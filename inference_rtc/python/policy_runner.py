#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略推理器

加载预训练模型，执行 10Hz 策略推理
"""

import os
import sys
import time
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque

import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.config import setup_rlft
from inference_rtc.python.config import RTCConfig


class PolicyRunner:
    """
    策略推理器
    
    负责:
    1. 加载预训练模型
    2. 维护观测历史
    3. 执行策略推理，输出关键帧动作
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: RTCConfig = None,
        device: str = "cuda",
        use_ema: bool = True,
        verbose: bool = True,
    ):
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.config = config or RTCConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_ema = use_ema
        self.verbose = verbose
        
        # 模型组件
        self.visual_encoder = None
        self.state_encoder = None
        self.agent = None
        self.algorithm = None
        
        # 观测历史
        self.obs_buffer: deque = deque(maxlen=self.config.obs_horizon)
        
        # 统计
        self.inference_count = 0
        self.total_inference_time = 0.0
    
    def load_model(self):
        """加载预训练模型"""
        print(f"\n[PolicyRunner] 加载策略: {self.checkpoint_path}")
        
        setup_rlft()
        
        try:
            from diffusion_policy.plain_conv import PlainConv
            from diffusion_policy.utils import StateEncoder
            from diffusion_policy.algorithms import (
                DiffusionPolicyAgent,
                FlowMatchingAgent,
                ConsistencyFlowAgent,
            )
            from diffusion_policy.algorithms.networks import VelocityUNet1D
            from diffusion_policy.conditional_unet1d import ConditionalUnet1D
            
            # 加载 checkpoint
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 自动检测算法类型
            agent_keys = list(ckpt.get('agent', ckpt.get('ema_agent', {})).keys())
            if any('velocity_net' in k for k in agent_keys):
                if any('velocity_net_ema' in k for k in agent_keys):
                    self.algorithm = "consistency_flow"
                else:
                    self.algorithm = "flow_matching"
            elif any('noise_pred_net' in k for k in agent_keys):
                self.algorithm = "diffusion_policy"
            else:
                path_lower = self.checkpoint_path.lower()
                if "diffusion_policy" in path_lower:
                    self.algorithm = "diffusion_policy"
                elif "consistency_flow" in path_lower:
                    self.algorithm = "consistency_flow"
                else:
                    self.algorithm = "flow_matching"
            
            print(f"  算法: {self.algorithm}")
            
            # 模型架构参数
            visual_feature_dim = 256
            state_encoder_hidden_dim = 128
            state_encoder_out_dim = 256
            diffusion_step_embed_dim = 64
            unet_dims = (64, 128, 256)
            n_groups = 8
            
            global_cond_dim = self.config.obs_horizon * (visual_feature_dim + state_encoder_out_dim)
            
            # 创建视觉编码器
            self.visual_encoder = PlainConv(
                in_channels=3,
                out_dim=visual_feature_dim,
                pool_feature_map=True,
            ).to(self.device)
            
            # 创建状态编码器
            self.state_encoder = StateEncoder(
                state_dim=self.config.state_dim,
                hidden_dim=state_encoder_hidden_dim,
                out_dim=state_encoder_out_dim,
            ).to(self.device)
            
            # 创建 agent
            if self.algorithm == "diffusion_policy":
                noise_pred_net = ConditionalUnet1D(
                    input_dim=self.config.action_dim,
                    global_cond_dim=global_cond_dim,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=unet_dims,
                    n_groups=n_groups,
                )
                self.agent = DiffusionPolicyAgent(
                    noise_pred_net=noise_pred_net,
                    action_dim=self.config.action_dim,
                    obs_horizon=self.config.obs_horizon,
                    pred_horizon=self.config.pred_horizon,
                    num_diffusion_iters=100,
                    device=str(self.device),
                )
            elif self.algorithm == "consistency_flow":
                velocity_net = VelocityUNet1D(
                    input_dim=self.config.action_dim,
                    global_cond_dim=global_cond_dim,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=unet_dims,
                    n_groups=n_groups,
                )
                self.agent = ConsistencyFlowAgent(
                    velocity_net=velocity_net,
                    action_dim=self.config.action_dim,
                    obs_horizon=self.config.obs_horizon,
                    pred_horizon=self.config.pred_horizon,
                    num_flow_steps=self.config.num_flow_steps,
                    ema_decay=0.999,
                    action_bounds=None,
                    device=str(self.device),
                )
            else:  # flow_matching
                velocity_net = VelocityUNet1D(
                    input_dim=self.config.action_dim,
                    global_cond_dim=global_cond_dim,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=unet_dims,
                    n_groups=n_groups,
                )
                self.agent = FlowMatchingAgent(
                    velocity_net=velocity_net,
                    action_dim=self.config.action_dim,
                    obs_horizon=self.config.obs_horizon,
                    pred_horizon=self.config.pred_horizon,
                    num_flow_steps=self.config.num_flow_steps,
                    action_bounds=None,
                    device=str(self.device),
                )
            
            self.agent = self.agent.to(self.device)
            
            # 加载权重
            agent_key = "ema_agent" if self.use_ema else "agent"
            if agent_key not in ckpt:
                agent_key = "agent"
                print(f"  警告: 未找到 EMA 权重，使用普通权重")
            
            self.agent.load_state_dict(ckpt[agent_key])
            self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
            self.state_encoder.load_state_dict(ckpt["state_encoder"])
            
            # 设置为评估模式
            self.agent.eval()
            self.visual_encoder.eval()
            self.state_encoder.eval()
            
            print(f"  obs_horizon: {self.config.obs_horizon}")
            print(f"  act_horizon: {self.config.act_horizon}")
            print(f"  pred_horizon: {self.config.pred_horizon}")
            print(f"  num_flow_steps: {self.config.num_flow_steps}")
            print(f"  EMA 权重: {agent_key == 'ema_agent'}")
            
        except Exception as e:
            print(f"[PolicyRunner] 加载策略失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def warmup(self, num_iters: int = 3):
        """GPU 预热"""
        if self.agent is None:
            return
        
        print("\n[PolicyRunner] GPU 预热中...")
        
        dummy_rgb = np.random.randint(
            0, 255, 
            (self.config.obs_horizon, 3, *self.config.image_size), 
            dtype=np.uint8
        )
        dummy_state = np.random.randn(
            self.config.obs_horizon, self.config.state_dim
        ).astype(np.float32)
        
        rgb = torch.from_numpy(dummy_rgb.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
        state = torch.from_numpy(dummy_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            B, T = rgb.shape[0], rgb.shape[1]
            rgb_flat = rgb.view(B * T, *rgb.shape[2:])
            state_flat = state.view(B * T, -1)
            
            for i in range(num_iters):
                visual_feat = self.visual_encoder(rgb_flat)
                visual_feat = visual_feat.view(B, T, -1)
                
                state_feat = self.state_encoder(state_flat)
                state_feat = state_feat.view(B, T, -1)
                
                obs_features = torch.cat([visual_feat, state_feat], dim=-1)
                obs_cond = obs_features.view(B, -1)
                
                _ = self.agent.get_action(obs_cond)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print("  GPU 预热完成!")
    
    def add_observation(self, rgb: np.ndarray, state: np.ndarray):
        """
        添加观测帧
        
        Args:
            rgb: [H, W, 3] uint8 RGB 图像
            state: [state_dim] float32 机器人状态
        """
        obs = {
            "rgb": rgb.copy(),
            "state": state.astype(np.float32).copy()
        }
        self.obs_buffer.append(obs)
    
    def predict_keyframes(self) -> np.ndarray:
        """
        执行策略推理，返回关键帧动作
        
        Returns:
            q_key: [act_horizon, action_dim] 关键帧动作
        """
        start_time = time.perf_counter()
        
        # 准备观测
        obs_dict = self._prepare_obs()
        
        rgb = torch.from_numpy(obs_dict["rgb"]).unsqueeze(0).to(self.device)
        state = torch.from_numpy(obs_dict["state"]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            B, T = rgb.shape[0], rgb.shape[1]
            rgb_flat = rgb.view(B * T, *rgb.shape[2:])
            state_flat = state.view(B * T, -1)
            
            visual_feat = self.visual_encoder(rgb_flat)
            visual_feat = visual_feat.view(B, T, -1)
            
            state_feat = self.state_encoder(state_flat)
            state_feat = state_feat.view(B, T, -1)
            
            obs_features = torch.cat([visual_feat, state_feat], dim=-1)
            obs_cond = obs_features.view(B, -1)
            
            action_seq = self.agent.get_action(obs_cond)
            
            if isinstance(action_seq, tuple):
                action_seq = action_seq[0]
        
        action_seq = action_seq[0].cpu().numpy()
        
        # 提取可执行动作 (关键帧)
        start = self.config.obs_horizon - 1
        end = start + self.config.act_horizon
        q_key = action_seq[start:end]
        
        # 统计
        inference_time = time.perf_counter() - start_time
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        if self.verbose and self.inference_count % 10 == 0:
            avg_time = self.total_inference_time / self.inference_count
            print(f"[PolicyRunner] 推理 #{self.inference_count}: {inference_time*1000:.1f}ms (avg: {avg_time*1000:.1f}ms)")
        
        return q_key
    
    def _prepare_obs(self) -> Dict[str, np.ndarray]:
        """准备观测数据"""
        obs_list = list(self.obs_buffer)
        
        if len(obs_list) == 0:
            # 没有观测，返回零
            dummy_obs = {
                "rgb": np.zeros((*self.config.image_size, 3), dtype=np.uint8),
                "state": np.zeros(self.config.state_dim, dtype=np.float32)
            }
            obs_list = [dummy_obs]
        
        # 填充到 obs_horizon
        while len(obs_list) < self.config.obs_horizon:
            obs_list.insert(0, obs_list[0])
        
        # 堆叠
        rgb_stack = np.stack([o["rgb"] for o in obs_list])
        state_stack = np.stack([o["state"] for o in obs_list])
        
        # RGB: [T, H, W, C] -> [T, C, H, W], 归一化
        rgb_nchw = np.transpose(rgb_stack, (0, 3, 1, 2))
        rgb_nchw = rgb_nchw.astype(np.float32) / 255.0
        
        return {"rgb": rgb_nchw, "state": state_stack}
    
    def clear_buffer(self):
        """清空观测缓冲区"""
        self.obs_buffer.clear()
    
    @property
    def avg_inference_time(self) -> float:
        """平均推理时间 (秒)"""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count
