#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consistency Policy 推理节点

作为 ZMQ 服务端运行，接收观测数据，返回动作预测。
参考: detached-umi-policy/detached_policy_inference.py

用法:
    python -m consistency_policy.policy_inference \
        --checkpoint /path/to/checkpoint.pt \
        --ip 0.0.0.0 \
        --port 8766

架构:
    - ZMQ REP 模式，监听推理请求
    - 输入: obs_dict_np (包含 rgb 和 state)
    - 输出: action (N, action_dim) 绝对关节角
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Optional, Dict, Any

import torch
import zmq

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from consistency_policy.config import setup_rlft


class PolicyInferenceNode:
    """策略推理节点"""
    
    def __init__(
        self,
        checkpoint_path: str,
        ip: str = "0.0.0.0",
        port: int = 8766,
        device: str = "cuda",
        use_ema: bool = True,
        verbose: bool = True,
    ):
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.ip = ip
        self.port = port
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_ema = use_ema
        self.verbose = verbose
        
        # 设置 rlft 路径
        setup_rlft()
        
        # 加载模块
        from diffusion_policy.utils import (
            get_real_robot_data_info,
        )
        self.get_real_robot_data_info = get_real_robot_data_info
        
        # 配置参数 (与训练一致)
        self.obs_horizon = 2
        self.act_horizon = 8
        self.pred_horizon = 16
        self.image_size = (128, 128)
        self.num_flow_steps = 10
        
        # 模型组件
        self.visual_encoder = None
        self.state_encoder = None
        self.agent = None
        
        # 加载模型
        self._load_model()
        
        if self.verbose:
            print(f"\n[PolicyInferenceNode] 初始化完成")
            print(f"  Checkpoint: {self.checkpoint_path}")
            print(f"  Device: {self.device}")
            print(f"  Use EMA: {self.use_ema}")
            print(f"  Obs horizon: {self.obs_horizon}")
            print(f"  Pred horizon: {self.pred_horizon}")
    
    def _load_model(self):
        """加载模型"""
        print(f"\n[PolicyInferenceNode] 加载模型: {self.checkpoint_path}")
        
        from diffusion_policy.plain_conv import PlainConv
        from diffusion_policy.utils import StateEncoder
        from diffusion_policy.algorithms import ConsistencyFlowAgent
        from diffusion_policy.algorithms.networks import VelocityUNet1D
        
        # 加载 checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 从 checkpoint 自动推断维度
        # state_encoder.mlp.0.weight 形状为 (hidden_dim, state_dim)
        state_encoder_state = ckpt.get("state_encoder", {})
        if "mlp.0.weight" in state_encoder_state:
            self.state_dim = state_encoder_state["mlp.0.weight"].shape[1]
        else:
            self.state_dim = 7  # 默认值
            print(f"  警告: 无法从 checkpoint 推断 state_dim，使用默认值 {self.state_dim}")
        
        # 从 agent/ema_agent 推断 action_dim
        agent_key = "ema_agent" if self.use_ema and "ema_agent" in ckpt else "agent"
        agent_state = ckpt.get(agent_key, {})
        
        # velocity_net.final_conv.2.weight 形状为 (action_dim, ...)
        action_dim_inferred = None
        for key, value in agent_state.items():
            if "final_conv.2.weight" in key:
                action_dim_inferred = value.shape[0]
                break
        
        if action_dim_inferred is not None:
            self.action_dim = action_dim_inferred
        else:
            self.action_dim = 7  # 默认值
            print(f"  警告: 无法从 checkpoint 推断 action_dim，使用默认值 {self.action_dim}")
        
        print(f"  推断维度: state_dim={self.state_dim}, action_dim={self.action_dim}")
        
        # 计算 global condition 维度
        visual_feature_dim = 256
        state_encoder_out_dim = 256
        global_cond_dim = self.obs_horizon * (visual_feature_dim + state_encoder_out_dim)
        
        # 创建视觉编码器
        self.visual_encoder = PlainConv(
            in_channels=3,
            out_dim=visual_feature_dim,
            pool_feature_map=True,
        ).to(self.device)
        
        # 创建状态编码器
        self.state_encoder = StateEncoder(
            state_dim=self.state_dim,
            hidden_dim=128,
            out_dim=state_encoder_out_dim,
        ).to(self.device)
        
        # 创建 agent
        velocity_net = VelocityUNet1D(
            input_dim=self.action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=64,
            down_dims=(64, 128, 256),
            n_groups=8,
        )
        
        self.agent = ConsistencyFlowAgent(
            velocity_net=velocity_net,
            action_dim=self.action_dim,
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            num_flow_steps=self.num_flow_steps,
            ema_decay=0.999,
            action_bounds=None,
            device=str(self.device),
        ).to(self.device)
        
        # 加载权重
        agent_key = "ema_agent" if self.use_ema else "agent"
        self.agent.load_state_dict(ckpt[agent_key])
        self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        
        # 设置为评估模式
        self.agent.eval()
        self.visual_encoder.eval()
        self.state_encoder.eval()
        
        param_count = sum(p.numel() for p in self.agent.parameters())
        print(f"  参数量: {param_count / 1e6:.2f}M")
        print(f"  EMA 权重: {self.use_ema}")
        print(f"  ✓ 模型加载完成")
    
    @torch.no_grad()
    def predict(self, obs_dict_np: Dict[str, np.ndarray]) -> np.ndarray:
        """
        单步预测
        
        Args:
            obs_dict_np: 观测字典
                - 'rgb': (T, H, W, C) or (T, C, H, W) uint8
                - 'state': (T, state_dim) float32
        
        Returns:
            action: (pred_horizon, action_dim) 动作序列
        """
        # 提取观测
        rgb = obs_dict_np['rgb']  # (T, H, W, C) or (T, C, H, W)
        state = obs_dict_np['state']  # (T, state_dim)
        
        # 状态维度检查
        actual_state_dim = state.shape[-1]
        if actual_state_dim != self.state_dim:
            raise ValueError(f"  警告: 状态维度不匹配! 期望 {self.state_dim}, 实际 {actual_state_dim}")
        
        # 转换为 tensor
        rgb = torch.from_numpy(rgb).to(self.device)
        state = torch.from_numpy(state).float().to(self.device)
        
        # 确保 batch 维度
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(0)  # (1, T, H, W, C) or (1, T, C, H, W)
        if state.dim() == 2:
            state = state.unsqueeze(0)  # (1, T, state_dim)
        
        B, T = rgb.shape[0], rgb.shape[1]
        
        # 处理图像格式: 确保是 (B*T, C, H, W)
        if rgb.shape[-1] == 3:  # NHWC format
            rgb = rgb.permute(0, 1, 4, 2, 3)  # -> (B, T, C, H, W)
        
        rgb_flat = rgb.view(B * T, *rgb.shape[2:]).float() / 255.0  # (B*T, C, H, W)
        
        # 编码视觉特征
        visual_feat = self.visual_encoder(rgb_flat)  # (B*T, visual_dim)
        visual_feat = visual_feat.view(B, T, -1)  # (B, T, visual_dim)
        
        # 编码状态特征
        state_flat = state.view(B * T, -1)  # (B*T, state_dim)
        state_feat = self.state_encoder(state_flat)  # (B*T, state_encoder_dim)
        state_feat = state_feat.view(B, T, -1)  # (B, T, state_encoder_dim)
        
        # 拼接观测特征
        obs_features = torch.cat([visual_feat, state_feat], dim=-1)  # (B, T, feat_dim)
        obs_cond = obs_features.view(B, -1)  # (B, T * feat_dim)
        
        # 推理
        action_seq = self.agent.get_action(obs_cond)  # (B, pred_horizon, action_dim)
        if isinstance(action_seq, tuple):
            action_seq = action_seq[0]
        
        # 返回动作序列
        return action_seq[0].cpu().numpy()  # (pred_horizon, action_dim)
    
    def run(self):
        """运行 ZMQ 服务"""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.ip}:{self.port}")
        
        print(f"\n[PolicyInferenceNode] 服务启动")
        print(f"  监听地址: tcp://{self.ip}:{self.port}")
        print(f"  等待推理请求...")
        
        # 推理统计
        inference_count = 0
        inference_times = []
        
        try:
            while True:
                # 接收观测
                t_recv = time.time()
                obs_dict_np = socket.recv_pyobj()
                
                try:
                    start_time = time.time()
                    
                    # 推理
                    action = self.predict(obs_dict_np)
                    
                    end_time = time.time()
                    inference_time = end_time - start_time
                    inference_times.append(inference_time)
                    if len(inference_times) > 100:
                        inference_times.pop(0)
                    
                    inference_count += 1
                    
                    if self.verbose:
                        avg_time = np.mean(inference_times) * 1000
                        print(f"[Inference #{inference_count}] 耗时: {inference_time * 1000:.1f}ms "
                              f"(avg: {avg_time:.1f}ms), 动作: {action.shape}, "
                              f"范围: [{action[:, :6].min():.3f}, {action[:, :6].max():.3f}]")
                    
                except Exception as e:
                    import traceback
                    err_str = traceback.format_exc()
                    print(f"[PolicyInferenceNode] 推理错误: {err_str}")
                    action = err_str
                
                # 发送结果
                socket.send_pyobj(action)
        
        except KeyboardInterrupt:
            print(f"\n[PolicyInferenceNode] 收到中断信号，关闭服务...")
        
        finally:
            socket.close()
            context.term()
            print(f"[PolicyInferenceNode] 服务已关闭")


def main():
    parser = argparse.ArgumentParser(description="Consistency Policy 推理节点")
    parser.add_argument("-c", "--checkpoint", required=True, help="Checkpoint 路径")
    parser.add_argument("--ip", default="0.0.0.0", help="监听 IP")
    parser.add_argument("--port", type=int, default=8766, help="监听端口")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--no-ema", action="store_true", help="不使用 EMA 权重")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Consistency Policy 推理节点")
    print("=" * 60)
    
    node = PolicyInferenceNode(
        checkpoint_path=args.checkpoint,
        ip=args.ip,
        port=args.port,
        device=args.device,
        use_ema=not args.no_ema,
        verbose=args.verbose or True,
    )
    
    node.run()


if __name__ == "__main__":
    main()
