#!/usr/bin/env python3
"""
CARM 离线评估脚本

使用数据集进行离线评估，验证推理 pipeline 并评估模型性能。
不需要机械臂，只需要 PyTorch 环境。

使用方法:
    python -m rlft.offline.eval_carm --model_path runs/consistency_flow/checkpoints/final.pt \
                                     --data_dir ~/rl-vla/recorded_data \
                                     --output_dir offline_results

    # EMA vs Non-EMA 对比
    python -m rlft.offline.eval_carm --model_path runs/consistency_flow/checkpoints/final.pt \
                                     --data_dir ~/rl-vla/recorded_data/mix \
                                     --compare_ema
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import h5py
import cv2
from collections import deque

import torch
import torch.nn as nn
from einops import rearrange

# rlft 训练框架中的共享模块
from rlft.networks import (
    StateEncoder, GripperHead,
    create_visual_encoder,
)
from rlft.datasets import (
    ActionNormalizer,
    load_carm_episode,
    compute_relative_pose_transform,
)
from rlft.utils.model_factory import create_agent_for_inference


class OfflinePolicy:
    """
    离线策略推理类
    与 RealPolicy (policy_loader.py) 完全对齐，用于离线测试验证推理 pipeline
    
    支持:
        - EMA 和非 EMA 模型推理对比
        - 不同推理步数对比
    """
    
    # 默认参数 — 与 policy_loader.py 中 POLICY_DEFAULTS 保持一致
    DEFAULTS = {
        'obs_horizon': 2,
        'pred_horizon': 16,
        'action_dim': 13,
        'action_dim_full': 15,
        'state_mode': 'joint_only',
        'target_image_size': (128, 128),
        'visual_feature_dim': 256,
        'state_encoder_hidden_dim': 128,
        'state_encoder_out_dim': 256,
        'use_state_encoder': True,
        'algorithm': 'consistency_flow',
        'num_inference_steps': 10,
        'gripper_threshold': 0.05,
        'gripper_open_val': 0.078,
        'gripper_close_val': 0.04,
        'gripper_head_hidden_dim': 256,
        'gripper_hysteresis_window': 1,
    }
    
    def __init__(self, model_path: str, device: str = 'cuda', use_ema: bool = False,
                 num_inference_steps: Optional[int] = None,
                 gripper_hysteresis_window: Optional[int] = None):
        """
        Args:
            model_path: 模型 checkpoint 路径
            device: 推理设备
            use_ema: 是否使用 EMA 模型进行推理
            num_inference_steps: 推理步数 (None = 使用默认值 10)
            gripper_hysteresis_window: Gripper 滞后窗口 (None = 使用默认值 1)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.use_ema = use_ema
        
        _d = self.DEFAULTS
        
        # 默认参数（会被 _load_model -> args.json 覆盖）
        self.obs_horizon = _d['obs_horizon']
        self.pred_horizon = _d['pred_horizon']
        self.action_dim = _d['action_dim']
        self.action_dim_full = _d['action_dim_full']
        self.state_dim = 7
        self.target_image_size = _d['target_image_size']
        self.visual_feature_dim = _d['visual_feature_dim']
        self.state_encoder_hidden_dim = _d['state_encoder_hidden_dim']
        self.state_encoder_out_dim = _d['state_encoder_out_dim']
        self.use_state_encoder = _d['use_state_encoder']
        self.algorithm = _d['algorithm']
        self.visual_encoder_type = 'plain_conv'
        
        # 推理参数
        self.num_inference_steps = num_inference_steps or _d['num_inference_steps']
        
        # State/Action mode
        self.state_mode = _d['state_mode']
        self.action_mode = 'full'
        self.normalize_actions = False
        self.action_norm_mode = 'standard'
        
        # 模型组件
        self.visual_encoder = None
        self.state_encoder = None
        self.agent = None
        self.gripper_head = None
        self.action_normalizer = None
        
        # Gripper 配置 — 与 policy_loader 一致
        self.gripper_threshold = _d['gripper_threshold']
        self.gripper_open_val = _d['gripper_open_val']
        self.gripper_close_val = _d['gripper_close_val']
        self.gripper_head_hidden_dim = _d['gripper_head_hidden_dim']
        
        # Hysteresis — 与 policy_loader 一致
        self.gripper_hysteresis_window = gripper_hysteresis_window or _d['gripper_hysteresis_window']
        self._gripper_history = deque(maxlen=self.gripper_hysteresis_window)
        self._last_gripper_state = 0  # 0=open, 1=close
        
        # 观测历史
        self.obs_history = {'rgb': [], 'state': []}
        
        # 加载模型
        self._load_model()
    
    @staticmethod
    def _get_state_dim_for_mode(state_mode: str) -> int:
        if state_mode == 'joint_only':
            return 7
        elif state_mode == 'ee_only':
            return 8
        elif state_mode == 'both':
            return 14
        else:
            print(f"Warning: Unknown state_mode: {state_mode}, defaulting to joint_only")
            return 7
    
    def _load_model(self):
        """加载模型 — 对齐 policy_loader.py 中 RealPolicy.load_model()"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")
        
        checkpoint_dir = os.path.dirname(self.model_path)
        
        # 1. 加载训练配置 ------------------------------------------------
        args_path = os.path.join(checkpoint_dir, "args.json")
        visual_encoder_type = 'plain_conv'
        
        if os.path.exists(args_path):
            print(f"Loading config from: {args_path}")
            with open(args_path, 'r') as f:
                args = json.load(f)
            
            self.obs_horizon = args.get('obs_horizon', self.obs_horizon)
            self.pred_horizon = args.get('pred_horizon', self.pred_horizon)
            self.action_dim = args.get('action_dim', self.action_dim)
            self.visual_feature_dim = args.get('visual_feature_dim', self.visual_feature_dim)
            self.state_encoder_hidden_dim = args.get('state_encoder_hidden_dim', self.state_encoder_hidden_dim)
            self.state_encoder_out_dim = args.get('state_encoder_out_dim', self.state_encoder_out_dim)
            self.use_state_encoder = args.get('use_state_encoder', self.use_state_encoder)
            self.algorithm = args.get('algorithm', self.algorithm)
            
            # State mode
            self.state_mode = args.get('state_mode', 'joint_only')
            self.state_dim = self._get_state_dim_for_mode(self.state_mode)
            
            # Visual encoder
            visual_encoder_type = args.get('visual_encoder_type', 'plain_conv')
            self.visual_encoder_type = visual_encoder_type
            
            # Image size — 与 policy_loader 对齐: auto_image_size 默认 True
            if args.get('auto_image_size', True):
                if visual_encoder_type in ['resnet18', 'resnet34', 'resnet50']:
                    self.target_image_size = (224, 224)
                else:
                    self.target_image_size = (128, 128)
            else:
                target_size = args.get('target_image_size', self.target_image_size)
                if isinstance(target_size, str):
                    import ast
                    target_size = ast.literal_eval(target_size)
                elif isinstance(target_size, list):
                    target_size = tuple(target_size)
                self.target_image_size = target_size
            
            # action_mode
            self.action_mode = args.get('action_mode', 'full')
            if self.action_mode == 'full':
                self.action_dim = 13
                self.action_dim_full = 15
            else:
                self.action_dim = 7
                self.action_dim_full = 8
            
            # Gripper
            self.gripper_threshold = args.get('gripper_threshold', self.gripper_threshold)
            self.gripper_open_val = args.get('gripper_open_val', self.gripper_open_val)
            self.gripper_close_val = args.get('gripper_close_val', self.gripper_close_val)
            self.gripper_head_hidden_dim = args.get('gripper_head_hidden_dim', self.gripper_head_hidden_dim)
            
            # Action normalization
            self.normalize_actions = args.get('normalize_actions', False)
            self.action_norm_mode = args.get('action_norm_mode', 'standard')
            
            print(f"Config: algorithm={self.algorithm}, action_dim={self.action_dim} (continuous), "
                  f"obs_horizon={self.obs_horizon}, pred_horizon={self.pred_horizon}")
            print(f"State mode: {self.state_mode}, state_dim={self.state_dim}")
            print(f"Discrete gripper: threshold={self.gripper_threshold}, open={self.gripper_open_val}, close={self.gripper_close_val}")
            print(f"Inference config: num_steps={self.num_inference_steps}, use_ema={self.use_ema}")
            print(f"Visual encoder: {visual_encoder_type}, image_size={self.target_image_size}")
            print(f"Action normalization: {self.normalize_actions}, mode={self.action_norm_mode}")
        else:
            print("Warning: args.json not found, using defaults")
        
        # 2. 创建模型 ------------------------------------------------
        print("Creating models...")
        
        # Visual encoder — 与 policy_loader 对齐: freeze_bn=False
        self.visual_encoder = create_visual_encoder(
            encoder_type=visual_encoder_type,
            out_dim=self.visual_feature_dim,
            pretrained=True,
            freeze_backbone=False,
            freeze_bn=False,
        ).to(self.device)
        print(f"Created visual encoder: {visual_encoder_type}")
        
        # State encoder
        encoded_state_dim = self.state_dim
        if self.use_state_encoder:
            self.state_encoder = StateEncoder(
                state_dim=self.state_dim,
                hidden_dim=self.state_encoder_hidden_dim,
                out_dim=self.state_encoder_out_dim,
            ).to(self.device)
            encoded_state_dim = self.state_encoder_out_dim
        
        # global_cond_dim
        global_cond_dim = self.obs_horizon * (self.visual_feature_dim + encoded_state_dim)
        print(f"global_cond_dim={global_cond_dim} = {self.obs_horizon} * ({self.visual_feature_dim} + {encoded_state_dim})")
        
        # Agent
        self.agent = self._create_agent(global_cond_dim)
        
        # 3. 加载权重 ------------------------------------------------
        print(f"Loading checkpoint from: {self.model_path}")
        ckpt = torch.load(self.model_path, map_location=self.device)
        
        # visual encoder
        if "visual_encoder" in ckpt:
            self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
            print("Loaded visual_encoder weights")
        
        # state encoder
        if self.state_encoder is not None and "state_encoder" in ckpt:
            self.state_encoder.load_state_dict(ckpt["state_encoder"])
            print("Loaded state_encoder weights")
        
        # agent
        if self.use_ema:
            if "ema_agent" in ckpt:
                self.agent.load_state_dict(ckpt["ema_agent"])
                print("Loaded EMA agent weights")
            elif "agent" in ckpt:
                print("Warning: EMA agent not found, falling back to regular agent")
                self.agent.load_state_dict(ckpt["agent"])
            else:
                raise ValueError("No agent weights in checkpoint")
        else:
            if "agent" in ckpt:
                self.agent.load_state_dict(ckpt["agent"])
                print("Loaded regular agent weights")
            elif "ema_agent" in ckpt:
                print("Warning: Regular agent not found, falling back to EMA agent")
                self.agent.load_state_dict(ckpt["ema_agent"])
            else:
                raise ValueError("No agent weights in checkpoint")
        
        # GripperHead — detect architecture from checkpoint keys
        gripper_input_dim = self.obs_horizon * (self.visual_feature_dim + encoded_state_dim)
        gripper_use_layernorm = True
        if "gripper_head" in ckpt:
            ckpt_keys = set(ckpt["gripper_head"].keys())
            if "net.2.weight" in ckpt_keys and "net.1.weight" not in ckpt_keys:
                gripper_use_layernorm = False
                print("Detected legacy GripperHead (no LayerNorm)")
            else:
                print("Detected GripperHead with LayerNorm")
        
        self.gripper_head = GripperHead(
            obs_dim=gripper_input_dim,
            hidden_dim=self.gripper_head_hidden_dim,
            pred_horizon=self.pred_horizon,
            use_layernorm=gripper_use_layernorm,
        ).to(self.device)
        print(f"Created GripperHead: obs_dim={gripper_input_dim}, hidden_dim={self.gripper_head_hidden_dim}, layernorm={gripper_use_layernorm}")
        
        if "gripper_head" in ckpt:
            self.gripper_head.load_state_dict(ckpt["gripper_head"])
            print("Loaded gripper_head weights")
        else:
            print("Warning: gripper_head not in checkpoint! Using random initialization")
        
        # Action normalizer
        if self.normalize_actions:
            loaded = False
            
            if "action_normalizer" in ckpt:
                self.action_normalizer = ActionNormalizer.from_checkpoint(ckpt["action_normalizer"])
                print("Loaded action normalizer from checkpoint")
                loaded = True
            else:
                normalizer_path = os.path.join(checkpoint_dir, "action_normalizer.json")
                if os.path.exists(normalizer_path):
                    self.action_normalizer = ActionNormalizer(mode=self.action_norm_mode)
                    self.action_normalizer.load(normalizer_path)
                    print(f"Loaded action normalizer from: {normalizer_path}")
                    loaded = True
            
            if loaded:
                print(f"  Mode: {self.action_normalizer.mode}")
                if self.action_normalizer.mode == 'standard' and self.action_normalizer.stats:
                    print(f"  Mean: {self.action_normalizer.stats['mean'][:3]}...")
                    print(f"  Std:  {self.action_normalizer.stats['std'][:3]}...")
            else:
                print("Warning: normalize_actions=True but action_normalizer not found!")
        
        # 评估模式
        self.visual_encoder.eval()
        if self.state_encoder is not None:
            self.state_encoder.eval()
        self.agent.eval()
        self.gripper_head.eval()
        
        print(f"Model loaded successfully! (use_ema={self.use_ema})")
    
    def _create_agent(self, global_cond_dim: int) -> nn.Module:
        """创建 agent — 使用 self.num_inference_steps（与 policy_loader 对齐）"""
        return create_agent_for_inference(
            algorithm=self.algorithm,
            action_dim=self.action_dim,
            global_cond_dim=global_cond_dim,
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            num_inference_steps=self.num_inference_steps,
            device=str(self.device),
        ).to(self.device)
    
    def reset(self):
        """重置观测历史和 gripper 状态"""
        self.obs_history = {'rgb': [], 'state': []}
        self._gripper_history.clear()
        self._last_gripper_state = 0
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像: resize + HWC -> CHW"""
        if self.target_image_size is not None:
            h, w = self.target_image_size
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = rearrange(image, 'h w c -> c h w')
        return image
    
    def _update_obs_history(self, rgb: np.ndarray, state: np.ndarray):
        """更新观测历史"""
        self.obs_history['rgb'].append(rgb)
        self.obs_history['state'].append(state)
        
        if len(self.obs_history['rgb']) > self.obs_horizon:
            self.obs_history['rgb'].pop(0)
            self.obs_history['state'].pop(0)
        
        while len(self.obs_history['rgb']) < self.obs_horizon:
            self.obs_history['rgb'].insert(0, self.obs_history['rgb'][0])
            self.obs_history['state'].insert(0, self.obs_history['state'][0])
    
    def _encode_observations(self) -> torch.Tensor:
        """编码观测"""
        B, T = 1, self.obs_horizon
        
        rgb_list = [torch.from_numpy(r).float() for r in self.obs_history['rgb']]
        rgb = torch.stack(rgb_list, dim=0).unsqueeze(0).to(self.device)
        rgb_flat = rgb.view(B * T, *rgb.shape[2:]) / 255.0
        visual_feat = self.visual_encoder(rgb_flat)
        visual_feat = visual_feat.view(B, T, -1)
        
        state_list = [torch.from_numpy(s).float() for s in self.obs_history['state']]
        state = torch.stack(state_list, dim=0).unsqueeze(0).to(self.device)
        
        if self.state_encoder is not None:
            state_flat = state.view(B * T, -1)
            state_feat = self.state_encoder(state_flat)
            state_feat = state_feat.view(B, T, -1)
        else:
            state_feat = state
        
        obs_features = torch.cat([visual_feat, state_feat], dim=-1)
        return obs_features
    
    @torch.no_grad()
    def predict(self, image: np.ndarray, qpos: np.ndarray, 
                num_steps: Optional[int] = None,
                deterministic: bool = True) -> np.ndarray:
        """
        执行推理 — 对齐 policy_loader.py 中 RealPolicy.__call__()
        
        Args:
            image: RGB 图像 [H, W, C]
            qpos: 关节状态，根据 state_mode 维度不同
            num_steps: 推理步数 (None = 使用默认值)
            deterministic: 是否使用确定性推理 (从零开始而非噪声)
            
        Returns:
            actions: 预测动作 [pred_horizon, action_dim_full]
        """
        # 预处理
        image_processed = self._preprocess_image(image)
        self._update_obs_history(image_processed, qpos)
        
        # 编码观测
        obs_features = self._encode_observations()
        
        # 1. continuous actions
        if deterministic:
            actions_cont = self.agent.get_action_deterministic(obs_features, num_steps=num_steps)
        else:
            actions_cont = self.agent.get_action(obs_features, num_steps=num_steps)
        
        # 2. inverse normalization
        if self.action_normalizer is not None:
            actions_np = actions_cont.cpu().numpy()
            batch_size, pred_horizon, action_dim = actions_np.shape
            actions_flat = actions_np.reshape(-1, action_dim)
            actions_denorm = self.action_normalizer.inverse_transform(actions_flat)
            actions_cont = torch.from_numpy(
                actions_denorm.reshape(batch_size, pred_horizon, action_dim)
            ).to(self.device).float()
        
        # 3. discrete gripper — flatten if 3D (对齐 policy_loader)
        obs_flat = obs_features.reshape(obs_features.shape[0], -1) if obs_features.dim() == 3 else obs_features
        gripper_logits = self.gripper_head(obs_flat)
        gripper_cls = gripper_logits.argmax(dim=-1)
        
        # 4. hysteresis — 使用与 policy_loader 相同的逻辑
        gripper_vals = self._apply_gripper_hysteresis(gripper_cls[0].cpu().numpy())
        
        # 5. reconstruct full action
        actions_full = self._reconstruct_full_action(actions_cont[0], gripper_vals)
        
        return actions_full.cpu().numpy()
    
    def _apply_gripper_hysteresis(self, gripper_cls: np.ndarray) -> np.ndarray:
        """
        Apply hysteresis to gripper predictions.
        与 policy_loader.py 中 RealPolicy._apply_gripper_hysteresis() 完全对齐。
        
        逻辑：
        1. 对 pred_horizon 前 8 帧做多数投票得到 current_vote
        2. 加入 temporal history (deque)
        3. 根据 gripper_hysteresis_window 做二次投票
        """
        # Step 1: horizon vote
        horizon_vote_frames = min(8, len(gripper_cls))
        horizon_votes = gripper_cls[:horizon_vote_frames]
        close_in_horizon = np.sum(horizon_votes == 1)
        current_vote = 1 if close_in_horizon > horizon_vote_frames / 2 else 0
        
        self._gripper_history.append(current_vote)
        
        # Step 2: temporal voting
        if self.gripper_hysteresis_window == 1:
            new_state = current_vote
        elif len(self._gripper_history) >= max(1, self.gripper_hysteresis_window // 2):
            close_votes = sum(self._gripper_history)
            total_votes = len(self._gripper_history)
            new_state = 1 if close_votes > total_votes / 2 else 0
        else:
            new_state = current_vote
        
        self._last_gripper_state = new_state
        
        gripper_val = self.gripper_close_val if new_state == 1 else self.gripper_open_val
        return np.full(len(gripper_cls), gripper_val, dtype=np.float32)
    
    def _reconstruct_full_action(self, actions_cont: torch.Tensor,
                                  gripper_vals: np.ndarray) -> torch.Tensor:
        """
        Reconstruct full action tensor by inserting gripper values.
        与 policy_loader.py 中 RealPolicy._reconstruct_full_action() 对齐。
        """
        pred_horizon = actions_cont.shape[0]
        actions_full = torch.zeros(pred_horizon, self.action_dim_full, device=actions_cont.device)
        
        if self.action_dim_full == 15:  # full mode
            actions_full[:, :6] = actions_cont[:, :6]
            actions_full[:, 6] = torch.from_numpy(gripper_vals).to(actions_cont.device)
            actions_full[:, 7:14] = actions_cont[:, 6:13]
            actions_full[:, 14] = torch.from_numpy(gripper_vals).to(actions_cont.device)
        else:  # ee_only mode
            actions_full[:, :7] = actions_cont[:, :7]
            actions_full[:, 7] = torch.from_numpy(gripper_vals).to(actions_cont.device)
        
        return actions_full


class OfflineEvaluator:
    """
    离线评估器
    使用数据集评估模型性能
    
    支持:
        - EMA vs 非 EMA 模型对比
        - 不同推理步数对比
    """
    
    def __init__(self, model_path: str, data_dir: str, output_dir: str = 'offline_results',
                 use_ema: bool = False):
        """
        Args:
            model_path: 模型 checkpoint 路径
            data_dir: 数据集目录
            output_dir: 输出目录
            use_ema: 是否使用 EMA 模型
        """
        self.model_path = model_path
        self.data_dir = os.path.expanduser(data_dir)
        self.output_dir = output_dir
        self.use_ema = use_ema
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型
        print(f"Loading model (use_ema={use_ema})...")
        self.policy = OfflinePolicy(model_path, use_ema=use_ema)
        
        # 获取数据集文件列表
        self.episode_files = sorted([
            f for f in os.listdir(self.data_dir) 
            if f.startswith('episode_') and f.endswith('.hdf5')
        ])
        print(f"Found {len(self.episode_files)} episodes in {self.data_dir}")
    
    def evaluate_episode(self, ep_idx: int, verbose: bool = False,
                         num_steps: Optional[int] = None,
                         deterministic: bool = True) -> Dict:
        """
        评估单个 episode
        
        Args:
            ep_idx: episode 索引
            verbose: 是否显示进度条
            num_steps: 推理步数 (None = 使用默认值)
            deterministic: 是否使用确定性推理
        
        Returns:
            Dict with predicted_actions, gt_actions, metrics
        """
        filepath = os.path.join(self.data_dir, self.episode_files[ep_idx])
        episode = load_carm_episode(filepath)
        
        self.policy.reset()
        
        predicted_actions = []
        gt_actions = []
        
        T = len(episode['qpos_joint'])
        
        # ======== 正确计算相对动作（对齐训练逻辑） ========
        raw_actions = episode['action']  # [T, 15] 原始动作（包含目标位姿）
        qpos_end = episode['qpos_end']   # [T, 8] 当前末端位姿 [x,y,z,qx,qy,qz,qw,gripper]
        
        # 构建相对动作：计算从当前位姿到目标位姿的相对变换（与训练一致）
        relative_actions = np.zeros_like(raw_actions)
        for t in range(T):
            relative_actions[t, :6] = raw_actions[t, :6]  # 关节角度（绝对值）
            relative_actions[t, 6] = raw_actions[t, 6]    # 夹爪状态
            
            # 计算相对位姿：从当前位姿到目标位姿（与训练时一致）
            ref_pose = qpos_end[t, :7]           # 当前帧末端位姿
            target_pose = raw_actions[t, 7:14]   # 目标位姿（来自 action）
            relative_pose = compute_relative_pose_transform(ref_pose, target_pose)
            relative_actions[t, 7:14] = relative_pose
            
            relative_actions[t, 14] = raw_actions[t, 14]  # 末端夹爪
        
        iterator = tqdm(range(T), desc=f"Episode {ep_idx}") if verbose else range(T)
        
        # 获取 state_mode 和 action_mode
        state_mode = self.policy.state_mode
        action_mode = self.policy.action_mode
        
        for t in iterator:
            # 获取当前帧数据
            image = episode['images'][t]  # [H, W, C]
            
            # 根据 state_mode 构建状态向量
            if state_mode == 'joint_only':
                qpos = episode['qpos_joint'][t]  # [7]
            elif state_mode == 'ee_only':
                qpos = episode['qpos_end'][t]    # [8]
            else:  # both
                qpos = np.concatenate([
                    episode['qpos_joint'][t][:6],  # [6] joint (no gripper)
                    episode['qpos_end'][t]          # [8] ee pose
                ])  # [14]
            
            # 根据 action_mode 构建 GT action
            if action_mode == 'full':
                gt_action = relative_actions[t]  # [15]
            else:  # ee_only
                gt_action = relative_actions[t, 7:15]  # [8] rel_pose(7) + gripper(1)
            
            # 推理
            pred_actions = self.policy.predict(
                image, qpos, 
                num_steps=num_steps,
                deterministic=deterministic
            )  # [pred_horizon, action_dim_full]
            pred_action = pred_actions[0]  # 取第一个预测动作
            
            predicted_actions.append(pred_action)
            gt_actions.append(gt_action)
        
        predicted_actions = np.array(predicted_actions)
        gt_actions = np.array(gt_actions)
        
        # 计算指标
        metrics = self._compute_metrics(predicted_actions, gt_actions)
        
        return {
            'predicted_actions': predicted_actions,
            'gt_actions': gt_actions,
            'metrics': metrics,
            'qpos_joint': episode['qpos_joint'],
            'qpos_end': episode['qpos_end'],
        }
    
    def _compute_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict:
        """计算评估指标（支持 full 和 ee_only 两种 action_mode）"""
        action_mode = self.policy.action_mode
        
        if action_mode == 'full':
            # full mode: [joint(6), gripper(1), relative_end_pose(7), gripper(1)]
            joint_pred = pred[:, :6]
            joint_gt = gt[:, :6]
            gripper_joint_pred = pred[:, 6]
            gripper_joint_gt = gt[:, 6]
            pose_pred = pred[:, 7:14]
            pose_gt = gt[:, 7:14]
            gripper_pose_pred = pred[:, 14]
            gripper_pose_gt = gt[:, 14]

            # 推理实际使用: relative_end_pose + gripper
            ee_pred = np.concatenate([pred[:, 6:7], pred[:, 7:14]], axis=1)
            ee_gt = np.concatenate([gt[:, 6:7], gt[:, 7:14]], axis=1)

            # 计算 MSE
            joint_mse = np.mean((joint_pred - joint_gt) ** 2)
            gripper_joint_mse = np.mean((gripper_joint_pred - gripper_joint_gt) ** 2)
            gripper_pose_mse = np.mean((gripper_pose_pred - gripper_pose_gt) ** 2)
            pose_mse = np.mean((pose_pred - pose_gt) ** 2)
            ee_mse = np.mean((ee_pred - ee_gt) ** 2)
            total_mse = np.mean((pred - gt) ** 2)

            # 计算 MAE
            joint_mae = np.mean(np.abs(joint_pred - joint_gt))
            gripper_joint_mae = np.mean(np.abs(gripper_joint_pred - gripper_joint_gt))
            gripper_pose_mae = np.mean(np.abs(gripper_pose_pred - gripper_pose_gt))
            pose_mae = np.mean(np.abs(pose_pred - pose_gt))
            ee_mae = np.mean(np.abs(ee_pred - ee_gt))
            total_mae = np.mean(np.abs(pred - gt))

            # 计算各关节的误差
            joint_errors = []
            for i in range(6):
                joint_errors.append({
                    'mse': np.mean((joint_pred[:, i] - joint_gt[:, i]) ** 2),
                    'mae': np.mean(np.abs(joint_pred[:, i] - joint_gt[:, i])),
                    'max': np.max(np.abs(joint_pred[:, i] - joint_gt[:, i])),
                })

            return {
                'joint_mse': joint_mse,
                'joint_mae': joint_mae,
                'gripper_joint_mse': gripper_joint_mse,
                'gripper_joint_mae': gripper_joint_mae,
                'gripper_pose_mse': gripper_pose_mse,
                'gripper_pose_mae': gripper_pose_mae,
                'pose_mse': pose_mse,
                'pose_mae': pose_mae,
                'ee_mse': ee_mse,
                'ee_mae': ee_mae,
                'total_mse': total_mse,
                'total_mae': total_mae,
                'joint_errors': joint_errors,
            }
        else:  # ee_only
            # ee_only mode: [relative_end_pose(7), gripper(1)]
            pose_pred = pred[:, :7]
            pose_gt = gt[:, :7]
            gripper_pred = pred[:, 7]
            gripper_gt = gt[:, 7]

            # 计算 MSE/MAE
            pose_mse = np.mean((pose_pred - pose_gt) ** 2)
            pose_mae = np.mean(np.abs(pose_pred - pose_gt))
            gripper_mse = np.mean((gripper_pred - gripper_gt) ** 2)
            gripper_mae = np.mean(np.abs(gripper_pred - gripper_gt))
            total_mse = np.mean((pred - gt) ** 2)
            total_mae = np.mean(np.abs(pred - gt))

            # EE = pose + gripper
            ee_mse = total_mse
            ee_mae = total_mae

            return {
                'joint_mse': 0.0,  # N/A for ee_only
                'joint_mae': 0.0,
                'gripper_joint_mse': gripper_mse,
                'gripper_joint_mae': gripper_mae,
                'gripper_pose_mse': gripper_mse,
                'gripper_pose_mae': gripper_mae,
                'pose_mse': pose_mse,
                'pose_mae': pose_mae,
                'ee_mse': ee_mse,
                'ee_mae': ee_mae,
                'total_mse': total_mse,
                'total_mae': total_mae,
                'joint_errors': [],  # N/A for ee_only
            }
    
    def plot_episode_comparison(self, result: Dict, ep_idx: int, save: bool = True):
        """绘制单个 episode 的对比曲线"""
        pred = result['predicted_actions']
        gt = result['gt_actions']
        num_steps = len(pred)
        time_steps = np.arange(num_steps)
        
        # 创建图形
        fig, axes = plt.subplots(3, 4, figsize=(22, 12))
        fig.suptitle(f'Episode {ep_idx}: Predicted vs Ground Truth', fontsize=14)
        
        # 1. 关节 1-3
        for i in range(3):
            ax = axes[0, i]
            ax.plot(time_steps, gt[:, i], 'b-', label='Ground Truth', alpha=0.7)
            ax.plot(time_steps, pred[:, i], 'r--', label='Predicted', alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'Joint {i+1} (rad)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Joint {i+1}')
        
        # 2. 关节 4-6
        for i in range(3):
            ax = axes[1, i]
            ax.plot(time_steps, gt[:, i+3], 'b-', label='Ground Truth', alpha=0.7)
            ax.plot(time_steps, pred[:, i+3], 'r--', label='Predicted', alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'Joint {i+4} (rad)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Joint {i+4}')

        # 3. EE 位置 (x, y, z)
        ax = axes[0, 3]
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']
        for i, (c, l) in enumerate(zip(colors, labels)):
            ax.plot(time_steps, gt[:, 7+i], f'{c}-', label=f'GT {l}', alpha=0.5)
            ax.plot(time_steps, pred[:, 7+i], f'{c}--', label=f'Pred {l}', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Relative Position (m)')
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_title('EE Relative Position')

        # 4. EE 四元数 (qx, qy, qz, qw)
        ax = axes[1, 3]
        q_labels = ['qx', 'qy', 'qz', 'qw']
        for i, l in enumerate(q_labels):
            ax.plot(time_steps, gt[:, 10+i], label=f'GT {l}', alpha=0.5)
            ax.plot(time_steps, pred[:, 10+i], '--', label=f'Pred {l}', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Quat')
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_title('EE Relative Rotation')
        
        # 5. 夹爪（joint 通道）
        ax = axes[2, 0]
        ax.plot(time_steps, gt[:, 6], 'b-', label='Ground Truth', alpha=0.7)
        ax.plot(time_steps, pred[:, 6], 'r--', label='Predicted', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Gripper (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Gripper (joint channel)')

        # 6. 夹爪（ee 通道）
        ax = axes[2, 1]
        ax.plot(time_steps, gt[:, 14], 'b-', label='Ground Truth', alpha=0.7)
        ax.plot(time_steps, pred[:, 14], 'r--', label='Predicted', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Gripper (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Gripper (ee channel)')
        
        # 7. EE MAE
        ax = axes[2, 2]
        ee_pred = np.concatenate([pred[:, 6:7], pred[:, 7:14]], axis=1)
        ee_gt = np.concatenate([gt[:, 6:7], gt[:, 7:14]], axis=1)
        ee_err = np.mean(np.abs(ee_pred - ee_gt), axis=1)
        ax.plot(time_steps, ee_err, 'k-', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('MAE')
        ax.grid(True, alpha=0.3)
        ax.set_title('EE MAE per step')

        # 8. Total MAE
        ax = axes[2, 3]
        total_err = np.mean(np.abs(pred - gt), axis=1)
        ax.plot(time_steps, total_err, 'k-', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('MAE')
        ax.grid(True, alpha=0.3)
        ax.set_title('Total MAE per step')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'comparison_ep{ep_idx:03d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_error_distribution(self, all_results: List[Dict], save: bool = True):
        """绘制误差分布直方图"""
        all_pred = np.concatenate([r['predicted_actions'] for r in all_results], axis=0)
        all_gt = np.concatenate([r['gt_actions'] for r in all_results], axis=0)
        errors = all_pred - all_gt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Prediction Error Distribution', fontsize=14)
        
        # 关节误差
        for i in range(6):
            ax = axes[i // 4, i % 4]
            ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Error (rad)')
            ax.set_ylabel('Count')
            ax.set_title(f'Joint {i+1} Error')
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # 夹爪误差（joint 通道）
        ax = axes[1, 2]
        ax.hist(errors[:, 6], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Error (m)')
        ax.set_ylabel('Count')
        ax.set_title('Gripper Error (joint channel)')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # EE 通道误差分布
        ax = axes[1, 3]
        ee_errors = np.concatenate([errors[:, 6:7], errors[:, 7:14]], axis=1)
        ee_mae = np.mean(np.abs(ee_errors), axis=1)
        ax.hist(ee_mae, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('EE MAE')
        ax.set_ylabel('Count')
        ax.set_title('EE MAE per Step')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'error_distribution.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_cumulative_error(self, result: Dict, ep_idx: int, save: bool = True):
        """绘制累积误差曲线"""
        pred = result['predicted_actions']
        gt = result['gt_actions']
        num_steps = len(pred)
        
        # 计算每步误差
        step_errors = np.mean(np.abs(pred - gt), axis=1)
        cumulative_errors = np.cumsum(step_errors)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 每步误差
        ax = axes[0]
        ax.plot(np.arange(num_steps), step_errors, 'b-', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(f'Episode {ep_idx}: Step-wise Error')
        ax.grid(True, alpha=0.3)
        
        # 累积误差
        ax = axes[1]
        ax.plot(np.arange(num_steps), cumulative_errors, 'r-', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Error')
        ax.set_title(f'Episode {ep_idx}: Cumulative Error')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'cumulative_error_ep{ep_idx:03d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_ee_pose_detailed(self, result: Dict, ep_idx: int, save: bool = True):
        """详细绘制 EE Relative Pose 的每个维度"""
        pred = result['predicted_actions']
        gt = result['gt_actions']
        num_steps = len(pred)
        time_steps = np.arange(num_steps)
        
        # 创建 2x4 的图形：上排是位置 x,y,z + 位置MAE，下排是四元数 qx,qy,qz,qw
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Episode {ep_idx}: EE Relative Pose Detailed Comparison', fontsize=14)
        
        # 位置维度
        pos_labels = ['Rel X', 'Rel Y', 'Rel Z']
        for i, label in enumerate(pos_labels):
            ax = axes[0, i]
            gt_val = gt[:, 7+i]
            pred_val = pred[:, 7+i]
            
            ax.plot(time_steps, gt_val, 'b-', label='Ground Truth', alpha=0.7, linewidth=1.5)
            ax.plot(time_steps, pred_val, 'r--', label='Predicted', alpha=0.7, linewidth=1.5)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Position (m)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{label}\nGT range: [{gt_val.min():.4f}, {gt_val.max():.4f}]\n'
                        f'Pred range: [{pred_val.min():.4f}, {pred_val.max():.4f}]')
            
            # 计算并显示误差统计
            err = np.abs(gt_val - pred_val)
            ax.text(0.02, 0.98, f'MAE: {err.mean():.4f}\nMax: {err.max():.4f}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 位置 MAE
        ax = axes[0, 3]
        pos_err = np.abs(gt[:, 7:10] - pred[:, 7:10])
        ax.plot(time_steps, pos_err[:, 0], 'r-', label='X error', alpha=0.7)
        ax.plot(time_steps, pos_err[:, 1], 'g-', label='Y error', alpha=0.7)
        ax.plot(time_steps, pos_err[:, 2], 'b-', label='Z error', alpha=0.7)
        ax.plot(time_steps, pos_err.mean(axis=1), 'k--', label='Mean', alpha=0.9, linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Absolute Error (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Position Error per Dimension')
        
        # 四元数维度
        quat_labels = ['Rel qx', 'Rel qy', 'Rel qz', 'Rel qw']
        for i, label in enumerate(quat_labels):
            ax = axes[1, i]
            gt_val = gt[:, 10+i]
            pred_val = pred[:, 10+i]
            
            ax.plot(time_steps, gt_val, 'b-', label='Ground Truth', alpha=0.7, linewidth=1.5)
            ax.plot(time_steps, pred_val, 'r--', label='Predicted', alpha=0.7, linewidth=1.5)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Quaternion')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{label}\nGT range: [{gt_val.min():.4f}, {gt_val.max():.4f}]\n'
                        f'Pred range: [{pred_val.min():.4f}, {pred_val.max():.4f}]')
            
            # 计算并显示误差统计
            err = np.abs(gt_val - pred_val)
            ax.text(0.02, 0.98, f'MAE: {err.mean():.4f}\nMax: {err.max():.4f}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'ee_pose_detailed_ep{ep_idx:03d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def run_evaluation(self, num_episodes: Optional[int] = None, 
                       plot_individual: bool = True,
                       verbose: bool = True,
                       num_inference_steps: Optional[int] = None):
        """
        运行完整评估
        
        Args:
            num_episodes: 评估的 episode 数量 (None = all)
            plot_individual: 是否绘制每个 episode 的对比图
            verbose: 是否显示进度
            num_inference_steps: 推理步数 (None = 使用默认值)
        """
        if num_episodes is None:
            num_episodes = len(self.episode_files)
        
        all_results = []
        all_metrics = []
        
        ema_str = "EMA" if self.use_ema else "Non-EMA"
        print(f"\n{'='*60}")
        print(f"Offline Evaluation ({ema_str}): {num_episodes} episodes")
        if num_inference_steps is not None:
            print(f"Inference steps: {num_inference_steps}")
        print(f"{'='*60}\n")
        
        for ep_idx in range(num_episodes):
            print(f"\nEvaluating episode {ep_idx + 1}/{num_episodes}...")
            result = self.evaluate_episode(
                ep_idx, verbose=verbose, 
                num_steps=num_inference_steps
            )
            all_results.append(result)
            all_metrics.append(result['metrics'])
            
            # 绘制单个 episode 对比图
            if plot_individual:
                self.plot_episode_comparison(result, ep_idx)
                self.plot_cumulative_error(result, ep_idx)
                self.plot_ee_pose_detailed(result, ep_idx)
            
            # 打印当前 episode 指标
            m = result['metrics']
            print(f"  Joint MAE: {m['joint_mae']:.4f}, Gripper MAE: {m['gripper_joint_mae']:.4f}, "
                f"Pose MAE: {m['pose_mae']:.4f}, EE MAE: {m['ee_mae']:.4f}, Total MAE: {m['total_mae']:.4f}")
        
        # 计算整体指标
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics]) 
            for key in all_metrics[0].keys() if key != 'joint_errors'
        }
        
        # 绘制误差分布
        self.plot_error_distribution(all_results)
        
        # 保存结果
        self._save_results(all_results, avg_metrics)
        
        # 打印总结
        print(f"\n{'='*60}")
        print("Evaluation Summary")
        print(f"{'='*60}")
        print(f"Total Episodes: {num_episodes}")
        print(f"Average Joint MAE:   {avg_metrics['joint_mae']:.4f} rad")
        print(f"Average Gripper MAE: {avg_metrics['gripper_joint_mae']:.4f} m")
        print(f"Average Pose MAE:    {avg_metrics['pose_mae']:.4f}")
        print(f"Average EE MAE:      {avg_metrics['ee_mae']:.4f}")
        print(f"Average Total MAE:   {avg_metrics['total_mae']:.4f}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return avg_metrics
    
    def _save_results(self, all_results: List[Dict], avg_metrics: Dict):
        """保存评估结果"""
        # 将 numpy 类型转换为 Python 原生类型
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 保存指标
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'avg_metrics': convert_to_native(avg_metrics),
                'model_path': self.model_path,
                'data_dir': self.data_dir,
                'num_episodes': len(all_results),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        
        # 保存详细结果到 HDF5
        results_path = os.path.join(self.output_dir, 'evaluation_results.hdf5')
        with h5py.File(results_path, 'w') as f:
            for i, result in enumerate(all_results):
                grp = f.create_group(f'episode_{i:03d}')
                grp.create_dataset('predicted_actions', data=result['predicted_actions'])
                grp.create_dataset('gt_actions', data=result['gt_actions'])
                
                metrics_grp = grp.create_group('metrics')
                for k, v in result['metrics'].items():
                    if k != 'joint_errors':
                        metrics_grp.attrs[k] = float(v) if isinstance(v, (np.floating, float)) else v


class EMAComparisonEvaluator:
    """
    EMA vs 非 EMA 模型对比评估器
    """
    
    def __init__(self, model_path: str, data_dir: str, output_dir: str = 'ema_comparison_results'):
        self.model_path = model_path
        self.data_dir = os.path.expanduser(data_dir)
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取数据集文件列表
        self.episode_files = sorted([
            f for f in os.listdir(self.data_dir) 
            if f.startswith('episode_') and f.endswith('.hdf5')
        ])
        print(f"Found {len(self.episode_files)} episodes in {self.data_dir}")
        
        # 加载两个模型
        print("\nLoading Non-EMA model...")
        self.policy_regular = OfflinePolicy(model_path, use_ema=False)
        
        print("\nLoading EMA model...")
        self.policy_ema = OfflinePolicy(model_path, use_ema=True)
    
    def compare_single_episode(self, ep_idx: int, verbose: bool = False,
                               num_inference_steps: Optional[int] = None) -> Dict:
        """
        对比单个 episode
        
        Returns:
            包含两个模型预测结果和对比指标的字典
        """
        filepath = os.path.join(self.data_dir, self.episode_files[ep_idx])
        episode = load_carm_episode(filepath)
        
        self.policy_regular.reset()
        self.policy_ema.reset()
        
        pred_regular = []
        pred_ema = []
        gt_actions = []
        
        T = len(episode['qpos_joint'])
        
        # 构建 GT actions (相对于自身 = identity)
        raw_actions = episode['action']
        relative_actions = np.zeros_like(raw_actions)
        for t in range(T):
            relative_actions[t, :6] = raw_actions[t, :6]
            relative_actions[t, 6] = raw_actions[t, 6]
            relative_actions[t, 7:10] = 0.0
            relative_actions[t, 10:14] = np.array([0.0, 0.0, 0.0, 1.0])
            relative_actions[t, 14] = raw_actions[t, 14]
        
        iterator = tqdm(range(T), desc=f"Episode {ep_idx}") if verbose else range(T)
        
        for t in iterator:
            image = episode['images'][t]
            qpos = episode['qpos_joint'][t]
            gt_action = relative_actions[t]
            
            # 两个模型分别推理
            pred_r = self.policy_regular.predict(
                image, qpos, num_steps=num_inference_steps
            )[0]
            pred_e = self.policy_ema.predict(
                image, qpos, num_steps=num_inference_steps
            )[0]
            
            pred_regular.append(pred_r)
            pred_ema.append(pred_e)
            gt_actions.append(gt_action)
        
        pred_regular = np.array(pred_regular)
        pred_ema = np.array(pred_ema)
        gt_actions = np.array(gt_actions)
        
        # 计算指标
        metrics_regular = self._compute_metrics(pred_regular, gt_actions)
        metrics_ema = self._compute_metrics(pred_ema, gt_actions)
        
        # 计算两个模型之间的差异
        pred_diff = np.abs(pred_regular - pred_ema)
        diff_metrics = {
            'mean_diff': np.mean(pred_diff),
            'max_diff': np.max(pred_diff),
            'joint_diff': np.mean(pred_diff[:, :6]),
            'pose_diff': np.mean(pred_diff[:, 7:14]),
        }
        
        return {
            'pred_regular': pred_regular,
            'pred_ema': pred_ema,
            'gt_actions': gt_actions,
            'metrics_regular': metrics_regular,
            'metrics_ema': metrics_ema,
            'diff_metrics': diff_metrics,
        }
    
    def _compute_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict:
        """计算评估指标"""
        joint_pred = pred[:, :6]
        joint_gt = gt[:, :6]
        gripper_joint_pred = pred[:, 6]
        gripper_joint_gt = gt[:, 6]
        pose_pred = pred[:, 7:14]
        pose_gt = gt[:, 7:14]
        gripper_pose_pred = pred[:, 14]
        gripper_pose_gt = gt[:, 14]

        ee_pred = np.concatenate([pred[:, 6:7], pred[:, 7:14]], axis=1)
        ee_gt = np.concatenate([gt[:, 6:7], gt[:, 7:14]], axis=1)
        
        return {
            'joint_mse': np.mean((joint_pred - joint_gt) ** 2),
            'joint_mae': np.mean(np.abs(joint_pred - joint_gt)),
            'gripper_joint_mse': np.mean((gripper_joint_pred - gripper_joint_gt) ** 2),
            'gripper_joint_mae': np.mean(np.abs(gripper_joint_pred - gripper_joint_gt)),
            'gripper_pose_mse': np.mean((gripper_pose_pred - gripper_pose_gt) ** 2),
            'gripper_pose_mae': np.mean(np.abs(gripper_pose_pred - gripper_pose_gt)),
            'pose_mse': np.mean((pose_pred - pose_gt) ** 2),
            'pose_mae': np.mean(np.abs(pose_pred - pose_gt)),
            'ee_mse': np.mean((ee_pred - ee_gt) ** 2),
            'ee_mae': np.mean(np.abs(ee_pred - ee_gt)),
            'total_mse': np.mean((pred - gt) ** 2),
            'total_mae': np.mean(np.abs(pred - gt)),
        }
    
    def plot_comparison(self, result: Dict, ep_idx: int, save: bool = True):
        """绘制 EMA vs 非 EMA 对比图"""
        pred_r = result['pred_regular']
        pred_e = result['pred_ema']
        gt = result['gt_actions']
        T = len(gt)
        time_steps = np.arange(T)
        
        fig, axes = plt.subplots(3, 4, figsize=(22, 12))
        fig.suptitle(f'Episode {ep_idx}: EMA vs Non-EMA Comparison', fontsize=14)
        
        # 关节 1-6
        for i in range(6):
            ax = axes[i // 3, i % 3]
            ax.plot(time_steps, gt[:, i], 'k-', label='GT', alpha=0.5, linewidth=2)
            ax.plot(time_steps, pred_r[:, i], 'b--', label='Non-EMA', alpha=0.7)
            ax.plot(time_steps, pred_e[:, i], 'r--', label='EMA', alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'Joint {i+1} (rad)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Joint {i+1}')
        
        # 误差对比
        ax = axes[2, 0]
        error_r = np.mean(np.abs(pred_r - gt), axis=1)
        error_e = np.mean(np.abs(pred_e - gt), axis=1)
        ax.plot(time_steps, error_r, 'b-', label='Non-EMA', alpha=0.7)
        ax.plot(time_steps, error_e, 'r-', label='EMA', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Step-wise Error Comparison')
        
        # 累积误差
        ax = axes[2, 1]
        cum_error_r = np.cumsum(error_r)
        cum_error_e = np.cumsum(error_e)
        ax.plot(time_steps, cum_error_r, 'b-', label='Non-EMA', alpha=0.7)
        ax.plot(time_steps, cum_error_e, 'r-', label='EMA', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Cumulative Error Comparison')
        
        # 模型间差异
        ax = axes[2, 2]
        pred_diff = np.mean(np.abs(pred_r - pred_e), axis=1)
        ax.plot(time_steps, pred_diff, 'g-', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Mean Abs Difference')
        ax.grid(True, alpha=0.3)
        ax.set_title('EMA vs Non-EMA Difference')

        # 夹爪对比（joint 通道）
        ax = axes[0, 3]
        ax.plot(time_steps, gt[:, 6], 'k-', label='GT', alpha=0.5, linewidth=2)
        ax.plot(time_steps, pred_r[:, 6], 'b--', label='Non-EMA', alpha=0.7)
        ax.plot(time_steps, pred_e[:, 6], 'r--', label='EMA', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Gripper (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Gripper (joint channel)')

        # 夹爪对比（ee 通道）
        ax = axes[1, 3]
        ax.plot(time_steps, gt[:, 14], 'k-', label='GT', alpha=0.5, linewidth=2)
        ax.plot(time_steps, pred_r[:, 14], 'b--', label='Non-EMA', alpha=0.7)
        ax.plot(time_steps, pred_e[:, 14], 'r--', label='EMA', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Gripper (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Gripper (ee channel)')

        # EE MAE 对比
        ax = axes[2, 3]
        ee_r = np.concatenate([pred_r[:, 6:7], pred_r[:, 7:14]], axis=1)
        ee_e = np.concatenate([pred_e[:, 6:7], pred_e[:, 7:14]], axis=1)
        ee_gt = np.concatenate([gt[:, 6:7], gt[:, 7:14]], axis=1)
        ee_err_r = np.mean(np.abs(ee_r - ee_gt), axis=1)
        ee_err_e = np.mean(np.abs(ee_e - ee_gt), axis=1)
        ax.plot(time_steps, ee_err_r, 'b-', label='Non-EMA', alpha=0.7)
        ax.plot(time_steps, ee_err_e, 'r-', label='EMA', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('EE MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('EE MAE Comparison')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'ema_comparison_ep{ep_idx:03d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def run_comparison(self, num_episodes: Optional[int] = None,
                       num_inference_steps: Optional[int] = None,
                       verbose: bool = True):
        """
        运行完整对比评估
        """
        if num_episodes is None:
            num_episodes = min(5, len(self.episode_files))  # 默认测试5个
        
        all_results = []
        
        print(f"\n{'='*60}")
        print(f"EMA vs Non-EMA Comparison: {num_episodes} episodes")
        if num_inference_steps is not None:
            print(f"Inference steps: {num_inference_steps}")
        print(f"{'='*60}\n")
        
        for ep_idx in range(num_episodes):
            print(f"\nComparing episode {ep_idx + 1}/{num_episodes}...")
            result = self.compare_single_episode(
                ep_idx, verbose=verbose,
                num_inference_steps=num_inference_steps
            )
            all_results.append(result)
            
            # 绘制对比图
            self.plot_comparison(result, ep_idx)
            
            # 打印对比结果
            m_r = result['metrics_regular']
            m_e = result['metrics_ema']
            d = result['diff_metrics']
            print(f"  Non-EMA: Joint MAE={m_r['joint_mae']:.4f}, EE MAE={m_r['ee_mae']:.4f}, Total MAE={m_r['total_mae']:.4f}")
            print(f"  EMA:     Joint MAE={m_e['joint_mae']:.4f}, EE MAE={m_e['ee_mae']:.4f}, Total MAE={m_e['total_mae']:.4f}")
            print(f"  Diff:    Mean={d['mean_diff']:.4f}, Max={d['max_diff']:.4f}")
        
        # 汇总统计
        avg_regular = {
            'joint_mae': np.mean([r['metrics_regular']['joint_mae'] for r in all_results]),
            'ee_mae': np.mean([r['metrics_regular']['ee_mae'] for r in all_results]),
            'total_mae': np.mean([r['metrics_regular']['total_mae'] for r in all_results]),
        }
        avg_ema = {
            'joint_mae': np.mean([r['metrics_ema']['joint_mae'] for r in all_results]),
            'ee_mae': np.mean([r['metrics_ema']['ee_mae'] for r in all_results]),
            'total_mae': np.mean([r['metrics_ema']['total_mae'] for r in all_results]),
        }
        avg_diff = np.mean([r['diff_metrics']['mean_diff'] for r in all_results])
        
        # 打印总结
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}")
        print(f"Total Episodes: {num_episodes}")
        print(f"\n[Non-EMA Model]")
        print(f"  Average Joint MAE: {avg_regular['joint_mae']:.4f}")
        print(f"  Average EE MAE:    {avg_regular['ee_mae']:.4f}")
        print(f"  Average Total MAE: {avg_regular['total_mae']:.4f}")
        print(f"\n[EMA Model]")
        print(f"  Average Joint MAE: {avg_ema['joint_mae']:.4f}")
        print(f"  Average EE MAE:    {avg_ema['ee_mae']:.4f}")
        print(f"  Average Total MAE: {avg_ema['total_mae']:.4f}")
        print(f"\n[Comparison]")
        improvement = (avg_regular['total_mae'] - avg_ema['total_mae']) / avg_regular['total_mae'] * 100
        print(f"  EMA Improvement: {improvement:+.2f}%")
        print(f"  Mean Prediction Diff: {avg_diff:.4f}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # 保存结果
        self._save_comparison_results(all_results, avg_regular, avg_ema)
        
        return {
            'avg_regular': avg_regular,
            'avg_ema': avg_ema,
            'improvement': improvement,
        }
    
    def _save_comparison_results(self, all_results: List[Dict], 
                                  avg_regular: Dict, avg_ema: Dict):
        """保存对比结果"""
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_path = os.path.join(self.output_dir, 'comparison_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'avg_regular': convert_to_native(avg_regular),
                'avg_ema': convert_to_native(avg_ema),
                'model_path': self.model_path,
                'data_dir': self.data_dir,
                'num_episodes': len(all_results),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='CARM Offline Evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='~/rl-vla/recorded_data',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='offline_results',
                        help='Output directory for results')
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='Number of episodes to evaluate (default: all)')
    parser.add_argument('--no_individual_plots', action='store_true',
                        help='Skip individual episode plots')
    parser.add_argument('--quiet', action='store_true',
                        help='Less verbose output')
    
    # EMA 相关参数
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA model for inference')
    parser.add_argument('--compare_ema', action='store_true',
                        help='Run EMA vs Non-EMA comparison test')
    
    # 推理步数
    parser.add_argument('--num_inference_steps', type=int, default=None,
                        help='Number of inference steps (default: use model default)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.compare_ema:
        # 运行 EMA vs 非 EMA 对比测试
        comparison_output = args.output_dir.replace('offline_results', 'ema_comparison_results')
        comparator = EMAComparisonEvaluator(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=comparison_output,
        )
        comparator.run_comparison(
            num_episodes=args.num_episodes,
            num_inference_steps=args.num_inference_steps,
            verbose=not args.quiet,
        )
    else:
        # 运行标准评估
        evaluator = OfflineEvaluator(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            use_ema=args.use_ema,
        )
        evaluator.run_evaluation(
            num_episodes=args.num_episodes,
            plot_individual=not args.no_individual_plots,
            verbose=not args.quiet,
            num_inference_steps=args.num_inference_steps,
        )


if __name__ == '__main__':
    main()
