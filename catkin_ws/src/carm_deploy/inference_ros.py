#!/usr/bin/env python3
"""
CARM 机械臂 ROS 策略推理主程序
基于 carm_real/infer_g3_api.py 重构，将 svar 通信替换为 ROS1 原生通信

支持的算法:
    - consistency_flow: Consistency Flow Matching (推荐)
    - flow_matching: Flow Matching Policy
    - diffusion_policy: DDPM-based Diffusion Policy

测试模式:
    --dry_run: 只推理不执行动作（安全测试）
    --slow_mode: 慢速执行（5Hz）
    --log_dir: 保存推理日志

使用方法:
    # 干运行模式（不执行动作）
    rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt --dry_run
    
    # 慢速模式（5Hz）
    rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt --slow_mode
    
    # 正常模式（30Hz）
    rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt
"""

import argparse
import threading
import time
import json
import signal
import numpy as np
import cv2
import rospy
from scipy.spatial.transform import Rotation as R
from einops import rearrange
from typing import Optional, Dict, List, Any

# 本地模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 添加训练代码路径
rl_vla_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(rl_vla_root, 'rlft', 'diffusion_policy'))

# PyTorch
import torch
import torch.nn as nn

# 训练代码中的模块
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.carm_utils import StateEncoder
from diffusion_policy.algorithms.networks import VelocityUNet1D
from diffusion_policy.algorithms import (
    ConsistencyFlowAgent,
    FlowMatchingAgent,
    DiffusionPolicyAgent,
)

# 安全控制和日志
from safety_controller import SafetyController
from inference_logger import InferenceLogger

from env_ros import RealEnvironment
from utils.trajectory_interpolator import VecTF, ActionChunkManager


def pose_to_transform_matrix(position, quaternion):
    """
    将位姿 (xyz + 四元数) 转换为 4x4 变换矩阵
    
    Args:
        position: 平移 [x, y, z]
        quaternion: 四元数 [qx, qy, qz, qw]
        
    Returns:
        4x4 变换矩阵
    """
    rotation = R.from_quat(quaternion).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def apply_relative_transform(relative_pose, current_pose, gripper):
    """
    将相对位姿变换应用到当前位姿，得到目标绝对位姿
    
    计算公式: target_pose = current_pose @ relative_transform
    
    This matches the inference behavior in infer_g3_api.py:
        - Model outputs relative transformations
        - All actions in prediction horizon are relative to observation frame's pose
        - target = current @ relative
    
    Args:
        relative_pose: 模型输出的相对位姿 [x,y,z,qx,qy,qz,qw]
        current_pose: 当前末端位姿 [x,y,z,qx,qy,qz,qw]
        gripper: 夹爪开度
        
    Returns:
        目标绝对位姿 [x,y,z,qx,qy,qz,qw,gripper]
    """
    # relative_pose 是模型输出的相对变换
    # current_pose 是当前末端位姿
    T_relative = pose_to_transform_matrix(relative_pose[:3], relative_pose[3:])
    T_current = pose_to_transform_matrix(current_pose[:3], current_pose[3:])
    
    # target = current @ relative
    T_target = T_current @ T_relative
    
    target_position = T_target[:3, 3]
    target_quat = R.from_matrix(T_target[:3, :3]).as_quat()
    
    return target_position.tolist() + target_quat.tolist() + [gripper]


class PolicyInterface:
    """
    策略模型接口（抽象基类）
    用户需要继承此类并实现 load_model 和 __call__ 方法
    """
    
    def __init__(self, config):
        """
        初始化策略接口
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
        """
        raise NotImplementedError("Subclass must implement load_model()")
    
    def __call__(self, inputs):
        """
        执行推理
        
        Args:
            inputs: 输入字典，包含 'qpos' 和 'image'
            
        Returns:
            输出字典，包含 'a_hat' (动作预测)
        """
        raise NotImplementedError("Subclass must implement __call__()")


class DummyPolicy(PolicyInterface):
    """
    虚拟策略（用于测试）
    返回零动作
    """
    
    def load_model(self, model_path):
        rospy.loginfo(f"DummyPolicy: would load model from {model_path}")
        self.model = True
    
    def __call__(self, inputs):
        # 返回零动作，形状为 [1, horizon, action_dim]
        batch_size = 1
        horizon = 16
        action_dim = 15
        
        # 获取当前 qpos 作为动作
        qpos = inputs['qpos'].cpu().numpy()  # [B, 7]
        
        # 扩展为 horizon 步
        actions = np.zeros((batch_size, horizon, action_dim))
        actions[:, :, :7] = qpos  # 关节位置
        actions[:, :, 7:14] = [0, 0, 0, 0, 0, 0, 1]  # 单位四元数表示零位移
        actions[:, :, 6] = qpos[:, -1]  # gripper
        actions[:, :, 14] = qpos[:, -1]  # gripper
        
        return {'a_hat': torch.from_numpy(actions).float()}


class RealPolicy(PolicyInterface):
    """
    真实策略实现
    加载训练好的 checkpoint 并执行推理
    
    支持的算法:
        - consistency_flow: ConsistencyFlowAgent
        - flow_matching: FlowMatchingAgent
        - diffusion_policy: DiffusionPolicyAgent
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 默认参数（会被 args.json 覆盖）
        self.obs_horizon = config.get('obs_horizon', 2)
        self.pred_horizon = config.get('pred_horizon', 16)
        self.action_dim = config.get('action_dim', 15)  # full mode
        self.state_dim = 7  # 6 joints + 1 gripper
        self.target_image_size = config.get('target_image_size', (128, 128))
        self.visual_feature_dim = config.get('visual_feature_dim', 256)
        self.state_encoder_hidden_dim = config.get('state_encoder_hidden_dim', 128)
        self.state_encoder_out_dim = config.get('state_encoder_out_dim', 256)
        self.use_state_encoder = config.get('use_state_encoder', True)
        self.algorithm = config.get('algorithm', 'consistency_flow')
        
        # 模型组件
        self.visual_encoder = None
        self.state_encoder = None
        self.agent = None
        
        # 观测历史缓冲区
        self.obs_history = {
            'rgb': [],    # 保存最近 obs_horizon 帧图像 (CHW 格式)
            'state': [],  # 保存最近 obs_horizon 帧状态
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded = False
    
    def load_model(self, model_path: str):
        """
        加载模型 checkpoint
        
        Args:
            model_path: checkpoint 文件路径 (例如 runs/exp_name/checkpoints/latest.pt)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        
        checkpoint_dir = os.path.dirname(model_path)
        
        # 1. 加载训练配置
        args_path = os.path.join(checkpoint_dir, "args.json")
        visual_encoder_type = 'plain_conv'  # 默认值
        
        if os.path.exists(args_path):
            rospy.loginfo(f"Loading config from: {args_path}")
            with open(args_path, 'r') as f:
                args = json.load(f)
            
            # 更新配置
            self.obs_horizon = args.get('obs_horizon', self.obs_horizon)
            self.pred_horizon = args.get('pred_horizon', self.pred_horizon)
            self.action_dim = args.get('action_dim', self.action_dim)
            self.visual_feature_dim = args.get('visual_feature_dim', self.visual_feature_dim)
            self.state_encoder_hidden_dim = args.get('state_encoder_hidden_dim', self.state_encoder_hidden_dim)
            self.state_encoder_out_dim = args.get('state_encoder_out_dim', self.state_encoder_out_dim)
            self.use_state_encoder = args.get('use_state_encoder', self.use_state_encoder)
            self.algorithm = args.get('algorithm', self.algorithm)
            
            # 视觉编码器类型
            visual_encoder_type = args.get('visual_encoder_type', 'plain_conv')
            
            # 解析 target_image_size (优先使用 auto_image_size 设置)
            if args.get('auto_image_size', True):
                # 根据编码器类型自动设置图像尺寸
                if visual_encoder_type in ['resnet18', 'resnet34', 'resnet50']:
                    self.target_image_size = (224, 224)
                else:
                    self.target_image_size = (128, 128)
            else:
                target_size = args.get('target_image_size', self.target_image_size)
                if isinstance(target_size, str):
                    # 处理字符串格式 "(128, 128)"
                    target_size = eval(target_size)
                elif isinstance(target_size, list):
                    target_size = tuple(target_size)
                self.target_image_size = target_size
            
            # action_mode 决定 action_dim
            action_mode = args.get('action_mode', 'full')
            if action_mode == 'full':
                self.action_dim = 15
            else:  # ee_only
                self.action_dim = 8
                
            rospy.loginfo(f"Config: algorithm={self.algorithm}, action_dim={self.action_dim}, "
                         f"obs_horizon={self.obs_horizon}, pred_horizon={self.pred_horizon}")
            rospy.loginfo(f"Visual encoder: {visual_encoder_type}, image_size={self.target_image_size}")
        else:
            rospy.logwarn(f"args.json not found, using default config")
        
        # 2. 创建模型
        rospy.loginfo("Creating models...")
        
        # Visual encoder - 根据类型创建不同的编码器
        if visual_encoder_type == 'plain_conv':
            self.visual_encoder = PlainConv(
                in_channels=3,
                out_dim=self.visual_feature_dim,
                pool_feature_map=True,
            ).to(self.device)
        elif visual_encoder_type in ['resnet18', 'resnet34', 'resnet50']:
            # 导入 ResNet 编码器
            from diffusion_policy.resnet_encoder import ResNetEncoder
            self.visual_encoder = ResNetEncoder(
                backbone_name=visual_encoder_type,
                out_dim=self.visual_feature_dim,
                pretrained=False,  # 推理时不需要预训练权重，会从 checkpoint 加载
                freeze_backbone=False,  # 推理时不需要冻结
                freeze_bn=False,  # 推理时使用 eval 模式
            ).to(self.device)
            rospy.loginfo(f"Created ResNetEncoder: {visual_encoder_type}")
        else:
            rospy.logwarn(f"Unknown visual_encoder_type: {visual_encoder_type}, using plain_conv")
            self.visual_encoder = PlainConv(
                in_channels=3,
                out_dim=self.visual_feature_dim,
                pool_feature_map=True,
            ).to(self.device)
        
        # State encoder
        encoded_state_dim = self.state_dim
        if self.use_state_encoder:
            self.state_encoder = StateEncoder(
                state_dim=self.state_dim,
                hidden_dim=self.state_encoder_hidden_dim,
                out_dim=self.state_encoder_out_dim,
            ).to(self.device)
            encoded_state_dim = self.state_encoder_out_dim
        
        # 计算 global conditioning 维度
        global_cond_dim = self.obs_horizon * (self.visual_feature_dim + encoded_state_dim)
        rospy.loginfo(f"global_cond_dim={global_cond_dim} = {self.obs_horizon} * ({self.visual_feature_dim} + {encoded_state_dim})")
        
        # Agent
        self.agent = self._create_agent(global_cond_dim)
        
        # 3. 加载权重
        rospy.loginfo(f"Loading checkpoint from: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device)
        
        # 加载 visual encoder
        if "visual_encoder" in ckpt:
            self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
            rospy.loginfo("Loaded visual_encoder weights")
        else:
            rospy.logwarn("visual_encoder not in checkpoint")
        
        # 加载 state encoder
        if self.state_encoder is not None and "state_encoder" in ckpt:
            self.state_encoder.load_state_dict(ckpt["state_encoder"])
            rospy.loginfo("Loaded state_encoder weights")
        
        # 加载 agent (使用 EMA 版本)
        if "ema_agent" in ckpt:
            self.agent.load_state_dict(ckpt["ema_agent"])
            rospy.loginfo("Loaded ema_agent weights")
        elif "agent" in ckpt:
            self.agent.load_state_dict(ckpt["agent"])
            rospy.loginfo("Loaded agent weights (non-EMA)")
        else:
            raise ValueError("No agent weights in checkpoint")
        
        # 4. 设置为评估模式
        self.visual_encoder.eval()
        if self.state_encoder is not None:
            self.state_encoder.eval()
        self.agent.eval()
        
        self.loaded = True
        rospy.loginfo(f"Model loaded successfully! Algorithm: {self.algorithm}")
    
    def _create_agent(self, global_cond_dim: int) -> nn.Module:
        """根据算法类型创建 agent"""
        
        # UNet 参数（使用训练时的默认值）
        diffusion_step_embed_dim = 64
        unet_dims = [64, 128, 256]
        n_groups = 8
        
        if self.algorithm == "consistency_flow":
            velocity_net = VelocityUNet1D(
                input_dim=self.action_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=tuple(unet_dims),
                n_groups=n_groups,
            )
            return ConsistencyFlowAgent(
                velocity_net=velocity_net,
                action_dim=self.action_dim,
                obs_horizon=self.obs_horizon,
                pred_horizon=self.pred_horizon,
                num_flow_steps=10,  # 默认推理步数
                device=str(self.device),
            ).to(self.device)
            
        elif self.algorithm == "flow_matching":
            velocity_net = VelocityUNet1D(
                input_dim=self.action_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=tuple(unet_dims),
                n_groups=n_groups,
            )
            return FlowMatchingAgent(
                velocity_net=velocity_net,
                action_dim=self.action_dim,
                obs_horizon=self.obs_horizon,
                pred_horizon=self.pred_horizon,
                num_flow_steps=10,
                device=str(self.device),
            ).to(self.device)
            
        elif self.algorithm == "diffusion_policy":
            from diffusion_policy.conditional_unet1d import ConditionalUnet1D
            noise_pred_net = ConditionalUnet1D(
                input_dim=self.action_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=tuple(unet_dims),
                n_groups=n_groups,
            )
            return DiffusionPolicyAgent(
                noise_pred_net=noise_pred_net,
                action_dim=self.action_dim,
                obs_horizon=self.obs_horizon,
                pred_horizon=self.pred_horizon,
                device=str(self.device),
            ).to(self.device)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像: resize + HWC -> CHW
        
        Args:
            image: RGB 图像 [H, W, C]
            
        Returns:
            处理后的图像 [C, H, W]
        """
        # Resize 到目标尺寸
        if self.target_image_size is not None:
            h, w = self.target_image_size
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # HWC -> CHW
        image = rearrange(image, 'h w c -> c h w')
        return image
    
    def _update_obs_history(self, rgb: np.ndarray, state: np.ndarray):
        """
        更新观测历史
        
        Args:
            rgb: 预处理后的图像 [C, H, W]
            state: 状态向量 [state_dim]
        """
        self.obs_history['rgb'].append(rgb)
        self.obs_history['state'].append(state)
        
        # 保持 obs_horizon 长度
        if len(self.obs_history['rgb']) > self.obs_horizon:
            self.obs_history['rgb'].pop(0)
            self.obs_history['state'].pop(0)
        
        # 如果不足，用第一帧填充
        while len(self.obs_history['rgb']) < self.obs_horizon:
            self.obs_history['rgb'].insert(0, self.obs_history['rgb'][0])
            self.obs_history['state'].insert(0, self.obs_history['state'][0])
    
    def _encode_observations(self) -> torch.Tensor:
        """
        编码观测历史为 obs_features
        
        Returns:
            obs_features: [B, obs_horizon, visual_dim + state_dim]
        """
        B, T = 1, self.obs_horizon
        
        # RGB: [B, T, C, H, W]
        rgb_list = [torch.from_numpy(r).float() for r in self.obs_history['rgb']]
        rgb = torch.stack(rgb_list, dim=0).unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        rgb_flat = rgb.view(B * T, *rgb.shape[2:]) / 255.0  # [T, C, H, W]
        
        visual_feat = self.visual_encoder(rgb_flat)  # [T, visual_dim]
        visual_feat = visual_feat.view(B, T, -1)  # [1, T, visual_dim]
        
        # State: [B, T, state_dim]
        state_list = [torch.from_numpy(s).float() for s in self.obs_history['state']]
        state = torch.stack(state_list, dim=0).unsqueeze(0).to(self.device)  # [1, T, state_dim]
        
        if self.state_encoder is not None:
            state_flat = state.view(B * T, -1)  # [T, state_dim]
            state_feat = self.state_encoder(state_flat)  # [T, encoded_state_dim]
            state_feat = state_feat.view(B, T, -1)  # [1, T, encoded_state_dim]
        else:
            state_feat = state
        
        # 拼接 [1, T, visual_dim + state_dim]
        obs_features = torch.cat([visual_feat, state_feat], dim=-1)
        return obs_features
    
    def reset(self):
        """重置观测历史"""
        self.obs_history = {'rgb': [], 'state': []}
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        执行推理
        
        Args:
            inputs: 包含 'qpos' (状态) 和 'image' (图像) 的字典
            
        Returns:
            {'a_hat': [1, pred_horizon, action_dim]}
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # 获取当前观测
        qpos = inputs['qpos'].cpu().numpy().squeeze()  # [7]
        image = inputs['image'].cpu().numpy().squeeze()  # [C, H, W] 或 [1, C, H, W]
        
        # 如果图像是 4D，取第一个
        if image.ndim == 4:
            image = image[0]  # [C, H, W]
        
        # 预处理图像（如果还是 HWC 格式）
        if image.shape[0] != 3:
            # 假设是 HWC 格式，需要转换
            image = self._preprocess_image(image)
        
        # 更新观测历史
        self._update_obs_history(image, qpos)
        
        # 编码观测
        with torch.no_grad():
            obs_features = self._encode_observations()
            
            # 调用 agent.get_action()
            actions = self.agent.get_action_deterministic(obs_features)  # [1, pred_horizon, action_dim]
        
        return {'a_hat': actions}


class InferenceNode:
    """
    ROS 推理节点
    
    支持测试模式:
        - dry_run: 只推理不执行动作（安全测试）
        - slow_mode: 慢速执行（5Hz）
        - normal: 正常速度
    """
    
    def __init__(self, config):
        """
        初始化推理节点
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 测试模式
        self.dry_run = config.get('dry_run', False)
        self.slow_mode = config.get('slow_mode', False)
        
        # 参数
        self.temporal_factor_k = config.get('temporal_factor_k', 0.05)  # 默认 0.05
        self.desire_inference_freq = config.get('desire_inference_freq', 30)  # 默认 30Hz
        self.pos_lookahead_step = config.get('pos_lookahead_step', 1)
        self.pos_lookahead_duration = config.get('pos_lookahead_duration', 0.015)
        self.joint_cmd_mode = config.get('joint_cmd_mode', False)
        
        # 如果是 slow_mode，降低推理频率并调整相关参数
        if self.slow_mode:
            self.desire_inference_freq = 5
            # slow_mode 下使用更小的时间加权因子，因为轨迹更新较慢
            self.temporal_factor_k = config.get('temporal_factor_k', 0.01)
            rospy.logwarn(f"SLOW MODE: Inference frequency set to {self.desire_inference_freq} Hz")
            rospy.logwarn(f"SLOW MODE: temporal_factor_k adjusted to {self.temporal_factor_k}")
        
        if self.dry_run:
            rospy.logwarn("DRY RUN MODE: Actions will NOT be executed on robot!")
        
        # 初始化环境
        rospy.loginfo("Initializing environment...")
        self.env = RealEnvironment(config)
        
        # 初始化策略
        rospy.loginfo("Initializing policy...")
        self.policy = self._create_policy(config)
        
        # 初始化安全控制器
        self.safety_controller = self._create_safety_controller(config)
        
        # 初始化推理日志记录器
        self.logger = self._create_logger(config)
        self.episode_started = False
        
        # 动作管理器
        self.action_manager = ActionChunkManager(temporal_factor_k=self.temporal_factor_k)
        self.lock_tfs = threading.Lock()
        
        # 控制变量
        self.running = True
        self.latest_obs = None
        self.pos_lookahead_step_start_idx = 0
        self.step_count = 0
        self.last_action = None
        
        # 启动推理线程
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        rospy.loginfo("InferenceNode initialized")
    
    def _create_policy(self, config):
        """
        创建策略实例
        
        根据 pretrain 路径决定使用 RealPolicy 还是 DummyPolicy
        """
        pretrain_path = config.get('pretrain', '')
        
        if pretrain_path and os.path.exists(pretrain_path):
            # 使用真实策略
            rospy.loginfo(f"Loading real policy from: {pretrain_path}")
            policy = RealPolicy(config)
            policy.load_model(pretrain_path)
            return policy
        else:
            # 使用虚拟策略（用于测试）
            rospy.logwarn("No valid pretrain model specified, using dummy policy")
            policy = DummyPolicy(config)
            return policy
    
    def _create_safety_controller(self, config):
        """
        创建安全控制器
        
        优先从 dataset_info.json 加载配置
        """
        safety_config_path = config.get('safety_config', '')
        data_dir = config.get('data_dir', '')
        
        if safety_config_path and os.path.exists(safety_config_path):
            rospy.loginfo(f"Loading safety config from: {safety_config_path}")
            return SafetyController.from_config(safety_config_path)
        elif data_dir and os.path.exists(os.path.join(data_dir, 'dataset_info.json')):
            rospy.loginfo(f"Creating safety controller from dataset stats: {data_dir}")
            return SafetyController.from_dataset_stats(data_dir, margin=0.1)
        else:
            # 使用默认参数
            rospy.logwarn("No safety config or dataset stats found, using default safety limits")
            return SafetyController()
    
    def _create_logger(self, config):
        """
        创建推理日志记录器
        """
        log_dir = config.get('log_dir', '')
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            return InferenceLogger(log_dir=log_dir)
        else:
            # 默认日志目录
            default_log_dir = os.path.expanduser('~/rl-vla/inference_logs')
            os.makedirs(default_log_dir, exist_ok=True)
            return InferenceLogger(log_dir=default_log_dir)
    
    def _preprocess_image(self, image: np.ndarray, target_size=(128, 128)) -> np.ndarray:
        """
        预处理单张图像: resize
        
        Args:
            image: RGB 图像 [H, W, C]
            target_size: 目标尺寸 (H, W)
            
        Returns:
            处理后的图像 [H, W, C]
        """
        h, w = target_size
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def _normalize_images(self, obs, target_size=(128, 128)):
        """
        归一化图像（对齐训练代码）
        
        Args:
            obs: 观测字典
            target_size: 目标图像尺寸 (H, W)
            
        Returns:
            torch.Tensor: 预处理后的图像 [C, H, W] (未归一化，RealPolicy 内部会归一化)
        """
        # 只使用第一个相机
        image = obs["images"][0]  # [H, W, C] RGB 格式
        
        # Resize 到目标尺寸
        image = self._preprocess_image(image, target_size)
        
        # HWC -> CHW
        image = rearrange(image, 'h w c -> c h w')
        
        return image
    
    def _inference_loop(self):
        """推理线程主循环"""
        rospy.loginfo("Inference thread started")
        
        desire_period = 1.0 / self.desire_inference_freq
        
        with torch.inference_mode():
            while self.running and not rospy.is_shutdown():
                # 获取观测
                self.latest_obs = self.env.get_observation()
                if self.latest_obs is None:
                    time.sleep(0.5)
                    rospy.loginfo_throttle(5.0, "Waiting for observation...")
                    continue
                
                # 启动 episode（如果尚未启动）
                if not self.episode_started:
                    self.logger.start_episode()
                    self.episode_started = True
                    rospy.loginfo("Episode started, logging enabled")
                
                last_start = time.time()
                
                try:
                    # 准备输入
                    qpos_joint = np.array(self.latest_obs['qpos_joint'])  # [7]
                    qpos_end = np.array(self.latest_obs['qpos_end']).tolist()  # [8]
                    
                    # 状态: qpos_joint 已经是 7D (6 joints + 1 gripper)
                    qpos = torch.from_numpy(qpos_joint).float().cuda().unsqueeze(0)  # [1, 7]
                    
                    # 图像预处理: resize + HWC->CHW
                    # 获取目标尺寸（从 policy 获取，如果是 RealPolicy）
                    target_size = (128, 128)
                    if hasattr(self.policy, 'target_image_size') and self.policy.target_image_size:
                        target_size = self.policy.target_image_size
                    
                    curr_image = self._normalize_images(self.latest_obs, target_size)  # [C, H, W]
                    
                    # 转换为 torch tensor
                    curr_image = torch.from_numpy(curr_image).float().cuda()  # [C, H, W]
                    
                    # 推理计时
                    inference_start = time.time()
                    ret = self.policy({"qpos": qpos, "image": curr_image})
                    inference_time = time.time() - inference_start
                    
                    all_actions = ret["a_hat"].squeeze(0).cpu().numpy()  # [pred_horizon, action_dim]
                    
                    # 取第一个动作进行安全检查
                    first_action = all_actions[0]  # [action_dim]
                    
                    # 安全检查
                    safety_events = []
                    if self.joint_cmd_mode:
                        # 关节模式：检查关节限位
                        joint_action = first_action[:7]  # [6 joints + 1 gripper]
                        is_safe, clipped_action, events = self.safety_controller.check_and_clip(
                            joint_action[:6],  # 只检查关节
                            self.last_action[:6] if self.last_action is not None else None
                        )
                        if events:
                            safety_events.extend(events)
                            for event in events:
                                rospy.logwarn(f"Safety event: {event}")
                    else:
                        # 末端位姿模式：检查工作空间
                        if first_action.shape[0] >= 15:
                            end_pose = first_action[7:14]  # [7] 相对位姿
                            # 检查相对位移是否过大
                            max_trans = 0.1  # 10cm
                            max_rot = 0.5    # ~30 degrees
                            trans_norm = np.linalg.norm(end_pose[:3])
                            if trans_norm > max_trans:
                                safety_events.append(f"Large translation: {trans_norm:.3f}m > {max_trans}m")
                                rospy.logwarn(f"Safety: Large translation {trans_norm:.3f}m")
                    
                    # 记录当前动作
                    self.last_action = first_action
                    
                    # 记录日志
                    self.logger.log_step(
                        timestamp=time.time(),
                        obs=self.latest_obs,  # 包含 images, qpos_joint, qpos_end
                        raw_action=first_action,
                        inference_time=inference_time,
                        safety_clipped=len(safety_events) > 0,
                        safety_warnings=safety_events if safety_events else None,
                    )
                    
                    # 如果是 dry_run 模式，跳过动作执行
                    if self.dry_run:
                        self.step_count += 1
                        rospy.loginfo_throttle(1.0, 
                            f"[DRY RUN] Step {self.step_count}, Inference: {inference_time:.4f}s, "
                            f"Actions: {all_actions.shape}, Safety events: {len(safety_events)}")
                        
                        wait_tm = desire_period - (time.time() - last_start)
                        if wait_tm > 0:
                            time.sleep(wait_tm)
                        continue
                    
                    # 转换动作空间 (full mode: 15D)
                    # all_actions: [joint(6), gripper(1), relative_end_pose(7), gripper(1)]
                    if not self.joint_cmd_mode:
                        all_endactions = []
                        for i in range(all_actions.shape[0]):
                            # 取 index 7 开始的 8D: relative_end_pose(7) + gripper(1)
                            relative_pose = all_actions[i][7:14]  # [7] 相对位姿
                            # 取 index 6: 第一个 gripper
                            grip = all_actions[i][6]
                            # 将相对位姿变换应用到当前位姿，得到目标绝对位姿
                            target_pose = apply_relative_transform(relative_pose, qpos_end[:7], grip)
                            all_endactions.append(target_pose)
                        all_actions = np.array(all_endactions)
                    else:
                        # joint mode: 取前 7D (6 joints + 1 gripper)
                        all_actions = all_actions[:, :7]
                    
                    # 创建轨迹并添加到管理器
                    stamp = self.latest_obs["stamp"]
                    tf = VecTF({})
                    
                    # 使用固定的动作执行间隔 (0.033s ≈ 30Hz)，而不是推理周期
                    # 这样即使在 slow_mode 下，动作轨迹也能保持合理的时间分布
                    action_interval = 1.0 / 30.0  # 30Hz 的动作执行频率
                    
                    self.pos_lookahead_step_start_idx += 1
                    for i in range(len(all_actions)):
                        if self.pos_lookahead_step == 1:
                            tf.append(stamp + i * action_interval, all_actions[i].tolist())
                        else:
                            if self.pos_lookahead_step_start_idx % self.pos_lookahead_step == 0:
                                tf.append(stamp + i * action_interval, all_actions[i].tolist())
                            else:
                                tf.append(stamp + i * self.pos_lookahead_duration, all_actions[i].tolist())
                    
                    with self.lock_tfs:
                        self.action_manager.add_trajectory(tf)
                    
                    self.step_count += 1
                    rospy.loginfo_throttle(1.0, 
                        f"Step {self.step_count}, Inference: {inference_time:.4f}s, "
                        f"Actions: {all_actions.shape}")
                    
                except Exception as e:
                    import traceback
                    rospy.logerr(f"Error in inference: {e}")
                    rospy.logerr(traceback.format_exc())
                
                # 等待下一个周期
                wait_tm = desire_period - (time.time() - last_start)
                if wait_tm > 0:
                    time.sleep(wait_tm)
    
    def control_loop(self):
        """控制主循环"""
        rospy.loginfo("Control loop started")
        
        # 控制频率: slow_mode 下稍微降低以减少 CPU 负载
        control_period = 0.005 if not self.slow_mode else 0.01  # 200Hz / 100Hz
        
        while self.running and not rospy.is_shutdown():
            # 获取融合后的动作
            tm = time.time()
            
            with self.lock_tfs:
                action = self.action_manager.get_fused_action(tm)
            
            if action is None:
                time.sleep(0.02)
                continue
            
            # 执行控制
            if self.joint_cmd_mode:
                rospy.logdebug("Joint control")
                self.env.joint_control_nostep(action)
            else:
                rospy.logdebug("End pose control")
                self.env.end_control_nostep(action)
            
            time.sleep(control_period)
    
    def shutdown(self):
        """关闭节点"""
        # 防止重复调用
        if hasattr(self, '_shutdown_called') and self._shutdown_called:
            return
        self._shutdown_called = True
        
        rospy.loginfo("Shutting down InferenceNode...")
        self.running = False
        
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        
        # 结束并保存日志
        if self.episode_started:
            log_path = self.logger.end_episode()
            if log_path:
                rospy.loginfo(f"Inference log saved to: {log_path}")
        
        self.env.shutdown()
        rospy.loginfo("InferenceNode shutdown complete")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARM Robot Policy Inference (ROS)')
    
    # 机械臂参数
    parser.add_argument('--robot_ip', type=str, default='10.42.0.101',
                        help='Robot IP address')
    parser.add_argument('--robot_mode', type=int, default=1,
                        help='Control mode (0=IDLE, 1=POSITION, 2=MIT, 3=DRAG)')
    parser.add_argument('--robot_tau', type=float, default=10,
                        help='Gripper torque')
    
    # 初始位置 (从实际机械臂读取 2026-01-13)
    parser.add_argument('--arm_init_pose', type=float, nargs=7,
                        default=[0.2475, 0.0014, 0.3251, 0.9996, -0.0034, 0.0255, -0.0074],
                        help='Initial end effector pose [x,y,z,qx,qy,qz,qw]')
    parser.add_argument('--arm_init_gripper', type=float, default=0.078,
                        help='Initial gripper position')
    
    # 相机参数
    parser.add_argument('--camera_topics', type=str,
                        default='/camera/color/image_raw',
                        help='Camera topic(s), comma separated')
    parser.add_argument('--sync_slop', type=float, default=0.1,
                        help='Image sync tolerance in seconds')
    
    # 策略参数
    parser.add_argument('--pretrain', type=str, default='',
                        help='Path to pretrained model checkpoint (e.g., runs/exp/checkpoints/latest.pt)')
    parser.add_argument('--algorithm', type=str, default='consistency_flow',
                        choices=['consistency_flow', 'flow_matching', 'diffusion_policy'],
                        help='Algorithm type (auto-detected from args.json if available)')
    parser.add_argument('--desire_inference_freq', type=float, default=30,
                        help='Desired inference frequency')
    parser.add_argument('--temporal_factor_k', type=float, default=0.05,
                        help='Temporal factor for action fusion')
    
    # 控制参数
    parser.add_argument('--pos_lookahead_step', type=int, default=1,
                        help='Position lookahead step')
    parser.add_argument('--pos_lookahead_duration', type=float, default=0.015,
                        help='Position lookahead duration')
    parser.add_argument('--joint_cmd_mode', action='store_true',
                        help='Use joint command mode instead of end-effector pose')
    parser.add_argument('--not_origin', action='store_true',
                        help='Skip moving to initial pose on startup')
    
    # 测试模式参数
    parser.add_argument('--dry_run', action='store_true',
                        help='Dry run mode: inference only, no action execution (safest)')
    parser.add_argument('--slow_mode', action='store_true',
                        help='Slow mode: run at 5Hz instead of 30Hz (safer)')
    
    # 安全和确认参数
    parser.add_argument('--no_confirm', action='store_true',
                        help='Skip user confirmation before initializing arm position')
    parser.add_argument('--no_return_home', action='store_true',
                        help='Do not return to any position on exit (robot stays in current position)')
    parser.add_argument('--return_to_init', action='store_true',
                        help='Return to init pose on exit instead of zero position (default: return to zero)')
    parser.add_argument('--init_speed', type=float, default=3.0,
                        help='Speed level for initialization movement (0-10, default: 3.0 = slow)')
    
    # 安全控制参数
    parser.add_argument('--safety_config', type=str, default='',
                        help='Path to safety config JSON file')
    parser.add_argument('--data_dir', type=str, default='',
                        help='Data directory for auto-loading safety limits from dataset_info.json')
    
    # 日志参数
    parser.add_argument('--log_dir', type=str, default='',
                        help='Directory to save inference logs (default: ~/rl-vla/inference_logs)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save images in inference log (increases file size)')
    
    # 可视化
    parser.add_argument('--vis', action='store_true',
                        help='Visualize images in OpenCV window')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 初始化 ROS 节点
    rospy.init_node('carm_inference', anonymous=True)
    
    # 解析参数
    args = parse_args()
    
    # 转换为配置字典
    config = vars(args)
    
    # 处理相机话题
    if isinstance(config['camera_topics'], str):
        config['camera_topics'] = config['camera_topics'].split(',')
    
    # 传递确认和退出参数
    config['skip_init_confirm'] = config.get('no_confirm', False)
    config['no_return_home'] = config.get('no_return_home', False)
    config['return_to_zero'] = not config.get('return_to_init', False)  # 默认回零位
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("CARM Policy Inference Node")
    rospy.loginfo("=" * 60)
    rospy.loginfo(f"Robot IP: {config['robot_ip']}")
    rospy.loginfo(f"Camera topics: {config['camera_topics']}")
    rospy.loginfo(f"Pretrain: {config['pretrain']}")
    rospy.loginfo(f"Joint cmd mode: {config['joint_cmd_mode']}")
    rospy.loginfo("-" * 60)
    rospy.loginfo("Test Mode Configuration:")
    rospy.loginfo(f"  dry_run: {config['dry_run']} (no action execution)")
    rospy.loginfo(f"  slow_mode: {config['slow_mode']} (5Hz inference)")
    rospy.loginfo(f"  no_confirm: {config['skip_init_confirm']} (skip init confirmation)")
    rospy.loginfo(f"  no_return_home: {config['no_return_home']} (skip return on exit)")
    rospy.loginfo(f"  return_to_zero: {config['return_to_zero']} (return to zero position on exit)")
    rospy.loginfo(f"  log_dir: {config['log_dir'] or '~/rl-vla/inference_logs'}")
    rospy.loginfo("=" * 60)
    
    # 安全警告
    if not config['dry_run'] and not config['slow_mode']:
        rospy.logwarn("=" * 60)
        rospy.logwarn("RUNNING IN NORMAL MODE - ROBOT WILL MOVE AT FULL SPEED!")
        rospy.logwarn("Ensure the workspace is clear and E-stop is ready!")
        rospy.logwarn("=" * 60)
        rospy.sleep(2.0)  # 给用户时间阅读警告
    
    # 创建推理节点
    node = InferenceNode(config)
    
    # 全局变量用于信号处理
    shutdown_in_progress = False
    
    def signal_handler(signum, frame):
        """处理 Ctrl+C 信号，确保安全退出"""
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            rospy.logwarn("Force exit requested, exiting immediately...")
            sys.exit(1)
        shutdown_in_progress = True
        rospy.loginfo("\nReceived shutdown signal, cleaning up...")
        node.shutdown()
        rospy.signal_shutdown("User interrupted")
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 注册 ROS 关闭回调
    rospy.on_shutdown(node.shutdown)
    
    try:
        # 运行控制循环
        node.control_loop()
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
    finally:
        if not shutdown_in_progress:
            node.shutdown()


if __name__ == '__main__':
    main()
