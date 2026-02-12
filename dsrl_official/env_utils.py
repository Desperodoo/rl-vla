"""
DSRL Official Environment Utils - 从官方 ajwagen/dsrl 移植

包含:
- ShortCutFlowEnvWrapper: ShortCut Flow 环境包装器 (VecEnv 版本)
- ManiSkillGPUFlowEnvWrapper: ManiSkill3 GPU 并行环境包装器
- ActionChunkWrapper: 动作分块包装器

用于 DSRL-SAC 模式：将噪声空间包装为动作空间。

Reference: https://github.com/ajwagen/dsrl/blob/main/env_utils.py
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium
from gymnasium import spaces
from typing import Optional, Any, Dict, Tuple, List
import sys
from pathlib import Path

# 尝试导入 VecEnv
try:
    from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv, VecFrameStack
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    VecEnvWrapper = object  # Fallback
    VecFrameStack = None

# 添加路径
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))


class ShortCutFlowEnvWrapper(VecEnvWrapper if HAS_SB3 else object):
    """ShortCut Flow 环境包装器 (VecEnv 版本)。
    
    对应官方 DiffusionPolicyEnvWrapper，将噪声空间包装为动作空间。
    SAC 输出的"动作"实际上是噪声 w ∈ [-action_magnitude, +action_magnitude]^(act_steps * action_dim)，
    环境内部通过 ShortCut Flow 将噪声解码为真实动作。
    
    Visual Encoder 集成:
        - 接收 visual_encoder 参数
        - 在 _encode_obs() 中处理 RGB 图像编码
        - 内置观察历史管理 (obs_horizon)
    
    官方实现:
        class DiffusionPolicyEnvWrapper(VecEnvWrapper):
            def step_async(self, actions):
                actions = actions.view(-1, action_horizon, action_dim)
                diffused_actions = self.base_policy(self.obs, actions)
                self.venv.step_async(diffused_actions)
    
    Args:
        env: 向量化环境 (VecEnv)
        base_policy: ShortCutFlowWrapper
        visual_encoder: 视觉编码器 (PlainConv)，可选
        action_magnitude: 噪声范围 [-mag, +mag]
        act_steps: 动作执行步数 (action horizon)
        action_dim: 动作维度
        state_dim: 状态维度 (不含视觉特征)
        visual_feature_dim: 视觉特征维度 (默认 256)
        obs_horizon: 观察历史长度 (默认 2)
        include_rgb: 是否包含 RGB 观察
        device: 设备
    """
    
    def __init__(
        self,
        env,
        base_policy,
        visual_encoder: Optional[nn.Module] = None,
        action_magnitude: float = 1.5,
        act_steps: int = 8,
        action_dim: int = 7,
        state_dim: int = 25,
        visual_feature_dim: int = 256,
        obs_horizon: int = 2,
        include_rgb: bool = True,
        device: str = "cuda",
    ):
        if HAS_SB3:
            super().__init__(env)
        else:
            # Fallback: 手动设置 VecEnv 必要属性
            self.venv = env
            self.num_envs = getattr(env, 'num_envs', 1)
            self.metadata = getattr(env, 'metadata', {})
        
        self.action_horizon = act_steps
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.visual_feature_dim = visual_feature_dim
        self.obs_horizon = obs_horizon
        self.include_rgb = include_rgb
        self.device = device
        self.base_policy = base_policy
        self.action_magnitude = action_magnitude
        
        # Visual encoder
        self.visual_encoder = visual_encoder
        if self.visual_encoder is not None:
            self.visual_encoder = self.visual_encoder.to(device)
            self.visual_encoder.eval()
        
        # 计算编码后的观察维度
        single_obs_dim = state_dim
        if include_rgb and visual_encoder is not None:
            single_obs_dim += visual_feature_dim
        
        self.single_obs_dim = single_obs_dim
        self.obs_dim = obs_horizon * single_obs_dim  # 用于 base_policy
        
        # 噪声空间作为动作空间
        self.action_space = spaces.Box(
            low=-action_magnitude * np.ones(action_dim * act_steps),
            high=action_magnitude * np.ones(action_dim * act_steps),
            dtype=np.float32
        )
        
        # 观察空间 - 使用编码后的特征维度
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(self.obs_dim),
            high=np.inf * np.ones(self.obs_dim),
            dtype=np.float32
        )
        
        self.env = env
        self.obs = None  # 缓存当前编码后的观察 (用于 base_policy)
        
        # 观察历史缓冲
        self._obs_history = None
    
    def step_async(self, actions: np.ndarray):
        """异步执行步骤 - 将噪声转换为真实动作。"""
        # 转换为张量
        actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        
        # 重塑为 (n_envs, act_steps, action_dim)
        actions = actions.view(-1, self.action_horizon, self.action_dim)
        
        # 通过 base_policy 解码为真实动作
        with torch.no_grad():
            diffused_actions = self.base_policy(self.obs, actions, return_numpy=True)
        
        # 存储动作序列，在 step_wait 中逐步执行
        self._pending_actions = diffused_actions
        self._pending_action_idx = 0
        
        # 执行第一步
        self.venv.step_async(self._pending_actions[:, 0, :])
    
    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """等待步骤完成并执行剩余动作 (action chunking)。"""
        # 执行第一步
        obs, rewards, dones, infos = self.venv.step_wait()
        total_rewards = rewards.copy()
        
        # 执行剩余动作
        for i in range(1, self.action_horizon):
            if np.any(dones):
                break
            self.venv.step_async(self._pending_actions[:, i, :])
            obs, rewards, step_dones, step_infos = self.venv.step_wait()
            total_rewards += rewards
            dones = np.logical_or(dones, step_dones)
            
            # 更新 infos
            for j, (info, step_info) in enumerate(zip(infos, step_infos)):
                if step_dones[j] and 'terminal_observation' in step_info:
                    info['terminal_observation'] = step_info['terminal_observation']
        
        # 更新缓存的观察
        encoded_obs = self._encode_and_update_history(obs)
        
        return encoded_obs, total_rewards, dones, infos
    
    def reset(self, **kwargs):
        """重置环境。"""
        result = self.venv.reset(**kwargs)
        
        # ManiSkill3 reset 返回 (obs, info)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = [{} for _ in range(self.num_envs)]
        
        # 初始化观察历史
        self._init_obs_history()
        
        # 编码并更新历史
        encoded_obs = self._encode_and_update_history(obs)
        
        return encoded_obs
    
    def _init_obs_history(self):
        """初始化观察历史缓冲。"""
        self._obs_history = torch.zeros(
            self.num_envs, self.obs_horizon, self.single_obs_dim,
            device=self.device, dtype=torch.float32
        )
    
    def _encode_single_obs(self, obs) -> torch.Tensor:
        """编码单帧观察。"""
        features_list = []
        
        if isinstance(obs, dict):
            # 视觉特征
            if self.include_rgb and self.visual_encoder is not None and 'rgb' in obs:
                rgb = obs['rgb']
                if isinstance(rgb, np.ndarray):
                    rgb = torch.from_numpy(rgb).to(self.device)
                else:
                    rgb = rgb.to(self.device)
                
                # (n_envs, H, W, C) -> (n_envs, C, H, W)
                if rgb.dim() == 4 and rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    rgb = rgb.permute(0, 3, 1, 2)
                
                rgb = rgb.float()
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                
                with torch.no_grad():
                    visual_feat = self.visual_encoder(rgb)
                features_list.append(visual_feat)
            
            # 状态特征
            state = obs.get('state', obs.get('agent', None))
            if state is not None:
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).to(self.device).float()
                else:
                    state = state.to(self.device).float()
                features_list.append(state)
            
            if len(features_list) == 0:
                return torch.zeros(self.num_envs, self.single_obs_dim, device=self.device)
            
            return torch.cat(features_list, dim=-1)
        
        # Array/tensor 格式
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device).float()
        else:
            obs = obs.to(self.device).float()
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        return obs
    
    def _encode_and_update_history(self, obs) -> np.ndarray:
        """编码观察并更新历史。"""
        encoded = self._encode_single_obs(obs)
        
        if self._obs_history is None:
            self._init_obs_history()
        
        # Roll and update
        self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
        self._obs_history[:, -1, :] = encoded
        
        # Flatten for base_policy
        self.obs = self._obs_history.reshape(self.num_envs, -1)
        
        return self.obs.cpu().numpy()
    
    def seed(self, seed: Optional[int] = None):
        """设置随机种子。"""
        if hasattr(self.venv, 'seed'):
            self.venv.seed(seed)


class ManiSkillGPUFlowEnvWrapper(gymnasium.Wrapper):
    """ManiSkill3 GPU 并行环境 ShortCut Flow 包装器。
    
    用于 ManiSkill3 physx_cuda 后端，直接使用 GPU 张量。
    
    特点:
        - 直接操作 GPU 张量
        - 集成 visual_encoder
        - 内置观察历史管理
        - Action chunking 内置
        - 提供 SB3 VecEnv 兼容接口
    """
    
    def __init__(
        self,
        env,
        base_policy,
        visual_encoder: Optional[nn.Module] = None,
        action_magnitude: float = 1.5,
        act_steps: int = 8,
        action_dim: int = 7,
        state_dim: int = 25,
        visual_feature_dim: int = 256,
        obs_horizon: int = 2,
        include_rgb: bool = True,
        device: str = "cuda",
    ):
        super().__init__(env)
        
        self.base_policy = base_policy
        self.visual_encoder = visual_encoder
        self.action_magnitude = action_magnitude
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.visual_feature_dim = visual_feature_dim
        self.obs_horizon = obs_horizon
        self.include_rgb = include_rgb
        self.device = device
        
        if self.visual_encoder is not None:
            self.visual_encoder = self.visual_encoder.to(device)
            self.visual_encoder.eval()
        
        # 计算维度
        self.single_obs_dim = state_dim
        if include_rgb and visual_encoder is not None:
            self.single_obs_dim += visual_feature_dim
        self.obs_dim = obs_horizon * self.single_obs_dim
        
        self.num_envs = getattr(env, 'num_envs', 1)
        
        # 噪声空间作为动作空间
        self.action_space = spaces.Box(
            low=-action_magnitude * np.ones(action_dim * act_steps),
            high=action_magnitude * np.ones(action_dim * act_steps),
            dtype=np.float32
        )
        
        # 观察空间 (编码后)
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(self.obs_dim),
            high=np.inf * np.ones(self.obs_dim),
            dtype=np.float32
        )
        
        self._obs_history = None
    
    def step(self, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """执行一步 (GPU 张量输入输出)。
        
        base_policy 会在 pred_horizon 上进行 flow 推理，
        并从 obs_horizon-1 开始返回 act_steps 步动作（通过 act_steps 参数）。
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device).float()
        else:
            action = action.to(self.device).float()
        
        action = action.view(-1, self.act_steps, self.action_dim)
        obs_cond = self._get_obs_cond()
        
        with torch.no_grad():
            # base_policy 会 pad 噪声到 pred_horizon，进行 flow 推理，
            # 然后从 obs_horizon-1 开始返回 act_steps 步动作
            real_actions = self.base_policy(
                obs_cond, action, return_numpy=False, act_steps=self.act_steps
            )
        
        # Action chunking
        total_reward = torch.zeros(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info = {}
        step_info = {}
        
        for i in range(self.act_steps):
            step_action = real_actions[:, i, :]
            obs, reward, term, trunc, step_info = self.env.step(step_action)
            
            total_reward += reward
            terminated = terminated | term
            truncated = truncated | trunc
            
            # 保留最后一步的 info (包含 final_info)
            if term.any() or trunc.any():
                info = step_info
                break
        
        # 如果循环正常结束，保留最后一步的 info
        if not info:
            info = step_info
        
        encoded_obs = self._encode_and_update_history(obs)
        
        return encoded_obs, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """重置环境。"""
        obs, info = self.env.reset(**kwargs)
        self._init_obs_history()
        encoded_obs = self._encode_and_update_history(obs)
        return encoded_obs, info
    
    def _init_obs_history(self):
        """初始化观察历史。"""
        self._obs_history = torch.zeros(
            self.num_envs, self.obs_horizon, self.single_obs_dim,
            device=self.device, dtype=torch.float32
        )
    
    def _encode_obs_with_history(self, obs) -> torch.Tensor:
        """编码带有 obs_horizon 的观察。
        
        ManiSkill3 环境返回的观察已经包含 obs_horizon 维度:
            - rgb: (B, T, H, W, C)
            - state: (B, T, state_dim)
        
        Returns:
            obs_cond: (B, T * (visual_dim + state_dim))
        """
        features_list = []
        
        if isinstance(obs, dict):
            # 处理 RGB 图像
            if self.include_rgb and self.visual_encoder is not None and 'rgb' in obs:
                rgb = obs['rgb']
                if isinstance(rgb, np.ndarray):
                    rgb = torch.from_numpy(rgb).to(self.device)
                else:
                    rgb = rgb.to(self.device)
                
                B = rgb.shape[0]
                T = rgb.shape[1] if rgb.dim() == 5 else 1
                
                # (B, T, H, W, C) -> (B*T, C, H, W)
                if rgb.dim() == 5:
                    rgb = rgb.reshape(B * T, *rgb.shape[2:])  # (B*T, H, W, C)
                
                if rgb.dim() == 4 and rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    rgb = rgb.permute(0, 3, 1, 2)  # (B*T, C, H, W)
                
                rgb = rgb.float()
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                
                with torch.no_grad():
                    visual_feat = self.visual_encoder(rgb)  # (B*T, visual_dim)
                
                # Reshape to (B, T, visual_dim)
                visual_feat = visual_feat.reshape(B, T, -1)
                features_list.append(visual_feat)
            
            # 处理状态
            state = obs.get('state', obs.get('agent', None))
            if state is not None:
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).to(self.device).float()
                else:
                    state = state.to(self.device).float()
                
                # state 已经是 (B, T, state_dim) 格式
                if state.dim() == 2:
                    state = state.unsqueeze(1)  # (B, 1, state_dim)
                
                features_list.append(state)
            
            if len(features_list) == 0:
                B = self.num_envs
                T = self.obs_horizon
                return torch.zeros(B, T * self.single_obs_dim, device=self.device)
            
            # Concatenate features: (B, T, visual_dim + state_dim)
            combined = torch.cat(features_list, dim=-1)
            
            # Flatten: (B, T * (visual_dim + state_dim))
            return combined.reshape(combined.shape[0], -1)
        
        # Fallback for array/tensor input
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device).float()
        else:
            obs = obs.to(self.device).float()
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        return obs
    
    def _encode_single_obs(self, obs) -> torch.Tensor:
        """编码单帧观察 (兼容性方法)。"""
        return self._encode_obs_with_history(obs)
    
    def _encode_and_update_history(self, obs) -> torch.Tensor:
        """编码观察并缓存结果。"""
        encoded = self._encode_obs_with_history(obs)
        self._cached_obs_cond = encoded  # 缓存用于 base_policy
        return encoded
    
    def _get_obs_cond(self) -> torch.Tensor:
        """获取当前编码的观察条件。"""
        if hasattr(self, '_cached_obs_cond') and self._cached_obs_cond is not None:
            return self._cached_obs_cond
        # Fallback
        if self._obs_history is None:
            self._init_obs_history()
        return self._obs_history.reshape(self.num_envs, -1)
    
    # === SB3 VecEnv 兼容接口 ===
    
    def step_async(self, actions):
        """SB3 兼容: 异步步骤。"""
        self._pending_actions = actions
    
    def step_wait(self):
        """SB3 兼容: 等待步骤完成。"""
        obs, reward, terminated, truncated, info = self.step(self._pending_actions)
        
        obs_np = obs.cpu().numpy()
        reward_np = reward.cpu().numpy()
        dones_np = (terminated | truncated).cpu().numpy()
        
        infos = []
        for i in range(self.num_envs):
            info_i = {}
            if dones_np[i]:
                info_i['terminal_observation'] = obs_np[i]
            infos.append(info_i)
        
        return obs_np, reward_np, dones_np, infos


class ActionChunkWrapper(gymnasium.Wrapper):
    """动作分块包装器。"""
    
    def __init__(
        self,
        env,
        act_steps: int = 8,
        max_episode_steps: int = 100,
    ):
        super().__init__(env)
        
        self.act_steps = act_steps
        self.max_episode_steps = max_episode_steps
        self.count = 0
        
        base_action_space = env.action_space
        self.action_space = spaces.Box(
            low=np.tile(base_action_space.low, act_steps),
            high=np.tile(base_action_space.high, act_steps),
            dtype=np.float32
        )
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """执行多步动作。"""
        if len(action.shape) == 1:
            action = action.reshape(self.act_steps, -1)
        
        total_reward = 0
        obs = None
        terminated = False
        truncated = False
        info = {}
        
        for i in range(action.shape[0]):
            self.count += 1
            obs, reward, terminated, truncated, info = self.env.step(action[i])
            total_reward += reward
            
            if terminated or truncated:
                break
        
        if self.count >= self.max_episode_steps:
            truncated = True
        
        if terminated or truncated:
            info['terminal_observation'] = obs
        
        return obs, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """重置环境。"""
        self.count = 0
        return self.env.reset(**kwargs)


# ============================================================================
# 工厂函数
# ============================================================================

def make_dsrl_env(
    env,
    base_policy,
    visual_encoder=None,
    action_magnitude: float = 1.5,
    act_steps: int = 8,
    action_dim: int = 7,
    state_dim: int = 25,
    visual_feature_dim: int = 256,
    obs_horizon: int = 2,
    include_rgb: bool = True,
    use_gpu_env: bool = False,
    device: str = "cuda",
):
    """创建 DSRL 环境包装器。"""
    if use_gpu_env:
        return ManiSkillGPUFlowEnvWrapper(
            env=env,
            base_policy=base_policy,
            visual_encoder=visual_encoder,
            action_magnitude=action_magnitude,
            act_steps=act_steps,
            action_dim=action_dim,
            state_dim=state_dim,
            visual_feature_dim=visual_feature_dim,
            obs_horizon=obs_horizon,
            include_rgb=include_rgb,
            device=device,
        )
    else:
        return ShortCutFlowEnvWrapper(
            env=env,
            base_policy=base_policy,
            visual_encoder=visual_encoder,
            action_magnitude=action_magnitude,
            act_steps=act_steps,
            action_dim=action_dim,
            state_dim=state_dim,
            visual_feature_dim=visual_feature_dim,
            obs_horizon=obs_horizon,
            include_rgb=include_rgb,
            device=device,
        )
