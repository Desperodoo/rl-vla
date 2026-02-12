"""
DSRL Official Utils - 从官方 ajwagen/dsrl 移植

包含:
- ShortCutFlowWrapper: ShortCut Flow 策略包装器 (对应官方 DPPOBasePolicyWrapper)
- LoggingCallback: 训练日志回调
- collect_rollouts: 初始数据收集
- load_offline_data: 离线数据加载

Reference: https://github.com/ajwagen/dsrl/blob/main/utils.py
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
import sys
from pathlib import Path

# 添加路径以导入 diffusion_policy 组件
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))


class ShortCutFlowWrapper:
    """ShortCut Flow 策略包装器。
    
    对应官方 DPPOBasePolicyWrapper，将 ShortCut Flow velocity_net
    包装为统一接口，支持从噪声生成动作。
    
    官方接口:
        cond = {"state": obs, "noise_action": initial_noise}
        samples = base_policy(cond=cond, deterministic=True)
        actions = samples.trajectories
    
    本实现:
        actions = wrapper(obs, initial_noise)
        # 内部调用 velocity_net 进行 flow 采样
    
    Args:
        velocity_net: 预训练的 ShortCutVelocityUNet1D
        visual_encoder: 可选的视觉编码器
        obs_horizon: 观察历史长度
        pred_horizon: 预测动作序列长度
        action_dim: 动作维度
        num_inference_steps: Flow 积分步数
        device: 设备
    """
    
    def __init__(
        self,
        velocity_net,
        visual_encoder=None,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_dim: int = 7,
        num_inference_steps: int = 8,
        device: str = "cuda",
    ):
        self.velocity_net = velocity_net
        self.visual_encoder = visual_encoder
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        self.device = device
        
        # 确保模型在正确设备上
        self.velocity_net = self.velocity_net.to(device)
        if self.visual_encoder is not None:
            self.visual_encoder = self.visual_encoder.to(device)
        
        # 设置为评估模式
        self.velocity_net.eval()
        if self.visual_encoder is not None:
            self.visual_encoder.eval()
    
    def __call__(
        self,
        obs: torch.Tensor,
        initial_noise: torch.Tensor,
        return_numpy: bool = True,
        act_steps: int = None,
    ) -> np.ndarray:
        """从观察和噪声生成动作序列。
        
        Flow 积分：从 t=0 (噪声) 到 t=1 (动作)
        
        支持两种模式：
        1. initial_noise 形状为 (B, pred_horizon, action_dim): 直接使用
        2. initial_noise 形状为 (B, act_steps, action_dim): 用零 pad 到 pred_horizon
        
        Args:
            obs: 观察张量 (B, obs_dim) 或 (B, obs_horizon, obs_dim)
            initial_noise: 初始噪声 (B, T, action_dim) 或 (B, T * action_dim)
            return_numpy: 是否返回 numpy 数组
            act_steps: 如果提供，只返回前 act_steps 步的动作
            
        Returns:
            actions: 动作序列 (B, T, action_dim)，T 取决于 act_steps 参数
        """
        with torch.no_grad():
            # 确保输入在正确设备上
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            else:
                obs = obs.to(self.device)
            
            if not isinstance(initial_noise, torch.Tensor):
                initial_noise = torch.tensor(initial_noise, device=self.device, dtype=torch.float32)
            else:
                initial_noise = initial_noise.to(self.device)
            
            B = initial_noise.shape[0]
            
            # 重塑噪声
            if initial_noise.dim() == 2:
                # (B, T * action_dim) -> (B, T, action_dim)
                T = initial_noise.shape[1] // self.action_dim
                initial_noise = initial_noise.view(B, T, self.action_dim)
            
            # 检查噪声长度
            noise_T = initial_noise.shape[1]
            
            # 总是 pad 到 pred_horizon 进行推理
            # 这是为了保持与 pretrained model 一致的行为
            if noise_T < self.pred_horizon:
                # 噪声长度小于 pred_horizon，需要 pad
                # 用零填充后面的位置
                pad_length = self.pred_horizon - noise_T
                padding = torch.zeros(B, pad_length, self.action_dim, device=self.device)
                x = torch.cat([initial_noise, padding], dim=1)
            elif noise_T > self.pred_horizon:
                # 噪声长度大于 pred_horizon，截断
                x = initial_noise[:, :self.pred_horizon, :]
            else:
                x = initial_noise
            
            # Flatten obs for global conditioning
            if obs.dim() == 3:
                obs = obs.reshape(B, -1)  # (B, obs_horizon * obs_dim)
            
            # 从 t=0 积分到 t=1 (噪声 -> 动作)
            dt = 1.0 / self.num_inference_steps
            step_size = torch.full((B,), dt, device=self.device)
            
            for i in range(self.num_inference_steps):
                t = torch.full((B,), i * dt, device=self.device)
                # velocity_net 需要 4 个参数: sample, timestep, step_size, global_cond
                v = self.velocity_net(x, t, step_size, obs)
                # Euler 步进
                x = x + v * dt
            
            # Clamp to action bounds
            actions = torch.clamp(x, -1.0, 1.0)
            
            # 总是返回完整的 pred_horizon 长度
            # 调用方需要自己处理 action indexing（从 obs_horizon-1 开始取 act_steps 步）
            # 如果显式指定了 act_steps，则从 obs_horizon-1 开始返回 act_steps 步
            if act_steps is not None:
                start_idx = self.obs_horizon - 1
                actions = actions[:, start_idx:start_idx + act_steps, :]
            
            if return_numpy:
                return actions.cpu().numpy()
            return actions
    
    def sample_with_latent(
        self,
        obs: torch.Tensor,
        latent_w: torch.Tensor,
        action_magnitude: float = 1.5,
        return_numpy: bool = True,
    ) -> np.ndarray:
        """使用 latent steering 向量生成动作。
        
        latent_w 已经是 TanhNormal 采样的结果，范围在 [-action_magnitude, +action_magnitude]。
        将其作为初始噪声输入 flow policy。
        
        Args:
            obs: 观察张量 (B, obs_dim)
            latent_w: Latent steering 向量 (B, pred_horizon, action_dim)
            action_magnitude: latent 范围 [-mag, +mag]
            return_numpy: 是否返回 numpy 数组
            
        Returns:
            actions: 动作序列 (B, pred_horizon, action_dim)
        """
        # latent_w 直接作为初始噪声
        return self(obs, latent_w, return_numpy=return_numpy)


def load_shortcut_flow_policy(
    checkpoint_path: str,
    visual_encoder_class=None,
    obs_horizon: int = 2,
    pred_horizon: int = 16,
    action_dim: int = 7,
    visual_feature_dim: int = 256,
    diffusion_step_embed_dim: int = 64,
    unet_dims: Tuple[int, ...] = (64, 128, 256),
    n_groups: int = 8,
    state_dim: int = None,  # 如果为 None，从 checkpoint 推断
    include_rgb: bool = True,
    use_ema: bool = True,
    device: str = "cuda",
) -> Tuple[ShortCutFlowWrapper, Optional[Any]]:
    """加载预训练的 ShortCut Flow 策略。
    
    Args:
        checkpoint_path: checkpoint 文件路径
        visual_encoder_class: 视觉编码器类 (如 PlainConv)
        state_dim: state 维度。如果为 None，从 checkpoint 推断
        其他参数: 模型配置
        
    Returns:
        (ShortCutFlowWrapper, visual_encoder)
    """
    from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
    from diffusion_policy.plain_conv import PlainConv
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取 agent state
    if use_ema and "ema_agent" in checkpoint:
        agent_state = checkpoint["ema_agent"]
    else:
        agent_state = checkpoint.get("agent", checkpoint)
    
    # 从 checkpoint 推断 state_dim
    if state_dim is None:
        # 找到 cond_encoder 的输入维度
        for key, value in agent_state.items():
            if "velocity_net" in key and "cond_encoder.1.weight" in key:
                cond_encoder_input_dim = value.shape[1]
                # cond_encoder 输入 = global_cond_dim + step_embed_dim
                global_cond_dim = cond_encoder_input_dim - diffusion_step_embed_dim
                # global_cond_dim = obs_horizon * (visual_dim + state_dim)
                visual_dim = visual_feature_dim if include_rgb else 0
                state_dim = (global_cond_dim // obs_horizon) - visual_dim
                print(f"Inferred state_dim from checkpoint: {state_dim}")
                print(f"  (cond_encoder_input={cond_encoder_input_dim}, global_cond_dim={global_cond_dim})")
                break
        
        if state_dim is None:
            raise ValueError("Could not infer state_dim from checkpoint")
    
    # 计算 global_cond_dim
    visual_dim = visual_feature_dim if include_rgb else 0
    global_cond_dim = obs_horizon * (visual_dim + state_dim)
    
    # 创建 velocity_net
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=unet_dims,
        n_groups=n_groups,
    ).to(device)
    
    # 创建 visual_encoder
    visual_encoder = None
    if include_rgb and visual_encoder_class is not None:
        visual_encoder = visual_encoder_class(
            in_channels=3,
            out_dim=visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
    
    # 提取 velocity_net 权重 (agent_state 已在上面加载)
    velocity_net_state = {}
    for key, value in agent_state.items():
        if key.startswith("velocity_net."):
            velocity_net_state[key.replace("velocity_net.", "")] = value
    
    if velocity_net_state:
        velocity_net.load_state_dict(velocity_net_state)
        print(f"Loaded velocity_net from {checkpoint_path} ({len(velocity_net_state)} keys)")
    else:
        raise ValueError(f"No velocity_net weights found in {checkpoint_path}")
    
    # 加载 visual_encoder
    if visual_encoder is not None and "visual_encoder" in checkpoint:
        visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        print("Loaded visual_encoder")
    
    # 创建 wrapper
    wrapper = ShortCutFlowWrapper(
        velocity_net=velocity_net,
        visual_encoder=visual_encoder,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        device=device,
    )
    
    return wrapper, visual_encoder


def collect_rollouts(
    model,
    env,
    num_steps: int,
    base_policy: ShortCutFlowWrapper,
    algorithm: str = "dsrl_sac",
    action_magnitude: float = 1.5,
    act_steps: int = 8,
    action_dim: int = 7,
    n_envs: int = 1,
    device: str = "cuda",
):
    """收集初始 rollout 数据到 replay buffer。
    
    从官方 collect_rollouts 移植，适配 ShortCut Flow。
    
    Args:
        model: SB3 模型 (SAC 或 DSRL)
        env: 向量化环境
        num_steps: 收集步数
        base_policy: ShortCutFlowWrapper
        algorithm: "dsrl_sac" 或 "dsrl_na"
        action_magnitude: 噪声范围
        act_steps: 动作执行步数
        action_dim: 动作维度
        n_envs: 环境数量
        device: 设备
    """
    obs = env.reset()
    
    for i in range(num_steps):
        # 采样噪声
        noise = torch.randn(n_envs, act_steps, action_dim, device=device)
        
        if algorithm == "dsrl_sac":
            # 裁剪噪声到 action_magnitude 范围
            noise = noise.clamp(-action_magnitude, action_magnitude)
        
        # 通过 base_policy 生成动作
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
        action = base_policy(obs_tensor, noise)
        
        # 执行动作
        next_obs, reward, done, info = env.step(action)
        
        # 存储到 replay buffer
        if algorithm == "dsrl_na":
            action_store = action  # 存储真实动作
        else:  # dsrl_sac
            action_store = noise.detach().cpu().numpy()  # 存储噪声
        
        # 扁平化动作
        action_store = action_store.reshape(-1, act_steps * action_dim)
        
        if algorithm == "dsrl_sac":
            action_store = model.policy.scale_action(action_store)
        
        model.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action_store,
            reward=reward,
            done=done,
            infos=info,
        )
        
        obs = next_obs
    
    # 标记离线数据结束
    if hasattr(model.replay_buffer, 'final_offline_step'):
        model.replay_buffer.final_offline_step()


def load_offline_data(
    model,
    offline_data_path: str,
    n_env: int = 1,
):
    """加载离线数据到 replay buffer。
    
    从官方 load_offline_data 移植。
    
    注意: 此函数应仅用于 DSRL-NA。
    
    Args:
        model: SB3 模型
        offline_data_path: 离线数据路径 (.npz 格式)
        n_env: 环境数量
    """
    offline_data = np.load(offline_data_path)
    
    obs = offline_data['states']
    next_obs = offline_data['states_next']
    actions = offline_data['actions']
    rewards = offline_data['rewards']
    terminals = offline_data['terminals']
    
    for i in range(int(obs.shape[0] / n_env)):
        model.replay_buffer.add(
            obs=obs[n_env * i : n_env * i + n_env],
            next_obs=next_obs[n_env * i : n_env * i + n_env],
            action=actions[n_env * i : n_env * i + n_env],
            reward=rewards[n_env * i : n_env * i + n_env],
            done=terminals[n_env * i : n_env * i + n_env],
            infos=[{}] * n_env,
        )
    
    if hasattr(model.replay_buffer, 'final_offline_step'):
        model.replay_buffer.final_offline_step()
