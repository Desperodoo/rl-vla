"""
ReinFlow Online Fine-tuning Script

Fine-tunes pre-trained flow-matching policies using PPO with SMDP formulation.

Usage:
    python -m rlft.online.train_reinflow \
        --env_id PushCube-v1 \
        --pretrained_path runs/flow_matching/checkpoint.pt \
        --total_updates 1000
"""

ALGO_NAME = "ReinFlow"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import tyro
from diffusers.training_utils import EMAModel
from torch.utils.tensorboard import SummaryWriter

# Import from rlft package
from rlft.networks import PlainConv, StateEncoder
from rlft.algorithms.online_rl import ReinFlowAgent
from rlft.buffers import RolloutBufferPPO
from rlft.envs import make_eval_envs, evaluate
from rlft.datasets.data_utils import ObservationStacker


@dataclass
class Args:
    """ReinFlow fine-tuning arguments."""
    # Experiment settings
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "ReinFlow"
    wandb_entity: Optional[str] = None
    capture_video: bool = True
    """whether to capture videos of the agent performances"""
    
    # Environment settings
    env_id: str = "PushCube-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_joint_delta_pos"
    sim_backend: str = "gpu"
    num_envs: int = 1024
    num_eval_envs: int = 5
    """number of parallel eval environments"""
    max_episode_steps: Optional[int] = None
    """max episode steps (None uses env default)"""
    
    # Pre-trained model
    pretrained_path: Optional[str] = None
    freeze_visual_encoder: bool = True
    
    # Training settings
    total_updates: int = 10000
    """total training updates"""
    rollout_steps: int = 64
    """number of SMDP chunks to collect before each update"""
    ppo_epochs: int = 20
    """number of PPO epochs per update"""
    minibatch_size: int = 5120
    """minibatch size for PPO updates"""
    lr: float = 1e-6
    """learning rate for policy"""
    lr_critic: float = 1e-6
    """learning rate for value network"""
    max_grad_norm: float = 10.0
    """maximum gradient norm for clipping"""
    
    # PPO hyperparameters
    gamma: float = 0.99
    """discount factor"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    clip_ratio: float = 0.1
    """PPO clip ratio"""
    entropy_coef: float = 0.0
    """entropy coefficient"""
    value_coef: float = 0.5
    """value loss coefficient"""
    normalize_advantage: bool = True
    """whether to normalize advantages per minibatch"""
    
    # Value network settings
    clip_value_loss: bool = True
    """whether to clip value loss to reduce vloss explosion"""
    value_clip_range: float = 10.0
    """clip value predictions to [-range, range] for stable training"""
    use_target_value_net: bool = True
    """whether to use target value network for stable value estimation"""
    target_update_rate: float = 0.005
    """soft update rate for target value network"""
    value_target_clip: float = 100.0
    """clip value targets to prevent extreme values"""
    
    # Reward settings
    reward_scale: float = 0.5
    """scale factor for rewards (helps stabilize critic)"""
    normalize_rewards: bool = True
    """whether to normalize rewards using running mean/std"""
    normalize_returns: bool = True
    """whether to normalize returns using running mean/std"""
    
    # Noise/exploration settings
    max_noise_std: float = 0.15
    """maximum exploration noise std"""
    min_noise_std: float = 0.01
    """minimum exploration noise std"""
    noise_decay_type: Literal["linear", "cosine", "constant", "exponential"] = "linear"
    """noise decay schedule type"""
    noise_decay_steps: int = 500
    """noise decay steps"""
    
    # Critic warmup
    critic_warmup_steps: int = 50
    
    # KL early stopping
    kl_early_stop: bool = True
    target_kl: float = 0.01
    
    # Policy architecture
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    
    # Flow settings
    num_flow_steps: int = 10
    ema_decay: float = 0.999
    
    # Visual encoder (if RGB obs)
    visual_feature_dim: int = 256
    
    # Logging settings
    log_freq: int = 5
    eval_freq: int = 20
    save_freq: int = 100
    num_eval_episodes: int = 100


class RunningMeanStd:
    """Running mean and standard deviation tracker."""
    def __init__(self, shape=(), epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = len(x)
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class RewardNormalizer:
    """Normalizes rewards using running statistics."""
    def __init__(self):
        self.rms = RunningMeanStd()
    
    def update(self, rewards):
        self.rms.update(rewards)
    
    def normalize(self, rewards):
        return rewards / (self.rms.std + 1e-8)


class ReturnNormalizer:
    """Normalizes returns using running statistics."""
    def __init__(self):
        self.rms = RunningMeanStd()
    
    def update(self, returns):
        self.rms.update(returns)
    
    @property
    def std(self):
        return self.rms.std


class AgentWrapper:
    """Wrapper for unified evaluation interface.
    
    This wrapper handles observation encoding for evaluation.
    Note: eval_envs already applies FrameStack, so obs comes in as
    (num_envs, obs_horizon, state_dim) format - no additional stacking needed.
    """
    def __init__(self, agent, visual_encoder, state_encoder, include_rgb, device):
        self.agent = agent
        self.visual_encoder = visual_encoder
        self.state_encoder = state_encoder
        self.include_rgb = include_rgb
        self.device = device
    
    def eval(self):
        """Set agent to eval mode."""
        self.agent.eval()
        if self.visual_encoder is not None:
            self.visual_encoder.eval()
        if self.state_encoder is not None:
            self.state_encoder.eval()
    
    def train(self):
        """Set agent to train mode."""
        self.agent.train()
        if self.visual_encoder is not None:
            self.visual_encoder.train()
        if self.state_encoder is not None:
            self.state_encoder.train()
    
    def reset(self, obs):
        """Reset is a no-op for eval since FrameStack handles history."""
        pass
    
    def encode_obs(self, obs):
        """Encode observation to conditioning features.
        
        Args:
            obs: Dict with 'state' key, shape (B, T, state_dim) from FrameStack
                 OR tensor shape (B, T, state_dim)
        
        Returns:
            Flattened features (B, T * feature_dim)
        """
        # Handle dict or tensor input
        if isinstance(obs, dict):
            state = obs["state"]
        else:
            state = obs
        
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        state = state.float()
        
        # state shape: (B, T, state_dim) from FrameStack
        B = state.shape[0]
        T = state.shape[1]
        features = []
        
        if self.include_rgb and self.visual_encoder is not None:
            rgb = obs.get("rgb") if isinstance(obs, dict) else None
            if rgb is not None:
                if isinstance(rgb, np.ndarray):
                    rgb = torch.from_numpy(rgb).to(self.device)
                rgb_flat = rgb.view(B * T, *rgb.shape[2:]).float() / 255.0
                if rgb_flat.ndim == 4 and rgb_flat.shape[-1] == 3:
                    rgb_flat = rgb_flat.permute(0, 3, 1, 2)
                visual_feat = self.visual_encoder(rgb_flat)
                visual_feat = visual_feat.view(B, T, -1)
                features.append(visual_feat)
        
        features.append(state)
        
        obs_features = torch.cat(features, dim=-1)
        return obs_features.reshape(B, -1)  # (B, T * feature_dim)
    
    @torch.no_grad()
    def get_action(self, obs, deterministic=True, use_ema=True, **kwargs):
        """Get action from agent.
        
        Args:
            obs: Observation from eval_envs (already FrameStacked)
                 Shape: (B, T, state_dim) or dict with 'state' key
            deterministic: Whether to use deterministic action
            use_ema: Whether to use EMA weights
        
        Returns:
            actions: (B, pred_horizon, act_dim)
        """
        obs_cond = self.encode_obs(obs)
        
        # ReinFlowAgent.get_action returns (actions, x_chain), only take actions
        actions, _ = self.agent.get_action(
            obs_cond, 
            deterministic=deterministic, 
            use_ema=use_ema,
            return_chains=False,
        )
        return actions


def make_train_envs(args):
    """Create training environments."""
    try:
        import gymnasium as gym
        import mani_skill.envs
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    except ImportError:
        raise ImportError("ManiSkill3 is required. Install with: pip install mani-skill")
    
    include_rgb = "rgb" in args.obs_mode or "rgbd" in args.obs_mode
    
    env_kwargs = dict(
        obs_mode="rgbd" if include_rgb else "state",
        control_mode=args.control_mode,
        reward_mode="dense",
        sim_backend=args.sim_backend,
    )
    
    # Check for GPU backend (either "gpu" or "physx_cuda")
    if args.sim_backend in ["gpu", "physx_cuda"]:
        envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
        if include_rgb:
            envs = FlattenRGBDObservationWrapper(envs)
    else:
        from gymnasium.vector import SyncVectorEnv
        def make_env():
            env = gym.make(args.env_id, **env_kwargs)
            if include_rgb:
                env = FlattenRGBDObservationWrapper(env)
            return env
        envs = SyncVectorEnv([make_env for _ in range(args.num_envs)])
    
    return envs


def main():
    args = tyro.cli(Args)
    
    # Generate experiment name
    if args.exp_name is None:
        args.exp_name = f"reinflow-{args.env_id}-seed{args.seed}"
    
    run_name = f"{args.exp_name}__{int(time.time())}"
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )
    
    writer = SummaryWriter(log_dir)
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # Create environments
    train_envs = make_train_envs(args)
    
    # Create evaluation environments WITH FrameStack matching obs_horizon
    include_rgb = "rgb" in args.obs_mode or "rgbd" in args.obs_mode
    
    eval_env_kwargs = dict(
        control_mode=args.control_mode,
        obs_mode="rgbd" if include_rgb else "state",
        render_mode="rgb_array",
    )
    eval_other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    # Add FlattenRGBDObservationWrapper for RGB observations
    if include_rgb:
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
        eval_wrappers = [FlattenRGBDObservationWrapper]
    else:
        eval_wrappers = []
    
    eval_envs = make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        sim_backend=args.sim_backend,
        env_kwargs=eval_env_kwargs,
        other_kwargs=eval_other_kwargs,
        video_dir=f"{log_dir}/videos" if args.capture_video else None,
        wrappers=eval_wrappers,
    )
    
    # Get environment info
    sample_obs, _ = train_envs.reset(seed=args.seed)
    
    if include_rgb:
        # With FlattenRGBDObservationWrapper, obs is {"state": ..., "rgb": ...}
        if isinstance(sample_obs, dict) and "state" in sample_obs:
            state_dim = sample_obs["state"].shape[-1]
            rgb_shape = sample_obs["rgb"].shape
        elif isinstance(sample_obs, dict) and "sensor_data" in sample_obs:
            # Fallback for non-flattened obs
            first_cam = list(sample_obs["sensor_data"].keys())[0]
            rgb_shape = sample_obs["sensor_data"][first_cam]["rgb"].shape
            state_dim = sample_obs["agent"]["qpos"].shape[-1] + sample_obs["agent"]["qvel"].shape[-1]
        else:
            rgb_shape = sample_obs["rgb"].shape
            state_dim = sample_obs["state"].shape[-1]
    else:
        if isinstance(sample_obs, dict) and "observation" in sample_obs:
            state_dim = sample_obs["observation"].shape[-1]
        elif isinstance(sample_obs, dict) and "state" in sample_obs:
            state_dim = sample_obs["state"].shape[-1]
        elif isinstance(sample_obs, dict):
            state_dim = sum(v.shape[-1] for v in sample_obs.values() if isinstance(v, (np.ndarray, torch.Tensor)))
        else:
            state_dim = sample_obs.shape[-1]
    
    action_dim = train_envs.single_action_space.shape[0]
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Include RGB: {include_rgb}")
    
    # Create visual encoder
    if include_rgb:
        visual_encoder = PlainConv(
            in_channels=3,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
        feature_dim = args.visual_feature_dim + state_dim
    else:
        visual_encoder = None
        feature_dim = state_dim
    
    state_encoder = None  # Optional, can add later
    obs_dim = feature_dim * args.obs_horizon
    
    # Create agent
    from rlft.networks import ShortCutVelocityUNet1D
    
    # Use ShortCutVelocityUNet1D which is compatible with ReinFlowAgent
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=tuple(args.unet_dims),
        n_groups=args.n_groups,
    )
    
    agent = ReinFlowAgent(
        velocity_net=velocity_net,
        act_dim=action_dim,  # ReinFlowAgent uses act_dim, not action_dim
        obs_dim=obs_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        act_horizon=args.act_horizon,
        num_inference_steps=args.num_flow_steps,  # ReinFlowAgent uses num_inference_steps
        ema_decay=args.ema_decay,
        max_noise_std=args.max_noise_std,
        min_noise_std=args.min_noise_std,
        noise_decay_type=args.noise_decay_type,
        noise_decay_steps=args.noise_decay_steps,
        value_target_tau=args.target_update_rate,  # ReinFlowAgent uses value_target_tau
        value_target_clip=args.value_target_clip,
        gamma=args.gamma,
    ).to(device)
    
    # Load pre-trained weights
    if args.pretrained_path is not None:
        print(f"Loading pre-trained model from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        if "agent" in checkpoint:
            agent.load_state_dict(checkpoint["agent"], strict=False)
        if "visual_encoder" in checkpoint and visual_encoder is not None:
            visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        
        # Freeze visual encoder if specified
        if args.freeze_visual_encoder and visual_encoder is not None:
            for param in visual_encoder.parameters():
                param.requires_grad = False
    
    # Create observation stacker for training rollouts
    obs_stacker = ObservationStacker(
        obs_horizon=args.obs_horizon,
        num_envs=args.num_envs,
    )
    
    # Create agent wrapper for evaluation (no obs_stacker needed - eval_envs uses FrameStack)
    agent_wrapper = AgentWrapper(
        agent, visual_encoder, state_encoder, include_rgb, device
    )
    
    # Separate optimizers for policy and critic
    policy_params = list(agent.noisy_velocity_net.parameters())
    if visual_encoder is not None and not args.freeze_visual_encoder:
        policy_params += list(visual_encoder.parameters())
    
    critic_params = list(agent.value_net.parameters())
    
    policy_optimizer = optim.Adam(policy_params, lr=args.lr)
    critic_optimizer = optim.Adam(critic_params, lr=args.lr_critic)
    
    # Create rollout buffer
    # RolloutBufferPPO uses: num_steps, num_envs, obs_dim, pred_horizon, act_dim, num_inference_steps
    buffer = RolloutBufferPPO(
        num_steps=args.rollout_steps,
        num_envs=args.num_envs,
        obs_dim=obs_dim,
        pred_horizon=args.pred_horizon,
        act_dim=action_dim,
        num_inference_steps=args.num_flow_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=device,
    )
    
    # Normalizers
    reward_normalizer = RewardNormalizer()
    return_normalizer = ReturnNormalizer()
    
    # Tracking variables
    global_step = 0
    num_updates = 0
    best_eval_metrics = {"success_once": 0.0, "success_at_end": 0.0}
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    env_episode_rewards = [0.0] * args.num_envs
    env_episode_lengths = [0] * args.num_envs
    timings = defaultdict(float)
    start_time = time.time()
    
    # Local encoding function
    def local_encode_observations(obs_seq):
        B = obs_seq["state"].shape[0]
        T = obs_seq["state"].shape[1]
        features = []
        
        if include_rgb and visual_encoder is not None:
            rgb = obs_seq["rgb"]
            rgb_flat = rgb.view(B * T, *rgb.shape[2:]).float() / 255.0
            # Convert NHWC to NCHW if needed
            if rgb_flat.ndim == 4 and rgb_flat.shape[-1] in [1, 3, 4]:
                rgb_flat = rgb_flat.permute(0, 3, 1, 2)
            visual_feat = visual_encoder(rgb_flat)
            visual_feat = visual_feat.view(B, T, -1)
            features.append(visual_feat)
        
        state = obs_seq["state"]
        features.append(state)
        
        obs_features = torch.cat(features, dim=-1)
        return obs_features.reshape(B, -1)
    
    def convert_obs_to_dict(obs):
        """Convert raw ManiSkill observations to consistent dict format.
        
        ManiSkill returns different formats depending on obs_mode:
        - state: returns tensor directly (num_envs, obs_dim) 
        - rgbd with FlattenRGBDObservationWrapper: returns dict {"state": ..., "rgb": ...}
        
        This function normalizes both to dict format preserving all keys.
        """
        if isinstance(obs, dict):
            result = {}
            # Preserve state
            if "state" in obs:
                s = obs["state"]
                result["state"] = s.float() if torch.is_tensor(s) else torch.from_numpy(s).float().to(device)
            elif "observation" in obs:
                s = obs["observation"]
                result["state"] = s.float() if torch.is_tensor(s) else torch.from_numpy(s).float().to(device)
            else:
                # Flatten nested dict to state tensor
                state_parts = []
                for k, v in obs.items():
                    if k == "rgb":
                        continue  # Handle rgb separately
                    if isinstance(v, dict):
                        for sk, sv in v.items():
                            if torch.is_tensor(sv):
                                state_parts.append(sv.float().reshape(sv.shape[0], -1))
                            elif isinstance(sv, np.ndarray):
                                state_parts.append(torch.from_numpy(sv).float().to(device).reshape(sv.shape[0], -1))
                    elif torch.is_tensor(v):
                        state_parts.append(v.float().reshape(v.shape[0], -1))
                    elif isinstance(v, np.ndarray):
                        state_parts.append(torch.from_numpy(v).float().to(device).reshape(v.shape[0], -1))
                result["state"] = torch.cat(state_parts, dim=-1)
            
            # Preserve rgb if present
            if "rgb" in obs:
                rgb = obs["rgb"]
                result["rgb"] = rgb if torch.is_tensor(rgb) else torch.from_numpy(rgb).to(device)
            
            return result
        elif torch.is_tensor(obs):
            return {"state": obs.float()}
        else:
            return {"state": torch.from_numpy(obs).float().to(device)}
    
    # Reset environments and initialize obs stacker
    obs_raw, _ = train_envs.reset(seed=args.seed)
    obs = convert_obs_to_dict(obs_raw)
    obs_stacker.reset(obs)
    
    pbar = tqdm(total=args.total_updates, desc="Training")
    
    while num_updates < args.total_updates:
        # Collect rollouts
        buffer.reset()
        
        for rollout_step in range(args.rollout_steps):
            obs_seq = obs_stacker.get_stacked()
            
            with torch.no_grad():
                obs_cond = local_encode_observations(obs_seq)
                
                actions, x_chain = agent.get_action(
                    obs_cond,
                    deterministic=False,
                    use_ema=False,
                    return_chains=True,
                )
                
                # Squeeze values from [B, 1] to [B]
                values = agent.compute_value(obs_cond, use_target=args.use_target_value_net).squeeze(-1)
                log_probs, _ = agent.compute_action_log_prob(obs_cond, actions, x_chain=x_chain)
            
            # Execute action chunk (SMDP)
            chunk_rewards = []
            chunk_dones = []
            
            for step_idx in range(args.act_horizon):
                action = actions[:, step_idx, :]
                next_obs_raw, rewards, terminations, truncations, infos = train_envs.step(action)
                next_obs = convert_obs_to_dict(next_obs_raw)
                
                dones = terminations | truncations
                chunk_rewards.append(rewards)
                chunk_dones.append(dones)
                
                # Track episode statistics
                for i in range(args.num_envs):
                    reward_val = rewards[i].item() if hasattr(rewards[i], "item") else float(rewards[i])
                    env_episode_rewards[i] += reward_val
                    env_episode_lengths[i] += 1
                    
                    done_val = dones[i].item() if hasattr(dones[i], "item") else bool(dones[i])
                    if done_val:
                        episode_rewards.append(env_episode_rewards[i])
                        episode_lengths.append(env_episode_lengths[i])
                        success = infos.get("success", [False] * args.num_envs)[i]
                        if hasattr(success, "item"):
                            success = success.item()
                        episode_successes.append(float(success))
                        env_episode_rewards[i] = 0.0
                        env_episode_lengths[i] = 0
                
                obs_stacker.append(next_obs)
                obs = next_obs
                global_step += args.num_envs
                
                if dones.any():
                    break
            
            # Compute SMDP rewards
            chunk_rewards_tensor = torch.stack(chunk_rewards) if isinstance(chunk_rewards[0], torch.Tensor) else torch.tensor(chunk_rewards, device=device)
            chunk_dones_tensor = torch.stack(chunk_dones) if isinstance(chunk_dones[0], torch.Tensor) else torch.tensor(chunk_dones, device=device)
            chunk_len = len(chunk_rewards)
            
            cum_rewards = torch.zeros(args.num_envs, device=device)
            chunk_done_flags = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            discount = 1.0
            
            for i in range(chunk_len):
                still_running = ~chunk_done_flags
                cum_rewards = cum_rewards + discount * chunk_rewards_tensor[i] * still_running.float()
                chunk_done_flags = chunk_done_flags | chunk_dones_tensor[i]
                discount *= args.gamma
            
            cum_rewards = cum_rewards * args.reward_scale
            
            if args.normalize_rewards:
                reward_normalizer.update(cum_rewards.cpu().numpy())
                cum_rewards = cum_rewards / (reward_normalizer.rms.std + 1e-8)
            
            # RolloutBufferPPO.add() uses: obs, action, log_prob, reward, done, value, x_chain
            buffer.add(
                obs=obs_cond,
                action=actions,  # 'action' not 'actions'
                log_prob=log_probs,  # 'log_prob' not 'log_probs'
                reward=cum_rewards,  # 'reward' not 'rewards'
                done=chunk_done_flags.float(),  # 'done' not 'dones'
                value=values,  # 'value' not 'values'
                x_chain=x_chain,
            )
            
            # Note: RolloutBufferPPO handles episode boundaries in compute_returns_and_advantages
            # via the dones flags, so we don't need explicit set_final_values
        
        # Compute returns and advantages
        with torch.no_grad():
            obs_seq = obs_stacker.get_stacked()
            obs_cond = local_encode_observations(obs_seq)
            last_value = agent.compute_value(obs_cond, use_target=args.use_target_value_net).squeeze(-1)
        
        # RolloutBufferPPO uses: last_value, last_done (not next_value, next_done)
        buffer.compute_returns_and_advantages(last_value=last_value, last_done=chunk_done_flags.float())
        
        if args.normalize_returns:
            returns_np = buffer.returns.cpu().numpy().flatten()
            return_normalizer.update(returns_np)
            buffer.returns = buffer.returns / (return_normalizer.std + 1e-8)
        
        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0
        kl_early_stopped = False
        
        for epoch in range(args.ppo_epochs):
            if kl_early_stopped:
                break
            
            # RolloutBufferPPO.get_batches(batch_size, normalize_advantages=True)
            batches = buffer.get_batches(args.minibatch_size, normalize_advantages=args.normalize_advantage)
            
            for batch in batches:
                loss_dict = agent.compute_ppo_loss(
                    obs_cond=batch["obs"],
                    actions=batch["actions"],
                    old_log_probs=batch["log_probs"],
                    advantages=batch["advantages"],
                    returns=batch["returns"],
                    x_chain=batch["x_chains"],  # key is 'x_chains' (plural)
                    old_values=batch.get("values"),
                    clip_value=args.clip_value_loss,
                    value_clip_range=args.value_clip_range,
                )
                
                if num_updates < args.critic_warmup_steps:
                    critic_optimizer.zero_grad()
                    loss_dict["loss"].backward()
                    nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                    critic_optimizer.step()
                else:
                    policy_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    loss_dict["loss"].backward()
                    nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm)
                    nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                    policy_optimizer.step()
                    critic_optimizer.step()
                
                total_policy_loss += loss_dict["policy_loss"].item()
                total_value_loss += loss_dict["value_loss"].item()
                total_entropy += loss_dict["entropy"].item()
                num_batches += 1
                
                if args.kl_early_stop and args.target_kl is not None:
                    kl_val = loss_dict.get("approx_kl", 0)
                    if isinstance(kl_val, torch.Tensor):
                        kl_val = kl_val.item()
                    if kl_val > args.target_kl * 1.5:
                        kl_early_stopped = True
                        break
        
        # Updates
        agent.update_ema()
        agent.update_noise_schedule()
        agent.update_target_value_net()
        num_updates += 1
        
        # Logging
        avg_policy_loss = total_policy_loss / max(1, num_batches)
        avg_value_loss = total_value_loss / max(1, num_batches)
        avg_entropy = total_entropy / max(1, num_batches)
        
        if num_updates % args.log_freq == 0:
            writer.add_scalar("losses/policy_loss", avg_policy_loss, num_updates)
            writer.add_scalar("losses/value_loss", avg_value_loss, num_updates)
            writer.add_scalar("losses/entropy", avg_entropy, num_updates)
            writer.add_scalar("training/global_step", global_step, num_updates)
            
            if len(episode_rewards) > 0:
                writer.add_scalar("charts/episode_reward", np.mean(episode_rewards[-100:]), num_updates)
                writer.add_scalar("charts/success_rate", np.mean(episode_successes[-100:]), num_updates)
            
            if args.track:
                wandb.log({
                    "losses/policy_loss": avg_policy_loss,
                    "losses/value_loss": avg_value_loss,
                    "losses/entropy": avg_entropy,
                    "training/global_step": global_step,
                }, step=num_updates)
        
        # Evaluation
        if num_updates % args.eval_freq == 0:
            eval_metrics = evaluate(
                args.num_eval_episodes,
                agent_wrapper,
                eval_envs,
                device,
                args.sim_backend,
                agent_kwargs={"deterministic": True, "use_ema": True},
            )
            
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], num_updates)
            
            # WandB logging for evaluation
            if args.track:
                wandb_eval = {f"eval/{k}": v for k, v in eval_metrics.items()}
                wandb.log(wandb_eval, step=num_updates)
            
            for k in ["success_once", "success_at_end"]:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    checkpoint = {
                        "agent": agent.state_dict(),
                        "visual_encoder": visual_encoder.state_dict() if visual_encoder else None,
                        "config": vars(args),
                    }
                    torch.save(checkpoint, f"{log_dir}/checkpoint_best_{k}.pt")
        
        # Save checkpoint
        if num_updates % args.save_freq == 0:
            checkpoint = {
                "agent": agent.state_dict(),
                "visual_encoder": visual_encoder.state_dict() if visual_encoder else None,
                "policy_optimizer": policy_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "num_updates": num_updates,
                "config": vars(args),
            }
            torch.save(checkpoint, f"{log_dir}/checkpoint_{num_updates}.pt")
        
        # Update progress bar
        pbar.set_postfix({
            "ploss": f"{avg_policy_loss:.4f}",
            "vloss": f"{avg_value_loss:.4f}",
            "reward": f"{np.mean(episode_rewards[-100:]) if episode_rewards else 0:.2f}",
        })
        pbar.update(1)
    
    pbar.close()
    
    # Final save
    checkpoint = {
        "agent": agent.state_dict(),
        "visual_encoder": visual_encoder.state_dict() if visual_encoder else None,
        "config": vars(args),
    }
    torch.save(checkpoint, f"{log_dir}/final_model.pt")
    
    # Cleanup
    train_envs.close()
    eval_envs.close()
    writer.close()
    
    if args.track:
        wandb.finish()
    
    print("Training complete!")


if __name__ == "__main__":
    main()
