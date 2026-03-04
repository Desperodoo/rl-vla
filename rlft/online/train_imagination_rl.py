"""
T-MBRL-BC-FINETUNE — SAC RL on ImaginationRLEnv (WM + VLM).

Trains a SAC agent in the imagination environment (Ctrl-World WM + VLM reward).
This is the core VLAW experiment: Policy-in-the-Loop Imagination RL.

Key differences from real-env PLD-SAC (train_pld.py):
    * Environment is ImaginationRLEnv (WM + VLM), not ManiSkill.
    * Observations are flat: [latent_4608 + state_25] = 4633 dim.
    * No visual encoder needed (latent already encoded via VAE).
    * No residual actions (no base policy inference on latent obs).
    * Direct SAC in full action space [-1, 1]^7.
    * WM + VLM on separate GPUs to fit in 24GB per card.
    * Lower UTD (20) due to WM compounding errors.

GPU Allocation (default):
    GPU 8: World Model (~13GB) + SAC agent (<1GB)
    GPU 9: VLM reward model (~9GB)

Usage::

    python -m rlft.online.train_imagination_rl \\
        --total_timesteps 10000 \\
        --wm_gpu 8 --vlm_gpu 9 --agent_gpu 8

    # Quick smoke test with mock environment:
    python -m rlft.online.train_imagination_rl \\
        --use_mock --total_timesteps 500 --eval_freq 100
"""

ALGO_NAME = "Imagination-SAC"

import os
import sys
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    import tyro
    HAS_TYRO = True
except ImportError:
    HAS_TYRO = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Ensure project root in path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# =====================================================================
# Arguments
# =====================================================================

@dataclass
class Args:
    """Imagination RL training arguments."""

    # ----- experiment -----
    exp_name: Optional[str] = None
    seed: int = 42
    cuda: bool = True
    track: bool = True
    wandb_project: str = "VLAW-ImaginationRL"
    wandb_entity: Optional[str] = None

    # ----- ImaginationRLEnv -----
    max_episode_steps: int = 60
    """Max steps per episode (num_interact=12 × act_steps=5)."""
    wm_act_steps: int = 5
    """Frames generated per WM rollout call."""
    num_history: int = 6
    """Number of history frames for WM."""
    vlm_reward_interval: int = 5
    """Call VLM every N steps (VLM is slow ~0.4s/call)."""
    use_continuous_reward: bool = True
    """Use p_yes as continuous reward (vs binary threshold)."""
    obs_mode: str = "flat"
    """Observation mode: 'flat' (latent+state) for RL."""
    task_instruction: str = "lift the peg upright"
    task_id: str = "LiftPegUpright-v1"

    # ----- model paths -----
    wm_ckpt: str = "checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt"
    vlm_model_path: str = "checkpoints/vlaw/reward_model/qwen_vl"
    vlm_lora_path: str = "checkpoints/vlaw/reward_model/lora_iter1_16frame/final"
    state_predictor_ckpt: str = "checkpoints/vlaw/state_predictor/LiftPegUpright-v1/state_predictor_iter1.pt"
    initial_frames_h5: str = "data/vlaw/encoded/eval_fixed/eval_set.h5"
    action_stat_path: str = "data/vlaw/meta_info/maniskill/stat.json"

    # ----- GPU allocation -----
    wm_gpu: int = 8
    """GPU for World Model (~13GB)."""
    vlm_gpu: int = 9
    """GPU for VLM reward model (~9GB)."""
    agent_gpu: int = 8
    """GPU for SAC agent (<1GB, can share with WM)."""

    # ----- mock mode -----
    use_mock: bool = False
    """Use mock WM/VLM for pipeline testing (no real model loading)."""

    # ----- SAC hyper-parameters -----
    total_timesteps: int = 10_000
    learning_rate: float = 1e-4
    """Conservative LR (prevents Q-divergence, PLD sweep §2.5)."""
    buffer_size: int = 100_000
    batch_size: int = 256
    """Smaller batch (single env, not vectorized)."""
    gamma: float = 0.95
    """Short horizon: matches imagination episode length (~60 steps)."""
    tau: float = 0.001
    """Slow target update (critical with high UTD, PLD v3)."""
    utd_ratio: int = 20
    """Lower UTD since WM has compounding errors (vs 60 in real env)."""
    init_temperature: float = 0.5
    """Moderate exploration (PLD v3: +0.14 vs temp=0.1)."""
    target_entropy: float = -3.5
    """PLD sweep optimal."""
    log_std_init: float = -3.0
    """Less conservative than PLD (-5.0) since we need more exploration in WM."""
    max_grad_norm: float = 10.0
    action_dim: int = 7
    state_dim: int = 25
    latent_shape: tuple = (4, 48, 24)

    # ----- network architecture -----
    num_layers: int = 3
    layer_size: int = 512
    """Smaller network for 4633-dim obs (vs 768 in PLD for ~562-dim obs)."""
    num_qs: int = 5
    use_layer_norm: bool = True

    # ----- warmup / seed -----
    num_seed_steps: int = 200
    """Random exploration steps before training (fill buffer)."""

    # ----- logging / eval / saving -----
    log_freq: int = 50
    eval_freq: int = 1000
    """Evaluate every N steps (run a few episodes in same env)."""
    num_eval_episodes: int = 5
    save_freq: int = 2000
    save_dir: str = "checkpoints/vlaw/imagination_rl"


# =====================================================================
# Environment creation
# =====================================================================

def create_imagination_env(args: Args):
    """Create ImaginationRLEnv with real or mock components.

    Returns:
        (env, component_info_dict)
    """
    from rlft.vlaw.world_model.imagination_rl_env import (
        ImaginationRLEnvConfig,
        make_imagination_rl_env,
    )

    config = ImaginationRLEnvConfig(
        max_steps=args.max_episode_steps,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        wm_act_steps=args.wm_act_steps,
        num_history=args.num_history,
        vlm_reward_interval=args.vlm_reward_interval,
        use_continuous_reward=args.use_continuous_reward,
        obs_mode=args.obs_mode,
        latent_shape=args.latent_shape,
        initial_frames_h5=str(Path(_ROOT) / args.initial_frames_h5),
        task_instruction=args.task_instruction,
        task_id=args.task_id,
        gpu_id=args.wm_gpu,
        state_predictor_ckpt=str(Path(_ROOT) / args.state_predictor_ckpt),
        verbose=True,
    )

    if args.use_mock:
        env = make_imagination_rl_env(config, use_mock=True)
        return env, {"mode": "mock"}

    # --- Load real World Model ---
    print(f"\n[1] Loading World Model on GPU {args.wm_gpu} ...")
    wm_adapter = _load_world_model(args)

    # --- Load real VLM Reward Model ---
    print(f"\n[2] Loading VLM Reward Model on GPU {args.vlm_gpu} ...")
    reward_model = _load_vlm_reward(args)

    # --- Load State Predictor ---
    print(f"\n[3] Loading State Predictor ...")
    state_predictor = _load_state_predictor(args)

    env = make_imagination_rl_env(
        config,
        wm_adapter=wm_adapter,
        reward_model=reward_model,
        state_predictor=state_predictor,
    )

    return env, {
        "mode": "real",
        "wm_gpu": args.wm_gpu,
        "vlm_gpu": args.vlm_gpu,
    }


def _load_world_model(args: Args):
    """Load Ctrl-World adapter."""
    # Add ctrl_world to path
    ctrl_world_root = _ROOT / "ctrl_world"
    if str(ctrl_world_root) not in sys.path:
        sys.path.insert(0, str(ctrl_world_root))

    from ctrl_world.config import wm_args_maniskill
    from rlft.vlaw.world_model.ctrl_world_adapter import CtrlWorldAdapter

    wm_config = wm_args_maniskill()
    # Override checkpoint path
    wm_config.ckpt_path = str(Path(_ROOT) / args.wm_ckpt)
    wm_config.data_stat_path = str(Path(_ROOT) / args.action_stat_path)

    adapter = CtrlWorldAdapter(
        wm_config,
        ckpt_path=wm_config.ckpt_path,
        device=f"cuda:{args.wm_gpu}",
        dtype=torch.float16,
    )

    mem = torch.cuda.max_memory_allocated(args.wm_gpu) / 1024**3
    print(f"  WM loaded. Peak VRAM: {mem:.1f} GB")
    return adapter


def _load_vlm_reward(args: Args):
    """Load VLM reward model."""
    from rlft.vlaw.reward.reward_model import VLAWRewardModel, VLAWRewardConfig

    vlm_config = VLAWRewardConfig(
        model_path=str(Path(_ROOT) / args.vlm_model_path),
        device=f"cuda:{args.vlm_gpu}",
        torch_dtype="bfloat16",
        num_frames=16,
        use_video_format=True,
    )
    reward_model = VLAWRewardModel(vlm_config)
    lora_path = str(Path(_ROOT) / args.vlm_lora_path)
    reward_model.load_model(lora_path=lora_path)

    mem = torch.cuda.max_memory_allocated(args.vlm_gpu) / 1024**3
    print(f"  VLM loaded. Peak VRAM: {mem:.1f} GB")
    return reward_model


def _load_state_predictor(args: Args):
    """Load StatePredictor model."""
    from rlft.vlaw.policy.state_predictor import StatePredictor

    sp = StatePredictor(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )
    ckpt_path = Path(_ROOT) / args.state_predictor_ckpt
    if ckpt_path.exists():
        state_dict = torch.load(str(ckpt_path), map_location="cpu")
        if "model_state_dict" in state_dict:
            sp.load_state_dict(state_dict["model_state_dict"])
        else:
            sp.load_state_dict(state_dict)
        print(f"  State Predictor loaded from {ckpt_path}")
    else:
        print(f"  ⚠️ State Predictor not found at {ckpt_path}, using random init")

    sp.to(f"cuda:{args.wm_gpu}").eval()
    return sp


# =====================================================================
# Simple Replay Buffer (single-env, no offline/online split)
# =====================================================================

class SimpleReplayBuffer:
    """Fixed-size ring buffer for single-env imagination RL."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """Add single transition."""
        idx = self.ptr
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """Sample random batch."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]).float().to(self.device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).float().to(self.device),
            "actions": torch.from_numpy(self.actions[idx]).float().to(self.device),
            "rewards": torch.from_numpy(self.rewards[idx]).float().to(self.device),
            "dones": torch.from_numpy(self.dones[idx]).float().to(self.device),
        }


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_imagination(
    env,
    agent,
    num_episodes: int,
    device: str,
) -> dict:
    """Evaluate SAC agent in ImaginationRLEnv.

    Returns:
        dict with episode stats (avg_reward, avg_length, avg_p_yes, etc.)
    """
    agent.eval()
    ep_rewards = []
    ep_lengths = []
    ep_p_yes_max = []
    ep_p_yes_last = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        p_yes_max = 0.0
        p_yes_last = 0.0
        step = 0

        while True:
            obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            action = agent.select_action(obs_t, deterministic=True).cpu().numpy().squeeze(0)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            if info.get("is_vlm_step", False):
                p = info.get("p_yes", 0.0)
                p_yes_max = max(p_yes_max, p)
                p_yes_last = p
            step += 1

            if terminated or truncated:
                break

        ep_rewards.append(total_reward)
        ep_lengths.append(step)
        ep_p_yes_max.append(p_yes_max)
        ep_p_yes_last.append(p_yes_last)

    return {
        "avg_reward": np.mean(ep_rewards),
        "std_reward": np.std(ep_rewards),
        "avg_length": np.mean(ep_lengths),
        "avg_p_yes_max": np.mean(ep_p_yes_max),
        "avg_p_yes_last": np.mean(ep_p_yes_last),
        "max_p_yes": np.max(ep_p_yes_max) if ep_p_yes_max else 0.0,
    }


# =====================================================================
# Main training loop
# =====================================================================

def main():
    if HAS_TYRO:
        args = tyro.cli(Args)
    else:
        args = Args()

    if args.exp_name is None:
        args.exp_name = (
            f"imagination-sac-{args.task_id}-"
            f"utd{args.utd_ratio}-lr{args.learning_rate}-seed{args.seed}"
        )
    run_name = f"{args.exp_name}__{int(time.time())}"
    log_dir = Path(_ROOT) / "runs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "checkpoints").mkdir(exist_ok=True)

    with open(log_dir / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, tuple) else v
                    for k, v in vars(args).items()}, indent=2, fp=f)

    # ---- seed ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.agent_gpu}" if args.cuda else "cpu")

    # ---- wandb ----
    if args.track and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )

    # =================================================================
    # 1. Create environment
    # =================================================================
    print("=" * 60)
    print(f"[{ALGO_NAME}] Creating ImaginationRLEnv ...")
    print("=" * 60)

    t0 = time.time()
    env, env_info = create_imagination_env(args)
    env_setup_time = time.time() - t0
    print(f"\n  Environment ready in {env_setup_time:.1f}s | mode={env_info['mode']}")

    obs_dim = int(np.prod(args.latent_shape)) + args.state_dim  # 4608 + 25 = 4633
    act_dim = args.action_dim  # 7

    print(f"  Obs dim: {obs_dim} | Action dim: {act_dim}")
    print(f"  Obs space: {env.observation_space}")
    print(f"  Act space: {env.action_space}")

    # =================================================================
    # 2. Create SAC agent
    # =================================================================
    print(f"\n[{ALGO_NAME}] Creating SAC agent ...")

    from rlft.algorithms.online_rl.pld_sac import PLDSACAgent

    hidden_dims = [args.layer_size] * args.num_layers

    agent = PLDSACAgent(
        obs_dim=obs_dim,
        act_steps=1,          # Single-step actions (not chunked)
        action_dim=act_dim,
        action_scale=1.0,     # Full action space (not residual)
        hidden_dims=hidden_dims,
        num_qs=args.num_qs,
        gamma=args.gamma,
        tau=args.tau,
        init_temperature=args.init_temperature,
        target_entropy=args.target_entropy,
        log_std_init=args.log_std_init,
        use_layer_norm=args.use_layer_norm,
        device=str(device),
    ).to(device)

    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.learning_rate)
    temp_optimizer = optim.Adam([agent.log_alpha], lr=args.learning_rate)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Agent parameters: {total_params / 1e6:.2f} M")
    print(f"  Architecture: {args.num_layers}×{args.layer_size}")
    print(f"  action_scale=1.0, act_steps=1 (direct action SAC)")

    # =================================================================
    # 3. Replay buffer
    # =================================================================
    buffer = SimpleReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=obs_dim,
        action_dim=act_dim,
        device=str(device),
    )

    # =================================================================
    # 4. Training loop
    # =================================================================
    print(f"\n{'='*60}")
    print(f"[{ALGO_NAME}] Starting training — {args.total_timesteps} steps")
    print(f"  UTD={args.utd_ratio} | LR={args.learning_rate} | "
          f"gamma={args.gamma} | tau={args.tau} | batch={args.batch_size}")
    print(f"  Seed steps: {args.num_seed_steps}")
    print(f"{'='*60}\n")

    obs, info = env.reset(seed=args.seed)
    total_steps = 0
    episode_count = 0
    episode_reward = 0.0
    episode_length = 0
    episode_p_yes_max = 0.0

    # Tracking
    ep_rewards_history = []
    ep_lengths_history = []
    ep_p_yes_history = []
    training_metrics = defaultdict(list)

    best_eval_reward = -float("inf")
    train_start = time.time()

    # Timing measurement
    step_times = []
    wm_call_count = 0

    pbar = tqdm(total=args.total_timesteps, desc=ALGO_NAME)

    while total_steps < args.total_timesteps:
        step_start = time.time()

        # ---- Select action ----
        if total_steps < args.num_seed_steps:
            # Random exploration
            action = env.action_space.sample()
        else:
            agent.eval()
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                action = agent.select_action(obs_t, deterministic=False)
                action = action.cpu().numpy().squeeze(0)

        # ---- Step environment ----
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ---- Store transition ----
        buffer.add(obs, action, reward, next_obs, done)

        episode_reward += reward
        episode_length += 1
        if info.get("is_vlm_step", False):
            p = info.get("p_yes", 0.0)
            episode_p_yes_max = max(episode_p_yes_max, p)

        obs = next_obs
        total_steps += 1
        pbar.update(1)

        step_times.append(time.time() - step_start)

        # ---- Episode done ----
        if done:
            ep_rewards_history.append(episode_reward)
            ep_lengths_history.append(episode_length)
            ep_p_yes_history.append(episode_p_yes_max)

            if total_steps % args.log_freq == 0 or episode_count < 5:
                elapsed = time.time() - train_start
                avg_step_time = np.mean(step_times[-100:]) if step_times else 0
                print(
                    f"  [Ep {episode_count}] steps={total_steps} | "
                    f"ep_rew={episode_reward:.3f} | ep_len={episode_length} | "
                    f"p_yes_max={episode_p_yes_max:.3f} | "
                    f"avg_step={avg_step_time:.2f}s | "
                    f"elapsed={elapsed:.0f}s"
                )

            # Log to wandb
            if args.track and HAS_WANDB:
                wandb.log({
                    "episode/reward": episode_reward,
                    "episode/length": episode_length,
                    "episode/p_yes_max": episode_p_yes_max,
                    "episode/count": episode_count,
                }, step=total_steps)

            episode_count += 1
            episode_reward = 0.0
            episode_length = 0
            episode_p_yes_max = 0.0
            obs, info = env.reset()

        # ---- Training updates (UTD) ----
        if total_steps >= args.num_seed_steps and buffer.size >= args.batch_size:
            agent.train()
            n_updates = min(args.utd_ratio, 1)  # At least 1 update per step
            if total_steps % max(1, 1) == 0:  # Update every step
                n_updates = args.utd_ratio

            for _ in range(n_updates):
                batch = buffer.sample(args.batch_size)

                # Critic update
                critic_optimizer.zero_grad()
                critic_loss, c_met = agent.compute_critic_loss(
                    batch["obs"], batch["actions"], batch["next_obs"],
                    batch["rewards"], batch["dones"],
                )
                critic_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                critic_optimizer.step()

                for k, v in c_met.items():
                    training_metrics[f"critic/{k}"].append(v)

                # Actor update
                actor_optimizer.zero_grad()
                actor_loss, a_met = agent.compute_actor_loss(batch["obs"])
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                actor_optimizer.step()

                for k, v in a_met.items():
                    training_metrics[f"actor/{k}"].append(v)

                # Temperature update
                temp_optimizer.zero_grad()
                temp_loss, t_met = agent.compute_temperature_loss(batch["obs"])
                temp_loss.backward()
                temp_optimizer.step()

                for k, v in t_met.items():
                    training_metrics[f"temp/{k}"].append(v)

                # Target network update
                agent.update_target()

        # ---- Logging ----
        if total_steps % args.log_freq == 0 and training_metrics:
            log_dict = {}
            for mk, mv in training_metrics.items():
                val = np.mean(mv)
                log_dict[f"train/{mk}"] = val

            log_dict["train/buffer_size"] = buffer.size
            log_dict["train/steps_per_sec"] = 1.0 / np.mean(step_times[-100:]) if step_times else 0
            log_dict["train/avg_ep_reward"] = np.mean(ep_rewards_history[-10:]) if ep_rewards_history else 0

            if args.track and HAS_WANDB:
                wandb.log(log_dict, step=total_steps)

            training_metrics.clear()

        # ---- Evaluation ----
        if total_steps % args.eval_freq == 0 and total_steps > 0:
            print(f"\n  [Eval @ step {total_steps}]")
            eval_metrics = evaluate_imagination(
                env, agent, args.num_eval_episodes, str(device),
            )
            print(
                f"    avg_reward={eval_metrics['avg_reward']:.3f} ± "
                f"{eval_metrics['std_reward']:.3f} | "
                f"avg_len={eval_metrics['avg_length']:.0f} | "
                f"avg_p_yes_max={eval_metrics['avg_p_yes_max']:.3f} | "
                f"max_p_yes={eval_metrics['max_p_yes']:.3f}"
            )

            if args.track and HAS_WANDB:
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=total_steps)

            if eval_metrics["avg_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["avg_reward"]
                _save_checkpoint(
                    log_dir / "checkpoints" / "best.pt",
                    agent, args, total_steps, eval_metrics,
                )
                print(f"    ✓ New best! (avg_reward={best_eval_reward:.3f})")

            # Reset env after eval (eval may have consumed the env state)
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_p_yes_max = 0.0

        # ---- Periodic checkpoint ----
        if total_steps % args.save_freq == 0 and total_steps > 0:
            _save_checkpoint(
                log_dir / "checkpoints" / f"step_{total_steps}.pt",
                agent, args, total_steps,
            )

        pbar.set_postfix({
            "ep_rew": f"{np.mean(ep_rewards_history[-10:]):.2f}" if ep_rewards_history else "N/A",
            "p_yes": f"{np.mean(ep_p_yes_history[-10:]):.3f}" if ep_p_yes_history else "N/A",
            "buf": buffer.size,
        })

    pbar.close()
    total_time = time.time() - train_start

    # =================================================================
    # Final save & summary
    # =================================================================
    _save_checkpoint(
        log_dir / "checkpoints" / "final.pt",
        agent, args, total_steps,
    )

    # Also save to canonical location
    canon_dir = Path(_ROOT) / args.save_dir
    canon_dir.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(
        canon_dir / "imagination_sac_10k.pt",
        agent, args, total_steps,
    )

    # Summary
    summary = {
        "total_steps": total_steps,
        "total_episodes": episode_count,
        "total_time_s": total_time,
        "steps_per_sec": total_steps / total_time if total_time > 0 else 0,
        "avg_step_time_s": np.mean(step_times) if step_times else 0,
        "best_eval_reward": best_eval_reward,
        "final_avg_ep_reward": np.mean(ep_rewards_history[-10:]) if ep_rewards_history else 0,
        "final_avg_p_yes": np.mean(ep_p_yes_history[-10:]) if ep_p_yes_history else 0,
        "final_max_p_yes": np.max(ep_p_yes_history) if ep_p_yes_history else 0,
        "ep_rewards": ep_rewards_history,
        "ep_p_yes": ep_p_yes_history,
    }

    with open(log_dir / "training_summary.json", "w") as f:
        json.dump({k: v if not isinstance(v, np.floating) else float(v)
                    for k, v in summary.items()}, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"[{ALGO_NAME}] Training complete!")
    print(f"  Total steps:   {total_steps}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Total time:    {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Steps/sec:      {total_steps/total_time:.1f}")
    print(f"  Best eval rew:  {best_eval_reward:.3f}")
    print(f"  Final avg rew:  {summary['final_avg_ep_reward']:.3f}")
    print(f"  Final p_yes:    {summary['final_avg_p_yes']:.3f}")
    print(f"  Log dir:        {log_dir}")
    print(f"  Checkpoint:     {canon_dir}/imagination_sac_10k.pt")
    print(f"{'='*60}")

    if args.track and HAS_WANDB:
        wandb.log({"final/best_eval_reward": best_eval_reward})
        wandb.finish()

    env.close()
    return summary


def _save_checkpoint(
    path: Path,
    agent,
    args: Args,
    total_steps: int,
    eval_metrics: dict = None,
):
    """Save agent checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "agent": agent.state_dict(),
        "total_steps": total_steps,
        "config": {k: str(v) if isinstance(v, tuple) else v
                    for k, v in vars(args).items()},
    }
    if eval_metrics:
        ckpt["eval_metrics"] = eval_metrics
    torch.save(ckpt, str(path))


if __name__ == "__main__":
    main()
