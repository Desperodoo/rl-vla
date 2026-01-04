"""
Stage 1: Offline Advantage-Weighted MLE Training for Latent Policy

This script trains the latent steering policy using offline data and
pretrained Q-network. The flow/shortcut policy remains frozen.

NOTE: Stage 1 is identical between on-policy and off-policy DSRL because
AW-MLE is inherently off-policy (uses pretrained Q-network for scoring).
This script is provided for consistency and self-contained off-policy pipeline.

Key idea:
- Sample M latent candidates from prior N(0, I)
- Generate actions via frozen flow policy
- Score actions with ensemble Q-network (UCB: μ - κσ)
- Compute softmax advantage weights
- Train latent policy with weighted MLE

Aligns with dsrl/train_latent_offline.py for unified data loading and evaluation.

Usage:
    python train_stage1_offline.py --env_id LiftPegUpright-v1 \
        --demo_path ~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
        --awsc_checkpoint ../dsrl/checkpoints/best_eval_success_once.pt
"""

ALGO_NAME = "DSRL_OffPolicy_Stage1"

import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import tyro

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

# Add parent dir for imports
_root = Path(__file__).parent.parent.parent
_dsrl_offpolicy = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl"))
sys.path.insert(0, str(_dsrl_offpolicy.parent))  # Add parent so dsrl_offpolicy is importable

from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.evaluate import evaluate
from diffusion_policy.utils import (
    AgentWrapper,
    IterationBasedBatchSampler,
    build_state_obs_extractor,
    convert_obs,
    worker_init_fn,
    encode_observations,
)
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.algorithms.networks import DoubleQNetwork
from diffusion_policy.rlpd.networks import EnsembleQNetwork

# Import from diffusion_policy for unified data loading
try:
    from train_offline_rl import OfflineRLDataset
except ImportError:
    # Fallback: define minimal dataset if not available
    from diffusion_policy.train_offline_rl import OfflineRLDataset

# Import from on-policy DSRL (shared components)
from latent_policy import LatentGaussianPolicy

# Import off-policy specific components
from dsrl_offpolicy.agents.dsrl_sac_agent import DSRLSACAgent
from dsrl_offpolicy.models.latent_q_network import DoubleLatentQNetwork


@dataclass
class Args:
    # Experiment settings
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "maniskill_dsrl_offpolicy"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""

    # Environment settings
    env_id: str = "LiftPegUpright-v1"
    """the id of the environment"""
    demo_path: str = "~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"
    """the path of demo dataset"""
    num_demos: Optional[int] = 745
    """number of trajectories to load from the demo dataset"""
    max_episode_steps: Optional[int] = 100
    """max episode steps for evaluation"""
    control_mode: str = "pd_ee_delta_pose"
    """the control mode"""
    obs_mode: str = "rgb"
    """observation mode: state or rgb"""
    sim_backend: str = "physx_cuda"
    """simulation backend for evaluation"""

    # Pretrained checkpoint (from AW-ShortCut Flow training)
    awsc_checkpoint: str = "/home/lizh/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
    """path to pretrained AW-ShortCut Flow checkpoint (contains velocity_net and q_network)"""
    use_ema: bool = True
    """use EMA weights from checkpoint (recommended for better performance)"""
    
    # Training settings
    total_iters: int = 10240
    """total training iterations"""
    batch_size: int = 256
    """batch size"""
    lr: float = 1e-4
    """learning rate for latent policy"""

    # Observation/Action horizons
    obs_horizon: int = 2
    """observation horizon"""
    act_horizon: int = 8
    """action execution horizon"""
    pred_horizon: int = 16
    """action prediction horizon"""
    
    # Visual encoder settings
    visual_feature_dim: int = 256
    """visual encoder output dimension"""
    diffusion_step_embed_dim: int = 64
    """timestep embedding dimension (for velocity net)"""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """U-Net channel dimensions"""
    n_groups: int = 8
    """GroupNorm groups"""
    
    # Latent policy architecture
    latent_hidden_dims: List[int] = field(default_factory=lambda: [2048, 2048, 2048])
    """hidden dimensions for latent policy MLP"""
    steer_mode: str = "full"
    """steering mode: 'full' for all timesteps, 'act_horizon' for first act_horizon only"""
    state_dependent_std: bool = True
    """whether to use state-dependent std for latent policy"""
    
    # Latent Q-network architecture (for Q^W in latent space)
    latent_q_hidden_dims: List[int] = field(default_factory=lambda: [2048, 2048, 2048])
    """hidden dimensions for latent Q-network MLP (for Q^W distillation)"""
    
    # Ensemble Q settings (must match pretrained checkpoint)
    use_double_q: bool = True
    """use DoubleQNetwork (default for AWSC) instead of EnsembleQNetwork"""
    num_qs: int = 10
    """number of Q-networks in ensemble (only used if use_double_q=False)"""
    num_min_qs: int = 2
    """number of Q-networks for subsample + min (only used if use_double_q=False)"""
    q_hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    """hidden dimensions for Q-network MLP"""
    
    # Stage 1 DSRL-NA hyperparameters
    num_candidates: int = 32
    """M: number of latent candidates to sample (for AWMLE scoring)"""
    tau: float = 5.0
    """soft baseline temperature for advantage computation (AWMLE)"""
    beta_latent: float = 1.0
    """temperature for advantage weighting softmax (AWMLE)"""
    advantage_clip: float = 20.0
    """maximum advantage value for numerical stability"""
    action_magnitude: float = 1.5
    """TanhNormal output range [-magnitude, magnitude]. Official DSRL: 1.0 (Libero), 2.0 (Aloha), 2.5 (real)"""
    
    # Flow inference
    num_inference_steps: int = 8
    """number of flow integration steps for action generation"""
    gamma: float = 0.99
    """discount factor for SMDP"""
    
    # Offline algorithm settings
    offline_algo: str = "sac"
    """policy update algorithm: 'awmle' (Advantage-Weighted MLE) or 'sac' (SAC-style)"""
    qw_warmup_iters: int = 128
    """number of iterations to warmup Q^W before updating policy (0 = no warmup)"""
    qw_distill_coef: float = 1.0
    """coefficient for Q^W distillation loss"""

    # Logging settings
    log_freq: int = 1
    """logging frequency"""
    eval_freq: int = 64
    """evaluation frequency"""
    save_freq: Optional[int] = None
    """checkpoint save frequency"""
    num_eval_episodes: int = 50
    """number of evaluation episodes"""
    num_eval_envs: int = 50
    """number of parallel eval environments"""
    num_dataload_workers: int = 0
    """dataloader workers"""


def create_dsrl_sac_agent(args, action_dim: int, global_cond_dim: int) -> DSRLSACAgent:
    """Create DSRL SAC agent for Stage 1 with Q^W distillation."""
    device = "cuda" if args.cuda else "cpu"
    
    # Create velocity network
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=tuple(args.unet_dims),
        n_groups=args.n_groups,
    )
    
    # Create Q-network (for action scoring)
    if args.use_double_q:
        q_network = DoubleQNetwork(
            action_dim=action_dim,
            obs_dim=global_cond_dim,
            action_horizon=args.act_horizon,
            hidden_dims=args.q_hidden_dims,
        )
    else:
        q_network = EnsembleQNetwork(
            action_dim=action_dim,
            obs_dim=global_cond_dim,
            action_horizon=args.act_horizon,
            hidden_dims=args.q_hidden_dims,
            num_qs=args.num_qs,
            num_min_qs=args.num_min_qs,
        )
    
    # Create latent policy (trainable)
    latent_policy = LatentGaussianPolicy(
        obs_dim=global_cond_dim,
        pred_horizon=args.pred_horizon,
        action_dim=action_dim,
        hidden_dims=args.latent_hidden_dims,
        steer_mode=args.steer_mode,
        act_horizon=args.act_horizon,
        state_dependent_std=args.state_dependent_std,
        action_magnitude=args.action_magnitude,
    )
    
    # Create latent Q-network for Q^W distillation (REQUIRED for DSRL-NA)
    from dsrl_offpolicy.models.latent_q_network import DoubleLatentQNetwork
    latent_q_network = DoubleLatentQNetwork(
        obs_dim=global_cond_dim,
        pred_horizon=args.pred_horizon,
        action_dim=action_dim,
        hidden_dims=args.latent_q_hidden_dims,
        steer_mode=args.steer_mode,
        act_horizon=args.act_horizon,
        tau=0.005,
    )
    
    # Create DSRL SAC agent
    agent = DSRLSACAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        latent_policy=latent_policy,
        latent_q_network=latent_q_network,
        action_dim=action_dim,
        obs_dim=global_cond_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        act_horizon=args.act_horizon,
        num_inference_steps=args.num_inference_steps,
        num_candidates=args.num_candidates,
        tau=args.tau,
        beta_latent=args.beta_latent,
        advantage_clip=args.advantage_clip,
        qw_distill_coef=args.qw_distill_coef,
        action_magnitude=args.action_magnitude,
        device=device,
    )
    
    return agent


def load_pretrained_checkpoint(
    agent: DSRLSACAgent,
    checkpoint_path: str,
    visual_encoder: Optional[nn.Module],
    device: str,
    use_ema: bool = True,
):
    """Load pretrained velocity net and Q-network from AWShortCutFlow checkpoint.
    
    Aligned with on-policy load_pretrained_checkpoint for consistency.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading pretrained checkpoint from {checkpoint_path}")
    print(f"  Using EMA weights: {use_ema}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get agent state from checkpoint
    if use_ema:
        if "ema_agent" not in checkpoint:
            raise ValueError(
                f"use_ema=True but checkpoint does not contain 'ema_agent' key. "
                f"Available keys: {list(checkpoint.keys())}."
            )
        agent_state = checkpoint["ema_agent"]
        print("  Loading from ema_agent")
    else:
        agent_state = checkpoint.get("agent", checkpoint)
        print("  Loading from agent")
    
    # Extract velocity_net weights
    velocity_net_state = {}
    q_network_state = {}
    
    for key, value in agent_state.items():
        if key.startswith("velocity_net."):
            velocity_net_state[key.replace("velocity_net.", "")] = value
        elif key.startswith("q_network.") or key.startswith("critic."):
            new_key = key.replace("q_network.", "").replace("critic.", "")
            q_network_state[new_key] = value
    
    # Load velocity net
    if velocity_net_state:
        agent.velocity_net.load_state_dict(velocity_net_state)
        agent.velocity_net_ema.load_state_dict(agent.velocity_net.state_dict())
        print(f"  Loaded velocity_net ({len(velocity_net_state)} keys)")
        print("  Synced velocity_net_ema from velocity_net")
    else:
        raise ValueError("No velocity_net weights found in checkpoint")
    
    # Load Q-network
    if q_network_state:
        try:
            agent.q_network.load_state_dict(q_network_state)
            print(f"  Loaded q_network ({len(q_network_state)} keys)")
        except Exception as e:
            raise ValueError(f"Failed to load q_network: {e}")
    else:
        raise ValueError("No q_network weights found in checkpoint")
    
    # Load visual encoder if present
    if visual_encoder is not None and "visual_encoder" in checkpoint:
        visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        print("  Loaded visual_encoder")


def save_ckpt(run_name: str, tag: str, agent: DSRLSACAgent, visual_encoder: Optional[nn.Module] = None, args=None):
    """Save checkpoint in unified format."""
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ckpt = {
        "latent_policy": agent.latent_policy.state_dict(),
        "agent": agent.state_dict(),
    }
    if visual_encoder is not None:
        ckpt["visual_encoder"] = visual_encoder.state_dict()
    if agent.latent_q_network is not None:
        ckpt["latent_q_network"] = agent.latent_q_network.state_dict()
    if args is not None:
        ckpt["args"] = vars(args)
    torch.save(ckpt, f"runs/{run_name}/checkpoints/{tag}.pt")


class DSRLSACAgentWrapper(AgentWrapper):
    """Wrapper for DSRL SAC agent evaluation."""
    
    def get_action(self, obs_seq, **kwargs):
        """Get action from observations using latent policy steering."""
        with torch.no_grad():
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            
            action_seq, _, _ = self.agent.get_action(
                obs_cond,
                use_latent_policy=True,
                deterministic=True,
            )
            
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            return action_seq[:, start:end]


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.exp_name is None:
        args.exp_name = f"dsrl_offpolicy_stage1-{args.env_id}-seed{args.seed}"
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # Set up logging
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
    
    # Initialize tracking
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
            group="dsrl_offpolicy_stage1",
            tags=["dsrl", "offpolicy", "stage1", "awmle"],
        )
    
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========== Load Dataset ==========
    print(f"Loading dataset from {args.demo_path}")
    include_rgb = "rgb" in args.obs_mode
    
    dataset = OfflineRLDataset(
        data_path=args.demo_path,
        include_rgb=include_rgb,
        device=device,
        num_traj=args.num_demos,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        act_horizon=args.act_horizon,
        control_mode=args.control_mode,
        env_id=args.env_id,
        gamma=args.gamma,
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )
    
    # Get dimensions from dataset
    sample_obs = dataset.trajectories["observations"][0]
    state_dim = sample_obs["state"].shape[-1]
    action_dim = dataset.trajectories["actions"][0].shape[-1]
    
    print(f"Action dimension: {action_dim}")
    print(f"State dimension: {state_dim}")
    
    # ========== Setup Visual Encoder ==========
    visual_encoder = None
    visual_feature_dim = 0
    
    if include_rgb:
        in_channels = 3
        visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
        visual_feature_dim = args.visual_feature_dim
    
    global_cond_dim = args.obs_horizon * (visual_feature_dim + state_dim)
    print(f"State dim: {state_dim}, Visual dim: {visual_feature_dim}, Total obs dim: {global_cond_dim}")
    
    # ========== Create Agent ==========
    agent = create_dsrl_sac_agent(args, action_dim, global_cond_dim).to(device)
    print(f"DSRL SAC Agent parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M")
    print(f"Latent policy parameters: {sum(p.numel() for p in agent.latent_policy.parameters()) / 1e6:.2f}M")
    
    # Load pretrained checkpoint
    load_pretrained_checkpoint(
        agent, args.awsc_checkpoint,
        visual_encoder, str(device), use_ema=args.use_ema
    )
    
    # Freeze velocity net and Q-network
    for param in agent.velocity_net.parameters():
        param.requires_grad = False
    for param in agent.q_network.parameters():
        param.requires_grad = False
    if visual_encoder is not None:
        for param in visual_encoder.parameters():
            param.requires_grad = False
    
    # ========== Create Evaluation Environments ==========
    print("Creating evaluation environments...")
    eval_env_kwargs = dict(
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        reward_mode="dense",
    )
    eval_other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    eval_envs = make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        sim_backend=args.sim_backend,
        env_kwargs=eval_env_kwargs,
        other_kwargs=eval_other_kwargs,
        video_dir=f"{log_dir}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    
    # Create wrapper for evaluation
    agent_wrapper = DSRLSACAgentWrapper(
        agent, visual_encoder, include_rgb,
        args.obs_horizon, args.act_horizon
    ).to(device)
    
    # ========== Create Optimizer ==========
    # Trainable parameters: latent policy + Q^W network (Q^W is now required)
    trainable_params = list(agent.latent_policy.parameters())
    trainable_params += agent.get_critic_parameters()
    print(f"Latent Q-network parameters: {sum(p.numel() for p in agent.get_critic_parameters()) / 1e6:.2f}M")
    
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=1e-5,
    )
    
    # Print training mode
    print(f"\nTraining Mode (DSRL-NA):")
    print(f"  Offline algorithm: {args.offline_algo}")
    print(f"  Q^W warmup iterations: {args.qw_warmup_iters}")
    print(f"  Q^W distill coefficient: {args.qw_distill_coef}")
    
    # ========== Training Loop ==========
    best_success_rate = 0.0
    best_success_at_end = 0.0
    
    print("\n" + "=" * 50)
    print(f"Starting Stage 1 Offline Training ({args.offline_algo.upper()})...")
    if args.qw_warmup_iters > 0:
        print(f"  Phase 1: Q^W warmup for {args.qw_warmup_iters} iterations")
        print(f"  Phase 2: Full training for {args.total_iters - args.qw_warmup_iters} iterations")
    print("=" * 50 + "\n")
    
    pbar = tqdm(dataloader, total=args.total_iters, desc="Training")
    
    for iteration, batch in enumerate(pbar):
        # Extract observations dict (batch["observations"] contains {"state": ..., "rgb": ...})
        obs_seq = batch["observations"]
        
        # Encode observations
        obs_cond = encode_observations(obs_seq, visual_encoder, include_rgb, device)
        
        # Determine training phase
        update_actor = (iteration >= args.qw_warmup_iters)
        
        # Compute loss using Q^W (DSRL-NA style)
        agent.latent_policy.train()
        agent.latent_q_network.train()
        
        loss_dict = agent.compute_offline_loss(
            obs_cond,
            algo=args.offline_algo,
            update_actor=update_actor,
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        
        # Logging
        if iteration % args.log_freq == 0:
            train_log = {
                "train/loss": loss_dict["loss"].item(),
                "train/qw_distill_loss": loss_dict["qw_distill_loss"].item(),
                "train/actor_loss": loss_dict["actor_loss"].item(),
                "train/qa_mean": loss_dict["qa_mean"].item(),
                "train/qw_mean": loss_dict["qw_mean"].item(),
                "train/eff_num": loss_dict["eff_num"].item(),
                "train/entropy": loss_dict["entropy"].item(),
                "train/log_std_mean": loss_dict["log_std_mean"].item(),
                "train/update_actor": float(update_actor),
                "train/phase": "full" if update_actor else "qw_warmup",
            }
            
            for key, value in train_log.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, iteration)
            
            if args.track:
                import wandb
                wandb.log(train_log, step=iteration)
            
            phase = "FULL" if update_actor else "QW"
            pbar.set_postfix({
                "phase": phase,
                "loss": f"{loss_dict['loss'].item():.4f}",
                "qw_d": f"{loss_dict['qw_distill_loss'].item():.3f}",
                "actor": f"{loss_dict['actor_loss'].item():.3f}",
                "qa": f"{loss_dict['qa_mean'].item():.2f}",
            })
        
        # Evaluation
        if iteration % args.eval_freq == 0:
            agent.latent_policy.eval()
            
            eval_metrics = evaluate(
                args.num_eval_episodes, agent_wrapper, eval_envs, device, args.sim_backend
            )
            
            phase_str = "Full" if update_actor else "Q^W Warmup"
            print(f"\nEvaluation at iteration {iteration} ({phase_str}):")
            eval_log = {}
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                eval_log[f"eval/{k}"] = eval_metrics[k]
                print(f"  {k}: {eval_metrics[k]:.4f}")
            
            if args.track:
                import wandb
                wandb.log(eval_log, step=iteration)
            
            # Save best checkpoints
            success_once = eval_metrics.get("success_once", 0)
            success_at_end = eval_metrics.get("success_at_end", 0)
            
            if success_once > best_success_rate:
                best_success_rate = success_once
                save_ckpt(run_name, "best_eval_success_once", agent, visual_encoder, args)
                print(f"  New best success_once! {success_once:.2%}")
            
            if success_at_end > best_success_at_end:
                best_success_at_end = success_at_end
                save_ckpt(run_name, "best_eval_success_at_end", agent, visual_encoder, args)
                print(f"  New best success_at_end! {success_at_end:.2%}")
        
        # Save checkpoint
        if args.save_freq and iteration % args.save_freq == 0 and iteration > 0:
            save_ckpt(run_name, f"iter_{iteration}", agent, visual_encoder, args)
    
    # Final evaluation
    agent.latent_policy.eval()
    eval_metrics = evaluate(
        args.num_eval_episodes, agent_wrapper, eval_envs, device, args.sim_backend
    )
    
    print(f"\nFinal evaluation:")
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])
        writer.add_scalar(f"eval/{k}", eval_metrics[k], args.total_iters)
        print(f"  {k}: {eval_metrics[k]:.4f}")
    
    # Save final checkpoint
    save_ckpt(run_name, "final", agent, visual_encoder, args)
    print(f"\nTraining complete! Final checkpoint saved to {log_dir}/checkpoints/final.pt")
    print(f"Best success_once: {best_success_rate:.2%}")
    print(f"Best success_at_end: {best_success_at_end:.2%}")
    
    eval_envs.close()
    writer.close()
    
    if args.track:
        import wandb
        wandb.finish()
