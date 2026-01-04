"""
Stage 1: Offline Advantage-Weighted MLE Training for Latent Policy

This script trains the latent steering policy using offline data and
pretrained Q-network. The flow/shortcut policy remains frozen.

Key idea:
- Sample M latent candidates from prior N(0, I)
- Generate actions via frozen flow policy
- Score actions with ensemble Q-network (UCB: μ - κσ)
- Compute softmax advantage weights
- Train latent policy with weighted MLE

Aligns with train_offline_rl.py for unified data loading, observation encoding,
and checkpoint management.

Usage:
    python train_latent_offline.py --env_id LiftPegUpright-v1 \
        --demo_path ~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
        --awsc_checkpoint runs/aw_shortcut_flow-LiftPegUpright-v1/checkpoints/best_eval_success_once.pt
"""

ALGO_NAME = "DSRL_Stage1"

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
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))

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

# Import from train_offline_rl for unified data loading
from train_offline_rl import OfflineRLDataset

# Local imports
from dsrl_agent import DSRLAgent
from latent_policy import LatentGaussianPolicy
from value_network import ValueNetwork


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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
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
    awsc_checkpoint: str = ""
    """path to pretrained AW-ShortCut Flow checkpoint (contains velocity_net and q_network)"""
    use_ema: bool = True
    """use EMA weights from checkpoint (recommended for better performance)"""
    
    # Training settings
    total_iters: int = 100_000
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
    latent_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    """hidden dimensions for latent policy MLP"""
    steer_mode: str = "full"
    """steering mode: 'full' for all timesteps, 'act_horizon' for first act_horizon only"""
    state_dependent_std: bool = True
    """whether to use state-dependent std for latent policy"""
    
    # Ensemble Q settings (must match pretrained checkpoint)
    use_double_q: bool = True
    """use DoubleQNetwork (default for AWSC) instead of EnsembleQNetwork"""
    num_qs: int = 10
    """number of Q-networks in ensemble (only used if use_double_q=False)"""
    num_min_qs: int = 2
    """number of Q-networks for subsample + min (only used if use_double_q=False)"""
    q_hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    """hidden dimensions for Q-network MLP"""
    
    # Stage 1 AW-MLE hyperparameters
    num_candidates: int = 32
    """M: number of latent candidates to sample"""
    kappa: float = 1.0
    """UCB coefficient for Q-value aggregation (μ - κσ)"""
    use_ucb: bool = False
    """use UCB aggregation (μ - κσ) instead of simple mean for Q-value scoring"""
    tau: float = 5.0
    """soft baseline temperature for advantage computation"""
    beta_latent: float = 1.0
    """temperature for advantage weighting softmax"""
    advantage_clip: float = 20.0
    """maximum advantage value for numerical stability"""
    kl_coef: float = 1e-3
    """KL-to-prior regularization coefficient"""
    
    # Adaptive KL settings
    adaptive_kl: bool = False
    """enable adaptive KL coefficient adjustment"""
    kl_target: float = 0.01
    """target KL divergence value for adaptive adjustment"""
    kl_adapt_factor: float = 1.5
    """multiplicative factor for KL coefficient adjustment"""
    kl_coef_min: float = 1e-5
    """minimum KL coefficient"""
    kl_coef_max: float = 1.0
    """maximum KL coefficient"""
    
    # Flow inference
    num_inference_steps: int = 8
    """number of flow integration steps for action generation"""
    gamma: float = 0.99
    """discount factor for SMDP"""

    # Logging settings
    log_freq: int = 1000
    """logging frequency"""
    eval_freq: int = 5000
    """evaluation frequency"""
    save_freq: Optional[int] = None
    """checkpoint save frequency"""
    num_eval_episodes: int = 100
    """number of evaluation episodes"""
    num_eval_envs: int = 10
    """number of parallel eval environments"""
    num_dataload_workers: int = 0
    """dataloader workers"""


def create_dsrl_agent(args, action_dim: int, global_cond_dim: int) -> DSRLAgent:
    """Create DSRL agent with pretrained velocity net and Q-network.
    
    Args:
        args: Training arguments
        action_dim: Action dimension from environment
        global_cond_dim: Global conditioning dimension (obs_horizon * feature_dim)
        
    Returns:
        Initialized DSRLAgent
    """
    device = "cuda" if args.cuda else "cpu"
    
    # Create velocity network (matches AWShortCutFlowAgent architecture)
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=tuple(args.unet_dims),
        n_groups=args.n_groups,
    )
    
    # Create Q-network
    if args.use_double_q:
        # DoubleQNetwork (default for AWSC checkpoints)
        q_network = DoubleQNetwork(
            action_dim=action_dim,
            obs_dim=global_cond_dim,
            action_horizon=args.act_horizon,
            hidden_dims=args.q_hidden_dims,
        )
    else:
        # EnsembleQNetwork (for RLPD fine-tuned checkpoints)
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
    )
    
    # Create DSRL agent (no value network for Stage 1)
    agent = DSRLAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        latent_policy=latent_policy,
        value_network=None,  # Not needed for Stage 1
        action_dim=action_dim,
        obs_dim=global_cond_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        act_horizon=args.act_horizon,
        num_inference_steps=args.num_inference_steps,
        num_candidates=args.num_candidates,
        kappa=args.kappa,
        use_ucb=args.use_ucb,
        tau=args.tau,
        beta_latent=args.beta_latent,
        advantage_clip=args.advantage_clip,
        kl_coef=args.kl_coef,
        device=device,
    )
    
    return agent


def load_pretrained_checkpoint(
    agent: DSRLAgent,
    checkpoint_path: str,
    visual_encoder: Optional[nn.Module],
    device: str,
    use_ema: bool = True,
):
    """Load pretrained velocity net and Q-network from AWShortCutFlow checkpoint.
    
    Args:
        agent: DSRLAgent to load weights into
        checkpoint_path: Path to AWShortCutFlow checkpoint
        visual_encoder: Optional visual encoder to load
        device: Device for loading
        use_ema: If True, load from ema_agent (recommended for better performance)
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("WARNING: No pretrained checkpoint provided, using random init!")
        return
    
    print(f"Loading pretrained checkpoint from {checkpoint_path}")
    print(f"  Using EMA weights: {use_ema}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats - prefer EMA weights
    if use_ema:
        if "ema_agent" not in checkpoint:
            raise ValueError(
                f"use_ema=True but checkpoint '{checkpoint_path}' does not contain 'ema_agent' key. "
                f"Available keys: {list(checkpoint.keys())}. "
                f"Either set use_ema=False or provide a checkpoint with EMA weights."
            )
        agent_state = checkpoint["ema_agent"]
        print("  Loading from ema_agent")
    elif "agent" in checkpoint:
        # Full checkpoint format (agent + ema_agent + visual_encoder)
        agent_state = checkpoint["agent"]
        print("  Loading from agent")
    elif "velocity_net" in checkpoint:
        # Direct component format
        agent_state = checkpoint
    else:
        # Try as direct state dict
        agent_state = checkpoint
    
    # Extract velocity_net weights
    velocity_net_state = {}
    q_network_state = {}
    
    for key, value in agent_state.items():
        if key.startswith("velocity_net."):
            velocity_net_state[key.replace("velocity_net.", "")] = value
        elif key.startswith("q_network.") or key.startswith("critic."):
            # Handle both q_network and critic naming
            new_key = key.replace("q_network.", "").replace("critic.", "")
            q_network_state[new_key] = value
    
    # Load velocity net
    if velocity_net_state:
        agent.velocity_net.load_state_dict(velocity_net_state)
        # Sync EMA velocity net (critical for correct inference)
        agent.velocity_net_ema.load_state_dict(agent.velocity_net.state_dict())
        print(f"  Loaded velocity_net ({len(velocity_net_state)} keys)")
        print("  Synced velocity_net_ema from velocity_net")
    else:
        print("  WARNING: No velocity_net keys found in checkpoint")
    
    # Load Q-network
    if q_network_state:
        try:
            agent.q_network.load_state_dict(q_network_state)
            print(f"  Loaded q_network ({len(q_network_state)} keys)")
        except Exception as e:
            print(f"  WARNING: Could not load q_network: {e}")
    else:
        print("  WARNING: No q_network keys found in checkpoint")
    
    # Load visual encoder
    if visual_encoder is not None and "visual_encoder" in checkpoint:
        visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        print("  Loaded visual_encoder")


def save_ckpt(run_name: str, tag: str, agent: DSRLAgent, visual_encoder: Optional[nn.Module] = None):
    """Save checkpoint in unified format.
    
    Args:
        run_name: Run directory name
        tag: Checkpoint tag (e.g., 'best_eval_success_once')
        agent: DSRLAgent model
        visual_encoder: Optional visual encoder model
    """
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ckpt = {
        "agent": agent.state_dict(),
        "latent_policy": agent.latent_policy.state_dict(),
    }
    if visual_encoder is not None:
        ckpt["visual_encoder"] = visual_encoder.state_dict()
    torch.save(ckpt, f"runs/{run_name}/checkpoints/{tag}.pt")


class DSRLAgentWrapper(AgentWrapper):
    """Wrapper for DSRL agent evaluation.
    
    Extends AgentWrapper to use latent policy steering during evaluation.
    """
    
    def get_action(self, obs_seq, **kwargs):
        """Get action from observations using latent policy steering."""
        with torch.no_grad():
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            
            # Use latent policy for evaluation (deterministic)
            action_seq, _, _ = self.agent.get_action(
                obs_cond,
                use_latent_policy=True,
                deterministic=True,
                use_prior_mixing=False,
            )
            
            # Only return act_horizon actions
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            return action_seq[:, start:end]


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.exp_name is None:
        args.exp_name = f"dsrl_stage1-{args.env_id}"
        run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1
    
    # Warning for experimental steer_mode
    if args.steer_mode == "act_horizon":
        import warnings
        warnings.warn(
            f"steer_mode='act_horizon' is experimental. Credit assignment may be suboptimal "
            f"compared to steer_mode='full' (default). Consider using 'full' mode for stability.",
            UserWarning
        )
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Create evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="dense",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
    )
    assert args.max_episode_steps is not None, "max_episode_steps must be specified"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    
    # Wandb tracking
    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_eval_envs,
            env_id=args.env_id,
            env_horizon=args.max_episode_steps,
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="dsrl_stage1",
            tags=["dsrl", "stage1", "offline"],
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Setup data processing
    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
        state_obs_extractor=build_state_obs_extractor(args.env_id),
    )
    
    # Get observation space from temp env
    tmp_env = gym.make(args.env_id, **env_kwargs)
    original_obs_space = tmp_env.observation_space
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    tmp_env.close()
    
    # Create dataset using unified OfflineRLDataset
    dataset = OfflineRLDataset(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=original_obs_space,
        include_rgb=include_rgb,
        device=device,
        num_traj=args.num_demos,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        act_horizon=args.act_horizon,
        control_mode=args.control_mode,
        gamma=args.gamma,
    )
    
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )
    
    # Get action dimension from environment
    action_dim = envs.single_action_space.shape[0]
    
    # Determine state dimension from dataset
    sample_obs = dataset.trajectories["observations"][0]
    state_dim = sample_obs["state"].shape[-1]
    
    # Create visual encoder
    include_rgb_flag = "rgb" in dataset.obs_keys
    
    in_channels = 0
    if include_rgb_flag:
        in_channels += 3
    
    visual_encoder = None
    visual_feature_dim = 0
    if in_channels > 0:
        visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
        visual_feature_dim = args.visual_feature_dim
    
    # Compute global conditioning dimension
    global_cond_dim = args.obs_horizon * (visual_feature_dim + state_dim)
    print(f"action_dim: {action_dim}, state_dim: {state_dim}, visual_feature_dim: {visual_feature_dim}")
    print(f"global_cond_dim: {global_cond_dim} = {args.obs_horizon} * ({visual_feature_dim} + {state_dim})")
    
    # Create DSRL agent
    agent = create_dsrl_agent(args, action_dim, global_cond_dim).to(device)
    print(f"DSRL Agent parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M")
    print(f"  Latent policy params: {sum(p.numel() for p in agent.latent_policy.parameters()) / 1e6:.2f}M (trainable)")
    print(f"  Velocity net params: {sum(p.numel() for p in agent.velocity_net.parameters()) / 1e6:.2f}M (frozen)")
    print(f"  Q-network params: {sum(p.numel() for p in agent.q_network.parameters()) / 1e6:.2f}M (frozen)")
    
    # Load pretrained checkpoint (use EMA weights for better performance)
    load_pretrained_checkpoint(
        agent, args.awsc_checkpoint, visual_encoder, str(device), use_ema=args.use_ema
    )
    
    # Freeze velocity net and Q-network (only train latent policy)
    for param in agent.velocity_net.parameters():
        param.requires_grad = False
    for param in agent.q_network.parameters():
        param.requires_grad = False
    if visual_encoder is not None:
        for param in visual_encoder.parameters():
            param.requires_grad = False
    
    # Create agent wrapper for evaluation
    agent_wrapper = DSRLAgentWrapper(
        agent, visual_encoder, include_rgb_flag,
        args.obs_horizon, args.act_horizon
    ).to(device)
    
    # Setup optimizer (only latent policy is trainable)
    optimizer = optim.AdamW(
        params=agent.latent_policy.parameters(),
        lr=args.lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
    )
    
    # Helper function to encode observations
    def encode_obs(obs_seq):
        """Encode observations to get obs_features for agents."""
        return encode_observations(obs_seq, visual_encoder, include_rgb_flag, device)
    
    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)
    
    def evaluate_and_save_best(iteration):
        if iteration % args.eval_freq == 0 and iteration > 0:
            last_tick = time.time()
            eval_metrics = evaluate(
                args.num_eval_episodes, agent_wrapper, envs, device, args.sim_backend
            )
            timings["eval"] += time.time() - last_tick
            
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            eval_log = {}
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                eval_log[f"eval/{k}"] = eval_metrics[k]
                print(f"{k}: {eval_metrics[k]:.4f}")
            
            # Log to wandb
            if args.track:
                import wandb
                wandb.log(eval_log, step=iteration)
            
            for k in ["success_once", "success_at_end"]:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}", agent, visual_encoder)
                    print(f"New best {k}: {eval_metrics[k]:.4f}. Saving checkpoint.")
    
    def log_metrics(iteration, losses):
        if iteration % args.log_freq == 0:
            train_log = {"charts/learning_rate": optimizer.param_groups[0]["lr"]}
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            for k, v in losses.items():
                writer.add_scalar(f"losses/{k}", v, iteration)
                train_log[f"losses/{k}"] = v
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)
                train_log[f"time/{k}"] = v
            
            # Log to wandb
            if args.track:
                import wandb
                wandb.log(train_log, step=iteration)
    
    # Training loop
    agent.latent_policy.train()
    agent.velocity_net.eval()
    agent.q_network.eval()
    if visual_encoder is not None:
        visual_encoder.eval()
    
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    
    # Track current KL coefficient for adaptive adjustment
    current_kl_coef = args.kl_coef
    
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick
        last_tick = time.time()
        
        # Encode observations
        obs_seq = data_batch["observations"]
        obs_features = encode_obs(obs_seq)
        obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        
        # Compute AW-MLE loss for latent policy
        loss_dict = agent.compute_offline_loss(obs_cond)
        loss = loss_dict["loss"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(agent.latent_policy.parameters(), 1.0)
        
        optimizer.step()
        
        # Extract metrics
        losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        
        # Adaptive KL coefficient adjustment
        if args.adaptive_kl:
            kl_value = losses.get("kl_loss", 0)
            if kl_value > 2.0 * args.kl_target:
                # KL too high: increase regularization
                current_kl_coef = min(current_kl_coef * args.kl_adapt_factor, args.kl_coef_max)
            elif kl_value < 0.5 * args.kl_target:
                # KL too low: decrease regularization
                current_kl_coef = max(current_kl_coef / args.kl_adapt_factor, args.kl_coef_min)
            
            # Update agent's kl_coef
            agent.kl_coef = current_kl_coef
            losses["current_kl_coef"] = current_kl_coef
        
        timings["forward"] += time.time() - last_tick
        
        # Evaluation and logging
        evaluate_and_save_best(iteration)
        log_metrics(iteration, losses)
        
        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration), agent, visual_encoder)
        
        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{losses.get('loss', 0):.4f}",
            "nll": f"{losses.get('nll_loss', 0):.4f}",
            "q_mean": f"{losses.get('q_mean', 0):.2f}",
        })
        last_tick = time.time()
    
    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters, losses)
    
    # Save final checkpoint
    save_ckpt(run_name, "final", agent, visual_encoder)
    print(f"Training complete! Final checkpoint saved to runs/{run_name}/checkpoints/final.pt")
    
    envs.close()
    writer.close()
