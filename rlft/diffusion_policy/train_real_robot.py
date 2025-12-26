"""
Real Robot Offline Training Script

Trains diffusion policy / flow matching algorithms on real robot demonstration data
collected from ARX5 robot. This script is adapted from train_offline_rl.py but:

1. Uses real robot data adapter (ARX5 format instead of ManiSkill format)
2. Uses StateEncoder MLP to encode state features for better multimodal fusion
3. Removes evaluation (real robot cannot be evaluated during offline training)
4. Only uses wrist camera (eye-in-hand) RGB/D data

Supported algorithms (IL only, offline RL algorithms not tested with real robot data):
- diffusion_policy: DDPM-based Diffusion Policy
- flow_matching: Flow Matching Policy
- reflected_flow: Reflected Flow for bounded actions
- consistency_flow: Consistency Flow with self-consistency
- shortcut_flow: ShortCut Flow with adaptive steps
"""

ALGO_NAME = "RealRobot_UNet"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.utils import (
    IterationBasedBatchSampler,
    worker_init_fn,
    StateEncoder,
    create_real_robot_obs_process_fn,
    get_real_robot_data_info,
    load_traj_hdf5,
)
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.algorithms import (
    DiffusionPolicyAgent,
    FlowMatchingAgent,
    ReflectedFlowAgent,
    ConsistencyFlowAgent,
    ShortCutFlowAgent,
    ShortCutVelocityUNet1D,
)
from diffusion_policy.algorithms.networks import VelocityUNet1D


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
    wandb_project_name: str = "RealRobot"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # Data settings
    demo_path: str = "~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5"
    """the path of demo dataset (ARX5 format)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    task_name: str = "pick_cube"
    """task name for logging"""

    # Camera settings
    camera_name: str = "wrist"
    """camera to use: 'wrist' (eye-in-hand) or 'external' (eye-to-hand)"""
    include_depth: bool = False
    """whether to include depth channel in visual input"""
    target_image_size: Optional[Tuple[int, int]] = (128, 128)
    """target image size (H, W) for resizing, None = no resize"""

    # Training settings
    total_iters: int = 30_000
    """total training iterations"""
    batch_size: int = 256
    """batch size"""
    lr: float = 1e-4
    """learning rate"""

    # Diffusion Policy / Flow Matching settings
    obs_horizon: int = 2
    """observation horizon"""
    act_horizon: int = 8
    """action execution horizon"""
    pred_horizon: int = 16
    """action prediction horizon"""
    diffusion_step_embed_dim: int = 64
    """timestep embedding dimension"""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """U-Net channel dimensions"""
    n_groups: int = 8
    """GroupNorm groups"""
    
    # Visual encoder settings
    visual_feature_dim: int = 256
    """visual encoder output dimension"""
    
    # State encoder settings (new for multimodal fusion)
    use_state_encoder: bool = True
    """whether to use StateEncoder MLP for state features"""
    state_encoder_hidden_dim: int = 128
    """hidden dimension for StateEncoder MLP"""
    state_encoder_out_dim: int = 256
    """output dimension for StateEncoder MLP (should match visual_feature_dim for best fusion)"""

    # Algorithm selection
    algorithm: Literal[
        "diffusion_policy",
        "flow_matching", 
        "reflected_flow", 
        "consistency_flow", 
        "shortcut_flow",
    ] = "flow_matching"
    """algorithm to train"""
    
    # Diffusion Policy specific hyperparameters
    num_diffusion_iters: int = 100
    """number of diffusion iterations for DDPM"""
    
    # Flow variant specific hyperparameters
    reflection_mode: Literal["hard", "soft"] = "hard"
    """reflection mode for reflected_flow"""
    boundary_reg_weight: float = 0.01
    """boundary regularization weight for reflected_flow"""
    max_denoising_steps: int = 8
    """max denoising steps for shortcut_flow"""
    self_consistency_k: float = 0.25
    """fraction of batch for self-consistency in shortcut_flow"""
    ema_decay: float = 0.999
    """EMA decay rate for consistency_flow and shortcut_flow"""
    
    # BC weight settings
    bc_weight: float = 1.0
    """BC/flow matching loss weight"""
    consistency_weight: float = 0.3
    """consistency/shortcut loss weight"""
    num_flow_steps: int = 10
    """ODE integration steps for flow matching"""

    # Consistency Flow specific hyperparameters
    cons_use_flow_t: bool = False
    """reuse flow t for consistency branch instead of resampling"""
    cons_full_t_range: bool = False
    """sample consistency t in [0,1] instead of clipped range"""
    cons_t_min: float = 0.05
    """minimum t for consistency sampling when not using full range"""
    cons_t_max: float = 0.95
    """maximum t for consistency sampling when not using full range"""
    cons_t_upper: float = 0.95
    """upper clamp for t_plus"""
    cons_delta_mode: Literal["random", "fixed"] = "random"
    """delta sampling strategy for consistency"""
    cons_delta_min: float = 0.02
    """minimum delta when using random delta"""
    cons_delta_max: float = 0.15
    """maximum delta when using random delta"""
    cons_delta_fixed: float = 0.01
    """fixed delta when cons_delta_mode=fixed"""
    cons_delta_dynamic_max: bool = False
    """cap random delta by remaining time"""
    cons_delta_cap: float = 0.99
    """ceiling used when cons_delta_dynamic_max is enabled"""
    cons_teacher_steps: int = 2
    """teacher rollout steps to t=1"""
    cons_teacher_from: Literal["t_plus", "t_cons"] = "t_plus"
    """where teacher rollout starts"""
    cons_student_point: Literal["t_plus", "t_cons"] = "t_plus"
    """student evaluation point for consistency loss"""
    cons_loss_space: Literal["velocity", "endpoint"] = "velocity"
    """consistency loss space"""

    # ShortCut Flow specific hyperparameters
    sc_t_min: float = 0.0
    """minimum t for time sampling in shortcut_flow"""
    sc_t_max: float = 1.0
    """maximum t for time sampling in shortcut_flow"""
    sc_t_sampling_mode: Literal["uniform", "truncated"] = "uniform"
    """time sampling mode"""
    sc_step_size_mode: Literal["power2", "uniform", "fixed"] = "fixed"
    """step size sampling mode"""
    sc_min_step_size: float = 0.0625
    """minimum step size"""
    sc_max_step_size: float = 0.5
    """maximum step size"""
    sc_fixed_step_size: float = 0.0625
    """fixed step size when sc_step_size_mode=fixed"""
    sc_target_mode: Literal["velocity", "endpoint"] = "velocity"
    """shortcut target mode"""
    sc_teacher_steps: int = 1
    """teacher rollout steps for shortcut target"""
    sc_use_ema_teacher: bool = True
    """whether to use EMA network as teacher"""
    sc_inference_mode: Literal["adaptive", "uniform"] = "uniform"
    """inference mode"""
    sc_num_inference_steps: int = 8
    """number of inference steps"""

    # Logging settings
    log_freq: int = 1000
    """logging frequency"""
    save_freq: int = 2000
    """checkpoint save frequency"""
    num_dataload_workers: int = 0
    """dataloader workers"""


class RealRobotDataset(Dataset):
    """Dataset for Real Robot demonstrations (ARX5 format).
    
    Loads demonstrations from ARX5 real robot data format:
    - obs/joint_pos: (T, 6) joint positions
    - obs/joint_vel: (T, 6) joint velocities
    - obs/gripper_pos: (T, 1) gripper position
    - obs/images/wrist/rgb: (T, H, W, 3) wrist camera RGB
    - obs/images/wrist/depth: (T, H, W) wrist camera depth (optional)
    - actions: (T, 7) actions (6 joint + 1 gripper)
    
    Note: Unlike ManiSkill, real robot data has obs and actions with same length T
    (no extra observation at the end).
    
    Args:
        data_path: Path to HDF5 trajectory file
        obs_process_fn: Function to process raw observations
        device: Device to store tensors on
        num_traj: Number of trajectories to load (None = all)
        obs_horizon: Observation stacking horizon
        pred_horizon: Action prediction horizon
    """
    
    def __init__(
        self,
        data_path: str,
        obs_process_fn,
        device,
        num_traj: Optional[int],
        obs_horizon: int,
        pred_horizon: int,
    ):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.device = device
        
        # Load demo dataset
        raw_data = load_traj_hdf5(data_path, num_traj=num_traj)
        
        print("Raw trajectory loaded, beginning observation pre-processing...")
        
        # Process trajectories
        trajectories = {
            "observations": [],
            "actions": [],
        }
        
        for traj_key in sorted(raw_data.keys(), key=lambda x: int(x.split("_")[-1])):
            traj = raw_data[traj_key]
            
            # Process observations using real robot adapter
            obs_dict = obs_process_fn(traj["obs"])
            
            processed_obs = {}
            processed_obs["rgb"] = torch.from_numpy(obs_dict["rgb"]).to(device)
            processed_obs["state"] = torch.from_numpy(obs_dict["state"]).to(device)
            
            trajectories["observations"].append(processed_obs)
            
            # Process actions
            trajectories["actions"].append(
                torch.Tensor(traj["actions"]).to(device=device)
            )
        
        self.obs_keys = list(processed_obs.keys())
        print(f"Obs keys: {self.obs_keys}")
        print("Obs/action pre-processing done, computing slice indices...")
        
        # Compute slices
        # For real robot data: obs and actions have same length T
        # We need to handle this differently from ManiSkill
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            # Real robot data: obs has same length as actions (T)
            obs_len = trajectories["observations"][traj_idx]["state"].shape[0]
            assert obs_len == L, f"Obs length {obs_len} != Action length {L}"
            
            total_transitions += L
            
            pad_before = obs_horizon - 1
            # We need at least pred_horizon actions, so start from -pad_before
            # and end at L - pred_horizon + 1 (so end index is L - pred_horizon + 1 + pred_horizon = L + 1)
            # But since actions are 0-indexed and we need [start, start+pred_horizon), 
            # the last valid start is L - pred_horizon
            
            for start in range(-pad_before, L - pred_horizon + 1):
                self.slices.append((traj_idx, start, start + pred_horizon))
        
        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")
        self.trajectories = trajectories
    
    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape
        
        obs_traj = self.trajectories["observations"][traj_idx]
        
        # Get observation sequence (for policy input)
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start):start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
        
        # Get action sequence
        act_seq = self.trajectories["actions"][traj_idx][max(0, start):end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            # Pad with last action
            pad_action = act_seq[-1]
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
        
        assert obs_seq["state"].shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon
        
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }
    
    def __len__(self):
        return len(self.slices)


def create_agent(algorithm: str, action_dim: int, global_cond_dim: int, args):
    """Create agent based on algorithm name.
    
    Args:
        algorithm: Algorithm name
        action_dim: Action dimension
        global_cond_dim: Global conditioning dimension
        args: Training arguments
        
    Returns:
        Initialized agent module
    """
    device = "cuda" if args.cuda else "cpu"
    
    if algorithm == "diffusion_policy":
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        
        return DiffusionPolicyAgent(
            noise_pred_net=noise_pred_net,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            num_diffusion_iters=args.num_diffusion_iters,
            device=device,
        )
    
    elif algorithm == "flow_matching":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        return FlowMatchingAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            num_flow_steps=args.num_flow_steps,
            action_bounds=None,  # No clipping for real robot data
            device=device,
        )
    
    elif algorithm == "reflected_flow":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        return ReflectedFlowAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            num_flow_steps=args.num_flow_steps,
            reflection_mode=args.reflection_mode,
            boundary_reg_weight=args.boundary_reg_weight,
            device=device,
        )
    
    elif algorithm == "consistency_flow":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        return ConsistencyFlowAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            num_flow_steps=args.num_flow_steps,
            flow_weight=args.bc_weight,
            consistency_weight=args.consistency_weight,
            ema_decay=args.ema_decay,
            cons_use_flow_t=args.cons_use_flow_t,
            cons_full_t_range=args.cons_full_t_range,
            cons_t_min=args.cons_t_min,
            cons_t_max=args.cons_t_max,
            cons_t_upper=args.cons_t_upper,
            cons_delta_mode=args.cons_delta_mode,
            cons_delta_min=args.cons_delta_min,
            cons_delta_max=args.cons_delta_max,
            cons_delta_fixed=args.cons_delta_fixed,
            cons_delta_dynamic_max=args.cons_delta_dynamic_max,
            cons_delta_cap=args.cons_delta_cap,
            teacher_steps=args.cons_teacher_steps,
            teacher_from=args.cons_teacher_from,
            student_point=args.cons_student_point,
            consistency_loss_space=args.cons_loss_space,
            action_bounds=None,  # No clipping for real robot data
            device=device,
        )
    
    elif algorithm == "shortcut_flow":
        shortcut_velocity_net = ShortCutVelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        return ShortCutFlowAgent(
            velocity_net=shortcut_velocity_net,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            max_denoising_steps=args.max_denoising_steps,
            self_consistency_k=args.self_consistency_k,
            flow_weight=args.bc_weight,
            shortcut_weight=args.consistency_weight,
            ema_decay=args.ema_decay,
            t_min=args.sc_t_min,
            t_max=args.sc_t_max,
            t_sampling_mode=args.sc_t_sampling_mode,
            step_size_mode=args.sc_step_size_mode,
            min_step_size=args.sc_min_step_size,
            max_step_size=args.sc_max_step_size,
            fixed_step_size=args.sc_fixed_step_size,
            target_mode=args.sc_target_mode,
            teacher_steps=args.sc_teacher_steps,
            use_ema_teacher=args.sc_use_ema_teacher,
            inference_mode=args.sc_inference_mode,
            num_inference_steps=args.sc_num_inference_steps,
            device=device,
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def save_ckpt(run_name, tag, agent, ema_agent, visual_encoder=None, state_encoder=None):
    """Save checkpoint.
    
    Args:
        run_name: Run directory name
        tag: Checkpoint tag (e.g., 'latest', 'iter_10000')
        agent: Agent model
        ema_agent: EMA agent model
        visual_encoder: Optional visual encoder model (PlainConv)
        state_encoder: Optional state encoder model (StateEncoder)
    """
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ckpt = {
        "agent": agent.state_dict(),
        "ema_agent": ema_agent.state_dict(),
    }
    if visual_encoder is not None:
        ckpt["visual_encoder"] = visual_encoder.state_dict()
    if state_encoder is not None:
        ckpt["state_encoder"] = state_encoder.state_dict()
    torch.save(ckpt, f"runs/{run_name}/checkpoints/{tag}.pt")


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.exp_name is None:
        args.exp_name = f"{args.algorithm}-{args.task_name}-real"
        run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Get dataset info
    print(f"Loading dataset info from: {args.demo_path}")
    data_info = get_real_robot_data_info(args.demo_path)
    print(f"Dataset info: {data_info}")
    
    action_dim = data_info["action_dim"]
    raw_state_dim = data_info["state_dim"]  # Before encoding
    
    # Wandb tracking
    if args.track:
        import wandb
        config = vars(args)
        config["data_info"] = data_info
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group=args.algorithm,
            tags=[args.algorithm, "real_robot", args.task_name],
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Create observation processing function
    obs_process_fn = create_real_robot_obs_process_fn(
        output_format="NCHW",
        camera_name=args.camera_name,
        include_depth=args.include_depth,
        target_size=args.target_image_size,
    )
    
    # Create dataset
    dataset = RealRobotDataset(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        device=device,
        num_traj=args.num_demos,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
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
    
    # Determine dimensions from dataset
    sample_obs = dataset.trajectories["observations"][0]
    state_dim = sample_obs["state"].shape[-1]
    
    # Determine image channels
    in_channels = 3  # RGB
    if args.include_depth:
        in_channels += 1  # RGBD
    
    # Create visual encoder
    visual_encoder = PlainConv(
        in_channels=in_channels,
        out_dim=args.visual_feature_dim,
        pool_feature_map=True,
    ).to(device)
    visual_feature_dim = args.visual_feature_dim
    
    # Create state encoder (optional but recommended for multimodal fusion)
    state_encoder = None
    encoded_state_dim = state_dim  # Default: no encoding
    if args.use_state_encoder:
        state_encoder = StateEncoder(
            state_dim=state_dim,
            hidden_dim=args.state_encoder_hidden_dim,
            out_dim=args.state_encoder_out_dim,
        ).to(device)
        encoded_state_dim = args.state_encoder_out_dim
        print(f"Using StateEncoder: {state_dim} -> {encoded_state_dim}")
    
    # Compute global conditioning dimension
    # obs_features = concat(visual_features, encoded_state) per timestep, then concat across obs_horizon
    global_cond_dim = args.obs_horizon * (visual_feature_dim + encoded_state_dim)
    print(f"action_dim: {action_dim}, raw_state_dim: {state_dim}, encoded_state_dim: {encoded_state_dim}")
    print(f"visual_feature_dim: {visual_feature_dim}")
    print(f"global_cond_dim: {global_cond_dim} = {args.obs_horizon} * ({visual_feature_dim} + {encoded_state_dim})")
    
    # Create agent
    agent = create_agent(args.algorithm, action_dim, global_cond_dim, args).to(device)
    print(f"Agent ({args.algorithm}) parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M")
    
    # Setup optimizer
    all_params = list(agent.parameters()) + list(visual_encoder.parameters())
    if state_encoder is not None:
        all_params += list(state_encoder.parameters())
    
    optimizer = optim.AdamW(
        params=all_params,
        lr=args.lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
    )
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )
    
    # EMA setup
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = create_agent(args.algorithm, action_dim, global_cond_dim, args).to(device)
    
    timings = defaultdict(float)
    
    def copy_ema_to_eval_agent():
        """Copy EMA parameters to evaluation agent."""
        ema.copy_to(ema_agent.parameters())
    
    def log_metrics(iteration, losses):
        if iteration % args.log_freq == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            for k, v in losses.items():
                writer.add_scalar(f"losses/{k}", v, iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)
    
    # Helper function to encode observations
    def encode_observations(obs_seq):
        """Encode observations to get obs_features for agents.
        
        Args:
            obs_seq: Dict with 'rgb' [B, T, C, H, W] and 'state' [B, T, state_dim]
        
        Returns:
            obs_features: [B, T, visual_dim + encoded_state_dim]
        """
        B = obs_seq["state"].shape[0]
        T = obs_seq["state"].shape[1]
        
        features_list = []
        
        # Visual features
        rgb = obs_seq["rgb"]  # [B, T, C, H, W]
        rgb_flat = rgb.view(B * T, *rgb.shape[2:]).float() / 255.0
        visual_feat = visual_encoder(rgb_flat)  # [B*T, visual_dim]
        visual_feat = visual_feat.view(B, T, -1)  # [B, T, visual_dim]
        features_list.append(visual_feat)
        
        # State features (with optional encoding)
        state = obs_seq["state"]  # [B, T, state_dim]
        if state_encoder is not None:
            # Encode state with MLP
            state_flat = state.view(B * T, -1)  # [B*T, state_dim]
            state_feat = state_encoder(state_flat)  # [B*T, encoded_state_dim]
            state_feat = state_feat.view(B, T, -1)  # [B, T, encoded_state_dim]
            features_list.append(state_feat)
        else:
            features_list.append(state)
        
        # Concatenate features: [B, T, visual_dim + encoded_state_dim]
        obs_features = torch.cat(features_list, dim=-1)
        
        return obs_features
    
    # Training loop
    agent.train()
    visual_encoder.train()
    if state_encoder is not None:
        state_encoder.train()
    
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick
        
        last_tick = time.time()
        
        # Encode observations
        obs_seq = data_batch["observations"]
        action_seq = data_batch["actions"]
        obs_features = encode_observations(obs_seq)
        
        # Compute loss
        loss_dict = agent.compute_loss(
            obs_features=obs_features,
            actions=action_seq,
        )
        
        if isinstance(loss_dict, dict):
            total_loss = loss_dict["loss"]
            losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        else:
            total_loss = loss_dict
            losses = {"total_loss": total_loss.item()}
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(visual_encoder.parameters(), 1.0)
        if state_encoder is not None:
            torch.nn.utils.clip_grad_norm_(state_encoder.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        
        # Update EMA for agents that have it
        if hasattr(agent, "update_ema"):
            agent.update_ema()
        
        timings["forward"] += time.time() - last_tick
        
        # EMA step
        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick
        
        # Logging
        log_metrics(iteration, losses)
        
        # Checkpoint
        if iteration > 0 and iteration % args.save_freq == 0:
            copy_ema_to_eval_agent()
            save_ckpt(run_name, f"iter_{iteration}", agent, ema_agent, visual_encoder, state_encoder)
            save_ckpt(run_name, "latest", agent, ema_agent, visual_encoder, state_encoder)
        
        pbar.update(1)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
        last_tick = time.time()
    
    # Final checkpoint
    copy_ema_to_eval_agent()
    save_ckpt(run_name, "final", agent, ema_agent, visual_encoder, state_encoder)
    log_metrics(args.total_iters, losses)
    
    writer.close()
    print(f"Training complete! Checkpoints saved to runs/{run_name}/checkpoints/")
