"""
CARM Robot Training Script

Trains diffusion policy / flow matching algorithms on CARM robot demonstration data.
Aligned with infer_g3_api.py for action space and state design.

Action Space Design (aligned with infer_g3_api.py):
    - Policy outputs: [joint(6), gripper(1), relative_end_pose(7), gripper(1)] = 15D
    - Or simplified: [relative_end_pose(7), gripper(1)] = 8D
    - relative_end_pose is a transformation relative to current pose

State Design (aligned with infer_g3_api.py):
    - qpos_joint: [6 joints + 1 gripper] = 7D

Supported algorithms:
    - diffusion_policy: DDPM-based Diffusion Policy
    - flow_matching: Flow Matching Policy
    - reflected_flow: Reflected Flow for bounded actions
    - consistency_flow: Consistency Flow with self-consistency
    - shortcut_flow: ShortCut Flow with adaptive steps

Usage:
    python train_carm.py --demo_path ~/rl-vla/recorded_data --algorithm flow_matching
"""

ALGO_NAME = "CARM_UNet"

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

from diffusion_policy.carm_utils import (
    load_carm_dataset,
    create_carm_obs_process_fn,
    get_carm_data_info,
    compute_relative_actions,
    compute_delta_actions_simple,
    ActionNormalizer,
    StateEncoder,
)
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.resnet_encoder import (
    ResNetEncoder,
    create_visual_encoder,
    get_encoder_input_size,
)
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
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "CARM"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # Data settings
    demo_path: str = "~/rl-vla/recorded_data"
    """the path of demo dataset directory (CARM format)"""
    num_demos: Optional[int] = None
    """number of episodes to load from the demo dataset"""
    task_name: str = "carm_teleop_pick_place"
    """task name for logging"""

    # Action space settings
    action_mode: Literal["full", "ee_only"] = "full"
    """action mode: 'full' (15D) or 'ee_only' (8D: relative_pose + gripper)"""
    use_delta_actions: bool = False
    """compute actions as delta between frames instead of using recorded actions"""
    normalize_actions: bool = False
    """whether to normalize actions for training"""
    action_norm_mode: Literal["standard", "minmax"] = "standard"
    """action normalization mode"""

    # Camera settings
    target_image_size: Optional[Tuple[int, int]] = (128, 128)
    """target image size (H, W) for resizing, None = no resize"""

    # Training settings
    total_iters: int = 100_000
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
    visual_encoder_type: Literal["plain_conv", "resnet18", "resnet34", "resnet50"] = "plain_conv"
    """visual encoder type: plain_conv (lightweight), resnet18/34/50 (pretrained)"""
    visual_feature_dim: int = 256
    """visual encoder output dimension"""
    pretrained_backbone: bool = True
    """whether to use ImageNet pretrained weights (ResNet only)"""
    freeze_backbone: bool = False
    """whether to freeze backbone parameters (ResNet only, for few-shot learning)"""
    freeze_bn: bool = True
    """whether to freeze BatchNorm layers (ResNet only, recommended for small batch)"""
    lr_backbone: float = 1e-5
    """learning rate for backbone (ResNet only, typically lower than main lr)"""
    auto_image_size: bool = True
    """automatically adjust image size based on encoder type (128 for plain_conv, 224 for ResNet)"""
    
    # State encoder settings
    use_state_encoder: bool = True
    """whether to use StateEncoder MLP for state features"""
    state_encoder_hidden_dim: int = 128
    """hidden dimension for StateEncoder MLP"""
    state_encoder_out_dim: int = 256
    """output dimension for StateEncoder MLP"""

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
    cons_full_t_range: bool = False
    cons_t_min: float = 0.05
    cons_t_max: float = 0.95
    cons_t_upper: float = 0.95
    cons_delta_mode: Literal["random", "fixed"] = "random"
    cons_delta_min: float = 0.02
    cons_delta_max: float = 0.15
    cons_delta_fixed: float = 0.01
    cons_delta_dynamic_max: bool = False
    cons_delta_cap: float = 0.99
    cons_teacher_steps: int = 2
    cons_teacher_from: Literal["t_plus", "t_cons"] = "t_plus"
    cons_student_point: Literal["t_plus", "t_cons"] = "t_plus"
    cons_loss_space: Literal["velocity", "endpoint"] = "velocity"

    # ShortCut Flow specific hyperparameters
    sc_t_min: float = 0.0
    sc_t_max: float = 1.0
    sc_t_sampling_mode: Literal["uniform", "truncated"] = "uniform"
    sc_step_size_mode: Literal["power2", "uniform", "fixed"] = "fixed"
    sc_min_step_size: float = 0.0625
    sc_max_step_size: float = 0.5
    sc_fixed_step_size: float = 0.0625
    sc_target_mode: Literal["velocity", "endpoint"] = "velocity"
    sc_teacher_steps: int = 1
    sc_use_ema_teacher: bool = True
    sc_inference_mode: Literal["adaptive", "uniform"] = "uniform"
    sc_num_inference_steps: int = 8

    # Logging settings
    log_freq: int = 1
    """logging frequency"""
    save_freq: int = 2000
    """checkpoint save frequency"""
    num_dataload_workers: int = 0
    """dataloader workers"""
    
    # Resume training
    resume_from: Optional[str] = None
    """path to checkpoint to resume training from (e.g., runs/exp_name/checkpoints/latest.pt)"""
    resume_optimizer: bool = True
    """whether to resume optimizer state (set False to reset learning rate)"""


class IterationBasedBatchSampler:
    """Wraps a BatchSampler, resampling until specified iterations."""
    
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def worker_init_fn(worker_id, base_seed=None):
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)


class CARMDataset(Dataset):
    """Dataset for CARM robot demonstrations.
    
    Loads demonstrations from CARM HDF5 files and processes them for training.
    
    Action modes:
        - 'full': [joint(6), gripper(1), relative_end_pose(7), gripper(1)] = 15D
        - 'ee_only': [relative_end_pose(7), gripper(1)] = 8D
    
    State (aligned with infer_g3_api.py):
        - qpos_joint: [6 joints + 1 gripper] = 7D
    
    Args:
        data_path: Path to directory containing HDF5 files
        obs_process_fn: Function to process observations
        device: Device to store tensors on
        num_episodes: Number of episodes to load (None = all)
        obs_horizon: Observation stacking horizon
        pred_horizon: Action prediction horizon
        action_mode: 'full' or 'ee_only'
        use_delta_actions: Whether to compute delta actions
        action_normalizer: Optional action normalizer
    """
    
    def __init__(
        self,
        data_path: str,
        obs_process_fn,
        device,
        num_episodes: Optional[int],
        obs_horizon: int,
        pred_horizon: int,
        action_mode: str = "full",
        use_delta_actions: bool = False,
        action_normalizer: Optional[ActionNormalizer] = None,
    ):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.device = device
        self.action_mode = action_mode
        self.use_delta_actions = use_delta_actions
        self.action_normalizer = action_normalizer
        
        # Load dataset
        print(f"Loading CARM dataset from {data_path}...")
        raw_data = load_carm_dataset(data_path, num_episodes=num_episodes)
        
        print("Processing trajectories...")
        
        trajectories = {
            "observations": [],
            "actions": [],
        }
        
        all_actions = []  # For computing normalization stats
        
        for ep_idx in tqdm(range(len(raw_data['images'])), desc="Processing episodes"):
            images = raw_data['images'][ep_idx]
            qpos_joint = raw_data['qpos_joint'][ep_idx]
            qpos_end = raw_data['qpos_end'][ep_idx]
            gripper = raw_data['gripper'][ep_idx]
            
            # Process observations
            obs_dict = obs_process_fn(images, qpos_joint, qpos_end)
            
            processed_obs = {
                'rgb': torch.from_numpy(obs_dict['rgb']).to(device),
                'state': torch.from_numpy(obs_dict['state']).to(device),
            }
            
            # Process actions
            if self.use_delta_actions or len(raw_data['action']) == 0:
                # Compute delta actions from trajectory
                delta_actions = compute_delta_actions_simple(qpos_end, gripper)
                T_actions = len(delta_actions)
                
                if self.action_mode == 'full':
                    # Pad with joint positions to match full format
                    full_actions = np.zeros((T_actions, 15), dtype=np.float32)
                    full_actions[:, :6] = qpos_joint[:T_actions, :6]  # joints
                    full_actions[:, 6] = gripper[:T_actions]  # gripper
                    full_actions[:, 7:14] = delta_actions[:, :7]  # relative pose
                    full_actions[:, 14] = delta_actions[:, 7]  # gripper
                    actions = full_actions
                else:  # ee_only
                    actions = delta_actions
                
                # Truncate observations to match actions
                for k in processed_obs:
                    processed_obs[k] = processed_obs[k][:T_actions]
            else:
                # Use recorded actions, convert to relative
                raw_actions = raw_data['action'][ep_idx]
                actions = compute_relative_actions(qpos_end, raw_actions, gripper)
                
                if self.action_mode == 'ee_only':
                    # Extract only end effector part
                    actions = np.concatenate([
                        actions[:, 7:14],  # relative pose
                        actions[:, 14:15],  # gripper
                    ], axis=-1)
            
            all_actions.append(actions)
            
            trajectories["observations"].append(processed_obs)
            trajectories["actions"].append(torch.from_numpy(actions).to(device))
        
        # Compute action normalization stats
        if self.action_normalizer is not None:
            all_actions_concat = np.concatenate(all_actions, axis=0)
            self.action_normalizer.fit(all_actions_concat)
            print(f"Action normalization stats computed on {len(all_actions_concat)} samples")
            
            # Normalize actions
            for i in range(len(trajectories["actions"])):
                actions_np = trajectories["actions"][i].cpu().numpy()
                actions_norm = self.action_normalizer.transform(actions_np)
                trajectories["actions"][i] = torch.from_numpy(actions_norm).float().to(device)
        
        self.obs_keys = list(processed_obs.keys())
        print(f"Obs keys: {self.obs_keys}")
        
        # Compute slices
        print("Computing slice indices...")
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            total_transitions += L
            
            pad_before = obs_horizon - 1
            for start in range(-pad_before, L - pred_horizon + 1):
                self.slices.append((traj_idx, start, start + pred_horizon))
        
        print(f"Total transitions: {total_transitions}, Total sequences: {len(self.slices)}")
        self.trajectories = trajectories
    
    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape
        
        obs_traj = self.trajectories["observations"][traj_idx]
        
        # Get observation sequence
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
    """Create agent based on algorithm name."""
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
            action_bounds=None,
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
            action_bounds=None,
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
            num_flow_steps=args.num_flow_steps,
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
            action_bounds=None,
            device=device,
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def save_ckpt(run_name, tag, agent, ema_agent, visual_encoder=None, state_encoder=None, 
              action_normalizer=None, args=None, optimizer=None, lr_scheduler=None, 
              ema=None, iteration=None):
    """Save checkpoint."""
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ckpt = {
        "agent": agent.state_dict(),
        "ema_agent": ema_agent.state_dict(),
    }
    if visual_encoder is not None:
        ckpt["visual_encoder"] = visual_encoder.state_dict()
    if state_encoder is not None:
        ckpt["state_encoder"] = state_encoder.state_dict()
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        ckpt["lr_scheduler"] = lr_scheduler.state_dict()
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    if iteration is not None:
        ckpt["iteration"] = iteration
    
    torch.save(ckpt, f"runs/{run_name}/checkpoints/{tag}.pt")
    
    # Save action normalizer stats
    if action_normalizer is not None and action_normalizer.stats is not None:
        action_normalizer.save(f"runs/{run_name}/checkpoints/action_normalizer.json")
    
    # Save args
    if args is not None:
        import json
        args_dict = vars(args)
        args_serializable = {k: v if not isinstance(v, (list, tuple)) or not any(isinstance(x, type) for x in v) else str(v) for k, v in args_dict.items()}
        with open(f"runs/{run_name}/checkpoints/args.json", 'w') as f:
            json.dump(args_serializable, f, indent=2, default=str)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.exp_name is None:
        run_name = f"{args.task_name}__{args.algorithm}__{args.action_mode}__{args.seed}__{int(time.time())}"
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
    data_info = get_carm_data_info(args.demo_path)
    print(f"Dataset info: {data_info}")
    
    # Determine action dimension based on mode
    if args.action_mode == "full":
        action_dim = 15  # joint(6) + gripper(1) + rel_pose(7) + gripper(1)
    else:  # ee_only
        action_dim = 8   # rel_pose(7) + gripper(1)
    
    state_dim = data_info['state_dim']  # 7 (qpos_joint)
    
    print(f"Action mode: {args.action_mode}, action_dim: {action_dim}")
    print(f"State dim: {state_dim}")
    
    # Wandb tracking
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Determine image size based on encoder type
    if args.auto_image_size:
        target_image_size = get_encoder_input_size(args.visual_encoder_type, default_size=(128, 128))
        print(f"Auto image size: {args.visual_encoder_type} -> {target_image_size}")
    else:
        target_image_size = args.target_image_size
        print(f"Manual image size: {target_image_size}")
    
    # Create observation processing function
    # Note: For ResNet, images should be in [0, 1] range (normalization done inside encoder)
    obs_process_fn = create_carm_obs_process_fn(
        output_format="NCHW",
        target_size=target_image_size,
        normalize_images=True,  # Output in [0, 255] range for PlainConv or [0, 1] for ResNet
    )
    
    # Create action normalizer
    action_normalizer = None
    if args.normalize_actions:
        action_normalizer = ActionNormalizer(mode=args.action_norm_mode)
    
    # Create dataset
    dataset = CARMDataset(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        device=device,
        num_episodes=args.num_demos,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_mode=args.action_mode,
        use_delta_actions=args.use_delta_actions,
        action_normalizer=action_normalizer,
    )
    
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    # Note: start_iter will be set after resume logic, use 0 here and skip in training loop
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters, start_iter=0)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )
    
    # Determine image channels
    in_channels = 3  # RGB
    
    # Create visual encoder based on encoder type
    print(f"Creating visual encoder: {args.visual_encoder_type}")
    if args.visual_encoder_type == "plain_conv":
        visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
        use_separate_backbone_lr = False
    else:
        # ResNet-based encoder
        visual_encoder = ResNetEncoder(
            backbone_name=args.visual_encoder_type,
            out_dim=args.visual_feature_dim,
            pretrained=args.pretrained_backbone,
            freeze_backbone=args.freeze_backbone,
            freeze_bn=args.freeze_bn,
        ).to(device)
        use_separate_backbone_lr = not args.freeze_backbone  # Only use separate lr if not frozen
        
        # Print parameter statistics
        total_params = sum(p.numel() for p in visual_encoder.parameters())
        trainable_params = sum(p.numel() for p in visual_encoder.parameters() if p.requires_grad)
        print(f"  Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
        if args.freeze_backbone:
            print(f"  Backbone FROZEN - only projection layer is trainable")
        if args.freeze_bn:
            print(f"  BatchNorm layers FROZEN")
    
    visual_feature_dim = args.visual_feature_dim
    
    # Create state encoder
    state_encoder = None
    encoded_state_dim = state_dim
    if args.use_state_encoder:
        state_encoder = StateEncoder(
            state_dim=state_dim,
            hidden_dim=args.state_encoder_hidden_dim,
            out_dim=args.state_encoder_out_dim,
        ).to(device)
        encoded_state_dim = args.state_encoder_out_dim
        print(f"Using StateEncoder: {state_dim} -> {encoded_state_dim}")
    
    # Compute global conditioning dimension
    global_cond_dim = args.obs_horizon * (visual_feature_dim + encoded_state_dim)
    print(f"action_dim: {action_dim}, state_dim: {state_dim}, encoded_state_dim: {encoded_state_dim}")
    print(f"visual_feature_dim: {visual_feature_dim}")
    print(f"global_cond_dim: {global_cond_dim} = {args.obs_horizon} * ({visual_feature_dim} + {encoded_state_dim})")
    
    # Create agent
    agent = create_agent(args.algorithm, action_dim, global_cond_dim, args).to(device)
    print(f"Agent ({args.algorithm}) parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M")
    
    # Setup optimizer with optional separate learning rates for backbone
    param_groups = []
    
    # Agent parameters
    param_groups.append({
        'params': list(agent.parameters()),
        'lr': args.lr,
        'name': 'agent'
    })
    
    # Visual encoder parameters (with potential separate lr for backbone)
    if use_separate_backbone_lr and hasattr(visual_encoder, 'get_param_groups'):
        # ResNet with separate backbone/head learning rates
        backbone_params = list(visual_encoder.get_backbone_params())
        head_params = list(visual_encoder.get_head_params())
        param_groups.append({
            'params': backbone_params,
            'lr': args.lr_backbone,
            'name': 'visual_backbone'
        })
        param_groups.append({
            'params': head_params,
            'lr': args.lr,
            'name': 'visual_head'
        })
        print(f"Using separate lr for backbone: {args.lr_backbone} (head: {args.lr})")
    else:
        # PlainConv or frozen backbone
        param_groups.append({
            'params': list(visual_encoder.parameters()),
            'lr': args.lr,
            'name': 'visual_encoder'
        })
    
    # State encoder parameters
    if state_encoder is not None:
        param_groups.append({
            'params': list(state_encoder.parameters()),
            'lr': args.lr,
            'name': 'state_encoder'
        })
    
    optimizer = optim.AdamW(
        params=param_groups,
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
    
    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from is not None:
        resume_path = os.path.expanduser(args.resume_from)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        
        # Load agent weights (strict=True will raise error if mismatch)
        agent.load_state_dict(ckpt["agent"], strict=True)
        print("  Loaded agent weights")
        
        # Load EMA agent weights
        if "ema_agent" in ckpt:
            ema_agent.load_state_dict(ckpt["ema_agent"], strict=True)
            print("  Loaded ema_agent weights")
        
        # Load visual encoder weights
        if "visual_encoder" in ckpt:
            visual_encoder.load_state_dict(ckpt["visual_encoder"], strict=True)
            print("  Loaded visual_encoder weights")
        
        # Load state encoder weights
        if state_encoder is not None and "state_encoder" in ckpt:
            state_encoder.load_state_dict(ckpt["state_encoder"], strict=True)
            print("  Loaded state_encoder weights")
        
        # Load EMA state
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            print("  Loaded EMA state")
        
        # Load optimizer and scheduler state
        if args.resume_optimizer:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("  Loaded optimizer state")
            if "lr_scheduler" in ckpt:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
                print("  Loaded lr_scheduler state")
        else:
            print("  Skipped optimizer/scheduler state (resume_optimizer=False)")
        
        # Get starting iteration
        if "iteration" in ckpt:
            start_iter = ckpt["iteration"] + 1
            print(f"  Resuming from iteration {start_iter}")
        
        print(f"Resume complete!")
    
    timings = defaultdict(float)
    
    def copy_ema_to_eval_agent():
        ema.copy_to(ema_agent.parameters())
    
    def log_metrics(iteration, losses):
        if iteration % args.log_freq == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            for k, v in losses.items():
                writer.add_scalar(f"losses/{k}", v, iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)
    
    def encode_observations(obs_seq):
        """Encode observations to get obs_features for agents."""
        B = obs_seq["state"].shape[0]
        T = obs_seq["state"].shape[1]
        
        features_list = []
        
        # Visual features
        rgb = obs_seq["rgb"]  # [B, T, C, H, W]
        rgb_flat = rgb.view(B * T, *rgb.shape[2:]).float() / 255.0
        visual_feat = visual_encoder(rgb_flat)  # [B*T, visual_dim]
        visual_feat = visual_feat.view(B, T, -1)  # [B, T, visual_dim]
        features_list.append(visual_feat)
        
        # State features
        state = obs_seq["state"]  # [B, T, state_dim]
        if state_encoder is not None:
            state_flat = state.view(B * T, -1).float()
            state_feat = state_encoder(state_flat)
            state_feat = state_feat.view(B, T, -1)
            features_list.append(state_feat)
        else:
            features_list.append(state.float())
        
        obs_features = torch.cat(features_list, dim=-1)
        return obs_features
    
    # Training loop
    agent.train()
    visual_encoder.train()
    if state_encoder is not None:
        state_encoder.train()
    
    pbar = tqdm(total=args.total_iters, initial=start_iter)
    last_tick = time.time()
    
    for iteration, data_batch in enumerate(train_dataloader):
        # Skip iterations if resuming
        if iteration < start_iter:
            continue
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
            save_ckpt(run_name, f"iter_{iteration}", agent, ema_agent, 
                     visual_encoder, state_encoder, action_normalizer, args,
                     optimizer, lr_scheduler, ema, iteration)
            save_ckpt(run_name, "latest", agent, ema_agent,
                     visual_encoder, state_encoder, action_normalizer, args,
                     optimizer, lr_scheduler, ema, iteration)
        
        pbar.update(1)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
        last_tick = time.time()
    
    # Final checkpoint
    copy_ema_to_eval_agent()
    save_ckpt(run_name, "final", agent, ema_agent, 
             visual_encoder, state_encoder, action_normalizer, args,
             optimizer, lr_scheduler, ema, args.total_iters)
    log_metrics(args.total_iters, losses)
    
    writer.close()
    print(f"Training complete! Checkpoints saved to runs/{run_name}/checkpoints/")
