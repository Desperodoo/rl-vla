"""
CARM Robot Finetune Script - Finetune from Real Robot Inference Data

Finetune a pretrained diffusion policy / flow matching model using data collected
during real robot deployment (with optional human intervention).

Key Differences from train_carm.py:
    1. InferenceDataset: Loads inference data where action[7:14] is ALREADY relative pose
       (no need to call compute_relative_pose_transform)
    2. Supports mixing teleop data (CARMDataset) with inference data (InferenceDataset)
    3. Supports sample weighting for intervention frames (intervention data is more valuable)
    4. Loads pretrained checkpoint and finetunes with lower learning rate

Data Format (inference_logs):
    - action_model: [T, pred_horizon, 15] - model's raw output (relative pose)
    - action_intervened: [T, pred_horizon, 15] - action after human intervention
    - intervention_mask: [T, pred_horizon, 15] - which dimensions were intervened

Usage:
    # Finetune from pretrained model using inference data only
    python train_finetune_from_realbot.py \\
        --pretrain_ckpt runs/exp_name/checkpoints/latest.pt \\
        --inference_data_dir ~/rl-vla/inference_logs \\
        --lr 1e-5

    # Finetune with mixed data (teleop + inference)
    python train_finetune_from_realbot.py \\
        --pretrain_ckpt runs/exp_name/checkpoints/latest.pt \\
        --inference_data_dir ~/rl-vla/inference_logs \\
        --teleop_data_dir ~/rl-vla/recorded_data/mix \\
        --mix_ratio 0.5 \\
        --lr 1e-5
"""

ALGO_NAME = "CARM_Finetune"

import os
import glob
import random
import time
import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Literal, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.carm_utils import (
    load_carm_dataset,
    create_carm_obs_process_fn,
    get_carm_data_info,
    compute_relative_pose_transform,
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

# Import CARMDataset from train_carm for mixed training
from train_carm import CARMDataset, GripperHead, IterationBasedBatchSampler, worker_init_fn, create_agent, save_ckpt


@dataclass
class FinetuneArgs:
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
    wandb_project_name: str = "CARM_Finetune"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # ============ Finetune-specific settings ============
    pretrain_ckpt: str = ""
    """path to pretrained checkpoint (required)"""
    inference_data_dir: str = "~/rl-vla/inference_logs"
    """directory containing inference HDF5 files"""
    teleop_data_dir: Optional[str] = None
    """directory containing teleop data for mixed training (optional)"""
    mix_ratio: float = 1.0
    """ratio of inference data in mixed training (1.0 = inference only, 0.5 = 50/50)"""
    use_intervened_action: bool = True
    """use action_intervened instead of action_model"""
    intervention_weight: float = 2.0
    """sample weight multiplier for frames with intervention (higher = more important)"""
    use_intervention_weighting: bool = True
    """whether to use intervention-based sample weighting"""

    # Data settings
    num_inference_demos: Optional[int] = None
    """number of inference episodes to load (None = all)"""
    num_teleop_demos: Optional[int] = None
    """number of teleop episodes to load (None = all)"""
    task_name: str = "carm_finetune"
    """task name for logging"""

    # Action space settings (should match pretrained model)
    action_mode: Literal["full", "ee_only"] = "full"
    """action mode: 'full' (15D) or 'ee_only' (8D: relative_pose + gripper)"""
    normalize_actions: bool = False
    """whether to normalize actions for training"""
    action_norm_mode: Literal["standard", "minmax"] = "standard"
    """action normalization mode"""
    
    # Discrete gripper settings (should match pretrained model)
    gripper_threshold: float = 0.05
    """threshold to discretize gripper: close (label=1) if g < threshold, else open (label=0)"""
    gripper_ce_weight: float = 1.0
    """weight for gripper cross-entropy loss"""
    gripper_class_weight_close: float = 3.0
    """class weight for close label in CE loss"""

    # Camera settings
    target_image_size: Optional[Tuple[int, int]] = (128, 128)
    """target image size (H, W) for resizing"""

    # Training settings
    total_iters: int = 10_000
    """total finetuning iterations (typically much smaller than pretraining)"""
    batch_size: int = 128
    """batch size (can be smaller for finetuning)"""
    lr: float = 1e-5
    """learning rate (typically lower for finetuning)"""
    warmup_steps: int = 100
    """warmup steps (shorter for finetuning)"""

    # Model settings (should match pretrained model - auto-loaded from checkpoint)
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
    """visual encoder type"""
    visual_feature_dim: int = 256
    """visual encoder output dimension"""
    pretrained_backbone: bool = True
    """whether to use ImageNet pretrained weights (ResNet only)"""
    freeze_backbone: bool = False
    """whether to freeze backbone parameters"""
    freeze_bn: bool = True
    """whether to freeze BatchNorm layers"""
    auto_image_size: bool = True
    """automatically adjust image size based on encoder type"""
    
    # State encoder settings
    use_state_encoder: bool = True
    """whether to use StateEncoder MLP"""
    state_encoder_hidden_dim: int = 128
    """hidden dimension for StateEncoder MLP"""
    state_encoder_out_dim: int = 256
    """output dimension for StateEncoder MLP"""
    gripper_head_hidden_dim: int = 256
    """hidden dimension for gripper classification head"""

    # Algorithm selection (should match pretrained model)
    algorithm: Literal[
        "diffusion_policy",
        "flow_matching", 
        "reflected_flow", 
        "consistency_flow", 
        "shortcut_flow",
    ] = "consistency_flow"
    """algorithm type"""
    
    # Algorithm hyperparameters (copied from train_carm)
    num_diffusion_iters: int = 100
    reflection_mode: Literal["hard", "soft"] = "hard"
    boundary_reg_weight: float = 0.01
    max_denoising_steps: int = 8
    self_consistency_k: float = 0.25
    ema_decay: float = 0.999
    bc_weight: float = 1.0
    consistency_weight: float = 0.3
    num_flow_steps: int = 10
    
    # Consistency Flow hyperparameters
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

    # ShortCut Flow hyperparameters
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
    save_freq: int = 1000
    """checkpoint save frequency"""
    num_dataload_workers: int = 0
    """dataloader workers"""


class InferenceDataset(Dataset):
    """Dataset for real robot inference data.
    
    Loads data collected during inference (with optional human intervention).
    
    Key difference from CARMDataset:
    - action[7:14] is ALREADY relative pose (model output), no need for conversion
    - Supports action_intervened (human-corrected actions)
    - Supports intervention_mask for sample weighting
    
    Data format:
        observations/
            images        [T, H, W, C]
            qpos_joint    [T, 7]
            qpos_end      [T, 8]
            timestamps    [T]
        action_model      [T, pred_horizon, 15]  # model output (relative pose)
        action_intervened [T, pred_horizon, 15]  # after human intervention
        intervention_mask [T, pred_horizon, 15]  # bool mask
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
        use_intervened_action: bool = True,
        gripper_threshold: float = 0.05,
        action_normalizer: Optional[ActionNormalizer] = None,
    ):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.device = device
        self.action_mode = action_mode
        self.use_intervened_action = use_intervened_action
        self.gripper_threshold = gripper_threshold
        self.action_normalizer = action_normalizer
        
        # Determine action dimension (continuous only, gripper is discrete)
        # full: joint(6) + relative_pose(7) = 13D
        # ee_only: relative_pose(7) = 7D
        self.action_dim = 13 if action_mode == 'full' else 7
        
        # Find inference files
        data_path = os.path.expanduser(data_path)
        files = sorted(glob.glob(os.path.join(data_path, "inference_episode_*.hdf5")))
        
        if num_episodes is not None:
            files = files[:num_episodes]
        
        if len(files) == 0:
            raise ValueError(f"No inference data found in {data_path}")
        
        print(f"Loading {len(files)} inference episodes from {data_path}...")
        
        # Store trajectory data
        trajectories = {
            "observations": [],
            "actions": [],           # [T, 15] action (relative pose already)
            "intervention_flags": [], # [T] bool, whether any dimension was intervened
        }
        
        self.slices = []
        self.slice_weights = []  # For intervention-based weighting
        
        for filepath in tqdm(files, desc="Loading inference data"):
            try:
                data = self._load_inference_episode(filepath, obs_process_fn)
                if data is None:
                    continue
                
                traj_idx = len(trajectories["observations"])
                trajectories["observations"].append(data["observations"])
                trajectories["actions"].append(data["actions"])
                trajectories["intervention_flags"].append(data["intervention_flags"])
                
                # Compute slices
                L = len(data["actions"])
                pad_before = obs_horizon - 1
                
                for start in range(-pad_before, L - pred_horizon + 1):
                    self.slices.append((traj_idx, start, start + pred_horizon))
                    
                    # Compute weight based on intervention
                    # Check if any frame in this slice has intervention
                    slice_start = max(0, start)
                    slice_end = min(L, start + pred_horizon)
                    has_intervention = np.any(data["intervention_flags"][slice_start:slice_end])
                    self.slice_weights.append(has_intervention)
                    
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
        
        self.trajectories = trajectories
        
        # Convert weights to float
        self.slice_weights = np.array(self.slice_weights, dtype=np.float32)
        
        print(f"Loaded {len(trajectories['observations'])} episodes, {len(self.slices)} sequences")
        print(f"Intervention frames: {self.slice_weights.sum():.0f}/{len(self.slice_weights)} "
              f"({self.slice_weights.mean()*100:.1f}%)")
    
    def _load_inference_episode(
        self, 
        filepath: str, 
        obs_process_fn
    ) -> Optional[Dict[str, Any]]:
        """Load a single inference episode."""
        with h5py.File(filepath, 'r') as f:
            obs = f['observations']
            
            # Load observations
            images = np.array(obs['images'])           # [T, H, W, C]
            qpos_joint = np.array(obs['qpos_joint'])   # [T, 7]
            qpos_end = np.array(obs['qpos_end'])       # [T, 8]
            
            # Load actions
            if self.use_intervened_action and 'action_intervened' in f:
                # Use action_intervened[:, 0, :] as the action for each step
                action_all = np.array(f['action_intervened'])  # [T, pred_horizon, 15]
                actions = action_all[:, 0, :]  # [T, 15]
            elif 'action_model' in f:
                action_all = np.array(f['action_model'])
                actions = action_all[:, 0, :]
            elif 'action' in f:
                actions = np.array(f['action'])  # [T, 15]
            else:
                print(f"  Warning: No action found in {filepath}")
                return None
            
            # Load intervention mask
            if 'intervention_mask' in f:
                mask_all = np.array(f['intervention_mask'])  # [T, pred_horizon, 15]
                # Intervention flag: only consider XYZ dimensions (7:10) for intervention
                # Gripper intervention mask persists and causes 99%+ "intervention" rate
                # which is not useful for sample weighting
                # XYZ intervention is the meaningful signal for position corrections
                intervention_flags = mask_all[:, 0, 7:10].any(axis=1)  # [T]
            else:
                intervention_flags = np.zeros(len(actions), dtype=bool)
            
            T = len(images)
            if T < self.pred_horizon + self.obs_horizon:
                print(f"  Warning: Episode too short ({T} steps) in {filepath}")
                return None
            
            # Process observations
            obs_dict = obs_process_fn(images, qpos_joint, qpos_end)
            
            processed_obs = {
                'rgb': torch.from_numpy(obs_dict['rgb']).to(self.device),
                'state': torch.from_numpy(obs_dict['state']).to(self.device),
            }
            
            return {
                "observations": processed_obs,
                "actions": actions,  # [T, 15] - already relative pose!
                "intervention_flags": intervention_flags,  # [T]
            }
    
    def __getitem__(self, index):
        """Get a training sample.
        
        For inference data, action[7:14] is ALREADY relative pose from model output.
        No need to call compute_relative_pose_transform.
        """
        traj_idx, start, end = self.slices[index]
        obs_traj = self.trajectories["observations"][traj_idx]
        actions = self.trajectories["actions"][traj_idx]
        L = len(actions)
        
        # Get observation sequence
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start):start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
        
        # Get action indices with padding
        act_indices = list(range(max(0, start), min(end, L)))
        if start < 0:
            act_indices = [0] * (-start) + act_indices
        if end > L:
            act_indices = act_indices + [L - 1] * (end - L)
        
        # Build action sequence
        act_seq_list = []
        gripper_label_list = []
        
        for idx in act_indices:
            raw_action = actions[idx]  # [15]
            
            # action[7:14] is ALREADY relative pose!
            relative_pose = raw_action[7:14]
            gripper_val = raw_action[14]
            
            # Build continuous action (no gripper)
            if self.action_mode == 'full':
                # full: joint(6) + relative_pose(7) = 13D
                rel_action = np.zeros(self.action_dim, dtype=np.float32)
                rel_action[:6] = raw_action[:6]  # joints
                rel_action[6:13] = relative_pose
            else:  # ee_only
                # ee_only: relative_pose(7) = 7D
                rel_action = relative_pose.astype(np.float32)
            
            act_seq_list.append(rel_action)
            
            # Gripper label: 1 = close, 0 = open
            gripper_label_list.append(1 if gripper_val < self.gripper_threshold else 0)
        
        act_seq = np.stack(act_seq_list, axis=0)
        gripper_label = np.array(gripper_label_list, dtype=np.int64)
        
        # Apply normalization if configured
        if self.action_normalizer is not None:
            act_seq = self.action_normalizer.transform(act_seq)
        
        act_seq = torch.from_numpy(act_seq).float().to(self.device)
        gripper_label = torch.from_numpy(gripper_label).long().to(self.device)
        
        # Get intervention weight for this sample
        intervention_weight = float(self.slice_weights[index])
        
        return {
            "observations": obs_seq,
            "actions_cont": act_seq,
            "gripper_label": gripper_label,
            "intervention_weight": intervention_weight,
        }
    
    def __len__(self):
        return len(self.slices)
    
    def get_sample_weights(self, intervention_multiplier: float = 2.0) -> np.ndarray:
        """Get sample weights for WeightedRandomSampler.
        
        Args:
            intervention_multiplier: Weight multiplier for intervention frames
        
        Returns:
            weights: [num_samples] sample weights
        """
        weights = np.ones(len(self.slices), dtype=np.float32)
        # Increase weight for intervention samples
        weights[self.slice_weights > 0] = intervention_multiplier
        return weights


class MixedDataset(Dataset):
    """Mixed dataset combining teleop and inference data.
    
    Supports different sampling ratios for the two data sources.
    """
    
    def __init__(
        self,
        teleop_dataset: CARMDataset,
        inference_dataset: InferenceDataset,
        mix_ratio: float = 0.5,
    ):
        """
        Args:
            teleop_dataset: CARMDataset for teleop demonstrations
            inference_dataset: InferenceDataset for inference data
            mix_ratio: Probability of sampling from inference_dataset (0.0-1.0)
        """
        self.teleop_dataset = teleop_dataset
        self.inference_dataset = inference_dataset
        self.mix_ratio = mix_ratio
        
        # Virtual length = max of both datasets
        self._len = max(len(teleop_dataset), len(inference_dataset))
        
        print(f"MixedDataset: {len(teleop_dataset)} teleop + {len(inference_dataset)} inference")
        print(f"Mix ratio: {mix_ratio:.1%} inference data")
    
    def __getitem__(self, index):
        # Randomly choose which dataset to sample from
        if random.random() < self.mix_ratio:
            # Sample from inference dataset
            idx = random.randint(0, len(self.inference_dataset) - 1)
            sample = self.inference_dataset[idx]
        else:
            # Sample from teleop dataset
            idx = random.randint(0, len(self.teleop_dataset) - 1)
            sample = self.teleop_dataset[idx]
            # Add dummy intervention_weight
            sample["intervention_weight"] = 0.0
        
        return sample
    
    def __len__(self):
        return self._len


def load_args_from_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    """Load args from checkpoint's args.json file."""
    ckpt_dir = os.path.dirname(ckpt_path)
    args_path = os.path.join(ckpt_dir, "args.json")
    
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            return json.load(f)
    
    return {}


if __name__ == "__main__":
    args = tyro.cli(FinetuneArgs)
    
    # Validate required arguments
    if not args.pretrain_ckpt:
        raise ValueError("--pretrain_ckpt is required for finetuning")
    
    pretrain_ckpt_path = os.path.expanduser(args.pretrain_ckpt)
    if not os.path.exists(pretrain_ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrain_ckpt_path}")
    
    # Load args from pretrained checkpoint (for model architecture consistency)
    pretrain_args = load_args_from_checkpoint(pretrain_ckpt_path)
    if pretrain_args:
        print("Loading model config from pretrained checkpoint...")
        # Override model architecture args (but not training args)
        model_args = [
            'algorithm', 'action_mode', 'obs_horizon', 'act_horizon', 'pred_horizon',
            'diffusion_step_embed_dim', 'unet_dims', 'n_groups',
            'visual_encoder_type', 'visual_feature_dim',
            'use_state_encoder', 'state_encoder_hidden_dim', 'state_encoder_out_dim',
            'gripper_head_hidden_dim', 'gripper_threshold',
        ]
        for key in model_args:
            if key in pretrain_args:
                old_val = getattr(args, key, None)
                new_val = pretrain_args[key]
                if old_val != new_val:
                    print(f"  {key}: {old_val} -> {new_val}")
                    setattr(args, key, new_val)
    
    if args.exp_name is None:
        run_name = f"{args.task_name}__{args.algorithm}__finetune__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Determine action dimension
    if args.action_mode == "full":
        action_dim = 13
    else:
        action_dim = 7
    
    state_dim = 7  # qpos_joint
    
    print("=" * 60)
    print("CARM Finetune from Real Robot Data")
    print("=" * 60)
    print(f"Pretrained checkpoint: {pretrain_ckpt_path}")
    print(f"Inference data: {args.inference_data_dir}")
    print(f"Teleop data: {args.teleop_data_dir}")
    print(f"Mix ratio: {args.mix_ratio:.1%} inference data")
    print(f"Algorithm: {args.algorithm}")
    print(f"Action mode: {args.action_mode}, action_dim: {action_dim}")
    print(f"Learning rate: {args.lr}")
    print(f"Intervention weighting: {args.use_intervention_weighting}, weight={args.intervention_weight}")
    print("=" * 60)
    
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
    
    # Determine image size
    if args.auto_image_size:
        target_image_size = get_encoder_input_size(args.visual_encoder_type, default_size=(128, 128))
    else:
        target_image_size = args.target_image_size
    
    # Create observation processing function
    obs_process_fn = create_carm_obs_process_fn(
        output_format="NCHW",
        target_size=target_image_size,
        normalize_images=True,
    )
    
    # Create action normalizer
    action_normalizer = None
    if args.normalize_actions:
        action_normalizer = ActionNormalizer(mode=args.action_norm_mode)
    
    # ============ Create Datasets ============
    
    # Create inference dataset
    inference_dataset = InferenceDataset(
        data_path=args.inference_data_dir,
        obs_process_fn=obs_process_fn,
        device=device,
        num_episodes=args.num_inference_demos,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_mode=args.action_mode,
        use_intervened_action=args.use_intervened_action,
        gripper_threshold=args.gripper_threshold,
        action_normalizer=action_normalizer,
    )
    
    # Create mixed dataset if teleop data is provided
    if args.teleop_data_dir and args.mix_ratio < 1.0:
        teleop_dataset = CARMDataset(
            data_path=args.teleop_data_dir,
            obs_process_fn=obs_process_fn,
            device=device,
            num_episodes=args.num_teleop_demos,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            action_mode=args.action_mode,
            precompute_actions=False,
            action_normalizer=action_normalizer,
            gripper_threshold=args.gripper_threshold,
        )
        
        dataset = MixedDataset(
            teleop_dataset=teleop_dataset,
            inference_dataset=inference_dataset,
            mix_ratio=args.mix_ratio,
        )
    else:
        dataset = inference_dataset
    
    # Create sampler (with optional intervention weighting)
    if args.use_intervention_weighting and isinstance(dataset, InferenceDataset):
        sample_weights = dataset.get_sample_weights(args.intervention_weight)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(dataset),
            replacement=True,
        )
        batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    else:
        sampler = RandomSampler(dataset, replacement=False)
        batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters, start_iter=0)
    
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )
    
    # ============ Create Models ============
    
    # Create visual encoder
    print(f"Creating visual encoder: {args.visual_encoder_type}")
    if args.visual_encoder_type == "plain_conv":
        visual_encoder = PlainConv(
            in_channels=3,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
    else:
        visual_encoder = ResNetEncoder(
            backbone_name=args.visual_encoder_type,
            out_dim=args.visual_feature_dim,
            pretrained=args.pretrained_backbone,
            freeze_backbone=args.freeze_backbone,
            freeze_bn=args.freeze_bn,
        ).to(device)
    
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
    
    # Compute global conditioning dimension
    global_cond_dim = args.obs_horizon * (visual_feature_dim + encoded_state_dim)
    
    # Create agent
    agent = create_agent(args.algorithm, action_dim, global_cond_dim, args).to(device)
    
    # Create gripper head
    gripper_head = GripperHead(
        obs_dim=visual_feature_dim + encoded_state_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        hidden_dim=args.gripper_head_hidden_dim,
        num_classes=2,
    ).to(device)
    
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M")
    print(f"Visual encoder parameters: {sum(p.numel() for p in visual_encoder.parameters()) / 1e6:.2f}M")
    print(f"GripperHead parameters: {sum(p.numel() for p in gripper_head.parameters()) / 1e6:.4f}M")
    
    # ============ Load Pretrained Checkpoint ============
    
    print(f"\nLoading pretrained checkpoint: {pretrain_ckpt_path}")
    ckpt = torch.load(pretrain_ckpt_path, map_location=device)
    
    # Load weights
    agent.load_state_dict(ckpt["agent"], strict=True)
    print("  Loaded agent weights")
    
    if "visual_encoder" in ckpt:
        visual_encoder.load_state_dict(ckpt["visual_encoder"], strict=True)
        print("  Loaded visual_encoder weights")
    
    if state_encoder is not None and "state_encoder" in ckpt:
        state_encoder.load_state_dict(ckpt["state_encoder"], strict=True)
        print("  Loaded state_encoder weights")
    
    if "gripper_head" in ckpt:
        gripper_head.load_state_dict(ckpt["gripper_head"], strict=True)
        print("  Loaded gripper_head weights")
    
    # Create EMA agent
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = create_agent(args.algorithm, action_dim, global_cond_dim, args).to(device)
    
    if "ema_agent" in ckpt:
        ema_agent.load_state_dict(ckpt["ema_agent"], strict=True)
        print("  Loaded ema_agent weights")
    
    if "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
        print("  Loaded EMA state")
    
    print("Pretrained checkpoint loaded!")
    
    # ============ Setup Optimizer (Fresh for Finetuning) ============
    
    param_groups = [
        {'params': list(agent.parameters()), 'lr': args.lr, 'name': 'agent'},
        {'params': list(visual_encoder.parameters()), 'lr': args.lr, 'name': 'visual_encoder'},
        {'params': list(gripper_head.parameters()), 'lr': args.lr, 'name': 'gripper_head'},
    ]
    
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
    
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_iters,
    )
    
    # ============ Training Functions ============
    
    timings = defaultdict(float)
    
    def copy_ema_to_eval_agent():
        ema.copy_to(ema_agent.parameters())
    
    def log_metrics(iteration, losses):
        if iteration % args.log_freq == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            for k, v in losses.items():
                writer.add_scalar(f"losses/{k}", v, iteration)
    
    def encode_observations(obs_seq):
        B = obs_seq["state"].shape[0]
        T = obs_seq["state"].shape[1]
        
        features_list = []
        
        # Visual features
        rgb = obs_seq["rgb"]
        rgb_flat = rgb.view(B * T, *rgb.shape[2:]).float() / 255.0
        visual_feat = visual_encoder(rgb_flat)
        visual_feat = visual_feat.view(B, T, -1)
        features_list.append(visual_feat)
        
        # State features
        state = obs_seq["state"]
        if state_encoder is not None:
            state_flat = state.view(B * T, -1).float()
            state_feat = state_encoder(state_flat)
            state_feat = state_feat.view(B, T, -1)
            features_list.append(state_feat)
        else:
            features_list.append(state.float())
        
        obs_features = torch.cat(features_list, dim=-1)
        return obs_features
    
    # ============ Training Loop ============
    
    agent.train()
    visual_encoder.train()
    gripper_head.train()
    if state_encoder is not None:
        state_encoder.train()
    
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick
        last_tick = time.time()
        
        # Encode observations
        obs_seq = data_batch["observations"]
        action_cont_seq = data_batch["actions_cont"]
        gripper_label = data_batch["gripper_label"]
        obs_features = encode_observations(obs_seq)
        
        # Compute flow/diffusion loss
        loss_dict = agent.compute_loss(
            obs_features=obs_features,
            actions=action_cont_seq,
        )
        
        if isinstance(loss_dict, dict):
            flow_loss = loss_dict["loss"]
            losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        else:
            flow_loss = loss_dict
            losses = {"flow_loss": flow_loss.item()}
        
        # Compute gripper classification loss
        gripper_logits = gripper_head(obs_features)
        gripper_class_weights = torch.tensor([1.0, args.gripper_class_weight_close], device=device)
        gripper_ce_loss = F.cross_entropy(
            gripper_logits.view(-1, 2),
            gripper_label.view(-1),
            weight=gripper_class_weights,
        )
        
        # Gripper accuracy
        with torch.no_grad():
            gripper_pred = gripper_logits.argmax(dim=-1)
            gripper_acc = (gripper_pred == gripper_label).float().mean()
        
        # Total loss
        total_loss = flow_loss + args.gripper_ce_weight * gripper_ce_loss
        
        losses["gripper_ce"] = gripper_ce_loss.item()
        losses["gripper_acc"] = gripper_acc.item()
        losses["total_loss"] = total_loss.item()
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(visual_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(gripper_head.parameters(), 1.0)
        if state_encoder is not None:
            torch.nn.utils.clip_grad_norm_(state_encoder.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        
        # Update EMA
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
                     optimizer, lr_scheduler, ema, iteration, gripper_head)
            save_ckpt(run_name, "latest", agent, ema_agent,
                     visual_encoder, state_encoder, action_normalizer, args,
                     optimizer, lr_scheduler, ema, iteration, gripper_head)
        
        pbar.update(1)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
        last_tick = time.time()
    
    # Final checkpoint
    copy_ema_to_eval_agent()
    save_ckpt(run_name, "final", agent, ema_agent,
             visual_encoder, state_encoder, action_normalizer, args,
             optimizer, lr_scheduler, ema, args.total_iters, gripper_head)
    
    writer.close()
    print(f"\nFinetuning complete! Checkpoints saved to runs/{run_name}/checkpoints/")
