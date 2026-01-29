"""
CARM Robot Training Script

Trains diffusion policy / flow matching algorithms on CARM robot demonstration data.

Usage:
    python -m rlft.offline.train_carm --demo_path ~/recorded_data --algorithm flow_matching
    python -m rlft.offline.train_carm --demo_path ~/recorded_data --algorithm shortcut_flow
"""

ALGO_NAME = "CARM_UNet"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

# Import from rlft package
from rlft.networks import (
    PlainConv, ResNetEncoder, StateEncoder, VelocityUNet1D, 
    ShortCutVelocityUNet1D, GripperHead,
)
from rlft.algorithms import (
    DiffusionPolicyAgent, FlowMatchingAgent, ShortCutFlowAgent,
    ConsistencyFlowAgent, ReflectedFlowAgent,
)
from rlft.datasets import (
    CARMDataset, ActionNormalizer, 
    load_carm_dataset, create_carm_obs_process_fn, get_carm_data_info,
)
from rlft.datasets.data_utils import IterationBasedBatchSampler, worker_init_fn


@dataclass
class Args:
    """Training arguments for CARM robot."""
    # Experiment settings
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "CARM"
    wandb_entity: Optional[str] = None

    # Data settings
    demo_path: str = "~/rl-vla/recorded_data/mix"
    num_demos: Optional[int] = None
    task_name: str = "carm_teleop_pick_place"

    # Action space settings
    action_mode: Literal["full", "ee_only"] = "full"
    state_mode: Literal["joint_only", "ee_only", "both"] = "joint_only"
    precompute_actions: bool = False
    normalize_actions: bool = True
    action_norm_mode: Literal["standard", "minmax"] = "standard"
    
    # Discrete gripper settings
    gripper_threshold: float = 0.05
    gripper_ce_weight: float = 1.0
    gripper_class_weight_close: float = 3.0
    gripper_open_val: float = 0.078
    gripper_close_val: float = 0.04
    gripper_head_hidden_dim: int = 256

    # Camera settings
    target_image_size: Optional[Tuple[int, int]] = (128, 128)

    # Training settings
    total_iters: int = 100_000
    batch_size: int = 256
    lr: float = 1e-4

    # Policy architecture settings
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    
    # Visual encoder settings
    visual_encoder_type: Literal["plain_conv", "resnet10", "resnet18", "resnet34", "resnet50"] = "resnet10"
    visual_feature_dim: int = 256
    pretrained_backbone: bool = True
    freeze_backbone: bool = False
    freeze_bn: bool = True
    lr_backbone: float = 1e-5
    auto_image_size: bool = True
    """automatically adjust image size based on encoder type (128 for plain_conv/resnet10, 224 for ResNet18+)"""
    
    # State encoder settings
    use_state_encoder: bool = True
    state_encoder_hidden_dim: int = 128
    state_encoder_out_dim: int = 256

    # Algorithm selection
    algorithm: Literal[
        "diffusion_policy", 
        "flow_matching", 
        "reflected_flow",
        "consistency_flow",
        "shortcut_flow",
    ] = "flow_matching"
    
    # Diffusion/Flow settings
    num_diffusion_iters: int = 100
    num_flow_steps: int = 10
    ema_decay: float = 0.999
    bc_weight: float = 1.0
    consistency_weight: float = 0.3
    
    # Reflected Flow settings
    reflection_mode: Literal["hard", "soft"] = "hard"
    """reflection mode for reflected_flow"""
    boundary_reg_weight: float = 0.01
    """boundary regularization weight for reflected_flow"""
    
    # Consistency Flow settings
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
    
    # ShortCut Flow settings
    max_denoising_steps: int = 8
    """max denoising steps for shortcut_flow"""
    self_consistency_k: float = 0.25
    """fraction of batch for self-consistency in shortcut_flow"""
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
    save_freq: int = 2000
    num_dataload_workers: int = 0
    
    # Resume training
    resume_from: Optional[str] = None
    resume_optimizer: bool = True


def create_visual_encoder(encoder_type: str, out_dim: int, pretrained: bool = True, 
                          freeze_backbone: bool = False, freeze_bn: bool = True):
    """Create visual encoder based on type."""
    if encoder_type == "plain_conv":
        return PlainConv(in_channels=3, out_dim=out_dim, pool_feature_map=True)
    elif encoder_type.startswith("resnet"):
        return ResNetEncoder(
            backbone_name=encoder_type,
            out_dim=out_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            freeze_bn=freeze_bn,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def create_agent(algorithm: str, action_dim: int, global_cond_dim: int, args):
    """Create agent based on algorithm name."""
    device = "cuda" if args.cuda else "cpu"
    
    if algorithm == "diffusion_policy":
        from rlft.networks import ConditionalUnet1D
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
            # Consistency-specific parameters
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
            flow_weight=args.bc_weight,
            shortcut_weight=args.consistency_weight,
            ema_decay=args.ema_decay,
            # ShortCut-specific parameters
            max_denoising_steps=args.max_denoising_steps,
            self_consistency_k=args.self_consistency_k,
            t_min=args.sc_t_min,
            t_max=args.sc_t_max,
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


def save_ckpt(run_name, tag, agent, ema_agent, visual_encoder, state_encoder, 
              gripper_head, action_normalizer, optimizer=None):
    """Save checkpoint."""
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    checkpoint = {
        "agent": agent.state_dict(),
        "ema_agent": ema_agent.state_dict() if ema_agent else None,
        "visual_encoder": visual_encoder.state_dict() if visual_encoder else None,
        "state_encoder": state_encoder.state_dict() if state_encoder else None,
        "gripper_head": gripper_head.state_dict() if gripper_head else None,
    }
    if action_normalizer is not None and action_normalizer.stats is not None:
        checkpoint["action_normalizer"] = {
            "mode": action_normalizer.mode,
            "stats": {k: v.tolist() for k, v in action_normalizer.stats.items()},
        }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, f"runs/{run_name}/checkpoints/{tag}.pt")


def main():
    args = tyro.cli(Args)
    
    # Generate experiment name
    if args.exp_name is None:
        args.exp_name = f"{args.algorithm}-{args.task_name}-seed{args.seed}"
    
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # Set up logging
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # Get data info
    data_info = get_carm_data_info(args.demo_path, state_mode=args.state_mode)
    state_dim = data_info["state_dim"]
    
    # Determine action dimension (continuous only, gripper is discrete)
    action_dim = 13 if args.action_mode == "full" else 7
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create observation processing function
    obs_process_fn = create_carm_obs_process_fn(
        output_format="NCHW",
        target_size=args.target_image_size,
        state_mode=args.state_mode,
    )
    
    # Create action normalizer
    action_normalizer = ActionNormalizer(mode=args.action_norm_mode) if args.normalize_actions else None
    
    # Create dataset
    dataset = CARMDataset(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        device=device,
        num_episodes=args.num_demos,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_mode=args.action_mode,
        precompute_actions=args.precompute_actions,
        action_normalizer=action_normalizer,
        gripper_threshold=args.gripper_threshold,
    )
    
    # Create dataloader
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id),
    )
    
    # Build visual encoder
    visual_encoder = create_visual_encoder(
        encoder_type=args.visual_encoder_type,
        out_dim=args.visual_feature_dim,
        pretrained=args.pretrained_backbone,
        freeze_backbone=args.freeze_backbone,
        freeze_bn=args.freeze_bn,
    ).to(device)
    
    # Build state encoder
    if args.use_state_encoder:
        state_encoder = StateEncoder(
            state_dim=state_dim,
            hidden_dim=args.state_encoder_hidden_dim,
            out_dim=args.state_encoder_out_dim,
        ).to(device)
        feature_dim = args.visual_feature_dim + args.state_encoder_out_dim
    else:
        state_encoder = None
        feature_dim = args.visual_feature_dim + state_dim
    
    obs_dim = feature_dim * args.obs_horizon
    print(f"Feature dim: {feature_dim}, Obs dim: {obs_dim}")
    
    # Create agent
    agent = create_agent(args.algorithm, action_dim, obs_dim, args).to(device)
    ema_agent = create_agent(args.algorithm, action_dim, obs_dim, args).to(device)
    
    # Create gripper head
    gripper_head = GripperHead(
        obs_dim=obs_dim,
        hidden_dim=args.gripper_head_hidden_dim,
        pred_horizon=args.pred_horizon,
    ).to(device)
    
    # Set up optimizer with different LRs for backbone vs rest
    param_groups = []
    
    # Visual encoder params
    if args.freeze_backbone:
        visual_params = [p for p in visual_encoder.parameters() if p.requires_grad]
    else:
        visual_params = list(visual_encoder.parameters())
    
    if args.visual_encoder_type.startswith("resnet") and not args.freeze_backbone:
        # Separate backbone and head parameters
        backbone_params = [p for n, p in visual_encoder.named_parameters() if "fc" not in n]
        head_params = [p for n, p in visual_encoder.named_parameters() if "fc" in n]
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})
        param_groups.append({"params": head_params, "lr": args.lr})
    else:
        param_groups.append({"params": visual_params, "lr": args.lr})
    
    # State encoder params
    if state_encoder is not None:
        param_groups.append({"params": state_encoder.parameters(), "lr": args.lr})
    
    # Agent params
    param_groups.append({"params": agent.parameters(), "lr": args.lr})
    
    # Gripper head params
    param_groups.append({"params": gripper_head.parameters(), "lr": args.lr})
    
    optimizer = optim.AdamW(param_groups)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )
    
    # EMA
    all_params = list(agent.parameters()) + list(visual_encoder.parameters()) + list(gripper_head.parameters())
    if state_encoder is not None:
        all_params += list(state_encoder.parameters())
    ema = EMAModel(parameters=all_params, power=0.75)
    
    # Gripper loss with class weighting
    gripper_class_weights = torch.tensor([1.0, args.gripper_class_weight_close], device=device)
    gripper_criterion = nn.CrossEntropyLoss(weight=gripper_class_weights)
    
    # Resume from checkpoint
    start_iter = 0
    if args.resume_from is not None:
        print(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        agent.load_state_dict(checkpoint["agent"])
        if checkpoint.get("ema_agent"):
            ema_agent.load_state_dict(checkpoint["ema_agent"])
        if checkpoint.get("visual_encoder"):
            visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        if checkpoint.get("state_encoder") and state_encoder is not None:
            state_encoder.load_state_dict(checkpoint["state_encoder"])
        if checkpoint.get("gripper_head"):
            gripper_head.load_state_dict(checkpoint["gripper_head"])
        if args.resume_optimizer and checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
    
    def encode_observations(obs_seq):
        """Encode observations to get obs_features."""
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
            state_flat = state.view(B * T, -1)
            state_feat = state_encoder(state_flat)
            state_feat = state_feat.view(B, T, -1)
        else:
            state_feat = state
        features_list.append(state_feat)
        
        obs_features = torch.cat(features_list, dim=-1)
        return obs_features
    
    # Training loop
    agent.train()
    visual_encoder.train()
    gripper_head.train()
    if state_encoder is not None:
        state_encoder.train()
    
    pbar = tqdm(total=args.total_iters)
    
    for iteration, data_batch in enumerate(train_dataloader):
        obs_seq = data_batch["observations"]
        action_seq = data_batch["actions_cont"]  # Continuous actions without gripper
        gripper_labels = data_batch["gripper_label"]  # Discrete gripper labels
        
        obs_features = encode_observations(obs_seq)
        
        # Policy loss (continuous actions)
        loss_dict = agent.compute_loss(obs_features=obs_features, actions=action_seq)
        policy_loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
        
        # Gripper classification loss
        obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        gripper_logits = gripper_head(obs_cond)  # [B, pred_horizon, 2]
        gripper_loss = gripper_criterion(
            gripper_logits.view(-1, 2), 
            gripper_labels.view(-1)
        )
        
        # Total loss
        total_loss = policy_loss + args.gripper_ce_weight * gripper_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(visual_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(gripper_head.parameters(), 1.0)
        if state_encoder is not None:
            torch.nn.utils.clip_grad_norm_(state_encoder.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        
        if hasattr(agent, "update_ema"):
            agent.update_ema()
        
        # EMA update
        ema.step(all_params)
        
        # Logging
        if iteration % args.log_freq == 0:
            writer.add_scalar("losses/policy_loss", policy_loss.item(), iteration)
            writer.add_scalar("losses/gripper_loss", gripper_loss.item(), iteration)
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            
            # Gripper accuracy
            with torch.no_grad():
                gripper_pred = gripper_logits.argmax(dim=-1)
                gripper_acc = (gripper_pred == gripper_labels).float().mean()
                writer.add_scalar("metrics/gripper_accuracy", gripper_acc.item(), iteration)
            
            # WandB logging
            if args.track:
                wandb.log({
                    "losses/policy_loss": policy_loss.item(),
                    "losses/gripper_loss": gripper_loss.item(),
                    "losses/total_loss": total_loss.item(),
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "metrics/gripper_accuracy": gripper_acc.item(),
                    "charts/iteration": iteration,
                }, step=iteration)
        
        # Save checkpoint
        if iteration % args.save_freq == 0 and iteration > 0:
            ema.copy_to(ema_agent.parameters())
            save_ckpt(
                run_name, f"iter_{iteration}", 
                agent, ema_agent, visual_encoder, state_encoder, 
                gripper_head, action_normalizer, optimizer
            )
        
        pbar.update(1)
        pbar.set_postfix({
            "policy": f"{policy_loss.item():.4f}",
            "gripper": f"{gripper_loss.item():.4f}",
        })
    
    # Final save
    ema.copy_to(ema_agent.parameters())
    save_ckpt(
        run_name, "final", 
        agent, ema_agent, visual_encoder, state_encoder, 
        gripper_head, action_normalizer, optimizer
    )
    
    writer.close()
    
    # Close WandB
    if args.track:
        wandb.finish()
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Run: {run_name}")
    print(f"Total iterations: {args.total_iters}")
    print("=" * 50)


if __name__ == "__main__":
    main()
