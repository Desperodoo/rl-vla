"""
Minimal reproduction of the training eval path from train_maniskill.py.
Loads checkpoint exactly as training does and evaluates.
Should reproduce the ~85% success_once.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy

import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

# ---- exact imports from train_maniskill.py ----
from rlft.envs import make_eval_envs, evaluate
from rlft.networks import PlainConv, ShortCutVelocityUNet1D


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
    
    device = torch.device("cuda")
    
    # ---- Training defaults from Args dataclass ----
    env_id = "LiftPegUpright-v1"
    obs_mode = "rgb"
    control_mode = "pd_ee_delta_pose"
    obs_horizon = 2
    pred_horizon = 16
    act_steps = 8
    num_eval_envs = 25
    num_eval_episodes = 100
    visual_feature_dim = 256
    diffusion_step_embed_dim = 64
    unet_dims = (64, 128, 256)
    n_groups = 8
    sim_backend = "physx_cuda"
    
    # ---- Load checkpoint ----
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    ema_agent_state = checkpoint["ema_agent"]
    
    # ---- Infer state_dim from weights ----
    for key, value in ema_agent_state.items():
        if "velocity_net" in key and "cond_encoder.1.weight" in key and not key.startswith("velocity_net_ema"):
            cond_input = value.shape[1]
            global_cond = cond_input - diffusion_step_embed_dim
            state_dim = (global_cond // obs_horizon) - visual_feature_dim
            break
    
    print(f"  state_dim={state_dim}, action_dim inferred from weights...")
    
    # Check action_dim from velocity_net input conv
    for key, value in ema_agent_state.items():
        if key == "velocity_net.unet.down_modules.0.0.residual_conv.weight":
            action_dim = value.shape[1]
            break
    
    print(f"  action_dim={action_dim}")
    
    global_cond_dim = obs_horizon * (visual_feature_dim + state_dim)
    print(f"  global_cond_dim={global_cond_dim}")
    
    # ---- Create velocity_net ----
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=unet_dims,
        n_groups=n_groups,
    ).to(device)
    
    # ---- Create full ShortCutFlowAgent (like training does) ----
    from rlft.algorithms.il.shortcut_flow import ShortCutFlowAgent
    
    agent = ShortCutFlowAgent(
        velocity_net=velocity_net,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        max_denoising_steps=8,
        num_inference_steps=8,
        inference_mode="uniform",
        action_bounds=None,
        device=str(device),
    )
    
    # ---- Load agent weights ----
    # During training: agent weights are loaded/trained, then at eval:
    #   ema.copy_to(ema_agent.parameters()) 
    #   ema_agent_wrapper = AgentWrapper(ema_agent, ...)
    # The checkpoint saves ema_agent after ema.copy_to
    
    # Load full agent state (including velocity_net AND velocity_net_ema)
    agent_state = {}
    for k, v in ema_agent_state.items():
        # Keep velocity_net.* and velocity_net_ema.* keys as-is
        agent_state[k] = v
    
    missing, unexpected = agent.load_state_dict(agent_state, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  Missing: {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected: {unexpected[:5]}...")
    
    agent.eval()
    
    # ---- Create ema_agent (like training) ----
    ema_agent = deepcopy(agent)
    ema_agent.load_state_dict(ema_agent_state, strict=False)
    ema_agent.eval()
    
    # ---- Create visual encoder ----
    include_rgb = "rgb" in obs_mode
    
    visual_encoder = None
    if include_rgb:
        # Determine in_channels from checkpoint
        ve_state = checkpoint["visual_encoder"]
        in_channels = None
        for k, v in ve_state.items():
            if k.endswith("cnn.0.weight"):
                in_channels = v.shape[1]
                break
        print(f"  visual_encoder in_channels={in_channels}")
        
        visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
        visual_encoder.load_state_dict(ve_state)
        visual_encoder.eval()
    
    # ---- Create AgentWrapper (exact copy from train_maniskill.py) ----
    # This is the EXACT class from train_maniskill.py
    class AgentWrapper(nn.Module):
        def __init__(self, agent, visual_encoder, include_rgb, obs_horizon, act_horizon=None):
            super().__init__()
            self.agent = agent
            self.visual_encoder = visual_encoder
            self.include_rgb = include_rgb
            self.obs_horizon = obs_horizon
            self.act_horizon = act_horizon if act_horizon else agent.act_horizon if hasattr(agent, 'act_horizon') else 8

        def get_action(self, obs, **kwargs):
            if self.include_rgb:
                state = obs["state"]
                B = state.shape[0]
                T = self.obs_horizon
                
                features_list = []
                
                if self.visual_encoder is not None:
                    rgb = obs["rgb"]
                    if rgb.dim() == 5 and rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                        rgb = rgb.permute(0, 1, 4, 2, 3)
                    rgb_flat = rgb.reshape(B * T, *rgb.shape[2:]).float()
                    rgb_flat = rgb_flat / 255.0
                    visual_input = rgb_flat
                    visual_feat = self.visual_encoder(visual_input)
                    visual_feat = visual_feat.view(B, T, -1)
                    features_list.append(visual_feat)
                
                features_list.append(state.float())
                obs_features = torch.cat(features_list, dim=-1)
            else:
                state = obs
                B = state.shape[0]
                obs_features = state.float()
            
            obs_cond = obs_features.reshape(B, -1)
            actions = self.agent.get_action(obs_cond, **kwargs)
            
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            action_seq = actions[:, start:end]
            return action_seq
        
        def eval(self):
            self.agent.eval()
            if self.visual_encoder is not None:
                self.visual_encoder.eval()
            return self
        
        def train(self, mode=True):
            self.agent.train(mode)
            if self.visual_encoder is not None:
                self.visual_encoder.train(mode)
            return self
    
    ema_agent_wrapper = AgentWrapper(
        ema_agent, visual_encoder, include_rgb, obs_horizon, act_steps
    )
    
    # ---- Create eval envs (exact same as training) ----
    print("\nCreating eval environments...")
    env_kwargs = dict(
        control_mode=control_mode,
        obs_mode=obs_mode,
        render_mode="rgb_array",
    )
    other_kwargs = dict(obs_horizon=obs_horizon)
    wrappers = [FlattenRGBDObservationWrapper] if include_rgb else []
    
    envs = make_eval_envs(
        env_id=env_id,
        num_envs=num_eval_envs,
        sim_backend=sim_backend,
        env_kwargs=env_kwargs,
        other_kwargs=other_kwargs,
        wrappers=wrappers,
    )
    
    # Check observation shapes
    obs, info = envs.reset()
    obs_t = common.to_tensor(obs, device)
    print(f"  obs keys: {list(obs_t.keys()) if isinstance(obs_t, dict) else type(obs_t)}")
    if isinstance(obs_t, dict):
        for k, v in obs_t.items():
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
    
    # ---- Evaluate using training's evaluate() function ----
    print(f"\nEvaluating with {num_eval_episodes} episodes...")
    eval_metrics = evaluate(
        num_eval_episodes, ema_agent_wrapper, envs, device, sim_backend
    )
    
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])
    
    print(f"\nResults:")
    for k, v in sorted(eval_metrics.items()):
        print(f"  {k}: {v:.4f}")
    
    envs.close()


if __name__ == "__main__":
    main()
