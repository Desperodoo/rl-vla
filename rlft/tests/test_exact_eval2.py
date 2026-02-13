"""
Minimal eval using AWShortCutFlowAgent directly (same as training).
"""
import sys, os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from rlft.envs import make_eval_envs, evaluate
from rlft.networks import PlainConv, ShortCutVelocityUNet1D, EnsembleQNetwork, DoubleQNetwork


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
    
    device = torch.device("cuda")
    
    # Training defaults
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
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    ema_state = checkpoint["ema_agent"]
    
    # Infer dimensions
    for key, value in ema_state.items():
        if "velocity_net.unet.down_modules.0.0.cond_encoder.1.weight" == key:
            cond_input = value.shape[1]
            global_cond = cond_input - diffusion_step_embed_dim
            state_dim = (global_cond // obs_horizon) - visual_feature_dim
            break
    
    for key, value in ema_state.items():
        if key == "velocity_net.unet.down_modules.0.0.residual_conv.weight":
            action_dim = value.shape[1]
            break
    
    global_cond_dim = obs_horizon * (visual_feature_dim + state_dim)
    obs_dim = global_cond_dim
    print(f"  state_dim={state_dim}, action_dim={action_dim}, obs_dim={obs_dim}")
    
    # Create velocity_net
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=unet_dims,
        n_groups=n_groups,
    ).to(device)
    
    # Detect critic type from checkpoint keys (AWShortCutFlow uses 'critic' prefix)
    q_keys = [k for k in ema_state.keys() if k.startswith("critic.")]
    has_ensemble = any("q_nets" in k for k in q_keys)
    has_double = any("q1" in k or "q2" in k for k in q_keys)
    
    print(f"  Critic keys: {len(q_keys)}, ensemble={has_ensemble}, double={has_double}")
    
    # Infer Q-network input dim from first weight
    q_input_dim = None
    for k, v in ema_state.items():
        if k.startswith("critic.") and "weight" in k and v.dim() == 2:
            q_input_dim = v.shape[1]
            print(f"  First critic weight: {k}: {v.shape} → input_dim={q_input_dim}")
            break
    
    # Q-network: obs_dim + action_dim * pred_horizon
    q_action_dim = action_dim * pred_horizon
    expected_q_input = obs_dim + q_action_dim
    print(f"  Expected Q input: {obs_dim} + {q_action_dim} = {expected_q_input}")
    if q_input_dim:
        print(f"  Actual Q input: {q_input_dim}")
    
    if has_ensemble:
        # Count num_qs from q_nets indices
        q_net_indices = set()
        for k in q_keys:
            if "q_nets" in k:
                parts = k.split(".")
                for i, p in enumerate(parts):
                    if p == "q_nets" and i + 1 < len(parts):
                        try:
                            q_net_indices.add(int(parts[i+1]))
                        except ValueError:
                            pass
        num_qs = max(len(q_net_indices), 2)
        print(f"  num_qs={num_qs}")
        
        q_network = EnsembleQNetwork(
            obs_dim=obs_dim,
            action_dim=q_action_dim,
            hidden_dims=[512, 512, 512],
            num_qs=num_qs,
        ).to(device)
    else:
        q_network = DoubleQNetwork(
            obs_dim=obs_dim,
            action_dim=q_action_dim,
            hidden_dims=[512, 512, 512],
        ).to(device)
    
    # Create AWShortCutFlowAgent
    from rlft.algorithms.offline_rl.aw_shortcut_flow import AWShortCutFlowAgent
    
    agent = AWShortCutFlowAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        act_horizon=act_steps,
        num_inference_steps=8,
        inference_mode="uniform",
        device=str(device),
    )
    
    # Load weights - only velocity_net parts matter for inference
    # (critic is only used during training for advantage weighting)
    vnet_state = {k: v for k, v in ema_state.items() 
                  if k.startswith("velocity_net.") or k.startswith("velocity_net_ema.")}
    missing, unexpected = agent.load_state_dict(vnet_state, strict=False)
    # Filter missing keys to show only non-critic ones
    non_critic_missing = [k for k in missing if not k.startswith("critic")]
    print(f"  Loaded {len(vnet_state)} velocity_net keys")
    print(f"  Non-critic missing keys: {len(non_critic_missing)}")
    if non_critic_missing:
        print(f"  Missing: {non_critic_missing[:5]}")
    agent.eval()
    
    # Visual encoder
    ve_state = checkpoint["visual_encoder"]
    in_channels = ve_state["cnn.0.weight"].shape[1]
    visual_encoder = PlainConv(
        in_channels=in_channels, out_dim=visual_feature_dim, pool_feature_map=True,
    ).to(device)
    visual_encoder.load_state_dict(ve_state)
    visual_encoder.eval()
    
    # Create AgentWrapper (exact copy from train_maniskill.py)
    class AgentWrapper(nn.Module):
        def __init__(self, agent, visual_encoder, include_rgb, obs_horizon, act_horizon):
            super().__init__()
            self.agent = agent
            self.visual_encoder = visual_encoder
            self.include_rgb = include_rgb
            self.obs_horizon = obs_horizon
            self.act_horizon = act_horizon

        def get_action(self, obs, **kwargs):
            if self.include_rgb:
                state = obs["state"]
                B = state.shape[0]
                T = self.obs_horizon
                features_list = []
                rgb = obs["rgb"]
                if rgb.dim() == 5 and rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    rgb = rgb.permute(0, 1, 4, 2, 3)
                rgb_flat = rgb.reshape(B * T, *rgb.shape[2:]).float() / 255.0
                visual_feat = self.visual_encoder(rgb_flat).view(B, T, -1)
                features_list.append(visual_feat)
                features_list.append(state.float())
                obs_features = torch.cat(features_list, dim=-1)
            else:
                obs_features = obs.float()
                B = obs_features.shape[0]
            
            obs_cond = obs_features.reshape(B, -1)
            actions = self.agent.get_action(obs_cond, **kwargs)
            start = self.obs_horizon - 1
            return actions[:, start:start + self.act_horizon]
        
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
    
    wrapper = AgentWrapper(agent, visual_encoder, True, obs_horizon, act_steps).to(device)
    wrapper.eval()
    
    # Create eval envs
    print("\nCreating eval environments...")
    env_kwargs = dict(control_mode=control_mode, obs_mode=obs_mode, render_mode="rgb_array")
    wrappers = [FlattenRGBDObservationWrapper]
    
    envs = make_eval_envs(
        env_id=env_id, num_envs=num_eval_envs, sim_backend=sim_backend,
        env_kwargs=env_kwargs, other_kwargs=dict(obs_horizon=obs_horizon),
        wrappers=wrappers,
    )
    
    # Quick sanity check on obs shapes
    obs, info = envs.reset()
    obs_t = common.to_tensor(obs, device)
    print(f"  state: {obs_t['state'].shape}, rgb: {obs_t['rgb'].shape}")
    
    # Test action output
    with torch.no_grad():
        test_action = wrapper.get_action(obs_t)
        print(f"  action output shape: {test_action.shape}")
        print(f"  action range: [{test_action.min():.4f}, {test_action.max():.4f}]")
        print(f"  action mean: {test_action.mean():.4f}, std: {test_action.std():.4f}")
    
    # Evaluate
    print(f"\nEvaluating with {num_eval_episodes} episodes...")
    eval_metrics = evaluate(num_eval_episodes, wrapper, envs, device, sim_backend)
    
    for k in eval_metrics:
        eval_metrics[k] = np.mean(eval_metrics[k])
    
    print(f"\nResults:")
    for k, v in sorted(eval_metrics.items()):
        print(f"  {k}: {v:.4f}")
    
    envs.close()


if __name__ == "__main__":
    main()
