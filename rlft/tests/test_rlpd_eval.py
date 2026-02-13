"""
Reproduce train_rlpd's 85% pretrained eval by using AWSCAgent exactly.
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

from rlft.envs import make_eval_envs, evaluate
from rlft.networks import PlainConv, ShortCutVelocityUNet1D
from rlft.algorithms.online_rl import AWSCAgent


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
    
    device = torch.device("cuda")
    
    # ---- Exact defaults from train_rlpd.py Args ----
    env_id = "LiftPegUpright-v1"
    obs_mode = "rgb"
    control_mode = "pd_ee_delta_pose"
    obs_horizon = 2
    pred_horizon = 16
    act_horizon = 8
    num_eval_envs = 16
    num_eval_episodes = 100
    visual_feature_dim = 256
    diffusion_step_embed_dim = 64
    unet_dims = (64, 128, 256)
    n_groups = 8
    sim_backend = "physx_cuda"
    max_episode_steps = 100
    action_dim = 7
    
    # ---- Create eval envs (same as train_rlpd) ----
    print("Creating eval environments...")
    include_rgb = "rgb" in obs_mode
    env_kwargs = dict(
        control_mode=control_mode,
        obs_mode=obs_mode,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=obs_horizon)
    wrappers = [FlattenRGBDObservationWrapper] if include_rgb else []
    
    eval_envs = make_eval_envs(
        env_id=env_id,
        num_envs=num_eval_envs,
        sim_backend=sim_backend,
        env_kwargs=env_kwargs,
        other_kwargs=other_kwargs,
        wrappers=wrappers,
    )
    
    # ---- Infer dimensions from env ----
    sample_obs = eval_envs.single_observation_space
    if include_rgb:
        state_dim = sample_obs["state"].shape[-1]
    else:
        state_dim = sample_obs.shape[-1]
    
    visual_dim = visual_feature_dim if include_rgb else 0
    obs_dim = obs_horizon * (visual_dim + state_dim)
    
    print(f"  state_dim={state_dim}, obs_dim={obs_dim}, action_dim={action_dim}")
    
    # ---- Create visual encoder ----
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    visual_encoder = None
    if include_rgb:
        ve_state = checkpoint["visual_encoder"]
        in_channels = None
        for k, v in ve_state.items():
            if k.endswith("cnn.0.weight"):
                in_channels = v.shape[1]
                break
        visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
        visual_encoder.load_state_dict(ve_state)
        visual_encoder.eval()
        print(f"  Loaded visual encoder (in_channels={in_channels})")
    
    # ---- Create AWSCAgent (same as train_rlpd) ----
    print("Creating AWSCAgent...")
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=unet_dims,
        n_groups=n_groups,
    ).to(device)
    
    agent = AWSCAgent(
        velocity_net=velocity_net,
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        num_inference_steps=8,
        inference_mode="uniform",
        action_bounds=None,
        device=str(device),
    ).to(device)
    
    # ---- Load pretrained weights (same as train_rlpd) ----
    print(f"Loading pretrained checkpoint from {checkpoint_path}...")
    agent.load_pretrained(
        checkpoint_path,
        load_critic=False,
        strict=False,
        use_ema=True,
    )
    
    # ---- Create EvalAgentWrapper (from train_rlpd) ----
    from rlft.online.train_rlpd import AgentWrapper as EvalAgentWrapper
    
    eval_wrapper = EvalAgentWrapper(
        agent=agent,
        visual_encoder=visual_encoder,
        include_rgb=include_rgb,
        obs_horizon=obs_horizon,
        act_horizon=act_horizon,
        device=device,
    )
    
    # ---- Check obs shapes ----
    obs, info = eval_envs.reset()
    obs_t = common.to_tensor(obs, device)
    print(f"\n  obs keys: {list(obs_t.keys()) if isinstance(obs_t, dict) else type(obs_t)}")
    if isinstance(obs_t, dict):
        for k, v in obs_t.items():
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
    
    # ---- Quick debug: check encode_obs consistency ----
    from rlft.datasets.data_utils import encode_observations
    obs_cond_wrapper = eval_wrapper.encode_obs(obs_t)
    obs_cond_shared = encode_observations(
        obs_seq=obs_t,
        visual_encoder=visual_encoder,
        include_rgb=include_rgb,
        device=device,
    )
    diff = (obs_cond_wrapper - obs_cond_shared).abs().max().item()
    print(f"\n  encode_obs diff between wrapper & shared: {diff:.8f}")
    if diff > 1e-5:
        print("  WARNING: encode_obs results differ!")
        # Show where they differ
        per_dim_diff = (obs_cond_wrapper - obs_cond_shared).abs().mean(dim=0)
        top_diffs = per_dim_diff.topk(5)
        for idx, val in zip(top_diffs.indices.tolist(), top_diffs.values.tolist()):
            print(f"    dim {idx}: avg_diff={val:.6f}, wrapper={obs_cond_wrapper[0, idx].item():.4f}, shared={obs_cond_shared[0, idx].item():.4f}")

    # ---- Evaluate ----
    print(f"\nEvaluating with {num_eval_episodes} episodes...")
    eval_metrics = evaluate(
        num_eval_episodes, eval_wrapper, eval_envs, device, sim_backend
    )
    
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])
    
    print(f"\nResults:")
    for k, v in sorted(eval_metrics.items()):
        print(f"  {k}: {v:.4f}")
    
    eval_envs.close()


if __name__ == "__main__":
    main()
