"""
Directly invoke train_rlpd's agent creation + initial eval, bypassing tyro CLI.
"""
import numpy as np
import torch
import sys

import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from rlft.envs import make_eval_envs, evaluate
from rlft.networks import PlainConv, ShortCutVelocityUNet1D
from rlft.algorithms.online_rl import AWSCAgent


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"

    device = torch.device("cuda")

    # Params matching train_rlpd defaults
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
    unet_down_dims = (64, 128, 256)
    n_groups = 8
    sim_backend = "physx_cuda"
    max_episode_steps = 100
    action_dim = 7
    action_bounds = (-1.0, 1.0)

    # ---- Eval envs (same as train_rlpd) ----
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

    # ---- Dimensions ----
    sample_obs = eval_envs.single_observation_space
    state_dim = sample_obs["state"].shape[-1] if include_rgb else sample_obs.shape[-1]
    visual_dim = visual_feature_dim if include_rgb else 0
    obs_dim = obs_horizon * (visual_dim + state_dim)

    print(f"state_dim={state_dim}, obs_dim={obs_dim}, action_dim={action_dim}")

    # ---- Visual encoder (exact same as train_rlpd) ----
    visual_encoder = None
    if include_rgb:
        rgb_obs = sample_obs["rgb"]
        in_channels = rgb_obs.shape[-1] if rgb_obs.shape[-1] <= 12 else rgb_obs.shape[1]
        visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=visual_feature_dim,
            pool_feature_map=True,
        ).to(device)

    # ---- AWSCAgent (exact same as train_rlpd) ----
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=unet_down_dims,
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
        action_bounds=action_bounds,
        device=str(device),
    ).to(device)

    # ---- Load pretrained (exact same as train_rlpd) ----
    print(f"\nLoading pretrained checkpoint: {checkpoint_path}")
    agent.load_pretrained(
        checkpoint_path,
        load_critic=False,
        strict=False,
        use_ema=True,
    )

    # Also load visual encoder
    if include_rgb and visual_encoder is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "visual_encoder" in checkpoint:
            visual_encoder.load_state_dict(checkpoint["visual_encoder"])
            print(f"Loaded visual encoder from checkpoint")

    # ---- Action normalizer (same as train_rlpd) ----
    action_normalizer = None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "action_normalizer" in checkpoint and checkpoint["action_normalizer"] is not None:
        from rlft.datasets import ActionNormalizer
        normalizer_info = checkpoint["action_normalizer"]
        action_normalizer = ActionNormalizer(mode=normalizer_info["mode"])
        action_normalizer.stats = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in normalizer_info["stats"].items()
        }
        print(f"Loaded action normalizer (mode={normalizer_info['mode']})")
    else:
        print(f"No action_normalizer in checkpoint — using raw actions")

    # ---- Agent wrapper (from train_rlpd) ----
    from rlft.online.train_rlpd import AgentWrapper
    agent_wrapper = AgentWrapper(
        agent=agent,
        visual_encoder=visual_encoder,
        include_rgb=include_rgb,
        obs_horizon=obs_horizon,
        act_horizon=act_horizon,
        device=device,
        action_normalizer=action_normalizer,
    )

    # ---- Evaluate ----
    print(f"\nEvaluating with {num_eval_episodes} episodes...")
    agent.eval()

    eval_metrics = evaluate(
        num_eval_episodes, agent_wrapper, eval_envs, device, sim_backend
    )
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])

    print(f"\nResults:")
    for k, v in sorted(eval_metrics.items()):
        print(f"  {k}: {v:.4f}")

    eval_envs.close()


if __name__ == "__main__":
    main()
