"""
Diagnostic test: Compare ShortCutFlowWrapper eval path vs standard ShortCutFlowAgent eval path.

The pretrained policy achieves ~85% success through the standard eval path 
(ShortCutFlowAgent.get_action), but only ~20% through the DSRL eval path
(ShortCutFlowWrapper). This test isolates the cause.

Key differences between the two paths:
1. Standard eval: x_0 ~ N(0,1) for ALL pred_horizon=16 positions
   DSRL eval: x_0 ~ noise for act_steps=8 positions + ZEROS for positions 8-15
2. Standard eval: NO clamping (action_bounds=None)
   DSRL eval: clamp(-1, 1)
3. Standard eval: AgentWrapper slices actions[:, obs_horizon-1 : obs_horizon-1+act_horizon]
   DSRL eval: ShortCutFlowWrapper also slices the same way

Tests:
  A) Standard path: ShortCutFlowAgent.get_action() + AgentWrapper slicing  (baseline, should be ~85%)
  B) Wrapper path, full noise: ShortCutFlowWrapper with full pred_horizon N(0,1) noise, no clamp
  C) Wrapper path, full noise + clamp: Same as B but with clamp(-1, 1)
  D) Wrapper path, act_steps noise + zero-pad: Current DSRL approach (expect ~20%)
  E) Wrapper path, act_steps noise + random-pad: N(0,1) for padding positions
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import common

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

# rlft
from rlft.envs import make_eval_envs
from rlft.networks import PlainConv, ShortCutVelocityUNet1D
from rlft.utils.flow_wrapper import ShortCutFlowWrapper, load_shortcut_flow_policy
from rlft.algorithms.il.shortcut_flow import ShortCutFlowAgent
from rlft.datasets.data_utils import encode_observations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="LiftPegUpright-v1")
    parser.add_argument("--num_envs", type=int, default=20)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--pred_horizon", type=int, default=16)
    parser.add_argument("--act_steps", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--control_mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--visual_feature_dim", type=int, default=256)
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class StandardAgent:
    """Replicate the exact standard eval path from train_maniskill.py.
    
    Uses ShortCutFlowAgent.get_action() (full pred_horizon N(0,1) noise),
    then slices actions[:, obs_horizon-1 : obs_horizon-1+act_steps].
    No clamping.
    """
    def __init__(self, velocity_net, visual_encoder, obs_horizon, pred_horizon,
                 act_steps, action_dim, device):
        self.agent = ShortCutFlowAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            num_inference_steps=8,
            inference_mode="uniform",
            action_bounds=None,  # No clamping
            device=device,
        )
        # Copy EMA weights into velocity_net_ema (they should be identical since we loaded EMA)
        self.agent.velocity_net_ema.load_state_dict(velocity_net.state_dict())
        
        self.visual_encoder = visual_encoder
        self.obs_horizon = obs_horizon
        self.act_steps = act_steps
        self.device = device

    def _encode_obs(self, obs):
        return encode_observations(
            obs_seq=obs, visual_encoder=self.visual_encoder,
            include_rgb=True, device=self.device,
        )

    @torch.no_grad()
    def get_action(self, obs):
        obs_cond = self._encode_obs(obs)
        # Full pred_horizon generation with N(0,1) noise at ALL positions
        actions = self.agent.get_action(obs_cond, use_ema=True)
        # Slice like AgentWrapper
        start = self.obs_horizon - 1
        end = start + self.act_steps
        return actions[:, start:end]

    def eval(self):
        self.agent.eval()
        return self


class WrapperAgent:
    """Use ShortCutFlowWrapper with configurable noise and clamping."""
    
    def __init__(self, flow_wrapper, visual_encoder, obs_horizon, pred_horizon,
                 act_steps, action_dim, device,
                 noise_mode="full_pred", do_clamp=True):
        self.flow_wrapper = flow_wrapper
        self.visual_encoder = visual_encoder
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.device = device
        self.noise_mode = noise_mode
        self.do_clamp = do_clamp

    def _encode_obs(self, obs):
        return encode_observations(
            obs_seq=obs, visual_encoder=self.visual_encoder,
            include_rgb=True, device=self.device,
        )

    @torch.no_grad()
    def get_action(self, obs):
        obs_cond = self._encode_obs(obs)
        B = obs_cond.shape[0]

        if self.noise_mode == "full_pred":
            # Full N(0,1) noise for all pred_horizon positions
            noise = torch.randn(B, self.pred_horizon, self.action_dim, device=self.device)
        elif self.noise_mode == "act_steps_zero_pad":
            # N(0,1) for act_steps, zero-pad the rest (current DSRL approach)
            noise = torch.randn(B, self.act_steps, self.action_dim, device=self.device)
        elif self.noise_mode == "act_steps_random_pad":
            # N(0,1) for act_steps, N(0,1) for padding too
            noise_act = torch.randn(B, self.act_steps, self.action_dim, device=self.device)
            noise_pad = torch.randn(B, self.pred_horizon - self.act_steps, self.action_dim, device=self.device)
            noise = torch.cat([noise_act, noise_pad], dim=1)
        elif self.noise_mode == "zero_zero_pad":
            # All zeros (act_steps zeros + zero-padding)
            noise = torch.zeros(B, self.act_steps, self.action_dim, device=self.device)
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

        # Call wrapper - but optionally skip clamping
        # We need to replicate the wrapper logic but with optional clamping
        initial_noise = noise
        if initial_noise.dim() == 2:
            T = initial_noise.shape[1] // self.action_dim
            initial_noise = initial_noise.view(B, T, self.action_dim)
        noise_T = initial_noise.shape[1]

        # Pad/truncate to pred_horizon
        if noise_T < self.pred_horizon:
            pad = torch.zeros(B, self.pred_horizon - noise_T, self.action_dim, device=self.device)
            x = torch.cat([initial_noise, pad], dim=1)
        elif noise_T > self.pred_horizon:
            x = initial_noise[:, :self.pred_horizon, :]
        else:
            x = initial_noise

        # Flatten obs for global conditioning
        if obs_cond.dim() == 3:
            obs_cond_flat = obs_cond.reshape(B, -1)
        else:
            obs_cond_flat = obs_cond

        # Euler integration
        dt = 1.0 / 8  # num_inference_steps = 8
        step_size = torch.full((B,), dt, device=self.device)
        
        for i in range(8):
            t = torch.full((B,), i * dt, device=self.device)
            v = self.flow_wrapper.velocity_net(x, t, step_size, obs_cond_flat)
            x = x + v * dt

        # Optional clamping
        if self.do_clamp:
            x = torch.clamp(x, -1.0, 1.0)

        # Slice act_steps
        start = self.obs_horizon - 1
        actions = x[:, start : start + self.act_steps, :]
        return actions

    def eval(self):
        return self


def evaluate_agent(agent, eval_envs, num_episodes, device):
    """Evaluate agent, returns metrics dict."""
    agent.eval()
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < num_episodes:
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs)
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break
            if truncated.any():
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                eps_count += eval_envs.num_envs
    results = {}
    for k in eval_metrics:
        results[k] = np.concatenate(eval_metrics[k]).mean()
    return results


def main():
    args = parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("DIAGNOSTIC: Standard eval path vs DSRL wrapper eval path")
    print("=" * 70)

    # Load pretrained policy
    print("\nLoading pretrained ShortCut Flow policy...")
    flow_wrapper, visual_encoder, state_dim = load_shortcut_flow_policy(
        args.checkpoint,
        visual_encoder_class=PlainConv,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_dim=args.action_dim,
        visual_feature_dim=args.visual_feature_dim,
        device=device,
    )
    # Share the same velocity_net reference
    velocity_net = flow_wrapper.velocity_net

    print(f"  state_dim={state_dim}")
    print(f"  noise_dim={args.act_steps * args.action_dim}")

    # Create eval environments
    print(f"\nCreating {args.num_envs} eval envs...")
    eval_envs = make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_envs,
        sim_backend="physx_cuda",
        env_kwargs=dict(
            obs_mode="rgbd",
            control_mode=args.control_mode,
            max_episode_steps=args.max_episode_steps,
        ),
        other_kwargs=dict(obs_horizon=args.obs_horizon),
        wrappers=[FlattenRGBDObservationWrapper],
    )

    tests = [
        {
            "name": "A) Standard path (ShortCutFlowAgent, full N(0,1), no clamp)",
            "agent_type": "standard",
        },
        {
            "name": "B) Wrapper: full pred_horizon N(0,1) + NO clamp",
            "agent_type": "wrapper",
            "noise_mode": "full_pred",
            "do_clamp": False,
        },
        {
            "name": "C) Wrapper: full pred_horizon N(0,1) + clamp(-1,1)",
            "agent_type": "wrapper",
            "noise_mode": "full_pred",
            "do_clamp": True,
        },
        {
            "name": "D) Wrapper: act_steps N(0,1) + zero-pad + clamp (current DSRL)",
            "agent_type": "wrapper",
            "noise_mode": "act_steps_zero_pad",
            "do_clamp": True,
        },
        {
            "name": "E) Wrapper: act_steps N(0,1) + random-pad + clamp",
            "agent_type": "wrapper",
            "noise_mode": "act_steps_random_pad",
            "do_clamp": True,
        },
        {
            "name": "F) Wrapper: act_steps N(0,1) + random-pad + NO clamp",
            "agent_type": "wrapper",
            "noise_mode": "act_steps_random_pad",
            "do_clamp": False,
        },
        {
            "name": "G) Wrapper: zero noise + zero-pad + clamp",
            "agent_type": "wrapper",
            "noise_mode": "zero_zero_pad",
            "do_clamp": True,
        },
    ]

    all_results = []
    for i, test in enumerate(tests):
        print(f"\n{'=' * 70}")
        print(f"TEST {test['name']}")
        print(f"{'=' * 70}")

        if test["agent_type"] == "standard":
            agent = StandardAgent(
                velocity_net=velocity_net,
                visual_encoder=visual_encoder,
                obs_horizon=args.obs_horizon,
                pred_horizon=args.pred_horizon,
                act_steps=args.act_steps,
                action_dim=args.action_dim,
                device=device,
            )
        else:
            agent = WrapperAgent(
                flow_wrapper=flow_wrapper,
                visual_encoder=visual_encoder,
                obs_horizon=args.obs_horizon,
                pred_horizon=args.pred_horizon,
                act_steps=args.act_steps,
                action_dim=args.action_dim,
                device=device,
                noise_mode=test["noise_mode"],
                do_clamp=test["do_clamp"],
            )

        results = evaluate_agent(agent, eval_envs, args.num_eval_episodes, device)
        all_results.append((test["name"], results))

        print(f"  Results:")
        for k, v in sorted(results.items()):
            print(f"    {k}: {v:.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Test':<65} {'success_once':>12}  {'success_end':>12}")
    print("-" * 95)
    for name, results in all_results:
        so = results.get("success_once", float("nan"))
        se = results.get("success_at_end", float("nan"))
        print(f"{name:<65} {so:>12.4f}  {se:>12.4f}")

    eval_envs.close()


if __name__ == "__main__":
    main()
