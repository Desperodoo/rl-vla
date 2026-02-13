"""
Test: Impact of noise magnitude on pretrained flow policy performance.

This script tests whether different noise initializations for the SAC actor
cause degradation of the pretrained ShortCut Flow policy's performance.

Key question: does adding a randomly initialized latent policy (SAC actor)
degrade the pretrained policy's ~85% success rate?

Note: The checkpoint was trained with pred_horizon=8 (offline default).
Using pred_horizon=16 (online default) causes Conv1d dimension mismatch
and reduces performance to ~17% regardless of noise configuration.

Usage:
    python -m rlft.tests.test_noise_impact \
        --checkpoint runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm

# ManiSkill
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

# rlft
from rlft.envs import make_eval_envs
from rlft.networks import PlainConv, ShortCutVelocityUNet1D
from rlft.utils.flow_wrapper import ShortCutFlowWrapper, load_shortcut_flow_policy
from rlft.envs.dsrl_env import ManiSkillFlowEnvWrapper
from rlft.algorithms.online_rl.dsrl_sac import DSRLActor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt")
    parser.add_argument("--env_id", type=str, default="LiftPegUpright-v1")
    parser.add_argument("--num_envs", type=int, default=50)
    parser.add_argument("--num_eval_episodes", type=int, default=50)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--pred_horizon", type=int, default=8,
                        help="Must match offline training checkpoint (default 8)")
    parser.add_argument("--act_steps", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--visual_feature_dim", type=int, default=256)
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--action_magnitude", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class FlowOnlyAgent:
    """Agent that uses flow policy with controlled noise injection.
    
    This bypasses the SAC actor and directly injects noise into the flow policy
    for testing purposes.
    """
    def __init__(self, base_policy, visual_encoder, obs_horizon, act_steps,
                 action_dim, visual_feature_dim, device,
                 noise_mode="zero", noise_std=0.0, action_magnitude=2.0,
                 pred_horizon=16):
        self.base_policy = base_policy
        self.visual_encoder = visual_encoder
        self.obs_horizon = obs_horizon
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.visual_feature_dim = visual_feature_dim
        self.device = device
        self.noise_mode = noise_mode
        self.noise_std = noise_std
        self.action_magnitude = action_magnitude
        self.pred_horizon = pred_horizon

    def _encode_obs(self, obs):
        """Encode stacked observation to flat features."""
        from rlft.datasets.data_utils import encode_observations
        return encode_observations(
            obs_seq=obs,
            visual_encoder=self.visual_encoder,
            include_rgb=True,
            device=self.device,
        )

    @torch.no_grad()
    def get_action(self, obs, **kwargs):
        obs_cond = self._encode_obs(obs)
        B = obs_cond.shape[0]

        if self.noise_mode == "zero":
            noise = torch.zeros(B, self.act_steps, self.action_dim, device=self.device)
        elif self.noise_mode == "gaussian":
            noise = torch.randn(B, self.act_steps, self.action_dim, device=self.device) * self.noise_std
        elif self.noise_mode == "gaussian_clamped":
            noise = torch.randn(B, self.act_steps, self.action_dim, device=self.device) * self.noise_std
            noise = noise.clamp(-self.action_magnitude, self.action_magnitude)
        elif self.noise_mode == "tanh_gaussian":
            # Mimics SAC actor: tanh(N(0, std)) * action_magnitude
            z = torch.randn(B, self.act_steps * self.action_dim, device=self.device) * self.noise_std
            noise = torch.tanh(z) * self.action_magnitude
            noise = noise.view(B, self.act_steps, self.action_dim)
        elif self.noise_mode == "full_pred_horizon":
            # Provide noise for ALL pred_horizon positions (no zero-padding)
            noise = torch.randn(B, self.pred_horizon, self.action_dim, device=self.device) * self.noise_std
        elif self.noise_mode == "zero_with_noise_pad":
            # Zero noise for act_steps positions, N(0,1) for the rest  
            noise_act = torch.zeros(B, self.act_steps, self.action_dim, device=self.device)
            noise_pad = torch.randn(B, self.pred_horizon - self.act_steps, self.action_dim, device=self.device) * self.noise_std
            noise = torch.cat([noise_act, noise_pad], dim=1)
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

        actions = self.base_policy(obs_cond, noise, return_numpy=False, act_steps=self.act_steps)
        return actions

    def eval(self):
        return self


class SACActorAgent:
    """Agent that uses a randomly initialized SAC actor + flow policy.
    
    This tests the exact setup used in DSRL training at step 0.
    """
    def __init__(self, base_policy, visual_encoder, obs_dim, obs_horizon,
                 act_steps, action_dim, visual_feature_dim, device,
                 action_magnitude=2.0, log_std_init=-3.0, hidden_dims=None,
                 deterministic=True):
        self.base_policy = base_policy
        self.visual_encoder = visual_encoder
        self.obs_horizon = obs_horizon
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.visual_feature_dim = visual_feature_dim
        self.device = device
        self.action_magnitude = action_magnitude
        self.deterministic = deterministic

        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        self.actor = DSRLActor(
            obs_dim=obs_dim,
            noise_dim=act_steps * action_dim,
            hidden_dims=hidden_dims,
            action_magnitude=action_magnitude,
            log_std_init=log_std_init,
        ).to(device)
        self.actor.eval()

    def _encode_obs(self, obs):
        from rlft.datasets.data_utils import encode_observations
        return encode_observations(
            obs_seq=obs,
            visual_encoder=self.visual_encoder,
            include_rgb=True,
            device=self.device,
        )

    @torch.no_grad()
    def get_action(self, obs, **kwargs):
        obs_cond = self._encode_obs(obs)
        noise, _ = self.actor.get_action(obs_cond, deterministic=self.deterministic)
        noise_3d = noise.view(-1, self.act_steps, self.action_dim)
        actions = self.base_policy(obs_cond, noise_3d, return_numpy=False, act_steps=self.act_steps)
        return actions

    def eval(self):
        self.actor.eval()
        return self


def evaluate_agent(agent, eval_envs, num_episodes, device, sim_backend="physx_cuda"):
    """Quick evaluation returning success_once."""
    agent.eval()
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < num_episodes:
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
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


def analyze_noise_distribution(actor, obs_cond, action_magnitude, deterministic=True):
    """Analyze the noise distribution from a randomly initialized actor."""
    with torch.no_grad():
        dist = actor.forward(obs_cond)
        # Mean analysis
        mean = dist.mean  # tanh(loc) * scale
        mean_abs = mean.abs()
        print(f"    Mean noise (deterministic output):")
        print(f"      per-dim abs mean: {mean_abs.mean():.6f}")
        print(f"      per-dim abs max:  {mean_abs.max():.6f}")
        print(f"      L2 norm:          {mean.norm(dim=-1).mean():.4f}")

        # Stochastic analysis
        samples = []
        for _ in range(10):
            noise, log_prob = dist.sample_with_log_prob()
            samples.append(noise)
        samples = torch.stack(samples)  # (10, B, D)
        noise_abs = samples.abs()
        print(f"    Stochastic noise (10 samples):")
        print(f"      per-dim abs mean: {noise_abs.mean():.6f}")
        print(f"      per-dim abs max:  {noise_abs.max():.6f}")
        print(f"      per-dim std:      {samples.std(dim=0).mean():.6f}")
        print(f"      L2 norm:          {samples.norm(dim=-1).mean():.4f}")
        print(f"      effective range:  [{samples.min():.4f}, {samples.max():.4f}]")


def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load pretrained flow policy ----
    print("=" * 70)
    print("Loading pretrained ShortCut Flow policy...")
    base_policy, visual_encoder, state_dim = load_shortcut_flow_policy(
        checkpoint_path=args.checkpoint,
        visual_encoder_class=PlainConv,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_dim=args.action_dim,
        visual_feature_dim=args.visual_feature_dim,
        include_rgb=True,
        use_ema=True,
        device=str(device),
    )
    visual_dim = args.visual_feature_dim
    obs_dim = args.obs_horizon * (visual_dim + state_dim)
    noise_dim = args.act_steps * args.action_dim
    print(f"  state_dim={state_dim}, visual_dim={visual_dim}, obs_dim={obs_dim}")
    print(f"  noise_dim={noise_dim} ({args.act_steps} x {args.action_dim})")

    # ---- Create eval environments ----
    print("\nCreating evaluation environments...")
    env_kwargs = dict(
        control_mode="pd_ee_delta_pose",
        obs_mode="rgb",
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    wrappers = [FlattenRGBDObservationWrapper]
    eval_envs = make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_envs,
        sim_backend="physx_cuda",
        env_kwargs=env_kwargs,
        other_kwargs=dict(obs_horizon=args.obs_horizon),
        wrappers=wrappers,
    )
    print(f"  {args.num_envs} eval envs created")

    # Compute actual action steps after temporal offset slice
    actual_act_steps = min(args.act_steps, args.pred_horizon - args.obs_horizon + 1)
    print(f"  pred_horizon={args.pred_horizon}, act_steps={args.act_steps}, "
          f"actual output steps (after temporal offset)={actual_act_steps}")

    all_results = []  # (name, success_once, return)

    def run_test(name, agent):
        print(f"\n{'=' * 70}")
        print(f"TEST: {name}")
        results = evaluate_agent(agent, eval_envs, args.num_eval_episodes, device)
        sr = results.get('success_once', 0)
        ret = results.get('return', 0)
        print(f"  success_once: {sr:.4f}  |  return: {ret:.2f}")
        all_results.append((name, sr, ret))
        return results

    # ================================================================
    # 1. Baseline: Full pred_horizon N(0,1) noise (standard inference)
    # ================================================================
    run_test("Full pred_horizon N(0,1) — standard inference", FlowOnlyAgent(
        base_policy, visual_encoder, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        noise_mode="full_pred_horizon", noise_std=1.0,
        pred_horizon=args.pred_horizon,
    ))

    # ================================================================
    # 2. Zero noise (x₀ = 0)
    # ================================================================
    run_test("Zero noise (x₀=0)", FlowOnlyAgent(
        base_policy, visual_encoder, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        noise_mode="zero", pred_horizon=args.pred_horizon,
    ))

    # ================================================================
    # 3. Small Gaussian noise (std=0.05, like log_std_init=-3)
    # ================================================================
    run_test("Gaussian std=0.05 (log_std_init=-3)", FlowOnlyAgent(
        base_policy, visual_encoder, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        noise_mode="gaussian", noise_std=0.05, pred_horizon=args.pred_horizon,
    ))

    # ================================================================
    # 4. Medium Gaussian noise (std=0.5)
    # ================================================================
    run_test("Gaussian std=0.5", FlowOnlyAgent(
        base_policy, visual_encoder, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        noise_mode="gaussian", noise_std=0.5, pred_horizon=args.pred_horizon,
    ))

    # ================================================================
    # 5. Full N(0,1) noise for act_steps positions
    # ================================================================
    run_test("Gaussian std=1.0 (act_steps only)", FlowOnlyAgent(
        base_policy, visual_encoder, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        noise_mode="gaussian", noise_std=1.0, pred_horizon=args.pred_horizon,
    ))

    # ================================================================
    # 6. N(0,1) clamped to [-am, am]
    # ================================================================
    run_test(f"N(0,1) clamped [-{args.action_magnitude},{args.action_magnitude}]", FlowOnlyAgent(
        base_policy, visual_encoder, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        noise_mode="gaussian_clamped", noise_std=1.0,
        action_magnitude=args.action_magnitude, pred_horizon=args.pred_horizon,
    ))

    # ================================================================
    # 7. tanh(N(0,1)) * am (SAC-like, log_std_init=0)
    # ================================================================
    run_test(f"tanh(N(0,1))*{args.action_magnitude} (SAC log_std=0)", FlowOnlyAgent(
        base_policy, visual_encoder, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        noise_mode="tanh_gaussian", noise_std=1.0,
        action_magnitude=args.action_magnitude, pred_horizon=args.pred_horizon,
    ))

    # ================================================================
    # 8. tanh(N(0,0.05)) * am (SAC-like, log_std_init=-3)
    # ================================================================
    run_test(f"tanh(N(0,0.05))*{args.action_magnitude} (SAC log_std=-3)", FlowOnlyAgent(
        base_policy, visual_encoder, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        noise_mode="tanh_gaussian", noise_std=0.05,
        action_magnitude=args.action_magnitude, pred_horizon=args.pred_horizon,
    ))

    # ================================================================
    # 9. Random SAC actor (log_std_init=-3, deterministic) — CURRENT
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Analyzing SAC actor noise (log_std_init=-3.0)...")
    dummy_obs = torch.randn(args.num_envs, obs_dim, device=device)
    actor_cur = DSRLActor(obs_dim, noise_dim, [512]*3, args.action_magnitude, -3.0).to(device)
    actor_cur.eval()
    analyze_noise_distribution(actor_cur, dummy_obs, args.action_magnitude)

    run_test("SAC actor (log_std=-3, det) — CURRENT", SACActorAgent(
        base_policy, visual_encoder, obs_dim, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        action_magnitude=args.action_magnitude, log_std_init=-3.0,
        deterministic=True,
    ))

    # ================================================================
    # 10. Random SAC actor (log_std_init=0, deterministic) — ORIG DSRL
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Analyzing SAC actor noise (log_std_init=0.0)...")
    actor_orig = DSRLActor(obs_dim, noise_dim, [512]*3, args.action_magnitude, 0.0).to(device)
    actor_orig.eval()
    analyze_noise_distribution(actor_orig, dummy_obs, args.action_magnitude)

    run_test("SAC actor (log_std=0, det) — ORIG DSRL", SACActorAgent(
        base_policy, visual_encoder, obs_dim, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        action_magnitude=args.action_magnitude, log_std_init=0.0,
        deterministic=True,
    ))

    # ================================================================
    # 11. Random SAC actor (log_std_init=0, stochastic)
    # ================================================================
    run_test("SAC actor (log_std=0, stochastic)", SACActorAgent(
        base_policy, visual_encoder, obs_dim, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        action_magnitude=args.action_magnitude, log_std_init=0.0,
        deterministic=False,
    ))

    # ================================================================
    # 12. Random SAC actor (log_std_init=-3, stochastic)
    # ================================================================
    run_test("SAC actor (log_std=-3, stochastic)", SACActorAgent(
        base_policy, visual_encoder, obs_dim, args.obs_horizon, args.act_steps,
        args.action_dim, args.visual_feature_dim, device,
        action_magnitude=args.action_magnitude, log_std_init=-3.0,
        deterministic=False,
    ))

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<50} {'success_once':>12} {'return':>10}")
    print("-" * 72)
    for name, sr, ret in all_results:
        print(f"{name:<50} {sr:>12.4f} {ret:>10.2f}")
    print("-" * 72)

    eval_envs.close()
    print("Done.")


if __name__ == "__main__":
    main()
