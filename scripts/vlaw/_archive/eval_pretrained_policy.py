#!/usr/bin/env python3
"""Evaluate pretrained ShortCut Flow policy in ManiSkill LiftPegUpright-v1.

Uses the TESTED evaluate() function from rlft/envs/evaluate.py
with the same env creation as PLD (make_flow_eval_envs).

Expected: success_once ~70%, success_at_end ~10%

Usage:
    CUDA_VISIBLE_DEVICES=8 conda run -n rlft_ms3 \
        python scripts/vlaw/eval/eval_pretrained_policy.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _root not in sys.path:
    sys.path.insert(0, _root)


class BaseFlowPolicyAgent(nn.Module):
    """Wrap ShortCutFlowWrapper + PlainConv as an agent for evaluate().

    evaluate() calls agent.get_action(obs) where obs is raw ManiSkill obs
    (already FrameStack-ed by the eval env, so rgb is 5D and state is 3D).

    Encoding matches BaseFlowEnvWrapper._encode_single_frame exactly:
    per-frame concat [visual_features | state], then flatten across frames.
    """

    def __init__(
        self,
        base_policy,
        visual_encoder,
        state_dim: int,
        visual_feature_dim: int = 256,
        obs_horizon: int = 2,
        act_steps: int = 8,
        action_dim: int = 7,
        device: str = "cuda",
    ):
        super().__init__()
        self.base_policy = base_policy
        self.visual_encoder = visual_encoder
        self.state_dim = state_dim
        self.visual_feature_dim = visual_feature_dim
        self.obs_horizon = obs_horizon
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.device = device

        self.single_obs_dim = state_dim
        if visual_encoder is not None:
            self.single_obs_dim += visual_feature_dim

    def _encode_obs(self, obs) -> torch.Tensor:
        """Encode FrameStack-ed observation into a flat obs_cond vector.

        Matches BaseFlowEnvWrapper._encode_single_frame + _encode_and_update_history:
        per-frame [visual_feat | state] concatenation, then flatten across frames.

        Returns: (B, obs_horizon * single_obs_dim)
        """
        parts = []
        B = None

        if isinstance(obs, dict):
            # ---- RGB → visual features (kept as 3D) ----
            if self.visual_encoder is not None and "rgb" in obs:
                rgb = obs["rgb"]
                if isinstance(rgb, np.ndarray):
                    rgb = torch.from_numpy(rgb).to(self.device)
                else:
                    rgb = rgb.to(self.device)
                rgb = rgb.float()

                B = rgb.shape[0]
                T = rgb.shape[1] if rgb.dim() == 5 else 1

                # (B, T, H, W, C) → (B*T, H, W, C)
                if rgb.dim() == 5:
                    rgb = rgb.reshape(B * T, *rgb.shape[2:])

                # channels last → channels first
                if rgb.dim() == 4 and rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    rgb = rgb.permute(0, 3, 1, 2)

                if rgb.max() > 1.0:
                    rgb = rgb / 255.0

                with torch.no_grad():
                    vfeat = self.visual_encoder(rgb)  # (B*T, feat_dim)

                # Keep 3D: (B, T, feat_dim)
                vfeat = vfeat.reshape(B, T, -1) if T > 1 else vfeat.unsqueeze(1)
                parts.append(vfeat)

            # ---- State / agent proprioception (kept as 3D) ----
            state = obs.get("state", obs.get("agent", None))
            if state is not None:
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).to(self.device).float()
                else:
                    state = state.to(self.device).float()

                if B is None:
                    B = state.shape[0]

                # Ensure 3D: (B, T, state_dim)
                if state.dim() == 2:
                    state = state.unsqueeze(1)

                # Truncate per-frame state if needed
                if state.shape[-1] > self.state_dim:
                    state = state[..., :self.state_dim]

                parts.append(state)

            # Per-frame concat: (B, T, vis_feat + state_dim)
            combined = torch.cat(parts, dim=-1)

            # Take last obs_horizon frames, truncate feature dim, flatten
            combined = combined[:, -self.obs_horizon:, :self.single_obs_dim]
            return combined.reshape(B, -1)  # (B, obs_horizon * single_obs_dim)
        else:
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).to(self.device).float()
            return obs

    @torch.no_grad()
    def get_action(self, obs, **kwargs) -> torch.Tensor:
        """Called by evaluate(): returns (B, actual_steps, action_dim).

        NOTE: The base policy with pred_horizon=8 + obs_horizon=2 returns only
        7 actions (sliced from index obs_horizon-1). Do NOT pad with zeros —
        evaluate()'s inner loop uses action_seq.shape[1] to iterate, so
        returning 7 actions means 7 real env steps per chunk (matching PLD).
        """
        obs_cond = self._encode_obs(obs)  # (B, obs_dim)
        B = obs_cond.shape[0]

        zero_noise = torch.zeros(B, self.act_steps, self.action_dim, device=self.device)
        actions = self.base_policy(
            obs_cond, zero_noise, return_numpy=False, act_steps=self.act_steps,
        )  # (B, actual_steps, action_dim) — may be fewer than act_steps
        return actions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(_root, "checkpoints/il/best_eval_success_once.pt"))
    parser.add_argument("--env_id", type=str, default="LiftPegUpright-v1")
    parser.add_argument("--num_envs", type=int, default=50)
    parser.add_argument("--num_eval_episodes", type=int, default=50)
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--pred_horizon", type=int, default=8)
    parser.add_argument("--act_steps", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--visual_feature_dim", type=int, default=256)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 1. Load policy ----
    print("[1/3] Loading pretrained ShortCut Flow policy ...")
    from rlft.utils.flow_wrapper import load_shortcut_flow_policy
    from rlft.networks import PlainConv

    base_policy, visual_encoder, state_dim = load_shortcut_flow_policy(
        checkpoint_path=args.checkpoint,
        visual_encoder_class=PlainConv,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_dim=args.action_dim,
        visual_feature_dim=args.visual_feature_dim,
        include_rgb=True,
        use_ema=True,
        device=device,
    )
    print(f"  state_dim={state_dim}, pred_horizon={args.pred_horizon}, act_steps={args.act_steps}")

    # ---- 2. Create eval env (same as PLD) ----
    print("[2/3] Creating eval env ...")
    from rlft.online._flow_helpers import make_flow_eval_envs
    eval_args = SimpleNamespace(
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        max_episode_steps=args.max_episode_steps,
        env_id=args.env_id,
        num_eval_envs=args.num_envs,
        sim_backend="physx_cuda",
        obs_horizon=args.obs_horizon,
    )
    eval_envs = make_flow_eval_envs(eval_args)
    print(f"  num_envs={args.num_envs}, max_episode_steps={args.max_episode_steps}")

    # ---- 3. Build agent wrapper ----
    agent = BaseFlowPolicyAgent(
        base_policy=base_policy,
        visual_encoder=visual_encoder,
        state_dim=state_dim,
        visual_feature_dim=args.visual_feature_dim,
        obs_horizon=args.obs_horizon,
        act_steps=args.act_steps,
        action_dim=args.action_dim,
        device=device,
    )

    # ---- 4. Evaluate using TESTED evaluate() ----
    print("[3/3] Running evaluation ...")
    from rlft.envs.evaluate import evaluate
    t0 = time.time()
    eval_metrics = evaluate(
        n=args.num_eval_episodes,
        agent=agent,
        eval_envs=eval_envs,
        device=device,
        sim_backend="physx_cuda",
    )
    elapsed = time.time() - t0

    # ---- 5. Report ----
    print(f"\nResults ({elapsed:.1f}s):")
    results = {}
    for k, v in eval_metrics.items():
        mean_val = float(np.mean(v))
        results[k] = round(mean_val, 4)
        print(f"  {k}: {mean_val:.4f}")

    success_once = results.get("success_once", 0) * 100
    success_at_end = results.get("success_at_end", 0) * 100
    print(f"\n  success_once:   {success_once:.1f}%  (expected ~70%)")
    print(f"  success_at_end: {success_at_end:.1f}%  (expected ~10%)")

    out_dir = os.path.join(_root, "results/vlaw")
    os.makedirs(out_dir, exist_ok=True)
    full_results = {
        "checkpoint": args.checkpoint,
        "env_id": args.env_id,
        "num_episodes": args.num_eval_episodes,
        "metrics": results,
        "success_once_pct": success_once,
        "success_at_end_pct": success_at_end,
        "pred_horizon": args.pred_horizon,
        "act_steps": args.act_steps,
        "state_dim": state_dim,
        "time_s": round(elapsed, 1),
    }
    out_path = os.path.join(out_dir, "pretrained_policy_eval.json")
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"  Saved to: {out_path}")

    eval_envs.close()


if __name__ == "__main__":
    main()
