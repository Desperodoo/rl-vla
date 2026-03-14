"""Multi-distribution ACP data collection script.

Collects rollouts under four data distributions to give the ACP value model
training data that spans the full quality spectrum encountered in practice:

    none         — clean pretrained-policy rollouts (baseline / offline-RL data)
    teleop       — Ornstein-Uhlenbeck noise on top of policy (real-robot teleop)
    rl_explore   — Gaussian noise on top of policy (online RL exploration prior)
    random       — pure random actions (distribution boundary / ablation)

The script is a thin wrapper around the existing ``VLAWDataCollector``.
It subclasses the collector to inject the desired noise wrapper at
``_load_policy`` time, so no changes to the collector itself are needed.

Usage::

    # Type B — clean pretrained policy (no noise)
    python scripts/collect_acp_data.py \\
        --noise_mode none \\
        --num_episodes 200 \\
        --output_dir data/vlaw/rollouts/pretrained_policy \\
        --gpu_id 2

    # Type C — teleop simulation (OU noise)
    python scripts/collect_acp_data.py \\
        --noise_mode teleop \\
        --ou_sigma 0.07 \\
        --pause_prob 0.04 \\
        --num_episodes 200 \\
        --output_dir data/vlaw/rollouts/teleop_sim \\
        --gpu_id 3

    # Type D — RL exploration prior (Gaussian noise)
    python scripts/collect_acp_data.py \\
        --noise_mode rl_explore \\
        --explore_sigma 0.25 \\
        --num_episodes 200 \\
        --output_dir data/vlaw/rollouts/rl_prior \\
        --gpu_id 4

    # Type E — random actions (ablation)
    python scripts/collect_acp_data.py \\
        --noise_mode random \\
        --random_sigma 0.80 \\
        --num_episodes 100 \\
        --output_dir data/vlaw/rollouts/random \\
        --gpu_id 5
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import tyro

# ---------------------------------------------------------------------------
# Allow running as a top-level script
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from rlft.vlaw.data.collector import VLAWDataCollector, CollectorConfig
from rlft.vlaw.data.noisy_policy import (
    OUNoisePolicyWrapper,
    GaussianNoisePolicyWrapper,
    ScaledRandomPolicy,
)


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------

@dataclass
class DataCollectionArgs:
    # ── Data distribution ───────────────────────────────────────────────────
    noise_mode: Literal["none", "teleop", "rl_explore", "random"] = "none"
    """
    none       → clean pretrained-policy output        (Type B)
    teleop     → OU-smoothed noise + pauses             (Type C)
    rl_explore → i.i.d. Gaussian noise                  (Type D)
    random     → ignore policy, pure random             (Type E)
    """

    # ── Environment / collection settings ───────────────────────────────────
    env_id: str = "LiftPegUpright-v1"
    num_envs: int = 32
    num_episodes: int = 200
    max_episode_steps: int = 100
    control_mode: str = "pd_ee_delta_pose"
    obs_mode: str = "rgb"
    sim_backend: str = "physx_cuda"
    frame_skip: int = 3
    min_traj_length: int = 10

    ignore_terminations: bool = False
    """True: 忽略 success 导致的 terminated，episode 持续到 max_episode_steps。
    用于 ACP v3 数据采集：产生"成功后掉落"轨迹，使 success_once ≠ success_at_end。"""

    # ── Checkpoint ──────────────────────────────────────────────────────────
    checkpoint_path: str = (
        "runs/fair_comparison/awsc/best_s42__1772570560/checkpoints/best.pt"
    )
    """Pretrained AWSC policy checkpoint.  Ignored when noise_mode='random'."""

    # ── Output ──────────────────────────────────────────────────────────────
    output_dir: str = "data/vlaw/rollouts/pretrained_policy"

    # ── GPU ─────────────────────────────────────────────────────────────────
    gpu_id: int = 2

    # ── OU noise params (teleop mode) ────────────────────────────────────────
    ou_theta: float = 0.15
    ou_sigma: float = 0.07
    ou_action_clip: float = 1.0
    pause_prob: float = 0.04
    hold_gripper_sigma: float = 0.02

    # ── Gaussian noise params (rl_explore mode) ──────────────────────────────
    explore_sigma: float = 0.25
    explore_action_clip: float = 1.0

    # ── Random policy params ─────────────────────────────────────────────────
    random_sigma: float = 0.80

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    verbose: bool = True
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Noise-injecting subclass of VLAWDataCollector
# ---------------------------------------------------------------------------

class NoisyDataCollector(VLAWDataCollector):
    """Subclass that injects a noise wrapper at ``_load_policy`` time."""

    def __init__(self, collector_cfg: CollectorConfig, args: DataCollectionArgs) -> None:
        super().__init__(collector_cfg)
        self._args = args

    def _load_policy(self, env):
        noise_mode = self._args.noise_mode
        a = self._args

        if noise_mode == "random":
            # Delegate to parent which has built-in RandomPolicy that
            # returns action chunks in the right shape (N, chunk_len, action_dim).
            # use_random_policy=True is already set in CollectorConfig.
            return super()._load_policy(env)

        # Load base policy from checkpoint
        base_policy, visual_encoder = super()._load_policy(env)

        if noise_mode == "none":
            return base_policy, visual_encoder

        action_dim = env.action_space.shape[-1] if hasattr(env.action_space, "shape") else 7

        if noise_mode == "teleop":
            policy = OUNoisePolicyWrapper(
                policy=base_policy,
                action_dim=action_dim,
                theta=a.ou_theta,
                sigma=a.ou_sigma,
                action_clip=a.ou_action_clip,
                pause_prob=a.pause_prob,
                hold_gripper_sigma=a.hold_gripper_sigma,
                rng_seed=a.seed,
            )
        elif noise_mode == "rl_explore":
            policy = GaussianNoisePolicyWrapper(
                policy=base_policy,
                action_dim=action_dim,
                sigma=a.explore_sigma,
                action_clip=a.explore_action_clip,
                rng_seed=a.seed,
            )
        else:
            raise ValueError(f"Unknown noise_mode: {noise_mode!r}")

        return policy, visual_encoder


# ---------------------------------------------------------------------------
# Source tags per distribution type
# ---------------------------------------------------------------------------

_SOURCE_TAGS = {
    "none": "pretrained_policy",
    "teleop": "teleop_sim",
    "rl_explore": "rl_prior",
    "random": "random",
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: DataCollectionArgs) -> None:
    # Set GPU visibility before any CUDA initialisation
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    source_tag = _SOURCE_TAGS[args.noise_mode]

    print(f"[collect_acp_data] noise_mode={args.noise_mode!r}  source_tag={source_tag!r}")
    print(f"[collect_acp_data] output_dir={args.output_dir!r}   num_episodes={args.num_episodes}")
    if args.noise_mode != "random":
        print(f"[collect_acp_data] checkpoint={args.checkpoint_path!r}")

    collector_cfg = CollectorConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        camera_width=128,
        camera_height=128,
        max_episode_steps=args.max_episode_steps,
        num_episodes=args.num_episodes,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        frame_skip=args.frame_skip,
        min_traj_length=args.min_traj_length,
        ignore_terminations=args.ignore_terminations,
        checkpoint_path="" if args.noise_mode == "random" else args.checkpoint_path,
        use_random_policy=(args.noise_mode == "random"),
        gpu_id=args.gpu_id,
        output_dir=args.output_dir,
        source_tag=source_tag,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    collector = NoisyDataCollector(collector_cfg, args)
    output_path = collector.run()
    print(f"[collect_acp_data] Saved → {output_path}")


if __name__ == "__main__":
    main(tyro.cli(DataCollectionArgs))
