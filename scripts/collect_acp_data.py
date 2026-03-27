"""Multi-distribution ACP raw-fps data collection script.

Collects rollouts under four ACP data distributions with an ACP-specific collector
that saves every control step by default instead of reusing the WM/VLAW collector.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from rlft.acp.data_collector import ACPCollectorConfig, ACPDataCollector
from rlft.vlaw.data.noisy_policy import GaussianNoisePolicyWrapper, OUNoisePolicyWrapper


@dataclass
class DataCollectionArgs:
    noise_mode: Literal["none", "teleop", "rl_explore", "random"] = "none"

    env_id: str = "LiftPegUpright-v1"
    num_envs: int = 32
    num_episodes: int = 200
    max_episode_steps: int = 200
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "physx_cuda"
    save_every_n_steps: int = 1
    min_traj_length: int = 10
    ignore_terminations: bool = True

    checkpoint_path: str = "runs/fair_comparison/fair_comparison/awsc/best_s42__1772570560/checkpoints/best.pt"
    output_dir: str = "data/vlaw/rollouts_acp/pretrained_policy_rawfps"
    gpu_id: int = 2

    ou_theta: float = 0.15
    ou_sigma: float = 0.07
    ou_action_clip: float = 1.0
    pause_prob: float = 0.04
    hold_gripper_sigma: float = 0.02

    explore_sigma: float = 0.25
    explore_action_clip: float = 1.0
    random_sigma: float = 0.80

    seed: int = 42
    verbose: bool = True
    dry_run: bool = False


class NoisyACPDataCollector(ACPDataCollector):
    """ACP collector with policy noise wrappers."""

    def __init__(self, collector_cfg: ACPCollectorConfig, args: DataCollectionArgs) -> None:
        super().__init__(collector_cfg)
        self._args = args

    def _load_policy(self, env):
        noise_mode = self._args.noise_mode
        args = self._args

        if noise_mode == "random":
            return super()._load_policy(env)

        base_policy, visual_encoder = super()._load_policy(env)
        if noise_mode == "none":
            return base_policy, visual_encoder

        action_dim = env.action_space.shape[-1] if hasattr(env.action_space, "shape") else 7
        if noise_mode == "teleop":
            policy = OUNoisePolicyWrapper(
                policy=base_policy,
                action_dim=action_dim,
                theta=args.ou_theta,
                sigma=args.ou_sigma,
                action_clip=args.ou_action_clip,
                pause_prob=args.pause_prob,
                hold_gripper_sigma=args.hold_gripper_sigma,
                rng_seed=args.seed,
            )
        elif noise_mode == "rl_explore":
            policy = GaussianNoisePolicyWrapper(
                policy=base_policy,
                action_dim=action_dim,
                sigma=args.explore_sigma,
                action_clip=args.explore_action_clip,
                rng_seed=args.seed,
            )
        else:
            raise ValueError(f"Unknown noise_mode: {noise_mode!r}")
        return policy, visual_encoder


_SOURCE_TAGS = {
    "none": "pretrained_policy",
    "teleop": "teleop_sim",
    "rl_explore": "rl_prior",
    "random": "random",
}


def main(args: DataCollectionArgs) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    source_tag = _SOURCE_TAGS[args.noise_mode]

    print(f"[collect_acp_data] noise_mode={args.noise_mode!r} source_tag={source_tag!r}")
    print(
        f"[collect_acp_data] output_dir={args.output_dir!r} num_episodes={args.num_episodes} "
        f"save_every_n_steps={args.save_every_n_steps}"
    )
    if args.noise_mode != "random":
        print(f"[collect_acp_data] checkpoint={args.checkpoint_path!r}")

    collector_cfg = ACPCollectorConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        camera_width=128,
        camera_height=128,
        max_episode_steps=args.max_episode_steps,
        num_episodes=args.num_episodes,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        save_every_n_steps=args.save_every_n_steps,
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

    collector = NoisyACPDataCollector(collector_cfg, args)
    output_path = collector.run()
    print(f"[collect_acp_data] Saved → {output_path}")


if __name__ == "__main__":
    main(tyro.cli(DataCollectionArgs))
