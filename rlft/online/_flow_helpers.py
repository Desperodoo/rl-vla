"""
Shared helpers for Flow-policy online training scripts (DSRL, PLD, etc.).

Provides:
    * ``make_flow_train_envs(args)``  — GPU-vectorized train env creation
    * ``make_flow_eval_envs(args)``   — GPU-vectorized eval env creation
    * ``FlowVecEnvAdapter``           — torch-env → numpy VecEnv adapter
    * ``extract_success(info, num_envs)`` — ManiSkill3 success extraction

All functions accept duck-typed ``args`` that must have the fields used
by both ``train_dsrl.Args`` and ``train_pld.Args``.

NOTE: ``train_dsrl.py`` keeps its own identical copies of these utilities
      for backward compatibility.  Only new training scripts should import
      from this module.
"""

import numpy as np
import torch
from typing import Dict, Any

from rlft.envs import make_eval_envs


# =====================================================================
# Environment creation helpers
# =====================================================================

def make_flow_train_envs(args):
    """Create GPU-vectorized training environments.

    ``args`` must have:
        obs_mode, control_mode, reward_mode, max_episode_steps,
        env_id, num_envs, sim_backend, obs_horizon
    """
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    env_kwargs = dict(
        obs_mode="rgbd" if "rgb" in args.obs_mode else "state",
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    wrappers = [FlattenRGBDObservationWrapper] if "rgb" in args.obs_mode else []

    return make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        env_kwargs=env_kwargs,
        other_kwargs=dict(obs_horizon=args.obs_horizon),
        video_dir=None,
        wrappers=wrappers,
    )


def make_flow_eval_envs(args):
    """Create GPU-vectorized evaluation environments.

    ``args`` must have:
        obs_mode, control_mode, reward_mode, max_episode_steps,
        env_id, num_eval_envs, sim_backend, obs_horizon
    """
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    env_kwargs = dict(
        control_mode=args.control_mode,
        obs_mode="rgbd" if "rgb" in args.obs_mode else "state",
        render_mode="rgb_array",
        reward_mode=args.reward_mode,
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    wrappers = [FlattenRGBDObservationWrapper] if "rgb" in args.obs_mode else []

    return make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        sim_backend=args.sim_backend,
        env_kwargs=env_kwargs,
        other_kwargs=dict(obs_horizon=args.obs_horizon),
        video_dir=None,
        wrappers=wrappers,
    )


# =====================================================================
# VecEnv adapter
# =====================================================================

class FlowVecEnvAdapter:
    """Minimal adapter: flow-wrapped ManiSkill3 env → numpy-based VecEnv.

    Works with any env that follows the ``gymnasium.Wrapper`` interface
    (returns torch tensors for obs/rew/term/trunc).

    If the wrapped env has a ``step_base_only()`` method (e.g.
    ``ManiSkillResidualEnvWrapper``), it is forwarded automatically.
    """

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        obs, info = self.env.reset()
        return self._t2n(obs), info

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        done = term | trunc
        return (
            self._t2n(obs),
            self._t2n(rew),
            self._t2n(done),
            self._t2n(term),
            self._t2n(trunc),
            info,
        )

    def step_base_only(self):
        """Run one step with only base policy (zero residual).

        Delegates to ``self.env.step_base_only()``.  Raises
        ``AttributeError`` if the wrapped env does not support it.
        """
        obs, rew, term, trunc, info = self.env.step_base_only()
        done = term | trunc
        return (
            self._t2n(obs),
            self._t2n(rew),
            self._t2n(done),
            self._t2n(term),
            self._t2n(trunc),
            info,
        )

    @staticmethod
    def _t2n(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    def close(self):
        self.env.close()


# =====================================================================
# Info extraction helper
# =====================================================================

def extract_success(info: Dict[str, Any], num_envs: int) -> np.ndarray:
    """Extract per-env success flags from ManiSkill3 info dict.

    Handles both ``ManiSkillVectorEnv`` format (``final_info``) and
    the legacy per-step ``success`` tensor.
    """
    if not isinstance(info, dict):
        return np.zeros(num_envs)

    # 1. ManiSkillVectorEnv format: info["final_info"]["episode"]["success_once"]
    final_info = info.get("final_info")
    if isinstance(final_info, dict) and "episode" in final_info:
        so = final_info["episode"].get("success_once")
        if so is not None:
            if isinstance(so, torch.Tensor):
                return so.float().cpu().numpy()
            return np.asarray(so, dtype=np.float32)

    # 2. Fallback: per-step success tensor
    success = info.get("success")
    if success is not None:
        if isinstance(success, torch.Tensor):
            return success.float().cpu().numpy()
        if isinstance(success, np.ndarray):
            return success.astype(np.float32)

    return np.zeros(num_envs)
