"""ACP 高帧率数据收集器（Phase P6）

与 WM/VLAW rollout collector 分离：
- 不做世界模型用的下采样语义
- 默认保存每个控制步的原始帧
- 默认更长 episode（200）
- 支持 ignore_terminations=True 保留 success 后视觉过程
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import tyro

from rlft.vlaw.data.collector import (
    PLDSACPolicy,
    RandomPolicy,
    ShortCutFlowPolicy,
    _get_render_frame,
    _np,
    build_obs_features,
    extract_agent_state,
)


@dataclass
class ACPCollectorConfig:
    """ACP rollout 数据收集配置。"""

    env_id: str = "LiftPegUpright-v1"
    num_envs: int = 32
    camera_width: int = 128
    camera_height: int = 128
    max_episode_steps: int = 200
    num_episodes: int = 200
    sim_backend: str = "physx_cuda"
    control_mode: str = "pd_ee_delta_pose"

    obs_horizon: int = 2
    act_steps: int = 8
    include_rgb: bool = True
    visual_feature_dim: int = 256

    checkpoint_path: str = ""
    use_random_policy: bool = False

    save_every_n_steps: int = 1
    min_traj_length: int = 10
    ignore_terminations: bool = True

    gpu_id: int = 0
    output_dir: str = "data/vlaw/rollouts_acp/pretrained_policy_rawfps"
    source_tag: str = "pretrained_policy"
    task_instruction: str = ""

    dry_run: bool = False
    verbose: bool = True


class Trajectory:
    """单条 ACP 轨迹缓冲。"""

    def __init__(self, env_idx: int) -> None:
        self.env_idx = env_idx
        self.rgb_base: list[np.ndarray] = []
        self.rgb_render: list[np.ndarray] = []
        self.state: list[np.ndarray] = []
        self.obs_agent: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.env_success: list[bool] = []

    def append(
        self,
        rgb_base: np.ndarray,
        rgb_render: np.ndarray,
        state: np.ndarray,
        obs_agent: np.ndarray,
        action: np.ndarray,
        success: bool,
    ) -> None:
        self.rgb_base.append(rgb_base)
        self.rgb_render.append(rgb_render)
        self.state.append(state)
        self.obs_agent.append(obs_agent)
        self.actions.append(action)
        self.env_success.append(success)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "rgb_base": np.stack(self.rgb_base).astype(np.uint8),
            "rgb_render": np.stack(self.rgb_render).astype(np.uint8),
            "state": np.stack(self.state).astype(np.float32),
            "obs_agent": np.stack(self.obs_agent).astype(np.float32),
            "actions": np.stack(self.actions).astype(np.float32),
            "env_success": np.array(self.env_success, dtype=bool),
        }

    def __len__(self) -> int:
        return len(self.actions)


class ACPDataCollector:
    """ACP 专用 ManiSkill collector。"""

    def __init__(self, cfg: ACPCollectorConfig) -> None:
        self.cfg = cfg
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ACP-COLLECT] device={self.device} gpu={cfg.gpu_id}")

    def _make_env(self):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

        env_kwargs: dict = dict(
            obs_mode="rgbd",
            render_mode="rgb_array",
            control_mode=self.cfg.control_mode,
            max_episode_steps=self.cfg.max_episode_steps,
        )
        if self.cfg.sim_backend == "physx_cpu":
            env = gym.make(self.cfg.env_id, **env_kwargs)
        else:
            env = gym.make(
                self.cfg.env_id,
                num_envs=self.cfg.num_envs,
                sim_backend=self.cfg.sim_backend,
                **env_kwargs,
            )
        if self.cfg.include_rgb:
            env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
        print(
            f"[ACP-COLLECT] env={self.cfg.env_id} num_envs={self.cfg.num_envs} "
            f"action_space={env.action_space.shape} save_every_n_steps={self.cfg.save_every_n_steps}"
        )
        return env

    def _load_policy(self, env) -> tuple:
        if self.cfg.use_random_policy:
            print("[ACP-COLLECT] using random policy")
            return RandomPolicy(env.action_space), None

        if not self.cfg.checkpoint_path:
            raise ValueError("checkpoint_path is required when use_random_policy=False")

        raw = torch.load(self.cfg.checkpoint_path, map_location="cpu", weights_only=False)
        agent_sd = raw.get("agent", {})
        has_velocity_net = any(k.startswith("velocity_net.") for k in agent_sd.keys())
        is_pld = (
            isinstance(raw, dict)
            and "agent" in raw
            and "config" in raw
            and not has_velocity_net
        )

        from rlft.networks import PlainConv
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy

        if is_pld:
            return self._load_pld_policy(raw)

        ckpt_config = raw.get("config", {})
        pred_horizon = int(ckpt_config.get("pred_horizon", 16))
        wrapper, visual_encoder, _state_dim = load_shortcut_flow_policy(
            self.cfg.checkpoint_path,
            visual_encoder_class=PlainConv if self.cfg.include_rgb else None,
            obs_horizon=self.cfg.obs_horizon,
            pred_horizon=pred_horizon,
            visual_feature_dim=self.cfg.visual_feature_dim,
            include_rgb=self.cfg.include_rgb,
            device=str(self.device),
        )
        policy = ShortCutFlowPolicy(
            wrapper=wrapper,
            visual_encoder=visual_encoder,
            device=self.device,
            include_rgb=self.cfg.include_rgb,
            obs_horizon=self.cfg.obs_horizon,
            action_pred_horizon=pred_horizon,
            act_steps=min(self.cfg.act_steps, pred_horizon),
        )
        return policy, visual_encoder

    def _load_pld_policy(self, ckpt: dict) -> tuple:
        from rlft.algorithms.online_rl.pld_sac import PLDActor
        from rlft.networks import PlainConv
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy

        cfg_dict = ckpt["config"]
        act_steps = int(cfg_dict.get("act_steps", 8))
        pred_horizon = int(cfg_dict.get("pred_horizon", 16))
        action_dim = int(cfg_dict.get("action_dim", 7))
        visual_feature_dim = int(cfg_dict.get("visual_feature_dim", 256))
        obs_horizon = int(cfg_dict.get("obs_horizon", 2))
        use_ema = bool(cfg_dict.get("use_ema", True))
        action_scale = float(cfg_dict.get("action_scale", 0.3))
        base_ckpt_path = cfg_dict.get("checkpoint", "")

        visual_encoder: Optional[torch.nn.Module] = None
        if self.cfg.include_rgb and "visual_encoder" in ckpt:
            ve_sd = ckpt["visual_encoder"]
            fc_in_dim = ve_sd["fc.0.weight"].shape[1] if "fc.0.weight" in ve_sd else None
            pool_feature_map = fc_in_dim == 128
            visual_encoder = PlainConv(
                in_channels=3,
                out_dim=visual_feature_dim,
                pool_feature_map=pool_feature_map,
            ).to(self.device)
            visual_encoder.load_state_dict(ve_sd)
            visual_encoder.eval()

        agent_sd = ckpt["agent"]
        obs_dim = int(agent_sd["actor.trunk.0.weight"].shape[1])
        residual_dim = int(agent_sd["actor.mean_head.weight"].shape[0])
        hidden_dims = []
        idx = 0
        while f"actor.trunk.{idx}.weight" in agent_sd:
            hidden_dims.append(int(agent_sd[f"actor.trunk.{idx}.weight"].shape[0]))
            idx += 2

        actor = PLDActor(
            obs_dim=obs_dim,
            residual_dim=residual_dim,
            hidden_dims=hidden_dims,
            action_scale=action_scale,
        ).to(self.device)
        actor.load_state_dict(
            {k.removeprefix("actor."): v for k, v in agent_sd.items() if k.startswith("actor.")}
        )
        actor.eval()

        if not base_ckpt_path:
            raise ValueError("PLD checkpoint missing config.checkpoint")
        root = Path(__file__).resolve().parents[2]
        base_path = Path(base_ckpt_path)
        if not base_path.is_absolute():
            base_path = root / base_path

        base_flow, _base_ve, _state_dim = load_shortcut_flow_policy(
            str(base_path),
            visual_encoder_class=None,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            visual_feature_dim=visual_feature_dim,
            include_rgb=self.cfg.include_rgb,
            use_ema=use_ema,
            device=str(self.device),
        )
        policy = PLDSACPolicy(
            actor=actor,
            base_flow=base_flow,
            visual_encoder=visual_encoder,
            device=self.device,
            act_steps=act_steps,
            action_dim=action_dim,
        )
        return policy, visual_encoder

    def collect_rollouts(self, policy=None, visual_encoder: Optional[torch.nn.Module] = None) -> list[dict]:
        cfg = self.cfg
        env = self._make_env()
        if policy is None:
            policy, visual_encoder = self._load_policy(env)

        n_envs = cfg.num_envs
        h, w = cfg.camera_height, cfg.camera_width
        target_episodes = 3 if cfg.dry_run else cfg.num_episodes
        task_instruction = cfg.task_instruction or cfg.env_id.replace("-v1", "")

        completed_trajs: list[dict] = []
        active_trajs = [Trajectory(i) for i in range(n_envs)]
        step_in_episode = np.zeros(n_envs, dtype=int)

        state_dim = None
        state_history: Optional[np.ndarray] = None
        rgb_history: Optional[np.ndarray] = None

        obs, _ = env.reset(seed=42)
        discarded_short = 0
        t_start = time.perf_counter()

        print(
            f"[ACP-COLLECT] start target_episodes={target_episodes} max_episode_steps={cfg.max_episode_steps} "
            f"ignore_terminations={cfg.ignore_terminations}"
        )

        while len(completed_trajs) < target_episodes:
            if isinstance(obs, dict) and "state" in obs and "rgb" in obs:
                rgb_base = _np(obs["rgb"]).astype(np.uint8)
                agent_state = _np(obs["state"]).astype(np.float32)
            else:
                rgb_base = _np(obs["rgb"]).astype(np.uint8) if isinstance(obs, dict) and "rgb" in obs else None
                if rgb_base is None:
                    raise RuntimeError("ACP collector requires flattened RGB observations")
                agent_state = extract_agent_state(obs)
            rgb_render = _get_render_frame(env, n_envs, h, w, rgb_base)

            if state_dim is None:
                state_dim = agent_state.shape[-1]
                state_history = np.tile(agent_state[:, np.newaxis, :], (1, cfg.obs_horizon, 1))
                rgb_history = np.tile(rgb_base[:, np.newaxis, :, :, :], (1, cfg.obs_horizon, 1, 1, 1))

            state_history = np.roll(state_history, shift=-1, axis=1)
            state_history[:, -1, :] = agent_state
            rgb_history = np.roll(rgb_history, shift=-1, axis=1)
            rgb_history[:, -1, :] = rgb_base

            if isinstance(policy, RandomPolicy):
                action_chunk = policy.get_actions(torch.zeros(n_envs, cfg.obs_horizon, 1))
                if action_chunk.ndim == 2:
                    action_chunk = action_chunk[:, np.newaxis, :]
            else:
                obs_features = build_obs_features(
                    obs=obs,
                    state_history=state_history,
                    rgb_history=rgb_history,
                    visual_encoder=visual_encoder,
                    include_rgb=cfg.include_rgb,
                    visual_feature_dim=cfg.visual_feature_dim,
                    device=self.device,
                )
                action_chunk = policy.get_actions(obs_features)

            done_in_chunk = np.zeros(n_envs, dtype=bool)
            for t_chunk in range(action_chunk.shape[1]):
                actions = action_chunk[:, t_chunk, :]

                if t_chunk > 0:
                    if isinstance(obs, dict) and "state" in obs and "rgb" in obs:
                        rgb_base = _np(obs["rgb"]).astype(np.uint8)
                        agent_state = _np(obs["state"]).astype(np.float32)
                    else:
                        raise RuntimeError("ACP collector requires flattened RGB observations")
                    rgb_render = _get_render_frame(env, n_envs, h, w, rgb_base)
                    state_history = np.roll(state_history, shift=-1, axis=1)
                    state_history[:, -1, :] = agent_state
                    rgb_history = np.roll(rgb_history, shift=-1, axis=1)
                    rgb_history[:, -1, :] = rgb_base

                obs, _reward, terminated, truncated, info = env.step(actions)
                terminated_np = _np(terminated).astype(bool)
                truncated_np = _np(truncated).astype(bool)
                if cfg.ignore_terminations:
                    terminated_np[:] = False
                done = np.logical_or(terminated_np, truncated_np)

                if "success" in info:
                    success_arr = _np(info["success"]).astype(bool)
                elif "episode" in info and "success" in info["episode"]:
                    success_arr = _np(info["episode"]["success"]).astype(bool)
                else:
                    success_arr = np.zeros(n_envs, dtype=bool)

                for i in range(n_envs):
                    if done_in_chunk[i]:
                        step_in_episode[i] += 1
                        continue
                    step_in_episode[i] += 1
                    is_first = step_in_episode[i] == 1
                    is_regular = step_in_episode[i] % cfg.save_every_n_steps == 0
                    is_final = bool(done[i])
                    if is_first or is_regular or is_final:
                        active_trajs[i].append(
                            rgb_base=rgb_base[i],
                            rgb_render=rgb_render[i],
                            state=agent_state[i],
                            obs_agent=agent_state[i],
                            action=actions[i],
                            success=bool(success_arr[i]),
                        )

                newly_done = np.where(done & ~done_in_chunk)[0]
                for i in newly_done:
                    if len(completed_trajs) >= target_episodes:
                        break
                    traj = active_trajs[i]
                    if len(traj) >= cfg.min_traj_length:
                        traj_dict = traj.to_arrays()
                        traj_dict["task_instruction"] = task_instruction
                        traj_dict["source"] = cfg.source_tag
                        completed_trajs.append(traj_dict)
                        if cfg.verbose:
                            print(
                                f"[ACP-COLLECT] episode {len(completed_trajs):4d}/{target_episodes} "
                                f"env={i:3d} T={len(traj):4d} success_at_end={'✅' if bool(traj_dict['env_success'][-1]) else '❌'}"
                            )
                    else:
                        discarded_short += 1
                    active_trajs[i] = Trajectory(i)
                    step_in_episode[i] = 0
                    done_in_chunk[i] = True

                if len(newly_done) > 0:
                    reset_idx = torch.tensor(list(newly_done), dtype=torch.int64, device="cuda")
                    try:
                        obs, _ = env.reset(options={"env_idx": reset_idx})
                    except Exception:
                        obs, _ = env.reset()
                    if isinstance(obs, dict) and "state" in obs and "rgb" in obs:
                        st = _np(obs["state"]).astype(np.float32)
                        rgb = _np(obs["rgb"]).astype(np.uint8)
                        for ri in newly_done:
                            for th in range(cfg.obs_horizon):
                                state_history[ri, th] = st[ri]
                                rgb_history[ri, th] = rgb[ri]

                if len(completed_trajs) >= target_episodes:
                    break

            if len(completed_trajs) >= target_episodes:
                break

        elapsed = time.perf_counter() - t_start
        success_rate = sum(bool(t["env_success"][-1]) for t in completed_trajs) / max(len(completed_trajs), 1)
        lengths = [t["actions"].shape[0] for t in completed_trajs]
        print("\n[ACP-COLLECT] === done ===")
        print(f"  valid={len(completed_trajs)} discarded_short={discarded_short} success_at_end={success_rate:.1%}")
        if lengths:
            print(f"  length min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f} median={np.median(lengths):.1f}")
        print(f"  elapsed={elapsed:.1f}s")
        env.close()
        return completed_trajs

    def save_hdf5(self, trajectories: list[dict], output_path: Optional[str] = None) -> Path:
        if output_path is None:
            out_dir = Path(self.cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            output_path = str(out_dir / f"{self.cfg.env_id}_{self.cfg.source_tag}_{ts}.h5")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        success_count = sum(bool(t["env_success"][-1]) for t in trajectories)
        success_rate = success_count / max(len(trajectories), 1)

        with h5py.File(str(out_path), "w") as f:
            meta = f.create_group("meta")
            meta.attrs["num_trajectories"] = len(trajectories)
            meta.attrs["success_rate"] = float(success_rate)
            meta.attrs["env_id"] = self.cfg.env_id
            meta.attrs["camera_hw"] = f"{self.cfg.camera_height},{self.cfg.camera_width}"
            meta.attrs["source"] = self.cfg.source_tag
            meta.attrs["save_every_n_steps"] = self.cfg.save_every_n_steps
            meta.attrs["collector"] = "acp_rawfps"
            meta.attrs["ignore_terminations"] = bool(self.cfg.ignore_terminations)

            for idx, traj in enumerate(trajectories):
                grp = f.create_group(f"traj_{idx:04d}")
                for key, arr in traj.items():
                    if isinstance(arr, np.ndarray):
                        grp.create_dataset(
                            key,
                            data=arr,
                            chunks=True,
                            compression="gzip",
                            compression_opts=1,
                        )
                grp.attrs["task_instruction"] = traj.get("task_instruction", "")
                grp.attrs["source"] = traj.get("source", "real")
                grp.attrs["success"] = bool(traj["env_success"][-1])

        print(f"[ACP-COLLECT] saved {out_path} ({len(trajectories)} trajs, success_at_end={success_rate:.1%})")
        return out_path

    def run(self) -> Path:
        trajs = self.collect_rollouts()
        if self.cfg.dry_run:
            print("[ACP-COLLECT] dry_run=True, skip saving")
            return Path("/dev/null")
        return self.save_hdf5(trajs)


if __name__ == "__main__":
    cfg = tyro.cli(ACPCollectorConfig)
    ACPDataCollector(cfg).run()
