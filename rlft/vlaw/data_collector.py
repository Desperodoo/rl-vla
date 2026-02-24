"""ManiSkill Rollout 数据收集器.

P1.1 阶段: 在 ManiSkill GPU 向量化环境中用 ShortCut Flow (或随机策略) 收集轨迹，
保存为 HDF5 格式供 Ctrl-World 世界模型训练使用。

数据格式见 VLAW_REPRODUCTION_PLAN.md 第 3.1.2 节。

所属阶段: P1.1 — ManiSkill Rollout 收集器
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import h5py
import numpy as np
import torch
import tyro


# ---------------------------------------------------------------------------
# 策略协议 (Protocol)
# ---------------------------------------------------------------------------

@runtime_checkable
class PolicyProtocol(Protocol):
    """数据收集器接受的策略接口."""

    def get_actions(
        self,
        obs_features: torch.Tensor,   # (N_env, obs_horizon, feat_dim)
    ) -> np.ndarray:
        """返回 (N_env, action_dim) 的动作数组."""
        ...


class RandomPolicy:
    """随机策略，用于基线数据收集。"""

    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def get_actions(self, obs_features: torch.Tensor) -> np.ndarray:
        """在向量化环境中 sample() 本身返回 (N, action_dim)."""
        sample = self.action_space.sample()
        if isinstance(sample, torch.Tensor):
            sample = sample.cpu().numpy()
        arr = np.asarray(sample)
        # 若 action_space 返回 (action_dim,) 则扩展到 (N, action_dim)
        if arr.ndim == 1:
            N = obs_features.shape[0]
            arr = np.stack([arr] * N)
        return arr


class ShortCutFlowPolicy:
    """包装 ShortCutFlowWrapper 适配 PolicyProtocol.

    Args:
        wrapper: 加载好的 ShortCutFlowWrapper 实例
        visual_encoder: PlainConv 视觉编码器 (可为 None)
        device: 计算设备
        include_rgb: 是否包含 RGB 特征
        obs_horizon: 观测历史长度
        action_pred_horizon: 动作预测长度
        act_steps: 每步执行的动作数量
    """

    def __init__(
        self,
        wrapper,
        visual_encoder: Optional[torch.nn.Module],
        device: torch.device,
        include_rgb: bool = True,
        obs_horizon: int = 2,
        action_pred_horizon: int = 16,
        act_steps: int = 8,
    ) -> None:
        self.wrapper = wrapper
        self.visual_encoder = visual_encoder
        self.device = device
        self.include_rgb = include_rgb
        self.obs_horizon = obs_horizon
        self.action_pred_horizon = action_pred_horizon
        self.act_steps = act_steps

    @torch.no_grad()
    def get_actions(self, obs_features: torch.Tensor) -> np.ndarray:
        """obs_features: (N_env, obs_horizon * feat_dim)."""
        B = obs_features.shape[0]
        noise = torch.randn(
            B, self.action_pred_horizon, self.wrapper.action_dim,
            device=self.device,
        )
        actions = self.wrapper(
            obs=obs_features,
            initial_noise=noise,
            return_numpy=True,
            act_steps=self.act_steps,
        )
        # actions: (B, act_steps, action_dim) → 取第一步 (B, action_dim)
        return actions[:, 0, :]


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class CollectorConfig:
    """P1.1 Rollout 数据收集配置."""

    # 环境
    env_id: str = "PickCube-v1"
    """ManiSkill 任务 ID"""

    num_envs: int = 64
    """并行环境数 (GPU 向量化)"""

    camera_width: int = 192
    """相机分辨率宽"""

    camera_height: int = 192
    """相机分辨率高"""

    max_episode_steps: int = 200
    """每幕最大步数"""

    num_episodes: int = 50
    """目标采集幕数 (不足时继续直到达到)"""

    sim_backend: str = "physx_cuda"
    """仿真后端: physx_cuda 或 physx_cpu"""

    control_mode: str = "pd_ee_delta_pose"
    """控制模式"""

    obs_horizon: int = 2
    """观测历史窗口大小"""

    act_steps: int = 8
    """每次策略调用实际执行的步数"""

    # 策略
    use_random_policy: bool = False
    """是否使用随机策略 (True: 随机; False: ShortCut Flow)"""

    checkpoint_path: str = ""
    """ShortCut Flow checkpoint 路径 (use_random_policy=False 时必填)"""

    include_rgb: bool = True
    """是否使用 RGB 观测 (False 时退化为 state-only)"""

    visual_feature_dim: int = 256
    """PlainConv 输出维度"""

    # 帧率控制
    frame_skip: int = 3
    """ManiSkill 控制频率 / 保存频率 = frame_skip (下采样到 ~5Hz)"""

    # GPU
    gpu_id: int = 4
    """使用的 GPU"""

    # 输出
    output_dir: str = "data/vlaw/rollouts/iter0"
    """HDF5 数据保存目录"""

    source_tag: str = "real"
    """数据来源标签: 'real' 或 'synthetic'"""

    task_instruction: str = ""
    """任务语言描述 (空则自动从 env_id 推断)"""

    # 调试
    dry_run: bool = False
    """True: 只运行 3 幕，不保存文件"""

    verbose: bool = True
    """是否打印详细日志"""


# ---------------------------------------------------------------------------
# 观测处理工具
# ---------------------------------------------------------------------------

def _np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Tensor → numpy, 保持在 CPU."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def extract_raw_frames(
    obs: dict,
    camera_height: int,
    camera_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """从 ManiSkill obs 提取 base_camera RGB 和 render 帧.

    Args:
        obs: ManiSkill step obs dict
        camera_height: 目标相机高度
        camera_width: 目标相机宽度

    Returns:
        (rgb_base, rgb_render): 各自 uint8 (N, H, W, 3) 数组
    """
    sensor_data = obs.get("sensor_data", {})
    base_rgb = sensor_data.get("base_camera", {}).get("rgb")

    if base_rgb is not None:
        rgb_base = _np(base_rgb).astype(np.uint8)  # (N, H, W, 3)
    else:
        N = next(iter(obs.get("agent", {}).values())).shape[0]
        rgb_base = np.zeros((N, camera_height, camera_width, 3), dtype=np.uint8)

    return rgb_base


def extract_agent_state(obs: dict) -> np.ndarray:
    """拼接 obs['agent'] 所有数组 → (N, state_dim) float32."""
    arrays = []
    for v in obs.get("agent", {}).values():
        arr = _np(v).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        arrays.append(arr)
    for v in obs.get("extra", {}).values():
        arr = _np(v).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.dtype == bool or arr.dtype == np.bool_:
            arr = arr.astype(np.float32)
        arrays.append(arr)
    if not arrays:
        raise RuntimeError("obs['agent'] is empty — cannot extract state")
    return np.concatenate(arrays, axis=-1)


def build_obs_features(
    obs: dict,
    state_history: np.ndarray,      # (N, obs_horizon, state_dim)
    rgb_history: np.ndarray,        # (N, obs_horizon, H, W, 3) uint8
    visual_encoder: Optional[torch.nn.Module],
    include_rgb: bool,
    visual_feature_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """拼接视觉特征 + 状态特征 → (N, obs_horizon * feat_dim)."""
    from rlft.datasets.data_utils import encode_observations
    N, T = state_history.shape[:2]

    obs_seq: dict = {
        "state": torch.from_numpy(state_history).to(device),   # (N, T, state_dim)
    }

    if include_rgb and visual_encoder is not None:
        # 转为 (N, T, C, H, W) float
        rgb_t = torch.from_numpy(rgb_history).to(device)  # (N, T, H, W, 3)
        rgb_t = rgb_t.permute(0, 1, 4, 2, 3).float() / 255.0  # → (N, T, 3, H, W)
        obs_seq["rgb"] = rgb_t

    obs_features = encode_observations(
        obs_seq=obs_seq,
        visual_encoder=visual_encoder,
        include_rgb=include_rgb,
        device=device,
        flatten=True,   # (N, T * feat_dim)
    )
    return obs_features


# ---------------------------------------------------------------------------
# 轨迹存储
# ---------------------------------------------------------------------------

class Trajectory:
    """单条轨迹的内存缓冲区."""

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
        rgb_base: np.ndarray,       # (H, W, 3)
        rgb_render: np.ndarray,     # (H, W, 3)
        state: np.ndarray,          # (state_dim,)
        obs_agent: np.ndarray,      # (agent_dim,)
        action: np.ndarray,         # (action_dim,)
        success: bool,
    ) -> None:
        self.rgb_base.append(rgb_base)
        self.rgb_render.append(rgb_render)
        self.state.append(state)
        self.obs_agent.append(obs_agent)
        self.actions.append(action)
        self.env_success.append(success)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """合并成 numpy 数组字典."""
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


# ---------------------------------------------------------------------------
# 主收集器
# ---------------------------------------------------------------------------

class VLAWDataCollector:
    """ManiSkill GPU 向量化环境的 VLAW 轨迹收集器.

    支持:
    - ShortCut Flow 策略 rollout (用于 D_real)
    - 随机策略 rollout (用于探索基线)
    - HDF5 格式保存

    Args:
        cfg: 收集器配置
    """

    def __init__(self, cfg: CollectorConfig) -> None:
        self.cfg = cfg
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VLAW-P1.1] 设备: {self.device} (GPU {cfg.gpu_id})")

    # ------------------------------------------------------------------
    # 环境创建
    # ------------------------------------------------------------------

    def _make_env(self):
        """创建 ManiSkill GPU 向量化环境."""
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        env_kwargs: dict = dict(
            obs_mode="rgbd",
            render_mode="rgb_array",
            control_mode=self.cfg.control_mode,
            sensor_configs=dict(
                width=self.cfg.camera_width,
                height=self.cfg.camera_height,
            ),
        )
        if self.cfg.max_episode_steps:
            env_kwargs["max_episode_steps"] = self.cfg.max_episode_steps

        if self.cfg.sim_backend == "physx_cpu":
            env = gym.make(self.cfg.env_id, **env_kwargs)
        else:
            env = gym.make(
                self.cfg.env_id,
                num_envs=self.cfg.num_envs,
                sim_backend=self.cfg.sim_backend,
                **env_kwargs,
            )

        print(f"[VLAW-P1.1] 环境创建: {self.cfg.env_id} "
              f"num_envs={self.cfg.num_envs} "
              f"action_space={env.action_space.shape}")
        return env

    # ------------------------------------------------------------------
    # 策略加载
    # ------------------------------------------------------------------

    def _load_policy(self, env) -> tuple:
        """加载策略，返回 (policy, visual_encoder)."""
        if self.cfg.use_random_policy:
            print("[VLAW-P1.1] 使用随机策略")
            return RandomPolicy(env.action_space), None

        if not self.cfg.checkpoint_path:
            raise ValueError(
                "use_random_policy=False 时必须提供 checkpoint_path"
            )

        print(f"[VLAW-P1.1] 加载 ShortCut Flow: {self.cfg.checkpoint_path}")
        from rlft.networks import PlainConv
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy

        wrapper, visual_encoder, _state_dim = load_shortcut_flow_policy(
            self.cfg.checkpoint_path,
            visual_encoder_class=PlainConv if self.cfg.include_rgb else None,
            obs_horizon=self.cfg.obs_horizon,
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
            act_steps=self.cfg.act_steps,
        )
        return policy, visual_encoder

    # ------------------------------------------------------------------
    # 主收集循环
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        policy=None,
        visual_encoder: Optional[torch.nn.Module] = None,
    ) -> list[dict]:
        """执行 rollout 并返回轨迹列表.

        Args:
            policy: 策略 (None 时自动加载)
            visual_encoder: 视觉编码器 (None 时自动加载)

        Returns:
            List of trajectory dicts, 每个 dict 包含 numpy 数组
        """
        cfg = self.cfg
        env = self._make_env()

        if policy is None:
            policy, visual_encoder = self._load_policy(env)

        N = cfg.num_envs
        H, W = cfg.camera_height, cfg.camera_width
        target_episodes = 3 if cfg.dry_run else cfg.num_episodes

        task_instruction = cfg.task_instruction or cfg.env_id.replace("-v1", "")

        completed_trajs: list[dict] = []
        active_trajs: list[Trajectory] = [Trajectory(i) for i in range(N)]

        # 历史缓冲 (用于 obs_horizon)
        state_dim = None
        state_history: Optional[np.ndarray] = None   # (N, T, state_dim)
        rgb_history: Optional[np.ndarray] = None     # (N, T, H, W, 3)

        # 重置
        obs, _ = env.reset(seed=42)
        step = 0
        episode_count = 0
        t_start = time.perf_counter()

        print(f"[VLAW-P1.1] 开始收集: 目标 {target_episodes} 条轨迹")

        while episode_count < target_episodes:
            # ---- 提取当前 obs ----
            rgb_base = extract_raw_frames(obs, H, W)       # (N, H, W, 3)
            agent_state = extract_agent_state(obs)          # (N, state_dim)

            # 获取 render 帧 (第二视角)
            try:
                render_out = env.render()
                if isinstance(render_out, torch.Tensor):
                    render_out = render_out.cpu().numpy()
                if render_out is not None:
                    if render_out.ndim == 4:
                        rgb_render = render_out.astype(np.uint8)  # (N, H, W, 3)
                    else:
                        rgb_render = np.stack([render_out] * N).astype(np.uint8)
                    # 缩放到目标分辨率
                    if rgb_render.shape[1] != H or rgb_render.shape[2] != W:
                        from PIL import Image as PILImage
                        resized = np.zeros((N, H, W, 3), dtype=np.uint8)
                        for i in range(N):
                            resized[i] = np.asarray(
                                PILImage.fromarray(rgb_render[i]).resize(
                                    (W, H), PILImage.BILINEAR
                                )
                            )
                        rgb_render = resized
                else:
                    rgb_render = np.zeros_like(rgb_base)
            except Exception:
                rgb_render = np.zeros_like(rgb_base)

            # 初始化历史缓冲
            if state_dim is None:
                state_dim = agent_state.shape[-1]
                state_history = np.zeros(
                    (N, cfg.obs_horizon, state_dim), dtype=np.float32
                )
                rgb_history = np.zeros(
                    (N, cfg.obs_horizon, H, W, 3), dtype=np.uint8
                )

            # 滚动历史
            state_history = np.roll(state_history, shift=-1, axis=1)
            state_history[:, -1, :] = agent_state
            rgb_history = np.roll(rgb_history, shift=-1, axis=1)
            rgb_history[:, -1, :] = rgb_base

            # ---- 生成动作 ----
            if isinstance(policy, RandomPolicy):
                actions = policy.get_actions(
                    torch.zeros(N, cfg.obs_horizon, 1)
                )  # obs_features 未使用
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
                actions = policy.get_actions(obs_features)
            # actions: (N, action_dim)

            # ---- 步进环境 ----
            obs, _reward, terminated, truncated, info = env.step(actions)
            done = np.logical_or(
                _np(terminated).astype(bool),
                _np(truncated).astype(bool),
            )
            step += 1

            # ---- 提取 success ----
            success_arr: np.ndarray
            if "success" in info:
                s = info["success"]
                success_arr = _np(s).astype(bool)
            elif "episode" in info and "success" in info["episode"]:
                s = info["episode"]["success"]
                success_arr = _np(s).astype(bool)
            else:
                success_arr = np.zeros(N, dtype=bool)

            # ---- 记录帧 (frame_skip 下采样) ----
            if step % cfg.frame_skip == 0:
                for i in range(N):
                    active_trajs[i].append(
                        rgb_base=rgb_base[i],
                        rgb_render=rgb_render[i],
                        state=agent_state[i],
                        obs_agent=agent_state[i],
                        action=actions[i],
                        success=bool(success_arr[i]),
                    )

            # ---- 幕结束处理 ----
            if np.any(done) or (step % cfg.max_episode_steps == 0 and step > 0):
                done_indices = np.where(done)[0]
                if len(done_indices) == 0 and step % cfg.max_episode_steps == 0:
                    done_indices = np.arange(N)

                for i in done_indices:
                    traj = active_trajs[i]
                    if len(traj) > 0:
                        traj_dict = traj.to_arrays()
                        traj_dict["task_instruction"] = task_instruction
                        traj_dict["source"] = cfg.source_tag
                        completed_trajs.append(traj_dict)
                        episode_count += 1
                        if cfg.verbose:
                            final_success = traj_dict["env_success"].any()
                            print(
                                f"[VLAW-P1.1] 幕 {episode_count:4d}/{target_episodes} "
                                f"env={i:3d} T={len(traj):4d} "
                                f"success={'✅' if final_success else '❌'}"
                            )

                    # 重置该环境
                    active_trajs[i] = Trajectory(i)
                    state_history[i] = 0
                    rgb_history[i] = 0

                    if episode_count >= target_episodes:
                        break

                # 批量重置 done 的环境
                if np.any(done) and not all(done):
                    try:
                        env.unwrapped.reset(
                            seed=None,
                            options={"env_idx": torch.tensor(done_indices)},
                        )
                    except Exception:
                        pass

                if episode_count >= target_episodes:
                    break

        elapsed = time.perf_counter() - t_start
        sr = sum(t["env_success"].any() for t in completed_trajs) / max(len(completed_trajs), 1)
        print(f"[VLAW-P1.1] 收集完成: {len(completed_trajs)} 条轨迹, "
              f"成功率={sr:.1%}, 耗时={elapsed:.1f}s")

        env.close()
        return completed_trajs

    # ------------------------------------------------------------------
    # HDF5 保存
    # ------------------------------------------------------------------

    def save_hdf5(
        self,
        trajectories: list[dict],
        output_path: Optional[str] = None,
    ) -> Path:
        """将轨迹列表保存为 HDF5 文件.

        文件结构:
            /traj_0000/   ← 第 0 条轨迹
                rgb_base       (T, H, W, 3) uint8
                rgb_render     (T, H, W, 3) uint8
                state          (T, state_dim) float32
                obs_agent      (T, agent_dim) float32
                actions        (T, action_dim) float32
                env_success    (T,) bool
                task_instruction  str attr
                source            str attr
            /traj_0001/ ...
            /meta/
                num_trajectories  int attr
                success_rate      float attr
                env_id            str attr
                camera_hw         str attr  "H,W"

        Args:
            trajectories: collect_rollouts 返回的轨迹列表
            output_path: HDF5 文件路径 (None 则自动生成)

        Returns:
            保存路径 Path 对象
        """
        if output_path is None:
            out_dir = Path(self.cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            output_path = str(
                out_dir / f"{self.cfg.env_id}_{self.cfg.source_tag}_{ts}.h5"
            )

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        success_count = sum(t["env_success"].any() for t in trajectories)
        sr = success_count / max(len(trajectories), 1)

        with h5py.File(str(out_path), "w") as f:
            # --- meta ---
            meta = f.create_group("meta")
            meta.attrs["num_trajectories"] = len(trajectories)
            meta.attrs["success_rate"] = float(sr)
            meta.attrs["env_id"] = self.cfg.env_id
            meta.attrs["camera_hw"] = f"{self.cfg.camera_height},{self.cfg.camera_width}"
            meta.attrs["source"] = self.cfg.source_tag
            meta.attrs["frame_skip"] = self.cfg.frame_skip

            # --- per-trajectory ---
            for idx, traj in enumerate(trajectories):
                grp = f.create_group(f"traj_{idx:04d}")
                for key, arr in traj.items():
                    if isinstance(arr, np.ndarray):
                        grp.create_dataset(
                            key, data=arr,
                            chunks=True, compression="gzip", compression_opts=1,
                        )
                grp.attrs["task_instruction"] = traj.get("task_instruction", "")
                grp.attrs["source"] = traj.get("source", "real")
                grp.attrs["success"] = bool(traj["env_success"].any())

        print(f"[VLAW-P1.1] HDF5 已保存: {out_path} "
              f"({len(trajectories)} 条, 成功率={sr:.1%})")
        return out_path

    # ------------------------------------------------------------------
    # 一站式入口
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """完整数据收集流程: 环境 → 策略 → 收集 → 保存.

        Returns:
            保存的 HDF5 文件路径
        """
        trajs = self.collect_rollouts()
        if self.cfg.dry_run:
            print("[VLAW-P1.1] dry_run=True, 跳过 HDF5 保存")
            return Path("/dev/null")
        return self.save_hdf5(trajs)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = tyro.cli(CollectorConfig)
    collector = VLAWDataCollector(cfg)
    collector.run()
