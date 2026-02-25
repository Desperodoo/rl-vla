"""ManiSkill Demo 数据准备工具 (P1.3).

将 ManiSkill 内置 demo HDF5 (trajectory.rgb.*.h5) 转换为 VLAW 统一 HDF5 格式，
供 Ctrl-World 世界模型训练使用。

工作流:
    1. 确认 trajectory.rgb.*.h5 存在 (否则先 replay trajectory.none.*.h5)
    2. 读取 obs/sensor_data/base_camera/rgb (T, H, W, 3)
    3. 若分辨率不符 (如 128→192)，PIL resize
    4. 由于标准 replay 只有 single base_camera，复用 rgb_base 作为 rgb_render
    5. 拼接 obs/agent/* + obs/extra/* → state
    6. 对齐 actions (T-1) 与 obs (T) — 按 actions 长度截断
    7. 写出 VLAW HDF5 格式 (与 data_collector.py 一致)

所属阶段: P1.3 — 演示数据准备

使用:
    # 直接转换已有 rgb demo (LiftPegUpright-v1 已有)
    python -m rlft.vlaw.demo_prep \\
        --env_id LiftPegUpright-v1 \\
        --num_trajs 25 \\
        --target_hw 192

    # 自动 replay + 转换 (PickCube-v1 等只有 none 文件)
    python -m rlft.vlaw.demo_prep \\
        --env_id PickCube-v1 \\
        --num_trajs 25 \\
        --auto_replay \\
        --num_envs 64 \\
        --target_hw 192
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import tyro


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class DemoPrepConfig:
    """P1.3 Demo 数据准备配置."""

    env_id: str = "LiftPegUpright-v1"
    """任务 ID"""

    control_mode: str = "pd_ee_delta_pose"
    """控制模式 (需与 ShortCut Flow 训练时一致)"""

    sim_backend: str = "physx_cuda"
    """仿真后端"""

    num_trajs: int = 25
    """提取的轨迹数 (VLAW 论文使用 25 条)"""

    target_hw: int = 192
    """目标图像分辨率 (若 demo 为 128 则 resize)"""

    auto_replay: bool = False
    """True: 若 rgb demo 不存在则自动触发 replay_trajectory"""

    num_envs: int = 64
    """replay 时的并行环境数"""

    output_dir: str = "data/vlaw/demos"
    """VLAW HDF5 输出目录"""

    task_instruction: str = ""
    """任务语言描述 (空则自动推断)"""

    frame_skip: int = 3
    """帧率下采样 (与 data_collector 一致, 降至 ~5Hz)"""

    demos_root: str = "~/.maniskill/demos"
    """ManiSkill demo 根目录"""

    gpu_id: int = 4
    """replay 时的 GPU"""

    dry_run: bool = False
    """True: 只转换前 3 条轨迹，不写入数据目录"""

    verbose: bool = True


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _find_rgb_demo_path(cfg: DemoPrepConfig) -> Optional[Path]:
    """在标准路径下查找 rgb demo H5 文件."""
    demos_root = Path(cfg.demos_root).expanduser()
    candidate = (
        demos_root / cfg.env_id / "rl"
        / f"trajectory.rgb.{cfg.control_mode}.{cfg.sim_backend}.h5"
    )
    if candidate.exists():
        return candidate
    # 尝试不含 sim_backend 后缀的旧版格式
    for p in (demos_root / cfg.env_id / "rl").glob("trajectory.rgb.*.h5"):
        if cfg.control_mode in p.name:
            return p
    return None


def _find_none_demo_path(cfg: DemoPrepConfig) -> Optional[Path]:
    """查找 none (无观测) 原始轨迹文件."""
    demos_root = Path(cfg.demos_root).expanduser()
    candidate = (
        demos_root / cfg.env_id / "rl"
        / f"trajectory.none.{cfg.control_mode}.{cfg.sim_backend}.h5"
    )
    if candidate.exists():
        return candidate
    return None


def replay_to_rgb(cfg: DemoPrepConfig, none_path: Path) -> Path:
    """调用 mani_skill.trajectory.replay_trajectory 生成 rgb demo.

    Args:
        cfg: 配置
        none_path: 原始 trajectory.none.*.h5 文件路径

    Returns:
        生成的 trajectory.rgb.*.h5 路径
    """
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu_id))

    out_h5 = none_path.parent / f"trajectory.rgb.{cfg.control_mode}.{cfg.sim_backend}.h5"
    print(f"[VLAW-P1.3] Replaying {none_path.name} → {out_h5.name} ...")

    cmd = [
        sys.executable, "-m", "mani_skill.trajectory.replay_trajectory",
        "--traj-path", str(none_path),
        "-o", "rgb",
        "-c", cfg.control_mode,
        "-b", cfg.sim_backend,
        "-n", str(cfg.num_envs),
        "--record-rewards",
        "--reward-mode", "dense",
        "--use-first-env-state",
        "--save-traj",
    ]
    import os as _os
    env_vars = dict(_os.environ)
    env_vars["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    result = subprocess.run(cmd, env=env_vars, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"replay_trajectory 失败 (返回码 {result.returncode})")

    if not out_h5.exists():
        raise FileNotFoundError(f"replay 后未找到预期输出: {out_h5}")
    print(f"[VLAW-P1.3] Replay 完成: {out_h5}")
    return out_h5


def resize_frames(
    frames: np.ndarray,  # (T, H, W, 3) uint8
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """批量 resize 帧序列."""
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames

    from PIL import Image as PILImage
    T = frames.shape[0]
    out = np.zeros((T, target_h, target_w, 3), dtype=np.uint8)
    for i in range(T):
        out[i] = np.asarray(
            PILImage.fromarray(frames[i]).resize((target_w, target_h), PILImage.BILINEAR)
        )
    return out


def extract_state(obs_grp: h5py.Group) -> np.ndarray:
    """从 HDF5 obs group 拼接 agent + extra → (T, state_dim) float32.

    ManiSkill demo obs 结构:
        obs/agent/qpos  (T, N)
        obs/agent/qvel  (T, N)
        obs/extra/*     (T, ...)
    """
    parts = []
    for v in obs_grp["agent"].values():
        arr = v[:].astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        parts.append(arr)
    if "extra" in obs_grp:
        for v in obs_grp["extra"].values():
            arr = v[:].astype(np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.dtype == bool or arr.dtype == np.bool_:
                arr = arr.astype(np.float32)
            parts.append(arr)
    return np.concatenate(parts, axis=-1)  # (T, state_dim)


# ---------------------------------------------------------------------------
# 主转换器
# ---------------------------------------------------------------------------

class DemoConverter:
    """将 ManiSkill trajectory.rgb.*.h5 转换为 VLAW HDF5 格式.

    VLAW HDF5 格式 (与 data_collector 输出一致):
        traj_i/
            rgb_base    (T, H, W, 3) uint8  — base_camera
            rgb_render  (T, H, W, 3) uint8  — 同 rgb_base (单相机 demo)
            state       (T, state_dim) float32
            obs_agent   (T, state_dim) float32
            actions     (T, action_dim) float32
            env_success (T,) bool
        traj_i.attrs:
            task_instruction  str
            source            "demo"
            success           bool

    注意事项:
        - ManiSkill demo obs 有 T 步，actions 有 T-1 步
          → 统一截为 T-1 步，与 data_collector 的最后一步一致
        - 单帧图像若为 128×128 且 target_hw=192 则 resize

    Args:
        cfg: DemoPrepConfig 配置
    """

    def __init__(self, cfg: DemoPrepConfig) -> None:
        self.cfg = cfg

    def convert(
        self,
        src_h5: Path,
        out_dir: Path,
        max_trajs: Optional[int] = None,
    ) -> Path:
        """执行转换并写出 VLAW HDF5.

        Args:
            src_h5: ManiSkill rgb demo 文件
            out_dir: 输出目录
            max_trajs: 最多处理轨迹数 (None=cfg.num_trajs)

        Returns:
            写出的 HDF5 路径
        """
        cfg = self.cfg
        tgt = cfg.target_hw
        max_n = max_trajs if max_trajs is not None else cfg.num_trajs
        task_instruction = cfg.task_instruction or cfg.env_id.replace("-v1", "")

        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        out_path = out_dir / f"{cfg.env_id}_demo_{ts}.h5"

        t0 = time.perf_counter()
        written = 0
        success_count = 0

        with h5py.File(str(src_h5), "r") as f_src, \
             h5py.File(str(out_path), "w") as f_dst:

            traj_keys = sorted(
                [k for k in f_src.keys() if k.startswith("traj_")],
                key=lambda x: int(x.split("_")[-1]),
            )[:max_n]

            for tkey in traj_keys:
                grp = f_src[tkey]
                obs = grp["obs"]

                # --- RGB ---
                sd = obs["sensor_data"]
                rgb_raw = sd["base_camera"]["rgb"][:]        # (T, H_raw, W_raw, 3)
                rgb_base = resize_frames(rgb_raw, tgt, tgt)  # (T, tgt, tgt, 3)

                # 单相机 demo → 复用 base 作为 render 视角
                rgb_render = rgb_base.copy()

                # --- State ---
                state_full = extract_state(obs)   # (T, state_dim)

                # --- Actions (T-1 vs T obs) ---
                actions = grp["actions"][:].astype(np.float32)  # (T-1, action_dim)
                T_act = actions.shape[0]

                # --- Success ---
                if "success" in grp:
                    success_arr = grp["success"][:].astype(bool)  # (T-1,)
                elif "terminated" in grp:
                    success_arr = grp["terminated"][:].astype(bool)
                else:
                    success_arr = np.zeros(T_act, dtype=bool)

                # 对齐到 T_act 步
                rgb_base = rgb_base[:T_act]
                rgb_render = rgb_render[:T_act]
                state_trunc = state_full[:T_act]

                # frame_skip 下采样
                if cfg.frame_skip > 1:
                    idx = np.arange(0, T_act, cfg.frame_skip)
                    rgb_base = rgb_base[idx]
                    rgb_render = rgb_render[idx]
                    state_trunc = state_trunc[idx]
                    actions = actions[idx]
                    success_arr = success_arr[idx]

                # --- 写 VLAW HDF5 ---
                dst_grp = f_dst.create_group(f"traj_{written:04d}")
                kw = dict(chunks=True, compression="gzip", compression_opts=1)
                dst_grp.create_dataset("rgb_base", data=rgb_base, **kw)
                dst_grp.create_dataset("rgb_render", data=rgb_render, **kw)
                dst_grp.create_dataset("state", data=state_trunc, **kw)
                dst_grp.create_dataset("obs_agent", data=state_trunc, **kw)
                dst_grp.create_dataset("actions", data=actions, **kw)
                dst_grp.create_dataset("env_success", data=success_arr, **kw)
                dst_grp.attrs["task_instruction"] = task_instruction
                dst_grp.attrs["source"] = "demo"
                dst_grp.attrs["success"] = bool(success_arr.any())

                if bool(success_arr.any()):
                    success_count += 1
                written += 1

                if cfg.verbose:
                    T_save = len(actions)
                    print(f"[VLAW-P1.3] {tkey} → traj_{written-1:04d}: "
                          f"T={T_save} rgb={rgb_base.shape[1]}px "
                          f"state={state_trunc.shape[1]}D "
                          f"act={actions.shape[1]}D "
                          f"success={'✅' if success_arr.any() else '❌'}")

            # --- meta ---
            meta = f_dst.create_group("meta")
            meta.attrs["num_trajectories"] = written
            meta.attrs["success_rate"] = success_count / max(written, 1)
            meta.attrs["env_id"] = cfg.env_id
            meta.attrs["camera_hw"] = f"{tgt},{tgt}"
            meta.attrs["source"] = "demo"
            meta.attrs["frame_skip"] = cfg.frame_skip
            meta.attrs["original_h5"] = str(src_h5)

        elapsed = time.perf_counter() - t0
        sr = success_count / max(written, 1)
        print(f"[VLAW-P1.3] 转换完成: {written} 条轨迹 → {out_path}")
        print(f"[VLAW-P1.3] 成功率={sr:.1%}, 耗时={elapsed:.1f}s")
        return out_path

    def run(self) -> Path:
        """一键准备 demo 数据: 查找/replay/转换."""
        cfg = self.cfg
        out_dir = Path(cfg.output_dir) / cfg.env_id
        max_trajs = 3 if cfg.dry_run else cfg.num_trajs

        # 1. 查找 rgb demo 文件
        rgb_path = _find_rgb_demo_path(cfg)
        if rgb_path is None:
            none_path = _find_none_demo_path(cfg)
            if none_path is None:
                raise FileNotFoundError(
                    f"未找到 {cfg.env_id} 的 demo 文件。"
                    "请先运行: python -m mani_skill.utils.download_demo {cfg.env_id}"
                )
            if not cfg.auto_replay:
                raise FileNotFoundError(
                    f"只找到 trajectory.none.*.h5，需先 replay。"
                    "请设置 --auto_replay 或手动运行 scripts/replay_demos.sh"
                )
            rgb_path = replay_to_rgb(cfg, none_path)

        print(f"[VLAW-P1.3] 使用 demo 文件: {rgb_path}")

        # 2. 转换
        out = self.convert(rgb_path, out_dir, max_trajs=max_trajs)
        if cfg.dry_run:
            print("[VLAW-P1.3] dry_run=True，输出仅供验证，实际部署请去掉 --dry_run")
        return out


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = tyro.cli(DemoPrepConfig)
    converter = DemoConverter(cfg)
    converter.run()
