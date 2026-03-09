"""VLAW MODIFICATION: ManiSkill HDF5 数据加载器.

P2.1 阶段: 为 Ctrl-World 提供 ManiSkill 环境的训练数据.

数据格式 (来自 rlft/vlaw/data_collector.py + data_pipeline.py):
    HDF5 文件结构:
        traj_XXXX/
            latent_concat:  (T, 4, 48, 24) float16   ← VAE 编码后的 latent
            actions:        (T, 7)         float32   ← delta pose (不再用于 WM conditioning)
            state:          (T, 25)        float32   ← 完整 agent state
            env_success:    (T,)           bool
        traj_XXXX.attrs["task_instruction"]: str

    state 布局 (25-D, LiftPegUpright-v1):
        [0:9]   = qpos   (7 arm joints + 2 gripper fingers)
        [9:18]  = qvel   (velocities)
        [18:25] = tcp_pose (x, y, z, qw, qx, qy, qz) — 绝对 EE 位姿

WM Action Conditioning (对齐 DROID):
    Ctrl-World 预训练使用 **绝对 EE 位姿** 做 action conditioning,
    而非 delta pose. 因此 WM 训练/推理时 "action" 字段的语义为:
        [tcp_x, tcp_y, tcp_z, euler_rx, euler_ry, euler_rz, gripper_norm]
    - tcp_xyz 来自 state[18:21]
    - euler_xyz 由 state[21:25] 的四元数转换得到
    - gripper_norm = qpos[7] / 0.04 ∈ [0, 1]

与 DROID dataset 的关键差异:
    - latent shape: (T, 4, 48, 24)  vs DROID (T, 4, 72, 40)
    - 相机数: 2 vs 3
    - 归一化: 使用 ManiSkill stat.json (EE 位姿百分位)
    - 数据存储: HDF5 vs DROID JSON + .pt
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import Dataset

# Panda gripper max finger opening (one finger, in metres).
PANDA_FINGER_MAX = 0.04


# Panda gripper max finger opening (one finger, in metres).
PANDA_FINGER_MAX = 0.04


def state_to_ee_pose_7d(state: np.ndarray) -> np.ndarray:
    """Convert ManiSkill 25-D state → 7-D EE conditioning vector.

    This matches the DROID convention where Ctrl-World is conditioned on
    per-frame absolute end-effector state, not delta actions.

    Args:
        state: (N, 25) or (25,) — raw state from HDF5.

    Returns:
        (N, 7) or (7,) float32 — [tcp_x, tcp_y, tcp_z,
                                    euler_rx, euler_ry, euler_rz,
                                    gripper_norm].
    """
    squeeze = state.ndim == 1
    if squeeze:
        state = state[None, :]

    N, D = state.shape
    # Need at least 25 dims; if shorter (e.g. test/mock), pad with zeros
    if D < 25:
        pad = np.zeros((N, 25 - D), dtype=state.dtype)
        state = np.concatenate([state, pad], axis=1)

    tcp_pos = state[:, 18:21].astype(np.float64)       # (N, 3) xyz
    tcp_quat_wxyz = state[:, 21:25].astype(np.float64)  # (N, 4) qw,qx,qy,qz
    # scipy expects xyzw ordering
    tcp_quat_xyzw = tcp_quat_wxyz[:, [1, 2, 3, 0]]
    # Handle zero-norm quaternions (e.g. from mock/test states):
    # replace with identity quaternion [0, 0, 0, 1]
    norms = np.linalg.norm(tcp_quat_xyzw, axis=1, keepdims=True)
    zero_mask = norms.squeeze() < 1e-8
    if np.any(zero_mask):
        tcp_quat_xyzw[zero_mask] = [0.0, 0.0, 0.0, 1.0]
    euler = Rot.from_quat(tcp_quat_xyzw).as_euler("xyz")  # (N, 3)
    gripper_norm = (state[:, 7] / PANDA_FINGER_MAX).clip(0.0, 1.0)  # (N,)
    result = np.column_stack([tcp_pos, euler, gripper_norm[:, None]]).astype(np.float32)

    if squeeze:
        return result[0]
    return result


class Dataset_ManiSkill(Dataset):
    """ManiSkill HDF5 轨迹数据集，兼容 Ctrl-World 训练接口.

    WM action conditioning 使用 **绝对 EE 位姿** (对齐 DROID),
    从 HDF5 的 ``state`` 字段在线计算 7-D EE pose.

    Args:
        args: wm_args_maniskill 实例 (或任何有相应属性的对象)
        mode: 'train' 或 'val'
        val_split: 验证集比例 (0~1), 按轨迹切分
    """

    def __init__(
        self,
        args,
        mode: str = "train",
        val_split: float = 0.1,
    ) -> None:
        super().__init__()
        self.args = args
        self.mode = mode
        self.val_split = val_split

        # ---- 找到所有 HDF5 文件 ----
        dataset_names = args.dataset_names.split("+")
        self.samples: list[dict] = []   # {'h5_path': str, 'traj_key': str, 'start_frame': int}
        self.norm_stats: Optional[tuple] = None   # (p01, p99) shape (1, 7) — EE pose percentiles

        for ds_name in dataset_names:
            ds_dir = Path(args.dataset_root_path) / ds_name
            if not ds_dir.exists():
                print(f"[WM-Dataset] ⚠️  目录不存在: {ds_dir}，跳过")
                continue
            h5_files = sorted(ds_dir.glob("**/*.h5")) + sorted(ds_dir.glob("**/*.hdf5"))
            print(f"[WM-Dataset] {ds_name}: 找到 {len(h5_files)} 个 HDF5 文件")
            for h5_path in h5_files:
                self._index_hdf5(h5_path)

        # ---- 归一化统计量 (EE pose percentiles) ----
        stat_path = getattr(args, "data_stat_path", None)
        if stat_path and Path(stat_path).exists():
            with open(stat_path, "r") as f:
                stat = json.load(f)
            p01 = np.array(stat["state_01"], dtype=np.float32)[None, :]  # (1, 7)
            p99 = np.array(stat["state_99"], dtype=np.float32)[None, :]
            self.norm_stats = (p01, p99)
            print(f"[WM-Dataset] 加载 EE pose 归一化统计量: {stat_path}")
        else:
            if mode == "train":
                raise FileNotFoundError(
                    f"[WM-Dataset] stat.json 不存在: {stat_path}\n"
                    f"请先运行: python scripts/vlaw/generate_stat_json.py 生成 EE pose 归一化统计量"
                )
            print(f"[WM-Dataset] ⚠️  未找到 stat.json ({stat_path})，EE pose 不归一化（仅 val 模式允许）")

        # ---- 训练/验证集切分 ----
        n = len(self.samples)
        n_val = max(1, int(n * val_split))
        if mode == "train":
            self.samples = self.samples[n_val:]
        else:
            self.samples = self.samples[:n_val]

        print(f"[WM-Dataset] mode={mode}, 样本数={len(self.samples)}")

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------

    def _index_hdf5(self, h5_path: Path) -> None:
        """从单个 HDF5 文件中索引所有可用窗口."""
        window_len = self.args.num_history + self.args.num_frames
        try:
            with h5py.File(str(h5_path), "r") as f:
                traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
                for tkey in traj_keys:
                    grp = f[tkey]
                    if "latent_concat" not in grp:
                        # 还未经过 VAE 编码 (data_pipeline.py 未跑) → 跳过
                        continue
                    T = grp["latent_concat"].shape[0]
                    if T < window_len:
                        continue
                    # 每条轨迹按 skip_step 生成多个滑动窗口
                    skip = getattr(self.args, "skip_step", 1)
                    for start in range(0, T - window_len + 1, max(1, skip)):
                        self.samples.append({
                            "h5_path": str(h5_path),
                            "traj_key": tkey,
                            "start_frame": start,
                        })
        except Exception as e:
            print(f"[WM-Dataset] ⚠️  读取 HDF5 失败 {h5_path}: {e}")

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        window_len = self.args.num_history + self.args.num_frames
        start = sample["start_frame"]
        end = start + window_len
        frame_ids = list(range(start, end))

        with h5py.File(sample["h5_path"], "r") as f:
            grp = f[sample["traj_key"]]

            # ---- latent (T, 4, lat_H, lat_W) float16 → float32 ----
            latent_raw = grp["latent_concat"][frame_ids]   # (T, 4, 48, 24)
            latent = torch.from_numpy(latent_raw.astype(np.float32))

            # ---- EE pose from state: absolute TCP pose + gripper ----
            state_key = "state" if "state" in grp else "obs_agent"
            if state_key in grp:
                state_raw = grp[state_key][frame_ids].astype(np.float32)  # (T, 25)
                ee_pose_raw = state_to_ee_pose_7d(state_raw)  # (T, 7)
            else:
                # Fallback: zeros (should not happen with properly collected data)
                ee_pose_raw = np.zeros((window_len, 7), dtype=np.float32)

            # ---- instruction text ----
            text = grp.attrs.get("task_instruction", "")

        # ---- EE pose 归一化 [-1, 1] ----
        ee_pose = self._normalize_action(ee_pose_raw)

        return {
            "latent": latent,                                    # (T, 4, 48, 24)
            "action": torch.tensor(ee_pose, dtype=torch.float32),  # (T, 7) normalized EE pose
            "text": text,
        }

    # ------------------------------------------------------------------
    # 归一化工具
    # ------------------------------------------------------------------

    def _normalize_action(self, ee_pose: np.ndarray) -> np.ndarray:
        """将 EE pose 归一化到 [-1, 1].

        若无统计量则原样返回 (训练阶段 stat.json 必须存在).
        """
        if self.norm_stats is None:
            return ee_pose
        p01, p99 = self.norm_stats
        eps = 1e-8
        ndata = 2.0 * (ee_pose - p01) / (p99 - p01 + eps) - 1.0
        return np.clip(ndata, -1.0, 1.0)

    def denormalize_action(self, ee_pose: np.ndarray) -> np.ndarray:
        """反归一化，恢复真实 EE pose 值."""
        if self.norm_stats is None:
            return ee_pose
        p01, p99 = self.norm_stats
        return (ee_pose + 1.0) / 2.0 * (p99 - p01) + p01


# ---------------------------------------------------------------------------
# 元信息创建工具
# ---------------------------------------------------------------------------

def create_meta_info(
    data_dir: str,
    output_dir: str,
    dataset_name: str = "maniskill",
) -> None:
    """从 HDF5 数据目录计算并保存 stat.json (EE pose percentiles).

    stat.json 格式与 DROID 一致:
        {"state_01": [...], "state_99": [...]}
    但计算基底为绝对 EE 位姿 7D, 而非 delta action.

    Args:
        data_dir: 包含 HDF5 文件的目录
        output_dir: 保存 stat.json 的目录
        dataset_name: 子目录名
    """
    all_ee_poses: list[np.ndarray] = []

    h5_files = sorted(Path(data_dir).glob("**/*.h5")) + sorted(Path(data_dir).glob("**/*.hdf5"))
    print(f"[MetaInfo] 扫描 {len(h5_files)} 个 HDF5 文件...")

    for h5_path in h5_files:
        with h5py.File(str(h5_path), "r") as f:
            for key in sorted(f.keys()):
                if not key.startswith("traj_"):
                    continue
                grp = f[key]
                state_key = "state" if "state" in grp else "obs_agent"
                if state_key in grp:
                    st = grp[state_key][:].astype(np.float32)
                    all_ee_poses.append(state_to_ee_pose_7d(st))

    if not all_ee_poses:
        raise RuntimeError(f"[MetaInfo] 未在 {data_dir} 中找到 state 数据")

    ee_poses = np.concatenate(all_ee_poses, axis=0)  # (N_total_frames, 7)
    p01 = np.percentile(ee_poses, 1, axis=0).tolist()
    p99 = np.percentile(ee_poses, 99, axis=0).tolist()

    out_dir = Path(output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stat_path = out_dir / "stat.json"
    with open(stat_path, "w") as f:
        json.dump({"state_01": p01, "state_99": p99}, f, indent=2)

    print(f"[MetaInfo] 保存 stat.json → {stat_path}")
    print(f"[MetaInfo] p01 (EE pose): {[f'{x:.4f}' for x in p01]}")
    print(f"[MetaInfo] p99 (EE pose): {[f'{x:.4f}' for x in p99]}")
    return str(stat_path)


# ---------------------------------------------------------------------------
# 快速验证
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import wm_args_maniskill

    args = wm_args_maniskill()
    print("=== 测试 Dataset_ManiSkill ===")
    print(f"dataset_root_path: {args.dataset_root_path}")
    print(f"dataset_names: {args.dataset_names}")
    print(f"num_history: {args.num_history}, num_frames: {args.num_frames}")

    ds = Dataset_ManiSkill(args, mode="train")
    if len(ds) == 0:
        print("⚠️  数据集为空，需先运行 data_collector + data_pipeline 生成数据")
    else:
        item = ds[0]
        print(f"latent: {item['latent'].shape} {item['latent'].dtype}")
        print(f"action: {item['action'].shape} {item['action'].dtype}")
        print(f"text:   {item['text']!r}")
        print("✅ Dataset_ManiSkill OK")
