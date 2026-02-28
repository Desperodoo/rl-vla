"""VLAW MODIFICATION: ManiSkill HDF5 数据加载器.

P2.1 阶段: 为 Ctrl-World 提供 ManiSkill 环境的训练数据.

数据格式 (来自 rlft/vlaw/data_collector.py + data_pipeline.py):
    HDF5 文件结构:
        traj_XXXX/
            latent_concat:  (T, 4, 48, 24) float16   ← VAE 编码后的 latent
            actions:        (T, 7)         float32   ← delta pose
            env_success:    (T,)           bool
        traj_XXXX.attrs["task_instruction"]: str

与 DROID dataset 的关键差异:
    - latent shape: (T, 4, 48, 24)  vs DROID (T, 4, 72, 40)
    - action: delta pose (增量)      vs DROID 绝对位姿
    - 归一化: 使用 ManiSkill stat.json
    - 数据存储: HDF5                  vs DROID JSON + .pt
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
from torch.utils.data import Dataset


class Dataset_ManiSkill(Dataset):
    """ManiSkill HDF5 轨迹数据集，兼容 Ctrl-World 训练接口.

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
        self.norm_stats: Optional[tuple] = None   # (p01, p99) shape (1, 7)

        for ds_name in dataset_names:
            ds_dir = Path(args.dataset_root_path) / ds_name
            if not ds_dir.exists():
                print(f"[WM-Dataset] ⚠️  目录不存在: {ds_dir}，跳过")
                continue
            h5_files = sorted(ds_dir.glob("**/*.h5")) + sorted(ds_dir.glob("**/*.hdf5"))
            print(f"[WM-Dataset] {ds_name}: 找到 {len(h5_files)} 个 HDF5 文件")
            for h5_path in h5_files:
                self._index_hdf5(h5_path)

        # ---- 归一化统计量 ----
        stat_path = getattr(args, "data_stat_path", None)
        if stat_path and Path(stat_path).exists():
            with open(stat_path, "r") as f:
                stat = json.load(f)
            p01 = np.array(stat["state_01"], dtype=np.float32)[None, :]  # (1, 7)
            p99 = np.array(stat["state_99"], dtype=np.float32)[None, :]
            self.norm_stats = (p01, p99)
            print(f"[WM-Dataset] 加载归一化统计量: {stat_path}")
        else:
            # VLAW MODIFICATION: train 模式下 stat.json 必须存在，否则直接报错；val 模式下允许跳过
            if mode == "train":
                raise FileNotFoundError(
                    f"[WM-Dataset] stat.json 不存在: {stat_path}\n"
                    f"请先运行: python ctrl_world/dataset/dataset_maniskill.py "
                    f"或 scripts/create_maniskill_meta_info.py 生成归一化统计量"
                )
            print(f"[WM-Dataset] ⚠️  未找到 stat.json ({stat_path})，动作不归一化（仅 val 模式允许）")

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

            # ---- action (T, 7) delta pose ----
            if "actions" in grp:
                action_raw = grp["actions"][frame_ids].astype(np.float32)
            else:
                action_raw = np.zeros((window_len, self.args.action_dim), dtype=np.float32)

            # ---- instruction text ----
            text = grp.attrs.get("task_instruction", "")

        # ---- 动作归一化 [-1, 1] ----
        action = self._normalize_action(action_raw)

        return {
            "latent": latent,                           # (T, 4, 48, 24)
            "action": torch.tensor(action, dtype=torch.float32),  # (T, 7)
            "text": text,
        }

    # ------------------------------------------------------------------
    # 归一化工具
    # ------------------------------------------------------------------

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """将动作归一化到 [-1, 1].

        若无统计量则原样返回 (训练阶段 stat.json 必须存在).
        """
        if self.norm_stats is None:
            return action
        p01, p99 = self.norm_stats
        eps = 1e-8
        ndata = 2.0 * (action - p01) / (p99 - p01 + eps) - 1.0
        return np.clip(ndata, -1.0, 1.0)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """反归一化，用于推理时恢复真实动作值."""
        if self.norm_stats is None:
            return action
        p01, p99 = self.norm_stats
        return (action + 1.0) / 2.0 * (p99 - p01) + p01


# ---------------------------------------------------------------------------
# 元信息创建工具
# ---------------------------------------------------------------------------

def create_meta_info(
    data_dir: str,
    output_dir: str,
    dataset_name: str = "maniskill",
) -> None:
    """从 HDF5 数据目录计算并保存 stat.json.

    stat.json 格式与 DROID 一致:
        {"state_01": [...], "state_99": [...]}  (p1 / p99 分位数)

    Args:
        data_dir: 包含 HDF5 文件的目录
        output_dir: 保存 stat.json 的目录
        dataset_name: 子目录名
    """
    all_actions: list[np.ndarray] = []

    h5_files = sorted(Path(data_dir).glob("**/*.h5")) + sorted(Path(data_dir).glob("**/*.hdf5"))
    print(f"[MetaInfo] 扫描 {len(h5_files)} 个 HDF5 文件...")

    for h5_path in h5_files:
        with h5py.File(str(h5_path), "r") as f:
            for key in sorted(f.keys()):
                if key.startswith("traj_") and "actions" in f[key]:
                    all_actions.append(f[key]["actions"][:].astype(np.float32))

    if not all_actions:
        raise RuntimeError(f"[MetaInfo] 未在 {data_dir} 中找到 actions 数据")

    actions = np.concatenate(all_actions, axis=0)   # (N_total_frames, 7)
    p01 = np.percentile(actions, 1, axis=0).tolist()
    p99 = np.percentile(actions, 99, axis=0).tolist()

    out_dir = Path(output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stat_path = out_dir / "stat.json"
    with open(stat_path, "w") as f:
        json.dump({"state_01": p01, "state_99": p99}, f, indent=2)

    print(f"[MetaInfo] 保存 stat.json → {stat_path}")
    print(f"[MetaInfo] p01: {[f'{x:.4f}' for x in p01]}")
    print(f"[MetaInfo] p99: {[f'{x:.4f}' for x in p99]}")
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
