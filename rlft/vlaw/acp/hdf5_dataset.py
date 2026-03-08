"""ACP HDF5 Dataset（Phase P6.B2）

从 VLAW HDF5 轨迹数据构建 PyTorch Dataset，供 value model 训练和推理。
每个样本 = 单帧（multi-camera images + value target）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from rlft.vlaw.acp.config import ValueModelConfig, ValueTargetConfig
from rlft.vlaw.acp.value_targets import compute_value_targets


class ACPValueDataset(Dataset):
    """ACP value model 训练/推理用 Dataset。

    从 HDF5 文件读取轨迹，每帧生成一个样本：
        - images: (N_cam, C, H, W) uint8 → 由 model 内部归一化
        - image_mask: (N_cam,) bool — 有效相机标记
        - value_target: float32 — GT value target（训练模式）
        - traj_key: str, frame_idx: int — 定位信息（推理标注用）

    Args:
        hdf5_paths: HDF5 文件路径列表
        camera_keys: 相机数据的 key 列表
        value_target_cfg: value target 配置（None 则跳过 target 计算）
        target_field: 如果 HDF5 中已有预计算 target，从此 key 读取
    """

    def __init__(
        self,
        hdf5_paths: list[str | Path],
        camera_keys: list[str] | None = None,
        value_target_cfg: ValueTargetConfig | None = None,
        target_field: str | None = None,
    ) -> None:
        self.hdf5_paths = [Path(p) for p in hdf5_paths]
        self.camera_keys = camera_keys or ["rgb_base", "rgb_render"]
        self.value_target_cfg = value_target_cfg
        self.target_field = target_field

        # 样本索引：(hdf5_path_idx, traj_key, frame_idx, value_target)
        self._samples: list[tuple[int, str, int, float]] = []
        self._scan_all()

    def _scan_all(self) -> None:
        """扫描所有 HDF5 文件，建立帧级索引。"""
        global_max_len = 0
        success_key = self.value_target_cfg.success_key if self.value_target_cfg else "env_success"

        # 第一遍：找全局 max episode length
        traj_meta: list[tuple[int, str, int, bool]] = []  # (path_idx, traj_key, length, success)
        for path_idx, path in enumerate(self.hdf5_paths):
            if not path.exists():
                print(f"[ACP] 警告: HDF5 不存在: {path}")
                continue
            with h5py.File(str(path), "r") as f:
                for key in sorted(f.keys()):
                    if not key.startswith("traj_"):
                        continue
                    grp = f[key]
                    # 至少需要一个相机
                    if not any(ck in grp for ck in self.camera_keys):
                        continue

                    # 确定轨迹长度和 success 信号
                    T, success = self._read_traj_meta(grp, success_key)
                    if T is None:
                        continue

                    traj_meta.append((path_idx, key, T, success))
                    global_max_len = max(global_max_len, T)

        if global_max_len == 0:
            print("[ACP] 警告: 未找到有效轨迹")
            return

        # 第二遍：计算 value target 并建立帧索引
        for path_idx, traj_key, length, success in traj_meta:
            if self.value_target_cfg is not None:
                with h5py.File(str(self.hdf5_paths[path_idx]), "r") as f:
                    env_success = self._read_success_array(
                        f[traj_key], success_key, length, success
                    )
                targets = compute_value_targets(
                    env_success=env_success,
                    episode_length=length,
                    max_episode_length=global_max_len,
                    cfg=self.value_target_cfg,
                )
            else:
                targets = np.full(length, float("nan"), dtype=np.float32)

            for t in range(length):
                self._samples.append((path_idx, traj_key, t, float(targets[t])))

        print(
            f"[ACP] ACPValueDataset: {len(self._samples)} 帧, "
            f"{len(traj_meta)} 轨迹, {len(self.hdf5_paths)} 文件, "
            f"max_len={global_max_len}"
        )

    @staticmethod
    def _read_traj_meta(
        grp: h5py.Group, success_key: str
    ) -> tuple[int | None, bool]:
        """从 HDF5 group 读取轨迹长度和 success 信号。

        Returns:
            (T, success) — 如果无法确定长度返回 (None, False)
        """
        if success_key == "env_success":
            # env_success: per-frame (T,) bool dataset
            if "env_success" not in grp:
                return None, False
            T = grp["env_success"].shape[0]
            success = bool(np.asarray(grp["env_success"]).any())
            return T, success
        else:
            # VLM 等外部标签：scalar attribute，需要从其他 dataset 推断长度
            # 尝试从任意已知 dataset 读取长度
            T = None
            for key in ("actions", "rgb_base", "rgb_render"):
                if key in grp:
                    T = grp[key].shape[0]
                    break
            if T is None:
                return None, False
            # 读取 scalar attribute
            if success_key in grp.attrs:
                success = bool(int(grp.attrs[success_key]))
            elif success_key in grp:
                # 也可能是 dataset（兼容未来 per-frame VLM）
                success = bool(np.asarray(grp[success_key]).any())
            else:
                return None, False
            return T, success

    @staticmethod
    def _read_success_array(
        grp: h5py.Group, success_key: str, length: int, success: bool
    ) -> np.ndarray:
        """读取 per-frame success array。

        对于 env_success: 直接返回 per-frame (T,) bool。
        对于 VLM scalar labels: 展开为 (T,) bool，成功时最后一帧为 True。
        """
        if success_key == "env_success" and "env_success" in grp:
            return np.asarray(grp["env_success"], dtype=bool)

        # scalar label → expand to per-frame array
        arr = np.zeros(length, dtype=bool)
        if success:
            arr[-1] = True  # 成功轨迹：最后一帧标记为 True
        return arr

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        """返回单帧样本。

        Returns:
            dict:
                "images": (N_cam, 3, H, W) uint8
                "image_mask": (N_cam,) bool
                "value_target": float32
                "traj_key": str
                "frame_idx": int
                "hdf5_path": str
        """
        path_idx, traj_key, frame_idx, value_target = self._samples[idx]
        path = self.hdf5_paths[path_idx]

        n_cam = len(self.camera_keys)
        images_list = []
        mask_list = []

        with h5py.File(str(path), "r") as f:
            grp = f[traj_key]

            # 如果有预计算 target 字段，优先使用
            if self.target_field is not None and self.target_field in grp:
                value_target = float(grp[self.target_field][frame_idx])

            for ck in self.camera_keys:
                if ck in grp:
                    img = np.asarray(grp[ck][frame_idx], dtype=np.uint8)  # (H, W, C)
                    img = np.transpose(img, (2, 0, 1))  # (C, H, W)
                    images_list.append(img)
                    mask_list.append(True)
                else:
                    # 缺失相机用零填充
                    if images_list:
                        h, w = images_list[0].shape[1], images_list[0].shape[2]
                    else:
                        h, w = 128, 128
                    images_list.append(np.zeros((3, h, w), dtype=np.uint8))
                    mask_list.append(False)

        images = np.stack(images_list, axis=0)  # (N_cam, C, H, W)
        image_mask = np.array(mask_list, dtype=bool)

        return {
            "images": torch.from_numpy(images),              # (N_cam, 3, H, W) uint8
            "image_mask": torch.from_numpy(image_mask),       # (N_cam,) bool
            "value_target": torch.tensor(value_target, dtype=torch.float32),
            "traj_key": traj_key,
            "frame_idx": frame_idx,
            "hdf5_path": str(path),
        }


def collate_acp(batch: list[dict]) -> dict:
    """ACP 专用 collate function。

    处理变长字符串字段（traj_key, hdf5_path）。
    """
    images = torch.stack([b["images"] for b in batch])
    image_mask = torch.stack([b["image_mask"] for b in batch])
    value_target = torch.stack([b["value_target"] for b in batch])
    traj_keys = [b["traj_key"] for b in batch]
    frame_idxs = [b["frame_idx"] for b in batch]
    hdf5_paths = [b["hdf5_path"] for b in batch]

    return {
        "images": images,                # (B, N_cam, C, H, W)
        "image_mask": image_mask,         # (B, N_cam)
        "value_target": value_target,     # (B,)
        "traj_keys": traj_keys,
        "frame_idxs": frame_idxs,
        "hdf5_paths": hdf5_paths,
    }
