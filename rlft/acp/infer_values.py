"""ACP 推理与标注（Phase P6.C2）

批量推理 Pistar06 value → 计算 N-step advantage → 写回 HDF5。
产出字段：acp_value_target, acp_value_pred, acp_advantage, acp_indicator, acp_weight。
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from rlft.acp.advantage import (
    binarize_advantages,
    compute_dense_rewards,
    compute_n_step_advantage,
    compute_task_threshold,
    normalize_advantages_to_weights,
)
from rlft.acp.config import ACPInferConfig
from rlft.acp.hdf5_dataset import ACPValueDataset, collate_acp
from rlft.acp.value_model import ManiSkillValueModel
from rlft.acp.value_targets import compute_value_targets

logger = logging.getLogger(__name__)


class ACPAnnotator:
    """ACP 推理标注器。

    1. 加载训练好的 value model
    2. 对所有帧做 value 推理
    3. 计算 N-step advantage + 二值化/归一化
    4. 写回 HDF5

    Args:
        cfg: ACPInferConfig
        device: GPU device 字符串
    """

    def __init__(self, cfg: ACPInferConfig, device: str = "cuda:0") -> None:
        self.cfg = cfg
        self.device = device

    def run(self) -> dict[str, float]:
        """执行推理标注流程。

        Returns:
            dict: 统计信息（num_frames, positive_ratio, value_mae 等）
        """
        cfg = self.cfg

        # ---- 加载模型 ----
        model = ManiSkillValueModel(cfg.value_model, device=self.device)
        ckpt_path = Path(cfg.checkpoint_path)
        if ckpt_path.exists():
            model.load(ckpt_path)
            print(f"[ACP] 模型加载: {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

        # ---- 数据 ----
        hdf5_files = self._discover_hdf5(cfg.data_dirs)
        if not hdf5_files:
            raise RuntimeError(f"未找到 HDF5 文件: {cfg.data_dirs}")

        dataset = ACPValueDataset(
            hdf5_paths=hdf5_files,
            camera_keys=cfg.value_model.camera_keys,
            value_target_cfg=cfg.value_target,
        )
        if len(dataset) == 0:
            raise RuntimeError("数据集为空")

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_acp,
        )

        # ---- 推理 ----
        print(f"[ACP] 开始推理 {len(dataset)} 帧...")
        all_preds: list[float] = []
        all_targets: list[float] = []
        all_meta: list[tuple[str, str, int]] = []  # (hdf5_path, traj_key, frame_idx)

        model.model.eval()
        with torch.no_grad():
            for batch in loader:
                pred = model.predict_values(
                    images=batch["images"],
                    image_mask=batch["image_mask"],
                )
                pred_np = pred.cpu().numpy().astype(np.float32)
                target_np = batch["value_target"].numpy().astype(np.float32)

                for i in range(len(pred_np)):
                    all_preds.append(float(pred_np[i]))
                    all_targets.append(float(target_np[i]))
                    all_meta.append((
                        batch["hdf5_paths"][i],
                        batch["traj_keys"][i],
                        batch["frame_idxs"][i],
                    ))

        preds = np.array(all_preds, dtype=np.float32)
        targets = np.array(all_targets, dtype=np.float32)
        value_mae = float(np.mean(np.abs(preds - targets)))
        print(f"[ACP] 推理完成: {len(preds)} 帧, value MAE={value_mae:.4f}")

        # ---- 按轨迹分组计算 advantage ----
        # 建立 (hdf5_path, traj_key) → [(idx, frame_idx)] 映射
        traj_groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for i, (hp, tk, fi) in enumerate(all_meta):
            key = (hp, tk)
            if key not in traj_groups:
                traj_groups[key] = []
            traj_groups[key].append((i, fi))

        # 为每帧计算 advantage
        advantages = np.zeros(len(preds), dtype=np.float32)
        indicators = np.zeros(len(preds), dtype=np.int32)
        weights = np.zeros(len(preds), dtype=np.float32)

        # 收集所有轨迹 advantage 做全局 threshold
        all_traj_advantages: list[np.ndarray] = []

        for (hp, tk), frame_list in traj_groups.items():
            frame_list.sort(key=lambda x: x[1])  # 按 frame_idx 排序
            idxs = [x[0] for x in frame_list]
            traj_targets = targets[idxs]
            traj_preds = preds[idxs]

            rewards = compute_dense_rewards(traj_targets)
            traj_adv = compute_n_step_advantage(rewards, traj_preds, cfg.advantage.n_step)
            all_traj_advantages.append(traj_adv)

            for j, global_idx in enumerate(idxs):
                advantages[global_idx] = traj_adv[j]

        # 全局 threshold（单任务场景）
        all_adv_flat = np.concatenate(all_traj_advantages) if all_traj_advantages else np.array([])
        threshold = compute_task_threshold(all_adv_flat, cfg.advantage.positive_ratio)
        indicators = binarize_advantages(advantages, threshold)

        if cfg.advantage.use_continuous_weights:
            weights = normalize_advantages_to_weights(advantages, cfg.advantage)
        else:
            weights = indicators.astype(np.float32)

        positive_ratio = float(np.mean(indicators.astype(np.float32)))
        print(
            f"[ACP] Advantage 统计: threshold={threshold:.4f}, "
            f"positive_ratio={positive_ratio:.3f}, "
            f"adv_mean={float(np.mean(advantages)):.4f}, "
            f"adv_std={float(np.std(advantages)):.4f}"
        )

        # ---- 写回 HDF5 ----
        if cfg.write_back:
            self._write_back(
                all_meta=all_meta,
                targets=targets,
                preds=preds,
                advantages=advantages,
                indicators=indicators,
                weights=weights,
                threshold=threshold,
                positive_ratio=positive_ratio,
            )

        return {
            "num_frames": len(preds),
            "value_mae": value_mae,
            "positive_ratio": positive_ratio,
            "threshold": threshold,
            "advantage_mean": float(np.mean(advantages)),
            "advantage_std": float(np.std(advantages)),
        }

    def _write_back(
        self,
        all_meta: list[tuple[str, str, int]],
        targets: np.ndarray,
        preds: np.ndarray,
        advantages: np.ndarray,
        indicators: np.ndarray,
        weights: np.ndarray,
        threshold: float,
        positive_ratio: float,
    ) -> None:
        """将标注结果写回 HDF5 文件。"""
        # 按 (hdf5_path, traj_key) 分组
        write_groups: dict[str, dict[str, dict[int, int]]] = {}  # path → traj → frame_idx → global_idx
        for i, (hp, tk, fi) in enumerate(all_meta):
            if hp not in write_groups:
                write_groups[hp] = {}
            if tk not in write_groups[hp]:
                write_groups[hp][tk] = {}
            write_groups[hp][tk][fi] = i

        for hp, trajs in write_groups.items():
            with h5py.File(hp, "a") as f:
                for tk, frames in trajs.items():
                    grp = f[tk]
                    T = len(frames)
                    frame_indices = sorted(frames.keys())

                    # 构建轨迹级数组
                    traj_targets = np.zeros(T, dtype=np.float32)
                    traj_preds = np.zeros(T, dtype=np.float32)
                    traj_adv = np.zeros(T, dtype=np.float32)
                    traj_ind = np.zeros(T, dtype=np.int32)
                    traj_w = np.zeros(T, dtype=np.float32)

                    for j, fi in enumerate(frame_indices):
                        gi = frames[fi]
                        traj_targets[j] = targets[gi]
                        traj_preds[j] = preds[gi]
                        traj_adv[j] = advantages[gi]
                        traj_ind[j] = indicators[gi]
                        traj_w[j] = weights[gi]

                    # 写入 datasets（已存在则覆盖）
                    for name, data in [
                        ("acp_value_target", traj_targets),
                        ("acp_value_pred", traj_preds),
                        ("acp_advantage", traj_adv),
                        ("acp_indicator", traj_ind),
                        ("acp_weight", traj_w),
                    ]:
                        if name in grp:
                            del grp[name]
                        grp.create_dataset(name, data=data)

                    # 写入 attrs
                    grp.attrs["acp_positive_ratio"] = float(np.mean(traj_ind.astype(np.float32)))
                    grp.attrs["acp_advantage_mean"] = float(np.mean(traj_adv))
                    grp.attrs["acp_threshold"] = threshold

            print(f"[ACP] 写回 {hp}: {len(trajs)} 条轨迹")

        print(
            f"[ACP] HDF5 标注完成: {len(write_groups)} 文件, "
            f"positive_ratio={positive_ratio:.3f}"
        )

    @staticmethod
    def _discover_hdf5(dirs: list[str]) -> list[Path]:
        files = []
        for d in dirs:
            p = Path(d)
            if not p.exists():
                continue
            files.extend(sorted(p.glob("**/*.h5")))
            files.extend(sorted(p.glob("**/*.hdf5")))
        return files


def main() -> None:
    """CLI 入口。"""
    import tyro

    cfg = tyro.cli(ACPInferConfig)
    annotator = ACPAnnotator(cfg, device="cuda:0")
    result = annotator.run()
    print(f"[ACP] 推理标注结果: {result}")


if __name__ == "__main__":
    main()
