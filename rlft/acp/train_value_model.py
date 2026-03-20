"""ACP value model 训练（Phase P6.C1）

训练 Pistar06 value model：distributional cross-entropy loss。
冻结 SigLIP + Gemma backbone，只训练 projector + value head。
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from rlft.acp.config import ACPTrainConfig
from rlft.acp.hdf5_dataset import ACPValueDataset, collate_acp
from rlft.acp.value_model import ManiSkillValueModel

logger = logging.getLogger(__name__)


class ACPValueTrainer:
    """ACP value model 训练器。

    Args:
        cfg: ACPTrainConfig
        device: GPU device 字符串
    """

    def __init__(self, cfg: ACPTrainConfig, device: str = "cuda:0") -> None:
        self.cfg = cfg
        self.device = device

    def train(self) -> dict[str, float]:
        """执行训练循环。

        Returns:
            dict: 最终指标（loss, value_mae, best_mae 等）
        """
        cfg = self.cfg
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- wandb ----
        run = None
        if cfg.use_wandb:
            try:
                import wandb
                run = wandb.init(
                    project="vlaw",
                    name=cfg.wandb_run_name,
                    config=self._cfg_to_dict(),
                    resume="allow",
                )
                print(f"[ACP] wandb 初始化: {run.url}")
            except Exception as e:
                print(f"[ACP] wandb 初始化失败: {e}")

        # ---- 数据 ----
        hdf5_files = self._discover_hdf5(cfg.data_dirs)
        if not hdf5_files:
            raise RuntimeError(f"未找到 HDF5 文件: {cfg.data_dirs}")
        print(f"[ACP] 发现 {len(hdf5_files)} 个 HDF5 文件")

        full_dataset = ACPValueDataset(
            hdf5_paths=hdf5_files,
            camera_keys=cfg.value_model.camera_keys,
            value_target_cfg=cfg.value_target,
        )
        if len(full_dataset) == 0:
            raise RuntimeError("数据集为空")

        # Train/val split
        n_val = max(1, int(len(full_dataset) * cfg.val_split))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        print(f"[ACP] train={n_train}, val={n_val}")

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_acp,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=min(2, cfg.num_workers),
            pin_memory=True,
            collate_fn=collate_acp,
        )

        # ---- 模型 ----
        model = ManiSkillValueModel(cfg.value_model, device=self.device)
        trainable = model.trainable_parameters()
        n_params = sum(p.numel() for p in trainable)
        print(f"[ACP] 可训练参数: {n_params:,}")

        # ---- 优化器 & scheduler ----
        optimizer = torch.optim.AdamW(
            trainable,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = self._build_lr_schedule(optimizer, cfg.warmup_steps, cfg.num_steps, cfg.learning_rate, cfg.lr_min)

        # ---- 训练循环 ----
        best_mae = float("inf")
        data_iter = iter(train_loader)
        running = {"loss": 0.0, "value_mae": 0.0, "count": 0}

        for step in range(1, cfg.num_steps + 1):
            # 无限循环
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            model.model.train()
            optimizer.zero_grad()

            # Frozen backbone 推理用 bfloat16 加速，trainable head 保持 float32
            amp_dtype = torch.bfloat16 if model.cfg.dtype == "bfloat16" else None
            ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else nullcontext()
            with ctx:
                loss, metrics = model.compute_loss(
                    images=batch["images"],
                    image_mask=batch["image_mask"],
                    value_targets=batch["value_target"],
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            running["loss"] += metrics["loss"]
            running["value_mae"] += metrics["value_mae"]
            running["count"] += 1

            # ---- 日志 ----
            if step % 50 == 0:
                n = running["count"]
                avg_loss = running["loss"] / n
                avg_mae = running["value_mae"] / n
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"[ACP] step={step}/{cfg.num_steps} "
                    f"loss={avg_loss:.4f} mae={avg_mae:.4f} lr={lr_now:.2e}"
                )
                if run is not None:
                    run.log({
                        "acp/train_loss": avg_loss,
                        "acp/train_mae": avg_mae,
                        "acp/lr": lr_now,
                        "acp/step": step,
                    })
                running = {"loss": 0.0, "value_mae": 0.0, "count": 0}

            # ---- 验证 ----
            if step % cfg.eval_interval == 0:
                val_metrics = self._validate(model, val_loader)
                print(
                    f"[ACP] [val] step={step} "
                    f"loss={val_metrics['loss']:.4f} mae={val_metrics['value_mae']:.4f}"
                )
                if run is not None:
                    run.log({
                        "acp/val_loss": val_metrics["loss"],
                        "acp/val_mae": val_metrics["value_mae"],
                        "acp/step": step,
                    })
                if val_metrics["value_mae"] < best_mae:
                    best_mae = val_metrics["value_mae"]
                    best_path = out_dir / "best.safetensors"
                    model.save(best_path)
                    print(f"[ACP] 新 best MAE={best_mae:.4f} → {best_path}")

            # ---- 定期保存 ----
            if step % cfg.save_interval == 0:
                ckpt_path = out_dir / f"step_{step}.safetensors"
                model.save(ckpt_path)

        # ---- 最终保存 ----
        final_path = out_dir / "final.safetensors"
        model.save(final_path)

        if run is not None:
            run.finish()

        result = {
            "final_loss": float(metrics["loss"]),
            "best_mae": best_mae,
            "num_steps": cfg.num_steps,
            "train_samples": n_train,
            "val_samples": n_val,
            "checkpoint_dir": str(out_dir),
        }
        print(f"[ACP] 训练完成: {result}")
        return result

    def _validate(self, model: ManiSkillValueModel, val_loader: DataLoader) -> dict[str, float]:
        """验证集评估。"""
        model.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n = 0
        amp_dtype = torch.bfloat16 if model.cfg.dtype == "bfloat16" else None
        ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else nullcontext()
        with torch.no_grad(), ctx:
            for batch in val_loader:
                _, metrics = model.compute_loss(
                    images=batch["images"],
                    image_mask=batch["image_mask"],
                    value_targets=batch["value_target"],
                )
                total_loss += metrics["loss"]
                total_mae += metrics["value_mae"]
                n += 1
        return {
            "loss": total_loss / max(n, 1),
            "value_mae": total_mae / max(n, 1),
        }

    @staticmethod
    def _build_lr_schedule(
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float = 5e-5,
        lr_min: float = 0.0,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Cosine decay with linear warmup and optional LR floor."""
        alpha = lr_min / peak_lr if peak_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(alpha, float(step) / max(warmup_steps, 1))
            progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(alpha, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _discover_hdf5(dirs: list[str]) -> list[Path]:
        """递归发现目录下所有 .h5 / .hdf5 文件。"""
        files = []
        for d in dirs:
            p = Path(d)
            if not p.exists():
                print(f"[ACP] 警告: 目录不存在: {p}")
                continue
            files.extend(sorted(p.glob("**/*.h5")))
            files.extend(sorted(p.glob("**/*.hdf5")))
        return files

    def _cfg_to_dict(self) -> dict:
        """将嵌套 dataclass config 转为 flat dict（wandb 友好）。"""
        import dataclasses
        result = {}
        for f in dataclasses.fields(self.cfg):
            v = getattr(self.cfg, f.name)
            if dataclasses.is_dataclass(v):
                for sf in dataclasses.fields(v):
                    result[f"{f.name}.{sf.name}"] = getattr(v, sf.name)
            else:
                result[f.name] = v
        return result


def main() -> None:
    """CLI 入口。"""
    import tyro

    cfg = tyro.cli(ACPTrainConfig)
    trainer = ACPValueTrainer(cfg, device="cuda:0")
    result = trainer.train()
    print(f"[ACP] 训练结果: {result}")


if __name__ == "__main__":
    main()
