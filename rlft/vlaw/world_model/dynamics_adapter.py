"""Dynamics Adapter — 学习 ManiSkill PD 控制器前向动力学.

参考 DROID 的 Dynamics MLP (ctrl_world/models/action_adapter/train2.py)，
将 (current_state, delta_action_chunk) 映射到 future EE poses (world frame 7D)。

DROID pipeline:  joint_pos + joint_vel → cumulative joint delta → FK → EE cartesian
ManiSkill 版:    state_25d + delta_actions → future_ee_pose_7d  (直接输出 Cartesian)

用途：在 Imagination 中替代 env.step()，从 policy delta actions 预测
WM 所需的 absolute EE pose conditioning，消除 BUG-D (tiled EE pose)。

所属阶段: Dynamics Adapter Track (ADR-047)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import tyro

# state_to_ee_pose_7d: 从 25D state 提取 7D EE pose (world frame)
from ctrl_world.dataset.dataset_maniskill import state_to_ee_pose_7d


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class DynamicsAdapterConfig:
    """Dynamics Adapter 训练/推理配置."""

    state_dim: int = 25
    """ManiSkill agent_state 维度 (qpos 9 + qvel 9 + tcp_pose 7 = 25)."""

    action_dim: int = 7
    """pd_ee_delta_pose 动作维度."""

    act_steps: int = 5
    """每个 chunk 的动作步数 (对齐 WM act_steps)."""

    hidden_dim: int = 512
    """MLP 隐藏层维度 (对齐 DROID Dynamics MLP)."""

    lr: float = 1e-4
    """Adam 学习率."""

    epochs: int = 20
    """训练轮数."""

    batch_size: int = 128
    """训练批大小."""

    val_ratio: float = 0.1
    """验证集比例."""

    checkpoint_dir: str = "checkpoints/vlaw/dynamics_adapter"
    """Checkpoint 保存目录."""

    gpu_id: int = 8
    """训练 GPU id."""

    mode: str = "test"
    """运行模式: train | eval | test."""

    hdf5_dir: str = "data/vlaw/rollouts/mixed/LiftPegUpright-v1"
    """训练数据 HDF5 目录."""


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


class DynamicsAdapter(nn.Module):
    """Chunk-level dynamics MLP: (state, action_chunk) → future EE poses.

    Output format: [tcp_x, tcp_y, tcp_z, sin_rx, cos_rx, sin_ry, cos_ry,
                     sin_rz, cos_rz, gripper_norm]  (10D per step)
    在 predict() 时自动转回 7D euler 格式供 WM 使用。

    Args:
        state_dim:  ManiSkill state 维度 (25).
        action_dim: pd_ee_delta_pose 维度 (7).
        act_steps:  chunk 内动作步数 (5).
        hidden_dim: 隐藏层宽度 (512).
    """

    EE_DIM: int = 7   # output 7D (after atan2 conversion)
    RAW_DIM: int = 10  # internal 10D: xyz(3) + sin/cos×3(6) + gripper(1)

    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 7,
        act_steps: int = 5,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.act_steps = act_steps

        input_dim = state_dim + action_dim * act_steps
        output_dim = self.RAW_DIM * act_steps

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """前向预测 (raw 10D output for training loss).

        Args:
            state:        (B, state_dim)  当前 agent_state.
            action_chunk: (B, act_steps, action_dim)  delta action chunk.

        Returns:
            (B, act_steps, 10)  raw output: [xyz, sin_rx, cos_rx, ..., grip].
        """
        B = state.shape[0]
        flat_actions = action_chunk.reshape(B, -1)
        x = torch.cat([state, flat_actions], dim=-1)
        out = self.net(x)
        return out.reshape(B, self.act_steps, self.RAW_DIM)

    def forward_7d(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """前向预测, 输出 7D EE pose (xyz + euler + gripper).

        将 sin/cos 编码通过 atan2 转回 euler angles.
        """
        raw = self.forward(state, action_chunk)  # (B, K, 10)
        return self._raw_to_7d(raw)

    @staticmethod
    def _raw_to_7d(raw: torch.Tensor) -> torch.Tensor:
        """Convert (B, K, 10) raw → (B, K, 7) euler format."""
        pos = raw[..., :3]
        sin_rx, cos_rx = raw[..., 3], raw[..., 4]
        sin_ry, cos_ry = raw[..., 5], raw[..., 6]
        sin_rz, cos_rz = raw[..., 7], raw[..., 8]
        grip = raw[..., 9:]
        euler_rx = torch.atan2(sin_rx, cos_rx)
        euler_ry = torch.atan2(sin_ry, cos_ry)
        euler_rz = torch.atan2(sin_rz, cos_rz)
        return torch.cat(
            [pos, euler_rx.unsqueeze(-1), euler_ry.unsqueeze(-1),
             euler_rz.unsqueeze(-1), grip], dim=-1
        )

    @torch.no_grad()
    def predict(
        self,
        state: np.ndarray,
        action_chunk: np.ndarray,
    ) -> np.ndarray:
        """Numpy inference — 用于 Imagination pipeline 集成.

        Args:
            state:        (state_dim,) or (B, state_dim)  当前 state.
            action_chunk: (act_steps, action_dim) or (B, act_steps, action_dim).

        Returns:
            (act_steps, 7) or (B, act_steps, 7)  np.float32 world frame EE.
        """
        squeeze = state.ndim == 1
        if squeeze:
            state = state[None, :]
            action_chunk = action_chunk[None, :]

        device = next(self.parameters()).device
        s = torch.from_numpy(state).float().to(device)
        a = torch.from_numpy(action_chunk).float().to(device)
        pred = self.forward_7d(s, a)  # (B, K, 7) — euler format
        result = pred.cpu().numpy()

        return result[0] if squeeze else result


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------


class DynamicsAdapterTrainer:
    """数据加载 + 训练 + 评估."""

    def __init__(self, config: DynamicsAdapterConfig) -> None:
        self.cfg = config
        self.device = torch.device(
            f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.model = DynamicsAdapter(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            act_steps=config.act_steps,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"[DynAdapter] 初始化完成: {n_params:,} params, device={self.device}"
        )

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    @staticmethod
    def _targets_7d_to_10d(targets: np.ndarray) -> np.ndarray:
        """Convert (N, K, 7) euler targets → (N, K, 10) sin/cos format.

        7D: [x, y, z, euler_rx, euler_ry, euler_rz, gripper]
        10D: [x, y, z, sin_rx, cos_rx, sin_ry, cos_ry, sin_rz, cos_rz, gripper]
        """
        pos = targets[..., :3]         # (N, K, 3)
        euler = targets[..., 3:6]      # (N, K, 3) — rx, ry, rz
        grip = targets[..., 6:7]       # (N, K, 1)
        sin_euler = np.sin(euler)      # (N, K, 3)
        cos_euler = np.cos(euler)      # (N, K, 3)
        # interleave: sin_rx, cos_rx, sin_ry, cos_ry, sin_rz, cos_rz
        sc = np.stack([sin_euler, cos_euler], axis=-1)  # (N, K, 3, 2)
        sc = sc.reshape(*euler.shape[:-1], 6)           # (N, K, 6)
        return np.concatenate([pos, sc, grip], axis=-1)  # (N, K, 10)

    # ------------------------------------------------------------------

    def _load_chunks(
        self, hdf5_dir: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从 HDF5 提取 (state_t, action_chunk[t:t+K], target_ee[t+1:t+K+1]).

        Returns:
            states:  (N, state_dim)
            actions: (N, act_steps, action_dim)
            targets: (N, act_steps, 7)
        """
        K = self.cfg.act_steps
        h5_files = sorted(Path(hdf5_dir).glob("**/*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"[DynAdapter] 未找到 HDF5: {hdf5_dir}")

        all_states: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        total_traj = 0

        for h5_file in h5_files:
            with h5py.File(str(h5_file), "r") as f:
                traj_keys = [k for k in f.keys() if k.startswith("traj_")]
                for key in traj_keys:
                    grp = f[key]
                    if "state" not in grp or "actions" not in grp:
                        continue
                    state_arr = grp["state"][:].astype(np.float32)  # (T, 25)
                    act_arr = grp["actions"][:].astype(np.float32)  # (T, 7)
                    T = min(state_arr.shape[0], act_arr.shape[0])
                    if T <= K:
                        continue

                    # 提取 chunk-level 训练对
                    for t in range(T - K):
                        all_states.append(state_arr[t])
                        all_actions.append(act_arr[t : t + K])
                        # target: 未来 K 步各帧的 EE pose (world frame)
                        future_states = state_arr[t + 1 : t + K + 1]
                        all_targets.append(state_to_ee_pose_7d(future_states))
                    total_traj += 1

        states = np.stack(all_states, axis=0)
        actions = np.stack(all_actions, axis=0)
        targets = np.stack(all_targets, axis=0)

        print(
            f"[DynAdapter] 数据加载: {total_traj} 条轨迹, "
            f"{states.shape[0]} 个训练对 (K={K})"
        )
        return states, actions, targets

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(self, hdf5_dir: str) -> dict:
        """完整训练流程.

        Returns:
            {"best_loss": float, "checkpoint_path": str, "eval_metrics": dict}
        """
        print(f"[DynAdapter] 开始训练, 数据: {hdf5_dir}")

        # ---- 加载数据 ----
        states, actions, targets = self._load_chunks(hdf5_dir)
        N = states.shape[0]

        # ---- 计算归一化统计量 (state input) ----
        state_mean = states.mean(axis=0)
        state_std = states.std(axis=0).clip(min=1e-6)

        # ---- Train/val split ----
        n_val = max(1, int(N * self.cfg.val_ratio))
        perm = np.random.RandomState(42).permutation(N)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).float().to(self.device)

        # 归一化 state
        states_n = (states - state_mean) / state_std

        # 将 7D targets 转为 10D sin/cos 格式 (训练用)
        targets_10d = self._targets_7d_to_10d(targets)

        train_s = _to_tensor(states_n[train_idx])
        train_a = _to_tensor(actions[train_idx])
        train_t = _to_tensor(targets_10d[train_idx])
        val_s = _to_tensor(states_n[val_idx])
        val_a = _to_tensor(actions[val_idx])
        val_t = _to_tensor(targets_10d[val_idx])
        # 保留 7D targets 用于验证指标
        val_t_7d = _to_tensor(targets[val_idx])

        print(
            f"[DynAdapter] Train: {len(train_idx)}, Val: {len(val_idx)}"
        )

        # ---- 优化器 ----
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        n_train = len(train_idx)
        steps_per_epoch = max(1, n_train // self.cfg.batch_size)
        total_steps = self.cfg.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        # ---- 训练循环 ----
        self.model.train()
        best_val_loss = float("inf")
        best_epoch = 0
        t0 = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                idx = torch.randint(0, n_train, (self.cfg.batch_size,))
                pred = self.model(train_s[idx], train_a[idx])
                loss = nn.functional.mse_loss(pred, train_t[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            epoch_loss /= steps_per_epoch

            # ---- Validation ----
            self.model.eval()
            with torch.no_grad():
                val_pred_raw = self.model(val_s, val_a)  # (B, K, 10)
                val_loss = nn.functional.mse_loss(val_pred_raw, val_t).item()

                # Per-dim MAE in 7D space (human-readable)
                val_pred_7d = DynamicsAdapter._raw_to_7d(val_pred_raw)
                diff = (val_pred_7d - val_t_7d).abs()
                pos_mae = diff[:, :, :3].mean().item()
                euler_mae = diff[:, :, 3:6].mean().item()
                grip_mae = diff[:, :, 6].mean().item()
            self.model.train()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                self._save_checkpoint(
                    "best.pt", state_mean, state_std, epoch, val_loss
                )

            elapsed = time.time() - t0
            mark = " ★" if is_best else ""
            print(
                f"[DynAdapter] epoch={epoch}/{self.cfg.epochs}  "
                f"train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}  "
                f"pos_mae={pos_mae:.5f}  euler_mae={euler_mae:.5f}  "
                f"grip_mae={grip_mae:.5f}  "
                f"elapsed={elapsed:.1f}s{mark}"
            )

        # ---- Final checkpoint ----
        self._save_checkpoint(
            "final.pt", state_mean, state_std, self.cfg.epochs, best_val_loss
        )

        ckpt_path = str(
            Path(self.cfg.checkpoint_dir) / "best.pt"
        )
        eval_metrics = {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }
        print(
            f"[DynAdapter] 训练完成: best_val_loss={best_val_loss:.6f} "
            f"(epoch {best_epoch}), checkpoint={ckpt_path}"
        )
        return {
            "best_loss": best_val_loss,
            "checkpoint_path": ckpt_path,
            "eval_metrics": eval_metrics,
        }

    def _save_checkpoint(
        self,
        filename: str,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        epoch: int,
        val_loss: float,
    ) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": {
                    "state_dim": self.cfg.state_dim,
                    "action_dim": self.cfg.action_dim,
                    "act_steps": self.cfg.act_steps,
                    "hidden_dim": self.cfg.hidden_dim,
                },
                "normalization": {
                    "state_mean": state_mean.tolist(),
                    "state_std": state_std.tolist(),
                },
                "epoch": epoch,
                "val_loss": val_loss,
            },
            str(ckpt_dir / filename),
        )

    @staticmethod
    def load_from_checkpoint(
        ckpt_path: str, device: str = "cpu"
    ) -> tuple["DynamicsAdapter", dict]:
        """Load a trained DynamicsAdapter from checkpoint.

        Args:
            ckpt_path: Path to checkpoint file (e.g., best.pt).
            device:    Target device ("cpu" or "cuda:X").

        Returns:
            (model, norm_dict) where norm_dict has "state_mean" and "state_std" as np.ndarray.
        """
        payload = torch.load(ckpt_path, map_location=device)
        cfg = payload["config"]
        model = DynamicsAdapter(
            state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"],
            act_steps=cfg["act_steps"],
            hidden_dim=cfg["hidden_dim"],
        )
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        model.eval()

        norm = payload["normalization"]
        norm_dict = {
            "state_mean": np.array(norm["state_mean"], dtype=np.float32),
            "state_std": np.array(norm["state_std"], dtype=np.float32),
        }
        return model, norm_dict

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(self, hdf5_dir: str) -> dict:
        """在全数据上评估已训练模型.

        Returns:
            {"pos_mae_mm": float, "euler_mae_rad": float, "grip_mae": float}
        """
        states, actions, targets = self._load_chunks(hdf5_dir)

        # 加载归一化参数
        ckpt_path = Path(self.cfg.checkpoint_dir) / "best.pt"
        payload = torch.load(str(ckpt_path), map_location="cpu")
        norm = payload["normalization"]
        state_mean = np.array(norm["state_mean"], dtype=np.float32)
        state_std = np.array(norm["state_std"], dtype=np.float32)

        states_n = (states - state_mean) / state_std

        self.model.eval()
        with torch.no_grad():
            s = torch.from_numpy(states_n).float().to(self.device)
            a = torch.from_numpy(actions).float().to(self.device)
            t = torch.from_numpy(targets).float().to(self.device)
            pred_7d = self.model.forward_7d(s, a)

            diff = (pred_7d - t).abs()
            pos_mae = diff[:, :, :3].mean().item()
            euler_mae = diff[:, :, 3:6].mean().item()
            grip_mae = diff[:, :, 6].mean().item()

        metrics = {
            "pos_mae_mm": pos_mae * 1000,  # m → mm
            "euler_mae_rad": euler_mae,
            "grip_mae": grip_mae,
        }
        print(
            f"[DynAdapter] Eval: pos_mae={metrics['pos_mae_mm']:.2f}mm, "
            f"euler_mae={metrics['euler_mae_rad']:.4f}rad, "
            f"grip_mae={metrics['grip_mae']:.4f}"
        )
        return metrics

    # ------------------------------------------------------------------
    # 加载 checkpoint
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_checkpoint(
        ckpt_path: str,
        device: str = "cuda",
    ) -> Tuple[DynamicsAdapter, dict]:
        """加载训练好的 adapter.

        Returns:
            (model, normalization_dict)  其中 normalization_dict 包含
            {"state_mean": np.ndarray, "state_std": np.ndarray}.
        """
        payload = torch.load(ckpt_path, map_location="cpu")
        cfg = payload["config"]
        model = DynamicsAdapter(
            state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"],
            act_steps=cfg["act_steps"],
            hidden_dim=cfg["hidden_dim"],
        )
        model.load_state_dict(payload["model_state_dict"])
        model.to(device).eval()

        norm = payload["normalization"]
        norm_dict = {
            "state_mean": np.array(norm["state_mean"], dtype=np.float32),
            "state_std": np.array(norm["state_std"], dtype=np.float32),
        }
        print(f"[DynAdapter] 加载自 {ckpt_path}, device={device}")
        return model, norm_dict


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cfg = tyro.cli(DynamicsAdapterConfig)

    if cfg.mode == "train":
        trainer = DynamicsAdapterTrainer(cfg)
        result = trainer.train(cfg.hdf5_dir)
        print(f"[DynAdapter] Result: {result}")

    elif cfg.mode == "eval":
        trainer = DynamicsAdapterTrainer(cfg)
        ckpt = Path(cfg.checkpoint_dir) / "best.pt"
        payload = torch.load(str(ckpt), map_location="cpu")
        trainer.model.load_state_dict(payload["model_state_dict"])
        metrics = trainer.evaluate(cfg.hdf5_dir)
        print(f"[DynAdapter] Metrics: {metrics}")

    elif cfg.mode == "test":
        print("[DynAdapter] 运行前向测试...")
        model = DynamicsAdapter(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            act_steps=cfg.act_steps,
            hidden_dim=cfg.hidden_dim,
        )
        B = 4
        state = torch.randn(B, cfg.state_dim)
        action_chunk = torch.randn(B, cfg.act_steps, cfg.action_dim)
        raw = model(state, action_chunk)
        assert raw.shape == (B, cfg.act_steps, model.RAW_DIM), f"Raw shape error: {raw.shape}"
        pred = model.forward_7d(state, action_chunk)
        assert pred.shape == (B, cfg.act_steps, model.EE_DIM), f"7D shape error: {pred.shape}"

        # Numpy predict test
        s0 = np.random.randn(cfg.state_dim).astype(np.float32)
        a0 = np.random.randn(cfg.act_steps, cfg.action_dim).astype(np.float32)
        ee = model.predict(s0, a0)
        assert ee.shape == (cfg.act_steps, 7), f"Numpy shape error: {ee.shape}"

        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[DynAdapter] 测试通过: raw={raw.shape}, pred_7d={pred.shape}, "
            f"numpy={ee.shape}, params={n_params:,}"
        )
    else:
        raise ValueError(f"未知 mode: {cfg.mode}")
