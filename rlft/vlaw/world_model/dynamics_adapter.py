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

    n_layers: int = 3
    """隐藏层数量 (v1=3, v2=5)."""

    model_version: str = "v1"
    """模型版本: v1 | v2 | single_step."""

    lr: float = 1e-4
    """Adam 学习率."""

    epochs: int = 20
    """训练轮数."""

    batch_size: int = 128
    """训练批大小."""

    val_ratio: float = 0.1
    """验证集比例."""

    pos_loss_weight: float = 10.0
    """Position loss 权重 (仅 v2 使用)."""

    checkpoint_dir: str = "checkpoints/vlaw/dynamics_adapter"
    """Checkpoint 保存目录."""

    gpu_id: int = 8
    """训练 GPU id."""

    mode: str = "test"
    """运行模式: train | eval | test."""

    hdf5_dir: str = "data/vlaw/rollouts/mixed/LiftPegUpright-v1"
    """训练数据 HDF5 目录 (单目录, 与 hdf5_dirs 互斥)."""

    hdf5_dirs: Tuple[str, ...] = ()
    """训练数据 HDF5 目录列表 (多目录, 优先于 hdf5_dir)."""

    # --- V3 improvements ---
    sincos_input: bool = False
    """将 input state 的 euler 角 (dims 21-23) 编码为 sin/cos (25D→28D)."""

    delta_target: bool = False
    """预测相对 current_ee 的 7D delta (而非绝对位置)."""

    weight_decay: float = 0.0
    """Adam weight decay."""

    grad_clip: float = 0.0
    """梯度裁剪 max_norm (0=不裁剪)."""

    early_stop_patience: int = 0
    """Early stopping patience (0=不启用)."""

    step_loss_decay: float = 0.0
    """Per-step loss 权重衰减 (0=均匀, 0.05=step1:1.0 step5:0.8)."""


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


class DynamicsAdapter(nn.Module):
    """Chunk-level dynamics MLP: (state, action_chunk) → future EE poses.

    Output format: [tcp_x, tcp_y, tcp_z, sin_rx, cos_rx, sin_ry, cos_ry,
                     sin_rz, cos_rz, gripper_norm]  (10D per step)
    在 predict() 时自动转回 7D euler 格式供 WM 使用。

    When output_7d=True (delta target mode), outputs raw 7D:
    [dx, dy, dz, d_rx, d_ry, d_rz, d_gripper]

    Args:
        state_dim:  ManiSkill state 维度 (25 or 28 with sincos).
        action_dim: pd_ee_delta_pose 维度 (7).
        act_steps:  chunk 内动作步数 (5).
        hidden_dim: 隐藏层宽度 (512).
        output_7d:  True=raw 7D delta output; False=10D sin/cos output (default).
    """

    EE_DIM: int = 7   # output 7D (after atan2 conversion)

    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 7,
        act_steps: int = 5,
        hidden_dim: int = 512,
        output_7d: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.act_steps = act_steps
        self.output_7d = output_7d
        self.raw_dim = 7 if output_7d else 10

        input_dim = state_dim + action_dim * act_steps
        output_dim = self.raw_dim * act_steps

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
        """前向预测 (raw output for training loss).

        Args:
            state:        (B, state_dim)  当前 agent_state.
            action_chunk: (B, act_steps, action_dim)  delta action chunk.

        Returns:
            (B, act_steps, raw_dim)  raw output (7D or 10D depending on output_7d).
        """
        B = state.shape[0]
        flat_actions = action_chunk.reshape(B, -1)
        x = torch.cat([state, flat_actions], dim=-1)
        out = self.net(x)
        return out.reshape(B, self.act_steps, self.raw_dim)

    def forward_7d(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """前向预测, 输出 7D EE pose (xyz + euler + gripper).

        将 sin/cos 编码通过 atan2 转回 euler angles (10D mode).
        For output_7d mode, returns raw output directly.
        """
        raw = self.forward(state, action_chunk)  # (B, K, raw_dim)
        if self.output_7d:
            return raw  # already 7D
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
# Single-Step Recursive Model (Exp A)
# ---------------------------------------------------------------------------


class SingleStepDynamicsAdapter(nn.Module):
    """Single-step dynamics MLP: (state, single_action, current_ee) → next_ee.

    Predicts one step at a time. During inference, chains predictions
    autoregressively (previous output becomes next input).

    Output format: 10D sin/cos (same as V1), converted to 7D for WM.

    Args:
        state_dim:  ManiSkill state dimension (25).
        action_dim: pd_ee_delta_pose dimension (7).
        ee_dim:     Current EE pose dimension (7).
        hidden_dim: Hidden layer width (512).
    """

    EE_DIM: int = 7   # output 7D (after atan2 conversion)
    RAW_DIM: int = 10  # internal 10D: xyz(3) + sin/cos×3(6) + gripper(1)

    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 7,
        ee_dim: int = 7,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ee_dim = ee_dim

        # Input: state + single_action + current_ee = 25 + 7 + 7 = 39
        input_dim = state_dim + action_dim + ee_dim
        output_dim = self.RAW_DIM

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
        self, state: torch.Tensor, action: torch.Tensor, current_ee: torch.Tensor
    ) -> torch.Tensor:
        """Single-step forward prediction.

        Args:
            state:      (B, state_dim)  Current agent_state.
            action:     (B, action_dim) Single delta action.
            current_ee: (B, 7)  Current EE pose in 7D format.

        Returns:
            (B, 10)  Raw output in sin/cos format.
        """
        x = torch.cat([state, action, current_ee], dim=-1)
        return self.net(x)

    def forward_7d(
        self, state: torch.Tensor, action: torch.Tensor, current_ee: torch.Tensor
    ) -> torch.Tensor:
        """Single-step prediction returning 7D."""
        raw = self.forward(state, action, current_ee)  # (B, 10)
        return DynamicsAdapter._raw_to_7d(raw.unsqueeze(1)).squeeze(1)

    def predict_chunk(
        self, state: torch.Tensor, action_chunk: torch.Tensor, current_ee_7d: torch.Tensor
    ) -> torch.Tensor:
        """Autoregressive prediction for K steps.

        Args:
            state:        (B, state_dim)  Current agent_state.
            action_chunk: (B, K, action_dim)  Action chunk.
            current_ee_7d: (B, 7)  Current EE pose.

        Returns:
            (B, K, 7)  Predicted future EE poses.
        """
        B, K, _ = action_chunk.shape
        preds = []
        ee = current_ee_7d

        for t in range(K):
            next_ee = self.forward_7d(state, action_chunk[:, t], ee)
            preds.append(next_ee)
            ee = next_ee  # Use prediction as next input

        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def predict(
        self,
        state: np.ndarray,
        action_chunk: np.ndarray,
        current_ee_7d: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Numpy inference — compatible with V1 interface.

        Args:
            state:        (state_dim,) or (B, state_dim)  Normalized state.
            action_chunk: (K, action_dim) or (B, K, action_dim).
            current_ee_7d: (7,) or (B, 7)  Current EE pose. If None, extracts from
                          state[18:25] (only valid if state is UN-normalized).

        Returns:
            (K, 7) or (B, K, 7)  np.float32 world frame EE.
        """
        squeeze = state.ndim == 1
        if squeeze:
            state = state[None, :]
            action_chunk = action_chunk[None, :]

        # Current EE: use provided or extract from state
        if current_ee_7d is None:
            # WARNING: This only works for unnormalized state!
            current_ee_7d = state_to_ee_pose_7d(state)
        elif current_ee_7d.ndim == 1:
            current_ee_7d = current_ee_7d[None, :]

        device = next(self.parameters()).device
        s = torch.from_numpy(state).float().to(device)
        a = torch.from_numpy(action_chunk).float().to(device)
        ee = torch.from_numpy(current_ee_7d).float().to(device)

        pred = self.predict_chunk(s, a, ee)
        result = pred.cpu().numpy()

        return result[0] if squeeze else result


# ---------------------------------------------------------------------------
# V2 模型：残差学习 + 分离头 + 更深网络
# ---------------------------------------------------------------------------


class DynamicsAdapterV2(nn.Module):
    """Improved Dynamics Adapter v2: Residual learning + deeper network.

    Key improvements over v1:
    1. Residual learning: predict delta_ee instead of absolute ee
    2. Separate position/rotation prediction heads
    3. Deeper network (5 layers) with residual connections
    4. Better input processing: state embedding + action embedding

    Args:
        state_dim:  ManiSkill state dimension (25).
        action_dim: pd_ee_delta_pose dimension (7).
        act_steps:  Chunk action steps (5).
        hidden_dim: Hidden layer width (768).
        n_layers:   Number of hidden layers (5).
    """

    EE_DIM: int = 7   # output 7D (after atan2 conversion)
    RAW_DIM: int = 10  # internal 10D: xyz(3) + sin/cos×3(6) + gripper(1)

    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 7,
        act_steps: int = 5,
        hidden_dim: int = 768,
        n_layers: int = 5,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.act_steps = act_steps
        self.hidden_dim = hidden_dim

        # State embedding: extract relevant features
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Action embedding: process each action step
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim * act_steps, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Trunk: deeper network with residual connections
        self.trunk = nn.ModuleList()
        for _ in range(n_layers):
            self.trunk.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            ))

        # Separate heads for position and rotation (delta prediction)
        # Position head: predicts delta xyz (3D per step)
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3 * act_steps),
        )

        # Rotation head: predicts sin/cos of delta euler (6D per step)
        self.rot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 6 * act_steps),
        )

        # Gripper head: predicts gripper state directly (1D per step)
        self.grip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, act_steps),
        )

    def forward(
        self, state: torch.Tensor, action_chunk: torch.Tensor, current_ee: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass predicting delta EE poses.

        Args:
            state:        (B, state_dim)  Current agent_state.
            action_chunk: (B, act_steps, action_dim)  Delta action chunk.
            current_ee:   (B, 10)  Current EE pose in 10D sin/cos format.

        Returns:
            (B, act_steps, 10)  Raw output: absolute EE in sin/cos format.
        """
        B, K, _ = action_chunk.shape

        # Embed state and actions
        h_state = self.state_embed(state)  # (B, hidden)
        h_action = self.action_embed(action_chunk.reshape(B, -1))  # (B, hidden)

        # Combine and pass through trunk with residual connections
        h = h_state + h_action
        for layer in self.trunk:
            h = h + layer(h)  # Residual connection

        # Predict delta components
        delta_pos = self.pos_head(h).reshape(B, K, 3)  # (B, K, 3)
        delta_rot_sc = self.rot_head(h).reshape(B, K, 6)  # (B, K, 6) sin/cos
        grip = self.grip_head(h).reshape(B, K, 1)  # (B, K, 1)

        # Accumulate deltas: pos[k] = current_pos + sum(delta_pos[:k+1])
        current_pos = current_ee[:, :3]  # (B, 3)
        cumsum_delta_pos = delta_pos.cumsum(dim=1)  # (B, K, 3)
        abs_pos = current_pos.unsqueeze(1) + cumsum_delta_pos  # (B, K, 3)

        # For rotation: add delta to current sin/cos and re-normalize
        # This is approximate but avoids angle discontinuities
        current_rot_sc = current_ee[:, 3:9].reshape(B, 3, 2)  # (B, 3, 2) sin/cos pairs
        delta_rot_sc = delta_rot_sc.reshape(B, K, 3, 2)  # (B, K, 3, 2)

        # Cumulative delta rotation (approximate by accumulating sin/cos deltas)
        cum_delta = delta_rot_sc.cumsum(dim=1)  # (B, K, 3, 2)
        abs_rot_sc = current_rot_sc.unsqueeze(1) + cum_delta  # (B, K, 3, 2)
        # Re-normalize each sin/cos pair
        abs_rot_sc = abs_rot_sc / (abs_rot_sc.norm(dim=-1, keepdim=True) + 1e-8)
        abs_rot_sc = abs_rot_sc.reshape(B, K, 6)  # (B, K, 6)

        # Combine: [pos(3) + rot_sc(6) + grip(1)] = 10D
        return torch.cat([abs_pos, abs_rot_sc, grip], dim=-1)

    def forward_7d(
        self, state: torch.Tensor, action_chunk: torch.Tensor, current_ee: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning 7D EE pose (xyz + euler + gripper)."""
        raw = self.forward(state, action_chunk, current_ee)  # (B, K, 10)
        return DynamicsAdapter._raw_to_7d(raw)

    @torch.no_grad()
    def predict(
        self,
        state: np.ndarray,
        action_chunk: np.ndarray,
        current_ee_7d: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Numpy inference — compatible interface with v1.

        If current_ee_7d is None, extracts from state (state[18:25] = tcp_pose).

        Args:
            state:        (state_dim,) or (B, state_dim)  Normalized state.
            action_chunk: (act_steps, action_dim) or (B, act_steps, action_dim).
            current_ee_7d: (7,) or (B, 7) optional current EE in 7D format.

        Returns:
            (act_steps, 7) or (B, act_steps, 7)  np.float32 world frame EE.
        """
        squeeze = state.ndim == 1
        if squeeze:
            state = state[None, :]
            action_chunk = action_chunk[None, :]

        # Extract current EE from state if not provided
        if current_ee_7d is None:
            current_ee_7d = state_to_ee_pose_7d(state)  # (B, 7)
        elif current_ee_7d.ndim == 1:
            current_ee_7d = current_ee_7d[None, :]

        # Convert 7D to 10D
        current_ee_10d = DynamicsAdapterTrainer._targets_7d_to_10d(
            current_ee_7d[:, None, :]
        )[:, 0, :]  # (B, 10)

        device = next(self.parameters()).device
        s = torch.from_numpy(state).float().to(device)
        a = torch.from_numpy(action_chunk).float().to(device)
        ee = torch.from_numpy(current_ee_10d).float().to(device)

        pred = self.forward_7d(s, a, ee)  # (B, K, 7)
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

        # Create model based on version
        if config.model_version == "v2":
            self.model = DynamicsAdapterV2(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                act_steps=config.act_steps,
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
            ).to(self.device)
        elif config.model_version == "single_step":
            self.model = SingleStepDynamicsAdapter(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                ee_dim=7,
                hidden_dim=config.hidden_dim,
            ).to(self.device)
        else:
            self.model = DynamicsAdapter(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                act_steps=config.act_steps,
                hidden_dim=config.hidden_dim,
            ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"[DynAdapter {config.model_version}] 初始化完成: {n_params:,} params, device={self.device}"
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

    @staticmethod
    def encode_state_sincos(states: np.ndarray) -> np.ndarray:
        """Encode euler angles in state vector as sin/cos.

        Input:  (..., 25) with euler at dims [21, 22, 23]
        Output: (..., 28) with sin/cos at dims [21-26]

        State layout: qpos(9) + qvel(9) + tcp_xyz(3) + euler_rx,ry,rz(3) + gripper(1)
        """
        pre = states[..., :21]         # qpos + qvel + tcp_xyz
        euler = states[..., 21:24]     # euler_rx, ry, rz
        grip = states[..., 24:25]      # gripper
        sin_e = np.sin(euler)
        cos_e = np.cos(euler)
        sc = np.stack([sin_e, cos_e], axis=-1)  # (..., 3, 2)
        sc = sc.reshape(*euler.shape[:-1], 6)    # (..., 6)
        return np.concatenate([pre, sc, grip], axis=-1)  # (..., 28)

    # ------------------------------------------------------------------

    def _load_chunks(
        self, hdf5_dir: str, extra_dirs: Tuple[str, ...] = ()
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """从 HDF5 提取 (state_t, action_chunk[t:t+K], target_ee[t+1:t+K+1], current_ee[t]).

        Returns:
            states:     (N, state_dim)  — 28D if sincos_input else 25D
            actions:    (N, act_steps, action_dim)
            targets:    (N, act_steps, 7)  — delta if delta_target else absolute
            current_ee: (N, 7)  Current EE pose at time t
        """
        K = self.cfg.act_steps
        all_dirs = [hdf5_dir] + list(extra_dirs)
        h5_files = []
        for d in all_dirs:
            found = sorted(Path(d).glob("**/*.h5"))
            h5_files.extend(found)
            if not found:
                print(f"[DynAdapter] ⚠️ 未找到 HDF5: {d}")
        if not h5_files:
            raise FileNotFoundError(f"[DynAdapter] 所有目录均无 HDF5: {all_dirs}")

        all_states: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_current_ee: list[np.ndarray] = []
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
                        # current_ee: 当前帧的 EE pose (world frame)
                        current_ee = state_to_ee_pose_7d(state_arr[t : t + 1])[0]
                        all_current_ee.append(current_ee)
                        # target: 未来 K 步各帧的 EE pose (world frame)
                        future_states = state_arr[t + 1 : t + K + 1]
                        all_targets.append(state_to_ee_pose_7d(future_states))
                    total_traj += 1

        states = np.stack(all_states, axis=0)
        actions = np.stack(all_actions, axis=0)
        targets = np.stack(all_targets, axis=0)
        current_ee = np.stack(all_current_ee, axis=0)

        # --- V3 preprocessing ---
        if self.cfg.sincos_input:
            states = self.encode_state_sincos(states)  # (N, 25) → (N, 28)
        if self.cfg.delta_target:
            targets = targets - current_ee[:, None, :]  # (N, K, 7) delta

        print(
            f"[DynAdapter] 数据加载: {total_traj} 条轨迹, "
            f"{states.shape[0]} 个训练对 (K={K}), "
            f"state_dim={states.shape[1]}, "
            f"delta_target={self.cfg.delta_target}"
        )
        return states, actions, targets, current_ee

    def _load_single_steps(
        self, hdf5_dir: str, extra_dirs: Tuple[str, ...] = ()
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """从 HDF5 提取单步训练对 (state_t, action_t, current_ee_t, next_ee_t+1).

        For single_step model: predicts one step at a time.

        Returns:
            states:     (N, state_dim)
            actions:    (N, action_dim)  single action per sample
            current_ee: (N, 7)  Current EE pose at time t
            next_ee:    (N, 7)  Next EE pose at time t+1 (target)
        """
        all_dirs = [hdf5_dir] + list(extra_dirs)
        h5_files = []
        for d in all_dirs:
            found = sorted(Path(d).glob("**/*.h5"))
            h5_files.extend(found)
            if not found:
                print(f"[DynAdapter] ⚠️ 未找到 HDF5: {d}")
        if not h5_files:
            raise FileNotFoundError(f"[DynAdapter] 所有目录均无 HDF5: {all_dirs}")

        all_states: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_current_ee: list[np.ndarray] = []
        all_next_ee: list[np.ndarray] = []
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
                    if T < 2:
                        continue

                    # Extract single-step pairs: (state[t], action[t]) → ee[t+1]
                    ee_all = state_to_ee_pose_7d(state_arr[:T])  # (T, 7)
                    for t in range(T - 1):
                        all_states.append(state_arr[t])
                        all_actions.append(act_arr[t])
                        all_current_ee.append(ee_all[t])
                        all_next_ee.append(ee_all[t + 1])
                    total_traj += 1

        states = np.stack(all_states, axis=0)
        actions = np.stack(all_actions, axis=0)
        current_ee = np.stack(all_current_ee, axis=0)
        next_ee = np.stack(all_next_ee, axis=0)

        print(
            f"[DynAdapter single_step] 数据加载: {total_traj} 条轨迹, "
            f"{states.shape[0]} 个单步训练对, state_dim={states.shape[1]}"
        )
        return states, actions, current_ee, next_ee

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(self, hdf5_dir: str) -> dict:
        """完整训练流程.

        Returns:
            {"best_loss": float, "checkpoint_path": str, "eval_metrics": dict}
        """
        # Dispatch to single_step training if needed
        if self.cfg.model_version == "single_step":
            return self._train_single_step(hdf5_dir)

        is_v2 = self.cfg.model_version == "v2"
        use_delta = self.cfg.delta_target
        extra_dirs = self.cfg.hdf5_dirs
        print(
            f"[DynAdapter {self.cfg.model_version}] 开始训练, "
            f"数据: {hdf5_dir} + {len(extra_dirs)} extra dirs, "
            f"sincos_input={self.cfg.sincos_input}, delta_target={use_delta}"
        )

        # ---- 加载数据 ----
        states, actions, targets, current_ee = self._load_chunks(
            hdf5_dir, extra_dirs=extra_dirs
        )
        N = states.shape[0]
        actual_state_dim = states.shape[1]  # 28 if sincos_input else 25

        # Rebuild model if state_dim changed due to sincos_input
        if actual_state_dim != self.model.state_dim:
            print(f"[DynAdapter] Rebuilding model: state_dim {self.model.state_dim} → {actual_state_dim}")
            self.model = DynamicsAdapter(
                state_dim=actual_state_dim,
                action_dim=self.cfg.action_dim,
                act_steps=self.cfg.act_steps,
                hidden_dim=self.cfg.hidden_dim,
                output_7d=use_delta,
            ).to(self.device)
        elif use_delta and not getattr(self.model, 'output_7d', False):
            print(f"[DynAdapter] Rebuilding model for delta_target (output_7d=True)")
            self.model = DynamicsAdapter(
                state_dim=actual_state_dim,
                action_dim=self.cfg.action_dim,
                act_steps=self.cfg.act_steps,
                hidden_dim=self.cfg.hidden_dim,
                output_7d=True,
            ).to(self.device)

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

        # For delta_target: train on 7D directly; otherwise use 10D sin/cos
        if use_delta:
            train_t_tensor = _to_tensor(targets[train_idx])    # (N, K, 7)
            val_t_tensor = _to_tensor(targets[val_idx])
            val_t_7d = val_t_tensor  # already 7D delta
        else:
            targets_10d = self._targets_7d_to_10d(targets)
            train_t_tensor = _to_tensor(targets_10d[train_idx])
            val_t_tensor = _to_tensor(targets_10d[val_idx])
            val_t_7d = _to_tensor(targets[val_idx])

        current_ee_10d = self._targets_7d_to_10d(current_ee[:, None, :])[:, 0, :]

        train_s = _to_tensor(states_n[train_idx])
        train_a = _to_tensor(actions[train_idx])
        train_ee = _to_tensor(current_ee_10d[train_idx])
        val_s = _to_tensor(states_n[val_idx])
        val_a = _to_tensor(actions[val_idx])
        val_ee = _to_tensor(current_ee_10d[val_idx])

        print(
            f"[DynAdapter] Train: {len(train_idx)}, Val: {len(val_idx)}, "
            f"state_dim={actual_state_dim}, target_dim={'7D delta' if use_delta else '10D abs'}"
        )

        # ---- Per-step loss weights ----
        step_weights = None
        if self.cfg.step_loss_decay > 0:
            K = self.cfg.act_steps
            w = [1.0 - self.cfg.step_loss_decay * i for i in range(K)]
            step_weights = torch.tensor(w, device=self.device).float()
            step_weights = step_weights / step_weights.sum() * K  # normalize to sum=K
            print(f"[DynAdapter] Step loss weights: {[f'{x:.3f}' for x in step_weights.tolist()]}")

        # ---- 优化器 ----
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
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
        no_improve_count = 0
        t0 = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                idx = torch.randint(0, n_train, (self.cfg.batch_size,))
                if is_v2:
                    pred = self.model(train_s[idx], train_a[idx], train_ee[idx])
                else:
                    pred = self.model(train_s[idx], train_a[idx])

                target_batch = train_t_tensor[idx]

                # V2 uses weighted loss: position more important than rotation
                if is_v2:
                    pos_loss = nn.functional.mse_loss(pred[..., :3], target_batch[..., :3])
                    rot_loss = nn.functional.mse_loss(pred[..., 3:9], target_batch[..., 3:9])
                    grip_loss = nn.functional.mse_loss(pred[..., 9:], target_batch[..., 9:])
                    loss = self.cfg.pos_loss_weight * pos_loss + rot_loss + grip_loss
                elif step_weights is not None:
                    # Per-step weighted MSE
                    per_step = ((pred - target_batch) ** 2).mean(dim=-1)  # (B, K)
                    loss = (per_step * step_weights[None, :]).mean()
                else:
                    loss = nn.functional.mse_loss(pred, target_batch)

                optimizer.zero_grad()
                loss.backward()
                if self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            epoch_loss /= steps_per_epoch

            # ---- Validation ----
            self.model.eval()
            with torch.no_grad():
                if is_v2:
                    val_pred_raw = self.model(val_s, val_a, val_ee)
                else:
                    val_pred_raw = self.model(val_s, val_a)
                val_loss = nn.functional.mse_loss(val_pred_raw, val_t_tensor).item()

                # Per-dim MAE in 7D space
                if use_delta:
                    val_pred_7d_out = val_pred_raw  # already 7D
                else:
                    val_pred_7d_out = DynamicsAdapter._raw_to_7d(val_pred_raw)
                diff = (val_pred_7d_out - val_t_7d).abs()
                pos_mae = diff[:, :, :3].mean().item()
                euler_mae = diff[:, :, 3:6].mean().item()
                grip_mae = diff[:, :, 6].mean().item()
            self.model.train()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve_count = 0
                self._save_checkpoint(
                    "best.pt", state_mean, state_std, epoch, val_loss
                )
            else:
                no_improve_count += 1

            elapsed = time.time() - t0
            mark = " ★" if is_best else ""
            print(
                f"[DynAdapter] epoch={epoch}/{self.cfg.epochs}  "
                f"train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}  "
                f"pos_mae={pos_mae:.5f}  euler_mae={euler_mae:.5f}  "
                f"grip_mae={grip_mae:.5f}  "
                f"elapsed={elapsed:.1f}s{mark}"
            )

            # Early stopping
            if self.cfg.early_stop_patience > 0 and no_improve_count >= self.cfg.early_stop_patience:
                print(f"[DynAdapter] Early stopping at epoch {epoch} (patience={self.cfg.early_stop_patience})")
                break

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

    def _train_single_step(self, hdf5_dir: str) -> dict:
        """Training flow for single_step model with teacher forcing.

        Returns:
            {"best_loss": float, "checkpoint_path": str, "eval_metrics": dict}
        """
        extra_dirs = self.cfg.hdf5_dirs
        print(
            f"[DynAdapter single_step] 开始训练, "
            f"数据: {hdf5_dir} + {len(extra_dirs)} extra dirs"
        )

        # ---- Load single-step data ----
        states, actions, current_ee, next_ee = self._load_single_steps(
            hdf5_dir, extra_dirs=extra_dirs
        )
        N = states.shape[0]

        # ---- Compute normalization stats ----
        state_mean = states.mean(axis=0)
        state_std = states.std(axis=0).clip(min=1e-6)

        # ---- Train/val split ----
        n_val = max(1, int(N * self.cfg.val_ratio))
        perm = np.random.RandomState(42).permutation(N)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).float().to(self.device)

        states_n = (states - state_mean) / state_std

        # Convert targets to 10D format
        next_ee_10d = self._targets_7d_to_10d(next_ee[:, None, :])[:, 0, :]

        train_s = _to_tensor(states_n[train_idx])
        train_a = _to_tensor(actions[train_idx])
        train_ee = _to_tensor(current_ee[train_idx])  # 7D input for teacher forcing
        train_t = _to_tensor(next_ee_10d[train_idx])  # 10D target
        val_s = _to_tensor(states_n[val_idx])
        val_a = _to_tensor(actions[val_idx])
        val_ee = _to_tensor(current_ee[val_idx])
        val_t = _to_tensor(next_ee_10d[val_idx])
        val_t_7d = _to_tensor(next_ee[val_idx])

        print(
            f"[DynAdapter single_step] Train: {len(train_idx)}, Val: {len(val_idx)}"
        )

        # ---- Optimizer ----
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        n_train = len(train_idx)
        steps_per_epoch = max(1, n_train // self.cfg.batch_size)
        total_steps = self.cfg.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        # ---- Training loop ----
        self.model.train()
        best_val_loss = float("inf")
        best_epoch = 0
        no_improve_count = 0
        t0 = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                idx = torch.randint(0, n_train, (self.cfg.batch_size,))
                pred = self.model(train_s[idx], train_a[idx], train_ee[idx])
                loss = nn.functional.mse_loss(pred, train_t[idx])

                optimizer.zero_grad()
                loss.backward()
                if self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            epoch_loss /= steps_per_epoch

            # ---- Validation ----
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(val_s, val_a, val_ee)
                val_loss = nn.functional.mse_loss(val_pred, val_t).item()

                # Convert to 7D for MAE
                val_pred_7d = DynamicsAdapter._raw_to_7d(val_pred.unsqueeze(1)).squeeze(1)
                diff = (val_pred_7d - val_t_7d).abs()
                pos_mae = diff[:, :3].mean().item()
                euler_mae = diff[:, 3:6].mean().item()
                grip_mae = diff[:, 6].mean().item()
            self.model.train()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve_count = 0
                self._save_checkpoint(
                    "best.pt", state_mean, state_std, epoch, val_loss
                )
            else:
                no_improve_count += 1

            elapsed = time.time() - t0
            mark = " ★" if is_best else ""
            print(
                f"[DynAdapter single_step] epoch={epoch}/{self.cfg.epochs}  "
                f"train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}  "
                f"pos_mae={pos_mae:.5f}  euler_mae={euler_mae:.5f}  "
                f"grip_mae={grip_mae:.5f}  "
                f"elapsed={elapsed:.1f}s{mark}"
            )

            if self.cfg.early_stop_patience > 0 and no_improve_count >= self.cfg.early_stop_patience:
                print(f"[DynAdapter single_step] Early stopping at epoch {epoch}")
                break

        # ---- Final checkpoint ----
        self._save_checkpoint(
            "final.pt", state_mean, state_std, self.cfg.epochs, best_val_loss
        )

        ckpt_path = str(Path(self.cfg.checkpoint_dir) / "best.pt")
        print(
            f"[DynAdapter single_step] 训练完成: best_val_loss={best_val_loss:.6f} "
            f"(epoch {best_epoch}), checkpoint={ckpt_path}"
        )
        return {
            "best_loss": best_val_loss,
            "checkpoint_path": ckpt_path,
            "eval_metrics": {"best_val_loss": best_val_loss, "best_epoch": best_epoch},
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
                    "state_dim": self.model.state_dim,
                    "action_dim": self.cfg.action_dim,
                    "act_steps": self.cfg.act_steps,
                    "hidden_dim": self.cfg.hidden_dim,
                    "model_version": self.cfg.model_version,
                    "n_layers": self.cfg.n_layers,
                    "sincos_input": self.cfg.sincos_input,
                    "delta_target": self.cfg.delta_target,
                    "output_7d": getattr(self.model, 'output_7d', False),
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
    ) -> tuple["DynamicsAdapter | DynamicsAdapterV2 | SingleStepDynamicsAdapter", dict]:
        """Load a trained DynamicsAdapter (V1, V2, or single_step) from checkpoint.

        Args:
            ckpt_path: Path to checkpoint file (e.g., best.pt).
            device:    Target device ("cpu" or "cuda:X").

        Returns:
            (model, norm_dict) where norm_dict has "state_mean", "state_std",
            and V3 flags "sincos_input", "delta_target" as needed.
        """
        payload = torch.load(ckpt_path, map_location=device)
        cfg = payload["config"]
        model_version = cfg.get("model_version", "v1")

        if model_version == "v2":
            model = DynamicsAdapterV2(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                act_steps=cfg["act_steps"],
                hidden_dim=cfg["hidden_dim"],
                n_layers=cfg.get("n_layers", 5),
            )
        elif model_version == "single_step":
            model = SingleStepDynamicsAdapter(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                ee_dim=7,
                hidden_dim=cfg["hidden_dim"],
            )
        else:
            model = DynamicsAdapter(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                act_steps=cfg["act_steps"],
                hidden_dim=cfg["hidden_dim"],
                output_7d=cfg.get("output_7d", False),
            )

        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        model.eval()

        norm = payload["normalization"]
        norm_dict = {
            "state_mean": np.array(norm["state_mean"], dtype=np.float32),
            "state_std": np.array(norm["state_std"], dtype=np.float32),
            "sincos_input": cfg.get("sincos_input", False),
            "delta_target": cfg.get("delta_target", False),
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
        is_v2 = self.cfg.model_version == "v2"
        states, actions, targets, current_ee = self._load_chunks(hdf5_dir)

        # 加载归一化参数
        ckpt_path = Path(self.cfg.checkpoint_dir) / "best.pt"
        payload = torch.load(str(ckpt_path), map_location="cpu")
        norm = payload["normalization"]
        state_mean = np.array(norm["state_mean"], dtype=np.float32)
        state_std = np.array(norm["state_std"], dtype=np.float32)

        states_n = (states - state_mean) / state_std
        current_ee_10d = self._targets_7d_to_10d(current_ee[:, None, :])[:, 0, :]

        self.model.eval()
        with torch.no_grad():
            s = torch.from_numpy(states_n).float().to(self.device)
            a = torch.from_numpy(actions).float().to(self.device)
            t = torch.from_numpy(targets).float().to(self.device)
            ee = torch.from_numpy(current_ee_10d).float().to(self.device)

            if is_v2:
                pred_7d = self.model.forward_7d(s, a, ee)
            else:
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
            f"[DynAdapter {self.cfg.model_version}] Eval: pos_mae={metrics['pos_mae_mm']:.2f}mm, "
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
    ) -> Tuple["DynamicsAdapter | DynamicsAdapterV2 | SingleStepDynamicsAdapter", dict]:
        """加载训练好的 adapter (V1, V2, 或 single_step).

        Returns:
            (model, normalization_dict)  其中 normalization_dict 包含
            {"state_mean", "state_std", "sincos_input", "delta_target"}.
        """
        payload = torch.load(ckpt_path, map_location="cpu")
        cfg = payload["config"]
        model_version = cfg.get("model_version", "v1")

        if model_version == "v2":
            model = DynamicsAdapterV2(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                act_steps=cfg["act_steps"],
                hidden_dim=cfg["hidden_dim"],
                n_layers=cfg.get("n_layers", 5),
            )
        elif model_version == "single_step":
            model = SingleStepDynamicsAdapter(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                ee_dim=7,
                hidden_dim=cfg["hidden_dim"],
            )
        else:
            model = DynamicsAdapter(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                act_steps=cfg["act_steps"],
                hidden_dim=cfg["hidden_dim"],
                output_7d=cfg.get("output_7d", False),
            )

        model.load_state_dict(payload["model_state_dict"])
        model.to(device).eval()

        norm = payload["normalization"]
        norm_dict = {
            "state_mean": np.array(norm["state_mean"], dtype=np.float32),
            "state_std": np.array(norm["state_std"], dtype=np.float32),
            "sincos_input": cfg.get("sincos_input", False),
            "delta_target": cfg.get("delta_target", False),
        }
        print(f"[DynAdapter {model_version}] 加载自 {ckpt_path}, device={device}, "
              f"sincos={norm_dict['sincos_input']}, delta={norm_dict['delta_target']}")
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
        assert raw.shape == (B, cfg.act_steps, model.raw_dim), f"Raw shape error: {raw.shape}"
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
