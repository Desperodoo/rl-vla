"""VLAW P4.1 — State Predictor（状态递推器，⚠️ 临时脚手架）

⚠️  本模块是 Imagination 流程的临时脚手架，仅用于跑通代码流程。
    最终方案（P4.3）：直接调用 ManiSkill env.step() 获取精确 state_{t+1}。
    - 理由：本项目用 ManiSkill 仿真替代真机，env.step() 完全可用且精确
    - P4.3 完成后，本模块将降为可选依赖
    参见 ADR-004 和 ADR-006（.github/knowledge/decisions.md）

当前工作方式：用轻量残差 MLP 近似状态转移，避免在调通流程阶段引入 env 依赖：
    state_{t+1} = state_t + MLP(concat(state_t, action_t))

ShortCut Flow 的 obs = concat([visual_feature, agent_state])
State Predictor 负责在 imagination 中维护 agent_state 序列（临时方案）。

所属阶段: P4.1 — State Predictor（临时方案）
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


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class StatePredictorConfig:
    """P4.1 State Predictor 训练配置."""

    state_dim: int = 25
    """ManiSkill agent_state 维度（实测: qpos+qvel=25D，非之前假设的 29D）"""

    action_dim: int = 7
    """delta pose 动作维度"""

    hidden_dim: int = 256
    """MLP 隐藏层维度"""

    num_layers: int = 2
    """MLP 隐藏层数 (不含输入/输出)"""

    lr: float = 1e-3
    """Adam 学习率"""

    max_steps: int = 5000
    """最大训练步数"""

    batch_size: int = 256
    """训练批大小"""

    checkpoint_dir: str = "checkpoints/vlaw/state_predictor"
    """checkpoint 保存目录"""

    gpu_id: int = 4
    """训练使用的 GPU id"""

    mode: str = "test"
    """运行模式: train | test"""

    hdf5_dir: str = "data/vlaw/rollouts/iter1"
    """训练数据 HDF5 目录 (mode=train 时使用)"""


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


class StatePredictor(nn.Module):
    """残差 MLP 状态递推器.

    预测 delta_state，然后加到 state_t 得到 state_{t+1}：
        state_{t+1} = state_t + MLP(concat(state_t, action_t))

    Architecture:
        Linear(state_dim + action_dim → hidden) → ReLU → LayerNorm
        → Linear(hidden → hidden) → ReLU
        → Linear(hidden → state_dim)   # 预测 delta

    Args:
        state_dim:  agent_state 向量维度
        action_dim: delta pose 动作维度
        hidden_dim: 隐藏层宽度
    """

    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 7,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        inp_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """单步递推.

        Args:
            state:  (B, state_dim)  当前 agent_state（已归一化）
            action: (B, action_dim) 执行的 delta pose 动作（已归一化）

        Returns:
            next_state: (B, state_dim) 预测的下一 agent_state
        """
        x = torch.cat([state, action], dim=-1)  # (B, state_dim + action_dim)
        delta = self.net(x)                      # (B, state_dim)
        return state + delta                     # 残差连接

    @torch.no_grad()
    def predict_sequence(
        self,
        state_0: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """给定初始状态和动作序列，递推完整状态序列.

        Args:
            state_0: (state_dim,) 初始 agent_state（原始尺度，会自动归一化/反归一化）
            actions: (T, action_dim) 动作序列

        Returns:
            states: (T+1, state_dim) 状态序列（与 state_0 同尺度）
        """
        T = actions.shape[0]
        device = next(self.parameters()).device

        state_t = torch.tensor(state_0, dtype=torch.float32, device=device).unsqueeze(0)  # (1, S)
        act_tensor = torch.tensor(actions, dtype=torch.float32, device=device)             # (T, A)

        seq = [state_t.cpu().numpy().squeeze(0)]
        for t in range(T):
            a_t = act_tensor[t].unsqueeze(0)   # (1, A)
            state_t = self(state_t, a_t)
            seq.append(state_t.cpu().numpy().squeeze(0))

        return np.stack(seq, axis=0)  # (T+1, state_dim)


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------


class StatePredictorTrainer:
    """P4.1 State Predictor 训练器.

    从 HDF5 rollout 数据中学习 (state_t, action_t) → state_{t+1} 的映射。

    Args:
        config: StatePredictorConfig 实例
    """

    def __init__(self, config: StatePredictorConfig) -> None:
        self.config = config
        self.device = torch.device(
            f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.model = StatePredictor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        print(f"[VLAW-P4.1] StatePredictor 初始化完成, device={self.device}")

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def collect_training_data(
        self, hdf5_dir: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从 HDF5 rollout 数据中提取 (states, actions, next_states) 三元组.

        遍历目录下所有 .h5 文件，从每条轨迹提取逐帧转换对。

        Args:
            hdf5_dir: 包含 .h5 文件的目录路径

        Returns:
            states:      (N, state_dim)  float32 Tensor
            actions:     (N, action_dim) float32 Tensor
            next_states: (N, state_dim)  float32 Tensor
        """
        hdf5_path = Path(hdf5_dir)
        h5_files = sorted(hdf5_path.glob("**/*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"[VLAW-P4.1] 未找到 HDF5 文件: {hdf5_dir}")

        all_states, all_actions, all_next_states = [], [], []
        total_traj = 0

        for h5_file in h5_files:
            with h5py.File(str(h5_file), "r") as f:
                traj_keys = [k for k in f.keys() if k.startswith("traj_")]
                for key in traj_keys:
                    grp = f[key]
                    # 优先用 obs_agent / state
                    if "obs_agent" in grp:
                        state_arr = grp["obs_agent"][:]   # (T, agent_dim)
                    elif "state" in grp:
                        state_arr = grp["state"][:]
                    else:
                        continue
                    if "actions" not in grp:
                        continue
                    act_arr = grp["actions"][:]           # (T, action_dim)
                    T = min(state_arr.shape[0], act_arr.shape[0]) - 1
                    if T <= 0:
                        continue
                    all_states.append(state_arr[:T].astype(np.float32))
                    all_actions.append(act_arr[:T].astype(np.float32))
                    all_next_states.append(state_arr[1:T+1].astype(np.float32))
                    total_traj += 1

        if not all_states:
            raise RuntimeError(f"[VLAW-P4.1] HDF5 中未找到有效轨迹数据: {hdf5_dir}")

        states = torch.from_numpy(np.concatenate(all_states, axis=0))
        actions = torch.from_numpy(np.concatenate(all_actions, axis=0))
        next_states = torch.from_numpy(np.concatenate(all_next_states, axis=0))

        # 截断到 config.state_dim / action_dim
        states = states[:, : self.config.state_dim]
        actions = actions[:, : self.config.action_dim]
        next_states = next_states[:, : self.config.state_dim]

        print(
            f"[VLAW-P4.1] 数据加载: {total_traj} 条轨迹, "
            f"{states.shape[0]} 个转换对"
        )
        return states, actions, next_states

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(self, hdf5_dir: str) -> dict:
        """完整训练流程.

        Args:
            hdf5_dir: HDF5 数据目录

        Returns:
            {"final_loss": float, "checkpoint_path": str}
        """
        print(f"[VLAW-P4.1] 开始训练 State Predictor, 数据目录: {hdf5_dir}")

        # ---- 加载数据 ----
        states, actions, next_states = self.collect_training_data(hdf5_dir)
        N = states.shape[0]

        # ---- 计算归一化统计量 ----
        state_mean = states.mean(dim=0).numpy()
        state_std = states.std(dim=0).clamp(min=1e-6).numpy()

        # 归一化
        states_n = (states - torch.from_numpy(state_mean)) / torch.from_numpy(state_std)
        next_states_n = (next_states - torch.from_numpy(state_mean)) / torch.from_numpy(state_std)

        states_n = states_n.to(self.device)
        actions = actions.to(self.device)
        next_states_n = next_states_n.to(self.device)

        # ---- 优化器 ----
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.max_steps
        )
        loss_fn = nn.MSELoss()

        # ---- 训练循环 ----
        self.model.train()
        t0 = time.time()
        final_loss = float("inf")
        log_interval = max(self.config.max_steps // 20, 1)

        for step in range(1, self.config.max_steps + 1):
            idx = torch.randint(0, N, (self.config.batch_size,), device=self.device)
            s = states_n[idx]
            a = actions[idx]
            ns = next_states_n[idx]

            pred = self.model(s, a)
            loss = loss_fn(pred, ns)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            final_loss = loss.item()
            if step % log_interval == 0 or step == 1:
                elapsed = time.time() - t0
                print(
                    f"[VLAW-P4.1] step={step}/{self.config.max_steps}  "
                    f"loss={final_loss:.6f}  elapsed={elapsed:.1f}s"
                )

        # ---- 保存 checkpoint ----
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "state_predictor.pt"

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": {
                    "state_dim": self.config.state_dim,
                    "action_dim": self.config.action_dim,
                    "hidden_dim": self.config.hidden_dim,
                },
            },
            str(ckpt_path),
        )

        # ---- 保存归一化统计量 ----
        stats_path = ckpt_dir / "state_stats.json"
        stats = {
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
        }
        with open(str(stats_path), "w") as fp:
            json.dump(stats, fp, indent=2)

        print(
            f"[VLAW-P4.1] 训练完成。final_loss={final_loss:.6f}, "
            f"checkpoint={ckpt_path}, stats={stats_path}"
        )
        return {"final_loss": float(final_loss), "checkpoint_path": str(ckpt_path)}

    # ------------------------------------------------------------------
    # 加载已有 checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_path: str,
        device: str = "cuda",
    ) -> StatePredictor:
        """从 checkpoint 加载 StatePredictor 模型.

        Args:
            ckpt_path: .pt 文件路径
            device:    目标设备字符串

        Returns:
            加载好权重的 StatePredictor 实例
        """
        payload = torch.load(ckpt_path, map_location="cpu")
        cfg = payload.get("config", {})
        model = StatePredictor(
            state_dim=cfg.get("state_dim", 29),
            action_dim=cfg.get("action_dim", 7),
            hidden_dim=cfg.get("hidden_dim", 256),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.to(device).eval()
        print(f"[VLAW-P4.1] StatePredictor 权重加载自: {ckpt_path}")
        return model


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

@dataclass
class _EntryConfig:
    """state_predictor.py 入口配置."""
    mode: str = "test"
    """运行模式: train | test"""
    hdf5_dir: str = "data/vlaw/rollouts/iter1"
    """训练数据目录 (mode=train 时使用)"""
    predictor: StatePredictorConfig = field(default_factory=StatePredictorConfig)
    """State Predictor 超参配置"""


if __name__ == "__main__":
    entry = tyro.cli(_EntryConfig)
    cfg = entry.predictor

    if entry.mode == "train":
        trainer = StatePredictorTrainer(cfg)
        result = trainer.train(entry.hdf5_dir)
        print(f"[VLAW-P4.1] 训练结果: {result}")

    elif entry.mode == "test":
        print("[VLAW-P4.1] 运行随机数据前向测试...")
        model = StatePredictor(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
        )
        B = 8
        state = torch.randn(B, cfg.state_dim)
        action = torch.randn(B, cfg.action_dim)
        next_s = model(state, action)
        assert next_s.shape == (B, cfg.state_dim), f"形状错误: {next_s.shape}"

        # predict_sequence 测试
        s0 = np.random.randn(cfg.state_dim).astype(np.float32)
        acts = np.random.randn(10, cfg.action_dim).astype(np.float32)
        seq = model.predict_sequence(s0, acts)
        assert seq.shape == (11, cfg.state_dim), f"序列形状错误: {seq.shape}"

        print(
            f"[VLAW-P4.1] forward 测试通过: "
            f"next_state={next_s.shape}, sequence={seq.shape}"
        )
    else:
        raise ValueError(f"未知 mode: {entry.mode}，请使用 train 或 test")
