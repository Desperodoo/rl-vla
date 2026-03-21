"""VLAW P8.2 — ImaginationRLEnv: WM+VLM 封装为标准 Gym 环境接口.

将 Ctrl-World 世界模型 + VLM 奖励模型封装为 gymnasium.Env 兼容接口，
支持 RLPD / DSRL / PLD 等 Model-based RL 算法直接使用。

核心流程 (每步):
    1. 收到 RL agent 动作 (7D delta pose)
    2. State Predictor 预测下一步 agent_state
    3. World Model (Ctrl-World) 生成下一帧 latent
    4. 定期调用 VLM 计算 reward (p_yes)
    5. 返回 (obs, reward, terminated, truncated, info)

观测空间:
    - latent 模式: Dict{"latent": (4,48,24), "agent_state": (state_dim,)}
    - flat 模式:   Box(4*48*24 + state_dim,) — 兼容现有 RL pipeline

动作空间: Box(7,) — pd_ee_delta_pose

所属阶段: P8.2 — Imagination RL (Model-based RL in World Model)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

if TYPE_CHECKING:
    from rlft.vlaw.reward.reward_model import VLAWRewardModel
    from rlft.vlaw.world_model.ctrl_world_adapter import CtrlWorldAdapter

try:
    from rlft.vlaw.world_model.imagination_env import get_history_indices, ee_pose_base_to_world
    from ctrl_world.dataset.dataset_maniskill import state_to_ee_pose_7d
except ImportError:
    # 作为脚本直接运行时的 fallback
    import sys as _sys
    import os as _os
    _script_root = _os.path.dirname(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    )
    if _script_root not in _sys.path:
        _sys.path.insert(0, _script_root)
    from rlft.vlaw.world_model.imagination_env import get_history_indices, ee_pose_base_to_world
    from ctrl_world.dataset.dataset_maniskill import state_to_ee_pose_7d


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class ImaginationRLEnvConfig:
    """ImaginationRLEnv 配置 (tyro dataclass).

    将 WM + VLM 封装为 Gym env 的全部超参。
    """

    # --- 环境核心 ---
    max_steps: int = 60
    """每幕最大步数 (默认 num_interact=12 × act_steps=5 = 60)"""

    action_dim: int = 7
    """动作维度 (pd_ee_delta_pose)"""

    state_dim: int = 25
    """agent_state 维度 (LiftPegUpright=25, PickCube=29, StackCube=25)"""

    # --- 世界模型 ---
    wm_act_steps: int = 5
    """每次 WM rollout 预测的帧数 (= Ctrl-World num_frames)"""

    num_history: int = 6
    """WM 历史帧数量 (官方 DROID 配置)"""

    history_idx: list[int] = field(
        default_factory=lambda: [0, 0, -12, -9, -6, -3]
    )
    """稀疏采样索引模板 (对齐官方 Ctrl-World)"""

    # --- VLM 奖励 ---
    vlm_reward_interval: int = 5
    """每 N 步调用一次 VLM 评估 reward (VLM 慢~0.4s, 不能每步调)"""

    vlm_reward_alpha: float = 0.8
    """VLM 二值化阈值"""

    use_continuous_reward: bool = True
    """True: 使用 p_yes 连续值; False: 二值化 (p_yes > alpha)"""

    vlm_failure_threshold: float = 0.05
    """若连续 VLM 得分低于此值, 判定为失败并终止"""

    vlm_failure_patience: int = 3
    """连续 N 次 VLM 得分低于 failure_threshold 才终止"""

    reward_on_non_vlm_steps: float = 0.0
    """VLM 未调用时的默认 reward (0.0 = sparse)"""

    # --- 观测空间 ---
    obs_mode: str = "flat"
    """观测模式: 'flat' (兼容 RL pipeline), 'dict' (结构化), 'latent_only'"""

    latent_shape: tuple[int, ...] = (4, 48, 24)
    """VAE latent 形状 (C, H, W)"""

    decode_for_obs: bool = False
    """是否解码 latent 为 RGB (192,192,3) 作为观测 (慢, 通常 False)"""

    # --- 初始帧数据 ---
    initial_frames_h5: str = ""
    """初始帧 HDF5 文件路径 (含 latent + state), 若空则随机噪声初始化"""

    task_instruction: str = "lift the peg upright"
    """任务描述文本 (传给 WM text cond 和 VLM prompt)"""

    task_id: str = "LiftPegUpright-v1"
    """任务 ID"""

    # --- GPU ---
    gpu_id: int = 0
    """目标 GPU"""

    # --- State Predictor ---
    use_state_predictor: bool = True
    """是否使用 State Predictor 预测下一步 agent_state
    (若 False, agent_state 保持不变 — 仅供测试)"""

    state_predictor_ckpt: str = ""
    """State Predictor checkpoint 路径"""

    dynamics_adapter_ckpt: str = ""
    """Dynamics Adapter checkpoint 路径 (若非空则用 adapter 预测 future EE poses)"""

    # --- 调试 ---
    verbose: bool = False
    """是否输出详细日志"""


# ---------------------------------------------------------------------------
# 初始帧加载器
# ---------------------------------------------------------------------------


def load_initial_frames_from_h5(
    h5_path: str,
    max_count: int = 1000,
) -> list[dict[str, np.ndarray]]:
    """从 HDF5 加载初始帧 (latent + state).

    Args:
        h5_path:   HDF5 文件路径
        max_count: 最多加载多少条

    Returns:
        列表, 每项 {"latent": (4,48,24) float32, "state": (D,) float32}
    """
    import h5py

    frames: list[dict[str, np.ndarray]] = []
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
        for key in traj_keys[:max_count]:
            grp = f[key]
            # 取第一帧
            if "latent_concat" in grp:
                lat = grp["latent_concat"][0].astype(np.float32)
            elif "latent" in grp:
                lat = grp["latent"][0].astype(np.float32)
            else:
                lat = np.zeros((4, 48, 24), dtype=np.float32)

            if "state" in grp:
                st = grp["state"][0].astype(np.float32)
            else:
                st = np.zeros(25, dtype=np.float32)

            frames.append({"latent": lat, "state": st})
    return frames


# ---------------------------------------------------------------------------
# ImaginationRLEnv — 核心实现
# ---------------------------------------------------------------------------


class ImaginationRLEnv(gym.Env):
    """gymnasium.Env 封装: Ctrl-World WM + VLM Reward.

    将 World Model 和 VLM 奖励模型封装为标准 RL 环境接口,
    供现有 RLPD/DSRL/PLD 管线直接使用。

    核心 step() 循环:
        1. action → State Predictor → next agent_state
        2. action → WM rollout → next latent frame
        3. 每 vlm_reward_interval 步: decode latents → VLM → p_yes reward
        4. 返回 (obs, reward, terminated, truncated, info)

    Args:
        wm_adapter:      CtrlWorldAdapter 实例 (世界模型推理)
        reward_model:    VLAWRewardModel 实例 (VLM 奖励), 可选
        state_predictor: StatePredictor 实例, 可选
        config:          ImaginationRLEnvConfig
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        wm_adapter: "CtrlWorldAdapter",
        reward_model: Optional["VLAWRewardModel"] = None,
        state_predictor: Optional[Any] = None,
        config: Optional[ImaginationRLEnvConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or ImaginationRLEnvConfig()
        cfg = self.config

        self.wm_adapter = wm_adapter
        self.reward_model = reward_model
        self.state_predictor = state_predictor

        self.device = torch.device(
            f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        # --- 动作空间 ---
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(cfg.action_dim,),
            dtype=np.float32,
        )

        # --- 观测空间 ---
        lat_flat_dim = int(np.prod(cfg.latent_shape))  # 4*48*24 = 4608
        self._lat_flat_dim = lat_flat_dim

        if cfg.obs_mode == "flat":
            obs_dim = lat_flat_dim + cfg.state_dim
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        elif cfg.obs_mode == "dict":
            self.observation_space = spaces.Dict({
                "latent": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=cfg.latent_shape, dtype=np.float32,
                ),
                "agent_state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(cfg.state_dim,), dtype=np.float32,
                ),
            })
        elif cfg.obs_mode == "latent_only":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=cfg.latent_shape, dtype=np.float32,
            )
        else:
            raise ValueError(f"不支持的 obs_mode: {cfg.obs_mode}")

        # --- 初始帧池 ---
        self._initial_frames: list[dict[str, np.ndarray]] = []
        if cfg.initial_frames_h5 and Path(cfg.initial_frames_h5).exists():
            self._initial_frames = load_initial_frames_from_h5(cfg.initial_frames_h5)
            if cfg.verbose:
                print(
                    f"[ImaginationRLEnv] 加载 {len(self._initial_frames)} 个初始帧 "
                    f"from {cfg.initial_frames_h5}"
                )

        # --- 内部状态 (reset 时初始化) ---
        self._step_count: int = 0
        self._current_latent: Optional[torch.Tensor] = None  # (4,48,24)
        self._current_state: Optional[np.ndarray] = None  # (state_dim,)
        # 列表式 latent history (对齐 imagination_env.py 官方做法)
        self._latent_history: list[torch.Tensor] = []
        # action history (用于 WM 历史动作输入)
        self._action_history: list[np.ndarray] = []
        # EE pose history (用于 WM 历史 EE 位姿输入, 对齐 DROID)
        self._ee_pose_history: list[np.ndarray] = []
        # 用于 WM rollout 预测缓冲 (action chunk 内部消费)
        self._pending_latents: list[torch.Tensor] = []
        self._pending_idx: int = 0
        # VLM 相关
        self._vlm_failure_count: int = 0
        self._last_vlm_reward: float = 0.0
        # 收集的帧序列 (用于 VLM 评估)
        self._collected_latents: list[torch.Tensor] = []
        self._episode_done: bool = False

        # --- Dynamics Adapter (optional) ---
        self._dynamics_adapter = None
        self._adapter_norm = None
        if cfg.dynamics_adapter_ckpt:
            from rlft.vlaw.world_model.dynamics_adapter import DynamicsAdapterTrainer
            adapter, norm_dict = DynamicsAdapterTrainer.load_from_checkpoint(
                cfg.dynamics_adapter_ckpt, device=str(self.device)
            )
            self._dynamics_adapter = adapter
            self._adapter_norm = norm_dict
            if cfg.verbose:
                print(f"[ImaginationRLEnv] Dynamics Adapter 已加载: {cfg.dynamics_adapter_ckpt}")

        if cfg.verbose:
            print(
                f"[ImaginationRLEnv] 初始化完成 | "
                f"obs_mode={cfg.obs_mode} | action_dim={cfg.action_dim} | "
                f"max_steps={cfg.max_steps} | vlm_interval={cfg.vlm_reward_interval}"
            )

    # ------------------------------------------------------------------
    # gymnasium.Env 核心接口
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Any, dict]:
        """重置环境, 从初始帧池随机选取起点.

        Args:
            seed:    RNG seed
            options: 可选覆盖, 支持:
                     - "initial_latent": (4,48,24) tensor/ndarray
                     - "initial_state":  (state_dim,) ndarray
                     - "instruction":    str

        Returns:
            (obs, info)
        """
        super().reset(seed=seed)
        cfg = self.config

        # --- 确定初始帧 ---
        initial_latent: Optional[np.ndarray] = None
        initial_state: Optional[np.ndarray] = None

        if options:
            if "initial_latent" in options:
                lat = options["initial_latent"]
                initial_latent = (
                    lat.cpu().numpy() if isinstance(lat, torch.Tensor) else np.asarray(lat)
                ).astype(np.float32)
            if "initial_state" in options:
                initial_state = np.asarray(options["initial_state"], dtype=np.float32)
            if "instruction" in options:
                cfg.task_instruction = options["instruction"]

        if initial_latent is None:
            if self._initial_frames:
                idx = self.np_random.integers(0, len(self._initial_frames))
                frame = self._initial_frames[idx]
                initial_latent = frame["latent"].copy()
                if initial_state is None:
                    initial_state = frame["state"].copy()
            else:
                # 随机噪声初始化 (仅供测试)
                initial_latent = np.random.randn(*cfg.latent_shape).astype(np.float32)

        if initial_state is None:
            initial_state = np.zeros(cfg.state_dim, dtype=np.float32)

        # --- 设置内部状态 ---
        self._step_count = 0
        self._current_latent = torch.from_numpy(initial_latent).to(self.device)
        self._current_state = initial_state.copy()
        self._episode_done = False

        # 初始化 latent history: 用初始帧填充 (对齐 imagination_env.py)
        # 足够多的初始帧以支持稀疏采样 (至少 13 帧)
        init_fill = max(13, cfg.num_history * 4)
        self._latent_history = [
            self._current_latent.clone() for _ in range(init_fill)
        ]
        self._action_history = []
        self._ee_pose_history = [state_to_ee_pose_7d(self._current_state)]

        # 清空 pending latent 缓冲
        self._pending_latents = []
        self._pending_idx = 0

        # VLM 状态
        self._vlm_failure_count = 0
        self._last_vlm_reward = 0.0
        self._collected_latents = [self._current_latent.clone()]

        obs = self._build_obs()
        info = {
            "step": 0,
            "vlm_reward": 0.0,
            "p_yes": 0.0,
            "is_vlm_step": False,
        }

        if cfg.verbose:
            print(f"[ImaginationRLEnv] reset | latent_shape={initial_latent.shape}")

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Any, float, bool, bool, dict]:
        """执行一步 imagination 推理.

        流程:
            1. State Predictor: state_t + action → state_{t+1}
            2. WM rollout: 使用 action chunk 预测下一帧 latent
               (每 wm_act_steps 步调用一次 WM, 中间步从缓冲取)
            3. VLM reward: 每 vlm_reward_interval 步评估一次
            4. 判定 terminated / truncated

        Args:
            action: (action_dim,) ndarray, 范围 [-1, 1]

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        cfg = self.config
        action = np.asarray(action, dtype=np.float32).flatten()[:cfg.action_dim]

        if self._episode_done:
            warnings.warn(
                "ImaginationRLEnv.step() called after episode done. "
                "Call reset() first.",
                stacklevel=2,
            )
            return self._build_obs(), 0.0, True, False, {"step": self._step_count}

        self._step_count += 1

        # ---- Step 1: State Predictor ----
        next_state = self._predict_next_state(self._current_state, action)
        self._current_state = next_state

        # ---- Step 2: WM Rollout (批量, 每 wm_act_steps 步调一次) ----
        next_latent = self._get_next_latent(action)
        self._current_latent = next_latent
        self._latent_history.append(next_latent.clone())
        self._action_history.append(action.copy())
        self._ee_pose_history.append(state_to_ee_pose_7d(self._current_state))
        self._collected_latents.append(next_latent.clone())

        # ---- Step 3: VLM Reward ----
        reward = cfg.reward_on_non_vlm_steps
        p_yes = self._last_vlm_reward
        is_vlm_step = False

        if (
            self.reward_model is not None
            and self._step_count % cfg.vlm_reward_interval == 0
        ):
            is_vlm_step = True
            p_yes = self._compute_vlm_reward()
            self._last_vlm_reward = p_yes

            if cfg.use_continuous_reward:
                reward = p_yes
            else:
                reward = 1.0 if p_yes > cfg.vlm_reward_alpha else 0.0

            # VLM 失败检测
            if p_yes < cfg.vlm_failure_threshold:
                self._vlm_failure_count += 1
            else:
                self._vlm_failure_count = 0

        # ---- Step 4: Termination ----
        terminated = False  # 在 imagination 中无自然终止
        truncated = False

        # VLM 连续失败 → 提前终止
        if self._vlm_failure_count >= cfg.vlm_failure_patience:
            terminated = True

        # 达到 max_steps → truncated
        if self._step_count >= cfg.max_steps:
            truncated = True

        if terminated or truncated:
            self._episode_done = True

        # ---- 构建 obs & info ----
        obs = self._build_obs()
        info = {
            "step": self._step_count,
            "vlm_reward": float(reward),
            "p_yes": float(p_yes),
            "is_vlm_step": is_vlm_step,
            "vlm_failure_count": self._vlm_failure_count,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """渲染当前帧 (decode latent → RGB).

        Returns:
            (H, W, 3) uint8 ndarray, 或 None
        """
        if self._current_latent is None:
            return None

        try:
            lat = self._current_latent.unsqueeze(0).float()  # (1, 4, 48, 24)
            rgb = self.wm_adapter.decode_latents(lat, decode_chunk_size=1)
            # rgb: (1, H, W, 3) uint8
            return rgb[0]
        except Exception:
            return None

    def close(self) -> None:
        """清理资源."""
        self._latent_history.clear()
        self._action_history.clear()
        self._ee_pose_history.clear()
        self._collected_latents.clear()
        self._pending_latents.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_obs(self) -> Any:
        """根据 obs_mode 构建观测."""
        cfg = self.config

        if self._current_latent is None:
            lat_np = np.zeros(cfg.latent_shape, dtype=np.float32)
        else:
            lat_np = self._current_latent.cpu().float().numpy()

        state_np = (
            self._current_state
            if self._current_state is not None
            else np.zeros(cfg.state_dim, dtype=np.float32)
        )

        if cfg.obs_mode == "flat":
            return np.concatenate([lat_np.flatten(), state_np]).astype(np.float32)
        elif cfg.obs_mode == "dict":
            return {
                "latent": lat_np.astype(np.float32),
                "agent_state": state_np.astype(np.float32),
            }
        elif cfg.obs_mode == "latent_only":
            return lat_np.astype(np.float32)
        else:
            return np.concatenate([lat_np.flatten(), state_np]).astype(np.float32)

    def _predict_next_state(
        self, state: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """使用 State Predictor 预测下一步 agent_state.

        若 state_predictor 为 None, 保持 state 不变。
        """
        if not self.config.use_state_predictor or self.state_predictor is None:
            return state.copy()

        try:
            device = next(self.state_predictor.parameters()).device
            state_t = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)  # (1, state_dim)
            action_t = torch.tensor(
                action, dtype=torch.float32, device=device
            ).unsqueeze(0)  # (1, action_dim)

            with torch.no_grad():
                next_state = self.state_predictor(state_t, action_t)  # (1, state_dim)
            return next_state.squeeze(0).cpu().numpy()
        except Exception as e:
            if self.config.verbose:
                print(f"[ImaginationRLEnv] State Predictor 失败: {e}")
            return state.copy()

    def _get_next_latent(self, action: np.ndarray) -> torch.Tensor:
        """获取下一帧 latent (WM rollout 或从缓冲消费).

        WM 每次生成 wm_act_steps 帧, 我们按步消费:
            - 若缓冲有剩余帧 → 直接取
            - 否则调用 WM rollout 生成 wm_act_steps 帧并缓冲
        """
        # 如果有缓冲帧, 消费一帧
        if self._pending_idx < len(self._pending_latents):
            lat = self._pending_latents[self._pending_idx]
            self._pending_idx += 1
            return lat

        # 缓冲耗尽, 需要新的 WM rollout
        return self._wm_rollout_and_buffer(action)

    def _predict_future_ee(
        self, state: np.ndarray, action_chunk: np.ndarray
    ) -> np.ndarray:
        """Predict future EE poses using dynamics adapter or fallback.

        Args:
            state:        (state_dim,) current agent_state (raw, unnormalized).
            action_chunk: (wm_act_steps, action_dim) delta actions from policy.

        Returns:
            (wm_act_steps, 7) future EE poses in world frame.
        """
        if self._dynamics_adapter is not None:
            norm = self._adapter_norm
            # V3: optionally encode euler → sin/cos before normalization
            if norm.get("sincos_input", False):
                from rlft.vlaw.world_model.dynamics_adapter import DynamicsAdapterTrainer
                state_enc = DynamicsAdapterTrainer.encode_state_sincos(state)
            else:
                state_enc = state
            state_n = (state_enc - norm["state_mean"]) / norm["state_std"]
            pred = self._dynamics_adapter.predict(state_n, action_chunk)
            # V3: delta_target → add current_ee back
            if norm.get("delta_target", False):
                current_ee = state_to_ee_pose_7d(state[None, :])[0]  # (7,)
                pred = pred + current_ee[None, :]  # (K, 7)
            return pred
        # Fallback: treat action_chunk as base-frame EE pose
        return ee_pose_base_to_world(action_chunk)

    def _wm_rollout_and_buffer(self, action: np.ndarray) -> torch.Tensor:
        """调用 WM rollout 生成 wm_act_steps 帧并缓冲.

        使用列表式 latent_history + 稀疏采样 history_idx (对齐官方 Ctrl-World)。

        Args:
            action: 当前单步 action (7,)

        Returns:
            本步的 latent (4, 48, 24) tensor
        """
        cfg = self.config

        # 构建 action chunk: 用当前 action 填充 wm_act_steps 步
        # (简化: 假设连续动作一致; 实际 RL agent 只给单步 action)
        action_chunk = np.tile(action[None, :], (cfg.wm_act_steps, 1))  # (T, 7)

        # ---- 稀疏采样历史帧 ----
        total_len = len(self._latent_history)
        hist_indices = get_history_indices(total_len, cfg.history_idx)

        his_latent = torch.stack(
            [self._latent_history[i] for i in hist_indices], dim=0
        )  # (num_history, 4, 48, 24)

        # 当前帧 padding 拼接
        cur_pad = (
            self._latent_history[-1]
            .unsqueeze(0)
            .expand(cfg.wm_act_steps, -1, -1, -1)
            .clone()
        )
        wm_input = torch.cat([his_latent, cur_pad], dim=0)
        # (num_history + wm_act_steps, 4, 48, 24)

        # ---- 历史 EE 位姿: 稀疏采样 + 当前 EE pose (对齐 DROID) ----
        hist_ee_list = []
        for i in hist_indices:
            if i < len(self._ee_pose_history):
                hist_ee_list.append(self._ee_pose_history[i])
            else:
                initial_ee = self._ee_pose_history[0] if self._ee_pose_history else np.zeros(cfg.action_dim, dtype=np.float32)
                hist_ee_list.append(initial_ee)
        hist_ee = np.stack(hist_ee_list, axis=0)  # (num_history, 7)
        # 未来帧 EE pose: 使用 Dynamics Adapter (若有) 或 ee_pose_base_to_world fallback
        future_ee = self._predict_future_ee(self._current_state, action_chunk)  # (wm_act_steps, 7)
        full_ee_poses = np.concatenate([hist_ee, future_ee], axis=0)

        # ---- WM 推理 ----
        try:
            pred_latents = self.wm_adapter.rollout(
                obs_latents=wm_input,
                actions=full_ee_poses,
                instruction=cfg.task_instruction,
            )
            # pred_latents: (N_CAMS, wm_act_steps, 4, lat_h_single, lat_w)

            cam0 = pred_latents[0]  # (T, 4, 24, 24)
            cam1 = pred_latents[1] if pred_latents.shape[0] > 1 else cam0
            # 拼接 2 相机: (T, 4, 48, 24)
            new_latents = torch.cat(
                [cam0.to(self.device), cam1.to(self.device)], dim=2
            )

            # 缓冲 wm_act_steps 帧
            self._pending_latents = [
                new_latents[i] for i in range(new_latents.shape[0])
            ]
            self._pending_idx = 1  # 消费第 0 帧
            return self._pending_latents[0]

        except Exception as e:
            if self.config.verbose:
                print(f"[ImaginationRLEnv] WM rollout 失败: {e}")
            # 失败时返回当前帧 + 小扰动
            return self._current_latent.clone() + 0.01 * torch.randn_like(
                self._current_latent
            )

    def _compute_vlm_reward(self) -> float:
        """调用 VLM 对最近帧序列评分, 返回 p_yes.

        使用最近收集的帧作为输入 (均匀采样到 num_frames):
            - decode latents → RGB images
            - VLM score_trajectory → p_yes
        """
        if self.reward_model is None:
            return 0.0

        try:
            # 取最近的帧 (最多 16 帧供 VLM)
            n_frames = min(len(self._collected_latents), 16)
            if n_frames == 0:
                return 0.0

            # 均匀采样
            total = len(self._collected_latents)
            indices = np.linspace(0, total - 1, n_frames, dtype=int)
            sampled_latents = torch.stack(
                [self._collected_latents[i] for i in indices], dim=0
            )  # (N, 4, 48, 24)

            # decode → RGB
            rgb_frames = self.wm_adapter.decode_latents(
                sampled_latents.float(), decode_chunk_size=4
            )  # (N, H, W, 3) uint8

            # VLM 评分
            result = self.reward_model.score_trajectory(
                rgb_frames, self.config.task_instruction
            )
            return float(result["p_yes"])

        except Exception as e:
            if self.config.verbose:
                print(f"[ImaginationRLEnv] VLM reward 计算失败: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # 便捷属性 & 工具
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        """兼容向量化 env 接口 (单 env = 1)."""
        return 1

    def get_episode_latents(self) -> list[torch.Tensor]:
        """获取当前 episode 收集的全部 latent 帧."""
        return list(self._collected_latents)

    def get_episode_rgb(self) -> Optional[np.ndarray]:
        """将当前 episode 的全部 latent 解码为 RGB 序列.

        Returns:
            (T, H, W, 3) uint8 ndarray, 或 None
        """
        if not self._collected_latents:
            return None
        try:
            all_lats = torch.stack(self._collected_latents, dim=0).float()
            return self.wm_adapter.decode_latents(all_lats, decode_chunk_size=4)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 向量化封装 (多个 ImaginationRLEnv 并行)
# ---------------------------------------------------------------------------


class VecImaginationRLEnv:
    """简易同步向量化: N 个 ImaginationRLEnv 并行.

    兼容 gymnasium VectorEnv API subset (reset, step, close)。
    当 N=1 时, 行为等价于单个 ImaginationRLEnv。

    Note:
        真正的性能敏感场景应考虑 GPU batch rollout,
        本封装仅供 prototype 和小规模实验使用。
    """

    def __init__(
        self,
        envs: list[ImaginationRLEnv],
    ) -> None:
        assert len(envs) > 0
        self.envs = envs
        self.num_envs = len(envs)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.single_observation_space = envs[0].observation_space
        self.single_action_space = envs[0].action_space

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """重置所有环境."""
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed, options=options)
            obs_list.append(obs)
            info_list.append(info)

        if isinstance(obs_list[0], dict):
            # Dict obs: stack each key
            stacked = {}
            for key in obs_list[0]:
                stacked[key] = np.stack([o[key] for o in obs_list])
            return stacked, _stack_infos(info_list)
        else:
            return np.stack(obs_list), _stack_infos(info_list)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """所有环境同步 step.

        Args:
            actions: (num_envs, action_dim)

        Returns:
            (obs, rewards, terminateds, truncateds, infos)
        """
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i, env in enumerate(self.envs):
            act = actions[i] if actions.ndim > 1 else actions
            obs, rew, term, trunc, info = env.step(act)

            # auto-reset
            if term or trunc:
                final_obs = obs
                obs, reset_info = env.reset()
                info["final_observation"] = final_obs
                info.update({f"reset_{k}": v for k, v in reset_info.items()})

            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)

        if isinstance(obs_list[0], dict):
            stacked_obs = {}
            for key in obs_list[0]:
                stacked_obs[key] = np.stack([o[key] for o in obs_list])
            obs_out = stacked_obs
        else:
            obs_out = np.stack(obs_list)

        return (
            obs_out,
            np.array(rew_list, dtype=np.float32),
            np.array(term_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            _stack_infos(info_list),
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


def _stack_infos(infos: list[dict]) -> dict:
    """将 info dict 列表合并."""
    if not infos:
        return {}
    stacked: dict = {}
    for key in infos[0]:
        vals = [info.get(key) for info in infos]
        if all(isinstance(v, (int, float, bool, np.floating, np.integer)) for v in vals):
            stacked[key] = np.array(vals)
        else:
            stacked[key] = vals
    return stacked


# ---------------------------------------------------------------------------
# Mock 对象 (仅供测试, 不加载真实模型)
# ---------------------------------------------------------------------------


class MockCtrlWorldAdapter:
    """Mock WM adapter: 返回随机 latent, 不加载模型."""

    class _MockArgs:
        num_history: int = 6
        num_frames: int = 5
        action_dim: int = 7
        text_cond: bool = False
        width: int = 192
        height: int = 384
        decode_chunk_size: int = 2
        fps: int = 7
        motion_bucket_id: int = 127
        data_stat_path: Optional[str] = None

    def __init__(self) -> None:
        self.args = self._MockArgs()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    @torch.no_grad()
    def rollout(
        self,
        obs_latents: torch.Tensor,
        actions: Union[np.ndarray, torch.Tensor],
        instruction: str = "",
    ) -> torch.Tensor:
        T = self.args.num_frames
        return torch.randn(2, T, 4, 24, 24, dtype=torch.float32)

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.Tensor,
        decode_chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        N = latents.shape[0] if latents.dim() >= 2 else 1
        return np.random.randint(0, 255, (N, 192, 192, 3), dtype=np.uint8)


class MockRewardModel:
    """Mock VLM reward: 返回随机 p_yes, 不加载模型."""

    def __init__(self, p_yes_range: Tuple[float, float] = (0.1, 0.9)) -> None:
        self._low, self._high = p_yes_range

    def score_trajectory(
        self,
        frames: Any,
        instruction: str,
    ) -> dict:
        p_yes = np.random.uniform(self._low, self._high)
        return {
            "p_yes": float(p_yes),
            "reward": int(p_yes > 0.8),
            "threshold": 0.8,
            "num_frames": 16,
        }

    def load_model(self, lora_path: Optional[str] = None) -> None:
        pass

    @property
    def _loaded(self) -> bool:
        return True


class MockStatePredictor(torch.nn.Module):
    """Mock State Predictor: 小随机扰动."""

    def __init__(self, state_dim: int = 25, action_dim: int = 7) -> None:
        super().__init__()
        self.state_dim = state_dim
        # 需要至少一个参数让 next(self.parameters()) 有效
        self._dummy = torch.nn.Linear(1, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # 小扰动 (模拟真实 predictor)
        noise = torch.randn_like(state) * 0.01
        return state + noise


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def make_imagination_rl_env(
    config: ImaginationRLEnvConfig,
    wm_adapter: Optional["CtrlWorldAdapter"] = None,
    reward_model: Optional["VLAWRewardModel"] = None,
    state_predictor: Optional[Any] = None,
    use_mock: bool = False,
) -> ImaginationRLEnv:
    """工厂函数: 创建 ImaginationRLEnv.

    Args:
        config:          环境配置
        wm_adapter:      CtrlWorldAdapter (若 use_mock=True 则忽略)
        reward_model:    VLAWRewardModel (可选)
        state_predictor: StatePredictor (可选)
        use_mock:        True 则使用 mock 组件 (测试用)

    Returns:
        ImaginationRLEnv 实例
    """
    if use_mock:
        wm_adapter = MockCtrlWorldAdapter()  # type: ignore[assignment]
        reward_model = MockRewardModel()  # type: ignore[assignment]
        state_predictor = MockStatePredictor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
        )

    if wm_adapter is None:
        raise ValueError(
            "wm_adapter 不能为 None (除非 use_mock=True)"
        )

    return ImaginationRLEnv(
        wm_adapter=wm_adapter,
        reward_model=reward_model,
        state_predictor=state_predictor,
        config=config,
    )


def make_vec_imagination_rl_env(
    config: ImaginationRLEnvConfig,
    num_envs: int = 4,
    wm_adapter: Optional["CtrlWorldAdapter"] = None,
    reward_model: Optional["VLAWRewardModel"] = None,
    state_predictor: Optional[Any] = None,
    use_mock: bool = False,
) -> VecImaginationRLEnv:
    """工厂函数: 创建向量化 ImaginationRLEnv.

    注意: 所有 env 共享同一个 wm_adapter 和 reward_model 实例
    (顺序调用, 非并行)。

    Args:
        config:     环境配置
        num_envs:   并行 env 数量
        use_mock:   True 则使用 mock 组件

    Returns:
        VecImaginationRLEnv 实例
    """
    if use_mock:
        wm_adapter = MockCtrlWorldAdapter()  # type: ignore[assignment]
        reward_model = MockRewardModel()  # type: ignore[assignment]
        state_predictor = MockStatePredictor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
        )

    envs = [
        ImaginationRLEnv(
            wm_adapter=wm_adapter,
            reward_model=reward_model,
            state_predictor=state_predictor,
            config=config,
        )
        for _ in range(num_envs)
    ]
    return VecImaginationRLEnv(envs)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # 支持从项目根目录直接运行: python rlft/vlaw/world_model/imagination_rl_env.py
    _root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    if _root not in sys.path:
        sys.path.insert(0, _root)

    # Re-import after path fix (needed when running as script)
    from rlft.vlaw.world_model.imagination_rl_env import (  # noqa: F811
        ImaginationRLEnvConfig,
        make_imagination_rl_env,
        make_vec_imagination_rl_env,
    )

    print("=" * 60)
    print("ImaginationRLEnv — Smoke Test (Mock)")
    print("=" * 60)

    cfg = ImaginationRLEnvConfig(
        max_steps=20,
        vlm_reward_interval=5,
        verbose=True,
    )

    env = make_imagination_rl_env(cfg, use_mock=True)

    # --- Test reset ---
    obs, info = env.reset(seed=42)
    print(f"\n[reset] obs shape: {obs.shape}, info: {info}")
    assert obs.shape == env.observation_space.shape, (
        f"obs shape mismatch: {obs.shape} vs {env.observation_space.shape}"
    )

    # --- Test step loop ---
    total_reward = 0.0
    for step in range(cfg.max_steps + 5):  # 超出 max_steps 测试截断
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step < 3 or step % 10 == 0 or terminated or truncated:
            print(
                f"  step {step+1}: reward={reward:.3f}, "
                f"term={terminated}, trunc={truncated}, "
                f"p_yes={info.get('p_yes', 0):.3f}"
            )

        if terminated or truncated:
            print(f"  Episode ended at step {step+1}, total_reward={total_reward:.3f}")
            break

    # --- Test render ---
    rgb = env.render()
    print(f"\n[render] shape: {rgb.shape if rgb is not None else None}")

    # --- Test Vec env ---
    print("\n--- VecImaginationRLEnv ---")
    vec_env = make_vec_imagination_rl_env(cfg, num_envs=3, use_mock=True)
    obs, info = vec_env.reset(seed=0)
    print(f"[vec reset] obs shape: {obs.shape}, num_envs: {vec_env.num_envs}")

    actions = np.stack([vec_env.action_space.sample() for _ in range(3)])
    obs, rewards, terms, truncs, info = vec_env.step(actions)
    print(f"[vec step] rewards: {rewards}, terms: {terms}, truncs: {truncs}")

    vec_env.close()
    env.close()

    print("\n✅ ImaginationRLEnv smoke test PASSED")
