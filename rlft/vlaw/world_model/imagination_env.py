"""VLAW P4.3 — Imagination Engine（env.step() 版）

本模块是 P4.2 `imagination.py`（State Predictor MLP 版）的最终替代方案。

## 与 imagination.py（MLP 版）的唯一区别：Step 5

```
MLP 版 (P4.2):
    state_seq = state_predictor.predict_sequence(state_0, actions)
    state_t = state_seq[-1]

Env 版 (P4.3, 本模块):
    for each action in action_chunk:
        obs, reward, terminated, truncated, info = env.step(action)
        state_t = extract_agent_state(obs)[env_idx]
```

## 关于 initial_env_state 参数

`rollout_single()` 接受可选的 `initial_env_state: dict | None`：
- **若提供** (`initial_env_state is not None`)：
    调用 `env.set_state(initial_env_state)` 将仿真环境精确还原到真实轨迹的起始状态，
    使 initial_latent 与 env 物理状态严格对齐。
    适用于 data_collector 保存了 `env_state_dict` 的场景（未来扩展）。
- **若为 None（默认）**：
    调用 `env.reset()` 随机初始化，initial_latent 来自真实轨迹第一帧，
    视觉-状态对齐是近似的，但足以驱动世界模型推理。
    这是当前默认路径（data_collector 尚未保存 env_state_dict）。

## 并行效率估算

设单步 env.step() ≈ 0.5ms（GPU 向量化），每条轨迹 K×act_steps = 60 步：
- num_envs=1:  ~30ms/条
- num_envs=16: ~30ms / 16 ≈ 2ms（有效吞吐）
- num_envs=64: ~0.5ms（有效吞吐，推荐生产配置）

实际瓶颈在世界模型推理（SVD UNet），env.step() 本身可忽略不计。

所属阶段: P4.3 — env.step() 版 Imagination Engine（最终方案）
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import h5py
import numpy as np
import torch

if TYPE_CHECKING:
    from rlft.vlaw.world_model.ctrl_world_adapter import CtrlWorldAdapter

# 复用 data_collector 中的状态提取工具
from rlft.vlaw.data.collector import extract_agent_state

# EE pose 提取 (WM action conditioning 对齐 DROID)
from ctrl_world.dataset.dataset_maniskill import state_to_ee_pose_7d

from scipy.spatial.transform import Rotation as Rot

# 复用 imagination.py 中的数据容器和工具函数
from rlft.vlaw.utils.imagination import SyntheticTrajectory, _load_initial_frames


# ---------------------------------------------------------------------------
# History 索引工具函数（对齐官方 Ctrl-World 稀疏采样方式）
# ---------------------------------------------------------------------------


def get_history_indices(total_len: int, history_idx: list[int] | None = None) -> list[int]:
    """返回用于构建 WM 历史输入的稀疏采样帧索引 (绝对索引).

    对齐官方 Ctrl-World 做法 (DROID 数据集):
        history_idx = [0, 0, -12, -9, -6, -3]
        含义: 前 2 个位置始终 = 初始帧 (锚定)，
              后 4 个 = 相对于当前帧末尾的偏移，
              负索引越界时 clamp 到 0（退化为初始帧）。

    Args:
        total_len:   latent_history 列表当前长度
        history_idx: 稀疏采样索引模板 (默认官方 DROID 配置)

    Returns:
        长度 == len(history_idx) 的绝对索引列表，每个值 ∈ [0, total_len)
    """
    if history_idx is None:
        history_idx = [0, 0, -12, -9, -6, -3]
    indices: list[int] = []
    for idx in history_idx:
        if idx >= 0:
            # 非负索引：直接使用 (通常为 0 = 初始帧锚定)
            abs_idx = min(idx, total_len - 1)
        else:
            # 负索引：相对于末尾的偏移，clamp 到 0
            abs_idx = max(0, total_len + idx)
        indices.append(abs_idx)
    return indices


def integrate_delta_to_ee_poses(
    current_ee: np.ndarray,
    action_chunk: np.ndarray,
) -> np.ndarray:
    """[DEPRECATED] Integrate policy delta actions into absolute EE pose sequence.

    This function was used with ``pd_ee_delta_pose`` control mode but is now
    superseded by ``ee_pose_base_to_world`` when using ``pd_ee_pose``.
    Kept for backward compatibility / diagnostic scripts.
    """
    T = action_chunk.shape[0]
    future_ee = np.zeros((T, 7), dtype=np.float32)
    ee = current_ee.copy().astype(np.float64)

    for t in range(T):
        delta = action_chunk[t].astype(np.float64)
        ee[:3] = ee[:3] + delta[:3]
        cur_rot = Rot.from_euler("xyz", ee[3:6])
        delta_rot = Rot.from_euler("xyz", delta[3:6])
        ee[3:6] = (delta_rot * cur_rot).as_euler("xyz")
        ee[6] = np.clip(delta[6], 0.0, 1.0)
        future_ee[t] = ee.astype(np.float32)

    return future_ee


# Panda robot root position in world frame (LiftPegUpright-v1, fixed base).
# world_pos = base_pos + ROBOT_ROOT_POS.  Root rotation is identity.
ROBOT_ROOT_POS = np.array([-0.615, 0.0, 0.0], dtype=np.float64)


def ee_pose_base_to_world(
    ee_pose_base: np.ndarray,
    root_pos: np.ndarray = ROBOT_ROOT_POS,
) -> np.ndarray:
    """Convert ``pd_ee_pose`` actions (robot base frame) to world frame for WM.

    With ``pd_ee_pose`` control mode the policy outputs absolute EE target
    poses in the robot base frame.  The WM was trained on world-frame EE
    poses (extracted from ``state[18:21]``).  Since the Panda root has
    identity rotation, the conversion is a simple position offset.

    Args:
        ee_pose_base: *(T, 7)* or *(7,)* — ``[x, y, z, euler_rx, euler_ry,
            euler_rz, gripper]`` in robot base frame (denormalized).
        root_pos: *(3,)* — robot root position in world frame.

    Returns:
        Same shape, with position shifted to world frame and Euler angles
        re-wrapped to [-pi, pi] (the range the WM was trained on).
    """
    squeeze = ee_pose_base.ndim == 1
    if squeeze:
        ee_pose_base = ee_pose_base[None, :]
    result = ee_pose_base.copy().astype(np.float64)
    result[:, :3] += root_pos[None, :]
    # Re-wrap Euler angles (dims 3:6) from policy's [0, 2*pi] back to
    # WM's expected [-pi, pi].  The policy trains on unwrapped euler_rx
    # to avoid bimodal discontinuity, but the WM uses standard [-pi, pi].
    for d in range(3, 6):
        result[:, d] = (result[:, d] + np.pi) % (2 * np.pi) - np.pi
    result = result.astype(np.float32)
    return result[0] if squeeze else result


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class ImaginationEnvConfig:
    """P4.3 ImaginationEnvEngine 配置.

    接口兼容 ImaginationConfig，新增 ManiSkill env 相关字段。
    """

    # --- 共享自 ImaginationConfig ---
    num_interact: int = 12
    """闭环交互轮数 (K_interact)"""

    act_steps: int = 5
    """每次策略调用在世界模型中执行的步数 (= world model num_frames)"""

    obs_horizon: int = 2
    """策略观测历史长度"""

    decode_for_policy: bool = True
    """是否将 latent 解码为 RGB 供策略视觉编码器使用"""

    save_decoded_rgb: bool = False
    """是否保存解码的 RGB 帧（调试用）"""

    output_dir: str = "data/vlaw/synthetic/iter1"
    """合成轨迹输出目录"""

    gpu_id: int = 0
    """使用的 GPU id"""

    batch_size: int = 16
    """并行 imagination 数量（对应 num_envs）"""

    tasks: list = field(
        default_factory=lambda: ["LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"]
    )
    """目标任务列表"""

    num_rollouts_per_task: int = 500
    """每个任务生成的合成轨迹数量"""

    initial_frames_source: str = "data/vlaw/rollouts/iter1"
    """初始帧来源目录"""

    dry_run: bool = False
    """dry_run=True 时只执行 2 轮 × 1 条轨迹，不保存，用于验证流程"""

    # --- P4.3 新增：ManiSkill env 相关 ---
    num_envs: int = 16
    """并行 ManiSkill env 数量（替代 state_predictor，获取精确状态）。
    batch_size 会自动对齐到此值。"""

    task_id: str = "LiftPegUpright-v1"
    """当前运行的任务 ID（run() 中会被逐任务覆盖）"""

    obs_mode: str = "rgbd"
    """ManiSkill obs 模式"""

    sim_backend: str = "physx_cuda"
    """仿真后端: physx_cuda (GPU) 或 physx_cpu"""

    control_mode: str = "pd_ee_delta_pose"
    """控制模式"""

    camera_width: int = 192
    """相机宽度"""

    camera_height: int = 192
    """相机高度"""

    max_episode_steps: int = 200
    """每幕最大步数"""

    dynamics_adapter_ckpt: str = ""
    """Dynamics Adapter checkpoint 路径 (若非空则用 adapter 预测 future EE poses)"""


# ---------------------------------------------------------------------------
# 核心引擎
# ---------------------------------------------------------------------------


class ImaginationEnvEngine:
    """Policy-in-the-Loop Imagination 引擎（VLAW P4.3 最终版）.

    与 P4.2 ImaginationEngine 接口完全兼容，唯一区别：
        Step 5 从 ``state_predictor.predict_sequence()`` 改为 ``env.step()``，
        利用 ManiSkill 仿真精确计算 state_{t+1}。

    本类 **不** 接受 ``state_predictor`` 参数，改为在内部管理 ManiSkill env。

    Args:
        wm_adapter:  CtrlWorldAdapter 实例
        policy:      符合 PolicyProtocol 的策略对象
        config:      ImaginationEnvConfig
    """

    def __init__(
        self,
        wm_adapter: "CtrlWorldAdapter",
        policy: Any,
        config: ImaginationEnvConfig,
    ) -> None:
        self.wm_adapter = wm_adapter
        self.policy = policy
        self.config = config
        self.device = torch.device(
            f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        # 缓存 env 实例，避免 SAPIEN Vulkan 资源泄漏（反复创建/销毁会耗尽 GPU 渲染资源）
        self._env_cache: dict[str, Any] = {}  # key: f"{task_id}_{num_envs}"

        # --- Dynamics Adapter (optional) ---
        self._dynamics_adapter = None
        self._adapter_norm = None
        if config.dynamics_adapter_ckpt:
            from rlft.vlaw.world_model.dynamics_adapter import DynamicsAdapterTrainer
            adapter, norm_dict = DynamicsAdapterTrainer.load_from_checkpoint(
                config.dynamics_adapter_ckpt, device=str(self.device)
            )
            self._dynamics_adapter = adapter
            self._adapter_norm = norm_dict
            print(f"[VLAW-P4.3] Dynamics Adapter 已加载: {config.dynamics_adapter_ckpt}")

        print(
            f"[VLAW-P4.3] ImaginationEnvEngine 初始化完成, "
            f"device={self.device}, num_envs={config.num_envs}"
        )

    # ------------------------------------------------------------------
    # Dynamics Adapter helper
    # ------------------------------------------------------------------

    def _predict_future_ee(
        self, state: np.ndarray, action_chunk: np.ndarray
    ) -> np.ndarray:
        """Predict future EE poses using dynamics adapter or fallback.

        Args:
            state:        (state_dim,) current agent_state (raw, unnormalized).
            action_chunk: (act_steps, action_dim) delta actions from policy.

        Returns:
            (act_steps, 7) future EE poses in world frame.
        """
        if self._dynamics_adapter is not None:
            # Normalize state before feeding to adapter
            norm = self._adapter_norm
            state_n = (state - norm["state_mean"]) / norm["state_std"]
            return self._dynamics_adapter.predict(state_n, action_chunk)
        # Fallback: treat action_chunk as base-frame EE pose
        return ee_pose_base_to_world(action_chunk)

    # ------------------------------------------------------------------
    # 环境创建
    # ------------------------------------------------------------------

    def _make_env(self, task_id: str, num_envs: int = 1):
        """创建 ManiSkill 向量化环境（内部方法）.

        Args:
            task_id:  任务 ID（e.g. 'LiftPegUpright-v1'）
            num_envs: 并行 env 数量

        Returns:
            gymnasium env 实例
        """
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        cfg = self.config
        env_kwargs: dict = dict(
            obs_mode=cfg.obs_mode,
            render_mode="rgb_array",
            control_mode=cfg.control_mode,
            sensor_configs=dict(
                width=cfg.camera_width,
                height=cfg.camera_height,
            ),
            max_episode_steps=cfg.max_episode_steps,
        )

        if num_envs == 1 or cfg.sim_backend == "physx_cpu":
            env = gym.make(task_id, **env_kwargs)
        else:
            env = gym.make(
                task_id,
                num_envs=num_envs,
                sim_backend=cfg.sim_backend,
                **env_kwargs,
            )

        return env

    def _get_or_create_env(self, task_id: str, num_envs: int = 1):
        """获取缓存的 env 或创建新的（解决 SAPIEN Vulkan 资源泄漏问题）.

        SAPIEN 的 RenderSystem 在反复创建/销毁后会耗尽 GPU 渲染资源，
        导致 'Failed to find a supported physical device' 错误。
        通过缓存并复用同一 env 实例来避免此问题。
        """
        cache_key = f"{task_id}_{num_envs}"
        if cache_key not in self._env_cache:
            env = self._make_env(task_id, num_envs=num_envs)
            self._env_cache[cache_key] = env
            print(f"[VLAW-P4.3] 创建并缓存 env: {cache_key}")
        return self._env_cache[cache_key]

    def close(self) -> None:
        """关闭所有缓存的 env 实例."""
        for key, env in self._env_cache.items():
            try:
                env.close()
                print(f"[VLAW-P4.3] 关闭 env: {key}")
            except Exception as e:
                print(f"[VLAW-P4.3] ⚠️  关闭 env {key} 失败: {e}")
        self._env_cache.clear()

    # ------------------------------------------------------------------
    # 单条轨迹生成（公开接口）
    # ------------------------------------------------------------------

    def rollout_single(
        self,
        initial_latent: torch.Tensor,
        initial_state: np.ndarray,
        instruction: str,
        task_id: str,
        initial_env_state: Optional[dict] = None,
    ) -> Optional[SyntheticTrajectory]:
        """生成单条合成轨迹.

        接口与 ImaginationEngine.rollout_single() 完全兼容，
        新增可选参数 ``initial_env_state`` 用于精确 env 初始化。

        Args:
            initial_latent:     (4, 48, 24) float32 Tensor — 初始帧 VAE latent
            initial_state:      (state_dim,) float32 array — 初始 agent_state（用于策略初始化）
            instruction:        任务文本描述
            task_id:            任务 ID
            initial_env_state:  可选，env 物理状态字典；None 则随机 reset

        Returns:
            SyntheticTrajectory，或失败时返回 None
        """
        try:
            return self._rollout_single_impl(
                initial_latent=initial_latent,
                initial_state=initial_state,
                instruction=instruction,
                task_id=task_id,
                initial_env_state=initial_env_state,
            )
        except Exception:
            print(
                f"[VLAW-P4.3] ⚠️  rollout_single 失败 (task={task_id}):\n"
                f"{traceback.format_exc()}"
            )
            return None

    def _rollout_single_impl(
        self,
        initial_latent: torch.Tensor,
        initial_state: np.ndarray,
        instruction: str,
        task_id: str,
        initial_env_state: Optional[dict] = None,
    ) -> SyntheticTrajectory:
        """单条轨迹生成核心实现（复用缓存 env，仅 reset）."""
        cfg = self.config
        num_interact = cfg.num_interact
        act_steps = cfg.act_steps
        obs_horizon = cfg.obs_horizon

        # ---- 复用缓存的 env（避免 SAPIEN Vulkan 资源泄漏）----
        env = self._get_or_create_env(task_id, num_envs=1)
        return self._run_rollout_in_env(
            env=env,
            env_idx=0,
            initial_latent=initial_latent,
            initial_state=initial_state,
            instruction=instruction,
            task_id=task_id,
            initial_env_state=initial_env_state,
            num_interact=num_interact,
            act_steps=act_steps,
            obs_horizon=obs_horizon,
        )

    # ------------------------------------------------------------------
    # 核心 rollout 实现（给定已创建的 env）
    # ------------------------------------------------------------------

    def _run_rollout_in_env(
        self,
        env: Any,
        env_idx: int,
        initial_latent: torch.Tensor,
        initial_state: np.ndarray,
        instruction: str,
        task_id: str,
        initial_env_state: Optional[dict],
        num_interact: int,
        act_steps: int,
        obs_horizon: int,
    ) -> SyntheticTrajectory:
        """在给定 env 中执行单条轨迹生成（Step 5 使用 env.step()）.

        设计参考 ImaginationEngine._rollout_single_impl()，
        仅 Step 5 不同：
          原 MLP 版:  state_seq = state_predictor.predict_sequence(state_0, actions)
          Env 版:     obs, *_ = env.step(action); state_t = extract_agent_state(obs)[env_idx]
        """
        wm_args = self.wm_adapter.args
        num_history = getattr(wm_args, "num_history", obs_horizon)

        lat_h, lat_w = 48, 24
        lat_ch = 4

        # ---- Step 0: 重置 env 到初始状态 ----
        if initial_env_state is not None:
            # 精确还原：将仿真状态设为真实轨迹第一帧物理状态
            try:
                obs, _ = env.reset(options={"env_states": initial_env_state})
                print(f"[VLAW-P4.3] env.reset with initial_env_state (task={task_id})")
            except Exception:
                # 部分版本 ManiSkill API 可能是 set_state
                try:
                    obs, _ = env.reset()
                    env.set_state(initial_env_state)
                    obs = env.get_obs()
                except Exception:
                    obs, _ = env.reset()
                    print(
                        f"[VLAW-P4.3] ⚠️  initial_env_state 设置失败，使用随机 reset"
                    )
        else:
            # 随机 reset（默认路径）：initial_latent 来自真实数据，视觉-物理对齐近似
            obs, _ = env.reset()

        # ---- 获取初始状态（覆盖参数中的 initial_state）----
        # env.reset() 后获取精确状态，比 HDF5 缓存的更可靠
        try:
            state_t = extract_agent_state(obs)[env_idx]  # (state_dim,)
        except Exception:
            state_t = initial_state.copy()

        # ---- 初始化 latent history (列表式, 对齐官方 Ctrl-World) ----
        initial_latent = initial_latent.to(self.device)
        # 列表式 buffer: 所有生成帧按时序追加, idx=0 永远是真实初始帧
        latent_history: list[torch.Tensor] = []
        # 用真实首帧填充 num_history*4 个位置 (官方做法: his_cond 初始填充)
        for _ in range(num_history * 4):
            latent_history.append(initial_latent.clone())
        # 同时维护 ee_pose_history 用于 WM 输入中的历史 EE 位姿
        # (对齐 DROID: WM conditioning 使用绝对 EE 位姿而非 delta action)
        ee_pose_history: list[np.ndarray] = []
        # 用初始状态的 EE pose 填充 (对齐 latent_history: num_history * 4 条)
        initial_ee_pose = state_to_ee_pose_7d(state_t)  # (7,)
        for _ in range(num_history * 4):
            ee_pose_history.append(initial_ee_pose.copy())

        # ---- obs 历史 ----
        obs_feat_dim = lat_ch * lat_h * lat_w  # 4608
        obs_history: list[np.ndarray] = []
        for _ in range(obs_horizon):
            obs_history.append(initial_latent.cpu().float().numpy().flatten())

        # ---- 收集列表 ----
        all_latents: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_states: list[np.ndarray] = []
        env_success = False

        # Reset policy obs history for new trajectory
        if hasattr(self.policy, 'reset_history'):
            self.policy.reset_history()

        for k in range(num_interact):
            if env_success:
                break

            # ---- Step 1-2: 构建策略输入 ----
            # Always use latent features for obs_history (consistent shape).
            # decoded_rgb is passed separately to policy via kwargs.
            vis_feat = latent_history[-1].cpu().float().numpy().flatten()  # (4608,)

            decoded_rgb = None
            if self.config.decode_for_policy:
                try:
                    cur_lat = latent_history[-1].unsqueeze(0)  # (1, 4, 48, 24)
                    decoded_rgb = self.wm_adapter.decode_latents(
                        cur_lat.float(), decode_chunk_size=1
                    )  # (1, H, W, 3) uint8
                except Exception:
                    decoded_rgb = None

            obs_history.append(vis_feat)
            if len(obs_history) > obs_horizon:
                obs_history.pop(0)

            obs_np = np.stack(obs_history, axis=0).flatten()
            obs_tensor = (
                torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
            )  # (1, obs_horizon * feat_dim)

            # ---- Step 3: 策略推理 → action_chunk ----
            try:
                raw_actions = self.policy.get_actions(
                    obs_tensor, decoded_rgb=decoded_rgb, agent_state=state_t,
                )
            except TypeError:
                # Policy doesn't support extra kwargs (e.g. old interface)
                try:
                    raw_actions = self.policy.get_actions(obs_tensor)
                except Exception:
                    raw_actions = np.zeros((1, 7), dtype=np.float32)
            except Exception:
                raw_actions = np.zeros((1, 7), dtype=np.float32)

            # Build action_chunk: support both (act_steps, 7) and (1, 7) returns
            if raw_actions.ndim == 2 and raw_actions.shape[0] >= act_steps:
                action_chunk = raw_actions[:act_steps]  # (act_steps, 7)
            elif raw_actions.ndim == 2 and raw_actions.shape[0] == 1:
                action_chunk = np.tile(raw_actions, (act_steps, 1))
            elif raw_actions.ndim == 1:
                action_chunk = np.tile(raw_actions[None, :], (act_steps, 1))
            else:
                action_chunk = np.zeros((act_steps, 7), dtype=np.float32)

            # ---- Step 4: 世界模型 rollout ----
            # 稀疏采样历史帧 (对齐官方 history_idx = [0, 0, -12, -9, -6, -3])
            total_len = len(latent_history)
            sparse_offsets = [-12, -9, -6, -3]
            hist_indices = [0, 0]  # 前2个: 第一帧锚定 (真实 VAE 编码)
            for off in sparse_offsets:
                idx = max(0, total_len + off)
                hist_indices.append(idx)
            his_latent = torch.stack(
                [latent_history[i] for i in hist_indices], dim=0
            )  # (num_history, 4, 48, 24)
            # 拼接 history + 当前帧 padding 作为 WM 输入
            cur_pad = latent_history[-1].unsqueeze(0).expand(act_steps, -1, -1, -1).clone()
            wm_input = torch.cat([his_latent, cur_pad], dim=0)
            # (num_history + act_steps, 4, 48, 24)

            # 历史 EE 位姿: 稀疏采样 + 当前帧 EE pose (对齐 DROID)
            # history 部分用 ee_pose_history 中对应索引的 EE pose
            ee_pose_dim = 7
            hist_ee_list = []
            for i in hist_indices:
                if i < len(ee_pose_history):
                    hist_ee_list.append(ee_pose_history[i])
                else:
                    hist_ee_list.append(initial_ee_pose.copy())
            hist_ee = np.stack(hist_ee_list, axis=0)  # (num_history, 7)

            # 未来帧: 使用 Dynamics Adapter (若有) 或 ee_pose_base_to_world fallback
            future_ee = self._predict_future_ee(state_t, action_chunk)  # (act_steps, 7)
            full_ee_poses = np.concatenate([hist_ee, future_ee], axis=0)

            pred_latents = self.wm_adapter.rollout(
                obs_latents=wm_input,
                actions=full_ee_poses,
                instruction=instruction,
            )
            # pred_latents: (N_CAMS, act_steps, 4, lat_h_single, lat_w)

            cam0 = pred_latents[0]  # (T, 4, lat_h_single, lat_w)
            cam1 = pred_latents[1] if pred_latents.shape[0] > 1 else cam0
            new_latents = torch.cat(
                [cam0.to(self.device), cam1.to(self.device)], dim=2
            )  # (T, 4, 48, 24) — 确保与 lat_buf 同设备

            # ========================================================
            # Step 5 (P4.3 核心): 用 env.step() 替代 State Predictor
            # ========================================================
            state_seq: list[np.ndarray] = []
            for step_i in range(act_steps):
                act_i = action_chunk[step_i]  # (action_dim,)
                # 向量化 env 需要 (num_envs, action_dim) 形状
                act_input = act_i[None, :]  # (1, action_dim)
                try:
                    step_obs, _reward, terminated, truncated, _info = env.step(act_input)
                    # 提取本体状态
                    cur_state = extract_agent_state(step_obs)[env_idx]  # (state_dim,)
                    # 检查是否提前完成
                    term_val = terminated
                    if isinstance(term_val, (torch.Tensor, np.ndarray)):
                        term_val = bool(
                            term_val.cpu().item()
                            if isinstance(term_val, torch.Tensor)
                            else term_val.item()
                        )
                    if term_val:
                        env_success = True
                        state_t = cur_state
                    else:
                        state_t = cur_state
                except Exception as e:
                    # env.step() 异常时保持前一状态不变
                    print(f"[VLAW-P4.3] ⚠️  env.step() step_i={step_i} 失败: {e}")
                    cur_state = state_t.copy()
                state_seq.append(cur_state)

            # ---- Step 6: 更新 history buffer (列表式追加) ----
            # 将最后一帧预测追加到 latent_history (官方: his_cond.append(...))
            latent_history.append(new_latents[-1].clone())
            # 追加当前 EE pose 到 ee_pose_history (用 env.step() 后的最新状态)
            ee_pose_history.append(state_to_ee_pose_7d(state_t))

            # ---- 收集 ----
            for step_i in range(act_steps):
                all_latents.append(
                    new_latents[step_i].cpu().float().numpy().astype(np.float16)
                )
                all_actions.append(action_chunk[step_i])
                all_states.append(state_seq[step_i])

        # ---- 组装 SyntheticTrajectory ----
        traj = SyntheticTrajectory(
            latents=np.stack(all_latents, axis=0),   # (T, 4, 48, 24) float16
            actions=np.stack(all_actions, axis=0),   # (T, 7) float32
            states=np.stack(all_states, axis=0),     # (T, state_dim) float32
            instruction=instruction,
            task_id=task_id,
        )
        return traj

    # ------------------------------------------------------------------
    # 批量生成（利用 num_envs 并行）
    # ------------------------------------------------------------------

    def rollout_batch(
        self,
        initial_latents: list[torch.Tensor],
        initial_states: list[np.ndarray],
        instructions: list[str],
        task_ids: list[str],
        initial_env_states: Optional[list[Optional[dict]]] = None,
    ) -> List[SyntheticTrajectory]:
        """批量生成合成轨迹（利用 num_envs 并行化 env.step()）.

        使用单个 num_envs 并行 env 池，所有样本必须属于同一 task_id。
        若 task_ids 包含多个不同任务，自动按任务分组。

        接口兼容 ImaginationEngine.rollout_batch()（新增可选 initial_env_states）。

        Args:
            initial_latents:    长度 B 的 list，每项 (4, 48, 24)
            initial_states:     长度 B 的 list，每项 (state_dim,)
            instructions:       长度 B 的任务描述列表
            task_ids:           长度 B 的任务 ID 列表
            initial_env_states: 可选，长度 B 的 env 状态字典列表；None 则全部随机 reset

        Returns:
            生成成功的 SyntheticTrajectory 列表（失败项已剔除）
        """
        if initial_env_states is None:
            initial_env_states = [None] * len(initial_latents)

        # 按任务分组（不同任务需要不同 env）
        from collections import defaultdict

        task_groups: dict[str, list[int]] = defaultdict(list)
        for i, tid in enumerate(task_ids):
            task_groups[tid].append(i)

        results: List[SyntheticTrajectory] = []

        for tid, indices in task_groups.items():
            batch_num_envs = min(self.config.num_envs, len(indices))
            print(
                f"[VLAW-P4.3] 批量 rollout: task={tid}, "
                f"n={len(indices)}, num_envs={batch_num_envs}"
            )

            # 复用缓存 env（关键修复：避免 SAPIEN Vulkan 资源泄漏）
            try:
                env = self._get_or_create_env(tid, num_envs=batch_num_envs)
            except Exception as e:
                print(f"[VLAW-P4.3] ⚠️  创建 env 失败 ({tid}): {e}，退回单条模式")
                env = None

            if env is not None:
                try:
                    batch_trajs = self._batch_rollout_in_env(
                        env=env,
                        indices=indices,
                        initial_latents=initial_latents,
                        initial_states=initial_states,
                        instructions=instructions,
                        initial_env_states=initial_env_states,
                        task_id=tid,
                        num_envs=batch_num_envs,
                    )
                    results.extend(batch_trajs)
                except Exception as e:
                    print(
                        f"[VLAW-P4.3] ⚠️  batch_rollout 失败 ({tid}): {e}，"
                        f"退回逐条单 env 模式"
                    )
                    # 从缓存中移除故障 env 并关闭
                    cache_key = f"{tid}_{batch_num_envs}"
                    if cache_key in self._env_cache:
                        try:
                            self._env_cache[cache_key].close()
                        except Exception:
                            pass
                        del self._env_cache[cache_key]
                    env = None
                # 注意：成功时不 close env，保留在缓存中供后续批次复用

            if env is None:
                # 退回逐条 rollout_single
                for i in indices:
                    traj = self.rollout_single(
                        initial_latents[i],
                        initial_states[i],
                        instructions[i],
                        task_ids[i],
                        initial_env_states[i],
                    )
                    if traj is not None:
                        results.append(traj)

        return results

    def _batch_rollout_in_env(
        self,
        env: Any,
        indices: list[int],
        initial_latents: list[torch.Tensor],
        initial_states: list[np.ndarray],
        instructions: list[str],
        initial_env_states: list[Optional[dict]],
        task_id: str,
        num_envs: int,
    ) -> List[SyntheticTrajectory]:
        """在 num_envs 并行 env 中批量顺序推进，逐组处理 ceil(len/num_envs) 批次."""
        cfg = self.config
        num_interact = cfg.num_interact
        act_steps = cfg.act_steps
        obs_horizon = cfg.obs_horizon
        wm_args = self.wm_adapter.args
        num_history = getattr(wm_args, "num_history", obs_horizon)
        window_len = num_history + act_steps

        results: List[SyntheticTrajectory] = []

        # 分批次，每批 num_envs 条
        for batch_start in range(0, len(indices), num_envs):
            batch_idx = indices[batch_start : batch_start + num_envs]
            B = len(batch_idx)

            # ---- Step 0: 重置所有 env ----
            env_resets_ok = [False] * B
            batch_obs_list: list[Any] = [None] * B

            # 尝试统一 reset（env 可能支持 options）
            try:
                obs, _ = env.reset()
                for ei in range(B):
                    batch_obs_list[ei] = obs
                    env_resets_ok[ei] = True
            except Exception:
                pass

            # 尝试逐 env 设置 initial_env_state（若有）
            for ei, global_i in enumerate(batch_idx):
                env_state = initial_env_states[global_i]
                if env_state is not None:
                    try:
                        env.set_state(env_state)
                    except Exception:
                        pass  # 忽略，使用已 reset 的随机状态

            # ---- 初始化每个 env 的状态 ----
            lat_h, lat_w, lat_ch = 48, 24, 4
            per_env_lats: list[torch.Tensor] = []
            per_env_states: list[np.ndarray] = []
            per_env_obs_hist: list[list[np.ndarray]] = []

            for ei, global_i in enumerate(batch_idx):
                lat = initial_latents[global_i].to(self.device)
                per_env_lats.append(
                    lat.unsqueeze(0).expand(window_len, -1, -1, -1).clone()
                )
                # 获取初始状态（优先从 env 拿）
                try:
                    st = extract_agent_state(batch_obs_list[ei])[ei]
                except Exception:
                    st = initial_states[global_i].copy()
                per_env_states.append(st)
                per_env_obs_hist.append(
                    [lat.cpu().float().numpy().flatten()] * obs_horizon
                )

            # ---- 收集列表 ----
            per_env_latents: list[list[np.ndarray]] = [[] for _ in range(B)]
            per_env_actions: list[list[np.ndarray]] = [[] for _ in range(B)]
            per_env_state_seqs: list[list[np.ndarray]] = [[] for _ in range(B)]
            per_env_done: list[bool] = [False] * B

            for k in range(num_interact):
                for ei in range(B):
                    if per_env_done[ei]:
                        continue

                    lat_buf = per_env_lats[ei]
                    obs_history = per_env_obs_hist[ei]

                    # ---- Step 1-2: 视觉特征 ----
                    decoded_rgb = None
                    if cfg.decode_for_policy:
                        try:
                            decoded_rgb = self.wm_adapter.decode_latents(
                                lat_buf[-1].unsqueeze(0).float(), decode_chunk_size=1
                            )
                            vis_feat = decoded_rgb.flatten().astype(np.float32) / 255.0
                        except Exception:
                            vis_feat = lat_buf[-1].cpu().float().numpy().flatten()
                    else:
                        vis_feat = lat_buf[-1].cpu().float().numpy().flatten()

                    obs_history.append(vis_feat)
                    if len(obs_history) > obs_horizon:
                        obs_history.pop(0)
                    per_env_obs_hist[ei] = obs_history

                    obs_np = np.stack(obs_history, axis=0).flatten()
                    obs_tensor = (
                        torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
                    )

                    # ---- Step 3: 策略 ----
                    try:
                        raw_actions = self.policy.get_actions(
                            obs_tensor,
                            decoded_rgb=decoded_rgb,
                            agent_state=per_env_states[ei],
                        )
                    except TypeError:
                        try:
                            raw_actions = self.policy.get_actions(obs_tensor)
                        except Exception:
                            raw_actions = np.zeros((1, 7), dtype=np.float32)
                    except Exception:
                        raw_actions = np.zeros((1, 7), dtype=np.float32)

                    # Build action_chunk: support (act_steps, 7) and (1, 7)
                    if raw_actions.ndim == 2 and raw_actions.shape[0] >= act_steps:
                        action_chunk = raw_actions[:act_steps]
                    elif raw_actions.ndim == 2 and raw_actions.shape[0] == 1:
                        action_chunk = np.tile(raw_actions, (act_steps, 1))
                    elif raw_actions.ndim == 1:
                        action_chunk = np.tile(raw_actions[None, :], (act_steps, 1))
                    else:
                        action_chunk = np.zeros((act_steps, 7), dtype=np.float32)

                    # ---- Step 4: 世界模型 ----
                    # EE pose conditioning: history from buffer, future from policy ee_pose
                    current_ee = state_to_ee_pose_7d(per_env_states[ei])  # (7,) world frame
                    hist_ee = np.tile(current_ee[None, :], (num_history, 1))  # (num_history, 7)
                    future_ee = self._predict_future_ee(per_env_states[ei], action_chunk)  # (act_steps, 7)
                    full_ee_poses = np.concatenate([hist_ee, future_ee], axis=0)
                    pred_latents = self.wm_adapter.rollout(
                        obs_latents=lat_buf.clone(),
                        actions=full_ee_poses,
                        instruction=instructions[batch_idx[ei]],
                    )
                    cam0 = pred_latents[0]
                    cam1 = pred_latents[1] if pred_latents.shape[0] > 1 else cam0
                    new_latents = torch.cat(
                        [cam0.to(self.device), cam1.to(self.device)], dim=2
                    )

                    # ---- Step 5 (P4.3): env.step() ----
                    state_seq: list[np.ndarray] = []
                    for step_i in range(act_steps):
                        act_input = action_chunk[step_i][None, :]  # (1, action_dim)
                        try:
                            step_obs, _r, terminated, truncated, _info = env.step(
                                act_input
                            )
                            cur_state = extract_agent_state(step_obs)[0]
                            term_val = terminated
                            if isinstance(term_val, (torch.Tensor, np.ndarray)):
                                term_val = bool(
                                    term_val.cpu().item()
                                    if isinstance(term_val, torch.Tensor)
                                    else term_val.item()
                                )
                            if term_val:
                                per_env_done[ei] = True
                        except Exception as exc:
                            print(f"[VLAW-P4.3] ⚠️  env.step env_i={ei}: {exc}")
                            cur_state = per_env_states[ei].copy()
                        per_env_states[ei] = cur_state
                        state_seq.append(cur_state)

                    # ---- Step 6: 更新 buffer ----
                    per_env_lats[ei] = (
                        new_latents[-window_len:].clone()
                        if new_latents.shape[0] >= window_len
                        else torch.cat(
                            [lat_buf[new_latents.shape[0] :], new_latents], dim=0
                        )
                    )

                    # ---- 收集 ----
                    for step_i in range(act_steps):
                        per_env_latents[ei].append(
                            new_latents[step_i].cpu().float().numpy().astype(np.float16)
                        )
                        per_env_actions[ei].append(action_chunk[step_i])
                        per_env_state_seqs[ei].append(state_seq[step_i])

            # ---- 组装 SyntheticTrajectory ----
            for ei, global_i in enumerate(batch_idx):
                if not per_env_latents[ei]:
                    continue
                traj = SyntheticTrajectory(
                    latents=np.stack(per_env_latents[ei], axis=0),
                    actions=np.stack(per_env_actions[ei], axis=0),
                    states=np.stack(per_env_state_seqs[ei], axis=0),
                    instruction=instructions[global_i],
                    task_id=task_id,
                )
                results.append(traj)

        return results

    # ------------------------------------------------------------------
    # 保存轨迹（与 imagination.py 格式完全一致）
    # ------------------------------------------------------------------

    def save_trajectories(
        self,
        trajectories: List[SyntheticTrajectory],
        output_dir: str,
    ) -> str:
        """保存合成轨迹为 HDF5（格式兼容 VLAWDataCollector 输出）."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        out_path = out_dir / f"synthetic_env_{ts}.h5"

        with h5py.File(str(out_path), "w") as f:
            meta = f.create_group("meta")
            meta.attrs["num_trajectories"] = len(trajectories)
            meta.attrs["source"] = "imagination_env"
            if trajectories:
                meta.attrs["env_id"] = trajectories[0].task_id
            meta.attrs["latent_shape"] = "T,4,48,24"
            meta.attrs["step5_method"] = "env.step()"  # 区分 P4.2 / P4.3

            for idx, traj in enumerate(trajectories):
                grp = f.create_group(f"traj_{idx:04d}")
                grp.create_dataset(
                    "latent", data=traj.latents,
                    chunks=True, compression="gzip", compression_opts=1,
                )
                grp.create_dataset(
                    "actions", data=traj.actions,
                    chunks=True, compression="gzip", compression_opts=1,
                )
                grp.create_dataset(
                    "state", data=traj.states,
                    chunks=True, compression="gzip", compression_opts=1,
                )
                grp.attrs["task_instruction"] = traj.instruction
                grp.attrs["task_id"] = traj.task_id
                grp.attrs["source"] = "imagination_env"

        print(
            f"[VLAW-P4.3] HDF5 已保存: {out_path} ({len(trajectories)} 条轨迹)"
        )
        return str(out_path)

    # ------------------------------------------------------------------
    # 主运行入口
    # ------------------------------------------------------------------

    def run(
        self,
        iter_id: int = 1,
        real_data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        num_trajectories: Optional[int] = None,
    ) -> dict:
        """完整 Imagination 运行流程.

        接口兼容 ImaginationEngine.run()，新增 real_data_dir / num_trajectories 参数。

        Args:
            iter_id:          当前迭代轮次
            real_data_dir:    初始帧目录（覆盖 config.initial_frames_source）
            output_dir:       输出目录（覆盖 config.output_dir）
            num_trajectories: 每任务总轨迹数（覆盖 config.num_rollouts_per_task）

        Returns:
            {"num_generated": int, "success_rate_est": float,
             "output_paths": list[str], "iter_id": int}
        """
        cfg = self.config
        src_dir = real_data_dir or cfg.initial_frames_source
        out_root = output_dir or cfg.output_dir
        out_dir = f"{out_root}/iter{iter_id}"
        num_rollouts = num_trajectories or cfg.num_rollouts_per_task

        dry_run = cfg.dry_run
        if dry_run:
            num_rollouts = 1

        total_target = 0
        total_success = 0
        all_output_paths: list[str] = []

        for task_id in cfg.tasks:
            print(
                f"[VLAW-P4.3] 任务: {task_id}, 目标生成 {num_rollouts} 条, "
                f"num_envs={cfg.num_envs}"
            )

            # ---- 加载初始帧 ----
            initial_frames = _load_initial_frames(src_dir, task_id, num_rollouts)

            lat_h, lat_w = 48, 24
            task_trajs: List[SyntheticTrajectory] = []

            # 构建全部 items
            batch_lats, batch_states, batch_ins, batch_ids = [], [], [], []
            for i in range(num_rollouts):
                if i < len(initial_frames) and initial_frames[i]["latent"] is not None:
                    lat = torch.from_numpy(initial_frames[i]["latent"])
                    st = initial_frames[i]["state"]
                    ins = initial_frames[i]["instruction"] or f"complete {task_id}"
                else:
                    lat = torch.randn(4, lat_h, lat_w, dtype=torch.float32)
                    st_dim = (
                        initial_frames[i]["state"].shape[0]
                        if i < len(initial_frames)
                        else 29
                    )
                    st = np.zeros(st_dim, dtype=np.float32)
                    ins = f"complete {task_id}"
                batch_lats.append(lat)
                batch_states.append(st)
                batch_ins.append(ins)
                batch_ids.append(task_id)

            # 分批调用 rollout_batch
            bs = cfg.num_envs
            n_batches = max(1, (num_rollouts + bs - 1) // bs)
            for b in range(n_batches):
                s, e = b * bs, min((b + 1) * bs, num_rollouts)
                res = self.rollout_batch(
                    batch_lats[s:e],
                    batch_states[s:e],
                    batch_ins[s:e],
                    batch_ids[s:e],
                )
                task_trajs.extend(res)
                print(
                    f"[VLAW-P4.3]   batch {b+1}/{n_batches}: "
                    f"生成 {len(res)}/{e-s} 条"
                )
                # 定期清理 GPU 缓存（防止显存碑片化）
                if (b + 1) % 5 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            # ---- 保存 ----
            if task_trajs and not dry_run:
                task_out = f"{out_dir}/{task_id}"
                out_path = self.save_trajectories(task_trajs, task_out)
                all_output_paths.append(out_path)

            total_target += num_rollouts
            total_success += len(task_trajs)
            print(
                f"[VLAW-P4.3] 任务 {task_id}: "
                f"{len(task_trajs)}/{num_rollouts} 条生成成功"
            )

        summary = {
            "num_generated": total_success,
            "success_rate_est": total_success / max(total_target, 1),
            "output_paths": all_output_paths,
            "iter_id": iter_id,
        }
        print(f"[VLAW-P4.3] 运行完成: {json.dumps(summary, indent=2)}")
        return summary


# ---------------------------------------------------------------------------
# Mock 对象（仅供 dry_run 测试）
# ---------------------------------------------------------------------------


class _MockPolicy:
    """零动作 mock 策略."""

    def get_actions(self, obs_features: torch.Tensor, **kwargs) -> np.ndarray:
        B = obs_features.shape[0]
        return np.zeros((B, 7), dtype=np.float32)


class _MockCtrlWorldAdapter:
    """mock 世界模型，返回随机 latent."""

    class _MockArgs:
        num_history: int = 2
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
        actions: "np.ndarray | torch.Tensor",
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
        N = latents.shape[0]
        return np.zeros((N, 192, 192, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 入口 / dry_run 验证
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys
    import os

    # 支持从项目根目录直接运行：python rlft/vlaw/imagination_env.py
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    parser = argparse.ArgumentParser(
        description="VLAW P4.3 ImaginationEnvEngine dry_run 验证"
    )
    parser.add_argument("--dry_run", action="store_true", help="仅验证接口，不启动 ManiSkill")
    args = parser.parse_args()

    if args.dry_run:
        print("[dry_run] ImaginationEnvEngine 接口验证...")

        # 1. 配置实例化
        cfg = ImaginationEnvConfig(
            num_envs=4,
            task_id="LiftPegUpright-v1",
            dry_run=True,
            decode_for_policy=False,
        )
        print(f"  ImaginationEnvConfig: num_envs={cfg.num_envs}, task_id={cfg.task_id}")

        # 2. 引擎实例化（使用 mock）
        mock_wm = _MockCtrlWorldAdapter()
        mock_policy = _MockPolicy()
        engine = ImaginationEnvEngine(
            wm_adapter=mock_wm,   # type: ignore[arg-type]
            policy=mock_policy,
            config=cfg,
        )
        print(f"  ImaginationEnvEngine 实例化成功, device={engine.device}")

        # 3. 验证方法签名
        import inspect

        sig_single = inspect.signature(engine.rollout_single)
        sig_batch = inspect.signature(engine.rollout_batch)
        sig_run = inspect.signature(engine.run)
        print(f"  rollout_single 参数: {list(sig_single.parameters.keys())}")
        print(f"  rollout_batch  参数: {list(sig_batch.parameters.keys())}")
        print(f"  run            参数: {list(sig_run.parameters.keys())}")

        # 4. 验证 SyntheticTrajectory 可构造
        dummy_traj = SyntheticTrajectory(
            latents=np.zeros((10, 4, 48, 24), dtype=np.float16),
            actions=np.zeros((10, 7), dtype=np.float32),
            states=np.zeros((10, 29), dtype=np.float32),
            instruction="pick the cube",
            task_id="PickCube-v1",
        )
        print(
            f"  SyntheticTrajectory: latents={dummy_traj.latents.shape}, "
            f"actions={dummy_traj.actions.shape}"
        )

        # 5. 验证 initial_env_state 参数存在
        assert "initial_env_state" in sig_single.parameters, \
            "rollout_single 缺少 initial_env_state 参数"
        assert "initial_env_states" in sig_batch.parameters, \
            "rollout_batch 缺少 initial_env_states 参数"
        print("  initial_env_state 参数: ✓")

        print("[dry_run] 通过 ✅")
    else:
        print("[VLAW-P4.3] 请使用 --dry_run 进行接口验证，或在外部脚本中实例化 ImaginationEnvEngine。")
        print("示例: python rlft/vlaw/imagination_env.py --dry_run")
