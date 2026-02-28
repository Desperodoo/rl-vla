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

# 复用 imagination.py 中的数据容器和工具函数
from rlft.vlaw.utils.imagination import SyntheticTrajectory, _load_initial_frames


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
        print(
            f"[VLAW-P4.3] ImaginationEnvEngine 初始化完成, "
            f"device={self.device}, num_envs={config.num_envs}"
        )

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
        """单条轨迹生成核心实现（创建独立 env，用完即关）."""
        cfg = self.config
        num_interact = cfg.num_interact
        act_steps = cfg.act_steps
        obs_horizon = cfg.obs_horizon

        # ---- 创建单 env ----
        env = self._make_env(task_id, num_envs=1)
        try:
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
        finally:
            env.close()

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

        # ---- 初始化 latent buffer ----
        initial_latent = initial_latent.to(self.device)
        window_len = num_history + act_steps
        lat_buf = initial_latent.unsqueeze(0).expand(window_len, -1, -1, -1).clone()
        # (window_len, 4, 48, 24)

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

        for k in range(num_interact):
            if env_success:
                break

            # ---- Step 1-2: 构建策略输入 ----
            if self.config.decode_for_policy:
                try:
                    cur_lat = lat_buf[-1].unsqueeze(0)  # (1, 4, 48, 24)
                    rgb = self.wm_adapter.decode_latents(
                        cur_lat.float(), decode_chunk_size=1
                    )  # (1, H, W, 3) uint8
                    vis_feat = rgb.flatten().astype(np.float32) / 255.0
                except Exception:
                    vis_feat = lat_buf[-1].cpu().float().numpy().flatten()
            else:
                vis_feat = lat_buf[-1].cpu().float().numpy().flatten()

            obs_history.append(vis_feat)
            if len(obs_history) > obs_horizon:
                obs_history.pop(0)

            obs_np = np.stack(obs_history, axis=0).flatten()
            obs_tensor = (
                torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
            )  # (1, obs_horizon * feat_dim)

            # ---- Step 3: 策略推理 → action_chunk ----
            try:
                actions_np = self.policy.get_actions(obs_tensor)  # (1, action_dim)
            except Exception:
                actions_np = np.zeros((1, 7), dtype=np.float32)

            action_t = actions_np[0]  # (action_dim,)
            action_chunk = np.tile(action_t[None, :], (act_steps, 1))  # (act_steps, 7)

            # ---- Step 4: 世界模型 rollout ----
            wm_input = lat_buf.clone()
            hist_acts = np.zeros((num_history, action_t.shape[0]), dtype=np.float32)
            full_acts = np.concatenate([hist_acts, action_chunk], axis=0)

            pred_latents = self.wm_adapter.rollout(
                obs_latents=wm_input,
                actions=full_acts,
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

            # ---- Step 6: 更新 history buffer ----
            lat_buf = (
                new_latents[-window_len:].clone()
                if new_latents.shape[0] >= window_len
                else torch.cat(
                    [lat_buf[new_latents.shape[0] :], new_latents], dim=0
                )
            )

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

            # 创建 num_envs 并行 env
            try:
                env = self._make_env(tid, num_envs=batch_num_envs)
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
                    env.close()
                    env = None
                else:
                    env.close()

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
                    if cfg.decode_for_policy:
                        try:
                            rgb = self.wm_adapter.decode_latents(
                                lat_buf[-1].unsqueeze(0).float(), decode_chunk_size=1
                            )
                            vis_feat = rgb.flatten().astype(np.float32) / 255.0
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
                        actions_np = self.policy.get_actions(obs_tensor)
                    except Exception:
                        actions_np = np.zeros((1, 7), dtype=np.float32)

                    action_t = actions_np[0]
                    action_chunk = np.tile(action_t[None, :], (act_steps, 1))

                    # ---- Step 4: 世界模型 ----
                    hist_acts = np.zeros((num_history, action_t.shape[0]), dtype=np.float32)
                    full_acts = np.concatenate([hist_acts, action_chunk], axis=0)
                    pred_latents = self.wm_adapter.rollout(
                        obs_latents=lat_buf.clone(),
                        actions=full_acts,
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

    def get_actions(self, obs_features: torch.Tensor) -> np.ndarray:
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
