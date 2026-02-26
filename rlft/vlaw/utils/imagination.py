"""VLAW P4.2 — Imagination Engine（Policy-in-the-Loop World Model Rollout）

VLAW 核心创新：在世界模型中用 ShortCut Flow 做闭环推理，生成合成轨迹。

流程：
    initial_latent (来自真实帧 VAE encode)
    for k in range(K_interact=12):
        1. decoder latent → RGB (2cam) [可选，供策略视觉编码]
        2. PlainConv(RGB) → visual_feature
        3. [visual_feature, state_t] → obs → ShortCut Flow → action_chunk
        4. CtrlWorldAdapter.rollout(history + current, action_chunk) → pred_latents
        5. [临时] StatePredictor(state_t, action_t) → state_{t+1}
        6. 更新 history buffer
    输出: latent 序列 → 可选 decode → VLM 评估

⚠️  Step 5 当前使用 State Predictor MLP（临时脚手架，仅用于跑通流程）。
    本项目基于 ManiSkill 仿真（仿真替代真机），env.step() 可精确返回 state_{t+1}，
    是 P4.3 必须实现的最终方案：
        - 将 Step 5 替换为 ManiSkill env.step() 调用
        - 通过 num_envs=1..N 控制并行规模，系统测试数据效率
    参见 ADR-004 和 ADR-006（.github/knowledge/decisions.md）。

所属阶段: P4.2 — Imagination Engine（当前版本）/ P4.3 — env.step() 版本（待实现）
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

from rlft.vlaw.policy.state_predictor import StatePredictor


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class ImaginationConfig:
    """P4.2 Imagination Engine 配置."""

    num_interact: int = 12
    """闭环交互轮数 (K_interact)"""

    act_steps: int = 5
    """每次策略调用在世界模型中执行的步数 (= world model num_frames)"""

    obs_horizon: int = 2
    """策略观测历史长度"""

    decode_for_policy: bool = True
    """是否将 latent 解码为 RGB 供策略视觉编码器使用"""

    save_decoded_rgb: bool = False
    """是否保存解码的 RGB 帧（调试用，会增加存储）"""

    output_dir: str = "data/vlaw/synthetic/iter1"
    """合成轨迹输出目录"""

    gpu_id: int = 0
    """使用的 GPU id"""

    batch_size: int = 4
    """并行 imagination 数量"""

    tasks: list = field(
        default_factory=lambda: ["LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"]
    )
    """目标任务列表"""

    num_rollouts_per_task: int = 500
    """每个任务生成的合成轨迹数量"""

    initial_frames_source: str = "data/vlaw/rollouts/iter1"
    """初始帧来源目录（从真实 rollout HDF5 中随机采样）"""

    dry_run: bool = False
    """dry_run=True 时只执行 2 轮 × 1 条轨迹，不保存，用于验证流程"""


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------


@dataclass
class SyntheticTrajectory:
    """单条合成轨迹数据容器.

    Attributes:
        latents:     (T, 4, 48, 24) float16  — VAE latent 序列
        actions:     (T, 7) float32          — delta pose 动作序列
        states:      (T, state_dim) float32  — agent_state 序列（State Predictor 递推）
        instruction: str                     — 任务文本描述
        task_id:     str                     — 任务 ID (e.g. "LiftPegUpright-v1")
    """

    latents: np.ndarray       # (T, 4, 48, 24) float16
    actions: np.ndarray       # (T, 7) float32
    states: np.ndarray        # (T, state_dim) float32
    instruction: str
    task_id: str


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _load_initial_frames(source_dir: str, task_id: str, n: int) -> list[dict]:
    """从 HDF5 rollout 数据中随机采样 n 条初始帧信息.

    每条返回:
        {
            "latent":      (4, 48, 24) float32  ← 第一帧 latent（若不存在则 None）
            "state":       (state_dim,) float32
            "instruction": str
        }

    Args:
        source_dir: 包含 .h5 文件的目录
        task_id:    任务 ID, 用于过滤
        n:          采样数量

    Returns:
        最多 n 条初始帧 dict 列表（不足时重复采样）
    """
    src = Path(source_dir)
    h5_files = sorted(src.glob("**/*.h5"))
    if not h5_files:
        print(f"[VLAW-P4.2] ⚠️  未找到 HDF5 文件: {source_dir}, 使用随机 latent")
        return []

    candidates: list[dict] = []
    for h5_file in h5_files:
        try:
            with h5py.File(str(h5_file), "r") as f:
                traj_keys = [k for k in f.keys() if k.startswith("traj_")]
                for key in traj_keys:
                    grp = f[key]
                    env_id = f.get("meta", {}).attrs.get("env_id", "")
                    if task_id not in env_id and env_id:
                        continue
                    instruction = grp.attrs.get("task_instruction", "")
                    # 尝试读取 latent（由 VAE pipeline 生成）
                    latent = None
                    if "latent" in grp:
                        latent = grp["latent"][0].astype(np.float32)  # 第一帧
                    # 读取状态
                    if "obs_agent" in grp:
                        state = grp["obs_agent"][0].astype(np.float32)
                    elif "state" in grp:
                        state = grp["state"][0].astype(np.float32)
                    else:
                        state = np.zeros(29, dtype=np.float32)
                    candidates.append(
                        {
                            "latent": latent,
                            "state": state,
                            "instruction": instruction,
                        }
                    )
        except Exception as exc:
            print(f"[VLAW-P4.2] ⚠️  读取 {h5_file} 失败: {exc}")

    if not candidates:
        return []

    rng = np.random.default_rng(42)
    idxs = rng.integers(0, len(candidates), size=n)
    return [candidates[i] for i in idxs]


# ---------------------------------------------------------------------------
# 核心引擎
# ---------------------------------------------------------------------------


class ImaginationEngine:
    """Policy-in-the-Loop Imagination 引擎（VLAW P4.2 核心）.

    在未训练或已训练的世界模型中，让 ShortCut Flow 策略做闭环推理，
    生成合成轨迹数据用于策略与世界模型的迭代共同改进。

    Args:
        wm_adapter:       CtrlWorldAdapter 实例（可接受未训练权重）
        policy:           符合 PolicyProtocol 的策略对象
        state_predictor:  StatePredictor 实例（可未训练）
        config:           ImaginationConfig
    """

    def __init__(
        self,
        wm_adapter: "CtrlWorldAdapter",
        policy: Any,
        state_predictor: StatePredictor,
        config: ImaginationConfig,
    ) -> None:
        self.wm_adapter = wm_adapter
        self.policy = policy
        self.state_predictor = state_predictor
        self.config = config
        self.device = torch.device(
            f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print(f"[VLAW-P4.2] ImaginationEngine 初始化完成, device={self.device}")

    # ------------------------------------------------------------------
    # 单条轨迹生成
    # ------------------------------------------------------------------

    def rollout_single(
        self,
        initial_latent: torch.Tensor,
        initial_state: np.ndarray,
        instruction: str,
        task_id: str,
    ) -> Optional[SyntheticTrajectory]:
        """生成单条合成轨迹.

        Args:
            initial_latent: (4, 48, 24) float32 Tensor — 初始帧 VAE latent
            initial_state:  (state_dim,) float32 array — 初始 agent_state
            instruction:    任务文本描述
            task_id:        任务 ID

        Returns:
            SyntheticTrajectory，或失败时返回 None
        """
        try:
            return self._rollout_single_impl(
                initial_latent, initial_state, instruction, task_id
            )
        except Exception:
            print(
                f"[VLAW-P4.2] ⚠️  rollout_single 失败 (task={task_id}), "
                f"世界模型可能未训练:\n{traceback.format_exc()}"
            )
            return None

    def _rollout_single_impl(
        self,
        initial_latent: torch.Tensor,
        initial_state: np.ndarray,
        instruction: str,
        task_id: str,
    ) -> SyntheticTrajectory:
        """单条轨迹生成的具体实现（内部方法）."""
        # ---- 参数 ----
        num_interact = self.config.num_interact
        act_steps = self.config.act_steps
        obs_horizon = self.config.obs_horizon

        # ---- 获取世界模型参数 ----
        wm_args = self.wm_adapter.args
        num_history = getattr(wm_args, "num_history", obs_horizon)

        # ---- 初始化历史 latent buffer ----
        # 形状: (T_buf, 4, 48, 24)
        lat_h, lat_w = 48, 24
        lat_ch = 4

        # 初始帧 latent 广播填充 obs_horizon + act_steps 帧
        initial_latent = initial_latent.to(self.device)
        window_len = num_history + act_steps
        lat_buf = initial_latent.unsqueeze(0).expand(window_len, -1, -1, -1).clone()
        # (window_len, 4, 48, 24)

        # ---- 状态 ----
        state_dim = initial_state.shape[0]
        state_t = initial_state.copy()  # (state_dim,)

        # ---- 收集列表 ----
        all_latents: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_states: list[np.ndarray] = []

        # obs 特征缓存（策略需要 obs_horizon 帧）
        # 使用 latent 作为简化视觉特征（若 decode_for_policy=False）
        obs_feat_dim = lat_ch * lat_h * lat_w  # 4*48*24 = 4608 (flattened latent)
        obs_history: list[np.ndarray] = []
        for _ in range(obs_horizon):
            obs_history.append(initial_latent.cpu().float().numpy().flatten())

        for k in range(num_interact):
            # ---- Step 1-2: 构建策略输入 obs 特征 ----
            if self.config.decode_for_policy:
                # 解码最新 latent 为 RGB，再归一化作为视觉特征
                try:
                    cur_latent_t = lat_buf[-1].unsqueeze(0)  # (1, 4, 48, 24)
                    rgb = self.wm_adapter.decode_latents(
                        cur_latent_t.float(), decode_chunk_size=1
                    )  # (1, H, W, 3) uint8
                    vis_feat = rgb.flatten().astype(np.float32) / 255.0
                except Exception:
                    vis_feat = lat_buf[-1].cpu().float().numpy().flatten()
            else:
                vis_feat = lat_buf[-1].cpu().float().numpy().flatten()

            obs_history.append(vis_feat)
            if len(obs_history) > obs_horizon:
                obs_history.pop(0)

            # obs: concat history features → 策略期望 (1, obs_horizon * feat_dim)
            obs_np = np.stack(obs_history, axis=0).flatten()  # (obs_horizon * feat_dim,)
            obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
            # (1, obs_horizon * feat_dim)

            # ---- Step 3: 策略推理 → action_chunk ----
            try:
                actions_np = self.policy.get_actions(obs_tensor)  # (1, action_dim)
            except Exception:
                # 策略失败时使用零动作
                actions_np = np.zeros((1, 7), dtype=np.float32)

            action_t = actions_np[0]  # (action_dim,)

            # 扩展到 act_steps 帧的动作块
            action_chunk = np.tile(action_t[None, :], (act_steps, 1))  # (act_steps, 7)

            # ---- Step 4: 世界模型 rollout ----
            # 构造 (window_len, 4, 48, 24) 的 latent 输入
            wm_input = lat_buf.clone()  # (window_len, 4, 48, 24)
            # 构造 (window_len, action_dim) 的动作输入（历史部分用 0）
            hist_acts = np.zeros((num_history, action_t.shape[0]), dtype=np.float32)
            full_acts = np.concatenate([hist_acts, action_chunk], axis=0)  # (window_len, 7)

            pred_latents = self.wm_adapter.rollout(
                obs_latents=wm_input,
                actions=full_acts,
                instruction=instruction,
            )
            # pred_latents: (N_CAMS, act_steps, 4, 24, 24)

            # 取第一个相机，拼接两相机恢复 (act_steps, 4, 48, 24)
            # CtrlWorldAdapter.rollout 返回 (N_CAMS, T, 4, lat_h/N_CAMS, lat_w)
            # 两相机纵向拼接: (T, 4, 48, 24)
            cam0 = pred_latents[0]  # (T, 4, lat_h_single, lat_w)
            cam1 = pred_latents[1] if pred_latents.shape[0] > 1 else cam0
            # 恢复 (T, 4, 48, 24)  ← cat on dim H
            new_latents = torch.cat([cam0, cam1], dim=2)  # (T, 4, 48, 24)

            # ---- Step 5: State Predictor 递推 ----
            state_seq = self.state_predictor.predict_sequence(
                state_0=state_t,
                actions=action_chunk,
            )  # (act_steps+1, state_dim)
            state_t = state_seq[-1]

            # ---- Step 6: 更新 history buffer ----
            lat_buf = new_latents[-window_len:].clone() if new_latents.shape[0] >= window_len \
                else torch.cat([lat_buf[new_latents.shape[0]:], new_latents], dim=0)

            # ---- 收集 ----
            for step_i in range(act_steps):
                all_latents.append(new_latents[step_i].cpu().float().numpy().astype(np.float16))
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
    # 批量生成
    # ------------------------------------------------------------------

    def rollout_batch(
        self,
        initial_latents: list[torch.Tensor],
        initial_states: list[np.ndarray],
        instructions: list[str],
        task_ids: list[str],
    ) -> List[SyntheticTrajectory]:
        """批量生成合成轨迹.

        Args:
            initial_latents: 长度 B 的 list，每项 (4, 48, 24)
            initial_states:  长度 B 的 list，每项 (state_dim,)
            instructions:    长度 B 的任务描述列表
            task_ids:        长度 B 的任务 ID 列表

        Returns:
            生成成功的 SyntheticTrajectory 列表（失败项已剔除）
        """
        results: List[SyntheticTrajectory] = []
        for lat, st, ins, tid in zip(initial_latents, initial_states, instructions, task_ids):
            traj = self.rollout_single(lat, st, ins, tid)
            if traj is not None:
                results.append(traj)
        return results

    # ------------------------------------------------------------------
    # 保存轨迹（与 data_collector.py 相同的 HDF5 格式）
    # ------------------------------------------------------------------

    def save_trajectories(
        self,
        trajectories: List[SyntheticTrajectory],
        output_dir: str,
    ) -> str:
        """将合成轨迹保存为 HDF5 文件（兼容 VLAWDataCollector 格式）.

        Args:
            trajectories: SyntheticTrajectory 列表
            output_dir:   输出目录

        Returns:
            保存的 HDF5 文件路径字符串
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        out_path = out_dir / f"synthetic_{ts}.h5"

        with h5py.File(str(out_path), "w") as f:
            # --- meta ---
            meta = f.create_group("meta")
            meta.attrs["num_trajectories"] = len(trajectories)
            meta.attrs["source"] = "imagination"
            if trajectories:
                meta.attrs["env_id"] = trajectories[0].task_id
            meta.attrs["latent_shape"] = "T,4,48,24"

            # --- per-trajectory ---
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
                grp.attrs["source"] = "imagination"

        print(
            f"[VLAW-P4.2] HDF5 已保存: {out_path} ({len(trajectories)} 条轨迹)"
        )
        return str(out_path)

    # ------------------------------------------------------------------
    # 主运行入口
    # ------------------------------------------------------------------

    def run(self, iter_id: int = 1) -> dict:
        """完整 Imagination 运行流程.

        从 initial_frames_source 采样初始帧 → 批量 rollout → 保存。

        Args:
            iter_id: 当前迭代轮次（用于日志与输出路径命名）

        Returns:
            统计摘要 dict:
                {"total_trajectories": int, "success_count": int,
                 "output_path": str, "iter_id": int}
        """
        cfg = self.config
        out_dir = f"{cfg.output_dir}/iter{iter_id}"
        total_gen = 0
        success_count = 0
        all_output_paths: list[str] = []

        num_interact_override = 2 if cfg.dry_run else cfg.num_interact
        num_rollouts_per_task = 1 if cfg.dry_run else cfg.num_rollouts_per_task

        for task_id in cfg.tasks:
            print(f"[VLAW-P4.2] 任务: {task_id}, 目标生成 {num_rollouts_per_task} 条")

            # ---- 加载初始帧 ----
            initial_frames = _load_initial_frames(
                cfg.initial_frames_source, task_id, num_rollouts_per_task
            )

            # ---- 若无真实帧，使用随机 latent ----
            task_trajs: List[SyntheticTrajectory] = []
            lat_h, lat_w = 48, 24
            n_batches = max(1, (num_rollouts_per_task + cfg.batch_size - 1) // cfg.batch_size)

            for b in range(n_batches):
                start = b * cfg.batch_size
                end = min(start + cfg.batch_size, num_rollouts_per_task)

                batch_lats, batch_states, batch_ins = [], [], []
                for i in range(start, end):
                    if i < len(initial_frames) and initial_frames[i]["latent"] is not None:
                        lat = torch.from_numpy(initial_frames[i]["latent"])
                        st = initial_frames[i]["state"]
                        ins = initial_frames[i]["instruction"] or f"complete the task {task_id}"
                    else:
                        lat = torch.randn(4, lat_h, lat_w, dtype=torch.float32)
                        st_dim = (
                            initial_frames[i]["state"].shape[0]
                            if i < len(initial_frames)
                            else 29
                        )
                        st = np.zeros(st_dim, dtype=np.float32)
                        ins = f"complete the task {task_id}"

                    batch_lats.append(lat)
                    batch_states.append(st)
                    batch_ins.append(ins)

                batch_ids = [task_id] * len(batch_lats)

                # ---- 临时覆盖 num_interact（dry_run）----
                orig_interact = self.config.num_interact
                if cfg.dry_run:
                    self.config.num_interact = num_interact_override

                batch_results = self.rollout_batch(
                    batch_lats, batch_states, batch_ins, batch_ids
                )

                if cfg.dry_run:
                    self.config.num_interact = orig_interact

                task_trajs.extend(batch_results)
                print(
                    f"[VLAW-P4.2]   batch {b+1}/{n_batches}: "
                    f"生成 {len(batch_results)}/{len(batch_lats)} 条"
                )

            # ---- 保存 ----
            if task_trajs and not cfg.dry_run:
                task_out_dir = f"{out_dir}/{task_id}"
                out_path = self.save_trajectories(task_trajs, task_out_dir)
                all_output_paths.append(out_path)

            success_count += len(task_trajs)
            total_gen += num_rollouts_per_task
            print(
                f"[VLAW-P4.2] 任务 {task_id} 完成: "
                f"{len(task_trajs)}/{num_rollouts_per_task} 条生成成功"
            )

        summary = {
            "total_trajectories": total_gen,
            "success_count": success_count,
            "output_paths": all_output_paths,
            "iter_id": iter_id,
        }
        print(f"[VLAW-P4.2] 运行完成: {json.dumps(summary, indent=2)}")
        return summary


# ---------------------------------------------------------------------------
# Mock 对象（仅用于 dry_run 测试）
# ---------------------------------------------------------------------------


class _MockPolicy:
    """零动作 mock 策略，用于 dry_run 测试."""

    def get_actions(self, obs_features: torch.Tensor) -> np.ndarray:
        B = obs_features.shape[0]
        return np.zeros((B, 7), dtype=np.float32)


class _MockCtrlWorldAdapter:
    """返回随机 latent 的 mock 世界模型，用于 dry_run 测试."""

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
        # state_p01/p99 for denormalize_action
        self.state_p01 = np.zeros((1, 7), dtype=np.float32)
        self.state_p99 = np.ones((1, 7), dtype=np.float32)

    @torch.no_grad()
    def rollout(
        self,
        obs_latents: torch.Tensor,
        actions: "np.ndarray | torch.Tensor",
        instruction: str = "",
    ) -> torch.Tensor:
        # 返回 (N_CAMS=2, act_steps, 4, 24, 24) 随机 latent
        T = self.args.num_frames
        return torch.randn(2, T, 4, 24, 24, dtype=torch.float32)

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.Tensor,
        decode_chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        # 返回 (N, H, W, 3) uint8
        N = latents.shape[0]
        return np.zeros((N, 192, 192, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


@dataclass
class _EntryConfig:
    """imagination.py 入口配置."""

    dry_run: bool = False
    """True: 用 mock 模型验证 2 轮 × 1 条轨迹流程，不保存"""

    gpu_id: int = 0
    """使用的 GPU id"""

    iter_id: int = 1
    """当前迭代轮次"""

    engine: ImaginationConfig = field(default_factory=ImaginationConfig)
    """Imagination Engine 配置"""


if __name__ == "__main__":
    import tyro

    entry = tyro.cli(_EntryConfig)
    cfg = entry.engine
    cfg.dry_run = entry.dry_run or cfg.dry_run
    cfg.gpu_id = entry.gpu_id

    if cfg.dry_run:
        print("[VLAW-P4.2] === dry_run 模式: 使用 Mock 模型验证流程 ===")
        cfg.tasks = ["LiftPegUpright-v1"]
        cfg.batch_size = 1
        cfg.num_rollouts_per_task = 1
        cfg.decode_for_policy = False  # mock 无真实 VAE

        mock_wm = _MockCtrlWorldAdapter()
        mock_policy = _MockPolicy()
        mock_sp = StatePredictor(state_dim=29, action_dim=7, hidden_dim=64)

        engine = ImaginationEngine(
            wm_adapter=mock_wm,  # type: ignore[arg-type]
            policy=mock_policy,
            state_predictor=mock_sp,
            config=cfg,
        )
        summary = engine.run(iter_id=entry.iter_id)
        print(f"[VLAW-P4.2] dry_run 完成: {summary}")
    else:
        print("[VLAW-P4.2] 请在外部脚本中实例化 ImaginationEngine 并调用 run()。")
        print("提示: 使用 --dry_run True 进行流程验证。")
