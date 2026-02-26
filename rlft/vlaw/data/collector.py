"""ManiSkill Rollout 数据收集器.

P1.1 阶段: 在 ManiSkill GPU 向量化环境中用 ShortCut Flow (或随机策略) 收集轨迹，
保存为 HDF5 格式供 Ctrl-World 世界模型训练使用。

数据格式见 VLAW_REPRODUCTION_PLAN.md 第 3.1.2 节。

所属阶段: P1.1 — ManiSkill Rollout 收集器
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import h5py
import numpy as np
import torch
import tyro


# ---------------------------------------------------------------------------
# 策略协议 (Protocol)
# ---------------------------------------------------------------------------

@runtime_checkable
class PolicyProtocol(Protocol):
    """数据收集器接受的策略接口."""

    def get_actions(
        self,
        obs_features: torch.Tensor,   # (N_env, obs_horizon, feat_dim)
    ) -> np.ndarray:
        """返回动作数组.

        对于 chunk 策略 (ShortCutFlow / PLD-SAC): 返回 (N_env, act_steps, action_dim)
        对于随机策略: 返回 (N_env, action_dim)
        """
        ...


class RandomPolicy:
    """随机策略，用于基线数据收集。"""

    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def get_actions(self, obs_features: torch.Tensor) -> np.ndarray:
        """在向量化环境中 sample() 本身返回 (N, action_dim)."""
        sample = self.action_space.sample()
        if isinstance(sample, torch.Tensor):
            sample = sample.cpu().numpy()
        arr = np.asarray(sample)
        # 若 action_space 返回 (action_dim,) 则扩展到 (N, action_dim)
        if arr.ndim == 1:
            N = obs_features.shape[0]
            arr = np.stack([arr] * N)
        return arr


class ShortCutFlowPolicy:
    """包装 ShortCutFlowWrapper 适配 PolicyProtocol.

    Args:
        wrapper: 加载好的 ShortCutFlowWrapper 实例
        visual_encoder: PlainConv 视觉编码器 (可为 None)
        device: 计算设备
        include_rgb: 是否包含 RGB 特征
        obs_horizon: 观测历史长度
        action_pred_horizon: 动作预测长度
        act_steps: 每步执行的动作数量
    """

    def __init__(
        self,
        wrapper,
        visual_encoder: Optional[torch.nn.Module],
        device: torch.device,
        include_rgb: bool = True,
        obs_horizon: int = 2,
        action_pred_horizon: int = 16,
        act_steps: int = 8,
    ) -> None:
        self.wrapper = wrapper
        self.visual_encoder = visual_encoder
        self.device = device
        self.include_rgb = include_rgb
        self.obs_horizon = obs_horizon
        self.action_pred_horizon = action_pred_horizon
        self.act_steps = act_steps

    @torch.no_grad()
    def get_actions(self, obs_features: torch.Tensor) -> np.ndarray:
        """obs_features: (N_env, obs_horizon * feat_dim)."""
        B = obs_features.shape[0]
        noise = torch.randn(
            B, self.action_pred_horizon, self.wrapper.action_dim,
            device=self.device,
        )
        actions = self.wrapper(
            obs=obs_features,
            initial_noise=noise,
            return_numpy=True,
            act_steps=self.act_steps,
        )
        # actions: (B, act_steps, action_dim) — 返回完整 chunk
        return actions


class PLDSACPolicy:
    """包装 PLD-SAC (PLDActor + base ShortCutFlow) 适配 PolicyProtocol.

    PLD 输出 = clamp(a_base + a_delta, -1, 1)，其中:
    - a_base: 来自 ShortCut Flow base policy（零噪声，确定性）
    - a_delta: 来自 PLD residual actor（确定性）

    Args:
        actor: 加载好的 PLDActor 实例
        base_flow: 加载好的 ShortCutFlowWrapper 实例
        visual_encoder: PlainConv 视觉编码器 (可为 None)
        device: 计算设备
        act_steps: 每步执行的动作数量
        action_dim: 动作维度
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        base_flow,
        visual_encoder: Optional[torch.nn.Module],
        device: torch.device,
        act_steps: int = 8,
        action_dim: int = 7,
    ) -> None:
        self.actor = actor
        self.base_flow = base_flow
        self.visual_encoder = visual_encoder
        self.device = device
        self.act_steps = act_steps
        self.action_dim = action_dim

    @torch.no_grad()
    def get_actions(self, obs_features: torch.Tensor) -> np.ndarray:
        """obs_features: (N_env, obs_horizon * feat_dim)."""
        if not isinstance(obs_features, torch.Tensor):
            obs_features = torch.as_tensor(obs_features, dtype=torch.float32,
                                           device=self.device)
        obs_features = obs_features.to(self.device)
        B = obs_features.shape[0]

        # 1. 残差动作 (确定性)
        dist = self.actor(obs_features)
        a_delta = dist.mean  # (B, act_steps * action_dim)
        a_delta_3d = a_delta.view(B, self.act_steps, self.action_dim)

        # 2. Base flow 动作 (零噪声 = 确定性)
        zero_noise = torch.zeros_like(a_delta_3d)
        a_base = self.base_flow(
            obs=obs_features,
            initial_noise=zero_noise,
            return_numpy=False,
            act_steps=self.act_steps,
        )  # (B, act_steps, action_dim)

        # 3. 对齐时间维度
        n_actual = a_base.shape[1]
        if n_actual < self.act_steps:
            a_delta_3d = a_delta_3d[:, :n_actual, :]

        # 4. 合成动作 — 返回完整 chunk (B, act_steps, action_dim)
        a_bar = torch.clamp(a_base + a_delta_3d, -1.0, 1.0)
        return a_bar.cpu().numpy()


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class CollectorConfig:
    """P1.1 Rollout 数据收集配置."""

    # 环境
    env_id: str = "PickCube-v1"
    """ManiSkill 任务 ID"""

    num_envs: int = 64
    """并行环境数 (GPU 向量化)"""

    camera_width: int = 128
    """相机分辨率宽 (ManiSkill 默认 128, 需与训练时一致)"""

    camera_height: int = 128
    """相机分辨率高 (ManiSkill 默认 128, 需与训练时一致)"""

    max_episode_steps: int = 200
    """每幕最大步数"""

    num_episodes: int = 50
    """目标采集幕数 (不足时继续直到达到)"""

    sim_backend: str = "physx_cuda"
    """仿真后端: physx_cuda 或 physx_cpu"""

    control_mode: str = "pd_ee_delta_pose"
    """控制模式"""

    obs_horizon: int = 2
    """观测历史窗口大小"""

    act_steps: int = 8
    """每次策略调用实际执行的步数"""

    # 策略
    use_random_policy: bool = False
    """是否使用随机策略 (True: 随机; False: ShortCut Flow)"""

    checkpoint_path: str = ""
    """ShortCut Flow checkpoint 路径 (use_random_policy=False 时必填)"""

    include_rgb: bool = True
    """是否使用 RGB 观测 (False 时退化为 state-only)"""

    visual_feature_dim: int = 256
    """PlainConv 输出维度"""

    # 帧率控制
    frame_skip: int = 3
    """ManiSkill 控制频率 / 保存频率 = frame_skip (下采样到 ~5Hz)"""

    # GPU
    gpu_id: int = 4
    """使用的 GPU"""

    # 输出
    output_dir: str = "data/vlaw/rollouts/iter0"
    """HDF5 数据保存目录"""

    source_tag: str = "real"
    """数据来源标签: 'real' 或 'synthetic'"""

    task_instruction: str = ""
    """任务语言描述 (空则自动从 env_id 推断)"""

    # 调试
    dry_run: bool = False
    """True: 只运行 3 幕，不保存文件"""

    verbose: bool = True
    """是否打印详细日志"""


# ---------------------------------------------------------------------------
# 观测处理工具
# ---------------------------------------------------------------------------

def _np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Tensor → numpy, 保持在 CPU."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def _get_render_frame(
    env,
    N: int,
    H: int,
    W: int,
    rgb_base: np.ndarray,
) -> np.ndarray:
    """获取 render 帧 (第二视角), 缩放到目标分辨率.

    Args:
        env: ManiSkill 环境
        N: 环境数量
        H: 目标高度
        W: 目标宽度
        rgb_base: 备用帧 (失败时返回同 shape 零数组)

    Returns:
        rgb_render: (N, H, W, 3) uint8
    """
    try:
        render_out = env.render()
        if isinstance(render_out, torch.Tensor):
            render_out = render_out.cpu().numpy()
        if render_out is not None:
            if render_out.ndim == 4:
                rgb_render = render_out.astype(np.uint8)
            else:
                rgb_render = np.stack([render_out] * N).astype(np.uint8)
            if rgb_render.shape[1] != H or rgb_render.shape[2] != W:
                from PIL import Image as PILImage
                resized = np.zeros((N, H, W, 3), dtype=np.uint8)
                for i in range(N):
                    resized[i] = np.asarray(
                        PILImage.fromarray(rgb_render[i]).resize(
                            (W, H), PILImage.BILINEAR
                        )
                    )
                rgb_render = resized
        else:
            rgb_render = np.zeros_like(rgb_base)
    except Exception:
        rgb_render = np.zeros_like(rgb_base)
    return rgb_render


def extract_raw_frames(
    obs: dict,
    camera_height: int,
    camera_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """从 ManiSkill obs 提取 base_camera RGB 和 render 帧.

    Args:
        obs: ManiSkill step obs dict
        camera_height: 目标相机高度
        camera_width: 目标相机宽度

    Returns:
        (rgb_base, rgb_render): 各自 uint8 (N, H, W, 3) 数组
    """
    sensor_data = obs.get("sensor_data", {})
    base_rgb = sensor_data.get("base_camera", {}).get("rgb")

    if base_rgb is not None:
        rgb_base = _np(base_rgb).astype(np.uint8)  # (N, H, W, 3)
    else:
        N = next(iter(obs.get("agent", {}).values())).shape[0]
        rgb_base = np.zeros((N, camera_height, camera_width, 3), dtype=np.uint8)

    return rgb_base


def extract_agent_state(obs: dict) -> np.ndarray:
    """拼接 obs['agent'] 所有数组 → (N, state_dim) float32."""
    arrays = []
    for v in obs.get("agent", {}).values():
        arr = _np(v).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        arrays.append(arr)
    for v in obs.get("extra", {}).values():
        arr = _np(v).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.dtype == bool or arr.dtype == np.bool_:
            arr = arr.astype(np.float32)
        arrays.append(arr)
    if not arrays:
        raise RuntimeError("obs['agent'] is empty — cannot extract state")
    return np.concatenate(arrays, axis=-1)


def build_obs_features(
    obs: dict,
    state_history: np.ndarray,      # (N, obs_horizon, state_dim)
    rgb_history: np.ndarray,        # (N, obs_horizon, H, W, 3) uint8
    visual_encoder: Optional[torch.nn.Module],
    include_rgb: bool,
    visual_feature_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """拼接视觉特征 + 状态特征 → (N, obs_horizon * feat_dim)."""
    from rlft.datasets.data_utils import encode_observations
    N, T = state_history.shape[:2]

    obs_seq: dict = {
        "state": torch.from_numpy(state_history).to(device),   # (N, T, state_dim)
    }

    if include_rgb and visual_encoder is not None:
        # 转为 (N, T, C, H, W) float
        rgb_t = torch.from_numpy(rgb_history).to(device)  # (N, T, H, W, C)
        # ManiSkill 某些任务会把多相机 RGB 拼在通道维上 (C=6)。
        # 现有 PLD/ShortCutFlow checkpoint 默认使用 3 通道 PlainConv，
        # 这里保留前 3 通道，保持与训练侧输入一致。
        if rgb_t.shape[-1] > 3:
            rgb_t = rgb_t[..., :3]
        rgb_t = rgb_t.permute(0, 1, 4, 2, 3).float() / 255.0  # → (N, T, 3, H, W)
        obs_seq["rgb"] = rgb_t

    obs_features = encode_observations(
        obs_seq=obs_seq,
        visual_encoder=visual_encoder,
        include_rgb=include_rgb,
        device=device,
        flatten=True,   # (N, T * feat_dim)
    )
    return obs_features


# ---------------------------------------------------------------------------
# 轨迹存储
# ---------------------------------------------------------------------------

class Trajectory:
    """单条轨迹的内存缓冲区."""

    def __init__(self, env_idx: int) -> None:
        self.env_idx = env_idx
        self.rgb_base: list[np.ndarray] = []
        self.rgb_render: list[np.ndarray] = []
        self.state: list[np.ndarray] = []
        self.obs_agent: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.env_success: list[bool] = []

    def append(
        self,
        rgb_base: np.ndarray,       # (H, W, 3)
        rgb_render: np.ndarray,     # (H, W, 3)
        state: np.ndarray,          # (state_dim,)
        obs_agent: np.ndarray,      # (agent_dim,)
        action: np.ndarray,         # (action_dim,)
        success: bool,
    ) -> None:
        self.rgb_base.append(rgb_base)
        self.rgb_render.append(rgb_render)
        self.state.append(state)
        self.obs_agent.append(obs_agent)
        self.actions.append(action)
        self.env_success.append(success)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """合并成 numpy 数组字典."""
        return {
            "rgb_base": np.stack(self.rgb_base).astype(np.uint8),
            "rgb_render": np.stack(self.rgb_render).astype(np.uint8),
            "state": np.stack(self.state).astype(np.float32),
            "obs_agent": np.stack(self.obs_agent).astype(np.float32),
            "actions": np.stack(self.actions).astype(np.float32),
            "env_success": np.array(self.env_success, dtype=bool),
        }

    def __len__(self) -> int:
        return len(self.actions)


# ---------------------------------------------------------------------------
# 主收集器
# ---------------------------------------------------------------------------

class VLAWDataCollector:
    """ManiSkill GPU 向量化环境的 VLAW 轨迹收集器.

    支持:
    - ShortCut Flow 策略 rollout (用于 D_real)
    - 随机策略 rollout (用于探索基线)
    - HDF5 格式保存

    Args:
        cfg: 收集器配置
    """

    def __init__(self, cfg: CollectorConfig) -> None:
        self.cfg = cfg
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VLAW-P1.1] 设备: {self.device} (GPU {cfg.gpu_id})")

    # ------------------------------------------------------------------
    # 环境创建
    # ------------------------------------------------------------------

    def _make_env(self):
        """创建 ManiSkill GPU 向量化环境.

        使用 FlattenRGBDObservationWrapper 确保 obs 格式与训练时一致:
        obs = {"state": (N, state_dim), "rgb": (N, H, W, 3)}
        不设 sensor_configs — 使用 ManiSkill 默认分辨率 (128×128)。
        """
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

        env_kwargs: dict = dict(
            obs_mode="rgbd",
            render_mode="rgb_array",
            control_mode=self.cfg.control_mode,
        )
        if self.cfg.max_episode_steps:
            env_kwargs["max_episode_steps"] = self.cfg.max_episode_steps

        if self.cfg.sim_backend == "physx_cpu":
            env = gym.make(self.cfg.env_id, **env_kwargs)
        else:
            env = gym.make(
                self.cfg.env_id,
                num_envs=self.cfg.num_envs,
                sim_backend=self.cfg.sim_backend,
                **env_kwargs,
            )

        # 应用 FlattenRGBDObservationWrapper，与训练管线保持一致
        if self.cfg.include_rgb:
            env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

        print(f"[VLAW-P1.1] 环境创建: {self.cfg.env_id} "
              f"num_envs={self.cfg.num_envs} "
              f"action_space={env.action_space.shape} "
              f"(FlattenRGBD={'ON' if self.cfg.include_rgb else 'OFF'})")
        return env

    # ------------------------------------------------------------------
    # 策略加载
    # ------------------------------------------------------------------

    def _load_policy(self, env) -> tuple:
        """加载策略，返回 (policy, visual_encoder)."""
        if self.cfg.use_random_policy:
            print("[VLAW-P1.1] 使用随机策略")
            return RandomPolicy(env.action_space), None

        if not self.cfg.checkpoint_path:
            raise ValueError(
                "use_random_policy=False 时必须提供 checkpoint_path"
            )

        # 检测 checkpoint 类型:
        # - PLD-SAC: dict 含 "agent" + "config" 两个 key
        # - ShortCut Flow: dict 含 "agent" 但 agent 的子 key 以 velocity_net 开头
        _raw = torch.load(
            self.cfg.checkpoint_path, map_location="cpu", weights_only=False
        )
        _is_pld = (
            isinstance(_raw, dict)
            and "agent" in _raw
            and "config" in _raw  # PLD-SAC checkpoint 包含 config 字典
        )

        from rlft.networks import PlainConv
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy

        if _is_pld:
            print(f"[VLAW-P1.1] 检测到 PLD-SAC checkpoint: {self.cfg.checkpoint_path}")
            return self._load_pld_policy(_raw)

        print(f"[VLAW-P1.1] 加载 ShortCut Flow: {self.cfg.checkpoint_path}")

        wrapper, visual_encoder, _state_dim = load_shortcut_flow_policy(
            self.cfg.checkpoint_path,
            visual_encoder_class=PlainConv if self.cfg.include_rgb else None,
            obs_horizon=self.cfg.obs_horizon,
            visual_feature_dim=self.cfg.visual_feature_dim,
            include_rgb=self.cfg.include_rgb,
            device=str(self.device),
        )
        policy = ShortCutFlowPolicy(
            wrapper=wrapper,
            visual_encoder=visual_encoder,
            device=self.device,
            include_rgb=self.cfg.include_rgb,
            obs_horizon=self.cfg.obs_horizon,
            act_steps=self.cfg.act_steps,
        )
        return policy, visual_encoder

    def _load_pld_policy(self, ckpt: dict) -> tuple:
        """从 PLD-SAC checkpoint 加载策略，返回 (PLDSACPolicy, visual_encoder)."""
        from rlft.networks import PlainConv
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy
        from rlft.algorithms.online_rl.pld_sac import PLDActor

        cfg_dict = ckpt["config"]
        act_steps: int = int(cfg_dict.get("act_steps", 8))
        pred_horizon: int = int(cfg_dict.get("pred_horizon", 16))
        action_dim: int = int(cfg_dict.get("action_dim", 7))
        visual_feature_dim: int = int(cfg_dict.get("visual_feature_dim", 256))
        obs_horizon: int = int(cfg_dict.get("obs_horizon", 2))
        use_ema: bool = bool(cfg_dict.get("use_ema", True))
        action_scale: float = float(cfg_dict.get("action_scale", 0.3))
        base_ckpt_path: str = cfg_dict.get("checkpoint", "")

        # 加载 visual encoder (PlainConv)
        visual_encoder: Optional[torch.nn.Module] = None
        if self.cfg.include_rgb and "visual_encoder" in ckpt:
            ve_sd = ckpt["visual_encoder"]
            # 自动检测 pool_feature_map: 若 fc.0.weight 输入维度 == 128，说明使用了 AdaptiveMaxPool
            fc_in_dim = ve_sd["fc.0.weight"].shape[1] if "fc.0.weight" in ve_sd else None
            pool_feature_map = (fc_in_dim == 128)
            visual_encoder = PlainConv(
                in_channels=3, out_dim=visual_feature_dim,
                pool_feature_map=pool_feature_map,
            ).to(self.device)
            visual_encoder.load_state_dict(ve_sd)
            visual_encoder.eval()

        # 推断 obs_dim 和 residual_dim 从 checkpoint weights
        agent_sd = ckpt["agent"]
        obs_dim: int = int(agent_sd["actor.trunk.0.weight"].shape[1])
        residual_dim: int = int(agent_sd["actor.mean_head.weight"].shape[0])
        # hidden_dims: 通过 trunk 层数推断
        # trunk: [Linear, Tanh, Linear, Tanh, ...] → 偶数索引是 Linear
        hidden_dims = []
        idx = 0
        while f"actor.trunk.{idx}.weight" in agent_sd:
            hidden_dims.append(int(agent_sd[f"actor.trunk.{idx}.weight"].shape[0]))
            idx += 2  # 跳过 Tanh

        actor = PLDActor(
            obs_dim=obs_dim,
            residual_dim=residual_dim,
            hidden_dims=hidden_dims,
            action_scale=action_scale,
        ).to(self.device)
        actor.load_state_dict(
            {k.removeprefix("actor."): v for k, v in agent_sd.items()
             if k.startswith("actor.")}
        )
        actor.eval()

        # 加载 base ShortCut Flow
        if not base_ckpt_path:
            raise ValueError("PLD checkpoint 缺少 config.checkpoint (base policy 路径)")
        # 支持相对路径（相对项目根）
        _root = Path(__file__).resolve().parents[3]
        base_path = Path(base_ckpt_path)
        if not base_path.is_absolute():
            base_path = _root / base_path
        print(f"[VLAW-P1.1] 加载 base ShortCut Flow: {base_path}")

        base_flow, _base_ve, _state_dim = load_shortcut_flow_policy(
            str(base_path),
            visual_encoder_class=None,  # visual_encoder 已从 PLD ckpt 加载
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            visual_feature_dim=visual_feature_dim,
            include_rgb=self.cfg.include_rgb,
            use_ema=use_ema,
            device=str(self.device),
        )

        policy = PLDSACPolicy(
            actor=actor,
            base_flow=base_flow,
            visual_encoder=visual_encoder,
            device=self.device,
            act_steps=act_steps,
            action_dim=action_dim,
        )
        print(f"[VLAW-P1.1] PLD-SAC 加载完成 obs_dim={obs_dim} "
              f"residual_dim={residual_dim} act_steps={act_steps} "
              f"pred_horizon={pred_horizon} use_ema={use_ema}")
        return policy, visual_encoder

    # ------------------------------------------------------------------
    # 主收集循环
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        policy=None,
        visual_encoder: Optional[torch.nn.Module] = None,
    ) -> list[dict]:
        """执行 rollout 并返回轨迹列表.

        Args:
            policy: 策略 (None 时自动加载)
            visual_encoder: 视觉编码器 (None 时自动加载)

        Returns:
            List of trajectory dicts, 每个 dict 包含 numpy 数组
        """
        cfg = self.cfg
        env = self._make_env()

        if policy is None:
            policy, visual_encoder = self._load_policy(env)

        N = cfg.num_envs
        H, W = cfg.camera_height, cfg.camera_width
        target_episodes = 3 if cfg.dry_run else cfg.num_episodes

        task_instruction = cfg.task_instruction or cfg.env_id.replace("-v1", "")

        completed_trajs: list[dict] = []
        active_trajs: list[Trajectory] = [Trajectory(i) for i in range(N)]

        # 历史缓冲 (用于 obs_horizon)
        state_dim = None
        state_history: Optional[np.ndarray] = None   # (N, T, state_dim)
        rgb_history: Optional[np.ndarray] = None     # (N, T, H, W, 3)

        # 重置
        obs, _ = env.reset(seed=42)
        step = 0
        episode_count = 0
        t_start = time.perf_counter()

        print(f"[VLAW-P1.1] 开始收集: 目标 {target_episodes} 条轨迹")

        while episode_count < target_episodes:
            # ---- 提取当前 obs ----
            # FlattenRGBDObservationWrapper 返回 {"state": ..., "rgb": ...}
            if isinstance(obs, dict) and "state" in obs and "rgb" in obs:
                rgb_base = _np(obs["rgb"]).astype(np.uint8)       # (N, H, W, 3)
                agent_state = _np(obs["state"]).astype(np.float32)  # (N, state_dim)
            else:
                # Fallback: 未包装的 raw obs
                rgb_base = extract_raw_frames(obs, H, W)   # (N, H, W, 3)
                agent_state = extract_agent_state(obs)      # (N, state_dim)

            # 获取 render 帧 (第二视角)
            rgb_render = _get_render_frame(env, N, H, W, rgb_base)

            # 初始化历史缓冲 — 模仿 FrameStack reset 行为：
            # 用当前帧（而非零）填充所有 obs_horizon 位置
            if state_dim is None:
                state_dim = agent_state.shape[-1]
                # 用第一帧 obs 填充所有历史位置（与 FrameStack.reset 一致）
                state_history = np.tile(
                    agent_state[:, np.newaxis, :],
                    (1, cfg.obs_horizon, 1),
                )  # (N, obs_horizon, state_dim)
                rgb_history = np.tile(
                    rgb_base[:, np.newaxis, :, :, :],
                    (1, cfg.obs_horizon, 1, 1, 1),
                )  # (N, obs_horizon, H, W, 3)

            # 滚动历史（chunk 第 0 步的历史更新）
            state_history = np.roll(state_history, shift=-1, axis=1)
            state_history[:, -1, :] = agent_state
            rgb_history = np.roll(rgb_history, shift=-1, axis=1)
            rgb_history[:, -1, :] = rgb_base

            # ---- 生成动作 chunk ----
            if isinstance(policy, RandomPolicy):
                action_chunk = policy.get_actions(
                    torch.zeros(N, cfg.obs_horizon, 1)
                )  # obs_features 未使用; (N, action_dim)
                # RandomPolicy 返回 (N, action_dim) → 扩展为 (N, 1, action_dim)
                if action_chunk.ndim == 2:
                    action_chunk = action_chunk[:, np.newaxis, :]
            else:
                obs_features = build_obs_features(
                    obs=obs,
                    state_history=state_history,
                    rgb_history=rgb_history,
                    visual_encoder=visual_encoder,
                    include_rgb=cfg.include_rgb,
                    visual_feature_dim=cfg.visual_feature_dim,
                    device=self.device,
                )
                action_chunk = policy.get_actions(obs_features)
            # action_chunk: (N, act_steps, action_dim)  或 (N, 1, action_dim) for Random

            # ---- 执行 action chunk 中的每一步 ----
            chunk_len = action_chunk.shape[1]
            for t_chunk in range(chunk_len):
                actions = action_chunk[:, t_chunk, :]  # (N, action_dim)

                # 对 sub-step > 0: 重新提取观测帧 + 更新历史
                if t_chunk > 0:
                    if isinstance(obs, dict) and "state" in obs and "rgb" in obs:
                        rgb_base = _np(obs["rgb"]).astype(np.uint8)
                        agent_state = _np(obs["state"]).astype(np.float32)
                    else:
                        rgb_base = extract_raw_frames(obs, H, W)
                        agent_state = extract_agent_state(obs)

                    rgb_render = _get_render_frame(env, N, H, W, rgb_base)

                    # 更新历史（保证下一次 policy query 时 history 正确）
                    state_history = np.roll(state_history, shift=-1, axis=1)
                    state_history[:, -1, :] = agent_state
                    rgb_history = np.roll(rgb_history, shift=-1, axis=1)
                    rgb_history[:, -1, :] = rgb_base

                # ---- 步进环境 ----
                obs, _reward, terminated, truncated, info = env.step(actions)
                terminated_np = _np(terminated).astype(bool)
                truncated_np = _np(truncated).astype(bool)
                done = np.logical_or(terminated_np, truncated_np)
                step += 1

                # ---- 提取 success ----
                success_arr: np.ndarray
                if (
                    "final_info" in info
                    and isinstance(info["final_info"], dict)
                    and "episode" in info["final_info"]
                    and "success_once" in info["final_info"]["episode"]
                ):
                    s = info["final_info"]["episode"]["success_once"]
                    success_arr = _np(s).astype(bool)
                elif "success" in info:
                    s = info["success"]
                    success_arr = _np(s).astype(bool)
                elif "episode" in info and "success" in info["episode"]:
                    s = info["episode"]["success"]
                    success_arr = _np(s).astype(bool)
                else:
                    success_arr = np.zeros(N, dtype=bool)

                # ---- 记录帧 (frame_skip 下采样) ----
                if step % cfg.frame_skip == 0:
                    for i in range(N):
                        active_trajs[i].append(
                            rgb_base=rgb_base[i],
                            rgb_render=rgb_render[i],
                            state=agent_state[i],
                            obs_agent=agent_state[i],
                            action=actions[i],
                            success=bool(success_arr[i]),
                        )

                # ---- 幕结束处理 ----
                if np.any(done):
                    done_indices = np.where(done)[0]

                    for i in done_indices:
                        traj = active_trajs[i]
                        if len(traj) > 0:
                            traj_dict = traj.to_arrays()
                            traj_dict["task_instruction"] = task_instruction
                            traj_dict["source"] = cfg.source_tag
                            completed_trajs.append(traj_dict)
                            episode_count += 1
                            if cfg.verbose:
                                final_success = traj_dict["env_success"].any()
                                print(
                                    f"[VLAW-P1.1] 幕 {episode_count:4d}/{target_episodes} "
                                    f"env={i:3d} T={len(traj):4d} "
                                    f"success={'✅' if final_success else '❌'}"
                                )

                        # 重置该环境的历史缓冲
                        # ManiSkill auto-reset 后 obs 已包含新幕的初始帧
                        active_trajs[i] = Trajectory(i)
                        # 用当前帧填充（模仿 FrameStack.reset 行为）
                        if isinstance(obs, dict) and "state" in obs and "rgb" in obs:
                            _st = _np(obs["state"]).astype(np.float32)
                            _rgb = _np(obs["rgb"]).astype(np.uint8)
                            for _th in range(cfg.obs_horizon):
                                state_history[i, _th] = _st[i]
                                rgb_history[i, _th] = _rgb[i]
                        else:
                            state_history[i] = 0
                            rgb_history[i] = 0

                        if episode_count >= target_episodes:
                            break

                    if episode_count >= target_episodes:
                        break

                    # 对齐 evaluate()：在 episode 结束时打断当前 chunk
                    break

            if episode_count >= target_episodes:
                break

        elapsed = time.perf_counter() - t_start
        sr = sum(t["env_success"].any() for t in completed_trajs) / max(len(completed_trajs), 1)
        print(f"[VLAW-P1.1] 收集完成: {len(completed_trajs)} 条轨迹, "
              f"成功率={sr:.1%}, 耗时={elapsed:.1f}s")

        env.close()
        return completed_trajs

    # ------------------------------------------------------------------
    # HDF5 保存
    # ------------------------------------------------------------------

    def save_hdf5(
        self,
        trajectories: list[dict],
        output_path: Optional[str] = None,
    ) -> Path:
        """将轨迹列表保存为 HDF5 文件.

        文件结构:
            /traj_0000/   ← 第 0 条轨迹
                rgb_base       (T, H, W, 3) uint8
                rgb_render     (T, H, W, 3) uint8
                state          (T, state_dim) float32
                obs_agent      (T, agent_dim) float32
                actions        (T, action_dim) float32
                env_success    (T,) bool
                task_instruction  str attr
                source            str attr
            /traj_0001/ ...
            /meta/
                num_trajectories  int attr
                success_rate      float attr
                env_id            str attr
                camera_hw         str attr  "H,W"

        Args:
            trajectories: collect_rollouts 返回的轨迹列表
            output_path: HDF5 文件路径 (None 则自动生成)

        Returns:
            保存路径 Path 对象
        """
        if output_path is None:
            out_dir = Path(self.cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            output_path = str(
                out_dir / f"{self.cfg.env_id}_{self.cfg.source_tag}_{ts}.h5"
            )

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        success_count = sum(t["env_success"].any() for t in trajectories)
        sr = success_count / max(len(trajectories), 1)

        with h5py.File(str(out_path), "w") as f:
            # --- meta ---
            meta = f.create_group("meta")
            meta.attrs["num_trajectories"] = len(trajectories)
            meta.attrs["success_rate"] = float(sr)
            meta.attrs["env_id"] = self.cfg.env_id
            meta.attrs["camera_hw"] = f"{self.cfg.camera_height},{self.cfg.camera_width}"
            meta.attrs["source"] = self.cfg.source_tag
            meta.attrs["frame_skip"] = self.cfg.frame_skip

            # --- per-trajectory ---
            for idx, traj in enumerate(trajectories):
                grp = f.create_group(f"traj_{idx:04d}")
                for key, arr in traj.items():
                    if isinstance(arr, np.ndarray):
                        grp.create_dataset(
                            key, data=arr,
                            chunks=True, compression="gzip", compression_opts=1,
                        )
                grp.attrs["task_instruction"] = traj.get("task_instruction", "")
                grp.attrs["source"] = traj.get("source", "real")
                grp.attrs["success"] = bool(traj["env_success"].any())

        print(f"[VLAW-P1.1] HDF5 已保存: {out_path} "
              f"({len(trajectories)} 条, 成功率={sr:.1%})")
        return out_path

    # ------------------------------------------------------------------
    # 一站式入口
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """完整数据收集流程: 环境 → 策略 → 收集 → 保存.

        Returns:
            保存的 HDF5 文件路径
        """
        trajs = self.collect_rollouts()
        if self.cfg.dry_run:
            print("[VLAW-P1.1] dry_run=True, 跳过 HDF5 保存")
            return Path("/dev/null")
        return self.save_hdf5(trajs)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = tyro.cli(CollectorConfig)
    collector = VLAWDataCollector(cfg)
    collector.run()
