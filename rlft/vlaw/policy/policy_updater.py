"""VLAW P5.1 — 策略更新器（Weighted Filtered BC）

VLAW 策略更新 = 在 D_real+ ∪ D_syn+ 上做标准 Flow Matching 训练。
D_real+: 真实 rollout 中 VLM 奖励 R=1 的轨迹
D_syn+:  Imagination 生成的 VLM 奖励 R=1 的合成轨迹

权重来源:
  - 二值 mask（vlm_reward=1 → weight=1, 否则 weight=0）
  - 或软标签（VLM P('yes') 概率直接作为 weight）

所属阶段: P5.1 — Weighted Flow Matching 实现
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class PolicyUpdaterConfig:
    """P5.1 策略更新配置."""

    # 起点 checkpoint（应已经过 IL 预训练）
    checkpoint_path: str = "checkpoints/il/best_eval_success_once.pt"
    """ShortCut Flow checkpoint 路径（预训练起点）"""

    # 输出
    output_dir: str = "checkpoints/vlaw/policy/iter1"
    """更新后 checkpoint 保存目录"""

    # 训练超参
    num_steps: int = 2000
    """梯度更新步数"""

    batch_size: int = 64
    """混合 mini-batch 大小"""

    learning_rate: float = 1e-5
    """AdamW 学习率"""

    warmup_steps: int = 100
    """线性 warmup 步数（之后 cosine 衰减）"""

    data_mix_ratio: float = 0.5
    """D_real+ 在混合 batch 中的比例；(1 - ratio) 来自 D_syn+"""

    # GPU
    gpu_id: int = 8
    """训练使用的 GPU ID"""

    # wandb
    use_wandb: bool = True
    """是否启用 wandb 日志"""

    wandb_run_name: str = "vlaw_policy_iter1"
    """wandb run 名称"""

    # 观测 / 动作维度（需与 checkpoint 一致）
    obs_horizon: int = 2
    """观测历史帧数"""

    action_horizon: int = 8
    """预测动作序列长度（pred_horizon）"""

    visual_feature_dim: int = 256
    """PlainConv 视觉编码器输出维度"""

    state_dim: int = 25
    """低维 state 维度（如 LiftPegUpright 为 25）"""

    # 视觉观测
    use_visual_obs: bool = True
    """True=视觉模式（RGB+state），False=仅 state 模式"""

    image_key: str = "rgb_base"
    """HDF5 中的 RGB 数据 key"""

    # UNet 架构超参（需与 checkpoint 一致）
    unet_down_dims: tuple[int, ...] = (64, 128, 256)
    """UNet 各层 channel 维度"""

    unet_step_embed_dim: int = 64
    """UNet timestep embedding 维度"""

    unet_n_groups: int = 8
    """UNet GroupNorm 分组数"""

    # 调试
    dry_run: bool = False
    """True: 用随机数据验证前向传播，不加载真实 checkpoint 和数据"""

    iter_id: int = 1
    """当前迭代 ID，用于 checkpoint 文件命名"""


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------


class VLAWSuccessDataset(Dataset):
    """从 HDF5 文件加载 VLM 标注为成功的轨迹.

    **成功轨迹识别策略**（按优先级）:
        1. group.attrs["vlm_reward"] == 1  — VLM 明确标注成功
        2. group.attrs["success"] == True  — 数据收集器标注的 env_success
        3. dataset["env_success"].any()    — 轨迹内任意帧 success=True

    每条轨迹被切分为多个 (obs_window, action_window) 训练样本：
        - obs 窗口：连续 obs_horizon 帧的 state（float32）
        - action 窗口：紧接其后的 action_horizon 帧动作

    Args:
        hdf5_path: HDF5 文件路径
        obs_horizon: 观测历史长度
        action_horizon: 动作序列长度（pred_horizon）
        source_tag: "real" 或 "synthetic"，存入样本 metadata
        weight: 该数据集内所有样本的默认权重（1.0 = 成功轨迹全权重）
        filter_by_vlm: True 则仅保留 vlm_reward=1 的轨迹；
                       False 则退化为按 env_success 过滤
    """

    def __init__(
        self,
        hdf5_path: str,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        source_tag: str = "real",
        weight: float = 1.0,
        filter_by_vlm: bool = True,
        image_key: str = "rgb_base",
        use_visual_obs: bool = True,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.source_tag = source_tag
        self.weight = weight
        self.filter_by_vlm = filter_by_vlm
        self.image_key = image_key
        self.use_visual_obs = use_visual_obs

        # 预扫描 HDF5：收集所有成功轨迹的样本索引
        self._samples: list[tuple[str, int]] = []  # (traj_key, start_frame_idx)
        self._state_dim: int = 0
        self._action_dim: int = 0

        self._scan_file()

    def _is_success_traj(self, grp: h5py.Group) -> bool:
        """判断轨迹是否为成功轨迹."""
        # 优先使用 vlm_reward 属性
        if "vlm_reward" in grp.attrs:
            return int(grp.attrs["vlm_reward"]) == 1
        # 次优先：success 属性
        if "success" in grp.attrs:
            return bool(grp.attrs["success"])
        # 降级：env_success 数据集
        if "env_success" in grp:
            return bool(np.asarray(grp["env_success"]).any())
        # 无法判断，保守起见返回 False
        return False

    def _scan_file(self) -> None:
        """扫描 HDF5 文件，建立样本索引."""
        if not self.hdf5_path.exists():
            print(f"[VLAW-P5.1] 警告: HDF5 文件不存在: {self.hdf5_path}")
            return

        min_len = self.obs_horizon + self.action_horizon

        with h5py.File(str(self.hdf5_path), "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj_")]
            for key in traj_keys:
                grp = f[key]
                if not self._is_success_traj(grp):
                    continue
                if "state" not in grp or "actions" not in grp:
                    continue
                if self.use_visual_obs and self.image_key not in grp:
                    continue

                T = grp["state"].shape[0]
                if T < min_len:
                    continue

                # 读取维度信息（仅一次）
                if self._state_dim == 0:
                    self._state_dim = grp["state"].shape[-1]
                if self._action_dim == 0:
                    self._action_dim = grp["actions"].shape[-1]

                # 滑动窗口切片
                for start in range(T - min_len + 1):
                    self._samples.append((key, start))

        print(
            f"[VLAW-P5.1] VLAWSuccessDataset [{self.source_tag}] "
            f"{self.hdf5_path.name}: "
            f"{len(self._samples)} 样本 "
            f"(state_dim={self._state_dim}, action_dim={self._action_dim})"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        """返回单个训练样本.

        Returns:
            dict with:
                "obs":     Tensor(obs_horizon, state_dim)  — 观测序列
                "actions": Tensor(action_horizon, action_dim) — 动作序列
                "weight":  float — 样本权重
                "source":  str   — "real" 或 "synthetic"
                "rgb":     Tensor(obs_horizon, 3, 192, 192) — 视觉观测（仅 use_visual_obs=True）
        """
        traj_key, start = self._samples[idx]

        with h5py.File(str(self.hdf5_path), "r") as f:
            grp = f[traj_key]
            obs_end = start + self.obs_horizon
            act_end = obs_end + self.action_horizon

            state = np.asarray(grp["state"][start:obs_end], dtype=np.float32)
            actions = np.asarray(grp["actions"][obs_end:act_end], dtype=np.float32)

            # 视觉观测：(obs_horizon, H, W, C) uint8 → (obs_horizon, C, H, W) float32
            rgb_np: Optional[np.ndarray] = None
            if self.use_visual_obs and self.image_key in grp:
                rgb_np = np.asarray(
                    grp[self.image_key][start:obs_end], dtype=np.float32
                )  # (T, H, W, C)

        obs_tensor = torch.from_numpy(state)        # (obs_horizon, state_dim)
        act_tensor = torch.from_numpy(actions)      # (action_horizon, action_dim)

        result = {
            "obs": obs_tensor,
            "actions": act_tensor,
            "weight": float(self.weight),
            "source": self.source_tag,
        }

        if rgb_np is not None:
            # NHWC → NCHW, 归一化到 [0, 1]
            rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2) / 255.0
            result["rgb"] = rgb_tensor  # (obs_horizon, C, H, W)

        return result

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


def _collate_fn(batch: list[dict]) -> dict:
    """自定义 collate，处理 source 字符串字段和可选的 rgb."""
    obs = torch.stack([b["obs"] for b in batch])
    actions = torch.stack([b["actions"] for b in batch])
    weights = torch.tensor([b["weight"] for b in batch], dtype=torch.float32)
    sources = [b["source"] for b in batch]
    result = {"obs": obs, "actions": actions, "weights": weights, "sources": sources}
    if "rgb" in batch[0]:
        rgb = torch.stack([b["rgb"] for b in batch])
        result["rgb"] = rgb  # (B, obs_horizon, C, H, W)
    return result


# ---------------------------------------------------------------------------
# 核心更新器
# ---------------------------------------------------------------------------


class VLAWPolicyUpdater:
    """VLAW P5.1 策略更新器.

    将 ShortCut Flow 在 D_real+ ∪ D_syn+ 上做 Weighted Filtered BC 更新。

    Args:
        config: PolicyUpdaterConfig
    """

    def __init__(self, config: PolicyUpdaterConfig) -> None:
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"[VLAW-P5.1] PolicyUpdater 初始化，device={self.device}")

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load_policy(self) -> nn.Module:
        """加载 ShortCut Flow checkpoint.

        在 use_visual_obs=True 时：
          - 构建 PlainConv 并从 ckpt["visual_encoder"] 加载权重
          - global_cond_dim = obs_horizon * (visual_feature_dim + state_dim)
          - velocity_net 从 ckpt["agent"] strict=False 加载（global_cond_dim
            变化导致的 size mismatch 会自动跳过）

        Returns:
            加载好的 ShortCutFlowAgent（位于 self.device）
        """
        from rlft.algorithms.il.shortcut_flow import ShortCutFlowAgent
        from rlft.networks import PlainConv, ShortCutVelocityUNet1D

        cfg = self.config
        ckpt_path = Path(cfg.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

        print(f"[VLAW-P5.1] 加载 checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=self.device)

        # 支持 canonical checkpoint 格式（build_checkpoint 产出）
        agent_sd = ckpt.get("agent") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        cfg_dict: dict = ckpt.get("config", {})

        # 从 checkpoint config 或默认值推断维度
        action_dim: int = cfg_dict.get("action_dim", 7)
        pred_horizon: int = cfg_dict.get("pred_horizon", cfg.action_horizon)
        obs_horizon: int = cfg_dict.get("obs_horizon", cfg.obs_horizon)

        # ---------- 视觉编码器 ----------
        if cfg.use_visual_obs:
            global_cond_dim = obs_horizon * (cfg.visual_feature_dim + cfg.state_dim)

            # 创建 PlainConv 并加载权重
            visual_encoder = PlainConv(
                in_channels=3,
                out_dim=cfg.visual_feature_dim,
                pool_feature_map=True,  # 与 train_maniskill 一致
            ).to(self.device)

            ve_sd = ckpt.get("visual_encoder")
            if ve_sd is not None:
                visual_encoder.load_state_dict(ve_sd, strict=True)
                print(f"[VLAW-P5.1] visual_encoder 权重已从 checkpoint 加载")
            else:
                print("[VLAW-P5.1] 警告: checkpoint 无 visual_encoder, 使用随机初始化")

            visual_encoder.eval()  # 冻结 BN/Dropout
            for p in visual_encoder.parameters():
                p.requires_grad = False
            self.visual_encoder: Optional[nn.Module] = visual_encoder
        else:
            global_cond_dim = obs_horizon * cfg.state_dim
            self.visual_encoder = None

        print(f"[VLAW-P5.1] global_cond_dim={global_cond_dim}")

        # ---------- velocity net（随机初始化新维度） ----------
        velocity_net = ShortCutVelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=cfg.unet_step_embed_dim,
            down_dims=tuple(cfg.unet_down_dims),
            n_groups=cfg.unet_n_groups,
        )
        agent = ShortCutFlowAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            device=str(self.device),
        )

        # strict=False: global_cond_dim 变化导致的首层 Linear size mismatch 会被跳过
        missing, unexpected = agent.load_state_dict(agent_sd, strict=False)
        if missing:
            print(f"[VLAW-P5.1] 加载 agent 时缺失 keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[VLAW-P5.1] 加载 agent 时多余 keys ({len(unexpected)}): {unexpected[:5]}...")

        agent = agent.to(self.device)
        print(
            f"[VLAW-P5.1] 模型加载完成 "
            f"(action_dim={action_dim}, global_cond_dim={global_cond_dim}, "
            f"visual={'ON' if cfg.use_visual_obs else 'OFF'})"
        )
        return agent

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def create_mixed_dataloader(
        self,
        real_success_dirs: str | list[str],
        syn_success_dirs: str | list[str] = "",
        demo_dirs: str | list[str] | None = None,
    ) -> DataLoader:
        """按 data_mix_ratio 混合真实、合成与演示成功数据集.

        Args:
            real_success_dirs: D_real+ HDF5 目录（str 或 list[str]）
            syn_success_dirs:  D_syn+ HDF5 目录（str 或 list[str]），可为空
            demo_dirs: D_demo 目录（高质量，weight=1.0）；None 或空则跳过

        Returns:
            混合 DataLoader（无限循环，shuffle=True）
        """
        cfg = self.config

        # 统一转 list
        if isinstance(real_success_dirs, str):
            real_success_dirs = [real_success_dirs] if real_success_dirs else []
        if isinstance(syn_success_dirs, str):
            syn_success_dirs = [syn_success_dirs] if syn_success_dirs else []
        if demo_dirs is None:
            demo_dirs = []
        elif isinstance(demo_dirs, str):
            demo_dirs = [demo_dirs] if demo_dirs else []

        real_datasets: list[VLAWSuccessDataset] = []
        for d in real_success_dirs:
            real_datasets.extend(self._load_hdf5_dir(d, source_tag="real", weight=1.0))

        syn_datasets: list[VLAWSuccessDataset] = []
        for d in syn_success_dirs:
            syn_datasets.extend(self._load_hdf5_dir(d, source_tag="synthetic", weight=1.0))

        demo_datasets: list[VLAWSuccessDataset] = []
        for d in demo_dirs:
            demo_datasets.extend(self._load_hdf5_dir(d, source_tag="demo", weight=1.0))

        all_datasets = real_datasets + syn_datasets + demo_datasets
        if not all_datasets:
            raise RuntimeError(
                f"未找到任何数据集！real_dirs={real_success_dirs}, "
                f"syn_dirs={syn_success_dirs}, demo_dirs={demo_dirs}"
            )

        combined = ConcatDataset(all_datasets)
        n_samples = len(combined)
        n_real = sum(len(d) for d in real_datasets)
        n_syn = sum(len(d) for d in syn_datasets)
        n_demo = sum(len(d) for d in demo_datasets)
        print(
            f"[VLAW-P5.1] 混合数据集: {n_samples} 样本 "
            f"(real={n_real}, syn={n_syn}, demo={n_demo})"
        )

        # 当样本数不足 batch_size 时自动降级
        actual_bs = min(cfg.batch_size, n_samples)
        drop_last = n_samples > actual_bs  # 只在够2个batch时drop_last
        if actual_bs < cfg.batch_size:
            print(
                f"[VLAW-P5.1] 警告: 样本({n_samples}) < batch_size({cfg.batch_size}), "
                f"自动降为 batch_size={actual_bs}, drop_last={drop_last}"
            )

        loader = DataLoader(
            combined,
            batch_size=actual_bs,
            shuffle=True,
            num_workers=min(4, max(1, n_samples // actual_bs)),
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=_collate_fn,
            persistent_workers=n_samples > actual_bs,
        )
        return loader

    def _load_hdf5_dir(
        self,
        directory: str,
        source_tag: str,
        weight: float,
    ) -> list[VLAWSuccessDataset]:
        """扫描目录下所有 .h5 / .hdf5 文件，构建 VLAWSuccessDataset 列表."""
        dir_path = Path(directory)
        datasets: list[VLAWSuccessDataset] = []

        if not dir_path.exists():
            print(f"[VLAW-P5.1] 警告: 目录不存在，跳过: {dir_path}")
            return datasets

        for h5file in sorted(dir_path.glob("**/*.h5")) + sorted(dir_path.glob("**/*.hdf5")):
            ds = VLAWSuccessDataset(
                hdf5_path=str(h5file),
                obs_horizon=self.config.obs_horizon,
                action_horizon=self.config.action_horizon,
                source_tag=source_tag,
                weight=weight,
                image_key=self.config.image_key,
                use_visual_obs=self.config.use_visual_obs,
            )
            if len(ds) > 0:
                datasets.append(ds)

        return datasets

    # ------------------------------------------------------------------
    # 学习率 schedule
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lr_schedule(
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Cosine decay with linear warmup."""

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(warmup_steps, 1)
            progress = float(current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # 主更新逻辑
    # ------------------------------------------------------------------

    def update(
        self,
        real_success_dirs: str | list[str],
        syn_success_dirs: str | list[str] = "",
        demo_dirs: str | list[str] | None = None,
    ) -> dict:
        """执行策略更新（Weighted Filtered BC）.

        Args:
            real_success_dirs: D_real+ 数据目录（str 或 list[str]）
            syn_success_dirs:  D_syn+ 数据目录（str 或 list[str]），可为空
            demo_dirs: D_demo 目录（高质量演示数据）；None 或空则跳过

        Returns:
            dict: {
                "final_loss": float,
                "num_steps": int,
                "checkpoint_path": str,
            }
        """
        cfg = self.config
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
                    config=vars(cfg),
                    resume="allow",
                )
                print(f"[VLAW-P5.1] wandb 已初始化: {run.url}")
            except Exception as e:
                print(f"[VLAW-P5.1] wandb 初始化失败，跳过: {e}")
                run = None

        # ---- 加载模型 ----
        if cfg.dry_run:
            policy = self._build_dry_run_policy()
        else:
            policy = self.load_policy()

        policy.train()

        # ---- 优化器 & schedule ----
        # 只优化 velocity_net 参数；visual_encoder 已冻结
        optimizer = torch.optim.AdamW(
            policy.velocity_net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=1e-5,
        )
        scheduler = self._build_lr_schedule(optimizer, cfg.warmup_steps, cfg.num_steps)

        # ---- 数据 ----
        if cfg.dry_run:
            dataloader = self._build_dry_run_dataloader()
        else:
            dataloader = self.create_mixed_dataloader(
                real_success_dirs, syn_success_dirs, demo_dirs
            )

        data_iter = iter(dataloader)

        print(f"[VLAW-P5.1] 开始策略更新，共 {cfg.num_steps} 步")
        running_loss = 0.0
        final_loss = 0.0

        for step in range(1, cfg.num_steps + 1):
            # 无限循环数据
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            state = batch["obs"].to(self.device)        # (B, obs_horizon, state_dim)
            actions = batch["actions"].to(self.device)    # (B, action_horizon, action_dim)
            weights = batch["weights"].to(self.device)    # (B,)

            # ---------- 构建 obs_cond ----------
            if cfg.use_visual_obs and "rgb" in batch:
                rgb = batch["rgb"].to(self.device)       # (B, obs_horizon, C, H, W)
                B, T = rgb.shape[:2]
                rgb_flat = rgb.reshape(B * T, *rgb.shape[2:])  # (B*T, C, H, W)
                with torch.no_grad():
                    visual_feat = self.visual_encoder(rgb_flat)  # (B*T, visual_feature_dim)
                visual_feat = visual_feat.view(B, T, -1)        # (B, T, visual_feature_dim)
                obs_features = torch.cat([visual_feat, state], dim=-1)  # (B, T, V+S)
            else:
                obs_features = state  # (B, obs_horizon, state_dim)

            optimizer.zero_grad()
            loss_dict = policy.compute_weighted_loss(obs_features, actions, weights)
            loss = loss_dict["loss"]
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy.velocity_net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            policy.update_ema()

            running_loss += loss.item()
            final_loss = loss.item()

            # ---- 日志 ----
            if step % 100 == 0:
                avg_loss = running_loss / 100
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"[VLAW-P5.1] step={step:4d}/{cfg.num_steps} "
                    f"loss={avg_loss:.5f} lr={lr_now:.2e}"
                )
                if run is not None:
                    run.log({
                        "train/loss": avg_loss,
                        "train/flow_loss": loss_dict["flow_loss"].item(),
                        "train/shortcut_loss": loss_dict["shortcut_loss"].item(),
                        "train/lr": lr_now,
                        "train/step": step,
                    })
                running_loss = 0.0

            # ---- 中间 checkpoint ----
            if step % 500 == 0:
                mid_path = out_dir / f"policy_iter{cfg.iter_id}_step{step}.pt"
                self._save_checkpoint(policy, optimizer, step, str(mid_path))

        # ---- 最终 checkpoint ----
        final_ckpt = str(out_dir / f"policy_iter{cfg.iter_id}.pt")
        self._save_checkpoint(policy, optimizer, cfg.num_steps, final_ckpt)

        if run is not None:
            run.finish()

        print(
            f"[VLAW-P5.1] 策略更新完成！"
            f"final_loss={final_loss:.5f}, checkpoint={final_ckpt}"
        )
        return {
            "final_loss": final_loss,
            "num_steps": cfg.num_steps,
            "checkpoint_path": final_ckpt,
        }

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        path: str,
    ) -> None:
        """保存 checkpoint（canonical 格式，含 visual_encoder + ema_agent）."""
        agent_sd = policy.state_dict()
        ckpt: dict = {
            "agent": agent_sd,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": vars(self.config),
        }

        # 提取 EMA 权重（与 base ckpt 格式一致）
        # velocity_net_ema.* → velocity_net.* 以匹配 load_shortcut_flow_policy 期望
        ema_agent = {
            k.replace("velocity_net_ema.", "velocity_net."): v
            for k, v in agent_sd.items()
            if k.startswith("velocity_net_ema.")
        }
        if ema_agent:
            ckpt["ema_agent"] = ema_agent
            print(f"[VLAW-P5.1] EMA agent 已提取 ({len(ema_agent)} keys)")

        if self.visual_encoder is not None:
            ckpt["visual_encoder"] = self.visual_encoder.state_dict()
        torch.save(ckpt, path)
        print(f"[VLAW-P5.1] Checkpoint 已保存: {path} (step={step})")

    def _build_dry_run_policy(self) -> nn.Module:
        """dry_run 模式：用随机初始化的策略（跳过 checkpoint 加载）."""
        from rlft.algorithms.il.shortcut_flow import ShortCutFlowAgent
        from rlft.networks import PlainConv, ShortCutVelocityUNet1D

        cfg = self.config
        action_dim = 7
        obs_horizon = cfg.obs_horizon
        pred_horizon = cfg.action_horizon

        if cfg.use_visual_obs:
            global_cond_dim = obs_horizon * (cfg.visual_feature_dim + cfg.state_dim)
            visual_encoder = PlainConv(
                in_channels=3,
                out_dim=cfg.visual_feature_dim,
                pool_feature_map=True,
            ).to(self.device)
            visual_encoder.eval()
            for p in visual_encoder.parameters():
                p.requires_grad = False
            self.visual_encoder = visual_encoder
        else:
            global_cond_dim = obs_horizon * cfg.state_dim
            self.visual_encoder = None

        velocity_net = ShortCutVelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=cfg.unet_step_embed_dim,
            down_dims=tuple(cfg.unet_down_dims),
            n_groups=cfg.unet_n_groups,
        )
        agent = ShortCutFlowAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            device=str(self.device),
        ).to(self.device)
        print(
            f"[VLAW-P5.1] dry_run: 随机初始化策略 "
            f"(global_cond_dim={global_cond_dim}, visual={'ON' if cfg.use_visual_obs else 'OFF'})"
        )
        return agent

    def _build_dry_run_dataloader(self) -> DataLoader:
        """dry_run 模式：用随机 Tensor 数据集."""
        cfg = self.config
        action_dim = 7
        state_dim = cfg.state_dim
        obs_horizon = cfg.obs_horizon
        action_horizon = cfg.action_horizon
        use_visual = cfg.use_visual_obs
        n_samples = cfg.batch_size * 20

        class _RandDataset(Dataset):
            def __len__(self_) -> int:
                return n_samples

            def __getitem__(self_, idx: int) -> dict:
                sample: dict = {
                    "obs": torch.randn(obs_horizon, state_dim),
                    "actions": torch.randn(action_horizon, action_dim),
                    "weight": 1.0,
                    "source": "dry_run",
                }
                if use_visual:
                    sample["rgb"] = torch.rand(obs_horizon, 3, 192, 192)
                return sample

        loader = DataLoader(
            _RandDataset(),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=_collate_fn,
        )
        print(
            f"[VLAW-P5.1] dry_run: 随机数据集，{n_samples} 样本 "
            f"(visual={'ON' if use_visual else 'OFF'})"
        )
        return loader


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tyro

    @dataclass
    class _CLI:
        config: PolicyUpdaterConfig = field(default_factory=PolicyUpdaterConfig)
        real_success_dirs: list[str] = field(
            default_factory=lambda: ["data/vlaw/success/real"]
        )
        syn_success_dirs: list[str] = field(
            default_factory=lambda: []
        )
        demo_dirs: list[str] = field(
            default_factory=lambda: []
        )

    args = tyro.cli(_CLI)
    updater = VLAWPolicyUpdater(args.config)
    result = updater.update(args.real_success_dirs, args.syn_success_dirs, args.demo_dirs)
    print(f"[VLAW-P5.1] 完成: {result}")
