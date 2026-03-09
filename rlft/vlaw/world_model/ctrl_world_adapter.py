"""VLAW P2.1 — Ctrl-World 推理封装层 (ManiSkill 适配).

为 Imagination 引擎 (P4.2) 提供统一接口，屏蔽 Ctrl-World 内部细节.

典型用法:
    adapter = CtrlWorldAdapter(args, ckpt_path="checkpoints/vlaw/world_model/...")
    pred_latents = adapter.rollout(
        obs_latents,   # (T_hist, 4, 48, 24) — 历史帧 latent
        actions,       # (T_hist + T_pred, 7) — 绝对 EE pose (未归一化)
        instruction,   # str — 任务描述
    )  # → (N_CAMS, T_pred, 4, lat_h_single, lat_w) float32 tensor

WM Action Conditioning — 对齐 DROID:
    Ctrl-World 预训练使用 **绝对 EE 位姿** 做 action conditioning.
    "action" 字段语义: [tcp_x, tcp_y, tcp_z, euler_rx, euler_ry, euler_rz, gripper_norm]
    归一化统计量 state_01/state_99 来自 EE pose 的 p1/p99 分位数.

DROID vs ManiSkill 关键差异:
    - latent shape:  (T, 4, 72, 40)  →  (T, 4, 48, 24)  (3-cam → 2-cam)
    - rearrange:     m=3,n=1         →  m=2,n=1
    - height:        int(192*3)=576  →  int(192*2)=384
    - width:         320             →  192
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn

# 将 ctrl_world/ 目录加入 Python 路径 (运行时自动解析)
_THIS_FILE = Path(__file__).resolve()
_CTRL_WORLD_ROOT = _THIS_FILE.parents[2] / "ctrl_world"
if str(_CTRL_WORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_CTRL_WORLD_ROOT))


class CtrlWorldAdapter:
    """Ctrl-World 世界模型的 ManiSkill 适配封装.

    提供简洁的 rollout() 接口供 Imagination 引擎调用.
    内部处理动作归一化、latent 形状适配、pipeline 调用细节.

    Args:
        args: wm_args_maniskill 实例 (含模型路径、分辨率等)
        ckpt_path: 可选, 覆盖 args.ckpt_path 指定的 checkpoint 路径
        device: 目标设备, 默认 cuda
        dtype: 推理精度, 默认 float16
    """

    # ManiSkill latent 网格尺寸 (2 相机纵向拼接后, 经 VAE 8× 下采样)
    # 单相机 192h×192w → VAE → 24h×24w; 2-cam concat → 48h×24w
    LATENT_H = 48
    LATENT_W = 24
    N_CAMS = 2

    def __init__(
        self,
        args,
        ckpt_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.args = args
        self.device = torch.device(device)
        self.dtype = dtype

        # ---- 加载归一化统计量 (EE pose percentiles) ----
        stat_path = getattr(args, "data_stat_path", None)
        if stat_path and Path(stat_path).exists():
            with open(stat_path, "r") as f:
                stat = json.load(f)
            self.state_p01 = np.array(stat["state_01"], dtype=np.float32)[None, :]  # (1, 7)
            self.state_p99 = np.array(stat["state_99"], dtype=np.float32)[None, :]
        else:
            print(f"[CtrlWorldAdapter] ⚠️  未找到 stat.json ({stat_path}), EE pose 不归一化")
            self.state_p01 = np.zeros((1, args.action_dim), dtype=np.float32)
            self.state_p99 = np.ones((1, args.action_dim), dtype=np.float32)

        # ---- 加载 Ctrl-World 模型 ----
        resolved_ckpt = ckpt_path or args.ckpt_path
        self._load_model(resolved_ckpt)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        obs_latents: torch.Tensor,
        actions: np.ndarray | torch.Tensor,
        instruction: str = "",
    ) -> torch.Tensor:
        """单次世界模型 rollout, 预测未来帧 latent.

        Args:
            obs_latents: (T_hist + T_pred, 4, 48, 24) 历史+当前帧 latent (float16/32)
                         前 num_history 帧作为历史条件, 当前帧(索引 num_history)作为条件帧.
                         T_hist = args.num_history, T_pred = args.num_frames
            actions:     (T_hist + T_pred, 7) **绝对 EE pose** (未归一化)
                         [tcp_x, tcp_y, tcp_z, euler_rx, euler_ry, euler_rz, gripper_norm]
            instruction: 任务文本描述

        Returns:
            pred_latents: (N_CAMS, T_pred, 4, lat_h_single, lat_w) float32
                          latent in VAE space (scaled), 每个相机独立
        """
        # ---- 输入验证与格式化 ----
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = np.asarray(actions, dtype=np.float32)

        if isinstance(obs_latents, np.ndarray):
            obs_latents = torch.from_numpy(obs_latents)
        obs_latents = obs_latents.to(self.device).to(self.dtype)

        window_len = self.args.num_history + self.args.num_frames
        assert obs_latents.shape[0] >= window_len, (
            f"obs_latents 帧数不足: 需要 {window_len}, 实际 {obs_latents.shape[0]}"
        )
        assert obs_latents.shape[1:] == (4, self.LATENT_H, self.LATENT_W), (
            f"latent 形状错误: {obs_latents.shape[1:]} != (4, {self.LATENT_H}, {self.LATENT_W})"
        )

        # ---- 动作归一化 ----
        actions_norm = self._normalize_action(actions_np[:window_len])
        action_cond = (
            torch.tensor(actions_norm, dtype=self.dtype)
            .unsqueeze(0)
            .to(self.device)
        )  # (1, T, 7)

        # ---- 条件帧 & 历史 ----
        # 取 num_history 帧作为历史条件
        num_hist = self.args.num_history
        image_cond = obs_latents[num_hist]          # (4, 48, 24) — 当前帧作为 img cond
        his_cond_list = [
            obs_latents[i].unsqueeze(0)             # (1, 4, 48, 24)
            for i in range(num_hist)
        ]
        his_cond = (
            torch.stack(his_cond_list, dim=1)       # (1, num_hist, 4, 48, 24)
            if his_cond_list
            else None
        )

        # ---- Action Encoder (含文本条件) ----
        text_list = [instruction]
        if self.args.text_cond:
            action_hidden = self.model.action_encoder(
                action_cond,
                text_list,
                self.model.tokenizer,
                self.model.text_encoder,
            )
        else:
            action_hidden = self.model.action_encoder(action_cond)

        # ---- CtrlWorld 扩散 Pipeline 推理 ----
        from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

        pipeline = self.model.pipeline
        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=image_cond.unsqueeze(0),                  # (1, 4, 48, 24)
            text=action_hidden,
            width=self.args.width,                          # 192
            height=self.args.height,                        # 384 (2-cam concat raw height)
            num_frames=self.args.num_frames,
            history=his_cond,
            num_inference_steps=getattr(self.args, "num_inference_steps", 25),
            decode_chunk_size=self.args.decode_chunk_size,
            max_guidance_scale=getattr(self.args, "guidance_scale", 3.0),
            fps=self.args.fps,
            motion_bucket_id=self.args.motion_bucket_id,
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=True,
        )  # pred_latents: (1, T_pred, 4, 48, 24)  (合并 m*n 后)

        # ---- rearrange: 拆分 2 个相机 ----
        # (B, T_pred, 4, H_concat, W) → (B*N_CAMS, T_pred, 4, H_single, W)
        pred_latents = einops.rearrange(
            pred_latents,
            "b f c (m h) (n w) -> (b m n) f c h w",
            m=self.N_CAMS,
            n=1,
        )  # (N_CAMS, T_pred, 4, 24, 24)

        return pred_latents.float()

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.Tensor,
        decode_chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """将 latent 解码为 RGB 像素帧 (uint8).

        Args:
            latents: (..., 4, lat_h, lat_w) VAE-scaled latent
            decode_chunk_size: VAE 分块解码大小

        Returns:
            frames: (..., H, W, 3) uint8, 值域 [0, 255]
        """
        vae = self.model.pipeline.vae
        chunk = decode_chunk_size or self.args.decode_chunk_size
        orig_shape = latents.shape[:-3]
        flat = latents.reshape(-1, *latents.shape[-3:]).to(self.device).to(self.dtype)

        decoded = []
        for i in range(0, flat.shape[0], chunk):
            x = flat[i : i + chunk] / vae.config.scaling_factor
            decoded.append(vae.decode(x, num_frames=x.shape[0]).sample)
        decoded = torch.cat(decoded, dim=0)
        decoded = (decoded / 2.0 + 0.5).clamp(0, 1) * 255
        decoded = decoded.float().cpu().numpy()     # (N, C, H, W)
        decoded = decoded.transpose(0, 2, 3, 1).astype(np.uint8)  # (N, H, W, C)
        return decoded.reshape(*orig_shape, *decoded.shape[1:])

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_model(self, ckpt_path: str) -> None:
        """加载 CrtlWorld 模型并迁移到目标设备."""
        from models.ctrl_world import CrtlWorld

        print(f"[CtrlWorldAdapter] 加载模型权重: {ckpt_path}")
        self.model = CrtlWorld(self.args)

        if ckpt_path and Path(ckpt_path).exists():
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[CtrlWorldAdapter] 权重加载 OK: {ckpt_path}")
        else:
            print(f"[CtrlWorldAdapter] ⚠️  未找到 checkpoint ({ckpt_path}), 使用随机初始化权重")

        self.model.to(self.device).to(self.dtype)
        self.model.eval()

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """将绝对 EE pose 归一化到 [-1, 1] (对齐 DROID percentile 归一化)."""
        eps = 1e-8
        ndata = 2.0 * (action - self.state_p01) / (self.state_p99 - self.state_p01 + eps) - 1.0
        return np.clip(ndata, -1.0, 1.0).astype(np.float32)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """反归一化 EE pose (推理后恢复真实值)."""
        return (action + 1.0) / 2.0 * (self.state_p99 - self.state_p01) + self.state_p01


# ---------------------------------------------------------------------------
# 快速验证 (仅当 checkpoint 和 stat.json 已存在时有意义)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # 加入项目根目录
    _ROOT = str(_THIS_FILE.parents[2])
    sys.path.insert(0, _ROOT)

    from ctrl_world.config import wm_args_maniskill  # type: ignore

    args = wm_args_maniskill()
    print("=== 测试 CtrlWorldAdapter ===")
    print(f"ckpt_path: {args.ckpt_path}")
    print(f"data_stat_path: {args.data_stat_path}")

    # 用随机 latent + action 跑一次 rollout (不验证质量, 只验证不 OOM)
    adapter = CtrlWorldAdapter(args, device="cuda", dtype=torch.float16)

    window_len = args.num_history + args.num_frames
    rand_latents = torch.randn(window_len, 4, 48, 24, dtype=torch.float16)
    rand_actions = np.random.randn(window_len, args.action_dim).astype(np.float32)

    print(f"输入 latent: {rand_latents.shape}, action: {rand_actions.shape}")
    pred = adapter.rollout(rand_latents, rand_actions, instruction="pick up the cube")
    print(f"预测 latent: {pred.shape}")  # 期望: (2, num_frames, 4, 24, 24)

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"峰值显存: {mem:.2f} GB")

    print("✅ CtrlWorldAdapter 推理 OK")
