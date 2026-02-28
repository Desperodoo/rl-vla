"""VLAW VAE 编码管线.

P1.2 阶段: 将 data_collector 收集的 HDF5 轨迹经过 Ctrl-World VAE 批量编码，
生成 latent_concat 张量写回 HDF5 供世界模型训练使用。

数据流:
    HDF5[rgb_base (T,H,W,3) + rgb_render (T,H,W,3)]
    → 垂直拼接 (T, 2H, W, 3)
    → AutoencoderKL encode
    → latent_concat (T, 4, 2H/8, W/8) float16
    → 写回同一 HDF5 / 新 HDF5

分辨率约定 (192×192 双相机):
    concat_hw = (384, 192, 3)    [2H × W × 3]
    latent_hw = (4, 48, 24)      [C × 2H/8 × W/8]

所属阶段: P1.2 — VAE 编码管线
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import tyro
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """P1.2 VAE 编码管线配置."""

    # 输入/输出
    input_dir: str = "data/vlaw/rollouts/iter0"
    """包含 HDF5 文件的输入目录"""

    output_dir: str = "data/vlaw/encoded/iter0"
    """编码结果输出目录 (空字符串则原地写回 HDF5)"""

    in_place: bool = False
    """True: 将 latent_concat 数据集写回原 HDF5; False: 写到新 HDF5"""

    # VAE
    vae_model_id: str = "stabilityai/sd-vae-ft-mse"
    """HuggingFace 模型 ID 或本地路径"""

    # VLAW MODIFICATION: 移除硬编码用户路径，改为空字符串并自动查找 HF 缓存
    vae_local_path: str = ""
    """VAE 本地缓存路径 (优先于 vae_model_id); 空字符串则尝试自动从 HF 缓存查找，找不到则用 vae_model_id 在线下载"""

    # 图像
    camera_height: int = 192
    """单张相机高度"""

    camera_width: int = 192
    """单张相机宽度"""

    concat_mode: str = "vertical"
    """相机拼接方式: 'vertical' (垂直) 或 'horizontal' (水平)"""

    # 批次
    batch_size: int = 16
    """每次 VAE encode 的帧数 (受 VRAM 限制)"""

    # GPU
    gpu_id: int = 4
    """使用的 GPU"""

    # 动作归一化
    compute_action_stats: bool = False
    """True: 计算并保存 action mean/std"""

    action_stats_output: str = "data/vlaw/action_stats.json"
    """动作统计量输出路径"""

    # 调试
    dry_run: bool = False
    """True: 只处理第一个 HDF5 的前 3 条轨迹"""

    verbose: bool = True


# ---------------------------------------------------------------------------
# VAE 工具
# ---------------------------------------------------------------------------

def load_vae(
    vae_model_id: str,
    vae_local_path: str,
    device: torch.device,
) -> "AutoencoderKL":
    """加载 stabilityai/sd-vae-ft-mse AutoencoderKL.

    Args:
        vae_model_id: HuggingFace 模型 ID
        vae_local_path: 本地缓存路径 (优先)
        device: 目标设备

    Returns:
        eval 模式下的 AutoencoderKL
    """
    from diffusers import AutoencoderKL

    # VLAW MODIFICATION: 支持自动查找 HF 缓存，避免硬编码路径
    if not vae_local_path:
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(vae_model_id, filename="config.json")
            if cached is not None:
                import os
                # try_to_load_from_cache 返回文件路径，取其目录作为模型路径
                vae_local_path = str(os.path.dirname(cached))
                print(f"[VLAW-P1.2] 自动发现 VAE 缓存: {vae_local_path}")
        except Exception:
            pass

    load_from = vae_local_path if vae_local_path else vae_model_id
    print(f"[VLAW-P1.2] 加载 VAE: {load_from}")
    vae = AutoencoderKL.from_pretrained(load_from, torch_dtype=torch.float32, low_cpu_mem_usage=False)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"[VLAW-P1.2] VAE 加载完成 (device={device})")
    return vae


def concat_cameras(
    rgb_base: np.ndarray,   # (T, H, W, 3) uint8
    rgb_render: np.ndarray,  # (T, H, W, 3) uint8
    mode: str = "vertical",
) -> np.ndarray:
    """将两相机帧拼接.

    Args:
        rgb_base: 基础相机帧 (T, H, W, 3)
        rgb_render: 渲染相机帧 (T, H, W, 3)
        mode: 'vertical' → (T, 2H, W, 3); 'horizontal' → (T, H, 2W, 3)

    Returns:
        拼接后 uint8 数组
    """
    if rgb_base.shape[-1] != rgb_render.shape[-1]:
        min_c = min(rgb_base.shape[-1], rgb_render.shape[-1])
        rgb_base = rgb_base[..., :min_c]
        rgb_render = rgb_render[..., :min_c]

    if rgb_base.shape[-1] > 3:
        rgb_base = rgb_base[..., :3]
    if rgb_render.shape[-1] > 3:
        rgb_render = rgb_render[..., :3]

    if mode == "vertical":
        return np.concatenate([rgb_base, rgb_render], axis=1)
    elif mode == "horizontal":
        return np.concatenate([rgb_base, rgb_render], axis=2)
    else:
        raise ValueError(f"Unknown concat_mode: {mode}")


@torch.no_grad()
def encode_frames_batch(
    vae,
    frames: np.ndarray,   # (T, cH, cW, 3) uint8
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """批量 VAE 编码帧序列.

    Args:
        vae: AutoencoderKL
        frames: uint8 帧序列 (T, cH, cW, 3), cH/cW 必须为 8 的倍数
        batch_size: 每批帧数
        device: 计算设备

    Returns:
        latent (T, 4, cH/8, cW/8) float16 numpy 数组
    """
    T, cH, cW, _ = frames.shape
    latents = []

    for start in range(0, T, batch_size):
        batch = frames[start : start + batch_size]   # (B, cH, cW, 3)
        B = batch.shape[0]

        # [0,255] → [0,1] → [-1,1]; NHWC → NCHW
        x = torch.from_numpy(batch).float().to(device)
        x = x.permute(0, 3, 1, 2) / 255.0   # (B, 3, cH, cW)
        x = x * 2.0 - 1.0                    # [-1, 1]

        dist = vae.encode(x).latent_dist
        z = dist.sample() * vae.config.scaling_factor  # (B, 4, cH/8, cW/8)
        latents.append(z.cpu().to(torch.float16).numpy())

    return np.concatenate(latents, axis=0)  # (T, 4, cH/8, cW/8)


# ---------------------------------------------------------------------------
# 动作统计
# ---------------------------------------------------------------------------

def compute_action_stats_from_dir(
    input_dir: str,
) -> dict:
    """从目录下所有 HDF5 计算动作 mean/std.

    Returns:
        {'mean': [action_dim], 'std': [action_dim]}
    """
    all_actions = []
    for h5_path in sorted(Path(input_dir).glob("**/*.h5")):
        with h5py.File(str(h5_path), "r") as f:
            for key in f.keys():
                if key.startswith("traj_"):
                    if "actions" in f[key]:
                        all_actions.append(f[key]["actions"][:])

    if not all_actions:
        raise RuntimeError(f"No action data found in {input_dir}")

    actions = np.concatenate(all_actions, axis=0)  # (N_total, action_dim)
    mean = actions.mean(axis=0).tolist()
    std = actions.std(axis=0).tolist()
    print(f"[VLAW-P1.2] 动作统计: mean={mean[:4]}..., std={std[:4]}...")
    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# 主管线
# ---------------------------------------------------------------------------

class VLAWDataPipeline:
    """VLAW 数据编码与转换管线.

    负责:
    1. 读取 data_collector 输出的 HDF5 轨迹
    2. 将两相机帧拼接后经 VAE 编码 → latent_concat
    3. 将 latent 写回 HDF5 (原地或新文件)
    4. 可选: 计算动作归一化统计量

    Args:
        cfg: 管线配置
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VLAW-P1.2] 设备: {self.device} (GPU {cfg.gpu_id})")
        self._vae = None

    @property
    def vae(self):
        if self._vae is None:
            self._vae = load_vae(
                self.cfg.vae_model_id,
                self.cfg.vae_local_path,
                self.device,
            )
        return self._vae

    # ------------------------------------------------------------------
    # 单文件处理
    # ------------------------------------------------------------------

    def encode_single_hdf5(
        self,
        src_path: Path,
        dst_path: Optional[Path] = None,
        max_trajs: Optional[int] = None,
    ) -> Path:
        """编码单个 HDF5 文件中的所有轨迹.

        Args:
            src_path: 源 HDF5 文件
            dst_path: 目标 HDF5 文件 (None 则原地)
            max_trajs: 最大处理轨迹数 (dry_run 用)

        Returns:
            输出文件路径
        """
        cfg = self.cfg

        # 计算目标分辨率
        if cfg.concat_mode == "vertical":
            tgt_h = cfg.camera_height * 2
            tgt_w = cfg.camera_width
        else:
            tgt_h = cfg.camera_height
            tgt_w = cfg.camera_width * 2

        # 确认可被 8 整除
        if tgt_h % 8 != 0 or tgt_w % 8 != 0:
            raise ValueError(
                f"拼接后分辨率 ({tgt_h}, {tgt_w}) 不能被 8 整除，"
                "VAE 需要输入尺寸为 8 的倍数"
            )

        lat_h, lat_w = tgt_h // 8, tgt_w // 8
        lat_shape = (4, lat_h, lat_w)

        if dst_path is None:
            # 原地写回
            dst_path = src_path
            in_place = True
        else:
            in_place = False

        # --- 读源文件，编码，写目标 ---
        t0 = time.perf_counter()

        if not in_place:
            import shutil
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_path), str(dst_path))

        traj_keys: list[str] = []
        with h5py.File(str(src_path), "r") as f_src:
            traj_keys = sorted(
                [k for k in f_src.keys() if k.startswith("traj_")]
            )

        if max_trajs is not None:
            traj_keys = traj_keys[:max_trajs]

        with h5py.File(str(dst_path), "a") as f_dst:
            for i, tkey in enumerate(traj_keys):
                with h5py.File(str(src_path), "r") as f_src:
                    grp = f_src[tkey]
                    rgb_base = grp["rgb_base"][:]    # (T, H, W, 3) uint8
                    rgb_render = grp["rgb_render"][:]  # (T, H, W, 3) uint8

                # 拼接
                concat_frames = concat_cameras(rgb_base, rgb_render, cfg.concat_mode)
                # (T, tgt_h, tgt_w, 3)

                # VAE encode
                latent = encode_frames_batch(
                    self.vae, concat_frames, cfg.batch_size, self.device
                )
                # latent: (T, 4, lat_h, lat_w) float16

                if latent.shape[1:] != lat_shape:
                    raise RuntimeError(
                        f"Latent shape 不符: {latent.shape[1:]} vs 期望 {lat_shape}"
                    )

                # 写入 HDF5
                grp_dst = f_dst[tkey]
                if "latent_concat" in grp_dst:
                    del grp_dst["latent_concat"]
                grp_dst.create_dataset(
                    "latent_concat",
                    data=latent,
                    chunks=True,
                    compression="gzip",
                    compression_opts=1,
                )
                grp_dst.attrs["latent_shape"] = str(latent.shape)

                if cfg.verbose:
                    T = latent.shape[0]
                    print(f"[VLAW-P1.2] {tkey}: T={T} → latent {latent.shape}")

        elapsed = time.perf_counter() - t0
        print(f"[VLAW-P1.2] {len(traj_keys)} 条轨迹编码完成 ({elapsed:.1f}s)")
        return dst_path

    # ------------------------------------------------------------------
    # 目录级批量处理
    # ------------------------------------------------------------------

    def encode_trajectories(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> list[Path]:
        """批量编码目录下所有 HDF5 文件.

        Args:
            input_dir: 源目录 (None 则用 cfg.input_dir)
            output_dir: 目标目录 (None 则用 cfg.output_dir 或原地)

        Returns:
            所有输出文件路径列表
        """
        cfg = self.cfg
        src_dir = Path(input_dir or cfg.input_dir)
        dst_dir_str = output_dir or cfg.output_dir

        h5_files = sorted(src_dir.glob("**/*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"在 {src_dir} 下未找到 HDF5 文件")

        max_trajs = 3 if cfg.dry_run else None
        if cfg.dry_run:
            h5_files = h5_files[:1]

        out_paths: list[Path] = []
        for h5_path in h5_files:
            if dst_dir_str and not cfg.in_place:
                rel = h5_path.relative_to(src_dir)
                dst_path = Path(dst_dir_str) / rel
                dst_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                dst_path = None  # 原地

            out = self.encode_single_hdf5(h5_path, dst_path, max_trajs=max_trajs)
            out_paths.append(out)

        return out_paths

    # ------------------------------------------------------------------
    # 动作统计入口
    # ------------------------------------------------------------------

    def compute_action_stats(
        self,
        traj_dir: Optional[str] = None,
    ) -> dict:
        """计算并保存动作归一化统计量.

        Args:
            traj_dir: 轨迹目录 (None 则用 cfg.input_dir)

        Returns:
            {'mean': [...], 'std': [...]}
        """
        import json

        src_dir = traj_dir or self.cfg.input_dir
        stats = compute_action_stats_from_dir(src_dir)

        out_path = Path(self.cfg.action_stats_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(out_path), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[VLAW-P1.2] 动作统计已保存: {out_path}")
        return stats

    # ------------------------------------------------------------------
    # 一站式入口
    # ------------------------------------------------------------------

    def run(self) -> list[Path]:
        """完整 VAE 编码流程."""
        cfg = self.cfg

        if cfg.compute_action_stats:
            self.compute_action_stats()

        out_paths = self.encode_trajectories()
        print(f"[VLAW-P1.2] 全部完成: {len(out_paths)} 个文件已编码")
        return out_paths


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = tyro.cli(PipelineConfig)
    pipeline = VLAWDataPipeline(cfg)
    pipeline.run()
