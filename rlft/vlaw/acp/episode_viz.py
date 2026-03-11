"""ACP Episode 级可视化 — 双相机图像 + Value 进度条/曲线 → PNG (关键帧) + GIF (连续帧).

每个 episode 生成:
  1. PNG: 选取关键帧（首帧、成功帧/中间帧、末帧等），上方双相机图像，下方 value 曲线
  2. GIF: 逐帧动画，每帧包含双相机图像 + 当前 value 进度指示

用法:
  python -m rlft.vlaw.acp.episode_viz \
    --hdf5_paths data/vlaw/rollouts/pretrained_policy/*.h5 \
    --output_dir docs/vlaw/figures/episodes \
    --num_success 1 --num_fail 1

也可作为库被 visualize.py 或 gen_acp_figures.py 调用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

from rlft.vlaw.acp.config import ValueTargetConfig
from rlft.vlaw.acp.value_targets import compute_value_targets

logger = logging.getLogger(__name__)


@dataclass
class EpisodeVizConfig:
    """Episode 可视化配置."""
    # 输入/输出
    hdf5_paths: list[str] = field(default_factory=list)
    output_dir: str = "docs/vlaw/figures/episodes"

    # 采样
    num_success: int = 1
    num_fail: int = 1
    seed: int = 42

    # PNG 关键帧选择
    num_keyframes: int = 6  # PNG 中展示的关键帧数量

    # GIF 设置
    gif_fps: int = 4  # GIF 帧率
    gif_dpi: int = 100

    # Value target 配置（用于无 ACP 标注时自行计算）
    max_episode_length: int = 35

    # 相机 key
    camera_keys: list[str] = field(default_factory=lambda: ["rgb_base", "rgb_render"])
    camera_labels: list[str] = field(default_factory=lambda: ["Base Camera", "Render Camera"])


def load_episode(
    f: h5py.File,
    traj_key: str,
    camera_keys: list[str],
    max_episode_length: int,
) -> dict[str, Any]:
    """从 HDF5 加载单个 episode 的全部数据.

    Returns:
        dict with keys: traj_key, images (dict[cam_key] -> (T,H,W,3) uint8),
        env_success (T,) bool, success bool, length int,
        value_target (T,) float32 (computed or loaded),
        value_pred (T,) float32 or None, source str
    """
    grp = f[traj_key]
    T = grp[camera_keys[0]].shape[0]

    images: dict[str, np.ndarray] = {}
    for ck in camera_keys:
        if ck in grp:
            images[ck] = grp[ck][:]  # (T, H, W, 3) uint8

    env_success = grp["env_success"][:].astype(bool) if "env_success" in grp else np.zeros(T, dtype=bool)
    is_success = bool(np.any(env_success))
    source = str(grp.attrs.get("source", "unknown"))

    # Value target: try loaded ACP annotation first, else compute
    if "acp_value_target" in grp:
        value_target = grp["acp_value_target"][:].astype(np.float32)
    else:
        cfg = ValueTargetConfig()
        value_target = compute_value_targets(
            env_success=env_success,
            episode_length=T,
            max_episode_length=max_episode_length,
            cfg=cfg,
        )

    # Value prediction (only if ACP-annotated)
    value_pred = grp["acp_value_pred"][:].astype(np.float32) if "acp_value_pred" in grp else None

    return {
        "traj_key": traj_key,
        "images": images,
        "env_success": env_success,
        "success": is_success,
        "length": T,
        "value_target": value_target,
        "value_pred": value_pred,
        "source": source,
    }


def select_keyframes(episode: dict[str, Any], num_keyframes: int) -> list[int]:
    """选择关键帧索引: 首帧 + 成功帧(或均匀间隔) + 末帧."""
    T = episode["length"]
    if T <= num_keyframes:
        return list(range(T))

    keyframes: list[int] = [0]  # 首帧

    # 成功帧
    success_frames = np.where(episode["env_success"])[0]
    if len(success_frames) > 0:
        first_success = int(success_frames[0])
        if first_success not in keyframes:
            keyframes.append(first_success)

    # 末帧
    last = T - 1
    if last not in keyframes:
        keyframes.append(last)

    # 用均匀间隔填充剩余
    remaining = num_keyframes - len(keyframes)
    if remaining > 0:
        candidates = [i for i in range(1, T - 1) if i not in keyframes]
        if candidates:
            step = max(1, len(candidates) // (remaining + 1))
            for i in range(1, remaining + 1):
                idx = min(i * step, len(candidates) - 1)
                keyframes.append(candidates[idx])

    keyframes = sorted(set(keyframes))[:num_keyframes]
    return keyframes


def render_keyframes_png(
    episode: dict[str, Any],
    camera_keys: list[str],
    camera_labels: list[str],
    keyframe_indices: list[int],
    output_path: Path,
) -> None:
    """生成 PNG: 上方关键帧双相机图像行, 下方 value 曲线.

    Layout:
      Row 0: camera_0 images at keyframes
      Row 1: camera_1 images at keyframes
      Row 2: value target (+pred) curve spanning full width, with keyframe markers
    """
    n_kf = len(keyframe_indices)
    T = episode["length"]
    n_cams = len(camera_keys)

    fig = plt.figure(figsize=(3.0 * n_kf, 3.0 * n_cams + 3.5))
    gs = gridspec.GridSpec(
        n_cams + 1, n_kf,
        height_ratios=[1] * n_cams + [1.2],
        hspace=0.25, wspace=0.05,
    )

    # -- Camera image rows --
    for cam_idx, (ck, cl) in enumerate(zip(camera_keys, camera_labels)):
        for kf_idx, frame_t in enumerate(keyframe_indices):
            ax = fig.add_subplot(gs[cam_idx, kf_idx])
            if ck in episode["images"]:
                img = episode["images"][ck][frame_t]  # (H,W,3) uint8
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])

            # Top row: frame index label
            if cam_idx == 0:
                is_success_frame = bool(episode["env_success"][frame_t])
                label = f"t={frame_t}"
                if is_success_frame:
                    label += " [S]"
                ax.set_title(label, fontsize=9, fontweight="bold" if is_success_frame else "normal",
                             color="#2E7D32" if is_success_frame else "black")

            # Left column: camera label
            if kf_idx == 0:
                ax.set_ylabel(cl, fontsize=9)

    # -- Value curve row (spanning all columns) --
    ax_val = fig.add_subplot(gs[n_cams, :])
    t_arr = np.arange(T)

    # Value target
    ax_val.plot(t_arr, episode["value_target"], "b-", linewidth=2.0, label="Value Target (GT)", zorder=3)

    # Value prediction (if available)
    if episode["value_pred"] is not None:
        ax_val.plot(t_arr, episode["value_pred"], "r--", linewidth=1.5, label="Value Pred (ACP)", zorder=3)

    # Mark keyframe positions
    kf_vals = episode["value_target"][keyframe_indices]
    ax_val.scatter(keyframe_indices, kf_vals, c="orange", s=60, zorder=5,
                   edgecolors="black", linewidths=0.8, label="Key frames")

    # Mark success frames
    success_frames = np.where(episode["env_success"])[0]
    if len(success_frames) > 0:
        for sf in success_frames:
            ax_val.axvline(sf, color="#4CAF50", linestyle=":", alpha=0.6, linewidth=1.0)
        ax_val.axvline(success_frames[0], color="#4CAF50", linestyle=":", alpha=0.6,
                       linewidth=1.0, label="Success frame")

    ax_val.set_xlim(-0.5, T - 0.5)
    ax_val.set_ylim(-1.05, 0.05)
    ax_val.set_xlabel("Frame", fontsize=10)
    ax_val.set_ylabel("Value", fontsize=10)
    ax_val.legend(fontsize=8, loc="lower right", ncol=2)
    ax_val.grid(True, alpha=0.3)

    # Title
    status = "SUCCESS" if episode["success"] else "FAIL"
    status_color = "#2E7D32" if episode["success"] else "#C62828"
    fig.suptitle(
        f'{episode["traj_key"]} | {status} | source={episode["source"]} | T={T}',
        fontsize=12, fontweight="bold", color=status_color, y=0.98,
    )

    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved keyframe PNG: %s", output_path)


def render_episode_gif(
    episode: dict[str, Any],
    camera_keys: list[str],
    camera_labels: list[str],
    output_path: Path,
    fps: int = 4,
    dpi: int = 100,
) -> None:
    """生成 GIF: 逐帧动画, 每帧包含双相机图像 + value 曲线进度.

    Layout per frame:
      Left:  camera images stacked vertically
      Right: value curve with current-frame marker
    """
    T = episode["length"]
    n_cams = len(camera_keys)
    pil_frames: list[Image.Image] = []

    for frame_t in range(T):
        fig = plt.figure(figsize=(9, 3.2 * n_cams / 2 + 1.0))
        gs = gridspec.GridSpec(n_cams, 2, width_ratios=[1, 1.8], hspace=0.15, wspace=0.2)

        # -- Left: camera images --
        for cam_idx, (ck, cl) in enumerate(zip(camera_keys, camera_labels)):
            ax = fig.add_subplot(gs[cam_idx, 0])
            if ck in episode["images"]:
                ax.imshow(episode["images"][ck][frame_t])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(cl, fontsize=8)

            # env_success indicator on image border
            if episode["env_success"][frame_t]:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#4CAF50")
                    spine.set_linewidth(3)

        # -- Right: value curve (spans all camera rows) --
        ax_val = fig.add_subplot(gs[:, 1])
        t_arr = np.arange(T)

        # Full curve (dim)
        ax_val.plot(t_arr, episode["value_target"], "b-", linewidth=1.5, alpha=0.4)
        # Already-traversed portion (solid)
        if frame_t > 0:
            ax_val.plot(t_arr[:frame_t + 1], episode["value_target"][:frame_t + 1],
                        "b-", linewidth=2.0, label="Value Target")

        if episode["value_pred"] is not None:
            ax_val.plot(t_arr, episode["value_pred"], "r--", linewidth=1.0, alpha=0.3)
            if frame_t > 0:
                ax_val.plot(t_arr[:frame_t + 1], episode["value_pred"][:frame_t + 1],
                            "r--", linewidth=1.5, label="Value Pred")

        # Current frame marker
        cur_val = episode["value_target"][frame_t]
        ax_val.scatter([frame_t], [cur_val], c="orange", s=80, zorder=5,
                       edgecolors="black", linewidths=1.0)

        # Progress text
        progress_pct = (frame_t + 1) / T * 100
        ax_val.text(
            0.98, 0.95,
            f"t={frame_t}/{T-1}  ({progress_pct:.0f}%)\nV={cur_val:.3f}",
            transform=ax_val.transAxes, ha="right", va="top",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        # Success frames
        success_frames = np.where(episode["env_success"])[0]
        for sf in success_frames:
            ax_val.axvline(sf, color="#4CAF50", linestyle=":", alpha=0.5, linewidth=0.8)

        ax_val.set_xlim(-0.5, T - 0.5)
        ax_val.set_ylim(-1.05, 0.05)
        ax_val.set_xlabel("Frame", fontsize=9)
        ax_val.set_ylabel("Value", fontsize=9)
        ax_val.grid(True, alpha=0.3)
        # Legend only when labeled artists exist (frame_t > 0 has plotted segments)
        if frame_t > 0:
            ax_val.legend(fontsize=7, loc="lower right")

        # Title
        status = "SUCCESS" if episode["success"] else "FAIL"
        status_color = "#2E7D32" if episode["success"] else "#C62828"
        fig.suptitle(
            f'{episode["traj_key"]} | {status} | {episode["source"]}',
            fontsize=10, fontweight="bold", color=status_color,
        )

        # Render to PIL image
        fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.10, hspace=0.15, wspace=0.2)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        pil_frames.append(Image.fromarray(buf[:, :, :3]))  # drop alpha
        plt.close(fig)

    # Save GIF
    if pil_frames:
        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            str(output_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        logger.info("Saved episode GIF (%d frames, %d fps): %s", len(pil_frames), fps, output_path)


def sample_episodes(
    hdf5_paths: list[Path],
    camera_keys: list[str],
    num_success: int = 1,
    num_fail: int = 1,
    max_episode_length: int = 35,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从 HDF5 文件中采样成功和失败 episode.

    Returns:
        (success_episodes, fail_episodes) 各为 episode dict 的列表
    """
    rng = np.random.default_rng(seed)
    success_pool: list[tuple[Path, str]] = []
    fail_pool: list[tuple[Path, str]] = []

    for hp in hdf5_paths:
        with h5py.File(str(hp), "r") as f:
            traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
            for tk in traj_keys:
                grp = f[tk]
                env_success = grp["env_success"][:] if "env_success" in grp else np.zeros(1, dtype=bool)
                is_success = bool(np.any(env_success))
                if is_success:
                    success_pool.append((hp, tk))
                else:
                    fail_pool.append((hp, tk))

    # Sample
    def _sample(pool: list[tuple[Path, str]], n: int) -> list[dict]:
        if not pool:
            return []
        n = min(n, len(pool))
        idxs = rng.choice(len(pool), size=n, replace=False)
        episodes = []
        for i in idxs:
            hp, tk = pool[i]
            with h5py.File(str(hp), "r") as f:
                ep = load_episode(f, tk, camera_keys, max_episode_length)
                ep["source_file"] = str(hp)
            episodes.append(ep)
        return episodes

    success_eps = _sample(success_pool, num_success)
    fail_eps = _sample(fail_pool, num_fail)

    logger.info(
        "Sampled %d success + %d fail episodes (pool: %d success, %d fail)",
        len(success_eps), len(fail_eps), len(success_pool), len(fail_pool),
    )
    return success_eps, fail_eps


def generate_episode_visualizations(cfg: EpisodeVizConfig) -> dict[str, Any]:
    """主入口: 生成 episode 级可视化 (PNG + GIF).

    Returns:
        dict with output paths and statistics
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_paths = [Path(p) for p in cfg.hdf5_paths]
    if not hdf5_paths:
        raise ValueError("hdf5_paths is empty")

    # Sample episodes
    success_eps, fail_eps = sample_episodes(
        hdf5_paths=hdf5_paths,
        camera_keys=cfg.camera_keys,
        num_success=cfg.num_success,
        num_fail=cfg.num_fail,
        max_episode_length=cfg.max_episode_length,
        seed=cfg.seed,
    )

    all_episodes = success_eps + fail_eps
    if not all_episodes:
        raise RuntimeError("No episodes found in the provided HDF5 files")

    results: dict[str, Any] = {
        "n_success": len(success_eps),
        "n_fail": len(fail_eps),
        "outputs": [],
    }

    for ep in all_episodes:
        tag = "success" if ep["success"] else "fail"
        prefix = f'{ep["traj_key"]}_{tag}'

        # Keyframe PNG
        keyframes = select_keyframes(ep, cfg.num_keyframes)
        png_path = output_dir / f"{prefix}_keyframes.png"
        render_keyframes_png(
            episode=ep,
            camera_keys=cfg.camera_keys,
            camera_labels=cfg.camera_labels,
            keyframe_indices=keyframes,
            output_path=png_path,
        )

        # GIF
        gif_path = output_dir / f"{prefix}_episode.gif"
        render_episode_gif(
            episode=ep,
            camera_keys=cfg.camera_keys,
            camera_labels=cfg.camera_labels,
            output_path=gif_path,
            fps=cfg.gif_fps,
            dpi=cfg.gif_dpi,
        )

        results["outputs"].append({
            "traj_key": ep["traj_key"],
            "success": ep["success"],
            "source": ep["source"],
            "length": ep["length"],
            "png": str(png_path),
            "gif": str(gif_path),
        })

        print(f"[EPISODE-VIZ] {tag.upper()} {ep['traj_key']} "
              f"(T={ep['length']}, src={ep['source']}): {png_path.name} + {gif_path.name}")

    print(f"[EPISODE-VIZ] Done! {len(all_episodes)} episodes -> {output_dir}")
    return results


# -- CLI entry point --
def main() -> None:
    """CLI 入口: 支持 tyro 配置."""
    import tyro
    cfg = tyro.cli(EpisodeVizConfig)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate_episode_visualizations(cfg)


if __name__ == "__main__":
    main()
