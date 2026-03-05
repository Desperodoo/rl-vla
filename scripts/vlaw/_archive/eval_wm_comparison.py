"""VLAW P2.3 — Ctrl-World 世界模型定量+定性对比评估脚本.

对比 3 个模型：
- pretrained (DROID 预训练)
- ckpt-8000 (ManiSkill Phase-A)
- ckpt-10000 (ManiSkill Phase-A)

评估输出：
- 定量：PSNR / SSIM / LPIPS（逐帧统计）
- 定性：关键帧对比图（GT + 3 个模型）

用法::
    CUDA_VISIBLE_DEVICES=4,5 conda run -n ctrl_world python rlft/vlaw/scripts/eval_wm_comparison.py

输出：
    - logs/vlaw/wm_comparison_frames/  PNG 对比图
    - 终端打印 PSNR/SSIM/LPIPS 对比
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

# ---- 项目根路径 ----
_ROOT = Path(__file__).resolve().parents[3]  # rl-vla/
# 直接插入所需模块路径，绕过 rlft/__init__.py 的 tyro 依赖
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "ctrl_world"))
sys.path.insert(0, str(_ROOT / "rlft" / "vlaw" / "world_model"))

os.chdir(str(_ROOT))  # 保证相对路径正确

# ---- 输出目录 ----
OUTPUT_DIR = _ROOT / "logs" / "vlaw" / "wm_comparison_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. PSNR 计算（无需 skimage）
# ============================================================

def compute_psnr(img_gt: np.ndarray, img_pred: np.ndarray, max_val: float = 255.0) -> float:
    """计算 PSNR（dB）. 输入为任意形状的 uint8 或 float 数组."""
    mse = np.mean((img_gt.astype(np.float32) - img_pred.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """计算 SSIM（使用 skimage 标准实现）."""
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def compute_lpips(
    img_gt: np.ndarray,
    img_pred: np.ndarray,
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> float:
    """计算 LPIPS（每帧）."""
    gt = torch.from_numpy(img_gt).permute(2, 0, 1).float() / 127.5 - 1.0
    pred = torch.from_numpy(img_pred).permute(2, 0, 1).float() / 127.5 - 1.0
    gt = gt.unsqueeze(0).to(device)
    pred = pred.unsqueeze(0).to(device)
    with torch.no_grad():
        val = lpips_model(gt, pred)
    return float(val.item())


def detect_checkpoints() -> Dict[str, str]:
    """自动探测 pretrained / ckpt-8000 / ckpt-10000 路径。"""
    phase_a_dir = _ROOT / "checkpoints" / "vlaw" / "world_model" / "phase_a"
    pretrained_dir = _ROOT / "checkpoints" / "vlaw" / "world_model" / "pretrained"

    def _pick(paths: List[Path]) -> Optional[Path]:
        for p in paths:
            if p.exists() and p.is_file():
                return p
        return None

    pretrained_candidates = [
        pretrained_dir / "Ctrl-World" / "checkpoint-10000.pt",
        pretrained_dir / "checkpoint-10000.pt",
    ]
    for pattern in ["**/checkpoint-10000.pt", "**/*10000*.pt", "**/checkpoint-*.pt"]:
        pretrained_candidates.extend(sorted(pretrained_dir.glob(pattern), reverse=True))

    ckpt8k_candidates = [
        phase_a_dir / "checkpoint-8000.pt",
        phase_a_dir / "ckpt-8000.pt",
    ]
    ckpt8k_candidates.extend(sorted(phase_a_dir.glob("**/*8000*.pt"), reverse=True))

    ckpt10k_candidates = [
        phase_a_dir / "checkpoint-10000.pt",
        phase_a_dir / "ckpt-10000.pt",
    ]
    ckpt10k_candidates.extend(sorted(phase_a_dir.glob("**/*10000*.pt"), reverse=True))

    model_paths = {
        "pretrained": _pick(pretrained_candidates),
        "ckpt-8000": _pick(ckpt8k_candidates),
        "ckpt-10000": _pick(ckpt10k_candidates),
    }

    missing = [k for k, v in model_paths.items() if v is None]
    if missing:
        raise FileNotFoundError(
            f"未找到以下 checkpoint: {missing} | "
            f"pretrained_dir={pretrained_dir}, phase_a_dir={phase_a_dir}"
        )

    return {k: str(v) for k, v in model_paths.items() if v is not None}


# ============================================================
# 2. 数据加载
# ============================================================

def collect_eval_trajectories(
    data_root: str = "data/vlaw/encoded",
    tasks: Optional[List[str]] = None,
    max_per_task: int = 5,
    min_frames: int = 9,
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """收集评估轨迹，优先选取帧数充足的数据.

    Returns:
        dict: {task_name: [(latent_concat, actions), ...]}
        latent_concat: (T, 4, 48, 24) float16
        actions:       (T, 7) float32
    """
    if tasks is None:
        tasks = ["LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"]

    result: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    # 按优先级尝试不同数据源（rollout 帧数更长）
    sources = [
        f"{data_root}/rollouts/iter1",
        f"{data_root}/demos",
    ]

    for task in tasks:
        trajs = []
        for src in sources:
            task_dir = Path(src) / task
            if not task_dir.exists():
                continue
            h5_files = sorted(task_dir.glob("*.h5"))
            for hfile in h5_files:
                with h5py.File(hfile, "r") as f:
                    traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
                    for tk in traj_keys:
                        latent = f[tk]["latent_concat"][:]  # (T, 4, 48, 24)
                        if latent.shape[0] < min_frames:
                            continue
                        actions = f[tk]["actions"][:]  # (T, 7)
                        trajs.append((latent.astype(np.float32), actions))
                        if len(trajs) >= max_per_task:
                            break
                if len(trajs) >= max_per_task:
                    break
            if len(trajs) >= max_per_task:
                break

        result[task] = trajs[:max_per_task]
        print(f"  [数据] {task}: {len(result[task])} 条轨迹 "
              f"(帧长 {[t[0].shape[0] for t in result[task]]})")

    return result


# ============================================================
# 3. 模型加载 & 推理
# ============================================================

def load_adapter(ckpt_path: Optional[str], device: str = "cuda") -> "CtrlWorldAdapter":  # noqa: F821
    """加载 CtrlWorldAdapter，指定 ckpt_path."""
    from ctrl_world.config import wm_args_maniskill  # type: ignore
    from ctrl_world_adapter import CtrlWorldAdapter  # type: ignore  (via rlft/vlaw/world_model/)

    args = wm_args_maniskill()
    # 减小推理步数以加速（定量评估不需要最优质量）
    args.num_inference_steps = 10  # 10步已足够评估差异
    # 明确设定预测 + 历史帧数
    args.num_frames = 5
    args.num_history = 4

    adapter = CtrlWorldAdapter(
        args,
        ckpt_path=ckpt_path,
        device=device,
        dtype=torch.float16,
    )
    return adapter


@torch.no_grad()
def run_prediction(
    adapter: "CtrlWorldAdapter",  # noqa: F821
    latent: np.ndarray,    # (T, 4, 48, 24) float32
    actions: np.ndarray,   # (T, 7) float32
    num_history: int = 4,
    num_frames: int = 5,
    instruction: str = "robot arm manipulation task",
) -> Tuple[np.ndarray, np.ndarray]:
    """运行一次 WM 推理，返回 GT 和 pred 的解码 RGB.

    Returns:
        gt_rgb:   (T_pred, H*n_cam, W, 3) uint8
        pred_rgb: (T_pred, H*n_cam, W, 3) uint8
        其中 H=192, W=192, n_cam=2 → (T_pred, 384, 192, 3)
    """
    T = latent.shape[0]
    window = num_history + num_frames
    if T < window:
        # 如果帧数不足，降级处理
        num_history = max(2, T - 3)
        num_frames = T - num_history
        window = T

    lat_tensor = torch.from_numpy(latent[:window]).to(adapter.device).to(adapter.dtype)

    # ---- GT RGB (解码真实 latent) ----
    gt_lat = lat_tensor[num_history:num_history + num_frames]   # (T_pred, 4, 48, 24)
    gt_rgb = adapter.decode_latents(gt_lat)   # (T_pred, H_concat, W, 3)

    # ---- 模型预测 ----
    try:
        pred_latents = adapter.rollout(
            lat_tensor,
            actions[:window],
            instruction=instruction,
        )  # (N_CAMS, T_pred, 4, 24, 24)  ← 拆分后
        # 重新拼接到 (T_pred, 4, 48, 24)
        N_C, T_p, C, H_s, W_s = pred_latents.shape
        # rearrange: (N_CAMS, T_pred, 4, H_s, W_s) → (T_pred, 4, N_CAMS*H_s, W_s)
        pred_lat_concat = pred_latents.permute(1, 2, 0, 3, 4).reshape(T_p, C, N_C * H_s, W_s)
        pred_rgb = adapter.decode_latents(pred_lat_concat)  # (T_pred, H_concat, W, 3)
    except Exception as e:
        print(f"  ⚠️  推理失败: {e}, 返回黑色帧")
        pred_rgb = np.zeros_like(gt_rgb)

    return gt_rgb, pred_rgb


# ============================================================
# 4. 可视化
# ============================================================

def save_comparison_figure(
    task: str,
    traj_idx: int,
    gt_rgb: np.ndarray,
    model_frames: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """保存 GT + 多模型关键帧对比图（PNG）."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T_pred = gt_rgb.shape[0]
    row_items = [("Ground Truth", gt_rgb)] + [
        ("Pretrained (DROID)" if k == "pretrained" else k, v)
        for k, v in model_frames.items()
    ]
    n_rows = len(row_items)

    fig, axes = plt.subplots(n_rows, T_pred, figsize=(3 * T_pred, 3 * n_rows))
    if T_pred == 1:
        axes = axes[:, np.newaxis]

    for row_i, (label, frames) in enumerate(row_items):
        for col_i in range(T_pred):
            ax = axes[row_i, col_i]
            ax.imshow(frames[col_i])
            ax.axis("off")
            if col_i == 0:
                ax.set_ylabel(label, fontsize=9, rotation=90, labelpad=5)
            if row_i == 0:
                ax.set_title(f"t+{col_i+1}", fontsize=8)

    plt.suptitle(f"{task} | Traj {traj_idx}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 图像已保存: {save_path.name}")


# ============================================================
# 5. 主评估流程
# ============================================================

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== WM 对比评估 ===")
    print(f"CUDA 设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Step 1: 收集评估数据 ----
    print("\n[1/5] 收集评估轨迹...")
    tasks = ["LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"]
    eval_data = collect_eval_trajectories(
        data_root="data/vlaw/encoded",
        tasks=tasks,
        max_per_task=3,   # 3 条/任务，加速评估
        min_frames=9,
    )

    total_trajs = sum(len(v) for v in eval_data.values())
    print(f"总计 {total_trajs} 条轨迹用于评估")

    # ---- Checkpoint 路径（自动探测） ----
    model_ckpts = detect_checkpoints()
    print("\n  探测到 checkpoint:")
    for k, v in model_ckpts.items():
        print(f"  - {k:<10}: {v}")

    # ---- 存储结果 ----
    results: Dict[str, Dict[str, Dict[str, float]]] = {k: {} for k in model_ckpts}
    pred_store: Dict[str, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = {k: {} for k in model_ckpts}

    # LPIPS 模型
    import lpips

    lpips_device = torch.device(device)
    lpips_model = lpips.LPIPS(net="alex").to(lpips_device)
    lpips_model.eval()

    # ============================================================
    # 6~8. 逐模型评估
    # ============================================================
    for eval_idx, (model_name, model_ckpt) in enumerate(model_ckpts.items(), start=2):
        print(f"\n[{eval_idx}/5] 加载 {model_name} 并评估...")
        adapter = load_adapter(model_ckpt, device=device)

        for task, trajs in eval_data.items():
            preds_task = []
            frame_psnrs: List[float] = []
            frame_ssims: List[float] = []
            frame_lpips: List[float] = []

            for i, (latent, actions) in enumerate(trajs):
                print(f"  {model_name:<10} | {task} | traj {i}: T={latent.shape[0]}")
                gt_rgb, pred_rgb = run_prediction(
                    adapter, latent, actions, instruction=f"{task} manipulation"
                )
                preds_task.append((gt_rgb, pred_rgb))

                for f in range(len(gt_rgb)):
                    frame_psnrs.append(compute_psnr(gt_rgb[f], pred_rgb[f]))
                    frame_ssims.append(compute_ssim(gt_rgb[f], pred_rgb[f]))
                    frame_lpips.append(
                        compute_lpips(gt_rgb[f], pred_rgb[f], lpips_model=lpips_model, device=lpips_device)
                    )

            pred_store[model_name][task] = preds_task
            results[model_name][task] = {
                "mean_psnr": float(np.mean(frame_psnrs)),
                "std_psnr": float(np.std(frame_psnrs)),
                "mean_ssim": float(np.mean(frame_ssims)),
                "std_ssim": float(np.std(frame_ssims)),
                "mean_lpips": float(np.mean(frame_lpips)),
                "std_lpips": float(np.std(frame_lpips)),
                "n_frames": float(len(frame_psnrs)),
            }
            print(
                f"  ▶ {task} {model_name}: "
                f"PSNR {results[model_name][task]['mean_psnr']:.2f}±{results[model_name][task]['std_psnr']:.2f}, "
                f"SSIM {results[model_name][task]['mean_ssim']:.4f}±{results[model_name][task]['std_ssim']:.4f}, "
                f"LPIPS {results[model_name][task]['mean_lpips']:.4f}±{results[model_name][task]['std_lpips']:.4f}"
            )

        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [{model_name} 模型已卸载]")

    # ============================================================
    # 8. 保存定性对比图（每任务前3条轨迹）
    # ============================================================
    print("\n[4/5] 保存定性对比图...")
    saved_paths = []
    for task in tasks:
        pre_list = pred_store["pretrained"].get(task, [])
        n_vis = min(3, len(pre_list))
        for i in range(n_vis):
            gt_rgb = pre_list[i][0]
            model_frames = {
                name: pred_store[name].get(task, [])[i][1]
                for name in model_ckpts
                if i < len(pred_store[name].get(task, []))
            }
            fname = f"{task}_traj{i:02d}_comparison.png"
            save_path = OUTPUT_DIR / fname
            save_comparison_figure(task, i, gt_rgb, model_frames, save_path)
            saved_paths.append(str(save_path))

    # ============================================================
    # 9. 打印结果表格
    # ============================================================
    print("\n[5/5] 定量结果汇总")
    print("=" * 120)
    print(f"{'任务':<20} {'模型':<12} {'PSNR':>16} {'SSIM':>16} {'LPIPS':>16}")
    print("-" * 120)

    overall_stats: Dict[str, Dict[str, float]] = {}
    for model_name in model_ckpts:
        model_psnr = []
        model_ssim = []
        model_lpips = []
        for task in tasks:
            res = results[model_name].get(task, {})
            if not res:
                continue
            print(
                f"  {task:<20} {model_name:<12} "
                f"{res['mean_psnr']:>7.2f}±{res['std_psnr']:.2f} "
                f"{res['mean_ssim']:>9.4f}±{res['std_ssim']:.4f} "
                f"{res['mean_lpips']:>8.4f}±{res['std_lpips']:.4f}"
            )
            model_psnr.append(res["mean_psnr"])
            model_ssim.append(res["mean_ssim"])
            model_lpips.append(res["mean_lpips"])
        if model_psnr:
            overall_stats[model_name] = {
                "psnr": float(np.mean(model_psnr)),
                "ssim": float(np.mean(model_ssim)),
                "lpips": float(np.mean(model_lpips)),
            }

    print("-" * 120)
    for model_name, agg in overall_stats.items():
        print(
            f"  {'总均值':<20} {model_name:<12} "
            f"{agg['psnr']:>16.2f} {agg['ssim']:>16.4f} {agg['lpips']:>16.4f}"
        )
    print("=" * 120)

    psnr_pass = {k: (v["psnr"] > 18.0) for k, v in overall_stats.items()}
    print("\n✅ PSNR>18 检查:")
    for k, passed in psnr_pass.items():
        print(f"  - {k:<10}: {'PASS' if passed else 'FAIL'} (avg PSNR={overall_stats[k]['psnr']:.2f})")
    print(f"\n📁 对比图保存位置: {OUTPUT_DIR}")
    print(f"   共 {len(saved_paths)} 张图片:")
    for p in saved_paths:
        print(f"   - {Path(p).name}")

    # ============================================================
    # 10. 写入报告文件
    # ============================================================
    result_file = os.environ.get(
        "RESULT_FILE",
        str(_ROOT / "logs/vlaw/wm-compare-result-auto.md")
    )
    with open(result_file, "a", encoding="utf-8") as rf:
        rf.write("\n## 定量结果（PSNR / SSIM / LPIPS）\n\n")
        rf.write("| 任务 | 模型 | PSNR (dB) | SSIM | LPIPS |\n")
        rf.write("|------|------|-----------|------|-------|\n")
        for task in tasks:
            for model_name in model_ckpts:
                res = results[model_name].get(task, {})
                if res:
                    rf.write(
                        f"| {task} | {model_name} "
                        f"| {res['mean_psnr']:.2f}±{res['std_psnr']:.2f} "
                        f"| {res['mean_ssim']:.4f}±{res['std_ssim']:.4f} "
                        f"| {res['mean_lpips']:.4f}±{res['std_lpips']:.4f} |\n"
                    )

        rf.write("\n### 总均值\n\n")
        rf.write("| 模型 | Avg PSNR | Avg SSIM | Avg LPIPS | PSNR>18 |\n")
        rf.write("|------|----------|----------|-----------|---------|\n")
        for model_name, agg in overall_stats.items():
            rf.write(
                f"| {model_name} | {agg['psnr']:.2f} | {agg['ssim']:.4f} "
                f"| {agg['lpips']:.4f} | {'PASS' if psnr_pass[model_name] else 'FAIL'} |\n"
            )

        rf.write(f"\n## 定性输出\n\n")
        rf.write(f"对比图保存路径: `{OUTPUT_DIR}`\n\n")
        for p in saved_paths:
            rf.write(f"- `{Path(p).name}`\n")

        rf.write("\n## 结论\n\n")
        all_pass = all(psnr_pass.values()) if psnr_pass else False
        rf.write(f"- 是否满足 PSNR>18: {'✅ 是' if all_pass else '⚠️ 部分满足/不满足'}\n")
        for model_name, agg in overall_stats.items():
            rf.write(
                f"- {model_name}: Avg PSNR={agg['psnr']:.2f}, "
                f"Avg SSIM={agg['ssim']:.4f}, Avg LPIPS={agg['lpips']:.4f}\n"
            )

        rf.write(f"\n## 状态：✅ 完成\n")

    print(f"\n📝 报告已追加写入: {result_file}")
    print("✅ 评估完成！")


if __name__ == "__main__":
    main()
