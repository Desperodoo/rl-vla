"""ACP v3_so vs v3_sae 对比分析脚本。

对比新训练的两个 ACP v3 模型：
- v3_so: success_once 标签, 12K steps, bs=128, best MAE=0.0724
- v3_sae: success_at_end 标签, 12K steps, bs=128, best MAE=0.0463

生成：
1. 数据统计：v3 数据中 success_once ≠ success_at_end 的 mismatch 统计
2. Value target 对比：同一轨迹在两种模式下的 target 差异
3. 模型预测对比：v3_so 和 v3_sae 对同一数据的 value 预测差异
4. TD-shaped reward 影响分析
5. 训练曲线对比（从 wandb log 解析）
6. 推理指标对比（从 HDF5 标注读取）

用法：
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/wjz/rl-vla conda run -n rlft_ms3 --no-capture-output \
        python scripts/analyze_acp_v3_so_vs_sae.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/wjz/rl-vla")
sys.path.insert(0, str(PROJECT_ROOT))

from rlft.acp.config import ValueTargetConfig
from rlft.acp.value_targets import compute_value_targets

FIG_DIR = PROJECT_ROOT / "docs" / "vlaw" / "figures" / "v3_comparison"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# v3 数据 (ignore_terminations=True, PLD-SAC s42)
DATA_DIRS = {
    "A: Demo": PROJECT_ROOT / "data/vlaw/rollouts/mixed",
    "B: PLD-Pretrained": PROJECT_ROOT / "data/vlaw/rollouts/v3_pld_pretrained",
    "C: PLD-Teleop": PROJECT_ROOT / "data/vlaw/rollouts/v3_pld_teleop",
    "D: PLD-RL Prior": PROJECT_ROOT / "data/vlaw/rollouts/v3_pld_rl_prior",
}

V3_SO_CKPT = PROJECT_ROOT / "checkpoints/vlaw/acp/v3_so/best.safetensors"
V3_SAE_CKPT = PROJECT_ROOT / "checkpoints/vlaw/acp/v3_sae/best.safetensors"

# 训练日志（实际保留的 v3 ACP 训练输出）
V3_SO_LOG = PROJECT_ROOT / "logs/vlaw/acp_exp_aligned.log"
V3_SAE_LOG = PROJECT_ROOT / "logs/vlaw/acp_exp_aligned.log"

# 推理标注后的 HDF5（当前已写回 ACP 结果的实际目录）
EVAL_DIR_SO = PROJECT_ROOT / "data/vlaw/encoded/train_v4/LiftPegUpright-v1"
EVAL_DIR_SAE = PROJECT_ROOT / "data/vlaw/encoded/train_v5/LiftPegUpright-v1"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.figsize": (12, 8),
})


# ── Part 1: Scan data for success mismatch ────────────────────────────────
def scan_trajectories():
    """扫描 v3 数据，统计 success_once vs success_at_end 差异。"""
    results = {}
    all_trajs = []

    for label, data_dir in DATA_DIRS.items():
        h5_files = sorted(data_dir.rglob("*.h5"))
        stats = {
            "total": 0, "success_once": 0, "success_at_end": 0,
            "mismatch": 0, "mismatch_trajs": [],
        }
        for h5_path in h5_files:
            with h5py.File(str(h5_path), "r") as f:
                for key in sorted(f.keys()):
                    if not key.startswith("traj_"):
                        continue
                    grp = f[key]
                    if "env_success" not in grp:
                        continue
                    env_success = np.asarray(grp["env_success"], dtype=bool)
                    T = len(env_success)
                    s_once = bool(env_success.any())
                    s_end = bool(env_success[-1])

                    stats["total"] += 1
                    if s_once:
                        stats["success_once"] += 1
                    if s_end:
                        stats["success_at_end"] += 1
                    if s_once != s_end:
                        stats["mismatch"] += 1
                        first_success = int(np.argmax(env_success)) if s_once else -1
                        if s_once:
                            last_success = T - 1 - int(np.argmax(env_success[::-1]))
                        else:
                            last_success = -1
                        stats["mismatch_trajs"].append({
                            "file": str(h5_path.name),
                            "traj": key,
                            "length": T,
                            "success_once": s_once,
                            "success_at_end": s_end,
                            "first_success_frame": first_success,
                            "last_success_frame": last_success,
                            "success_frames": int(env_success.sum()),
                        })

                    all_trajs.append({
                        "label": label,
                        "file": str(h5_path.name),
                        "traj": key,
                        "length": T,
                        "env_success": env_success,
                        "success_once": s_once,
                        "success_at_end": s_end,
                    })

        results[label] = stats

    return results, all_trajs


# ── Part 2: Value target comparison ──────────────────────────────────────
def compute_target_comparison(all_trajs: list[dict]):
    """对所有轨迹分别计算 success_once 和 success_at_end 的 value target。"""
    max_len = max(t["length"] for t in all_trajs)

    cfg_once = ValueTargetConfig(success_mode="success_once")
    cfg_end = ValueTargetConfig(success_mode="success_at_end")

    comparisons = []
    for traj in all_trajs:
        targets_once = compute_value_targets(
            traj["env_success"], traj["length"], max_len, cfg_once
        )
        targets_end = compute_value_targets(
            traj["env_success"], traj["length"], max_len, cfg_end
        )
        comparisons.append({
            **traj,
            "targets_once": targets_once,
            "targets_end": targets_end,
            "target_diff": targets_end - targets_once,
            "mean_diff": float(np.mean(targets_end - targets_once)),
        })

    return comparisons


# ── Part 3: Model inference comparison ────────────────────────────────────
def run_model_comparison(all_trajs: list[dict], device: str = "cuda:0", max_trajs: int = 100):
    """用 v3_so 和 v3_sae 模型分别推理，比较 value 预测。"""
    from safetensors.torch import load_file

    from rlft.acp.config import ValueModelConfig
    from rlft.acp.value_model import ManiSkillValueModel

    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_trajs), min(max_trajs, len(all_trajs)), replace=False)
    subset = [all_trajs[i] for i in sorted(indices)]

    vm_cfg = ValueModelConfig()

    print("[分析] 加载 v3_so (success_once) 模型...")
    model_so = ManiSkillValueModel(vm_cfg, device=device)
    state_so = load_file(str(V3_SO_CKPT))
    model_so.model.load_state_dict(state_so, strict=False)

    print("[分析] 加载 v3_sae (success_at_end) 模型...")
    model_sae = ManiSkillValueModel(vm_cfg, device=device)
    state_sae = load_file(str(V3_SAE_CKPT))
    model_sae.model.load_state_dict(state_sae, strict=False)

    camera_keys = vm_cfg.camera_keys
    model_results = []

    for traj_info in subset:
        h5_path = None
        for d in DATA_DIRS.values():
            candidate = list(d.rglob(traj_info["file"]))
            if candidate:
                h5_path = candidate[0]
                break
        if h5_path is None:
            continue

        with h5py.File(str(h5_path), "r") as f:
            grp = f[traj_info["traj"]]
            T = traj_info["length"]

            values_so = []
            values_sae = []

            for t in range(T):
                images = []
                for ck in camera_keys:
                    if ck in grp:
                        img = np.asarray(grp[ck][t])
                        if img.ndim == 3 and img.shape[-1] == 3:
                            img = img.transpose(2, 0, 1)
                        images.append(img)

                if len(images) != len(camera_keys):
                    break

                img_tensor = torch.tensor(np.stack(images), dtype=torch.uint8).unsqueeze(0).to(device)
                mask_tensor = torch.ones(1, len(camera_keys), dtype=torch.bool, device=device)

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    so_val = model_so.predict_values(img_tensor, mask_tensor).item()
                    sae_val = model_sae.predict_values(img_tensor, mask_tensor).item()

                values_so.append(so_val)
                values_sae.append(sae_val)

            if len(values_so) == T:
                model_results.append({
                    **traj_info,
                    "pred_so": np.array(values_so),
                    "pred_sae": np.array(values_sae),
                    "pred_diff": np.array(values_sae) - np.array(values_so),
                })
                print(f"  [{len(model_results)}/{len(subset)}] {traj_info['traj']} "
                      f"T={T} s_once={traj_info['success_once']} s_end={traj_info['success_at_end']} "
                      f"so_mean={np.mean(values_so):.4f} sae_mean={np.mean(values_sae):.4f}")

    del model_so, model_sae
    torch.cuda.empty_cache()
    return model_results


# ── Part 4: Read inference annotations from HDF5 ─────────────────────────
def read_infer_annotations(eval_dir: Path) -> dict:
    """读取推理标注后的 HDF5, 汇总统计指标。"""
    h5_files = sorted(eval_dir.rglob("*.h5"))
    all_targets, all_preds, all_advantages, all_indicators, all_weights = [], [], [], [], []

    for hp in h5_files:
        with h5py.File(str(hp), "r") as f:
            for key in sorted(f.keys()):
                if not key.startswith("traj_"):
                    continue
                grp = f[key]
                if "acp_value_target" not in grp:
                    continue
                all_targets.append(np.asarray(grp["acp_value_target"]))
                all_preds.append(np.asarray(grp["acp_value_pred"]))
                all_advantages.append(np.asarray(grp["acp_advantage"]))
                all_indicators.append(np.asarray(grp["acp_indicator"]))
                all_weights.append(np.asarray(grp["acp_weight"]))

    if not all_targets:
        return {}

    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)
    advantages = np.concatenate(all_advantages)
    indicators = np.concatenate(all_indicators)
    weights = np.concatenate(all_weights)

    mae = float(np.mean(np.abs(preds - targets)))
    positive_ratio = float(indicators.mean())

    return {
        "num_frames": len(targets),
        "num_trajs": len(all_targets),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean((preds - targets)**2))),
        "pearson_r": float(np.corrcoef(targets, preds)[0, 1]),
        "positive_ratio": positive_ratio,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "weight_mean": float(weights.mean()),
        "weight_max": float(weights.max()),
        "target_mean": float(targets.mean()),
        "pred_mean": float(preds.mean()),
    }


def load_infer_records(eval_dir: Path) -> list[dict]:
    """读取带 ACP 标注的 HDF5，并按轨迹返回明细记录。"""
    records: list[dict] = []
    h5_files = sorted(eval_dir.rglob("*.h5"))

    for hp in h5_files:
        with h5py.File(str(hp), "r") as f:
            for key in sorted(f.keys()):
                if not key.startswith("traj_"):
                    continue
                grp = f[key]
                required = [
                    "acp_value_target",
                    "acp_value_pred",
                    "acp_advantage",
                    "acp_indicator",
                    "acp_weight",
                ]
                if any(name not in grp for name in required):
                    continue
                if "env_success" not in grp:
                    continue

                env_success = np.asarray(grp["env_success"], dtype=bool)
                records.append({
                    "file": hp.name,
                    "traj": key,
                    "length": int(len(env_success)),
                    "success_once": bool(env_success.any()),
                    "success_at_end": bool(env_success[-1]),
                    "targets": np.asarray(grp["acp_value_target"], dtype=np.float32),
                    "preds": np.asarray(grp["acp_value_pred"], dtype=np.float32),
                    "advantages": np.asarray(grp["acp_advantage"], dtype=np.float32),
                    "indicators": np.asarray(grp["acp_indicator"], dtype=np.float32),
                    "weights": np.asarray(grp["acp_weight"], dtype=np.float32),
                    "threshold": float(grp.attrs.get("acp_threshold", np.nan)),
                    "positive_ratio": float(grp.attrs.get("acp_positive_ratio", np.nan)),
                })

    return records


def _split_records(records: list[dict]) -> dict[str, list[dict]]:
    return {
        "stable": [r for r in records if r["success_once"] and r["success_at_end"]],
        "drop": [r for r in records if r["success_once"] and not r["success_at_end"]],
        "fail": [r for r in records if not r["success_once"] and not r["success_at_end"]],
    }


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    a_var = float(np.var(a, ddof=1))
    b_var = float(np.var(b, ddof=1))
    pooled = ((len(a) - 1) * a_var + (len(b) - 1) * b_var) / (len(a) + len(b) - 2)
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def summarize_layerwise_effects(records: list[dict]) -> dict[str, dict[str, float]]:
    """按轨迹统计 target/pred/advantage/weight 的 hold-vs-drop 效应量。"""
    groups = _split_records(records)
    layer_map = {
        "target": "targets",
        "pred": "preds",
        "advantage": "advantages",
        "indicator": "indicators",
        "weight": "weights",
    }
    summary: dict[str, dict[str, float]] = {}

    for layer_name, key in layer_map.items():
        stable = np.array([float(np.mean(r[key])) for r in groups["stable"]], dtype=np.float64)
        drop = np.array([float(np.mean(r[key])) for r in groups["drop"]], dtype=np.float64)
        fail = np.array([float(np.mean(r[key])) for r in groups["fail"]], dtype=np.float64)

        summary[layer_name] = {
            "stable_mean": float(np.mean(stable)) if len(stable) else 0.0,
            "drop_mean": float(np.mean(drop)) if len(drop) else 0.0,
            "fail_mean": float(np.mean(fail)) if len(fail) else 0.0,
            "stable_minus_drop": float(np.mean(stable) - np.mean(drop)) if len(stable) and len(drop) else 0.0,
            "stable_vs_drop_d": _cohen_d(stable, drop),
            "stable_vs_fail_d": _cohen_d(stable, fail),
            "drop_vs_fail_d": _cohen_d(drop, fail),
            "stable_count": float(len(stable)),
            "drop_count": float(len(drop)),
            "fail_count": float(len(fail)),
        }

    return summary


def compare_threshold_scopes(records: list[dict]) -> dict[str, object]:
    """比较全局阈值和 per-trajectory 阈值的正样本覆盖情况。"""
    groups = _split_records(records)
    valid_ratios = [r["positive_ratio"] for r in records if np.isfinite(r["positive_ratio"])]
    positive_ratio = float(np.mean(valid_ratios)) if valid_ratios else 0.3
    valid_thresholds = [r["threshold"] for r in records if np.isfinite(r["threshold"])]
    if valid_thresholds:
        global_threshold = float(np.median(valid_thresholds))
    else:
        all_adv = np.concatenate([r["advantages"] for r in records]) if records else np.array([], dtype=np.float32)
        global_threshold = float(np.quantile(all_adv, 1.0 - positive_ratio)) if len(all_adv) else 0.0

    def _mean_positive_rate(recs: list[dict], threshold_fn) -> float:
        if not recs:
            return 0.0
        rates = []
        for r in recs:
            threshold = threshold_fn(r)
            rates.append(float(np.mean(r["advantages"] >= threshold)))
        return float(np.mean(rates))

    global_rates = {
        name: _mean_positive_rate(recs, lambda _r: global_threshold)
        for name, recs in groups.items()
    }
    local_rates = {
        name: _mean_positive_rate(recs, lambda r: float(np.quantile(r["advantages"], 1.0 - positive_ratio)))
        for name, recs in groups.items()
    }

    return {
        "positive_ratio_target": positive_ratio,
        "global_threshold": global_threshold,
        "threshold_spread": float(np.max(valid_thresholds) - np.min(valid_thresholds)) if valid_thresholds else 0.0,
        "global_positive_rates": global_rates,
        "local_positive_rates": local_rates,
        "global_gap_stable_drop": float(global_rates["stable"] - global_rates["drop"]),
        "local_gap_stable_drop": float(local_rates["stable"] - local_rates["drop"]),
        "gap_gain_from_local": float(
            (local_rates["stable"] - local_rates["drop"]) - (global_rates["stable"] - global_rates["drop"])
        ),
        "local_minus_global": {
            name: float(local_rates[name] - global_rates[name])
            for name in groups.keys()
        },
    }


def parse_training_log(log_path: Path) -> dict[str, dict[str, list[float]]]:
    """解析训练日志中的 train/val 曲线；缺失日志时返回空序列。"""
    series: dict[str, dict[str, list[float]]] = {
        "train": {"steps": [], "mae": [], "loss": []},
        "val": {"steps": [], "mae": [], "loss": []},
    }
    if not log_path.exists():
        return series

    num = r"-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
    generic_step_re = re.compile(r"(?:global_step|step|steps)\D*(\d+)", re.IGNORECASE)
    generic_metric_re = re.compile(
        r"(?:wandb:\s*)?(?P<key>(?:train|val|eval)/(?:mae|loss|critic_loss|actor_loss))\s+"
        rf"(?P<value>{num})"
    )
    acp_train_re = re.compile(
        rf"^\[ACP\]\s+step=(?P<step>\d+)/\d+\s+loss=(?P<loss>{num})\s+mae=(?P<mae>{num})",
        re.IGNORECASE,
    )
    acp_val_re = re.compile(
        rf"^\[ACP\]\s+\[val\]\s+step=(?P<step>\d+)\s+loss=(?P<loss>{num})\s+mae=(?P<mae>{num})",
        re.IGNORECASE,
    )

    records: dict[str, dict[int, dict[str, float]]] = {"train": {}, "val": {}}
    last_step: dict[str, int | None] = {"train": None, "val": None}
    fallback_step: dict[str, int] = {"train": 0, "val": 0}

    for line in log_path.read_text(errors="ignore").splitlines():
        acp_match = acp_val_re.search(line)
        if acp_match:
            step = int(acp_match.group("step"))
            records["val"].setdefault(step, {})["loss"] = float(acp_match.group("loss"))
            records["val"].setdefault(step, {})["mae"] = float(acp_match.group("mae"))
            last_step["val"] = step
            continue

        acp_match = acp_train_re.search(line)
        if acp_match:
            step = int(acp_match.group("step"))
            records["train"].setdefault(step, {})["loss"] = float(acp_match.group("loss"))
            records["train"].setdefault(step, {})["mae"] = float(acp_match.group("mae"))
            last_step["train"] = step
            continue

        step_match = generic_step_re.search(line)
        metric_matches = list(generic_metric_re.finditer(line))
        if not metric_matches:
            continue

        for metric_match in metric_matches:
            raw_key = metric_match.group("key").lower()
            split, metric = raw_key.split("/", 1)
            split = "val" if split == "eval" else split
            if split not in records:
                continue

            value = float(metric_match.group("value"))
            metric = "loss" if metric in {"loss", "critic_loss", "actor_loss"} else metric
            if metric not in {"mae", "loss"}:
                continue

            step = int(step_match.group(1)) if step_match else last_step[split]
            if step is None:
                step = fallback_step[split]
                fallback_step[split] += 1
            last_step[split] = step
            records[split].setdefault(step, {})[metric] = value

    for split in ["train", "val"]:
        step_keys = sorted(records[split].keys())
        series[split]["steps"] = [int(step) for step in step_keys]
        mae_vals: list[float] = []
        loss_vals: list[float] = []
        for step in step_keys:
            rec = records[split][step]
            mae = rec.get("mae")
            loss = rec.get("loss")
            if mae is None and loss is not None:
                mae = loss
            if loss is None and mae is not None:
                loss = mae
            mae_vals.append(float(mae) if mae is not None else float("nan"))
            loss_vals.append(float(loss) if loss is not None else float("nan"))
        series[split]["mae"] = mae_vals
        series[split]["loss"] = loss_vals

    return series


def build_infer_diagnostics() -> dict[str, dict[str, object]]:
    """汇总 v3_so / v3_sae 的逐层信号保真度诊断。"""
    diagnostics: dict[str, dict[str, object]] = {}
    for name, eval_dir in {"v3_so": EVAL_DIR_SO, "v3_sae": EVAL_DIR_SAE}.items():
        records = load_infer_records(eval_dir)
        diagnostics[name] = {
            "num_records": len(records),
            "layerwise_effects": summarize_layerwise_effects(records) if records else {},
            "threshold_scope": compare_threshold_scopes(records) if records else {},
        }
    return diagnostics


# ── Figures ──────────────────────────────────────────────────────────────
def plot_data_statistics(scan_results: dict):
    """Fig 1: v3 数据 success mismatch 统计。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = list(scan_results.keys())
    total = [scan_results[l]["total"] for l in labels]
    s_once = [scan_results[l]["success_once"] for l in labels]
    s_end = [scan_results[l]["success_at_end"] for l in labels]
    mismatch = [scan_results[l]["mismatch"] for l in labels]

    x = np.arange(len(labels))
    w = 0.25

    ax = axes[0]
    ax.bar(x - w, s_once, w, label="success_once", color="#4CAF50", alpha=0.8)
    ax.bar(x, s_end, w, label="success_at_end", color="#2196F3", alpha=0.8)
    ax.bar(x + w, mismatch, w, label="mismatch (once!=end)", color="#FF5722", alpha=0.8)
    for i in range(len(labels)):
        ax.text(i - w, s_once[i] + 1, str(s_once[i]), ha="center", fontsize=8)
        ax.text(i, s_end[i] + 1, str(s_end[i]), ha="center", fontsize=8)
        ax.text(i + w, mismatch[i] + 1, str(mismatch[i]), ha="center", fontsize=8, color="red")
    ax.set_xticks(x)
    ax.set_xticklabels([l.split(":")[1].strip() for l in labels], fontsize=9)
    ax.set_ylabel("Trajectories")
    ax.set_title("(a) Success Signal Statistics per Dataset (v3 data)")
    ax.legend()

    ax = axes[1]
    rates = [m / t * 100 if t > 0 else 0 for m, t in zip(mismatch, total)]
    ax.bar(x, rates, 0.5, color="#FF5722", alpha=0.8)
    for i, (r, m, t) in enumerate(zip(rates, mismatch, total)):
        ax.text(i, r + 0.3, f"{r:.1f}%\n({m}/{t})", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split(":")[1].strip() for l in labels], fontsize=9)
    ax.set_ylabel("Mismatch Rate (%)")
    ax.set_title("(b) success_once != success_at_end Rate")
    ax.set_ylim(0, max(rates) * 1.5 + 5)

    fig.suptitle("Fig 1. v3 Data — success_once vs success_at_end Mismatch", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3cmp_fig1_data_statistics.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 1] {FIG_DIR / 'v3cmp_fig1_data_statistics.png'}")


def plot_value_target_comparison(comparisons: list[dict]):
    """Fig 2: Value target 差异分析。"""
    match_success = [c for c in comparisons if c["success_once"] and c["success_at_end"]]
    match_fail = [c for c in comparisons if not c["success_once"] and not c["success_at_end"]]
    mismatch = [c for c in comparisons if c["success_once"] != c["success_at_end"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Distribution of mean target diffs
    ax = axes[0, 0]
    match_diffs = [c["mean_diff"] for c in match_success + match_fail]
    mismatch_diffs = [c["mean_diff"] for c in mismatch]
    ax.hist(match_diffs, bins=50, alpha=0.7, label=f"Matching ({len(match_diffs)})", color="#4CAF50")
    if mismatch_diffs:
        ax.hist(mismatch_diffs, bins=50, alpha=0.7, label=f"Mismatch ({len(mismatch_diffs)})", color="#FF5722")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean Target Diff (success_at_end - success_once)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Distribution of Target Differences")
    ax.legend()

    # (b) Mismatch example trajectory
    ax = axes[0, 1]
    if mismatch:
        ex = max(mismatch, key=lambda c: abs(c["mean_diff"]))
        T = ex["length"]
        frames = np.arange(T)
        ax.plot(frames, ex["targets_once"], "g-", linewidth=2, label="success_once target")
        ax.plot(frames, ex["targets_end"], "r-", linewidth=2, label="success_at_end target")
        ax.fill_between(frames, ex["targets_once"], ex["targets_end"], alpha=0.3, color="orange")
        succ_frames = np.where(ex["env_success"])[0]
        if len(succ_frames) > 0:
            ax.axvspan(succ_frames[0], succ_frames[-1], alpha=0.15, color="green", label="success=True")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value Target")
        ax.set_title(f"(b) Mismatch: {ex['traj']} (T={T})")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No mismatch trajectories", transform=ax.transAxes, ha="center")

    # (c) Scatter: length vs target diff (mismatch only)
    ax = axes[1, 0]
    if mismatch:
        lengths = [c["length"] for c in mismatch]
        diffs = [c["mean_diff"] for c in mismatch]
        ax.scatter(lengths, diffs, c="#FF5722", alpha=0.6, s=30, edgecolors="black", linewidths=0.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Episode Length")
        ax.set_ylabel("Mean Target Diff (at_end - once)")
        ax.set_title(f"(c) Mismatch: Length vs Target Diff (N={len(mismatch)})")

    # (d) Synthetic example
    ax = axes[1, 1]
    T_demo = 50
    env_s = np.zeros(T_demo, dtype=bool)
    env_s[20:40] = True
    cfg_once = ValueTargetConfig(success_mode="success_once")
    cfg_end = ValueTargetConfig(success_mode="success_at_end")
    t_once = compute_value_targets(env_s, T_demo, T_demo, cfg_once)
    t_end = compute_value_targets(env_s, T_demo, T_demo, cfg_end)
    frames = np.arange(T_demo)
    ax.plot(frames, t_once, "g-", linewidth=2, label="success_once (success)")
    ax.plot(frames, t_end, "r-", linewidth=2, label="success_at_end (fail)")
    ax.axvspan(20, 39, alpha=0.15, color="green")
    ax.annotate("success=True\n(frame 20-39)", xy=(30, -0.3), fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    ax.annotate("dropped!\n(frame 40+)", xy=(45, -0.6), fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsalmon", alpha=0.5))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value Target")
    ax.set_title(f"(d) Synthetic: Lift@f20, Drop@f40 (gap={np.mean(t_end - t_once):.3f})")
    ax.legend(fontsize=8)

    fig.suptitle("Fig 2. Value Target Comparison — success_once vs success_at_end", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3cmp_fig2_value_targets.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 2] {FIG_DIR / 'v3cmp_fig2_value_targets.png'}")


def plot_model_predictions(model_results: list[dict]):
    """Fig 3: v3_so vs v3_sae 模型预测对比。"""
    if not model_results:
        print("[Fig 3] 跳过: 无模型预测结果")
        return

    match_succ = [r for r in model_results if r["success_once"] and r["success_at_end"]]
    match_fail = [r for r in model_results if not r["success_once"] and not r["success_at_end"]]
    mismatch = [r for r in model_results if r["success_once"] != r["success_at_end"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Scatter: so mean value vs sae mean value
    ax = axes[0, 0]
    for subset, label, color, marker in [
        (match_succ, "Both success", "#4CAF50", "o"),
        (match_fail, "Both fail", "#9E9E9E", "x"),
        (mismatch, "Mismatch", "#FF5722", "D"),
    ]:
        if subset:
            so_means = [np.mean(r["pred_so"]) for r in subset]
            sae_means = [np.mean(r["pred_sae"]) for r in subset]
            ax.scatter(so_means, sae_means, c=color, marker=marker, alpha=0.6,
                       s=30, label=f"{label} (N={len(subset)})", edgecolors="black", linewidths=0.3)
    lims = [-1.05, 0.05]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("v3_so (success_once) Mean Value")
    ax.set_ylabel("v3_sae (success_at_end) Mean Value")
    ax.set_title("(a) Per-Trajectory Mean Value: v3_so vs v3_sae")
    ax.legend(fontsize=7)

    # (b) Distribution of prediction diffs
    ax = axes[0, 1]
    match_mean_diffs = [np.mean(r["pred_diff"]) for r in match_succ + match_fail]
    mismatch_mean_diffs = [np.mean(r["pred_diff"]) for r in mismatch]
    ax.hist(match_mean_diffs, bins=40, alpha=0.7, label=f"Matching ({len(match_mean_diffs)})", color="#4CAF50")
    if mismatch_mean_diffs:
        ax.hist(mismatch_mean_diffs, bins=20, alpha=0.7, label=f"Mismatch ({len(mismatch_mean_diffs)})", color="#FF5722")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean Prediction Diff (v3_sae - v3_so)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Distribution of Prediction Differences")
    ax.legend()

    # (c) Example trajectory curves
    ax = axes[1, 0]
    if mismatch:
        ex = max(mismatch, key=lambda r: abs(np.mean(r["pred_diff"])))
        title_suffix = "Mismatch"
    else:
        ex = max(model_results, key=lambda r: abs(np.mean(r["pred_diff"])))
        title_suffix = "Max Diff"
    T = ex["length"]
    frames = np.arange(T)
    ax.plot(frames, ex["pred_so"], "g-", linewidth=1.5, label="v3_so prediction")
    ax.plot(frames, ex["pred_sae"], "r-", linewidth=1.5, label="v3_sae prediction")
    ax.fill_between(frames, ex["pred_so"], ex["pred_sae"], alpha=0.3, color="orange")
    succ_frames = np.where(ex["env_success"])[0]
    if len(succ_frames) > 0:
        ax.axvspan(succ_frames[0], succ_frames[-1], alpha=0.1, color="green")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Predicted Value")
    ax.set_title(f"(c) {title_suffix}: {ex['traj']} (s_once={ex['success_once']}, s_end={ex['success_at_end']})")
    ax.legend(fontsize=8)

    # (d) Per-frame avg diff by group
    ax = axes[1, 1]
    max_T = 35
    for group, label, color in [
        (match_succ, "Both success", "#4CAF50"),
        (match_fail, "Both fail", "#9E9E9E"),
        (mismatch, "Mismatch (once=T, end=F)", "#FF5722"),
    ]:
        if not group:
            continue
        diffs_padded = []
        for r in group:
            d = r["pred_diff"]
            if len(d) >= max_T:
                diffs_padded.append(d[:max_T])
            else:
                padded = np.full(max_T, np.nan)
                padded[:len(d)] = d
                diffs_padded.append(padded)
        arr = np.array(diffs_padded)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        frames = np.arange(max_T)
        ax.plot(frames, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(frames, mean - std, mean + std, color=color, alpha=0.15)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean Pred Diff (v3_sae - v3_so)")
    ax.set_title("(d) Average Per-Frame Prediction Diff by Group")
    ax.legend(fontsize=7)

    fig.suptitle("Fig 3. Model Prediction — v3_so (success_once) vs v3_sae (success_at_end)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3cmp_fig3_model_predictions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 3] {FIG_DIR / 'v3cmp_fig3_model_predictions.png'}")


def plot_td_reward_impact(comparisons: list[dict]):
    """Fig 4: TD-shaped reward 影响分析。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scale = 100.0

    # (a) Mismatch trajectory TD rewards
    ax = axes[0]
    mismatch_trajs = [c for c in comparisons if c["success_once"] != c["success_at_end"]]
    if mismatch_trajs:
        ex = max(mismatch_trajs, key=lambda c: abs(c["mean_diff"]))
        T = ex["length"]
        frames = np.arange(T - 1)
        td_once = np.diff(ex["targets_once"]) * scale
        td_end = np.diff(ex["targets_end"]) * scale
        ax.plot(frames, td_once, "g-", linewidth=1.5, label="TD (success_once)")
        ax.plot(frames, td_end, "r-", linewidth=1.5, label="TD (success_at_end)")
        succ_frames = np.where(ex["env_success"])[0]
        if len(succ_frames) > 0:
            ax.axvspan(succ_frames[0], min(succ_frames[-1], T-2), alpha=0.1, color="green")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Frame")
        ax.set_ylabel(f"TD Reward (scale={scale:.0f})")
        ax.set_title("(a) Target-based TD Reward (mismatch traj)")
        ax.legend(fontsize=8)

    # (b) Synthetic lift-and-drop
    ax = axes[1]
    T_demo = 50
    env_s = np.zeros(T_demo, dtype=bool)
    env_s[20:40] = True
    cfg_once = ValueTargetConfig(success_mode="success_once")
    cfg_end = ValueTargetConfig(success_mode="success_at_end")
    t_once = compute_value_targets(env_s, T_demo, T_demo, cfg_once)
    t_end = compute_value_targets(env_s, T_demo, T_demo, cfg_end)
    td_once = np.diff(t_once) * scale
    td_end = np.diff(t_end) * scale
    frames = np.arange(T_demo - 1)
    ax.plot(frames, td_once, "g-", linewidth=2, label="TD (success_once)")
    ax.plot(frames, td_end, "r-", linewidth=2, label="TD (success_at_end)")
    ax.axvspan(20, 39, alpha=0.1, color="green")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"TD Reward (scale={scale:.0f})")
    ax.set_title("(b) Synthetic: Lift@f20, Drop@f40")
    ax.legend(fontsize=8)

    # (c) Cumulative
    ax = axes[2]
    cum_once = np.cumsum(td_once)
    cum_end = np.cumsum(td_end)
    ax.plot(frames, cum_once, "g-", linewidth=2, label="Cumulative (success_once)")
    ax.plot(frames, cum_end, "r-", linewidth=2, label="Cumulative (success_at_end)")
    ax.axvspan(20, 39, alpha=0.1, color="green")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cumulative TD Reward")
    ax.set_title(f"(c) Cumulative: once={cum_once[-1]:.1f}, end={cum_end[-1]:.1f}")
    ax.legend(fontsize=8)

    fig.suptitle("Fig 4. TD-Shaped Reward — success_once vs success_at_end", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3cmp_fig4_td_reward.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 4] {FIG_DIR / 'v3cmp_fig4_td_reward.png'}")


def plot_training_curves(log_so: dict, log_sae: dict):
    """Fig 5: v3_so vs v3_sae 训练曲线对比。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Val MAE
    ax = axes[0]
    if log_so["val"]["steps"]:
        ax.plot(log_so["val"]["steps"], log_so["val"]["mae"], "g-o", markersize=3,
                linewidth=1.5, label="v3_so (success_once)")
    if log_sae["val"]["steps"]:
        ax.plot(log_sae["val"]["steps"], log_sae["val"]["mae"], "r-o", markersize=3,
                linewidth=1.5, label="v3_sae (success_at_end)")
    ax.axhline(0.1, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Quality gate")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation MAE")
    ax.set_title("(a) Validation MAE")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.2)

    # (b) Val loss
    ax = axes[1]
    if log_so["val"]["steps"]:
        ax.plot(log_so["val"]["steps"], log_so["val"]["loss"], "g-o", markersize=3,
                linewidth=1.5, label="v3_so")
    if log_sae["val"]["steps"]:
        ax.plot(log_sae["val"]["steps"], log_sae["val"]["loss"], "r-o", markersize=3,
                linewidth=1.5, label="v3_sae")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("(b) Validation Loss")
    ax.legend(fontsize=8)

    # (c) Train MAE (smoothed)
    ax = axes[2]
    window = 10
    for log, label, color in [
        (log_so, "v3_so", "green"),
        (log_sae, "v3_sae", "red"),
    ]:
        if log["train"]["steps"]:
            steps = np.array(log["train"]["steps"])
            mae = np.array(log["train"]["mae"])
            if len(mae) >= window:
                smooth = np.convolve(mae, np.ones(window)/window, mode="valid")
                ax.plot(steps[window-1:], smooth, color=color, linewidth=1.5, label=label)
            else:
                ax.plot(steps, mae, color=color, linewidth=1.5, label=label)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Train MAE (smoothed)")
    ax.set_title(f"(c) Train MAE (MA-{window})")
    ax.legend(fontsize=8)

    fig.suptitle("Fig 5. Training Curves — v3_so vs v3_sae (12K steps, bs=128)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3cmp_fig5_training_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 5] {FIG_DIR / 'v3cmp_fig5_training_curves.png'}")


def plot_inference_comparison(infer_so: dict, infer_sae: dict):
    """Fig 6: 推理指标对比柱状图。"""
    if not infer_so or not infer_sae:
        print("[Fig 6] 跳过: 推理标注数据不完整")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    versions = ["v3_so\n(success_once)", "v3_sae\n(success_at_end)"]
    x = np.arange(2)

    # (a) MAE
    ax = axes[0]
    maes = [infer_so["mae"], infer_sae["mae"]]
    bars = ax.bar(x, maes, 0.4, color=["#4CAF50", "#FF5722"], alpha=0.8)
    for i, v in enumerate(maes):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0.1, color="orange", linestyle="--", label="Gate: MAE<0.1")
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel("MAE")
    ax.set_title("(a) Inference MAE")
    ax.legend()

    # (b) Positive ratio
    ax = axes[1]
    pos = [infer_so["positive_ratio"], infer_sae["positive_ratio"]]
    ax.bar(x, pos, 0.4, color=["#4CAF50", "#FF5722"], alpha=0.8)
    for i, v in enumerate(pos):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0.3, color="orange", linestyle="--", label="Target: 0.30")
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel("Positive Ratio")
    ax.set_title("(b) Advantage Positive Ratio")
    ax.legend()

    # (c) Pearson r
    ax = axes[2]
    rs = [infer_so["pearson_r"], infer_sae["pearson_r"]]
    ax.bar(x, rs, 0.4, color=["#4CAF50", "#FF5722"], alpha=0.8)
    for i, v in enumerate(rs):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel("Pearson r")
    ax.set_title("(c) Prediction-Target Correlation")

    fig.suptitle("Fig 6. Inference Metrics — v3_so vs v3_sae", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3cmp_fig6_inference_metrics.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 6] {FIG_DIR / 'v3cmp_fig6_inference_metrics.png'}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("[ACP v3_so vs v3_sae 对比分析] 开始")
    print("=" * 70)

    # Step 1: Scan data
    print("\n[Step 1/7] 扫描 v3 数据...")
    scan_results, all_trajs = scan_trajectories()
    total_all = sum(s["total"] for s in scan_results.values())
    total_mismatch = sum(s["mismatch"] for s in scan_results.values())
    print(f"  总轨迹: {total_all}, Mismatch: {total_mismatch} ({total_mismatch/total_all*100:.1f}%)")
    for label, stats in scan_results.items():
        print(f"  {label}: total={stats['total']}, once={stats['success_once']}, "
              f"end={stats['success_at_end']}, mismatch={stats['mismatch']}")

    # Step 2: Value target comparison
    print("\n[Step 2/7] 计算 value target 对比...")
    comparisons = compute_target_comparison(all_trajs)
    mismatch_comps = [c for c in comparisons if c["success_once"] != c["success_at_end"]]
    if mismatch_comps:
        print(f"  Mismatch target diff: mean={np.mean([c['mean_diff'] for c in mismatch_comps]):.4f}")

    # Step 3: Model inference comparison
    print("\n[Step 3/7] 模型推理对比...")
    model_results = []
    if V3_SO_CKPT.exists() and V3_SAE_CKPT.exists():
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_results = run_model_comparison(all_trajs, device=device, max_trajs=80)
        print(f"  完成: {len(model_results)} 轨迹")
    else:
        print(f"  跳过: v3_so={V3_SO_CKPT.exists()}, v3_sae={V3_SAE_CKPT.exists()}")

    # Step 4: Read inference annotations
    print("\n[Step 4/7] 读取推理标注...")
    infer_so = read_infer_annotations(EVAL_DIR_SO)
    infer_sae = read_infer_annotations(EVAL_DIR_SAE)
    infer_diag = build_infer_diagnostics()
    if infer_so:
        print(f"  v3_so: MAE={infer_so['mae']:.4f}, pos_ratio={infer_so['positive_ratio']:.3f}, r={infer_so['pearson_r']:.4f}")
    else:
        print("  v3_so: 无标注数据 (需先运行 `python -m rlft.acp.infer_values`)")
    if infer_sae:
        print(f"  v3_sae: MAE={infer_sae['mae']:.4f}, pos_ratio={infer_sae['positive_ratio']:.3f}, r={infer_sae['pearson_r']:.4f}")
    else:
        print("  v3_sae: 无标注数据 (需先运行 `python -m rlft.acp.infer_values`)")

    for name, diag in infer_diag.items():
        layerwise = diag.get("layerwise_effects", {})
        threshold_scope = diag.get("threshold_scope", {})
        print(f"  {name}: records={diag.get('num_records', 0)}")
        for layer_name in ["target", "pred", "advantage", "indicator", "weight"]:
            layer = layerwise.get(layer_name, {})
            if not layer:
                continue
            print(
                "    "
                f"{layer_name}: stable-drop={layer['stable_minus_drop']:.4f}, "
                f"d={layer['stable_vs_drop_d']:.3f}, "
                f"stable={layer['stable_mean']:.4f}, drop={layer['drop_mean']:.4f}"
            )
        if threshold_scope:
            print(
                "    threshold: "
                f"global_gap={threshold_scope.get('global_gap_stable_drop', 0.0):.4f}, "
                f"local_gap={threshold_scope.get('local_gap_stable_drop', 0.0):.4f}, "
                f"gain={threshold_scope.get('gap_gain_from_local', 0.0):.4f}"
            )

    # Step 5: Parse training logs
    print("\n[Step 5/7] 解析训练日志...")
    log_so = parse_training_log(V3_SO_LOG)
    log_sae = parse_training_log(V3_SAE_LOG)
    print(f"  v3_so: {len(log_so['val']['steps'])} val evals, {len(log_so['train']['steps'])} train steps")
    print(f"  v3_sae: {len(log_sae['val']['steps'])} val evals, {len(log_sae['train']['steps'])} train steps")

    # Step 6: Generate figures
    print("\n[Step 6/7] 生成图表...")
    plot_data_statistics(scan_results)
    plot_value_target_comparison(comparisons)
    if model_results:
        plot_model_predictions(model_results)
    plot_td_reward_impact(comparisons)
    plot_training_curves(log_so, log_sae)
    plot_inference_comparison(infer_so, infer_sae)

    # Step 7: Save JSON summary
    print("\n[Step 7/7] 保存分析数据...")
    summary = {
        "scan_results": {k: {kk: vv for kk, vv in v.items() if kk != "mismatch_trajs"}
                         for k, v in scan_results.items()},
        "total_trajs": total_all,
        "total_mismatch": total_mismatch,
        "mismatch_rate": total_mismatch / total_all * 100 if total_all > 0 else 0,
        "training": {
            "v3_so": {"best_mae": 0.0724, "final_val_mae": 0.0725, "steps": 12000, "batch_size": 128},
            "v3_sae": {"best_mae": 0.0463, "final_val_mae": 0.0466, "steps": 12000, "batch_size": 128},
        },
        "inference": {"v3_so": infer_so, "v3_sae": infer_sae},
        "diagnostics": infer_diag,
    }
    if model_results:
        summary["model_comparison"] = {
            "num_trajs": len(model_results),
            "mean_so": float(np.mean([np.mean(r["pred_so"]) for r in model_results])),
            "mean_sae": float(np.mean([np.mean(r["pred_sae"]) for r in model_results])),
            "mean_diff": float(np.mean([np.mean(r["pred_diff"]) for r in model_results])),
        }

    json_path = FIG_DIR / "v3_comparison_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  保存 {json_path}")

    print("\n" + "=" * 70)
    print("[ACP v3_so vs v3_sae 对比分析] 完成")
    print(f"  图表目录: {FIG_DIR}")
    print(f"  生成图表: v3cmp_fig1~fig6")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
