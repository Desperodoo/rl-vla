from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import tyro


TRAIN_METRIC_RE = re.compile(
    r"INFO (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"step:(?P<step>[0-9.]+K?) .*?"
    r"loss:(?P<loss>[-+0-9.eE]+) "
    r"grdn:(?P<grdn>[-+0-9.eE]+) "
    r"lr:(?P<lr>[-+0-9.eE]+) "
    r"updt_s:(?P<updt_s>[-+0-9.eE]+) "
    r"data_s:(?P<data_s>[-+0-9.eE]+)"
)
PROGRESS_RE = re.compile(
    r"Training:\s+\d+%.*?(?P<current>\d+)/(?P<total>\d+)\s*\[[^\]]*?,\s*(?P<rate>[0-9.]+)(?P<unit>s/step|step/s)\]"
)


@dataclass
class Args:
    output_dir: str
    report_path: str
    title: str = "PI05 Official vs Batch2 vs Batch64 Training / Eval Report"

    base_batch2_train: str = ""
    libero_batch2_train: str = ""
    base_batch64_train: str = ""
    libero_batch64_train: str = ""

    base_batch2_launch_config: str = ""
    libero_batch2_launch_config: str = ""
    base_batch64_launch_config: str = ""
    libero_batch64_launch_config: str = ""

    base_batch2_val: str = ""
    base_batch2_test: str = ""
    libero_batch2_val: str = ""
    libero_batch2_test: str = ""

    base_official_val: str | None = None
    base_official_test: str | None = None
    libero_official_val: str | None = None
    libero_official_test: str | None = None

    base_batch64_val: str | None = None
    base_batch64_test: str | None = None
    libero_batch64_val: str | None = None
    libero_batch64_test: str | None = None

    base_batch64_resource_monitor: str | None = None
    libero_batch64_resource_monitor: str | None = None

    carm_consistency_val: str | None = None
    carm_consistency_test: str | None = None
    carm_probe_comparison: str | None = None


def _parse_step_token(token: str) -> int:
    token = token.strip().upper()
    if token.endswith("K"):
        return int(float(token[:-1]) * 1000)
    return int(float(token))


def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None or math.isnan(value):
        return "pending"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "pending"
    return f"{value * 100:.1f}%"


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text())


def _load_launch_config(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_eval(path: str | None) -> dict[str, Any] | None:
    return _load_json(path)


def _load_carm_eval(path: str | None) -> dict[str, Any] | None:
    payload = _load_json(path)
    if payload is None:
        return None
    avg_metrics = payload.get("avg_metrics", payload)
    return {
        "avg_metrics": avg_metrics,
        "num_episodes": payload.get("num_episodes"),
        "timestamp": payload.get("timestamp"),
        "data_dir": payload.get("data_dir"),
    }


def _parse_train_metrics(path: str) -> pd.DataFrame:
    text = Path(path).read_text(errors="ignore")
    rows: list[dict[str, Any]] = []
    for match in TRAIN_METRIC_RE.finditer(text):
        rows.append(
            {
                "timestamp": pd.to_datetime(match.group("ts")),
                "step": _parse_step_token(match.group("step")),
                "loss": float(match.group("loss")),
                "grad_norm": float(match.group("grdn")),
                "lr": float(match.group("lr")),
                "update_s": float(match.group("updt_s")),
                "data_s": float(match.group("data_s")),
            }
        )
    return pd.DataFrame(rows)


def _parse_last_progress(path: str) -> dict[str, Any] | None:
    text = Path(path).read_text(errors="ignore")
    matches = list(PROGRESS_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    current = int(last.group("current"))
    total = int(last.group("total"))
    rate = float(last.group("rate"))
    unit = last.group("unit")
    seconds_per_step = rate if unit == "s/step" else 1.0 / rate
    return {
        "current_step": current,
        "total_steps": total,
        "seconds_per_step": seconds_per_step,
        "progress": current / total if total else None,
    }


def _parse_monitor(path: str | None, gpus: list[int]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for line in file_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        gpu_rows = [gpu for gpu in item.get("gpus", []) if int(gpu["index"]) in gpus]
        if not gpu_rows:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(item["time"], unit="s"),
                "memory_mean_gb": sum(g["memory_used_mib"] for g in gpu_rows) / len(gpu_rows) / 1024.0,
                "memory_max_gb": max(g["memory_used_mib"] for g in gpu_rows) / 1024.0,
                "util_mean_pct": sum(g["utilization_gpu_pct"] for g in gpu_rows) / len(gpu_rows),
            }
        )
    return pd.DataFrame(rows)


def _extract_gpu_list(launch_config: dict[str, Any]) -> list[int]:
    return [int(token.strip()) for token in str(launch_config["gpus"]).split(",") if token.strip()]


def _latest_metric(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None
    return df.sort_values("step").iloc[-1].to_dict()


def _status_row(
    label: str,
    metrics_df: pd.DataFrame,
    progress_info: dict[str, Any] | None,
    launch_config: dict[str, Any],
) -> dict[str, Any]:
    latest = _latest_metric(metrics_df)
    batch_size = int(launch_config["batch_size"])
    num_processes = int(launch_config["num_processes"])
    global_batch = batch_size * num_processes
    latest_step = int(progress_info["current_step"]) if progress_info else (int(latest["step"]) if latest else 0)
    total_steps = int(progress_info["total_steps"]) if progress_info else int(launch_config["steps"])
    sec_per_step = None
    if progress_info:
        sec_per_step = float(progress_info["seconds_per_step"])
    elif latest:
        sec_per_step = float(latest["update_s"] + latest["data_s"])
    remaining_hours = None
    eta_text = "pending"
    if sec_per_step is not None and total_steps and latest_step <= total_steps:
        remaining_hours = (total_steps - latest_step) * sec_per_step / 3600.0
        eta_text = f"{remaining_hours:.1f} h"

    sample_throughput = None
    if sec_per_step and sec_per_step > 0:
        sample_throughput = global_batch / sec_per_step

    latest_ts = latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if latest else "pending"
    return {
        "label": label,
        "latest_step": latest_step,
        "total_steps": total_steps,
        "progress": latest_step / total_steps if total_steps else None,
        "loss": float(latest["loss"]) if latest else None,
        "grad_norm": float(latest["grad_norm"]) if latest else None,
        "lr": float(latest["lr"]) if latest else None,
        "update_s": float(latest["update_s"]) if latest else None,
        "global_batch": global_batch,
        "sec_per_step": sec_per_step,
        "samples_per_sec": sample_throughput,
        "remaining_hours": remaining_hours,
        "latest_timestamp": latest_ts,
        "eta_text": eta_text,
    }


def _plot_training_curves(
    output_dir: Path,
    base_batch2_df: pd.DataFrame,
    base_batch64_df: pd.DataFrame,
    libero_batch2_df: pd.DataFrame,
    libero_batch64_df: pd.DataFrame,
) -> tuple[str, str]:
    loss_path = output_dir / "training_loss_curves.png"
    speed_path = output_dir / "training_update_time_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    pairs = [
        ("pi05_base", axes[0], base_batch2_df, base_batch64_df),
        ("pi05_libero", axes[1], libero_batch2_df, libero_batch64_df),
    ]
    for title, ax, batch2_df, batch64_df in pairs:
        if not batch2_df.empty:
            ax.plot(batch2_df["step"], batch2_df["loss"], label="batch2", linewidth=2)
        if not batch64_df.empty:
            ax.plot(batch64_df["step"], batch64_df["loss"], label="batch64", linewidth=2, linestyle="--")
        ax.set_title(f"{title} loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(loss_path, dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    pairs = [
        ("pi05_base", axes[0], base_batch2_df, base_batch64_df),
        ("pi05_libero", axes[1], libero_batch2_df, libero_batch64_df),
    ]
    for title, ax, batch2_df, batch64_df in pairs:
        if not batch2_df.empty:
            ax.plot(batch2_df["step"], batch2_df["update_s"], label="batch2", linewidth=2)
        if not batch64_df.empty:
            ax.plot(batch64_df["step"], batch64_df["update_s"], label="batch64", linewidth=2, linestyle="--")
        ax.set_title(f"{title} update_s")
        ax.set_xlabel("step")
        ax.set_ylabel("update_s")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(speed_path, dpi=200)
    plt.close(fig)
    return loss_path.name, speed_path.name


def _plot_batch64_resource_curves(
    output_dir: Path,
    base_monitor_df: pd.DataFrame,
    libero_monitor_df: pd.DataFrame,
) -> str | None:
    if base_monitor_df.empty and libero_monitor_df.empty:
        return None

    out_path = output_dir / "batch64_resource_curves.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True, sharex="col")
    pairs = [
        ("pi05_base", base_monitor_df, axes[0][0], axes[1][0]),
        ("pi05_libero", libero_monitor_df, axes[0][1], axes[1][1]),
    ]
    for title, df, mem_ax, util_ax in pairs:
        if df.empty:
            mem_ax.set_title(f"{title} memory (pending)")
            util_ax.set_title(f"{title} utilization (pending)")
            continue
        t0 = df["timestamp"].iloc[0]
        hours = (df["timestamp"] - t0).dt.total_seconds() / 3600.0
        mem_ax.plot(hours, df["memory_mean_gb"], label="mean", linewidth=2)
        mem_ax.plot(hours, df["memory_max_gb"], label="max", linewidth=2, linestyle="--")
        mem_ax.set_title(f"{title} GPU memory")
        mem_ax.set_xlabel("hours since launch")
        mem_ax.set_ylabel("GB")
        mem_ax.grid(alpha=0.3)
        mem_ax.legend()

        util_ax.plot(hours, df["util_mean_pct"], color="tab:orange", linewidth=2)
        util_ax.set_title(f"{title} GPU utilization")
        util_ax.set_xlabel("hours since launch")
        util_ax.set_ylabel("%")
        util_ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path.name


def _eval_metric_row(model_name: str, setup: str, split: str, data: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "model": model_name,
        "setup": setup,
        "split": split,
        "mse": None if data is None else float(data["mean_action_mse"]),
        "mae": None if data is None else float(data["mean_action_mae"]),
    }


def _carm_metric_row(model_name: str, split: str, data: dict[str, Any] | None) -> dict[str, Any]:
    avg = None if data is None else data["avg_metrics"]
    return {
        "model": model_name,
        "split": split,
        "total_mae": None if avg is None else float(avg["total_mae"]),
        "ee_mae": None if avg is None else float(avg["ee_mae"]),
        "pose_mae": None if avg is None else float(avg["pose_mae"]),
        "gripper_mae": None if avg is None else float(avg["gripper_joint_mae"]),
        "num_episodes": None if data is None else data.get("num_episodes"),
    }


def _plot_eval_overview(
    output_dir: Path,
    eval_rows: list[dict[str, Any]],
    carm_rows: list[dict[str, Any]] | None = None,
) -> str:
    out_path = output_dir / "eval_metric_overview.png"
    df = pd.DataFrame(eval_rows)
    order = ["base-val", "base-test", "libero-val", "libero-test"]
    setup_order = ["official", "batch2", "batch64"]
    setup_labels = {"official": "official", "batch2": "batch2", "batch64": "batch64"}
    metric_keys = [("mse", "mean_action_mse"), ("mae", "mean_action_mae")]

    include_carm = bool(carm_rows)
    fig, axes = plt.subplots(1, 3 if include_carm else 2, figsize=(20 if include_carm else 14, 5), constrained_layout=True)
    axes_list = list(axes if isinstance(axes, (list, tuple)) else axes.flat if hasattr(axes, "flat") else [axes])
    for ax, (metric_key, title) in zip(axes_list[:2], metric_keys):
        x = range(len(order))
        width = 0.24
        all_series: dict[str, list[float]] = {setup: [] for setup in setup_order}
        pending_marks: list[tuple[float, float]] = []
        for idx, label in enumerate(order):
            model_tag, split = label.split("-")
            model = "pi05_base" if model_tag == "base" else "pi05_libero"
            for setup_idx, setup in enumerate(setup_order):
                series = df[(df["model"] == model) & (df["setup"] == setup) & (df["split"] == split)][metric_key]
                value = series.iloc[0] if not series.empty else math.nan
                all_series[setup].append(value)

        max_visible = max(
            [value for values in all_series.values() for value in values if not math.isnan(value)],
            default=0.01,
        )
        offsets = [-width, 0.0, width]
        for setup, offset in zip(setup_order, offsets):
            visible_values = [0.0 if math.isnan(v) else v for v in all_series[setup]]
            ax.bar([idx + offset for idx in x], visible_values, width=width, label=setup_labels[setup])
            for idx, value in enumerate(all_series[setup]):
                if math.isnan(value):
                    pending_marks.append((idx + offset, max_visible * 0.04))
        for x_pos, y_pos in pending_marks:
            ax.text(x_pos, y_pos, "pending", rotation=90, ha="center", va="bottom")
        ax.set_xticks(list(x))
        ax.set_xticklabels(order)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()

    if include_carm:
        carm_ax = axes_list[2]
        carm_df = pd.DataFrame(carm_rows)
        split_order = ["val", "test"]
        metric_order = ["total_mae", "pose_mae", "gripper_mae"]
        metric_labels = {
            "total_mae": "total_mae",
            "pose_mae": "pose_mae",
            "gripper_mae": "gripper_mae",
        }
        width = 0.22
        offsets = [-width, 0.0, width]
        max_visible = 0.01
        for idx, metric_key in enumerate(metric_order):
            values = []
            for split in split_order:
                series = carm_df[carm_df["split"] == split][metric_key]
                value = series.iloc[0] if not series.empty else math.nan
                values.append(value)
                if not math.isnan(value):
                    max_visible = max(max_visible, float(value))
            carm_ax.bar(
                [split_idx + offsets[idx] for split_idx in range(len(split_order))],
                [0.0 if math.isnan(v) else v for v in values],
                width=width,
                label=metric_labels[metric_key],
            )
            for split_idx, value in enumerate(values):
                if math.isnan(value):
                    carm_ax.text(split_idx + offsets[idx], max_visible * 0.04, "pending", rotation=90, ha="center", va="bottom")
        carm_ax.set_xticks(list(range(len(split_order))))
        carm_ax.set_xticklabels(split_order)
        carm_ax.set_title("consistency_flow_resnet18 (eval_carm metrics)")
        carm_ax.set_ylabel("MAE")
        carm_ax.grid(alpha=0.3, axis="y")
        carm_ax.legend()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path.name


def _plot_per_dim_mae(output_dir: Path, eval_payloads: dict[str, dict[str, Any] | None]) -> str:
    out_path = output_dir / "per_dim_mae_val.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    configs = [
        (
            "pi05_base",
            axes[0],
            [
                ("official", eval_payloads.get("base_official_val")),
                ("batch2", eval_payloads["base_batch2_val"]),
                ("batch64", eval_payloads.get("base_batch64_val")),
            ],
        ),
        (
            "pi05_libero",
            axes[1],
            [
                ("official", eval_payloads.get("libero_official_val")),
                ("batch2", eval_payloads["libero_batch2_val"]),
                ("batch64", eval_payloads.get("libero_batch64_val")),
            ],
        ),
    ]
    for title, ax, eval_configs in configs:
        reference_eval = next(data for _, data in eval_configs if data is not None)
        reference_vals = reference_eval["per_dim_mae"]
        dims = list(range(1, len(reference_vals) + 1))
        width = 0.24
        offsets = {"official": -width, "batch2": 0.0, "batch64": width}
        max_visible = max(reference_vals)
        for label, payload in eval_configs:
            x_positions = [dim + offsets[label] for dim in dims]
            if payload is None:
                for x_pos in x_positions:
                    ax.text(x_pos, max_visible * 0.05, "pending", rotation=90, ha="center", va="bottom")
                continue
            values = payload["per_dim_mae"]
            max_visible = max(max_visible, max(values))
            ax.bar(x_positions, values, width=width, label=label)
        ax.set_title(f"{title} val per-dim MAE")
        ax.set_xlabel("action dimension")
        ax.set_ylabel("MAE")
        ax.set_xticks(dims)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path.name


def _plot_per_episode_mae(output_dir: Path, eval_payloads: dict[str, dict[str, Any] | None]) -> str:
    out_path = output_dir / "per_episode_mae_val.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    configs = [
        (
            "pi05_base",
            axes[0],
            [
                ("official", eval_payloads.get("base_official_val"), ":"),
                ("batch2", eval_payloads["base_batch2_val"], "-"),
                ("batch64", eval_payloads.get("base_batch64_val"), "--"),
            ],
        ),
        (
            "pi05_libero",
            axes[1],
            [
                ("official", eval_payloads.get("libero_official_val"), ":"),
                ("batch2", eval_payloads["libero_batch2_val"], "-"),
                ("batch64", eval_payloads.get("libero_batch64_val"), "--"),
            ],
        ),
    ]
    for title, ax, eval_configs in configs:
        max_visible = 0.01
        has_pending = False
        for label, payload, linestyle in eval_configs:
            if payload is None:
                has_pending = True
                continue
            values = sorted(float(v) for v in payload["per_episode_mean_mae"].values())
            max_visible = max(max_visible, max(values))
            ax.plot(values, label=label, linewidth=2, linestyle=linestyle)
        if has_pending:
            ax.text(0.98, 0.05, "pending setup omitted", transform=ax.transAxes, ha="right", va="bottom")
        ax.set_title(f"{title} val per-episode mean MAE")
        ax.set_xlabel("episode (sorted)")
        ax.set_ylabel("mean MAE")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path.name


def _markdown_status_table(rows: list[dict[str, Any]]) -> str:
    header = [
        "| run | step | progress | loss | grad_norm | lr | update_s | global_batch | samples/s | ETA to finish | last metric ts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    lines = []
    for row in rows:
        lines.append(
            "| {label} | {latest_step}/{total_steps} | {progress} | {loss} | {grad_norm} | {lr:.2e} | {update_s} | {global_batch} | {samples_per_sec} | {eta_text} | {latest_timestamp} |".format(
                label=row["label"],
                latest_step=row["latest_step"],
                total_steps=row["total_steps"],
                progress=_fmt_pct(row["progress"]),
                loss=_fmt_float(row["loss"], 4),
                grad_norm=_fmt_float(row["grad_norm"], 4),
                lr=row["lr"] if row["lr"] is not None else float("nan"),
                update_s=_fmt_float(row["update_s"], 3),
                global_batch=row["global_batch"],
                samples_per_sec=_fmt_float(row["samples_per_sec"], 1),
                eta_text=row["eta_text"],
                latest_timestamp=row["latest_timestamp"],
            )
        )
    return "\n".join(header + lines)


def _markdown_eval_table(eval_rows: list[dict[str, Any]]) -> str:
    header = [
        "| model | setup | split | mean_action_mse | mean_action_mae |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    lines = []
    for row in eval_rows:
        lines.append(
            f"| {row['model']} | {row['setup']} | {row['split']} | {_fmt_float(row['mse'], 9)} | {_fmt_float(row['mae'], 9)} |"
        )
    return "\n".join(header + lines)


def _markdown_carm_eval_table(carm_rows: list[dict[str, Any]]) -> str:
    header = [
        "| model | split | total_mae | ee_mae | pose_mae | gripper_mae | episodes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines = []
    for row in carm_rows:
        lines.append(
            "| {model} | {split} | {total_mae} | {ee_mae} | {pose_mae} | {gripper_mae} | {episodes} |".format(
                model=row["model"],
                split=row["split"],
                total_mae=_fmt_float(row["total_mae"], 6),
                ee_mae=_fmt_float(row["ee_mae"], 6),
                pose_mae=_fmt_float(row["pose_mae"], 6),
                gripper_mae=_fmt_float(row["gripper_mae"], 6),
                episodes=row["num_episodes"] if row["num_episodes"] is not None else "pending",
            )
        )
    return "\n".join(header + lines)


def main() -> None:
    args = tyro.cli(Args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    base_batch2_df = _parse_train_metrics(args.base_batch2_train)
    libero_batch2_df = _parse_train_metrics(args.libero_batch2_train)
    base_batch64_df = _parse_train_metrics(args.base_batch64_train)
    libero_batch64_df = _parse_train_metrics(args.libero_batch64_train)

    base_batch2_cfg = _load_launch_config(args.base_batch2_launch_config)
    libero_batch2_cfg = _load_launch_config(args.libero_batch2_launch_config)
    base_batch64_cfg = _load_launch_config(args.base_batch64_launch_config)
    libero_batch64_cfg = _load_launch_config(args.libero_batch64_launch_config)

    base_batch2_status = _status_row(
        "pi05_base batch2",
        base_batch2_df,
        _parse_last_progress(args.base_batch2_train),
        base_batch2_cfg,
    )
    libero_batch2_status = _status_row(
        "pi05_libero batch2",
        libero_batch2_df,
        _parse_last_progress(args.libero_batch2_train),
        libero_batch2_cfg,
    )
    base_batch64_status = _status_row(
        "pi05_base batch64",
        base_batch64_df,
        _parse_last_progress(args.base_batch64_train),
        base_batch64_cfg,
    )
    libero_batch64_status = _status_row(
        "pi05_libero batch64",
        libero_batch64_df,
        _parse_last_progress(args.libero_batch64_train),
        libero_batch64_cfg,
    )
    status_rows = [base_batch2_status, libero_batch2_status, base_batch64_status, libero_batch64_status]

    eval_payloads = {
        "base_official_val": _load_eval(args.base_official_val),
        "base_official_test": _load_eval(args.base_official_test),
        "base_batch2_val": _load_eval(args.base_batch2_val),
        "base_batch2_test": _load_eval(args.base_batch2_test),
        "libero_official_val": _load_eval(args.libero_official_val),
        "libero_official_test": _load_eval(args.libero_official_test),
        "libero_batch2_val": _load_eval(args.libero_batch2_val),
        "libero_batch2_test": _load_eval(args.libero_batch2_test),
        "base_batch64_val": _load_eval(args.base_batch64_val),
        "base_batch64_test": _load_eval(args.base_batch64_test),
        "libero_batch64_val": _load_eval(args.libero_batch64_val),
        "libero_batch64_test": _load_eval(args.libero_batch64_test),
    }
    carm_payloads = {
        "val": _load_carm_eval(args.carm_consistency_val),
        "test": _load_carm_eval(args.carm_consistency_test),
    }
    carm_probe = _load_json(args.carm_probe_comparison)
    eval_rows = [
        _eval_metric_row("pi05_base", "official", "val", eval_payloads["base_official_val"]),
        _eval_metric_row("pi05_base", "official", "test", eval_payloads["base_official_test"]),
        _eval_metric_row("pi05_base", "batch2", "val", eval_payloads["base_batch2_val"]),
        _eval_metric_row("pi05_base", "batch2", "test", eval_payloads["base_batch2_test"]),
        _eval_metric_row("pi05_base", "batch64", "val", eval_payloads["base_batch64_val"]),
        _eval_metric_row("pi05_base", "batch64", "test", eval_payloads["base_batch64_test"]),
        _eval_metric_row("pi05_libero", "official", "val", eval_payloads["libero_official_val"]),
        _eval_metric_row("pi05_libero", "official", "test", eval_payloads["libero_official_test"]),
        _eval_metric_row("pi05_libero", "batch2", "val", eval_payloads["libero_batch2_val"]),
        _eval_metric_row("pi05_libero", "batch2", "test", eval_payloads["libero_batch2_test"]),
        _eval_metric_row("pi05_libero", "batch64", "val", eval_payloads["libero_batch64_val"]),
        _eval_metric_row("pi05_libero", "batch64", "test", eval_payloads["libero_batch64_test"]),
    ]
    carm_rows = [
        _carm_metric_row("consistency_flow_resnet18", "val", carm_payloads["val"]),
        _carm_metric_row("consistency_flow_resnet18", "test", carm_payloads["test"]),
    ]

    base_monitor_df = _parse_monitor(
        args.base_batch64_resource_monitor,
        _extract_gpu_list(base_batch64_cfg),
    )
    libero_monitor_df = _parse_monitor(
        args.libero_batch64_resource_monitor,
        _extract_gpu_list(libero_batch64_cfg),
    )

    loss_plot, speed_plot = _plot_training_curves(
        output_dir,
        base_batch2_df,
        base_batch64_df,
        libero_batch2_df,
        libero_batch64_df,
    )
    resource_plot = _plot_batch64_resource_curves(output_dir, base_monitor_df, libero_monitor_df)
    eval_overview_plot = _plot_eval_overview(output_dir, eval_rows, carm_rows)
    per_dim_plot = _plot_per_dim_mae(output_dir, eval_payloads)
    per_episode_plot = _plot_per_episode_mae(output_dir, eval_payloads)

    official_eval_ready = all(
        eval_payloads[key] is not None
        for key in ["base_official_val", "base_official_test", "libero_official_val", "libero_official_test"]
    )
    batch64_eval_ready = all(
        eval_payloads[key] is not None
        for key in ["base_batch64_val", "base_batch64_test", "libero_batch64_val", "libero_batch64_test"]
    )
    carm_eval_ready = all(carm_payloads[key] is not None for key in ["val", "test"])

    observations = [
        f"- `pi05_base batch64` 当前已到 `{base_batch64_status['latest_step']}/{base_batch64_status['total_steps']}`，预计剩余 `{base_batch64_status['eta_text']}`。",
        f"- `pi05_libero batch64` 当前已到 `{libero_batch64_status['latest_step']}/{libero_batch64_status['total_steps']}`，预计剩余 `{libero_batch64_status['eta_text']}`。",
        f"- `official/untrained` eval {'已经齐全' if official_eval_ready else '正在补跑或待回收'}；第 4 节统一按 `official / batch2 / batch64` 三组 setup 展示。",
        f"- `batch2` 已有完整 val/test eval；`batch64` eval {'已经齐全' if batch64_eval_ready else '仍在等待 020000 checkpoint 触发'}。",
        f"- `consistency_flow_resnet18` 离线评估 {'已经齐全' if carm_eval_ready else '正在运行或待回收'}；由于 `eval_carm.py` 与 `eval_pi05.py` 指标族不同，第 4 节使用同一张总览图中的独立子图展示。",
        f"- 从最近 logged `update_s` 看，`batch64` 单步开销约为 batch2 的 {base_batch64_status['update_s'] / base_batch2_status['update_s']:.1f}x（base）和 {libero_batch64_status['update_s'] / libero_batch2_status['update_s']:.1f}x（libero）。",
    ]
    if carm_probe is not None:
        improvement = None
        if "improvement" in carm_probe:
            improvement = float(carm_probe["improvement"])
        elif "avg_regular" in carm_probe and "avg_ema" in carm_probe:
            regular_total = float(carm_probe["avg_regular"]["total_mae"])
            ema_total = float(carm_probe["avg_ema"]["total_mae"])
            if regular_total != 0:
                improvement = (regular_total - ema_total) / regular_total * 100.0
        if improvement is not None:
            observations.append(
                f"- `consistency_flow_resnet18` 的 EMA probe 已完成，`EMA` 相比 `non-EMA` 的 `total_mae` 改善约 {improvement:+.2f}%（基于 probe episodes）。"
            )

    lines = [
        f"# {args.title}",
        "",
        f"- Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Report assets dir: `{output_dir}`",
        "",
        "## 1. 训练状态总览",
        "",
        _markdown_status_table(status_rows),
        "",
        "### 1.1 关键信息",
        "",
        *observations,
        "",
        "## 2. 训练曲线",
        "",
        f"![training_loss_curves]({output_dir.name}/{loss_plot})",
        "",
        f"![training_update_time_curves]({output_dir.name}/{speed_plot})",
        "",
    ]
    if resource_plot is not None:
        lines.extend(
            [
                "## 3. batch64 资源监控",
                "",
                f"![batch64_resource_curves]({output_dir.name}/{resource_plot})",
                "",
            ]
        )
    lines.extend(
        [
            "## 4. 离线评估与 official-vs-batch2-vs-batch64 对比",
            "",
            "### 4.1 PI05",
            "",
            _markdown_eval_table(eval_rows),
            "",
            f"![per_dim_mae_val]({output_dir.name}/{per_dim_plot})",
            "",
            f"![per_episode_mae_val]({output_dir.name}/{per_episode_plot})",
            "",
            "### 4.2 PI05 + consistency_flow_resnet18 总览图",
            "",
            "- 左两幅子图为 PI05 的 `mean_action_mse / mean_action_mae`。",
            "- 右侧子图为 `consistency_flow_resnet18` 的 `eval_carm.py` 指标：`total_mae / pose_mae / gripper_mae`。",
            "- 两套指标不是严格同口径，因此放在同一张总览图中做并列展示，而不是强行混成同一根柱子。",
            "",
            f"![eval_metric_overview]({output_dir.name}/{eval_overview_plot})",
            "",
            "### 4.3 consistency_flow_resnet18",
            "",
            _markdown_carm_eval_table(carm_rows),
            "",
            "## 5. 结论",
            "",
        ]
    )
    if batch64_eval_ready:
        lines.extend(
            [
                "- `batch64` 四份 eval 已齐，可以直接把本报告视为最终对比版。",
                "- 建议下一步把 `batch64` 的 checkpoint 和 `batch2` 结果一起做更正式的结论归纳。",
            ]
        )
    else:
        lines.extend(
            [
                "- 当前报告是进度版：训练曲线和资源曲线已经能稳定反映 batch64 的中期状态。",
                "- `batch64` 四份 eval json 一旦落盘，只需要用同一脚本再跑一次，就会自动刷新成最终对比版。",
                "- 远端现有 watcher 仍在等待 `020000/pretrained_model`，当前不需要额外人工补触发。",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n")
    print(f"report: {report_path}")
    print(f"assets: {output_dir}")


if __name__ == "__main__":
    main()
