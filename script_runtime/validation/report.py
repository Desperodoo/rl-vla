"""Visualization helpers for script-runtime validation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_validation_report(
    summary_path: str | Path,
    output_dir: str | Path,
    timeline_episode_index: int = 0,
) -> Dict[str, Path]:
    summary = load_json(summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_png = output_dir / "success_overview.png"
    failure_png = output_dir / "failure_codes.png"
    timeline_png = output_dir / f"episode_{timeline_episode_index:03d}_timeline.png"
    markdown_path = output_dir / "REPORT.md"

    _plot_success_overview(summary, success_png)
    _plot_failure_codes(summary, failure_png)

    trace_path = _resolve_trace_path(summary, timeline_episode_index)
    if trace_path is not None and trace_path.exists():
        rows = load_jsonl(trace_path)
        _plot_timeline(rows, timeline_png)
    else:
        timeline_png = None

    markdown_path.write_text(
        _build_markdown(summary, success_png, failure_png, timeline_png, trace_path),
        encoding="utf-8",
    )
    outputs = {
        "report_md": markdown_path,
        "success_png": success_png,
        "failure_png": failure_png,
    }
    if timeline_png is not None:
        outputs["timeline_png"] = timeline_png
    return outputs


def _resolve_trace_path(summary: Dict[str, Any], episode_index: int) -> Path | None:
    episodes = summary.get("episode_results", [])
    for row in episodes:
        if int(row.get("episode_index", -1)) == episode_index and row.get("trace_path"):
            return Path(row["trace_path"])
    return None


def _plot_success_overview(summary: Dict[str, Any], out_path: Path) -> None:
    labels = ["Runtime Success", "Sim Success", "Episodes"]
    values = [
        int(summary.get("runtime_successes", 0)),
        int(summary.get("sim_successes", 0)),
        int(summary.get("num_episodes", 0)),
    ]
    colors = ["#2E7D32", "#1565C0", "#616161"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Script Runtime Validation Overview")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_failure_codes(summary: Dict[str, Any], out_path: Path) -> None:
    counts = dict(summary.get("failure_code_counts", {}))
    if not counts:
        counts = {"NONE": int(summary.get("num_episodes", 0))}
    labels = list(counts.keys())
    values = [int(counts[key]) for key in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color="#C62828")
    ax.set_title("Failure Code Distribution")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center", va="bottom")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_timeline(rows: List[Dict[str, Any]], out_path: Path) -> None:
    labels = [row.get("skill_name", "unknown") for row in rows]
    durations = [float(row.get("duration_s", 0.0)) for row in rows]
    colors = ["#2E7D32" if row.get("failure_code") in ("", "NONE", None) else "#C62828" for row in rows]

    fig_height = max(3.5, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_pos = list(range(len(labels)))
    ax.barh(y_pos, durations, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Duration (s)")
    ax.set_title("Episode Skill Timeline")
    ax.grid(axis="x", alpha=0.25)

    for idx, row in enumerate(rows):
        failure = row.get("failure_code", "NONE")
        ax.text(durations[idx] + 0.001, idx, failure, va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _build_markdown(
    summary: Dict[str, Any],
    success_png: Path,
    failure_png: Path,
    timeline_png: Path | None,
    trace_path: Path | None,
) -> str:
    lines = [
        "# Script Runtime Validation Report",
        "",
        f"- Episodes: {summary.get('num_episodes', 0)}",
        f"- Runtime successes: {summary.get('runtime_successes', 0)}",
        f"- Sim successes: {summary.get('sim_successes', 0)}",
        f"- Failure code counts: `{summary.get('failure_code_counts', {})}`",
        f"- Failed skill counts: `{summary.get('failed_skill_counts', {})}`",
        "",
        "## Visuals",
        "",
        f"![Success Overview]({success_png.name})",
        "",
        f"![Failure Codes]({failure_png.name})",
        "",
    ]
    if timeline_png is not None:
        lines.extend(
            [
                f"![Episode Timeline]({timeline_png.name})",
                "",
            ]
        )
    if trace_path is not None:
        lines.append(f"- Timeline source: `{trace_path}`")
    episodes = summary.get("episode_results", [])
    if episodes:
        lines.extend(["", "## Episodes", ""])
        for row in episodes:
            lines.append(
                f"- Episode {row.get('episode_index')}: task={row.get('task_status')} sim_success={row.get('sim_success')}"
            )
            if row.get("rollout_gif"):
                lines.append(f"  rollout: `{row.get('rollout_gif')}`")
            if row.get("grounding_topdown_png"):
                lines.append(f"  grounding: `{row.get('grounding_topdown_png')}`")
            if row.get("grounding_json"):
                lines.append(f"  grounding_json: `{row.get('grounding_json')}`")
    lines.append("")
    return "\n".join(lines)
