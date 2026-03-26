#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/home/wjz/rl-vla")
OUT_DIR = PROJECT_ROOT / "docs" / "vlaw" / "figures" / "acp_retention_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})


@dataclass
class RetentionRow:
    source: str
    algo: str
    reward: str
    config: str
    best_so: float
    best_sae: float
    final_sae: float
    retention_ratio: float
    gap: float
    note: str = ""


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _safe_retention(best_so: float, best_sae: float) -> float:
    if best_so <= 1e-8:
        return 0.0
    return best_sae / best_so


def build_rows() -> tuple[list[RetentionRow], dict]:
    rows: list[RetentionRow] = []

    v3_rlpd = _load_json(PROJECT_ROOT / "docs/vlaw/figures/v3_rlpd/rlpd_results_summary.json")
    v3_cmp = _load_json(PROJECT_ROOT / "docs/vlaw/figures/v3_comparison/v3_comparison_summary.json")
    v7_diag = _load_json(PROJECT_ROOT / "docs/vlaw/figures/rlpd_acp_v7_diag/diagnosis_summary.json")

    # v3 ACP rows
    for key, value in v3_rlpd.items():
        algo, reward_tag = key.split("+")
        best_so = float(value["best_so"])
        best_sae = float(value["best_sae"])
        final_sae = float(value["final_sae"])
        rows.append(
            RetentionRow(
                source="v3",
                algo=algo.lower(),
                reward="acp",
                config=reward_tag,
                best_so=best_so,
                best_sae=best_sae,
                final_sae=final_sae,
                retention_ratio=_safe_retention(best_so, best_sae),
                gap=best_so - best_sae,
                note="ACP v3 online runs",
            )
        )

    # v5 rows from archived report table
    v5_table = {
        "awsc_v_reward": ("awsc", "acp", 0.96, 0.58, 0.46, "v5 potential reward best SO"),
        "awsc_td_clip": ("awsc", "acp", 0.90, 0.70, 0.52, "v5 td+clip best SAE"),
        "pld_stable_g05": ("pld", "acp", 0.92, 0.04, 0.00, "v5 stable PLD"),
        "pld_v_reward_sae": ("pld", "acp", 0.92, 0.04, 0.00, "v5 potential+sAE PLD"),
        "dsrl_v_reward_g05": ("dsrl", "acp", 0.92, 0.08, 0.02, "v5 best DSRL SAE"),
        "dsrl_stable_g05": ("dsrl", "acp", 0.90, 0.04, 0.00, "v5 stable DSRL"),
    }
    for config, (algo, reward, best_so, best_sae, final_sae, note) in v5_table.items():
        rows.append(
            RetentionRow(
                source="v5",
                algo=algo,
                reward=reward,
                config=config,
                best_so=best_so,
                best_sae=best_sae,
                final_sae=final_sae,
                retention_ratio=_safe_retention(best_so, best_sae),
                gap=best_so - best_sae,
                note=note,
            )
        )

    # v6 rows from archived report table
    v6_table = {
        "pld_entropy_grasp": ("pld", "acp", 0.86, 0.04, 0.02, "v6 best PLD SO"),
        "dsrl_grasp1_td": ("dsrl", "acp", 0.92, 0.04, 0.02, "v6 grasp bonus baseline"),
        "dsrl_long_grasp": ("dsrl", "acp", 0.92, 0.14, 0.02, "v6 long training breakthrough"),
    }
    for config, (algo, reward, best_so, best_sae, final_sae, note) in v6_table.items():
        rows.append(
            RetentionRow(
                source="v6",
                algo=algo,
                reward=reward,
                config=config,
                best_so=best_so,
                best_sae=best_sae,
                final_sae=final_sae,
                retention_ratio=_safe_retention(best_so, best_sae),
                gap=best_so - best_sae,
                note=note,
            )
        )

    # v7 direct sim/acp controlled comparison
    for run_name, value in v7_diag.items():
        algo = str(value["algo"])
        reward = "acp" if "_acp_" in run_name else "sim"
        best_so = float(value["best_success_once"])
        best_sae = float(value["best_success_at_end"])
        final_sae = float(value["final_success_at_end"])
        rows.append(
            RetentionRow(
                source="v7",
                algo=algo,
                reward=reward,
                config=run_name,
                best_so=best_so,
                best_sae=best_sae,
                final_sae=final_sae,
                retention_ratio=float(value.get("sae_retention", _safe_retention(best_so, best_sae))),
                gap=best_so - best_sae,
                note="v7 controlled sim/acp comparison",
            )
        )

    aux = {
        "v3_mismatch_rate": float(v3_cmp["mismatch_rate"]),
        "v3_total_mismatch": int(v3_cmp["total_mismatch"]),
        "v3_total_trajs": int(v3_cmp["total_trajs"]),
        "v3_inference": v3_cmp["inference"],
        "v3_model_comparison": v3_cmp["model_comparison"],
        "v3_scan_results": v3_cmp["scan_results"],
    }
    return rows, aux


def save_summary(rows: list[RetentionRow], aux: dict) -> None:
    payload = {
        "rows": [asdict(r) for r in rows],
        "aux": aux,
    }
    (OUT_DIR / "retention_summary.json").write_text(json.dumps(payload, indent=2))


def plot_v7_retention_matrix(rows: list[RetentionRow]) -> None:
    v7 = [r for r in rows if r.source == "v7"]
    algos = ["awsc", "dsrl", "pld"]
    rewards = ["sim", "acp"]

    best_sae = np.zeros((len(algos), len(rewards)))
    best_so = np.zeros((len(algos), len(rewards)))
    retention = np.zeros((len(algos), len(rewards)))
    gap = np.zeros((len(algos), len(rewards)))

    for i, algo in enumerate(algos):
        for j, reward in enumerate(rewards):
            row = next(r for r in v7 if r.algo == algo and r.reward == reward)
            best_so[i, j] = row.best_so
            best_sae[i, j] = row.best_sae
            retention[i, j] = row.retention_ratio
            gap[i, j] = row.gap

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    im0 = axes[0].imshow(retention, cmap="YlGn", vmin=0.0, vmax=1.0)
    axes[0].set_xticks(range(len(rewards)), rewards)
    axes[0].set_yticks(range(len(algos)), [a.upper() for a in algos])
    axes[0].set_title("V7 retention ratio = SAE / SO")
    for i in range(len(algos)):
        for j in range(len(rewards)):
            axes[0].text(j, i, f"{retention[i,j]:.3f}\nSO={best_so[i,j]:.2f}\nSAE={best_sae[i,j]:.2f}",
                         ha="center", va="center", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(gap, cmap="OrRd", vmin=0.0, vmax=1.0)
    axes[1].set_xticks(range(len(rewards)), rewards)
    axes[1].set_yticks(range(len(algos)), [a.upper() for a in algos])
    axes[1].set_title("V7 SO-SAE gap")
    for i in range(len(algos)):
        for j in range(len(rewards)):
            axes[1].text(j, i, f"{gap[i,j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("ACP v7 controlled comparison: retention is algorithm-dependent, not ACP-only", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_v7_retention_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_v7_four_quadrants(rows: list[RetentionRow]) -> None:
    v7 = [r for r in rows if r.source == "v7"]
    order = [
        ("awsc", "sim"), ("awsc", "acp"),
        ("dsrl", "sim"), ("dsrl", "acp"),
        ("pld", "sim"), ("pld", "acp"),
    ]
    labels: list[str] = []
    stable_success: list[float] = []   # A = SO=True, SAE=True ≈ SAE
    drop_after_success: list[float] = []  # B = SO=True, SAE=False ≈ SO-SAE
    never_success: list[float] = []    # D = 1-SO

    for algo, reward in order:
        row = next(r for r in v7 if r.algo == algo and r.reward == reward)
        labels.append(f"{algo.upper()}\n{reward}")
        stable_success.append(row.best_sae)
        drop_after_success.append(max(0.0, row.best_so - row.best_sae))
        never_success.append(max(0.0, 1.0 - row.best_so))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, stable_success, label="A: SO=True, SAE=True", color="#4CAF50")
    ax.bar(x, drop_after_success, bottom=stable_success, label="B: SO=True, SAE=False", color="#FF9800")
    ax.bar(x, never_success, bottom=np.array(stable_success) + np.array(drop_after_success),
           label="D: SO=False, SAE=False", color="#BDBDBD")

    for i, (a, b, d) in enumerate(zip(stable_success, drop_after_success, never_success)):
        ax.text(i, a / 2 if a > 0.03 else a + 0.02, f"A {a:.2f}", ha="center", va="center", fontsize=8)
        if b > 0.05:
            ax.text(i, a + b / 2, f"B {b:.2f}", ha="center", va="center", fontsize=8)
        if d > 0.08:
            ax.text(i, a + b + d / 2, f"D {d:.2f}", ha="center", va="center", fontsize=8)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Episode fraction (approximated from best SO/SAE)")
    ax.set_xticks(x, labels)
    ax.set_title("Four-quadrant decomposition: PLD/DSRL fail mostly as drop-after-success")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_v7_four_quadrants.png", bbox_inches="tight")
    plt.close(fig)


def plot_cross_version_best(rows: list[RetentionRow]) -> None:
    selected = [
        next(r for r in rows if r.source == "v3" and r.config == "v3_so" and r.algo == "awsc"),
        next(r for r in rows if r.source == "v5" and r.config == "awsc_td_clip"),
        next(r for r in rows if r.source == "v7" and r.config == "awsc_v7_diag_acp_s42"),
        next(r for r in rows if r.source == "v3" and r.config == "v3_so" and r.algo == "dsrl"),
        next(r for r in rows if r.source == "v5" and r.config == "dsrl_v_reward_g05"),
        next(r for r in rows if r.source == "v6" and r.config == "dsrl_long_grasp"),
        next(r for r in rows if r.source == "v7" and r.config == "dsrl_v7_diag_acp_s42"),
        next(r for r in rows if r.source == "v3" and r.config == "v3_so" and r.algo == "pld"),
        next(r for r in rows if r.source == "v5" and r.config == "pld_stable_g05"),
        next(r for r in rows if r.source == "v6" and r.config == "pld_entropy_grasp"),
        next(r for r in rows if r.source == "v7" and r.config == "pld_v7_diag_acp_s42"),
    ]

    labels = [
        "AWSC\nv3_so", "AWSC\nv5_td_clip", "AWSC\nv7_acp",
        "DSRL\nv3_so", "DSRL\nv5_v_reward", "DSRL\nv6_long", "DSRL\nv7_acp",
        "PLD\nv3_so", "PLD\nv5_stable", "PLD\nv6_entropy", "PLD\nv7_acp",
    ]
    so = [r.best_so for r in selected]
    sae = [r.best_sae for r in selected]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.bar(x - width / 2, so, width, label="Best SO", color="#64B5F6")
    ax.bar(x + width / 2, sae, width, label="Best SAE", color="#81C784")
    for i, (so_i, sae_i) in enumerate(zip(so, sae)):
        ax.plot([i - width / 2, i + width / 2], [so_i, sae_i], color="#F44336", linewidth=1.2)
        ax.text(i + width / 2, sae_i + 0.02, f"gap={so_i - sae_i:.2f}", ha="center", fontsize=7, rotation=90)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Best metric")
    ax.set_title("Across ACP track, the bottleneck is persistent SO→SAE retention")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_cross_version_best_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_v3_mismatch_bridge(aux: dict) -> None:
    scan = aux["v3_scan_results"]
    datasets = list(scan.keys())
    mismatch_rates = [scan[k]["mismatch"] / max(1, scan[k]["total"]) for k in datasets]

    inf = aux["v3_inference"]
    mae_vals = [inf["v3_so"]["mae"], inf["v3_sae"]["mae"]]
    adv_std_vals = [inf["v3_so"]["advantage_std"], inf["v3_sae"]["advantage_std"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    x0 = np.arange(len(datasets))
    axes[0].bar(x0, mismatch_rates, color="#FFB74D")
    axes[0].set_xticks(x0, [d.replace(": ", "\n") for d in datasets], rotation=0)
    axes[0].set_ylim(0, 0.35)
    axes[0].set_ylabel("Mismatch rate")
    axes[0].set_title("v3 dataset mismatch: SO=True, SAE=False")
    for i, rate in enumerate(mismatch_rates):
        axes[0].text(i, rate + 0.01, f"{rate:.1%}", ha="center", fontsize=8)

    x1 = np.arange(2)
    axes[1].bar(x1, mae_vals, color=["#4CAF50", "#FF5722"])
    axes[1].set_xticks(x1, ["v3_so", "v3_sae"])
    axes[1].set_ylabel("Inference MAE")
    axes[1].set_title("v3_sae predicts value more accurately")
    for i, val in enumerate(mae_vals):
        axes[1].text(i, val + 0.002, f"{val:.3f}", ha="center", fontsize=8)

    axes[2].bar(x1, adv_std_vals, color=["#4CAF50", "#FF5722"])
    axes[2].set_xticks(x1, ["v3_so", "v3_sae"])
    axes[2].set_ylabel("Advantage std")
    axes[2].set_title("v3_sae advantage is more concentrated")
    for i, val in enumerate(adv_std_vals):
        axes[2].text(i, val + 0.003, f"{val:.3f}", ha="center", fontsize=8)

    fig.suptitle("Mismatch evidence bridge: label semantics matter, but do not fully solve retention")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_v3_mismatch_bridge.png", bbox_inches="tight")
    plt.close(fig)


def write_report(rows: list[RetentionRow], aux: dict) -> None:
    v7 = [r for r in rows if r.source == "v7"]
    awsc_sim = next(r for r in v7 if r.algo == "awsc" and r.reward == "sim")
    awsc_acp = next(r for r in v7 if r.algo == "awsc" and r.reward == "acp")
    dsrl_sim = next(r for r in v7 if r.algo == "dsrl" and r.reward == "sim")
    dsrl_acp = next(r for r in v7 if r.algo == "dsrl" and r.reward == "acp")
    pld_sim = next(r for r in v7 if r.algo == "pld" and r.reward == "sim")
    pld_acp = next(r for r in v7 if r.algo == "pld" and r.reward == "acp")
    mismatch_rate = aux["v3_mismatch_rate"] / 100.0

    text = f"""# ACP retention diagnosis summary

## Key observations

1. **The dominant failure mode for PLD/DSRL is retention failure, not pure progress failure.**
   - In v7 controlled comparison, AWSC retains a high SAE/SO ratio under both rewards.
   - DSRL and PLD show very large SO-SAE gaps under both `sim` and `acp`.

2. **This is not ACP-only.**
   - AWSC sim retention: {awsc_sim.retention_ratio:.3f}
   - AWSC acp retention: {awsc_acp.retention_ratio:.3f}
   - DSRL sim retention: {dsrl_sim.retention_ratio:.3f}
   - DSRL acp retention: {dsrl_acp.retention_ratio:.3f}
   - PLD sim retention: {pld_sim.retention_ratio:.3f}
   - PLD acp retention: {pld_acp.retention_ratio:.3f}

3. **ACP still does not provide a strong enough hold-sensitive signal.**
   - v3 mismatch rate (`SO=True, SAE=False`) is {mismatch_rate:.1%} over 1250 trajectories.
   - v3_sae improves value prediction accuracy over v3_so, but downstream retention remains poor for PLD/DSRL in v5/v6/v7.
   - This supports: label semantics matter, but algorithmic retention capacity is still the deeper bottleneck.

## Generated files
- `fig_v7_retention_matrix.png`
- `fig_v7_four_quadrants.png`
- `fig_cross_version_best_metrics.png`
- `fig_v3_mismatch_bridge.png`
- `retention_summary.json`
"""
    (OUT_DIR / "diagnosis_report.md").write_text(text)


def main() -> None:
    rows, aux = build_rows()
    save_summary(rows, aux)
    plot_v7_retention_matrix(rows)
    plot_v7_four_quadrants(rows)
    plot_cross_version_best(rows)
    plot_v3_mismatch_bridge(aux)
    write_report(rows, aux)
    print(f"[OK] Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
