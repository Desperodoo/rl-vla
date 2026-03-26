#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path("/home/wjz/rl-vla")
OUT_DIR = PROJECT_ROOT / "docs" / "vlaw" / "figures" / "acp_hold_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})


@dataclass
class RunSummary:
    name: str
    algo: str
    reward: str
    run_dir: str
    best_so: float
    final_so: float
    best_sae: float
    final_sae: float
    retention: float
    gap: float
    q_final: float | None
    critic_loss_final: float | None
    temp_final: float | None
    grasp_rate_final: float | None = None
    acp_base_final: float | None = None
    acp_bonus_final: float | None = None


CORRECTED_V7_RUNS = {
    "dsrl_acp": PROJECT_ROOT / "runs/dsrl_v7_qclip0_acp_mirror_s42__1774237641",
    "dsrl_sim": PROJECT_ROOT / "runs/dsrl_v7_reg_qclip0_sim_s42__1774197495",
    "pld_acp": PROJECT_ROOT / "runs/pld_v7_qclip0_acp_mirror_s42__1774237641",
    "pld_sim": PROJECT_ROOT / "runs/pld_v7_reg_qclip0_sim_s42__1774197493",
}

V3_COMPARISON_JSON = PROJECT_ROOT / "docs/vlaw/figures/v3_comparison/v3_comparison_summary.json"
EPISODE_FIGS = {
    "v3_so_success": PROJECT_ROOT / "docs/vlaw/figures/v3_so/episodes/traj_0007_success_keyframes.png",
    "v3_so_fail": PROJECT_ROOT / "docs/vlaw/figures/v3_so/episodes/traj_0232_fail_keyframes.png",
    "v3_sae_success": PROJECT_ROOT / "docs/vlaw/figures/v3_sae/episodes/traj_0007_success_keyframes.png",
    "v3_sae_fail": PROJECT_ROOT / "docs/vlaw/figures/v3_sae/episodes/traj_0232_fail_keyframes.png",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _ea(run_dir: Path) -> EventAccumulator:
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    return ea


def _scalar(ea: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return np.array([]), np.array([])
    vals = ea.Scalars(tag)
    return np.array([v.step for v in vals]), np.array([v.value for v in vals], dtype=float)


def _last(arr: np.ndarray) -> float | None:
    return float(arr[-1]) if len(arr) else None


def _best(arr: np.ndarray) -> float | None:
    return float(np.max(arr)) if len(arr) else None


def load_corrected_v7() -> tuple[list[RunSummary], dict[str, dict[str, np.ndarray]]]:
    summaries: list[RunSummary] = []
    raw: dict[str, dict[str, np.ndarray]] = {}

    for key, run_dir in CORRECTED_V7_RUNS.items():
        algo, reward = key.split("_")
        ea = _ea(run_dir)

        step_so, so = _scalar(ea, "eval/success_once")
        step_sae, sae = _scalar(ea, "eval/success_at_end")
        _, q = _scalar(ea, "train/critic/q_mean")
        _, critic_loss = _scalar(ea, "train/critic/critic_loss")
        _, temp = _scalar(ea, "train/temp/temperature")
        _, entropy = _scalar(ea, "train/actor/actor_entropy")
        _, grasp_rate = _scalar(ea, "train/reward/is_grasping_rate")
        _, acp_base = _scalar(ea, "train/reward/acp_base_mean")
        _, acp_bonus = _scalar(ea, "train/reward/acp_grasp_bonus_mean")
        _, acp_total = _scalar(ea, "train/reward/acp_total_mean")

        best_so = _best(so) or 0.0
        best_sae = _best(sae) or 0.0
        retention = best_sae / best_so if best_so > 1e-8 else 0.0

        summaries.append(
            RunSummary(
                name=key,
                algo=algo,
                reward=reward,
                run_dir=str(run_dir),
                best_so=best_so,
                final_so=_last(so) or 0.0,
                best_sae=best_sae,
                final_sae=_last(sae) or 0.0,
                retention=retention,
                gap=best_so - best_sae,
                q_final=_last(q),
                critic_loss_final=_last(critic_loss),
                temp_final=_last(temp),
                grasp_rate_final=_last(grasp_rate),
                acp_base_final=_last(acp_base),
                acp_bonus_final=_last(acp_bonus),
            )
        )

        raw[key] = {
            "step_so": step_so,
            "so": so,
            "step_sae": step_sae,
            "sae": sae,
            "q": q,
            "critic_loss": critic_loss,
            "temp": temp,
            "entropy": entropy,
            "grasp_rate": grasp_rate,
            "acp_base": acp_base,
            "acp_bonus": acp_bonus,
            "acp_total": acp_total,
        }

    return summaries, raw


def save_summary(summaries: list[RunSummary], v3_cmp: dict) -> None:
    payload = {
        "corrected_v7": [asdict(s) for s in summaries],
        "v3_mismatch": {
            "rate": v3_cmp["mismatch_rate"],
            "total_mismatch": v3_cmp["total_mismatch"],
            "total_trajs": v3_cmp["total_trajs"],
            "scan_results": v3_cmp["scan_results"],
            "inference": v3_cmp["inference"],
        },
    }
    (OUT_DIR / "corrected_v7_summary.json").write_text(json.dumps(payload, indent=2))


def plot_corrected_retention_matrix(summaries: list[RunSummary]) -> None:
    algos = ["dsrl", "pld"]
    rewards = ["sim", "acp"]
    ret = np.zeros((2, 2))
    so = np.zeros((2, 2))
    sae = np.zeros((2, 2))
    gap = np.zeros((2, 2))

    for i, algo in enumerate(algos):
        for j, reward in enumerate(rewards):
            s = next(x for x in summaries if x.algo == algo and x.reward == reward)
            ret[i, j] = s.retention
            so[i, j] = s.best_so
            sae[i, j] = s.best_sae
            gap[i, j] = s.gap

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    im0 = axes[0].imshow(ret, cmap="YlGn", vmin=0, vmax=1)
    axes[0].set_xticks(range(2), rewards)
    axes[0].set_yticks(range(2), [a.upper() for a in algos])
    axes[0].set_title("Corrected v7 retention ratio")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{ret[i,j]:.3f}\nSO={so[i,j]:.2f}\nSAE={sae[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(gap, cmap="OrRd", vmin=0, vmax=1)
    axes[1].set_xticks(range(2), rewards)
    axes[1].set_yticks(range(2), [a.upper() for a in algos])
    axes[1].set_title("Corrected v7 SO-SAE gap")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{gap[i,j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Corrected qclip0 comparison: ACP does not rescue hold for DSRL/PLD", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_corrected_v7_retention_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_training_dynamics(raw: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    algo_order = ["dsrl", "pld"]
    colors = {"sim": "#1E88E5", "acp": "#E53935"}

    for col, algo in enumerate(algo_order):
        for reward in ["sim", "acp"]:
            key = f"{algo}_{reward}"
            d = raw[key]
            label = reward.upper()
            step_eval = d["step_so"] / 1000.0 if len(d["step_so"]) else np.arange(len(d["so"]))
            step_train = np.linspace(0, step_eval[-1] if len(step_eval) else 71, len(d["q"])) if len(d["q"]) else np.array([])

            axes[0, col].plot(step_eval, d["so"], color=colors[reward], linestyle="--", label=f"{label} SO")
            axes[0, col].plot(step_eval, d["sae"], color=colors[reward], linestyle="-", label=f"{label} SAE")
            if len(step_train):
                axes[1, col].plot(step_train, d["q"], color=colors[reward], label=f"{label} q_mean")
                axes[2, col].plot(step_train, d["temp"], color=colors[reward], label=f"{label} temp")

        axes[0, col].set_title(f"{algo.upper()}: success_once vs success_at_end")
        axes[1, col].set_title(f"{algo.upper()}: critic q_mean")
        axes[2, col].set_title(f"{algo.upper()}: temperature")
        for row in range(3):
            axes[row, col].grid(True, alpha=0.25)
            axes[row, col].legend(loc="best")
            axes[row, col].set_xlabel("Steps (K)")
        axes[0, col].set_ylim(0, 1.02)

    fig.suptitle("Corrected v7 dynamics: progress can stay high while retention collapses", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_corrected_v7_training_dynamics.png", bbox_inches="tight")
    plt.close(fig)


def plot_acp_reward_components(raw: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, algo in enumerate(["dsrl", "pld"]):
        d = raw[f"{algo}_acp"]
        n = max(len(d["grasp_rate"]), len(d["acp_base"]), len(d["acp_bonus"]), 1)
        steps = np.linspace(0, 71, n)

        if len(d["grasp_rate"]):
            axes[row, 0].plot(np.linspace(0, 71, len(d["grasp_rate"])), d["grasp_rate"], color="#43A047", label="is_grasping_rate")
        if len(d["step_sae"]):
            axes[row, 0].plot(d["step_sae"] / 1000.0, d["sae"], color="#E53935", linestyle="--", label="eval SAE")
        axes[row, 0].set_title(f"{algo.upper()}: grasping vs SAE")
        axes[row, 0].set_ylim(0, 1.02)
        axes[row, 0].grid(True, alpha=0.25)
        axes[row, 0].legend(loc="best")

        if len(d["acp_base"]):
            axes[row, 1].plot(np.linspace(0, 71, len(d["acp_base"])), d["acp_base"], color="#1E88E5", label="acp_base_mean")
        if len(d["acp_bonus"]):
            axes[row, 1].plot(np.linspace(0, 71, len(d["acp_bonus"])), d["acp_bonus"], color="#FB8C00", label="acp_grasp_bonus_mean")
        if len(d["acp_total"]):
            axes[row, 1].plot(np.linspace(0, 71, len(d["acp_total"])), d["acp_total"], color="#8E24AA", label="acp_total_mean")
        axes[row, 1].set_title(f"{algo.upper()}: ACP reward components")
        axes[row, 1].grid(True, alpha=0.25)
        axes[row, 1].legend(loc="best")

        for col in range(2):
            axes[row, col].set_xlabel("Steps (K)")

    fig.suptitle("ACP reward emphasizes grasp/progress, but not stable hold-to-end", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_acp_reward_components.png", bbox_inches="tight")
    plt.close(fig)


def plot_v3_mismatch_evidence(v3_cmp: dict) -> None:
    scan = v3_cmp["scan_results"]
    datasets = list(scan.keys())
    mismatch = np.array([scan[k]["mismatch"] / scan[k]["total"] for k in datasets], dtype=float)

    inf = v3_cmp["inference"]
    mae = [inf["v3_so"]["mae"], inf["v3_sae"]["mae"]]
    adv_std = [inf["v3_so"]["advantage_std"], inf["v3_sae"]["advantage_std"]]
    pred_mean = [inf["v3_so"]["pred_mean"], inf["v3_sae"]["pred_mean"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(datasets))
    axes[0].bar(x, mismatch, color="#FFB74D")
    axes[0].set_xticks(x, [d.replace(": ", "\n") for d in datasets])
    axes[0].set_ylim(0, 0.35)
    axes[0].set_title("SO=True, SAE=False mismatch rate")
    for i, val in enumerate(mismatch):
        axes[0].text(i, val + 0.01, f"{val:.1%}", ha="center", fontsize=8)

    x2 = np.arange(2)
    axes[1].bar(x2, mae, color=["#43A047", "#E53935"])
    axes[1].set_xticks(x2, ["v3_so", "v3_sae"])
    axes[1].set_title("Value prediction MAE")
    for i, val in enumerate(mae):
        axes[1].text(i, val + 0.002, f"{val:.3f}", ha="center", fontsize=8)

    axes[2].bar(x2 - 0.15, adv_std, width=0.3, color="#1E88E5", label="adv std")
    axes[2].bar(x2 + 0.15, pred_mean, width=0.3, color="#8E24AA", label="pred mean")
    axes[2].set_xticks(x2, ["v3_so", "v3_sae"])
    axes[2].set_title("Sharper value / advantage under v3_sae")
    axes[2].legend(loc="best")

    fig.suptitle("Label semantics matter: mismatch exists, but deeper retention gap remains")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_v3_mismatch_evidence.png", bbox_inches="tight")
    plt.close(fig)


def write_report(summaries: list[RunSummary], v3_cmp: dict) -> None:
    dsrl_sim = next(s for s in summaries if s.name == "dsrl_sim")
    dsrl_acp = next(s for s in summaries if s.name == "dsrl_acp")
    pld_sim = next(s for s in summaries if s.name == "pld_sim")
    pld_acp = next(s for s in summaries if s.name == "pld_acp")

    text = f"""# ACP hold diagnostics

## Corrected v7 comparison

Using the corrected qclip0 comparison runs:
- DSRL sim: `{Path(dsrl_sim.run_dir).name}`
- DSRL acp: `{Path(dsrl_acp.run_dir).name}`
- PLD sim: `{Path(pld_sim.run_dir).name}`
- PLD acp: `{Path(pld_acp.run_dir).name}`

### Headline numbers
- DSRL sim: SO={dsrl_sim.best_so:.2f}, SAE={dsrl_sim.best_sae:.2f}, retention={dsrl_sim.retention:.3f}
- DSRL acp: SO={dsrl_acp.best_so:.2f}, SAE={dsrl_acp.best_sae:.2f}, retention={dsrl_acp.retention:.3f}
- PLD sim: SO={pld_sim.best_so:.2f}, SAE={pld_sim.best_sae:.2f}, retention={pld_sim.retention:.3f}
- PLD acp: SO={pld_acp.best_so:.2f}, SAE={pld_acp.best_sae:.2f}, retention={pld_acp.retention:.3f}

## What the deeper analysis supports

1. **The previous v7 sim baseline used in the first pass should be replaced by these corrected runs.**
   Under the corrected qclip0-controlled comparison, sim is much stronger than the earlier `diag_sim` runs.

2. **ACP is not simply “worse than sim” in a scalar sense; it changes what gets reinforced.**
   The ACP runs expose grasp/progress-related reward components (`is_grasping_rate`, `acp_grasp_bonus_mean`, `acp_base_mean`), but those signals still do not translate into strong `success_at_end`.

3. **This gets closer to the root cause than SO/SAE alone:**
   - if grasping-related signals rise but SAE stays near zero,
   - then the problem is not “the agent never reaches/grips”,
   - but rather **the reward is not sufficiently discriminating stable hold vs imminent drop**.

4. **v3 mismatch evidence remains important but not sufficient.**
   `v3_sae` improves value semantics on mismatch trajectories, yet PLD/DSRL still fail to retain success at the policy level.

## Generated files
- `fig_corrected_v7_retention_matrix.png`
- `fig_corrected_v7_training_dynamics.png`
- `fig_acp_reward_components.png`
- `fig_v3_mismatch_evidence.png`
- `corrected_v7_summary.json`
"""
    (OUT_DIR / "hold_diagnostics_report.md").write_text(text)


def main() -> None:
    v3_cmp = _load_json(V3_COMPARISON_JSON)
    summaries, raw = load_corrected_v7()
    save_summary(summaries, v3_cmp)
    plot_corrected_retention_matrix(summaries)
    plot_training_dynamics(raw)
    plot_acp_reward_components(raw)
    plot_v3_mismatch_evidence(v3_cmp)
    write_report(summaries, v3_cmp)
    print(f"[OK] Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
