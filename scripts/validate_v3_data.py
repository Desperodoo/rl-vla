"""Validate ACP v3 data collected with ignore_terminations=True.

Checks:
1. Episode length distribution (should be uniform ~max_steps/frame_skip)
2. success_once vs success_at_end mismatch rate (should be significant, not 0.16%)
3. Per-dataset statistics (total trajs, success rates, frame counts)
4. success_at_end binary label quality (has both True and False)
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Data directories (v3 = ignore_terminations)
# ---------------------------------------------------------------------------
DATA_DIRS = {
    "B: PLD_pretrained": _ROOT / "data/vlaw/rollouts/v3_pld_pretrained",
    "C: PLD_teleop": _ROOT / "data/vlaw/rollouts/v3_pld_teleop",
    "D: PLD_rl_prior": _ROOT / "data/vlaw/rollouts/v3_pld_rl_prior",
    "E: PLD_random": _ROOT / "data/vlaw/rollouts/v3_pld_random",
}

# Also scan old data for comparison
OLD_DIRS = {
    "B: Pretrained": _ROOT / "data/vlaw/rollouts/pretrained_policy",
    "C: Teleop": _ROOT / "data/vlaw/rollouts/teleop_sim",
    "D: RL_Prior": _ROOT / "data/vlaw/rollouts/rl_prior",
    "E: Random": _ROOT / "data/vlaw/rollouts/random",
}

# Include demo data (Type A, unchanged)
DEMO_DIR = _ROOT / "data/vlaw/rollouts/mixed/LiftPegUpright-v1"


def scan_dataset(label: str, directory: Path) -> dict:
    """Scan all HDF5 files in a directory, return statistics."""
    h5_files = sorted(directory.glob("*.h5"))
    if not h5_files:
        return {"label": label, "files": 0, "total": 0, "error": "No HDF5 files found"}

    total_trajs = 0
    success_once_count = 0
    success_at_end_count = 0
    mismatch_count = 0
    lengths = []
    mismatch_details = []

    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
            for tk in traj_keys:
                grp = f[tk]
                total_trajs += 1

                if "env_success" in grp:
                    es = np.asarray(grp["env_success"], dtype=bool)
                    T = len(es)
                    lengths.append(T)

                    s_once = bool(np.any(es))
                    s_end = bool(es[-1])

                    if s_once:
                        success_once_count += 1
                    if s_end:
                        success_at_end_count += 1
                    if s_once != s_end:
                        mismatch_count += 1
                        mismatch_details.append({
                            "file": h5_path.name,
                            "traj": tk,
                            "T": T,
                            "success_once": s_once,
                            "success_at_end": s_end,
                            # Find first and last success frames
                            "first_success_frame": int(np.argmax(es)) if s_once else -1,
                            "last_success_frame": int(T - 1 - np.argmax(es[::-1])) if s_once else -1,
                        })
                else:
                    lengths.append(0)

    lengths_arr = np.array(lengths)
    mismatch_rate = mismatch_count / total_trajs * 100 if total_trajs > 0 else 0

    return {
        "label": label,
        "files": len(h5_files),
        "total": total_trajs,
        "frames": int(lengths_arr.sum()),
        "success_once": success_once_count,
        "success_at_end": success_at_end_count,
        "mismatch": mismatch_count,
        "mismatch_rate": mismatch_rate,
        "length_mean": float(lengths_arr.mean()) if len(lengths_arr) > 0 else 0,
        "length_std": float(lengths_arr.std()) if len(lengths_arr) > 0 else 0,
        "length_min": int(lengths_arr.min()) if len(lengths_arr) > 0 else 0,
        "length_max": int(lengths_arr.max()) if len(lengths_arr) > 0 else 0,
        "mismatch_details": mismatch_details[:10],  # first 10 examples
    }


def print_table(results: list[dict], title: str) -> None:
    """Pretty-print results as a table."""
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    print(f"{'Dataset':<22} {'Files':>5} {'Trajs':>6} {'Frames':>7} "
          f"{'S_once':>6} {'S_end':>6} {'Mismatch':>8} {'Rate%':>6} "
          f"{'Len_mean':>8} {'Len_std':>7} {'Len_min':>7} {'Len_max':>7}")
    print("-" * 100)

    for r in results:
        if "error" in r:
            print(f"{r['label']:<22} {'—':>5} {r['error']}")
            continue
        print(f"{r['label']:<22} {r['files']:>5} {r['total']:>6} {r['frames']:>7} "
              f"{r['success_once']:>6} {r['success_at_end']:>6} {r['mismatch']:>8} "
              f"{r['mismatch_rate']:>5.1f}% "
              f"{r['length_mean']:>8.1f} {r['length_std']:>7.1f} "
              f"{r['length_min']:>7} {r['length_max']:>7}")


def print_mismatch_examples(results: list[dict]) -> None:
    """Print mismatch trajectory examples."""
    has_mismatch = False
    for r in results:
        if "error" in r:
            continue
        if r.get("mismatch_details"):
            if not has_mismatch:
                print(f"\n{'='*80}")
                print("  Mismatch Trajectory Examples (success_once ≠ success_at_end)")
                print(f"{'='*80}")
                has_mismatch = True
            print(f"\n  [{r['label']}]")
            for d in r["mismatch_details"][:5]:
                print(f"    {d['traj']} (T={d['T']}): "
                      f"success_once={d['success_once']}, success_at_end={d['success_at_end']}, "
                      f"first_success={d['first_success_frame']}, last_success={d['last_success_frame']}")

    if not has_mismatch:
        print("\n  ⚠️ No mismatch trajectories found!")


def main():
    print("=" * 80)
    print("  ACP v3 Data Validation (ignore_terminations=True)")
    print("=" * 80)

    # Scan v3 data
    v3_results = []
    for label, d in DATA_DIRS.items():
        if d.exists():
            v3_results.append(scan_dataset(label, d))
        else:
            v3_results.append({"label": label, "files": 0, "total": 0, "error": f"Dir not found: {d}"})

    # Scan demo data
    if DEMO_DIR.exists():
        demo_result = scan_dataset("A: Demo", DEMO_DIR)
        v3_results.insert(0, demo_result)

    print_table(v3_results, "V3 Data (ignore_terminations=True)")
    print_mismatch_examples(v3_results)

    # Scan old data for comparison
    old_results = []
    for label, d in OLD_DIRS.items():
        if d.exists():
            old_results.append(scan_dataset(label, d))

    if old_results:
        print_table(old_results, "V2 Data (original, early termination)")

    # Summary comparison
    v3_total = sum(r.get("total", 0) for r in v3_results if "error" not in r)
    v3_mismatch = sum(r.get("mismatch", 0) for r in v3_results if "error" not in r)
    v3_s_once = sum(r.get("success_once", 0) for r in v3_results if "error" not in r)
    v3_s_end = sum(r.get("success_at_end", 0) for r in v3_results if "error" not in r)

    old_total = sum(r.get("total", 0) for r in old_results if "error" not in r)
    old_mismatch = sum(r.get("mismatch", 0) for r in old_results if "error" not in r)

    print(f"\n{'='*80}")
    print("  Summary Comparison")
    print(f"{'='*80}")
    print(f"  V3 (ignore_terminations): {v3_total} trajs, "
          f"success_once={v3_s_once} ({v3_s_once/max(v3_total,1)*100:.1f}%), "
          f"success_at_end={v3_s_end} ({v3_s_end/max(v3_total,1)*100:.1f}%), "
          f"mismatch={v3_mismatch} ({v3_mismatch/max(v3_total,1)*100:.1f}%)")
    if old_total > 0:
        print(f"  V2 (early termination):  {old_total} trajs, "
              f"mismatch={old_mismatch} ({old_mismatch/max(old_total,1)*100:.1f}%)")
        print(f"\n  Mismatch improvement: {old_mismatch/max(old_total,1)*100:.1f}% → "
              f"{v3_mismatch/max(v3_total,1)*100:.1f}%")

    # Quality checks
    print(f"\n{'='*80}")
    print("  Quality Checks")
    print(f"{'='*80}")

    checks = []

    # Check 1: Has both success and failure
    has_success = v3_s_end > 0
    has_failure = (v3_total - v3_s_end) > 0
    c1 = "✅" if (has_success and has_failure) else "❌"
    checks.append(f"  {c1} Has both success_at_end=True and False: "
                  f"True={v3_s_end}, False={v3_total - v3_s_end}")

    # Check 2: Meaningful mismatch rate
    c2 = "✅" if v3_mismatch > 10 else "⚠️" if v3_mismatch > 0 else "❌"
    checks.append(f"  {c2} Meaningful mismatch (success_once ≠ success_at_end): "
                  f"{v3_mismatch} trajs ({v3_mismatch/max(v3_total,1)*100:.1f}%)")

    # Check 3: Episode lengths uniform (not early terminated)
    v3_no_demo = [r for r in v3_results if "error" not in r and "Demo" not in r["label"]]
    if v3_no_demo:
        length_stds = [r["length_std"] for r in v3_no_demo]
        all_uniform = all(s < 3.0 for s in length_stds)
        c3 = "✅" if all_uniform else "⚠️"
        checks.append(f"  {c3} Episode lengths uniform (std < 3.0): "
                      f"stds = {[f'{s:.1f}' for s in length_stds]}")

    # Check 4: Enough data
    c4 = "✅" if v3_total >= 500 else "⚠️"
    checks.append(f"  {c4} Sufficient data volume: {v3_total} trajs")

    for c in checks:
        print(c)

    print()


if __name__ == "__main__":
    main()
