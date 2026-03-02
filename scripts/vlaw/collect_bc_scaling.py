#!/usr/bin/env python3
"""T-BC-SCALING: Collect results and plot Demo Scaling Curve.

Parses training logs from run_bc_scaling.sh and produces:
1. A summary table (printed + saved as JSON)
2. A scaling curve plot (success_rate vs num_demos)

Usage:
    python scripts/vlaw/collect_bc_scaling.py
"""

import os
import re
import json
import glob
from collections import defaultdict

LOG_DIR = "/home/wjz/rl-vla/logs/vlaw/bc_scaling"
RUNS_DIR = "/home/wjz/rl-vla/runs"
RESULTS_DIR = "/home/wjz/rl-vla/results/vlaw/bc_scaling"
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_log(log_file: str) -> dict:
    """Parse training log to extract eval metrics at each checkpoint."""
    metrics = {
        "eval_steps": [],       # step numbers where eval was done
        "success_once": [],     # success_once at each eval
        "success_at_end": [],   # success_at_end at each eval
        "losses": [],           # loss values
        "best_success_once": 0.0,
        "best_success_at_end": 0.0,
        "completed": False,
    }
    
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            # Check for completion
            if "Training completed successfully" in line:
                metrics["completed"] = True
            
            # Look for eval metrics in tqdm output or print statements
            # The train script logs via tensorboard/wandb, not directly to stdout
            # But we can check for the completion message and wandb run name
    
    return metrics


def scan_wandb_runs() -> dict:
    """Scan wandb offline runs for bc_scaling experiments."""
    results = {}
    wandb_dir = "/home/wjz/rl-vla/wandb"
    
    for run_dir in glob.glob(f"{wandb_dir}/*/"):
        config_path = os.path.join(run_dir, "files", "config.yaml")
        summary_path = os.path.join(run_dir, "files", "wandb-summary.json")
        
        if not os.path.exists(summary_path):
            continue
        
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            
            # Check if this is a bc_scaling run
            exp_name = summary.get("_wandb", {}).get("config", {}).get("exp_name", "")
            if "bc_scaling" not in exp_name:
                continue
            
            num_demos = summary.get("_wandb", {}).get("config", {}).get("num_demos", 0)
            success_once = summary.get("eval/success_once", 0.0)
            success_at_end = summary.get("eval/success_at_end", 0.0)
            best_success_once = summary.get("final/best_success_once", 0.0)
            best_success_at_end = summary.get("final/best_success_at_end", 0.0)
            
            results[num_demos] = {
                "success_once": max(success_once, best_success_once),
                "success_at_end": max(success_at_end, best_success_at_end),
                "run_dir": run_dir,
            }
        except (json.JSONDecodeError, KeyError):
            continue
    
    return results


def scan_runs_dir() -> dict:
    """Scan runs/ directory for bc_scaling run folders."""
    results = {}
    
    for run_dir in sorted(glob.glob(f"{RUNS_DIR}/bc_scaling_*")):
        basename = os.path.basename(run_dir)
        # Extract num_demos from folder name like "bc_scaling_25demos_5000steps__1234"
        match = re.match(r"bc_scaling_(\d+)demos_", basename)
        if not match:
            continue
        
        num_demos = int(match.group(1))
        
        # Check for best checkpoint
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        has_best_ckpt = os.path.exists(os.path.join(ckpt_dir, "best_eval_success_once.pt"))
        
        # Look for tensorboard events
        event_files = glob.glob(os.path.join(run_dir, "events.out.*"))
        
        results[num_demos] = {
            "run_dir": run_dir,
            "has_best_ckpt": has_best_ckpt,
            "has_events": len(event_files) > 0,
        }
    
    return results


def main():
    print("=" * 60)
    print("T-BC-SCALING: Results Collection")
    print("=" * 60)
    
    # Scan logs
    log_results = {}
    for log_file in sorted(glob.glob(f"{LOG_DIR}/bc_scaling_*demos_*.log")):
        basename = os.path.basename(log_file).replace(".log", "")
        match = re.match(r"bc_scaling_(\d+)demos_", basename)
        if match:
            num_demos = int(match.group(1))
            log_results[num_demos] = parse_log(log_file)
            log_results[num_demos]["log_file"] = log_file
    
    # Scan runs/ directory
    run_results = scan_runs_dir()
    
    # Merge results
    all_demos = sorted(set(list(log_results.keys()) + list(run_results.keys())))
    
    print(f"\n{'Demos':>6} | {'Completed':>9} | {'Best Ckpt':>9} | {'Run Dir'}")
    print("-" * 80)
    
    summary = {}
    for nd in all_demos:
        completed = log_results.get(nd, {}).get("completed", False)
        has_best = run_results.get(nd, {}).get("has_best_ckpt", False)
        run_dir = run_results.get(nd, {}).get("run_dir", "N/A")
        
        print(f"{nd:>6} | {'Yes' if completed else 'No':>9} | {'Yes' if has_best else 'No':>9} | {run_dir}")
        
        summary[nd] = {
            "completed": completed,
            "has_best_ckpt": has_best,
            "run_dir": run_dir,
        }
    
    # Save
    out_file = f"{RESULTS_DIR}/bc_scaling_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_file}")
    
    # Hint for plotting
    print("\nTo plot the scaling curve, run after all experiments complete:")
    print("  python scripts/vlaw/plot_bc_scaling.py")


if __name__ == "__main__":
    main()
