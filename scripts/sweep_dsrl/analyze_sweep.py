#!/usr/bin/env python3
"""
DSRL Sweep Experiment Analysis Tool
====================================

Analysis tool for DSRL-SAC (Diffusion-based SAC in Residual/Noise space)
hyperparameter sweep experiments:
1. Data extraction from training logs and TensorBoard events
2. Parameter sensitivity analysis (controlled variable analysis)
3. Training curve analysis (convergence, stability)
4. Visualization (bar charts, training curves, heatmaps)
5. Markdown report generation

Usage:
    python analyze_sweep.py --sweep-dir runs/dsrl_sweep
    python analyze_sweep.py --sweep-dir runs/dsrl_sweep --output-dir analysis_results
    python analyze_sweep.py --sweep-dir runs/dsrl_sweep --config-version v2
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Try to import tensorboard
try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed. Training curve analysis will be limited.")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExperimentResult:
    """Single experiment result"""
    algorithm: str
    config_name: str
    params: Dict[str, Any]
    success_once: Optional[float] = None
    success_at_end: Optional[float] = None
    status: str = "unknown"  # success, failed, not_started
    exp_dir: Optional[str] = None
    training_curves: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SweepAnalysis:
    """Analysis results for a sweep"""
    algorithm: str
    experiments: List[ExperimentResult]
    param_sensitivity: Dict[str, float] = field(default_factory=dict)
    best_config: Optional[str] = None
    best_score: float = 0.0
    param_values: Dict[str, List[Any]] = field(default_factory=dict)


# =============================================================================
# Configuration — DSRL-SAC Default Parameters
# =============================================================================

DEFAULT_PARAMS = {
    "dsrl_sac": {
        "action_magnitude": 2.0,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "utd_ratio": 40,
        "batch_size": 256,
        "init_temperature": 1.0,
        "target_entropy": 0.0,
        "log_std_init": -3.0,
        "num_seed_steps": 1000,
        "num_layers": 3,
        "layer_size": 512,
        "num_qs": 2,
        "use_layer_norm": True,
        "max_grad_norm": 10.0,
        "buffer_size": 1_000_000,
    },
}

# Mapping from config name patterns to parameter changes
CONFIG_PARAM_PATTERNS = {
    # Action magnitude (noise scale)
    r"am_([\d.]+)": lambda m: {"action_magnitude": float(m.group(1))},

    # UTD ratio
    r"utd_(\d+)": lambda m: {"utd_ratio": int(m.group(1))},

    # Learning rate
    r"lr_([\d.]+e?-?\d*)": lambda m: {"learning_rate": float(m.group(1))},

    # Discount factor
    r"gamma_([\d.]+)": lambda m: {"gamma": float(m.group(1))},

    # Tau
    r"tau_([\d.]+)": lambda m: {"tau": float(m.group(1))},

    # Temperature
    r"init_temp_([\d.]+)": lambda m: {"init_temperature": float(m.group(1))},

    # Target entropy
    r"target_ent_([-\d.]+)": lambda m: {"target_entropy": float(m.group(1))},

    # Log std init
    r"log_std_([-\d.]+)": lambda m: {"log_std_init": float(m.group(1))},

    # Seed steps
    r"seed_steps_(\d+)": lambda m: {"num_seed_steps": int(m.group(1))},

    # Batch size
    r"batch_(\d+)": lambda m: {"batch_size": int(m.group(1))},

    # Network architecture
    r"num_qs_(\d+)": lambda m: {"num_qs": int(m.group(1))},
    r"arch_small": lambda m: {"num_layers": 2, "layer_size": 512},
    r"arch_medium": lambda m: {"num_layers": 2, "layer_size": 1024},
    r"arch_deep": lambda m: {"num_layers": 4, "layer_size": 2048},
    r"arch_wide": lambda m: {"num_layers": 3, "layer_size": 4096},
}


# =============================================================================
# Data Extraction
# =============================================================================

class DataExtractor:
    """Extract data from DSRL sweep experiments"""

    def __init__(self, sweep_dir: str, config_version: str = "v1"):
        self.sweep_dir = Path(sweep_dir)
        self.config_version = config_version

    def find_experiment_dirs(self, algorithm: str) -> List[Path]:
        """Find all experiment directories for an algorithm"""
        algo_dir = self.sweep_dir / algorithm
        if not algo_dir.exists():
            return []

        exp_dirs = []
        for item in algo_dir.iterdir():
            if not item.is_dir():
                continue
            if "__" in item.name:
                exp_dirs.append(item)
            else:
                timestamped = list(algo_dir.glob(f"{item.name}__*"))
                if timestamped:
                    exp_dirs.extend(timestamped)
                elif (item / "train.log").exists() or (item / "checkpoints").exists():
                    exp_dirs.append(item)

        return sorted(set(exp_dirs))

    def get_config_name(self, exp_dir: Path) -> str:
        """Extract config name from experiment directory"""
        name = exp_dir.name
        if "__" in name:
            name = name.rsplit("__", 1)[0]
        return name

    def parse_config_params(self, algorithm: str, config_name: str) -> Dict[str, Any]:
        """Parse hyperparameters from config name"""
        params = DEFAULT_PARAMS.get(algorithm, {}).copy()

        for pattern, extractor in CONFIG_PARAM_PATTERNS.items():
            match = re.search(pattern, config_name)
            if match:
                try:
                    extracted = extractor(match)
                    params.update(extracted)
                except Exception:
                    pass

        return params

    def parse_metrics_from_log(self, log_file: Path) -> Dict[str, float]:
        """Parse metrics from train.log file

        train_dsrl.py logs:
          - wandb Run summary: eval/success_once, eval/success_at_end (may truncate others)
          - wandb.log: final/best_success_rate (often hidden behind '+10 ...' in log)
          - Plain text at end: 'Done. Best success rate: XX.XX%'
          - Plain text eval: '  success_once: X.XXXX', '  success_at_end: X.XXXX'
        """
        metrics = {}
        if not log_file.exists():
            return metrics

        with open(log_file, 'r', errors='ignore') as f:
            content = f.read()

        # --- wandb Run summary metrics ---
        wandb_patterns = [
            # Best success rate (may be truncated in wandb summary)
            (r"wandb:\s*final/best_success_rate\s+([\d.]+)", "best_success_once"),
            # Final eval values from wandb summary
            (r"wandb:\s*eval/success_once\s+([\d.]+)", "final_success_once"),
            (r"wandb:\s*eval/success_at_end\s+([\d.]+)", "final_success_at_end"),
            # Training metrics
            (r"wandb:\s*train/critic_loss\s+([\d.]+)", "critic_loss"),
            (r"wandb:\s*train/actor_loss\s+(-?[\d.]+)", "actor_loss"),
            (r"wandb:\s*train/alpha\s+([\d.]+)", "alpha"),
            (r"wandb:\s*train/entropy\s+(-?[\d.]+)", "entropy"),
        ]

        for pattern, metric_name in wandb_patterns:
            matches = re.findall(pattern, content)
            if matches:
                try:
                    value = float(matches[-1])
                    if metric_name.startswith("best_success") or metric_name.startswith("final_success"):
                        if value <= 1.0:
                            metrics[metric_name] = value
                    else:
                        metrics[metric_name] = value
                except ValueError:
                    pass

        # --- Fallback: parse 'Done. Best success rate: XX.XX%' ---
        if "best_success_once" not in metrics:
            match = re.search(r"Done\.\s*Best success rate:\s*([\d.]+)%", content)
            if match:
                try:
                    pct = float(match.group(1))
                    metrics["best_success_once"] = pct / 100.0
                except ValueError:
                    pass

        # --- Fallback: parse plain text eval output ---
        # '  success_once: X.XXXX' (printed by train_dsrl.py in eval blocks)
        if "final_success_once" not in metrics:
            matches = re.findall(r"^\s+success_once:\s+([\d.]+)", content, re.MULTILINE)
            if matches:
                try:
                    metrics["final_success_once"] = float(matches[-1])
                except ValueError:
                    pass

        if "final_success_at_end" not in metrics:
            matches = re.findall(r"^\s+success_at_end:\s+([\d.]+)", content, re.MULTILINE)
            if matches:
                try:
                    metrics["final_success_at_end"] = float(matches[-1])
                except ValueError:
                    pass

        return metrics

    def load_tensorboard_data(self, exp_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
        """Load training curves from TensorBoard event files"""
        curves = {}
        if not HAS_TENSORBOARD:
            return curves

        event_files = list(exp_dir.glob("events.out.tfevents.*"))
        if not event_files:
            return curves

        try:
            ea = event_accumulator.EventAccumulator(str(exp_dir))
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            for tag in tags:
                events = ea.Scalars(tag)
                curves[tag] = [(e.step, e.value) for e in events]
        except Exception as e:
            print(f"Warning: Failed to load TensorBoard data from {exp_dir}: {e}")

        return curves

    def load_config_json(self, exp_dir: Path) -> Dict[str, Any]:
        """Load config.json if available"""
        config_file = exp_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def extract_experiment(self, algorithm: str, exp_dir: Path) -> ExperimentResult:
        """Extract all data from a single experiment"""
        config_name = self.get_config_name(exp_dir)
        params = self.parse_config_params(algorithm, config_name)

        config_json = self.load_config_json(exp_dir)
        if config_json:
            for key in params:
                if key in config_json:
                    params[key] = config_json[key]

        # Check status
        # train_dsrl.py saves best.pt (not best_eval_success_once.pt)
        checkpoint_dir = exp_dir / "checkpoints"
        has_checkpoint = (
            (checkpoint_dir / "best.pt").exists() or
            (checkpoint_dir / "best_eval_success_once.pt").exists() or
            (checkpoint_dir / "final.pt").exists()
        )

        log_file = exp_dir / "train.log"
        if not log_file.exists():
            base_dir = exp_dir.parent / config_name
            if (base_dir / "train.log").exists():
                log_file = base_dir / "train.log"

        metrics = self.parse_metrics_from_log(log_file)

        if has_checkpoint or metrics.get("best_success_once") is not None:
            status = "success"
        elif log_file.exists():
            status = "failed"
        else:
            status = "not_started"

        training_curves = self.load_tensorboard_data(exp_dir)

        # best_success_once = final/best_success_rate (or parsed from 'Done. Best success rate')
        # Fallback chain: best_success_once > final_success_once
        success_once = metrics.get("best_success_once")
        if success_once is None:
            success_once = metrics.get("final_success_once")

        # For success_at_end, use final eval value
        success_at_end = metrics.get("final_success_at_end")

        return ExperimentResult(
            algorithm=algorithm,
            config_name=config_name,
            params=params,
            success_once=success_once,
            success_at_end=success_at_end,
            status=status,
            exp_dir=str(exp_dir),
            training_curves=training_curves,
            final_metrics=metrics,
        )

    def extract_algorithm(self, algorithm: str) -> List[ExperimentResult]:
        """Extract all experiments for an algorithm"""
        exp_dirs = self.find_experiment_dirs(algorithm)
        results = []
        for exp_dir in exp_dirs:
            result = self.extract_experiment(algorithm, exp_dir)
            results.append(result)
        return results


# =============================================================================
# Parameter Sensitivity Analysis
# =============================================================================

class SensitivityAnalyzer:
    """Analyze parameter sensitivity using controlled variable analysis"""

    def __init__(self, experiments: List[ExperimentResult]):
        self.experiments = [e for e in experiments if e.success_once is not None]
        self.df = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        data = []
        for exp in self.experiments:
            row = {
                "config_name": exp.config_name,
                "success_once": exp.success_once,
                "success_at_end": exp.success_at_end,
                **exp.params,
            }
            data.append(row)
        return pd.DataFrame(data)

    def analyze_single_param(self, param: str) -> Dict[str, Any]:
        if param not in self.df.columns:
            return {}

        grouped = self.df.groupby(param)["success_once"]
        stats = grouped.agg(["mean", "std", "min", "max", "count"]).reset_index()

        if len(stats) > 1:
            sensitivity = stats["mean"].max() - stats["mean"].min()
        else:
            sensitivity = 0.0

        best_idx = stats["mean"].idxmax()
        best_value = stats.loc[best_idx, param]
        best_mean = stats.loc[best_idx, "mean"]

        return {
            "param": param,
            "sensitivity": sensitivity,
            "best_value": best_value,
            "best_mean": best_mean,
            "values": stats[param].tolist(),
            "means": stats["mean"].tolist(),
            "stds": stats["std"].fillna(0).tolist(),
            "counts": stats["count"].tolist(),
            "stats_df": stats,
        }

    def analyze_all_params(self) -> Dict[str, Dict[str, Any]]:
        exclude = {"config_name", "success_once", "success_at_end", "config_type", "mode"}
        param_cols = [c for c in self.df.columns if c not in exclude]
        results = {}

        for param in param_cols:
            unique_values = self.df[param].dropna().unique()
            if len(unique_values) <= 1:
                continue

            if isinstance(unique_values[0], (tuple, list)):
                self.df[f"{param}_str"] = self.df[param].apply(str)
                analysis = self.analyze_single_param(f"{param}_str")
                if analysis:
                    analysis["param"] = param
                    results[param] = analysis
            else:
                analysis = self.analyze_single_param(param)
                if analysis:
                    results[param] = analysis

        return results

    def get_sensitivity_ranking(self) -> List[Tuple[str, float]]:
        analyses = self.analyze_all_params()
        ranking = [(p, a["sensitivity"]) for p, a in analyses.items()]
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def get_best_params(self) -> Dict[str, Any]:
        analyses = self.analyze_all_params()
        return {p: a["best_value"] for p, a in analyses.items()}


# =============================================================================
# Training Curve Analysis
# =============================================================================

class TrainingCurveAnalyzer:
    """Analyze training curves for convergence, stability"""

    def __init__(self, experiments: List[ExperimentResult]):
        self.experiments = experiments

    def get_curve(self, exp: ExperimentResult, metric: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if metric not in exp.training_curves:
            return None
        data = exp.training_curves[metric]
        if not data:
            return None
        steps = np.array([d[0] for d in data])
        values = np.array([d[1] for d in data])
        return steps, values

    def analyze_convergence(self, exp: ExperimentResult, metric: str = "train/critic_loss") -> Dict[str, float]:
        curve = self.get_curve(exp, metric)
        if curve is None:
            return {}

        steps, values = curve
        final_value = values[-1] if len(values) > 0 else np.nan

        last_portion = int(len(values) * 0.2)
        stability = np.std(values[-last_portion:]) if last_portion > 1 else np.nan

        if len(values) > 1:
            diffs = np.diff(values)
            monotonicity = np.mean(diffs < 0)
        else:
            monotonicity = np.nan

        return {
            "final_value": final_value,
            "stability": stability,
            "monotonicity": monotonicity,
        }

    def compare_curves(self, metric: str, configs: Optional[List[str]] = None) -> pd.DataFrame:
        data = []
        for exp in self.experiments:
            if configs and exp.config_name not in configs:
                continue
            analysis = self.analyze_convergence(exp, metric)
            if analysis:
                row = {"config_name": exp.config_name, "success_once": exp.success_once, **analysis}
                data.append(row)
        return pd.DataFrame(data)


# =============================================================================
# Visualization
# =============================================================================

class SweepVisualizer:
    """Generate visualizations for DSRL sweep analysis"""

    def __init__(self, output_dir: str = "analysis_results"):
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.base_output_dir

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10

    def set_algorithm_dir(self, algorithm: str):
        self.output_dir = self.base_output_dir / algorithm
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def reset_to_base_dir(self):
        self.output_dir = self.base_output_dir

    def plot_param_sensitivity_bar(self, sensitivity_data: Dict[str, Dict],
                                    algorithm: str, top_n: int = 15) -> str:
        """Plot parameter sensitivity as horizontal bar chart"""
        fig, ax = plt.subplots(figsize=(10, 8))

        sorted_params = sorted(
            sensitivity_data.items(),
            key=lambda x: x[1]["sensitivity"],
            reverse=True
        )[:top_n]

        params = [p[0] for p in sorted_params]
        sensitivities = [p[1]["sensitivity"] for p in sorted_params]

        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(params)))
        bars = ax.barh(params, sensitivities, color=colors)

        ax.set_xlabel("Sensitivity (Δ success_once)")
        ax.set_title(f"Parameter Sensitivity Analysis — {algorithm} (DSRL)")
        ax.invert_yaxis()

        for bar, val in zip(bars, sensitivities):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va='center', fontsize=9)

        plt.tight_layout()
        filepath = self.output_dir / "param_sensitivity.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return str(filepath)

    def plot_param_effect(self, sensitivity_data: Dict[str, Dict],
                          param: str, algorithm: str) -> str:
        """Plot effect of a single parameter on success rate"""
        if param not in sensitivity_data:
            return ""

        data = sensitivity_data[param]
        fig, ax = plt.subplots(figsize=(10, 6))

        values = data["values"]
        means = data["means"]
        stds = data["stds"]

        x_labels = [str(v) for v in values]
        x_pos = np.arange(len(values))

        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=plt.cm.Blues(0.6), edgecolor='navy', alpha=0.8)

        ax.set_xlabel(param)
        ax.set_ylabel("success_once")
        ax.set_title(f"Effect of {param} on Success Rate — {algorithm}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        best_idx = means.index(max(means))
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(1.0)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{mean:.3f}", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        filepath = self.output_dir / f"{param}_effect.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return str(filepath)

    def plot_config_ranking(self, experiments: List[ExperimentResult],
                            algorithm: str, top_n: int = 20) -> str:
        """Plot ranking of all configurations"""
        sorted_exps = sorted(
            [e for e in experiments if e.success_once is not None],
            key=lambda x: x.success_once,
            reverse=True
        )[:top_n]

        if not sorted_exps:
            return ""

        fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_exps) * 0.4)))

        configs = [e.config_name for e in sorted_exps]
        scores = [e.success_once for e in sorted_exps]

        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(configs)))
        bars = ax.barh(configs, scores, color=colors, edgecolor='black', alpha=0.8)

        ax.set_xlabel("success_once")
        ax.set_title(f"Configuration Ranking — {algorithm} (Top {top_n})")
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)

        for bar, score in zip(bars, scores):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{score:.3f}", va='center', fontsize=9)

        plt.tight_layout()
        filepath = self.output_dir / "config_ranking.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return str(filepath)

    def plot_training_curves(self, experiments: List[ExperimentResult],
                             metric: str, algorithm: str,
                             top_n: int = 5, metric_label: Optional[str] = None) -> str:
        """Plot training curves for top N experiments"""
        sorted_exps = sorted(
            [e for e in experiments if e.success_once is not None],
            key=lambda x: x.success_once,
            reverse=True
        )[:top_n]

        has_data = any(metric in exp.training_curves and exp.training_curves[metric]
                       for exp in sorted_exps)
        if not has_data:
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_exps)))

        plotted = False
        for exp, color in zip(sorted_exps, colors):
            if metric not in exp.training_curves:
                continue
            data = exp.training_curves[metric]
            if not data:
                continue
            steps = [d[0] for d in data]
            values = [d[1] for d in data]
            label = f"{exp.config_name} ({exp.success_once:.2f})"
            ax.plot(steps, values, label=label, color=color, linewidth=1.5)
            plotted = True

        if not plotted:
            plt.close()
            return ""

        ax.set_xlabel("Training Steps")
        ax.set_ylabel(metric_label or metric)

        if "loss" in metric.lower():
            filename = f"{metric.replace('/', '_')}_curves.png"
            ax.set_title(f"{metric} Curves — {algorithm} (Top {top_n})")
        elif "success" in metric.lower():
            filename = "eval_success_curves.png"
            ax.set_title(f"Evaluation Success Rate — {algorithm} (Top {top_n})")
        else:
            filename = f"{metric.replace('/', '_')}_curves.png"
            ax.set_title(f"{metric} — {algorithm} (Top {top_n})")

        ax.legend(loc='best', fontsize=8)
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return str(filepath)

    def plot_success_over_time(self, experiments: List[ExperimentResult],
                               algorithm: str, top_n: int = 5) -> str:
        return self.plot_training_curves(
            experiments, "eval/success_once", algorithm,
            top_n=top_n, metric_label="Success Rate (success_once)"
        )

    def plot_critic_loss(self, experiments: List[ExperimentResult],
                         algorithm: str, top_n: int = 5) -> str:
        return self.plot_training_curves(
            experiments, "train/critic_loss", algorithm,
            top_n=top_n, metric_label="Critic Loss"
        )

    def plot_actor_loss(self, experiments: List[ExperimentResult],
                        algorithm: str, top_n: int = 5) -> str:
        return self.plot_training_curves(
            experiments, "train/actor_loss", algorithm,
            top_n=top_n, metric_label="Actor Loss"
        )

    def plot_alpha_curve(self, experiments: List[ExperimentResult],
                         algorithm: str, top_n: int = 5) -> str:
        """Plot SAC temperature (alpha) curve — DSRL-specific"""
        return self.plot_training_curves(
            experiments, "train/alpha", algorithm,
            top_n=top_n, metric_label="SAC Temperature (α)"
        )

    def plot_heatmap(self, df: pd.DataFrame, x_param: str, y_param: str,
                     algorithm: str) -> str:
        """Plot heatmap of success rate for two parameters"""
        if x_param not in df.columns or y_param not in df.columns:
            return ""

        pivot = df.pivot_table(
            values="success_once", index=y_param, columns=x_param, aggfunc="mean"
        )
        if pivot.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha='right')
        ax.set_yticklabels([str(v) for v in pivot.index])
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f"Success Rate Heatmap — {algorithm}")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("success_once")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                           color='black' if val > 0.5 else 'white', fontsize=9)

        plt.tight_layout()
        filepath = self.output_dir / f"{x_param}_vs_{y_param}_heatmap.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return str(filepath)


# =============================================================================
# Report Generation
# =============================================================================

class ReportGenerator:
    """Generate Markdown analysis report"""

    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_algorithm_report(self, analysis: SweepAnalysis,
                                   sensitivity_data: Dict[str, Dict],
                                   figure_paths: Dict[str, str]) -> str:
        lines = []
        lines.append(f"## {analysis.algorithm}")
        lines.append("")

        total = len(analysis.experiments)
        completed = len([e for e in analysis.experiments if e.status == "success"])
        failed = len([e for e in analysis.experiments if e.status == "failed"])
        lines.append(f"**Experiments**: {completed}/{total} completed ({failed} failed)")

        if analysis.best_config:
            lines.append(f"**Best Config**: `{analysis.best_config}` (success_once = {analysis.best_score:.3f})")
        lines.append("")

        # Results table
        sorted_exps = sorted(
            [e for e in analysis.experiments if e.success_once is not None],
            key=lambda x: x.success_once, reverse=True
        )
        if sorted_exps:
            lines.append("### Results")
            lines.append("")
            lines.append("| Rank | Config | success_once | success_at_end | Status |")
            lines.append("|------|--------|-------------|---------------|--------|")
            for rank, exp in enumerate(sorted_exps, 1):
                so = f"{exp.success_once:.3f}" if exp.success_once is not None else "-"
                se = f"{exp.success_at_end:.3f}" if exp.success_at_end is not None else "-"
                lines.append(f"| {rank} | `{exp.config_name}` | {so} | {se} | {exp.status} |")
            lines.append("")

        # Parameter sensitivity ranking
        if sensitivity_data:
            lines.append("### Parameter Sensitivity Ranking")
            lines.append("")
            lines.append("| Rank | Parameter | Sensitivity | Best Value | Best Mean |")
            lines.append("|------|-----------|-------------|------------|-----------|")

            sorted_params = sorted(
                sensitivity_data.items(),
                key=lambda x: x[1]["sensitivity"],
                reverse=True
            )
            for rank, (param, data) in enumerate(sorted_params[:10], 1):
                lines.append(f"| {rank} | `{param}` | {data['sensitivity']:.4f} | "
                           f"{data['best_value']} | {data['best_mean']:.3f} |")
            lines.append("")

        # Figures
        for key, label in [
            ("sensitivity", "Parameter Sensitivity"),
            ("ranking", "Configuration Ranking"),
            ("eval_curves", "Evaluation Success Curves"),
            ("critic_curves", "Critic Loss Curves"),
            ("actor_curves", "Actor Loss Curves"),
            ("alpha_curves", "SAC Temperature (α) Curves"),
        ]:
            if key in figure_paths and figure_paths[key]:
                lines.append(f"### {label}")
                rel_path = f"{analysis.algorithm}/{Path(figure_paths[key]).name}"
                lines.append(f"![{label}]({rel_path})")
                lines.append("")

        # Key findings
        lines.append("### Key Findings")
        lines.append("")
        if sensitivity_data:
            top_params = sorted(
                sensitivity_data.items(),
                key=lambda x: x[1]["sensitivity"],
                reverse=True
            )[:3]
            for param, data in top_params:
                lines.append(f"- **{param}**: Best value = `{data['best_value']}` "
                           f"(mean = {data['best_mean']:.3f}, sensitivity = {data['sensitivity']:.3f})")
        lines.append("")
        return "\n".join(lines)

    def generate_full_report(self, all_analyses: Dict[str, SweepAnalysis],
                             all_sensitivity: Dict[str, Dict],
                             all_figures: Dict[str, Dict[str, str]]) -> str:
        lines = []
        lines.append("# DSRL Sweep Experiment Analysis Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        best_algo = None
        best_score = 0
        best_config = None
        for algo, analysis in all_analyses.items():
            if analysis.best_score > best_score:
                best_score = analysis.best_score
                best_algo = algo
                best_config = analysis.best_config

        if best_algo:
            lines.append(f"**Overall Best**: `{best_algo}` / `{best_config}` "
                       f"with success_once = **{best_score:.3f}**")
            lines.append("")

        # Algorithm ranking table
        lines.append("### Algorithm Performance Ranking")
        lines.append("")
        lines.append("| Rank | Algorithm | Best success_once | Best Config |")
        lines.append("|------|-----------|-------------------|-------------|")

        sorted_algos = sorted(
            all_analyses.items(),
            key=lambda x: x[1].best_score,
            reverse=True
        )
        for rank, (algo, analysis) in enumerate(sorted_algos, 1):
            lines.append(f"| {rank} | `{algo}` | {analysis.best_score:.3f} | "
                       f"`{analysis.best_config}` |")
        lines.append("")

        # Per-algorithm sections
        for algo_name, _ in sorted_algos:
            if algo_name in all_analyses:
                figures = all_figures.get(algo_name, {})
                sensitivity = all_sensitivity.get(algo_name, {})
                lines.append(self.generate_algorithm_report(
                    all_analyses[algo_name], sensitivity, figures
                ))

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        lines.append("Based on the analysis:")
        lines.append("")
        for algo, analysis in sorted_algos[:3]:
            if analysis.best_config:
                lines.append(f"### {algo}")
                lines.append(f"- Use config: `{analysis.best_config}` (success_once = {analysis.best_score:.3f})")
                if algo in all_sensitivity:
                    sens = all_sensitivity[algo]
                    top_params = sorted(sens.items(), key=lambda x: x[1]["sensitivity"], reverse=True)[:5]
                    for param, data in top_params:
                        lines.append(f"- `{param}`: {data['best_value']}")
                lines.append("")

        return "\n".join(lines)

    def save_report(self, content: str, filename: str = "analysis_report.md"):
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        return str(filepath)


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

class SweepAnalyzer:
    """Main analysis pipeline for DSRL sweeps"""

    def __init__(self, sweep_dir: str, config_version: str = "v1",
                 output_dir: str = "analysis_results"):
        self.sweep_dir = Path(sweep_dir)
        self.config_version = config_version
        self.output_dir = Path(output_dir)

        self.extractor = DataExtractor(sweep_dir, config_version)
        self.visualizer = SweepVisualizer(output_dir)
        self.report_gen = ReportGenerator(output_dir)

        # DSRL algorithm
        self.algorithms = ["dsrl_sac"]

    def analyze_algorithm(self, algorithm: str) -> Tuple[Optional[SweepAnalysis], Dict, Dict[str, str]]:
        """Run full analysis for one algorithm"""
        print(f"Analyzing {algorithm}...")
        self.visualizer.set_algorithm_dir(algorithm)

        experiments = self.extractor.extract_algorithm(algorithm)
        if not experiments:
            print(f"  No experiments found for {algorithm}")
            return None, {}, {}

        print(f"  Found {len(experiments)} experiments")

        analysis = SweepAnalysis(algorithm=algorithm, experiments=experiments)

        completed = [e for e in experiments if e.success_once is not None]
        if completed:
            best_exp = max(completed, key=lambda x: x.success_once)
            analysis.best_config = best_exp.config_name
            analysis.best_score = best_exp.success_once

        # Sensitivity analysis
        sensitivity_data = {}
        if len(completed) >= 2:
            analyzer = SensitivityAnalyzer(completed)
            sensitivity_data = analyzer.analyze_all_params()
            analysis.param_sensitivity = {p: d["sensitivity"] for p, d in sensitivity_data.items()}

        # Generate figures
        figures = {}

        if sensitivity_data:
            figures["sensitivity"] = self.visualizer.plot_param_sensitivity_bar(
                sensitivity_data, algorithm
            )
            top_params = sorted(sensitivity_data.items(),
                               key=lambda x: x[1]["sensitivity"], reverse=True)[:3]
            for param, _ in top_params:
                fig_path = self.visualizer.plot_param_effect(sensitivity_data, param, algorithm)
                if fig_path:
                    figures[f"effect_{param}"] = fig_path

        if completed:
            figures["ranking"] = self.visualizer.plot_config_ranking(experiments, algorithm)

            has_curves = any(exp.training_curves for exp in experiments)
            if has_curves:
                eval_path = self.visualizer.plot_success_over_time(experiments, algorithm)
                if eval_path:
                    figures["eval_curves"] = eval_path

                critic_path = self.visualizer.plot_critic_loss(experiments, algorithm)
                if critic_path:
                    figures["critic_curves"] = critic_path

                actor_path = self.visualizer.plot_actor_loss(experiments, algorithm)
                if actor_path:
                    figures["actor_curves"] = actor_path

                # DSRL-specific: alpha (temperature) curve
                alpha_path = self.visualizer.plot_alpha_curve(experiments, algorithm)
                if alpha_path:
                    figures["alpha_curves"] = alpha_path

        return analysis, sensitivity_data, figures

    def analyze_all(self, algorithms: Optional[List[str]] = None) -> str:
        """Run full analysis for all algorithms"""
        if algorithms is None:
            algorithms = self.algorithms

        all_analyses = {}
        all_sensitivity = {}
        all_figures = {}

        for algo in algorithms:
            result = self.analyze_algorithm(algo)
            if result[0] is not None:
                all_analyses[algo] = result[0]
                all_sensitivity[algo] = result[1]
                all_figures[algo] = result[2]

        self.visualizer.reset_to_base_dir()

        # Generate report
        report = self.report_gen.generate_full_report(
            all_analyses, all_sensitivity, all_figures
        )
        report_path = self.report_gen.save_report(report)

        # Save raw data
        self.save_raw_data(all_analyses, all_sensitivity)

        print(f"\nAnalysis complete!")
        print(f"Report saved to: {report_path}")
        print(f"Figures saved to: {self.output_dir}")

        return report_path

    def save_raw_data(self, all_analyses: Dict[str, SweepAnalysis],
                      all_sensitivity: Dict[str, Dict]):
        """Save raw analysis data as JSON"""

        def to_json(obj):
            if isinstance(obj, (bool,)):
                return bool(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: to_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_json(v) for v in obj]
            return obj

        data = {
            "timestamp": datetime.now().isoformat(),
            "config_version": self.config_version,
            "algorithms": {}
        }

        for algo, analysis in all_analyses.items():
            algo_data = {
                "best_config": analysis.best_config,
                "best_score": to_json(analysis.best_score),
                "experiments": []
            }

            for exp in analysis.experiments:
                exp_data = {
                    "config_name": exp.config_name,
                    "success_once": to_json(exp.success_once),
                    "success_at_end": to_json(exp.success_at_end),
                    "status": exp.status,
                    "params": {k: to_json(v) for k, v in exp.params.items()},
                    "final_metrics": {k: to_json(v) for k, v in exp.final_metrics.items()},
                }
                algo_data["experiments"].append(exp_data)

            if algo in all_sensitivity:
                algo_data["sensitivity"] = {
                    param: {
                        "sensitivity": to_json(d["sensitivity"]),
                        "best_value": to_json(d["best_value"]),
                        "best_mean": to_json(d["best_mean"]),
                    }
                    for param, d in all_sensitivity[algo].items()
                }

            data["algorithms"][algo] = algo_data

        filepath = self.output_dir / "analysis_data.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Raw data saved to: {filepath}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze DSRL-SAC sweep experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--sweep-dir", "-d",
        default=None,
        help="Directory containing sweep experiments (default: auto by config-version)"
    )

    parser.add_argument(
        "--config-version", "-v",
        default="v1",
        choices=["v1", "v2"],
        help="Config version"
    )

    parser.add_argument(
        "--algorithm", "-a",
        help="Analyze only specific algorithm (dsrl_sac)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for results (default: <sweep-dir>/analysis_results)"
    )

    parser.add_argument(
        "--no-curves",
        action="store_true",
        help="Skip training curve analysis (faster)"
    )

    args = parser.parse_args()

    if args.sweep_dir is None:
        if args.config_version == "v2":
            args.sweep_dir = "runs/dsrl_sweep_v2"
        else:
            args.sweep_dir = "runs/dsrl_sweep"

    if args.output_dir is None:
        args.output_dir = os.path.join(args.sweep_dir, "analysis_results")

    analyzer = SweepAnalyzer(
        sweep_dir=args.sweep_dir,
        config_version=args.config_version,
        output_dir=args.output_dir
    )

    if args.algorithm:
        analyzer.analyze_all([args.algorithm])
    else:
        analyzer.analyze_all()


if __name__ == "__main__":
    main()
