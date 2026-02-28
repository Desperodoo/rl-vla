#!/usr/bin/env python3
"""
超参数扫描结果分析脚本

功能：
1. 收集所有实验的训练指标 (WandB / TensorBoard / 日志)
2. 为每个算法找到最优超参数配置
3. 生成分析报告和可视化

用法：
    python scripts/analyze_sweep.py --log_dir logs/sweep_xxx
    python scripts/analyze_sweep.py --wandb_project ManiSkill-Sweep
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import glob

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="分析超参数扫描结果")
    parser.add_argument("--log_dir", type=str, help="扫描日志目录")
    parser.add_argument("--wandb_project", type=str, help="WandB 项目名称")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB 实体")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--metric", type=str, default="eval/success_once", 
                        help="用于比较的主指标")
    parser.add_argument("--secondary_metrics", type=str, nargs="*",
                        default=["eval/success_at_end", "eval/return"],
                        help="次要指标")
    parser.add_argument("--top_k", type=int, default=3, help="每个算法显示 Top K")
    parser.add_argument("--format", type=str, choices=["table", "json", "csv", "all"],
                        default="all", help="输出格式")
    # 新增选项
    parser.add_argument("--algorithm", type=str, default=None,
                        help="只分析指定算法")
    parser.add_argument("--export_best", action="store_true",
                        help="导出最优参数为 shell 脚本")
    parser.add_argument("--recursive", action="store_true",
                        help="递归搜索子目录中的日志")
    return parser.parse_args()


# =============================================================================
# 数据收集
# =============================================================================

def load_running_tasks(log_dir: str) -> Dict[str, Dict[str, Any]]:
    """从 running_tasks.txt 加载实验参数
    
    格式: pid|exp_name|algorithm|gpu_id|extra_params|timestamp
    """
    tasks_file = os.path.join(log_dir, "running_tasks.txt")
    params_map = {}
    
    if not os.path.exists(tasks_file):
        return params_map
    
    try:
        with open(tasks_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 5:
                    exp_name = parts[1]
                    extra_params = parts[4]
                    
                    # 解析参数字符串为字典
                    hp_dict = {}
                    # 匹配 --key value 格式，支持科学记数法（如 1e-4）
                    # 使用两步解析：先分割参数，再解析每个键值对
                    tokens = extra_params.split()
                    i = 0
                    while i < len(tokens):
                        if tokens[i].startswith("--"):
                            key = tokens[i][2:]  # 去掉 --
                            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                                value = tokens[i + 1]
                                try:
                                    # 尝试转换为数字（支持科学记数法）
                                    if "." in value or "e" in value.lower():
                                        hp_dict[key] = float(value)
                                    else:
                                        hp_dict[key] = int(value)
                                except ValueError:
                                    hp_dict[key] = value
                                i += 2
                            else:
                                # 布尔标志
                                hp_dict[key] = True
                                i += 1
                        else:
                            i += 1
                    
                    params_map[exp_name] = hp_dict
    except Exception as e:
        print(f"   ⚠️ 读取 running_tasks.txt 失败: {e}")
    
    return params_map


def collect_from_logs(log_dir: str, recursive: bool = False) -> pd.DataFrame:
    """从训练日志文件收集结果"""
    print(f"📂 从日志目录收集: {log_dir}")
    
    results = []
    
    if recursive:
        log_files = glob.glob(os.path.join(log_dir, "**/*.log"), recursive=True)
    else:
        log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    # 加载参数映射（从 running_tasks.txt）
    # 对于递归模式，尝试从每个子目录加载
    all_params_map = {}
    if recursive:
        for log_file in log_files:
            parent_dir = os.path.dirname(log_file)
            if parent_dir not in all_params_map:
                params = load_running_tasks(parent_dir)
                all_params_map.update(params)
    else:
        all_params_map = load_running_tasks(log_dir)
    
    for log_file in log_files:
        exp_name = Path(log_file).stem
        
        # 解析实验名称: task_algo_cfgN_obsmode
        match = re.match(r"(.+?)_([a-z_]+)_cfg(\d+)_(.+)", exp_name)
        if not match:
            continue
            
        task, algo, cfg_idx, obs_mode = match.groups()
        
        # 解析日志文件
        metrics = parse_log_file(log_file)
        if metrics:
            metrics.update({
                "exp_name": exp_name,
                "task": task,
                "algorithm": algo,
                "config_idx": int(cfg_idx),
                "obs_mode": obs_mode,
                "log_file": log_file
            })
            
            # 添加从 running_tasks.txt 读取的超参数
            if exp_name in all_params_map:
                for key, value in all_params_map[exp_name].items():
                    metrics[f"hp_{key}"] = value
            
            results.append(metrics)
    
    df = pd.DataFrame(results)
    print(f"   收集到 {len(df)} 个实验结果")
    return df


def parse_log_file(log_file: str) -> Optional[Dict[str, Any]]:
    """解析单个日志文件，提取最终指标"""
    try:
        with open(log_file, "r") as f:
            content = f.read()
        
        metrics = {}
        
        # 常见指标的正则表达式
        patterns = {
            "eval/success_once": r"eval/success_once[:\s]+([0-9.]+)",
            "eval/success_at_end": r"eval/success_at_end[:\s]+([0-9.]+)",
            "eval/return": r"eval/return[:\s]+([0-9.-]+)",
            "train/loss": r"train/loss[:\s]+([0-9.]+)",
            "final_success": r"Final success.*?([0-9.]+)",
            "best_success": r"Best success.*?([0-9.]+)",
        }
        
        for metric_name, pattern in patterns.items():
            # 找最后一个匹配（最终值）
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                try:
                    metrics[metric_name] = float(matches[-1])
                except ValueError:
                    pass
        
        # 检查训练是否完成
        if "Training completed" in content or "Final" in content:
            metrics["completed"] = True
        else:
            metrics["completed"] = False
            
        # 提取超参数（从命令行参数）
        hp_patterns = {
            "lr": r"--lr\s+([0-9.e-]+)",
            "beta": r"--beta\s+([0-9.]+)",
            "alpha": r"--alpha\s+([0-9.]+)",
            "consistency_weight": r"--consistency_weight\s+([0-9.]+)",
            "shortcut_weight": r"--shortcut_weight\s+([0-9.]+)",
            "num_flow_steps": r"--num_flow_steps\s+(\d+)",
            "num_diffusion_iters": r"--num_diffusion_iters\s+(\d+)",
            "reward_scale": r"--reward_scale\s+([0-9.]+)",
            "sc_num_inference_steps": r"--sc_num_inference_steps\s+(\d+)",
        }
        
        for hp_name, pattern in hp_patterns.items():
            match = re.search(pattern, content)
            if match:
                val = match.group(1)
                try:
                    metrics[f"hp_{hp_name}"] = float(val) if "." in val or "e" in val else int(val)
                except ValueError:
                    metrics[f"hp_{hp_name}"] = val
        
        return metrics if metrics else None
        
    except Exception as e:
        print(f"   ⚠️ 解析失败 {log_file}: {e}")
        return None


def collect_from_wandb(project: str, entity: Optional[str] = None) -> pd.DataFrame:
    """从 WandB 收集结果"""
    print(f"📡 从 WandB 收集: {project}")
    
    try:
        import wandb
        api = wandb.Api()
        
        # 获取所有 runs
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path)
        
        results = []
        for run in runs:
            # 提取配置和指标
            config = run.config
            summary = run.summary._json_dict
            
            # 解析实验名称
            exp_name = run.name
            match = re.match(r"(.+?)_([a-z_]+)_cfg(\d+)_(.+)", exp_name)
            if not match:
                continue
            
            task, algo, cfg_idx, obs_mode = match.groups()
            
            metrics = {
                "exp_name": exp_name,
                "task": task,
                "algorithm": algo,
                "config_idx": int(cfg_idx),
                "obs_mode": obs_mode,
                "run_id": run.id,
                "state": run.state,
            }
            
            # 添加关键指标
            for key in ["eval/success_once", "eval/success_at_end", "eval/return"]:
                if key in summary:
                    metrics[key] = summary[key]
            
            # 添加超参数
            for key, val in config.items():
                metrics[f"hp_{key}"] = val
            
            results.append(metrics)
        
        df = pd.DataFrame(results)
        print(f"   收集到 {len(df)} 个实验结果")
        return df
        
    except ImportError:
        print("   ⚠️ wandb 未安装")
        return pd.DataFrame()
    except Exception as e:
        print(f"   ⚠️ WandB 收集失败: {e}")
        return pd.DataFrame()


def collect_from_tensorboard(log_dir: str) -> pd.DataFrame:
    """从 TensorBoard 日志收集结果"""
    print(f"📊 从 TensorBoard 收集: {log_dir}")
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        results = []
        tb_dirs = glob.glob(os.path.join(log_dir, "**/events.out.tfevents.*"), recursive=True)
        
        for tb_file in tb_dirs:
            tb_dir = os.path.dirname(tb_file)
            exp_name = os.path.basename(tb_dir)
            
            try:
                ea = event_accumulator.EventAccumulator(tb_dir)
                ea.Reload()
                
                metrics = {"exp_name": exp_name}
                
                # 读取标量
                for tag in ea.Tags()["scalars"]:
                    events = ea.Scalars(tag)
                    if events:
                        # 取最后一个值
                        metrics[tag.replace("/", "_")] = events[-1].value
                
                results.append(metrics)
            except Exception:
                continue
        
        df = pd.DataFrame(results)
        print(f"   收集到 {len(df)} 个实验结果")
        return df
        
    except ImportError:
        print("   ⚠️ tensorboard 未安装")
        return pd.DataFrame()


# =============================================================================
# 分析
# =============================================================================

def analyze_results(
    df: pd.DataFrame,
    metric: str = "eval/success_once",
    secondary_metrics: List[str] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """分析结果，为每个算法找到最优配置"""
    
    if df.empty:
        return {"error": "没有数据可分析"}
    
    results = {
        "summary": {},
        "best_per_algorithm": {},
        "all_results": {}
    }
    
    # 按算法分组
    algorithms = df["algorithm"].unique()
    
    for algo in algorithms:
        algo_df = df[df["algorithm"] == algo].copy()
        
        if metric not in algo_df.columns:
            continue
        
        # 按主指标排序
        algo_df_sorted = algo_df.sort_values(metric, ascending=False)
        
        # 获取 top K
        top_k_rows = algo_df_sorted.head(top_k)
        
        # 找出超参数列
        hp_cols = [c for c in algo_df.columns if c.startswith("hp_")]
        
        best_config = top_k_rows.iloc[0] if len(top_k_rows) > 0 else None
        
        results["best_per_algorithm"][algo] = {
            "best_metric": float(best_config[metric]) if best_config is not None else None,
            "best_config": {
                col.replace("hp_", ""): best_config[col] 
                for col in hp_cols 
                if pd.notna(best_config[col])
            } if best_config is not None else {},
            "top_k": [
                {
                    "exp_name": row["exp_name"],
                    metric: float(row[metric]) if pd.notna(row[metric]) else None,
                    **{m: float(row[m]) if m in row and pd.notna(row[m]) else None 
                       for m in (secondary_metrics or [])},
                    "config": {
                        col.replace("hp_", ""): row[col] 
                        for col in hp_cols 
                        if pd.notna(row[col])
                    }
                }
                for _, row in top_k_rows.iterrows()
            ],
            "num_experiments": len(algo_df)
        }
        
        # 统计信息
        results["summary"][algo] = {
            "count": len(algo_df),
            "mean": float(algo_df[metric].mean()) if metric in algo_df else None,
            "std": float(algo_df[metric].std()) if metric in algo_df else None,
            "max": float(algo_df[metric].max()) if metric in algo_df else None,
            "min": float(algo_df[metric].min()) if metric in algo_df else None,
        }
    
    # 全局最佳
    if metric in df.columns:
        best_overall = df.loc[df[metric].idxmax()]
        results["best_overall"] = {
            "algorithm": best_overall["algorithm"],
            "exp_name": best_overall["exp_name"],
            "metric": float(best_overall[metric])
        }
    
    return results


def generate_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """基于分析结果生成推荐"""
    recommendations = []
    
    if "best_overall" in analysis:
        best = analysis["best_overall"]
        recommendations.append(
            f"🏆 全局最佳: {best['algorithm']} 达到 {best['metric']:.4f}"
        )
    
    # 按算法性能排序
    algo_scores = {
        algo: data["best_metric"]
        for algo, data in analysis.get("best_per_algorithm", {}).items()
        if data["best_metric"] is not None
    }
    sorted_algos = sorted(algo_scores.items(), key=lambda x: x[1], reverse=True)
    
    recommendations.append("\n📊 算法排名 (按最佳配置表现):")
    for rank, (algo, score) in enumerate(sorted_algos, 1):
        recommendations.append(f"   {rank}. {algo}: {score:.4f}")
    
    # 超参数建议
    recommendations.append("\n💡 最优超参数建议:")
    for algo, data in analysis.get("best_per_algorithm", {}).items():
        config = data.get("best_config", {})
        if config:
            config_str = ", ".join(f"{k}={v}" for k, v in config.items())
            recommendations.append(f"   {algo}: {config_str}")
    
    return recommendations


# =============================================================================
# 输出
# =============================================================================

def print_table(analysis: Dict[str, Any], metric: str):
    """打印表格形式的结果"""
    print("\n" + "=" * 80)
    print("超参数扫描结果分析")
    print("=" * 80)
    
    # 统计表格
    print(f"\n📈 算法统计 (指标: {metric})")
    print("-" * 60)
    print(f"{'算法':<25} {'实验数':<8} {'最佳':<10} {'平均':<10} {'标准差':<10}")
    print("-" * 60)
    
    for algo, stats in sorted(analysis.get("summary", {}).items()):
        best = stats.get("max", 0) or 0
        mean = stats.get("mean", 0) or 0
        std = stats.get("std", 0) or 0
        count = stats.get("count", 0)
        print(f"{algo:<25} {count:<8} {best:<10.4f} {mean:<10.4f} {std:<10.4f}")
    
    # 最佳配置表格
    print(f"\n🏆 各算法最佳配置")
    print("-" * 80)
    
    for algo, data in analysis.get("best_per_algorithm", {}).items():
        print(f"\n【{algo}】 最佳: {data.get('best_metric', 'N/A'):.4f}")
        config = data.get("best_config", {})
        if config:
            print(f"   配置: {config}")
        
        print(f"   Top 配置:")
        for i, entry in enumerate(data.get("top_k", []), 1):
            entry_metric = entry.get(metric, "N/A")
            if isinstance(entry_metric, (int, float)):
                print(f"      {i}. {entry_metric:.4f} - {entry.get('config', {})}")
    
    # 推荐
    recommendations = generate_recommendations(analysis)
    print("\n" + "\n".join(recommendations))


def save_json(analysis: Dict[str, Any], output_path: str):
    """保存 JSON 格式结果"""
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    print(f"📄 JSON 已保存: {output_path}")


def save_csv(df: pd.DataFrame, output_path: str):
    """保存 CSV 格式结果"""
    df.to_csv(output_path, index=False)
    print(f"📄 CSV 已保存: {output_path}")


def save_markdown_report(
    analysis: Dict[str, Any], 
    df: pd.DataFrame, 
    metric: str,
    output_path: str
):
    """生成 Markdown 分析报告"""
    lines = ["# 超参数扫描分析报告\n"]
    
    # 概览
    lines.append("## 📊 概览\n")
    lines.append(f"- 总实验数: {len(df)}")
    lines.append(f"- 算法数: {len(analysis.get('summary', {}))}")
    lines.append(f"- 主指标: `{metric}`")
    
    if "best_overall" in analysis:
        best = analysis["best_overall"]
        lines.append(f"\n**🏆 全局最佳**: {best['algorithm']} ({best['metric']:.4f})\n")
    
    # 统计表格
    lines.append("\n## 📈 算法性能统计\n")
    lines.append("| 算法 | 实验数 | 最佳 | 平均 | 标准差 |")
    lines.append("|------|--------|------|------|--------|")
    
    for algo, stats in sorted(
        analysis.get("summary", {}).items(), 
        key=lambda x: x[1].get("max", 0) or 0,
        reverse=True
    ):
        best = stats.get("max", 0) or 0
        mean = stats.get("mean", 0) or 0
        std = stats.get("std", 0) or 0
        count = stats.get("count", 0)
        lines.append(f"| {algo} | {count} | {best:.4f} | {mean:.4f} | {std:.4f} |")
    
    # 最佳配置
    lines.append("\n## 🎯 各算法最佳配置\n")
    
    for algo, data in analysis.get("best_per_algorithm", {}).items():
        lines.append(f"### {algo}\n")
        lines.append(f"- 最佳指标: **{data.get('best_metric', 'N/A'):.4f}**")
        
        config = data.get("best_config", {})
        if config:
            config_str = ", ".join(f"`{k}={v}`" for k, v in config.items())
            lines.append(f"- 最佳配置: {config_str}")
        
        lines.append(f"\n**Top 配置:**\n")
        lines.append("| 排名 | 指标 | 配置 |")
        lines.append("|------|------|------|")
        for i, entry in enumerate(data.get("top_k", []), 1):
            entry_metric = entry.get(metric, "N/A")
            if isinstance(entry_metric, (int, float)):
                cfg_str = ", ".join(f"{k}={v}" for k, v in entry.get("config", {}).items())
                lines.append(f"| {i} | {entry_metric:.4f} | {cfg_str} |")
        lines.append("")
    
    # 推荐
    lines.append("\n## 💡 推荐\n")
    recommendations = generate_recommendations(analysis)
    for rec in recommendations:
        lines.append(rec)
    
    # 写入文件
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"📄 Markdown 报告已保存: {output_path}")


def export_best_params(
    analysis: Dict[str, Any],
    output_dir: str,
    algorithm: Optional[str] = None
):
    """导出最优参数为 shell 脚本，供下游算法继承
    
    输出格式:
    # best_params_flow_matching.sh
    BEST_LR="3e-4"
    BEST_NUM_FLOW_STEPS="10"
    BEST_OBS_HORIZON="2"
    ...
    """
    algos_to_export = [algorithm] if algorithm else analysis.get("best_per_algorithm", {}).keys()
    
    for algo in algos_to_export:
        if algo not in analysis.get("best_per_algorithm", {}):
            print(f"⚠️ 算法 {algo} 无结果，跳过导出")
            continue
        
        data = analysis["best_per_algorithm"][algo]
        config = data.get("best_config", {})
        
        if not config:
            print(f"⚠️ 算法 {algo} 无最优配置，跳过导出")
            continue
        
        output_path = os.path.join(output_dir, f"best_params_{algo}.sh")
        
        lines = [
            f"# 最优参数: {algo}",
            f"# 最佳指标: {data.get('best_metric', 'N/A')}",
            f"# 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        for key, value in config.items():
            # 转换为大写环境变量格式
            env_key = f"BEST_{key.upper()}"
            # 确保值被引号包围
            if isinstance(value, str):
                lines.append(f'{env_key}="{value}"')
            else:
                lines.append(f'{env_key}="{value}"')
        
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        
        print(f"📄 最优参数已导出: {output_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    args = parse_args()
    
    print("=" * 60)
    print("超参数扫描结果分析器")
    print("=" * 60)
    
    # 收集数据
    dfs = []
    
    if args.log_dir:
        df_logs = collect_from_logs(args.log_dir, recursive=args.recursive)
        if not df_logs.empty:
            dfs.append(df_logs)
        
        df_tb = collect_from_tensorboard(args.log_dir)
        if not df_tb.empty:
            dfs.append(df_tb)
    
    if args.wandb_project:
        df_wandb = collect_from_wandb(args.wandb_project, args.wandb_entity)
        if not df_wandb.empty:
            dfs.append(df_wandb)
    
    if not dfs:
        print("❌ 没有找到任何数据")
        return
    
    # 合并数据
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["exp_name"], keep="last")
    
    # 如果指定了算法，过滤数据
    if args.algorithm:
        df = df[df["algorithm"] == args.algorithm]
        if df.empty:
            print(f"❌ 没有找到算法 {args.algorithm} 的数据")
            return
    
    print(f"\n📊 总共收集到 {len(df)} 个实验结果")
    print(f"   算法: {df['algorithm'].unique().tolist()}")
    
    # 分析
    analysis = analyze_results(
        df,
        metric=args.metric,
        secondary_metrics=args.secondary_metrics,
        top_k=args.top_k
    )
    
    # 输出
    output_dir = args.output_dir or args.log_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    
    if args.format in ["table", "all"]:
        print_table(analysis, args.metric)
    
    if args.format in ["json", "all"]:
        save_json(analysis, os.path.join(output_dir, "sweep_analysis.json"))
    
    if args.format in ["csv", "all"]:
        save_csv(df, os.path.join(output_dir, "sweep_results.csv"))
    
    if args.format == "all":
        save_markdown_report(
            analysis, df, args.metric,
            os.path.join(output_dir, "sweep_report.md")
        )
    
    # 导出最优参数
    if args.export_best:
        export_best_params(analysis, output_dir, args.algorithm)
    
    print("\n✅ 分析完成!")


if __name__ == "__main__":
    main()
