#!/usr/bin/env python3
"""
Action Chunk 时间线分析脚本 (v2)

解析 timeline JSONL 日志，分析 action chunk 之间的重叠关系，
为真机 RL 数据采集和 training-time RTC 接入提供时间语义诊断。

功能：
    1. 基础统计（与 v1 兼容）
    2. Chunk 重叠分析：计算相邻 chunk 重叠率，act_horizon 内 chunk 切换次数
    3. Gantt 风格时间线可视化

使用方法:
    # 基础统计
    python analyze_timeline.py --logs timeline_*.jsonl --out summary.json

    # 可视化（单 episode）
    python analyze_timeline.py --logs timeline.jsonl --visualize --fig_out timeline.png

    # 指定 act_horizon（覆盖日志中的值）
    python analyze_timeline.py --logs timeline.jsonl --act_horizon 8 --visualize
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class InitEvent:
    """初始化事件"""
    timestamp: float
    desire_inference_freq: float = 30.0
    temporal_factor_k: float = 0.05
    chunk_time_base: str = 'sys_time'
    act_horizon: int = 16
    pred_horizon: int = 16
    obs_horizon: int = 2


@dataclass
class ChunkEvent:
    """Chunk 创建事件"""
    timestamp: float
    chunk_id: int
    chunk_base_time: float
    obs_stamp_ros: Optional[float]
    t_obs_ready_sys: Optional[float]
    action_interval: float  # 通常 1/30 秒
    pred_horizon: int       # chunk 长度（动作数）
    act_horizon: int        # 执行 horizon（用于标注）
    delta_chunk_obs: Optional[float]  # chunk_base_time - obs_stamp_ros
    chunk_targets: List[float] = field(default_factory=list)  # 目标时间序列
    
    @property
    def chunk_start(self) -> float:
        """Chunk 起始时间"""
        return self.chunk_targets[0] if self.chunk_targets else self.chunk_base_time
    
    @property
    def chunk_end(self) -> float:
        """Chunk 结束时间"""
        return self.chunk_targets[-1] if self.chunk_targets else self.chunk_base_time
    
    @property
    def chunk_duration(self) -> float:
        """Chunk 持续时间"""
        return self.chunk_end - self.chunk_start
    
    @property
    def act_horizon_end(self) -> float:
        """act_horizon 截止时间"""
        if len(self.chunk_targets) >= self.act_horizon:
            return self.chunk_targets[self.act_horizon - 1]
        return self.chunk_end


@dataclass
class ControlEvent:
    """控制下发事件"""
    timestamp: float
    query_time: float
    t_send_sys: float
    candidate_timestamps: List[float] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    num_candidates: int = 0
    used_chunk_ids: List[int] = field(default_factory=list)


@dataclass
class InferenceEvent:
    """推理事件"""
    timestamp: float
    t_infer_start: float
    t_infer_end: float
    inference_time: float


@dataclass
class RecordStepEvent:
    """采集步事件"""
    timestamp: float
    episode: int
    step: int
    obs_stamp_ros: Optional[float]
    t_obs_ready_sys: float
    delta_obs: Optional[float]
    t_action_query_sys: float
    delta_action_obs: Optional[float]
    action_present: bool


# ============================================================================
# 事件解析
# ============================================================================

def load_events(paths: List[str]) -> List[Dict]:
    """加载 JSONL 事件"""
    events = []
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def parse_events(raw_events: List[Dict]) -> Dict[str, List]:
    """将原始事件解析为结构化对象"""
    parsed = {
        'init': [],
        'chunk': [],
        'control': [],
        'inference': [],
        'record_step': [],
        'obs': [],
        'unknown': [],
    }

    for e in raw_events:
        event_type = e.get('event', 'unknown')
        ts = e.get('timestamp', e.get('t_sys', 0.0))
        
        if event_type == 'init':
            parsed['init'].append(InitEvent(
                timestamp=ts,
                desire_inference_freq=e.get('desire_inference_freq', 30.0),
                temporal_factor_k=e.get('temporal_factor_k', 0.05),
                chunk_time_base=e.get('chunk_time_base', 'sys_time'),
                act_horizon=e.get('act_horizon', 16),
                pred_horizon=e.get('pred_horizon', 16),
                obs_horizon=e.get('obs_horizon', 2),
            ))
        
        elif event_type == 'chunk':
            parsed['chunk'].append(ChunkEvent(
                timestamp=ts,
                chunk_id=e.get('chunk_id', -1),
                chunk_base_time=e.get('chunk_base_time', ts),
                obs_stamp_ros=e.get('obs_stamp_ros'),
                t_obs_ready_sys=e.get('t_obs_ready_sys'),
                action_interval=e.get('action_interval', 1/30),
                pred_horizon=e.get('pred_horizon', 16),
                act_horizon=e.get('act_horizon', 16),
                delta_chunk_obs=e.get('delta_chunk_obs'),
                chunk_targets=e.get('chunk_targets', []),
            ))
        
        elif event_type == 'control':
            parsed['control'].append(ControlEvent(
                timestamp=ts,
                query_time=e.get('query_time', ts),
                t_send_sys=e.get('t_send_sys', ts),
                candidate_timestamps=e.get('candidate_timestamps', []),
                weights=e.get('weights', []),
                num_candidates=e.get('num_candidates', 0),
                used_chunk_ids=e.get('used_chunk_ids', []),
            ))
        
        elif event_type == 'inference':
            parsed['inference'].append(InferenceEvent(
                timestamp=ts,
                t_infer_start=e.get('t_infer_start', ts),
                t_infer_end=e.get('t_infer_end', ts),
                inference_time=e.get('inference_time', 0.0),
            ))
        
        elif event_type == 'record_step':
            parsed['record_step'].append(RecordStepEvent(
                timestamp=ts,
                episode=e.get('episode', 0),
                step=e.get('step', 0),
                obs_stamp_ros=e.get('obs_stamp_ros'),
                t_obs_ready_sys=e.get('t_obs_ready_sys', ts),
                delta_obs=e.get('delta_obs'),
                t_action_query_sys=e.get('t_action_query_sys', ts),
                delta_action_obs=e.get('delta_action_obs'),
                action_present=e.get('action_present', False),
            ))
        
        elif event_type == 'obs':
            # obs 事件用于基础统计
            parsed['obs'].append(e)
        
        else:
            parsed['unknown'].append(e)
    
    return parsed


# ============================================================================
# 统计分析
# ============================================================================

def stats(arr: List[float]) -> Optional[Dict]:
    """计算统计量"""
    if len(arr) == 0:
        return None
    arr = np.array(arr, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def analyze_basic(parsed: Dict[str, List]) -> Dict:
    """基础统计分析（兼容 v1）"""
    result = {}
    
    # delta_obs (obs + record_step)
    delta_obs = []
    for e in parsed.get('obs', []):
        d = e.get('delta_obs')
        if d is not None:
            delta_obs.append(d)
    for e in parsed.get('record_step', []):
        if e.delta_obs is not None:
            delta_obs.append(e.delta_obs)
    result['delta_obs'] = stats(delta_obs)
    
    # inference_time
    inference_time = [e.inference_time for e in parsed.get('inference', []) if e.inference_time > 0]
    result['inference_time'] = stats(inference_time)
    
    # chunk 统计
    chunks: List[ChunkEvent] = parsed.get('chunk', [])
    delta_chunk_obs = [c.delta_chunk_obs for c in chunks if c.delta_chunk_obs is not None]
    chunk_len = [c.pred_horizon for c in chunks]
    chunk_target_offsets = []
    for c in chunks:
        for t in c.chunk_targets:
            chunk_target_offsets.append(t - c.chunk_base_time)
    
    result['delta_chunk_obs'] = stats(delta_chunk_obs)
    result['chunk_len'] = stats(chunk_len)
    result['chunk_target_offsets'] = stats(chunk_target_offsets)
    
    # control 统计
    controls: List[ControlEvent] = parsed.get('control', [])
    control_lag = [c.t_send_sys - c.query_time for c in controls]
    candidate_age = []
    candidate_span = []
    for c in controls:
        if c.candidate_timestamps:
            candidate_age.append(c.query_time - max(c.candidate_timestamps))
            candidate_span.append(max(c.candidate_timestamps) - min(c.candidate_timestamps))
    
    result['control_lag'] = stats(control_lag)
    result['candidate_age'] = stats(candidate_age)
    result['candidate_span'] = stats(candidate_span)
    
    # record_step 统计
    record_steps: List[RecordStepEvent] = parsed.get('record_step', [])
    if record_steps:
        present_rate = sum(1 for r in record_steps if r.action_present) / len(record_steps)
        result['record_action_present_rate'] = present_rate
        delta_action_obs = [r.delta_action_obs for r in record_steps if r.delta_action_obs is not None]
        result['delta_action_obs'] = stats(delta_action_obs)
    
    # 事件计数
    result['event_counts'] = {
        'init': len(parsed.get('init', [])),
        'chunk': len(chunks),
        'control': len(controls),
        'inference': len(parsed.get('inference', [])),
        'record_step': len(record_steps),
        'obs': len(parsed.get('obs', [])),
    }
    
    return result


# ============================================================================
# Chunk 重叠分析
# ============================================================================

def analyze_chunk_overlap(chunks: List[ChunkEvent], act_horizon_override: Optional[int] = None) -> Dict:
    """
    分析 chunk 之间的重叠关系
    
    Args:
        chunks: ChunkEvent 列表（按时间排序）
        act_horizon_override: 覆盖日志中的 act_horizon 值
        
    Returns:
        重叠分析结果
    """
    if len(chunks) < 2:
        return {
            'error': 'Need at least 2 chunks for overlap analysis',
            'num_chunks': len(chunks),
        }
    
    # 按 chunk_base_time 排序
    sorted_chunks = sorted(chunks, key=lambda c: c.chunk_base_time)
    
    # 计算相邻 chunk 的间隔和重叠
    inter_chunk_gaps = []  # 相邻 chunk 起始时间间隔
    chunk_overlaps = []    # 相邻 chunk 重叠时间
    overlap_ratios = []    # 重叠率 = overlap / chunk_duration
    
    for i in range(1, len(sorted_chunks)):
        prev = sorted_chunks[i-1]
        curr = sorted_chunks[i]
        
        # 间隔 = 当前 chunk 起始 - 上一个 chunk 起始
        gap = curr.chunk_start - prev.chunk_start
        inter_chunk_gaps.append(gap)
        
        # 重叠 = max(0, 上一个 chunk 结束 - 当前 chunk 起始)
        overlap = max(0, prev.chunk_end - curr.chunk_start)
        chunk_overlaps.append(overlap)
        
        # 重叠率
        if prev.chunk_duration > 0:
            ratio = overlap / prev.chunk_duration
            overlap_ratios.append(ratio)
    
    # 分析 act_horizon 内的 chunk 切换
    # 对于每个 chunk，统计在其 act_horizon 时间范围内有多少个后续 chunk 产生
    act_horizon_switches = []  # act_horizon 期间产生的新 chunk 数
    
    for i, chunk in enumerate(sorted_chunks):
        act_horizon = act_horizon_override if act_horizon_override else chunk.act_horizon
        act_end_time = chunk.chunk_base_time + act_horizon * chunk.action_interval
        
        # 统计在 [chunk_base_time, act_end_time] 内产生的后续 chunk
        new_chunks_in_act = 0
        for j in range(i + 1, len(sorted_chunks)):
            next_chunk = sorted_chunks[j]
            if next_chunk.chunk_base_time <= act_end_time:
                new_chunks_in_act += 1
            else:
                break
        act_horizon_switches.append(new_chunks_in_act)
    
    # 分析每个时刻的有效 chunk 数量
    # 采样时间点：从第一个 chunk 开始到最后一个 chunk 结束
    if sorted_chunks:
        t_start = sorted_chunks[0].chunk_start
        t_end = sorted_chunks[-1].chunk_end
        sample_interval = sorted_chunks[0].action_interval
        
        sample_times = np.arange(t_start, t_end, sample_interval)
        active_chunk_counts = []
        
        for t in sample_times:
            count = 0
            for chunk in sorted_chunks:
                if chunk.chunk_start <= t <= chunk.chunk_end:
                    count += 1
            active_chunk_counts.append(count)
    else:
        active_chunk_counts = []
    
    return {
        'num_chunks': len(sorted_chunks),
        'inter_chunk_gap': stats(inter_chunk_gaps),
        'chunk_overlap_time': stats(chunk_overlaps),
        'overlap_ratio': stats(overlap_ratios),
        'act_horizon_switches': stats(act_horizon_switches),
        'active_chunk_count': stats(active_chunk_counts),
        # 详细数据用于可视化
        '_inter_chunk_gaps': inter_chunk_gaps,
        '_overlap_ratios': overlap_ratios,
        '_act_horizon_switches': act_horizon_switches,
    }


def analyze_control_chunk_usage(controls: List[ControlEvent], chunks: List[ChunkEvent]) -> Dict:
    """
    分析控制步使用的 chunk 分布
    
    Args:
        controls: ControlEvent 列表
        chunks: ChunkEvent 列表
        
    Returns:
        chunk 使用分析结果
    """
    if not controls or not chunks:
        return {'error': 'No control or chunk events'}
    
    # 构建 chunk_id -> ChunkEvent 映射
    chunk_map = {c.chunk_id: c for c in chunks if c.chunk_id >= 0}
    
    # 统计每个控制步使用的 chunk 数量
    num_chunks_per_control = [c.num_candidates for c in controls]
    
    # 统计 chunk 被使用的次数
    chunk_usage_count = defaultdict(int)
    for c in controls:
        for cid in c.used_chunk_ids:
            chunk_usage_count[cid] += 1
    
    # chunk 使用次数分布
    usage_counts = list(chunk_usage_count.values()) if chunk_usage_count else [0]
    
    return {
        'num_chunks_per_control': stats(num_chunks_per_control),
        'chunk_usage_count': stats(usage_counts),
        'unique_chunks_used': len(chunk_usage_count),
        'total_chunks': len(chunks),
    }


# ============================================================================
# 可视化
# ============================================================================

def visualize_chunk_timeline(
    chunks: List[ChunkEvent],
    controls: List[ControlEvent],
    init_event: Optional[InitEvent] = None,
    act_horizon_override: Optional[int] = None,
    save_path: Optional[str] = None,
    time_range: Optional[Tuple[float, float]] = None,
    max_chunks: int = 50,
):
    """
    生成 Gantt 风格 chunk 时间线图
    
    Args:
        chunks: ChunkEvent 列表
        controls: ControlEvent 列表
        init_event: 初始化事件（用于获取参数）
        act_horizon_override: 覆盖 act_horizon
        save_path: 保存路径
        time_range: 显示时间范围 (start, end)
        max_chunks: 最大显示 chunk 数
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return
    
    if not chunks:
        print("Warning: No chunk events to visualize")
        return
    
    # 按时间排序
    sorted_chunks = sorted(chunks, key=lambda c: c.chunk_base_time)
    
    # 限制显示数量
    if len(sorted_chunks) > max_chunks:
        print(f"Warning: Limiting to first {max_chunks} chunks (total: {len(sorted_chunks)})")
        sorted_chunks = sorted_chunks[:max_chunks]
    
    # 确定时间范围
    if time_range:
        t_min, t_max = time_range
    else:
        t_min = sorted_chunks[0].chunk_start
        t_max = sorted_chunks[-1].chunk_end
    
    # 归一化时间（相对于 t_min）
    def norm_t(t):
        return t - t_min
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[3, 1, 1])
    fig.suptitle('Action Chunk Timeline Analysis', fontsize=14)
    
    # ---- 子图 1: Chunk Gantt 图 ----
    ax1 = axes[0]
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(sorted_chunks))))
    
    for idx, chunk in enumerate(sorted_chunks):
        y = idx
        start = norm_t(chunk.chunk_start)
        end = norm_t(chunk.chunk_end)
        duration = end - start
        
        # 完整 chunk 区域（浅色）
        ax1.barh(y, duration, left=start, height=0.8, 
                color=colors[idx % len(colors)], alpha=0.3, edgecolor='gray', linewidth=0.5)
        
        # act_horizon 区域（深色）
        act_horizon = act_horizon_override if act_horizon_override else chunk.act_horizon
        act_duration = min(act_horizon * chunk.action_interval, duration)
        ax1.barh(y, act_duration, left=start, height=0.8,
                color=colors[idx % len(colors)], alpha=0.7, edgecolor='black', linewidth=1)
        
        # 标注 chunk_id
        ax1.text(start + 0.01, y, f'C{chunk.chunk_id}', va='center', fontsize=8)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Chunk ID')
    ax1.set_yticks(range(len(sorted_chunks)))
    ax1.set_yticklabels([f'{c.chunk_id}' for c in sorted_chunks])
    ax1.set_xlim(0, norm_t(t_max))
    ax1.set_title('Chunk Timeline (dark = act_horizon, light = full pred_horizon)')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 图例
    act_patch = mpatches.Patch(color='steelblue', alpha=0.7, label='act_horizon')
    pred_patch = mpatches.Patch(color='steelblue', alpha=0.3, label='pred_horizon (remaining)')
    ax1.legend(handles=[act_patch, pred_patch], loc='upper right')
    
    # ---- 子图 2: 每个时刻的有效 chunk 数量 ----
    ax2 = axes[1]
    
    sample_interval = sorted_chunks[0].action_interval if sorted_chunks else 0.033
    sample_times = np.arange(t_min, t_max, sample_interval)
    active_counts = []
    
    for t in sample_times:
        count = 0
        for chunk in sorted_chunks:
            if chunk.chunk_start <= t <= chunk.chunk_end:
                count += 1
        active_counts.append(count)
    
    ax2.fill_between(sample_times - t_min, active_counts, alpha=0.5, label='Active chunks')
    ax2.plot(sample_times - t_min, active_counts, 'b-', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Active Chunks')
    ax2.set_xlim(0, norm_t(t_max))
    ax2.set_title('Number of Active Chunks Over Time')
    ax2.grid(True, alpha=0.3)
    
    # ---- 子图 3: Control 事件的 chunk 使用 ----
    ax3 = axes[2]
    
    if controls:
        # 过滤时间范围内的 control 事件
        filtered_controls = [c for c in controls if t_min <= c.query_time <= t_max]
        
        if filtered_controls:
            control_times = [norm_t(c.query_time) for c in filtered_controls]
            num_candidates = [c.num_candidates for c in filtered_controls]
            
            ax3.scatter(control_times, num_candidates, s=10, alpha=0.5, label='Num candidates')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Chunks Used')
            ax3.set_xlim(0, norm_t(t_max))
            ax3.set_title('Number of Chunks Used per Control Step')
            ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_overlap_histogram(
    overlap_result: Dict,
    save_path: Optional[str] = None,
):
    """
    生成重叠分析直方图
    
    Args:
        overlap_result: analyze_chunk_overlap 的返回结果
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Chunk Overlap Analysis', fontsize=14)
    
    # 1. 相邻 chunk 间隔分布
    ax1 = axes[0, 0]
    gaps = overlap_result.get('_inter_chunk_gaps', [])
    if gaps:
        ax1.hist(gaps, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(gaps), color='r', linestyle='--', label=f'Mean: {np.mean(gaps)*1000:.1f}ms')
    ax1.set_xlabel('Inter-chunk Gap (s)')
    ax1.set_ylabel('Count')
    ax1.set_title('Inter-chunk Gap Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 重叠率分布
    ax2 = axes[0, 1]
    ratios = overlap_result.get('_overlap_ratios', [])
    if ratios:
        ax2.hist(ratios, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(ratios), color='r', linestyle='--', label=f'Mean: {np.mean(ratios)*100:.1f}%')
    ax2.set_xlabel('Overlap Ratio')
    ax2.set_ylabel('Count')
    ax2.set_title('Chunk Overlap Ratio Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. act_horizon 期间新 chunk 产生数分布
    ax3 = axes[1, 0]
    switches = overlap_result.get('_act_horizon_switches', [])
    if switches:
        unique_vals = sorted(set(switches))
        counts = [switches.count(v) for v in unique_vals]
        ax3.bar(unique_vals, counts, edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(switches), color='r', linestyle='--', label=f'Mean: {np.mean(switches):.2f}')
    ax3.set_xlabel('New Chunks in act_horizon')
    ax3.set_ylabel('Count')
    ax3.set_title('Chunk Switches within act_horizon')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 统计摘要文本
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Summary Statistics\n" + "=" * 30 + "\n\n"
    
    if overlap_result.get('inter_chunk_gap'):
        gap_stats = overlap_result['inter_chunk_gap']
        summary_text += f"Inter-chunk Gap:\n"
        summary_text += f"  Mean: {gap_stats['mean']*1000:.2f} ms\n"
        summary_text += f"  Std:  {gap_stats['std']*1000:.2f} ms\n"
        summary_text += f"  P95:  {gap_stats['p95']*1000:.2f} ms\n\n"
    
    if overlap_result.get('overlap_ratio'):
        ratio_stats = overlap_result['overlap_ratio']
        summary_text += f"Overlap Ratio:\n"
        summary_text += f"  Mean: {ratio_stats['mean']*100:.1f}%\n"
        summary_text += f"  Std:  {ratio_stats['std']*100:.1f}%\n"
        summary_text += f"  P95:  {ratio_stats['p95']*100:.1f}%\n\n"
    
    if overlap_result.get('act_horizon_switches'):
        switch_stats = overlap_result['act_horizon_switches']
        summary_text += f"Switches in act_horizon:\n"
        summary_text += f"  Mean: {switch_stats['mean']:.2f}\n"
        summary_text += f"  Max:  {switch_stats['max']:.0f}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved overlap histogram to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze timeline JSONL logs (v2)')
    parser.add_argument('--logs', type=str, nargs='+', required=True, 
                        help='Timeline JSONL log paths')
    parser.add_argument('--out', type=str, default='', 
                        help='Output JSON path for statistics')
    parser.add_argument('--visualize', action='store_true', 
                        help='Generate visualization')
    parser.add_argument('--fig_out', type=str, default='', 
                        help='Output path for timeline figure (requires --visualize)')
    parser.add_argument('--hist_out', type=str, default='', 
                        help='Output path for histogram figure (requires --visualize)')
    parser.add_argument('--act_horizon', type=int, default=None, 
                        help='Override act_horizon value from logs')
    parser.add_argument('--max_chunks', type=int, default=50, 
                        help='Max chunks to display in timeline visualization')
    parser.add_argument('--time_range', type=float, nargs=2, default=None,
                        help='Time range to display (start end) in seconds relative to first chunk')
    args = parser.parse_args()
    
    # 加载事件
    print(f"Loading events from {len(args.logs)} file(s)...")
    raw_events = load_events(args.logs)
    print(f"Loaded {len(raw_events)} raw events")
    
    # 解析事件
    parsed = parse_events(raw_events)
    print(f"Parsed: {len(parsed['init'])} init, {len(parsed['chunk'])} chunks, "
          f"{len(parsed['control'])} controls, {len(parsed['inference'])} inferences")
    
    # 基础统计
    basic_stats = analyze_basic(parsed)
    
    # Chunk 重叠分析
    chunks = parsed.get('chunk', [])
    overlap_result = analyze_chunk_overlap(chunks, args.act_horizon)
    
    # Control chunk 使用分析
    controls = parsed.get('control', [])
    usage_result = analyze_control_chunk_usage(controls, chunks)
    
    # 合并结果
    result = {
        'basic': basic_stats,
        'chunk_overlap': {k: v for k, v in overlap_result.items() if not k.startswith('_')},
        'control_chunk_usage': usage_result,
    }
    
    # 添加 init 参数信息
    if parsed.get('init'):
        init_evt = parsed['init'][0]
        result['config'] = {
            'act_horizon': init_evt.act_horizon,
            'pred_horizon': init_evt.pred_horizon,
            'obs_horizon': init_evt.obs_horizon,
            'desire_inference_freq': init_evt.desire_inference_freq,
            'temporal_factor_k': init_evt.temporal_factor_k,
        }
    
    # 输出 JSON
    print("\n" + "=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved statistics to: {args.out}")
    
    # 可视化
    if args.visualize:
        init_evt = parsed['init'][0] if parsed.get('init') else None
        
        # 时间线图
        time_range = None
        if args.time_range:
            # 转换为绝对时间
            if chunks:
                t0 = sorted(chunks, key=lambda c: c.chunk_base_time)[0].chunk_start
                time_range = (t0 + args.time_range[0], t0 + args.time_range[1])
        
        fig_path = args.fig_out if args.fig_out else None
        visualize_chunk_timeline(
            chunks, controls, init_evt, 
            args.act_horizon, fig_path, time_range, args.max_chunks
        )
        
        # 重叠直方图
        hist_path = args.hist_out if args.hist_out else None
        visualize_overlap_histogram(overlap_result, hist_path)


if __name__ == '__main__':
    main()
