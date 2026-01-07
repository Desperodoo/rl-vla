#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据可视化对比分析

用法:
    python -m inference_rtc.tests.visualize_debug \
        --inference debug_inference_*.npz \
        --servo debug_servo_*.npz \
        --demo ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_demo(demo_path: str):
    """加载 demo 数据"""
    demo_path = os.path.expanduser(demo_path)
    
    from inference.config import setup_rlft
    setup_rlft()
    from diffusion_policy.utils import load_traj_hdf5
    
    raw = load_traj_hdf5(demo_path, num_traj=1)
    return {
        'actions': raw['traj_0']['actions'],
        'obs': raw['traj_0']['obs'],
    }


def plot_joint_comparison(inference_data, servo_data, demo_data, joint_idx=0):
    """绘制单个关节的对比图"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    
    joint_names = ['关节1', '关节2', '关节3', '关节4', '关节5', '关节6', '夹爪']
    
    # 1. 推理输出的关键帧
    ax = axes[0]
    ax.set_title(f'{joint_names[joint_idx]} - 推理输出关键帧')
    
    if inference_data is not None and 'output_keyframes' in inference_data.files:
        kf = inference_data['output_keyframes']
        if len(kf) > 0:
            for i in range(min(len(kf), 20)):  # 显示前 20 次推理
                ax.plot(kf[i, :, joint_idx], 'o-', alpha=0.5, markersize=3)
            ax.set_ylabel('关节位置 (rad)')
            ax.set_xlabel('关键帧索引')
            ax.legend(['各次推理输出'])
    
    # 2. Servo 实际位置 vs 目标
    ax = axes[1]
    ax.set_title(f'{joint_names[joint_idx]} - Servo 实际位置 vs 插值目标')
    
    if servo_data is not None and 'actual_pos' in servo_data.files:
        actual = servo_data['actual_pos']
        interp = servo_data['interp_outputs']
        safe = servo_data['safe_outputs']
        
        if len(actual) > 0:
            # 降采样显示
            step = max(1, len(actual) // 2000)
            indices = np.arange(0, len(actual), step)
            
            t = (servo_data['timestamps'] - servo_data['timestamps'][0])[indices]
            
            ax.plot(t, actual[indices, joint_idx], label='实际位置', alpha=0.8)
            ax.plot(t, interp[indices, joint_idx], label='插值输出', alpha=0.6)
            ax.plot(t, safe[indices, joint_idx], '--', label='安全限制后', alpha=0.6)
            ax.set_ylabel('关节位置 (rad)')
            ax.set_xlabel('时间 (s)')
            ax.legend()
    
    # 3. 位置变化率
    ax = axes[2]
    ax.set_title(f'{joint_names[joint_idx]} - 位置变化率')
    
    if servo_data is not None and 'actual_pos' in servo_data.files:
        actual = servo_data['actual_pos']
        if len(actual) > 1:
            dt = np.diff(servo_data['timestamps'])
            dt[dt < 1e-6] = 1e-6  # 防止除零
            vel = np.diff(actual[:, joint_idx]) / dt
            
            step = max(1, len(vel) // 2000)
            indices = np.arange(0, len(vel), step)
            t = (servo_data['timestamps'][:-1] - servo_data['timestamps'][0])[indices]
            
            ax.plot(t, vel[indices], alpha=0.7)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_ylabel('变化率 (rad/s)')
            ax.set_xlabel('时间 (s)')
    
    # 4. Demo 对比
    ax = axes[3]
    ax.set_title(f'{joint_names[joint_idx]} - Demo 轨迹对比')
    
    if demo_data is not None:
        demo_actions = demo_data['actions']
        t_demo = np.arange(len(demo_actions)) * (1.0 / 30.0)  # 假设 30Hz
        ax.plot(t_demo, demo_actions[:, joint_idx], label='Demo 动作', linewidth=2)
        
        if 'joint_pos' in demo_data['obs']:
            demo_pos = demo_data['obs']['joint_pos']
            t_pos = np.arange(len(demo_pos)) * (1.0 / 30.0)
            ax.plot(t_pos, demo_pos[:, joint_idx], '--', label='Demo 位置', alpha=0.7)
        
        ax.set_ylabel('关节位置 (rad)')
        ax.set_xlabel('时间 (s)')
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_all_joints_summary(inference_data, servo_data, demo_data):
    """绘制所有关节的汇总图"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # 1. 动作分布对比
    ax = fig.add_subplot(gs[0, :])
    ax.set_title('关节动作分布对比 (各关节均值和标准差)')
    
    joint_indices = np.arange(6)
    width = 0.25
    
    if inference_data is not None and 'output_keyframes' in inference_data.files:
        kf = inference_data['output_keyframes']
        if len(kf) > 0:
            all_kf = kf.reshape(-1, kf.shape[-1])
            infer_mean = all_kf[:, :6].mean(axis=0)
            infer_std = all_kf[:, :6].std(axis=0)
            ax.bar(joint_indices - width, infer_mean, width, yerr=infer_std, 
                   label='推理输出', alpha=0.8, capsize=3)
    
    if demo_data is not None:
        demo_actions = demo_data['actions']
        demo_mean = demo_actions[:, :6].mean(axis=0)
        demo_std = demo_actions[:, :6].std(axis=0)
        ax.bar(joint_indices, demo_mean, width, yerr=demo_std, 
               label='Demo 动作', alpha=0.8, capsize=3)
    
    if servo_data is not None and 'actual_pos' in servo_data.files:
        actual = servo_data['actual_pos']
        if len(actual) > 0:
            actual_mean = actual[:, :6].mean(axis=0)
            actual_std = actual[:, :6].std(axis=0)
            ax.bar(joint_indices + width, actual_mean, width, yerr=actual_std, 
                   label='实际位置', alpha=0.8, capsize=3)
    
    ax.set_xticks(joint_indices)
    ax.set_xticklabels([f'J{i+1}' for i in range(6)])
    ax.set_ylabel('位置 (rad)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. 变化率分布
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title('位置变化率分布')
    
    if servo_data is not None and 'actual_pos' in servo_data.files:
        actual = servo_data['actual_pos']
        if len(actual) > 1:
            dt = np.diff(servo_data['timestamps'])
            dt[dt < 1e-6] = 1e-6
            vel = np.diff(actual[:, :6], axis=0) / dt[:, np.newaxis]
            vel_flat = vel.flatten()
            
            ax.hist(vel_flat, bins=100, density=True, alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('变化率 (rad/s)')
            ax.set_ylabel('密度')
            
            # 标记高频抖动阈值
            percentile_99 = np.percentile(np.abs(vel_flat), 99)
            ax.axvline(x=percentile_99, color='orange', linestyle='--', label=f'P99={percentile_99:.3f}')
            ax.axvline(x=-percentile_99, color='orange', linestyle='--')
            ax.legend()
    
    # 3. 推理 vs 实际差异
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title('插值目标 vs 实际位置 差异')
    
    if servo_data is not None and 'actual_pos' in servo_data.files:
        actual = servo_data['actual_pos']
        interp = servo_data['interp_outputs']
        
        if len(actual) > 0:
            diff = np.linalg.norm(interp[:, :6] - actual[:, :6], axis=1)
            step = max(1, len(diff) // 2000)
            t = (servo_data['timestamps'] - servo_data['timestamps'][0])[::step]
            
            ax.plot(t, diff[::step], alpha=0.7)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('差异范数 (rad)')
            ax.axhline(y=np.mean(diff), color='r', linestyle='--', 
                       label=f'均值={np.mean(diff):.4f}')
            ax.legend()
    
    # 4. 关键帧时间线
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title('关键帧更新时间线')
    
    if servo_data is not None and 'keyframe_versions' in servo_data.files:
        versions = servo_data['keyframe_versions']
        t_writes = servo_data['keyframe_t_writes']
        
        if len(versions) > 0:
            t_writes_rel = t_writes - t_writes[0] if len(t_writes) > 1 else t_writes
            ax.stem(t_writes_rel, versions, basefmt=' ')
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('版本号')
    
    # 5. 循环时间分布
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title('Servo 循环时间分布')
    
    if servo_data is not None and 'loop_times' in servo_data.files:
        loop_times = servo_data['loop_times'] * 1000  # 转为 ms
        
        ax.hist(loop_times, bins=50, density=True, alpha=0.7)
        ax.axvline(x=2.0, color='r', linestyle='--', label='目标 2ms')
        ax.set_xlabel('循环时间 (ms)')
        ax.set_ylabel('密度')
        
        mean_time = np.mean(loop_times)
        p99_time = np.percentile(loop_times, 99)
        ax.axvline(x=mean_time, color='g', linestyle='--', 
                   label=f'均值={mean_time:.2f}ms')
        ax.axvline(x=p99_time, color='orange', linestyle='--', 
                   label=f'P99={p99_time:.2f}ms')
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_input_state_analysis(inference_data, demo_data):
    """分析输入状态"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 输入状态对比
    ax = axes[0, 0]
    ax.set_title('输入状态 vs Demo 状态')
    
    if inference_data is not None and 'input_robot_pos' in inference_data.files:
        robot_pos = inference_data['input_robot_pos']
        if len(robot_pos) > 0:
            for j in range(6):
                ax.plot(robot_pos[:, j], label=f'J{j+1} 输入', alpha=0.7)
    
    if demo_data is not None and 'joint_pos' in demo_data['obs']:
        demo_pos = demo_data['obs']['joint_pos']
        # 缩放到相同长度
        for j in range(min(6, demo_pos.shape[1])):
            ax.plot(demo_pos[:50, j], '--', label=f'J{j+1} Demo', alpha=0.5)
    
    ax.set_xlabel('帧')
    ax.set_ylabel('位置 (rad)')
    ax.legend(fontsize=8, ncol=2)
    
    # 2. 输入状态是否为零
    ax = axes[0, 1]
    ax.set_title('输入状态范数 (检测零状态)')
    
    if inference_data is not None and 'input_robot_pos' in inference_data.files:
        robot_pos = inference_data['input_robot_pos']
        if len(robot_pos) > 0:
            norms = np.linalg.norm(robot_pos[:, :6], axis=1)
            ax.plot(norms, label='状态范数')
            ax.axhline(y=0.01, color='r', linestyle='--', label='零状态阈值')
            ax.set_xlabel('帧')
            ax.set_ylabel('范数')
            ax.legend()
            
            zero_count = np.sum(norms < 0.01)
            ax.text(0.02, 0.98, f'零状态帧: {zero_count}/{len(norms)}', 
                   transform=ax.transAxes, va='top')
    
    # 3. 推理时间分布
    ax = axes[1, 0]
    ax.set_title('推理时间分布')
    
    if inference_data is not None and 'inference_times' in inference_data.files:
        infer_times = inference_data['inference_times'] * 1000  # ms
        
        ax.hist(infer_times, bins=30, alpha=0.7)
        ax.axvline(x=100, color='r', linestyle='--', label='目标 100ms')
        ax.set_xlabel('推理时间 (ms)')
        ax.set_ylabel('次数')
        ax.legend()
        
        ax.text(0.98, 0.98, 
                f'均值: {np.mean(infer_times):.1f}ms\n'
                f'P99: {np.percentile(infer_times, 99):.1f}ms',
                transform=ax.transAxes, va='top', ha='right')
    
    # 4. 关键帧版本演进
    ax = axes[1, 1]
    ax.set_title('关键帧版本演进')
    
    if inference_data is not None and 'output_versions' in inference_data.files:
        versions = inference_data['output_versions']
        ax.plot(versions, 'o-')
        ax.set_xlabel('推理次数')
        ax.set_ylabel('版本号')
    
    plt.tight_layout()
    return fig


def plot_images(inference_data, demo_data, num_samples=5):
    """对比输入图像"""
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 6))
    
    # 推理输入图像
    if inference_data is not None and 'input_images' in inference_data.files:
        images = inference_data['input_images']
        if len(images) > 0:
            step = max(1, len(images) // num_samples)
            for i in range(num_samples):
                idx = min(i * step, len(images) - 1)
                ax = axes[0, i]
                ax.imshow(images[idx])
                ax.set_title(f'推理帧 {idx}')
                ax.axis('off')
    
    # Demo 图像
    if demo_data is not None and 'wrist_rgb' in demo_data['obs']:
        demo_images = demo_data['obs']['wrist_rgb']
        if len(demo_images) > 0:
            step = max(1, len(demo_images) // num_samples)
            for i in range(num_samples):
                idx = min(i * step, len(demo_images) - 1)
                ax = axes[1, i]
                ax.imshow(demo_images[idx])
                ax.set_title(f'Demo 帧 {idx}')
                ax.axis('off')
    
    axes[0, 0].set_ylabel('推理输入', fontsize=12)
    axes[1, 0].set_ylabel('Demo', fontsize=12)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="调试数据可视化")
    parser.add_argument("--inference", type=str, help="Inference 数据文件")
    parser.add_argument("--servo", type=str, help="Servo 数据文件")
    parser.add_argument("--demo", type=str, help="Demo 文件")
    parser.add_argument("--joint", type=int, default=0, help="关注的关节索引 (0-6)")
    parser.add_argument("--save", type=str, help="保存图像前缀")
    parser.add_argument("--show", action="store_true", help="显示图像")
    
    args = parser.parse_args()
    
    # 加载数据
    inference_data = None
    servo_data = None
    demo_data = None
    
    if args.inference and os.path.exists(args.inference):
        inference_data = np.load(args.inference)
        print(f"加载 inference 数据: {args.inference}")
    
    if args.servo and os.path.exists(args.servo):
        servo_data = np.load(args.servo)
        print(f"加载 servo 数据: {args.servo}")
    
    if args.demo:
        try:
            demo_data = load_demo(args.demo)
            print(f"加载 demo 数据: {args.demo}")
        except Exception as e:
            print(f"加载 demo 失败: {e}")
    
    # 绘制图像
    figs = []
    
    # 1. 单关节详细分析
    fig = plot_joint_comparison(inference_data, servo_data, demo_data, args.joint)
    figs.append(('joint_detail', fig))
    
    # 2. 所有关节汇总
    fig = plot_all_joints_summary(inference_data, servo_data, demo_data)
    figs.append(('all_joints_summary', fig))
    
    # 3. 输入状态分析
    if inference_data is not None:
        fig = plot_input_state_analysis(inference_data, demo_data)
        figs.append(('input_state', fig))
    
    # 4. 图像对比
    if inference_data is not None and 'input_images' in inference_data.files:
        fig = plot_images(inference_data, demo_data)
        figs.append(('images', fig))
    
    # 保存或显示
    if args.save:
        for name, fig in figs:
            filename = f"{args.save}_{name}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"保存: {filename}")
    
    if args.show:
        plt.show()
    elif not args.save:
        # 默认保存
        timestamp = __import__('time').strftime("%Y%m%d_%H%M%S")
        for name, fig in figs:
            filename = f"debug_plot_{timestamp}_{name}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"保存: {filename}")
    
    plt.close('all')


if __name__ == "__main__":
    main()
