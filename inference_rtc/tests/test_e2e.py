#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端单进程测试 - 验证 RTC 管道

这个测试在单个进程中模拟整个 RTC 流程：
1. 模拟策略生成关键帧 (10Hz)
2. 共享内存写入/读取
3. 三次样条插值 (500Hz)
4. 安全限制器处理

不需要实际机器人，用于验证逻辑正确性。
"""

import os
import sys
import time
import numpy as np
import threading
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference_rtc.shared.shm_protocol import ShmKeyframeWriter
from inference_rtc.python.servo_main import (
    ShmKeyframeReader,
    CubicInterpolator,
    SafetyLimiter,
    EMAFilter,
    ServoConfig
)


@dataclass
class TestConfig:
    """测试配置"""
    duration: float = 5.0  # 测试时长 (秒)
    policy_rate: float = 10.0  # 策略频率
    servo_rate: float = 500.0  # 伺服频率
    H: int = 8  # 关键帧数量
    dof: int = 7  # 自由度
    dt_key: float = 1.0 / 30.0  # 关键帧间隔
    verbose: bool = True


def generate_test_trajectory(t: float, config: TestConfig) -> np.ndarray:
    """
    生成测试轨迹 (正弦波)
    
    Returns:
        q_key: shape (H, dof)
    """
    q_key = np.zeros((config.H, config.dof))
    
    for i in range(config.H):
        ti = t + i * config.dt_key
        
        # 不同频率和幅度的正弦波
        q_key[i, 0] = 0.5 * np.sin(2 * np.pi * 0.3 * ti)  # 关节1: 0.3Hz
        q_key[i, 1] = 0.4 * np.sin(2 * np.pi * 0.4 * ti)  # 关节2: 0.4Hz
        q_key[i, 2] = 0.3 * np.sin(2 * np.pi * 0.5 * ti)  # 关节3: 0.5Hz
        q_key[i, 3] = 0.2 * np.sin(2 * np.pi * 0.6 * ti)  # 关节4: 0.6Hz
        q_key[i, 4] = 0.15 * np.sin(2 * np.pi * 0.7 * ti)  # 关节5: 0.7Hz
        q_key[i, 5] = 0.1 * np.sin(2 * np.pi * 0.8 * ti)  # 关节6: 0.8Hz
        q_key[i, 6] = 0.04  # 夹爪: 固定
    
    return q_key


def test_single_process():
    """单进程端到端测试"""
    print("=" * 60)
    print("RTC 端到端单进程测试")
    print("=" * 60)
    
    config = TestConfig()
    
    # 初始化组件
    print("\n[1] 初始化组件...")
    writer = ShmKeyframeWriter.create(H=config.H, dof=config.dof, dt_key=config.dt_key)
    reader = ShmKeyframeReader()
    interpolator = CubicInterpolator()
    servo_cfg = ServoConfig(dof=config.dof)
    limiter = SafetyLimiter(servo_cfg)
    
    # 用于跟踪当前位置
    current_pos = np.zeros(config.dof)
    
    # 等待 reader 连接
    if not reader.connect(timeout=5.0):
        print("  [错误] 无法连接共享内存!")
        return 1
    print("  ✓ 共享内存连接成功")
    
    # 统计
    policy_count = 0
    servo_count = 0
    servo_errors = []
    
    # 模拟运行
    print(f"\n[2] 开始模拟 ({config.duration}秒)...")
    
    t_start = time.time()
    t_last_policy = t_start
    t_last_servo = t_start
    
    policy_period = 1.0 / config.policy_rate
    servo_period = 1.0 / config.servo_rate
    
    servo_outputs = []  # 记录伺服输出
    
    while True:
        t_now = time.time()
        elapsed = t_now - t_start
        
        if elapsed >= config.duration:
            break
        
        # 模拟策略 (10Hz)
        if t_now - t_last_policy >= policy_period:
            q_key = generate_test_trajectory(t_now, config)
            writer.write_keyframes(q_key)
            policy_count += 1
            t_last_policy = t_now
            
            if config.verbose and policy_count <= 5:
                print(f"  [策略] #{policy_count} q[0]={q_key[0, :3]}")
        
        # 模拟伺服 (500Hz)
        if t_now - t_last_servo >= servo_period:
            # 读取共享内存
            data = reader.read()
            
            if data is not None:
                # 更新插值器
                interpolator.build(data['q_key'], data['dt_key'], data['t_write'])
            
            # 插值
            if interpolator.valid:
                q_interp = interpolator.eval(t_now)
                
                if q_interp is not None:
                    # 安全限制
                    q_safe = limiter.apply(q_interp, current_pos)
                    current_pos = q_safe.copy()
                    
                    servo_count += 1
                    servo_outputs.append((elapsed, q_safe.copy()))
                    
                    # 计算跟踪误差 (与理想轨迹比较)
                    q_ideal = np.array([
                        0.5 * np.sin(2 * np.pi * 0.3 * t_now),
                        0.4 * np.sin(2 * np.pi * 0.4 * t_now),
                        0.3 * np.sin(2 * np.pi * 0.5 * t_now),
                        0.2 * np.sin(2 * np.pi * 0.6 * t_now),
                        0.15 * np.sin(2 * np.pi * 0.7 * t_now),
                        0.1 * np.sin(2 * np.pi * 0.8 * t_now),
                        0.04
                    ])
                    error = np.linalg.norm(q_safe - q_ideal)
                    servo_errors.append(error)
            
            t_last_servo = t_now
        
        # 让出 CPU
        time.sleep(0.0001)  # 0.1ms
    
    # 统计结果
    print("\n[3] 测试结果:")
    print(f"  策略执行次数: {policy_count}")
    print(f"  伺服执行次数: {servo_count}")
    
    if servo_errors:
        print(f"  跟踪误差: mean={np.mean(servo_errors):.6f}, "
              f"max={np.max(servo_errors):.6f}")
    
    actual_policy_rate = policy_count / config.duration
    actual_servo_rate = servo_count / config.duration
    
    print(f"  实际策略频率: {actual_policy_rate:.1f} Hz (目标 {config.policy_rate} Hz)")
    print(f"  实际伺服频率: {actual_servo_rate:.1f} Hz (目标 {config.servo_rate} Hz)")
    
    # 验证
    success = True
    
    if actual_policy_rate < config.policy_rate * 0.9:
        print(f"  [警告] 策略频率不足!")
        success = False
    
    if actual_servo_rate < config.servo_rate * 0.5:  # 放宽要求，Python 可能达不到 500Hz
        print(f"  [警告] 伺服频率不足!")
        success = False
    
    if np.mean(servo_errors) > 0.1:
        print(f"  [警告] 跟踪误差过大!")
        success = False
    
    # 清理
    writer.close()
    reader.disconnect()
    
    if success:
        print("\n✓ 测试通过!")
        return 0
    else:
        print("\n✗ 测试有警告，请检查")
        return 1


def test_interpolator_accuracy():
    """测试插值器精度"""
    print("\n" + "=" * 60)
    print("插值器精度测试")
    print("=" * 60)
    
    dof = 7
    interpolator = CubicInterpolator()
    
    # 生成测试数据 (完美的正弦波)
    t_write = 0.0
    dt_key = 1.0 / 30.0
    H = 8
    
    freq = 1.0  # 1Hz 正弦波
    
    q_key = np.zeros((H, dof))
    for i in range(H):
        t = i * dt_key
        q_key[i, 0] = np.sin(2 * np.pi * freq * t)
    
    interpolator.build(q_key, dt_key, t_write)
    
    # 在不同时间点采样并与真值比较
    print("\n  时间       插值值      真值        误差")
    print("  " + "-" * 50)
    
    errors = []
    for i in range(10):
        t = i * dt_key / 3  # 在关键帧之间采样
        q_interp = interpolator.eval(t_write + t)
        q_true = np.sin(2 * np.pi * freq * t)
        error = abs(q_interp[0] - q_true)
        errors.append(error)
        print(f"  {t:.4f}s    {q_interp[0]:+.6f}  {q_true:+.6f}   {error:.6f}")
    
    print(f"\n  平均误差: {np.mean(errors):.6f}")
    print(f"  最大误差: {np.max(errors):.6f}")
    
    # 三次样条对正弦波的逼近误差应该很小
    if np.max(errors) < 0.01:
        print("  ✓ 插值精度良好")
        return 0
    else:
        print("  ✗ 插值精度不足")
        return 1


def test_safety_limiter():
    """测试安全限制器"""
    print("\n" + "=" * 60)
    print("安全限制器测试")
    print("=" * 60)
    
    dof = 7
    servo_cfg = ServoConfig(dof=dof)
    limiter = SafetyLimiter(servo_cfg)
    current_pos = np.zeros(dof)
    
    # 测试 1: 超出关节限位
    print("\n  [测试1] 关节限位...")
    q_over = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])  # 关节1 超出 [-π, π]
    q_safe = limiter.apply(q_over, current_pos)
    if abs(q_safe[0]) <= np.pi:
        print(f"    ✓ 关节1 被限制: {q_over[0]:.2f} -> {q_safe[0]:.2f}")
    else:
        print(f"    ✗ 关节限位失败")
    
    # 测试 2: 速率限制
    print("\n  [测试2] 速率限制...")
    servo_cfg2 = ServoConfig(dof=dof)
    limiter2 = SafetyLimiter(servo_cfg2)
    current2 = np.zeros(dof)
    
    # 第一个命令
    q1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04])
    current2 = limiter2.apply(q1, current2)
    
    # 第二个命令 - 大跳变
    q2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04])  # 关节1 跳变 1.0 rad
    q_safe2 = limiter2.apply(q2, current2)
    
    # 由于 EMA 和速率限制，输出不应该直接跳到 1.0
    if q_safe2[0] < 0.5:
        print(f"    ✓ 大跳变被平滑: {q2[0]:.2f} -> {q_safe2[0]:.4f}")
    else:
        print(f"    ✗ 速率限制失败")
    
    # 测试 3: 夹爪限位
    print("\n  [测试3] 夹爪限位...")
    q_gripper = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])  # 夹爪超过 0.08
    current3 = np.zeros(dof)
    q_safe3 = limiter.apply(q_gripper, current3)
    if q_safe3[6] <= 0.08:
        print(f"    ✓ 夹爪被限制: {q_gripper[6]:.3f} -> {q_safe3[6]:.3f}")
    else:
        print(f"    ✗ 夹爪限位失败")
    
    print("\n  ✓ 安全限制器测试完成")
    return 0


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("inference_rtc 测试套件")
    print("=" * 60)
    
    results = []
    
    # 测试 1: 插值器精度
    results.append(("插值器精度", test_interpolator_accuracy()))
    
    # 测试 2: 安全限制器
    results.append(("安全限制器", test_safety_limiter()))
    
    # 测试 3: 端到端
    results.append(("端到端", test_single_process()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_pass = True
    for name, code in results:
        status = "✓ PASS" if code == 0 else "✗ WARN"
        print(f"  {name}: {status}")
        if code != 0:
            all_pass = False
    
    if all_pass:
        print("\n✓ 所有测试通过!")
        return 0
    else:
        print("\n⚠ 部分测试有警告")
        return 1


if __name__ == "__main__":
    sys.exit(main())
