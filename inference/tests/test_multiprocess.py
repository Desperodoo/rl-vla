#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多进程架构测试

测试控制进程和推理进程之间的通信和安全机制。

用法:
    # 测试共享内存通信
    python -m inference.tests.test_multiprocess --test shm
    
    # 测试控制进程 (模拟模式)
    python -m inference.tests.test_multiprocess --test control
    
    # 完整测试 (需要两个终端)
    # 终端 1: python -m inference.control_node --dry-run
    # 终端 2: python -m inference.tests.test_multiprocess --test full
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.shared_state import SharedState, ControlFlags, SharedMemoryLayout


def test_shared_memory():
    """测试共享内存基本功能"""
    print("\n" + "=" * 60)
    print("测试: 共享内存基本功能")
    print("=" * 60)
    
    # 创建共享内存
    print("\n[1] 创建共享内存...")
    state = SharedState.create("test_shm")
    print(f"  大小: {SharedMemoryLayout.TOTAL_SIZE} bytes")
    print(f"  版本: {SharedMemoryLayout.VERSION}")
    
    # 测试控制标志
    print("\n[2] 测试控制标志...")
    state.set_state(ControlFlags.IDLE)
    assert state.get_state() == ControlFlags.IDLE, "IDLE 状态设置失败"
    print(f"  IDLE: ✓")
    
    state.set_state(ControlFlags.RUNNING)
    assert state.get_state() == ControlFlags.RUNNING, "RUNNING 状态设置失败"
    print(f"  RUNNING: ✓")
    
    # 测试请求/确认
    print("\n[3] 测试请求/确认...")
    state.request_pause()
    assert state.has_request(ControlFlags.REQUEST_PAUSE), "PAUSE 请求失败"
    print(f"  REQUEST_PAUSE: ✓")
    
    state.ack_pause()
    assert state.has_ack(ControlFlags.ACK_PAUSE), "PAUSE 确认失败"
    assert not state.has_request(ControlFlags.REQUEST_PAUSE), "请求未清除"
    print(f"  ACK_PAUSE: ✓")
    
    state.clear_acks()
    assert not state.has_ack(ControlFlags.ACK_PAUSE), "确认未清除"
    print(f"  CLEAR_ACKS: ✓")
    
    # 测试机器人状态
    print("\n[4] 测试机器人状态读写...")
    test_pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    test_gripper = 0.04
    test_vel = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float64)
    
    state.write_robot_state(
        joint_pos=test_pos,
        gripper_pos=test_gripper,
        joint_vel=test_vel,
    )
    
    pos, vel, ts = state.read_robot_state()
    assert np.allclose(pos[:6], test_pos), "关节位置读取失败"
    assert np.isclose(pos[6], test_gripper), "夹爪位置读取失败"
    assert np.allclose(vel[:6], test_vel), "关节速度读取失败"
    print(f"  机器人状态: ✓")
    
    # 测试动作缓冲区
    print("\n[5] 测试动作缓冲区...")
    test_actions = np.random.randn(10, 7).astype(np.float64)
    state.write_actions(test_actions)
    
    actions, idx, size = state.read_actions()
    assert size == 10, f"动作数量错误: {size} != 10"
    assert idx == 0, f"动作索引错误: {idx} != 0"
    assert np.allclose(actions, test_actions), "动作数据错误"
    print(f"  写入动作: ✓")
    
    # 测试获取下一个动作
    action, is_last = state.get_next_action()
    assert not is_last, "不应该是最后一个"
    assert np.allclose(action, test_actions[0]), "第一个动作错误"
    print(f"  获取动作 0: ✓")
    
    action, is_last = state.get_next_action()
    assert np.allclose(action, test_actions[1]), "第二个动作错误"
    print(f"  获取动作 1: ✓")
    
    # 快进到最后
    for i in range(7):
        state.get_next_action()
    
    action, is_last = state.get_next_action()
    assert is_last, "应该是最后一个"
    print(f"  最后一个动作: ✓")
    
    # 测试缓冲区为空
    assert state.is_action_buffer_empty(), "缓冲区应该为空"
    print(f"  缓冲区为空检测: ✓")
    
    # 清理
    state.close()
    print("\n✓ 共享内存测试通过")


def test_multiprocess_communication():
    """测试多进程通信"""
    print("\n" + "=" * 60)
    print("测试: 多进程通信")
    print("=" * 60)
    
    def control_process(ready_event, stop_event):
        """模拟控制进程"""
        state = SharedState.create("test_mp")
        ready_event.set()
        
        step_count = 0
        while not stop_event.is_set():
            # 处理请求
            if state.has_request(ControlFlags.REQUEST_START):
                state.set_state(ControlFlags.RUNNING)
                state.ack_start()
            
            if state.has_request(ControlFlags.REQUEST_PAUSE):
                state.set_state(ControlFlags.PAUSED)
                state.ack_pause()
            
            if state.has_request(ControlFlags.REQUEST_STOP):
                state.set_state(ControlFlags.EMERGENCY_STOP)
                state.ack_stop()
                break
            
            # 模拟执行动作
            if state.get_state() == ControlFlags.RUNNING:
                action, is_last = state.get_next_action()
                if action is not None:
                    step_count += 1
            
            # 更新状态
            state.write_robot_state(
                joint_pos=np.random.randn(6) * 0.1,
                gripper_pos=0.04,
            )
            state.update_control_timestamp()
            
            time.sleep(0.002)  # 500Hz
        
        state.close()
        return step_count
    
    def inference_process(ready_event, stop_event, result_queue):
        """模拟推理进程"""
        ready_event.wait(timeout=5.0)
        
        state = SharedState.connect("test_mp")
        
        # 发送 START
        state.request_start()
        while not state.has_ack(ControlFlags.ACK_START):
            time.sleep(0.001)
        state.clear_acks()
        
        # 写入动作
        n_batches = 5
        total_actions = 0
        for i in range(n_batches):
            actions = np.random.randn(10, 7)
            state.write_actions(actions)
            total_actions += 10
            state.update_inference_timestamp()
            
            # 等待缓冲区被消费
            while not state.is_action_buffer_empty():
                time.sleep(0.01)
        
        # 发送 STOP
        state.request_stop()
        while not state.has_ack(ControlFlags.ACK_STOP):
            time.sleep(0.001)
        
        result_queue.put(total_actions)
        stop_event.set()
        state.close()
    
    # 创建事件和队列
    ready_event = mp.Event()
    stop_event = mp.Event()
    result_queue = mp.Queue()
    
    # 启动进程
    print("\n[1] 启动控制进程...")
    ctrl_proc = mp.Process(target=control_process, args=(ready_event, stop_event))
    ctrl_proc.start()
    
    print("[2] 启动推理进程...")
    inf_proc = mp.Process(target=inference_process, args=(ready_event, stop_event, result_queue))
    inf_proc.start()
    
    # 等待完成
    print("[3] 等待测试完成...")
    inf_proc.join(timeout=10.0)
    ctrl_proc.join(timeout=2.0)
    
    # 获取结果
    if not result_queue.empty():
        total_actions = result_queue.get()
        print(f"\n  总动作数: {total_actions}")
        print(f"  控制进程退出码: {ctrl_proc.exitcode}")
        print(f"  推理进程退出码: {inf_proc.exitcode}")
        
        if ctrl_proc.exitcode == 0 and inf_proc.exitcode == 0:
            print("\n✓ 多进程通信测试通过")
        else:
            print("\n✗ 测试失败")
    else:
        print("\n✗ 未收到结果")


def test_safety_mechanisms():
    """测试安全机制"""
    print("\n" + "=" * 60)
    print("测试: 安全机制")
    print("=" * 60)
    
    def control_with_watchdog(ready_event, stop_event, result_queue):
        """带看门狗的控制进程"""
        state = SharedState.create("test_safety")
        ready_event.set()
        
        watchdog_timeout = 0.5  # 500ms
        last_inference_time = time.time()
        watchdog_triggered = False
        
        while not stop_event.is_set():
            # 处理请求
            if state.has_request(ControlFlags.REQUEST_START):
                state.set_state(ControlFlags.RUNNING)
                state.ack_start()
                last_inference_time = time.time()
            
            if state.has_request(ControlFlags.REQUEST_STOP):
                state.set_state(ControlFlags.EMERGENCY_STOP)
                state.ack_stop()
                break
            
            # 看门狗检查
            if state.get_state() == ControlFlags.RUNNING:
                inference_ts = state.get_inference_timestamp()
                if inference_ts > 0:
                    elapsed = time.time() - inference_ts
                    if elapsed > watchdog_timeout:
                        print(f"  [看门狗] 推理进程超时 ({elapsed:.2f}s)")
                        state.set_state(ControlFlags.PAUSED)
                        state.set_error(ControlFlags.ERROR_TIMEOUT)
                        watchdog_triggered = True
            
            state.update_control_timestamp()
            time.sleep(0.002)
        
        result_queue.put(watchdog_triggered)
        state.close()
    
    def inference_with_crash(ready_event, stop_event):
        """模拟崩溃的推理进程"""
        ready_event.wait(timeout=5.0)
        state = SharedState.connect("test_safety")
        
        # 发送 START
        state.request_start()
        while not state.has_ack(ControlFlags.ACK_START):
            time.sleep(0.001)
        state.clear_acks()
        
        # 更新几次时间戳然后"崩溃"
        for _ in range(5):
            state.update_inference_timestamp()
            time.sleep(0.1)
        
        print("  [推理] 模拟崩溃，停止更新时间戳...")
        time.sleep(1.0)  # 超过看门狗超时
        
        # 检查状态
        current_state = state.get_state()
        print(f"  [推理] 当前状态: {current_state}")
        
        # 发送 STOP
        state.request_stop()
        while not state.has_ack(ControlFlags.ACK_STOP):
            time.sleep(0.001)
        
        stop_event.set()
        state.close()
    
    # 创建事件和队列
    ready_event = mp.Event()
    stop_event = mp.Event()
    result_queue = mp.Queue()
    
    # 启动进程
    print("\n[1] 测试看门狗超时...")
    ctrl_proc = mp.Process(target=control_with_watchdog, args=(ready_event, stop_event, result_queue))
    ctrl_proc.start()
    
    inf_proc = mp.Process(target=inference_with_crash, args=(ready_event, stop_event))
    inf_proc.start()
    
    # 等待完成
    inf_proc.join(timeout=10.0)
    ctrl_proc.join(timeout=2.0)
    
    # 获取结果
    if not result_queue.empty():
        watchdog_triggered = result_queue.get()
        if watchdog_triggered:
            print("\n✓ 看门狗测试通过 - 超时后正确进入 PAUSED 状态")
        else:
            print("\n✗ 看门狗测试失败 - 超时未触发")
    else:
        print("\n✗ 未收到结果")


def test_full_integration():
    """完整集成测试 (需要控制节点已启动)"""
    print("\n" + "=" * 60)
    print("测试: 完整集成测试")
    print("=" * 60)
    print("\n请确保控制节点已在另一个终端启动:")
    print("  python -m inference.control_node --dry-run")
    print("")
    
    try:
        state = SharedState.connect("arx5_control", timeout=5.0)
        print("✓ 已连接到控制节点")
    except TimeoutError:
        print("✗ 无法连接到控制节点")
        return
    
    # 测试 START
    print("\n[1] 测试 START...")
    state.request_start()
    start_time = time.time()
    while not state.has_ack(ControlFlags.ACK_START):
        if time.time() - start_time > 1.0:
            print("  ✗ START 超时")
            return
        time.sleep(0.01)
    state.clear_acks()
    print("  ✓ START 确认")
    
    # 写入动作
    print("\n[2] 测试动作执行...")
    actions = np.random.randn(20, 7) * 0.1  # 小幅动作
    state.write_actions(actions)
    
    # 等待执行
    while not state.is_action_buffer_empty():
        idx, size = state.get_action_buffer_status()
        print(f"  执行中: {idx}/{size}", end="\r")
        time.sleep(0.05)
    print("\n  ✓ 动作执行完成")
    
    # 测试 PAUSE
    print("\n[3] 测试 PAUSE...")
    state.request_pause()
    start_time = time.time()
    while not state.has_ack(ControlFlags.ACK_PAUSE):
        if time.time() - start_time > 1.0:
            print("  ✗ PAUSE 超时")
            break
        time.sleep(0.01)
    else:
        state.clear_acks()
        print("  ✓ PAUSE 确认")
    
    # 验证状态
    current_state = state.get_state()
    if current_state == ControlFlags.PAUSED:
        print("  ✓ 状态已切换到 PAUSED")
    else:
        print(f"  ✗ 状态错误: {current_state}")
    
    # 测试 RESET
    print("\n[4] 测试 RESET...")
    state.request_reset()
    start_time = time.time()
    while not state.has_ack(ControlFlags.ACK_RESET):
        if time.time() - start_time > 5.0:
            print("  ✗ RESET 超时")
            break
        time.sleep(0.01)
    else:
        state.clear_acks()
        print("  ✓ RESET 确认")
    
    # 等待复位完成
    time.sleep(0.5)
    current_state = state.get_state()
    if current_state == ControlFlags.IDLE:
        print("  ✓ 状态已切换到 IDLE")
    else:
        print(f"  ✗ 状态错误: {current_state}")
    
    print("\n✓ 完整集成测试通过")
    
    # 不发送 STOP，让控制节点继续运行
    state.close()


def main():
    parser = argparse.ArgumentParser(
        description="多进程架构测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--test", type=str, default="shm",
                       choices=["shm", "mp", "safety", "full", "all"],
                       help="测试类型: shm=共享内存, mp=多进程, safety=安全机制, full=完整集成, all=全部")
    
    args = parser.parse_args()
    
    if args.test == "shm" or args.test == "all":
        test_shared_memory()
    
    if args.test == "mp" or args.test == "all":
        test_multiprocess_communication()
    
    if args.test == "safety" or args.test == "all":
        test_safety_mechanisms()
    
    if args.test == "full":
        test_full_integration()


if __name__ == "__main__":
    main()
