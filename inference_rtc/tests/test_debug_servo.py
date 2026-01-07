#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试版 Servo 测试

记录所有数据用于分析：
1. 关键帧数据 (从共享内存读取)
2. 插值输出
3. 安全限制后的命令
4. 实际机械臂状态

运行:
    # 终端 1: 启动 inference
    python -m inference_rtc.python.inference_main --auto-start -c <checkpoint>
    
    # 终端 2: 启动调试 servo (记录 5 秒数据)
    python -m inference_rtc.tests.test_debug_servo --duration 5
"""

import os
import sys
import time
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from scipy.interpolate import CubicSpline

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.config import setup_arx5


@dataclass
class DebugRecord:
    """调试记录"""
    timestamp: float
    
    # 关键帧数据
    keyframe_version: int = 0
    keyframe_t_write: float = 0.0
    keyframe_data: np.ndarray = None  # [H, dof]
    
    # 插值数据
    interp_t_query: float = 0.0
    interp_output: np.ndarray = None  # [dof]
    interp_valid: bool = False
    
    # 安全限制后
    safe_output: np.ndarray = None  # [dof]
    
    # 实际状态
    actual_pos: np.ndarray = None  # [dof]
    
    # 误差
    cmd_vs_actual_error: float = 0.0


class DebugServo:
    """调试版 Servo"""
    
    def __init__(self, duration: float = 5.0, dry_run: bool = False):
        self.duration = duration
        self.dry_run = dry_run
        
        self.robot = None
        self.arx5 = None
        self.shm = None
        self.buf = None
        
        self.records: List[DebugRecord] = []
        self.keyframes_received: List[Dict] = []
        
        # 插值器状态
        self._splines = []
        self._t0 = 0.0
        self._t_end = 0.0
        self._valid = False
        self._dof = 7
        
        # 当前位置
        self._current_pos = np.zeros(7)
        
    def connect_shm(self) -> bool:
        """连接共享内存"""
        from multiprocessing import shared_memory
        
        print("[1] 连接共享内存...")
        try:
            self.shm = shared_memory.SharedMemory(name="rtc_keyframes")
            self.buf = np.ndarray((self.shm.size,), dtype=np.uint8, buffer=self.shm.buf)
            print(f"  ✓ 连接成功, size={self.shm.size}")
            return True
        except FileNotFoundError:
            print("  ✗ 共享内存不存在，请先启动 inference")
            return False
    
    def connect_robot(self) -> bool:
        """连接机械臂"""
        print("\n[2] 连接机械臂...")
        
        if self.dry_run:
            print("  [模拟模式]")
            return True
        
        try:
            setup_arx5()
            import arx5_interface as arx5
            self.arx5 = arx5
            
            robot_config = arx5.RobotConfigFactory.get_instance().get_config("X5")
            controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", robot_config.joint_dof
            )
            
            self.robot = arx5.Arx5JointController(
                robot_config, controller_config, "can0"
            )
            self.robot.set_log_level(arx5.LogLevel.INFO)
            self.robot.enable_background_send_recv()
            
            print("  正在复位...")
            self.robot.reset_to_home()
            
            # 获取初始状态
            state = self.robot.get_state()
            self._current_pos[:6] = np.array(state.pos())
            self._current_pos[6] = state.gripper_pos
            
            print(f"  初始位置: {self._current_pos[:3]}...")
            print("  ✓ 机械臂连接成功")
            return True
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def read_keyframes(self) -> Optional[Dict]:
        """读取关键帧"""
        import struct
        
        header = bytes(self.buf[:32])
        version = struct.unpack_from('<Q', header, 0)[0]
        
        if not hasattr(self, '_last_version'):
            self._last_version = 0
        
        if version == self._last_version or version == 0:
            return None
        
        t_write = struct.unpack_from('<d', header, 8)[0]
        dof = struct.unpack_from('<i', header, 16)[0]
        H = struct.unpack_from('<i', header, 20)[0]
        dt_key = struct.unpack_from('<d', header, 24)[0]
        
        payload_size = H * dof * 8
        q_key_bytes = bytes(self.buf[32:32 + payload_size])
        q_key = np.frombuffer(q_key_bytes, dtype=np.float64).reshape(H, dof).copy()
        
        self._last_version = version
        
        return {
            'version': version,
            't_write': t_write,
            'dof': dof,
            'H': H,
            'dt_key': dt_key,
            'q_key': q_key,
        }
    
    def build_interpolator(self, q_key: np.ndarray, dt_key: float, t0: float):
        """构建插值器"""
        H, dof = q_key.shape
        if H < 2:
            return
        
        self._dof = dof
        self._t0 = t0
        self._t_end = t0 + (H - 1) * dt_key
        
        t = np.arange(H) * dt_key + t0
        
        self._splines = []
        for j in range(dof):
            cs = CubicSpline(t, q_key[:, j], bc_type='clamped')
            self._splines.append(cs)
        
        self._valid = True
        self._last_q_key = q_key.copy()
    
    def eval_interpolator(self, t_query: float) -> np.ndarray:
        """采样插值器"""
        if not self._valid:
            return self._current_pos.copy()
        
        t_clipped = np.clip(t_query, self._t0, self._t_end)
        
        q = np.zeros(self._dof)
        for j, cs in enumerate(self._splines):
            q[j] = cs(t_clipped)
        
        return q
    
    def apply_safety(self, target: np.ndarray, current: np.ndarray, 
                     alpha: float = 0.3, max_delta: float = 0.1) -> np.ndarray:
        """安全限制"""
        # EMA 滤波
        filtered = alpha * target + (1 - alpha) * current
        
        # 速率限制
        delta = filtered - current
        delta_norm = np.linalg.norm(delta[:6])
        if delta_norm > max_delta:
            delta[:6] = delta[:6] / delta_norm * max_delta
        
        safe = current + delta
        return safe
    
    def run(self):
        """运行调试循环"""
        print(f"\n[3] 开始调试循环 ({self.duration}s)...")
        print("  按 Ctrl+C 提前结束\n")
        
        start_time = time.time()
        loop_count = 0
        update_count = 0
        
        try:
            while time.time() - start_time < self.duration:
                t_now = time.clock_gettime(time.CLOCK_MONOTONIC)
                
                record = DebugRecord(timestamp=time.time() - start_time)
                
                # 1. 读取关键帧
                data = self.read_keyframes()
                if data is not None:
                    record.keyframe_version = data['version']
                    record.keyframe_t_write = data['t_write']
                    record.keyframe_data = data['q_key'].copy()
                    
                    self.keyframes_received.append({
                        'version': data['version'],
                        't_write': data['t_write'],
                        'q_key': data['q_key'].copy(),
                        'delay_ms': (t_now - data['t_write']) * 1000,
                    })
                    
                    self.build_interpolator(data['q_key'], data['dt_key'], data['t_write'])
                    update_count += 1
                
                # 2. 插值
                record.interp_t_query = t_now
                record.interp_valid = self._valid
                
                if self._valid:
                    interp_out = self.eval_interpolator(t_now)
                    record.interp_output = interp_out.copy()
                    
                    # 3. 安全限制
                    safe_out = self.apply_safety(interp_out, self._current_pos)
                    record.safe_output = safe_out.copy()
                    
                    # 4. 发送命令
                    if not self.dry_run and self.robot is not None:
                        cmd = self.arx5.JointState(6)
                        cmd.pos()[:] = safe_out[:6]
                        cmd.gripper_pos = float(safe_out[6])
                        self.robot.set_joint_cmd(cmd)
                        
                        # 读取实际状态
                        state = self.robot.get_state()
                        record.actual_pos = np.zeros(7)
                        record.actual_pos[:6] = np.array(state.pos())
                        record.actual_pos[6] = state.gripper_pos
                        
                        record.cmd_vs_actual_error = np.linalg.norm(
                            safe_out[:6] - record.actual_pos[:6]
                        )
                        
                        self._current_pos = record.actual_pos.copy()
                    else:
                        self._current_pos = safe_out.copy()
                
                self.records.append(record)
                loop_count += 1
                
                # 打印进度
                if loop_count % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"  loops={loop_count}, updates={update_count}, "
                          f"elapsed={elapsed:.1f}s")
                
                # 频率控制 (500Hz)
                time.sleep(0.002)
        
        except KeyboardInterrupt:
            print("\n  中断")
        
        print(f"\n  完成: {loop_count} loops, {update_count} updates")
    
    def analyze(self):
        """分析数据"""
        print("\n" + "=" * 60)
        print("数据分析")
        print("=" * 60)
        
        if not self.records:
            print("无数据")
            return
        
        # 1. 关键帧分析
        print("\n[关键帧分析]")
        if self.keyframes_received:
            print(f"  收到 {len(self.keyframes_received)} 个关键帧")
            
            # 打印前几个关键帧的第一帧
            print("\n  前 5 个关键帧的第一帧动作:")
            for i, kf in enumerate(self.keyframes_received[:5]):
                q0 = kf['q_key'][0]
                print(f"    v={kf['version']:3d}: {q0[:3]}... delay={kf['delay_ms']:.1f}ms")
            
            # 检查关键帧变化
            print("\n  关键帧变化检测:")
            for i in range(1, min(5, len(self.keyframes_received))):
                prev = self.keyframes_received[i-1]['q_key']
                curr = self.keyframes_received[i]['q_key']
                diff = np.linalg.norm(curr - prev)
                print(f"    v{self.keyframes_received[i-1]['version']} -> "
                      f"v{self.keyframes_received[i]['version']}: diff={diff:.4f}")
        
        # 2. 插值输出分析
        print("\n[插值输出分析]")
        interp_outputs = [r.interp_output for r in self.records if r.interp_output is not None]
        if interp_outputs:
            interp_arr = np.array(interp_outputs)
            print(f"  样本数: {len(interp_arr)}")
            print(f"  均值: {interp_arr.mean(axis=0)[:3]}...")
            print(f"  标准差: {interp_arr.std(axis=0)[:3]}...")
            print(f"  范围: [{interp_arr.min(axis=0)[:3]}...] ~ [{interp_arr.max(axis=0)[:3]}...]")
            
            # 相邻帧差异
            diffs = np.diff(interp_arr, axis=0)
            diff_norms = np.linalg.norm(diffs[:, :6], axis=1)
            print(f"\n  相邻帧差异 (关节):")
            print(f"    Mean: {diff_norms.mean():.6f} rad")
            print(f"    Max:  {diff_norms.max():.6f} rad")
            print(f"    Std:  {diff_norms.std():.6f} rad")
        
        # 3. 安全限制分析
        print("\n[安全限制分析]")
        safe_outputs = [r.safe_output for r in self.records if r.safe_output is not None]
        if safe_outputs:
            safe_arr = np.array(safe_outputs)
            
            # 与插值输出的差异
            if interp_outputs and len(interp_arr) == len(safe_arr):
                safety_diff = np.linalg.norm(interp_arr - safe_arr, axis=1)
                print(f"  安全限制修正量:")
                print(f"    Mean: {safety_diff.mean():.6f}")
                print(f"    Max:  {safety_diff.max():.6f}")
        
        # 4. 命令 vs 实际
        print("\n[命令 vs 实际]")
        errors = [r.cmd_vs_actual_error for r in self.records if r.cmd_vs_actual_error > 0]
        if errors:
            errors = np.array(errors)
            print(f"  跟踪误差:")
            print(f"    Mean: {errors.mean():.4f} rad")
            print(f"    Max:  {errors.max():.4f} rad")
    
    def save_data(self, filename: str = None):
        """保存数据"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"/home/lizh/rl-vla/inference_rtc/tests/debug_data_{timestamp}.npz"
        
        print(f"\n保存数据到: {filename}")
        
        # 收集数据
        timestamps = [r.timestamp for r in self.records]
        
        interp_outputs = []
        safe_outputs = []
        actual_poses = []
        
        for r in self.records:
            if r.interp_output is not None:
                interp_outputs.append(r.interp_output)
            if r.safe_output is not None:
                safe_outputs.append(r.safe_output)
            if r.actual_pos is not None:
                actual_poses.append(r.actual_pos)
        
        keyframe_q_keys = [kf['q_key'] for kf in self.keyframes_received]
        keyframe_versions = [kf['version'] for kf in self.keyframes_received]
        keyframe_delays = [kf['delay_ms'] for kf in self.keyframes_received]
        
        np.savez(
            filename,
            timestamps=np.array(timestamps),
            interp_outputs=np.array(interp_outputs) if interp_outputs else np.array([]),
            safe_outputs=np.array(safe_outputs) if safe_outputs else np.array([]),
            actual_poses=np.array(actual_poses) if actual_poses else np.array([]),
            keyframe_q_keys=np.array(keyframe_q_keys) if keyframe_q_keys else np.array([]),
            keyframe_versions=np.array(keyframe_versions),
            keyframe_delays=np.array(keyframe_delays),
        )
        
        print(f"  ✓ 数据已保存")
        return filename
    
    def cleanup(self):
        """清理"""
        print("\n[清理]")
        if self.robot is not None:
            print("  设置阻尼模式...")
            self.robot.set_to_damping()
        if self.shm is not None:
            self.shm.close()
        print("  ✓ 清理完成")


def compare_with_demo(debug_file: str, demo_file: str):
    """与 demo 数据比较"""
    print("\n" + "=" * 60)
    print("与 Demo 数据比较")
    print("=" * 60)
    
    # 加载调试数据
    debug_data = np.load(debug_file)
    
    # 加载 demo 数据
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from inference.config import setup_rlft
    setup_rlft()
    from diffusion_policy.utils import load_traj_hdf5
    
    demo_file = os.path.expanduser(demo_file)
    raw_data = load_traj_hdf5(demo_file, num_traj=1)
    demo_actions = raw_data['traj_0']['actions']
    
    print(f"\n[Demo 数据]")
    print(f"  轨迹长度: {len(demo_actions)}")
    print(f"  动作范围:")
    print(f"    关节: [{demo_actions[:, :6].min():.3f}, {demo_actions[:, :6].max():.3f}]")
    print(f"    夹爪: [{demo_actions[:, 6].min():.3f}, {demo_actions[:, 6].max():.3f}]")
    print(f"  前 5 帧:")
    for i in range(min(5, len(demo_actions))):
        print(f"    [{i}]: {demo_actions[i, :3]}...")
    
    print(f"\n[推理输出]")
    keyframes = debug_data['keyframe_q_keys']
    if len(keyframes) > 0:
        all_keyframe_actions = keyframes.reshape(-1, keyframes.shape[-1])
        print(f"  关键帧数: {len(keyframes)}")
        print(f"  动作范围:")
        print(f"    关节: [{all_keyframe_actions[:, :6].min():.3f}, {all_keyframe_actions[:, :6].max():.3f}]")
        print(f"    夹爪: [{all_keyframe_actions[:, 6].min():.3f}, {all_keyframe_actions[:, 6].max():.3f}]")
        print(f"  前 5 个关键帧的第一帧:")
        for i in range(min(5, len(keyframes))):
            print(f"    [v={debug_data['keyframe_versions'][i]}]: {keyframes[i, 0, :3]}...")
    
    print(f"\n[对比分析]")
    print(f"  Demo 动作均值: {demo_actions.mean(axis=0)[:3]}...")
    if len(keyframes) > 0:
        print(f"  推理输出均值: {all_keyframe_actions.mean(axis=0)[:3]}...")
        
        # 检查是否输出接近零
        infer_mean_norm = np.linalg.norm(all_keyframe_actions.mean(axis=0)[:6])
        demo_mean_norm = np.linalg.norm(demo_actions.mean(axis=0)[:6])
        print(f"\n  Demo 均值范数: {demo_mean_norm:.4f}")
        print(f"  推理均值范数: {infer_mean_norm:.4f}")
        
        if infer_mean_norm < 0.1:
            print(f"\n  ⚠ 警告: 推理输出接近零! 策略可能没有正确加载或输入有问题")


def main():
    parser = argparse.ArgumentParser(description="调试版 Servo 测试")
    parser.add_argument("-d", "--duration", type=float, default=5.0, help="运行时长 (秒)")
    parser.add_argument("--dry-run", action="store_true", help="模拟模式")
    parser.add_argument("--compare", type=str, default=None, help="与 demo 比较")
    parser.add_argument("--load", type=str, default=None, help="加载已有数据分析")
    
    args = parser.parse_args()
    
    # 如果只是加载分析
    if args.load:
        if args.compare:
            compare_with_demo(args.load, args.compare)
        else:
            data = np.load(args.load)
            print("加载数据:")
            for key in data.files:
                print(f"  {key}: {data[key].shape}")
        return 0
    
    print("=" * 60)
    print("调试版 Servo 测试")
    print("=" * 60)
    
    tester = DebugServo(duration=args.duration, dry_run=args.dry_run)
    
    try:
        if not tester.connect_shm():
            return 1
        
        if not tester.connect_robot():
            return 1
        
        tester.run()
        tester.analyze()
        
        debug_file = tester.save_data()
        
        # 如果指定了 demo 比较
        if args.compare:
            compare_with_demo(debug_file, args.compare)
        
    except KeyboardInterrupt:
        print("\n中断")
    
    finally:
        tester.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
