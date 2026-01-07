#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTC 推理主程序

10Hz 推理循环:
1. 从相机采集 RGB (复用 CameraManager)
2. 从控制进程读取机器人状态 (复用 SharedState)
3. 策略推理
4. 将关键帧写入共享内存 (供 C++ 伺服进程读取)
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
from typing import Optional

import cv2

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference_rtc.python.config import InferenceConfig
from inference_rtc.python.policy_runner import PolicyRunner
from inference_rtc.shared.shm_protocol import ShmKeyframeWriter

# 复用现有模块
from inference.camera_manager import CameraManager, DEFAULT_CAMERA_CONFIGS
from inference.shared_state import SharedState, ControlFlags


class RTCInferenceNode:
    """
    RTC 推理节点
    
    10Hz 策略推理，输出关键帧到共享内存
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: InferenceConfig = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.config = config or InferenceConfig()
        
        # 策略推理器
        self.policy_runner: Optional[PolicyRunner] = None
        
        # 共享内存 (关键帧输出)
        self.shm_writer: Optional[ShmKeyframeWriter] = None
        
        # 机器人状态 (从控制进程读取)
        self.robot_state: Optional[SharedState] = None
        
        # 相机
        self.camera_manager: Optional[CameraManager] = None
        
        # 状态
        self._running = False
        self._is_inferencing = False
    
    def setup(self):
        """初始化"""
        print("\n" + "=" * 60)
        print("RTC 推理节点")
        print("=" * 60)
        
        # 1. 加载策略
        self.policy_runner = PolicyRunner(
            checkpoint_path=self.checkpoint_path,
            config=self.config,
            device=self.config.device,
            use_ema=self.config.use_ema,
            verbose=self.config.verbose,
        )
        self.policy_runner.load_model()
        self.policy_runner.warmup()
        
        # 2. 创建共享内存 (关键帧输出)
        print("\n创建共享内存 (关键帧输出)...")
        self.shm_writer = ShmKeyframeWriter.create(
            name=self.config.shm_name,
            H=self.config.act_horizon,
            dof=self.config.dof,
            dt_key=self.config.dt_key,
        )
        
        # 3. 连接控制进程 (读取机器人状态)
        print("\n连接控制进程 (等待 servo 启动)...")
        if self.config.dry_run:
            print("  [模拟模式] 创建本地共享内存")
            self.robot_state = SharedState.create(self.config.control_shm_name)
        else:
            # 等待 servo 进程创建控制共享内存
            print("  等待 servo 进程...")
            try:
                self.robot_state = SharedState.connect(
                    self.config.control_shm_name, timeout=30.0  # 增加超时时间
                )
                print(f"  已连接到控制进程")
            except TimeoutError:
                print("  [警告] 无法连接到控制进程，将使用零状态")
                self.robot_state = None
        
        # 4. 初始化相机
        print("\n初始化相机...")
        try:
            self.camera_manager = CameraManager(DEFAULT_CAMERA_CONFIGS)
            self.camera_manager.initialize(auto_assign=True)
            self.camera_manager.start()
            print("  相机初始化完成")
        except Exception as e:
            print(f"  [警告] 相机初始化失败: {e}")
            print("  将使用零图像")
            self.camera_manager = None
        
        print("\n✓ RTC 推理节点初始化完成")
        print(f"  策略频率: {self.config.policy_rate} Hz")
        print(f"  关键帧数: {self.config.act_horizon}")
        print(f"  关键帧间隔: {self.config.dt_key*1000:.1f} ms")
        print(f"  规划时域: {self.config.planning_horizon*1000:.1f} ms")
    
    def run(self):
        """主推理循环"""
        print("\n" + "=" * 60)
        print("RTC 推理循环启动 (@10Hz)")
        print("=" * 60)
        
        # 检查是否可以使用 GUI
        # 可以通过设置 RTC_HEADLESS=1 环境变量强制 headless 模式
        import os
        if os.environ.get('RTC_HEADLESS', '').lower() in ('1', 'true', 'yes'):
            print("  [环境变量] RTC_HEADLESS=1, 强制 headless 模式")
            self.config.headless = True
        
        gui_available = False
        if not self.config.headless:
            # 检查 DISPLAY 环境变量
            display = os.environ.get('DISPLAY')
            if not display:
                print("  [警告] DISPLAY 环境变量未设置，切换到 headless 模式")
                self.config.headless = True
            else:
                # 假设 GUI 可用，如果崩溃请使用 --headless 或 RTC_HEADLESS=1
                gui_available = True
        
        if gui_available:
            cv2.namedWindow("RTC Inference", cv2.WINDOW_NORMAL)
            print("  [OpenCV 窗口]")
            print("    [Space] - 开始/暂停")
            print("    [R]     - 复位")
            print("    [Q]     - 退出")
        else:
            self.config.headless = True
        
        if self.config.headless:
            print("  [Headless 模式]")
            print("    按 Ctrl+C 退出")
        
        self._running = True
        
        # 自动开始推理 (headless 模式或 auto_start)
        if self.config.headless or self.config.auto_start:
            self._is_inferencing = True
            print("  → 自动开始推理")
        else:
            self._is_inferencing = False
        
        policy_dt = 1.0 / self.config.policy_rate
        last_inference_time = 0.0
        loop_count = 0
        
        try:
            while self._running:
                loop_start = time.time()
                
                # 1. 获取相机帧
                rgb = self._get_camera_frame()
                
                # 2. 获取机器人状态
                robot_pos, robot_vel = self._get_robot_state()
                
                # 3. 构建观测并添加到缓冲区
                state = self._build_state(robot_pos, robot_vel)
                rgb_resized = cv2.resize(rgb, self.config.image_size[::-1])
                self.policy_runner.add_observation(rgb_resized, state)
                
                # 4. 检查是否应该推理
                # RTC 架构下，只检查 _is_inferencing 标志
                # control_state 用于可视化显示
                should_infer = self._is_inferencing
                control_state = ControlFlags.RUNNING if self._is_inferencing else ControlFlags.IDLE
                
                if should_infer:
                    # 10Hz 推理
                    if time.time() - last_inference_time >= policy_dt:
                        self._do_inference()
                        last_inference_time = time.time()
                        
                        # 每 10 次打印一次状态
                        if self.policy_runner.inference_count % 10 == 1:
                            print(f"[RTC] 推理中... count={self.policy_runner.inference_count}")
                
                # 5. 可视化
                if not self.config.headless:
                    self._update_preview(rgb, control_state)
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key)
                
                loop_count += 1
                
                # 频率控制 (目标 30Hz 采集，10Hz 推理)
                elapsed = time.time() - loop_start
                obs_dt = 1.0 / 30.0
                sleep_time = obs_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time * 0.8)
        
        except KeyboardInterrupt:
            print("\n[RTC] Ctrl+C 中断")
        
        finally:
            self._cleanup()
    
    def _get_camera_frame(self) -> np.ndarray:
        """获取相机帧"""
        if self.camera_manager is None:
            return np.zeros((*self.config.image_size, 3), dtype=np.uint8)
        
        try:
            frames = self.camera_manager.get_frames()
            if "wrist" in frames and frames["wrist"] is not None:
                return frames["wrist"].rgb.copy()
        except:
            pass
        
        return np.zeros((*self.config.image_size, 3), dtype=np.uint8)
    
    def _get_robot_state(self) -> tuple:
        """获取机器人状态"""
        if self.robot_state is None:
            return np.zeros(7), np.zeros(7)
        
        try:
            pos, vel, _ = self.robot_state.read_robot_state()
            return pos, vel
        except:
            return np.zeros(7), np.zeros(7)
    
    def _build_state(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """构建状态向量"""
        # state_dim = 13: 6 joint_pos + 6 joint_vel + 1 gripper_pos
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        state[:6] = pos[:6]      # joint_pos
        state[6:12] = vel[:6]    # joint_vel
        state[12] = pos[6]       # gripper_pos
        return state
    
    def _get_control_state(self) -> int:
        """获取控制状态"""
        if self.robot_state is None:
            return ControlFlags.RUNNING if self._is_inferencing else ControlFlags.IDLE
        
        try:
            return self.robot_state.get_state()
        except:
            return ControlFlags.IDLE
    
    def _do_inference(self):
        """执行推理并写入共享内存"""
        # 策略推理
        q_key = self.policy_runner.predict_keyframes()
        
        # 写入共享内存
        version = self.shm_writer.write_keyframes(q_key)
        
        # 每 10 次打印一次（无论 verbose 设置）
        if self.policy_runner.inference_count % 10 == 0:
            print(f"[RTC] 写入关键帧 v={version}, shape={q_key.shape}")
    
    def _update_preview(self, rgb: np.ndarray, control_state: int):
        """更新预览窗口"""
        # RGB -> BGR 转换 (OpenCV 使用 BGR 格式显示)
        display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # 状态文本
        state_text = {
            ControlFlags.IDLE: "IDLE",
            ControlFlags.RUNNING: "RUNNING",
            ControlFlags.PAUSED: "PAUSED",
            ControlFlags.RESETTING: "RESETTING",
        }.get(control_state, "UNKNOWN")
        
        color = (0, 255, 0) if control_state == ControlFlags.RUNNING else (0, 255, 255)
        cv2.putText(display, f"State: {state_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(display, f"Infer: {'ON' if self._is_inferencing else 'OFF'}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(display, f"Count: {self.policy_runner.inference_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("RTC Inference", display)
    
    def _handle_key(self, key: int):
        """处理按键"""
        if key == ord('q') or key == ord('Q'):
            print("[RTC] 按键 Q - 退出")
            self._running = False
        elif key == ord(' '):  # Space - 开始/暂停
            self._is_inferencing = not self._is_inferencing
            print(f"[RTC] 推理: {'开始' if self._is_inferencing else '暂停'}")
        elif key == ord('r') or key == ord('R'):  # Reset
            print("[RTC] 请求复位...")
            self._is_inferencing = False
            self.policy_runner.clear_buffer()
        elif key == 255 or key == -1:
            pass  # 无按键
        elif key > 0:
            # 调试：打印未知按键
            print(f"[RTC] 未知按键: {key} ('{chr(key) if 32 <= key < 127 else '?'}')")
    
    def _cleanup(self):
        """清理资源"""
        print("\n[RTC] 正在清理...")
        
        if not self.config.headless:
            cv2.destroyAllWindows()
        
        if self.camera_manager:
            self.camera_manager.stop()
        
        if self.shm_writer:
            self.shm_writer.close()
        
        if self.robot_state:
            if hasattr(self.robot_state, '_is_owner') and self.robot_state._is_owner:
                self.robot_state.close()
        
        print("[RTC] 清理完成")


def main():
    parser = argparse.ArgumentParser(description="RTC 推理节点")
    parser.add_argument("-c", "--checkpoint", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--headless", action="store_true", help="无头模式 (无 GUI)")
    parser.add_argument("--auto-start", action="store_true", help="自动开始推理 (不等待按键)")
    parser.add_argument("--dry-run", action="store_true", help="模拟模式 (不连接机器人)")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    config = InferenceConfig(
        device=args.device,
        headless=args.headless,
        auto_start=args.auto_start,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    node = RTCInferenceNode(
        checkpoint_path=args.checkpoint,
        config=config,
    )
    
    node.setup()
    node.run()


if __name__ == "__main__":
    main()
