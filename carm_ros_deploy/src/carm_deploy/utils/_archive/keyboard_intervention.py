#!/usr/bin/env python3
"""
键盘干预模块 - 已退役的历史实验实现

状态说明:
- 该模块保留仅用于回看早期 keyboard-intervention 实验设计。
- 当前 `inference_ros` 主 pipeline 已不再接入本模块。
- 当前现役推理链路只保留 episode 级别的 `EpisodeKeyboardHandler` 控制，
  不再在 policy 输出与执行之间插入人工键盘干预。

如果后续要做 Human-in-the-loop inference，请不要直接复用这条旧链路，
应基于当前 `action_model -> action_executed` 语义重新设计接入点。

支持:
- xyz 平移 (W/S=X, A/D=Y, Q/E=Z)
- gripper 二值开合 (G 开 / H 关)
- 两种干预模式: replace (替换) / additive (叠加)

使用方法:
    handler = KeyboardInterventionHandler(
        xyz_scale=0.005,      # xyz 单步位移 (米)
        gripper_open=1.0,     # 夹爪打开值
        gripper_close=0.0,    # 夹爪关闭值
        mode='replace',       # 干预模式
    )
    handler.start()
    
    # 在推理循环中
    intervention = handler.get_intervention()
    if intervention is not None:
        delta, mode, mask = intervention
        # 应用干预...
    
    handler.stop()
"""

import threading
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import deque

try:
    import rospy
    HAS_ROSPY = True
except ImportError:
    HAS_ROSPY = False


class KeyboardInterventionHandler:
    """
    键盘干预处理器。

    该类属于历史实验接口，不再属于当前现役 inference pipeline。
    
    键位映射:
        W/S: X 轴正/负方向移动
        A/D: Y 轴负/正方向移动  
        Q/E: Z 轴正/负方向移动
        G: 夹爪打开
        H: 夹爪关闭
        
        R: 开始/停止录制
        Y: 确认保存 episode
        N: 丢弃 episode
        Ctrl+C: 退出
    
    干预模式:
        - replace: 替换模型输出的对应维度
        - additive: 在模型输出基础上叠加增量
    """
    
    # 键位到动作的映射 (dx, dy, dz)
    # W/S = X轴, A/D = Y轴, Q/E = Z轴
    KEY_TO_XYZ = {
        'w': (1, 0, 0),    # X+
        's': (-1, 0, 0),   # X-
        'a': (0, -1, 0),   # Y-
        'd': (0, 1, 0),    # Y+
        'q': (0, 0, 1),    # Z+
        'e': (0, 0, -1),   # Z-
    }
    
    def __init__(
        self,
        xyz_scale: float = 0.005,
        gripper_open: float = 1.0,
        gripper_close: float = 0.0,
        mode: str = 'replace',
        max_xyz_per_step: float = 0.01,
    ):
        """
        初始化键盘干预处理器
        
        Args:
            xyz_scale: xyz 单步位移幅度 (米)，默认 5mm
            gripper_open: 夹爪打开值
            gripper_close: 夹爪关闭值
            mode: 干预模式 'replace' 或 'additive'
            max_xyz_per_step: xyz 单步最大位移 (米)，安全限制
        """
        self.xyz_scale = min(xyz_scale, max_xyz_per_step)
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.mode = mode
        self.max_xyz_per_step = max_xyz_per_step
        
        # 状态
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # 当前按下的键 (用于持续移动)
        self._pressed_keys = set()
        
        # 当前干预状态
        self._current_xyz_delta = np.zeros(3)
        self._current_gripper = None  # None 表示不干预 gripper
        self._intervention_active = False
        
        # 录制控制回调
        self._record_callback = None  # (action: str) -> None
        self._quit_callback = None    # () -> None
        
        # 干预事件队列 (用于日志)
        self._event_queue = deque(maxlen=100)
        
    def set_record_callback(self, callback):
        """设置录制控制回调"""
        self._record_callback = callback
        
    def set_quit_callback(self, callback):
        """设置退出回调"""
        self._quit_callback = callback
    
    def start(self):
        """启动键盘监听线程。"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._thread.start()
        
        if HAS_ROSPY:
            rospy.logwarn("KeyboardInterventionHandler started in archived mode; current inference_ros no longer uses this path")
            rospy.loginfo("Controls: WS=X, AD=Y, QE=Z, G=Open, H=Close, R=Record, Y/N=Save/Discard, S/F=Success/Failure")
        else:
            print("KeyboardInterventionHandler started in archived mode; current inference_ros no longer uses this path")
            print("Controls: WS=X, AD=Y, QE=Z, G=Open, H=Close, R=Record, Y/N=Save/Discard, S/F=Success/Failure")
    
    def stop(self):
        """停止键盘监听"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        if HAS_ROSPY:
            rospy.loginfo("KeyboardInterventionHandler stopped")
    
    def _keyboard_loop(self):
        """键盘监听主循环"""
        try:
            import sys
            import tty
            import termios
            import select
            
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                
                while self._running:
                    # 使用 select 实现非阻塞读取
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        key = sys.stdin.read(1)
                        self._handle_key(key)
                    
                    # 更新持续移动状态
                    self._update_continuous_movement()
                    
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                
        except Exception as e:
            if HAS_ROSPY:
                rospy.logwarn(f"Keyboard listener error: {e}")
            else:
                print(f"Keyboard listener error: {e}")
    
    def _handle_key(self, key: str):
        """处理按键事件"""
        key_lower = key.lower()
        
        with self._lock:
            # XYZ 移动
            if key_lower in self.KEY_TO_XYZ:
                dx, dy, dz = self.KEY_TO_XYZ[key_lower]
                self._current_xyz_delta = np.array([dx, dy, dz]) * self.xyz_scale
                self._intervention_active = True
                self._log_event('xyz', key=key_lower, delta=self._current_xyz_delta.tolist())
                
            # 夹爪控制: G=打开, H=关闭 (一次性干预，不持续覆盖模型预测)
            elif key_lower == 'g':  # G - 打开
                self._current_gripper = self.gripper_open
                self._intervention_active = True
                self._log_event('gripper', action='open', value=self.gripper_open)
                
            elif key_lower == 'h':  # H - 关闭
                self._current_gripper = self.gripper_close
                self._intervention_active = True
                self._log_event('gripper', action='close', value=self.gripper_close)
                
            # 录制控制
            elif key_lower == 'r':
                if self._record_callback:
                    self._record_callback('toggle')
                self._log_event('record', action='toggle')
                    
            elif key_lower == 'y':
                if self._record_callback:
                    self._record_callback('confirm')
                self._log_event('record', action='confirm')

            elif key_lower == 'n':
                if self._record_callback:
                    self._record_callback('discard')
                self._log_event('record', action='discard')

            elif key_lower == 's':
                if self._record_callback:
                    self._record_callback('mark_success')
                self._log_event('record', action='mark_success')

            elif key_lower == 'f':
                if self._record_callback:
                    self._record_callback('mark_failure')
                self._log_event('record', action='mark_failure')
            
            # 退出
            elif key == '\x03':  # Ctrl+C
                if self._quit_callback:
                    self._quit_callback()
                self._log_event('quit')
    
    def _update_continuous_movement(self):
        """更新持续移动状态（键松开后清零）"""
        # 简化实现：按键触发后保持一小段时间
        # 完整实现需要检测 key release 事件
        pass
    
    def _log_event(self, event_type: str, **kwargs):
        """记录干预事件"""
        self._event_queue.append({
            'timestamp': time.time(),
            'type': event_type,
            **kwargs,
        })
    
    def get_intervention(self) -> Optional[Tuple[np.ndarray, str, np.ndarray]]:
        """
        获取当前干预
        
        Returns:
            如果没有干预，返回 None
            否则返回 (delta_or_target, mode, mask):
                - delta_or_target: 干预值 [dx, dy, dz, gripper] 或 None
                - mode: 'replace' 或 'additive'
                - mask: bool 数组，标记哪些维度被干预 [x, y, z, gripper]
        """
        with self._lock:
            if not self._intervention_active:
                return None
            
            # 构建干预值和掩码
            # [dx, dy, dz, gripper]
            intervention = np.zeros(4)
            mask = np.zeros(4, dtype=bool)
            
            # XYZ 干预
            if np.any(self._current_xyz_delta != 0):
                intervention[:3] = self._current_xyz_delta
                mask[:3] = True
            
            # Gripper 干预
            if self._current_gripper is not None:
                intervention[3] = self._current_gripper
                mask[3] = True
            
            # 清除状态（一次性干预）
            # 修复：gripper 干预也应该是一次性的，否则会持续覆盖模型预测
            self._current_xyz_delta = np.zeros(3)
            self._current_gripper = None  # 清除 gripper 干预，让模型恢复控制
            self._intervention_active = np.any(mask)
            
            if not np.any(mask):
                return None
            
            return intervention, self.mode, mask
    
    def is_intervening(self) -> bool:
        """检查当前是否有干预"""
        with self._lock:
            return self._intervention_active
    
    def get_events(self) -> list:
        """获取并清空事件队列"""
        with self._lock:
            events = list(self._event_queue)
            self._event_queue.clear()
            return events
    
    def reset(self):
        """重置干预状态"""
        with self._lock:
            self._current_xyz_delta = np.zeros(3)
            self._current_gripper = None
            self._intervention_active = False
            self._pressed_keys.clear()


class InterventionApplier:
    """
    干预应用器 - 将键盘干预应用到模型输出的 action。

    该类仅保留为历史参考，不再由当前 inference_ros 主流程调用。
    
    支持两种模式:
        - replace: 替换对应维度
        - additive: 叠加增量
    """
    
    @staticmethod
    def apply_to_action_chunk(
        all_actions: np.ndarray,
        intervention: Tuple[np.ndarray, str, np.ndarray],
        action_format: str = 'ee_delta',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将干预应用到整个 action chunk
        
        Args:
            all_actions: 模型输出 [pred_horizon, action_dim]
                - ee_delta 格式: [relative_pose(7), gripper(1)] 或 full 15D
            intervention: (delta_or_target, mode, mask) 从 KeyboardInterventionHandler.get_intervention()
            action_format: 动作格式 'ee_delta' 或 'joint'
            
        Returns:
            (modified_actions, intervention_mask):
                - modified_actions: 干预后的 actions [pred_horizon, action_dim]
                - intervention_mask: [pred_horizon, action_dim] bool 数组
        """
        delta_or_target, mode, mask = intervention
        modified_actions = all_actions.copy()
        intervention_mask = np.zeros_like(all_actions, dtype=bool)
        
        if action_format == 'ee_delta':
            # ee_delta 格式: action = [relative_pose(7), gripper(1)]
            # relative_pose = [dx, dy, dz, dqx, dqy, dqz, dqw]
            # 或 full 15D: [joint(6), gripper(1), relative_pose(7), gripper(1)]
            
            action_dim = all_actions.shape[-1]
            
            if action_dim == 8:
                # [relative_pose(7), gripper(1)]
                xyz_idx = slice(0, 3)
                gripper_idx = 7
            elif action_dim == 15:
                # [joint(6), gripper(1), relative_pose(7), gripper(1)]
                xyz_idx = slice(7, 10)  # relative_pose 的 xyz 部分
                gripper_idx = 14  # 最后一个 gripper
            else:
                raise ValueError(f"Unsupported action_dim: {action_dim}")
            
            for i in range(len(modified_actions)):
                # XYZ 干预
                if np.any(mask[:3]):
                    if mode == 'replace':
                        # replace 模式：直接设置相对位移
                        modified_actions[i, xyz_idx][mask[:3]] = delta_or_target[:3][mask[:3]]
                    else:  # additive
                        # additive 模式：叠加到相对位移
                        modified_actions[i, xyz_idx][mask[:3]] += delta_or_target[:3][mask[:3]]
                    intervention_mask[i, xyz_idx] = mask[:3]
                
                # Gripper 干预
                if mask[3]:
                    if mode == 'replace':
                        modified_actions[i, gripper_idx] = delta_or_target[3]
                    else:  # additive
                        modified_actions[i, gripper_idx] += delta_or_target[3]
                    intervention_mask[i, gripper_idx] = True
                    
                    # 如果是 15D，同步更新第一个 gripper
                    if action_dim == 15:
                        modified_actions[i, 6] = modified_actions[i, gripper_idx]
                        intervention_mask[i, 6] = True
        
        elif action_format == 'joint':
            # joint 格式: [j1, j2, j3, j4, j5, j6, gripper]
            # 键盘干预 xyz 在关节空间下不直接适用，需要 IK
            # 这里只支持 gripper 干预
            gripper_idx = 6
            
            for i in range(len(modified_actions)):
                if mask[3]:  # gripper
                    if mode == 'replace':
                        modified_actions[i, gripper_idx] = delta_or_target[3]
                    else:
                        modified_actions[i, gripper_idx] += delta_or_target[3]
                    intervention_mask[i, gripper_idx] = True
        
        return modified_actions, intervention_mask


if __name__ == '__main__':
    # 简单测试
    print("Testing archived KeyboardInterventionHandler...")
    
    handler = KeyboardInterventionHandler(
        xyz_scale=0.005,
        gripper_open=1.0,
        gripper_close=0.0,
        mode='replace',
    )
    
    def on_record(action):
        print(f"Record action: {action}")
    
    def on_quit():
        print("Quit requested")
        handler.stop()
    
    handler.set_record_callback(on_record)
    handler.set_quit_callback(on_quit)
    handler.start()
    
    print("Press keys to test (WS=X, AD=Y, QE=Z, G/H=Gripper, R=Record, Ctrl+C to quit)")
    
    try:
        while handler._running:
            intervention = handler.get_intervention()
            if intervention is not None:
                delta, mode, mask = intervention
                print(f"Intervention: delta={delta}, mode={mode}, mask={mask}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    handler.stop()
    print("Test complete")
