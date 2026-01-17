#!/usr/bin/env python3
"""
CARM 机械臂 ROS 策略推理主程序
基于 carm_real/infer_g3_api.py 重构，将 svar 通信替换为 ROS1 原生通信

使用方法:
    rosrun carm_ros_deploy inference_ros.py --pretrain /path/to/model.pt
"""

import argparse
import threading
import time
import numpy as np
import cv2
import rospy
from scipy.spatial.transform import Rotation as R
from einops import rearrange

# 本地模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_ros import RealEnvironment
from utils.trajectory_interpolator import VecTF, ActionChunkManager


def pose_to_transform_matrix(position, quaternion):
    """
    将位姿 (xyz + 四元数) 转换为 4x4 变换矩阵
    
    Args:
        position: 平移 [x, y, z]
        quaternion: 四元数 [qx, qy, qz, qw]
        
    Returns:
        4x4 变换矩阵
    """
    rotation = R.from_quat(quaternion).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def compute_relative_pose(pose_absolute, pose_init, gripper):
    """
    计算相对位姿
    
    Args:
        pose_absolute: 绝对位姿 [x,y,z,qx,qy,qz,qw]
        pose_init: 初始位姿 [x,y,z,qx,qy,qz,qw]
        gripper: 夹爪开度
        
    Returns:
        相对位姿 [x,y,z,qx,qy,qz,qw,gripper]
    """
    start2current = pose_to_transform_matrix(pose_absolute[:3], pose_absolute[3:])
    start = pose_to_transform_matrix(pose_init[:3], pose_init[3:])
    
    current2global = start @ start2current
    
    cur_position = current2global[:3, 3]
    cur_euler = R.from_matrix(current2global[:3, :3]).as_quat()
    
    pose_relative = cur_position.tolist() + cur_euler.tolist() + [gripper]
    return pose_relative


class PolicyInterface:
    """
    策略模型接口（抽象基类）
    用户需要继承此类并实现 load_model 和 __call__ 方法
    """
    
    def __init__(self, config):
        """
        初始化策略接口
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
        """
        raise NotImplementedError("Subclass must implement load_model()")
    
    def __call__(self, inputs):
        """
        执行推理
        
        Args:
            inputs: 输入字典，包含 'qpos' 和 'image'
            
        Returns:
            输出字典，包含 'a_hat' (动作预测)
        """
        raise NotImplementedError("Subclass must implement __call__()")


class DummyPolicy(PolicyInterface):
    """
    虚拟策略（用于测试）
    返回零动作
    """
    
    def load_model(self, model_path):
        rospy.loginfo(f"DummyPolicy: would load model from {model_path}")
        self.model = True
    
    def __call__(self, inputs):
        # 返回零动作，形状为 [1, horizon, action_dim]
        # 假设 horizon=16, action_dim=15 (7 joint + 8 end pose)
        batch_size = inputs['qpos'].shape[0]
        horizon = 16
        action_dim = 15
        
        # 获取当前 qpos 作为动作
        qpos = inputs['qpos'].cpu().numpy()  # [B, 7]
        
        # 扩展为 horizon 步
        actions = np.tile(qpos, (1, horizon, 1))
        
        # 添加末端位姿（使用零位姿）
        end_pose = np.zeros((batch_size, horizon, 8))
        actions = np.concatenate([actions, end_pose], axis=-1)
        
        import torch
        return {'a_hat': torch.from_numpy(actions).float()}


class InferenceNode:
    """
    ROS 推理节点
    """
    
    def __init__(self, config):
        """
        初始化推理节点
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 参数
        self.temporal_factor_k = config.get('temporal_factor_k', 0.01)
        self.desire_inference_freq = config.get('desire_inference_freq', 20)
        self.pos_lookahead_step = config.get('pos_lookahead_step', 1)
        self.pos_lookahead_duration = config.get('pos_lookahead_duration', 0.015)
        self.joint_cmd_mode = config.get('joint_cmd_mode', False)
        
        # 初始化环境
        rospy.loginfo("Initializing environment...")
        self.env = RealEnvironment(config)
        
        # 初始化策略（暂时使用虚拟策略）
        rospy.loginfo("Initializing policy...")
        self.policy = self._create_policy(config)
        
        # 动作管理器
        self.action_manager = ActionChunkManager(temporal_factor_k=self.temporal_factor_k)
        self.lock_tfs = threading.Lock()
        
        # 控制变量
        self.running = True
        self.latest_obs = None
        self.pos_lookahead_step_start_idx = 0
        
        # 启动推理线程
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        rospy.loginfo("InferenceNode initialized")
    
    def _create_policy(self, config):
        """
        创建策略实例
        
        用户可以修改此方法以加载实际的策略模型
        """
        # TODO: 替换为实际的策略加载逻辑
        # 例如:
        # from your_policy_module import YourPolicy
        # policy = YourPolicy(config)
        # policy.load_model(config.get('pretrain', ''))
        
        policy = DummyPolicy(config)
        pretrain_path = config.get('pretrain', '')
        if pretrain_path:
            policy.load_model(pretrain_path)
        else:
            rospy.logwarn("No pretrain model specified, using dummy policy")
        
        return policy
    
    def _normalize_images(self, obs):
        """
        归一化图像
        
        Args:
            obs: 观测字典
            
        Returns:
            torch.Tensor: 归一化后的图像 [B, C, H, W]
        """
        import torch
        
        curr_images = []
        for image in obs["images"]:
            curr_image = rearrange(image, 'h w c -> c h w')
            curr_images.append(curr_image)
        
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        
        # 只使用第一个相机
        curr_image = curr_image[:, 0].unsqueeze(1)  # [1, 1, C, H, W]
        
        return curr_image
    
    def _inference_loop(self):
        """推理线程主循环"""
        rospy.loginfo("Inference thread started")
        
        import torch
        desire_period = 1.0 / self.desire_inference_freq
        
        with torch.inference_mode():
            while self.running and not rospy.is_shutdown():
                # 获取观测
                self.latest_obs = self.env.get_observation()
                if self.latest_obs is None:
                    time.sleep(0.5)
                    rospy.loginfo_throttle(5.0, "Waiting for observation...")
                    continue
                
                last_start = time.time()
                
                try:
                    # 准备输入
                    qpos_joint = np.array(self.latest_obs['qpos_joint'])
                    qpos_end = np.array(self.latest_obs['qpos_end']).tolist()
                    qpos = torch.from_numpy(qpos_joint).float().cuda().unsqueeze(0)
                    
                    curr_image = self._normalize_images(self.latest_obs)
                    
                    # 推理
                    ret = self.policy({"qpos": qpos, "image": curr_image})
                    all_actions = ret["a_hat"].squeeze(0).cpu().numpy()
                    
                    # 转换动作空间
                    if not self.joint_cmd_mode:
                        all_endactions = []
                        for i in range(all_actions.shape[0]):
                            end_action = all_actions[i][7:]
                            grip = all_actions[i][6]
                            action = compute_relative_pose(end_action[:7], qpos_end[:7], grip)
                            all_endactions.append(action)
                        all_actions = np.array(all_endactions)
                    
                    # 创建轨迹并添加到管理器
                    stamp = self.latest_obs["stamp"]
                    tf = VecTF({})
                    
                    self.pos_lookahead_step_start_idx += 1
                    for i in range(len(all_actions)):
                        if self.pos_lookahead_step == 1:
                            tf.append(stamp + i * desire_period, all_actions[i].tolist())
                        else:
                            if self.pos_lookahead_step_start_idx % self.pos_lookahead_step == 0:
                                tf.append(stamp + i * desire_period, all_actions[i].tolist())
                            else:
                                tf.append(stamp + i * self.pos_lookahead_duration, all_actions[i].tolist())
                    
                    with self.lock_tfs:
                        self.action_manager.add_trajectory(tf)
                    
                    inference_time = time.time() - last_start
                    rospy.logdebug(f"Inference time: {inference_time:.4f}s")
                    
                except Exception as e:
                    rospy.logerr(f"Error in inference: {e}")
                
                # 等待下一个周期
                wait_tm = desire_period - (time.time() - last_start)
                if wait_tm > 0:
                    time.sleep(wait_tm)
    
    def control_loop(self):
        """控制主循环"""
        rospy.loginfo("Control loop started")
        
        while self.running and not rospy.is_shutdown():
            # 获取融合后的动作
            tm = time.time()
            
            with self.lock_tfs:
                action = self.action_manager.get_fused_action(tm)
            
            if action is None:
                time.sleep(0.02)
                continue
            
            # 执行控制
            if self.joint_cmd_mode:
                rospy.logdebug("Joint control")
                self.env.joint_control_nostep(action)
            else:
                rospy.logdebug("End pose control")
                self.env.end_control_nostep(action)
            
            time.sleep(0.005)
    
    def shutdown(self):
        """关闭节点"""
        rospy.loginfo("Shutting down InferenceNode...")
        self.running = False
        
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        
        self.env.shutdown()
        rospy.loginfo("InferenceNode shutdown complete")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARM Robot Policy Inference (ROS)')
    
    # 机械臂参数
    parser.add_argument('--robot_ip', type=str, default='10.42.0.101',
                        help='Robot IP address')
    parser.add_argument('--robot_mode', type=int, default=4,
                        help='Control mode (0=IDLE, 1=POSITION, 2=MIT, 3=DRAG)')
    parser.add_argument('--robot_tau', type=float, default=10,
                        help='Gripper torque')
    
    # 初始位置
    parser.add_argument('--arm_init_pose', type=float, nargs=7,
                        default=[0.26, -0.02, 0.22, 1, 0, 0, 0],
                        help='Initial end effector pose [x,y,z,qx,qy,qz,qw]')
    parser.add_argument('--arm_init_gripper', type=float, default=0.05,
                        help='Initial gripper position')
    
    # 相机参数
    parser.add_argument('--camera_topics', type=str,
                        default='/camera/color/image_raw',
                        help='Camera topic(s), comma separated')
    parser.add_argument('--sync_slop', type=float, default=0.1,
                        help='Image sync tolerance in seconds')
    
    # 策略参数
    parser.add_argument('--pretrain', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--desire_inference_freq', type=float, default=30,
                        help='Desired inference frequency')
    parser.add_argument('--temporal_factor_k', type=float, default=0.05,
                        help='Temporal factor for action fusion')
    
    # 控制参数
    parser.add_argument('--pos_lookahead_step', type=int, default=1,
                        help='Position lookahead step')
    parser.add_argument('--pos_lookahead_duration', type=float, default=0.015,
                        help='Position lookahead duration')
    parser.add_argument('--joint_cmd_mode', action='store_true',
                        help='Use joint command mode')
    parser.add_argument('--not_origin', action='store_true',
                        help='Skip initial pose')
    
    # 可视化
    parser.add_argument('--vis', action='store_true',
                        help='Visualize images')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 初始化 ROS 节点
    rospy.init_node('carm_inference', anonymous=True)
    
    # 解析参数
    args = parse_args()
    
    # 转换为配置字典
    config = vars(args)
    
    # 处理相机话题
    if isinstance(config['camera_topics'], str):
        config['camera_topics'] = config['camera_topics'].split(',')
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("CARM Policy Inference Node")
    rospy.loginfo("=" * 50)
    rospy.loginfo(f"Robot IP: {config['robot_ip']}")
    rospy.loginfo(f"Camera topics: {config['camera_topics']}")
    rospy.loginfo(f"Pretrain: {config['pretrain']}")
    rospy.loginfo(f"Joint cmd mode: {config['joint_cmd_mode']}")
    rospy.loginfo("=" * 50)
    
    # 创建推理节点
    node = InferenceNode(config)
    
    # 注册关闭回调
    rospy.on_shutdown(node.shutdown)
    
    try:
        # 运行控制循环
        node.control_loop()
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
    finally:
        node.shutdown()


if __name__ == '__main__':
    main()
