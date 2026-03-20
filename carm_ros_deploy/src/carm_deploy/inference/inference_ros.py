#!/usr/bin/env python3
"""
CARM 机械臂 ROS 策略推理入口

支持的算法:
    - consistency_flow: Consistency Flow Matching (推荐)
    - flow_matching: Flow Matching Policy
    - diffusion_policy: DDPM-based Diffusion Policy
    - reflected_flow: Reflected Flow Matching
    - shortcut_flow: Shortcut Flow Matching

使用方法:
    # 正常推理 (30Hz)
    rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt

    # 启用干预和采集
    rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt --intervention --record_inference
"""

import argparse
import signal
import sys
import os

# 添加 carm_deploy 根目录到路径
carm_deploy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, carm_deploy_root)

# 添加 rl-vla 根目录到路径（使 rlft 包可用）
rl_vla_root = os.path.dirname(os.path.dirname(os.path.dirname(carm_deploy_root)))
sys.path.insert(0, rl_vla_root)

import rospy

from rlft.utils.model_factory import SUPPORTED_ALGORITHMS
from inference.config import InferenceConfig

# Re-export InferenceNode for backward compatibility
from inference.inference_node import InferenceNode  # noqa: F401


# ============================================================================
# parse_args
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARM Robot Policy Inference (ROS)')

    # 机械臂参数
    parser.add_argument('--robot_ip', type=str, default='10.42.0.101',
                        help='Robot IP address')
    parser.add_argument('--robot_mode', type=int, default=4,
                        help='Control mode (0=IDLE, 1=POSITION, 2=MIT, 3=DRAG, 4=PF)')
    parser.add_argument('--robot_tau', type=float, default=10,
                        help='Gripper torque')

    # 初始位置
    parser.add_argument('--arm_init_pose', type=float, nargs=7,
                        default=[0.2475, 0.0014, 0.3251, 0.9996, -0.0034, 0.0255, -0.0074],
                        help='Initial end effector pose [x,y,z,qx,qy,qz,qw]')
    parser.add_argument('--arm_init_gripper', type=float, default=0.078,
                        help='Initial gripper position')

    # 相机参数
    parser.add_argument('--camera_topics', type=str,
                        default='/camera/color/image_raw',
                        help='Camera topic(s), comma separated')
    parser.add_argument('--sync_slop', type=float, default=0.02,
                        help='Image sync tolerance in seconds')

    # 时间线 — BUG-6 fix: 只保留 --timeline_disabled
    parser.add_argument('--timeline_disabled', action='store_true',
                        help='Disable timeline logging (enabled by default)')
    parser.add_argument('--timeline_log', type=str, default='',
                        help='Timeline log path (JSONL). Empty uses log_dir')
    parser.add_argument('--timeline_control_stride', type=int, default=10,
                        help='Log every N control steps (control loop)')
    parser.add_argument('--chunk_time_base', type=str, default='sys_time',
                        choices=['sys_time', 'obs_stamp'],
                        help='Chunk base time: sys_time (recommended) or obs_stamp')

    # 策略参数
    parser.add_argument('--pretrain', type=str, default='',
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--algorithm', type=str, default='consistency_flow',
                        choices=SUPPORTED_ALGORITHMS,
                        help='Algorithm type (auto-detected from args.json if available)')
    parser.add_argument('--desire_inference_freq', type=float, default=30,
                        help='Desired inference frequency')
    parser.add_argument('--temporal_factor_k', type=float, default=0.05,
                        help='Temporal factor for action fusion')
    parser.add_argument('--num_inference_steps', type=int, default=10,
                        help='Number of flow/diffusion steps for inference')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA model for inference')

    # Action Chunk 执行模式参数
    parser.add_argument('--execution_mode', type=str, default='receding_horizon',
                        choices=['temporal_ensemble', 'receding_horizon'],
                        help='Action chunk execution mode')
    parser.add_argument('--max_active_chunks', type=int, default=None,
                        help='Max active chunks in manager')
    parser.add_argument('--crossfade_steps', type=int, default=0,
                        help='Steps for crossfade smoothing (receding_horizon only)')
    # BUG-8 fix: use --no_truncate_at_act_horizon to disable
    parser.add_argument('--no_truncate_at_act_horizon',
                        dest='truncate_at_act_horizon', action='store_false',
                        help='Disable action chunk truncation at act_horizon')
    parser.set_defaults(truncate_at_act_horizon=True)
    parser.add_argument('--act_horizon', type=int, default=8,
                        help='Action horizon for chunk truncation')

    # 控制参数
    parser.add_argument('--pos_lookahead_step', type=int, default=1,
                        help='Position lookahead step')
    parser.add_argument('--pos_lookahead_duration', type=float, default=0.015,
                        help='Position lookahead duration')
    parser.add_argument('--joint_cmd_mode', action='store_true',
                        help='[DEPRECATED] Will raise error if used.')
    parser.add_argument('--teleop_scale', type=float, default=1.0,
                        help='[DEPRECATED] Fixed to 1.0. Use --inference_speed_scale.')
    parser.add_argument('--inference_speed_scale', type=float, default=1.0,
                        help='Runtime speed scaling for predicted actions')
    parser.add_argument('--control_freq', type=int, default=50,
                        help='Control loop frequency in Hz')
    parser.add_argument('--gripper_hysteresis_window', type=int, default=1,
                        help='Gripper hysteresis voting window size')

    # 安全
    parser.add_argument('--safety_config', type=str, default='',
                        help='Path to safety config JSON file')
    parser.add_argument('--init_speed', type=float, default=2.0,
                        help='Speed level for initialization movement (0-10)')
    parser.add_argument('--skip_init_confirm', action='store_true',
                        help='Skip arm init confirmation prompt (for scripted/automated launch)')

    # 日志
    parser.add_argument('--log_dir', type=str, default='',
                        help='Directory to save inference logs')
    parser.add_argument('--save_images', action='store_true',
                        help='Save images in inference log')
    parser.add_argument('--vis', action='store_true', default=True,
                        help='Visualize images in OpenCV window')

    # 干预和采集
    parser.add_argument('--record_inference', action='store_true',
                        help='Enable inference data recording')
    parser.add_argument('--intervention', action='store_true',
                        help='Enable keyboard intervention')
    parser.add_argument('--intervention_mode', type=str, default='replace',
                        choices=['replace', 'additive'],
                        help='Intervention mode')
    parser.add_argument('--intervention_xyz_scale', type=float, default=0.01,
                        help='XYZ movement scale per keypress in meters')
    parser.add_argument('--intervention_gripper_open', type=float, default=1.0,
                        help='Gripper open value for intervention')
    parser.add_argument('--intervention_gripper_close', type=float, default=0.0,
                        help='Gripper close value for intervention')
    parser.add_argument('--record_dir', type=str, default='',
                        help='Directory to save recorded inference data')
    parser.add_argument('--max_steps', type=int, default=99999,
                        help='Maximum steps per episode')

    # 兼容 roslaunch remap 参数
    return parser.parse_args(args=rospy.myargv()[1:])


# ============================================================================
# main
# ============================================================================

def main():
    """主函数"""
    rospy.init_node('carm_inference', anonymous=True)
    args = parse_args()

    # Build typed config
    cfg = InferenceConfig.from_argparse(args)

    # ROS param overlay
    ros_keys = [
        'robot_ip', 'robot_mode', 'robot_tau', 'arm_init_pose', 'arm_init_gripper',
        'camera_topics', 'sync_slop', 'timeline_disabled', 'timeline_log',
        'timeline_control_stride', 'chunk_time_base',
        'pretrain', 'algorithm', 'desire_inference_freq', 'temporal_factor_k',
        'num_inference_steps', 'use_ema', 'pos_lookahead_step', 'pos_lookahead_duration',
        'safety_config',
        'log_dir', 'save_images', 'vis',
        'execution_mode', 'max_active_chunks', 'crossfade_steps',
        'truncate_at_act_horizon', 'act_horizon',
        'inference_speed_scale', 'control_freq', 'gripper_hysteresis_window',
        'record_inference', 'intervention', 'intervention_mode',
        'intervention_xyz_scale', 'intervention_gripper_open', 'intervention_gripper_close',
        'record_dir', 'max_steps',
    ]
    for key in ros_keys:
        if rospy.has_param(f'~{key}'):
            setattr(cfg, key, rospy.get_param(f'~{key}'))

    # Normalize
    cfg.normalize_camera_topics()
    cfg.normalize_arm_init_pose()

    # Safety config resolution
    safety_path = cfg.resolve_safety_config(carm_deploy_root)
    if not os.path.exists(safety_path):
        rospy.logfatal("=" * 60)
        rospy.logfatal("Safety config not found: %s", safety_path)
        rospy.logfatal("Run: cd carm_ros_deploy/src/carm_deploy/tools && python record_workspace.py")
        rospy.logfatal("=" * 60)
        raise SystemExit(1)

    # Deprecated param checks
    if cfg.joint_cmd_mode:
        rospy.logfatal("joint_cmd_mode is no longer supported!")
        raise SystemExit(1)

    # Log config
    rospy.loginfo("=" * 60)
    rospy.loginfo("CARM Policy Inference Node")
    rospy.loginfo("=" * 60)
    rospy.loginfo(f"Robot IP: {cfg.robot_ip}")
    rospy.loginfo(f"Camera topics: {cfg.camera_topics}")
    rospy.loginfo(f"Pretrain: {cfg.pretrain}")
    rospy.loginfo(f"Execution mode: {cfg.execution_mode}")
    rospy.loginfo(f"Inference speed scale: {cfg.inference_speed_scale}")
    rospy.loginfo(f"Control freq: {cfg.control_freq}Hz")
    rospy.loginfo(f"Safety config: {cfg.safety_config}")
    rospy.loginfo("=" * 60)

    # Create node
    node = InferenceNode(cfg)

    # Signal handling
    shutdown_in_progress = False

    def signal_handler(signum, frame):
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            rospy.logwarn("Force exit requested, exiting immediately...")
            sys.exit(1)
        shutdown_in_progress = True
        rospy.loginfo("\nReceived shutdown signal, cleaning up...")
        node.shutdown()
        rospy.signal_shutdown("User interrupted")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    rospy.on_shutdown(node.shutdown)

    try:
        node.control_loop()
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
    finally:
        if not shutdown_in_progress:
            node.shutdown()


if __name__ == '__main__':
    main()
