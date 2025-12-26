#!/usr/bin/env python3
import time

import os
import sys

# ====== ARX5 Interface Path Configuration ======
# 使用 rl-vla 统一配置模块
try:
    from inference.config import setup_arx5
    setup_arx5()
except ImportError:
    # 兼容旧方式: 通过环境变量或默认路径
    ARX5_SDK_PATH = os.environ.get('ARX5_SDK_PATH', os.path.expanduser('~/rl-vla/arx5-sdk'))
    ROOT_DIR = os.path.join(ARX5_SDK_PATH, 'python')
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import arx5_interface as arx5
import click
import numpy as np


def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):

    # To initialize robot with different configurations,
    # you can create RobotConfig and ControllerConfig by yourself and modify based on it
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    # Modify the default configuration here
    # controller_config.controller_dt = 0.01 # etc.

    arx5_joint_controller = arx5.Arx5JointController(
        robot_config, controller_config, interface
    )

    # Or you can directly use the model and interface name
    # arx5_joint_controller = arx5.Arx5JointController(model, interface)

    USE_MULTITHREADING = False

    np.set_printoptions(precision=3, suppress=True)
    arx5_joint_controller.set_log_level(arx5.LogLevel.DEBUG)
    robot_config = arx5_joint_controller.get_robot_config()
    controller_config = arx5_joint_controller.get_controller_config()

    step_num = 1500
    if USE_MULTITHREADING:
        # Will create another thread that communicates with the arm, so each send_recv_once() will take no time
        # for the main thread to execute. Otherwise (without background send/recv), send_recv_once() will block the
        # main thread until the arm responds (usually 2ms).
        arx5_joint_controller.enable_background_send_recv()
    arx5_joint_controller.reset_to_home()

    target_joint_poses = np.array([1.0, 2.0, 2.0, 1.5, 1.5, -1.57])

    for i in range(step_num):
        cmd = arx5.JointState(robot_config.joint_dof)
        # i = 0
        cmd.pos()[0:4] = easeInOutQuad(float(i) / step_num) * target_joint_poses[0:4]
        cmd.gripper_pos = easeInOutQuad((i / (step_num - 1))) * 0.08
        arx5_joint_controller.set_joint_cmd(cmd)
        if not USE_MULTITHREADING:
            arx5_joint_controller.send_recv_once()
        else:
            time.sleep(controller_config.controller_dt)
        JointState = arx5_joint_controller.get_state()
        arm_dof_pos = JointState.pos().copy()
        arm_dof_vel = JointState.vel().copy()
        # print(arm_dof_pos, arm_dof_vel)
        # print(f"gripper: {JointState.gripper_pos:.05f}")

    for i in range(step_num):
        cmd = arx5.JointState(robot_config.joint_dof)
        cmd.pos()[0:4] = (
            easeInOutQuad((1 - float(i) / step_num)) * target_joint_poses[0:4]
        )
        cmd.gripper_pos = easeInOutQuad((1 - i / (step_num - 1))) * 0.08
        arx5_joint_controller.set_joint_cmd(cmd)
        if not USE_MULTITHREADING:
            arx5_joint_controller.send_recv_once()
        else:
            time.sleep(controller_config.controller_dt)
        JointState = arx5_joint_controller.get_state()
        # print(f"gripper: {JointState.gripper_pos:.05f}")

    arx5_joint_controller.reset_to_home()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Info] 程序被用户中断 (Ctrl+C)")
        print("[Info] 退出完成")
    except Exception as e:
        print(f"\n[Error] 发生错误: {e}")
        import traceback
        traceback.print_exc()
