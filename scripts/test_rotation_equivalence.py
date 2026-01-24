#!/usr/bin/env python3
"""
旋转乘法顺序数学等价性验证

对比两种位姿变换方法：
1. inference_ros.py: apply_relative_transform - 使用齐次变换矩阵右乘
   T_target = T_current @ T_relative
   
2. joystick.py / pose_diff.py: apply_transform_to_pose - 位置加法 + 四元数左乘
   new_pos = pose[:3] + pos_diff
   new_rot = diff_quat * current_rot

目的：验证这两种方法在什么情况下等价/不等价
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# ==========================================
# 方法1: inference_ros.py 的实现
# ==========================================
def pose_to_transform_matrix(position, quaternion):
    """将位姿转换为 4x4 齐次变换矩阵"""
    rotation = R.from_quat(quaternion).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def apply_relative_transform_v1(relative_pose, current_pose):
    """
    inference_ros.py 的方法: 齐次变换矩阵右乘
    T_target = T_current @ T_relative
    
    语义：relative_pose 是在当前帧坐标系下的相对变换
    """
    T_relative = pose_to_transform_matrix(relative_pose[:3], relative_pose[3:7])
    T_current = pose_to_transform_matrix(current_pose[:3], current_pose[3:7])
    
    T_target = T_current @ T_relative  # 右乘
    
    target_position = T_target[:3, 3]
    target_quat = R.from_matrix(T_target[:3, :3]).as_quat()
    
    return np.concatenate([target_position, target_quat])


# ==========================================
# 方法2: pose_diff.py 的实现  
# ==========================================
def apply_transform_to_pose_v2(pose, pos_diff, quat_diff):
    """
    pose_diff.py 的方法: 位置加法 + 四元数左乘
    new_pos = pose[:3] + pos_diff
    new_rot = diff_quat * current_rot  (左乘)
    
    语义：pos_diff 和 quat_diff 是在世界坐标系下的差值
    """
    # 更新位置：直接加法
    new_pos = pose[:3] + pos_diff
    
    # 更新姿态：四元数左乘
    current_quat = pose[3:7]
    diff_rot = R.from_quat(quat_diff)
    current_rot = R.from_quat(current_quat)
    new_rot = diff_rot * current_rot  # 左乘
    new_quat = new_rot.as_quat()
    
    return np.concatenate([new_pos, new_quat])


# ==========================================
# 方法2变体: 四元数右乘（假设的等价形式）
# ==========================================
def apply_transform_to_pose_v2_right(pose, pos_diff, quat_diff):
    """
    四元数右乘变体（用于对比）
    new_rot = current_rot * diff_quat  (右乘)
    
    语义：quat_diff 是在当前末端坐标系下的旋转
    """
    new_pos = pose[:3] + pos_diff
    
    current_quat = pose[3:7]
    diff_rot = R.from_quat(quat_diff)
    current_rot = R.from_quat(current_quat)
    new_rot = current_rot * diff_rot  # 右乘
    new_quat = new_rot.as_quat()
    
    return np.concatenate([new_pos, new_quat])


# ==========================================
# homogeneous_diff_with_scale 实现（从 pose_diff.py）
# ==========================================
def quaternion_slerp(q0, q1, t):
    """球面线性插值 (Slerp)"""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-7:
        return (1 - t) * q0 + t * q1
    
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def homogeneous_diff_with_scale(T1, T2, scale):
    """
    计算两个齐次变换矩阵之间的差值，位移和旋转都缩放 scale 倍
    
    返回：pos_diff (世界坐标系下), quat_diff (世界坐标系下)
    """
    pos_diff = T2[:3, 3] - T1[:3, 3]
    pos_diff = pos_diff * scale
    
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    R_diff = R2 @ R1.T  # 相对旋转矩阵: R_diff = R2 @ R1^T，即 R2 = R_diff @ R1
    
    q1 = np.array([0, 0, 0, 1])  # 单位四元数
    q2 = R.from_matrix(R_diff).as_quat()
    rot_diff = quaternion_slerp(q1, q2, scale)
    
    return pos_diff, rot_diff


# ==========================================
# 测试函数
# ==========================================
def test_case_identity():
    """测试1：单位变换（无位移、无旋转）"""
    print("\n" + "="*60)
    print("测试1：单位变换（relative = identity）")
    print("="*60)
    
    current_pose = np.array([0.5, 0.2, 0.3, 0.0, 0.0, 0.707, 0.707])  # 绕 z 轴旋转 90°
    relative_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # 单位变换
    
    # 方法1：齐次矩阵右乘
    result_v1 = apply_relative_transform_v1(relative_pose, current_pose)
    
    # 方法2：位置加法 + 四元数左乘
    result_v2 = apply_transform_to_pose_v2(current_pose, relative_pose[:3], relative_pose[3:7])
    
    print(f"当前位姿: pos={current_pose[:3]}, quat={current_pose[3:7]}")
    print(f"相对变换: pos={relative_pose[:3]}, quat={relative_pose[3:7]}")
    print(f"方法1 (齐次右乘): pos={result_v1[:3]}, quat={result_v1[3:7]}")
    print(f"方法2 (加法+左乘): pos={result_v2[:3]}, quat={result_v2[3:7]}")
    print(f"位置差异: {np.linalg.norm(result_v1[:3] - result_v2[:3]):.6f}")
    print(f"四元数差异: {np.linalg.norm(result_v1[3:7] - result_v2[3:7]):.6f}")


def test_case_pure_translation():
    """测试2：纯平移（无旋转）"""
    print("\n" + "="*60)
    print("测试2：纯平移（relative = 只有平移）")
    print("="*60)
    
    current_pose = np.array([0.5, 0.2, 0.3, 0.0, 0.0, 0.707, 0.707])  # 绕 z 轴旋转 90°
    
    # 在当前末端坐标系下的相对平移 [0.1, 0, 0] (末端坐标系的 x 轴方向)
    relative_pose = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    
    # 方法1：齐次矩阵右乘 - relative 是在当前帧坐标系下
    result_v1 = apply_relative_transform_v1(relative_pose, current_pose)
    
    # 方法2：位置加法 - pos_diff 是在世界坐标系下
    result_v2 = apply_transform_to_pose_v2(current_pose, relative_pose[:3], relative_pose[3:7])
    
    print(f"当前位姿: pos={current_pose[:3]}, quat={current_pose[3:7]}")
    print(f"相对变换: pos={relative_pose[:3]}, quat={relative_pose[3:7]}")
    print(f"")
    print(f"方法1 (齐次右乘): pos={result_v1[:3]}")
    print(f"  -> 平移 [0.1, 0, 0] 是在末端坐标系下（绕z旋转90°后，x轴指向世界y轴）")
    print(f"  -> 因此世界坐标系下平移应该是 [0, 0.1, 0]")
    print(f"")
    print(f"方法2 (加法+左乘): pos={result_v2[:3]}")
    print(f"  -> 平移 [0.1, 0, 0] 直接加到世界坐标系位置")
    print(f"  -> 因此世界坐标系下平移是 [0.1, 0, 0]")
    print(f"")
    print(f"⚠️  位置差异: {np.linalg.norm(result_v1[:3] - result_v2[:3]):.6f}")
    print(f"结论：两种方法在纯平移时，坐标系语义不同！")


def test_case_pure_rotation():
    """测试3：纯旋转（无平移）"""
    print("\n" + "="*60)
    print("测试3：纯旋转（relative = 只有旋转）")
    print("="*60)
    
    current_pose = np.array([0.5, 0.2, 0.3, 0.0, 0.0, 0.707, 0.707])  # 绕 z 轴旋转 90°
    
    # 相对旋转：绕 z 轴再转 45°
    relative_quat = R.from_euler('z', 45, degrees=True).as_quat()
    relative_pose = np.array([0.0, 0.0, 0.0, *relative_quat])
    
    # 方法1：齐次矩阵右乘
    result_v1 = apply_relative_transform_v1(relative_pose, current_pose)
    
    # 方法2：位置加法 + 四元数左乘
    result_v2 = apply_transform_to_pose_v2(current_pose, relative_pose[:3], relative_pose[3:7])
    
    # 方法2变体：四元数右乘
    result_v2_right = apply_transform_to_pose_v2_right(current_pose, relative_pose[:3], relative_pose[3:7])
    
    current_euler = R.from_quat(current_pose[3:7]).as_euler('xyz', degrees=True)
    result_v1_euler = R.from_quat(result_v1[3:7]).as_euler('xyz', degrees=True)
    result_v2_euler = R.from_quat(result_v2[3:7]).as_euler('xyz', degrees=True)
    result_v2_right_euler = R.from_quat(result_v2_right[3:7]).as_euler('xyz', degrees=True)
    
    print(f"当前姿态 (euler): {current_euler}")
    print(f"相对旋转 (euler): z+45°")
    print(f"")
    print(f"方法1 (齐次右乘):   euler={result_v1_euler}")
    print(f"方法2 (四元数左乘): euler={result_v2_euler}")
    print(f"方法2 (四元数右乘): euler={result_v2_right_euler}")
    print(f"")
    print(f"方法1 与 方法2(右乘) 四元数差异: {np.linalg.norm(result_v1[3:7] - result_v2_right[3:7]):.6f}")
    print(f"方法1 与 方法2(左乘) 四元数差异: {np.linalg.norm(result_v1[3:7] - result_v2[3:7]):.6f}")


def test_case_joystick_scenario():
    """测试4：模拟 joystick 的实际使用场景"""
    print("\n" + "="*60)
    print("测试4：模拟 joystick 使用场景")
    print("="*60)
    
    # 手柄初始位姿（VR坐标系，已转换到机械臂坐标系）
    vr_init = np.eye(4)
    vr_init[:3, 3] = [0.0, 0.0, 0.0]
    vr_init[:3, :3] = np.eye(3)
    
    # 手柄当前位姿（用户移动后）
    vr_current = np.eye(4)
    vr_current[:3, 3] = [0.1, 0.05, 0.02]  # 移动了 (0.1, 0.05, 0.02)
    vr_current[:3, :3] = R.from_euler('z', 30, degrees=True).as_matrix()  # 绕z轴旋转30°
    
    # 机械臂初始末端位姿
    arm_init_pose = np.array([0.5, 0.2, 0.3, 0.0, 0.0, 0.707, 0.707])  # 绕z轴旋转90°
    
    # 计算手柄位姿差值（使用 joystick 的方法）
    scale = 0.4
    pos_diff, quat_diff = homogeneous_diff_with_scale(vr_init, vr_current, scale)
    
    print(f"VR初始位姿: pos={vr_init[:3,3]}, rot=identity")
    print(f"VR当前位姿: pos={vr_current[:3,3]}, rot=z+30°")
    print(f"缩放因子: {scale}")
    print(f"计算的差值: pos_diff={pos_diff}, quat_diff={quat_diff}")
    print(f"")
    print(f"机械臂初始: pos={arm_init_pose[:3]}, euler={R.from_quat(arm_init_pose[3:7]).as_euler('xyz', degrees=True)}")
    print(f"")
    
    # 方法2：joystick 使用的 apply_transform_to_pose（位置加法 + 四元数左乘）
    result_joystick = apply_transform_to_pose_v2(arm_init_pose, pos_diff, quat_diff)
    result_joystick_euler = R.from_quat(result_joystick[3:7]).as_euler('xyz', degrees=True)
    
    print(f"Joystick 方法结果:")
    print(f"  pos={result_joystick[:3]}")
    print(f"  euler={result_joystick_euler}")
    print(f"")
    
    # 如果 inference 要达到相同效果，relative_pose 需要如何定义？
    # 由于 inference 使用齐次矩阵右乘（末端坐标系下的相对变换）
    # 而 joystick 使用位置加法（世界坐标系下的差值）
    # 两者语义不同！
    
    print("="*60)
    print("关键发现：")
    print("="*60)
    print("1. inference_ros: T_target = T_current @ T_relative")
    print("   -> relative_pose 是在当前末端坐标系下的变换")
    print("")
    print("2. joystick/pose_diff: new_pos = pos + delta, new_rot = delta_rot * current_rot")
    print("   -> pos_diff 是在世界坐标系下的平移")
    print("   -> quat_diff 是在世界坐标系下的旋转（左乘施加于当前姿态）")
    print("")
    print("3. 如果训练数据是用 joystick 采集的，而模型输出是 relative_pose：")
    print("   -> 需要确保数据预处理时，relative_pose 的计算方式与推理时一致！")


def test_case_verify_equivalence_condition():
    """测试5：验证等价条件"""
    print("\n" + "="*60)
    print("测试5：验证等价条件")
    print("="*60)
    
    # 设置当前位姿
    current_pose = np.array([0.5, 0.2, 0.3, 0.0, 0.0, 0.707, 0.707])
    T_current = pose_to_transform_matrix(current_pose[:3], current_pose[3:7])
    
    # 设置目标位姿（世界坐标系下）
    target_pose_world = np.array([0.6, 0.25, 0.35, 0.0, 0.0, 0.866, 0.5])  # 不同位姿
    T_target = pose_to_transform_matrix(target_pose_world[:3], target_pose_world[3:7])
    
    # 计算 inference 需要的 relative_pose（末端坐标系下）
    # T_target = T_current @ T_relative => T_relative = T_current^-1 @ T_target
    T_relative = np.linalg.inv(T_current) @ T_target
    relative_pos = T_relative[:3, 3]
    relative_quat = R.from_matrix(T_relative[:3, :3]).as_quat()
    relative_pose_for_inference = np.concatenate([relative_pos, relative_quat])
    
    # 计算 joystick 需要的 pos_diff, quat_diff（世界坐标系下）
    pos_diff_world = target_pose_world[:3] - current_pose[:3]
    
    # R_target = R_diff @ R_current => R_diff = R_target @ R_current^T
    R_current = R.from_quat(current_pose[3:7])
    R_target = R.from_quat(target_pose_world[3:7])
    R_diff_world = R_target * R_current.inv()
    quat_diff_world = R_diff_world.as_quat()
    
    # 验证
    result_inference = apply_relative_transform_v1(relative_pose_for_inference, current_pose)
    result_joystick = apply_transform_to_pose_v2(current_pose, pos_diff_world, quat_diff_world)
    
    print(f"目标位姿（世界坐标系）: pos={target_pose_world[:3]}, quat={target_pose_world[3:7]}")
    print(f"")
    print(f"Inference 需要的 relative_pose（末端坐标系下）:")
    print(f"  pos={relative_pos}, quat={relative_quat}")
    print(f"")
    print(f"Joystick 需要的差值（世界坐标系下）:")
    print(f"  pos_diff={pos_diff_world}, quat_diff={quat_diff_world}")
    print(f"")
    print(f"Inference 结果: {result_inference[:3]}")
    print(f"Joystick 结果:  {result_joystick[:3]}")
    print(f"目标位姿:       {target_pose_world[:3]}")
    print(f"")
    print(f"位置误差 (inference): {np.linalg.norm(result_inference[:3] - target_pose_world[:3]):.6f}")
    print(f"位置误差 (joystick):  {np.linalg.norm(result_joystick[:3] - target_pose_world[:3]):.6f}")
    print(f"")
    print("结论：两种方法可以达到相同的目标位姿，但需要使用不同的输入格式！")
    print("  - inference: relative_pose 是在末端坐标系下计算的")
    print("  - joystick:  pos_diff/quat_diff 是在世界坐标系下计算的")


if __name__ == "__main__":
    print("="*60)
    print("旋转乘法顺序数学等价性验证")
    print("="*60)
    
    test_case_identity()
    test_case_pure_translation()
    test_case_pure_rotation()
    test_case_joystick_scenario()
    test_case_verify_equivalence_condition()
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("""
关键发现：

1. 两种方法的坐标系语义不同：
   - inference_ros: relative_pose 定义在当前末端坐标系下
   - joystick/pose_diff: pos_diff 定义在世界坐标系下

2. 对于相同的"目标位姿"，两种方法需要的输入不同：
   - inference: T_relative = T_current^-1 @ T_target
   - joystick: pos_diff = target[:3] - current[:3] (世界坐标系)

3. 数据采集与推理一致性：
   - 如果训练数据的 action 是用 joystick 的方式计算的（世界坐标系差值）
   - 但推理时使用 inference_ros 的方式（末端坐标系相对变换）
   - 则存在坐标系不匹配问题！

4. 建议的对齐方案：
   - 方案A: 修改数据预处理，确保 action 是末端坐标系下的相对变换
   - 方案B: 修改推理代码，使用与数据采集一致的变换方式
""")
