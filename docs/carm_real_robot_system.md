# CARM 真机部署系统架构文档

## 1. 系统总览

CARM 真机部署系统由四大模块组成，覆盖从数据采集到模型部署的完整闭环：

```
┌────────────────────────────────────────────────────────────────────┐
│                     CARM 真机系统架构                              │
│                                                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐   │
│  │   backend    │   │arm_control_sdk│  │   carm_ros_deploy    │   │
│  │  (Flask IPC) │   │  (carm_py)   │   │   (ROS1 catkin)     │   │
│  │   Port 1999  │   │  TCP→机械臂  │   │ 数据采集+模型推理   │   │
│  └──────┬───────┘   └──────┬───────┘   └──────────┬───────────┘   │
│         │                  │                       │               │
│  Surreal Touch ─BLE→ 坐标变换 ─┐   ROS topics ────┤               │
│                                │                   │               │
│                    ┌───────────▼───────────────────▼──────┐        │
│                    │          CARM 机械臂 (6DOF)          │        │
│                    │         IP: 10.42.0.101              │        │
│                    │     EtherCAT → 电机驱动 → 关节       │        │
│                    └─────────────────────────────────────-┘        │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │              rlft/ (训练+评估)                            │     │
│  │  offline/train_carm.py ← datasets/carm_dataset.py       │     │
│  │  utils/pose_utils.py (训练/推理共享坐标变换)              │     │
│  └──────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────┘
```

## 2. arm_control_sdk — 机械臂控制 SDK

C++ 库 + pybind11 Python 绑定 (`carm_py`)。

### 2.1 硬件规格
- 6-DOF 协作机械臂 + 平行夹爪
- 通信：TCP/IP（默认 `10.42.0.101:8090`）
- 底层：EtherCAT → 伺服驱动

### 2.2 核心 API

| 方法 | 用途 | 阻塞 |
|------|------|------|
| `CArmSingleCol(ip)` | 创建连接实例 | - |
| `set_ready()` | 使能伺服 | - |
| `set_control_mode(mode)` | 设置控制模式 (0=IDLE, 2=MIT, 3=Drag, 4=PF) | - |
| `get_joint_pos()` → `[6]` | 读取 6 轴关节角 (rad) | 否 |
| `get_cart_pose()` → `[7]` | 读取末端位姿 [x,y,z,qx,qy,qz,qw] | 否 |
| `get_gripper_pos()` → `float` | 读取夹爪开度 (0-0.08m) | 否 |
| `track_pose([7])` | 连续位姿跟踪 (用于 50Hz/200Hz 控制) | 否 |
| `track_joint([6])` | 连续关节跟踪 | 否 |
| `move_pose([7])` | 运动到目标位姿 | 是 |
| `move_joint([6])` | 运动到目标关节角 | 是 |
| `set_gripper(pos, tau)` | 设置夹爪 (pos: 0-0.08m, tau: 0-20N) | 否 |
| `forward_kine(0, [6])` → `([6], [7])` | 正运动学 | 否 |
| `emergency_stop()` | 紧急停止 | - |

### 2.3 控制模式
- **Mode 1 (Position)**: 已被 `env_ros.py` **禁用**，强制切换到 Mode 2
- **Mode 2 (MIT)**: 力矩控制模式，推理时默认使用
- **Mode 3 (Drag)**: 拖动示教模式
- **Mode 4 (PF)**: 力位混合模式，遥操作时使用

## 3. backend — 下位机 Flask 后端

运行在机器人 IPC (ARM aarch64) 上的中间件，提供 HTTP REST API + SSE 实时推送。

### 3.1 遥操作数据流（关键路径）

```
Surreal Touch (BLE VR 手柄)
    │  BLE 通知 (76 bytes: timestamp + xyz + qxyzw + vel + acc)
    ▼
SurrealSdk._notification_handler()
    │  struct.unpack → PoseData
    ▼
JoystickState (线程安全共享状态)
    │  50Hz 采样
    ▼
CommandController.surreal_callback()
    │  1. 解析按钮: trigger(夹爪), grip(离合), A(回零), B(锁旋转)
    │  2. 构建 4×4 齐次矩阵: [xyz, R.from_quat(qxyzw)]
    │  3. surreal_to_arm_projection(T, calibration_mode)  ← 坐标系变换
    ▼
CommandController.compute_control()
    │  1. 离合检查: grip > 100 才控制
    │  2. 增量计算: homogeneous_diff_with_scale(init_vr, curr_vr, scale)
    │     - pos_diff = (T2[:3,3] - T1[:3,3]) * scale
    │     - rot_diff = slerp(identity, R_diff, scale)    ← scale 缩放!
    │  3. 积分: target = apply_transform_to_pose(init_arm, delta_pos, delta_quat)
    │  4. 夹爪: (1 - trigger/127) * 0.08
    ▼
CommandController.loop() — 50Hz
    │  carm_.track_pose(target[:7])
    │  carm_.set_gripper(pose, tau=10)
    ▼
机械臂执行
```

### 3.2 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `scale` | 0.4 | 遥操作灵敏度，影响位移和旋转幅度 |
| `calibration_mode` | X_LEFT (1) | 手柄坐标系到机械臂坐标系的映射方式 |
| `gripper_max_value` | 0.08m | 夹爪最大开度 |
| `gripper_tau` | 10N | 夹爪力矩 |
| 控制频率 | 50Hz | `time.sleep(0.02)` |

### 3.3 坐标变换

`surreal_to_arm_projection(T, mode)` 通过投影矩阵将 VR 手柄坐标变换到机械臂坐标系：

| 模式 | 投影矩阵 R |
|------|-----------|
| X_LEFT | `[[0,0,-1],[1,0,0],[0,1,0]]` |
| X_FORWARD | `[[1,0,0],[0,0,1],[0,1,0]]` |
| X_RIGHT | `[[0,0,1],[-1,0,0],[0,1,0]]` |
| X_BACKWARD | `[[-1,0,0],[0,0,-1],[0,1,0]]` |

### 3.4 增量计算方式

`homogeneous_diff_with_scale(T_init_vr, T_curr_vr, scale)`:
- **位移差**: `pos_diff = (T2[:3,3] - T1[:3,3]) * scale`
- **旋转差**: `R_diff = T2_rot @ T1_rot.T` → 转四元数 → `slerp(identity, q_diff, scale)`

`apply_transform_to_pose(init_arm_pose, pos_diff, quat_diff)`:
- **位移**: `new_pos = pose[:3] + pos_diff` (世界坐标系加法)
- **旋转**: `new_rot = R.from_quat(quat_diff) * R.from_quat(pose[3:])` (scipy 四元数左乘)

## 4. carm_ros_deploy — ROS1 数据采集与推理

### 4.1 数据采集 (`record_data_ros.py`)

**被动模式**：不干扰手柄遥操作，只读取机械臂状态。

采集的数据（30Hz）：

| 字段 | 维度 | 来源 |
|------|------|------|
| `images` | `[T, H, W, 3]` | ROS 相机话题 `/camera/color/image_raw` |
| `qpos_joint` | `[T, 7]` | `arm.get_joint_pos() + [gripper]` — 6 关节 + 夹爪 |
| `qpos_end` | `[T, 8]` | `arm.get_cart_pose() + [gripper]` — 7 EE位姿 + 夹爪 |
| `timestamps` | `[T]` | ROS 时间戳 |
| `action` | `[T, 15]` | `get_last_action()` — 见下文 |
| `gripper` | `[T]` | `get_gripper_pos()` |

`get_last_action()` 的实现 (`env_ros.py:262-280`):
```python
def get_last_action(self):
    joint_cmd = self.arm.get_plan_joint_pos()  # 规划的关节位置[6]
    end_cmd = self.arm.forward_kine(0, joint_cmd)[1]  # FK 得到末端位姿[7]
    gripper = self.arm.get_gripper_pos()
    return [*joint_cmd, gripper, *end_cmd, gripper]  # 15D
```

**关键问题**：`action` 记录的是 SDK 规划层输出的关节命令（经过 FK 得到末端位姿），而**不是**遥操作者发出的原始 `track_pose()` 命令。

### 4.2 策略推理 (`inference_ros.py`)

双线程架构：
- **推理线程** (30Hz)：获取观测 → 图像预处理 → 策略前向 → 动作后处理 → action chunk
- **控制线程** (200Hz)：从 action chunk 取动作 → 安全检查 → `track_pose(target[:7])`

推理流水线：
```
观测获取 → 图像 resize(128×128) → CHW → 策略推理
                                       ↓
                            model output: [pred_horizon, action_dim]
                            - full mode: action_dim=13 (joints[6] + rel_pose[7])
                            - ee_only:   action_dim=7  (rel_pose[7])
                                       ↓
                            inverse normalize (ActionNormalizer)
                                       ↓
                            apply_teleop_scale(rel_pose, scale)  ← 参数对齐!
                                       ↓
                            safety clip (关节限位 / 工作空间 / 增量限制)
                                       ↓
                            apply_relative_transform(rel_pose, current_ee, gripper)
                            → target_absolute_pose [x,y,z,qx,qy,qz,qw,gripper]
                                       ↓
                            track_pose(target[:7]) + set_gripper(target[7])
```

### 4.3 安全控制器 (`safety_controller.py`)

4 层安全机制：
1. **关节限位**: 官方限位 + 10% 裕度
2. **动作增量限制**: 每步最大关节变化 0.1rad, 夹爪 0.02m
3. **工作空间边界**: X[0.10, 0.50], Y[-0.30, 0.30], Z[0.05, 0.40]
4. **低通滤波**: α=0.3 指数移动平均

## 5. rlft/ — 训练代码

### 5.1 数据集构建 (`carm_dataset.py`)

从 HDF5 加载后，action 处理流程：

```
raw_action[15] = [joints(6), gripper(1), end_pose(7), gripper(1)]
    ↓
target_pose = raw_action[7:14]   # 末端位姿 7D
ref_pose = qpos_end[t, :7]       # 当前帧末端位姿
    ↓
relative_pose = compute_relative_pose_transform(ref_pose, target_pose)
    = T_ref^{-1} @ T_target   # SE(3) 相对变换
    ↓
if action_mode == 'full':
    action = [raw_joints(6), relative_ee_pose(7)]  # 13D
elif action_mode == 'ee_only':
    action = relative_ee_pose                        # 7D
    ↓
gripper: raw_action[14] → 二值分类 (threshold=0.05)
    ↓
action_normalizer.transform(action)  # 标准化
```

### 5.2 训练入口 (`train_carm.py`)

- 算法: diffusion_policy / flow_matching / shortcut_flow / consistency_flow / reflected_flow
- 视觉编码器: PlainConv / ResNet10-50
- 状态编码器: MLP (可选)
- 离散夹爪: CrossEntropyLoss + class weighting

### 5.3 推理时动作还原 (`inference_ros.py`)

```
model output: [pred_horizon, action_dim]
    ↓
action_normalizer.inverse_transform(action)
    ↓
rel_pose[7] = action[7:14]  (full mode) 或 action[0:7]  (ee_only mode)
    ↓
apply_teleop_scale(rel_pose, teleop_scale)   # delta 缩放
    ↓
target = apply_relative_transform(rel_pose, current_ee)
       = T_current @ T_relative
    ↓
track_pose(target[:7])
```

## 6. 已有数据

| 目录 | 内容 |
|------|------|
| `recorded_data/` | 49 条遥操作轨迹 (2026-01-12) |
| `inference_logs/` | 20+ 条推理记录 (含 model_output + intervention) |
