# 手柄遥操作 vs 推理脚本 控制逻辑对比分析报告

> 生成日期：2026-01-24  
> 目的：分析 `joystick.py` 手柄遥操作与 `inference_ros.py` 推理脚本的控制逻辑差异，为统一/对齐两者逻辑提供依据

---

## 1. 概述

| 维度 | joystick.py (手柄遥操) | inference_ros.py (模型推理) |
|------|------------------------|----------------------------|
| **文件位置** | `archive/carm_backend/server/basic/joystick.py` | `carm_ros_deploy/src/carm_deploy/inference/inference_ros.py` |
| **动作来源** | VR手柄传感器实时数据 | 神经网络策略模型推理 |
| **控制频率** | 50Hz | 推理30Hz + 控制200Hz (双线程) |
| **核心类** | `CommandController` | `InferenceNode` + `RealPolicy` |

---

## 2. 位姿变换方法对比（核心差异！）

### 2.1 inference_ros.py 的方法

```python
# 文件: carm_ros_deploy/src/carm_deploy/inference/inference_ros.py

def apply_relative_transform(relative_pose, current_pose, gripper):
    """
    将相对位姿变换应用到当前位姿，得到目标绝对位姿
    
    计算公式: target_pose = current_pose @ relative_transform
    """
    T_relative = pose_to_transform_matrix(relative_pose[:3], relative_pose[3:])
    T_current = pose_to_transform_matrix(current_pose[:3], current_pose[3:])
    
    # target = current @ relative  (齐次矩阵右乘)
    T_target = T_current @ T_relative
    
    target_position = T_target[:3, 3]
    target_quat = R.from_matrix(T_target[:3, :3]).as_quat()
    
    return target_position.tolist() + target_quat.tolist() + [gripper]
```

**语义**：`relative_pose` 定义在**当前末端坐标系**下的相对变换。

### 2.2 joystick.py / pose_diff.py 的方法

```python
# 文件: archive/carm_backend/dependencies/utils/pose_diff.py

def apply_transform_to_pose(pose, pos_diff, quat_diff, swap_x=False):
    """
    将差值应用到目标位姿
    """
    # 更新位置：直接加法（世界坐标系）
    new_pos = pose[:3] + pos_diff
    
    # 更新姿态：四元数左乘
    current_quat = pose[3:]
    diff_quat = R.from_quat(quat_diff)
    current_rot = R.from_quat(current_quat)
    new_rot = diff_quat * current_rot  # 左乘
    new_quat = new_rot.as_quat()
    
    return np.concatenate([new_pos, new_quat])
```

**语义**：`pos_diff` 定义在**世界坐标系**下的平移差值。

### 2.3 数学验证结果

通过 [test_rotation_equivalence.py](scripts/test_rotation_equivalence.py) 验证：

```
测试2：纯平移（relative = 只有平移）
============================================================
当前位姿: pos=[0.5 0.2 0.3], quat=[0. 0. 0.707 0.707] (绕z轴旋转90°)
相对变换: pos=[0.1 0. 0.], quat=[0. 0. 0. 1.]

方法1 (齐次右乘): pos=[0.5 0.3 0.3]
  -> 平移 [0.1, 0, 0] 是在末端坐标系下
  -> 因此世界坐标系下平移是 [0, 0.1, 0]

方法2 (加法+左乘): pos=[0.6 0.2 0.3]
  -> 平移 [0.1, 0, 0] 直接加到世界坐标系位置

⚠️ 位置差异: 0.141421m
结论：两种方法坐标系语义不同！
```

### 2.4 关键结论

| 特征 | inference_ros.py | joystick.py |
|------|------------------|-------------|
| **位置变换** | 末端坐标系下的相对位移 | 世界坐标系下的绝对位移差 |
| **旋转变换** | 齐次矩阵右乘 | 四元数左乘 |
| **等价条件** | `T_relative = T_current^-1 @ T_target` | `pos_diff = target[:3] - current[:3]` |

**✅ 好消息**：训练数据预处理 (`compute_relative_pose_transform`) 使用的公式与推理时 (`apply_relative_transform`) 一致：
```python
# carm_utils.py
T_relative = np.linalg.inv(T_current) @ T_target  # 训练时计算 relative_pose
# inference_ros.py  
T_target = T_current @ T_relative  # 推理时恢复 target_pose
```

---

## 3. 控制流程对比

### 3.1 joystick.py 控制流程

```
VR手柄数据 → surreal_to_arm_projection (坐标系转换)
                      ↓
              homogeneous_diff_with_scale (计算差值并缩放)
                      ↓
              apply_transform_to_pose (应用到机械臂初始位姿)
                      ↓
              target_end_arm_pose (目标位姿)
                      ↓
              carm.track_pose() (50Hz 控制)
```

**关键代码** (joystick.py L388-405):
```python
def compute_control(self, end_vr_pose, flag_squeeze, ...):
    # 离合逻辑: 松开不控制，同时记录初始位姿
    if not flag_squeeze:
        self.init_end_vr_pose = end_vr_pose.copy()
        self.init_end_arm_pose = self.end_arm_pose[:7]
        return

    # 增量计算
    delta_pos, delta_quat = homogeneous_diff_with_scale(
        self.init_end_vr_pose, end_vr_pose, self.scale  # scale=0.4
    )

    # 积分：应用到初始位姿
    self.target_end_arm_pose = apply_transform_to_pose(
        self.init_end_arm_pose, delta_pos, delta_quat
    )
```

### 3.2 inference_ros.py 控制流程

```
观测数据 (图像+状态) → 模型推理 (30Hz)
                           ↓
                    相对位姿 (relative_pose)
                           ↓
              apply_relative_transform (转换为绝对位姿)
                           ↓
              ActionChunkManager (时间加权融合)
                           ↓
              env.end_control_nostep() (200Hz 控制)
```

**关键代码** (inference_ros.py L1280-1300):
```python
if not self.joint_cmd_mode:
    all_endactions = []
    for i in range(all_actions.shape[0]):
        relative_pose = all_actions[i][rel_pose_start:rel_pose_end]
        grip = all_actions[i][gripper_idx]
        
        # 将相对位姿变换应用到当前位姿，得到目标绝对位姿
        target_pose = apply_relative_transform(relative_pose, qpos_end[:7], grip)
        all_endactions.append(target_pose)
    all_actions = np.array(all_endactions)
```

---

## 4. 夹爪控制对比

### 4.1 joystick.py 夹爪控制（连续映射）

```python
# joystick.py L400-404
# trigger 值 [0, 127] → 夹爪开度 [0, 0.08]
gripper_val = (1.0 - value_trigger/127.0) * self.gripper_max_value  # max=0.08
gripper_val = max(0.0, min(gripper_val, self.gripper_max_value))
self.end_gripper_pose = gripper_val
```

**特点**：
- 连续控制，trigger 压下程度线性映射到夹爪开度
- 完全松开 trigger (=0) → 夹爪全开 (0.08m)
- 完全压下 trigger (=127) → 夹爪全闭 (0m)

### 4.2 inference_ros.py 夹爪控制（离散分类 + 滞后平滑）

```python
# inference_ros.py L650-720
# 模型输出 2 分类概率
gripper_logits = self.gripper_head(obs_features)  # [1, pred_horizon, 2]
gripper_cls = gripper_logits.argmax(dim=-1)  # 0=open, 1=close

# 滞后处理：5帧多数投票 + "any close in act_horizon" 逻辑
def _apply_gripper_hysteresis(self, gripper_cls):
    act_horizon = min(8, len(gripper_cls))
    chunk_has_close = np.any(gripper_cls[:act_horizon] == 1)
    current_vote = 1 if chunk_has_close else 0
    
    self._gripper_history.append(current_vote)
    
    # 多数投票
    if len(self._gripper_history) >= 3:
        vote_result = sum(self._gripper_history) > len(self._gripper_history) / 2
        new_state = 1 if vote_result else 0
    
    # 映射到连续值
    gripper_val = self.gripper_close_val if new_state == 1 else self.gripper_open_val
    # 默认: open=0.078, close=0.04, threshold=0.05
```

**特点**：
- 离散预测 (open/close) + 滞后平滑
- 安全策略：只要 act_horizon 内有任意帧预测 close，就投 close
- 防止快速切换：5帧多数投票

### 4.3 夹爪一致性分析

| 维度 | joystick (采集) | inference (推理) |
|------|-----------------|-----------------|
| **输入** | trigger [0, 127] | 模型离散分类 [0, 1] |
| **输出范围** | [0, 0.08]m 连续 | {0.04, 0.078}m 离散 |
| **open 值** | 0.08m | 0.078m |
| **close 值** | 0m (理论) | 0.04m |
| **阈值** | 无 | 0.05m |

**数据预处理**（训练时）:
```python
# train_carm.py
gripper_threshold = args.gripper_threshold  # 默认 0.05
gripper_label = 1 if gripper_value < gripper_threshold else 0  # 0=open, 1=close
```

---

## 5. 坐标系转换

### 5.1 joystick.py 坐标系转换

```python
# rotation_utils.py L48-70
def surreal_to_arm_projection(homogeneous_matrix, calibration_mode):
    """将 surreal 手柄坐标系变换至机械臂坐标系"""
    T = np.eye(4)
    if calibration_mode == CalibrationMode.X_LEFT.value:
        T[:3, :3] = [[0,0,-1],
                     [1,0,0],
                     [0,1,0]]
    elif calibration_mode == CalibrationMode.X_FORWARD.value:
        T[:3, :3] = [[1,0,0],
                     [0,0,1],
                     [0,1,0]]
    # ...
    transformed_matrix = T @ homogeneous_matrix  # 左乘坐标变换
    return transformed_matrix
```

### 5.2 inference_ros.py 坐标系转换

**无显式坐标系转换**：假设模型输出已经在机械臂坐标系下。

---

## 6. 控制接口对比

### 6.1 joystick.py 控制接口

```python
# joystick.py L411-420
def loop(self):
    """控制循环（50Hz）"""
    while not self.isStop:
        if self.target_end_arm_pose is not None:
            self.carm_.track_pose(self.target_end_arm_pose[:7])  # 末端位姿跟踪
            if self.end_gripper_pose is not None:
                self.carm_.set_gripper(self.end_gripper_pose, self.end_gripper_tau)
        time.sleep(0.02)  # 50Hz
```

### 6.2 inference_ros.py 控制接口

```python
# env_ros.py L282-290
def end_control_nostep(self, action):
    """末端空间控制（不阻塞）"""
    self.arm.track_pose(list(action[:7]))
    self.arm.set_gripper(action[-1], self.tau)
```

**✅ 两者最终都调用相同的底层接口**：`arm.track_pose()` + `arm.set_gripper()`

---

## 7. record_data_ros.py 数据采集分析

### 7.1 当前采集的数据

```python
# record_data_ros.py
self.episode_data['action'].append(action)  # 通过 env.get_last_action() 获取

# env_ros.py
def get_last_action(self):
    end_cmd = self.arm.forward_kine(0, self.joint_cmd)[1]  # 通过正运动学计算
    gripper = self.arm.get_gripper_pos()
    joints_cmd = list(self.joint_cmd) + [gripper]
    end_cmd = list(end_cmd) + [gripper]
    return np.concatenate([joints_cmd, end_cmd], axis=0)  # [15D]
```

**问题**：`get_last_action()` 返回的是**当前实际位姿/关节角**，不是 joystick 计算的**目标位姿**！

### 7.2 joystick 真实的 action

joystick 计算的 action 是 `target_end_arm_pose`（通过 `apply_transform_to_pose` 计算），但这个值没有被 `record_data_ros.py` 捕获。

---

## 8. 差异总结

| 维度 | joystick.py | inference_ros.py | 是否一致 |
|------|-------------|------------------|----------|
| **位姿变换坐标系** | 世界坐标系差值 | 末端坐标系相对变换 | ⚠️ 不同 |
| **位姿变换公式** | `new_pos = pos + delta` | `T_target = T_current @ T_relative` | ⚠️ 不同 |
| **训练数据预处理** | N/A | `T_relative = T_current^-1 @ T_target` | ✅ 与推理一致 |
| **夹爪控制** | 连续映射 [0, 0.08] | 离散分类 + 滞后 | ⚠️ 不同 |
| **坐标系转换** | surreal→arm 投影 | 无 | ⚠️ 不同 |
| **控制底层接口** | `track_pose()` | `track_pose()` | ✅ 一致 |
| **控制频率** | 50Hz | 200Hz | 不同但无影响 |

---

## 9. 建议

### 9.1 对于数据采集 (record_data_ros.py)

**现状**：采集的 action 是通过 `forward_kine` 计算的实际末端位姿，不是 joystick 的目标位姿。

**影响**：如果要对齐 joystick 的控制逻辑，需要：
- 方案A：修改 joystick 发布 `target_end_arm_pose` 到 ROS topic
- 方案B：直接在 joystick 所在进程中采集数据

### 9.2 对于训练数据预处理

**现状**：`compute_relative_pose_transform` 计算的是**末端坐标系下的相对变换**，与 `inference_ros.py` 的 `apply_relative_transform` 一致。

**✅ 无需修改**：训练数据预处理与推理已对齐。

### 9.3 对于推理脚本

如果希望推理时使用 joystick 风格的变换（世界坐标系差值），需要：
- 修改模型输出为世界坐标系下的 `pos_diff, quat_diff`
- 修改 `apply_relative_transform` 使用加法+左乘

**不建议**：这会破坏现有的训练-推理一致性。

---

## 10. 附录：验证脚本

详见 [scripts/test_rotation_equivalence.py](scripts/test_rotation_equivalence.py)

运行方式：
```bash
conda run -n carm python scripts/test_rotation_equivalence.py
```
