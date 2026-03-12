# 遥操作-训练-推理 全链路 Gap 分析报告

**日期**: 2026-03-12
**分析范围**: backend 遥操作 → record_data_ros 数据采集 → train_carm 训练 → inference_ros 推理

---

## 0. 执行摘要

经过对全链路代码的逐行分析，我发现了 **5 个确认的 gap** 和 **2 个潜在风险点**。其中最严重的是 **GAP-1（action 语义错配）**和 **GAP-2（scale 参数未传播到训练数据）**，它们会系统性地导致策略学到的 action 与推理时需要的 action 之间存在偏差。

| 编号 | 严重程度 | 问题 | 影响 |
|------|---------|------|------|
| GAP-1 | **致命** | action 记录的是规划层输出而非遥操作命令 | 训练目标不是真正的 action |
| GAP-2 | **严重** | 遥操作 scale=0.4 的语义在训练 action 中丢失 | 推理 teleop_scale 补偿方向可能错误 |
| GAP-3 | **严重** | 旋转增量计算方式不一致（左乘 vs SE(3)逆） | 旋转 action 存在系统性偏差 |
| GAP-4 | **中等** | 频率和延迟不匹配 | 动态行为与训练数据不一致 |
| GAP-5 | **低** | 夹爪处理不一致（连续 vs 离散） | 夹爪动作精度下降 |
| RISK-1 | **潜在** | 四元数约定不统一 | 可能导致旋转反向 |
| RISK-2 | **潜在** | 坐标系基准不确定 | 依赖标定模式正确 |

---

## 1. GAP-1: Action 语义错配（致命）

### 问题本质

数据采集记录的 `action` 不是遥操作者发出的真正控制信号，而是 SDK 规划层的输出。

### 详细分析

**遥操作时真正的 action**（backend `CommandController.compute_control()`）:

```python
# 遥操作命令生成过程:
delta_pos, delta_quat = homogeneous_diff_with_scale(init_vr, curr_vr, scale)  # ①
target_arm_pose = apply_transform_to_pose(init_arm, delta_pos, delta_quat)    # ②
carm_.track_pose(target_arm_pose[:7])                                          # ③ 真正的命令
```

这里的 action 是一个**绝对末端位姿目标** `target_arm_pose[:7]`，通过 `track_pose()` 发送给 SDK。

**数据采集记录的 action**（`env_ros.py:get_last_action()`）:

```python
def get_last_action(self):
    joint_cmd = self.arm.get_plan_joint_pos()     # SDK 内部规划的关节位置
    end_cmd = self.arm.forward_kine(0, joint_cmd)  # FK 计算末端位姿
    gripper = self.arm.get_gripper_pos()
    return [*joint_cmd, gripper, *end_cmd, gripper]  # 15D
```

`get_plan_joint_pos()` 返回的是 SDK 内部控制器经过逆运动学、轨迹规划、插值后的**规划关节位置**，而不是遥操作者发送的 `track_pose()` 目标。

### 差异来源

```
遥操作者意图 → track_pose(target[:7])
                    ↓
               SDK 内部处理:
               1. IK 求解 (可能有多解选择)
               2. 轨迹规划 (加减速曲线)
               3. 关节空间插值
               4. PID 控制器
                    ↓
               get_plan_joint_pos() ← 数据采集记录的
                    ↓
               实际关节运动
                    ↓
               get_joint_pos() / get_cart_pose() ← 也被记录为 observation
```

这意味着：
- 记录的 `action` 是经过 SDK 内部加工后的**中间结果**，不等于操作者的意图
- 由于 50Hz track_pose 和 200Hz 规划的频率差异，`get_plan_joint_pos()` 在两次 track_pose 之间会持续插值，导致记录的 action **过于平滑/超前/滞后**
- 更严重的是，训练时用 `obs` 和 `action` 做差分得到的相对位姿，其实是 `当前真实状态` 到 `SDK规划目标` 的相对位姿，而非 `当前状态` 到 `遥操作目标` 的相对位姿

### 数值估计

假设遥操作发送 `track_pose([0.30, 0.0, 0.20, 1, 0, 0, 0])`，在 SDK 内部经过 IK + 轨迹规划后，`get_plan_joint_pos()` 的 FK 结果可能是插值中间点 `[0.295, 0.001, 0.198, 0.999, 0.001, 0.01, 0.001]`。两者的末端位姿差异约 2-5mm 位移 + 0.5-2° 旋转。**在厘米级操作任务中这是显著误差**。

### 修复方案

**方案 A（推荐）: 直接记录 track_pose 命令**

修改 `env_ros.py` 或 `record_data_ros.py`，向 backend 请求当前的 `target_end_arm_pose`：
- backend 的 `CommandController.target_end_arm_pose` 就是发送给 `track_pose` 的值
- 可通过新增一个 `/api/joystick/target_pose` HTTP endpoint，或通过 SSE events 推送
- 采集端以 30Hz 轮询此接口

```python
# record_data_ros.py 中的修改:
def get_teleop_action(self):
    """从 backend API 获取遥操作目标位姿"""
    resp = requests.get(f"http://{self.robot_ip}:1999/api/joystick/target_pose")
    data = resp.json()
    return data['target_pose']  # [x,y,z,qx,qy,qz,qw]
```

**方案 B: 用 observation 差分近似**

如果无法修改 backend，可以从连续的 `qpos_end` 差分得到动作：
```python
action_approx_t = qpos_end[t+1] - qpos_end[t]  # 位移差分
```
缺点：受控制延迟和采样率影响，高频运动时误差大。

**方案 C: 用 LCM/ROS bridge 监听**

backend 的 `SurrealDataPublisher` 支持通过 LCM 发布 `/carm/end_cmd`，可直接订阅。

---

## 2. GAP-2: Scale 参数未传播到训练数据（严重）

### 问题分析

遥操作时，VR 手柄的位移/旋转被 `scale=0.4` 缩放后才发送给机械臂：

```python
# backend compute_control():
delta_pos, delta_quat = homogeneous_diff_with_scale(init_vr, curr_vr, scale=0.4)
# → pos_diff = (vr_pos_diff) * 0.4
# → rot_diff = slerp(identity, vr_rot_diff, 0.4)
```

这意味着遥操作数据中的动作幅度已经被 scale=0.4 缩放过了。

**训练时**：`CARMDataset` 从 `qpos_end` 计算相对位姿作为 action label，这些位姿是机械臂实际运动的结果，**已经包含了 scale 的效果**。模型学到的 action 分布反映的是 scale=0.4 下的运动幅度。

**推理时**：`inference_ros.py` 有一个 `teleop_scale` 参数，对模型输出的 rel_pose 再次应用 `apply_teleop_scale(rel_pose, teleop_scale)`：

```python
# inference_ros.py:
if self.teleop_scale != 1.0:
    for i in range(len(all_actions)):
        rel_pose = all_actions[i, rel_pose_start:rel_pose_end]
        scaled_rel_pose = apply_teleop_scale(rel_pose, self.teleop_scale)
        all_actions[i, rel_pose_start:rel_pose_end] = scaled_rel_pose
```

### 问题

1. 如果 `teleop_scale=0.4`，模型输出的 action（已经是 scale=0.4 下的幅度）会被**再次缩放**到 0.4×0.4=0.16 的幅度 → 动作过小
2. 如果 `teleop_scale=1.0`（默认值），则不缩放，此时模型输出的 action 幅度应该是正确的
3. 但 `teleop_scale` 的**设计意图**是做推理时速度调节——用户可能期望调小 scale 让机器人更慢更精确

### 语义混淆

`teleop_scale` 在 `pose_utils.py` 中的 `apply_teleop_scale()` 和 backend 中的 `homogeneous_diff_with_scale()` 做的事情不完全一样：

```python
# pose_utils.py (推理侧):
def apply_teleop_scale(delta_pose, scale):
    scaled_pose[:3] = delta_pose[:3] * scale          # 位移线性缩放
    scaled_pose[3:7] = slerp(identity, delta_quat, scale)  # 旋转 slerp

# backend pose_diff.py (遥操作侧):
def homogeneous_diff_with_scale(T1, T2, scale):
    pos_diff = (T2[:3,3] - T1[:3,3]) * scale          # 位移线性缩放 ✓
    R_diff = T2_rot @ T1_rot.T                         # 旋转差
    rot_diff = slerp(identity, R.from_matrix(R_diff).as_quat(), scale)  # 旋转 slerp ✓
```

两者的数学操作一致，但应用在不同的语境：一个是对 VR 手柄增量缩放，一个是对模型输出增量缩放。

### 修复方案

训练数据中的 action 来自连续帧的 `qpos_end` 差分，已经自然包含了 scale 的效果。因此：

1. **推理时 `teleop_scale` 应设为 1.0**（已是默认值），不额外缩放
2. 在 checkpoint 或 config 中**明确记录训练数据对应的 teleop scale**
3. 如需推理时调速，应使用新参数 `inference_speed_scale` 并明确文档其含义

---

## 3. GAP-3: 旋转增量计算方式不一致（严重）

### 遥操作侧（backend）

```python
# apply_transform_to_pose (pose_diff.py):
new_pos = pose[:3] + pos_diff               # 世界系加法
new_rot = R.from_quat(quat_diff) * R.from_quat(current_quat)  # 左乘: R_diff * R_current
```

这里是 **R_diff 左乘 R_current**，即在世界坐标系中施加旋转增量。

### 训练数据侧（carm_dataset.py）

```python
# compute_relative_pose_transform (pose_utils.py):
T_relative = T_current^{-1} @ T_target      # 右乘约定: T_target = T_current @ T_relative
```

这里的相对位姿遵循 **SE(3) 右乘约定**：`T_target = T_current @ T_relative`。

### 推理侧（inference_ros.py）

```python
# apply_relative_transform (pose_utils.py):
T_target = T_current @ T_relative           # 右乘: 在末端坐标系中施加变换
```

### 不一致性分析

| 阶段 | 旋转增量含义 | 公式 |
|------|------------|------|
| 遥操作 (backend) | 世界系旋转差 `R_diff * R_cur` | 左乘 |
| 训练标签 (pose_utils) | 末端系相对变换 `T_cur^{-1} @ T_tgt` | SE(3) 逆 |
| 推理执行 (pose_utils) | 末端系相对变换 `T_cur @ T_rel` | SE(3) 右乘 |

训练标签和推理执行是一致的（都是 SE(3) 右乘），这是正确的。

**但问题在于**：遥操作时 backend 的旋转增量是在世界系中施加的（左乘），而训练时我们从 observation 差分计算出来的 label 是 SE(3) 右乘约定。这两个**在纯旋转时是不等价的**：

- 世界系左乘: `R_new = R_diff @ R_cur`，其中 `R_diff` 与机械臂当前朝向无关
- 末端系右乘: `T_new = T_cur @ T_rel`，其中 `T_rel` 在末端坐标系中表达

当机械臂末端朝向不是单位姿态（即 R_cur ≠ I）时，同一个目标姿态 `R_new`，用世界系和末端系表达的增量是不同的：
```
世界系增量: R_world_diff = R_new @ R_cur^T
末端系增量: R_local_diff = R_cur^T @ R_new
一般情况: R_world_diff ≠ R_local_diff
```

**但这个 gap 被数据"吸收"了**：因为训练标签不是从遥操作命令得到的，而是从实际的 `qpos_end` 前后帧差分得到的。差分出来的就是 SE(3) 右乘语义，推理时也是 SE(3) 右乘语义，所以**训练-推理是自洽的**。

### 结论

GAP-3 在当前代码中**不构成实际 gap**——前提是训练 label 来自 observation 差分（而非遥操作命令）。但如果修复 GAP-1（改为记录遥操作命令），就需要将 backend 的左乘语义转换为 SE(3) 右乘语义，否则会引入新的不一致。

---

## 4. GAP-4: 频率和时间对齐不匹配（中等）

### 频率不匹配

| 环节 | 频率 | 说明 |
|------|------|------|
| BLE 手柄数据 | ~100Hz | 蓝牙通知频率 |
| backend 控制循环 | 50Hz | `time.sleep(0.02)` |
| backend track_pose 发送 | 50Hz | 每次 loop 一次 |
| 数据采集 | 30Hz | `rospy.Rate(record_freq)` |
| SDK 内部规划 | 200Hz | `get_plan_joint_pos()` 更新频率 |
| 推理线程 | ~10-30Hz | 取决于模型和硬件 |
| 推理控制线程 | 200Hz | 发送 track_pose |

### 问题

1. **采集 30Hz < 控制 50Hz**: 每采一帧，遥操作已发了 1-2 个控制命令，部分中间动作被跳过
2. **采集的 action 来自 200Hz 规划**: 而 observation 是 30Hz，两者的时间戳不严格对齐
3. **推理控制 200Hz vs 遥操作控制 50Hz**: 推理时发送 track_pose 的频率是遥操作的 4 倍，SDK 内部的响应特性可能不同

### timeline 日志分析

`record_data_ros.py` 已经有 `timeline_logger` 记录 `obs_stamp_ros` 和 `t_action_query_sys` 的时间差，可以用已有数据量化这个 gap:
```python
delta_action_obs = t_action_query_sys - obs_stamp_ros  # action 和 obs 的时间差
```

### 修复方案

1. **提高采集频率到 50Hz**: 与遥操作控制循环对齐
2. **记录精确时间戳**: 在 action 中附加时间戳，训练时做插值对齐
3. **推理时匹配频率**: 控制线程频率可调，默认与训练数据一致（50Hz 或 30Hz）

---

## 5. GAP-5: 夹爪处理不一致（低）

### 遥操作侧

```python
# backend: 连续值
gripper_pose = (1 - trigger/127) * 0.08  # 范围 [0, 0.08]，连续映射
```

### 数据记录

```python
# 记录为连续值
gripper = arm.get_gripper_pos()  # 0 ~ 0.08
```

### 训练侧

```python
# 离散化: 二值分类
gripper_label = 1 if gripper_val < 0.05 else 0  # threshold=0.05
# 训练: CrossEntropyLoss
```

### 推理侧

```python
# 离散→连续: 二选一
gripper = 0.04 if predicted_class==1 else 0.078  # 硬编码两个值
```

### 问题

- 遥操作时夹爪是**连续控制**（任意开度），但训练后变成**半开/全开两档**
- threshold=0.05 的选择将 [0, 0.05) → close, [0.05, 0.08] → open
- 推理时 close=0.04, open=0.078 与阈值 0.05 不完全对应
- 实际操作中"半抓"状态完全丢失

### 修复方案

如果任务只需要 open/close 两种状态，当前方案够用。否则应改为连续夹爪预测。

---

## 6. RISK-1: 四元数约定潜在风险

### 约定统计

| 模块 | 四元数顺序 | 来源 |
|------|-----------|------|
| SDK `get_cart_pose()` | `[qx, qy, qz, qw]` | C++ 端确认 |
| SDK `track_pose()` | `[qx, qy, qz, qw]` | C++ 端确认 |
| BLE `PoseData` | `(qx, qy, qz, qw)` | `surreal_data_model.py` 定义 |
| scipy `R.from_quat()` | `[qx, qy, qz, qw]` | scipy 约定 |
| `pose_utils.py` | `[qx, qy, qz, qw]` | 注释明确 |
| `pose_diff.py` | `[qx, qy, qz, qw]` | scipy 约定 |

各模块约定一致，但**没有运行时断言**验证四元数是否归一化，也没有检查 `qw` 的符号。建议添加 `assert abs(np.linalg.norm(q) - 1.0) < 0.01` 断言。

---

## 7. RISK-2: 坐标系标定模式依赖

遥操作使用 `calibration_mode`（4 种之一）将 VR 手柄坐标系映射到机械臂坐标系。这个参数：
- 在 backend 中通过 `/api/joystick/calibration` 设置
- **没有记录在 HDF5 数据中**
- 如果标定模式错误或不一致，所有数据的动作方向会系统性偏转

---

## 8. 全链路数据流对比图

```
                    遥操作链路 (真实)                训练链路 (当前)                  推理链路
                    ================                ==============                  ========

用户意图         VR手柄 pose[xyz,qxyzw]
                        │
坐标变换          surreal_to_arm_projection()
                        │
缩放              homogeneous_diff * scale=0.4
                        │
积分              apply_transform_to_pose()          ╳ 未记录
                        │
SDK 命令           track_pose(target[:7])            ╳ 未记录                   track_pose(target[:7])
                        │                                                            ▲
SDK 规划          IK → 轨迹规划 → 插值                                               │
                        │                                                            │
规划输出          get_plan_joint_pos()                                                │
                        │                                                            │
FK               forward_kine() → end_cmd                                             │
                        │                                                            │
记录的action      [joints, gripper, end, gripper]     ← 15D                          │
                                                       │                              │
实际状态          get_joint_pos()                       │                              │
                  get_cart_pose()                       │                              │
                        │                              │                              │
记录的obs         qpos_joint, qpos_end                 │                              │
                                                       ▼                              │
                                                  CARMDataset:                        │
                                                  relative_pose =                     │
                                                  inv(qpos_end[t]) @                  │
                                                    raw_action[7:14]                  │
                                                       │                              │
                                                       ▼                              │
                                                  normalize()                         │
                                                       │                              │
                                                       ▼                              │
                                                  训练 → model                        │
                                                       │                              │
                                                       ▼                              │
                                                  推理 →  inverse_normalize()         │
                                                       │                              │
                                                       ▼                              │
                                                  apply_teleop_scale()                │
                                                       │                              │
                                                       ▼                              │
                                                  apply_relative_transform()  ────────┘
                                                  = T_cur @ T_rel → target
```

## 9. 修复优先级

| 优先级 | Gap | 修复方案 | 状态 |
|--------|-----|---------|------|
| **P0** | GAP-1 | backend 新增 `/api/joystick/teleop_target` API | **已修复** |
| **P0** | GAP-1 | record_data_ros 改为记录遥操作目标 (8D action) | **已修复** |
| **P1** | GAP-2 | `teleop_scale` 固定 1.0，新增 `inference_speed_scale` | **已修复** |
| **P1** | GAP-3 | 直接记录绝对目标位姿，SE(3) 逆自然吸收旋转约定差异 | **已修复（无需转换）** |
| **P2** | GAP-4 | 采集频率提到 50Hz + 时间戳对齐 | 待做 |
| **P3** | GAP-5 | 评估是否需要连续夹爪 | 视任务而定 |

## 10. 补充说明: 为什么当前策略可能"看起来还行"

尽管存在上述 gap，当前策略在简单任务上可能仍有一定表现。原因：

1. **Gap-1 的 action label 虽然不精确，但大方向正确**：SDK 的规划输出是在向目标位姿运动，FK 结果与真实目标的偏差在毫米——对于粗粒度任务足够
2. **Gap-2 被默认参数规避**：teleop_scale 默认 1.0，不额外缩放
3. **Gap-3 实际不存在**：训练-推理使用同一套 SE(3) 右乘约定
4. **闭环修正**：推理时每步重新观测+重新预测，误差不会无限累积

但对于**精细操作**（如 mm 级对位、旋转敏感任务），这些 gap 会成为性能瓶颈。
