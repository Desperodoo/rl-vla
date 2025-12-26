# ARX5 策略推理调试记录

## 概述

本文档记录在开发 `arx5_policy_inference.py` 策略推理脚本时遇到的问题及解决方案。

## Bug 1: 机械臂完全不动（kp=0 问题）

### 现象
- 运行推理脚本后，机械臂完全不响应位置命令
- 日志显示命令在发送，但机械臂纹丝不动
- 退出时日志显示：`Current kp is zero. Setting to default kp kd`

### 根因
ARX5 SDK 的 `Arx5JointController` 在初始化后，位置增益 **kp=0**。只有调用 `reset_to_home()` 后才会设置正确的 kp/kd 增益。

当代码中使用 `--init-pose` 参数时，直接调用 `_move_to_pose()` 而跳过了 `reset_to_home()`，导致：
- kp=0，位置控制器不工作
- 命令发送了但机械臂不响应

### 解决方案
**始终先调用 `reset_to_home()`**，无论是否指定了初始位姿：

```python
# 错误做法
if self.config.task_initial_pose is not None:
    self._move_to_pose(...)  # kp=0, 不工作！
else:
    self.robot_ctrl.reset_to_home()

# 正确做法
self.robot_ctrl.reset_to_home()  # 先设置 kp/kd
if self.config.task_initial_pose is not None:
    self._move_to_pose(...)  # 现在可以工作
```

---

## Bug 2: send_recv_once() 必须调用

### 现象
- 使用简化构造函数 `Arx5JointController("X5", "can0")` 初始化
- 只调用 `set_joint_cmd()` + `sleep()`
- 日志显示 "Background send_recv task is running"
- 但机械臂仍然不动

### 根因
即使后台线程在运行，**必须显式调用 `send_recv_once()`** 来触发命令发送。

这与 `ArxTeleop` 类的实现一致：
```python
# ArxTeleop.send_cmd() 中
self.ctrl.set_joint_cmd(js)
self.ctrl.send_recv_once()  # 必须调用！
```

### 验证测试
```python
# 测试1: 只有 set_joint_cmd + sleep → 不动
ctrl.set_joint_cmd(js)
time.sleep(dt)  # ❌ 不工作

# 测试2: 加上 send_recv_once → 正常
ctrl.set_joint_cmd(js)
ctrl.send_recv_once()  # ✓ 工作
```

### 解决方案
在所有发送命令的地方都调用 `send_recv_once()`：
- `_move_to_pose()`
- `execute_action()`
- 任何需要控制机械臂的地方

---

## Bug 3: 初始位姿选择错误

### 现象
- 设置 `--init-pose` 从数据集加载
- 使用默认 frame 0
- 机械臂移动到 home 位置附近
- 策略输出无意义的小动作

### 根因
数据集的 **frame 0 是数据采集的起始点（home 位置）**，不是任务执行姿态。

```
Frame 0:   shoulder = -0.001 rad (0.1°)   ← home 位置
Frame 150: shoulder = 1.618 rad (92.7°)   ← 任务姿态（手臂抬起）
```

策略模型是在任务姿态下训练的，当机械臂在 home 位置时：
- 相机看到的图像与训练数据不同
- 机器人状态与训练数据不同
- 模型输出无意义的动作

### 解决方案
使用 `--init-frame` 参数选择任务执行中的帧：

```bash
# 分析数据集找到任务开始帧
python -c "
import h5py
import numpy as np
with h5py.File('trajectory.h5', 'r') as f:
    shoulder = f['traj_0/obs/joint_pos'][:, 1]
    task_frames = np.where(np.abs(shoulder) > 0.5)[0]
    print(f'Task starts at frame: {task_frames[0]}')
"

# 使用正确的初始帧
python -m data_collection.arx5_policy_inference \
    --init-pose "dataset:trajectory.h5" \
    --init-frame 150
```

---

## Bug 4: 推理时机械臂卡顿

### 现象
- 机械臂执行策略时周期性停顿
- 每隔约 0.3-0.5 秒停一下再继续
- 运动不流畅，呈现 "走走停停" 的模式

### 根因分析
这是一个**时序问题**，源于推理时间超过了 action chunk 的执行时间：

#### 时序计算
```
参数:
- act_horizon = 8 帧 (策略输出的 action 帧数)
- record_freq = 30Hz (原始数据记录频率)
- control_freq = 500Hz (机械臂控制频率)
- inference_time ≈ 400ms (策略推理耗时)

计算:
- steps_per_chunk = act_horizon / record_freq * control_freq
                  = 8 / 30 * 500 = 133 步
- chunk_duration = 133 / 500 = 0.266 秒 (266ms)
```

#### 问题时序图
```
时间轴: 0    100   200   300   400   500   600   700   800ms
        |-----|-----|-----|-----|-----|-----|-----|-----|
推理1:   [=========推理中=========>] 产出 chunk_1
Action:  [执行chunk_0==]              [执行chunk_1==]
空白期:              ^^^^^^^^^等待^^^^^^^^
                     (266ms~400ms 约134ms 无新动作)
```

当 action buffer 耗尽时（266ms），推理还没完成（需要400ms），控制循环只能等待，导致停顿。

### 解决方案

#### 方案1: 提前触发推理（推荐）
在 action buffer 还剩一定步数时就开始下一次推理：

```python
# 原来: idx >= steps_per_chunk 时才开始推理
# 改为: 提前 inference_time * control_freq 步开始

inference_steps = int(inference_time * control_freq)  # 400ms * 500 = 200 步
trigger_threshold = steps_per_chunk - inference_steps  # 133 - 200 = -67 (无法实现)
```

但由于推理时间 > chunk 执行时间，这个方案无法完全解决问题。

#### 方案2: 增大 act_horizon
增加每次推理输出的 action 数量：

```python
# 当前: act_horizon = 8, chunk_duration = 266ms
# 修改: act_horizon = 16, chunk_duration = 533ms > 400ms

config = InferenceConfig(
    act_horizon=16,  # 增大 action horizon
    pred_horizon=24,  # 相应增大预测 horizon
)
```

#### 方案3: 使用 Temporal Ensemble
重叠执行多个 action chunk，平滑过渡：

```python
# 当新 chunk 到达时，与旧 chunk 的剩余部分混合
# 而不是直接切换，可以消除不连续性
def temporal_ensemble(old_actions, new_actions, overlap_steps):
    weights = np.linspace(0, 1, overlap_steps)
    blended = old_actions[-overlap_steps:] * (1 - weights[:, None]) + \
              new_actions[:overlap_steps] * weights[:, None]
    return np.concatenate([old_actions[:-overlap_steps], blended, new_actions[overlap_steps:]])
```

#### 方案4: 异步推理流水线
将推理移到独立线程，始终保持 buffer 充足：

```
推理线程: [chunk0] -> queue -> [chunk1] -> queue -> [chunk2] ...
控制线程:       消费 queue.get() ...
```

### 选择建议
1. **快速修复**: 增大 `act_horizon` 到 16 或更大
2. **最佳效果**: 实现 temporal ensemble + 异步流水线
3. **折中方案**: 降低 control_freq（如 200Hz），延长 chunk 持续时间

---

## 调试经验总结

### 1. ARX5 SDK 使用要点
- **必须调用 `reset_to_home()`** 初始化增益
- **必须调用 `send_recv_once()`** 发送命令
- 使用完整构造函数比简化构造函数更可控

### 2. 策略推理要点
- **初始状态必须与训练数据分布匹配**
- 数据集 frame 0 通常是 home 位置，不是任务姿态
- 验证策略输出时，使用数据集中间帧测试
- **推理时间必须小于 action chunk 执行时间**

### 3. 调试方法
1. 先用简单测试脚本验证硬件控制
2. 检查 SDK 日志中的 kp/kd 信息
3. 对比工作代码（如 `ArxTeleop`）找差异
4. 使用数据集验证模型输出是否正确
5. 计算时序，确保推理和执行的节奏匹配

---

## 参考代码

### 正确的机器人初始化
```python
import arx5_interface as arx5

# 使用完整构造函数
robot_cfg = arx5.RobotConfigFactory.get_instance().get_config("X5")
ctrl_cfg = arx5.ControllerConfigFactory.get_instance().get_config(
    "joint_controller", robot_cfg.joint_dof
)
ctrl = arx5.Arx5JointController(robot_cfg, ctrl_cfg, "can0")

# 设置增益
gain = arx5.Gain(robot_cfg.joint_dof)
gain.kd()[:] = 0.01
ctrl.set_gain(gain)

# 初始化位置控制
ctrl.reset_to_home()
```

### 正确的命令发送
```python
js = arx5.JointState(robot_cfg.joint_dof)
js.pos()[:] = target_position
js.gripper_pos = target_gripper

ctrl.set_joint_cmd(js)
ctrl.send_recv_once()  # 必须调用！
```
