# CARM 机械臂调试脚本

本文件夹包含 CARM 机械臂的调试和测试脚本。

## 快速开始

### 1. 配置环境

每次使用前，先运行配置脚本：

```bash
cd /home/lizh/rl-vla
source carm_scripts/setup_carm_env.sh
```

这将自动完成：
- 激活 `carm` conda 环境 (Python 3.10)
- 配置 arm_control_sdk 环境变量
- 配置 ROS1 noetic 环境

### 2. 运行测试脚本

```bash
# 测试连接
python carm_scripts/test_connection.py

# 测试运动（需要交互确认）
python carm_scripts/test_motion.py

# 测试夹爪（需要交互确认）
python carm_scripts/test_gripper.py
```

### 3. 安全退出（下使能）

调试完毕后，建议执行安全退出脚本：

```bash
python carm_scripts/safe_shutdown.py
```

这将执行：
1. 设置控制模式为 IDLE（空闲）
2. **下使能** - 关闭伺服电机，机械臂失去保持力
3. 断开与控制器的连接

⚠️ **注意**：下使能后机械臂会进入自由状态（可以手动拖动），请确保机械臂处于安全位置！

如果只是临时断开连接（保持使能状态），可以直接调用 `disconnect()`：
```python
carm.disconnect()  # 断开连接，但保持使能状态
```

## 机械臂连接信息

| 参数 | 值 |
|------|-----|
| IP 地址 | 10.42.0.101 |
| 端口 | 8090 |
| 本机 IP | 10.42.0.100 |

## 关节限位

| 关节 | 下限 (rad) | 上限 (rad) |
|------|-----------|-----------|
| 关节1 | -2.79 | 2.79 |
| 关节2 | 0.0 | 3.14 |
| 关节3 | -3.14 | 0.0 |
| 关节4 | -2.65 | 2.65 |
| 关节5 | -1.57 | 1.57 |
| 关节6 | -2.88 | 2.88 |

## 夹爪参数

- 间隔范围：0 - 0.08m (0mm - 80mm)
- 力矩范围：0 - 20N

## C++ 示例

```bash
source carm_scripts/setup_carm_env.sh
cd carm_demo/cpp_test_demo/build
./carm_demo  # 单臂交互式控制
```

## ROS1 使用

```bash
# 终端1：启动 roscore
source /opt/ros/noetic/setup.bash
roscore

# 终端2：启动机械臂节点
source carm_scripts/setup_carm_env.sh
rosrun carm_api carm_ros_node

# 终端3：查看话题
rostopic list
rostopic echo /real_joint_state
```

### ROS 话题列表

| 话题 | 类型 | 说明 |
|------|------|------|
| /connect | String | 连接/断开机械臂 |
| /ready | Bool | 复位机械臂 |
| /emergency_stop | Bool | 急停 |
| /move_joint | JointState | 关节空间运动 |
| /move_pose | Pose | 笛卡尔空间运动 |
| /move_tracking_joint | JointState | 高频关节跟随 |
| /move_tracking_pose | Pose | 高频末端跟随 |
| /set_gripper | JointState | 夹爪控制 |
| /real_joint_state | JointState | 当前关节状态（发布） |
| /flange_cart_state | PoseStamped | 末端位姿（发布） |

## Python API 快速参考

```python
from carm import carm_py

# 创建并连接（构造函数自动连接）
carm = carm_py.CArmSingleCol("10.42.0.101")

# 复位（清除错误、上使能、进入位置控制模式）
carm.set_ready()

# 获取状态
carm.get_joint_pos()      # 关节位置 [6]
carm.get_cart_pose()      # 末端位姿 [x,y,z,Qx,Qy,Qz,Qw]
carm.get_gripper_pos()    # 夹爪位置 (m)

# 运动控制
carm.move_joint([0,0,0,0,0,0])  # 关节运动
carm.set_gripper(0.08, 10.0)    # 夹爪控制 (间隔m, 力矩N)

# 安全退出
carm.set_control_mode(0)        # 设置为 IDLE 模式
carm.set_servo_enable(False)    # 下使能（机械臂失去保持力）
carm.disconnect()               # 断开连接
```

### 控制模式说明

| 模式值 | 名称 | 说明 |
|--------|------|------|
| 0 | IDLE | 空闲模式 |
| 1 | POSITION | 点位控制模式（默认） |
| 2 | MIT | 力矩控制模式（高频跟随） |
| 3 | DRAG | 拖动示教模式 |
| 4 | PF | 力位混合模式 |

### 使能状态说明

- **上使能 `set_servo_enable(True)`**：伺服电机启动，机械臂有保持力
- **下使能 `set_servo_enable(False)`**：伺服电机关闭，机械臂可自由拖动
- **`set_ready()`**：自动完成清除错误 + 上使能 + 进入位置模式

## 文件列表

- `setup_carm_env.sh` - 环境配置脚本
- `test_connection.py` - 连接测试
- `test_motion.py` - 运动测试
- `test_gripper.py` - 夹爪测试
- `safe_shutdown.py` - 安全退出（下使能并断开）
