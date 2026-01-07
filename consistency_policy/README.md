# Consistency Policy 真机推理框架

基于 UMI-ARX 框架设计，针对 ARX5 机械臂和关节空间控制优化。

## 目录结构

```
consistency_policy/
├── __init__.py                      # 模块初始化
├── config.py                        # 路径配置
├── policy_inference.py              # 策略推理节点 (ZMQ 服务端)
├── robot_controller.py              # ARX5 控制器 (单进程，简单版)
├── robot_controller_mp.py           # ARX5 控制器 (多进程，200Hz) ★推荐
├── realsense_camera.py              # RealSense 相机 (多进程)
├── joint_trajectory_interpolator.py # 关节空间轨迹插值器
├── eval_real.py                     # 评估脚本 (旧版，单进程)
├── eval_real_mp.py                  # 评估脚本 (新版，多进程) ★推荐
├── README.md                        # 本文档
└── tests/
    ├── __init__.py
    ├── test_offline_inference.py    # 离线推理测试
    └── test_robot_replay.py         # 机械臂回放测试
```

## 架构概述 (多进程版本)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        eval_real_mp.py (主进程)                              │
│                                                                              │
│   ┌────────────┐    ┌────────────────┐    ┌───────────────────────────┐     │
│   │ 键盘/UI    │    │ 观测组装       │    │ 动作调度                  │     │
│   │ 控制       │    │ get_obs()      │    │ schedule_actions()        │     │
│   └────────────┘    └───────┬────────┘    └──────────┬────────────────┘     │
│                             │                         │                      │
└─────────────────────────────┼─────────────────────────┼──────────────────────┘
                              │                         │
    ┌─────────────────────────┼─────────────────────────┼─────────────────────┐
    │         SharedMemory    │                         │                     │
    │        ┌────────────────┴───────────┐  ┌─────────┴───────────┐         │
    │        │ SharedMemoryRingBuffer     │  │ SharedMemoryQueue   │         │
    │        │ (相机帧/机器人状态)        │  │ (控制命令)          │         │
    │        └────────────────┬───────────┘  └─────────┬───────────┘         │
    └─────────────────────────┼─────────────────────────┼─────────────────────┘
                              │                         │
         ┌────────────────────┼───────────┬─────────────┼──────────────┐
         │                    │           │             │              │
         ▼                    ▼           │             ▼              │
┌─────────────────┐  ┌─────────────────┐  │  ┌─────────────────────────┴─┐
│ 腕部相机 (D435i) │  │ 外部相机 (D455) │  │  │ 控制器进程 (200Hz)         │
│ RealSenseCamera │  │ RealSenseCamera │  │  │ Arx5JointControllerProcess │
│ (30Hz 采集)     │  │ (30Hz+录制)     │  │  │ JointTrajectoryInterpolator │
└─────────────────┘  └─────────────────┘  │  └─────────────────────────────┘
                                          │               │
                                          │               ▼
                                          │  ┌─────────────────────────────┐
                                          │  │ arx5-sdk                    │
                                          │  │ (arx-py310 conda 环境)      │
                                          │  └─────────────────────────────┘
                                          │
                           ┌──────────────┘
                           ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │            PolicyInferenceNode (独立进程/ZMQ Server)             │
        │  - ZMQ REP 服务端 (tcp://localhost:8766)                        │
        │  - Consistency Policy 模型推理 (~70ms)                          │
        │  - 输入: obs_dict {rgb: (T,C,H,W), state: (T,7)}                │
        │  - 输出: action (pred_horizon, 7) 绝对关节角度                   │
        └─────────────────────────────────────────────────────────────────┘
```

## 硬件配置

### 相机

| 相机 | 型号 | 序列号 | 用途 |
|------|------|--------|------|
| 腕部相机 | D435i | 036222071712 | 策略推理输入 |
| 外部相机 | D455 | 037522250003 | 第三人称视角录制 |

### 机械臂

- **型号**: ARX5 X5
- **接口**: CAN (can0)
- **控制模式**: 关节空间控制 (非末端位姿)
- **控制频率**: 200Hz (通过插值实现)

## 快速开始

### 1. 环境准备

```bash
# 激活 arx5-sdk 环境
conda activate arx-py310

# 验证 arx5-sdk
python -c "import arx5_interface; print('OK')"

# 验证 pyrealsense2
python -c "import pyrealsense2 as rs; print('RealSense OK')"
```

### 2. 启动策略推理服务

在独立终端中启动策略推理节点：

```bash
python -m consistency_policy.policy_inference
```

### 3. 离线推理测试 (无需真机)

验证策略推理是否正确：

```bash
python -m consistency_policy.tests.test_offline_inference \
    --checkpoint /path/to/checkpoint.pt \
    --demo ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5
```

### 4. 机械臂回放测试 (需要真机)

验证控制链路是否正确 (安全起见先用 30% 速度)：

```bash
python -m consistency_policy.tests.test_robot_replay \
    --demo ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5 \
    --speed 0.3
```

### 5. 真机评估 (推荐多进程版本)

```bash
# 终端 1: 策略推理服务 (已在步骤2启动)

# 终端 2: 评估脚本 (多进程版本)
python -m consistency_policy.eval_real_mp \
    --output ./eval_output \
    -v
```

## 键盘控制

在评估脚本运行时：

| 按键 | 功能 |
|------|------|
| `q` | 退出程序 |
| `c` | 开始策略控制 (同时开始录制) |
| `s` | 停止策略控制 |
| `r` | 复位机械臂到 home |
| `v` | 开始/停止录制视频 |

## 配置参数

### policy_inference.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | (必需) | Checkpoint 路径 |
| `--ip` | 0.0.0.0 | 监听 IP |
| `--port` | 8766 | 监听端口 |
| `--device` | cuda | 推理设备 |
| `--no-ema` | False | 不使用 EMA 权重 |

### eval_real_mp.py (推荐)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output` | ./eval_output | 输出目录 |
| `--policy-endpoint` | tcp://localhost:8766 | 策略服务端点 |
| `--model` | X5 | 机械臂型号 |
| `--interface` | can0 | CAN 接口 |
| `--control-freq` | 200 | 控制器频率 Hz |
| `--eval-freq` | 10 | 评估循环频率 Hz |
| `--no-external-camera` | False | 禁用外部相机 |
| `--max-episodes` | 10 | 最大 episode 数 |
| `--max-steps` | 1000 | 每 episode 最大步数 |
| `-v` | False | 详细输出 |

## 与 UMI-ARX 框架的区别

| 特性 | UMI-ARX | Consistency Policy |
|------|---------|-------------------|
| 控制空间 | 末端位姿 (6D pose) | 关节空间 (6 joints) |
| 策略类型 | Diffusion Policy | Consistency Policy |
| 输出格式 | 末端位姿 + 夹爪 | 关节角度 + 夹爪 |
| 控制频率 | 100Hz | 200Hz |
| 轨迹插值 | 6D pose + Slerp | CubicSpline |
| 相机类型 | USB UVC | RealSense (D435i/D455) |
| 通信方式 | ZMQ + IPC | ZMQ + TCP/IPC |

## 关键模块说明

### JointTrajectoryInterpolator

关节空间轨迹插值器，使用三次样条插值实现平滑运动：

```python
from consistency_policy.joint_trajectory_interpolator import JointTrajectoryInterpolator

# 创建插值器
interp = JointTrajectoryInterpolator(
    times=np.array([0.0, 1.0, 2.0]),
    joints=np.array([...])  # (3, 7) 关节角度
)

# 查询任意时刻的关节位置
joints = interp(0.5)  # (7,)

# 添加新航点
new_interp = interp.schedule_waypoint(
    joints=target_joints,
    time=target_time,
    curr_time=current_time,
)
```

### Arx5JointControllerProcess

多进程控制器，在独立进程中以 200Hz 运行：

```python
from multiprocessing.managers import SharedMemoryManager
from consistency_policy.robot_controller_mp import Arx5JointControllerProcess

shm_manager = SharedMemoryManager()
shm_manager.start()

controller = Arx5JointControllerProcess(
    shm_manager=shm_manager,
    model="X5",
    interface="can0",
    frequency=200.0,
)

with controller:
    # 获取状态
    state = controller.get_state()
    
    # 调度航点
    controller.add_waypoint(joint_pos, gripper_pos, target_time)
    controller.update_trajectory()
```

### RealSenseCameraProcess

多进程相机，在独立进程中采集图像：

```python
from consistency_policy.realsense_camera import RealSenseCameraProcess, RealSenseCameraConfig

config = RealSenseCameraConfig(
    name='wrist',
    serial_number='036222071712',
    resolution=(640, 480),
    fps=30,
    enable_recording=True,
)

camera = RealSenseCameraProcess(
    shm_manager=shm_manager,
    config=config,
)

with camera:
    # 获取最新帧
    frame = camera.get_frame()  # {'rgb': (H,W,3), 'timestamp': float, ...}
    
    # 开始录制
    camera.start_recording('./video.mp4')
```

## 故障排除

### 1. 相机未检测到

```bash
# 检测连接的相机
python -c "
from consistency_policy.realsense_camera import RealSenseCameraManager
cameras = RealSenseCameraManager.detect_cameras()
print('检测到的相机:', cameras)
"
```

### 2. CAN 接口未就绪

```bash
# 检查 CAN 接口
ip link show can0

# 启用 CAN 接口 (如需要)
sudo ip link set can0 up type can bitrate 1000000
```

### 3. 策略推理超时

- 检查策略服务是否运行
- 检查网络连接 (`ping localhost`)
- 增加 ZMQ 超时时间

### 4. 控制频率不稳定

- 检查 CPU 负载
- 考虑使用 `taskset` 绑定核心
- 降低其他进程优先级

## 版本历史

- **v0.2.0**: 多进程架构，RealSense 支持，200Hz 控制
- **v0.1.0**: 初始版本，单进程架构
