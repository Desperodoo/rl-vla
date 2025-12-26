# rl-vla

**ARX5 机械臂遥操作 + 模仿学习训练 + 真机推理** 一体化项目

## 📁 项目结构

```
rl-vla/
├── scripts/                    # 统一脚本
│   ├── setup_env.sh           # 环境配置
│   └── build_all.sh           # 一键编译
├── inference/                  # 推理模块 (独立, 不依赖 ROS)
│   ├── config.py              # 路径配置
│   ├── camera_manager.py      # RealSense 相机管理
│   └── arx5_inference.py      # ARX5 策略推理
├── arx5-sdk/                   # ARX5 机械臂 SDK
│   ├── python/                # Python 绑定
│   ├── lib/                   # 预编译库
│   └── models/                # URDF 模型
├── LeRobot-Anything-U-Arm/     # 遥操作数据采集 (需要 ROS)
│   └── src/uarm/scripts/      # 数据采集脚本
└── rlft/                       # 模仿学习训练
    ├── diffusion_policy/      # Diffusion Policy / Flow Matching
    ├── act/                   # ACT 算法
    ├── ppo/                   # PPO
    └── rlpd/                  # RLPD 在线训练
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活 conda 环境
conda activate arx-py310

# 加载环境变量
source scripts/setup_env.sh
```

### 2. 编译

```bash
# 一键编译所有组件
./scripts/build_all.sh

# 如需清理重编译
./scripts/build_all.sh --clean
```

### 3. 验证安装

```bash
# 测试 ARX5 SDK
python -c "import arx5_interface; print('arx5 OK')"

# 测试 RLFT
python -c "from diffusion_policy.algorithms import FlowMatchingAgent; print('rlft OK')"
```

## 🎯 工作流程

### 流程 1: 遥操作采集数据

```bash
# 需要 ROS 环境
source /opt/ros/noetic/setup.bash
source LeRobot-Anything-U-Arm/devel/setup.bash

# 启动遥操作
roslaunch uarm teleop.launch
```

### 流程 2: 训练模型

```bash
cd rlft/diffusion_policy

# Flow Matching 训练
python train_real_robot.py \
    --dataset ~/data/pick_cube.hdf5 \
    --algorithm flow_matching \
    --epochs 500
```

### 流程 3: 真机推理

```bash
# 加载环境
source scripts/setup_env.sh

# 运行推理 (不需要 ROS!)
python -m inference.arx5_inference \
    -c ~/rlft/runs/exp/checkpoints/final.pt \
    --init-pose dataset:~/data/pick_cube.hdf5

# 模拟运行 (不执行机器人动作)
python -m inference.arx5_inference \
    -c checkpoint.pt --dry-run
```

## ⚙️ 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RL_VLA_ROOT` | 项目根目录 | 自动检测 |
| `ARX5_SDK_PATH` | ARX5 SDK 路径 | `$RL_VLA_ROOT/arx5-sdk` |
| `RLFT_PATH` | RLFT 路径 | `$RL_VLA_ROOT/rlft` |

### 相机配置

修改 `inference/camera_manager.py` 中的 `DEFAULT_CAMERA_CONFIGS`:

```python
DEFAULT_CAMERA_CONFIGS = {
    "wrist": CameraConfig(
        name="wrist",
        serial_number="YOUR_WRIST_CAMERA_SN",  # 修改为你的相机序列号
        resolution=(640, 480),
        fps=30,
    ),
    "external": CameraConfig(
        name="external",
        serial_number="YOUR_EXTERNAL_CAMERA_SN",
        resolution=(640, 480),
        fps=30,
    )
}
```

## 📖 模块说明

### inference/ - 推理模块

独立的推理模块，不依赖 ROS，可直接部署。

**主要文件:**
- `config.py` - 统一路径配置，支持环境变量覆盖
- `camera_manager.py` - RealSense 相机管理器
- `arx5_inference.py` - 策略推理主程序

**推理参数:**

```bash
python -m inference.arx5_inference --help

# 常用参数:
#   -c, --checkpoint    Checkpoint 文件路径
#   --dry-run          模拟运行
#   --init-pose        初始姿态 (dataset:xxx.hdf5 或 逗号分隔值)
#   --flow-steps       Flow 步数 (默认 10, 越少越快)
#   --filter-alpha     EMA 滤波系数 (默认 0.3, 越小越平滑)
```

### arx5-sdk/ - 机械臂 SDK

ARX5 机械臂的 C++ 和 Python SDK。

**主要功能:**
- 关节空间控制 (500Hz)
- 笛卡尔空间控制
- 夹爪控制

### LeRobot-Anything-U-Arm/ - 遥操作

基于 ROS 的主从臂遥操作系统。

**主要功能:**
- 主臂角度读取
- 从臂跟随控制
- 数据录制

### rlft/ - 训练

模仿学习算法实现。

**支持算法:**
- Diffusion Policy
- Flow Matching
- Consistency Flow
- ACT
- PPO / RLPD

## 🔧 故障排除

### 问题: arx5_interface 导入失败

```bash
# 确保已编译
./scripts/build_all.sh

# 检查环境变量
echo $PYTHONPATH
echo $LD_LIBRARY_PATH

# 重新加载环境
source scripts/setup_env.sh
```

### 问题: CAN 通信失败

```bash
# 设置 CAN 设备
sudo ./arx5-sdk/setup_can_devices.sh

# 检查 CAN 状态
ip link show can0
```

### 问题: 相机检测失败

```bash
# 列出 RealSense 设备
rs-enumerate-devices

# 测试相机
python -m inference.camera_manager
```

## 📝 开发说明

### 添加新的推理算法

1. 在 `rlft/diffusion_policy/algorithms/` 中实现算法
2. 在 `inference/arx5_inference.py` 的 `_load_policy()` 中添加支持

### 修改硬件配置

- 机器人型号: 修改 `_setup_robot()` 中的 `"X5"` 参数
- 相机序列号: 修改 `camera_manager.py` 中的 `DEFAULT_CAMERA_CONFIGS`

## 📜 许可证

各子模块保留其原始许可证。

---

**维护者:** lizh  
**最后更新:** 2024.12
