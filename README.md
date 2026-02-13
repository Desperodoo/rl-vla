# rl-vla

**CARM 机械臂真机部署 + 模仿学习训练** 一体化项目

## 📁 项目结构

```
rl-vla/
├── carm_ros_deploy/            # ROS1 部署工作空间
│   └── src/
│       ├── carm_deploy/       # 部署包 (主要模块)
│       │   ├── core/          # 核心模块 (环境、安全控制)
│       │   ├── inference/     # 推理模块
│       │   ├── data/          # 数据采集和分析
│       │   ├── tools/         # 工具脚本
│       │   ├── utils/         # 工具模块
│       │   ├── camera/        # 相机工具
│       │   ├── config/        # 配置文件
│       │   ├── launch/        # Launch 文件
│       │   └── safety_config.json  # 安全配置文件
│       ├── realsense-ros/     # RealSense ROS 驱动
│       └── carm_api/          # CARM ROS 消息
│
├── arm_control_sdk/            # CARM 机械臂 SDK (库、头文件、Python 绑定)
│
├── scripts/                    # 全局脚本
│   ├── build_catkin.sh        # catkin 编译脚本
│   └── setup_carm_env.sh      # 环境设置脚本
│
├── recorded_data/              # 采集的数据
│   ├── mix/                   # 混合数据集
│   └── random_pos/            # 随机位置数据集
│
├── inference_logs/             # 推理日志
│
├── rlft/                       # 模仿学习训练
│   ├── diffusion_policy/      # Diffusion Policy / Flow Matching
│   ├── act/                   # ACT 算法
│   └── ppo/                   # PPO 算法
│

```

## 🚀 快速开始

### 1. 首次安装

```bash
# 1) 创建 conda 环境
conda create -n carm python=3.10 -y
conda activate carm

# 2) 安装 Python 依赖
pip install numpy scipy h5py opencv-python einops pyrealsense2
pip install empy==3.3.4 catkin_pkg rospkg   # catkin 编译依赖

# 3) 编译并安装 CARM SDK Python 绑定
cd arm_control_sdk/python
python build_carm.py --Release     # 从源码编译 carm_py.so
python install_carm.py             # 安装到当前 conda 环境
cd ../..

# 4) 编译 ROS 工作空间 (自动加载 conda / ROS / SDK)
./scripts/build_catkin.sh
```

### 2. 日常使用 (每次开终端)

```bash
source scripts/setup_carm_env.sh
```

该脚本自动完成：激活 carm conda → 加载 ROS Noetic → 加载 SDK 库 → 加载 catkin 工作空间。

### 3. 验证安装

```bash
# 测试 SDK
python -c "from carm import carm_py; print('carm OK')"

# 测试机械臂连接 (需要机械臂上电)
python carm_ros_deploy/src/carm_deploy/tools/arm_test/test_connection.py
```

### 脚本说明

| 脚本 | 用途 |
|------|------|
| `scripts/setup_carm_env.sh` | **source** 一次加载全部运行环境 |
| `scripts/build_catkin.sh` | 编译 catkin 工作空间 (`--clean` 可清理重编) |

## 🤖 机械臂操作

### 连接参数

| 参数 | 值 |
|------|-----|
| IP | 10.42.0.101 |
| Port | 8090 |
| 关节数 | 6 + 夹爪 |
| 夹爪范围 | 0 ~ 0.073m |

### 控制模式

| 模式 | 值 | 说明 |
|------|-----|------|
| IDLE | 0 | 空闲 |
| POSITION | 1 | **禁用** (危险) |
| MIT | 2 | 阻抗控制 |
| DRAG | 3 | 拖动示教 |
| PF | 4 | 力控（**推荐**） |

### 调试脚本

```bash
cd carm_ros_deploy/src/carm_deploy/tools/arm_test

# 测试连接
python test_connection.py

# 测试关节运动
python test_motion.py

# 测试夹爪
python test_gripper.py

# 安全关闭
python safe_shutdown.py
```

## 📷 相机操作

### 相机参数

| 参数 | 值 |
|------|-----|
| 型号 | Intel RealSense D405 |
| 序列号 | 218622279840 |
| 分辨率 | 640x480 @ 30fps |

### 启动相机

```bash
# ROS Launch
roslaunch carm_deploy camera.launch

# 直接测试 (不需要 ROS)
python carm_ros_deploy/src/carm_deploy/camera/test_realsense.py
```

## 🎯 数据采集

### 拖动示教录制

```bash
# 启动相机
roslaunch carm_deploy camera.launch

# 另一个终端: 启动录制
roslaunch carm_deploy record.launch output_dir:=~/rl-vla/recorded_data

# 控制键:
#   's' - 开始/停止录制
#   'q' - 保存并退出
```

### 数据分析

```bash
cd carm_ros_deploy/src/carm_deploy/data
python analyze_dataset.py --data_dir ~/rl-vla/recorded_data/mix
```

### 数据格式

```
episode_0001.hdf5
├── observations/
│   ├── images          # [T, H, W, 3] RGB
│   ├── qpos_joint      # [T, 7]
│   ├── qpos_end        # [T, 8]
│   └── timestamps      # [T]
├── action              # [T, 15]
└── attrs/
```

## 🧠 策略推理

### 测试模式

```bash
# 干运行模式（不执行动作，最安全）
rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt --dry_run

# 慢速模式（5Hz）
rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt --slow_mode
```

### 正常推理

```bash
# 完整系统 (相机 + 推理)
roslaunch carm_deploy full_system.launch pretrain:=/path/to/model.pt

# 或分开启动
roslaunch carm_deploy camera.launch
roslaunch carm_deploy inference.launch pretrain:=/path/to/model.pt
```

### 离线测试

```bash
cd carm_ros_deploy/src/carm_deploy/tools
python offline_test.py \
    --model_path /path/to/model.pt \
    --data_dir ~/rl-vla/recorded_data/mix \
    --compare_ema
```

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_inference_steps | 5 | 推理步数 |
| use_ema | False | 使用 EMA 模型 |
| temporal_factor_k | 0.05 | 时序融合系数 |
| desire_inference_freq | 30 | 推理频率 (Hz) |

## 🔒 安全控制

> **首次使用必须录制安全边界。** 推理启动时会检查 `safety_config.json` 是否存在，不存在则拒绝运行。

### 1. 录制安全边界

```bash
cd carm_ros_deploy/src/carm_deploy/tools

# 拖动示教模式录制工作空间边界 (退出时自动回零位并下使能)
python record_workspace.py
```

操作说明：按 Enter 开启拖动模式 → 拖动机械臂覆盖工作空间 → 按 `s` 保存 → 按 `q` 退出。

### 2. 验证安全配置

```bash
# 开启拖动示教，实时查看当前位置是否在安全范围内
python verify_safety_config.py --test_mode visual

# 仅检查当前位置 (不进入拖动模式)
python verify_safety_config.py
```

退出时自动回零位并下使能。

## 📖 模块说明

### carm_deploy - 部署模块

| 目录 | 功能 |
|------|------|
| core/ | 环境封装、安全控制器 |
| inference/ | 策略推理、日志记录 |
| data/ | 数据采集、加载、分析 |
| tools/ | 离线测试、配置验证 |
| utils/ | 图像同步、轨迹插值、路径配置 |

### rlft - 训练模块

| 算法 | 目录 |
|------|------|
| Consistency Flow (推荐) | diffusion_policy/ |
| Flow Matching | diffusion_policy/ |
| Diffusion Policy | diffusion_policy/ |
| ACT | act/ |

## 🔧 故障排除

### carm_py 导入失败

```bash
cd arm_control_sdk/python
python build_carm.py --Release
python install_carm.py
```

### 机械臂连接失败

```bash
ping 10.42.0.101
nc -zv 10.42.0.101 8090
```

### catkin_make 失败

```bash
./scripts/build_catkin.sh --clean
```

### 相机无图像

```bash
rs-enumerate-devices | grep Serial
rostopic hz /camera/color/image_raw
```

## 📜 许可证

各子模块保留其原始许可证。

---

**维护者:** lizh  
**最后更新:** 2026.01
