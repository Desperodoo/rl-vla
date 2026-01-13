# rl-vla

**CARM 机械臂真机部署 + 模仿学习训练** 一体化项目

## 📁 项目结构

```
rl-vla/
├── catkin_ws/                  # ROS1 工作空间
│   └── src/
│       ├── carm_deploy/       # 部署包
│       │   ├── config/        # 配置文件
│       │   ├── launch/        # Launch 文件
│       │   ├── camera/        # 相机工具
│       │   ├── utils/         # 工具模块
│       │   └── *_ros.py       # 主程序
│       ├── realsense-ros/     # RealSense ROS 驱动
│       └── carm_api/          # CARM ROS 消息
│
├── carm_sdk/                   # CARM SDK (符号链接 → carm_demo/arm_control_sdk)
│
├── carm_demo/                  # CARM 官方 SDK (上游参考)
│   ├── arm_control_sdk/       # SDK 源码
│   ├── carm_ros/              # ROS1 消息包
│   ├── cpp_test_demo/         # C++ 示例
│   └── python/                # Python 示例
│
├── scripts/                    # 统一脚本
│   ├── build_catkin.sh        # catkin 编译脚本
│   └── carm/                  # CARM 调试脚本
│
├── rlft/                       # 模仿学习训练
│
└── archive/                    # 归档 (旧版参考)
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建 conda 环境 (如果没有)
conda create -n carm python=3.10 -y
conda activate carm

# 安装 CARM SDK
pip install carm_sdk/lib/amd64/carm_py-1.0-cp310-cp310-linux_x86_64.whl

# 安装其他依赖
pip install numpy scipy h5py opencv-python einops pyrealsense2
pip install empy==3.3.4 catkin_pkg rospkg  # catkin 编译依赖
```

### 2. 编译 ROS 工作空间

```bash
conda activate carm
source /opt/ros/noetic/setup.bash

# 使用统一编译脚本
./scripts/build_catkin.sh

# 加载编译结果
source catkin_ws/devel/setup.bash
```

### 3. 验证安装

```bash
# 测试 SDK
python -c "import carm_py; print('carm_py OK')"

# 测试机械臂连接
python scripts/carm/test_connection.py
```

## 🤖 机械臂操作

### 连接参数

| 参数 | 值 |
|------|-----|
| IP | 10.42.0.101 |
| Port | 8090 |
| 关节数 | 6 + 夹爪 |
| 夹爪范围 | 0 ~ 0.08m |

### 控制模式

| 模式 | 值 | 说明 |
|------|-----|------|
| IDLE | 0 | 空闲 |
| POSITION | 1 | 位置控制 |
| MIT | 2 | MIT 控制 |
| DRAG | 3 | 拖动示教 |
| PF | 4 | 力控 |

### 调试脚本

```bash
# 测试连接
python scripts/carm/test_connection.py

# 测试关节运动
python scripts/carm/test_motion.py

# 测试夹爪
python scripts/carm/test_gripper.py

# 安全关闭
python scripts/carm/safe_shutdown.py
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
# 方式1: ROS Launch
roslaunch carm_deploy camera.launch

# 方式2: 直接测试 (不需要 ROS)
python carm_deploy/camera/test_realsense.py
```

## 🎯 数据采集

### 拖动示教录制

```bash
# 启动相机
roslaunch carm_deploy camera.launch

# 另一个终端: 启动录制
roslaunch carm_deploy record.launch output_dir:=~/recorded_data

# 控制键:
#   's' - 开始/停止录制
#   'q' - 保存并退出
```

### 数据格式

录制的数据保存为 HDF5 格式:

```
episode_0001_20240108_120000.hdf5
├── observations/
│   ├── images          # [T, H, W, C] uint8
│   ├── qpos_joint      # [T, 7] float64
│   ├── qpos_end        # [T, 8] float64 (xyz + quat + gripper)
│   └── timestamps      # [T] float64
└── attrs/
    ├── num_steps
    └── record_freq
```

## 🧠 策略推理

### 运行推理

```bash
# 启动完整系统 (相机 + 推理)
roslaunch carm_deploy full_system.launch pretrain:=/path/to/model.pt

# 或分开启动
roslaunch carm_deploy camera.launch
roslaunch carm_deploy inference.launch pretrain:=/path/to/model.pt
```

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| pretrain | "" | 模型路径 |
| desire_inference_freq | 30 | 推理频率 (Hz) |
| temporal_factor_k | 0.05 | 时序融合系数 |
| joint_cmd_mode | false | 关节控制模式 |

### 自定义策略

继承 `PolicyInterface` 实现自定义策略:

```python
from carm_deploy.inference_ros import PolicyInterface

class MyPolicy(PolicyInterface):
    def load_model(self, model_path):
        self.model = torch.load(model_path)
    
    def __call__(self, inputs):
        # inputs: {'qpos': [B,7], 'image': [B,C,H,W]}
        return {'a_hat': self.model(inputs)}
```

## 📖 模块说明

### carm_deploy/ - 部署模块

纯 ROS1 实现的部署框架，替代旧版 svar 方案。

| 文件 | 功能 |
|------|------|
| env_ros.py | 机械臂环境封装 |
| inference_ros.py | 策略推理主程序 |
| record_data_ros.py | 数据采集程序 |
| utils/image_sync.py | 多相机图像同步 |
| utils/trajectory_interpolator.py | 动作轨迹插值与融合 |

### rlft/ - 训练模块

模仿学习算法实现。

| 算法 | 目录 |
|------|------|
| Diffusion Policy | diffusion_policy/ |
| ACT | act/ |
| PPO | ppo/ |
| RLPD | rlpd/ |

### archive/ - 归档

旧版代码参考，包含 svar 依赖的实现。

## 🔧 故障排除

### 问题: carm_py 导入失败

```bash
# 检查 wheel 文件
ls carm_sdk/lib/amd64/*.whl

# 重新安装
pip install --force-reinstall carm_sdk/lib/amd64/carm_py-*.whl
```

### 问题: 机械臂连接失败

```bash
# 检查网络
ping 10.42.0.101

# 检查端口
nc -zv 10.42.0.101 8090
```

### 问题: catkin_make 失败

```bash
# 确保在 carm 环境
conda activate carm

# 检查依赖
pip install empy==3.3.4 catkin_pkg rospkg

# 清理重编译
./scripts/build_catkin.sh --clean
```

### 问题: 相机无图像

```bash
# 检查相机连接
rs-enumerate-devices | grep Serial

# 检查话题
rostopic hz /camera/color/image_raw
```

### 问题: 关节超限报错

J2 限位 [0, 3.14]，J3 限位 [-3.14, 0]，其他关节 [-2.6, 2.6]。确保目标位置在限位内。

## 📜 许可证

各子模块保留其原始许可证。

---

**维护者:** lizh  
**最后更新:** 2026.01
