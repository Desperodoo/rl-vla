# CARM ROS 部署包

基于 ROS1 原生通信的 CARM 机械臂部署框架，替代 svar 方案。

## 项目结构

```
carm_deploy/
├── config/
│   └── default.yaml          # 默认配置
├── launch/
│   ├── camera.launch         # 相机启动
│   ├── realsense_d405.launch # D405 专用配置
│   ├── inference.launch      # 推理节点
│   ├── record.launch         # 数据记录
│   └── full_system.launch    # 完整系统
├── camera/
│   ├── test_realsense.py     # 相机测试 (pyrealsense2)
│   └── ros_camera_subscriber.py  # ROS 相机订阅
├── utils/
│   ├── __init__.py
│   ├── image_sync.py         # 图像同步 (替代 svar TopicSync)
│   └── trajectory_interpolator.py  # 轨迹插值 (替代 svar VecTF)
├── env_ros.py                # 环境封装 (替代 carm_real/env_api.py)
├── inference_ros.py          # 推理程序 (替代 carm_real/infer_g3_api.py)
├── record_data_ros.py        # 数据记录 (替代 carm_real/record_data_surreal3576.py)
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 依赖

### 系统依赖
- Ubuntu 20.04
- ROS Noetic
- Intel RealSense SDK 2.0

### ROS 包
```bash
sudo apt install ros-noetic-realsense2-camera ros-noetic-cv-bridge
```

### Python 依赖
```bash
conda activate carm
pip install numpy scipy h5py opencv-python einops
```

### 机械臂 SDK
```bash
pip install /path/to/carm_sdk/lib/amd64/carm_py-1.0-cp310-cp310-linux_x86_64.whl
```

## 快速开始

### 1. 编译 ROS 包

```bash
# 使用项目统一编译脚本
cd /path/to/rl-vla
./scripts/build_catkin.sh
source catkin_ws/devel/setup.bash
```

### 2. 启动相机

```bash
roslaunch carm_deploy camera.launch
# 或使用 D405 专用配置
roslaunch carm_deploy realsense_d405.launch
```

### 3. 数据记录（拖动示教）

```bash
roslaunch carm_deploy record.launch output_dir:=~/recorded_data
```

控制键:
- `s`: 开始/停止记录
- `q`: 保存并退出

### 4. 策略推理

```bash
roslaunch carm_deploy inference.launch pretrain:=/path/to/model.pt
```

### 5. 完整系统

```bash
roslaunch carm_deploy full_system.launch pretrain:=/path/to/model.pt
```

## 配置说明

### 机械臂参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| robot_ip | 10.42.0.101 | 机器人 IP |
| robot_mode | 1 | 控制模式 (0=IDLE, 1=POSITION, 2=MIT, 3=DRAG) |
| robot_tau | 10 | 夹爪扭矩 |

### 相机参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| camera_serial | 218622279840 | 相机序列号 |
| camera_topics | /camera/color/image_raw | 图像话题 |

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| pretrain | "" | 模型路径 |
| desire_inference_freq | 30 | 推理频率 |
| temporal_factor_k | 0.05 | 时序融合系数 |
| joint_cmd_mode | false | 关节控制模式 |

## 与 carm_real 的对应关系

| carm_real | carm_ros_deploy | 说明 |
|-----------|-----------------|------|
| svar ros2.TopicSync | utils/image_sync.py | 图像同步 |
| svar vectf.VecTF | utils/trajectory_interpolator.py | 轨迹插值 |
| svar messenger | rospy.Publisher/Subscriber | ROS 通信 |
| env_api.py | env_ros.py | 环境封装 |
| infer_g3_api.py | inference_ros.py | 推理程序 |
| record_data_surreal3576.py | record_data_ros.py | 数据记录 |

## 策略接口

推理程序使用 `PolicyInterface` 抽象类，用户需要继承并实现：

```python
from inference_ros import PolicyInterface

class MyPolicy(PolicyInterface):
    def load_model(self, model_path):
        # 加载模型
        self.model = torch.load(model_path)
    
    def __call__(self, inputs):
        # 执行推理
        # inputs: {'qpos': Tensor[B,7], 'image': Tensor[B,C,H,W]}
        # 返回: {'a_hat': Tensor[B,H,D]}
        return self.model(inputs)
```

然后修改 `inference_ros.py` 中的 `_create_policy` 方法加载自定义策略。

## 数据格式

记录的数据保存为 HDF5 格式：

```
episode_0001_20240108_120000.hdf5
├── observations/
│   ├── images          # [T, H, W, 3] uint8, RGB 格式
│   ├── qpos_joint      # [T, 7] float64 (6 joints + gripper)
│   ├── qpos_end        # [T, 8] float64 (xyz + quat + gripper)
│   ├── qpos            # [T, 15] float64 (兼容格式)
│   ├── gripper         # [T] float64
│   └── timestamps      # [T] float64
├── action              # [T, 15] float64 (joint cmd + end pose cmd)
└── attrs/
    ├── num_steps
    ├── record_freq
    └── ...
```

### 图像格式协议

**重要**：本项目统一使用 **RGB** 格式存储和处理图像。

| 阶段 | 格式 | 说明 |
|------|------|------|
| ROS 话题 | RGB8 | RealSense ROS 节点发布 |
| ImageSynchronizer | RGB | `get_images()` 返回 RGB |
| HDF5 存储 | RGB | `observations/images` |
| 训练加载 | RGB | 与 PyTorch/预训练模型兼容 |
| 推理输入 | RGB | 同训练一致 |

**注意**：
- 使用 OpenCV (`cv2.imshow`, `cv2.imwrite`) 时需要转换为 BGR：
  ```python
  img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
  cv2.imwrite('output.jpg', img_bgr)
  ```
- 使用 matplotlib 或 PIL 时直接使用 RGB，无需转换

## 常见问题

### 1. 相机无法启动

检查相机序列号：
```bash
rs-enumerate-devices | grep Serial
```

### 2. 机械臂连接失败

检查网络连接：
```bash
ping 10.42.0.101
```

### 3. 关节限位错误

J2 限位 [0, 3.14]，J3 限位 [-3.14, 0]，确保目标位置在限位内。

## 许可证

MIT License
