# ARX5 遥操作数据采集模块

本模块用于 ARX5 机械臂的遥操作数据采集，支持双 RealSense RGB-D 相机（腕部 D435i + 外部 D455），输出 ManiSkill 兼容的 HDF5 数据格式。

## 📁 模块结构

```
data_collection/
├── __init__.py              # 模块入口
├── dataset_config.py        # 数据集配置（相机、机器人参数）
├── camera_manager.py        # RealSense 相机管理（自动检测、同步采集）
├── data_recorder.py         # 数据记录器（原始数据存储）
├── arx5_collect_data.py     # 主采集脚本 ⭐
├── preprocess_dataset.py    # 数据预处理与清洗
└── replay_trajectory.py     # 轨迹回放与验证
```

## 🚀 快速开始

### 1. 检测相机

首先检测连接的 RealSense 相机：

```bash
cd ~/LeRobot-Anything-U-Arm/src/uarm/scripts
python -m data_collection.arx5_collect_data --list-cameras
```

输出示例：
```
=== Connected RealSense Cameras ===
[Camera 1]
  Name: Intel RealSense D435I
  Serial: 123456789012
[Camera 2]
  Name: Intel RealSense D455
  Serial: 987654321098
```

### 2. 开始采集数据

#### 基本使用（自动检测相机）

```bash
python -m data_collection.arx5_collect_data --task pick_cube
```

#### 指定相机序列号

```bash
python -m data_collection.arx5_collect_data --task pick_cube \
    --wrist-camera 123456789012 \
    --external-camera 987654321098
```

#### 仅相机测试模式（不连接机器人）

```bash
python -m data_collection.arx5_collect_data --task test --camera-only
```

#### Headless 模式（无显示器/远程环境）

```bash
python -m data_collection.arx5_collect_data --task pick_cube --headless
```

在 headless 模式下，使用文本命令控制（输入后按 Enter）：
- `s` - 开始/暂停录制
- `save` - 保存为成功
- `f` - 保存为失败  
- `d` - 丢弃当前轨迹
- `q` - 退出
- `h` - 显示帮助

### 3. 采集控制

启动后使用键盘控制（GUI 模式）：

| 按键 | 功能 |
|------|------|
| `Space` | 开始/暂停录制 |
| `Enter` | 保存当前轨迹（标记为成功） |
| `F` | 保存当前轨迹（标记为失败） |
| `Backspace` | 丢弃当前轨迹 |
| `Q` | 退出 |

> 💡 **提示**：Headless 模式使用文本命令，详见上文。

### 4. 数据预处理

采集完成后，对原始数据进行预处理：

```bash
python -m data_collection.preprocess_dataset \
    --input ~/.arx_demos/raw/pick_cube/20231215_120000 \
    --output ~/.arx_demos/processed/pick_cube
```

常用选项：

```bash
# 处理短轨迹（最小帧数设为 2）
python -m data_collection.preprocess_dataset \
    --input ~/.arx_demos/raw/test/session \
    --output ~/.arx_demos/processed/test \
    --min-length 2

# 跳过清洗步骤（保留所有帧）
python -m data_collection.preprocess_dataset \
    --input ~/.arx_demos/raw/test/session \
    --output ~/.arx_demos/processed/test \
    --no-clean
```

预处理功能：
- 图像缩放（640x480 → 256x256）
- 去除静止帧（关节位移 < 阈值）
- 去除速度异常帧
- 轨迹平滑（Savitzky-Golay 滤波）
- 计算归一化统计量

### 5. 轨迹回放验证

#### 可视化回放

```bash
python -m data_collection.replay_trajectory \
    --traj-path ~/.arx_demos/processed/pick_cube/trajectory.h5 \
    --visual-only
```

#### 数据完整性验证

```bash
python -m data_collection.replay_trajectory \
    --traj-path ~/.arx_demos/processed/pick_cube/trajectory.h5 \
    --verify --summary
```

#### 物理回放（在机械臂上执行）

```bash
python -m data_collection.replay_trajectory \
    --traj-path ~/.arx_demos/processed/pick_cube/trajectory.h5 \
    --execute --traj-idx 0 --speed 0.5
```

## 📊 数据格式

### 原始数据目录结构

```
~/.arx_demos/raw/{task_name}/{timestamp}/
├── config.yaml                 # 采集配置
├── episode_0000/
│   ├── robot_data.h5          # 机器人状态和动作
│   ├── metadata.json          # 轨迹元数据
│   ├── wrist_rgb/             # 腕部相机 RGB
│   │   ├── 000000.png
│   │   └── ...
│   ├── wrist_depth/           # 腕部相机深度（16-bit PNG, mm）
│   │   ├── 000000.png
│   │   └── ...
│   ├── external_rgb/          # 外部相机 RGB
│   └── external_depth/        # 外部相机深度
├── episode_0001/
└── ...
```

### 处理后 HDF5 格式（ManiSkill 兼容）

```
trajectory.h5
├── traj_0/
│   ├── obs/
│   │   ├── joint_pos    [T, 6]     # 关节位置 (rad)
│   │   ├── joint_vel    [T, 6]     # 关节速度 (rad/s)
│   │   ├── gripper_pos  [T, 1]     # 夹爪位置 (m)
│   │   └── images/
│   │       ├── wrist/
│   │       │   ├── rgb      [T, H, W, 3]  uint8
│   │       │   └── depth    [T, H, W]     uint16 (mm)
│   │       └── external/
│   │           ├── rgb
│   │           └── depth
│   ├── actions          [T, 7]     # 目标关节位置 + 夹爪
│   └── attrs:
│       ├── success      bool
│       ├── num_steps    int
│       └── episode_id   int
├── traj_1/
└── ...

trajectory.json                      # ManiSkill 元数据格式
stats.json                          # 归一化统计量
```

## ⚙️ 配置说明

### 默认配置

```python
# 相机配置
resolution = (640, 480)      # 采集分辨率
fps = 30                     # 帧率
enable_depth = True          # 启用深度

# 机器人配置
joint_dof = 6               # 关节自由度
gripper_range = (0.0, 0.08) # 夹爪范围 (m)

# 采集配置
control_freq = 30           # 采集频率 (Hz)
max_episode_steps = 1000    # 最大步数
```

### 自定义配置

创建配置文件：

```bash
python -m data_collection.arx5_collect_data --save-config my_config.yaml
# 编辑 my_config.yaml
python -m data_collection.arx5_collect_data --config my_config.yaml --task my_task
```

## 🔧 依赖

```bash
pip install pyrealsense2 opencv-python h5py scipy pyyaml tqdm numpy
```

## 📝 注意事项

1. **相机自动分配**：未指定序列号时，按检测顺序分配（第一个→wrist，第二个→external）
2. **深度数据**：以 16-bit PNG 格式存储（单位：毫米），读取时需除以 1000 转换为米
3. **坐标系**：关节角度为弧度制，夹爪位置为米
4. **时间同步**：相机帧和机器人状态通过时间戳进行软同步
5. **数据量**：30Hz 双相机 RGB-D 约 50MB/min，建议使用 SSD 存储
