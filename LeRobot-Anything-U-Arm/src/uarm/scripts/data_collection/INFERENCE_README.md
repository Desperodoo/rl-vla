# ARX5 真机策略推理脚本

## 概述

`arx5_policy_inference.py` 是一个用于在 ARX5 机械臂上运行训练好的扩散策略（Diffusion Policy）的推理脚本。

### 特性

- **双线程架构**：控制线程（500Hz）和推理线程（~30Hz）分离，确保控制频率稳定
- **双相机支持**：同时显示 wrist 和 external 相机画面
- **多种控制方式**：支持 OpenCV 窗口键盘控制和终端输入
- **安全限制**：关节角度限制、速度限制、夹爪限制
- **平滑处理**：EMA滤波 + 三次样条插值

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Thread                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │            _inference_and_ui_loop() @ 30Hz              ││
│  │  - 获取相机图像                                          ││
│  │  - 运行策略推理                                          ││
│  │  - 更新动作缓冲区                                        ││
│  │  - 更新可视化                                            ││
│  │  - 处理键盘输入                                          ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                        共享状态（线程安全）
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Control Thread                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │             _control_loop() @ 500Hz                     ││
│  │  - 读取动作缓冲区                                        ││
│  │  - 应用 EMA 滤波                                        ││
│  │  - 应用安全限制                                          ││
│  │  - 发送命令到机械臂                                      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 使用方法

### 基本命令

```bash
# 进入脚本目录
cd ~/LeRobot-Anything-U-Arm/src/uarm/scripts

# 基本推理（需要连接机械臂和相机）
python -m data_collection.arx5_policy_inference \
  -c ~/rlft/diffusion_policy/runs/YOUR_EXP/checkpoints/iter_XXXXX.pt

# 测试模式（无机械臂执行）
python -m data_collection.arx5_policy_inference \
  -c checkpoint.pt --dry-run

# 无可视化（headless 模式）
python -m data_collection.arx5_policy_inference \
  -c checkpoint.pt --no-viz

# 更平滑的动作（降低 filter-alpha）
python -m data_collection.arx5_policy_inference \
  -c checkpoint.pt --filter-alpha 0.2

# 禁用滤波
python -m data_collection.arx5_policy_inference \
  -c checkpoint.pt --no-filter
```

### 完整参数列表

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | `-c` | 必填 | 模型检查点路径 (.pt) |
| `--dry-run` | - | False | 测试模式，不执行机械臂命令 |
| `--no-viz` | - | False | 禁用可视化 |
| `--max-steps` | - | 1000 | 最大控制步数 |
| `--control-freq` | - | 500.0 | 控制频率 (Hz) |
| `--no-filter` | - | False | 禁用 EMA 动作滤波 |
| `--filter-alpha` | - | 0.3 | EMA 系数，越小越平滑 |
| `--no-ema` | - | False | 不使用检查点中的 EMA 权重 |
| `--device` | - | cuda | 推理设备 |
| `--quiet` | `-q` | False | 减少输出 |

## 控制方式

### OpenCV 窗口控制

在预览窗口点击使其获得焦点后：

| 按键 | 功能 |
|------|------|
| `Space` | 开始/暂停推理 |
| `R` | 复位机械臂到初始位置 |
| `Q` | 退出程序 |

### 终端控制

随时在终端输入命令（按 Enter 确认）：

| 命令 | 功能 |
|------|------|
| `s` / `start` / `pause` | 开始/暂停推理 |
| `r` / `reset` | 复位机械臂 |
| `q` / `quit` / `exit` | 退出程序 |
| `h` / `help` | 显示帮助 |

## 状态说明

| 状态 | 颜色 | 说明 |
|------|------|------|
| `IDLE` | 灰色 | 空闲状态，等待开始 |
| `RUNNING` | 绿色 | 正在执行推理和控制 |
| `PAUSED` | 黄色 | 暂停，保持当前位置 |
| `RESETTING` | - | 复位中 |

## 配置参数

关键配置参数在 `InferenceConfig` 中：

```python
@dataclass
class InferenceConfig:
    # 图像尺寸
    image_size: Tuple[int, int] = (240, 320)  # (H, W)
    
    # 频率设置
    control_freq: float = 500.0    # 控制频率 (Hz)
    record_freq: float = 30.0      # 策略输出频率 (Hz)
    inference_freq: int = 30       # 推理循环频率 (Hz)
    
    # Horizon 设置
    obs_horizon: int = 2           # 观测历史长度
    pred_horizon: int = 16         # 预测长度
    act_horizon: int = 8           # 执行长度
    
    # 滤波设置
    enable_filter: bool = True     # 启用 EMA 滤波
    filter_alpha: float = 0.3      # EMA 系数 (0.1-0.5)
    
    # 安全限制
    max_joint_delta: float = 0.3   # 最大关节变化 (rad/step)
    max_gripper_delta: float = 0.02  # 最大夹爪变化 (m/step)
```

## 相机配置

默认使用两个 RealSense 相机：

- **wrist**: D435i (SN: 036222071712) - 安装在手腕
- **external**: D455 (SN: 037522250003) - 外部视角

分辨率：640x480 @ 30fps

## 故障排除

### 1. 相机连接失败
```bash
# 检查相机
rs-enumerate-devices
# 重新插拔 USB 或重启 realsense 服务
```

### 2. CAN 连接失败
```bash
# 检查 CAN 设备
ip link show can0
# 重新配置
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

### 3. 推理太慢
- 确保使用 GPU (`--device cuda`)
- 检查是否有其他 GPU 进程占用显存

### 4. 机械臂抖动
- 增加滤波强度：`--filter-alpha 0.15`
- 检查控制频率是否稳定

## 与数据采集脚本的对比

| 特性 | data_recorder.py | arx5_policy_inference.py |
|------|------------------|--------------------------|
| 主要功能 | 遥操作数据采集 | 策略推理执行 |
| 控制来源 | 主臂 teleoperation | 神经网络推理 |
| 双线程 | ✅ | ✅ |
| 双相机 | ✅ | ✅ |
| 键盘控制 | ✅ | ✅ |
| 500Hz 插值 | ✅ | ✅ |
| 安全限制 | ✅ | ✅ |
