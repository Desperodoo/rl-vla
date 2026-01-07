# 真机部署指南

本指南描述如何在真实 ARX5 机械臂上部署 RTC (Real-Time Chunking) 推理系统。

## 目录

1. [前置条件](#前置条件)
2. [架构概述](#架构概述)
3. [部署步骤](#部署步骤)
4. [参数调优](#参数调优)
5. [监控与调试](#监控与调试)
6. [常见问题](#常见问题)

---

## 前置条件

### 硬件要求

- ARX5 X5 机械臂（带夹爪）
- 2 个 RealSense 相机（wrist + external）
- CAN 适配器（USB-CAN 或 PCIe CAN）
- GPU（推荐 RTX 3080 或更高）

### 软件要求

```bash
# 1. 激活 conda 环境
conda activate arx-py310

# 2. 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/lizh/rl-vla/arx5-sdk/python
export PYTHONPATH=$PYTHONPATH:/home/lizh/rl-vla/rlft/diffusion_policy

# 3. 设置 CAN 设备
cd /home/lizh/rl-vla
bash scripts/setup_can_devices.sh
```

### 验证环境

```bash
# 检查 arx5 SDK
python -c "import arx5_interface as arx5; print('arx5 OK')"

# 检查 RealSense
python -c "import pyrealsense2 as rs; print(f'Found {len(rs.context().query_devices())} cameras')"

# 检查 PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 架构概述

### RTC 核心概念

参考 [Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix):

```
时间线:
                    |<- inference_delay ->|<- execute_horizon - inference_delay ->|
                    |                     |                                        |
    t=0         t=33ms              t=100ms                               t=266ms
     |--------------|---------------------|--------------------------------------|
     ^              ^                     ^                                      ^
     |              |                     |                                      |
  推理开始      完成推理          开始执行新 chunk                         chunk 结束
     
  执行上一轮 chunk[0:inference_delay]
                    |
                    执行本轮 chunk[inference_delay:execute_horizon]
```

**关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `action_chunk_size` | 8 | 每次推理输出的关键帧数 |
| `inference_delay` | 1-3 | 推理期间执行的动作数（取决于推理延迟）|
| `execute_horizon` | 3-8 | 每轮执行的动作数 |
| `dt_key` | 33.3ms | 关键帧时间间隔 (30Hz) |
| `policy_dt` | 100ms | 推理间隔 (10Hz) |

### 进程架构

```
┌────────────────────────────────────────────────────┐
│           Python 推理进程 (inference_main.py)        │
│                                                     │
│   Camera (30Hz) → Policy (10Hz) → SHM Writer       │
└────────────────────────────────────────────────────┘
                         │
                    共享内存
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│           Python/C++ 伺服进程                        │
│                                                     │
│   SHM Reader → Interpolator → Safety → Robot       │
│             @500Hz                                  │
└────────────────────────────────────────────────────┘
```

---

## 部署步骤

### 第一步：延迟测试

在真机部署前，先测量系统延迟：

```bash
cd /home/lizh/rl-vla

# 完整延迟测试（相机 + 推理）
python -m inference_rtc.tests.test_inference_latency \
    --checkpoint /home/lizh/rl-vla/rlft/diffusion_policy/runs/consistency_flow-pick_cube-real__1__1766132160/checkpoints/latest.pt \
    --samples 100

# 仅测试相机延迟
python -m inference_rtc.tests.test_inference_latency --no-checkpoint --samples 50
```

**期望结果**:
- 相机采集: < 10ms
- 预处理: < 5ms
- 推理: < 80ms
- E2E: < 100ms

根据延迟结果设置 `inference_delay`:
- 如果推理 P95 < 33ms: `inference_delay = 1`
- 如果推理 P95 < 66ms: `inference_delay = 2`
- 如果推理 P95 < 100ms: `inference_delay = 3`

### 第二步：集成测试（模拟）

在不连接机器人的情况下测试全流程：

```bash
# Dry-run 模式测试
python -m inference_rtc.tests.test_integration \
    --npz /home/lizh/rl-vla/trajectory_multi_control.npz \
    --dry-run \
    --speed 3.0 \
    --verbose
```

检查输出：
- Write frequency 应接近 333.3ms
- Read latency 应 < 10ms
- 无错误或警告

### 第三步：真机慢速测试

使用低速度进行首次真机测试：

```bash
# 30% 速度测试
python -m inference_rtc.tests.test_integration \
    --npz /home/lizh/rl-vla/trajectory_multi_control.npz \
    --speed 0.3 \
    --verbose
```

**安全检查清单**:
- [ ] 机器人周围无障碍物
- [ ] 夹爪处于打开状态
- [ ] 急停按钮在手边
- [ ] 先观察 2-3 个周期确认运动正常

### 第四步：完整推理部署

#### 方案 A：单进程 Python 部署（推荐新手）

```bash
# 使用 Python servo_main.py
cd /home/lizh/rl-vla

# 终端 1：启动伺服进程
python -m inference_rtc.python.servo_main \
    --shm rtc_keyframes \
    --control-shm robot_control_state

# 终端 2：启动推理进程
python -m inference_rtc.python.inference_main \
    -c /home/lizh/rl-vla/rlft/diffusion_policy/runs/consistency_flow-pick_cube-real__1__1766132160/checkpoints/latest.pt \
    --shm rtc_keyframes \
    --control-shm robot_control_state
```

#### 方案 B：C++ 伺服 + Python 推理（推荐生产环境）

```bash
# 终端 1：启动 C++ 伺服进程
cd /home/lizh/rl-vla/inference_rtc/build
./servo_main --model X5 --interface can0 --shm rtc_keyframes

# 终端 2：启动 Python 推理进程
python -m inference_rtc.python.inference_main \
    -c /path/to/checkpoint.pt \
    --shm rtc_keyframes
```

---

## 参数调优

### 基于延迟测试结果调优

修改 [inference_rtc/python/config.py](inference_rtc/python/config.py):

```python
@dataclass
class InferenceConfig(RTCConfig):
    # 推理相关
    policy_rate: float = 10.0           # 推理频率 (Hz)
    num_flow_steps: int = 5             # Flow 采样步数
    
    # 关键帧输出
    act_horizon: int = 8                # 输出关键帧数
    dt_key: float = 1.0 / 30.0          # 关键帧间隔
    
    # 设备
    device: str = "cuda"
```

### 伺服参数调优

修改 [inference_rtc/python/servo_main.py](inference_rtc/python/servo_main.py) 中的安全参数:

```python
# 安全限制 (根据机器人能力调整)
MAX_JOINT_DELTA = 0.1      # rad/step @ 500Hz
MAX_GRIPPER_DELTA = 0.001  # m/step @ 500Hz
EMA_ALPHA = 0.3            # 平滑系数 (越小越平滑，延迟越大)
```

### 推理优化选项

如果推理延迟过高：

1. **减少 flow steps**:
   ```python
   num_flow_steps = 3  # 默认 5，减少可加速
   ```

2. **启用 FP16 推理**:
   ```python
   with torch.cuda.amp.autocast():
       keyframes = policy_runner.infer(images, state)
   ```

3. **使用更小的模型**（需要重新训练）

---

## 监控与调试

### 实时监控

推理进程会显示：
```
[Policy] t=1.234s | infer=45.2ms | write=0.3ms | total=45.5ms
```

伺服进程会显示：
```
[Servo] read_lat=2.1ms | track_err=0.012rad | update=yes
```

### 常用指标

| 指标 | 健康范围 | 含义 |
|------|---------|------|
| `infer` | < 100ms | 单次推理延迟 |
| `read_lat` | < 10ms | 共享内存读取延迟 |
| `track_err` | < 0.1rad | 实际位置 vs 目标位置误差 |
| `write_freq` | ~333ms | 关键帧写入频率 |

### Debug 模式

```bash
# 启用详细日志
python -m inference_rtc.python.inference_main ... --verbose

# 保存轨迹日志
python -m inference_rtc.python.inference_main ... --log-trajectory /tmp/traj.npz
```

### 紧急停止

- **Ctrl+C**: 安全停止，机器人回到 home 位置
- **急停按钮**: 立即断电停止

---

## 常见问题

### Q1: 推理延迟不稳定

**症状**: 推理时间波动大 (10ms ~ 200ms)

**解决方案**:
1. 首次推理会有 JIT 编译，需要 warmup
2. 禁用 GPU 电源管理:
   ```bash
   sudo nvidia-smi -pm 1
   ```
3. 检查其他 GPU 进程

### Q2: 机器人运动不平滑

**症状**: 运动有抖动或顿挫

**解决方案**:
1. 降低 EMA alpha:
   ```python
   EMA_ALPHA = 0.2  # 更平滑
   ```
2. 检查 CAN 通信是否稳定:
   ```bash
   candump can0  # 观察数据流
   ```
3. 增加 safety limiter 的 rate limit

### Q3: 共享内存连接失败

**症状**: `TimeoutError: 无法连接到共享内存`

**解决方案**:
1. 确保推理进程先启动（创建共享内存）
2. 检查共享内存名称是否匹配
3. 清理残留共享内存:
   ```bash
   ls /dev/shm/
   rm /dev/shm/rtc_keyframes*  # 清理
   ```

### Q4: 相机帧丢失

**症状**: `[警告] 相机帧为空`

**解决方案**:
1. 检查相机连接:
   ```bash
   realsense-viewer  # GUI 检查
   ```
2. 检查 USB 带宽（多相机需要 USB3.0）
3. 降低相机分辨率或 FPS

### Q5: Tracking error 过大

**症状**: 实际位置与目标偏差 > 0.3 rad

**原因**: 这是正常的！由于安全限制和 EMA 滤波，实际轨迹会"落后"于目标

**如果需要更精确跟踪**:
1. 增加 EMA alpha (但会更不平滑)
2. 减少 rate limit (需要确保安全)
3. 使用更慢的演示轨迹训练

---

## 完整启动脚本

创建 `run_rtc.sh`:

```bash
#!/bin/bash
set -e

# 环境设置
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arx-py310
export PYTHONPATH=$PYTHONPATH:/home/lizh/rl-vla/arx5-sdk/python
export PYTHONPATH=$PYTHONPATH:/home/lizh/rl-vla/rlft/diffusion_policy

cd /home/lizh/rl-vla

# 检查 CAN
if ! ip link show can0 > /dev/null 2>&1; then
    echo "设置 CAN 设备..."
    bash scripts/setup_can_devices.sh
fi

# 清理旧共享内存
rm -f /dev/shm/rtc_keyframes* /dev/shm/robot_control_state* 2>/dev/null || true

# 启动
echo "启动推理系统..."
echo "按 Ctrl+C 安全停止"

# 单进程集成测试模式
python -m inference_rtc.python.inference_main \
    -c /home/lizh/rl-vla/rlft/diffusion_policy/runs/consistency_flow-pick_cube-real__1__1766132160/checkpoints/latest.pt \
    --verbose
```

使用：
```bash
chmod +x run_rtc.sh
./run_rtc.sh
```

---

## 参考资料

- [Real-Time Chunking 论文](https://arxiv.org/abs/2506.07339)
- [Training-Time Action Conditioning](https://arxiv.org/abs/2512.05964)
- [官方仓库](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
- [ARX5 SDK 文档](../arx5-sdk/README.md)
