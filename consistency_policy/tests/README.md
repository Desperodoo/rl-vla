# Consistency Policy 测试文档

本目录包含 Consistency Policy 机器人推理框架的测试脚本。

## 测试环境准备

### 1. 激活 Conda 环境

```bash
conda activate arx-py310
```

### 2. 硬件要求

- **机器人**: ARX5 机械臂 (通过 CAN 总线连接)
- **相机**: 
  - Intel RealSense D435i (腕部相机, SN: 036222071712)
  - Intel RealSense D455 (外部相机, SN: 037522250003) - 可选
- **CAN 接口**: 已配置 `can0` 设备

### 3. CAN 设备设置

```bash
# 运行 CAN 设置脚本
bash scripts/setup_can_devices.sh

# 或手动设置
sudo ip link set can0 up type can bitrate 1000000
```

## 测试脚本

### test_camera.py - RealSense 相机测试

验证 RealSense 相机模块是否正常工作。

**用法**:

```bash
# 检测连接的相机
python -m consistency_policy.tests.test_camera --detect

# 测试腕部相机 (D435i)
python -m consistency_policy.tests.test_camera --wrist

# 测试外部相机 (D455) 并录制
python -m consistency_policy.tests.test_camera --external --record

# 测试双相机同步
python -m consistency_policy.tests.test_camera --both

# 自定义测试时长
python -m consistency_policy.tests.test_camera --wrist -d 30
```

**测试项**:
- ✅ 相机检测
- ✅ 单相机采集 (RGB/深度)
- ✅ 录制功能 (仅 D455)
- ✅ 双相机同步

---

### test_controller.py - 多进程控制器测试

验证 Arx5JointControllerProcess 及其 Manager 接口。

**用法**:

```bash
# 运行所有测试
python -m consistency_policy.tests.test_controller

# 仅测试状态获取 (安全，不移动机器人)
python -m consistency_policy.tests.test_controller --state-only

# 测试 servoL 点对点移动
python -m consistency_policy.tests.test_controller --servo

# 测试轨迹调度
python -m consistency_policy.tests.test_controller --trajectory

# 跳过运动测试 (仅启动/停止和状态读取)
python -m consistency_policy.tests.test_controller --skip-motion
```

**测试项**:
- ✅ 控制器启动/停止
- ✅ 状态读取 (关节位置、速度、力矩)
- ✅ 状态读取性能 (期望 >1kHz)
- ⚠️ servoL 点对点移动 (需确认)
- ⚠️ add_waypoint + update_trajectory 轨迹调度 (需确认)

---

### test_robot_replay.py - 轨迹回放测试

从 NPZ 文件加载轨迹并在真实机器人上回放。

**用法**:

```bash
# 回放默认轨迹
python -m consistency_policy.tests.test_robot_replay

# 指定轨迹文件
python -m consistency_policy.tests.test_robot_replay --trajectory path/to/trajectory.npz

# 调整速度
python -m consistency_policy.tests.test_robot_replay --speed-scale 0.5

# 使用较大批次
python -m consistency_policy.tests.test_robot_replay --batch-size 32

# 跳过 HOME 位置
python -m consistency_policy.tests.test_robot_replay --skip-home
```

**测试项**:
- ✅ NPZ 轨迹加载
- ✅ 多进程控制器初始化
- ✅ 批量轨迹调度 (add_waypoint + update_trajectory)
- ✅ 轨迹跟踪精度分析
- ✅ 状态记录和保存

---

### test_inference.py - 策略推理测试

验证 Consistency Policy 模型推理。

**用法**:

```bash
# 测试模型加载
python -m consistency_policy.tests.test_inference --model-only

# 测试完整推理 (需要相机)
python -m consistency_policy.tests.test_inference

# 指定模型检查点
python -m consistency_policy.tests.test_inference -c path/to/checkpoint.pt
```

**测试项**:
- ✅ 模型加载
- ✅ 推理时间测量 (~70ms 目标)
- ✅ 输出格式验证

---

## 测试顺序建议

1. **相机测试** (`test_camera.py`)
   - 确保相机正确连接和识别
   - 验证图像质量

2. **控制器测试** (`test_controller.py --state-only`)
   - 确保机器人连接正常
   - 验证状态读取

3. **控制器运动测试** (`test_controller.py`)
   - 在安全环境下测试运动
   - 验证定位精度

4. **轨迹回放测试** (`test_robot_replay.py`)
   - 验证完整控制链
   - 测试轨迹跟踪性能

5. **推理测试** (`test_inference.py`)
   - 验证策略模型
   - 测试推理速度

## 故障排查

### 相机未检测到

```bash
# 检查 USB 连接
lsusb | grep Intel

# 重置 USB
sudo usbreset /dev/bus/usb/XXX/YYY
```

### CAN 通信失败

```bash
# 检查 CAN 状态
ip link show can0
candump can0

# 重启 CAN
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
```

### 控制器进程崩溃

```bash
# 检查日志
dmesg | tail -20

# 确保没有残留进程
pkill -f arx5
```

## 期望测试结果

参见 [TEST_RESULTS.md](TEST_RESULTS.md) 获取测试结果模板和基准值。
