# 多进程推理系统

这是 ARX5 机器人的多进程推理系统，将安全关键的控制逻辑与 GPU 推理完全分离，确保机器人控制的稳定性和安全性。

## 架构概述

```
┌─────────────────────┐     SharedMemory     ┌─────────────────────┐
│   推理进程          │◄───────────────────►│   控制进程          │
│   (inference_node)  │                      │   (control_node)    │
│                     │   action_buffer      │                     │
│  - 摄像头采集       │   robot_state        │  - 500Hz 控制循环   │
│  - GPU 推理         │   control_flags      │  - 机器人硬件独占   │
│  - UI 显示          │                      │  - 安全看门狗       │
└─────────────────────┘                      └─────────────────────┘
```

### 为什么用多进程？

原来的单进程多线程架构存在以下问题：

1. **竞态条件**: 状态切换时主线程和控制线程同时访问 `robot_ctrl`，导致不一致
2. **推理崩溃影响控制**: GPU 推理或摄像头出问题时，整个程序崩溃，机器人失去控制
3. **GIL 限制**: Python 的 GIL 使得多线程无法真正并行

多进程架构的优势：

1. **故障隔离**: 推理进程崩溃不影响控制进程
2. **真正并行**: 推理和控制在独立的 CPU 核心上运行
3. **明确的所有权**: 只有控制进程能操作机器人硬件

## 文件结构

```
inference/
├── __init__.py           # 包初始化
├── shared_state.py       # 共享内存通信模块
├── control_node.py       # 控制进程 (安全核心)
├── inference_node.py     # 推理进程 (GPU + 摄像头)
└── tests/
    ├── __init__.py
    ├── test_multiprocess.py   # 单元测试
    └── benchmark.py           # 性能基准测试
```

## 快速开始

### 1. 运行测试

```bash
# 运行所有测试
./scripts/run_inference.sh test

# 或单独运行
python -m inference.tests.test_multiprocess --test shm      # 共享内存
python -m inference.tests.test_multiprocess --test mp       # 多进程通信
python -m inference.tests.test_multiprocess --test safety   # 安全机制
```

### 2. 模拟模式

不连接真实硬件，测试通信流程：

```bash
# 终端 1: 启动控制节点
python -m inference.control_node --dry-run

# 终端 2: 启动推理节点 (需要 checkpoint)
python -m inference.inference_node -c /path/to/checkpoint.pt --dry-run
```

### 3. 真实硬件

```bash
# 终端 1: 启动控制节点
python -m inference.control_node --model X5 --interface can0

# 终端 2: 启动推理节点
python -m inference.inference_node -c /path/to/checkpoint.pt
```

### 4. 性能基准

```bash
# 测试不同 flow_steps 的推理延迟
python -m inference.tests.benchmark -c /path/to/checkpoint.pt --flow-steps 5,10,15,20 --components
```

## 键盘控制

在推理节点窗口中：

| 按键 | 功能 |
|------|------|
| Space | 开始/暂停推理 |
| R | 复位到初始位置 |
| Q | 安全退出 |
| Esc | 强制退出 |

## 共享内存布局

```
偏移量      内容                大小
0x000      版本号 (uint32)      4 bytes
0x004      控制标志 (uint32)    4 bytes
0x008      请求标志 (uint32)    4 bytes
0x00C      确认标志 (uint32)    4 bytes
0x010      错误标志 (uint32)    4 bytes
0x020      控制时间戳 (float64) 8 bytes
0x028      推理时间戳 (float64) 8 bytes
0x040      关节位置 (7xfloat64) 56 bytes
0x080      关节速度 (7xfloat64) 56 bytes
0x100      动作缓冲区索引       4 bytes
0x104      动作缓冲区大小       4 bytes
0x108      动作数据 (36x7)      2016 bytes
```

## 状态机

```
                    ┌─────────────┐
                    │    IDLE     │
                    └──────┬──────┘
                           │ START
                           ▼
    ┌─────────┐     ┌─────────────┐
    │ PAUSED  │◄────│   RUNNING   │
    └────┬────┘     └──────┬──────┘
         │                 │
         │ RESET           │ STOP
         ▼                 ▼
    ┌─────────────┐  ┌─────────────┐
    │  RESETTING  │  │EMERGENCY_STOP│
    └──────┬──────┘  └──────┬──────┘
           │                │
           ▼                ▼
    ┌─────────────┐  ┌─────────────┐
    │    IDLE     │  │   DAMPING   │
    └─────────────┘  └─────────────┘
```

## 安全机制

### 1. 看门狗超时

控制进程监控推理进程的心跳时间戳，如果超过 `watchdog_timeout`（默认 1 秒），自动进入 PAUSED 状态保持当前位置。

### 2. 命令超时

如果动作缓冲区为空超过 `command_timeout`（默认 200ms），控制进程会保持当前位置，不会执行无效动作。

### 3. 进程崩溃处理

- 推理进程崩溃：控制进程继续运行，保持当前位置
- 控制进程崩溃：机器人进入阻尼模式（由 SDK 保证）

### 4. 优雅退出

- PAUSE：保存当前位置，持续发送该位置命令
- STOP：保存当前位置，进入阻尼模式后退出

## 配置参数

### 控制节点

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | X5 | 机器人型号 |
| `--interface` | can0 | CAN 接口 |
| `--urdf` | auto | URDF 文件路径 |
| `--control-freq` | 500 | 控制频率 (Hz) |
| `--command-timeout` | 0.2 | 命令超时 (秒) |
| `--watchdog-timeout` | 1.0 | 看门狗超时 (秒) |
| `--dry-run` | False | 模拟模式 |

### 推理节点

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-c, --checkpoint` | 必需 | 模型检查点路径 |
| `--num-flow-steps` | 10 | 扩散步数 |
| `--inference-freq` | 30 | 推理频率 (Hz) |
| `--camera-id` | 0 | 摄像头 ID |
| `--no-display` | False | 禁用显示 |
| `--dry-run` | False | 模拟模式 |

## 常见问题

### Q: 推理进程崩溃后机器人怎么办？

A: 控制进程会继续运行，看门狗检测到推理超时后自动进入 PAUSED 状态，机器人会保持当前位置。你可以重启推理进程继续工作。

### Q: 如何调整推理延迟？

A: 使用基准测试找到合适的 `num_flow_steps`：
```bash
python -m inference.tests.benchmark -c checkpoint.pt --flow-steps 5,10,15,20
```

### Q: 共享内存泄漏警告怎么处理？

A: 这是 Python 的已知问题，不影响功能。如果想手动清理：
```bash
# 查看共享内存
ls /dev/shm/

# 删除残留
rm /dev/shm/arx5_control
```

### Q: 为什么动作有时会卡顿？

A: 可能原因：
1. GPU 推理延迟波动 → 降低 `num_flow_steps`
2. 摄像头采集延迟 → 检查 USB 连接
3. 动作缓冲区不够大 → 增加 `chunk_size`

## 开发指南

### 添加新的传感器

在 `inference_node.py` 中添加传感器读取，通过共享内存或 ZeroMQ 发送给控制进程。

### 修改控制频率

修改 `control_node.py` 中的 `--control-freq` 参数，但要确保 SDK 支持该频率。

### 添加新的状态

在 `shared_state.py` 的 `ControlFlags` 类中添加新状态，并在两个节点中处理。
