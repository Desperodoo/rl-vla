# inference_rtc 已完成工作

## 概述

RTC (Real-Time Chunking) 架构的核心框架已搭建完成，包括共享内存通信、插值器、安全限制器等关键组件。

---

## 1. 目录结构 ✅

```
inference_rtc/
├── README.md                    # 项目文档
├── CMakeLists.txt              # C++ 构建配置
├── run_rtc.sh                  # 启动脚本
├── shared/
│   ├── __init__.py
│   ├── shm_protocol.py         # Python 共享内存协议
│   └── shm_protocol.hpp        # C++ 共享内存协议头文件
├── cpp/
│   ├── servo_main.cpp          # C++ 伺服主程序 (未编译通过)
│   ├── cubic_interpolator.hpp  # 三次样条插值器
│   ├── safety_limiter.hpp      # 安全限制器
│   ├── trajectory_buffer.hpp   # 轨迹缓冲区 (Phase B)
│   └── test_interpolator.cpp   # 插值器测试
├── python/
│   ├── __init__.py
│   ├── config.py               # 配置类
│   ├── policy_runner.py        # 策略推理器
│   ├── inference_main.py       # Python 推理主程序
│   └── servo_main.py           # Python 伺服 (备用方案)
└── tests/
    ├── __init__.py
    ├── test_e2e.py             # 端到端测试
    └── test_shm_comm.py        # 共享内存通信测试
```

---

## 2. 共享内存协议 ✅

### 内存布局
```
Offset  Size    Field       Description
0       8       version     写入版本号 (uint64, 每次写入递增)
8       8       t_write     写入时间戳 (double, time.time())
16      4       dof         自由度 (int32, 固定为 7)
20      4       H           关键帧数量 (int32, 默认 8)
24      8       dt_key      关键帧时间间隔 (double, 1/30s)
32      H*dof*8 q_key       关键帧数据 (double[H][dof])
```

### Python 端 (shared/shm_protocol.py)
- `ShmKeyframeWriter`: 创建共享内存，写入关键帧
- `write_keyframes(q_key)`: 写入并递增 version

### C++ 端 (shared/shm_protocol.hpp)
- `ShmKeyframeReader`: 连接共享内存，读取关键帧
- 双读校验保证数据一致性

---

## 3. 三次样条插值器 ✅

### C++ 版本 (cpp/cubic_interpolator.hpp)
- `CubicSpline1D`: 单关节三次样条
- `CubicInterpolator`: 多关节插值器
- 支持 30Hz 关键帧 → 500Hz 采样

### Python 版本 (python/servo_main.py)
- 使用 `scipy.interpolate.CubicSpline`
- API: `build(q_key, dt_key, t0)` + `eval(t_query)`

---

## 4. 安全限制器 ✅

### 功能
1. **关节位置限位**: 默认 [-π, π]，夹爪 [0, 0.08]
2. **速率限制**: max_joint_delta = 0.1 rad/step @500Hz
3. **EMA 平滑**: alpha = 0.3

### 实现
- C++: `cpp/safety_limiter.hpp`
- Python: `python/servo_main.py` 中的 `SafetyLimiter` 类

---

## 5. Python 推理进程 ✅

### policy_runner.py
- 加载模型 checkpoint (自动检测 FlowMatching/Diffusion/ConsistencyFlow)
- 维护观测缓冲区 (obs_buffer)
- `predict_keyframes()`: 返回 shape (H, dof) 的关键帧

### inference_main.py
- `RTCInferenceNode`: 主控类
- 复用 `CameraManager` 获取图像
- 复用 `SharedState` 获取机器人状态
- 10Hz 主循环：采集观测 → 推理 → 写入共享内存
- OpenCV 预览窗口 (Space/R/Q 控制)

---

## 6. Python 伺服进程 (备用) ✅

### python/servo_main.py
完整的 Python 伺服实现，作为 C++ 编译不通过时的备用方案：
- `ShmKeyframeReader`: 读取共享内存
- `CubicInterpolator`: 三次样条插值
- `EMAFilter`: 指数移动平均滤波
- `SafetyLimiter`: 安全限制
- `PythonServo`: 500Hz 主循环

---

## 7. 轨迹缓冲区 (Phase B) ✅

### cpp/trajectory_buffer.hpp
- `TrajectoryBuffer` 类
- 支持 committed/editable 区域划分
- `update_simple()`: Phase A 简单模式
- `update_rtc()`: Phase B RTC 模式，带混合窗口

---

## 8. 启动脚本 ✅

### run_rtc.sh
```bash
./run_rtc.sh -c /path/to/checkpoint.pt [--dry-run] [--cpp] [--verbose]
```
- 默认使用 Python 伺服
- `--cpp` 使用 C++ 伺服
- `--dry-run` 不连接真实机器人

---

## 9. 依赖安装 (rlft_ms3 环境) ✅

在 `rlft_ms3` conda 环境中已安装：
```bash
mamba install -c robostack-staging ros-humble-kdl-parser ros-humble-ament-cmake
mamba install conda-forge::soem=1.4.0  # 必须是 1.4.0，2.0.0 有兼容问题
mamba install pyparsing=3.0.9          # 避免与 ROS1 冲突
pip install scipy
```

arx5-sdk CMakeLists.txt 已修改，添加了：
```cmake
link_directories($ENV{CONDA_PREFIX}/lib)
```

arx5_interface Python 模块可正常导入。

---

## 10. 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| policy_rate | 10 Hz | 策略推理频率 |
| servo_rate | 500 Hz | 伺服控制频率 |
| H | 8 | 每次输出的关键帧数量 |
| dt_key | 1/30 s | 关键帧时间间隔 |
| commit_time | 100 ms | 已提交轨迹时长 (Phase B) |
| blend_window | 30 ms | 混合窗口时长 (Phase B) |
| ema_alpha | 0.3 | EMA 平滑系数 |
| max_joint_delta | 0.1 rad | 单步最大关节变化 |
