# inference_rtc 待完成工作

## 概述

Phase A 的核心代码已完成，但尚未在真实机械臂上验证。以下是迁移到机械臂平台后需要完成的工作。

---

## 🔴 优先级 1: 环境验证

### 1.1 验证 arx5_interface 导入
```bash
source ~/miniforge-pypy3/etc/profile.d/conda.sh
conda activate rlft_ms3
cd /path/to/rl-vla
export PYTHONPATH=$PYTHONPATH:$(pwd)/arx5-sdk/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/arx5-sdk/lib/x86_64:$CONDA_PREFIX/lib
python -c "import arx5_interface; print('OK')"
```

### 1.2 如果 arx5-sdk 需要重新编译
```bash
conda activate rlft_ms3
# 确保 soem 版本是 1.4.0
conda install conda-forge::soem=1.4.0

cd arx5-sdk
rm -rf build && mkdir build && cd build
cmake .. && make -j4
```

### 1.3 验证 scipy 安装
```bash
pip install scipy
```

---

## 🔴 优先级 2: 修复测试代码

### 2.1 修复 test_e2e.py API 不匹配

当前测试文件使用的 API 与实际实现不匹配，需要修改：

**错误的调用方式:**
```python
interpolator = CubicInterpolator(config.dof)  # 错误
interpolator.update(...)                       # 错误
interpolator.is_ready()                        # 错误
interpolator.sample(t)                         # 错误
limiter = SafetyLimiter(config.dof)           # 错误
limiter.apply(q)                              # 错误
```

**正确的调用方式:**
```python
from inference_rtc.python.servo_main import CubicInterpolator, SafetyLimiter, ServoConfig

interpolator = CubicInterpolator()
interpolator.build(q_key, dt_key, t0)          # 构建样条
interpolator.valid                             # 检查是否有效
interpolator.eval(t)                           # 采样

servo_config = ServoConfig(dof=7)
limiter = SafetyLimiter(servo_config)
limiter.apply(target, current)                 # 需要两个参数
```

### 2.2 运行测试
```bash
cd /path/to/rl-vla
python -m inference_rtc.tests.test_e2e
```

---

## 🟡 优先级 3: Phase A 端到端验证

### 3.1 准备 checkpoint

确保有可用的模型 checkpoint，包含：
- `config.yaml` 或 `config.json`
- `model_*.pt` 或 `policy_*.pt`

### 3.2 设置 CAN 设备
```bash
# USB-CAN
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# 或使用脚本
./scripts/setup_can_devices.sh
```

### 3.3 Python 伺服模式测试 (推荐先用这个)
```bash
cd inference_rtc
./run_rtc.sh -c /path/to/checkpoint.pt --verbose
```

### 3.4 观察要点
- 共享内存是否正常创建和读取
- 插值是否平滑
- 机械臂是否响应
- 是否有抖动或异常

---

## 🟡 优先级 4: 调试与调参

### 4.1 需要调整的参数 (python/config.py)

```python
@dataclass
class RTCConfig:
    # 这些可能需要根据实际情况调整
    policy_rate: float = 10.0      # 如果推理慢，可降到 5Hz
    ema_alpha: float = 0.3         # 平滑程度，越小越平滑但延迟越大
    max_joint_delta: float = 0.1   # 单步最大变化，太大会抖动
    timeout_threshold: float = 0.15 # 超时阈值
```

### 4.2 调试工具

**查看共享内存状态:**
```python
# 在另一个终端运行
python -m inference_rtc.tests.test_shm_comm --reader
```

**查看推理帧率:**
```bash
# inference_main.py 会打印帧率统计
```

---

## 🟢 优先级 5: Phase B 实现

### 5.1 启用 RTC 模式

修改 `python/servo_main.py` 中的 `PythonServo` 类：

```python
# 当前是 Phase A 简单模式
# 需要实现 committed/editable 区域管理

class PythonServo:
    def __init__(self, config: ServoConfig):
        # 添加
        self.commit_time = 0.1      # 100ms
        self.blend_window = 0.03    # 30ms
        self.committed_traj = None
        self.editable_traj = None
```

### 5.2 实现混合窗口

在 `_step_rtc()` 中实现：
```python
def _step_rtc(self, t_now):
    # 1. 检查是否有新 chunk
    # 2. 如果有，只更新 editable 区域
    # 3. 在 committed 末尾附近使用混合窗口平滑过渡
    pass
```

---

## 🟢 优先级 6: C++ 伺服 (可选)

### 6.1 编译 C++ 伺服

如果 Python 伺服达不到 500Hz，需要用 C++：

```bash
cd inference_rtc
mkdir build && cd build
cmake ..
make -j4
```

### 6.2 可能的编译问题

如果遇到 `kdl_parser` 找不到：
```bash
# 检查 conda 环境是否激活
conda activate rlft_ms3

# 检查库路径
echo $CONDA_PREFIX/lib
ls $CONDA_PREFIX/lib/libkdl_parser.so
```

### 6.3 运行 C++ 伺服
```bash
./run_rtc.sh -c /path/to/checkpoint.pt --cpp
```

---

## 🔵 优先级 7: 安全兜底逻辑

### 7.1 超时处理 (待实现)

在 `PythonServo._loop()` 中添加：

```python
def _loop(self):
    while self._running:
        t_now = time.time()
        
        # 检查超时
        if t_now - self._last_update > self._config.timeout_threshold:
            self._handle_timeout(t_now)
        else:
            self._step(t_now)
```

### 7.2 超时行为

1. 0-150ms: 沿旧轨迹继续
2. 150-500ms: 保持当前位姿
3. >500ms: 缓慢回归 home 位置

---

## 📋 检查清单

迁移后按顺序完成：

- [ ] 验证 conda 环境 (rlft_ms3)
- [ ] 验证 arx5_interface 导入
- [ ] 验证 scipy 安装
- [ ] 修复 test_e2e.py API 调用
- [ ] 运行 test_e2e.py 通过
- [ ] 设置 CAN 设备
- [ ] 准备模型 checkpoint
- [ ] 运行 Python 伺服模式
- [ ] 观察机械臂响应
- [ ] 调整参数 (如需要)
- [ ] (可选) 编译 C++ 伺服
- [ ] (可选) 实现 Phase B RTC 模式

---

## 💡 常见问题

### Q1: ImportError: cannot import name 'xxx'
确保 PYTHONPATH 包含项目根目录：
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/rl-vla
```

### Q2: undefined symbol: EcatError
soem 版本问题，必须用 1.4.0：
```bash
conda install conda-forge::soem=1.4.0
```

### Q3: 机械臂不响应
1. 检查 CAN 连接: `ip link show can0`
2. 检查权限: `sudo setcap ...`
3. 检查机械臂电源

### Q4: 抖动严重
- 增大 `ema_alpha` (更平滑)
- 减小 `max_joint_delta` (限制速率)
- 检查推理帧率是否稳定

### Q5: 延迟太大
- 减小 `ema_alpha` (响应更快)
- 检查模型推理时间
- 考虑使用 C++ 伺服
