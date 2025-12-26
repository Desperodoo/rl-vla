# 数据采集与回放时序问题分析

## 问题现象

在使用 `replay_trajectory.py` 回放采集的轨迹时，观察到以下现象：

1. **动作速度过快**：回放速度比实际录制时快很多（约 15-17 倍）
2. **动作不流畅**：机械臂运动出现抖动和跳跃
3. **总时长不匹配**：原本 30 秒的轨迹在几秒内就回放完成

## 问题原理

### 时序系统概述

ARX5 机械臂的遥操作系统涉及两个不同的时钟频率：

```
┌─────────────────────────────────────────────────────────────┐
│                    数据采集系统                              │
├─────────────────────────────────────────────────────────────┤
│  控制循环频率:  ~500 Hz  (controller_dt ≈ 2ms)              │
│  ├── 读取主臂角度                                           │
│  ├── 发送命令到从臂                                         │
│  └── 更新机器人状态                                         │
│                                                             │
│  数据录制频率:  ~30 Hz   (record_dt ≈ 33ms)                 │
│  ├── 获取相机帧                                             │
│  ├── 保存机器人状态                                         │
│  └── 保存图像到磁盘                                         │
└─────────────────────────────────────────────────────────────┘
```

### 问题根源

在 `replay_trajectory.py` 的 `physical_replay()` 方法中，原始代码使用了**控制器 dt** 而不是**录制 dt**：

```python
# 错误代码
ctrl_cfg = ctrl.get_controller_config()
dt = float(ctrl_cfg.controller_dt)  # ≈ 0.002s (500Hz)
...
time.sleep(dt / speed_factor)  # 每帧只等待 2ms！
```

而数据是以 **30Hz (~33ms/帧)** 的频率录制的。

### 速度计算

```
实际回放速度 = 录制频率 / 回放频率
            = (1/0.033) / (1/0.002)
            = 30 / 500
            = 0.06x  （太慢）

等等，这里的计算反了...

正确计算：
录制时每帧间隔: 33ms
回放时每帧间隔: 2ms
速度倍数 = 33ms / 2ms = 16.5x  （太快！）
```

这解释了为什么 30 秒的轨迹在 ~2 秒内就回放完了。

## 相关数据

从实际采集的数据中验证：

```
=== 原始数据 ===
总帧数: 870
时间跨度: 29.82s
平均帧间隔: 34.3ms
帧间隔 std: 1.7ms
采集频率: 29.1Hz

=== 处理后数据 ===
traj_0:
  帧数: 870
  success: False
  ⚠️ 没有时间戳数据!   <-- 问题：预处理丢失了时间戳
```

## 问题链条

```
1. 数据采集 (30Hz, dt=33ms)
   ↓
2. 预处理 (preprocess_dataset.py)
   ↓ ⚠️ 时间戳未保存到输出 HDF5
   ↓
3. 轨迹回放 (replay_trajectory.py)
   ↓ ⚠️ 无法获取原始 dt，使用了控制器 dt (2ms)
   ↓
4. 回放速度 = 33ms/2ms ≈ 16.5x 原速
```

## 受影响的代码

### 1. preprocess_dataset.py

`_save_consolidated_hdf5()` 方法没有保存时间戳：

```python
# 当前代码只保存了这些字段：
obs.create_dataset("joint_pos", data=ep["joint_pos"], compression="gzip")
obs.create_dataset("joint_vel", data=ep["joint_vel"], compression="gzip")
obs.create_dataset("gripper_pos", data=ep["gripper_pos"], compression="gzip")
# ❌ 缺少: obs.create_dataset("timestamps", ...)
# ❌ 缺少: grp.attrs["control_dt"] = ...
```

### 2. replay_trajectory.py

`physical_replay()` 方法使用了错误的时间间隔：

```python
# 当前代码：
dt = float(ctrl_cfg.controller_dt)  # 2ms - 控制器频率
time.sleep(dt / speed_factor)

# 应该是：
record_dt = 1.0 / 30.0  # 33ms - 录制频率
time.sleep(record_dt / speed_factor)
```

## 修复方案

### 方案 1: 在预处理时保存时序信息

```python
# preprocess_dataset.py - _save_consolidated_hdf5()

# 保存时间戳
if "timestamps" in ep:
    obs.create_dataset("timestamps", data=ep["timestamps"], compression="gzip")

# 保存帧率信息到属性
if "timestamps" in ep and len(ep["timestamps"]) > 1:
    dt = np.diff(ep["timestamps"]).mean()
    grp.attrs["control_dt"] = float(dt)
    grp.attrs["control_freq"] = float(1.0 / dt)
```

### 方案 2: 回放时正确获取时间间隔

```python
# replay_trajectory.py - physical_replay()

# 优先从轨迹属性获取 dt
if "control_dt" in traj_grp.attrs:
    record_dt = float(traj_grp.attrs["control_dt"])
elif "obs/timestamps" in traj_grp:
    timestamps = np.array(traj_grp["obs/timestamps"])
    record_dt = np.diff(timestamps).mean()
else:
    # 回退到默认 30Hz
    record_dt = 1.0 / 30.0
    print("Warning: No timing info, using default 30Hz")

# 使用录制 dt 而不是控制器 dt
time.sleep(record_dt / speed_factor)
```

## 验证方法

修复后，可以通过以下方式验证：

```bash
# 1. 重新预处理数据
python -m data_collection.preprocess_dataset \
    --input ~/.arx_demos/raw/pick_cube/20251218_222509 \
    --output ~/.arx_demos/processed/pick_cube/20251218_222509_fixed

# 2. 检查时间戳是否保存
python -c "
import h5py
with h5py.File('~/.arx_demos/processed/.../trajectory.h5', 'r') as f:
    grp = f['traj_0']
    print('control_dt:', grp.attrs.get('control_dt', 'MISSING'))
    print('timestamps:', 'obs/timestamps' in grp)
"

# 3. 回放并对比时长
python -m data_collection.replay_trajectory \
    --traj-path ~/.arx_demos/processed/.../trajectory.h5 \
    --execute --traj-idx 0 --speed 1.0
# 预期：870帧 * 33ms ≈ 29秒
```

## 总结

| 项目 | 错误值 | 正确值 |
|------|--------|--------|
| 回放 dt | 2ms (500Hz) | 33ms (30Hz) |
| 870帧回放时长 | ~1.7s | ~29s |
| 速度比 | 16.5x | 1.0x |

## 相关文件

- `data_collection/data_recorder.py` - 数据采集（正确以 30Hz 录制）
- `data_collection/preprocess_dataset.py` - 预处理（需要保存时间戳）
- `data_collection/replay_trajectory.py` - 轨迹回放（需要使用正确的 dt）

---

*文档创建日期: 2025-12-18*
*问题状态: ✅ 已修复 (2025-12-18)*

## 修复记录

### 修改内容

1. **preprocess_dataset.py** - `_save_consolidated_hdf5()`:
   - 新增 `record_dt_mean`, `record_dt_std`, `record_fps_nominal` 属性
   - 保留 `control_dt`, `control_freq` 作为兼容属性
   - 无时间戳时输出 warning 并使用 30Hz 默认值

2. **replay_trajectory.py** - `physical_replay()`:
   - 使用逐帧 dt (`np.diff(timestamps)`) 进行精确回放
   - 优先级: `obs/timestamps` → `record_dt_mean` → `control_dt` → 30Hz
   - 添加 dt clamp 保护 (1ms < dt < 100ms)
   - 输出预期/实际回放时长对比

### 验证步骤

```bash
# 1. 重新预处理数据
python -m data_collection.preprocess_dataset \
    --input ~/.arx_demos/raw/pick_cube/20251218_222509 \
    --output ~/.arx_demos/processed/pick_cube/test_fixed

# 2. 回放验证 (dry-run)
python -m data_collection.replay_trajectory \
    --traj-path ~/.arx_demos/processed/pick_cube/test_fixed/trajectory.h5 \
    --traj-idx 0 --speed 1.0

# 预期输出:
# Using per-frame timestamps: mean=34.3ms, std=1.7ms (29.1Hz)
# Expected replay duration: 29.8s (speed: 1.0x)
```
