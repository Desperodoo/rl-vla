# CARM Inference Log Format Specification

本文档描述推理日志系统的数据格式，包含三类文件：
- `run_info.json`: 运行配置和元数据（人类可读的快速查看）
- `*.hdf5`: 详细的数值数据记录
- `timeline_*.jsonl`: 时间线事件记录（用于时序分析）

## 1. run_info.json

每次推理运行开始时生成，包含完整的运行配置，便于快速了解本次推理的设置。

### 结构

```json
{
  "version": "2.0",
  "created_at": "2026-01-25T14:30:00.123456",
  "model": {
    "path": "/path/to/model.pt",
    "algorithm": "consistency_flow",
    "action_mode": "full",
    "state_mode": "joint_only",
    "obs_horizon": 2,
    "pred_horizon": 16,
    "action_dim": 13,
    "action_dim_full": 15,
    "visual_encoder_type": "resnet18",
    "use_ema": false,
    "num_inference_steps": 10
  },
  "normalizer": {
    "enabled": true,
    "mode": "standard",
    "action_stats": {
      "mean": [0.0, 1.5, -0.8, ...],
      "std": [0.1, 0.2, 0.15, ...]
    }
  },
  "control": {
    "control_freq": 50,
    "teleop_scale": 0.4,
    "joint_cmd_mode": false,
    "gripper_hysteresis_window": 1
  },
  "execution": {
    "mode": "temporal_ensemble",
    "act_horizon": 10,
    "max_active_chunks": null,
    "crossfade_steps": 0,
    "truncate_at_act_horizon": true,
    "temporal_factor_k": 0.05,
    "pos_lookahead_step": 1,
    "chunk_time_base": "sys_time"
  },
  "safety": {
    "config_path": "/path/to/carm_deploy/safety_config.json",
    "check_workspace": true,
    "max_relative_translation": 0.1
  },
  "files": {
    "hdf5": "inference_20260125_143000.hdf5",
    "timeline": "timeline_20260125_143000.jsonl"
  }
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `version` | string | 日志格式版本 |
| `model.path` | string | 模型 checkpoint 路径 |
| `model.algorithm` | string | 算法类型: consistency_flow, flow_matching, diffusion_policy |
| `model.action_mode` | string | 动作模式: full (15D), ee_only (8D) |
| `model.state_mode` | string | 状态模式: joint_only, ee_only, both |
| `normalizer.enabled` | bool | 是否启用动作归一化 |
| `normalizer.action_stats` | object | 归一化统计量 (mean, std) |
| `control.teleop_scale` | float | 遥操作缩放因子 (0-1)，影响动作幅度 |
| `execution.mode` | string | 执行模式: temporal_ensemble, basic_chunking |

---

## 2. HDF5 数据文件 (*.hdf5)

存储详细的数值数据，支持高效随机访问。

### 结构

```
inference_YYYYMMDD_HHMMSS.hdf5
├── observations/                    # 观测数据
│   ├── step_000000/
│   │   ├── qpos_joint    [7]       # 关节角度 [6 joints + 1 gripper]
│   │   └── qpos_end      [8]       # 末端位姿 [xyz, qxyzw, gripper]
│   ├── step_000001/
│   │   └── ...
│   └── ...
│
├── predictions/                     # 预测数据
│   ├── step_000000/
│   │   ├── raw_action       [action_dim_full]  # 模型原始输出（反归一化后）
│   │   └── executed_action  [8]                # 实际发送给机械臂的动作
│   ├── step_000001/
│   │   └── ...
│   └── ...
│
├── timing/                          # 时间数据
│   ├── step_000000/
│   │   ├── timestamp        float  # Unix 时间戳
│   │   └── inference_time   float  # 推理耗时 (秒)
│   └── ...
│
├── safety/                          # 安全数据
│   ├── step_000000/
│   │   ├── clipped     bool        # 是否被安全裁剪
│   │   └── warnings    string      # 安全警告 (JSON 数组)
│   └── ...
│
└── attrs (HDF5 attributes)          # 元数据
    ├── start_time     string       # 开始时间 (ISO 格式)
    ├── end_time       string       # 结束时间 (ISO 格式)
    ├── num_steps      int          # 总步数
    ├── model_path     string       # 模型路径
    └── config         string       # 运行配置 (JSON)
```

### 动作维度说明

| action_mode | raw_action | executed_action | 说明 |
|-------------|------------|-----------------|------|
| full (15D)  | [15] | [8] | raw: [joint(6), grip(1), rel_pose(7), grip(1)]<br>exec: [abs_pose(7), grip(1)] |
| ee_only (8D)| [8]  | [8] | raw: [rel_pose(7), grip(1)]<br>exec: [abs_pose(7), grip(1)] |

**注意**：
- `raw_action` 是模型输出经过反归一化后的动作（相对位姿）
- `executed_action` 是经过 `apply_relative_transform` 转换后的绝对位姿
- 两者的区别对于调试"动作幅度"问题非常重要

---

## 3. Timeline 文件 (timeline_*.jsonl)

JSONL 格式的时间线记录，每行一个 JSON 对象。专注于时间语义分析。

### 事件类型

#### 3.1 init 事件

推理开始时记录一次，包含执行参数。

```json
{
  "event": "init",
  "t_sys": 1737793639.123,
  "execution_mode": "temporal_ensemble",
  "act_horizon": 10,
  "pred_horizon": 16,
  "obs_horizon": 2,
  "control_freq": 50,
  "teleop_scale": 0.4,
  "temporal_factor_k": 0.05,
  "chunk_time_base": "sys_time"
}
```

#### 3.2 inference 事件

每次模型推理时记录。

```json
{
  "event": "inference",
  "t_sys": 1737793639.500,
  "step": 100,
  "inference_time": 0.030,
  "action_norm": 0.0152
}
```

| 字段 | 说明 |
|------|------|
| `step` | 推理步数 |
| `inference_time` | 推理耗时 (秒) |
| `action_norm` | 第一个动作的位移幅度 (米) |

#### 3.3 chunk 事件

新 chunk 创建时记录。

```json
{
  "event": "chunk",
  "t_sys": 1737793639.520,
  "chunk_id": 5,
  "chunk_base_time": 1737793639.456,
  "num_actions": 10,
  "action_interval": 0.02
}
```

#### 3.4 control 事件

控制命令发送时记录（每 100 步记录一次，或异常时记录）。

```json
{
  "event": "control",
  "t_sys": 1737793639.600,
  "step": 100,
  "num_candidates": 3,
  "used_chunk_ids": [4, 5]
}
```

#### 3.5 episode_end 事件

Episode 结束时记录。

```json
{
  "event": "episode_end",
  "t_sys": 1737793700.000,
  "total_steps": 1500,
  "duration": 60.877
}
```

---

## 4. 文件命名约定

```
inference_logs/
├── run_info_20260125_143000.json       # 运行配置
├── inference_20260125_143000.hdf5      # 数据文件
└── timeline_20260125_143000.jsonl      # 时间线
```

所有文件使用相同的时间戳后缀，便于关联。

---

## 5. 分析示例

### 5.1 快速查看运行配置

```bash
cat run_info_*.json | jq '.control.teleop_scale, .normalizer.enabled'
```

### 5.2 提取动作数据

```python
import h5py
import numpy as np

with h5py.File('inference_*.hdf5', 'r') as f:
    # 获取所有 raw_action
    raw_actions = []
    for step_key in sorted(f['predictions'].keys()):
        if 'raw_action' in f['predictions'][step_key]:
            raw_actions.append(f['predictions'][step_key]['raw_action'][:])
    raw_actions = np.array(raw_actions)
    
    # 分析相对位姿幅度 (full mode: 维度 7-13)
    rel_pos = raw_actions[:, 7:10]
    print(f"位移幅度: {np.linalg.norm(rel_pos, axis=1).mean():.6f} m")
```

### 5.3 分析时间线

```python
import json

with open('timeline_*.jsonl', 'r') as f:
    events = [json.loads(line) for line in f]

# 计算推理频率
infer_events = [e for e in events if e['event'] == 'inference']
times = [e['t_sys'] for e in infer_events]
freq = 1.0 / np.diff(times).mean()
print(f"实际推理频率: {freq:.1f} Hz")
```

---

## 6. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 2.0 | 2026-01-25 | 添加 run_info.json，精简 timeline，添加 executed_action |
| 1.0 | 2026-01-20 | 初始版本 |
