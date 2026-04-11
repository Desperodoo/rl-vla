# CARM Inference Logging Format

本文档描述当前 `inference_ros.py` 推理日志与回流采集格式。

当前实现已经收敛到两条主线：

- `run_info_*.json`
  记录本次运行的配置、摘要和产物映射
- `inference_episode_*.hdf5`
  由 `InferenceRecorder` 记录 canonical 推理回流数据
- `timeline_*.jsonl`
  记录推理、chunk、control 的时序事件

旧阶段的 intervention / action_intervened / intervention_mask 语义已经移除，不再是当前格式的一部分。

## 1. run_info_*.json

`InferenceLogger` 在一次运行结束后写出 `run_info_*.json`。

### 结构

```json
{
  "version": "2.1",
  "created_at": "2026-04-11T20:50:16.123456",
  "model": {
    "path": "/path/to/checkpoint.pt",
    "algorithm": "consistency_flow",
    "action_mode": "ee_only",
    "state_mode": "joint_only",
    "obs_horizon": 2,
    "pred_horizon": 16,
    "action_dim": 8,
    "action_dim_full": 8,
    "use_ema": false,
    "num_inference_steps": 10
  },
  "normalizer": {
    "enabled": true,
    "mode": "standard"
  },
  "control": {
    "control_freq": 50,
    "teleop_scale": 1.0,
    "gripper_hysteresis_window": 1
  },
  "execution": {
    "mode": "receding_horizon",
    "act_horizon": 8,
    "max_active_chunks": null,
    "crossfade_steps": 0,
    "truncate_at_act_horizon": true,
    "temporal_factor_k": 0.05,
    "pos_lookahead_step": 1,
    "chunk_time_base": "sys_time",
    "desire_inference_freq": 30
  },
  "safety": {
    "config_path": "/path/to/safety_config.json",
    "check_workspace": true,
    "max_relative_translation": 0.1
  },
  "files": {
    "episode_hdf5": [
      "inference_episode_0001_20260411_205134.hdf5"
    ],
    "timeline": "timeline_20260411_205014.jsonl",
    "run_info": "run_info_20260411_205016.json"
  },
  "summary": {
    "num_steps": 900,
    "avg_inference_time": 0.021,
    "max_inference_time": 0.041,
    "safety_clips": 0,
    "safety_clip_rate": 0.0,
    "safety_reason_counts": {},
    "episode_success": true,
    "episode_outcome_label": "success"
  },
  "total_steps": 900,
  "ended_at": "2026-04-11T21:05:14.456789"
}
```

### 关键字段

| 字段 | 说明 |
|------|------|
| `files.episode_hdf5` | 本次运行保存的 canonical 推理回流 HDF5 |
| `summary.num_steps` | 实际写入 recorder 的步数 |
| `summary.safety_clips` | 被 safety layer 裁剪的步数 |
| `summary.safety_clip_rate` | safety clip 比例 |
| `summary.episode_success` | episode 结果标签 |

## 2. inference_episode_*.hdf5

`InferenceRecorder` 在 `--record_inference` 模式下记录 canonical 推理回流 HDF5。

### 结构

```text
inference_episode_0001_YYYYMMDD_HHMMSS.hdf5
├── observations/
│   ├── images                   [T, H, W, C]
│   ├── images_by_camera/
│   │   └── <camera_name>        [T, H, W, C]
│   ├── qpos_joint               [T, 7]
│   ├── qpos_end                 [T, 8]
│   ├── qpos                     [T, 15]
│   ├── gripper                  [T]
│   └── timestamps               [T]
├── action_model                 [T, pred_horizon, action_dim]
├── action_executed              [T, pred_horizon, action_dim]
├── action                       [T, action_dim]
└── attrs
    ├── num_steps
    ├── pred_horizon
    ├── action_dim
    ├── data_source = inference_rollout
    ├── timestamp_semantics = obs_stamp_ros
    ├── action_semantics_version = absolute_ee_target_pose_v2
    ├── action_space = ee_target_pose_absolute
    ├── compat_action_source = action_executed[:,0,:]
    ├── camera_topics
    ├── camera_names
    ├── primary_camera
    ├── success
    └── outcome_label
```

### 动作语义

| dataset | shape | 语义 |
|---------|-------|------|
| `action_model` | `[T, pred_horizon, 8]` | 模型输出经过安全检查后的 absolute target chunk |
| `action_executed` | `[T, pred_horizon, 8]` | 实际送入执行链的 absolute target chunk |
| `action` | `[T, 8]` | `action_executed[:, 0, :]` 的单步兼容视图 |

说明：

- 当前 rollout 语义以 `action_executed` 为准。
- `action_model` 用于分析“模型原本想发什么”与“执行链实际发了什么”之间的差异。
- 当前格式不再写入 `action_intervened`、`intervention_mask`、`has_intervention`、`intervention_ratio`。

## 3. timeline_*.jsonl

时间线日志由 `TimelineLogger` 写出，每行一个 JSON 事件。

### `init`

记录运行初始化参数。

```json
{
  "event": "init",
  "desire_inference_freq": 30,
  "temporal_factor_k": 0.05,
  "chunk_time_base": "sys_time",
  "act_horizon": 8,
  "pred_horizon": 16,
  "execution_mode": "receding_horizon",
  "teleop_scale": 1.0,
  "control_freq": 50
}
```

### `obs`

记录观测可用时刻与 `obs_stamp_ros` 的差值。

```json
{
  "event": "obs",
  "obs_stamp_ros": 1712843414.10,
  "t_obs_ready_sys": 1712843414.12,
  "delta_obs": 0.02
}
```

### `inference`

记录单次模型推理耗时。

```json
{
  "event": "inference",
  "t_infer_start": 1712843414.12,
  "t_infer_end": 1712843414.15,
  "inference_time": 0.03
}
```

### `chunk`

记录 action chunk 入队信息。

```json
{
  "event": "chunk",
  "chunk_id": 42,
  "chunk_base_time": 1712843414.15,
  "obs_stamp_ros": 1712843414.10,
  "action_interval": 0.02,
  "pred_horizon": 16,
  "act_horizon": 8,
  "num_actions_added": 8,
  "truncated": true,
  "delta_chunk_obs": 0.05
}
```

### `control`

记录控制线程取样与下发时刻。

```json
{
  "event": "control",
  "query_time": 1712843414.18,
  "t_send_sys": 1712843414.18,
  "candidate_timestamps": [1712843414.15],
  "weights": [1.0],
  "num_candidates": 1,
  "used_chunk_ids": [42]
}
```

## 4. inference_staging HDF5

`scripts/convert_inference_to_training_staging.py` 会把 canonical rollout 转成训练 staging HDF5。

### 结构

```text
episode_0001_YYYYMMDD_HHMMSS.hdf5
├── observations/...
├── action                       [T, 8]
└── attrs
    ├── dataset_type = inference_staging
    ├── staging_schema_version = inference_staging_v2
    ├── source_file
    ├── source_num_steps
    ├── kept_steps
    ├── dropped_steps
    ├── action_source_used = action_executed[:,0,:]
    ├── source_run_info
    ├── source_timeline
    ├── admission_label
    ├── admission_pass
    ├── admission_reason
    ├── admission_bucket
    ├── admission_policy
    ├── policy_version
    ├── min_steps
    ├── gold_max_safety_clip_rate
    ├── silver_max_safety_clip_rate
    ├── safety_clips
    ├── safety_clip_rate
    └── ...
```

### Admission 语义

- 当前 staging admission 不再使用 intervention ratio。
- 当前 bucket 主要由以下因素决定：
  - 是否缺失关键字段
  - 是否太短
  - 是否存在 NaN
  - `run_info.summary.safety_clip_rate`

## 5. 诊断建议

- 回看真实执行语义时，优先看 `action_executed`。
- 分析模型与执行差异时，对比 `action_model` vs `action_executed`。
- 检查计数一致性时，三处应基本一致：
  - `HDF5 attrs.num_steps`
  - `run_info.total_steps`
  - timeline 中 `obs` / `inference` / `chunk` 事件数
