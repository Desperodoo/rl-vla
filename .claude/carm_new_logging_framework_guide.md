# CARM 新推理记录框架使用说明

本文档说明当前新的推理记录框架、推理启动方式，以及两类分析脚本的用法。

## 1. 框架概览

现在 inference 侧只保留三类 canonical 产物：

- `inference_episode_*.hdf5`：由 `InferenceRecorder` 生成，作为训练回流和样本级分析的主文件
- `timeline_*.jsonl`：由 `TimelineLogger` 生成，用于时间线与 chunk 分析
- `run_info_*.json`：由 `InferenceLogger` 生成，保存本次运行配置、文件映射和摘要

### 职责分工

- `InferenceRecorder`
  - 记录 episode 级数据
  - 保存图像、qpos、qpos_end、gripper、timestamps、action、intervention 标记
- `TimelineLogger`
  - 记录 `obs / inference / chunk / control` 等事件
  - 用于分析时延、chunk 重叠、control 频率
- `InferenceLogger`
  - 只保存运行配置快照与文件映射
  - 不再写旧的 `inference_*.hdf5` 诊断文件

## 2. 推理脚本用法

推理入口：

- `carm_ros_deploy/src/carm_deploy/inference/inference_ros.py`

基本启动方式：

```bash
rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt
```

### 2.1 仅推理，不录制

```bash
rosrun carm_deploy inference_ros.py \
  --pretrain /path/to/model.pt
```

效果：

- 正常执行推理与控制
- 默认会写 timeline
- 不生成 `inference_episode_*.hdf5`
- 不需要按 `R/Y/N`

### 2.2 推理 + 录制

```bash
rosrun carm_deploy inference_ros.py \
  --pretrain /path/to/model.pt \
  --record_inference
```

效果：

- 启用 recorder
- `R`：开始 / 停止一个 episode
- `Y`：确认保存当前 episode
- `N`：丢弃当前 episode
- 保存后的训练回流文件格式为 `inference_episode_*.hdf5`

### 2.3 推理 + 录制 + 人工干预

```bash
rosrun carm_deploy inference_ros.py \
  --pretrain /path/to/model.pt \
  --record_inference \
  --intervention
```

效果：

- 允许键盘干预动作
- 同时记录 recorder 数据
- 仍然支持 `R/Y/N`

### 2.4 重要参数

#### 记录相关

- `--record_inference`
  - 开启 episode 录制
- `--record_dir`
  - recorder 输出目录
  - 默认使用 `--log_dir`，再不指定则使用默认 inference logs 目录
- `--log_dir`
  - `run_info_*.json` 的输出目录
  - 也作为默认日志目录

#### 时间线相关

- `--timeline_enabled`
  - 开启 timeline 记录
- `--timeline_disabled`
  - 显式关闭 timeline 记录
- `--timeline_log`
  - 指定 JSONL 输出路径
- `--timeline_control_stride`
  - control 事件记录间隔
- `--chunk_time_base {sys_time, obs_stamp}`
  - chunk 时间基准

#### 推理执行相关

- `--execution_mode {temporal_ensemble, receding_horizon}`
- `--act_horizon`
- `--crossfade_steps`
- `--truncate_at_act_horizon`
- `--control_freq`
- `--inference_speed_scale`

### 2.5 输出文件怎么对应

一次完整记录后，通常会看到：

- `inference_episode_0001_YYYYMMDD_HHMMSS.hdf5`
- `timeline_YYYYMMDD_HHMMSS.jsonl`
- `run_info_YYYYMMDD_HHMMSS.json`

`run_info` 里会记录这些文件名的映射关系。

## 3. 数据分析脚本用法

### 3.1 样本级分析：`analyze_inference_data.py`

脚本路径：

- `carm_ros_deploy/src/carm_deploy/inference/analyze_inference_data.py`

这个脚本面向 `InferenceRecorder` 产出的 `inference_episode_*.hdf5`。

#### 最常见用法

```bash
python carm_ros_deploy/src/carm_deploy/inference/analyze_inference_data.py \
  --data_dir /home/amax/rl-vla/inference_logs
```

#### 指定文件

```bash
python carm_ros_deploy/src/carm_deploy/inference/analyze_inference_data.py \
  --files /home/amax/rl-vla/inference_logs/inference_episode_0001_20260401_221942.hdf5
```

#### 指定匹配模式

默认 pattern 是：

```text
inference_episode_*.hdf5
```

如果想改匹配模式：

```bash
python carm_ros_deploy/src/carm_deploy/inference/analyze_inference_data.py \
  --data_dir /home/amax/rl-vla/inference_logs \
  --pattern 'inference_episode_*.hdf5'
```

#### 关闭可视化

```bash
python carm_ros_deploy/src/carm_deploy/inference/analyze_inference_data.py \
  --data_dir /home/amax/rl-vla/inference_logs \
  --no_viz
```

#### 保存图表到指定目录

```bash
python carm_ros_deploy/src/carm_deploy/inference/analyze_inference_data.py \
  --data_dir /home/amax/rl-vla/inference_logs \
  --save_dir /home/amax/rl-vla/inference_logs/analysis
```

#### 这个脚本会看什么

- `action`
- `action_model`
- `action_intervened`
- `intervention_mask`
- `observations/images`
- `observations/gripper`
- `observations/qpos`
- `observations/qpos_end`
- `observations/qpos_joint`
- `observations/timestamps`

#### 注意

- 这个脚本不分析 `timeline_*.jsonl`
- 也不读取 `run_info_*.json`
- 它只针对 recorder 的 HDF5 样本文件

### 3.2 时间线分析：`analyze_timeline.py`

脚本路径：

- `carm_ros_deploy/src/carm_deploy/tools/analyze_timeline.py`

这个脚本面向 `timeline_*.jsonl`。

#### 基本统计

```bash
python carm_ros_deploy/src/carm_deploy/tools/analyze_timeline.py \
  --logs /home/amax/rl-vla/inference_logs/timeline_20260401_221855.jsonl
```

#### 多文件合并分析

```bash
python carm_ros_deploy/src/carm_deploy/tools/analyze_timeline.py \
  --logs /home/amax/rl-vla/inference_logs/timeline_*.jsonl
```

#### 输出统计 JSON

```bash
python carm_ros_deploy/src/carm_deploy/tools/analyze_timeline.py \
  --logs /home/amax/rl-vla/inference_logs/timeline_20260401_221855.jsonl \
  --out /home/amax/rl-vla/inference_logs/timeline_summary.json
```

#### 生成可视化

```bash
python carm_ros_deploy/src/carm_deploy/tools/analyze_timeline.py \
  --logs /home/amax/rl-vla/inference_logs/timeline_20260401_221855.jsonl \
  --visualize \
  --fig_out /home/amax/rl-vla/inference_logs/timeline.png
```

#### 只看指定时间范围

```bash
python carm_ros_deploy/src/carm_deploy/tools/analyze_timeline.py \
  --logs /home/amax/rl-vla/inference_logs/timeline_20260401_221855.jsonl \
  --visualize \
  --time_range 0 30
```

#### 覆盖 act_horizon

```bash
python carm_ros_deploy/src/carm_deploy/tools/analyze_timeline.py \
  --logs /home/amax/rl-vla/inference_logs/timeline_20260401_221855.jsonl \
  --act_horizon 8
```

#### 可用参数

- `--logs`
  - 一个或多个 timeline JSONL 文件
- `--out`
  - 输出统计 JSON
- `--visualize`
  - 开启可视化
- `--fig_out`
  - timeline 图输出路径
- `--hist_out`
  - 直方图输出路径
- `--act_horizon`
  - 覆盖日志中的 act_horizon
- `--max_chunks`
  - 最多显示多少个 chunk
- `--time_range start end`
  - 只展示指定秒数范围

#### 这个脚本会看什么

- `init`
- `obs`
- `inference`
- `chunk`
- `control`
- `record_step`

#### 常见指标

- `delta_obs`
- `delta_chunk_obs`
- `delta_action_obs`
- `control_lag`
- chunk overlap
- act_horizon 内 chunk 切换次数

## 4. 建议的日常流程

### 推理后检查

1. 启动推理并录制
2. 保存 episode
3. 检查目录中是否出现：
   - `inference_episode_*.hdf5`
   - `timeline_*.jsonl`
   - `run_info_*.json`
4. 用 `analyze_inference_data.py` 看样本级质量
5. 用 `analyze_timeline.py` 看时间线与 chunk 行为

### 快速排查

如果你发现：

- 没有生成 episode 文件
  - 确认是否加了 `--record_inference`
- `R/Y/N` 没反应
  - 确认是否开启了 `--record_inference` 或 `--intervention`
- 样本分析找不到文件
  - 检查 `--pattern` 是否还是 `inference_episode_*.hdf5`
- 时间线分析没数据
  - 检查 `timeline_enabled` 是否关闭，或 `--timeline_log` 路径是否正确

## 5. 备注

- 旧的 `inference_*.hdf5` 诊断日志已不作为默认产物
- 如果你只关心训练回流数据，优先看 `inference_episode_*.hdf5`
- 如果你只关心推理时延和 chunk 语义，优先看 `timeline_*.jsonl`
- 如果你想恢复一次完整运行的参数和文件映射，优先看 `run_info_*.json`
