# CARM 审计记录（Phase 1 / Phase 2）

日期：2026-03-30
范围：CARM 真机 teleop 采集、inference 记录、离线训练加载主路径
状态：进行中（基于代码阅读 + 样本文件/时间线抽查）

---

## 1. 本阶段结论摘要

### 已确认
1. **teleop 与训练主路径的数据语义大体一致**：
   - teleop 采集主文件名为 `episode_*.hdf5`
   - 训练 loader 当前也是按 `episode_*.hdf5` 扫描
   - teleop 主 action 语义为 `8D = target_pose(7) + gripper(1)`
   - `CARMDataset` 当前主线也是围绕 8D ee-only 语义构建 relative action

2. **teleop 与 inference 的 timestamp 语义不一致，必须修改**：
   - teleop `observations/timestamps` 保存的是 **ROS 图像时间戳**
   - inference `observations/timestamps` 保存的是 **system time / wall-clock**

3. **inference 的主视角保存语义与 teleop 不一致，必须对齐**：
   - teleop 显式支持 `primary_camera`
   - inference 当前默认直接使用 `obs['images'][0]`
   - 用户要求：**inference 也需要保存双视角，并对齐 teleop setting**

4. **teleop fallback 行为按用户要求保留，但训练数据侧需要过滤 inactive 帧**：
   - teleop inactive 时，采集脚本会把当前 `qpos_end` 写成 action，同时 `teleop_scale=0`
   - 用户明确要求：采集行为保留；后续在 dataset 里把 inactive 数据滤掉

5. **inference 的 `sys_time` chunk 基准不一定是 bug，但需要继续量化分析**：
   - 可能只是策略推理延迟造成的正常现象
   - 仍需进一步分析 obs→infer→chunk→control 的延迟链路

6. **新的 canonical inference 样本已验证可用，且三件套产物完整**：
   - `inference_episode_*.hdf5`、`timeline_*.jsonl`、`run_info_*.json` 均已生成并可解析
   - 当前架构在不做进一步重构的前提下可继续用于采样与审计
   - 仍需关注 recorder / timeline 的 1-step 边界差异，但这次样本没有暴露新的结构性问题

### 用户批注（已吸收）
- “P0-1: inference 数据现在不能直接进入训练主路径”：**当前阶段不是大问题**，因为此阶段还不打算让 inference 数据进入训练。
- “P0-2: teleop 与 inference 的时间戳语义不一致”：**确实需要修改**。
- “P1-1: primary camera 语义在 teleop 和 inference 两边不一致”：**inference 需要保存双视角，并对齐 teleop setting**。
- “P1-2: teleop fallback action 会把未激活遥操写成当前末端状态”：**保留 teleop 侧设置，但 dataset 要滤掉 inactive 数据**。
- “P1-3: inference 虽然是 receding_horizon，但 chunk 时间基默认是 sys_time”：**可能正常，但需要继续分析**。
- 新发现 1：**209 个文件里出现的 15D action 属于旧数据，已被手动删除**；后续统计只以当前保留样本为准。
- 新发现 2：**`inference_logs/inference_*.hdf5` 不是单一 recorder 产物，而是 recorder / logger 两条链路并存**；其中既有重复内容，也有不同内容，后续要合并链路并统一格式。

---

## 2. Phase 1：CARM 数据契约梳理

## 2.1 Teleop 采集主路径

关键文件：
- `carm_ros_deploy/src/carm_deploy/data/record_data_ros.py`
- `carm_ros_deploy/src/carm_deploy/core/env_ros.py`
- `carm_ros_deploy/src/carm_deploy/utils/image_sync.py`
- `carm_ros_deploy/src/carm_deploy/utils/timeline_logger.py`

### 2.1.1 teleop 观测来源
`RealEnvironment.get_observation()` 返回：
- `stamp`
- `images`
- `qpos_joint`
- `qpos_end`
- `gripper`

代码位置：
- `carm_ros_deploy/src/carm_deploy/core/env_ros.py:221-260`

说明：
- `stamp` 来源于 `ImageSynchronizer.get_images()`，底层取 ROS 图像消息 header timestamp
- `images` 是相机列表
- `qpos_joint` 是 7 维（6 joints + gripper）
- `qpos_end` 是 8 维（xyz + quat + gripper）

### 2.1.2 teleop episode 落盘格式
落盘逻辑在：
- `carm_ros_deploy/src/carm_deploy/data/record_data_ros.py:349-427`

文件名：
- `episode_{episode_count:04d}_{timestamp}.hdf5`

HDF5 结构：

```text
observations/
  images              [T,H,W,C]      # 主视角（兼容字段）
  images_by_camera/   group          # 多相机新格式
  qpos_joint          [T,7]
  qpos_end            [T,8]
  qpos                [T,15]         # 兼容旧版
  gripper             [T]
  timestamps          [T]            # ROS image stamp

action                [T,8]          # target_pose(7) + gripper(1)
teleop_scale          [T]
attrs:
  num_steps
  record_freq
  image_width
  image_height
  camera_topics
  camera_names
  primary_camera
  robot_ip
  created_at
  data_version='v3'
```

### 2.1.3 teleop action 语义
记录逻辑在：
- `carm_ros_deploy/src/carm_deploy/data/record_data_ros.py:468-484`

两种情况：
1. teleop active：
   - action = `target_pose + [gripper]`
   - teleop_scale = backend 返回的 scale
2. teleop inactive：
   - action = 当前 `obs['qpos_end']`
   - teleop_scale = 0.0

这意味着：
- teleop 数据里天然混有 active 和 inactive 帧
- inactive 帧可以通过 `teleop_scale == 0` 识别
- 用户要求：后续 dataset 过滤 inactive 帧，而不是改采集逻辑

### 2.1.4 teleop 时间线语义
在 `record_step()` 中记录 timeline：
- `obs_stamp_ros`
- `t_obs_ready_sys`
- `delta_obs = t_obs_ready_sys - obs_stamp_ros`
- `t_action_query_sys`
- `delta_action_obs = t_action_query_sys - obs_stamp_ros`

代码位置：
- `carm_ros_deploy/src/carm_deploy/data/record_data_ros.py:486-503`

抽样 timeline 观察：
- `recorded_data/fixed_dual_light/timeline_record_20260319_235212.jsonl`
- `recorded_data/random_no_light/timeline_record_20260320_231709.jsonl`
- `recorded_data/fixed_no_light/timeline_record_20260318_225939.jsonl`

当前观察：
- 多数 `delta_obs` / `delta_action_obs` 在约 **40ms ~ 60ms**
- 存在重复 `obs_stamp_ros` 被连续记录多次的情况，说明 recorder 可能重复采到了同一帧图像对应的 observation
- 这不会立刻破坏 schema，但会影响后续“真实采样频率”和“样本独立性”分析

---

## 2.2 训练加载主路径

关键文件：
- `rlft/datasets/data_utils.py`
- `rlft/datasets/carm_dataset.py`
- `rlft/offline/train_carm.py`

### 2.2.1 loader 识别规则
`load_carm_dataset()` 当前只扫描：
- `episode_*.hdf5`

代码位置：
- `rlft/datasets/data_utils.py:333-383`

这意味着：
- teleop 主数据会被训练主路径识别
- inference recorder 的 `inference_episode_*.hdf5` 当前不会自动被训练主路径识别

> 用户批注：这一点在当前调试阶段不是主要问题，因为 inference 数据本阶段本来就不打算直接进训练。

### 2.2.2 loader 读取字段
`load_carm_episode()` 读取：
- `observations/images`
- `qpos_joint`
- `qpos_end`
- `gripper`
- `timestamps`
- 可选 `action`
- 可选 `teleop_scale`

代码位置：
- `rlft/datasets/data_utils.py:289-321`

重要含义：
- 当前训练 loader 不读取 `observations/images_by_camera`
- 所以 teleop 虽然已保存多相机，但训练主路径实际只吃 `observations/images` 这个兼容字段

### 2.2.3 训练状态与图像预处理语义
`create_carm_obs_process_fn()`：
- 可 resize 到 `target_size`
- 图像支持 HWC → NCHW
- state 模式支持：`joint_only / ee_only / both`

代码位置：
- `rlft/datasets/data_utils.py:398-449`

状态定义：
- `joint_only` → `qpos_joint`（7D）
- `ee_only` → `qpos_end`（8D）
- `both` → `qpos_joint + qpos_end[:7]`（14D）

### 2.2.4 训练 action 语义
`CARMDataset`：
- 当前主线 action_dim 固定为 7（continuous relative ee pose）
- gripper 走离散 label
- `action_mode='full'` 已被弱化/弃用，内部强制落到 ee_only 主线

代码位置：
- `rlft/datasets/carm_dataset.py:161-167`

relative action 构造逻辑：
- 用 observation 对应帧的 `qpos_end` 当 reference pose
- 将未来 raw target pose 转为 relative pose
- gripper 单独用 threshold 做离散分类

代码位置：
- `rlft/datasets/carm_dataset.py:215-223`
- `rlft/datasets/carm_dataset.py:329-358`

### 2.2.5 对 inactive 数据过滤的启示
当前 `CARMDataset` 没有任何基于 `teleop_scale` 的过滤逻辑。
也就是说：
- 如果 teleop episode 含有大量 inactive 帧
- 它们会直接进入 relative action 构造
- 造成“保持当前 pose”的 target 被当成正常监督样本

这与用户要求冲突，因此需要后续改 dataset：
- 基于 `teleop_scale == 0` 过滤 inactive 样本
- 或至少在 slice 层过滤 reference/action 窗口中含 inactive 的样本

---

## 2.3 Inference 主路径

关键文件：
- `carm_ros_deploy/src/carm_deploy/inference/inference_ros.py`
- `carm_ros_deploy/src/carm_deploy/inference/policy_loader.py`
- `carm_ros_deploy/src/carm_deploy/inference/inference_recorder.py`
- `carm_ros_deploy/src/carm_deploy/utils/trajectory_interpolator.py`

### 2.3.1 inference 状态构造语义
`RealPolicy.build_state_from_obs()`：
- `joint_only` → `qpos_joint`
- `ee_only` → `qpos_end`
- `both` → `qpos_joint + qpos_end[:7]`

代码位置：
- `carm_ros_deploy/src/carm_deploy/inference/policy_loader.py:176-189`

这与训练侧 `create_carm_obs_process_fn()` 是一致的。

### 2.3.2 inference 图像语义
当前 inference 图像预处理默认：
- 只用 `obs['images'][0]`
- resize 到 policy 目标分辨率
- HWC → CHW

代码位置：
- `carm_ros_deploy/src/carm_deploy/inference/inference_ros.py:525-545`

这与 teleop 的 `primary_camera` 机制不一致。

> 用户要求：**inference 也要保存双视角，并对齐 teleop setting**。

### 2.3.3 inference 记录格式
`InferenceRecorder` 当前写出：
- 文件名：`inference_episode_*.hdf5`
- `observations/images`：当前只保存第一个相机
- `observations/timestamps`：系统时间
- `action_model`
- `action_intervened`
- `intervention_mask`
- root `action = action_intervened[:,0,:]`

代码位置：
- `carm_ros_deploy/src/carm_deploy/inference/inference_recorder.py:205-318`

### 2.3.4 inference 时间线语义
抽样 timeline：
- `inference_logs/timeline_20260327_195606.jsonl`

可见：
- `execution_mode = receding_horizon`
- `truncate_at_act_horizon = true`
- `chunk_time_base = sys_time`
- 常见 `delta_obs` 在约 **25ms ~ 60ms**
- 常见 `delta_chunk_obs` 在约 **90ms ~ 120ms**
- 首帧存在约 **0.48s** inference latency

当前解释：
- 这可能是策略推理开销、首帧 warmup 或 chunking 调度造成
- 暂不能直接判定为 bug
- 但必须在后续阶段继续分析 obs→infer→chunk→control 延迟闭环

---

## 3. 风险列表（基于当前阶段）

## 3.1 必须修复 / 高优先级

### R1. teleop 与 inference 的 timestamp 语义不一致
严重度：高

现状：
- teleop timestamp = ROS 图像时间戳
- inference timestamp = system wall-clock

影响：
- 两类数据无法用统一语义做时间线分析
- 后续如果要比较 teleop / inference 数据质量，会出现基准混乱
- 对 online RL 数据准入门槛设计不利

用户批注：
- **确实需要修改**

建议方向：
- 明确 `observations/timestamps` 的唯一语义（建议统一成 ROS obs stamp）
- 如有需要，额外新增 `timestamps_sys`，但不要混用一个字段表达两种时间基

### R2. inference 主视角与 teleop `primary_camera` 机制不一致
严重度：高

现状：
- teleop 支持 `primary_camera`
- inference 默认直接使用 `obs['images'][0]`

影响：
- 训练和部署图像语义可能漂移
- 多相机场景下，`images[0]` 不一定等于 teleop 配置的主视角

用户批注：
- **inference 时也需要保存双视角，请对齐 teleop setting**

建议方向：
- inference 侧引入与 teleop 相同的 `camera_names / primary_camera / images_by_camera` 语义
- `observations/images` 应始终表示“primary camera 兼容字段”
- 双视角完整保存到 `images_by_camera`

### R3. inactive teleop 数据会污染训练监督，需在 dataset 侧过滤
严重度：高

现状：
- teleop inactive 时 action = 当前 `qpos_end`
- `teleop_scale = 0`
- `CARMDataset` 当前不会过滤这些帧

影响：
- 训练样本中会混入大量“保持当前 pose”的伪监督
- 影响 relative action 分布
- 对后续 online RL warm-start 质量不利

用户批注：
- **保留 teleop 侧设置，但要在 dataset 中把 inactive 数据滤掉**

建议方向：
- dataset 加入 `filter_inactive=True` 路径
- 依据 `teleop_scale == 0` 过滤帧或 slice

## 3.2 当前阶段不作为 blocker 的项

### R4. inference 数据文件名不进入训练主路径
严重度：当前阶段降级

现状：
- 训练 loader 只认 `episode_*.hdf5`
- inference recorder 写 `inference_episode_*.hdf5`

原始判断：
- 这是训练回流兼容性缺口

用户批注：
- **当前调试阶段这不是大问题，因为本阶段本来不打算把 inference 数据直接送进训练**

结论：
- 在当前阶段先降级，不作为 Phase 1/2 blocker
- 但后续若进入 online RL 回流阶段，仍需重新提升优先级

### R5. `chunk_time_base=sys_time` 可能只是正常延迟表现
严重度：待分析

现状：
- timeline 显示 obs→chunk 有约 0.1s 量级延迟
- execution mode 是 receding_horizon

用户批注：
- **这个可能是策略推理延迟，可能正常，但需要进一步分析**

结论：
- 当前保留为“待量化问题”
- 进入后续 inference timeline 审计阶段再判断是否为真正风险

---

## 4. Phase 2：teleop 样本级 QA 当前进度

当前已完成：
- 样本文件定位与抽查
- timeline 抽查
- 代码层的 schema / append 对齐逻辑确认

当前仍在继续：
- episode 长度一致性全量统计
- timestamp 单调性/重复率统计
- fallback/inactive 比例统计
- action / qpos_end / quaternion 数值分布统计

### 已观察到的样本现象
1. `record_step()` 中所有核心字段是在同一次循环里同步 append 的：
   - `images`
   - `qpos_joint`
   - `qpos_end`
   - `qpos`
   - `gripper`
   - `timestamps`
   - `action`
   - `teleop_scale`

   代码位置：
   - `carm_ros_deploy/src/carm_deploy/data/record_data_ros.py:457-484`

   推断：
   - 结构上 episode 内各字段长度应当大多数情况下对齐

2. timeline 抽查中出现重复 `obs_stamp_ros`：
   - 例如 `timeline_record_20260319_235212.jsonl` 和 `timeline_record_20260320_231709.jsonl` 都能看到相邻 step 复用同一 obs stamp 的情况

   推断：
   - recorder 可能重复消费了同一帧图像的最新 observation
   - 需要在后续统计里区分“timestamp 非单调”与“timestamp 重复但非回退”

---

### 4.3 新增 recorder / logger 样本结论（2026-04-01）

已保存并核对这组新样本：
- `inference_logs/inference_episode_0001_20260401_221942.hdf5`
- `inference_logs/inference_20260401_221857.hdf5`
- `inference_logs/run_info_20260401_221857.json`
- `inference_logs/timeline_20260401_221855.jsonl`

#### 4.3.1 recorder HDF5 结果
`inference_episode_0001_20260401_221942.hdf5` 主要结论：
- `num_steps = 571`
- `action_dim = 8`，与当前 8D ee_only 目标一致
- `action_model / action_intervened / intervention_mask` 结构完整
- `has_intervention = False`，`intervention_ratio = 0.0`
- `observations/images` 为单相机主字段，shape = `[571, 480, 640, 3]`
- `observations/qpos_joint`, `qpos_end`, `qpos`, `gripper`, `timestamps` 全部存在且长度对齐
- `observations/timestamps` 单调递增，未见回退；相邻步长约 **67ms ~ 247ms**，均值约 **73.8ms**

初步判断：
- recorder 文件已经是“可继续做训练侧 QA”的主格式，和 `CARMDataset` 的单轨数据契约更接近
- 但它仍然沿用 **system time** 作为 `observations/timestamps`，这与 teleop 侧 ROS stamp 语义仍不一致，P0-2 仍成立
- 当前样本没有干预，说明 record-only 接线已经生效，recorder 没有把 intervention 误当成开启状态

#### 4.3.2 logger HDF5 结果
`inference_20260401_221857.hdf5` 主要结论：
- 结构仍是 step-wise 的诊断日志：`observations / predictions / timing / safety`
- `num_steps = 572`
- 每个 step 都有 `qpos_joint / qpos_end`，但默认没有保存图像（`save_images=False`）
- 与 recorder 不是同一类产物：logger 更适合时间线与安全分析，不适合作为训练主路径直接回流

#### 4.3.3 run_info / timeline 结果
`run_info_20260401_221857.json`：
- `model.action_mode = ee_only`
- `model.state_mode = joint_only`
- `model.action_dim = 7`，`model.action_dim_full = 8`
- `execution.mode = receding_horizon`
- `execution.truncate_at_act_horizon = true`
- `execution.chunk_time_base = sys_time`
- `control.control_freq = 50`

`timeline_20260401_221855.jsonl`：
- 共 **1925** 行事件，包含 **572** 个 `obs`、**572** 个 `chunk`、**208** 个 `control`
- `delta_obs` 平均约 **35.7ms**，范围约 **17.7ms ~ 75.2ms**
- `delta_chunk_obs` 平均约 **105.7ms**，范围约 **84.8ms ~ 433.5ms**
- 最后一个 chunk 的 `delta_chunk_obs` 约 **97.8ms**，和“推理 + chunk 调度”引入的系统时延相符

初步判断：
- `chunk_time_base = sys_time` 的 0.1s 量级延迟仍像是正常推理/调度开销，不足以直接判定为 bug
- 但它已经足够影响“线上实际执行动作”的时间基准分析，后续需要把 recorder / logger / timeline 三者联合起来看

#### 4.3.4 recorder vs logger 的直接结论
- recorder 与 logger 确实是两条并行链路，不是同一个产物的不同视图
- 两者 episode 边界不同：logger 从节点启动就开始记，recorder 只在按键开始后记
- 这次样本中 recorder 571 steps、logger 572 steps，说明二者并非严格一一对应
- 后续 audit 必须把“训练可回流数据”与“诊断日志”分开处理，避免把 logger 误当训练格式

### 5.1 继续做的两块
1. **Teleop 样本级 QA**
   - 长度一致性
   - timestamp 单调性 / 重复率
   - fallback / inactive 比例
   - action / qpos_end / quaternion 数值分布

2. **训练加载一致性**
   - 真实样本走 `load_carm_episode()` / `create_carm_obs_process_fn()` / `CARMDataset`
   - 验证 teleop / inference 数据谁能直接吃、谁吃不了、差在哪里

### 5.2 后续待落实的改动方向（不是本阶段立即改）
- 统一 teleop / inference timestamp 语义
- inference 对齐 teleop 的双视角/primary_camera 机制
- dataset 过滤 inactive teleop 帧
- **方案A 当前执行阻塞**：`rlft_ms3` 环境里 `cv2` 仍缺失，loader probe 不能直接跑；先补依赖，再继续全样本 QA / loader probe

---

## 6. 关键证据文件

代码：
- `carm_ros_deploy/src/carm_deploy/data/record_data_ros.py`
- `carm_ros_deploy/src/carm_deploy/core/env_ros.py`
- `carm_ros_deploy/src/carm_deploy/inference/inference_ros.py`
- `carm_ros_deploy/src/carm_deploy/inference/policy_loader.py`
- `carm_ros_deploy/src/carm_deploy/inference/inference_recorder.py`
- `carm_ros_deploy/src/carm_deploy/utils/image_sync.py`
- `carm_ros_deploy/src/carm_deploy/utils/trajectory_interpolator.py`
- `rlft/datasets/data_utils.py`
- `rlft/datasets/carm_dataset.py`

时间线样本：
- `recorded_data/fixed_dual_light/timeline_record_20260319_235212.jsonl`
- `recorded_data/fixed_no_light/timeline_record_20260318_225939.jsonl`
- `recorded_data/random_no_light/timeline_record_20260320_231709.jsonl`
- `inference_logs/timeline_20260327_195606.jsonl`
