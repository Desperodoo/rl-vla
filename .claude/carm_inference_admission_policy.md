# CARM Inference Admission Policy v1

日期：2026-04-07  
状态：Draft v1  
范围：CARM 真机 inference raw episode -> training staging -> second-stage training 的准入规则

---

## 1. 目标

本策略用于回答一个明确问题：

> 哪些 inference episode 可以进入下一轮训练，哪些只能保留为审计材料，哪些必须拒绝。

它服务于以下目标：

1. 保证回流进训练的数据语义稳定
2. 保证 second-stage 训练结果可解释、可复现
3. 避免把低质量 / 强干预 / 强裁剪 episode 无控制混入训练
4. 为后续 `teleop-only / inference-only / mixed` baseline 提供统一数据准入标准

---

## 2. 适用范围

本策略当前仅适用于：

- `inference_episode_*.hdf5` raw inference recorder 数据
- 由 conversion 产出的 `inference_staging` 数据
- episode-level admission

当前**不包含**：
- step-level curriculum
- online ingestion
- reward-based filtering
- 多视角直接进 policy 输入的策略约束

---

## 3. 设计原则

### 3.1 先 episode-level，后 step-level

当前阶段优先保证 episode 级回流稳定、可审计。

原因：
- 当前主问题是“哪些 episode 值得进训练”
- 不是“每个 step 的 credit assignment 如何最优”
- 过早引入 step-level policy 会显著增加复杂度，且不利于快速建立标准 baseline

### 3.2 先分桶，再训练

当前 admission 不应只有 pass / fail 两种结果，至少要支持：
- `gold`
- `silver`
- `reject`

原因：
- mixed training 需要区分高质量数据与边界数据
- 只做 pass / fail 会让后续 ablation 丢失解释力

### 3.3 准入规则必须显式写入 metadata

policy 结果不能只存在脚本参数或终端输出里。

每个 episode 的 admission 结果必须可追溯到：
- 使用的 policy version
- 具体阈值
- pass / reject 原因
- episode 级原始统计量

---

## 4. 输入数据前提

raw inference episode 至少应具备以下字段或语义：

### 4.1 必备 observations / action 字段

- `observations/images`
- `observations/qpos_joint`
- `observations/qpos_end`
- `observations/timestamps`
- `action`

### 4.2 必备 attrs / schema 信息

- `timestamp_semantics=obs_stamp_ros`
- `primary_camera`
- `camera_names`
- `camera_topics`
- `data_source`
- `action_semantics_version`
- `action_space`

### 4.3 推荐但非 admission hard requirement 的信息

- `has_intervention`
- `intervention_ratio`
- `source_run_info`
- `source_timeline`
- 成功 / 质量标签
- safety clip 摘要

说明：
- 若某些质量字段暂时缺失，不应阻塞整个 raw episode 被审计
- 但缺失关键 schema 字段时，不得进入正式训练桶

---

## 5. Admission 输出标签

每个 episode 最终必须被标到以下三类之一：

### 5.1 gold

适用于：
- 数据契约完整
- episode 长度充足
- intervention 很低或没有
- safety clip 很低或没有
- 若存在成功 / 高质量标签，则满足较高标准

用途：
- 作为 inference-only baseline 的首选训练数据
- 作为 teleop + inference mixed 的高置信度补充数据

### 5.2 silver

适用于：
- 数据仍可训练
- 但存在边界问题，例如：
  - intervention 略高
  - safety clip 略高
  - 质量标签一般
  - episode 长度刚过门槛

用途：
- 可用于 mixed training
- 不建议单独作为最纯 inference-only baseline 的主桶

### 5.3 reject

适用于：
- 关键 schema 不满足
- episode 过短
- intervention / safety clip 明显过高
- 质量明显不满足训练准入要求

用途：
- 不进入训练
- 保留 metadata 供审计与失败分析

---

## 6. Admission v1 默认规则

以下为建议默认规则。当前目标不是“最优阈值”，而是先形成一版可执行、可复现、可比较的正式 policy。

### 6.1 Hard requirements（不满足直接 reject）

1. **schema 完整性**
   - 必须存在：
     - `observations/images`
     - `observations/qpos_joint`
     - `observations/qpos_end`
     - `observations/gripper`
     - `observations/timestamps`
     - `action`
     - `intervention_mask`

2. **时间戳语义一致**
   - `timestamp_semantics` 必须为 `obs_stamp_ros`

3. **主视角契约完整**
   - 必须存在：
     - `primary_camera`
     - `camera_names`
     - `camera_topics`

4. **episode 长度下限**
   - `num_steps >= min_steps`
   - v1 建议默认：`min_steps = 32`

5. **动作语义信息完整**
   - 必须存在：
     - `action_semantics_version`
     - `action_space`

### 6.2 Soft quality rules（决定 gold / silver / reject）

#### intervention ratio
- `intervention_ratio_raw <= 0.02` -> 倾向 `gold`
- `0.02 < intervention_ratio_raw <= 0.10` -> 倾向 `silver`
- `intervention_ratio_raw > 0.10` -> `reject`

#### safety clip ratio
若能从 `run_info` / sidecar 中提取：
- `safety_clip_rate <= 0.01` -> 可维持 `gold`
- `0.01 < safety_clip_rate <= 0.05` -> 至少降为 `silver`
- `safety_clip_rate > 0.05` -> `reject`

#### success / quality
若当前已有成功或质量标签：
- success + high quality -> 提升到 `gold`
- success 但质量一般 -> 可保留为 `silver`
- 明确失败且伴随高 intervention / 高 clip -> 倾向 `reject`

### 6.3 缺失质量信息时的处理

若 success / safety / quality 标签暂时不存在：
- 允许仅基于 schema + step count + intervention ratio 进入 `silver`
- 不建议直接提升到 `gold`

也就是说：
- `gold` 需要更强证据
- `silver` 可以作为“结构合法且风险可控”的默认保守桶

---

## 7. v1 推荐阈值

建议在 policy 文件或 conversion 配置中显式固定以下默认值：

- `policy_version = carm_inference_admission_v1`
- `policy_level = episode`
- `min_steps = 32`
- `gold_max_intervention_ratio = 0.02`
- `silver_max_intervention_ratio = 0.10`
- `gold_max_safety_clip_ratio = 0.01`
- `silver_max_safety_clip_ratio = 0.05`
- `require_timestamp_semantics = obs_stamp_ros`
- `require_primary_camera = true`
- `require_qpos_joint = true`
- `require_action_semantics_version = true`

说明：
- 这些阈值当前首先服务于工程标准化，而不是最终最优训练效果
- 后续如有实验结论，可升级为 v1.1 / v2

---

## 8. Metadata 契约

### 8.1 Episode-level staging attrs

每个 staging episode 建议至少保留：

- `dataset_type=inference_staging`
- `staging_schema_version`
- `admission_policy_version`
- `admission_policy_level=episode`
- `admission_label`
- `admission_pass`
- `admission_reason`
- `kept_steps`
- `dropped_steps`
- `intervention_ratio_raw`
- `intervention_ratio_kept`
- `source_run_info`
- `source_timeline`

### 8.2 Sidecar metadata

每个 `episode_*.hdf5` 对应 sidecar `.meta.json` 应包含：

- source raw file path
- source run_info path
- source timeline path
- raw attrs 摘要
- conversion config snapshot
- admission policy snapshot
- admission result
- reject / downgrade reason
- 过滤前后统计

### 8.3 Directory-level summary

目录级 `conversion_metadata.json` 应至少包含：

- total episodes
- gold count
- silver count
- reject count
- reason distribution
- min / median / max step count
- intervention ratio summary
- safety clip ratio summary（若可用）

---

## 9. 与训练入口的衔接建议

### 9.1 v1 的推荐训练用法

第一轮 second-stage baseline 建议这样使用：

1. **teleop-only**
   - 不使用 inference 数据
   - 作为对照组

2. **inference-only (gold only)**
   - 只用 `gold` 桶
   - 测试高质量 inference 数据单独训练的效果

3. **teleop + inference mixed (gold + optional silver)**
   - 默认先用 `gold`
   - `silver` 只作为扩增 ablation

### 9.2 当前不建议的训练用法

在 admission v1 阶段，不建议：
- 把 `reject` 桶重新混回训练
- 不经分桶直接把所有 inference staging 一股脑加入训练
- 在没有 manifest 的情况下做 mixed training

---

## 10. v1 的非目标

以下内容明确不属于当前 admission policy v1 的职责：

1. 判定某个 step 的 credit 是否高
2. 直接做 reward-based ranking
3. 决定 AWSC / RLPD 的算法细节
4. 决定部署时多视角 policy 输入策略
5. 解决全部 timeline latency 问题

admission policy v1 的职责只有一个：

> 让 inference 数据进入训练之前，先变得可筛、可分桶、可追溯。

---

## 11. 建议的下一步实现顺序

### P0
把当前 conversion 脚本的准入逻辑正式参数化：
- 支持 policy version
- 支持 gold / silver / reject 分桶
- 支持目录级 summary 输出

### P1
让 second-stage 训练入口支持：
- `bucket_allowlist=[gold]`
- `bucket_allowlist=[gold,silver]`
- used / rejected manifest 保存

### P2
在完成第一轮 baseline 后，再根据结果收敛：
- `min_steps`
- intervention ratio 阈值
- safety clip ratio 阈值
- success / quality 是否作为 gold hard requirement

---

## 12. 最简结论

CARM inference admission policy v1 的核心思想是：

```text
先把 inference 数据分成 gold / silver / reject
再决定如何进入 second-stage 训练
而不是先把所有数据混进去再看训练结果
```

这样可以让下一阶段的离线回流实验具备：
- 更强的可解释性
- 更强的可复现性
- 更低的训练分布失控风险
