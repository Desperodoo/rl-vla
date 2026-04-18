# CARM Inference Admission Policy

日期基线：2026-04-16  
状态：v1 仍以 policy-first inference 回流为主；HITL live schema 已出现，但默认仍偏审计优先  
作用范围：CARM 真机 inference raw episode -> training staging -> second-stage 训练准入

## 1. 目标

本策略回答一个非常具体的问题：

哪些 inference episode 可以进入下一轮训练，哪些只能用于审计，哪些必须拒绝。

目标包括：
- 保证回流训练的数据语义稳定
- 保证 second-stage 训练结果可解释、可复现
- 避免把低质量、强干预、强裁剪 episode 无控制混入训练
- 为 `teleop-only / inference-only / mixed` baseline 提供统一准入标准

## 2. 适用范围

当前适用对象：
- `inference_episode_*.hdf5` 原始 inference recorder 数据
- conversion 产出的 `inference_staging` 数据
- episode-level admission

当前不包含：
- step-level curriculum
- online ingestion
- reward-based filtering
- 多视角直接进入 policy 输入时的额外策略约束

当前还要额外强调：
- 已经出现带 `HITL live owner/source` 与 `control_provenance` 的新 inference episode
- 但 admission v1 还没有把这类样本自动视为“可直接进训”的稳定来源
- 对这类样本，当前策略是：
  - schema 可以接纳并保留
  - 分析与审计优先
  - 真正放入训练前，需要单独确认其 action 语义是否满足当轮训练目标

## 3. 设计原则

### 3.1 先 episode-level，后 step-level

当前优先解决“哪些 episode 值得进入训练”，不是“每个 step 的 credit assignment 怎么做最优”。

### 3.2 先分桶，再训练

准入结果不能只有 pass / fail，至少支持：
- `gold`
- `silver`
- `reject`

这样后续 mixed training、ablation 和结果解释才有基础。

### 3.3 准入规则必须写入 metadata

每个 episode 的 admission 结果都要可追溯到：
- policy version
- 阈值
- pass / reject 原因
- episode 级原始统计量

## 4. 输入数据前提

### 4.1 必备 observations / action 字段

- `observations/images`
- `observations/qpos_joint`
- `observations/qpos_end`
- `observations/gripper`
- `observations/timestamps`
- `action`
- `intervention_mask`

### 4.2 必备 attrs / schema 信息

- `timestamp_semantics=obs_stamp_ros`
- `primary_camera`
- `camera_names`
- `camera_topics`
- `data_source`
- `action_semantics_version`
- `action_space`

若是新版 HITL / live inference episode，推荐同时存在：
- `hitl_mode`
- `hitl_live_execute_enabled`
- `hitl_arbitration_mode`
- `hitl_human_execute_mode`

### 4.3 推荐但不是 hard requirement 的信息

- `has_intervention`
- `intervention_ratio`
- `source_run_info`
- `source_timeline`
- 成功 / 质量标签
- safety clip 摘要
- `control_provenance` 相关存在性与长度摘要
- `shared_source` / `execute_source` 分布摘要

缺失关键质量字段时：
- 可以允许 episode 被审计
- 但不能默认进入高置信度训练桶

## 5. Admission 输出标签

### 5.1 gold

适用于：
- 数据契约完整
- episode 长度充足
- intervention 很低或没有
- safety clip 很低或没有
- 如存在成功 / 高质量标签，则满足较高标准

用途：
- inference-only baseline 首选训练桶
- teleop + inference mixed 的高置信度补充数据

### 5.2 silver

适用于：
- 数据仍可训练
- 但存在边界问题，如 intervention 略高、safety clip 略高、质量一般、长度刚过门槛

用途：
- mixed training 可用
- 不建议单独作为最纯 inference-only baseline 的主桶

### 5.3 reject

适用于：
- 关键 schema 不满足
- episode 过短
- intervention / safety clip 明显过高
- 质量明显不满足训练要求

用途：
- 不进入训练
- 保留 metadata 供审计与失败分析

## 6. v1 默认规则

### 6.1 Hard requirements

不满足则直接 `reject`：

1. schema 完整性
   - 必须存在：
     - `observations/images`
     - `observations/qpos_joint`
     - `observations/qpos_end`
     - `observations/gripper`
     - `observations/timestamps`
     - `action`
     - `intervention_mask`

2. 时间戳语义一致
   - `timestamp_semantics` 必须为 `obs_stamp_ros`

3. 主视角契约完整
   - 必须存在：
     - `primary_camera`
     - `camera_names`
     - `camera_topics`

4. episode 长度下限
   - `num_steps >= min_steps`
   - v1 默认建议：`min_steps = 32`

5. 动作语义信息完整
   - 必须存在：
     - `action_semantics_version`
     - `action_space`

### 6.1.1 对 HITL live episode 的补充要求

若 episode 声明：
- `hitl_mode=live`
或
- 存在 `control_provenance/`

则至少应满足：

1. live/source 语义可识别
   - 推荐存在：
     - `hitl_live_execute_enabled`
     - `hitl_human_execute_mode`
     - `hitl_arbitration_mode`

2. provenance 可回放
   - 推荐存在：
     - `action_policy_chunk`
     - `action_shared_chunk`
     - `action_live_execute_target`
   - 若走 direct / scheduled human path，推荐同时保留：
     - `action_human_direct_target`
     - `action_human_chunk`
     - `action_human_sched_t`
     - `action_human_exec_t`

3. control-loop 真值优先
   - 若存在 `control_provenance/`，后续 admission 与分析应优先使用它判断真实执行来源
   - 不应只靠 step-level snapshot 推断 live execute truth

### 6.2 Soft quality rules

#### intervention ratio

- `intervention_ratio_raw <= 0.02`：倾向 `gold`
- `0.02 < intervention_ratio_raw <= 0.10`：倾向 `silver`
- `intervention_ratio_raw > 0.10`：`reject`

#### safety clip ratio

若可以从 `run_info` / sidecar 中提取：
- `safety_clip_rate <= 0.01`：可维持 `gold`
- `0.01 < safety_clip_rate <= 0.05`：至少降到 `silver`
- `safety_clip_rate > 0.05`：`reject`

#### success / quality

若当前已有成功或质量标签：
- success + high quality：可提升到 `gold`
- success 但质量一般：可保留为 `silver`
- 明确失败且伴随高 intervention / 高 clip：倾向 `reject`

### 6.3 缺失质量信息时的保守处理

若 success / safety / quality 标签不存在：
- 允许仅基于 schema + step count + intervention ratio 进入 `silver`
- 不建议直接提升到 `gold`

换言之：
- `gold` 需要更强证据
- `silver` 是“结构合法且风险可控”的保守默认桶

### 6.4 HITL live episode 的当前保守处理

截至当前阶段，默认不建议把 HITL live episode 直接并入常规 `gold` policy-only 回流桶，原因是：

1. 它们已经不再是单纯 policy-only rollout
2. 当前系统仍保留一个明确待解项：
   - `human -> policy` 回切边界连续性问题
3. human scheduled path 虽已跑通，但还处在“部署语义收敛中”的阶段

因此当前更稳妥的 admission 习惯是：
- `policy-only inference episode`
  - 按原 v1 规则分 `gold / silver / reject`
- `HITL live episode`
  - 默认先进入 audit / analysis 桶
  - 只有在某轮训练明确要研究 HITL 数据时，再单独制定 allowlist 与 policy version

## 7. v1 推荐阈值

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

这些阈值的目的首先是工程标准化，不是声称它们已经是最优训练阈值。

## 8. metadata 与 sidecar 契约

每个通过 conversion 生成的 staging episode 应同时保留：

### 8.1 attrs 级关键信息

- `dataset_type = inference_staging`
- `staging_schema_version`
- `kept_steps`
- `dropped_steps`
- `intervention_ratio_raw`
- `intervention_ratio_kept`
- `source_run_info`
- `source_timeline`
- `admission_label`
- `admission_pass`
- `admission_reason`
- `policy_level`
- `policy_version`
- `min_steps`

若 episode 属于 HITL live，还建议补：
- `hitl_mode`
- `hitl_live_execute_enabled`
- `hitl_human_execute_mode`
- `hitl_arbitration_mode`
- `has_control_provenance`
- `execute_source_counts`
- `shared_source_counts`

### 8.2 sidecar `.meta.json`

建议记录：
- source file 指针
- run_info / timeline 指针
- conversion 配置快照
- 原始 attrs 摘要
- admission 结果
- 过滤前后统计

### 8.3 目录级汇总

目录级 `conversion_metadata.json` 至少应包含：
- pass / fail 计数
- gold / silver / reject 计数
- reason 分布
- 使用的 policy version

## 9. 与 second-stage 训练的连接方式

训练入口必须能够显式记录：
- 本轮训练使用哪些 `inference_staging_paths`
- 允许哪些 `admission_buckets`
- 固定哪一个 `policy_version`
- 哪些 episode 被选中，哪些被拒绝

因此训练 run 目录至少应产出：
- `training_manifest.json`
- `used_episodes.jsonl`
- `rejected_episodes.jsonl`
- `admission_summary.json`
- `data_mix_summary.json`

## 10. baseline 建议

第一轮 baseline 只建议包含以下三组：

1. teleop-only
2. inference-only gold
3. teleop + inference gold mixed

当前不建议在第一轮纳入：
- `gold + silver` mixed
- `reject` 参与训练
- 比例采样 ablation
- reward-based ranking
- robometer reward

原因很简单：
- 第一轮目标是建立稳定、容易解释的对照组
- 不是一上来就搜索最优组合

## 11. 当前 admission 记忆点

当前 `.codex` 层面最重要的准入记忆点只有三条：

1. inference admission v1 仍然以 policy-first episode 为主
2. HITL live 新 schema 必须保留，但默认先审计、后训练
3. 一旦未来要把 HITL episode 正式纳入 RL/finetune，必须同时检查：
   - `a_policy_chunk`
   - `a_sched_t`
   - `a_exec_t`
   - `shared_source`
   - `control_provenance`

否则训练会重新掉回“记录的不是系统真正执行语义”的老问题。
