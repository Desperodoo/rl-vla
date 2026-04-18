# CARM Pipeline Guide

日期基线：2026-04-16  
范围：CARM 真机 teleop / inference / HITL / train 主链、timeline 分析、离线回流、second-stage 训练与 baseline

## 1. 当前状态

### 1.1 已闭环完成的主线修复

以下能力已经完成并经过样本或 smoke 验证：

1. teleop / inference 时间戳语义对齐
   - `observations/timestamps` 统一到 `obs_stamp_ros`
   - inference recorder 已写入 `timestamp_semantics=obs_stamp_ros`

2. teleop / inference camera 契约对齐
   - `primary_camera`
   - `images_by_camera`
   - `camera_names`
   - `camera_topics`
   - `observations/images` 继续保留为 primary camera 兼容字段

3. teleop backend 假超时问题修复
   - 根因不是 backend 变慢，而是代理污染
   - 代码层已在 `env_ros.py` 中显式禁用代理
   - shell 层已补 `no_proxy / NO_PROXY`

4. 主训练链默认启用 inactive teleop 过滤
   - `train_carm` 主路径默认过滤 `teleop_scale == 0` 样本
   - `CARMDataset` 会输出过滤统计

5. inference recorder -> 训练读取契约守卫补齐
   - inference recorder 新样本已包含 `observations/qpos_joint`
   - `load_carm_episode()` 已能直接读取新版 inference 样本
   - 缺字段的旧样本会更早报错

6. inference 离线回流最小闭环已打通

```text
inference_episode_*.hdf5
-> staging conversion
-> load_carm_episode / CARMDataset
-> train_carm second-training smoke
```

### 1.2 当前已经可用的工具入口

1. timeline baseline 包装入口
   - `scripts/run_carm_timeline_baseline.py`

2. teleop vs inference gap 分析脚本
   - `scripts/analyze_carm_gap.py`

3. inference 离线 staging 转换脚本
   - `scripts/convert_inference_to_training_staging.py`

4. 中文分析报告
   - `docs/carm_timeline_analysis_report.md`

### 1.3 2026-04-16 当前停靠点

CARM 主线最近新增并完成第一阶段验证的内容：

1. `record_data_ros` teleop owner uplift 已闭环
   - `passive_shadow`
   - `upper_control`
   - backend `control_state` owner gate

2. `inference_ros` HITL live owner/source 最小闭环已闭环
   - live 时 `upper_machine` 成为 sole writer
   - `teleop active` 作为 `human` source-select 信号
   - 不做 blending，只做 `policy / human` source-select

3. human execute 已推进到两条路径
   - `direct`
   - `scheduled`

4. logging 已补到 control-loop truth 级别
   - HDF5 内新增 `control_provenance/`
   - 用于区分 inference-step 记录与 control tick 真实执行

5. 当前 HITL 主线先暂停继续扩展
   - 现存唯一需要长期挂账的问题是：
     - `human -> policy` 回切边界不完全平滑
   - 下一次重启 HITL 主线时，应先处理边界连续性，而不是继续扩展更复杂仲裁

## 2. 关键审计结论

### 2.1 teleop 数据契约

主采集链：
- `carm_ros_deploy/src/carm_deploy/data/record_data_ros.py`
- `carm_ros_deploy/src/carm_deploy/core/env_ros.py`

teleop episode 关键信息：
- `observations/images`
- `observations/images_by_camera/`
- `observations/qpos_joint`
- `observations/qpos_end`
- `observations/gripper`
- `observations/timestamps`
- `action`
- `teleop_scale`

teleop action 语义：
- active 时：`target_pose(7) + gripper(1)`
- inactive 时：当前 `qpos_end`

结论：
- 采集层保留 fallback 行为是合理的
- 训练层必须过滤 inactive 样本

### 2.2 inference 数据契约

当前新版 inference recorder 样本已验证满足：
- `qpos_joint` 存在
- `images_by_camera` / `primary_camera` / `timestamp_semantics` 存在
- 可被训练侧 loader 直接读取

历史结论：
- 旧版 `15D action` 语义对应旧数据，后续统计不应混入
- `inference_logs/inference_*.hdf5` 历史上存在 recorder / logger 双链路并存的问题，后续应继续朝统一格式收敛

### 2.3 timeline 与行为差异结论

当前最重要的分析结论：

1. 最大 gap 不在 observation freshness
   - teleop `delta_obs ~= 0.0324s`
   - inference `delta_obs ~= 0.0370s`

2. inference timing gap 主要来自模型推理与 chunk staging
   - `inference_time ~= 0.0657s`
   - `delta_chunk_obs ~= 0.1066s`
   - `control_send_minus_query ~= 0.00015s`

3. 更大的差异在行为语义，而不仅是 timing
   - teleop `position-gap mean ~= 0.0142`
   - inference `position-gap mean ~= 0.3930`
   - inference 明显比 teleop 更激进

工程含义：
- 单纯再抠几毫秒 latency，不能自动消除真实表现差异
- 当前更应该优先处理 action-amplitude semantics mismatch

### 2.4 最近补充确认

以下结论已经在后续推进中得到验证：

1. `gripper_max` 假阳性问题已收敛
   - 安全配置与真实硬件上限对齐后，最新 4 组 inference 对照实验不再出现 `gripper_clip` 假阳性
   - `safety_clips=0`

2. gap 分析已从 same-step gap 扩展到 motion-centric 指标
   - `realized motion / velocity`
   - `target motion / velocity`
   - `realized-vs-target ratio`
   - `future-k improvement`
   - `early / middle / late window summary`

3. inference 回流已推进到“带准入信息的可训练 staging”
   - 保留 source attrs
   - 新增 staging attrs
   - 每个 episode 生成 `.meta.json` sidecar
   - 目录级有 `conversion_metadata.json`

4. verifier 已能区分 `teleop_v2` 与 `inference_staging`

5. 已成功跑通更真实的 second-training smoke
   - 基于 admission 后的 staging
   - `train_carm` 10 iter smoke 成功

## 3. 当前 blocker

当前 blocker 已不再是“能不能把 inference 数据读进训练”。

真正 blocker 是：
- 什么 inference 数据该进训练
- 进入训练前如何正式分桶
- 如何把 second-stage 从 smoke 升级为标准化 pipeline
- 训练后如何完成 redeploy smoke

## 4. 下一阶段计划

### 4.1 主线 A：冻结 inference admission 规则

目标：
- 让 inference 数据回流从“最小工程可用”升级为“正式规范”

执行要点：
- episode-level policy 先于 step-level
- 先分桶，再讨论混合策略
- 所有准入规则写入 metadata

建议固定字段：
- `min_steps`
- `max_intervention_ratio`
- `max_safety_clip_ratio`
- `require_timestamp_semantics=obs_stamp_ros`
- `require_qpos_joint=true`
- `require_primary_camera=true`
- `require_schema_version>=...`

分桶：
- `gold`
- `silver`
- `reject`

### 4.2 主线 B：搭建正式 second-stage 训练入口

目标：
- 把零散的 conversion + smoke 链路升级为标准主路径

推荐入口命名：
- `rlft/offline/train_carm_second_stage.py`
或当前已有演进命名：
- `rlft/offline/train_carm_stage2.py`

入口至少需要解决：
- 指定 teleop dataset
- 指定 inference raw / staging dataset
- 指定 admission policy
- 指定混合方式
- 输出统一 run 目录
- 保存 used / rejected / bucket summary
- 导出 deploy-ready checkpoint

建议配置分层：

#### DataConfig
- `teleop_dirs`
- `inference_staging_dirs`
- `admission_policy_path`
- `mixing_mode`
- `mix_ratio`
- `bucket_allowlist`

#### TrainingConfig
- 复用 `train_carm` 主训练超参
- 增加 warm-start checkpoint / finetune steps

#### EvaluationConfig
- 是否自动导出 deploy checkpoint
- 是否自动跑 offline eval
- 是否自动准备 redeploy smoke 所需产物

#### AuditConfig
- `used_episodes.json`
- `rejected_episodes.json`
- `training_manifest.json`
- `admission_summary.json`

### 4.3 主线 C：预留 reward / offline RL 扩展位

在 admission 与 second-stage 训练入口稳定后，再补：
- reward source 统一接口
- reward attachment 统一接口
- 为 robometer 和真机 AWSC / RLPD 留出扩展点

## 5. 第一轮 baseline 矩阵

当前固定输入：
- inference staging：`/tmp/carm_staging_admission_v1`
- admission 汇总：`gold=4, silver=3, reject=1`
- policy version：`carm_inference_admission_v1`
- teleop 总 episode 数：127

可用 teleop 数据集：
- `recorded_data/fixed_dual_light`
- `recorded_data/fixed_left_light`
- `recorded_data/fixed_no_light`
- `recorded_data/random_no_light`

### 5.1 Baseline A：teleop-only

目的：
- 验证 second-stage 入口在不接 inference 数据时仍保持稳定

期望：
- `source_counts.teleop > 0`
- `source_counts.inference == 0`

### 5.2 Baseline B：inference-only gold

目的：
- 建立最保守的 inference-only 回流基线

输入：
- `inference_staging_paths=/tmp/carm_staging_admission_v1`
- `admission_buckets=gold`
- `policy_version=carm_inference_admission_v1`

期望：
- `source_counts.teleop == 0`
- `source_counts.inference == 4`
- `bucket_counts.gold == 4`

### 5.3 Baseline C：teleop + inference gold mixed

目的：
- 建立最小混合回流基线

期望：
- `source_counts.teleop == 127`
- `source_counts.inference == 4`
- `bucket_counts.teleop == 127`
- `bucket_counts.gold == 4`
- `total_selected_episodes == 131`

### 5.4 当前暂不纳入第一轮的组合

- `gold + silver` mixed
- `reject` 桶参与训练
- 不同比例采样
- reward-based ranking
- robometer reward

原因：
- 第一轮 baseline 目标是稳定和可解释
- 不急着追求组合最优

## 6. 产物要求

每次 dry-run / smoke / 正式训练后，至少检查：
- `training_manifest.json`
- `used_episodes.jsonl`
- `rejected_episodes.jsonl`
- `admission_summary.json`
- `data_mix_summary.json`

重点核对：
- `source_counts`
- `bucket_counts`
- `policy_version`
- 被选中 episode 列表与目录扫描输入是否一致

## 7. 与系统架构的关系

系统总览可结合仓库主文档理解：
- `README.md`
- `docs/carm_real_robot_system.md`

这里额外强调几个对 second-stage 训练直接重要的系统事实：

1. `record_data_ros.py` 采集的是被动观测，不干扰手柄控制
2. inference 节点是推理线程 + 控制线程双线程架构
3. 安全控制器有四层守卫：
   - 关节限位
   - 动作增量限制
   - 工作空间边界
   - 低通滤波

因此：
- 训练侧不应简单把“被 clip 过的行为”当成理想行为
- admission 与 sidecar 审计必须保留这些运行时信息
