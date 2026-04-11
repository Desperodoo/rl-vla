# CARM Next-Stage Plan（离线 inference 回流与 second-stage 训练）

日期：2026-04-07  
范围：CARM 真机 inference 数据准入、离线回流、second-stage 训练脚手架、reward / offline RL 预留接口

---

## 1. 本文目的

当前 CARM 主线已经从“修数据契约 / 打通最小 smoke 闭环”进入“定义正式回流规则 + 搭建标准化 second-stage pipeline”的阶段。

本文用于明确下一阶段的执行重点，避免继续把工作停留在：
- 仅补字段
- 仅做一次性 smoke
- 仅靠临时脚本拼接闭环

下一阶段的目标是把 inference 数据回流变成：
1. **有准入规则**
2. **有 staging 契约**
3. **有正式训练入口**
4. **有可比较的 baseline 与 redeploy 闭环**

---

## 2. 当前阶段判断

### 2.1 已经完成并稳定的部分

以下能力已基本具备：

1. **teleop / inference / training 核心数据契约已基本稳定**
   - timestamp 语义已对齐到 `obs_stamp_ros`
   - camera 契约已对齐：`primary_camera` / `images_by_camera` / `camera_names`
   - inference recorder 新样本已满足训练侧关键字段要求
   - `load_carm_episode()` 可直接读取新版 inference recorder 样本

2. **离线 inference 回流最小闭环已打通**
   已可执行：
   ```text
   inference_episode_*.hdf5
   -> staging conversion
   -> verifier / metadata audit
   -> train_carm second-training smoke
   ```

3. **分析结论已从“是否可读”推进到“行为差异是什么”**
   当前已明确：
   - teleop vs inference 的最大 gap 不在 observation freshness
   - timing burden 主要来自 inference / chunk staging
   - 更大的 gap 在行为语义：inference action 相比 teleop 更激进

### 2.2 当前真正的 blocker

当前 blocker 已不再是“inference 数据能不能进入训练”，而是：

1. **什么 inference 数据应该进入下一轮训练**
2. **进入训练前应如何做正式准入与分桶**
3. **如何把 second-training 从 smoke 提升为标准化可复现 pipeline**
4. **如何在训练后补一轮 redeploy smoke 验证行为是否改善**

---

## 3. 下一阶段总目标

下一阶段建议围绕两条主线推进：

### 主线 A：冻结 inference 数据准入规则
把当前“最小工程可用”的 admission，收敛成正式 policy。

### 主线 B：搭建离线 second-stage 训练正式入口
把当前 conversion + smoke 的零散链路，升级为与 `rlft/offline/train_carm.py` 同级的训练入口。

在这两条主线稳定后，再推进：

### 主线 C：补 reward / offline RL 预留脚手架
优先统一 reward source / reward attachment 接口，为 robometer 和真机版 AWSC/RLPD 留好扩展位。

---

## 4. 主线 A：inference 数据准入规则

### 4.1 目标

把当前 admission 从“最小可用”变成“正式规范”，确保回流进训练的数据分布可解释、可复现、可比较。

### 4.2 原则

1. **先做 episode-level policy，再考虑 step-level policy**
   - 当前阶段先避免把复杂度推到 step curriculum
   - 优先保证目录级 / episode 级回流可稳定执行

2. **先分桶，再决定训练混合策略**
   - 不只做 pass / fail
   - 让后续 mixed training / ablation 更自然

3. **准入规则必须写入 metadata，而不是只存在脚本参数里**
   - 保证后续训练结果可追溯

### 4.3 建议的 admission v1 字段

建议先固定以下 episode-level 规则：

- `min_steps`
- `max_intervention_ratio`
- `max_safety_clip_ratio`
- `require_timestamp_semantics=obs_stamp_ros`
- `require_qpos_joint=true`
- `require_primary_camera=true`
- `require_schema_version>=...`

可选准入条件：
- `success_only`
- `high_quality_only`
- `no_intervention_only`

### 4.4 建议的分桶方式

建议至少分三档：

1. **gold**
   - 成功
   - 高质量
   - 无 intervention 或极低 intervention
   - 无明显 safety clip

2. **silver**
   - 可训练
   - 有少量 intervention / safety clip / 质量边界问题
   - 可以用于 mixed 或扩增实验，但不应作为最纯 baseline

3. **reject**
   - 不满足准入门槛
   - 不进入训练，仅保留审计信息

### 4.5 这一阶段的交付物

建议新增一份正式文档：
- `.claude/carm_inference_admission_policy.md`

文档建议包含：
- raw inference episode 的必备字段
- admission 指标定义
- 默认阈值
- 分桶规则
- sidecar metadata 契约
- conversion summary 契约

---

## 5. 主线 B：离线 second-stage 训练正式入口

### 5.1 目标

把当前“staging conversion + second-training smoke”提升为正式训练入口，使其成为与 `train_carm.py` 同层级的标准主路径。

### 5.2 设计原则

1. **不要继续把核心逻辑散落在临时脚本里**
2. **训练入口必须同时管理数据、准入、混合策略、产物审计**
3. **必须显式记录本轮训练到底使用了哪些 episode / 哪些 admission bucket**

### 5.3 建议的入口形态

建议新增一个正式入口，候选名：
- `rlft/offline/train_carm_second_stage.py`

更偏向这个命名，因为它清楚表达：
- 不是普通 baseline train
- 而是 teleop 基础上的 inference 回流 / continual finetune / second-stage training

### 5.4 该入口至少需要解决的问题

1. 指定 teleop dataset
2. 指定 inference raw / staging dataset
3. 指定 admission policy
4. 指定混合方式：
   - teleop-only
   - inference-only
   - teleop + inference mixed
   - warm-start + second-stage finetune
5. 输出统一 run dir
6. 保存 used / rejected / bucket summary
7. 为后续 redeploy smoke 输出 deploy-ready checkpoint

### 5.5 建议的配置分层

建议沿用当前 `tyro dataclass` 风格，分成以下配置块：

#### DataConfig
- `teleop_dirs`
- `inference_staging_dirs`
- `admission_policy_path`
- `mixing_mode`
- `mix_ratio`
- `bucket_allowlist`

#### TrainingConfig
- 复用 `train_carm` 主训练超参
- 增加 second-stage 专用项（如 warm-start checkpoint、finetune steps）

#### EvaluationConfig
- 是否训练后自动导出 deploy checkpoint
- 是否自动跑 offline eval
- 是否自动触发 deployment smoke 所需产物导出

#### AuditConfig
- 是否保存 `used_episodes.json`
- 是否保存 `rejected_episodes.json`
- 是否保存 `training_manifest.json`
- 是否保存 `admission_summary.json`

### 5.6 建议的标准产物

second-stage 训练 run 目录建议至少落以下文件：

- `training_manifest.json`
- `used_episodes.json`
- `rejected_episodes.json`
- `admission_summary.json`
- `data_mix_summary.json`
- deploy-ready checkpoint / export metadata

这样后续才能回答：
- 本轮训练用了哪些数据
- 被拒绝的 episode 为什么被拒绝
- mixed 比例是多少
- 行为变化来自数据规则、数据量还是训练配置变化

---

## 6. 主线 C：reward / offline RL 预留接口

### 6.1 目标

为后续真机版 reward model（如 robometer）以及 AWSC / RLPD 风格回流预留统一接口，但当前阶段**先搭脚手架，不急着直接上完整 RL 主线**。

### 6.2 当前判断

现阶段最缺的不是 RL 算法壳，而是：
- inference 数据准入是否稳定
- second-stage 训练是否能稳定提升
- reward 信号是否可追踪、可解释、可挂载

因此更合理的顺序应为：
1. 先跑稳 offline second-stage baseline
2. 再接 robometer reward
3. 最后再评估是否推进真机版 RLPD-AWSC

### 6.3 建议先统一 reward source 抽象

建议 reward source 至少支持：
- `none`
- `robometer`
- `heuristic`
- `human_label`
- `from_run_info`

这样后续：
- BC / imitation-only
- weighted BC
- reward-conditioned filtering
- offline RL

都能复用同一套数据附着与审计逻辑。

### 6.4 建议先做 dataset-level reward attachment

优先明确：
- reward 写在 HDF5 还是 sidecar
- episode reward 与 step reward 如何表达
- reward version / reward source 如何写入 metadata

建议至少保留以下字段：
- `reward_source`
- `reward_version`
- `episode_reward`
- `step_rewards`
- `reward_metadata`

---

## 7. 推荐执行顺序

### 第一优先级：规则先行

#### 任务 1
冻结 inference admission policy v1：
- `min_steps`
- `max_intervention_ratio`
- `max_safety_clip_ratio`
- `success / high-quality / intervention` 分桶规则
- metadata / sidecar 契约

#### 任务 2
把当前 conversion 升级成稳定 staging builder：
- 输入 raw inference dir
- 输出 staging dir
- 输出 admission report
- 输出 episode sidecar / used-rejected 清单

### 第二优先级：形成正式 second-training 入口

#### 任务 3
新增与 `train_carm.py` 同级的 second-stage 入口：
- 支持 teleop-only / inference-only / mixed
- 支持 admission policy
- 支持 manifest / audit 产物
- 支持 deploy checkpoint 导出

#### 任务 4
定义第一轮标准实验矩阵：
- A: teleop-only baseline
- B: inference admitted-only
- C: teleop + inference mixed

### 第三优先级：补部署闭环

#### 任务 5
训练后自动产出 deploy-ready checkpoint

#### 任务 6
补 deployment smoke，验证：
- 真实 receding-horizon 下行为是否变化
- aggressive semantics 是否改善
- safety clip / intervention 是否恶化

### 第四优先级：reward / offline RL 脚手架

#### 任务 7
抽象 reward attachment schema（robometer 优先）

#### 任务 8
预留真机版 AWSC / RLPD 风格入口，但暂不在当前阶段重投入算法实现

---

## 8. 下一阶段建议的最小交付物

建议把下一阶段收敛为 4 个明确 deliverables：

1. **一份 admission policy 文档**
   - 明确 inference 数据如何筛选、如何分桶、如何挂 metadata

2. **一个稳定的 inference -> staging builder**
   - 不只是 smoke 脚本，而是标准化目录级入口

3. **一个 second-stage 正式训练入口**
   - 与 `train_carm.py` 同级
   - 支持 teleop-only / inference-only / mixed

4. **一组标准 baseline + redeploy smoke 实验矩阵**
   - 用于比较二次训练是否真的改善线上行为

---

## 9. 当前阶段不建议优先投入的项

以下内容当前不建议排到 next-stage 的最前面：

1. 继续大规模补 recorder / logger / timeline 全架构重构
2. 让 inference 数据自动混入训练目录扫描
3. 立即把多视角直接接入 policy 输入
4. 立即投入完整真机版 AWSC / RLPD 训练实现

原因：
- 这些项不是当前闭环的最短板
- 当前更需要先把“回流的数据规则 + 正式 second-stage pipeline”做扎实

---

## 10. 最简结论

下一阶段最值得做的，不是继续补字段，而是把 inference 数据离线回流升级成一条正式主线：

```text
inference raw data
-> admission policy
-> staging builder
-> second-stage training entry
-> deploy-ready export
-> redeploy smoke
```

优先级建议为：

1. **先定 inference 数据准入规则**
2. **再建离线 second-stage 正式入口**
3. **再补 reward / offline RL 的扩展接口**

如果这条主线建立起来，后续无论是 robometer reward、真机版 AWSC，还是更复杂的 online/offline 混合回流，都会更容易落到稳定、可复现、可比较的工程路径上。
