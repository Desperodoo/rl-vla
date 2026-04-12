# Human Chunk Online Rollout 计划

**日期**: 2026-04-12  
**状态**: 收敛稿，作为 `HumanChunkProposalBuilder` 下一阶段实现基线  
**定位**: 明确如何由当前 teleop intent 在线构造 `human_chunk proposal`，并为后续 `human_sched_t / human_exec_t` 对齐铺路

---

## 1. Summary

当前版本的 live HITL 已验证：

- owner uplift 到 `upper_machine` 可工作
- `teleop active -> human source` 的切换可工作
- current-version 的 `human direct execute` 路径是丝滑且可用的

下一阶段的目标不再是继续优化 direct execute，而是补齐：

- `human_chunk`
- `human_sched_t`
- `human_exec_t`

让 human 与 policy 在部署语义上真正对齐。

这一步的核心不是“把 teleop 单步 target 包装成 chunk 形状”，而是：

> 在当前时刻 `t_k`，基于人当前以及最近一小段时间内的 teleop intent，在线构造一个严格因果、短时未来、与控制时钟对齐的 `human_chunk proposal`。

---

## 2. 设计目标

`human_chunk proposal` 的构造必须同时满足以下约束：

1. 与 `policy_chunk` 同层语义  
   表达的是“当前时刻对未来短时执行目标的提案”，而不是单步 override。

2. 严格因果  
   只能使用当前与过去已知的 teleop signal，不允许用未来 signal 回填。

3. 与部署时钟对齐  
   chunk 内每个 step 对应未来某个具体控制时刻，而不是纯数组索引。

4. step-0 与当前 human direct target 对齐  
   以避免一切入 `human_chunk -> scheduler` 就先产生人为跳变。

5. 远端保守  
   人类 teleop 的未来意图只在短时范围内可信，不能对整个 `pred_horizon` 做激进外推。

---

## 3. Teleop Intent 的定义

当前收敛后的 human intent 定义，不再使用 raw joystick，也不使用 lower machine 的 SDK 单步命令，而是使用当前已经过真机验证的 processed teleop v2 signal：

- `processed.target_pose_abs`
- `gripper_pose`
- `active`
- `sequence`
- `signal_age_ms`

因此 human 当前时刻的意图锚点定义为：

- `intent_now = processed_target_abs_now`

这保证：

- current-version live direct execute
- 下一阶段的 `human_chunk`

都建立在同一个语义锚点上。

---

## 4. 收敛方案

### 4.1 核心方案

`HumanChunkProposalBuilder` 采用：

- history buffer
- short-horizon rollout
- long-tail hold

的组合方案。

具体为：

1. 维护最近一小段 processed target 历史窗口
2. 用历史窗口估计短时 teleop intent velocity
3. 用该 velocity 对未来前 `act_horizon` 步做 causal rollout
4. 对 `pred_horizon` 余下部分直接 hold

### 4.2 为什么不采用纯 hold

纯 hold 的优点是简单、严格因果，但会让 human chunk 对连续推动动作显得过于僵硬，不能合理表达“人正在持续推动”的意图。

因此纯 hold 只作为 fallback，而不是正式主方案。

### 4.3 为什么不对整个 `pred_horizon` 长距离外推

如果对整个 horizon 都做速度外推，会带来两个问题：

1. 人一松手，proposal 还会继续“飞”
2. 姿态和夹爪在远端的外推会迅速失真

因此当前收敛方案明确采用：

- 近端 rollout
- 远端 hold

的二段式定义。

---

## 5. Human Chunk 的正式定义

### 5.1 输入

`HumanChunkProposalBuilder` 输入为：

- 当前观测时刻的 `qpos_end`
- 当前 teleop snapshot
- 最近一段 teleop processed target 历史
- `control_freq`
- `pred_horizon`
- `act_horizon`

### 5.2 step-0 锚定

强制约定：

- `human_chunk[0] = processed_target_abs_now`

这是必须约束。

它保证：

- `human direct execute`
- `human_chunk -> scheduler`

在第一个执行点上语义一致。

### 5.3 历史窗口

第一版采用短窗口历史，例如：

- `80ms - 150ms`

只使用最近、连续、可用的 processed target 估计意图速度。

### 5.4 rollout 规则

第一版 rollout 采用最小可解释模型：

- 位置：constant linear velocity
- 姿态：constant angular velocity
- 夹爪：constant gripper velocity

rollout 只作用于前 `act_horizon` 步。

### 5.5 远端 hold

对 `i >= act_horizon` 的 step：

- 直接 hold `human_chunk[act_horizon - 1]`

### 5.6 历史不足时的退化

若出现以下任一情况：

- history sample 不足
- history span 太短
- teleop stale
- teleop inactive
- teleop invalid

则 builder 不做速度外推，退化为：

- hold chunk
- 或 unavailable proposal

具体取决于 teleop 当前是否仍有可用 processed target。

---

## 6. 与 `human_sched_t / human_exec_t` 的关系

当前计划中的 `human_chunk proposal` 只是第一步。

后续完整对齐路径为：

1. `human_chunk proposal`
2. `human_chunk -> ActionChunkManager`
3. 控制线程 query 得到 `human_sched_t`
4. 执行链输出 `human_exec_t`

因此，这里定义的 chunk proposal 必须天然适合进入现有 `ActionChunkManager`，而不是仅仅服务于 recorder 展示。

---

## 7. 第一版诊断字段

为了让后续真机验证能解释 proposal 是怎么构造出来的，第一版同时增加 diagnostics：

- `history_count`
- `history_span_ms`
- `history_usable`
- `rollout_step_count`
- `rollout_dt_ms`
- `linear_velocity`
- `angular_velocity`
- `gripper_velocity`

这些字段会进入：

- recorder HDF5
- timeline `hitl_human_chunk`

这样后续如果 human 手感异常、边界跳变异常、或 rollout 过冲，我们可以直接判断是：

- 没历史
- history 太短
- velocity 估计异常
- rollout 过长
- 还是 scheduler/query 问题

---

## 8. 本阶段实现边界

本阶段只做：

1. 新版 `HumanChunkProposalBuilder`
2. recorder / timeline diagnostics 落盘
3. 为 live human source 接通显式 execute-path 开关：
   - `direct`
   - `scheduled`
4. 默认仍保留 `direct` 作为 fallback，`scheduled` 先作为可切换验证路径

本阶段暂不做：

1. 在没有验证前直接废弃 current-version 的 `direct` 路径
2. human/policy 边界平滑优化
3. human 与 policy 的 chunk blending
4. learned human future predictor

---

## 9. 当前结论

当前收敛后的正式约定是：

1. `human_chunk proposal` 不等于 teleop 单步 target 的简单复制
2. 它必须是“当前 teleop intent 的短时未来 causal rollout”
3. `step-0` 必须锚定当前 processed target
4. 前 `act_horizon` 做 rollout，后续 `pred_horizon - act_horizon` 做 hold
5. history 不足时退化为 hold / unavailable

这条定义既保留了 current-version direct execute 的已验证语义，又为后续 `human_sched_t / human_exec_t` 对齐提供了干净起点。
