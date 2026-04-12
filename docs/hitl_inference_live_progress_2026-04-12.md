# HITL Inference Live 进展

**日期**: 2026-04-12  
**状态**: `hitl_mode=live` 第一版 owner/source 语义已完成首轮真机验证  
**定位**: 记录 2026-04-12 关于 live owner uplift、human direct execute、以及 source 切换边界的阶段性结论

---

## 1. 本轮完成内容

本轮围绕 `Human-in-the-loop inference live owner/source` 的第一版目标，完成了以下工作：

1. 新增 live owner/source 执行计划文档
2. 在 `inference_ros` 中接入 `hitl_mode=live` 的基础骨架
3. live episode 开始前显式请求 backend `control_state -> upper_machine`
4. episode 停止与 shutdown 时 best-effort 恢复 `control_state -> lower_machine`
5. 在当前版本中实现 human source 的 direct execute 路径
6. 在 recorder / timeline 中补充 current-version human direct execute provenance
7. 完成两条 live rollout 的真机验证
8. 对 human -> policy / policy -> human 切换边界进行跳变量分析

---

## 2. 当前 live 版本语义

当前版本的 `hitl_mode=live` 已明确采用以下最小语义：

- live 开始前切 owner 到 `upper_machine`
- lower-machine teleop loop 不再真实下发
- 上位机成为唯一 writer
- `shared_source` 仅做 `policy / human` 二选一 source-select
- `teleop active` 本身就是 human source 接管信号
- 当前版本 human execute 先直接复用 teleop processed absolute target 路径
- 不做 `human_chunk` 与 `policy_chunk` blending

这意味着当前 live 版本的重点不是 human scheduler/query 对齐，而是：

> 先把 owner/source 语义和 human direct execute 路径跑通，并确认它在真机上可用且安全。

---

## 3. 本轮真机样本

本轮主要样本为：

- [inference_episode_0001_20260412_154604.hdf5](/home/amax/rl-vla/inference_logs/inference_episode_0001_20260412_154604.hdf5)
- [inference_episode_0002_20260412_154654.hdf5](/home/amax/rl-vla/inference_logs/inference_episode_0002_20260412_154654.hdf5)
- [run_info_20260412_154542.json](/home/amax/rl-vla/inference_logs/run_info_20260412_154542.json)
- [timeline_20260412_154536.jsonl](/home/amax/rl-vla/inference_logs/timeline_20260412_154536.jsonl)

操作者现场反馈：

1. 启动过程中无报错
2. active 按下后手柄接管非常丝滑
3. 松开 active 后能正常回到 policy，但衔接有一点不太流畅
4. 停止录制后，网页端手柄能正常恢复 lower-machine 控制

---

## 4. 关键验证结论

### 4.1 human direct execute 已真实生效

当前版本中新增的 provenance 字段包括：

- `action_human_direct_target`
- `action_live_execute_target`
- `hitl_live_execute_source`

分析结果表明：

- 当 `hitl_live_execute_source=human` 时
- `action_human_direct_target` 与 `action_live_execute_target` 完全一致

对应两条 rollout 的最大误差都是：

- `max_abs_diff(human_direct, live_execute) = 0.0`

这说明 human source 下，实际执行的就是当前 human direct execute target，而不是 policy 仍在偷偷写 SDK。

### 4.2 live source 切换不是抖动，而是大段稳定切换

两条 rollout 的 source run-length 如下。

#### episode 1

- `policy_fallback`: 72 steps
- `human`: 113 steps
- `policy_fallback`: 15 steps

#### episode 2

- `policy_fallback`: 90 steps
- `human`: 110 steps

其中：

- episode 1 的 source changes = 2
- episode 2 的 source changes = 1

这说明 source-select 没有出现高频抖动，human 接管“丝滑”是和数据一致的。

### 4.3 signal freshness 正常

两条 rollout 的 teleop freshness 都处于健康范围：

#### episode 1

- `signal_age_ms` mean 约 `8.51ms`
- `signal_age_ms` max 约 `18.63ms`

#### episode 2

- `signal_age_ms` mean 约 `9.39ms`
- `signal_age_ms` max 约 `17.96ms`

因此当前 live 体验中的差异，主要不是由 teleop signal 延迟异常导致的。

---

## 5. 边界跳变量分析

这部分是本轮最关键的新结论。

### 5.1 episode 1: human -> policy 回切

在 episode 1 中出现了一次 `human -> policy` 回切，对应 index `185`。

该回切处的 live execute target 跳变量为：

- 位置跳变：约 `8.995 mm`
- 姿态跳变：约 `3.755 deg`
- 夹爪跳变：约 `2.0 mm`

进一步分析表明：

- `policy_chunk[0]` 与回切后的 `live_execute_target` 完全一致
- 也就是说，这次回切时系统是直接切回了 policy 当前第一执行点
- 而该点与上一时刻 human direct execute target 之间确实存在可感知跳变

这和操作者的主观反馈高度一致：

> human 控制本身是丝滑的，但松开 active 回到 policy 时会感到一点不顺。

### 5.2 episode 2: 只有 policy -> human 接管

episode 2 没有发生 `human -> policy` 回切，只发生了一次 `policy -> human` 接管，对应 index `90`。

该处跳变量为：

- 位置跳变：约 `2.735 mm`
- 姿态跳变：约 `0.397 deg`
- 夹爪跳变：约 `2.0 mm`

这个量级明显小于 episode 1 的 human -> policy 回切，因此操作者更容易觉得“接管很丝滑”。

---

## 6. 当前判断

基于本轮数据，当前可明确判断：

1. 之前的主要风险确实是双 writer / owner 语义不清
2. 现在 `live owner uplift + human direct execute` 已基本解决这一主风险
3. 当前版本的主要残余问题不是 human 接管时不跟手
4. 当前版本的主要残余问题是：
   - human -> policy 回切时，policy 当前执行点与上一时刻 human target 之间存在边界跳变

因此，当前系统的短板已经从“接管不丝滑”转移为“回切边界不平滑”。

---

## 7. 代码与记录补充

本轮已在 recorder 中新增 current-version live provenance：

- `action_human_direct_target`
- `action_live_execute_target`
- `hitl_live_execute_source`

实现位置：

- [inference_recorder.py](/home/amax/rl-vla/carm_ros_deploy/src/carm_deploy/inference/inference_recorder.py)
- [inference_ros.py](/home/amax/rl-vla/carm_ros_deploy/src/carm_deploy/inference/inference_ros.py)

同时修正了 recorder metadata：

- `hitl_live_execute_enabled`

后续新生成的 live rollout 应正确写为 `True`。

注意：

- 本轮已生成的两个 live rollout 是在修正 metadata 之前录制的
- 因此它们在 HDF5 attrs 中仍显示 `hitl_live_execute_enabled=False`
- 这是 metadata 历史遗留问题，不影响这两条样本对 owner/source 与 live execute 语义的分析

---

## 8. 下一步建议

当前最合理的下一步不是再改 owner 架构，而是针对回切边界做小步优化。

建议优先顺序：

1. 分析 `human -> policy` 回切时 policy 当前执行点与 human 末目标之间的差距分布
2. 设计一个最小边界平滑方案
   - 例如短窗口 warm-start
   - 或 policy 恢复前的单步对齐策略
3. 保持当前 owner/source 语义不变
4. 在此基础上再讨论后续 human scheduler/query 对齐版链路

---

## 9. 相关文档

- [hitl_inference_live_owner_source_plan_2026-04-12.md](/home/amax/rl-vla/docs/hitl_inference_live_owner_source_plan_2026-04-12.md)
- [project_memory_baseline_2026-04-11.md](/home/amax/rl-vla/docs/project_memory_baseline_2026-04-11.md)
- [chunk_action_semantics_and_teleop_alignment.md](/home/amax/rl-vla/docs/chunk_action_semantics_and_teleop_alignment.md)
- [teleop_uplift_progress_2026-04-11.md](/home/amax/rl-vla/docs/teleop_uplift_progress_2026-04-11.md)
