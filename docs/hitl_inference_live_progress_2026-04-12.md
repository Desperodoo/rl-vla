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

## 8. 速度问题排查与修复

在继续推进 `human_chunk / human_sched_t / human_exec_t` 之前，我们额外排查了一个会干扰主观手感判断的问题：

- 开启 `inference_ros` 后，teleop 手感会明显变慢
- 停掉 `inference_ros` 后，teleop 仍然维持慢速

这部分最终确认不是 HITL source-select 逻辑导致的，而是速度档位残留问题。

### 8.1 下位机实测事实

通过下位机状态与 SDK 状态核对，确认到：

- lower-machine teleop 初始化后，`speed_percentage` 可到 `1.0`
- inference 运行期间，`speed_percentage` 会降到约 `0.3`
- inference 退出后，机械臂仍可能停留在约 `0.2`
- lower-machine teleop 本身不会主动把该档位恢复到正常速度

也就是说：

> teleop 变慢的直接原因，不是 inference 和 teleop 同时写控制，而是 inference/`env_ros` 修改了速度相关状态，退出后又把机械臂留在了低速档位。

### 8.2 代码修复方向

本轮把环境速度语义拆成了两层：

- `init_speed`
  仅用于初始化 / 回位等大动作
- `normal_speed_level`
  用于运行阶段与退出恢复后的默认速度

当前默认约定为：

- `init_speed = 2.0`
- `normal_speed_level = 10.0`

这对应的目标是：

- 初始化仍保持保守慢速
- 进入正常推理后恢复到与 teleop 一致的正常速度语义
- inference 退出后也恢复到正常速度，而不是把 lower teleop 留在低速档位

### 8.3 2026-04-12 复测结果

复测样本：

- [inference_episode_0001_20260412_173714.hdf5](/home/amax/rl-vla/inference_logs/inference_episode_0001_20260412_173714.hdf5)
- [run_info_20260412_173632.json](/home/amax/rl-vla/inference_logs/run_info_20260412_173632.json)
- [timeline_20260412_173627.jsonl](/home/amax/rl-vla/inference_logs/timeline_20260412_173627.jsonl)

操作者反馈：

- teleop 手感已恢复正常

这轮数据本身显示：

- `hitl_mode=live`
- HDF5 attrs 中 `hitl_live_execute_enabled=True`
- `run_info.control.init_speed=2.0`
- `run_info.control.normal_speed_level=10.0`

同时这轮 rollout 没有发生 `human` takeover：

- `hitl_live_execute_source` 全程为 `policy_fallback`
- `hitl_human_active=0`
- `hitl_human_valid=0`

因此这轮样本的定位应当是：

- speed 修复后的 live 结构回归验证
- 不是新的 human 接管质量验证

### 8.4 当前结论

到目前为止，可以把这条问题收敛为：

1. “inference 开过之后 teleop 一直变慢”的主因已经明确
2. 根因是 speed state 残留，不是 owner/source 架构本身
3. 当前修复方向与复测主观反馈一致
4. 后续再讨论 human/policy 语义对齐时，可以把“速度档位残留”从 HITL 架构问题中剥离开

## 9. Scheduled Human 链路推进

在 speed 问题收敛后，本轮继续推进了 `human_chunk / human_sched_t / human_exec_t` 的对齐工作。

### 9.1 Human Chunk Online Rollout

新增了独立计划文档：

- [human_chunk_online_rollout_plan_2026-04-12.md](/home/amax/rl-vla/docs/human_chunk_online_rollout_plan_2026-04-12.md)

并实现了第一版 `HumanChunkProposalBuilder` 升级：

- 引入 processed teleop target 的 history buffer
- 用短窗口历史估计 human intent velocity
- 对前 `act_horizon` 做 short-horizon rollout
- 对后续 `pred_horizon - act_horizon` 做 hold

同时新增了一组 diagnostics，用于解释 proposal 是如何构造出来的：

- `hitl_human_history_count`
- `hitl_human_history_span_ms`
- `hitl_human_history_usable`
- `hitl_human_rollout_step_count`
- `hitl_human_rollout_dt_ms`
- `hitl_human_linear_velocity`
- `hitl_human_angular_velocity`
- `hitl_human_gripper_velocity`

### 9.2 引入 scheduled execute path

在 `live` 模式下，为 human source 新增了显式执行开关：

- `--hitl_human_execute_mode direct`
- `--hitl_human_execute_mode scheduled`

其中：

- `direct`
  保留 current-version 路径，继续直接执行 current processed teleop absolute target
- `scheduled`
  让 human source 经过：
  - `human_chunk`
  - `ActionChunkManager`
  - `human_sched_t`
  - `human_exec_t`

这一步的关键结论是：

> `scheduled` 模式已经可以稳定工作，而且从操作者主观手感上看，与 `direct` 基本没有明显差异。

### 9.3 真机验证结论

后续多轮真机验证表明：

1. `policy -> human` 切换时，human takeover 仍然丝滑
2. `human -> policy` 回切时，存在轻微不顺，但通常表现为“停顿感”而不是明显大跳变
3. 当前体感上的主要残余问题，不在 human scheduler 本体，而在 source boundary

进一步地，基于 `scheduled` rollout 的分析可以明确判断：

- `policy -> human` 的边界跳变，不是 `human_chunk -> ActionChunkManager -> human_sched_t` 这条链额外引入的
- `human -> policy` 的不顺，也不是 human scheduler/query 的稳定性问题
- 剩余问题主要来自 policy 恢复时，policy 当前执行点与 human 末目标之间的接续关系

### 9.4 Control-loop 对齐 provenance

为了把 `human_sched_t / human_exec_t` 从“最新状态快照”升级成真正可分析的数据，本轮又继续补了严格按 control loop 对齐的记录。

HDF5 新增独立组：

- `control_provenance/`

其中包含：

- `timestamps`
- `t_send_sys`
- `execute_source`
- `human_execute_mode`
- `live_execute_target`
- `human_direct_target`
- `human_sched_target`
- `human_exec_target`
- `shared_source`

新增这层以后，可以明确看到：

- 在 `human_scheduled` 控制段里：
  - `human_sched_t == human_exec_t`
  - `live_execute_target == human_exec_t`
- 因此当前 `scheduled` 路径本身没有额外 execute 偏移

这一步非常重要，因为它把剩余问题进一步收敛到了：

- source 切换边界
- 特别是 `human -> policy` 恢复点的接续逻辑

### 9.5 本轮停点

到当前为止，`scheduled` human 链路已经完成了：

- online human chunk proposal
- scheduled execute path
- control-loop aligned provenance

因此本轮决定：

- 暂时不再继续修 HITL 主链
- 先把当前进展冻结
- 将剩余的 `human -> policy` 边界跳变问题保留为后续待解决事项

当前明确列为待办的问题是：

1. `human -> policy` 回切时，policy 恢复点与 human 末目标之间仍存在接续不自然
2. 该问题更像 source boundary 对齐问题，而不是 human scheduler 稳定性问题
3. 后续若继续推进，优先方向应是：
   - policy 恢复前的 warm-start / re-anchor
   - 边界对齐策略
   - 而不是继续修改 human scheduler 本体

## 10. 当前状态

截至当前，HITL inference live 的结论可以压缩为：

1. owner uplift 与 lower-owner 恢复语义已验证
2. speed 残留问题已排查并修复
3. `direct` 与 `scheduled` 两条 human execute path 都已验证
4. `scheduled` 已具备语义上的价值，且没有明显损伤手感
5. 当前未解决问题主要只剩：
   - `human -> policy` 边界接续

## 11. 下一步建议

当前最合理的下一步不是再改 owner 架构，而是针对回切边界做小步优化。

建议优先顺序：

1. 分析 `human -> policy` 回切时 policy 当前执行点与 human 末目标之间的差距分布
2. 设计一个最小边界平滑方案
   - 例如短窗口 warm-start
   - 或 policy 恢复前的单步对齐策略
3. 保持当前 owner/source 语义不变
4. 在此基础上再讨论后续 human scheduler/query 对齐版链路

---

## 12. 相关文档

- [hitl_inference_live_owner_source_plan_2026-04-12.md](/home/amax/rl-vla/docs/hitl_inference_live_owner_source_plan_2026-04-12.md)
- [human_chunk_online_rollout_plan_2026-04-12.md](/home/amax/rl-vla/docs/human_chunk_online_rollout_plan_2026-04-12.md)
- [project_memory_baseline_2026-04-11.md](/home/amax/rl-vla/docs/project_memory_baseline_2026-04-11.md)
- [chunk_action_semantics_and_teleop_alignment.md](/home/amax/rl-vla/docs/chunk_action_semantics_and_teleop_alignment.md)
- [teleop_uplift_progress_2026-04-11.md](/home/amax/rl-vla/docs/teleop_uplift_progress_2026-04-11.md)
