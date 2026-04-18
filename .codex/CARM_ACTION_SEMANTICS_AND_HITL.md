# CARM Action Semantics And HITL Guide

日期基线：2026-04-16  
状态：HITL live owner/source 第一阶段已验证；human scheduled execute 与 control-loop provenance 已接入；后续 HITL 继续推进前先暂停收敛

本文件整理 `docs/` 中与 CARM 真机部署最长期、最稳定的工程结论，重点覆盖：
- action 三层语义
- teleop owner 与信号来源事实
- teleop uplift / ownership uplift 的阶段性结论
- HITL live 的 owner/source 约束
- human chunk proposal 的设计边界
- inference 侧 canonical 日志产物

## 1. 当前最核心的项目判断

对当前 CARM 主线而言，最关键的问题已经不是单纯的模型结构，而是：

> 采集、训练、部署、回流、HITL 之间的 action 与控制通道语义是否一致。

如果 action 语义漂移，那么：
- 模仿学习会学到错误目标
- 部署分析会混淆 policy 输出与系统执行
- inference 回流无法正确解释真实状态转移
- HITL / RL 会建立在不稳定的闭环之上

当前最新收敛点也因此变得更明确：
- `record_data_ros` 路线已经完成 teleop owner uplift 的验证
- `inference_ros` 路线已经完成 HITL live owner/source 最小闭环
- 现阶段最需要保留的不是更多试验分支，而是稳定的 action provenance 与边界问题诊断

## 2. Action 三层语义

当前系统里，`action` 不能再被视为单一对象。部署链路已经明确存在三层语义：

### 2.1 `a_policy_chunk`

- policy 一次推理输出的整段 chunk
- 最接近“agent 真正输出的动作”

### 2.2 `a_sched_t`

- 控制线程在某个控制时刻，从 `ActionChunkManager` 中取出的单步动作
- 已包含时间调度、receding horizon 覆盖、推理频率与控制频率不一致带来的采样效应

### 2.3 `a_exec_t`

- 最终真正发给 SDK 的动作
- 若存在安全裁剪、人工接管、限幅、低层门控，它会进一步偏离 `a_sched_t`

## 3. 当前收敛的 action 定义

### 3.1 学习动作仍定义为 chunk-level action

当前主张是：

> 真正的学习动作定义为 chunk-level policy action，也就是“一次推理输出的整个 chunk”。

原因：
- 与当前 policy 输出形式一致
- 不把 chunk policy 人为降格为单步控制 policy
- 使训练语义与 agent 输出语义保持一致

### 3.2 部署通道被视为执行系统的一部分

链路：

```text
a_policy_chunk -> a_sched_t -> a_exec_t
```

这里：
- policy 输出 chunk
- scheduler / control loop / delay 负责在控制时钟上落地
- 真机真正执行的是最终单步动作流

因此：
- `a_exec_t` 不应偷换成“学习动作”
- 但必须完整记录

### 3.3 回流与分析必须记录三层 action

工程上必须同时记录：
- `a_policy_chunk`
- `a_sched_t`
- `a_exec_t`

否则无法：
- 量化 train-deploy gap
- 解释真机状态转移
- 支撑后续 HITL / RL 设计

## 4. Lower-machine teleop 的当前事实

当前真实 teleop owner 在 lower machine：
- backend 位于 `/var/www/backend/carm_backend`
- 真实控制逻辑在 `server/basic/joystick.py`
- `CommandController.loop()` 以 `50Hz` 运行
- 在 lower machine 本地直接调用：
  - `track_pose(...)`
  - `set_gripper(...)`

这意味着：
- 上位机目前不是 teleop 控制 owner
- 上位机最初只是旁路观察者

## 5. Teleop 信号的两层事实

当前对外信号应明确区分：

### 5.1 raw teleop signal

更接近：
- 手柄 pose
- 按钮状态
- trigger / grip / clutch

它适合未来如果真的要把 teleop 控制律搬到上位机时使用。

### 5.2 processed teleop target

当前对应：
- `/api/joystick/teleop_target_v2`

它的语义是：
- 已经过 lower-machine teleop 控制律处理
- 已积分成绝对末端目标
- 可直接送 SDK 的目标位姿

当前阶段最重要的工程结论是：

> 先做“信号上移与 shadow 复现”，再讨论控制权迁移。

## 6. Teleop uplift 的当前工程约定

当前已锁定的迁移约定包括：

- 过渡宿主选 `record_data_ros`，不是 `inference_ros`
- `record_data_ros` 升级为双模式宿主：
  - `passive_shadow`
  - `upper_control`
- 第一阶段不接管真实控制，只把 teleop shadow 能力并入 recorder
- 共享组件拆分为：
  - `TeleopSignalClient`
  - `TeleopShadowTransformer`
  - `TeleopUpperControlBridge`
- 候选阶段先只生成 upper candidate command，不真实下发
- 只有候选验证通过后，才进入真实接管
- 真实接管只在 `recording=True` 期间由 upper machine 持有 teleop 控制权
- backend 必须通过显式 `control_state` 接口切 owner，不能靠隐式约定
- fail-safe 采用“停止更新并保位”，不自动回零

## 7. Teleop uplift 已验证结论

已完成并验证：

1. backend ownership gate 可控
   - `local_control_enabled=false` 时，lower machine 停止真实 SDK 下发
   - 但 `processed/raw` 状态仍持续更新

2. `passive_shadow` 几何闭环成立
   - `processed.target_pose_abs -> human_chunk_rel -> reconstructed_target_abs` 几何重建成立

3. `upper_control` candidate-only 稳定
   - candidate loop 已接近稳定 `50Hz`

4. `upper_control` live execute 已完成真机验证
   - `recording=True` 时 owner 切到 `upper_machine`
   - `recording=False` 时 owner 恢复 `lower_machine`
   - 上位机确实在真实下发控制

5. 异常退出恢复已验证
   - recorder 中断或 `Ctrl+C` 不会把系统卡死在 `upper_machine`

6. stale 主要来自 inactive，而不是 active 链路真的 timeout

7. 速度档位残留问题已查清并修复
   - 之前的“开启 inference 后 teleop 变慢，并且退出后仍然慢”不是双 writer 问题
   - 根因是 `env_ros` / inference 会把机械臂留在低速档位
   - 当前已拆分为：
     - `init_speed`
     - `normal_speed_level`
   - 正常运行与退出恢复后应回到 teleop 对齐的正常速度语义

## 8. HITL 三阶段语义

`hitl_mode` 当前按三阶段演进：

1. `shadow`
2. `candidate`
3. `live`

### 8.1 `shadow`

- 不切 backend owner
- lower machine 仍持有真实 teleop 控制权
- upper machine 只读取 teleop processed signal
- 只生成：
  - `human_chunk_proposal`
  - `shared_chunk`
  - provenance
- 不参与真实执行

### 8.2 `candidate`

- 仍不切 backend owner
- upper machine 只做候选调度与候选执行观测
- 不真实下发

### 8.3 `live`

- episode 开始前显式切 owner 到 `upper_machine`
- lower machine 停止真实 SDK 下发，但继续更新 teleop processed/raw signal
- upper machine 成为唯一 writer
- 上位机内部做 source-select：
  - `policy`
  - `human`

当前实际推进状态：
- `shadow`：已完成并稳定
- `candidate`：已完成并稳定
- `live`：第一版已真机验证，但当前先暂停继续改动，保留待解问题后再进入下一阶段

## 9. Live 阶段 owner 与 source 的正式约定

### 9.1 owner 固定在 upper machine

进入 `live` 后：
- `local_control_enabled=false`
- `control_owner=upper_machine`

此时：
- lower-machine teleop loop 不再真实写 SDK
- 但 teleop signal 继续可读

### 9.2 source 在上位机内部切换

owner 固定，source 只在以下两者中切换：
- `shared_source=policy`
- `shared_source=human`

这不是 blending，而是 source-select。

### 9.3 source 切换规则

第一版最小规则：
- 当 `teleop active=true`、signal valid 且 not stale 时：
  - `shared_source=human`
- 其他情况：
  - `shared_source=policy`

也就是说：

> 在 live 阶段，`teleop active` 本身就是 human 接管信号。

### 9.4 owner 切换与 source 切换不是一回事

必须强调：
- human active 不等于把 owner 切回 lower machine
- human active 只意味着“当前由 upper machine 内部的人类链路成为 active source”

owner 切回 lower machine 只发生在：
- inference 进程退出
- ROS shutdown
- live session 结束
- 上位机异常需要 fail-safe 恢复

## 10. 当前明确不做 blending

当前版本明确排除：
- `human_chunk` 与 `policy_chunk` 的数值加权融合
- per-dim mixing
- learned mixer

原因：
- action 语义不干净
- provenance 不清晰
- 不利于回流分析
- 容易把 source 决策问题与数值融合问题耦合

当前 live 只接受：
- `policy_chunk`
- `human_chunk`
- `shared_chunk = source_select(policy_chunk | human_chunk)`

## 11. Human chunk proposal 的收敛定义

当前 `human_chunk proposal` 不是 teleop 单步 target 的简单复制，而是：

> 基于当前及最近一小段时间内的 teleop intent，在线构造一个严格因果、短时未来、与控制时钟对齐的 `human_chunk proposal`。

设计原则：
- 与 `policy_chunk` 同层语义
- 只使用当前与过去可见信息，严格因果
- `step-0` 必须锚定当前 processed target
- 未来只做短时 rollout，不对整个 `pred_horizon` 激进外推

收敛方案：
- 用当前 processed absolute target 作为 step-0 锚点
- 用历史窗口估计短时 teleop intent velocity
- 只在短时窗口内 rollout
- 远端用 hold

当前版本的工程边界是：
- 先保证 human 接管质量和 owner/source 语义成立
- 后续再补齐：
  - `human_chunk`
  - `human_sched_t`
  - `human_exec_t`

截至 2026-04-16，以上“后续再补齐”已经推进到第一批可运行版本：
- `HumanChunkProposalBuilder` 已具备：
  - processed-target history buffer
  - short-horizon rollout
  - tail hold
- human execute 路径已支持两种模式：
  - `direct`
  - `scheduled`
- `scheduled` 路径已真机验证，主观手感上不明显差于 `direct`

但要明确：
- 当前 `human_chunk / human_sched_t / human_exec_t` 仍处于第一阶段实现
- 还没有进入“可直接作为正式 RL 微调数据定义”的最终收敛版本
- 当前最关键的剩余问题不是 human scheduler 不稳定，而是 `human -> policy` 回切边界

## 12. Teleop 与 inference 的关键语义差异

`joystick.py` 与 `inference_ros.py` 存在重要差别：

### 12.1 位姿变换语义

`inference_ros.py`：
- 使用 SE(3) 右乘
- `target = current @ relative`
- `relative_pose` 定义在末端坐标系

`joystick.py` / backend：
- 位置差值更接近世界坐标系加法
- 姿态用四元数左乘

关键判断：
- 训练标签与推理恢复若都走 `compute_relative_pose_transform` / `apply_relative_transform`，二者是自洽的
- 但若未来直接把 backend 命令语义并入训练标签，需要非常小心左乘 / 右乘差异

### 12.2 夹爪语义

teleop：
- 连续 trigger 映射到连续 gripper 开度

inference：
- 离散 open/close 分类
- 再经过 hysteresis 与映射到连续值

这意味着训练和部署里夹爪其实有“连续采集、离散推理”的语义差异，后续做更细致对齐时要单独考虑。

## 13. inference 侧 canonical 产物

当前 inference 只保留三类 canonical 产物：

1. `inference_episode_*.hdf5`
   - 由 `InferenceRecorder` 生成
   - 用于训练回流和样本级分析

2. `timeline_*.jsonl`
   - 由 `TimelineLogger` 生成
   - 用于分析时延、chunk、control 节奏

3. `run_info_*.json`
   - 由 `InferenceLogger` 生成
   - 保存配置快照、文件映射、运行摘要

职责划分：
- `InferenceRecorder`：episode 数据
- `TimelineLogger`：时间线事件
- `InferenceLogger`：run 元信息

这是当前应该坚持的 canonical logging surface，不再回到多套并存的旧诊断文件体系。

此外，HITL live 现已新增一层必须长期保留的 control-loop truth surface：

4. `control_provenance/` HDF5 group
   - 用于记录 control tick 级别的真实执行链路
   - 目的不是替代 step-level recorder，而是补足“推理步记录”和“控制 tick 真相”之间的语义空隙

当前 `control_provenance/` 至少包含：
- `timestamps`
- `t_send_sys`
- `execute_source`
- `human_execute_mode`
- `live_execute_target`
- `human_direct_target`
- `human_sched_target`
- `human_exec_target`
- `shared_source`

这层记录的意义很大：
- 它让我们能区分“inference step 当时认为会执行什么”与“control loop 实际发了什么”
- 它是后续分析 `policy -> human` / `human -> policy` 边界问题的关键依据
- 它也是将来 RL 微调若要严肃使用 HITL episode 时必须保留的基础 provenance

## 14. 当前 stop point 与待解问题

当前对 HITL inference 的建议不是继续快速扩功能，而是先明确 stop point：

- `live` owner/source 最小闭环已经成立
- `direct` 与 `scheduled` human execute 都已经打通
- control-loop aligned provenance 已补齐
- 当前可以暂停继续修改 HITL 主体逻辑

必须显式保留的待解项只有一个主问题：

- `human -> policy` 回切边界仍存在不完全平滑的问题

当前判断：
- 这更像 source boundary alignment 问题
- 不是 human 接管不丝滑
- 也不是 human scheduler 本身明显不稳定

因此，后续若重启 HITL 主线，优先级应是：
- 先研究回切边界如何平滑和对齐
- 再讨论是否继续增强 human rollout / scheduler 细节
- 不要在主问题未收敛前继续扩展 blending 或更复杂 arbitration

## 15. 当前建议

如果要继续沿着 `.codex` 里的长期约束推进，优先级应是：

1. 在分析与回流里显式保留三层 action provenance
2. 继续保持 owner/source 分离的 live HITL 架构
3. 保留 `control_provenance/` 这层 control-loop truth，不要退回只有 step-level snapshot 的记录方式
4. 把 `human -> policy` boundary continuity 作为明确待解事项长期挂账
5. 在任何新的回流或 RL 设计里，不要再把 lower-machine 本地闭环与 upper-machine 部署闭环混成一个 action 语义
