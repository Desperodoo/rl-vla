# Human-in-the-loop Inference Live Owner/Source 执行计划

**日期**: 2026-04-12  
**状态**: 方案收敛稿，作为 `inference_ros` 下一阶段实现基线  
**定位**: 明确 HITL inference 在 `shadow / candidate / live` 三阶段中的 owner/source 关系，以及 live 阶段的人类执行链路  

---

## 1. Summary

基于当前项目记忆基线、`chunk_action_semantics_and_teleop_alignment` 的三层 action 语义、以及 `record_data_ros` teleop uplift 已完成的 ownership 验证，Human-in-the-loop inference 下一阶段收敛为以下原则：

1. `shadow` 与 `candidate` 继续不切 backend owner，只做记录与候选验证
2. `live` 阶段必须显式 owner uplift：
   - `local_control_enabled=false`
   - `control_owner=upper_machine`
3. `live` 阶段由上位机成为唯一 writer，lower-machine teleop loop 不再真实向 SDK 下发
4. `live` 阶段不做 `human_chunk` 与 `policy_chunk` 的 blending
5. `live` 阶段只做 chunk-level source-select：
   - `shared_source=policy`
   - `shared_source=human`
6. `teleop active` 本身就是 human source 进入接管的信号，不再单独设计第二套 override 机制
7. 当前版本的 `human` live execute 先直接复用 `record_data_ros` ownership uplift 中已验证过的 upper teleop target 控制路径
8. 后续版本再补齐：
   - `human_chunk -> human_sched_t -> human_exec_t`
   使 human 与 policy 在部署语义上进一步对齐

这意味着：

> 后续 HITL inference live 不是“人在下位机直接抢控制，上位机旁观记录”，而是“owner 已在 upper machine，人类 signal 在上位机内部被接入为当前 active source；当前版本先走已验证过的 upper teleop target 执行路径，后续再补齐 human 的调度/执行对齐语义”。

---

## 2. 背景与问题重述

当前 `inference_ros` 第一版已经完成：

- `hitl_mode=disabled`
- `hitl_mode=shadow`
- `hitl_mode=candidate`

并已支持：

- `policy_chunk`
- `human_chunk_proposal`
- `shared_chunk`
- HITL provenance 落盘

但首轮真机联调暴露出一个关键事实：

- 在 `shadow` 验证期间，上位机 inference live path 仍在持续向 SDK 下发 policy 控制
- 下位机 teleop 在网页端初始化后，本地 `50Hz` control loop 也具备真实下发能力
- 当操作者试图用手柄“救场”时，可能出现上位机 policy live path 与 lower-machine teleop loop 同时写机械臂的风险

因此后续只要进入真正的 live HITL 阶段，就不能继续把 owner 问题后置为实现细节。

---

## 3. 当前收敛结论

### 3.1 三阶段关系

`hitl_mode` 继续保留三阶段演进：

1. `shadow`
2. `candidate`
3. `live`

但三阶段的 owner/source 语义明确区分如下。

### 3.2 `shadow`

- 不切 backend owner
- lower machine 仍持有 teleop 真实控制权
- 上位机只读取 teleop processed v2 signal
- 上位机只生成：
  - `human_chunk_proposal`
  - `shared_chunk`
  - provenance
- `shared_chunk` 不参与 live execute

### 3.3 `candidate`

- 仍不切 backend owner
- 上位机继续只做候选调度与候选执行观测
- lower machine 仍持有真实 teleop owner
- candidate 当前只要求先补 human 侧候选执行观测
- 与 policy 完全对齐的 `human_sched_t / human_exec_t` 作为下一阶段增强项
- 不真实下发

### 3.4 `live`

- 开始 live episode 前，显式切 owner 到 `upper_machine`
- lower machine 停止真实 SDK 下发，但继续更新 teleop processed/raw signal
- 上位机成为唯一 writer
- 上位机内部根据当前 source-select 决定由：
  - `policy` 驱动执行
  - 或 `human` 驱动执行

---

## 4. Live 阶段的 owner 与 source 关系

## 4.1 owner 语义

`live` 阶段的 owner 固定为：

- `local_control_enabled=false`
- `control_owner=upper_machine`

此时：

- lower-machine teleop loop 不再真实调用 SDK
- 但 teleop processed signal 仍持续可读
- 上位机负责最终 `track_pose / set_gripper` 等实际下发

## 4.2 source 语义

owner 在 `live` 期间固定于 upper machine，但 source 在上位机内部二选一切换：

- `shared_source=policy`
- `shared_source=human`

这是 source-select，不是 blending。

## 4.3 source 切换条件

第一版固定为最小规则：

- 当 `teleop active=true`、signal valid、且 not stale 时：
  - `shared_source=human`
- 其他情况：
  - `shared_source=policy`

因此：

> 在 `live` 阶段，`teleop active` 本身就是人工接管信号。

不再额外设计第二套 override UI、第二套 mode switch、或独立的人工接管状态机。

## 4.4 owner 切换与 source 切换不是同一件事

这部分必须单独强调。

在 `live` 阶段：

- owner 固定在 `upper_machine`
- source 在 `policy/human` 之间切换

也就是说：

- human active 不等于把 owner 切回 lower machine
- human active 只表示“当前由 upper machine 内部的 human 链路成为 active source”

只有在以下场景才切 owner 回 lower machine：

- inference 进程退出
- ROS shutdown
- recorder/live session 明确结束
- 上位机异常，需要 fail-safe 恢复

---

## 5. 不做 blending 的正式约定

当前版本明确排除以下方向：

- `human_chunk` 与 `policy_chunk` 的数值加权融合
- per-dim mixing
- learned mixer
- “既有人类又有 policy 同时贡献一个 chunk” 的共享数值方案

原因如下：

1. 从 action 语义上不干净
2. provenance 不清晰
3. 不利于回流分析
4. 容易把“source 决策问题”和“数值融合算法问题”耦在一起

当前 live 版本只接受：

- `policy_chunk`
- `human_chunk`
- `shared_chunk = source_select(policy_chunk | human_chunk)`

---

## 6. Human 模式的当前版本与后续版本

这是本轮新增的关键约定。

当前 policy 路径已经有一套较成熟的：

- `policy_chunk`
- `policy_sched_t`
- `policy_exec_t`

对应地，human 最终不能永久停在 `human_chunk_proposal`，否则 live 阶段会长期存在“policy 有 chunking/scheduling/executing，而 human 只是直接 teleop target”的语义不对称。

但当前已经明确的实现顺序是：

### 6.1 当前版本

- 先依赖 `record_data_ros` ownership uplift 已验证过的 upper teleop target 路径
- 在 `live` 下当 `shared_source=human` 时，直接使用当前 processed teleop absolute target 作为真实执行目标
- 先保证 human 接管质量和安全 owner 语义成立

### 6.2 后续版本

- 再补齐：
  - `human_chunk`
  - `human_sched_t`
  - `human_exec_t`
- 让 human 与 policy 在部署语义上完全对齐

---

## 7. Human 执行链路设计

## 7.1 输入来源

human 路径继续使用当前已验证可用的 teleop v2 processed signal：

- `processed.target_pose_abs`
- `gripper_pose`
- `active`
- `sequence`
- `signal_age_ms`

不重新引入 raw teleop 控制律迁移。

## 7.2 当前版本的 human execute target

当前版本中，`shared_source=human` 时的真实执行目标先定义为：

- 当前 teleop processed absolute target
- 即 `processed.target_pose_abs + gripper_pose`

它直接参考 `record_data_ros` ownership uplift 后的上位机 teleop 控制路径。

这样做的原因不是认为它已经是最终语义，而是：

1. 这条路径已经在真机上验证过
2. 当前最优先的是 owner/source 安全语义收口
3. 当前版本更关心“human 接管时是否稳定、安全、可用”

## 7.3 `human_chunk`

`human_chunk` 的目标不是表达 lower-machine 当前那一瞬间的单步命令，而是要提升成与 `policy_chunk` 同层的 chunk-level action。

第一版约定：

- 以当前 processed absolute target 为 human 当前意图
- 结合当前观测参考位姿
- 构造与 policy 相同 horizon、相同 shape、相同 absolute EE target 语义的 `human_chunk`

`human_chunk` 在当前版本仍然继续生成并记录，用于 provenance、分析和后续演进。

但当前版本不要求它直接参与 live execute 调度。

## 7.4 `human_sched_t`

human chunk 进入上位机的调度路径后，也必须经过与 policy 对齐的 scheduler/query 过程。

这部分是后续增强项，不是当前 live 第一版的硬门槛。

后续目标不是重新发明新的 scheduler，而是：

- 尽量复用现有 `ActionChunkManager`
- 或复用与 policy 同构的 query 语义

这样 human 的调度语义才与 policy 对齐。

## 7.5 `human_exec_t`

在后续对齐版中，真实下发给 SDK 的动作应来自 human 路径的 executed action，而不是旁路的 teleop direct target。

也就是说：

- human 先变成 `human_chunk`
- 再进入调度得到 `human_sched_t`
- 再得到真正下发的 `human_exec_t`

这样 human 与 policy 才在部署语义上对齐。

## 7.6 与 `record_data_ros` 的关系

当前 human live execute 参考 `record_data_ros` ownership uplift 后的 teleop upper-control 经验，但不完全等同于其最终目标实现：

- 相同点：
  - owner 在 upper machine
  - lower machine 只保留 signal，不真实下发
  - fail-safe 恢复 lower owner 已验证可行
- 不同点：
  - `record_data_ros` 的上位机 teleop 控制更接近单一路径 upper teleop host
  - `inference_ros` 当前 live 第一版先复用这条已验证路径
  - `inference_ros` 的后续版本才进一步让 human 路径与 policy 路径在 chunk/schedule/execute 语义上完全对齐

因此 inference live 第一版会先“复制 record_data_ros 可用的 upper teleop target 路径”，然后在后续版本再补齐 human 的调度与执行链路。

---

## 8. Live 阶段的运行语义

## 8.1 进入 live

进入 live episode 前执行：

1. 初始化上位机 teleop signal client
2. 检查 backend `control_state`
3. 请求切换到：
   - `local_control_enabled=false`
   - `control_owner=upper_machine`
4. 若切换失败：
   - 不允许进入 live
   - 回退为非 live 状态
   - 记录错误

## 8.2 live 中的 source-select

每个调度周期：

1. policy 产生 `policy_chunk`
2. teleop 产生 human 当前 execute target，并继续记录 `human_chunk`
3. 根据 teleop active/valid/stale 判定 source
4. 当前版本中：
   - 若 `shared_source=policy`，沿用 policy live path
   - 若 `shared_source=human`，直接下发当前 processed teleop target
5. 后续版本再把第 4 步收敛为统一的 `shared_sched_t / shared_exec_t`

## 8.3 teleop active 进入 human source

当操作者按下 hand controller 的 active/squeeze 等有效控制按键后：

- human signal 变为 active
- `shared_source` 切为 `human`
- live execute 不再继续执行 policy source

这就是当前版本中的人工接管。

## 8.4 teleop inactive 回退到 policy

当操作者松开 active 控制后：

- 若 signal 有效但 inactive：
  - `shared_source` 回退为 `policy`
- owner 不切回 lower
- live 继续在 upper owner 下运行

---

## 9. 失败模式与 fail-safe

## 9.1 signal stale

当 teleop signal stale 时：

- 若当前 `shared_source=human`
  - 自动回退到 `policy`
- 不切 owner
- 记录 fallback reason

## 9.2 inference 异常退出

当 inference 节点异常退出、Ctrl+C、ROS shutdown、或 recorder/live session 非正常中断时：

- 先停止上位机 live execute
- 再 best-effort 调 backend 恢复：
  - `local_control_enabled=true`
  - `control_owner=lower_machine`

这部分直接复用 `record_data_ros` ownership uplift 已验证的恢复语义。

## 9.3 恢复失败

若恢复 lower owner 失败：

- 记录明确错误
- 不额外自动发回零动作
- 不自动补额外 move_joint/move_pose

## 9.4 退出时的目标

当前 fail-safe 的核心目标不是“自动把机械臂拉回一个固定姿态”，而是：

- 停止 upper 继续写 SDK
- 尽快把真实控制权交还给 lower teleop loop

---

## 10. Recorder / Timeline / Provenance 要求

live 版本需要完整记录以下三类 provenance。

## 10.1 policy 侧

- `action_policy_chunk`
- `action_policy_sched`
- `action_policy_exec`

## 10.2 human 侧

- `action_human_chunk`
- `action_human_sched`
- `action_human_exec`

## 10.3 shared 侧

- `action_shared_chunk`
- `action_shared_sched`
- `action_shared_exec`
- `hitl_shared_source`
- `hitl_shared_valid_mask`
- `hitl_signal_age_ms`
- `hitl_human_active`
- `hitl_human_valid`

当前版本至少要记录：

- `human direct execute target`
- `human active/valid/stale`
- `shared_source`

后续版本再补齐 `human_sched / human_exec` 的完整 provenance。

---

## 11. Public Interfaces / Config

下一阶段需要稳定以下外部约定。

### 11.1 `hitl_mode`

- `disabled`
- `shadow`
- `candidate`
- `live`

### 11.2 live owner 相关约定

live 模式显式依赖 backend：

- `GET /api/joystick/control_state`
- `POST /api/joystick/control_state`

### 11.3 signal source 约定

- 默认 `teleop_v2_processed`
- 不新增 raw teleop control law 迁移开关

### 11.4 不新增的参数

当前阶段明确不开放：

- blending weight
- per-dim override
- learned arbitration

---

## 12. Test Plan

## 12.1 文档一致性验证

- 与 `project_memory_baseline_2026-04-11.md` 不冲突
- 与 `chunk_action_semantics_and_teleop_alignment.md` 的三层 action 语义一致
- 与 `record_data_ros` ownership uplift 结论一致

## 12.2 单元测试

新增或扩展：

- 当前版本：
  - human execute target 提取测试
  - source-select 触发 human direct execute 测试
- 后续版本：
  - `human_chunk -> human_sched_t -> human_exec_t` 语义测试
- source-select 规则测试：
  - policy
  - human active
  - stale fallback
- live owner 状态机测试：
  - enter upper owner
  - keep upper owner while source switches
  - restore lower owner on shutdown

## 12.3 集成验证：shadow

- 行为与当前第一版一致
- 继续不切 owner

## 12.4 集成验证：candidate

- 继续不切 owner
- 当前先补 human candidate/direct execute 观测
- 后续再补 scheduler/query 对齐版 candidate

## 12.5 集成验证：live

重点验证：

1. live 开始前 owner 成功切到 `upper_machine`
2. lower teleop loop 不再真实下发
3. `teleop active` 时 `shared_source` 正确切到 `human`
4. `teleop inactive` 时 `shared_source` 正确回到 `policy`
5. 当前版本 human direct execute 路径稳定成立
6. inference 退出后 owner 恢复到 `lower_machine`
7. 后续版本再验证 human 路径的 `chunk -> scheduled -> executed` 完整成立

---

## 13. 下一步实现顺序

建议实现顺序如下：

1. 先补文档与语义冻结
2. 在 `live` 模式中先落 human direct execute 与 owner/source 语义
3. 补 recorder/timeline 的 direct human execute provenance
4. 做最小幅度 live 真机验证
5. 再补 human 的 scheduler/query 对齐版链路
6. 最后再讨论更复杂的人机协同策略

---

## 14. 相关文档

- [project_memory_baseline_2026-04-11.md](/home/amax/rl-vla/docs/project_memory_baseline_2026-04-11.md)
- [chunk_action_semantics_and_teleop_alignment.md](/home/amax/rl-vla/docs/chunk_action_semantics_and_teleop_alignment.md)
- [teleop_uplift_progress_2026-04-11.md](/home/amax/rl-vla/docs/teleop_uplift_progress_2026-04-11.md)
- [lower_machine_teleop_recon_2026-04-10.md](/home/amax/rl-vla/docs/lower_machine_teleop_recon_2026-04-10.md)
