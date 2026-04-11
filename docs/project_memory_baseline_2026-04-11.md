# 项目记忆基线

**日期**: 2026-04-11  
**定位**: 当前 `rl-vla` 项目的压缩记忆入口  
**用途**: 为后续 Teleop Uplift、真机部署、推理回流与 RL 微调提供统一上下文

---

## 1. 项目主线

当前仓库覆盖一条完整的真机闭环：

1. 遥操作真机数据采集
2. policy 模仿学习训练
3. 真机 policy 部署推理
4. 真机推理数据回流
5. 后续真机 RL 微调

当前最关键的问题不是模型结构，而是：

> 采集、训练、部署、回流之间的 action 与控制通道语义是否一致。

---

## 2. Action 三层语义

当前系统里，`action` 不能再被视为单一对象。部署链路已经确认存在三层语义：

### 2.1 `a_policy_chunk`

- policy 一次推理输出的整段 chunk
- 最接近“agent 真正输出的动作”

### 2.2 `a_sched_t`

- 控制线程在某个控制时刻，从 `ActionChunkManager` 中取出的单步动作
- 已包含时间戳调度、`receding_horizon` 覆盖语义、推理频率与控制频率不一致带来的采样效应

### 2.3 `a_exec_t`

- 最终真正发给 SDK 的动作
- 若后续存在安全裁剪、人工干预、限幅等，它还会继续偏离 `a_sched_t`

---

## 3. 当前收敛结论

### 3.1 学习动作仍是 chunk-level policy action

当前项目继续把学习语义定义为 chunk-level policy action，而不是把系统降格成单步控制学习。

也就是说：

- 学习动作的核心仍是 `a_policy_chunk`
- `a_policy_chunk -> a_sched_t -> a_exec_t` 被视为部署通道
- 本轮 Teleop Uplift 不修改 action 定义

### 3.2 部署与回流必须同时记录三层 action

虽然学习动作不偷换成 `a_exec_t`，但工程上必须记录：

- `a_policy_chunk`
- `a_sched_t`
- `a_exec_t`

否则无法：

- 量化 train-deploy gap
- 解释真机状态转移
- 支撑后续 Human-in-the-loop 与 RL 设计

### 3.3 train-deploy gap 是结构性的

当前 gap 不是简单噪声，而是数据生成闭环与部署闭环不同：

- teleop 当前更像“人通过下位机本地闭环直接控制机械臂”
- inference 当前是“policy 输出 chunk，再经过 scheduler / control loop / SDK 执行”

因此 gap 的根源在闭环结构本身，而不只是 learner 结构。

---

## 4. 下位机 Teleop 当前事实

根据第一轮摸底，当前真实 teleop owner 在 lower machine：

- 下位机 backend 位于 `/var/www/backend/carm_backend`
- 真实控制逻辑位于 `server/basic/joystick.py`
- `CommandController.loop()` 以 `50Hz` 运行
- 在 lower machine 本地直接调用：
  - `track_pose(...)`
  - `set_gripper(...)`

当前上位机不是 teleop 控制 owner，而是旁路观察者。

当前对外信号可分为两类：

- raw teleop signal
  - 接近手柄姿态、按钮、trigger/grip 等输入
- processed teleop target
  - 已经过下位机 teleop 控制律处理
  - 语义是“可直接送 SDK 的绝对目标位姿”

---

## 5. Shadow 阶段已验证结论

第一轮 signal uplift / shadow 验证已经表明：

- 下位机双通道接口可用：
  - `/api/joystick/teleop_target_v2`
  - `/api/joystick/events_v2`
- 上位机可稳定读取 `processed.target_pose_abs`
- `processed.target_pose_abs -> human_chunk_rel -> reconstructed_target_abs` 几何闭环成立
- 位置重建误差为数值误差量级，姿态重建误差为 `0`
- HTTP 轮询延迟稳定，均值约 `6ms`，P95 约 `8ms`
- 当前无需迁移真实控制权，也已经验证了 signal uplift 的核心可行性

因此当前阶段结论是：

> 先做“信号上移与 shadow 复现”，再讨论控制权迁移。

---

## 6. 当前迁移约定

当前已经锁定以下工程约定：

- 过渡宿主选 `record_data_ros`，不是 `inference_ros`
- `record_data_ros` 升级为双模式宿主：
  - `passive_shadow`
  - `upper_control`
- 第一阶段仍不接管，只把 `teleop_shadow` 能力并入 `record_data_ros`
- 共享组件拆为：
  - `TeleopSignalClient`
  - `TeleopShadowTransformer`
  - `TeleopUpperControlBridge`
- 候选控制阶段先只生成 upper candidate command，不真实下发
- 只有候选验证通过后，才进入真实接管
- 真实接管时只在 `recording=True` 期间由上位机持有 teleop 控制权
- 下位机 backend 增加显式 `control_state` 接口，不能靠隐式约定切 owner
- fail-safe 采用“停止更新并保位”，不自动回零
- 当前迁移阶段刻意不引入新的 scheduler / buffer，也不改变 teleop 控制语义，避免把“控制权变化”和“控制语义变化”耦合在一起

---

## 7. 当前未决问题

以下问题仍属于后续工作，而不是本基线文档内已定结论：

- Human-in-the-loop inference 的最终接入形态
- Teleop raw 信号是否在未来上移到上位机并重跑控制律
- 真机推理回流数据如何系统性组织成后续 RL 样本
- 三层 action 在训练分析工具链中的最终落盘与对齐规范

---

## 8. 相关文档索引

- [chunk_action_semantics_and_teleop_alignment.md](/home/amax/rl-vla/docs/chunk_action_semantics_and_teleop_alignment.md)
- [lower_machine_teleop_recon_2026-04-10.md](/home/amax/rl-vla/docs/lower_machine_teleop_recon_2026-04-10.md)
- [teleop_signal_uplift_shadow_plan.md](/home/amax/rl-vla/docs/teleop_signal_uplift_shadow_plan.md)
