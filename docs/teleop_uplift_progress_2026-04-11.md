# Teleop Uplift 最新进展

**日期**: 2026-04-11  
**状态**: `record_data_ros` 宿主化已落地，`passive_shadow` / `upper_control` candidate-only / `upper_control` live execute 已完成首轮真机与异常退出验证，已开始清理 `inference_ros` 中旧 intervention 链路  
**定位**: 作为 `project_memory_baseline_2026-04-11.md` 之后的最新实现与现场验证续篇

---

## 1. 本轮完成内容

本轮已经不再停留在方案设计，而是完成了从本地实现到真机核对的一整轮闭环：

1. 在 `docs/` 下新增项目记忆基线文档，作为后续统一入口
2. 将 `record_data_ros` 升级为 teleop uplift 过渡宿主
3. 将 `teleop_shadow` 的观测与几何重建逻辑并入 recorder
4. 为下位机 backend 增加显式 `control_state` 接口与本地下发门控
5. 完成 `passive_shadow` 真机录制验证
6. 完成 `upper_control` candidate-only 真机录制验证
7. 完成 `upper_control` live execute 最小幅度真机验证
8. 修正 `signal_age_ms` 的跨机绝对时钟问题
9. 修正 candidate 误差统计中的 `NaN` 污染问题
10. 完成 live execute 长时稳定性验证
11. 完成 recorder 异常退出 / Ctrl+C 下的 owner 恢复演练
12. 开始清理 `inference_ros` 中旧 intervention pipeline，为 Human-in-the-loop inference 准备更干净的入口

---

## 2. 当前代码落地状态

### 2.1 上位机 recorder 侧

当前 `record_data_ros` 已经从纯 recorder 升级为双模式 teleop 宿主：

- `passive_shadow`
- `upper_control`

已落地的新参数包括：

- `teleop_bridge_mode`
- `backend_url_v2`
- `events_v2_url`
- `pred_horizon`
- `act_horizon`
- `teleop_candidate_control_freq`
- `teleop_signal_timeout_ms`
- `upper_control_enabled`

已落地的共享组件包括：

- `TeleopSignalClient`
- `TeleopShadowTransformer`
- `TeleopUpperControlBridge`

当前实现位置主要在：

- [record_data_ros.py](/home/amax/rl-vla/carm_ros_deploy/src/carm_deploy/data/record_data_ros.py)
- [teleop_bridge.py](/home/amax/rl-vla/carm_ros_deploy/src/carm_deploy/data/teleop_bridge.py)
- [record.launch](/home/amax/rl-vla/carm_ros_deploy/src/carm_deploy/launch/record.launch)

### 2.2 下位机 backend 侧

下位机 backend 已增加显式 ownership 接口与本地下发 gate：

- `GET /api/joystick/control_state`
- `POST /api/joystick/control_state`

当前行为是：

- `local_control_enabled=True` 时，下位机本地 `50Hz` loop 正常向 SDK 下发
- `local_control_enabled=False` 时，下位机仍持续更新 `processed/raw` 状态，但不再向 SDK 下发

当前实现位置主要在：

- [joystick.py](/home/amax/rl-vla/backend/carm_backend/server/api/joystick.py)
- [joystick.py](/home/amax/rl-vla/backend/carm_backend/server/basic/joystick.py)

---

## 3. 数据落盘状态

当前 HDF5 在保留旧字段兼容性的同时，已经新增 teleop uplift 所需字段。

保留兼容字段：

- `action`
- `teleop_scale`

新增 shadow / candidate / owner 相关字段：

- `teleop_processed_target_abs`
- `teleop_human_chunk_abs`
- `teleop_human_chunk_rel`
- `teleop_reconstructed_target_abs`
- `teleop_active`
- `teleop_processed_sequence`
- `teleop_raw_sequence`
- `teleop_signal_age_ms`
- `teleop_abs_reconstruction_pos_error`
- `teleop_abs_reconstruction_rot_error`
- `upper_candidate_target_abs`
- `upper_candidate_pos_error`
- `upper_candidate_rot_error`
- `teleop_candidate_loop_dt_ms`
- `teleop_candidate_stale`
- `teleop_candidate_applied`

当前 HDF5 metadata 已记录：

- `data_version=v4`
- `teleop_bridge_mode`
- `backend_url_v2`
- `events_v2_url`
- `teleop_signal_timeout_ms`
- `pred_horizon`
- `act_horizon`
- `lower_control_enabled_at_start`
- `control_owner_at_start`
- `upper_control_enabled_at_start`

timeline JSONL 也已同步记录：

- teleop v2 sequence / active / signal age
- reconstruction error
- candidate loop dt / stale / applied

---

## 4. 真机验证结果

### 4.1 backend ownership gate 验证

在真实下位机上已经确认：

1. 切到 `upper_machine` 后，机器人会停止跟随下位机本地 teleop 输入
2. 即使停止本地下发，`teleop_target_v2` / `events_v2` 中的 `processed/raw` 仍持续更新
3. 这说明当前 backend gate 已经把“状态更新”和“真实 SDK 下发”正确解耦

这一步验证了：

> ownership 迁移在 backend 层是可控的，不是靠隐式约定碰运气。

### 4.2 `passive_shadow` 验证

已完成 `record_data_ros --teleop_bridge_mode passive_shadow` 的真机录制。

关键结论：

1. recorder 可以稳定录制并保留旧字段兼容
2. 新增 teleop v2 / shadow 字段已成功落盘
3. `processed.target_pose_abs -> human_chunk_rel -> reconstructed_target_abs` 重建闭环成立
4. 位置重建误差仍为数值误差量级，姿态重建误差为 `0`

### 4.3 `upper_control` candidate-only 验证

已完成 `teleop_bridge_mode=upper_control` 且 `upper_control_enabled=False` 的真机录制。

关键结论：

1. 独立 candidate 线程可以稳定运行
2. 录制主循环未出现明显阻塞
3. `teleop_candidate_applied` 全程为 `False`
4. backend owner 始终保持为 `lower_machine`

首轮 candidate-only 样本中，候选线程指标为：

- loop dt p50 约 `20.18ms`
- loop dt p95 约 `23.80ms`
- loop dt max 约 `26.33ms`

这说明：

> candidate 骨架已经达到接近 `50Hz` 的稳定运行状态。

### 4.4 `upper_control` live execute 最小幅度验证

已完成 `teleop_bridge_mode=upper_control --upper_control_enabled` 的首轮最小幅度真机验证。

关键结论：

1. `recording=True` 后 backend 确实切到：
   - `local_control_enabled=false`
   - `control_owner=upper_machine`
2. `recording=False` 后 backend 能恢复到：
   - `local_control_enabled=true`
   - `control_owner=lower_machine`
3. live 模式下上位机确实在向机械臂真实下发命令
4. 退出路径没有触发自动回零，符合当前 fail-safe 约定

对应 live 录制样本：

- [episode_0001_20260411_182820.hdf5](/home/amax/rl-vla/tmp_record_data/episode_0001_20260411_182820.hdf5)
- [timeline_record_20260411_182725.jsonl](/home/amax/rl-vla/tmp_record_data/timeline_record_20260411_182725.jsonl)

该样本的关键指标：

- `teleop_candidate_applied = 830 / 847`
- `teleop_candidate_stale = 17`
- candidate loop dt p50 约 `20.35ms`
- candidate loop dt p95 约 `24.83ms`
- `teleop_signal_age_ms` p50 约 `8.59ms`
- `teleop_signal_age_ms` p95 约 `16.78ms`

### 4.5 `upper_control` live execute 长时验证

已完成一轮更长时间的 live execute 真机录制。

对应样本：

- [episode_0001_20260411_190005.hdf5](/home/amax/rl-vla/recorded_data/tmp_record_data/episode_0001_20260411_190005.hdf5)
- [timeline_record_20260411_185845.jsonl](/home/amax/rl-vla/recorded_data/tmp_record_data/timeline_record_20260411_185845.jsonl)

关键结论：

1. `recording=True` 期间 owner 持续保持在 `upper_machine`
2. 停录后 owner 能恢复到 `lower_machine`
3. candidate/live 控制线程在长时窗口下仍保持稳定

该样本关键指标：

- `teleop_candidate_applied = 1489 / 1751`
- `teleop_candidate_stale = 262 / 1751`
- `teleop_signal_age_ms` p50 约 `8.91ms`
- `teleop_signal_age_ms` p95 约 `17.30ms`
- candidate loop dt p50 约 `20.32ms`
- candidate loop dt p95 约 `22.23ms`

### 4.6 stale 拆分结果

为避免把“正常松手”与“真实 timeout”混在一起，现已在分析脚本中将 stale 拆分为：

- `inactive_stale`
- `timeout_stale`

对长时 live 样本的拆分结果是：

- `inactive_stale = 254 / 1751`
- `timeout_stale = 8 / 1751`

这说明长时样本中大部分 stale 都来自 teleop inactive，而不是 active 期间的链路陈旧。

### 4.7 异常退出 / owner 恢复演练

已完成多轮 recorder 中途中断与 `Ctrl+C` 异常退出演练。

当前确认：

1. live execute 运行中中断 recorder 后，backend owner 仍可恢复回 `lower_machine`
2. shutdown 路径中原先存在的 timeline logger 关闭竞态已修正
3. shutdown 路径中原先存在的 owner 恢复超时误报已修正

当前阶段可以明确确认：

> recorder 异常退出并不会把系统卡死在 `upper_machine` ownership。

---

## 5. 关键修正

### 5.1 `signal_age_ms` 修正

最初的 `teleop_signal_age_ms` 直接使用：

- 下位机 `source_timestamp`
- 上位机本地当前时间

来计算 age，这在没有严格跨机对时的前提下会产生巨大的伪数值。

现已修正为：

1. 在下位机直接计算并返回：
   - `processed.source_age_ms`
   - `raw.source_age_ms`
2. 上位机优先使用 backend 返回的 `source_age_ms`
3. 若缺失，再退回到本地 arrival-gap / sequence-change 的启发式近似

修正后的真机观测表明：

- active 样本 `teleop_signal_age_ms` 已回到合理量级
- 不再出现几十亿毫秒级别的跨机伪延迟

### 5.2 candidate 误差统计 `NaN` 修正

candidate-only 首轮样本中，`upper_candidate_rot_error` 曾出现 `NaN`。

根因是：

- inactive / stale 且无有效 target 时
- 候选目标会回落到全零占位
- 全零四元数参与旋转误差计算会产生 `NaN`

现已修正为：

1. 只有在观测姿态和 candidate 姿态都具有有效四元数时才计算旋转误差
2. 否则旋转误差安全置为 `0.0`

修正后，live 样本里的 `upper_candidate_rot_error` 已无 `NaN`。

### 5.3 metadata 语义修正

此前 `upper_control_enabled_at_start` 曾被写成“backend 启动时是否已经是 upper owner”，这会和“本次录制是否启用 live upper control”混淆。

现已调整为：

- `control_owner_at_start`: 记录录制启动时 backend owner
- `upper_control_enabled_at_start`: 记录本次 recorder 是否启用了 live upper control

### 5.4 shutdown 竞态与 owner 恢复误报修正

在异常退出演练中，曾暴露出两个退出路径问题：

1. `timeline_logger` 已关闭，但主循环仍有最后一步尝试写日志
2. owner 实际已经恢复，但上位机使用过短 HTTP timeout 导致误报“恢复失败”

现已修正为：

1. `record_data_ros` 的 shutdown 变为幂等，并增加 `shutting_down` 保护
2. `TimelineLogger` 在关闭后会忽略后续写入
3. `TeleopUpperControlBridge.deactivate_owner()` 使用更宽松 timeout 与少量重试做 best-effort 恢复

修正后，live execute 中途中断退出日志已恢复干净。

### 5.5 `inference_ros` intervention 链路清理

为加速进入 Human-in-the-loop inference，本轮已开始清理当前 `inference_ros` 主 pipeline 中的旧 intervention 内容。

当前已完成：

1. `inference_ros` 主循环不再在 policy 输出与执行之间插入 `InterventionApplier`
2. episode 控制从 `KeyboardInterventionHandler` 解耦，改为专用的 `EpisodeKeyboardHandler`
3. `InferenceRecorder` 主语义开始从：
   - `action_model + action_intervened + intervention_mask`
   收敛为：
   - `action_model + action_executed`
4. 为避免立刻打断旧分析链路，短期仍保留兼容字段：
   - `action_intervened`
   - `intervention_mask`

当前阶段目标不是立刻完成 HITL，而是：

> 先把旧键盘 intervention 路径从推理主链路中清出来，再为 HITL 接入腾出更明确的入口。

---

## 6. 当前测试状态

当前已补充并通过的测试包括：

- [test_teleop_bridge.py](/home/amax/rl-vla/carm_ros_deploy/src/carm_deploy/tests/test_teleop_bridge.py)
- [test_backend_control_state.py](/home/amax/rl-vla/carm_ros_deploy/src/carm_deploy/tests/test_backend_control_state.py)
- [test_inference_node.py](/home/amax/rl-vla/carm_ros_deploy/src/carm_deploy/tests/test_inference_node.py)

当前本地回归状态：

- `test_teleop_bridge.py`: `9 passed`
- `test_backend_control_state.py`: `4 passed`
- `test_inference_node.py`: `7 passed`

为运行这两组测试，已在 `carm` 环境中安装：

- `pytest`
- `flask`

---

## 7. 当前阶段结论

截至 `2026-04-11`，项目已经从“shadow 方案设计”进入“可运行的 ownership uplift 原型”阶段。

当前可以明确确认：

1. `record_data_ros` 已经能作为唯一过渡宿主承接 teleop uplift
2. `passive_shadow` 已经稳定可用
3. `upper_control` candidate-only 已经稳定可用
4. `upper_control` live execute 已完成最小幅度与长时两轮真机验证
5. backend ownership gate、owner 切换与退出恢复链路已经真实跑通
6. recorder 异常退出 / `Ctrl+C` 下的 owner 恢复链路已经演练通过
7. `signal_age_ms`、candidate 误差统计、shutdown race 与 owner 恢复误报等关键观测问题已经修正
8. `inference_ros` 中旧 intervention 主链路已经开始清理，为 HITL inference 做前置收口

因此当前项目状态可以概括为：

> Teleop ownership uplift 的第一版工程闭环已经基本跑通，系统性验证已覆盖到长时 live 与异常退出恢复；下一阶段可以开始把重心从 ownership 验证转向 Human-in-the-loop inference 接入前的 pipeline 收口。

---

## 8. 下一步建议

当前建议从“继续验证 ownership”转向“开始收口 inference pipeline，为 HITL 接入做准备”。

建议优先顺序：

1. 继续清理 `inference_ros` / `InferenceRecorder` 中遗留的 intervention 语义
2. 设计并接入 Human-in-the-loop inference 所需的显式 owner/source arbitration
3. 复用 `TeleopSignalClient` / `TeleopUpperControlBridge` 到 inference 侧
4. 将 inference rollout 落盘语义与 teleop uplift 数据语义做统一
5. 在 HITL 主链路跑通后，再按需回补更复杂动作任务上的对比验证

---

## 9. 相关文档

- [project_memory_baseline_2026-04-11.md](/home/amax/rl-vla/docs/project_memory_baseline_2026-04-11.md)
- [teleop_signal_uplift_shadow_plan.md](/home/amax/rl-vla/docs/teleop_signal_uplift_shadow_plan.md)
- [lower_machine_teleop_recon_2026-04-10.md](/home/amax/rl-vla/docs/lower_machine_teleop_recon_2026-04-10.md)
- [chunk_action_semantics_and_teleop_alignment.md](/home/amax/rl-vla/docs/chunk_action_semantics_and_teleop_alignment.md)
