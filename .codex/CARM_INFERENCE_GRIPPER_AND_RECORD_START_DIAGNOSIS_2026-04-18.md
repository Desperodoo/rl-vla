# CARM Inference Gripper And Record-Start Diagnosis Plan

日期基线：2026-04-18  
状态：本轮诊断与修复已完成阶段性收敛；`gripper` 问题已修复，`R` 开始时第二条 episode 的明显下坠在 `eval_fixed_4` 中已不再复现

## 1. 背景

当前 inference 脚本暴露出两个新的真机问题：

1. 抓取阶段夹爪闭合力度不足
   - 现象：推理抓取时更像“轻轻夹住”，容易掉物体
   - 对照事实：网页 teleop / 手柄遥操时夹爪表现正常，可以明显更紧地抓住物体

2. 按 `R` 开始 `record inference` 时机械臂会瞬时下坠
   - 现象：episode 开始瞬间机械臂会向下掉一小段，量级可到约 `10cm` 或更多
   - 对照事实：这个现象在添加 HITL 之前没有出现

这两个问题都优先按“部署链路诊断”处理，而不是先怀疑训练本身。

## 2. 当前初步判断

### 2.1 gripper 问题更像 inference pipeline 语义问题

当前 evidence：

- lower-machine teleop 在 `backend/carm_backend/server/basic/joystick.py` 中，夹爪目标按 trigger 连续映射：
  - `end_gripper_pose = (1 - trigger/127) * 0.08`
  - 因此 teleop 可以自然到达接近 `0.0` 的真实闭合位
- inference 在 `carm_ros_deploy/src/carm_deploy/inference/policy_loader.py` 中，当前离散 gripper 默认值是：
  - `gripper_open_val = 0.078`
  - `gripper_close_val = 0.04`

这意味着：

- teleop 的“close”语义接近“尽量闭合”
- inference 的“close”语义更接近“半闭合”

因此当前更合理的主判断是：

- 主因更像 inference pipeline 的 gripper target 定义不够激进
- 不是 SDK 本身没有力，也不是 `robot_tau` 明显不够

### 2.2 `R` 开始时下坠更像 live owner handoff 空窗

当前 evidence：

- `inference_ros.py` 的 `_start_new_episode()` 会在 live 模式下先 `_activate_hitl_live_owner()`
- 这一步会先让 backend 切到：
  - `local_control_enabled = false`
  - `control_owner = upper_machine`
- 但此时上位机新的 hold/action chunk 还没有保证已经可执行
- `control_loop()` 在 `action_manager` 暂时为空时不会主动发当前位姿 hold target，而是直接等待下一条 action

这意味着在 episode 开始时可能出现：

```text
lower_machine 停止真实下发
-> upper_machine 已接 owner
-> 但第一条有效 hold/track_pose 还没送出
```

对真机表现而言，这就是最可疑的“开始瞬间下坠窗口”。

## 3. 现有数据是否足够

现有可用目录：

- `inference_logs/eval_fixed/`

其中已确认：

- 多条 `inference_episode_*.hdf5`
- `hitl_mode=live`
- `hitl_live_execute_enabled=True`
- HDF5 已包含：
  - `action_policy_chunk`
  - `action_shared_chunk`
  - `action_live_execute_target`
  - `action_human_direct_target`
  - `action_human_sched_target`
  - `action_human_exec_target`
  - `hitl_live_execute_source`
- HDF5 已包含 `control_provenance/`
  - `execute_source`
  - `human_execute_mode`
  - `live_execute_target`
  - `human_direct_target`
  - `human_sched_target`
  - `human_exec_target`
  - `shared_source`

### 3.1 对 gripper 问题：现有数据已经足够做第一轮分析

离线检查已经显示：

- `action_live_execute_target[..., 7]` 的主值基本是：
  - `0.078`
  - `0.04`
- 也就是说，policy 路径下的 close 语义确实主要停在 `0.04`
- 这与 teleop 的连续闭合语义明显不一致

因此：

- gripper 问题不需要先重采
- 可以直接基于现有数据与当前代码先修

### 3.2 对 `R` 开始下坠问题：现有数据只够做间接判断，不够最终定责

现有 rollout 能提供：

- episode 开始后的第一批 observation / action / control_provenance
- 第一个 recorded step 附近的 `qpos_end.z` 与 `live_execute_target.z`

但它缺少更关键的“handoff 瞬间”显式记录，例如：

- `R` 按下时刻
- `_activate_hitl_live_owner()` 成功时刻
- live owner 切换完成后的第一条 hold/track_pose 发送时刻
- owner handoff 前最后一条 lower-machine 实际控制状态

因此当前结论是：

- 现有数据足够支持“handoff 空窗”是强嫌疑根因
- 但还不够做最终定责
- 真正修这个问题前，最好补 start-transition instrumentation

## 4. 固定执行顺序

当前执行顺序固定为：

1. 先修 gripper
2. 再修 `R` 开始时的 owner handoff 空窗

不建议反过来，原因：

- gripper 问题的根因更单纯、现有数据更充分
- 先修 gripper 可以先把“抓不紧”这个明显语义问题从系统里拿掉
- 之后再聚焦 handoff 时序问题，会更容易判断剩余异常

## 5. 诊断与修复计划

### 5.1 阶段 A：gripper 诊断与修复

目标：

- 让 inference 的“close”语义与 teleop 的真实抓取需求更一致

执行步骤：

1. 保持使用 `eval_fixed` 现有 rollout 做离线确认
   - 验证 `action_live_execute_target[..., 7]`
   - 验证 `control_provenance/live_execute_target[..., 7]`
   - 确认 policy 真实下发值主要停在 `0.04`

2. 检查 inference 夹爪链路
   - `policy_loader.py` 中的：
     - `gripper_open_val`
     - `gripper_close_val`
     - hysteresis
   - `inference_ros.py` 中的 safety 后处理
   - `env_ros.py` 中的 `set_gripper(..., tau)`

3. 第一版修复策略
   - 将 inference 默认 `gripper_close_val` 从 `0.04` 调整为更接近真实闭合的值
   - 保留 CLI / config override，避免后续实验被硬编码卡死

4. 验证目标
   - policy 抓取阶段的实际下发值明显小于当前的 `0.04`
   - 真机抓取更稳，不再表现为“轻轻夹住”

### 5.2 阶段 B：`R` 开始时 handoff 空窗诊断与修复

目标：

- 消除开始 episode 时 lower writer 停止与 upper 第一条有效命令之间的时序空窗

执行步骤：

1. 先补 instrumentation
   - episode start requested
   - owner acquire started
   - owner acquire success
   - first upper hold target ready
   - first upper control send

2. 修复策略优先顺序
   - 在切到 `upper_machine` 之前，先准备一个当前位姿 hold target
   - owner handoff 成功后，确保上位机立刻有一条有效 `track_pose`
   - 不允许出现 `action_manager` 为空且无人继续 hold 的窗口

3. 预期改法方向
   - `_start_new_episode()` 中先读取当前 `qpos_end`
   - 基于当前 `qpos_end` 构造 `hold_action`
   - 让上位机在 owner 切换前后都能无缝延续 hold

4. 验证目标
   - 反复按 `R` 开始 episode 时，不再出现明显瞬时下坠
   - 即使 policy 第一条 chunk 还没生成，上位机也能先稳定保位

## 6. 数据重采策略

当前策略不是立即要求重采。

### 6.1 gripper

- 先不重采
- 用 `eval_fixed` 分析后直接修

### 6.2 handoff 空窗

- 若仅靠现有数据和代码时序检查还不能完全坐实
- 则先补 recorder / timeline 的 start-transition 事件
- 再让操作者重采一条“只测试 `R` 开始”的最小真机样本

## 7. 修复结果与证据

### 7.1 gripper 已修复

根因已明确：

- `carm_ros_deploy/src/carm_deploy/inference/policy_loader.py` 会从 checkpoint 的 `args.json` 恢复历史参数
- 目标 checkpoint 中保存的 `gripper_close_val=0.04`
- 这会覆盖运行时默认值，导致 inference 的“close”长期停在偏松的半闭合语义

已完成修复：

- `policy_loader.py` 在加载 checkpoint 配置后重新应用运行时 gripper 配置
- `inference_ros.py` 新增显式 CLI：
  - `--gripper_threshold`
  - `--gripper_open_val`
  - `--gripper_close_val`
- 当前默认 close 路径会落到更接近真实闭合的位置，之后再经过 safety clip

修复证据：

- `inference_logs/eval_fixed_3/` 中已观察到 `policy0_gripper` 与 `live_gripper` 主值变为 `[0.008, 0.078]`
- 操作者真机反馈：夹爪“闭合非常实”，抓取力度恢复正常

当前结论：

- gripper 问题已经不是当前 inference pipeline 的阻塞项
- 后续若再做实验，只需要通过 CLI 显式覆写即可，不需要再改 checkpoint 历史参数

### 7.2 `R` 开始时的第二条 episode 下坠已修复

根因已从“owner handoff 空窗”的宽泛怀疑，收敛为更具体的问题：

- 第二条 episode 开始时，hold seed 复用了 stale `latest_obs.qpos_end`
- 旧坏样本里，hold seed 的 `z` 约为 `0.230`
- 而真机当时实际位置约为 `0.325`
- 上位机接 owner 后第一条保位命令向旧低位姿回拉，因此出现明显下坠

已完成修复：

- `_start_new_episode()` 优先通过 `env.get_state_observation()` 获取 fresh state，用于 seed hold chunk
- 仅在 fresh snapshot 不可用时才回退到 `latest_obs`
- `_reinitialize_arm()` 后主动清空 `self.latest_obs`
- 保留 episode start instrumentation，持续记录：
  - `start_requested`
  - `hold_chunk_seeded`
  - `owner_acquire_started`
  - `owner_acquired`
  - `episode_unpaused`
  - `first_policy_chunk_added`
  - `first_control_sent`

坏样本证据：

- `inference_logs/eval_fixed_3/inference_episode_0002_20260418_202419.hdf5`
- 第二条 episode 中：
  - `hold_target z = 0.230422`
  - 首帧 `obs_z = 0.325017`
  - 前几帧观测快速跌落到 `0.317 -> 0.305 -> 0.291`

修复后证据：

- `inference_logs/eval_fixed_4/inference_episode_0002_20260419_125335.hdf5`
- `inference_logs/eval_fixed_4/timeline_20260419_125205.jsonl`
- 第二条 episode 的 `start_requested / hold_chunk_seeded / episode_unpaused / first_control_sent` 中：
  - `current_qpos_end z ≈ 0.325020`
  - `hold_target z ≈ 0.325020`
- 同一 episode 前 10 帧观测 `z` 为：
  - `0.325020, 0.325023, 0.325023, 0.324455, 0.324171, ...`
- 不再出现此前约 `10cm` 量级的启动下坠
- 操作者体感反馈：第二次按 `R` 时机械臂不会明显下坠

当前结论：

- 本轮关于 `R` 开始时下坠的修复已经命中根因
- 现阶段可以将该问题从“待修”降级为“已修复，后续观察”

## 8. 当前决策

当前固定决策如下：

1. `gripper` 问题已完成诊断和修复，可继续作为稳定默认行为使用
2. 第二条 episode 启动下坠问题已通过 `eval_fixed_4` 验证为“不再复现”
3. start-transition instrumentation 继续保留，作为后续 live/HITL 调试基线
4. 当前 inference 侧剩余更值得继续推进的事项，重新回到 HITL 边界切换与 human/policy 语义对齐
