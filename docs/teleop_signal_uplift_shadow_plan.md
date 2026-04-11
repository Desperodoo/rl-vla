# Teleop 信号上移与 Shadow 验证计划

**日期**: 2026-04-10
**状态**: 第一版已实现并完成首轮现场验证
**目标**: 在不立即迁移控制权的前提下，验证“下位机输出 teleop 信号，上位机复现 teleop 控制与 chunk proposal 逻辑”的可行性

---

## 1. 背景

当前 teleop 链路是：

1. 下位机 backend 接收 BLE 手柄数据
2. 下位机本地执行 teleop 控制逻辑
3. 下位机直接调用 SDK，下发 `track_pose()` / `set_gripper()`
4. 上位机通过 `1999` 端口旁路订阅 teleop 状态，用于数据采集

为了后续实现：

- teleop 与 inference 共享同一套部署通道
- teleop 侧也能接入“与部署一致的 scheduler / buffer -> chunk proposal”
- Human-in-the-loop inference 复用相同的人类输入通道

需要探索把 `teleop -> sdk` 逻辑逐步上移到上位机的方案。

---

## 2. 基本判断

不建议一步到位直接把控制权迁移到上位机。  
更稳妥的路线是：

> 先做“信号上移，不做控制上移”的 shadow 验证。

也就是：

- 下位机继续保持当前 teleop -> sdk 的真实控制闭环
- 上位机只接收 teleop 信号并在本地复现 teleop 控制逻辑
- 上位机先不对真实机械臂发控制
- 先比较“上位机复现结果”和“下位机真实控制结果”的一致性

只有 shadow 验证稳定后，再考虑迁移控制权。

---

## 3. 阶段 A：信号上移，不做控制上移

### 3.1 目标

验证上位机是否能基于下位机输出的 teleop 信号，稳定复现：

1. teleop 控制逻辑
2. teleop 对应的 chunk proposal 逻辑
3. 后续拟采用的 scheduler / buffer 语义

### 3.2 下位机保持不变的部分

- BLE 设备连接与管理
- 网页 UI 与用户交互
- 当前 teleop 本地闭环
- 当前对 SDK 的直接控制

### 3.3 上位机新增的部分

- 订阅下位机 teleop 信号
- 在上位机本地复现 teleop 控制逻辑
- 在上位机本地生成 `human_chunk_proposal`
- 记录 shadow 路径中的中间量与时序

### 3.4 建议的验证输出

shadow 阶段至少记录以下量：

1. 下位机 teleop 原始信号时间戳
2. 上位机接收时间戳
3. 上位机重建的 teleop target
4. 上位机生成的 `human_chunk_proposal`
5. 下位机实际 teleop target
6. 机械臂真实状态 `qpos_end`

### 3.5 需要重点比较的指标

1. 信号传输延迟
2. 传输抖动
3. 上位机重建 target 与下位机 target 的偏差
4. 上位机 chunk proposal 的稳定性
5. 长时间运行下的丢包、阻塞与线程稳定性

---

## 4. 阶段 B：控制权迁移

只有在阶段 A 验证通过后，才进入控制权迁移。

### 4.1 目标

将真实控制链路切换为：

`下位机 teleop signal -> 上位机 teleop controller / scheduler / buffer -> SDK`

### 4.2 迁移后的职责边界

下位机负责：

- BLE
- 网页 UI
- teleop 原始输入输出

上位机负责：

- teleop 控制逻辑
- scheduler / buffer
- chunk proposal 生成
- SDK 控制
- 数据采集
- inference
- Human-in-the-loop source arbitration

### 4.3 迁移前必须回答的问题

1. 下位机向上位机输出什么信号
2. 上位机由哪个节点持有唯一控制权
3. 网络中断时如何 fail-safe
4. teleop / inference / Human-in-the-loop 之间如何切换 ownership

---

## 5. 实施前的关键调研项

在正式改造前，需要先对下位机进行充分调研，至少包括：

1. backend 实际部署目录与启动方式
2. teleop 相关 Python 模块、依赖和运行进程
3. `1999` 端口上当前可用的 API / 流式接口
4. 网页 teleop 打开时的网络行为
5. 下位机本地 teleop loop 的频率、线程与 CPU 占用
6. teleop 信号从 BLE 到 backend 内部状态的更新时间语义
7. 是否已有可复用的原始信号输出接口

---

## 6. 当前建议

当前建议不是直接改代码迁移控制权，而是按下面顺序推进：

1. 先充分摸清下位机实际部署情况
2. 明确下位机当前到底能输出哪些 teleop 信号
3. 设计 shadow 验证链路
4. 在上位机完成 teleop 控制逻辑复现，但不发 SDK
5. 比对 shadow 与真实执行的偏差
6. 通过后再决定是否迁移真实控制权

---

## 7. 后续文档

后续至少补充两份文档：

1. 下位机 teleop 部署与运行现状调研报告
2. shadow 验证链路设计与实验记录

---

## 8. 第一轮实现落地

截至 `2026-04-10`，第一版已经按“信号上移，不做控制上移”完成落地：

1. 下位机 backend 新增了双通道接口：
   - `/api/joystick/teleop_target_v2`
   - `/api/joystick/events_v2`
2. 下位机仍保留真实 teleop 控制权，不改变本地 `50Hz` 控制闭环
3. 上位机新增独立 `teleop_shadow_ros.py` 节点：
   - 仅读取双通道接口
   - 读取本机 `qpos_end`
   - 基于 `processed.target_pose_abs` 生成学习层 `human_chunk_rel`
   - 输出 JSONL 日志与周期摘要
4. `record_data_ros` 未被改写，仍可与 `teleop_shadow` 并行运行

---

## 9. 第一轮现场验证结果

第一轮现场验证包含：

1. 真实下位机 backend 备份、同步与重启
2. 网页 teleop 实际操作联动
3. 上位机 `teleop_shadow` 并行观测

### 9.1 下位机双通道接口验证

真实操作期间已确认：

1. `processed.active` 能正确反映 teleop 的 active / inactive 切换
2. `processed.target_pose_abs` 在 active 时连续变化，在 inactive 时返回 `null`
3. `raw.button` 中的 `grip / trigger` 会同步反映手柄输入
4. `processed.sequence` 与 `raw.sequence` 同步递增，符合“同源快照”预期

### 9.2 上位机 shadow 验证

已确认：

1. `processed absolute target -> human_chunk_rel -> reconstructed absolute target` 变换闭环成立
2. 位置重建误差为数值误差量级，姿态重建误差为 `0`
3. 运行过程中 `exceptions=0`
4. `teleop_shadow` 不持有控制权，不向 SDK 发控制

### 9.3 网络与时序观察

第一轮有效结论：

1. HTTP 轮询延迟稳定，均值约 `6.2ms`
2. HTTP P95 约 `8.0ms`
3. 在真实 teleop 操作期间，接口与 shadow 节点都能稳定工作

### 9.4 对 SSE 统计的修正

第一版最初曾直接用上下位机各自系统时钟计算 SSE “延迟”，结果会出现极大的伪数值。  
原因不是链路异常，而是两台机器未做严格时钟同步。

现已修正为：

1. SSE 日志中不再把跨机器时钟差当作延迟
2. 周期摘要改为统计本地观测到的 `SSE arrival gap`
3. 这样能更真实地反映 SSE 到达节奏与抖动，而不依赖跨机器对时

### 9.5 当前阶段结论

第一轮现场验证支持以下判断：

1. 当前双通道接口设计是可用的
2. `processed` 作为第一阶段 shadow 主输入是成立的
3. 上位机可以稳定生成学习层 `human_chunk_rel`
4. 第一阶段无需迁移控制权，就已经能验证 signal uplift 的核心可行性

---

## 10. 下一步建议

当前更合适的推进顺序是：

1. 继续积累更多 teleop 现场样本
2. 将 shadow 日志与正式录制 session 做更明确的对齐
3. 在此基础上再进入 Human-in-the-loop inference 的 implementation 设计
