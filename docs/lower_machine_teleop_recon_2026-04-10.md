# 下位机 Teleop 第一轮摸底记录

**日期**: 2026-04-10
**对象**: 下位机 `10.42.0.101`
**账号**: `cvte`
**范围**: backend 部署情况、teleop 运行链路、对外接口、初步网络与时序观察

---

## 1. 基本环境

通过 SSH 登录下位机后确认：

- 主机名：`cvte`
- 用户目录：`/home/cvte`
- 架构：`aarch64`
- 内核：`Linux 6.1.99-rt36`
- 内核特性：`PREEMPT_RT`

这说明下位机具备较好的实时控制基础，适合运行当前本地 teleop 闭环。

---

## 2. backend 实际部署位置

backend 实际部署在：

`/var/www/backend/carm_backend`

目录结构第一轮确认结果：

- `auto_start.sh`
- `config/`
- `dependencies/`
- `server/`

teleop 相关关键文件位于：

- `server/app.py`
- `server/api/joystick.py`
- `server/basic/joystick.py`

---

## 3. backend 启动方式

当前 backend 由下位机本地脚本拉起：

`/var/www/backend/carm_backend/auto_start.sh`

第一轮确认到的启动流程：

1. 设置 `PYTHONPATH`
2. 寻找并 `source` SDK 的 `setup.bash`
3. 进入 `/var/www/backend/carm_backend`
4. 启动：

`python3 server/app.py --project ARM`

这说明 teleop backend 运行依赖：

- 本地 Python 环境
- `/var/www/backend/carm_backend/dependencies`
- 本地 arm SDK 环境变量与动态库配置

---

## 4. 当前运行中的相关进程

第一轮观察到的关键进程：

- `/usr/bin/bash /var/www/backend/carm_backend/auto_start.sh`
- `python3 server/app.py --project ARM`
- `/usr/bin/python3 /opt/ros/foxy/bin/ros2 run arm_control arm_control_exe`

这表明：

- backend 已经在下位机常驻运行
- teleop 与 arm control 相关链路都在下位机侧就绪

---

## 5. 当前开放端口

第一轮看到的关键监听端口：

- `80`
- `1999`
- `8090`
- `22`

其中比较关键的是：

- `80`：网页服务入口
- `1999`：backend API
- `8090`：疑似机械臂 SDK / 控制相关端口

---

## 6. teleop 当前控制链路

当前 teleop 真正控制机械臂的核心逻辑在：

`server/basic/joystick.py`

第一轮确认到的关键事实：

1. `CommandController.compute_control()` 会根据手柄姿态与按钮状态，计算 `target_end_arm_pose`
2. `CommandController.loop()` 以 `50Hz` 运行
3. 在该 loop 内，如果 `target_end_arm_pose` 非空，就直接调用：
   - `track_pose(target[:7])`
   - `set_gripper(gripper_pose, tau)`

这意味着当前系统是：

> 下位机本地 teleop 控制律 + 下位机本地 SDK 下发

上位机目前只是旁路观察者，而不是 teleop 控制 owner。

---

## 7. teleop 对外接口现状

### 7.1 `/api/joystick/status`

当前可返回：

- `ble_connected`
- `arm_initialized`
- `calibration`
- `scale`
- `status`
- `battery_level`
- `rssi`

在本轮摸底时，该接口返回：

- `ble_connected=true`
- `arm_initialized=true`
- `status=connected`

说明 teleop backend 处于正常待命状态。

### 7.2 `/api/joystick/teleop_target`

当前可返回：

- `target_pose`
- `gripper_pose`
- `gripper_tau`
- `scale`
- `active`

本轮确认它的语义是：

> 已经过下位机 teleop 控制律处理后的绝对目标位姿

也就是说它不是 raw 手柄信号，也不是学习语义下的 `ee_delta_pose`。

### 7.3 `/api/joystick/events`

当前 backend 还暴露了 SSE 接口：

`/api/joystick/events`

它会推送：

- `pose`
- `button`
- `gripper_pose`
- `gripper_tau`
- `status`
- `device`
- `timestamp`
- `sequence`

这个接口比 `teleop_target` 更接近 raw teleop 信号层。

---

## 8. 当前接口的频率观察

### 8.1 `teleop_target`

在网页端主动 teleop 期间，从上位机无代理访问 `teleop_target`，测得：

- 平均 HTTP 往返：约 `6ms`
- P95：约 `7.9ms`
- 最大值：约 `8.6ms`

这说明：

- 下位机到上位机的局域网网络条件很好
- 以 `teleop_target` 为基础做 shadow 验证是可行的

### 8.2 `events` SSE

通过实际抓流确认：

- `events` 服务端显式 `sleep(0.05)`
- 因此对外推送频率约 `20Hz`

同时观察到：

- `sequence` 每次跳变约 `6-9`

这说明：

- backend 内部手柄状态更新频率高于对外 SSE 推送频率
- 当前 SSE 并不是全频原始信号出口，而是一个节流后的观察接口

---

## 9. 当前语义上的关键判断

第一轮摸底后，可以明确区分三类信号：

### 9.1 raw teleop signal

更接近：

- 手柄 `pose`
- 手柄 `button`
- clutch / trigger / button 状态

这类信号更适合未来如果要把 teleop 控制律真正搬到上位机时使用。

### 9.2 processed teleop target

当前对应：

- `/api/joystick/teleop_target`

它的语义是：

- 已经过下位机 teleop 控制律处理
- 已积分成绝对末端目标
- 可直接送 SDK 的目标位姿

### 9.3 当前上位机训练语义

虽然采集时 `record_data_ros` 把 `teleop_target` 直接存成 8D action：

- `[target_pose(7), gripper(1)]`

但训练阶段会再把它相对化成：

- `relative_pose = current_pose^{-1} @ target_pose`

所以：

- 存盘 action 是 absolute target
- 模型真正学习的是 relative action

---

## 10. 对后续改造的影响

第一轮摸底对后续方案的启发如下。

### 10.1 为什么 shadow 验证先做是合理的

因为当前网络层面已经证明：

- 下位机到上位机延迟较低
- `teleop_target` 可稳定读取

因此先做：

> 信号上移，不做控制上移

是低风险且高信息量的路径。

### 10.2 为什么只拿 `processed` 也可以先启动方案

如果当前目标是：

- 构造学习层的 human chunk action
- 验证上位机能否稳定重建并记录 human chunk

那么现阶段直接用 `processed teleop target` 作为 shadow 输入是可行的。

### 10.3 为什么 `raw` 仍然值得保留

虽然第一阶段可以只靠 `processed` 启动方案，但 `raw` 仍然值得在新接口中保留，因为它有助于：

- 以后复现或替换 teleop 控制律
- 对比 raw 与 processed 的控制语义差异
- 分析 clutch / trigger / button 等人机交互细节
- 在后续需要更强统一性时，提供升级空间

---

## 11. 当前建议

第一轮摸底后的建议是：

1. 下位机新增 `processed + raw` 双通道接口
2. 第一阶段 shadow 验证以上位机订阅 `processed` 为主
3. 同时保留 `raw`，作为后续升级与分析接口
4. 上位机先新建独立 `teleop_shadow` 节点，不直接接管控制权

---

## 12. 后续待补充

下一轮需要继续补充：

1. 下位机新接口的具体返回结构
2. 上位机 `teleop_shadow` 节点的输入输出与日志字段
3. `human_chunk_abs` / `human_chunk_rel` 的生成流程
4. 是否需要额外的高频 raw 信号出口，而不仅是当前 20Hz SSE

---

## 13. 第一轮改造后的补充确认

在第一轮摸底之后，已经完成了第一版实现与现场联动，新增确认如下。

### 13.1 下位机新接口已上线

真实下位机 `10.42.0.101` 已新增：

1. `/api/joystick/teleop_target_v2`
2. `/api/joystick/events_v2`

并确认：

1. 旧接口仍兼容可用
2. backend 已在真实设备上完成备份后再同步
3. 同步后服务已重启并正常对外提供新接口

### 13.2 双通道 live 行为确认

在网页 teleop 实际操作期间，已确认：

1. `processed.active` 会正确从 `False` 切换到 `True`
2. `processed.target_pose_abs` 会随操作持续变化
3. `processed.gripper_pose` 会反映夹爪状态变化
4. `raw.button` 中的 `grip / trigger` 与实际操作同步
5. `processed.sequence` 与 `raw.sequence` 一致递增

这说明新增接口不是静态可访问而已，而是已经通过 live 操作验证。

### 13.3 shadow 节点验证结论

上位机 `teleop_shadow` 节点已成功运行，并确认：

1. 可稳定读取双通道接口
2. 可读取本机 `qpos_end`
3. 可将 `processed.target_pose_abs` 转换为学习层 `human_chunk_rel`
4. 再反解回绝对目标时，位置误差仅为数值误差量级，姿态误差为 `0`

### 13.4 对 SSE 统计口径的补充

第一版最初将 SSE 到达时刻与下位机 `server_timestamp` 直接做差，得到过大的伪延迟值。  
原因是上下位机系统时钟未严格同步。

现阶段更合理的 SSE 指标口径是：

1. 记录本地接收顺序与事件数量
2. 统计本地 `arrival gap`
3. 将跨机器 `server_timestamp` 仅作为辅助时间信息保留

因此，后续若再看 shadow 摘要，应优先参考：

1. `http_mean / http_p95`
2. `sse_events`
3. `sse_gap_p95`
4. reconstruction 误差
