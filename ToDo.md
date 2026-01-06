好，我们把 plan 直接改成你要的形态：策略推理≈10Hz、相机观测≈30Hz（下采样）、但机械臂控制/插值执行保持 500Hz；动作是 ee_delta_pose；且不做 training-time RTC（纯推理侧/系统侧 RTC）。

核心思想变成一句话：

用 10Hz 的 chunk 产生“稀疏关键帧”（8 个 30Hz 的 ee Δpose），再把它实时重定时/平滑成 500Hz 的连续 EE 参考轨迹（servo），并在推理回来时只覆盖未提交的未来段。

0) 重新定参数：把 30Hz chunk 映射到 500Hz 执行

你的 ACT：act_horizon=8，每步原始 dt≈1/30s
一次 chunk 覆盖：8/30 ≈ 0.2667s

500Hz 控制 dt：0.002s
对应每个 30Hz 步长（0.0333s）要展开成 0.0333/0.002 ≈ 16~17 个 500Hz 子步。

因此，我们把系统分成三层时钟：

Cam/Obs 时钟（≈30Hz）：生成 obs 帧（可能有抖动，以 timestamp 为准）

Policy 时钟（10Hz）：每 0.1s 生成一次未来 8 步的 ee_delta_pose@30Hz

Servo 时钟（500Hz）：每 0.002s 输出一个“平滑 EE 目标增量/目标位姿”

1) 你要实现的 RTC（不重训版）应该长这样
1.1 两段队列：Committed / Editable（和之前一样，但基于“时间”而非“步数”）

因为最终要 500Hz，最稳妥的做法是用绝对时间定义提交窗口：

commit_time = 0.10s（≈一次推理周期）

planning_horizon = 0.2667s（act_horizon=8 的覆盖时长）

每次推理回来，你只允许更新：

从 now+commit_time 到 now+planning_horizon 这段未来轨迹
而 [now, now+commit_time] 这段已经“承诺给控制器”不再动（避免抖动/跳变）。

这样你的系统天然适配 10Hz 推理延迟和 500Hz servo。

1.2 队列里存的不是“8 个动作”，而是“未来 0.2667s 的轨迹表示”

建议你存两种表示（方便实现）：

Keyframe 轨迹（30Hz）：8 个 ee_delta_pose（这是模型原生输出）

Dense 轨迹（500Hz）：把 keyframe 展开/插值后的 500Hz “参考”序列（或者一个可在线采样的 spline）

控制线程（500Hz）永远只需要一个函数：

x_ref(t_now)：给出此刻应该追踪的 EE 目标（或目标增量）

2) 系统结构：3 线程 + 共享内存（推荐）
Thread A：Servo 线程（500Hz，最高优先级）

职责：

读取机器人当前状态（关节、EEF pose/速度）

从 TrajectoryBuffer 获取 x_ref(t)（或 delta_ref(t)）

调用 arx5-sdk 的笛卡尔伺服/轨迹接口发送目标（最好让 SDK 内部 500Hz 控制器做跟踪）

关键点：

Servo 线程绝不能等推理、不能做重计算

任何时刻推理没更新，也能沿着“已承诺轨迹”继续平滑走

Thread B：Obs 线程（≈30Hz）

职责：

采相机帧 + 同步机器人状态（取最近的 state）

写入 ObsRingBuffer（带 timestamp）

Thread C：Policy 线程（10Hz）

职责：

从 ObsRingBuffer 取最近两帧 obs（obs_horizon=2）

运行 ACT，输出 8 个 ee_delta_pose@30Hz

调用 RTC Scheduler 更新 TrajectoryBuffer：

冻结 [now, now+commit_time]

覆盖更新 (now+commit_time, now+planning_horizon]

产出新的 keyframe + dense（或 spline）表示

3) 关键实现细节（这决定你能不能真的 500Hz 丝滑）
3.1 把 ee_delta_pose 变成“绝对 EE 参考轨迹”，而不是每次直接发 delta

如果你每 500Hz 都发 delta，很容易累计误差/出现噪声放大。更推荐：

Policy 输出：ΔT_i（i=1..8，基于 30Hz 的增量）

在 policy 更新时，把它们在 SE(3) 上积分成绝对关键帧：

T_{k+1} = T_k ⊕ ΔT_1

T_{k+2} = T_{k+1} ⊕ ΔT_2

...
得到 8 个绝对目标位姿关键帧 T_target(t0 + i*dt30)

Servo 线程只需追踪 T_target(t)（插值），不会因 delta 噪声导致“抖”。

这里的 ⊕ 用李群 SE(3) 组合（位置直接加，旋转用 axis-angle/quat 乘法）。

3.2 30Hz→500Hz 的插值：建议用“分段三次/最小 jerk”而不是线性

最小可行：

position：三次样条（保证速度连续）

rotation：slerp + 对角速度做限幅（或用 squad）

并在插值前后做约束：

限制每 2ms 的最大平移增量、最大角度增量（防止某一步突然很大）

3.3 “无重训版 RTC”如何避免接缝跳变（最容易抖的点）

每次推理回来更新未来轨迹时，会出现“新轨迹和旧轨迹在 commit_time 边界不连续”。

解决方法（工程上非常有效）：

边界匹配（C1 stitching）：强制新轨迹在 t = now+commit_time 处与旧轨迹的 pose/velocity 一致

pose：让新轨迹从旧轨迹边界 pose 开始

velocity：用插值的导数调整（或用 20~40ms 的 blend window 做平滑过渡）

短窗口 blending：在 [now+commit_time, now+commit_time+blend] 里，用

T = s(t)*T_new + (1-s(t))*T_old（位姿用 SE(3) 插值方式）
blend 取 20~50ms 通常就很稳

3.4 推理慢/卡顿怎么办（必须有 fallback）

如果 policy 线程超过 150ms 没产出新 chunk：

继续沿着旧轨迹走（最多走到 planning_horizon）

planning_horizon 不够就进入“安全维持”：保持当前 pose 或缓慢回到中立姿态

4) 在 arx5-sdk 上的落地建议（与你的需求对齐）

你要 500Hz 丝滑，本质上就是：把“高频伺服”交给 SDK/控制器，外部只负责提供连续参考。

实际落地有两种路线（选其一）：

路线 1（优先推荐）：使用 SDK 的 trajectory / waypoint scheduling 能力

你每次推理生成未来 0.266s 的关键帧轨迹

把它作为“短轨迹片段”提交给 SDK（并允许 trajectory update）

SDK 内部 500Hz 控制器去跟踪

优点：你自己的 500Hz 线程负担小、稳定性高。

路线 2：自己做 500Hz servo loop（只有在 SDK 不支持时用）

你自己 500Hz 算出每一步 EE 目标（或关节目标），每 2ms 调 SDK “send command”

这要求你的进程实时性很好（Linux RT / 高优先级），否则 500Hz 在用户态容易抖

如果你现在是在 Python 里做控制，我不建议直接 Python 500Hz；最好 C++/Rust 或者让 SDK 负责 500Hz。

5) 你可以直接照着写的模块划分（调整版）

obs/obs_ringbuffer.py：相机帧+机器人状态（timestamp 对齐）

policy/act_infer.py：10Hz 推理，输出 ΔT[8]

rtc/keyframe_integrator.py：ΔT -> T_keyframes

rtc/spline_builder.py：T_keyframes -> T_ref(t)（可采样）

rtc/scheduler.py：

commit_time=0.10

planning_horizon=0.2667

blend_window=0.03（可调）

control/servo_500hz.cpp（或用 SDK trajectory 接口替代）：

只做 T_ref(t_now) 采样并发送

6) 一套你可以直接用的初始超参数（很“工程正确”）

policy_rate = 10Hz

obs_rate = 30Hz（按相机实际 timestamp）

servo_rate = 500Hz

commit_time = 0.10s

planning_horizon = 0.2667s

blend_window = 0.03s

pos_step_limit = 1~2 mm / 2ms（视机械臂性能）

rot_step_limit = 0.1~0.2 deg / 2ms（先保守）

7) 我需要你补一条信息，我才能把“ARX5 侧接口”写到可直接照抄

你说动作是 ee_delta_pose，但我还不知道你在 arx5-sdk 里现在走的是哪条控制链路：

A) 你是直接发 Cartesian pose 目标？

B) 还是发 Cartesian twist/velocity？

C) 还是你先 IK 成 joint 目标再发？

你不用解释太多，只要告诉我你现在用 SDK 的哪个示例/函数名（或者贴 20 行调用片段）。
我就能把上面路线里“servo 线程到底每 2ms 调哪个 API、数据结构长什么样”写成非常具体的执行步骤（包括时间戳、队列格式、如何做 SE(3) 积分与插值）。