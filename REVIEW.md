1) 你当前推理框架的关键事实（决定不一致的根源）

从报告看，你现在的在线执行不是“执行最新 chunk 的前 act_horizon 步”，而是：

每次推理都会把 pred_horizon=16 全部塞进 Manager，而且 act_horizon 并没有用于截断 

chunk_timeline_analysis_report

控制侧在任意时刻会有 ~9 个 chunk 同时有效，并用 w_i = exp(-k Δt) 做时间加权融合 

chunk_timeline_analysis_report

相邻 chunk 的平均间隔 54ms（≈18.5Hz），但每个 chunk 覆盖 ~0.53s，因此平均重叠率 ~89% 

chunk_timeline_analysis_report

一个 chunk 的覆盖时间里平均会产生 9–10 个新 chunk，所以“后半段动作”几乎注定被后续 chunk 融合稀释 

chunk_timeline_analysis_report

这意味着：你的系统在稳态下并不是“16步 chunk”在工作，而是“每次只让新预测影响一点点，然后被旧预测+新预测共同平滑”。

2) 基于这些事实，train–infer mismatch 主要来自两块
不一致 A：动作语义不一致（最致命）

训练里（无论你是 diffusion/flow policy 的 RTC，还是常规 BC），数据通常对应的是：给定 obs_t，监督一个 action_t 或一段 future chunk（但“执行时”往往假设是最新 chunk 的前几步）。
而你在线执行的是：

action_send(t) = 多个 chunk 在同一时间 t 的动作加权平均

这会导致在线系统的动作等价于一个“时间卷积/低通滤波”后的结果，新 chunk 的即时纠错能力被稀释。报告里 overlap 高、active chunks≈9，就意味着“新预测的边际贡献很小”。 

chunk_timeline_analysis_report

训练时如果没把这个“滤波器”也加进去，模型学到的是“直接控制”，上线却变成“被滤波后的控制”——典型后果是：

学到的策略在仿真/离线评估 OK，上线变慢、拐弯变钝、纠错滞后

或者你为了补偿滞后，训练出来动作更激进，上线又被滤波压掉，产生奇怪的欠/过补偿

不一致 B：观测时延不一致（你已经量化出来了）

遥操作采集端 delta_action_obs 均值 ~41.8ms

推理端从观测到 chunk 生成 delta_chunk_obs 均值 ~90ms（约 3 帧@30Hz） 


这会让模型在推理时看到的“世界状态”比训练时更旧。你确实有 obs_horizon=2，能缓解一点，但本质上这是系统级延迟差异，会在高速/精细操作里被放大。 


3) 这种“多 chunk 融合”的方式常见吗？

**不算主流。**更常见的在线执行范式是下面两类：

Receding-horizon / MPC 风格（最常见）
每次推理生成一段未来动作序列，只执行前 act_horizon（常小于 pred_horizon），下一次推理重新规划；旧 chunk 直接丢弃或仅用于 warm-start，不参与融合。

单动作低通滤波（常见，但幅度很小）
如果要平滑，很多系统是在最终 action 上做轻量滤波（如 EMA：a = α a_new + (1-α) a_prev），而不是维护 9 个 chunk 做时间衰减融合。你现在这个更像“时间域 ensemble”，平滑很强、响应会明显变慢。

你的做法不是“错”，它是一种工程上的强平滑策略，但它的代价就是：控制语义变了，而这件事会让训练/部署一致性变得很难。

4) 推理框架要不要调整？我建议：要，而且方向很明确

我建议你把推理框架往“标准 receding-horizon”靠拢，原因非常直接：
RLFT 的 credit assignment 需要动作具有清晰归因。
你现在的动作是 9 个 chunk 混出来的，你很难说“这一步的 reward 应该 credit 给哪次推理输出”。

推荐调整方向（从“最小改动”到“结构性改动”）
方案 1：保留框架，但让“最新 chunk 主导”（最小补丁）

目标：让融合退化成“几乎只用最新 chunk”，同时保留你现有接口。

怎么做（原则级）：

提高 temporal_factor_k，让权重对旧 chunk 衰减更快（否则旧 chunk 贡献太大）。你现在 k=0.05 衰减很慢。


限制 active chunks 数量：只保留最近 M 个（如 2～3 个），其余直接剔除

把 act_horizon 真正用起来：只把前 act_horizon 步加入 chunk（或在 Manager 里只认为前 act_horizon 有效）


这会显著降低“动作语义漂移”，也会让 RLFT 更可做。

方案 2：标准 action chunking（我更推荐）

目标：让在线执行与训练数据的定义天然一致。

做法（原则级）：

推理输出 pred_horizon

只执行前 act_horizon 步（通常 < pred_horizon）

下一次推理到来时，直接切到新 chunk（可以做一个很轻的交接平滑，但别把 9 个 chunk 混在一起）

你报告里已经指出：现在 act_horizon=pred_horizon=16 且无截断，导致“独占执行步数≈1.7步”这个现象。把它改成标准行为，很多怪问题会自动消失。

