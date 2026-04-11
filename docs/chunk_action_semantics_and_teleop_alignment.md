# Chunk Action 语义与 Teleop 对齐设计笔记

**日期**: 2026-04-10
**状态**: 讨论收敛稿，已加入 Human-in-the-loop 第一轮方案
**范围**: 真机遥操作采集、模仿学习训练、真机部署推理、推理数据回流、后续真机 RL 微调

---

## 1. 背景

当前仓库覆盖了一条完整的真机闭环：

1. 遥操作真机数据采集
2. 真机 policy 模仿学习训练
3. 真机 policy 部署推理
4. 真机推理数据回流
5. 真机 RL 微调 policy

这条链路里，最容易产生语义漂移的部分是 `action`。如果 action 在采集、训练、推理、回流几个阶段含义不一致，那么后续的模仿学习和 RL 都会建立在不稳定的目标之上。

此前代码与讨论里已经确认过几个关键事实：

- 推理侧当前默认执行模式是 `receding_horizon`，而不是旧的 `temporal_ensemble`。
- 每次推理会输出一个 action chunk，但控制线程并不会“原样完整执行整个 chunk”。
- chunk 进入 `ActionChunkManager` 后，还会经历时间戳调度、chunk 覆盖与控制线程查询。
- 控制线程最终以更高频率从 manager 中取出单步动作，再下发到 SDK。

因此，系统里天然存在多层 action 语义，不能再把“模型刚输出的东西”和“最后发给 SDK 的东西”混为一谈。

---

## 2. 已确认的推理链路语义

当前 `inference_ros.py` 的核心时序可以概括为：

1. policy 在推理线程输出一个 chunk
2. chunk 经过安全检查与可选干预
3. chunk 被赋予时间戳并写入 `ActionChunkManager`
4. 控制线程按 `control_freq` 查询 manager
5. manager 在 `receding_horizon` 语义下返回“当前时刻最新有效 chunk 对应的待执行动作”
6. 该动作最终下发给 SDK

对应地，当前系统至少存在三层 action：

### 2.1 第一层：policy action

记为 `a_policy_chunk`

含义是某次推理时 policy 输出的整段 chunk。  
这是最接近“agent/policy 真正输出”的 action 语义。

### 2.2 第二层：scheduled action

记为 `a_sched_t`

含义是控制线程在某个控制时刻，从 `ActionChunkManager` 中按时间戳规则取出的单步动作。  
它已经包含了：

- chunk 时间戳调度
- `receding_horizon` 下“最新 chunk 覆盖旧 chunk”
- 推理频率与控制频率不一致带来的采样效应

### 2.3 第三层：executed action

记为 `a_exec_t`

含义是最终发给 SDK 的单步动作。  
如果后续还存在安全裁剪、人工干预、限幅、低层屏蔽等，这一层还会继续偏离 `a_sched_t`。

---

## 3. 讨论中形成的核心判断

### 3.1 MDP 里的 action 不应直接偷换成 executed action

讨论中一个关键分歧是：RL/回流数据里的 action 是否应该直接定义为 `a_exec_t`。

当前结论是否定的。

原因是：

- 从 RL/MDP 定义上看，action 更应是 agent/policy 的输出，而不是经过执行通道后的结果。
- 从 `a_policy_chunk` 到 `a_sched_t / a_exec_t` 的环节，应视为环境或执行系统的一部分。
- 如果把 `a_exec_t` 直接当成“policy action”，会模糊 agent 与 environment 的边界。

因此，理论上的学习动作仍应优先定义为 `a_policy_chunk`。

### 3.2 但推理和回流阶段仍必须记录三层 action

虽然学习动作不应直接改写成 `a_exec_t`，但工程上仍必须记录：

- `a_policy_chunk`
- `a_sched_t`
- `a_exec_t`

原因是：

- 真机状态转移真正受到的是后两层 action 的影响
- 只有同时记录三层 action，才能量化 train-deploy gap
- 后续无论做误差分析、回流清洗，还是 Human-in-the-loop / RL 设计，都需要这三层 provenance

---

## 4. 当前收敛下来的 action 定义

### 4.1 统一 action 语义

当前收敛后的主张是：

> 真正的学习动作定义为 chunk-level policy action，也就是“一次推理输出的整个 chunk”。

这一定义将贯穿：

- 遥操作数据采集
- 模仿学习训练
- 真机 policy 推理
- 真机推理数据回流
- 后续真机 RL 微调

这样做的主要好处是：

- 与当前 policy 的输出形式一致
- 避免把 chunk policy 人为降格成单步控制 policy
- 整个 pipeline 的 action 语义保持一致

### 4.2 部署通道的定位

在这一定义下，应把以下链路视为执行系统的一部分：

`a_policy_chunk -> a_sched_t -> a_exec_t`

也就是说：

- policy 输出的是 chunk
- manager / scheduler / delay / control loop 决定 chunk 如何在控制时钟上落地
- 真机真正执行的是最终的单步动作流

因此，部署通道不属于“action 定义本身”，但必须尽量简单、稳定、可观测、可记录。

---

## 5. 对已有方案的判断

### 5.1 为什么倾向保留 chunk 作为 action

当前 policy 本来就是输出 action chunk。  
因此，继续把学习动作定义为 chunk，而不是单步动作，是自然且自洽的选择。

这也意味着后续优化重点应放在：

- 清理旧的 `temporal_ensemble` 语义
- 把 `chunk -> scheduled -> executed` 的部署通道做得更干净
- 降低推理耗时，减小异步时序带来的额外 gap

### 5.2 为什么不接受“只在训练时显式建 action channel 就够了”

一个被讨论并否定的方案是：

> 在训练/建模时显式加入 `policy output -> scheduled/executed action` 的已知模块，并在训练时 unroll 进去。

当前对它的判断是：**仅靠训练时补建模，不足以修复采集与部署之间的闭环错位。**

原因是：

- 如果采集数据时没有经历与部署一致的 action channel
- 那么状态转移在数据生成阶段就已经固定
- 训练时再建模，只能在输出后补一个变换，无法反向改变数据对应的真实闭环

所以问题的根源主要在数据生成分布，而不只是 learner 的结构。

### 5.3 “输出短 horizon chunk”不是新的替代方案

另一个被反驳的方案是：

> 直接把 policy 的 action space 改成更接近部署输入，比如输出短 horizon chunk，而不是单步控制。

当前不采用它的原因很直接：

- 当前 policy 已经是在输出 action chunk
- 这并不是新的替代方向，而是当前系统本来就在做的事情

真正需要解决的不是“要不要 chunk”，而是：

- chunk 的语义如何定义
- teleop 如何产生同语义的 chunk
- deployment channel 如何尽量对齐

---

## 6. Teleop 侧的收敛方案

为了让“`teleop action = chunk`”真正成立，当前收敛后的 teleop 方案是：

> 在观测时刻 `t_k`，让 teleop 在线跑一个与部署一致的 scheduler / buffer，把人当前输入注入进去，实时形成一个 chunk proposal。

这意味着：

- 人在 teleop 里不再被视为高频单步控制器，之后再离线重组为 chunk
- 而是尽量被视为“在线 chunk policy”
- 也就是“让人真的扮演 deployment-time policy”

### 6.1 为什么更倾向这个方案

相比“先采单步控制，再事后回填成 chunk”，这个方案更符合因果性：

- 每个 chunk proposal 都是在当时观测下在线形成
- 不会把未来已看到的新观测偷偷打包回过去的 action
- 更贴近真实部署时 policy 的决策形式

### 6.2 Teleop 侧还需要模拟部署延迟

仅仅让 teleop 侧也输出 chunk 还不够。

为了进一步缩小 train-deploy gap，还需要在 teleop 侧加入适度的部署通道模拟，包括但不限于：

- 推理延迟的等效模拟
- scheduler / buffer 的一致化
- 控制频率与推理频率不一致时的时间调度语义

目的不是机械地“制造噪声”，而是让 teleop 数据生成时的闭环，更接近真实部署闭环。

---

## 7. 当前结论

经过本轮讨论，当前可以收敛为以下结论。

### 7.1 action 语义结论

1. 真正的学习动作是 chunk-level policy action。
2. 在当前系统里，这个 action 定义为“一次推理输出的整个 chunk”。
3. `chunk -> scheduled -> executed` 是部署通道的一部分，而不是 action 定义本身。

### 7.2 数据记录结论

推理与回流阶段应同时记录三层 action：

1. `a_policy_chunk`
2. `a_sched_t`
3. `a_exec_t`

这样才能：

- 量化 train-deploy gap
- 解释真机状态转移
- 为后续 RL 和 Human-in-the-loop 提供足够上下文

### 7.3 Teleop 对齐结论

为了减小 train-deploy gap，teleop 采集时要尽量复现部署通道：

1. teleop 侧在线运行与部署一致的 scheduler / buffer
2. 将人当前输入实时注入，形成 chunk proposal
3. 必要时加入模拟推理延迟

这样才能更合理地说：

> 遥操作阶段的人类，扮演的是 deployment-time 的 chunk policy，而不是一个事后再被重组的高频单步控制器。

---

## 8. 下一步议题

在上述共识基础上，下一步进入 Human-in-the-loop 讨论时，重点不再是“action 到底定义成什么”，而是：

1. 人为干预应该介入哪一层 action
2. 如何在不破坏当前 chunk 语义的前提下，实现更自然的 shared autonomy
3. 如何把 Human-in-the-loop 数据组织成后续 RL 可用的回流样本

---

## 9. Human-in-the-loop 第一轮收敛

在继续讨论 implementation 之前，当前已经就 Human-in-the-loop 真机部署推理形成了第一轮较清晰的收敛。

### 9.1 当前问题背景

`inference_ros.py` 里现有的键盘 intervention 方案，只能视为临时调试机制，不应作为后续 Human-in-the-loop 的正式形态。

它的主要局限包括：

- 输入设备语义不对，键盘并不是实际 teleop 使用的手柄
- 介入层级不理想，它是在推理线程里直接改写 chunk 内容
- 它没有和当前已经确定的 chunk-level action 语义严格对齐

因此，后续正式的 Human-in-the-loop 方案，将不再以键盘 intervention 为基础，而是切换到基于下位机 teleop 信号的干预方式。

### 9.2 Human signal 的语义定位

当前 backend 可提供 teleop 侧的实时信号，例如：

- `target_pose`
- `gripper_pose`
- `scale`
- `active`

这些信号本身更接近“当前时刻人类 teleop 控制器想发给 SDK 的单步绝对目标”，而不是一个已经成形的 chunk action。

因此，Human-in-the-loop 的关键并不是把这类单步信号直接和 policy chunk 数值融合，而是先把 human signal 提升到与 policy 同层的 chunk proposal 语义。

### 9.3 human signal -> human chunk proposal

关于“第二层：怎么把 human signal 提升成 chunk proposal”，当前已明确选择：

> 完全复用 teleop 采集阶段那套“在线 scheduler / buffer -> chunk proposal”机制。

也就是说：

- human teleop 输入不会直接作为最终执行动作插入 control loop
- inference 侧会维护一个与 teleop 采集时一致的 human chunk 生成过程
- 人当前的 teleop 输入先进入这套在线机制
- 再形成一个与 policy 输出同语义的 `human_chunk_proposal`

这样做的目的，是让 Human-in-the-loop 继续遵循前文已经确定的 action 定义：

- 真正的学习动作仍然是 chunk-level action
- human 侧也在 chunk 层表达自己的意图
- 部署通道仍然保持统一

### 9.4 policy chunk 与 human chunk 的协调方式

关于“第三层：policy chunk 和 human chunk 如何协调”，当前已明确选择：

> `Authority gating`

其含义是：

- policy 仍持续推理，不停机
- 只有当 `human.active == False` 时，policy chunk 才进入 active manager
- 当 `human.active == True` 时，切换到 human chunk source
- 暂时不考虑 teleop action 与 policy action 的数值融合

这本质上仍然是一种 takeover，只是 policy 后台持续运行，方便后续 handoff。

### 9.5 为什么当前不考虑 human/policy 融合

当前阶段不考虑把 teleop action 和 policy action 做数值 blend，主要原因是：

- teleop 原始输入天然更像单步控制信号，而 policy 输出是整段 chunk
- 两者在时间基准、horizon、延迟和控制语义上都不天然一致
- 过早做 blend，容易重新打乱刚刚建立起来的 chunk-level action 定义

因此，第一阶段更强调：

- source-level arbitration，而不是 value-level blending
- 先把 human chunk 和 policy chunk 的层级对齐
- 再通过 gating 选择当前由谁拥有控制权

### 9.6 当前 Human-in-the-loop 主线

到这一轮讨论为止，Human-in-the-loop 的主线可以概括为：

1. human teleop 信号先通过在线 scheduler / buffer 机制提升成 `human_chunk_proposal`
2. policy 持续生成 `policy_chunk_proposal`
3. 系统在 source 层执行 `Authority gating`
4. 当前被选中的 chunk source 再进入统一的部署通道
5. 调度后继续产生 `scheduled action` 和 `executed action`

因此，当前 Human-in-the-loop 的重点已经不再是：

> teleop 信号怎么和 policy action 直接数值融合

而是：

> 如何把 teleop 单步意图提升为 `human_chunk_proposal`，并在 source 层与 `policy_chunk_proposal` 做稳定仲裁

### 9.7 后续 implementation 讨论的入口

基于当前收敛，后续 implementation 相关讨论可以集中在以下几个问题上：

1. `human_chunk_generator` 的输入、状态与输出格式如何定义
2. `Authority gating` 的切换条件、滞回与 handoff 机制如何设计
3. `inference_ros.py` 中 human chunk 与 policy chunk 的接入点应该放在哪一层
4. Human-in-the-loop 场景下需要新增哪些日志字段与回流字段
