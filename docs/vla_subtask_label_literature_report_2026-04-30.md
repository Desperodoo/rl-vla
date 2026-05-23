# VLA 长程任务中 Subtask Label 的用法与效果综述

- 生成日期：`2026-04-30`
- 关注问题：同一个长程任务，微调数据/系统里引入 subtask label 与不引入 subtask label，具体会带来什么差别。
- 结论先行：公开文献里的共识并不是“subtask label 一定降低全局 action MSE”，而是：subtask label 通常被用来把长程任务拆成短程可控阶段，并显式支持当前阶段条件、阶段切换、失败恢复、阶段相关视觉输入选择。它的收益更常体现在 rollout success、完成子任务数、错序/重复/过早切换减少，而不是 episode 平均动作误差必然下降。

## 1. 共识用法

| 共识用法 | 训练/推理里怎么用 | 代表文献与位置 | 对应效果 |
| --- | --- | --- | --- |
| 当前 subtask instruction 作为 policy condition | 每个阶段把高层任务拆成当前子任务语言，policy 根据 `observation + current subtask` 出动作 | PALO Sec. 3.3：把任务 instruction 分解为 subtask sequence，执行时用 subtask instruction 而不是只用原始高层 instruction；PALO Appendix D/G 还区分 high-level 与 low-level language | 减少动作分布混淆；在同一任务中避免只按高层语义做平均轨迹 |
| 显式 subtask generation / planning | 模型先预测下一步 atomic subtask，再用该 subtask 作为 action context | LoHoVLA Sec. 3.1：形式化为 `π(a_t, ĝ_t | o_t, g)`，先生成 subtask 再预测 action；Sec. 3.4 用 text loss + action loss 训练 | 长程任务中比 vanilla VLA 更稳定，尤其减少“只靠隐式推断 subtask”的错误 |
| completion / transition label | 不只训练动作，还训练“当前 subtask 是否完成”的二分类/完成检测头，用它触发切换 | SeqVLA Sec. III.B/III.C：在 π0 上加 completion detection head；输出 action chunk 和 completion indicator，低于阈值时切到下一 subtask | 减少过早切换、过晚停留、重复已完成 subtask、错序执行 |
| phase / progress label | 把每个 subtask 进一步分成 movement / interaction phase，用 phase id 或 mask 指导模型关注不同输入 | Long-VLA Sec. 3.1/3.2：将 trajectory 分解为 moving 与 interaction phase，加入 phase identifier，并做 phase-aware input masking | 改善 skill chaining；边界处状态偏移更小，后续阶段初始状态更接近训练分布 |
| failure-aware replanning | 失败后不是盲目继续原动作，也不是每次都重规划，而是根据当前 subtask 失败次数决定 action retry 或 subtask replanning | LoHoVLA Sec. 3.3 / Sec. 4.3：比较 action-only retry、每次 replanning、分层闭环 replanning | 避免错误 subtask 继续执行造成 deadlock；同时减少不必要高层重规划 |

## 2. “具体表现”与文献对应

### 2.1 过早/过晚切换减少

具体表现：

- 还没完成 pick，就进入 place 的动作模式减少。
- 已经完成当前阶段后，不再继续执行冗余动作。
- 阶段边界附近动作更干净，少出现“抓取动作”和“放置动作”混合。

文献对应：

- SeqVLA Abstract / Introduction 明确把长程任务问题归因到 subtask completion 检测错误会级联到后续失败；还指出 vanilla VLA 会 premature switch、linger redundantly、propagate early-stage mistakes。
- SeqVLA Sec. III.B/III.C：completion head 预测当前 subtask 是否完成；检测到完成后停止当前 action chunk、回 home pose、切换到下一个 prompt。
- SeqVLA Fig. 7 / Sec. IV.C：比较 joint vs sequential finetuning 的 completion prediction，joint finetuning 的输出在执行期和完成期区分更清楚，说明边界判断更可靠。

对我们任务的可观测指标：

- 在 `pick_tape -> place_tape_in_cup` 边界前后 1-2 秒单独算 MAE。
- 人工/规则统计：tape 未稳定夹住时是否已经开始朝 cup 运动。
- 统计边界后是否仍输出明显抓取/抬升动作。

### 2.2 错序、重复、漏执行减少

具体表现：

- 长程任务中不再跳过某个子任务。
- 不再重复已经完成的子任务。
- 多个相似子任务连续出现时，顺序更稳定。

文献对应：

- SeqVLA Sec. IV.D 明确设置 baseline：baseline π0 用完整长程 demonstrations 微调，推理时没有 subtask monitoring；SeqVLA-J 用 subtask completion head 管理阶段。
- SeqVLA Fig. 8/9/10 与 Sec. IV.D：baseline π0 会重复 completed subtasks 或 wrong ordering；SeqVLA-J 消除了 sequence-related failures，失败主要来自真实 manipulation difficulty 而不是顺序错误。
- SeqVLA 的 candy packing 任务包含重复执行的子任务，正是为了检验 repeated subtask execution 的顺序稳定性。

对我们任务的可观测指标：

- 如果抓取失败，模型是否仍进入 place。
- 如果 place 失败，模型是否无意义地回到 pick 轨迹。
- rollout 中是否出现“重复抬起/重复寻找/重复 release”的阶段循环。

### 2.3 减少只看高层任务导致的动作模式混淆

具体表现：

- 无 subtask label 时，模型只看到总任务，可能在相似视觉状态下学到平均动作。
- 有 subtask label 时，同样的任务总体描述被拆成局部动作分布：pick 段输出抓取/抬升，place 段输出靠近 cup/release。

文献对应：

- LoHoVLA Sec. 4.1 的 baseline 设计很接近这个问题：Vanilla VLA 用同样数据、不带 sub-task labels 训练，直接从高层 goal 预测低层 action；LoHoVLA 显式预测 subtask 并用它指导 action。
- LoHoVLA Sec. 4.2：Vanilla VLA 在多个任务上最差，并出现若干 zero success；作者的定性解释是缺少 sub-task supervision 导致模型过拟合训练数据中的 frequent patterns，例如把 block 放到错误 bowl，忽略 goal condition。
- LoHoVLA Table 2：报告 long-horizon benchmark 上 average reward / success rate；LoHoVLA 在大多数 seen/unseen tasks 上高于 Vanilla VLA 与 LoHoRavens baseline。

对我们任务的可观测指标：

- 同一视觉阶段下，固定 prompt vs subtask prompt 的动作方向是否更分离。
- `pick_tape` 段的 gripper close / lift 行为是否更集中。
- `place_tape_in_cup` 段的 cup approaching / release 行为是否更集中。

### 2.4 边界后的状态偏移更小，错误传播减少

具体表现：

- 前一阶段留下的轻微位置误差不再显著拖垮下一阶段。
- 后续阶段对“非理想初始状态”更鲁棒。
- 长序列越长，subtask/phase 方法优势越明显。

文献对应：

- Long-VLA Introduction / Related Work 把 long-horizon VLA 的核心难点定义为 skill chaining：subtask 边界处动态耦合和 error propagation 会降低整体任务表现。
- Long-VLA Appendix A.2 / Fig. 9：对 CALVIN 做 Independent vs Continuous 对比；连续执行时，即使单个 subtask 本身能做，性能也会因前序状态偏移而下降。
- Long-VLA Sec. 4.4 / Table 3：加入 decomposition strategy 后，Real Sorting、Real Cleaning、Sim D-D 的平均完成长度都提高；作者解释为 decomposition 能缓解 imperfect prior executions 的负面影响。
- Long-VLA Appendix D.3 / Fig. 10：人为在前一阶段后加入目标位置、光照、干扰物扰动，base policy 成功率约下降 50%，Long-VLA 保持约 80% 成功率，说明 skill chaining 阶段更稳。

对我们任务的可观测指标：

- pick 完成后 tape 的高度/姿态略有偏差时，place 是否仍能找到 cup。
- 随机位置/光照 subset 上，subtask prompt 是否比 fixed prompt 更不容易在 place 段漂移。
- 边界后 2 秒内 wrist 中 cup 出现后的动作是否更稳定。

### 2.5 失败恢复更合理

具体表现：

- 失败来自 action 误差时，优先 retry 当前动作/当前 subtask。
- 失败来自 subtask planning 错误时，才重新规划 subtask。
- 避免“错误阶段继续执行”导致死锁。

文献对应：

- LoHoVLA Sec. 3.3 定义三类失败：sub-task planning error、action prediction error、external disturbance。
- LoHoVLA Sec. 4.3 / Table 3 比较三种策略：只重新预测 action、每次失败都重规划 subtask、超过阈值才重规划。只 action retry 在 planning 错误时会继续执行错误计划，可能 deadlock；分层闭环策略性能相近但减少不必要 high-level planning。
- PALO Sec. 3.3 同时优化 subtask sequence `c` 和时间 partition `u`，承认同一 subtask 序列在不同 episode 里持续时间会变；这说明“什么时候切”本身是任务的一部分。

对我们任务的可观测指标：

- 抓取失败后是否继续 pick/retry，而不是把空夹爪移动到 cup。
- cup 没进入 wrist 视角时是否继续搜索/调整，而不是直接 release。
- place release 失败后是否局部修正，而不是重新执行整段 pick。

### 2.6 高层/低层语言缺一不可

具体表现：

- 只有低层动作提示，缺少当前任务语义，容易 overshoot 或朝错误对象移动。
- 只有高层任务，缺少低层分解，长程动作不够稳定。
- 固定时间切分不如按演示/状态优化切分。

文献对应：

- PALO Sec. 4.3 / Fig. 6 / Table 3 做 ablation：No high-level instruction、No low-level instruction、Fixed Times、Zero-shot Decomposition、No VLM。除部分场景外，去掉 low-level、固定时间、无 VLM decomposition 都降低成功率。
- PALO Appendix G / Fig. 12：mask low-level instruction 后出现 spatial reasoning failure。
- PALO Appendix G / Fig. 13/14：mask high-level instruction 后，低层“move left/forward”等指令本身合理，但缺少 subtask context，导致 overshoot 或 grounding failure。

对我们任务的可观测指标：

- 只用 `Current subtask: Pick up...` vs “总任务 + Current subtask”比较，后者应更稳，因为总任务保留目标 cup。
- 固定 50/50 切分 vs rule-detector 边界切分比较，固定切分应更容易在个别 episode 中错位。

## 3. 更贴近 “w/wo subtask-label” 的证据排序

| 贴近程度 | 文献 | 为什么贴近 | 局限 |
| --- | --- | --- | --- |
| 高 | LoHoVLA Sec. 4.1/4.2 | 明确训练 Vanilla VLA on same dataset without sub-task labels；目标是 isolate explicit sub-task prediction | 仿真 Ravens 任务；subtask 是模型生成，不只是 frame prompt |
| 高 | SeqVLA Sec. IV.D | baseline π0 用完整长程 demos、无 subtask monitoring；SeqVLA 有 subtask completion head 和 prompt switching | SeqVLA 的训练数据含单独 subtask demonstrations，不是完全同一批 raw demos |
| 中高 | PALO Sec. 4.3 / Table 3 | 有 No VLM、No low-level、Fixed Times 等 ablation，直接说明 decomposition/partition 的作用 | 主要是 few-shot adaptation，不是 full fine-tune VLA |
| 中 | Long-VLA Sec. 4.4 / Table 3 | decomposition/phase/mask 的 ablation 很清楚，效果指标是长程 avg length/success | 重点是 phase-aware input masking，不是自然语言 subtask prompt |

## 4. 对当前 PI05 tape-to-cup 实验的含义

### 4.1 当前做法属于哪一类

我们现在的数据做法：

```text
Pick up the black tape roll and place it into the blue cup.
Current subtask: Pick up the black tape roll.

Pick up the black tape roll and place it into the blue cup.
Current subtask: Place the tape roll into the blue cup.
```

这主要对应“当前 subtask instruction 作为 policy condition”。它没有显式 completion head，也没有 phase id/mask。因此它最像 PALO/LoHoVLA 的“用 subtask language 约束动作分布”，但不像 SeqVLA 那样真正学习“什么时候切换”。

### 4.2 为什么 offline MAE 不一定立刻变好

文献里 subtask label 的主要收益常出现在 rollout-level 或 transition-level：

- success rate；
-完成子任务数；
-错序/重复/漏执行；
-subtask planning success；
-skill chaining robustness。

这些指标和全 episode mean_action_mae 不是一回事。对我们的两阶段任务，全局 MAE 可能被大量非边界帧稀释；同时如果边界标签有 0.5-1 秒噪声，prompt 切换反而会增加局部方差。

### 4.3 建议补充的评估

| 评估 | 目的 | 对应文献动机 |
| --- | --- | --- |
| transition-window MAE：边界前后 1-2 秒 | 看 subtask label 是否让动作模式切换更干净 | SeqVLA completion boundary；PALO partition `u` |
| failed-grasp-then-place 率 | 看抓取没完成时是否仍盲目 place | SeqVLA sequence-related failures |
| repeated/incorrect phase 率 | 看是否重复 pick、错序 place、过早 release | SeqVLA Fig. 8/9/10 |
| subset robustness：fixed/random/light 条件分开 | 看视觉扰动下 place 阶段是否更稳 | Long-VLA skill chaining / perturbation analysis |
| prompt ablation：总任务 vs 总任务+subtask vs subtask-only | 验证 high-level 与 low-level language 是否互补 | PALO No cH / No cL ablation |
| rule boundary vs fixed ratio boundary | 验证“什么时候切”是否重要 | PALO Fixed Times ablation；SeqVLA completion head |

## 5. 参考文献

1. SeqVLA: Sequential Vision-Language-Action Models for Long-Horizon Robotic Manipulation. arXiv:2509.14138. https://arxiv.org/abs/2509.14138
2. LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon Embodied Tasks. arXiv:2506.00411. https://arxiv.org/abs/2506.00411
3. PALO: Policy Adaptation via Language Optimization. OpenReview. https://openreview.net/forum?id=qUSa3F79am
4. Long-VLA: The First End-to-End Vision-Language-Action Model for Long-horizon Manipulation. arXiv:2508.19958. https://arxiv.org/abs/2508.19958
