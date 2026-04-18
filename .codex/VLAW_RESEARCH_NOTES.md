# VLAW Research Notes

本文件把 `docs/vlaw/` 中值得长期保留的研究结论整理成一份工作笔记，避免每次重新翻大量归档报告。

重点收敛：
- ACP 当前共识
- retention / hold 失败的根因判断
- WM 的关键时间尺度与 autoregressive 经验
- ManiSkill 复现中最重要的实现警示

## 1. ACP 当前共识

根据 `ACP_INDEX.md`、v7 诊断链、retention/hold/archetype 报告，当前 ACP 方向的共识是：

1. 历史 shared drift 包括 `q_target_clip=20`
   - 它曾压低早期 PLD/DSRL sim baseline

2. 在 corrected `qclip0` 对照下：
   - PLD/DSRL 的 sim baseline 会恢复
   - 但 ACP mirror 仍然表现很差

3. 因此当前 ACP 的主问题已经不是“baseline reproduction drift”
   - 而是 ACP reward semantics / hold signal 本身不足

4. AWSC + ACP 不是这个诊断的反例
   - AWSC 依赖更强的 BC/flow anchor
   - 它不要求 ACP 从零“发明” hold 行为

## 2. ACP 模块定位

ACP 模块路径：
- `rlft/acp/`

目标：
- 用 per-frame 稠密 advantage 替代 VLM 的 per-trajectory 稀疏二元标注

核心思路：
- 训练 value model 预测每帧价值 `V(s_t) in [-1, 0]`
- 用实际 N-step 回报与预测值的差得到 advantage
- 高 advantage 帧在策略更新时获得更高权重

它可同时服务：
- offline weighted policy update
- online RL reward shaping

## 3. ACP 的重要工程结论

### 3.1 value target 不是 0/1 成功标签

ACP 训练标签虽然基于 `env_success`，但会被转换成连续 per-frame target：
- 范围 `[-1, 0]`
- 成功 / 失败之间保留稳定 gap
- 同一轨迹内随时间单调上升

这让 supervision 比单纯轨迹级标签更细。

### 3.2 当前问题不在“完全没有 dense signal”

更准确的表述是：

> ACP 现在不是没有信号，而是 dense signal 的语义偏了。

它更容易奖励：
- progress
- grasp proxy
- transient success

但不足以稳定地区分：
- stable hold
- imminent drop

## 4. retention / hold 的当前诊断

### 4.1 retention failure 是主失败模式

根据 retention 诊断：
- PLD / DSRL 的主失败模式是 retention failure，不是纯 progress failure
- 大量轨迹是 `SO=True, SAE=False`

关键指标：
- AWSC sim retention: 0.844
- AWSC acp retention: 0.778
- DSRL sim retention: 0.087
- DSRL acp retention: 0.152
- PLD sim retention: 0.050
- PLD acp retention: 0.073

结论：
- retention 问题不是 ACP 独有
- 但 ACP 也没有提供足够强的 hold-sensitive signal

### 4.2 corrected qclip0 的重要含义

在 corrected qclip0 对照中：
- DSRL sim: SO=0.94, SAE=0.66, retention=0.702
- DSRL acp: SO=0.94, SAE=0.06, retention=0.064
- PLD sim: SO=0.98, SAE=0.82, retention=0.837
- PLD acp: SO=0.82, SAE=0.02, retention=0.024

这非常关键，因为它说明：

> PLD / DSRL 本身不是“不会 hold”，在 sim reward 下它们能 hold。问题更像是 ACP 奖励把错误的行为强化了。

### 4.3 hold credit 错位的直接证据

在 archetype 诊断里，对比 stable 与 drop-after-success 的 post-success 窗口，有非常关键的现象：

DSRL + ACP：
- stable 的 post-success success 占比更高
- 但 stable 的 ACP total reward 反而更低
- drop 的累计 ACP reward 反而更高

这几乎是直接证据：

> 在成功后窗口里，ACP 没有把更多 credit 分给 stable hold，反而给了更容易掉落的轨迹更多 reward。

因此当前更精确的结论应是：

> ACP 失败的更深层原因，不是缺少 reward，而是 reward 的 credit emphasis 错了。

## 5. 训练内部健康的高价值结论

根据 v7 内科诊断，最有价值的长期观察是：

1. AWSC + sim 整体最健康
   - critic / actor / reward 都相对稳定

2. AWSC + acp 主要问题是 reward 几乎死掉
   - online/offline reward gap 可高到 1000x 量级
   - `acp_step_mean` 接近 0
   - advantage mean 偏高，说明 critic 难以区分好坏样本

3. DSRL / PLD + acp 在 aggregate 指标里并不总是显示 reward 崩掉
   - 但这不代表 reward 语义是对的
   - 问题更多体现在“强化的东西不对”

4. 常见处方仍然包括：
   - 提高 `online_ratio`
   - 提高 `acp_reward_scale`

但这些属于“把信号放大”的工程手段，不等于从根本修正 hold credit 语义。

## 6. WM 的关键研究结论

### 6.1 WM 的两个长期结构性问题

1. 动作空间桥接问题
   - policy space 与 world-model conditioning space 不同
   - 需要 adapter / FK / pose bridge

2. autoregressive compounding error
   - 长 rollouts 中图像逐步模糊
   - 这是视频扩散自回归的经典问题，不是一个本仓库独有的 bug

### 6.2 world model action space 的基本判断

当前最应记住的结构：
- policy 操作空间与 world model 条件空间不必相同
- 二者之间由 adapter / FK / pose sequence 连接

在 VLAW / Ctrl-World 语境里：
- policy 负责控制
- world model 负责视觉未来预测
- world model 更适合条件在几何空间而不是原始 low-level control space

这意味着：

> `policy space != world model space` 本身不是 bug，但 bridge 错了就会直接变成 imagination 的主故障。

### 6.3 autoregressive blur 的当前工程结论

即使完全对齐 Ctrl-World 官方实现，自回归 rollout 里的模糊趋势也不会彻底消失。

当前最有价值的控制手段是：
- 保留真实第一帧锚点
- 采用官方 history buffer 逻辑
- 控制 imagination horizon
- 用 VLM 做质量筛选

### 6.4 history buffer 的高优先级经验

官方有效做法：
- `history_idx = [0, 0, -12, -9, -6, -3]`
- 初始真实帧持续作为锚点参与条件生成

如果改成简单滑动窗口并让初始真实帧被挤出：
- 模糊和漂移会显著加速

因此：
- 第一帧锚定是必须保留的经验
- `num_history=6` 与稀疏采样应优先与官方对齐

## 7. 时间尺度与 imagination horizon

### 7.1 Ctrl-World / DROID 的时间尺度

Ctrl-World 原版是：
- DROID 原始 15Hz
- 下采样到 5Hz
- `pred_step=5` 对应约 1 秒

### 7.2 ManiSkill 复现的关键不匹配

当前 ManiSkill 复现里常见设定会导致：
- 工作帧率接近 6.67Hz，而不是 5Hz
- imagination 总时长明显长于真实任务时长

如果任务中位完成时间约 2.4 秒，却让 WM rollout 到 9 秒：
- 后半大部分帧都在描述“任务后期的无效状态”
- 这会拉低 VLM 判定并加剧 blur

因此一个重要经验是：

> imagination horizon 要匹配任务时间尺度，不要让 WM 长时间外推无意义尾部。

最小改动优先级通常是：
- 先缩短 `num_interact`
- 再视情况考虑帧率精确对齐

## 8. ManiSkill 复现里最值得反复提醒的点

1. 不要丢掉第一帧锚定
2. 不要把纯滑动窗口误当成与官方等价
3. 不要让 imagination 时长远超任务典型完成时长
4. 不要把 corrected qclip0 之前的 sim baseline 当作最终结论
5. 不要把 ACP 的 aggregate reward 健康误当作 hold credit 语义正确

## 9. 当前一句话总结

当前 VLAW 研究主线最值得记住的结论可以压缩成两句：

1. **ACP 的关键问题不是没有 dense reward，而是 dense reward 没有把 credit 正确给到 stable hold。**
2. **WM 的关键问题不是单一指标过不过线，而是 action bridge 与 autoregressive rollout 语义是否真正对齐任务时间尺度。**
