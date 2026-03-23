# ACP v7 深层失效分析：为什么 DSRL / PLD + ACP 失效，而 AWSC + ACP 仍然有效

**日期**：2026-03-22
**依据**：
- 正式诊断报告：`docs/vlaw/figures/rlpd_acp_v7_diag/diagnosis_report.md`
- 诊断汇总：`docs/vlaw/figures/rlpd_acp_v7_diag/diagnosis_summary.json`
- ACP wrapper：`rlft/envs/acp_reward_wrapper.py`
- PLD 训练入口：`rlft/online/train_pld.py`
- DSRL 训练入口：`rlft/online/train_dsrl.py`
- PLD 算法实现：`rlft/algorithms/online_rl/pld_sac.py`
- DSRL 算法实现：`rlft/algorithms/online_rl/dsrl_sac.py`
- 历史背景：`docs/vlaw/acp_v6_rlpd_report.md`

---

## 1. 本轮正式诊断真正回答了什么

v7 的 6-cell controlled comparison 给出的最重要信息，不是“ACP 会不会把算法弄坏”，而是：

> **在当前这套 matched recipe 下，PLD / DSRL 的主要失败模式并不是 ACP 特有的失败，而是它们本来就学不会稳定的 hold / retain 行为；ACP 只是没有提供足够强的机制去修复这个问题。**

诊断报告里的对照结论非常清楚：

| Algo | Reward | Best SO | Best SAE | Final SAE | SAE Retention | SO-SAE Gap |
|---|---:|---:|---:|---:|---:|---:|
| AWSC | acp | 0.900 | 0.700 | 0.520 | 0.778 | 0.200 |
| AWSC | sim | 0.900 | 0.760 | 0.740 | 0.844 | 0.140 |
| DSRL | acp | 0.920 | 0.140 | 0.020 | 0.152 | 0.780 |
| DSRL | sim | 0.920 | 0.080 | 0.000 | 0.087 | 0.840 |
| PLD | acp | 0.820 | 0.060 | 0.000 | 0.073 | 0.760 |
| PLD | sim | 0.800 | 0.040 | 0.000 | 0.050 | 0.760 |

见正式报告 `Controlled Comparison Summary`。

这张表已经足够说明：

1. **AWSC 在 sim / acp 下都能工作**，只是 ACP 版本退化一些。
2. **PLD / DSRL 在 sim 下也并不健康**。
3. 因此，“PLD / DSRL + ACP 失效”的更深层说法不是“ACP 单独害死了它们”，而是：
   - **它们缺少能把 “抓到 → 保持 → 到 episode 结束仍 upright” 这一行为链稳定下来的学习机制**；
   - ACP 的奖励形状又刚好**不能主动补上这个缺口**。

---

## 2. 从算法原理看，三类方法本质上在学什么

### 2.1 AWSC：不是“从零找动作”，而是在 BC 先验上做 advantage-weighted 微调

AWSC/RLPD 的本质不是纯 SAC，而是：
- 有一个已经学好的 ShortCut Flow policy 作为强行为先验；
- actor 的训练目标里保留了 flow matching / shortcut consistency；
- critic 的作用更多是**重加权**，而不是完全决定策略朝哪个动作族坍缩。

因此 AWSC 的关键性质是：

> **即使 reward 不完美，actor 也不会轻易偏离 demo 中已经包含的“抓住并保持”的动作结构。**

这解释了为什么 AWSC + ACP 即便 reward 很弱，仍然能保住较高 SAE。

### 2.2 PLD：在 base policy 之上学 residual，但 critic 只看 residual 动作

PLD 的实现里，actor 输出的是残差动作 `a_delta`，critic 学的是 `Q(s, a_delta)`，而不是 `Q(s, a_full)`：
- 训练定义见 `rlft/algorithms/online_rl/pld_sac.py`
- 注释明确写到 critic 评估的是 residual action space，而非组合后的 full action。

这意味着 PLD 的学习问题不是“直接学任务动作”，而是：

> **在 frozen base policy 的动作附近，学一个小修正，让轨迹更好。**

这类方法在 reward 足够明确、且局部修正足以完成目标时很有效；但对 `LiftPegUpright` 这种任务，
“hold upright until the end” 往往不是一个单步局部修正问题，而是一个**时序稳定性问题**。

也就是说，PLD 更擅长：
- 修正 reach/grasp 的局部偏差；
- 改善“接近成功”的短时行为；

但不擅长：
- 在长时间 horizon 上维持夹持与姿态稳定；
- 在 reward plateau 区域继续保持正确微操控。

### 2.3 DSRL：在 noise space 里做 SAC，动作语义比 PLD 更间接

DSRL 的 actor 并不直接输出 env action，而是输出 noise / latent，然后再通过 flow policy decode 成实际动作：
- `train_dsrl.py` 中明确写了 “train standard SAC agent in the noise space”
- `dsrl_sac.py` 里 actor/critic 直接建模的是 `Q(s, noise)`

这带来一个关键差别：

> **DSRL 的 policy gradient 优化的是“哪种 noise 经由 base policy 解码后更有价值”，而不是直接优化“哪组物理动作更利于稳定 holding”。**

优点是动作空间更光滑、更受先验约束。
缺点是如果 reward 对“hold”不敏感，那么 actor 学到的 latent 改动仍然会偏向“更容易拿到阶段性正反馈”的方向，而不是“更容易维持末态稳定”的方向。

DSRL 比 PLD 稍强，是因为：
- 它的探索更健康；
- latent space 比 residual action 更连续；
- 长训练时确实有机会把“抓住后继续稳住”的行为慢慢学出来。

但在没有强 hold-aware signal 的情况下，它还是容易停在：
- success_once 很高；
- success_at_end 很低。

---

## 3. ACP reward 的原理，决定了它天然更擅长“进展”，而不是“保持”

ACP wrapper 的核心定义在 `rlft/envs/acp_reward_wrapper.py`：

- TD shaping：`r = (V(s') - V(s)) * scale`
- potential shaping：`r = V(s') * scale`

这里最关键的是 TD reward 的性质：

> **它只奖励 value 的变化，不奖励“已经处在好状态并继续维持”。**

也就是说，如果 agent：
1. 成功把 peg 抬正；
2. 之后只是维持这个状态；
3. `V(s') ≈ V(s)`；

那么 TD reward 就会接近 0。

这和任务要求之间有一个本质错位：

- 任务真正关心的是 **episode end 仍 upright**；
- ACP TD reward 真正鼓励的是 **value 继续上涨**；
- 在“已经基本成功但尚未终止”的平台区间，reward 会变得非常弱。

所以 ACP 本身并不是“错”，但它对这个任务提供的是：
- 强 progress signal
- 弱 retention signal

这和 v6 报告里的历史结论完全一致：
- `docs/vlaw/acp_v6_rlpd_report.md` 已明确写过：ACP value function 难以区分 “holding” 与 “about to drop” 状态；
- TD reward 在保持成功时近似为零。

本轮 v7 正式报告只是把这个现象进一步分解到了算法层面：
- 对 **AWSC**，这个缺陷还能被 BC anchor 抵消一部分；
- 对 **PLD / DSRL**，这个缺陷会直接暴露出来。

---

## 4. 为什么 AWSC + ACP 还能工作，而 PLD / DSRL + ACP 不行

这是这轮分析里最重要的问题。

### 4.1 AWSC 的成功不是因为 ACP reward 很好，而是因为 actor 不依赖 reward 去“发明 hold”

正式报告显示：
- AWSC + ACP 的 critic 是健康的；
- actor 也是健康的；
- 真正差的是 reward 与 advantage：
  - online/offline reward gap ≈ **1049x**
  - `acp_step_mean ≈ 1e-4`
  - `advantage_mean ≈ 1.03`

这说明：

> **AWSC + ACP 并不是因为 reward 足够好才成功，而是即便 reward 很差，actor 仍然被 BC / flow 先验钉在一个不错的策略流形上。**

所以它的退化表现是：
- sim 版本 final SAE 0.74
- acp 版本 final SAE 0.52

也就是：**能用，但被弱 reward 拖了一截。**

### 4.2 PLD / DSRL 的失败在于：它们必须靠 RL objective 自己把 hold 学出来

而 PLD / DSRL 都没有 AWSC 那种强 BC actor anchor。

因此对它们来说，reward 的作用不是“轻微重加权”，而是：
- 直接决定什么行为会被保留；
- 决定 critic 会把哪些 action/noise 评为高价值；
- 决定 actor 会往哪一类策略坍缩。

如果 reward 对 holding 不敏感，就会出现一个典型偏差：

> 策略不断被鼓励去学“能更快 reach / grasp / lift”的行为，而不是“能稳定维持直到 episode end”的行为。

这正是正式报告里的 SO-SAE gap 所揭示的：
- PLD gap ≈ 0.76
- DSRL gap ≈ 0.78–0.84
- AWSC gap ≈ 0.14–0.20

换句话说：

- PLD / DSRL **不是没学会任务前半段**；
- 它们是 **只学会了任务前半段**。

---

## 5. PLD + ACP 为什么尤其容易失败

### 5.1 residual policy 天然更偏向“短时修正”，不擅长长时保持

PLD 学的是 `a_delta`，即对 base policy 的局部修正。

这个设计的隐含假设是：
- base policy 已经大体正确；
- residual 只需要做小范围补偿。

但“保持 upright 到 episode end”往往需要的是：
- 连续多步微调；
- 对抓取稳定性、姿态扰动、末端控制误差的长期补偿；
- 对“已经成功但仍需维持”的平台期策略。

而 residual RL 最容易学到的是：
- 哪种局部改动能更快把成功率从 0 提到有一次成功；
- 哪种局部改动能让“触碰/抬起”更容易发生。

因此 PLD 的结构天然更偏向 **progress correction**，不偏向 **retention control**。

### 5.2 PLD 的 entropy collapse 会把这个偏差进一步放大

正式报告里，PLD sim / acp 都出现：
- `entropy_min ≈ -55`
- sim 下 `temperature_final = 0.0032`
- acp 下虽然 final temperature 没那么低，但历史上同样发生过 collapse

这说明 PLD 的问题不是简单的“当前温度低”，而是：

> **训练过程中它曾多次塌到一个近似确定性的 residual policy，然后后续很难恢复出真正有用的 hold 微操控探索。**

一旦 collapse 发生，策略就更可能锁死在一类“能碰到/抓到，但 hold 不稳”的局部最优。

### 5.3 ACP 对 PLD 没有提供足够的反向纠偏

如果 ACP 能显著奖励 holding，那么 residual 还有机会慢慢学出稳定保持；
但 ACP 提供的主要仍是：
- value improvement
- grasp bonus

其中 grasp bonus 只能表达：
- “还抓着”

却不能表达：
- “虽然还抓着，但姿态在恶化，马上要掉”
- “保持 upright 的质量是否在提升/退化”

因此 PLD + ACP 的最终机制就是：

1. residual 学会帮助 reach / grasp；
2. 由于 hold 的优势信号不足，critic 无法把“稳住”与“快要掉”清晰区分；
3. entropy collapse 又让策略无法继续搜索更细的 hold 动作；
4. 最终得到高 SO、低 SAE、final SAE≈0 的结果。

**结论**：PLD + ACP 的失效本质上是 **结构性时序控制缺陷 + hold-aware reward 缺失 + entropy collapse** 的乘积，而不是单一 reward scale 问题。

---

## 6. DSRL + ACP 为什么比 PLD 好一点，但仍然失败

DSRL 比 PLD 更接近“可救”，因为它至少具备两点：

1. **探索没有明显塌掉**
   报告中 DSRL + ACP 的 exploration 是 A：
   - `temperature_avg = 0.1411`
   - `temperature_final = 0.1329`
   - `entropy_min = -8.56`

2. **ACP 确实给到了可学习的阶段性信号**
   报告显示：
   - `acp_base_mean = 0.0344`
   - `acp_grasp_bonus_mean = 0.7778`
   - `is_grasping_rate = 0.7778`

所以 DSRL + ACP 相比 sim 能把 best SAE 从 0.08 拉到 0.14，不是偶然，而是说明：

> **DSRL 的探索机制足以利用 ACP 提供的 progress + grasp signal，把策略推到“更常短暂成功”的区域。**

但它为什么最后还是只有 final SAE 0.02？

### 6.1 latent/noise control 优化的是“到达好区域”，不天然优化“停留在好区域”

DSRL 的 actor 更新目标仍然是标准 SAC：
- 最大化 `Q(s, noise)`
- 同时满足 entropy regularization

这意味着如果 critic 学到的是：
- 某些 latent 扰动很容易把轨迹推进到一次成功；

而不是：
- 某些 latent 扰动更容易让 agent 在剩余几十步里维持 upright；

那么 actor 会优先选择前者。

DSRL 相比 PLD 更强，只是因为 latent policy 的搜索空间更平滑；
但它依然没有任何显式机制，把“成功后继续保持”强行写进 actor objective。

### 6.2 DSRL 的 critic 虽不爆炸，但始终不够强，难以分辨“暂时成功”与“稳定成功”

正式报告给 DSRL + ACP 的 critic 是 C：
- `q_range = 13.6`
- `critic_loss_final = 16.33`
- `td_target_std = 1.192`

这说明 critic 并未发散，但也没有足够锐利。

在这个任务里，最重要的价值区分不是“有没有抓到”，而是：
- 抓到后能不能稳住；
- 当前状态离最终 drop 有多近；
- 当前姿态是否可持续。

如果 critic 只能学到一个粗糙的“progress score”，它就会把很多“已抓住但并不稳定”的状态也打成高值。

这样一来，actor 学出来的就更像：
- **高 success_once policy**
- 而不是 **高 success_at_end policy**

### 6.3 为什么 v6 的 DSRL long-grasp 能到 14%，而这轮仍然要判定其本质问题没解决

这并不矛盾。

v6 的结论已经说明：
- DSRL 是 PLD/DSRL 里唯一对长训练比较敏感的算法；
- grasp bonus + 长训练可以把 best SAE 顶到 14%。

但 v7 正式报告同时说明：
- 即使在 ACP 下，DSRL final SAE 仍只有 0.02；
- SO-SAE gap 仍极大；
- 也就是说它并没有真正解决 retention，只是偶尔更容易撞到短暂成功峰值。

所以更准确的表述是：

> **DSRL 不是“已经解决，只是被 ACP 拖累”；而是“它具备一定学到 hold 的潜力，但目前缺少一个能把这种潜力稳定固化下来的机制”。**

---

## 7. 真正的根因：不是“ACP 奖励不够大”，而是“算法目标与任务目标不对齐”

如果只看表面，容易得出一个肤浅结论：
- “把 ACP scale 调大一点就行。”

但从这轮正式报告看，这只是 **AWSC 问题的一部分**，不是 PLD / DSRL 失效的核心。

### 7.1 任务真正目标
任务最终看的是：
- **episode 结束时仍然 upright**

### 7.2 PLD / DSRL 当前实际优化的目标
在当前实现下，它们更接近在优化：
- 提高局部 progress；
- 提高 reach / grasp / transient lift 概率；
- 让 critic 对短期成功更敏感；

而不是：
- 明确优化 stable holding / long-horizon retention。

### 7.3 ACP 又进一步强化了这种错位
因为 ACP TD reward 的天然偏好就是：
- 奖励“继续变好”
- 不奖励“已经足够好并继续保持”

于是最终形成一个三层错位：

1. **任务层**：需要 retention
2. **奖励层**：主要表达 progress
3. **算法层**：PLD/DSRL 又缺少 BC-style hold anchor

所以 PLD / DSRL + ACP 的失败不是单点 bug，而是一个 **reward semantics × algorithm objective × task requirement** 的系统性失配。

---

## 8. 最终诊断结论

### 8.1 PLD + ACP 的失效原因

**主因**：
1. residual RL 只擅长局部修正，不擅长长时 hold control；
2. critic 只在 residual action space 上估值，难以精确刻画“稳定保持”这种细粒度长时价值；
3. ACP TD reward 对 holding 近似无梯度；
4. PLD 存在显著 entropy collapse，导致策略很早锁死在“能抓但 hold 不住”的局部最优。

**一句话总结**：

> **PLD + ACP 失败，不是因为它学不到成功，而是因为它只能学到“短时纠偏后的成功接触”，学不到“末态稳定保持”。**

### 8.2 DSRL + ACP 的失效原因

**主因**：
1. DSRL 在 noise space 优化，能学到更平滑的阶段性成功策略，但目标仍偏 progress；
2. ACP + grasp bonus 能提高短时成功峰值，却不能提供足够强的 hold-aware value ordering；
3. critic 虽稳定但不够锐利，无法持续区分“已成功但不稳定”与“稳定保持成功”；
4. 缺少 BC / imitation anchor，导致即便偶尔学到 hold，也难以长期保真。

**一句话总结**：

> **DSRL + ACP 不是完全没学到东西，而是只能把策略推到“经常能短暂成功”的区域，却没有机制把这种短暂成功固化成末态成功。**

### 8.3 AWSC 为什么是例外

**原因不是 ACP 更适合 AWSC，而是 AWSC 对 reward 缺陷不那么敏感。**

- 它有强 BC / flow anchor；
- actor 不需要从 reward 中“发明 hold”；
- critic 更多做 sample reweighting；
- 所以 reward 弱时只是性能下降，不会完全失效。

---

## 9. 对后续研究方向的直接启示

如果目标是**真正解释并修复 DSRL / PLD + ACP 失效**，下一轮不该再把重点放在“单纯调 scale / grasp bonus”上，而应该放在以下问题上：

### 9.1 方向一：把 hold / retention 写进 reward semantics
核心问题不是 reward 太弱，而是 reward **缺少 hold-aware ordering**。

可行方向：
- 让 ACP value 模型显式区分：
  - stable hold
  - unstable hold / about-to-drop
- 让 reward 不仅看 value 增量，还看 stability indicator
- 引入 end-state proximity / hold duration / pose stability 相关项

### 9.2 方向二：给 PLD / DSRL 增加“不会丢掉 hold 行为”的 actor anchor
AWSC 的成功说明：
- 有 anchor 的 actor 对 reward 缺陷容忍度更高。

所以可行方向是：
- 给 PLD / DSRL 增加某种 BC regularization / imitation anchor；
- 或者在成功片段上做 behavior retention regularization；
- 或者显式偏向成功末段片段，而不是全程等权。

### 9.3 方向三：不要只看 Best SAE，要重点看 SAE retention 与 SO-SAE gap
这轮报告最有价值的指标不是 best SAE，而是：
- `SAE Retention`
- `SO-SAE Gap`

因为它们直接区分了两类算法：
- **会短暂成功**
- **会稳定保持成功**

后续所有 sweep 都应该默认把这两个指标纳入主表，而不是只看峰值 SAE。

---

## 10. 最终一句话结论

> **DSRL / PLD + ACP 失效的深层原因，不是“ACP 奖励太小”这么简单，而是：ACP 主要提供 progress signal、缺少 hold-aware signal；而 PLD / DSRL 又都缺少像 AWSC 那样能把 hold 行为保留下来的强 actor anchor，于是训练自然收敛到“能抓到、能短暂成功、但无法稳定保持到 episode end”的策略。**
