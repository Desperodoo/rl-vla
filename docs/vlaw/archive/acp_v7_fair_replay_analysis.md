# [ARCHIVE] ACP v7 Fair Replay 分析：为什么 exact recipe 也没有复现历史高 SAE

**定位**：本文件是中间阶段归档，回答“为什么 exact historical replay 在当前代码下没有复现高 SAE sim baseline”。当前最终结论请优先参考：
- `docs/vlaw/archive/acp_v7_qclip0_acp_analysis.md`
- `docs/vlaw/archive/acp_v7_failure_analysis.md`
- `docs/vlaw/archive/acp_v7_diagnosis_progress.md`

## 依据
- 历史 fair-comparison recipe：
  - `runs/fair_comparison/pld/best__1772537094/config.json`
  - `runs/fair_comparison/dsrl/best__1772537094/config.json`
- replay 正式报告：
  - `docs/vlaw/figures/rlpd_acp_v7_fair_replay/diagnosis_report.md`
  - `docs/vlaw/figures/rlpd_acp_v7_fair_replay/diagnosis_summary.json`
- 当前训练入口：
  - `rlft/online/train_pld.py`
  - `rlft/online/train_dsrl.py`
- 当前算法实现：
  - `rlft/algorithms/online_rl/pld_sac.py`
  - `rlft/algorithms/online_rl/dsrl_sac.py`

---

## 1. 结论先说

这次 `fair_replay` 最重要的结果不是 ACP 的结论，而是：

> **即便严格切回 `runs/fair_comparison` 的 71K historical recipe，当前代码路径下也没有复现出历史上出现过的高 SAE sim baseline。**

因此当前最核心的问题已经进一步收敛为：

1. `PLD/DSRL + sim_reward` 的历史高 SAE 结果当前不可复现；
2. 这说明主问题已经不是“ACP 会不会破坏算法”，而是 **historical fair-comparison regime 与当前实现之间出现了 code-path drift / behavior drift**；
3. 在这个 drift 没有定位清楚之前，所有“ACP 为什么不如 sim”的解释都只能算次级问题。

---

## 2. replay 实验结果

来自 `docs/vlaw/figures/rlpd_acp_v7_fair_replay/diagnosis_report.md`：

| Algo | Reward | Best SO | Best SAE | Final SAE | SAE Retention | SO-SAE Gap |
|---|---:|---:|---:|---:|---:|---:|
| DSRL | acp | 0.940 | 0.060 | 0.000 | 0.064 | 0.880 |
| DSRL | sim | 0.880 | 0.020 | 0.000 | 0.023 | 0.860 |
| PLD | acp | 0.920 | 0.060 | 0.000 | 0.065 | 0.860 |
| PLD | sim | 0.800 | 0.060 | 0.000 | 0.075 | 0.740 |

这张表揭示了三个关键事实：

### A. sim baseline 没有复现
- PLD sim 本应是历史高 SAE recipe，但当前 best SAE 只有 **0.06**，final SAE 为 **0.00**。
- DSRL sim 当前更差，best SAE 只有 **0.02**，final SAE 为 **0.00**。

### B. ACP mirror 并没有比 sim 明显更差
- PLD acp 与 PLD sim 都是 best SAE 0.06。
- DSRL acp 甚至比 DSRL sim 略高（0.06 vs 0.02）。

这意味着：

> 在当前实现下，ACP 不是主导性的额外破坏因素；算法已经在 sim 下先失败了。

### C. 失败模式高度一致：高 SO，低 SAE，final SAE 清零
- 所有 4 个 run 都能把 `success_once` 拉到 0.8–0.94。
- 但 `success_at_end` 都只能到 0.02–0.06，最后全部回到 0。

这是一种非常强的统一 failure signature：

> **策略能 reach / grasp / briefly lift，但无法稳定保持到 episode end。**

---

## 3. 这次 replay 实际上排除了什么

### 3.1 排除了“只是 v7 diag 参数不对”这一弱解释

之前可以怀疑：`diag_core` 用的是为了诊断而匹配的 recipe，不是历史 best recipe，所以 sim 低并不奇怪。

现在这个解释已经不够了。

因为 `fair_replay` 已经切回：
- PLD：71K, gamma=0.99, init_temperature=0.1, action_scale=0.3, eval_freq=5000
- DSRL：71K, gamma=0.95, action_magnitude=2.5, num_seed_steps=0, eval_freq=5000

仍然没有恢复历史高 SAE。

所以问题不能再简单归因于：
- 训练步数不对
- diag_core 的 match 方式不对
- sim/acp 对照预算不对

### 3.2 进一步削弱了“ACP 独有失效”解释

如果当前 failure 是 ACP 专属，那么在 exact replay 下应该看到：
- sim 恢复正常
- acp 继续失败

但现在看到的是：
- sim 没恢复
- acp 也没恢复
- 两者 failure signature 极其相似

因此：

> 当前最优先的解释不再是 reward semantics，而是 **当前实现和历史运行条件之间存在更底层的行为漂移。**

---

## 4. 从 replay 报告看，当前失败更像什么

### 4.1 不是 actor 明显崩了

报告里 4 个 run 的 actor 全是 `A`：
- PLD acp: entropy_final -2.91
- PLD sim: entropy_final -0.06
- DSRL acp: entropy_final -4.13
- DSRL sim: entropy_final 1.46

说明这次 failure 不是简单的“actor 完全塌缩，什么都学不到”。

### 4.2 critic 依然普遍偏弱

4 个 run 的 critic 全是 `C`：
- PLD acp: critic_loss_final 6.46, td_target_std 2.2
- PLD sim: critic_loss_final 1.10, td_target_std 1.277
- DSRL acp: critic_loss_final 8.82
- DSRL sim: critic_loss_final 1.08

这更像是：
- critic 能学到 reach / grasp / partial lift 的价值排序；
- 但学不到足够稳定的 `hold-until-end` 区分；
- 或者当前实现里的某个路径已经改变了历史上 critic 可用的训练分布。

### 4.3 failure signature 与之前 v7 一致

这次 replay 和之前 `diag_core` 的共同点是：
- SO 高
- SAE 极低
- final SAE 归零

因此这不是偶发 seed 波动，而是当前代码路径下相当稳定的系统性行为。

---

## 5. 当前最值得怀疑的，不是 reward，而是 code-path drift

在 replay 已经对齐 historical recipe 的前提下，最值得优先查的是：

### A. 训练入口 drift
重点文件：
- `rlft/online/train_pld.py`
- `rlft/online/train_dsrl.py`

优先查：
- 当前默认值和 historical config 一致，但**实际生效逻辑**是否仍一致；
- eval agent 的 deterministic action 路径是否改变；
- env wrapper / reset / done 处理是否改变。

### B. 算法实现 drift
重点文件：
- `rlft/algorithms/online_rl/pld_sac.py`
- `rlft/algorithms/online_rl/dsrl_sac.py`

优先查：
- actor loss / critic loss / alpha update 是否与历史 fair-comparison 运行时期一致；
- critic 输入语义是否改变；
- target update 或 q clipping 的启用方式是否改变；
- replay buffer 中 action / reward / done 的写入语义是否改变。

### C. 评估路径 drift
当前需要特别怀疑：
- `success_at_end` 统计口径是否变了；
- eval cadence 与历史一样，但 eval env / wrapper 是否已变化；
- `best.pt` 保存条件是否还是围绕 `success_at_end`。

---

## 6. 现在对 ACP 应该怎么解读

在 replay 结果出来后，对 ACP 最稳妥的结论应当更新为：

1. ACP 仍然**不是**一个强 hold-aware reward；
2. 但在当前代码路径下，它也**不是** PLD / DSRL 失效的首要解释；
3. 当前第一优先级是恢复 historical sim baseline；
4. 只有在 sim baseline 恢复后，再谈 ACP 与 sim 的精细差异才有意义。

---

## 7. 下一步建议

### 第一优先级
做一次面向 `fair_comparison` 的 **code-path drift audit**：
- 逐文件比对 `train_pld.py` / `train_dsrl.py` / `pld_sac.py` / `dsrl_sac.py`
- 目标不是再调参，而是定位“为什么 exact recipe 在当前实现下不再产生历史行为”

### 第二优先级
如果能找到 drift 点，重新跑最小复现实验：
- `pld_fair_sim`
- `dsrl_fair_sim`

### 第三优先级
只有在 sim baseline 恢复后，再继续用同一 recipe 比较：
- `pld_fair_acp`
- `dsrl_fair_acp`

这样才能得到可信的 ACP 对照结论。
