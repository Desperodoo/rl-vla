# [ARCHIVE] ACP v7 Drift Regression 分析报告

**定位**：本文件是中间阶段归档，回答“共享 drift 是否主要来自 q-target clipping，以及 sim baseline 如何恢复”。当前最终结论请优先参考：
- `docs/vlaw/archive/acp_v7_qclip0_acp_analysis.md`
- `docs/vlaw/archive/acp_v7_failure_analysis.md`
- `docs/vlaw/archive/acp_v7_diagnosis_progress.md`

## 依据
- 回归正式报告：`docs/vlaw/figures/rlpd_acp_v7_drift_reg/diagnosis_report.md`
- 回归汇总：`docs/vlaw/figures/rlpd_acp_v7_drift_reg/diagnosis_summary.json`
- 历史 fair replay 报告：`docs/vlaw/figures/rlpd_acp_v7_fair_replay/diagnosis_report.md`
- 提交级 drift 审计：`docs/vlaw/archive/acp_v7_codepath_drift_audit.md`

---

## 1. 这轮回归真正回答了什么

这轮回归实验的目标是回答两个更具体的问题：

1. **是不是 q-target clipping 改变了历史 sim baseline 的训练形态？**
2. **是不是某个更早的历史 regime（PLD 的 `54faf40` 风格、DSRL 的 `592df92` 之前风格）才是高 SAE 的真正来源？**

现在答案已经比较清楚：

> **是的，`q_target_clip` 对当前复现结果影响非常大；它不像单纯“稳定 critic”，而是在当前代码路径下几乎决定了 PLD/DSRL 能否重新出现高 SAE sim baseline。**

同时：

> **PLD 的高 SAE 更接近“关掉 q-target clipping 后即可恢复”，而 DSRL 的高 SAE 更接近“当前 tuned regime + 关掉 q-target clipping”恢复；反而 pre-`592df92` 的旧 DSRL regime 并没有恢复成功。**

---

## 2. 回归实验结果总表

来自 `docs/vlaw/figures/rlpd_acp_v7_drift_reg/diagnosis_report.md`：

| Experiment | Algo | Reward | Best SO | Best SAE | Final SAE |
|---|---|---|---:|---:|---:|
| `pld_v7_reg_qclip0_sim_s42` | PLD | sim | 0.98 | **0.82** | **0.82** |
| `pld_v7_reg_54faf40_sim_s42` | PLD | sim | 0.96 | **0.80** | **0.80** |
| `pld_v7_reg_qclip0_acp_s42` | PLD | acp | 0.70 | 0.02 | 0.00 |
| `dsrl_v7_reg_qclip0_sim_s42` | DSRL | sim | 0.94 | **0.66** | **0.66** |
| `dsrl_v7_reg_qclip0_acp_s42` | DSRL | acp | 0.94 | 0.06 | 0.00 |
| `dsrl_v7_reg_pre592df92_sim_s42` | DSRL | sim | 0.94 | 0.04 | 0.02 |

和上一轮 `fair_replay` 结果对比：

| Setting | PLD sim best SAE | DSRL sim best SAE |
|---|---:|---:|
| `fair_replay`（当前代码，默认 qclip=20） | 0.06 | 0.02 |
| `qclip0` 回归 | **0.82** | **0.66** |
| `54faf40` 风格 PLD | **0.80** | — |
| `pre-592df92` 风格 DSRL | — | 0.04 |

---

## 3. 最重要的实证结论

### 3.1 PLD：当前无法复现历史高 SAE 的主因，极大概率就是 `q_target_clip=20`

这是目前最强的结论。

- `fair_replay`（默认 `q_target_clip=20`）:
  - PLD sim `best SAE = 0.06`
  - `final SAE = 0.00`
- `pld_v7_reg_qclip0_sim_s42`：
  - PLD sim `best SAE = 0.82`
  - `final SAE = 0.82`
- `pld_v7_reg_54faf40_sim_s42`：
  - PLD sim `best SAE = 0.80`
  - `final SAE = 0.80`

也就是说：

> **PLD 几乎不需要复杂回退，只要在当前 fair recipe 上把 `q_target_clip` 去掉，就能重新回到高 SAE 区间。**

这说明之前的失败并不是“PLD 本体已经坏了”，也不是“历史 baseline 根本不可复现”，而是：

> **后续为了 ACP/v5/v6 critic 稳定化引入的 `q_target_clip=20`，在 sim reward 下把 PLD 的有效价值差异压扁了。**

### 3.2 DSRL：当前高 SAE 也主要依赖于“关掉 q-target clipping”，而不是回退到旧 regime

DSRL 结果更有判别力：

- `fair_replay`（默认 `q_target_clip=20`）:
  - DSRL sim `best SAE = 0.02`
- `dsrl_v7_reg_qclip0_sim_s42`：
  - DSRL sim `best SAE = 0.66`
  - `final SAE = 0.66`
- `dsrl_v7_reg_pre592df92_sim_s42`：
  - DSRL sim `best SAE = 0.04`
  - `final SAE = 0.02`

这说明：

1. **旧 DSRL regime（pre-592df92）并不是答案**；
2. **当前 tuned DSRL regime 本身并没有坏**；
3. 真正压制 DSRL SAE 的关键因子，同样是 `q_target_clip=20`。

换句话说：

> DSRL 不需要回到旧参数；它需要的是：**保留当前 tuned regime，但去掉当前对 sim baseline 过强的 q-target clipping。**

---

## 4. 为什么 q-target clipping 会造成这种现象

这轮回归给出的行为模式非常一致：

### 当 `q_target_clip=20`（fair replay）时：
- PLD/DSRL sim 都是高 SO、低 SAE、final SAE 清零
- 看起来像“能抓到，但 hold 不住”

### 当 `q_target_clip=0` 时：
- PLD sim 直接恢复到 `0.82` SAE
- DSRL sim 恢复到 `0.66` SAE
- 两者 final SAE 都不再崩回 0

这说明 clipping 不只是“让 critic 更稳定”，而是在当前任务里改变了 critic 对成功末态的排序能力。

更具体地说：

- `LiftPegUpright` 的关键并不是“有没有正回报”，而是**末态保持质量的细粒度差异**；
- `q_target_clip=20` 会把高回报尾部和成功末态的价值差别压缩；
- 一旦这些差异被压平，critic 仍能鼓励 reach / grasp / short lift，
  但对 **hold until end** 的区分能力就不够；
- 这就导致：
  - `success_once` 还能很高；
  - `success_at_end` 学不出来；
  - final retention 崩掉。

因此当前最合理的解释是：

> **`q_target_clip=20` 对 ACP reward 是稳定化，但对 sim reward 的历史 fair baseline 来说，它是“过强稳定化”，把最需要的长期末态价值差异切掉了。**

---

## 5. 这轮结果如何重写当前 root-cause 认识

### 旧判断
之前我们只能说：
- 当前代码下 sim baseline 也坏了；
- 所以先别急着怪 ACP。

### 新判断
现在可以更明确地说：

#### 对 PLD
- 当前 sim baseline 失效的**主因**基本锁定为：`q_target_clip=20`
- `54faf40` 风格参数也能工作，但不是必要条件
- 当前 PLD code path 并没有“彻底坏掉”，而是被后续稳定化改造压制了

#### 对 DSRL
- 当前 sim baseline 失效的**主因也高度怀疑是 `q_target_clip=20`**
- pre-`592df92` 旧 DSRL regime 无法恢复高 SAE
- 所以真正有效的是：**当前 tuned DSRL + no qclip**，不是“回退到旧 DSRL”

#### 对 ACP
- 这轮结果没有洗白 ACP 问题
- 反而更清楚地区分了两件事：
  1. **sim baseline 失效主要是 qclip drift**
  2. **ACP 组即便在 qclip0 下也没恢复**，说明 ACP 仍然有独立问题

也就是说：

> 当前有两个层级的问题：
>
> 1. **共享 drift 问题**：`q_target_clip=20` 破坏了历史 sim baseline 复现；
> 2. **ACP 专属问题**：即便移除 qclip，ACP mirror 仍然不行，说明 reward semantics / scale / hold-signal 依旧不足。

---

## 6. 接下来最值得做的事

### 第一优先级：恢复 sim baseline
建议把 sim baseline 的官方 replay 路径切换为：
- PLD sim: `q_target_clip=0`
- DSRL sim: `q_target_clip=0`

先把历史 sim baseline 重新立住。

### 第二优先级：重新做 ACP 镜像对照
在 sim baseline 已恢复的前提下，再比较：
- PLD sim qclip0 vs PLD acp qclip0
- DSRL sim qclip0 vs DSRL acp qclip0

这样才能把 **qclip drift** 和 **ACP 本身问题** 真正分开。

### 第三优先级：如还要继续细化 ACP 问题
优先关注：
- ACP reward scale
- TD vs potential shaping
- grasp bonus 只能鼓励抓住、不能鼓励稳住 的问题

但这些都应排在 sim baseline 恢复之后。

---

## 7. 最终结论（一句话版）

> **当前复现失败的第一根因已经基本锁定：`q_target_clip=20` 破坏了历史 PLD/DSRL 的 sim baseline；而 ACP 组在去掉 qclip 后仍未恢复，说明 ACP 还存在第二层独立问题。**
