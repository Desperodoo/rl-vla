# Sweep Baselines 知识汇总 — LiftPegUpright-v1 (ManiSkill3)

> **汇总日期**: 2026-03-01  
> **环境**: LiftPegUpright-v1, ManiSkill3, GPU 向量化  
> **数据来源**:
> - `runs/maniskill_sweep_v3/analysis_results/analysis_report.md` — Offline IL/RL (72 configs)
> - `runs/dsrl_sweep/analysis_results/deep_analysis.md` — DSRL-SAC (53 configs, 200K steps)
> - `runs/pld_sweep_v3/analysis_results/deep_analysis.md` — PLD-SAC v3 (34 configs, 500K steps)
> - `runs/rlpd_sweep_v4/wave4_deep_analysis.md` — AWSC wave4 (12 configs, 500K steps)
> - `runs/rlpd_sweep_v4/analysis_results/analysis_report.md` — AWSC auto analysis
> - `results.json` — 综合结果 JSON（8 个算法族，200+ configs）

---

## 一、方法总览表

### 1.1 Offline / IL 方法（maniskill_sweep_v3, ≤100K train steps）

| 方法 | 最优配置 | success_once | success_at_end | 备注 |
|------|---------|:-----------:|:--------------:|------|
| **AW-ShortCut-Flow (AWSC)** | `cw0.3_step0.15` | **0.85** | — | 21 configs, lr=2e-4 关键 |
| **AWCP** | `beta10_rs0.05` | **0.85** | 0.49 (`ema_0.99`) | 17 configs, lr=2e-4 关键 |
| **Consistency Flow** | `ema_0.9995` | 0.71 | 0.33 | 12 configs |
| **CPQL** | `a0.0005_bc0.5` | 0.71 | 0.38 (`tau_0.01`) | 12 configs |
| **ShortCut Flow (纯 IL)** | `sc_k_0.25` | 0.60 | 0.26 | 10 configs, 1-step 推理 |
| Flow Matching | `pred_horizon_8` | 0.71 | 0.31 | 16 configs (results.json) |
| Diffusion Policy | `obs_horizon_4` | 0.45 | 0.23 | 16 configs (results.json) |
| Reflected Flow | `reflection_soft` | 0.48 | 0.23 | 22 configs (results.json) |

### 1.2 Online RL 方法（在预训练 ShortCut Flow 基础上微调）

| 方法 | 最优配置 | success_once | success_at_end | 训练步数 | 备注 |
|------|---------|:-----------:|:--------------:|:--------:|------|
| **DSRL-SAC** | `arch_3x2048` | 0.98 | **0.50** | 200K | 噪声空间 RL |
| **PLD-SAC v3** | `ablate_temp_0.5` | 1.00 | **0.80** | 500K | 动作空间 RL |
| **AWSC (wave4)** | `combo_or25` | 0.88 | **0.66** | 500K | advantage-weighted |
| **AWSC (wave4)** | `lr_7e5` | 0.94 | **0.66** | 500K | 最稳定 |

### 1.3 全局排名（按 success_at_end）

| 排名 | 方法 | 配置 | success_once | success_at_end | 类型 |
|:----:|------|------|:-----------:|:--------------:|------|
| 1 | PLD-SAC v3 | `ablate_temp_0.5` | 1.00 | **0.80** | Online RL |
| 2 | PLD-SAC v3 | `interact_gamma0.99_tau0.001` | 1.00 | **0.78** | Online RL |
| 3 | PLD-SAC v3 | `ablate_batch_256` | 0.96 | **0.78** | Online RL |
| 4 | PLD-SAC v3 | `ablate_calql_alpha_5.0` | 0.96 | **0.74** | Online RL |
| 5 | AWSC wave4 | `combo_or25` | 0.88 | **0.66** | Online RL |
| 6 | AWSC wave4 | `lr_7e5` | 0.94 | **0.66** | Online RL |
| 7 | DSRL-SAC | `arch_3x2048` | 0.98 | **0.50** | Online RL |
| 8 | AWCP (offline) | `ema_0.99` | 0.81 | **0.49** | Offline |
| 9 | AW-ShortCut-Flow | `cw0.3_step0.15` | 0.85 | — | Offline |

---

## 二、各方法详解

### 2.1 ShortCut Flow（纯 IL 基线）

**角色**: 所有在线 RL 方法的预训练策略基础。

| 排名 | 配置 | success_once | success_at_end |
|:----:|------|:-----------:|:--------------:|
| 1 | `weights_1.0_1.0` | 0.64 | 0.23 |
| 2 | `sc_k_0.25` | 0.60 | — |
| 3 | `ema_0.99` | 0.50 | 0.19 |

**核心发现**:
1. bc_weight=1.0 + consistency_weight=1.0 的等权组合最优
2. 1-step 推理（shortcut 核心能力）是部署时的关键优势
3. success_at_end 仅 0.23-0.26，说明纯 IL 的"保持"能力不足 → 需要 RL/ORL 微调

### 2.2 AWCP（Advantage-Weighted Critic Policy, Offline RL）

| 排名 | 配置 | success_once | success_at_end |
|:----:|------|:-----------:|:--------------:|
| 1 | `beta10_rs0.05` | 0.85 | — |
| 2 | `ema_0.99` | 0.81 | 0.49 |
| 3 | `conservative` | 0.81 | 0.45 |

**关键超参**: lr=2e-4（sensitivity=0.14, 最敏感参数）  
**核心发现**:
1. 学习率是最关键参数，2e-4 最优
2. EMA 显著提升稳定性，ema=0.99 在 success_at_end 上最优(0.49)

### 2.3 AW-ShortCut-Flow（Offline RL + ShortCut）

| 排名 | 配置 | success_once | success_at_end |
|:----:|------|:-----------:|:--------------:|
| 1 | `cw0.3_step0.15` | 0.85 | — |
| 2 | `aggressive` | 0.91 | 0.25 |
| 3 | `inference_16` | 0.84 | 0.45 |

**关键超参**: lr=2e-4（sensitivity=0.78, **极度敏感**）, num_qs=5  
**核心发现**:
1. LR 是最敏感参数（0.78），**必须使用 2e-4**
2. `aggressive` 配置 success_once 最高(0.91) 但 success_at_end 仅 0.25 → 典型"能做到但保持不住"
3. tau=0.001 和 gamma=0.999 对 success_at_end 有显著帮助(0.47-0.48)

### 2.4 DSRL-SAC（噪声空间 RL, 200K steps）

**核心机制**: Actor 输出噪声向量 w ∈ [-mag, +mag]^(T×d)，通过冻结的 ShortCut Flow 解码为实际动作。

| 排名 | 配置 | success_once | success_at_end |
|:----:|------|:-----------:|:--------------:|
| 1 | `arch_3x2048` | 0.98 | **0.50** |
| 2 | `utd_80` | 0.94 | 0.48 |
| 3 | `combined_high_utd_large_batch` | 0.92 | 0.46 |
| 4 | `num_qs_10` | 0.94 | 0.46 |
| 5 | `utd_60` | 0.92 | 0.46 |

**推荐配置**:

| 参数 | 推荐值 | 理由 |
|------|:------:|------|
| 架构 | 3×2048 或 3×1024 | 足够容量学习精细控制 |
| UTD | 60-80 | Critic 精度是稳定性关键 |
| gamma | 0.95 | 匹配 episode 长度(100步) |
| action_magnitude | 2.5 | 平衡修正幅度与精度 |
| num_seed_steps | 0 | 预训练策略已保证探索质量 |
| num_qs | 10 | 抑制 Q 过估计 |
| target_entropy | -3.5 | 平衡探索与利用 |
| log_std_init | -5.0 | 保守初始探索 |

**核心发现**:
1. **success_once 饱和**（全部 ≥0.90），真正区分配置好坏的是 success_at_end
2. **success_once 与 success_at_end 几乎不相关**（r=0.15）→ 典型"抬起又放下"行为
3. 关键改善三因素：Critic 更新量(UTD/num_qs) > 网络容量(layer_size) > 折扣因子匹配(gamma≈0.95)
4. seed_steps=0 最优 → 预训练策略足够好，无需 warmup

### 2.5 PLD-SAC v3（动作空间 RL, 500K steps）— 当前最强

**核心机制**: 在预训练 ShortCut Flow 的动作空间上直接运行 SAC + Cal-QL 正则化。

| 排名 | 配置 | success_once | success_at_end |
|:----:|------|:-----------:|:--------------:|
| 1 | `ablate_temp_0.5` | 1.00 | **0.80** |
| 2 | `interact_gamma0.99_tau0.001` | 1.00 | **0.78** |
| 3 | `ablate_batch_256` | 0.96 | **0.78** |
| 4 | `ablate_calql_alpha_5.0` | 0.96 | **0.74** |
| 5 | `bound_arch_3x768` | 0.98 | **0.74** |

**v3 Baseline**: lr=1e-4, layer=1024, batch=1024, num_qs=5, as=0.3, calql_alpha=0, gamma=0.99, or=1.0, temp=0.1, utd=60

**推荐最优配置（基于消融实验）**:

| 参数 | v3 baseline | 推荐改进 | 依据 |
|------|:-----------:|:--------:|------|
| init_temperature | 0.1 | **0.5** | 消融 +0.14 |
| tau | 0.005 | **0.001** | 高 UTD 下必须小 tau |
| calql_alpha | 0.0 | **5.0** | 在 lr=1e-4 下变为有效正则化 |
| layer_size | 1024 | **768** | 500K 步内收敛更快 |
| lr | 1e-4 | **1e-4** | 恢复到 1e-3 会崩(0.46) |
| action_scale | 0.3 | **0.3** | 恢复会降 |
| online_ratio | 1.0 | **1.0** | 恢复大幅降 |
| gamma | 0.99 | **0.99** | 恢复会降 |
| num_qs | 5 | **5** | 恢复会降 |

**核心发现**:
1. **冗余稳定化问题**: lr=1e-4 已从根源解决 Q-divergence，额外的保守措施(temp=0.1, batch=1024)反而抑制学习
2. **tau=0.001 是关键但被忽视的参数**: 高 UTD(60)下，tau=0.005 等效于 target 几乎 hard update
3. **LR 甜点极窄**: 1e-4 是唯一可行值，向任何方向偏移都大幅恶化
4. **网络深度>宽度**: 3×768 > 3×1024 > 3×2048 ≫ 2×1024 ≫ 4×512
5. **Cal-QL 正则化在稳定训练条件下有价值**: calql_alpha=5.0 + 预训练1000步提升 +0.08

### 2.6 AWSC（Advantage-Weighted ShortCut, Online, 500K steps）

**核心机制**: ShortCut Flow 策略 + SAC critic，用 advantage-weighting 进行 actor 更新。

| 排名 | 配置 | success_once | success_at_end | 综合得分 |
|:----:|------|:-----------:|:--------------:|:--------:|
| 1 | `lr_7e5` | 0.94 | **0.66** | 0.660 |
| 2 | `combo_or25` | 0.88 | **0.66** | 0.661 |
| 3 | `beta_60` | 0.92 | 0.62 | 0.655 |
| 4 | `combo_k2` | 0.90 | 0.54 | 0.647 |
| 5 | `lr_5e5` | 0.94 | 0.56 | 0.645 |

**v4 Baseline**: actor_policy_mode=all, or=0.15, beta=50, K=4, utd=20, lr=1e-4

**推荐配置**:
- **部署推荐（最稳定）**: lr=7e-5, beta=50, or=0.15, K=4 → 唯一 final_se=best_se, 趋势平稳
- **峰值追求**: beta=80, K=2, or=0.25 → best_se=0.72, 但 s_once 下降严重(-0.102)

**核心发现**:
1. **超参组合不具备加性效应**: beta=80 + K=2 的交互是对抗性的，不要盲目组合各自最优
2. **LR 与训练步数的关系**: $\text{lr}_\text{optimal} \propto 1/\sqrt{T}$ — 100K→2e-4, 250K→1e-4, 500K→7e-5
3. **450K 步系统性 s_once 崩塌**: 所有 12 个配置同步出现，原因可能是 buffer overflow + critic 过拟合
4. **500K 边际收益递减**: 多数配置在 250K 已达 best_se 峰值

---

## 三、方法间对比分析

### 3.1 success_at_end 排名（跨方法最终指标）

```
PLD-SAC v3   ████████████████████████████████████████  0.80  (ablate_temp_0.5)
AWSC wave4   █████████████████████████████████         0.66  (lr_7e5)
DSRL-SAC     █████████████████████████                 0.50  (arch_3x2048)
AWCP offline ████████████████████████                  0.49  (ema_0.99)
CPQL offline ███████████████████                       0.38  (tau_0.01)
Cons. Flow   ████████████████                          0.33  (ema_0.9999)
Flow Match   ███████████████                           0.31  (obs_horizon_4)
SC Flow      ████████████                              0.26  (inference_4)
Diff Policy  ███████████                               0.23  (obs_horizon_4)
```

### 3.2 各方法优劣对比

| 维度 | 最强方法 | 分析 |
|------|---------|------|
| **success_at_end** (保持能力) | PLD-SAC v3 (0.80) | 动作空间 RL + 深度消融，大幅领先 |
| **success_once** (达成能力) | PLD-SAC v3 (1.00) / DSRL-SAC (0.98) | 在线 RL 方法均可轻松"做到" |
| **纯 IL 任务达成率** | AWSC offline / AWCP (0.85) | 无需在线交互即可达到高 success_once |
| **训练效率（200K内）** | DSRL-SAC | 200K 步即可 success_once≥0.90 |
| **训练稳定性** | AWSC lr_7e5 | 唯一 final_se=best_se（不衰退） |
| **推理速度** | ShortCut Flow 系列 | 1-step 推理，其他 flow 方法需多步 |

### 3.3 关键跨方法模式

1. **success_once vs success_at_end 的 gap 普遍存在**: 所有方法都能"做到"但"保持不住"，gap 在纯 IL 中最大（SC Flow: 0.64 vs 0.23）
2. **高 UTD 对 success_at_end 至关重要**: DSRL-SAC(UTD=80: 0.48) 和 PLD-SAC(UTD=60: 0.80) 都受益于高 UTD
3. **LR 是所有方法中最敏感的超参**: 每个方法都有极窄的 LR 甜点
4. **网络容量需要与 LR 匹配**: 低 LR → 小网络收敛更快(PLD 3×768)；高 LR → 大网络更稳定(DSRL 3×2048)
5. **EMA 在 Offline 方法中普遍有效**: AWCP(ema=0.99) 和 Consistency Flow(ema=0.9995) 都受益
6. **tau 在高 UTD 设置中需要极小值**: PLD-SAC tau=0.001, AW-SC tau=0.001 均显著提升

---

## 四、VLAW 策略微调参考信息

### 4.1 VLAW 场景特点回顾

VLAW 的策略微调流程: 
1. ShortCut Flow 预训练策略（IL）→ 2. 世界模型(Ctrl-World)生成虚拟轨迹 → 3. VLM 奖励模型打分 → 4. 用虚拟数据+奖励微调策略 → 迭代

关键约束:
- **无真实环境交互**（纯离线 + 虚拟数据）
- **奖励来自 VLM**（二分类 P('yes') > 0.8，非稠密奖励）
- **虚拟轨迹质量受世界模型限制**（分布偏移更严重）

### 4.2 方法适用性分析

| 方法 | 适用 VLAW？ | 理由 |
|------|:----------:|------|
| **ShortCut Flow (纯 IL)** | ✅ 是 | 作为预训练基线，1-step 推理速度快，适合 WM roll-in |
| **AWCP / AWSC (Offline)** | ✅✅ 最适合 | 纯离线 advantage weighting，天然适合 WM 生成的虚拟数据 |
| **PLD-SAC** | ⚠️ 有条件 | 需要在线交互；若用 ImaginationEnv 代替真实环境则可行，但 Q-value 更新依赖 WM 准确性 |
| **DSRL-SAC** | ❌ 不适合 | 噪声空间 RL 高度依赖精确环境反馈，WM 误差在噪声空间被放大 |

### 4.3 推荐的 VLAW 策略微调方案

**首选: AWSC / AWCP 风格的 advantage-weighted regression**

理由:
1. **不需要在线环境交互**——直接用 WM 生成的虚拟轨迹 + VLM 奖励做 offline RL
2. **对数据质量容忍度高**——advantage weighting 自动 downweight 低质量虚拟轨迹
3. **beta 参数可调节 exploitation 力度**——当 WM 质量差时用较低 beta(30-50) 更保守；WM 质量好时用较高 beta(60-80) 更积极
4. **与 ShortCut Flow 架构天然兼容**——AWSC 就是在 SCF 上做 advantage weighting

**关键超参参考（基于 sweep 经验）**:

| 参数 | 建议值 | 来源 |
|------|:------:|------|
| lr | 2e-4 (短训练) / 1e-4 (长训练) | AWSC/AWCP 一致结论 |
| beta | 50-60 | AWSC v4 在 or=0.15 下最优 |
| num_v_samples(K) | 4 | K=2 + 高 beta 有对抗交互 |
| EMA | 0.99-0.9995 | 所有 offline 方法受益 |
| num_qs | 5 | PLD v3 sweet spot |

**备选: PLD-SAC + ImaginationEnv**

适用条件: 当世界模型足够准确时，可将 ImaginationEnv 作为虚拟环境，PLD-SAC 提供最高的 success_at_end 上限(0.80)。  
风险: WM 的 compounding error 会让 Q-value 估计偏离，可能需要:
- 较低 UTD (20-40 而非 60-80) → 减缓 Q-value 发散速度
- 较短的 imagination horizon → 限制 compounding error
- Cal-QL 正则化 (alpha=5.0) → 压制 OOD Q 值

---

## 五、关键超参经验总结（跨方法通用）

| 超参 | 通用规律 | 证据来源 |
|------|---------|---------|
| **Learning Rate** | 最敏感参数；$\text{lr}_\text{opt} \propto 1/\sqrt{T}$ | AWSC v4, PLD v3 |
| **UTD Ratio** | 越高 → success_at_end 越好，但需配合小 tau 和大网络 | DSRL, PLD |
| **tau** | 高 UTD 下 $\tau \leq 0.001$，否则 target network 形同虚设 | PLD v3 |
| **num_qs** | 5 是 pessimism-accuracy 最佳平衡，2 太少 10 太多 | PLD v3, DSRL |
| **gamma** | 匹配 episode 长度: 100步→0.95, 长 episode →0.99 | DSRL, PLD |
| **EMA** | offline 方法普遍受益，0.99-0.9995 | AWCP, Cons. Flow |
| **网络架构** | 3 层必需，宽度与 LR 匹配: 低 LR用小网络, 高 LR 用大网络 | PLD v3 |
| **init_temperature** | 在稳定训练条件(低LR)下，0.5 优于 0.1 | PLD v3 |
| **超参组合** | **不具备加性效应**，必须做消融实验验证 | PLD v3, AWSC v4 |

---

## 附录: results.json 完整算法概览

| 算法 | 完成/总计 | 最优配置 | best success_once | best success_at_end |
|------|:--------:|---------|:-----------------:|:-------------------:|
| aw_shortcut_flow | 39/40 | `aggressive` | 0.91 | 0.48 (`tau_0.001`) |
| awcp | 32/32 | `ema_0.99` | 0.81 | 0.49 |
| flow_matching | 16/18 | `pred_horizon_8` | 0.71 | 0.31 |
| cpql | 30/30 | `bc_weight_0.5` | 0.68 | 0.38 (`tau_0.01`) |
| shortcut_flow | 25/26 | `weights_1.0_1.0` | 0.64 | 0.26 |
| consistency_flow | 23/25 | `ema_0.9999` | 0.62 | 0.33 |
| reflected_flow | 22/24 | `reflection_soft` | 0.48 | 0.23 |
| diffusion_policy | 16/20 | `obs_horizon_4` | 0.45 | 0.23 |
