# VLAW × Online RL: 策略微调方案分析与计划

> **日期**: 2026-03-01
> **背景**: Iter-1 Weighted FM 策略严重退化 (78.1% → 17.2%)，需要重新审视策略微调方案
> **目标**: 分析 VLAW 官方方案 + 现有 3 种 Online RL 方案，提出融合 WM+VLM 的改进计划

---

## 一、VLAW 官方策略微调方案回顾

### 1.1 Algorithm 1 — Weighted Flow Matching (Filtered BC)

VLAW 论文 (arXiv:2602.12063) Section 4.1, Eq. 4:

$$\mathcal{L}_{WFM} = \mathbb{E}_{(o,a) \sim D_{real+} \cup D_{syn+}} [\mathcal{L}_{FM}(\theta; o, a)]$$

**本质**：在 VLM 标记为成功的轨迹 (D_real+ ∪ D_syn+) 上做标准 Flow Matching 监督学习。
- **不是 RL**：无 critic、无 Q 值、无环境交互
- **等价于 Filtered BC**：只用成功轨迹做 behavior cloning
- **权重来源**：VLM 二值过滤 (vlm=1 → weight=1, vlm=0 → weight=0)

### 1.2 论文中的训练细节

| 参数 | 值 | 说明 |
|------|-----|------|
| 策略模型 | π₀.₅ (Transformer + FM, ~3B) | 大模型更鲁棒 |
| 更新步数 | 2000 | 每轮迭代 |
| Batch size | 256 | |
| 学习率 | 1e-5 | |
| 数据混合 | D_real+ : D_syn+ ≈ 1:1 | |
| 数据量 | D_real + D_syn ≈ 数千条/任务 | DROID 95K 轨迹基数 |

### 1.3 Iter-1 失败分析

在我们的复现中，Weighted FM 严重退化的**根本原因**：

| 因素 | VLAW 原版 | 我们的复现 | 影响 |
|------|-----------|-----------|------|
| **策略规模** | π₀.₅ ~3B 参数 | ShortCut Flow ~1.5M 参数 | 小模型更易被少量数据破坏 |
| **数据量** | D_real 50条 + D_syn 数百条成功 | D_real 4条(vlm=1) + D_syn **0条**(vlm=0) | Effective data 几乎为零 |
| **预训练基数** | DROID 95K 多任务 | 25 条单任务 demo | 预训练知识更脆弱 |
| **微调方法** | Full fine-tune (大模型容忍) | Full fine-tune (小模型灾难性遗忘) | |

**关键洞察**：VLAW 论文的 Weighted FM 之所以有效，依赖于：
1. 大规模预训练模型 (3B) 本身对微调鲁棒
2. D_syn+ 数据量充足 (WM 质量足够生成 VLM-正向轨迹)
3. D_real 数据量大 (DROID 95K 基础上 50 条/task rollout，背景丰富)

**当我们的 D_syn+ = 0 时，纯 Weighted FM 退化为在极少数据上的 over-fitting，必然灾难性遗忘。**

---

## 二、现有 Online RL 微调方案分析

### 2.1 三种方案对比

| 特性 | RLPD (SAC/AWSC) | DSRL-SAC | PLD-SAC |
|------|-----------------|----------|---------|
| **核心思想** | Demo + Online 混合训练 | Noise-space RL (冻结 base) | Residual RL (冻结 base) |
| **Base policy** | 可学，可冻结 | ❄️ 冻结 | ❄️ 冻结 |
| **动作空间** | 原始 action space | ShortCut Flow 的 noise space | 残差 δa ∈ [-ξ, +ξ] |
| **Critic** | Ensemble Q (10 Qs) | Ensemble Q (10 Qs) | Ensemble Q (5 Qs) |
| **Demo 使用** | 混合 replay buffer (or=0.15) | 无 (base 已提供探索) | 冻结 rollout 作为 offline data |
| **网络规模** | SAC: 3×256 / AWSC: UNet + 3×256 | 3×2048 MLP | 3×1024 MLP |
| **最佳表现** | AWSC: **91%** (aggressive) | 依据 sweep 数据 | 依据 sweep 数据 |
| **总训练步** | 250K-500K env steps | 1M env steps | 500K env steps |
| **UTD ratio** | 20 | 60 | 60 |
| **灾难性遗忘** | AWSC: bc_weight=2.0 防止 | 基底冻结，无遗忘 | 基底冻结，无遗忘 |
| **环境交互** | ✅ 必须 | ✅ 必须 | ✅ 必须 |

### 2.2 RLPD + AWSC (最有潜力的方案)

**AWSC (Advantage-Weighted ShortCut Flow) 是本项目最强基线**：`success_once = 91%`

核心组件：
1. **Velocity Network (ShortCut Flow UNet)**: 与 VLAW 相同的策略架构
2. **Ensemble Q-Network**: 提供 Q(s,a) 估计
3. **Advantage Weighting**: $w(s,a) = \exp(\beta \cdot A(s,a))$, 用 Q 值加权 FM loss
4. **BC Regularization**: bc_weight=2.0 保留预训练知识
5. **EMA**: ema_decay=0.9995 缓慢追踪在线权重

**为什么 AWSC 已经解决了 VLAW 的核心问题**：
- `bc_weight=2.0` → 防止灾难性遗忘 (VLAW Weighted FM 没有)
- `advantage_weighting` → 类似 VLM 的成功/失败区分，但用 Q 值量化
- `online_ratio=0.15` → 主要从 demo 学习，在线数据仅占 15%
- Ensemble Q → 保守 Q 估计，避免 over-optimistic policy update

### 2.3 DSRL-SAC (Noise-Space RL)

**关键特点**：在 ShortCut Flow 的 noise space 中训练 SAC
- 基底策略完全冻结 → 零遗忘风险
- 学到的是"噪声偏差"，等价于策略空间中的校正
- `action_magnitude=2.5` → 探索范围受限但安全
- 无需 demo 数据——冻结的 base policy 提供足够初始探索质量

### 2.4 PLD-SAC (Residual RL)

**关键特点**：学习残差动作 δa，叠加到 base policy 输出上
- `action_scale=0.3` → 生理上限制偏离幅度
- Cal-QL critic pretraining → 利用离线数据初始化 Q 值
- Probing α=0.6 → episode 开头用 base policy，后续加入残差
- 更轻量 (3×1024 vs 3×2048) → 更适合小数据场景

---

## 三、WM + VLM 如何提升 Online RL

VLAW 框架提供两个核心资产：
1. **World Model (Ctrl-World)**: 生成合成轨迹 D_syn
2. **VLM Reward Model**: 提供二值成功/失败奖励信号

这两个资产可以从多个维度提升上述 Online RL 方法的样本效率：

### 3.1 方案矩阵

| 增强维度 | 具体手段 | 适用方法 | 复杂度 |
|----------|---------|---------|--------|
| **A. 合成数据增强** | D_syn+ 加入 demo buffer | RLPD/AWSC | ★★ |
| **B. VLM Reward Shaping** | VLM p(yes) 作为额外 reward 信号 | DSRL/PLD/RLPD | ★★★ |
| **C. Model-based Value Estimation** | WM rollout 用于 Q-value bootstrap | 所有方法 | ★★★★ |
| **D. Imagination Pretraining** | D_syn 用于 critic 预训练 (Cal-QL) | PLD/DSRL | ★★ |
| **E. Iterative WM→Policy 改进** | 策略改善 → 更好 rollout → WM 改善 → 更好合成数据 | 所有方法 (VLAW 核心) | ★★★ |

### 3.2 详细方案描述

#### 方案 A: AWSC + WM 数据增强 (推荐首选)

```
融合逻辑:
  1. 正常 AWSC 训练 (env 交互 + demo replay)
  2. 每 N 步, 用当前策略通过 WM 生成 D_syn
  3. VLM 过滤 D_syn → D_syn+ (成功轨迹)
  4. D_syn+ 加入 demo replay buffer (作为"伪 demo")
  5. AWSC 继续训练, bc_weight 覆盖 D_demo + D_syn+
  
优势:
  - bc_weight 天然保护 base policy → 无灾难性遗忘
  - D_syn+ 扩充了 demo buffer → 更好的 BC 正则化
  - Q-critic 用环境真实 reward → 无 VLM FP 污染
  - AWSC 已是 91% baseline → 进一步提升空间明确

实现改动:
  - train_rlpd.py: 添加 D_syn 动态加载 (定期从 WM 生成)
  - 或 offline 方式: 先生成 D_syn, 再作为额外 demo_path 传入
```

#### 方案 B: DSRL/PLD + VLM Reward Shaping

```
融合逻辑:
  1. 标准 DSRL/PLD 训练 (env 交互)
  2. 每个 episode 结束, 用 VLM 对该 rollout 打分
  3. VLM score 作为额外 reward 信号:
     r_total = r_env + λ_vlm * r_vlm  (λ_vlm 作为超参)
  4. 或: VLM 成功的 episode 加入 success buffer (AWSC 已有此机制)
  
优势:
  - 不依赖 WM 合成质量 → 避免 D_syn=0 问题
  - VLM 信号可补充 dense reward 不足的场景
  - DSRL/PLD 冻结 base → 零遗忘
  
局限:
  - VLM 推理成本高 (~2s/trajectory on 4090)
  - Online 场景下 VLM 评估瓶颈明显
  - LiftPegUpright 已有精确 env reward, VLM 附加值有限
```

#### 方案 C: Imagination 辅助 Critic 预训练

```
融合逻辑:
  1. 用 WM 生成 D_syn (无需 VLM 过滤)
  2. VLM 对 D_syn 打二值 reward (替代 env 的稠密 reward)
  3. D_syn + VLM reward 用于 Cal-QL critic 预训练
  4. 预训练后的 critic 提供更好的 Q 初值 → 在线 RL 更快收敛
  
优势:
  - 充分利用 WM 生成能力
  - 不需要 env 交互就能预训练 critic
  - 与 PLD 的 Cal-QL 流程天然兼容
  
局限:
  - 需要 WM 质量足够 (当前 D_syn vlm=0)
  - VLM FP 会污染 critic
```

#### 方案 D: Full VLAW Iterative Loop + Online RL

```
融合逻辑（最完整版）:
  for i = 1 to K_iter:
    Step 1-3: Rollout → VAE → VLM 标注 (与 VLAW 相同)
    Step 4:   WM 微调 (与 VLAW 相同)
    Step 5-6: Imagination → VLM 标注 (与 VLAW 相同)
    Step 7':  策略更新 = AWSC/DSRL/PLD (替代 Weighted FM)
              - D_syn+ 作为额外 demo/reward 信号
              - 环境交互提供 on-policy 数据 + 真实 reward
    Step 8:   评估 (与 VLAW 相同)
    
核心区别 vs 纯 VLAW:
  - Step 7 从 Filtered BC → Online RL + 环境交互
  - 合成数据是"辅助"而非"唯一"训练信号
  - Critic 提供细粒度 Q 值, 优于 VLM 二值标签
```

---

## 四、推荐计划

### 4.1 优先级排序

| 优先级 | 方案 | 理由 | 预估时间 | GPU |
|--------|------|------|---------|-----|
| **P1** | **AWSC baseline** | 先跑裸 AWSC 确认 91% 可复现, 确立最强 baseline | 4-6h | 8-9 |
| **P2** | **AWSC + D_syn 增强** | 最小改动: D_syn+ 加入 demo buffer | 需先解决 D_syn=0 | 8-9 |
| **P3** | **PLD + Cal-QL (with D_syn)** | D_syn 用于 critic 预训练, 改善 Q 初值 | 4-6h | 8-9 |
| **P4** | **DSRL baseline** | 冻结 base 方案, 完全无遗忘 | 4-6h | 8-9 |
| **P5** | **Full Loop (AWSC)** | WM 迭代改进 + AWSC 在线微调 | 需 P1+P2 结果 | 全部 |

### 4.2 Phase 1: 建立 Online RL Baselines (前置)

> **在尝试 WM+VLM 增强之前，先确认裸 Online RL 的表现。**

| Task ID | 任务 | 方法 | GPU | 训练步 | 预估时间 |
|---------|------|------|-----|--------|---------|
| T-RL-BASELINE-AWSC | AWSC baseline on LiftPegUpright | train_rlpd --algorithm awsc | 8-9 | 250K | 4-6h |
| T-RL-BASELINE-PLD | PLD-SAC baseline on LiftPegUpright | train_pld | 8-9 | 500K | 6-8h |
| T-RL-BASELINE-DSRL | DSRL-SAC baseline on LiftPegUpright | train_dsrl | 8-9 | 500K | 6-8h |

**目标**: 复现 results.json 中的最佳结果 — AWSC aggressive=91%

### 4.3 Phase 2: WM + VLM 增强 Experiments

#### 实验 2.1: AWSC + D_syn Demo Buffer

```python
# 修改 train_rlpd.py:
# 新增参数 --synthetic_demo_path 接收 D_syn+ HDF5
# D_syn+ 加入 OfflineRLDataset, 与原始 demo 混合
# 无需修改 AWSC 算法本身
```

**前置条件**: `D_syn+` 非空 (当前 0/200, 需先改善 WM 或降低 VLM 阈值)

#### 实验 2.2: PLD + Imagination Cal-QL

```python
# 修改 train_pld.py:
# Cal-QL pretrain 阶段加入 WM imagination rollouts
# VLM reward 作为二值 reward 信号
# critic 学习: Q(s, a_base+δ) → VLM_reward + env_future_reward
```

#### 实验 2.3: AWSC Iterative (Full VLAW + Online RL)

```
每轮迭代:
  1. AWSC 训练 N 步 (用当前 demo + 当前 D_syn+)
  2. 从训练过程中收集 D_real (成功和失败 episodes)
  3. WM 微调 on D_real (+ D_demo)
  4. Imagination → D_syn → VLM 过滤 → D_syn+
  5. D_syn+ 加入 demo buffer, 继续 AWSC 训练
```

### 4.4 Phase 3: 分析与消融

| 实验 | 变量 | 目标 |
|------|------|------|
| D_syn 比例消融 | D_syn% = 0/10/30/50 in demo buffer | 确认 WM 数据是否真有正面贡献 |
| VLM 阈值消融 | α = 0.5/0.6/0.7/0.8 | 平衡 D_syn 数量 vs 质量 |
| Online vs Pure WFM | AWSC vs Weighted FM (相同数据) | 验证 RL critic 的增量价值 |
| WM 质量 vs RL 提升 | 不同 WM ckpt × 相同 AWSC | 验证 WM 改善是否传导到策略 |

---

## 五、关键技术问题与解答

### Q1: 为什么 Online RL 不会像 Weighted FM 一样灾难性遗忘？

**AWSC**: `bc_weight=2.0` 的 FM loss 确保策略始终锚定在 demo 分布附近，即使 Q-critic 给出错误信号。

**DSRL/PLD**: Base policy 完全冻结，RL agent 只学增量/噪声。即使 RL agent 质量为零，组合输出至少等于 base policy。

### Q2: ManiSkill 仿真中 WM 的价值是什么？

ManiSkill 提供精确的 `env.step()` 和 `info["success"]`，WM 并非必须。但 WM 的价值在于：
1. **数据效率**: `env.step()` 需要 250K-500K 步交互，WM imagination 可以离线生成数千条轨迹
2. **迭代改进验证**: 验证 VLAW 核心假说 — WM 和策略能否通过迭代共同改进
3. **未来迁移**: 验证后可迁移到真实机器人 (无 env.step() 的场景)

### Q3: 在 ManiSkill 仿真中用 VLM reward 还是 env reward？

**推荐**：Online RL 用 `env reward` (精确且无 FP)，VLM 仅用于 D_syn 过滤。

| 场景 | 奖励源 | 理由 |
|------|--------|------|
| Online RL (env 交互) | env dense reward | 精确、无 FP、稠密 |
| D_syn 过滤 | VLM binary | 合成数据无 env reward |
| 训练数据权重 | VLM p(yes) as soft weight | 可选增强 |

### Q4: 是否需要修改 WM 训练来支持 Online RL？

**不需要**。WM 训练与策略微调方法无关。只是将 WM 的输出 (D_syn) 用于不同的 downstream task（从 Filtered BC 改为 demo buffer augmentation）。

---

## 六、实施路线图

```
Week 1: Online RL Baselines
  ├── Day 1-2: T-RL-BASELINE-AWSC (4-6h 训练 + 评估)
  ├── Day 2-3: T-RL-BASELINE-PLD  (6-8h 训练 + 评估)
  ├── Day 3-4: T-RL-BASELINE-DSRL (6-8h 训练 + 评估)
  └── Day 4:   对比报告 (AWSC vs PLD vs DSRL vs WFM vs Base)

Week 2: WM + VLM 增强 (基于 Week 1 最佳方法)
  ├── Day 1-2: 实现 D_syn demo buffer 增强代码
  ├── Day 2-3: AWSC + D_syn 实验 (需先有 D_syn+ > 0)
  ├── Day 3-4: PLD + Imagination Cal-QL
  └── Day 4-5: 消融实验

Week 3: Full Iterative Loop
  ├── 选择 Week 2 最佳方案
  ├── 实现 VLAW Loop 中的 Step 7' 替换
  └── 跑完整 2 轮迭代
```

---

## 七、总结

| 维度 | VLAW 原版 (Weighted FM) | 推荐方案 (AWSC + WM 增强) |
|------|------------------------|--------------------------|
| 策略微调 | Filtered BC (无 RL) | AWSC (有 Critic, 有 BC 正则) |
| 防遗忘机制 | 无 (依赖大模型鲁棒性) | bc_weight=2.0 + demo replay |
| 环境交互 | 无 (纯离线) | 有 (online_ratio=0.15) |
| D_syn 角色 | 唯一训练信号之一 | 辅助 demo buffer |
| D_syn=0 时 | 灾难性失败 (已验证) | 仍可正常训练 (有 env 交互) |
| Q 值估计 | 无 | Ensemble Q (10 networks) |
| 最佳表现上限 | 取决于 D_syn 质量 | **91%** (裸 AWSC 已验证) |
| 实现复杂度 | 低 (已有) | 中 (需集成 D_syn 路径) |

**核心结论**：

1. **单纯 Weighted FM 在小数据 + 小模型场景下不可行**，已被 Iter-1 实验证明
2. **AWSC 是最有前景的替代方案**：它已经内置了防遗忘机制 (bc_weight)、优势加权 (类似 VLM 过滤)、和在线探索
3. **WM + VLM 的增量价值应体现在 D_syn 数据增强上**，而非替代 RL critic 的信号
4. **建议先建立 Online RL baseline，再叠加 WM+VLM 增强**，以精确量化每个组件的贡献
