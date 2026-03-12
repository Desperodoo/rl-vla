# ACP Mirror Experiments — 分析报告

> 日期：2026-03-11 | 入口脚本：`scripts/run_acp_mirror_experiments.sh` | WandB project: `rlpd-acp-mirror`

## 1. 实验设计

用 ACP v2_combined reward（TD-shaped: `r = (V(s') - V(s)) × reward_scale`）**完全替代** ManiSkill sim dense reward，在 3 种算法上运行，与 `runs/fair_comparison/` 的 sim-reward 基线直接对比。

| 项目 | 配置 |
|------|------|
| Pretrained checkpoint | `runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt` |
| ACP checkpoint | `checkpoints/vlaw/acp/v2_combined/best.safetensors`（A+B+C+D 数据，1250 条轨迹） |
| ACP reward_scale | 100.0 |
| 任务 | LiftPegUpright-v1 |
| num_envs / num_eval_envs | 50 / 50 |
| max_episode_steps | 100 |
| seed | 42（单次运行） |

### 算法超参（与 fair_comparison 完全一致）

| 算法 | total_steps | UTD | 特有参数 |
|------|-------------|-----|---------|
| AWSC | 500K | 20 | beta=50, bc_weight=2.0, act_horizon=8 |
| PLD-SAC | 71K | 60 | action_scale=0.3, gamma=0.99, calql_pretrain=1000 |
| DSRL-SAC | 71K | 60 | action_magnitude=2.5, gamma=0.95, log_std_init=-5.0 |

---

## 2. 结果总览

### 2.1 success_once（轨迹内任意时刻成功）

| 算法 | Reward | Best SR | @Step | Final SR | @Step |
|------|--------|---------|-------|----------|-------|
| AWSC | **Sim** | **92%** | 58K | 76% | 500K |
| AWSC | ACP | 90% | 140K | 62% | 490K |
| PLD-SAC | **Sim** | **100%** | 40K | **98%** | 70K |
| PLD-SAC | ACP | 82% | 0 (pretrained) | 58% | 70K |
| DSRL-SAC | **Sim** | **98%** | 54K | **96%** | 70K |
| DSRL-SAC | ACP | 92% | 20K | 88% | 70K |

### 2.2 success_at_end（最终时刻成功 — Online RL finetuning 核心指标）

| 算法 | Reward | Best SR | @Step | Final SR | @Step |
|------|--------|---------|-------|----------|-------|
| **AWSC** | **Sim** | **72%** | 281K | 56% | 500K |
| **AWSC** | **ACP** | **66%** | 460K | **56%** | 490K |
| PLD-SAC | **Sim** | **86%** | 70K | **86%** | 70K |
| PLD-SAC | ACP | 2% | 0 | 0% | 70K |
| DSRL-SAC | **Sim** | **60%** | 70K | **60%** | 70K |
| DSRL-SAC | ACP | 6% | 40K | 2% | 70K |

### 2.3 关键发现

**success_at_end 维度的发现与 success_once 截然不同：**

1. **AWSC + ACP 是唯一在 success_at_end 上接近 sim 基线的算法**
   - Best: 66% (ACP) vs 72% (sim) — 差距仅 6%
   - Final: 56% (ACP) = 56% (sim) — **完全持平**
   - AWSC 在 success_at_end 上的表现远好于其 success_once 退化趋势所暗示的

2. **PLD-SAC + ACP 的 success_at_end 始终为 ~0%**
   - success_once 维持 58-82%（能触碰到 peg），但 success_at_end = 0%（无法保持竖直）
   - 说明 ACP reward 未能教会 PLD 策略"保持"的行为
   - Sim 基线 PLD 的 success_at_end = 86% 说明 PLD 算法本身有此能力

3. **DSRL-SAC + ACP 同样 success_at_end ~0%**
   - success_once 92%（看起来很好），但 success_at_end 仅 6%
   - 与 sim 基线（60%）差距巨大
   - DSRL 的保守更新策略保护了 success_once 但无法弥补 success_at_end 缺失

---

## 3. 完整训练曲线

### 3.1 AWSC + ACP（GPU 0+1，500K steps）

```
eval/success_at_end:
step         0: 2%    (pretrained baseline)
step    40,400: 12%   (开始学习)
step   110,400: 60%   ← 首次突破
step   280,400: 58%   (plateau)
step   330,400: 60%
step   440,400: 62%
step   460,400: 66%   ← best
step   490,400: 56%   (final)

eval/success_once:
step         0: 80%   (pretrained baseline)
step   140,400: 90%   ← best
step   240,400: 58%   (开始退化)
step   490,400: 62%   (final)
```

特征：success_at_end 持续上升（2%→66%），但 success_once 中后期退化（90%→62%）。
解读：策略学会了更好地"保持竖直"（success_at_end↑），但牺牲了"到达"的能力（success_once↓）。

### 3.2 PLD-SAC + ACP（GPU 2+3，71K steps）

```
eval/success_at_end:
step         0: 2%    → 全程 0-2%，完全未学会

eval/success_once:
step         0: 82%   (pretrained)
step    10,000: 34%   ← catastrophic forgetting
step    40,000: 72%   (部分恢复)
step    70,000: 58%   (final, 仍低于 pretrained)
```

### 3.3 DSRL-SAC + ACP（GPU 4+5，71K steps）

```
eval/success_at_end:
step         0: 0%     → 全程 0-6%，基本未学会

eval/success_once:
step         0: 82%    (pretrained)
step    20,000: 92%    ← best (improved!)
step    60,000: 74%    (退化)
step    70,000: 88%    (final)
```

---

## 4. ACP Reward 信号诊断

### 4.1 `acp_step_mean` = 0 的真相

AWSC 实验中 `train/reward/acp_step_mean` 在 step 5200 后降至 0.0（1237/1238 个数据点为 0），曾怀疑 ACP value model 输出常量。

**诊断过程：**

| 测试 | 结果 | 结论 |
|------|------|------|
| Random images → value model | V 值 std=0.024, range=[-0.39, -0.31] | ✅ 非常量 |
| Brightness gradient → value model | V 值随 brightness 变化 [-0.56, -0.54] | ✅ 有区分度 |
| ManiSkill env 50步 → value model | V 值 std=0.059, range=[-0.90, -0.66] | ✅ 正常 |
| Softmax logits 熵 | 2.86 / 5.30 (54% of max) | ✅ 健康分布 |
| 50 envs wrapper 模拟 | reward mean=±0.5, std=2.8, n_zero=0/50 | ✅ 非零 |
| 近零动作 (smooth policy) | reward mean=±0.3, std=1.2, n_zero=0/50 | ✅ 非零 |
| `online_cum_reward_mean` (buffer) | 0.5→0.04 (全程非零) | ✅ 实际训练接收到奖励 |

**结论：`acp_step_mean = 0` 是日志记录层面的问题（疑似 TensorBoard writer 的标量值丢失），不影响实际训练。**

证据链：
1. ACP value model 在所有测试配置下均产生非零、有区分度的输出
2. `DualCameraRewardWrapper` 在完整 wrapper 栈（含 FlattenRGBDObservationWrapper）中正确返回非零奖励
3. `online_cum_reward_mean`（从 replay buffer 采样的累计奖励）全程非零
4. 策略在 ACP reward 下确实学到了 success_at_end 的提升（2% → 66%）

### 4.2 ACP Reward 量级特征

| 场景 | 每步 reward mean | 每步 reward std | 100步累计 |
|------|-----------------|----------------|-----------|
| Random policy | ±0.5 | 2.8 | ~3-5 |
| Near-zero actions | ±0.3 | 1.2 | ~1-2 |
| Trained policy (buffer) | ~0.001 | — | 0.01-0.04 |

随训练进行，策略收敛于稳定状态，V(s') - V(s) 趋近于 0 是**正常行为**。TD reward 的本质决定了：当策略达到 value landscape 的 plateau 时，reward signal 自然衰减。

---

## 5. 根因分析：为什么 ACP reward 不足以替代 sim reward？

### 5.1 success_at_end 差距的根因

ACP value model 的训练目标是 `env_success`（二值: 0/1），通过 MC return 计算 per-frame value target：
- 成功帧：`V → 0`（轨迹末尾的 value 为 0）
- 失败帧：`V → -1 * c_fail_coef`

**核心问题：ACP 的 value target 不区分 success_once 和 success_at_end。**

LiftPegUpright 中 `env_success = True` 在任意时刻 peg 竖直时就触发（success_once 语义），但 RL finetuning 需要的是 success_at_end（最终保持竖直）。ACP reward 引导策略"到达"竖直状态，但不激励"保持"。

### 5.2 Per-algorithm 差异

| 算法 | 核心特性 | ACP reward 适配性 | 结果 |
|------|---------|-----------------|------|
| **AWSC** | BC loss 提供策略约束 + conservative critic | BC loss 提供隐式的 success_at_end 信号（demo 数据包含完整轨迹） | ⚠️ 部分成功（success_at_end=66%） |
| **PLD-SAC** | Pure RL, no BC constraint | 完全依赖 ACP reward，缺乏保持信号 → 灾难性遗忘 | ❌ 失败（0%） |
| **DSRL-SAC** | Action magnitude clipping | 保守更新保护了 success_once，但 ACP 无法教 success_at_end | ❌ 失败（2%） |

AWSC 在 success_at_end 上的成功，很可能是因为其 **BC loss（bc_weight=2.0）** 从 demo 回放中学到了保持行为，而非 ACP reward 的功劳。

---

## 6. 结论与建议

### 结论

| 维度 | 结论 |
|------|------|
| **success_once** | ACP reward 使 DSRL 维持 92%，AWSC 达到 90% 但退化，PLD 灾难性遗忘 |
| **success_at_end**（核心） | 仅 AWSC 达到 66%（得益于 BC loss），PLD/DSRL 均为 ~0% |
| **对比 sim baseline** | ACP 在 success_at_end 上全面劣于 sim reward（AWSC 持平，PLD/DSRL 差距 >50%） |
| **ACP reward 信号** | Value model 输出正常，`acp_step_mean=0` 为日志 bug，不影响训练 |
| **根本限制** | ACP value 目标为 success_once 语义，无法引导 success_at_end 行为 |

### 建议

1. **修改 ACP value target 为 success_at_end**：将 `success_key` 从 `env_success`（success_once）改为对应 success_at_end 的 GT label，重训 ACP value model
2. **尝试 acp_blend 模式**：保留部分 sim reward（`reward_mode=acp_blend, blend_weight=0.5`），测试是否可以兼得 ACP 的泛化性和 sim 的精确性
3. **修复日志 bug**：在 `train_rlpd.py` 中排查 `acp_step_mean` 始终为 0 的原因（可能是 TensorBoard writer flush timing 或 defaultdict 行为）
4. **多 seed 验证**：当前仅 seed=42 单次运行，AWSC success_at_end 66% vs sim 72% 的差异可能在统计误差范围内

### 实验数据路径

| 实验 | Run 路径 |
|------|---------|
| AWSC + ACP | `runs/awsc_acp_mirror_s42__1773208674/` |
| PLD + ACP | `runs/pld_acp_mirror_s42__1773208675/` |
| DSRL + ACP | `runs/dsrl_acp_mirror_s42__1773208675/` |
| AWSC sim | `runs/fair_comparison/awsc/best_s42__1772570560/` |
| PLD sim | `runs/fair_comparison/pld/best_s42__1772557687/` |
| DSRL sim | `runs/fair_comparison/dsrl/best_s42__1772557691/` |
