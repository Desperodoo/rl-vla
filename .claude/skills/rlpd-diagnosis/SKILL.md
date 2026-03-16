# rlpd-diagnosis — RLPD 训练"内科"诊断

当用户调用 `/rlpd-diagnosis` 时，对指定的 RLPD 在线训练实验进行深层诊断。分析训练过程中的 loss、Q-value、entropy 等内部指标，而非仅看最终 success rate。

**参数**：`/rlpd-diagnosis [wandb_project] [run_ids...]`
- 如未指定，读取最近的实验状态从 CLAUDE.md

---

## 五维诊断框架

每个 RLPD 实验从 5 个维度进行诊断，每个维度有明确的健康/异常判据：

### 维度 1: Critic 健康度

**数据源**：`train/critic/q_mean`, `train/critic/q_std`, `train/critic/td_target_mean`, `train/critic/critic_loss`

| 指标 | 健康 | 警告 | 危险 |
|------|------|------|------|
| Q-value 范围 | < 10 | 10-50 | > 50 |
| Critic loss (final 20%) | < 1.0 | 1-50 | > 50 |
| TD target std | < 1.0 | 1-10 | > 10 |
| Q-value 趋势 | 平稳/缓升 | 震荡 | 发散/暴涨 |

**诊断逻辑**：
- Q 范围 > 50 → "Critic 病态：Q-value 震荡"，通常因 gamma 过高或 reward scale 不匹配
- Critic loss 不收敛 → "Critic 未学到有用的 value function"
- TD target std > 10 → "Value estimation 不稳定"

### 维度 2: Actor 偏移

**数据源（AWSC）**：`train/actor/flow_loss`, `train/actor/shortcut_loss`, `train/actor/actor_loss`
**数据源（SAC/PLD/DSRL）**：`train/actor/actor_loss`, `train/actor/actor_entropy`, `train/actor/actor_q`

| 指标 | 健康 | 警告 | 危险 |
|------|------|------|------|
| Flow loss 趋势 (AWSC) | 缓降 | 急降 >50% | ↓>80% 且 SO 同步↓ |
| Actor entropy 范围 (SAC) | [-10, 10] | [-50, -10] | < -50 |
| Actor loss (SAC) | 缓降 | 震荡 | 快速变化 |

**诊断逻辑**：
- Flow loss ↓80% 且 SO ↓ → "策略过拟合 demo：BC loss 降低但泛化退化"
- Entropy < -50 → "策略坍缩：探索能力消失"
- Actor loss 震荡 → "Critic 信号不稳定传导到 actor"

### 维度 3: 探索 (Entropy & Temperature)

**数据源**：`train/temp/temperature`, `train/temp/entropy`, `train/temp/temperature_loss`

| 指标 | 健康 | 警告 | 危险 |
|------|------|------|------|
| Temperature 范围 | 0.1-0.5 | 0.05-0.1 | < 0.05 或 > 1.0 |
| Entropy 趋势 | 缓降至 target | 骤降后恢复 | 骤降不恢复 |
| Entropy min | > -20 | [-50, -20] | < -50 |

**诊断逻辑**：
- Temperature < 0.05 → "探索过度压缩"
- Entropy min < -50 → "历史上发生过策略坍缩"
- Temperature 快速上升 → "Entropy 远低于 target，系统在 struggle"

### 维度 4: Reward 信号

**数据源（AWSC）**：`train/smdp/online_cum_reward_mean`, `train/smdp/offline_cum_reward_mean`, `train/reward/acp_step_mean`, `train/reward/sim_step_mean`
**数据源（PLD/DSRL）**：通过 eval/return 和 Q-value 间接推断

| 指标 | 健康 | 警告 | 危险 |
|------|------|------|------|
| Online/Offline reward gap | < 10x | 10-100x | > 100x |
| ACP step reward | > 0.01 | 0.001-0.01 | < 0.001 |
| ACP/sim reward 比例 | > 0.3 | 0.1-0.3 | < 0.1 |

**诊断逻辑**：
- Gap > 100x → "Critic 被 offline demo 主导，online 信号被忽视"
- ACP reward ≈ 0 → "ACP reward scale 不足，TD reward 信号死亡"

### 维度 5: Advantage 加权 (AWSC 特有)

**数据源**：`train/actor/advantage_mean`, `train/actor/advantage_std`, `train/actor/weight_mean`, `train/actor/weight_max`, `train/actor/n_demo_samples`, `train/actor/n_online_kept`

| 指标 | 健康 | 警告 | 危险 |
|------|------|------|------|
| Advantage mean | [-0.5, 0.5] | [0.5, 1.0] | > 1.0 |
| Weight mean | 0.8-1.2 | < 0.5 或 > 2.0 | — |
| Weight max | < 5.0 | 5-20 | > 20 |

**诊断逻辑**：
- Advantage mean ≈ 1.0 → "Critic 无法区分好坏 action，advantage 无区分力"
- Weight max > 20 → "少数样本被过度放大，训练不稳定"

---

## 执行步骤

### Step 1: 获取 WandB 数据

```bash
# 获取指定实验的训练 history
http_proxy=http://10.20.93.149:7890 https_proxy=http://10.20.93.149:7890 \
conda run -n rlft_ms3 --no-capture-output \
env PYTHONPATH=/home/wjz/rl-vla \
python scripts/sweep_acp/fetch_wandb.py \
    --project {WANDB_PROJECT} \
    --run_ids {RUN_IDS} \
    --output_dir logs/vlaw/wandb_analysis/{OUTPUT_NAME} \
    --save_csv
```

### Step 2: 五维诊断

对每个实验的 CSV 数据，按五维框架逐一检查。重点关注：

1. **Q-value 尺度**：比较 Q_mean 的范围。如果 > 50，critic 可能病态
2. **Reward gap**：计算 online/offline cumulative reward 比值
3. **Entropy 异常**：找 entropy 最小值，判断是否发生过策略坍缩
4. **Loss 趋势**：flow_loss / critic_loss 是否正确收敛
5. **Advantage 偏差**：advantage_mean 是否偏离 0

### Step 3: 生成诊断图表

运行 `scripts/analyze_rlpd_internals.py` 或基于其模式创建针对性分析脚本：

```bash
PYTHONPATH=/home/wjz/rl-vla python scripts/analyze_rlpd_internals.py
```

生成 7 类诊断图：
- **Critic Health**: Q-value + critic loss 时序图（每算法一列）
- **Collapse Diagnosis**: 受影响实验的多指标并排（1 实验 6 panel）
- **Actor Internals**: Loss 分解 + advantage 动态（AWSC 特有）
- **Reward Gap**: Online vs offline cumulative reward + ratio
- **Entropy/Temperature**: PLD/DSRL 的探索情况
- **Loss vs Eval**: Flow loss 与 success rate 的对照（检测过拟合）
- **Q-value Scale**: 跨算法 Q-value 尺度对比

### Step 4: 输出诊断报告

以 Markdown 格式输出：
1. **五维评分卡**：每个实验 A/B/C/D/F 评级
2. **病因诊断**：用因果链描述问题根因
3. **定量证据**：关键指标的数值
4. **处方**：对应的调参建议

---

## 常见病因速查表

| 症状 | 可能病因 | 证据指标 | 处方 |
|------|---------|---------|------|
| SO 退化但 flow_loss 下降 | 策略过拟合 demo | flow_loss ratio < 0.3 | 增大 bc_weight 或 early stop |
| SAE ≈ 0% 但 SO 高 | Reward 信号淹没 | ACP/Q ratio < 5% | 增大 acp_reward_scale |
| SO 灾难性崩溃 | Critic 引导策略偏移 | Q膨胀 + entropy骤降 | 加 BC loss 或降 gamma |
| Q-value 震荡 | gamma 过高 + reward 不稳定 | Q range > 50 | 降 gamma |
| Advantage mean ≈ 1.0 | Critic 无区分力 | advantage_std 过低 | 增大 online_ratio |
| Online reward ≈ 0 | ACP reward scale 不足 | reward gap > 100x | 增大 scale |

---

## 参考

- 方法论来源：ADR-042（AWSC+ACP 数据驱动诊断）
- 分析脚本模板：`scripts/analyze_rlpd_internals.py`
- WandB 数据获取：`scripts/sweep_acp/fetch_wandb.py`
- 已有诊断报告：
  - `docs/vlaw/acp_v3_rlpd_internals_report.md` — ACP v3 内科分析
  - `logs/vlaw/wandb_analysis/awsc_acp_mirror/analysis_report.md` — ACP mirror AWSC 分析
