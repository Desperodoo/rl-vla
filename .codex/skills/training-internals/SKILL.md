# training-internals

此技能用于诊断 RLPD/ACP 在线训练实验的内部健康状况，聚焦 loss、Q 值、entropy、reward signal 与 advantage weighting。

适用对象：
- `/training-internals`
- 任何“帮我诊断在线训练内部指标是否健康”的请求

## 1. 输入约定

格式：

```text
/training-internals [wandb_project] [run_ids...]
```

若未指定项目：
- 从仓库当前实验上下文中推断

若未指定 run_ids：
- 默认分析项目下所有相关 runs

## 2. 五维诊断框架

### Dimension 1: Critic Health

数据源：
- `train/critic/q_mean`
- `train/critic/q_std`
- `train/critic/td_target_mean`
- `train/critic/critic_loss`

| 指标 | Healthy | Warning | Danger |
|------|---------|---------|--------|
| Q-value range | < 10 | 10-50 | > 50 |
| Critic loss（final 20%） | < 1.0 | 1-50 | > 50 |
| TD target std | < 1.0 | 1-10 | > 10 |
| Q-value trend | 稳定或渐升 | 振荡 | 发散 |

### Dimension 2: Actor Drift

AWSC 数据源：
- `train/actor/flow_loss`
- `train/actor/shortcut_loss`

SAC / PLD / DSRL 数据源：
- `train/actor/actor_loss`
- `train/actor/actor_entropy`

| 指标 | Healthy | Warning | Danger |
|------|---------|---------|--------|
| Flow loss trend（AWSC） | 缓慢下降 | 降幅 > 50% | 降幅 > 80% 且 SO 下滑 |
| Actor entropy（SAC） | [-10, 10] | [-50, -10] | < -50 |

### Dimension 3: Exploration

数据源：
- `train/temp/temperature`
- `train/temp/entropy`

AWSC 无此维度。

| 指标 | Healthy | Warning | Danger |
|------|---------|---------|--------|
| Temperature | 0.1-0.5 | 0.05-0.1 | < 0.05 或 > 1.0 |
| Entropy min | > -20 | [-50, -20] | < -50 |

### Dimension 4: Reward Signal

AWSC 数据源：
- `train/smdp/online_cum_reward_mean`
- `train/smdp/offline_cum_reward_mean`
- `train/reward/acp_step_mean`

PLD / DSRL：
- 可结合 `train/critic/q_mean` 尺度间接判断

| 指标 | Healthy | Warning | Danger |
|------|---------|---------|--------|
| Online / Offline reward gap | < 10x | 10-100x | > 100x |
| ACP step reward | > 0.01 | 0.001-0.01 | < 0.001 |

### Dimension 5: Advantage Weighting

仅 AWSC 使用：
- `train/actor/advantage_mean`
- `train/actor/advantage_std`
- `train/actor/weight_max`

| 指标 | Healthy | Warning | Danger |
|------|---------|---------|--------|
| Advantage mean | [-0.5, 0.5] | [0.5, 1.0] | > 1.0 |
| Weight max | < 5.0 | 5-20 | > 20 |

## 3. 执行步骤

### Step 1：抓取 WandB 数据

```bash
http_proxy=http://10.20.93.149:7890 https_proxy=http://10.20.93.149:7890 \
conda run -n rlft_ms3 --no-capture-output \
env PYTHONPATH=/home/amax/rl-vla \
python scripts/sweep_acp/fetch_wandb.py \
  --project {WANDB_PROJECT} \
  --run_ids {RUN_IDS} \
  --output_dir logs/vlaw/wandb_analysis/{PROJECT} \
  --save_csv
```

### Step 2：运行自动诊断

```bash
PYTHONPATH=/home/amax/rl-vla python scripts/analyze_training_internals.py \
  --project {WANDB_PROJECT} \
  --data_dir logs/vlaw/wandb_analysis/{PROJECT} \
  --output_dir docs/vlaw/figures/{PROJECT}_internals \
  --no_fetch_wandb
```

自动输出应包括：
- algorithm auto-detect
- 五维评分
- 诊断图
- markdown 报告

### Step 3：人工补充

重点补充自动分级不容易表达的内容：
- 跨实验比较
- 结合项目上下文的优先级建议
- 图表肉眼质量评估

## 4. 常见病灶速查

| 症状 | 可能原因 | 证据 | 建议 |
|------|----------|------|------|
| SO 下滑而 flow_loss 下降 | demo 过拟合 | flow_loss ratio < 0.3 | 增大 BC 锚定或提早停止 |
| SAE 接近 0 但 SO 高 | reward drowned | ACP/Q ratio 很小 | 提高 `acp_reward_scale` |
| SO 崩塌 | critic 驱动漂移 | Q 爆炸 + entropy 崩 | 增 BC 或降 gamma |
| Q-value 振荡 | gamma 过高 + reward 不稳 | Q range > 50 | 降 gamma |
| Advantage mean 接近 1 | critic 无法区分 | advantage_std 太低 | 提高 online_ratio |
| Online reward 接近 0 | ACP reward 太弱 | reward gap > 100x | 提高 scale |

## 5. 相关路径

- `scripts/analyze_training_internals.py`
- `scripts/sweep_acp/fetch_wandb.py`
- `docs/vlaw/figures/*_internals/`
