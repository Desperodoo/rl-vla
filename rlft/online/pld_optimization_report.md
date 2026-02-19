# `train_pld` Pipeline 优化分析报告

> 生成时间：2026-02-18 | 对比对象：train_pld / train_dsrl / train_rlpd / train_maniskill

---

## 一、可直接复用的模块

### 1. `pld_env.py` 的观测编码逻辑 — 与 `dsrl_env.py` 完全相同

`pld_env.py` 中的 `_init_obs_history`, `_encode_single_frame`, `_encode_and_update_history`, `_get_obs_cond` 四个方法与 `dsrl_env.py` **逐行相同**。

**建议**：将观测编码逻辑提取到一个 mixin 类或公共基类（如 `BaseFlowEnvWrapper`），`ManiSkillFlowEnvWrapper` 和 `ManiSkillResidualEnvWrapper` 都继承它。好处是未来修 obs encoding 时只需改一处。

### 2. `train_pld.py` 的环境创建函数 — 与 `train_dsrl.py` 完全相同

`train_pld.py` 的 `_make_train_envs` / `_make_eval_envs` 与 `train_dsrl.py` 的同名函数**完全相同**。

**建议**：提取到 `rlft/envs/make_env.py` 中作为公共工具函数（它已有 `make_eval_envs`），例如新增 `make_flow_train_envs(args)`。也可以简单定义一个 `_make_envs_common` 放在 `rlft/online/_env_helpers.py` 中。

### 3. `_extract_success` 辅助函数 — 三个在线训练脚本中出现

`train_pld.py` / `train_dsrl.py` 的 `_extract_success` 完全相同，且 `train_rlpd.py` 内部也有类似逻辑（通过 `info.get("success")` 提取）。

**建议**：提取到 `rlft/envs/` 或 `rlft/utils/` 中作为公共函数。

### 4. `_VecEnvAdapter` 类 — PLD 与 DSRL 几乎相同

`train_pld.py` 的 `_VecEnvAdapter` 与 `train_dsrl.py` 的 `_VecEnvAdapter` 核心逻辑相同，PLD 版只多了一个 `step_base_only()` 方法。

**建议**：可以让 PLD 的 `_VecEnvAdapter` 继承 DSRL 的版本并扩展。或者将 `_VecEnvAdapter` 提取到 `rlft/envs/` 中。

### 5. `PLDEvalAgentWrapper._encode_obs` — 可复用 `data_utils.encode_observations`

`PLDEvalAgentWrapper._encode_obs` 和 `DSRLEvalAgentWrapper._encode_obs` 都调用同一个 `rlft.datasets.data_utils.encode_observations`。AgentWrapper 结构几乎相同，唯一差异是 PLD 需要在 `get_action` 中做 `a_base + a_delta` 合成。

**建议**：可定义 `BaseFlowEvalWrapper` 基类包含公共的 `_encode_obs`、`eval()`、`train()` 方法，PLD 和 DSRL 各自只覆写 `get_action()`。

---

## 二、基于 DSRL Sweep 报告的超参数优化建议

DSRL 和 PLD 在结构上高度相似——都是在冻结 ShortCut Flow 策略之上运行 SAC，核心差异仅在于 **action space**（noise space vs. residual space）。因此 DSRL sweep 的结论对 PLD 有强参考价值。

以下对比 PLD 当前默认值与 DSRL sweep 最优值，并给出建议：

| 参数 | PLD 当前值 | DSRL sweep 最优 | 建议调整 | 理由 |
|------|-----------|----------------|---------|------|
| **layer_size** | `256` | `2048` | **→ 2048** (或至少 1024) | sweep 明确表明 2×256 的 critic 容量严重不足（success_at_end=0.0），3×2048 的 success_at_end=0.5 遥遥领先。PLD 论文的 256 是因为其 VLA 基座策略更强（GPT-4V），ManiSkill3 环境下需要更大 critic 来学习精细控制 |
| **num_qs** | `2` | `10` | **→ 10** | sweep 显示 num_qs=10 (se=0.46) >> num_qs=2 (se=0.22)。更多 Q-network 的 min-Q 能有效抑制噪声空间/残差空间中的 Q 过估计 |
| **utd_ratio** | `2` | `60-80` | **→ 60** | sweep 中最关键的发现：UTD 与 success_at_end 强单调正相关。UTD=5 se=0.0，UTD=80 se=0.48。PLD 的 UTD=2 严重不足。residual space 同样需要精确的 Q 梯度来学习精细控制 |
| **gamma** | `0.99` | `0.95` | **→ 0.95** | episode 长度 100 步，每 RL step 执行 8 步 real action → 实际 ~12 RL steps/episode。γ=0.95 的有效视野≈20 RL steps 刚好覆盖完整 episode，γ=0.99 视野过长导致方差大 |
| **log_std_init** | `-3.0` | `-5.0` | **→ -5.0** | 预训练策略已经很好，初始阶段应保守探索（std≈0.007 vs 0.05）。sweep 显示 log_std=-5.0 的 se=0.36 远好于 -3.0 的 se=0.22 |
| **init_temperature** | `1.0` | `0.5` | **→ 0.5** | sweep 显示 init_temp=0.5 se=0.38 > 1.0 se=0.22。较低初始温度避免开始时过度随机化 |
| **target_entropy** | `None` (auto: -56) | `-3.5` | **→ -3.5** | auto 模式计算出 -56（= -act_steps × action_dim），对 residual space 来说过度负，会让策略坍缩到几乎零方差。sweep 显示 -3.5 se=0.42 >> 0.0 se=0.22。适度的负 target_entropy 平衡探索和利用 |
| **num_seed_steps** | (通过 `offline_demo_episodes` 替代) | `0` | **保持当前设计** | DSRL sweep 确认不需要 warmup，因为预训练策略已提供足够探索。PLD 通过离线 demo 收集 + Cal-QL 预训练实现了更好的 warmup，无需额外 seed steps |
| **action_scale** | `0.5` | action_magnitude = `2.5` | **不直接适用，但需注意** | PLD 在 residual space（[-0.5, 0.5]），DSRL 在 noise space（[-2.5, 2.5]）。这两个不可直接对应，因为 residual 直接加到 [-1,1] 的真实动作上，而 noise 经过 ODE 解码。PLD 的 0.5 意味着最多修正真实动作 50%，需要实验确认是否足够 |
| **buffer_size** | online=500K, offline=200K | `100K` (se最高) | **考虑减小 online→200K** | sweep 发现小 buffer 更快淘汰早期低质量数据，但 PLD 有 offline buffer 混合所以影响可能不同 |
| **learning_rate** | `3e-4` | `3e-3` (se最高) | **→ 1e-3 或更高** | sweep 显示 lr=1e-4 se=0.0，3e-3 se=0.38。在大 UTD 设置下，较高的学习率帮助 critic 更快收敛 |

### 最关键的 3 个调整（优先级排序）

1. **UTD 2 → 60**：这是 DSRL sweep 中对 success_at_end 影响最大的参数。PLD 的 UTD=2 遵循了原论文（VLA 基座），但在 ManiSkill + ShortCut Flow 设置下严重不足。
2. **网络规模 256 → 2048 + num_qs 2 → 10**：决定了 critic 能否建模精细控制所需的 Q landscape。
3. **target_entropy auto(-56) → -3.5**：auto 值对 56 维 action space 来说过度 conservative，会压制所有探索。

---

## 三、其他可优化的结构性问题

### 1. Probing 逻辑会浪费大量 timestep

当前 probing 实现中，probe 阶段的 `total_steps` 仍然累加（`total_steps += train_adapter.num_envs`），但 transition 不存入 buffer。这意味着如果 `probe_steps=5` 且 `probing_alpha=0.6`，相当于大约 60% × 5/12 ≈ 25% 的 total_timesteps 被"浪费"在不产生训练数据的 probe 步骤上。

**建议**：probe 步数不计入 `total_timesteps`，或者作为单独的统计量跟踪实际有效训练步数。

### 2. Probing 对所有 env 统一决策有问题

当前逻辑对所有 env 统一随机决定是否 probe（一次 `np.random.random()`），而每个 env 可能处于不同的 episode step。更合理的做法是 **per-env 独立 probing**：只对 `ep_step_counters[i] < probe_steps` 的 env probe，其余正常 RL。当前实现 broadcast 到所有 env 会导致：如果某些 env 已过 probe 阶段而另一些刚 reset，成为 probe 阶段的那些 env 仍会被 base policy 推进，不存 buffer。

### 3. `save_checkpoint` 未保存 `visual_encoder`

`train_pld.py` 的 checkpoint 保存调用 `save_checkpoint(agent=agent, args=args, ...)` 但没传 `visual_encoder=visual_encoder`。对比 `train_rlpd.py` 的 checkpoint 保存会传 `visual_encoder=visual_encoder`。虽然 PLD 的 visual_encoder 是冻结的不需要保存，但如果需要从 PLD checkpoint 恢复完整推理环境，最好也保存。

### 4. 离线 demo 收集时 env 可能会 auto-reset

`_collect_offline_demos` 中，ManiSkill3 的 `auto_reset` 机制意味着 `done=True` 时 env 已自动 reset，`next_obs` 已经是新 episode 的 obs。当前代码将 `(obs, action, rew, next_obs, done=True)` 存入 buffer，但 `next_obs` 是 reset 后的状态，这在 RL 中通常要标记为 terminal（`done=True`），这样这条 transition 的 TD-target 会忽略 `next_obs`。所以行为上是正确的，但可以加个注释说明。

### 5. PLD critic 的 action input 逻辑不一致

`pld_sac.py` 的 `compute_critic_loss` 注释说 "actions 是 combined actions a_bar"，但 `train_pld.py` 存入 buffer 的是 `residual`（`buffer.add_online(obs, residual, ...)`），而不是 `info["combined_action"]`。这意味着 critic 实际上在评估 `Q(s, a_delta)` 而非 `Q(s, a_bar)`。

这不一定是 bug——DSRL 的 critic 也评估 `Q(s, noise)` 而非 `Q(s, real_action)`——但 docstring 和实际行为不一致。**建议**：统一为 `Q(s, a_delta)`，并修正 docstring，或者改为存储 a_bar 并训练 `Q(s, a_bar)`。从 PLD 论文看，critic 应该在 residual action 上训练，与 actor 一致。

---

## 四、总结

### 复用优先级（从高到低）

1. 环境创建函数 `_make_train_envs` / `_make_eval_envs` → 提取公共模块
2. `_extract_success` → 提取到 `rlft/utils/`
3. Obs 编码逻辑 → 抽取 `BaseFlowEnvWrapper` 基类
4. `_VecEnvAdapter` → 提取公共版本

### 超参数调整优先级（从高到低）

1. `utd_ratio`: 2 → **60**（影响最大）
2. `layer_size`: 256 → **2048**，`num_qs`: 2 → **10**（critic 容量）
3. `target_entropy`: auto → **-3.5**（避免过度 conservative）
4. `gamma`: 0.99 → **0.95**（匹配 episode 时间尺度）
5. `log_std_init`: -3.0 → **-5.0**，`init_temperature`: 1.0 → **0.5**（保守初始探索）
