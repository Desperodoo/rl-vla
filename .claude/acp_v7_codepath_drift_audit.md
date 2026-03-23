# Code-path Drift Audit vs 54faf40 (`更新pld参数`)

**日期**：2026-03-23

## 目标
围绕提交 `54faf40` 做定向 drift audit，解释为什么当前代码路径下，即便使用 `runs/fair_comparison` 的 exact recipe，也没有复现历史高 SAE `PLD/DSRL + sim_reward` baseline。

## 1. 54faf40 直接改了什么

该提交与复现问题直接相关的代码文件只有两个：
- `rlft/online/train_pld.py`
- `rlft/algorithms/online_rl/pld_sac.py`

其核心意图是把 PLD 默认参数从较保守的一组切到更激进/更新的一组：
- `init_temperature`: `0.1 -> 0.5`
- `tau`: `0.005 -> 0.001`（在 agent 默认值）
- `hidden_dims`: `1024 -> 768`（agent 默认值）
- `layer_size`: `1024 -> 768`（train args 默认值）
- `calql_alpha`: `0.0 -> 5.0`（train args 默认值）

但这个提交本身的 diff 已经暴露出一个异常：`train_pld.py` 中带有冲突痕迹（`<<<<<<< Updated upstream` / `>>>>>>> Stashed changes`），说明当时的参数更新并不是一个非常干净的单一路径替换，而是混合过两套配置方向。

## 2. 当前 HEAD 相对 54faf40 的 PLD 漂移

### 2.1 当前 `train_pld.py` 与 `pld_sac.py` 已经不一致

当前代码里：
- `train_pld.py` 默认值：
  - `tau = 0.005`
  - `init_temperature = 0.5`
  - `layer_size = 1024`
  - `calql_alpha = 0.0`
  - `batch_size = 1024`
- `pld_sac.py` 默认值：
  - `tau = 0.001`
  - `init_temperature = 0.5`
  - `hidden_dims = [768, 768, 768]`

这意味着：
- 训练入口默认配置和 agent 默认配置已经发生分叉；
- 实际运行值取决于 `train_pld.py` 是否显式传参；
- 这类“入口默认值”和“agent 默认值”不一致，本身就是一种高风险 drift 信号。

不过就实际主路径而言，`train_pld.py` 会显式传：
- `hidden_dims=[args.layer_size] * args.num_layers`
- `tau=args.tau`
- `init_temperature=args.init_temperature`

所以**实际生效值**更接近 `train_pld.py`，而不是 `pld_sac.py` 的内部默认值。

### 2.2 当前 HEAD 已经把 54faf40 的关键 PLD 更新基本“回退掉了”

相对 `54faf40`，当前 HEAD 的 PLD 主路径实际变成：
- `tau`: 回到 `0.005`
- `layer_size`: 回到 `1024`
- `calql_alpha`: 回到 `0.0`
- `init_temperature`: 保持 `0.5`

因此如果以 `54faf40` 为锚点，当前 HEAD 的 PLD 不是继续沿着那次“更新pld参数”的方向前进，而是一个**混合态**：
- 一部分保留新配置（`init_temperature=0.5`）
- 一部分回退旧配置（`tau=0.005`, `layer_size=1024`, `calql_alpha=0.0`）

这会造成一个问题：

> 当前 PLD 默认配置并不对应一套历史上被完整验证过的单一 recipe，而是不同 sweep 阶段参数的拼接体。

## 3. 为什么这件事对 fair replay 很重要

`fair_replay` 虽然在 CLI 上显式传入了 historical recipe 的关键参数，但当前代码路径里 PLD 相关行为并不只由这些表层超参决定。

当前 HEAD 在 `54faf40` 之后新增了多项行为级变更：
- `q_target_clip`
- `min_temperature`
- `entropy_bonus_coef`
- `best_sae.pt` 单独保存逻辑
- ACP wrapper 路径
- reward diagnostic logging
- `sae_retention` 记录

其中最值得注意的是：
- `pld_sac.py` 新增了 **q_target_clip** 逻辑
- `pld_sac.py` 新增了 **temperature floor** 逻辑
- `pld_sac.py` 新增了 **entropy bonus** 逻辑

即使这些在 replay CLI 中使用默认关闭值，它们也说明当前 PLD agent 已经不是 `54faf40` 时期的同一个实现表面。

## 4. 当前最强的 PLD drift 假设

### Hypothesis A — 配置层发生了“拼接式回退”
当前 PLD 默认值不是一套完整历史 recipe，而是：
- 部分来自 `54faf40`
- 部分来自其前后的回退/重调

这会让“exact replay”虽然在命令行上看起来对齐，但运行行为仍可能不处于历史最佳区域。

### Hypothesis B — `batch_size` 是一个非常可疑的非对齐点
历史 fair config：
- `batch_size = 256`

当前 `train_pld.py` 默认：
- `batch_size = 1024`

在本次 replay 中我已经显式按 fair recipe 传了 `256`，所以这不是 replay 直接失败的原因；但它说明当前主代码默认和历史 baseline 已经明显分离。

### Hypothesis C — PLD 当前真正的问题更像“实现演化后不再对应历史 fair-comparison 时代的训练分布”
尤其是：
- q clipping
- agent 默认值和入口默认值分叉
- 后续多轮 v5/v6 的 ACP-oriented 稳定化改造

这些都可能让当前代码在 sim reward 下也走向一种新的局部最优：
- high `success_once`
- low `success_at_end`
- final retention ≈ 0

## 5. DSRL 相对 54faf40 的状态

`54faf40` 本身不改 DSRL。

当前 DSRL 相对历史更重要的 drift 来自后续提交：
- `train_dsrl.py` 新增 ACP wrapper 路径
- `dsrl_sac.py` 新增 `q_target_clip`
- 新增 `best_sae.pt` / retention logging

也就是说：
- DSRL 的 drift 不是从 `54faf40` 开始
- 而是来自后续 ACP / critic stabilization 系列提交

这与 replay 结果一致：
- DSRL sim 与 acp 都失败
- 更像是当前 DSRL code path 整体偏离了历史 fair-comparison 行为，而不是被 `54faf40` 单独影响

## 6. 当前 audit 的首轮结论

### 已确认
1. `54faf40` 是一个 **PLD 参数切换提交**，而不是 DSRL 提交。
2. 当前 HEAD 的 PLD 配置状态已经**不是** `54faf40` 的状态，而是部分回退 + 部分保留的混合态。
3. 当前 HEAD 的 `train_pld.py` 与 `pld_sac.py` 默认值已经分叉。
4. 当前 HEAD 的 PLD/DSRL agent 都已经叠加了后续 v5/v6 的 critic stabilization / ACP instrumentation 逻辑。

### 这意味着
> 如果要恢复历史高 SAE baseline，不能只看 replay CLI 是否复用了 `runs/fair_comparison` 的数值；还必须确认当前代码是否仍然等价于历史时期的训练实现。

## 8. 更细的提交级 drift 结果

### 8.1 PLD 演化路径（`c055034 -> 5adf3d4 -> 54faf40 -> HEAD`）

#### `c055034 -> 5adf3d4`
这一段是一次非常大的 PLD regime 切换：
- `action_scale: 0.5 -> 0.3`
- `gamma: 0.95 -> 0.99`
- `learning_rate: 1e-3 -> 1e-4`
- `online_ratio: 0.5 -> 1.0`
- `num_qs: 10 -> 5`
- `layer_size: 2048 -> 1024`
- `offline_demo_episodes: 200 -> 50`
- `calql_pretrain_steps: 2000 -> 1000`
- `calql_alpha: 5.0 -> 0.0`
- `init_temperature: 0.5 -> 0.1`
- `batch_size: 256 -> 1024`

这说明 PLD 的 fair-comparison 时代并不是“小修小补”，而是一整个训练 regime 的重写。

#### `5adf3d4 -> 54faf40`
这一段又发生第二次 PLD regime 调整：
- `tau: 0.005 -> 0.001`
- `init_temperature: 0.1 -> 0.5`
- `layer_size: 1024 -> 768`
- `calql_alpha: 0.0 -> 5.0`

也就是：
> `54faf40` 代表的是一条比 fair-comparison `best__1772537094` 更靠后的 PLD v3 方向，而不是历史 baseline 本身。

#### `54faf40 -> HEAD`
当前 HEAD 又把其中一部分回退了：
- `tau` 回到 `0.005`
- `layer_size` 回到 `1024`
- `calql_alpha` 回到 `0.0`
- 但 `init_temperature=0.5` 仍被保留

因此当前 PLD 是一个混合态，而不是对齐任何一个明确历史节点。

### 8.2 DSRL 演化路径（`5cbdc32 -> 592df92 -> d13f59c -> HEAD`）

#### `5cbdc32 -> 592df92`
这是 DSRL 最大的一次 regime flip：
- `gamma: 0.99 -> 0.95`
- `utd_ratio: 40 -> 60`
- `num_seed_steps: 5000 -> 0`
- `target_entropy: 0.0 -> -3.5`
- `init_temperature: 1.0 -> 0.5`
- `log_std_init: -3.0 -> -5.0`
- `action_magnitude: 2.0 -> 2.5`
- `layer_size: 512 -> 2048`
- `num_qs: 2 -> 10`

这意味着 replay 用到的当前 DSRL path，和旧 regime 已经不是同一个算法工作点。

#### `592df92 -> d13f59c`
这一段引入了两个非常关键的后续 drift：
- `q_target_clip`
- ACP reward wrapper path + shaping/clip options

这两者都会直接改变 critic target 与训练 reward 语义。

## 10. Drift regression 结果如何更新 root cause

基于 `docs/vlaw/figures/rlpd_acp_v7_drift_reg/diagnosis_report.md`，现在可以把 root cause 进一步收敛：

### 10.1 `q_target_clip=20` 是当前最主要的共享 drift

这是目前最强结论。

- `fair_replay`（默认 `q_target_clip=20`）:
  - PLD sim: `best SAE = 0.06`, `final SAE = 0.00`
  - DSRL sim: `best SAE = 0.02`, `final SAE = 0.00`
- `qclip0` 回归：
  - `pld_v7_reg_qclip0_sim_s42`: `best/final SAE = 0.82 / 0.82`
  - `dsrl_v7_reg_qclip0_sim_s42`: `best/final SAE = 0.66 / 0.66`

这说明后续为了 ACP/v5/v6 critic 稳定化引入的 `q_target_clip=20`，在 sim reward 路径下不是“无害稳定器”，而是会显著压制历史 high-SAE baseline。

### 10.2 PLD 不需要整体回到旧时代，只要移除 qclip 就能恢复

- `pld_v7_reg_qclip0_sim_s42`: `0.82 / 0.82`
- `pld_v7_reg_54faf40_sim_s42`: `0.80 / 0.80`

解释：
- `54faf40` 风格参数是可工作的；
- 但它不是必要条件；
- 当前 tuned-ish PLD 路径只要移除 `q_target_clip=20`，就已经能恢复高 SAE。

### 10.3 DSRL 不是应该回退到 pre-592df92；真正有效的是“当前 tuned DSRL + no qclip”

- `dsrl_v7_reg_qclip0_sim_s42`: `0.66 / 0.66`
- `dsrl_v7_reg_pre592df92_sim_s42`: `0.04 / 0.02`

这说明：
- `592df92` 之前的旧 DSRL regime 并不是当前应当回退的目标；
- 当前 tuned DSRL regime 本身是可工作的；
- 关键共享问题仍然是 `q_target_clip=20`。

### 10.4 ACP 仍然是第二层独立问题

虽然去掉 qclip 后 sim baseline 恢复了，但 ACP mirror 没恢复：
- `pld_v7_reg_qclip0_acp_s42`: `0.02 / 0.00`
- `dsrl_v7_reg_qclip0_acp_s42`: `0.06 / 0.00`

因此当前最终诊断应拆成两层：

1. **共享 drift 问题**：`q_target_clip=20` 破坏了历史 sim baseline 复现；
2. **ACP 专属问题**：即使移除 qclip，ACP 仍然无法让 PLD/DSRL 恢复高 SAE。

## 11. 对下一步实验的含义

优先级已经可以重排为：

### Priority 1
把 sim baseline 的 replay / baseline path 切成：
- PLD sim → `q_target_clip=0`
- DSRL sim → `q_target_clip=0`

### Priority 2
在这个前提下重新做 ACP 对照：
- PLD sim qclip0 vs PLD acp qclip0
- DSRL sim qclip0 vs DSRL acp qclip0

### Priority 3
再继续分析 ACP reward semantics（scale / shaping / grasp-bonus / hold-signal）
