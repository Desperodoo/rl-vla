# Off-Policy DSRL（含 DSRL-NA）实现 ToDo（对照 On-Policy 版）

> 目标：**新建一个独立文件夹**（与 on-policy 版完全分开），实现 **Off-Policy 两阶段 DSRL**，并**补齐原文的 DSRL-NA（Noise Aliasing）**。
>
> 你当前 on-policy 版参考文件（仅用于对照，不复用代码）：
>
> * `dsrl_agent.py`（Stage1 AW-MLE + Stage2 PPO）
> * `latent_policy.py`（对角高斯 latent policy）
> * `train_latent_offline.py`（Stage1 训练脚本）
> * `train_latent_online.py`（Stage2 rollout + PPO 更新）
> * `macro_rollout_buffer.py`（SMDP/macro 轨迹 buffer）
> * `value_network.py`（PPO value）

---

## 0. 新建目录结构（独立实现）

**[新建]** `dsrl_offpolicy/`

* `agents/`

  * `dsrl_sac_agent.py`  ← Off-policy 主 agent（SAC + 可选 DSRL-NA）
* `models/`

  * `latent_policy.py`   ← 复制思路：对角高斯 actor（可参考 on-policy 的 `latent_policy.py` 接口）
  * `latent_q_network.py`← **新建**：Q^W(s,w)（double Q + target）
  * `action_q_network.py`← **新建**：Q^A(s,a)（double Q + target，用于 DSRL-NA）
* `buffers/`

  * `macro_replay_buffer.py` ← **新建**：SMDP off-policy replay（存 (s,w,r,discount,done,s')）
  * `action_replay_buffer.py`← **新建**：给 Q^A 用的 (s,a,r,discount,done,s')（可复用你原 AWSC 的离线 buffer 逻辑）
* `samplers/`

  * `base_policy_sampler.py`  ← **新建**：冻结 base policy（flow/shortcut-flow）解码 w→a_seq
  * `noise_aliasing.py`       ← **新建**：DSRL-NA 的 alias/inversion 工具
* `train/`

  * `train_stage1_offline.py` ← **新建**：离线 warm start（含 NA）
  * `train_stage2_online.py`  ← **新建**：在线 off-policy（rollout→replay→UTD 更新）
* `configs/`

  * `stage1.yaml`, `stage2.yaml`
* `utils/`

  * `loggers.py`, `math.py`, `schedules.py`, `seed.py`

> ✅ 对照点：on-policy 版的训练脚本分别对应 `train_latent_offline.py` 与 `train_latent_online.py`；但新实现**不继承** PPO/GAE/value。

---

## 1. 定义统一的“宏步 SMDP”接口（与 on-policy 对齐）

### 1.1 Macro transition 的数学定义

* 真实环境以 **act_horizon=H_a** 执行一个 chunk
* latent policy 每次输出一个 **macro-action**：`w`
* base policy（冻结）用 `w` 生成 `a_seq`，环境执行 `a_seq[:H_a]`
* 得到 cumulative reward `R`，done `D`，以及 discount factor `γ^k`（k 为有效步数）

### 1.2 代码接口（参考 on-policy 的 chunk collector 思路）

* **[新建]** `samplers/base_policy_sampler.py`

  * `decode_actions(obs_cond, w) -> a_seq`
  * 必须支持：`use_ema=True`（对照 on-policy 里“用 EMA base policy”）
  * 必须支持：只执行 head（act_horizon）与 full 序列（用于 NA）

> ✅ 对照点：on-policy 版 Stage2 rollout 中的 “w→a_seq→env.step(act_horizon)” 流程。

---

## 2. Stage2（在线）：Off-Policy DSRL = SAC-on-latent（核心）

> 目标：替换 on-policy PPO，改为 **SAC / TD3 风格**，实现高数据复用（UTD）。

### 2.1 Actor：latent policy π^W(w|s)

* **[新建/复用结构]** `models/latent_policy.py`

  * 建议复制 on-policy 的高斯 policy 结构与 API：

    * `sample(obs_cond) -> w, logp`
    * `rsample(obs_cond) -> w, logp`（reparameterization）
    * `log_prob(obs_cond, w)`
  * **必须**支持：`steer_mode`（full / act_horizon）

### 2.2 Critic：Q^W(s,w)（double Q + target）

* **[新建]** `models/latent_q_network.py`

  * 输入：`[obs_cond, w]`（flatten）
  * 输出：标量 Q
  * 双 Q：`Q1, Q2` + target `Q1_t, Q2_t`

### 2.3 Replay：MacroReplayBuffer

* **[新建]** `buffers/macro_replay_buffer.py`

  * 存：`obs_cond, w, R, discount, done, next_obs_cond`
  * API：`add_batch(...)`, `sample(batch_size)`

### 2.4 SAC 更新（必须写清楚公式）

* target：

  * `w' ~ π(w|s')`
  * `V_targ(s') = min(Q1_t(s',w'), Q2_t(s',w')) - α logπ(w'|s')`
  * `y = R + discount * (1-done) * V_targ(s')`
* critic loss：

  * `L_Q = MSE(Q1(s,w), y) + MSE(Q2(s,w), y)`
* actor loss：

  * `w ~ π(w|s)`（rsample）
  * `L_π = E[ α logπ(w|s) - min(Q1(s,w),Q2(s,w)) ]`
* temperature（可选 auto-α）：

  * `L_α = E[ -α (logπ + target_entropy) ]`
* target soft update：`θ_t ← τ θ + (1-τ) θ_t`

### 2.5 在线训练脚本：rollout + UTD

* **[新建]** `train/train_stage2_online.py`

  * 循环：

    1. rollout N 个 macro steps（多环境并行）
    2. 写入 replay
    3. 做 UTD：每收集 1 批数据，做 K 次 SAC update（K=10~100 作为 sweep）
  * 日志（必须）：

    * `approx_kl` 不再需要；改为 `alpha`, `q_mean`, `q_std`, `td_error`, `policy_entropy`, `utd`

> ✅ 对照点：on-policy 里你看过 `approx_kl/clip_frac/adv_std`；off-policy 版重点监控 `td_error/alpha/q`。

---

## 3. Stage1（离线）：Off-Policy warm start（含 DSRL-NA）

> 目标：用离线数据把 actor 初始化到“比 prior 更好”的区域；并通过 NA 让离线 (s,a) 数据能训练噪声空间。

### 3.1 先训练动作空间 critic：Q^A(s,a)

* **[新建]** `models/action_q_network.py`
* **[新建]** `buffers/action_replay_buffer.py`

  * 数据来源：你已有 offline dataset（或从环境采集）
  * 存：`obs_cond, a_head(or a_seq), R, discount, done, next_obs_cond`
* **[新建]** `train/train_qA_offline.py`（也可写在 stage1 脚本里）

  * 用 TD 学 `Q^A(s,a)`（double Q + target）

> ✅ 对照点：你 on-policy Stage1 用的是“已有 Q(s,a) 打分候选 w”；这里把它显式化为 `Q^A`，并可完全离线训练。

### 3.2 DSRL-NA：从 Q^A 蒸馏出噪声空间 critic Q^W（Noise Aliasing）

* **[新建]** `samplers/noise_aliasing.py`

  * `alias_noise(obs_cond, a_head) -> w_alias`
  * 推荐 2 个实现模式：

    1. **优化式 inversion（稳）**：

       * init `w ~ N(0,I)`
       * 迭代 T 次：`a_hat = decode(obs,w)`；最小化 `||a_hat_head - a_head||^2`
       * 得到 `w_alias`
    2. **检索式 aliasing（快）**：

       * 对每个 obs 采样 M 个 w
       * 生成 a_hat，选最接近 a_head 的 w
       * 缓存到 LMDB/npz

* 蒸馏训练 `Q^W`：

  * 采样 `s`（来自离线 obs）
  * 得到 `w_alias`
  * 目标：`Q^W(s,w_alias) ≈ Q^A(s,a_head)`
  * loss：`MSE(Q^W(s,w_alias), stopgrad(Q^A(s,a_head)))`

> ✅ 对照点：这一步是你之前讨论的 “NA 先不管”，但这里明确要实现。它是官方提高数据利用率的关键。

### 3.3 用 Q^W 做离线 actor warm start（两种方案任选其一）

**方案 S1：AW-MLE（对照你 on-policy Stage1）**

* 对每个 obs：采样候选 `w_i`（M=16~64）
* 用 `Q^W(s,w_i)` 计算优势/权重
* 最小化：`L = -Σ ω_i logπ(w_i|s)`

**方案 S2：SAC-style offline actor（更统一）**

* 直接用离线 replay（如果你存了 (s,w)）

* 或用 `w_alias` 当作行为动作，做离线 SAC（注意 OOD 风险）

* 初期建议先用 S1，稳且实现简单

* **[新建]** `train/train_stage1_offline.py`

  * 支持开关：`--use_na`、`--aw_mle`、`--m_candidates`、`--beta_latent`、`--tau_baseline`

---

## 4. 训练流程串联（两阶段）

### 4.1 Stage1 输出

* 保存：

  * actor `latent_policy.ckpt`
  * `Q^A.ckpt`（可选）
  * `Q^W.ckpt`（建议保存，用于 debug）

### 4.2 Stage2 初始化

* actor：加载 Stage1 的 `latent_policy.ckpt`
* critic Q^W：

  * 方案：随机初始化（简单）
  * 更强：加载 Stage1 蒸馏得到的 `Q^W.ckpt`（推荐）

---

## 5. 必做的诊断/验证（避免“跑了但不提升”）

### 5.1 Stage1 验证

* 固定一批 obs：比较

  * `E[Q^W(s,w)]` for `w~prior`
  * `E[Q^W(s,w)]` for `w~π_w_stage1`
* 期望：stage1 后者显著更高（哪怕提升很小）

### 5.2 Stage2 验证

* 核心曲线：

  * `td_error` 下降
  * `q_mean` 不爆炸
  * `alpha` 不飙升到极大（否则策略变纯随机）
  * `policy_entropy` 不瞬间塌到 0
* 关键对照：

  * base policy（prior w）
  * Stage1 only
  * Stage2 (off-policy)

---

## 6. 建议的最小实现顺序（给 Copilot）

1. ✅ 先实现 **Stage2 SAC-on-latent**（不带 NA）：

   * latent policy + latent Q + macro replay + UTD
   * 先在小环境/短训练验证 td_error/q/alpha 正常

2. ✅ 再实现 **Stage1（不带 NA）的 AW-MLE warm start**：

   * 直接复刻你 on-policy 的候选采样 + 打分 + weighted logprob
   * 打分先临时用 `Q^A(s, decode(s,w)_head)`（如果你已有动作空间 Q）

3. ✅ 最后补齐 **DSRL-NA**：

   * 先做检索式 aliasing（快）
   * 再做优化式 inversion（稳）
   * 蒸馏 Q^W，再用 Q^W 训练 actor

---

## 7. 关键超参（建议起步值）

* Stage2 SAC:

  * `batch_size=256`
  * `gamma=0.99`，macro discount 用 `discount_factor`（来自 env rollout）
  * `tau_target=0.005`
  * `actor_lr=1e-4`, `critic_lr=3e-4`
  * `UTD=20` 起步（sweep 10/20/50/100）
  * `target_entropy = -dim(w_steer)` 的比例（可先用 `-0.5*dim`）

* Stage1 AW-MLE:

  * `M=32`（候选数）
  * `beta_latent=0.3~1.0`（softmax 温度倒数）
  * baseline `tau=1.0`（别太大）

* NA inversion:

  * `steps=20~50`, `lr=1e-1~1e-2`, `multi_start=4`

---

## 8. 交付物清单（最终你应该看到的文件/功能）

* ✅ `dsrl_offpolicy/train/train_stage1_offline.py`：能跑完并保存 Stage1 actor
* ✅ `dsrl_offpolicy/train/train_stage2_online.py`：能在线训练，且 UTD>1 能显著加速
* ✅ `noise_aliasing.py`：能从 (s,a_head) 得到可复用的 w_alias（带缓存）
* ✅ `Q^A` 离线训练 + `Q^W` 蒸馏：可独立跑通
* ✅ 三组对照 eval：prior / stage1 / stage2
