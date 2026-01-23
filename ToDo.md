# 方案A：Critic 训练与评估（Copilot 实现用 ToDo / Prompt）

> **目的**：
> 在不改动 policy 的前提下，仅基于已标注的 **离散 1–5 Reward（来自 VLM）+ done 标签**，构建 **SMDP（chunk）buffer**，训练一个 **排序式（ranking-based）Critic V(s)**，并系统性评估：
>
> 1. Critic 是否学到了“接近完成 → 更高价值”；
> 2. 基于 Critic 计算的 **chunk-level advantage** 是否与真实进度变化一致、是否具备区分能力。

本文件包含 **两套实现任务**：

* **A1：仅成功数据（success-only）**
* **A2：成功 + 失败数据（success + failure）**

所有内容均以 **Copilot 可直接实现** 为目标编写（偏工程说明，少理论描述）。

---

## 通用约定（两套方案共用）

### 输入数据格式（episode 级）

```python
episode = {
  "obs_img": Tensor[T, 3, H, W],      # wrist 相机图像
  "obs_state": Tensor[T, D],          # 机械臂关节/状态
  "reward_1_5": Tensor[T],            # 离散 1~5 reward（VLM 打分）
  "done": Tensor[T],                  # episode 终止标记
  "is_success": bool,                 # 是否成功（A2 中必需）
  "teleop_mask": Tensor[T] | None     # 是否人类接管（可选）
}
```

### 关键超参数（统一放在 config）

```python
ACT_HORIZON = K          # chunk 长度，例如 8
GAMMA = 0.99
RANK_TAU = 0.1
LAMBDA_ANCHOR = 1.0
LAMBDA_SMOOTH = 0.1
```

---

# A1：仅成功数据（Success-only）

## A1-1. 构造 episode 内进度标签 p_t

> 目的：把离散 1–5 reward 转为 **长度无关、可排序的进度信号**。

### Step A1-1.1：reward 映射到 [0,1]

```python
r_tilde = (reward_1_5 - 1) / 4.0
```

### Step A1-1.2：episode 内归一化

对每条 episode 单独处理：

```python
p_t = (r_tilde - r_tilde.min()) / (r_tilde.max() - r_tilde.min() + eps)
```

### Step A1-1.3：时间平滑（EMA）

```python
p_t = ema(p_t, alpha=0.2)
```

保存 `p_t` 作为 critic 的监督信号（不要求严格单调）。

---

## A1-2. 构建 SMDP（chunk）buffer

> 使用 **SMDP 形式**，在 chunk 粒度上定义转移，用于后续 advantage 计算。

### Step A1-2.1：chunk 采样规则

* 仅采样 `t = 0 ... T-K-1`
* 尾部不足 K 的部分 **直接丢弃**

### Step A1-2.2：SMDP item 结构

```python
SMDPItem = {
  "s_t": (obs_img[t], obs_state[t]),
  "s_tpK": (obs_img[t+K], obs_state[t+K]),
  "p_t": p_t[t],
  "p_tpK": p_t[t+K],
  "done_tpK": done[t+K],
  "teleop_ratio": mean(teleop_mask[t:t+K]) if teleop_mask else 0.0
}
```

保存为 list / replay buffer（仅用于 critic）。

---

## A1-3. Critic 网络

### 网络定义

* 输入：`(image, state)`
* 输出：`V(s) ∈ [0,1]`（Sigmoid）
* **不输入 action**

---

## A1-4. Critic 训练目标（排序式）

### A1-4.1 Episode 内 Pairwise Ranking Loss（核心）

在同一条 episode 内采样 `t1 < t2`：

```math
L_rank = log(1 + exp(-(V(s_{t2}) - V(s_{t1})) / tau)) * w
```

其中：

```python
w = clip(p[t2] - p[t1], 0.0, 1.0)
```

> 采样建议：
>
> * t2 - t1 ∈ {K, 2K, 4K} 混合

---

### A1-4.2 Anchor Loss（尺度稳定）

* `reward_1_5 == 5` 或 `done == True` → V ≈ 1
* episode 内 p 最低的 10% → V ≈ 0

```math
L_anchor = ||V(s_done) - 1||^2 + ||V(s_low) - 0||^2
```

---

### A1-4.3 Temporal Smoothness

```math
L_smooth = E[ |V(s_{t+1}) - V(s_t)| ]
```

可按 `(1 - teleop_ratio)` 加权，减弱人类接管带来的突变影响。

---

### A1-4.4 总损失

```math
L = L_rank + λ_a * L_anchor + λ_s * L_smooth
```

---

## A1-5. Advantage 计算（chunk-level）

训练完成后，在 SMDP buffer 上计算：

```math
A_t^{(K)} = V(s_{t+K}) - V(s_t)
```

（评估阶段可不加折扣；如需，使用 `gamma^K`）

保存 `A_t^{(K)}` 供评估使用。

---

## A1-6. Critic / Advantage 评估方案

### A1-6.1 Pairwise 排序准确率

* 真实：`p[t2] > p[t1]`
* 预测：`V(s_{t2}) > V(s_{t1})`
* 指标：accuracy / AUC

### A1-6.2 相关性

* Spearman ρ：`corr(V(s_t), p_t)`

### A1-6.3 单条轨迹可视化

* t vs p_t
* t vs V(s_t)

### A1-6.4 Advantage 方向一致性

* GT：`Δp = p[t+K] - p[t]`
* Pred：`A_t^{(K)}`
* 指标：sign accuracy / Spearman ρ

---

# A2：成功 + 失败数据（Success + Failure）

> 在 A1 基础上，引入失败 episode，使 critic 学会 **区分成功/失败终态**，并提升 advantage 的判别能力。

---

## A2-1. 进度标签的校准（引入 done 锚点）

### 规则

* 成功 episode：终止帧强制 `p_T = 1`
* 失败 episode：终止帧强制 `p_T = 0`

中间帧：

* 成功：同 A1 的 episode 内归一化
* 失败：使用归一化值后整体乘缩放因子 `η ∈ [0.2, 0.4]`

---

## A2-2. SMDP buffer（新增字段）

```python
SMDPItem.update({
  "is_success_ep": is_success,
  "is_terminal_fail": done[t+K] and not is_success
})
```

---

## A2-3. Critic 训练新增约束

### A2-3.1 跨 episode Ranking（成功 > 失败）

采样：

* `s_pos`：成功 episode 的后 10% 状态
* `s_neg`：失败 episode 的后 10% 状态

```math
L_rank_inter = log(1 + exp(-(V(s_pos) - V(s_neg) - m) / tau))
```

`m ≈ 0.2`

---

### A2-3.2 Anchor Loss（成功/失败终态）

```math
L_anchor = ||V(s_done_succ) - 1||^2 + ||V(s_done_fail) - 0||^2
```

---

### A2-3.3 （可选）Success/Fail 分类头（仅用于评估）

* 在 critic backbone 上加一层 `q(s) ∈ [0,1]`
* 仅在 terminal states 上训练：

```math
L_cls = BCE(q(s_done), is_success)
```

---

### A2-3.4 总损失

```math
L = L_rank_intra + λ_inter * L_rank_inter + λ_a * L_anchor + λ_s * L_smooth + λ_c * L_cls
```

---

## A2-4. Advantage 计算（同 A1）

```math
A_t^{(K)} = V(s_{t+K}) - V(s_t)
```

额外用于分析失败 episode：

* 失败后段的 `A_t^{(K)}` 应更偏负

---

## A2-5. 评估方案（成功 + 失败）

### A2-5.1 Terminal 可分性

* 画 `V(s_done)` 的直方图（success vs failure）
* 指标：AUC

### A2-5.2 Advantage 的失败预警能力

* 标签：该 chunk 之后最终 success / failure
* 指标：AUC

### A2-5.3 Ranking Accuracy（intra / inter 分开统计）

---

## 最终交付物（Copilot 应生成）

1. `build_progress_labels.py`
2. `build_smdp_buffer.py`
3. `critic_model.py`
4. `train_critic_success_only.py`
5. `train_critic_success_fail.py`
6. `eval_critic.py`

> **注意**：本阶段不涉及 policy 更新，仅评估 critic 与 advantage 的质量。
