# Action Chunking 下的 Bellman 方程与 Offline DQL

## 1. 标准 DQL 的 Bellman 方程（不 chunk）

普通一步 DQL：

$$Q(s_t, a_t) = \mathbb{E}[r_t + \gamma \max_{a'} Q(s_{t+1}, a')]$$

采样 batch，目标值：

$$y_t = r_t + \gamma (1 - d_t) \max_{a'} Q_{\theta^-}(s_{t+1}, a')$$

loss：

$$L(\theta) = \mathbb{E}[(Q_\theta(s_t, a_t) - y_t)^2]$$

## 2. Action Chunking 下的 Bellman 方程（SMDP 形式）

### 2.1 设定

- 每个 chunk 长度为 $H$（固定），或者更一般地为 $\tau$（可变）
- chunk 行为：从 $t$ 开始，环境连续执行 $\tau$ 步原子动作 $(a_t, a_{t+1}, \ldots, a_{t+\tau-1})$，直到 chunk 结束才重新决策
- 中间获得奖励 $r_t, r_{t+1}, \ldots, r_{t+\tau-1}$
- chunk 结束后的状态为 $s_{t+\tau}$

### 2.2 SMDP Bellman 方程

把这一段视作一个"宏动作" $\hat{a}_t$（即 action chunk），那么 SMDP 形式的 Bellman 方程是：

$$Q(s_t, \hat{a}_t) = \mathbb{E}\left[\sum_{i=0}^{\tau-1} \gamma^i r_{t+i} + \gamma^\tau \max_{\hat{a}'} Q(s_{t+\tau}, \hat{a}')\right]$$

### 2.3 Episode 提前结束的情况

若 chunk 中途 episode 提前结束（done）：
- 设在第 $j$ 步终止（0-based），则 $\tau = j + 1$
- 后面那项就没有了：

$$Q(s_t, \hat{a}_t) = \mathbb{E}\left[\sum_{i=0}^{\tau-1} \gamma^i r_{t+i}\right]$$

### 2.4 Offline 条件下

- 不能与环境交互，只能用数据中已有的轨迹
- 但公式不变，只是所有量来自数据集

## 3. Offline DQL 下 Q 网络的 Loss 和 Update

### 3.1 构造 Chunk-Level Transition

假设你有原始 step-level 数据：$(s_t, a_t, r_t, s_{t+1}, d_t)$

在训练 DQL 时，先在数据里构造出 chunk 样本：

1. 从数据里随机采样起点 index $t$，以及对应状态 $s_t$
2. 从这一条轨迹里取接下来的 $H$ 步（或直到 episode 结束）
3. 形成一个 chunk（宏动作）

**Chunk 数据结构：**

| 字段 | 说明 | 公式 |
|------|------|------|
| 状态 | 起始状态 | $s_t$ |
| 动作 chunk | 宏动作表示 | $\hat{a}_t = (a_t, \ldots, a_{t+\tau-1})$ |
| 累计折扣奖励 | N-step return | $R_t^{(\tau)} = \sum_{i=0}^{\tau-1} \gamma^i r_{t+i}$ |
| 终止状态 | chunk 结束后状态 | $s' = s_{t+\tau}$ |
| Done 标志 | 中间是否有 done | $d_t^{(\tau)} = 1$ if any done else $0$ |
| 有效折扣 | 跨越步数的折扣 | $\Gamma^{(\tau)} = \gamma^\tau$ |
| 有效长度 | 实际 chunk 长度 | $\tau$ |

最终得到一个 chunk-level transition：

$$(s_t, \hat{a}_t, R_t^{(\tau)}, s', d_t^{(\tau)}, \tau)$$

### 3.2 Bellman 目标（Target）

用 target 网络 $Q_{\theta^-}$：

$$y_t = R_t^{(\tau)} + (1 - d_t^{(\tau)}) \cdot \gamma^\tau \max_{\hat{a}'} Q_{\theta^-}(s', \hat{a}')$$

**关键差异：**
- 折扣因子是 $\gamma^\tau$（不是 $\gamma$），因为 chunk 跨越了 $\tau$ 步
- 如果 chunk 中 episode 提前结束，$d_t^{(\tau)} = 1$，第二项为 0
- 如果 chunk 不足 $H$（比如到 episode 末尾只剩 3 步），那 $\tau$ 就是实际长度

### 3.3 Q 网络的输入与输出

在我们的框架中（连续 latent chunk）：
- chunk 由动作序列本身表示 $(a_t, \ldots, a_{t+\tau-1})$
- Q 网络：$Q_\theta(s, \text{action\_seq})$
- 实现上：state encoder + action sequence encoder (U-Net) + output head

### 3.4 Loss 与参数更新

构建好 batch 后，loss 就是标准 TD loss，只是用 chunk target：

$$L(\theta) = \mathbb{E}_{(s_t, \hat{a}_t, R_t^{(\tau)}, s', d_t^{(\tau)})} \left[(Q_\theta(s_t, \hat{a}_t) - y_t)^2\right]$$

其中：

$$y_t = R_t^{(\tau)} + (1 - d_t^{(\tau)}) \cdot \gamma^\tau \max_{\hat{a}'} Q_{\theta^-}(s', \hat{a}')$$

然后像普通 DQN 一样：
1. 对 $\theta$ 做梯度下降
2. 定期/软更新 target 网络 $\theta^- \leftarrow \theta$

## 4. 实现方案

### 4.1 Dataset 修改

需要修改 `OfflineRLDataset` 来输出 chunk-level transition：

```python
# 每个样本需要返回：
{
    "observations": obs_seq,           # (obs_horizon, ...) 起始观测
    "next_observations": next_obs_seq, # (obs_horizon, ...) chunk 结束后的观测
    "actions": action_seq,             # (pred_horizon, action_dim) 动作序列
    "cumulative_reward": R_tau,        # scalar: 累计折扣奖励
    "chunk_done": d_tau,               # scalar: chunk 内是否有 done
    "effective_length": tau,           # scalar: 有效 chunk 长度
    "discount_factor": gamma ** tau,   # scalar: 有效折扣因子
}
```

### 4.2 算法修改

需要修改 `DiffusionDoubleQAgent` 和 `CPQLAgent` 的 critic loss 计算：

```python
def _compute_critic_loss(self, ...):
    # 使用 cumulative_reward 而不是单步 reward
    # 使用 gamma^tau 而不是 gamma
    # 使用 chunk 结束后的状态而不是下一步状态
    
    with torch.no_grad():
        next_actions = self._sample_actions_batch(next_obs_cond)
        target_q1, target_q2 = self.critic_target(next_actions, next_obs_cond)
        target_q = torch.min(target_q1, target_q2)
        
        # 关键修改：使用 cumulative_reward 和 gamma^tau
        target_q = cumulative_reward + (1 - chunk_done) * discount_factor * target_q
    
    current_q1, current_q2 = self.critic(action_seq, obs_cond)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    
    return critic_loss
```

## 5. 与当前实现的对比

| 方面 | 当前实现 | 正确实现 |
|------|----------|----------|
| Reward | 单步 $r_t$ | 累计 $\sum_{i=0}^{\tau-1} \gamma^i r_{t+i}$ |
| Next State | $s_{t+1}$ | $s_{t+\tau}$ |
| Discount | $\gamma$ | $\gamma^\tau$ |
| Done | 单步 $d_t$ | chunk 内任意 $d_t^{(\tau)}$ |

## 6. 参考文献

- Semi-Markov Decision Processes (SMDP)
- Temporal Abstraction in Reinforcement Learning
- Diffusion-QL: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
