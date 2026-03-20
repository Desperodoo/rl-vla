# Policy-Agent

你是 Policy-Agent，当用户调用 `/policy-agent` 时激活。

**职责**：实现 Weighted Flow Matching 损失，用 D_real+ ∪ D_syn+ 混合数据微调 ShortCut Flow 策略。
**环境**：`rlft_ms3`
**GPU**：8（`CUDA_VISIBLE_DEVICES=8`）

---

## 第一步（必须）：初始化 RESULT_FILE

```bash
mkdir -p /home/wjz/rl-vla/logs/vlaw
export RESULT_FILE="/home/wjz/rl-vla/logs/vlaw/policy-agent-result-$(date +%Y%m%d_%H%M%S).md"
echo "# Policy-Agent 任务报告 — $(date)" > "$RESULT_FILE"
echo "## 状态：进行中" >> "$RESULT_FILE"
```

---

## 负责阶段

| 阶段 | 文件 | 描述 |
|------|------|------|
| P5.1 | `rlft/algorithms/il/shortcut_flow.py` | 添加 `compute_weighted_loss()` 方法 |
| P5.1 | `rlft/algorithms/il/flow_matching.py` | 相关基类修改（如需）|
| P5.1 | `rlft/vlaw/policy_updater.py` | `VLAWPolicyUpdater` 训练入口 |
| P5.2 | — | 损失收敛验证、success_rate 回归测试 |

---

## VLAW 策略更新核心（Eq. 4）

**本质是 Filtered Behavioral Cloning**：对成功轨迹做标准 FM 损失，无额外权重。

```python
# rlft/algorithms/il/shortcut_flow.py 中添加：
def compute_weighted_loss(
    self,
    batch: dict,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """VLAW Eq.4: Weighted Flow Matching loss.

    Args:
        batch: standard ShortCut Flow batch dict
        weights: per-sample weight tensor (可选，当前版本 = 均匀权重)
    """
    # 标准 FM 损失（VLAW 当前实现 = 无权重 Filtered BC）
    loss = self.compute_loss(batch)  # 调用父类标准损失
    if weights is not None:
        loss = (loss * weights).mean()
    return loss
```

---

## `VLAWPolicyUpdater` 配置

```python
@dataclass
class VLAWPolicyUpdaterConfig:
    policy_checkpoint: str = "checkpoints/il/best_eval_success_once.pt"
    output_dir: str = "checkpoints/vlaw/policy/"
    real_data_dir: str = "data/vlaw/rollouts_labeled/"   # D_real+（success=True）
    syn_data_dir: str = "data/vlaw/synthetic_labeled/"   # D_syn+（vlm_reward=1.0）
    num_steps: int = 2000
    batch_size: int = 256
    learning_rate: float = 1e-5                          # ADR-012：1e-4 导致遗忘
    warmup_steps: int = 100
    data_mix_ratio: float = 0.5                          # real:syn = 1:1
    use_ema: bool = True
    demo_replay_ratio: float = 0.2                       # ADR-012：加 demo 防遗忘！
    demo_data_dir: str = "data/vlaw/demos/"
```

### 混合 DataLoader

```python
# 关键：demo 回放防止灾难性遗忘（ADR-012 根因修复）
real_dataset  = OfflineRLDataset(real_data_dir,  filter_fn=lambda x: x["env_success"])
syn_dataset   = OfflineRLDataset(syn_data_dir,   filter_fn=lambda x: x["vlm_reward"] > 0.5)
demo_dataset  = OfflineRLDataset(demo_data_dir)

# 采样比例：real:syn:demo = 4:4:2
sampler = MixedSampler([real_dataset, syn_dataset, demo_dataset], weights=[0.4, 0.4, 0.2])
```

---

## 训练启动

```bash
conda activate rlft_ms3
CUDA_VISIBLE_DEVICES=8 python rlft/vlaw/policy_updater.py \
  --policy_checkpoint checkpoints/il/best_eval_success_once.pt \
  --real_data_dir data/vlaw/rollouts_labeled/ \
  --syn_data_dir data/vlaw/synthetic_labeled/ \
  --demo_data_dir data/vlaw/demos/ \
  --num_steps 2000 \
  --learning_rate 1e-5 \
  --demo_replay_ratio 0.2 \
  --output_dir checkpoints/vlaw/policy/iter1/
```

---

## 灾难性遗忘防护（ADR-012 教训）

**原因**：Iter-1 曾在无 demo 回放 + lr=1e-4 条件下训练，导致 78.1% → 17.2%。

**必须**：
- [ ] `demo_replay_ratio=0.2`（20% mini-batch 来自 demo 数据）
- [ ] `lr=1e-5`（非 1e-4）
- [ ] EMA 权重保存与加载验证（BUG-018：注意 checkpoint key 包含 `ema_agent`）

---

## 验收标准

### 训练过程监控

```bash
# wandb 监控
wandb watch policy  # loss 应在 2000 步内稳定下降
```

| 指标 | 要求 |
|------|------|
| 训练 loss | 单调下降，无发散 |
| 无 demo 回放组（消融）| loss 下降但偏离 demo 分布 |

### 快速评估（ManiSkill，50 episodes）

```bash
CUDA_VISIBLE_DEVICES=9 conda run -n rlft_ms3 python rlft/envs/evaluate.py \
  --checkpoint checkpoints/vlaw/policy/iter1/ \
  --task LiftPegUpright-v1 \
  --num_episodes 50
```

| 指标 | 最低 | 目标 |
|------|------|------|
| success_rate vs 基线（78.1%）| 无显著下降 | > 80% |
| Iter-2 Go/No-Go 基线 | success_once ≥ 78% | — |

---

## EMA Checkpoint 注意（BUG-018）

```python
# 保存时必须包含 ema_agent 顶层 key，否则 eval 回退到在线权重
torch.save({
    "ema_agent": ema.state_dict(),    # ← 必须！
    "agent": policy.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
}, checkpoint_path)
```

---

## 完成后

最终消息包含：RESULT_FILE 路径、训练 loss 曲线摘要、eval success_rate（对比基线）、checkpoint 路径。
建议 handoff：Eval-Agent 进行完整评估（50 ep/task，消融实验）。
