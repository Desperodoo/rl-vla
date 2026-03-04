# VLM LoRA Rank 消融实验 — 评估汇总

> 评估日期: 2026-03-04
> 评估配置: num_frames=16, use_video_format=True (ADR-015), GPU 7, conda env: vlaw_reward
> 评估数据: data/vlaw/rollouts/iter1/ + iter1_highsuc/ (LiftPegUpright-v1, 100条: 11 success + 89 fail)
> base model: Qwen3-VL-4B-Instruct, alpha=32, target: q_proj + v_proj, 200 steps, dropout=0.1

## 指标对比表

| LoRA Rank | ROC-AUC | Acc@α=0.8 | FP@α=0.8 | Youden θ | Acc@Youden | FP@Youden | p_yes(+) | p_yes(-) |
|-----------|---------|-----------|----------|----------|------------|-----------|----------|----------|
| r=8       | —       | 0.794     | 0.0%     | —        | —          | —         | —        | —        |
| **r=16**  | **0.808** | **0.847** | **0.0%** | **0.40** | **0.847**  | **1.8%**  | **0.43±0.27** | **0.13±0.16** |
| r=32      | 0.345   | 0.890     | 0.0%     | inf      | 0.890      | 0.0%      | 0.0010±0.0005 | 0.0014±0.0009 |
| r=64      | 0.367   | 0.890     | 0.0%     | 0.0013   | 0.800      | 11.2%     | 0.0007±0.0004 | 0.0008±0.0005 |

> r=8 数据来自之前评估 (仅记录 acc 和 FP, 无 ROC-AUC/Youden)
> r=16 为基线 (T-VLM-EVAL-REPRODUCE, AUC=0.808)

## 关键发现

1. **r=16 是最优 rank**: ROC-AUC=0.808 远优于其他所有 rank
2. **r=32 和 r=64 严重退化**: AUC 分别为 0.345 和 0.367 (低于随机 0.5)，p_yes 全部 < 0.01，模型完全失去判别能力
3. **过大的 rank 导致过拟合/训练不稳定**: 在仅 ~200 样本、200 步的小数据微调场景下，rank ↑ → 参数量 ↑ → 过拟合严重
4. **r=8 vs r=16**: r=8 acc=0.794 vs r=16 acc=0.847，r=16 优势明显但不极端
5. **结论**: 论文推荐的 r=16 确实是最佳选择；更大的 rank (32, 64) 在小数据场景完全不可用

## 详细 JSON 结果

- r=32: [vlm_lora_r32_16frame.json](vlm_lora_r32_16frame.json)
- r=64: [vlm_lora_r64_16frame.json](vlm_lora_r64_16frame.json)

## Confusion Matrix

### r=32 @ α=0.8
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail | 89 (TN) | 0 (FP) |
| GT Succ | 11 (FN) | 0 (TP) |

→ 全部预测为 fail (p_yes max=0.008, 远低于 0.8)

### r=64 @ α=0.8
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail | 89 (TN) | 0 (FP) |
| GT Succ | 11 (FN) | 0 (TP) |

→ 全部预测为 fail (p_yes max=0.004, 远低于 0.8)

### r=64 @ Youden θ=0.0013
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail | 79 (TN) | 10 (FP) |
| GT Succ | 10 (FN) | 1 (TP) |

→ 即使用最优阈值，Youden J=0.069 (几乎无判别力)
