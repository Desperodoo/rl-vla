# VLM Baseline Report — LiftPegUpright-v1

## 数据统计
- 轨迹总数: 160 (success=63, fail=97)
- 数据源: `data/vlaw/rollouts/iter1_lift_only/LiftPegUpright-v1/`
- 评估方式: 最后一帧 (single frame)
- Instruction: "Pick up the peg and lift it upright."

## 结果

### Zero-shot (Qwen3-VL-4B)
- ROC-AUC: **0.5852**
- p_yes (success): 0.0833 ± 0.0691
- p_yes (fail): 0.0675 ± 0.0729
- Best threshold (Youden's J): 0.0141

**Confusion Matrix @ α=0.8:**
|  | Pred Fail | Pred Success |
|--|-----------|--------------|
| GT Fail    | 97 (TN) | 0 (FP) |
| GT Success | 63 (FN) | 0 (TP) |

Accuracy: 60.6%, FP Rate: 0.0%

**Confusion Matrix @ Best threshold=0.0141:**
|  | Pred Fail | Pred Success |
|--|-----------|--------------|
| GT Fail    | 37 (TN) | 60 (FP) |
| GT Success | 3 (FN) | 60 (TP) |

Accuracy: 60.6%, FP Rate: 61.9%
### LoRA fine-tuned
- ROC-AUC: **0.6165**
- p_yes (success): 0.0091 ± 0.0057
- p_yes (fail): 0.0071 ± 0.0068
- Best threshold: 0.0041

**Confusion Matrix @ α=0.8:**
|  | Pred Fail | Pred Success |
|--|-----------|--------------|
| GT Fail    | 97 (TN) | 0 (FP) |
| GT Success | 63 (FN) | 0 (TP) |

Accuracy: 60.6%, FP Rate: 0.0%

**Confusion Matrix @ Best threshold=0.0041:**
|  | Pred Fail | Pred Success |
|--|-----------|--------------|
| GT Fail    | 37 (TN) | 60 (FP) |
| GT Success | 9 (FN) | 54 (TP) |

Accuracy: 56.9%, FP Rate: 61.9%


## 结论
Zero-shot p_yes 极低 (< 0.15)，α=0.8 阈值无法使用。需 LoRA fine-tune 后才能有效标注 D_syn。
