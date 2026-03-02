# VLM 1-Frame vs 16-Frame 评估报告 — LiftPegUpright-v1

## 数据
- 轨迹: 170 (success=59, fail=111)
- 来源: /home/wjz/rl-vla/data/vlaw/rollouts/iter1/LiftPegUpright-v1, /home/wjz/rl-vla/data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1
- Instruction: "Lift the peg and insert it upright into the holder."
- 帧数: min=1, max=16

## 结果

### zero_shot_single (1 frame (last))
- **ROC-AUC: 0.5793**
- p_yes (success): 0.0006 ± 0.0009
- p_yes (fail): 0.0009 ± 0.0019
- Youden's J threshold: 0.0001 (recall=89.8%, FP=59.5%)
- FP<20% threshold: 0.0012 (recall=18.6%, FP=16.2%)

**CM @ α=0.8:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail    | 111 (TN) | 0 (FP) |
| GT Succ    | 59 (FN) | 0 (TP) |

Acc=65.3%, FP rate=0.0%

**CM @ Youden threshold=0.0001:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail    | 45 (TN) | 66 (FP) |
| GT Succ    | 11 (FN) | 48 (TP) |

Acc=54.7%, FP rate=59.5%
### zero_shot_multi (≤16 frames (uniform))
- **ROC-AUC: 0.6452**
- p_yes (success): 0.0004 ± 0.0006
- p_yes (fail): 0.0001 ± 0.0001
- Youden's J threshold: 0.0001 (recall=54.2%, FP=22.5%)
- FP<20% threshold: 0.0001 (recall=44.1%, FP=18.9%)

**CM @ α=0.8:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail    | 111 (TN) | 0 (FP) |
| GT Succ    | 59 (FN) | 0 (TP) |

Acc=65.3%, FP rate=0.0%

**CM @ Youden threshold=0.0001:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail    | 86 (TN) | 25 (FP) |
| GT Succ    | 28 (FN) | 31 (TP) |

Acc=68.8%, FP rate=22.5%
### lora_single (1 frame (last))
- **ROC-AUC: 0.7553**
- p_yes (success): 0.6642 ± 0.1150
- p_yes (fail): 0.4780 ± 0.2190
- Youden's J threshold: 0.5927 (recall=81.4%, FP=36.0%)
- FP<20% threshold: 0.7058 (recall=44.1%, FP=18.9%)

**CM @ α=0.8:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail    | 105 (TN) | 6 (FP) |
| GT Succ    | 53 (FN) | 6 (TP) |

Acc=65.3%, FP rate=5.4%

**CM @ Youden threshold=0.5927:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail    | 71 (TN) | 40 (FP) |
| GT Succ    | 13 (FN) | 46 (TP) |

Acc=68.8%, FP rate=36.0%
### lora_multi (≤16 frames (uniform))
- **ROC-AUC: 0.8084**
- p_yes (success): 0.4783 ± 0.2872
- p_yes (fail): 0.1408 ± 0.1064
- Youden's J threshold: 0.5000 (recall=61.0%, FP=1.8%)
- FP<20% threshold: 0.2018 (recall=69.5%, FP=18.9%)

**CM @ α=0.8:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail    | 111 (TN) | 0 (FP) |
| GT Succ    | 55 (FN) | 4 (TP) |

Acc=67.6%, FP rate=0.0%

**CM @ Youden threshold=0.5000:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail    | 109 (TN) | 2 (FP) |
| GT Succ    | 24 (FN) | 35 (TP) |

Acc=84.7%, FP rate=1.8%


## 对比表

| Config | AUC | p_yes(+) | p_yes(-) | Youden_θ | Recall@FP<20% |
|--------|-----|----------|----------|----------|---------------|
| zero_shot_single | 0.5793 | 0.0006 | 0.0009 | 0.0001 | 18.6% |
| zero_shot_multi | 0.6452 | 0.0004 | 0.0001 | 0.0001 | 44.1% |
| lora_single | 0.7553 | 0.6642 | 0.4780 | 0.5927 | 44.1% |
| lora_multi | 0.8084 | 0.4783 | 0.1408 | 0.5000 | 69.5% |

## 结论
16帧显著优于单帧
