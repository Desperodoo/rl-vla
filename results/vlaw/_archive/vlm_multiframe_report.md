# VLM 16帧视频序列评估报告 — LiftPegUpright-v1

## 数据
- 轨迹: 170 (成功=59, 失败=111)
- 来源: /home/wjz/rl-vla/data/vlaw/rollouts/iter1/LiftPegUpright-v1, /home/wjz/rl-vla/data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1
- 指令: "Lift the peg and insert it upright into the holder."
- 帧数: min=1, max=16, mean=10.1

## 旧基线
- 单帧 zero-shot AUC = 0.5852

## 结果

### single_frame_images (1帧 (最后一帧))
- **ROC-AUC: 0.5793**
- p_yes (成功): 0.0006 ± 0.0009
- p_yes (失败): 0.0009 ± 0.0019
- p_yes 范围: [0.0000, 0.0110]
- Youden 阈值: 0.0001 (recall=89.8%, FP=59.5%)
- FP<20% 阈值: 0.0012 (recall=18.6%, FP=16.2%)

**CM @ α=0.8:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail | 111 (TN) | 0 (FP) |
| GT Succ | 59 (FN) | 0 (TP) |

Acc=65.3%, FP rate=0.0%

**CM @ Youden 阈值=0.0001:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail | 45 (TN) | 66 (FP) |
| GT Succ | 11 (FN) | 48 (TP) |

Acc=54.7%, FP rate=59.5%
### 16frame_images (≤16帧 (images))
- **ROC-AUC: 0.8153**
- p_yes (成功): 0.0004 ± 0.0006
- p_yes (失败): 0.0001 ± 0.0002
- p_yes 范围: [0.0000, 0.0025]
- Youden 阈值: 0.0000 (recall=91.5%, FP=35.1%)
- FP<20% 阈值: 0.0001 (recall=67.8%, FP=18.0%)

**CM @ α=0.8:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail | 111 (TN) | 0 (FP) |
| GT Succ | 59 (FN) | 0 (TP) |

Acc=65.3%, FP rate=0.0%

**CM @ Youden 阈值=0.0000:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail | 72 (TN) | 39 (FP) |
| GT Succ | 6 (FN) | 53 (TP) |

Acc=73.5%, FP rate=35.1%
### 16frame_video (≤16帧 (video))
- **ROC-AUC: 0.6452**
- p_yes (成功): 0.0004 ± 0.0006
- p_yes (失败): 0.0001 ± 0.0001
- p_yes 范围: [0.0000, 0.0025]
- Youden 阈值: 0.0001 (recall=54.2%, FP=22.5%)
- FP<20% 阈值: 0.0001 (recall=44.1%, FP=18.9%)

**CM @ α=0.8:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail | 111 (TN) | 0 (FP) |
| GT Succ | 59 (FN) | 0 (TP) |

Acc=65.3%, FP rate=0.0%

**CM @ Youden 阈值=0.0001:**
|  | Pred Fail | Pred Succ |
|--|-----------|----------|
| GT Fail | 86 (TN) | 25 (FP) |
| GT Succ | 28 (FN) | 31 (TP) |

Acc=68.8%, FP rate=22.5%


## 对比表

| Config | AUC | p_yes(+) | p_yes(-) | Youden_θ | Recall@FP<20% |
|--------|-----|----------|----------|----------|---------------|
| single_frame_images | 0.5793 | 0.0006 | 0.0009 | 0.0001 | 18.6% |
| 16frame_images | 0.8153 | 0.0004 | 0.0001 | 0.0000 | 67.8% |
| 16frame_video | 0.6452 | 0.0004 | 0.0001 | 0.0001 | 44.1% |

## AUC 提升
- 旧基线: 0.5852
- 最佳16帧 (16frame_images): 0.8153
- **ΔAUC = +0.2301**
