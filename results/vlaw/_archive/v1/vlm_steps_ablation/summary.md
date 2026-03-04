# T-EXP-VLM-03: VLM Training Steps Ablation

| Steps | ROC-AUC | Acc@α=0.8 | FP@α=0.8 | Source |
|-------|---------|-----------|----------|--------|
| 100 | 0.3146 | 0.8900 | 0.0000 | ablation_100steps |
| 200 | 0.8080 | 0.8240 | 0.0370 | baseline (lora_iter1_16frame) |
| 400 | 0.3830 | 0.8900 | 0.0000 | ablation_400steps |
| 800 | 0.3897 | 0.8900 | 0.0000 | ablation_800steps |
