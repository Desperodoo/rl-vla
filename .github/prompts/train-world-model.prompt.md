---
name: train-world-model
description: "训练/微调 Ctrl-World 世界模型"
agent: WM-Agent
tools: ['runCommands', 'read', 'edit']
---

# 训练世界模型

使用收集的数据训练或微调 Ctrl-World 世界模型。

## 步骤
1. 读取 `.github/vlaw-status.md` 确认训练数据就绪
2. 确认训练配置:
   - Phase A (预热): 仅 Action Encoder + temporal attention, ~10K steps
   - Phase B (全量): UNet 全部解冻, ~50K steps
3. 在 GPU 0-3 上启动 DDP 训练:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
     rlft/vlaw/train_world_model.py \
     --data_dir data/vlaw/encoded/ \
     --output_dir checkpoints/vlaw/world_model/ \
     --mixed_precision fp16 \
     --gradient_checkpointing
   ```
4. 监控训练 loss (WandB)
5. 验证: Action Replay PSNR > 18
6. 更新 `.github/vlaw-status.md`

## 参数
- 训练阶段: ${input:phase:A}
- 训练步数: ${input:num_steps:10000}
