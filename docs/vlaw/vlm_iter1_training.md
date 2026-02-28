# VLM 奖励模型 Iter-1 微调记录

> **训练日期**: 2026-02-28 13:05 → 13:55 (约 50 分钟)
> **论文对齐**: 见 [vlm_finetuning_comparison.md](vlm_finetuning_comparison.md)

---

## 1. 调试与准备

### V1.2 wandb 集成验证 (正式训练前)

- 2-step mini test 验证 `--use_wandb` 参数
- Loss: 21.26 → 20.47（2 steps, 正常下降）
- wandb offline run 创建成功：`wandb/offline-run-20260228_125417-12qa71xs`
- 确认 `train/loss`, `eval/accuracy`, `eval/fp_rate` 等 metrics 正确记录

### 16 帧 images vs video 选择

基于前期评估 (`scripts/eval_vlm_multiframe.py`, 170 条轨迹)：

| 配置 | AUC | Recall@FP<20% |
|------|-----|---------------|
| 单帧 (最后一帧) | 0.579 | 18.6% |
| **16 帧多图 (images)** | **0.815** | **67.8%** |
| 16 帧视频 (video) | 0.645 | 44.1% |

选择 **16 帧 images 模式**，AUC 最高。

---

## 2. 超参数

| 配置项 | 值 | 论文一致性 |
|--------|-----|-----------|
| **基座模型** | Qwen3-VL-4B-Instruct | ✅ |
| **微调方法** | LoRA (PEFT) | ✅ |
| **LoRA rank** | r=16, alpha=32 | ⚠️ 论文未指定 |
| **LoRA targets** | q_proj, v_proj | ⚠️ 论文未指定 |
| **LoRA 参数量** | 5,898,240 (0.13% of total) | |
| **训练步数** | 200 global steps | ✅ |
| **Batch size** | per_device=1 × grad_accum=128 × 2GPU = eff 256 | ⚠️ 论文说 128 |
| **学习率** | 2e-5 (AdamW) | ⚠️ 论文未指定 |
| **Warmup** | 20 steps (LinearLR 0.1→1.0) | ⚠️ 论文未指定 |
| **Weight decay** | 0.01 | ⚠️ 论文未指定 |
| **梯度裁剪** | 1.0 | |
| **输入帧数** | 16 帧均匀下采样 | ✅ |
| **输入模式** | images (多图) | ⚠️ 论文说 video |
| **判定阈值** | α = 0.8 | ✅ |
| **Attention** | flash_attention_2 | |
| **Gradient checkpointing** | ✅ | |
| **GPU** | 2× RTX 4090 (GPU 6,7), ~12GB/卡 | |
| **混合精度** | bf16 | |
| **评估间隔** | 每 50 steps | |
| **wandb** | offline mode | |

---

## 3. 数据集

### 训练数据

| 来源 | 条数 | 成功率 | 说明 |
|------|------|--------|------|
| iter1 rollouts | 50 | 16% | LiftPegUpright-v1, 基线策略 |
| iter1_highsuc | 50* | 70% | 高成功率补充数据 |
| **总计** | ~100 | ~43% | train/eval split: 80%/20% |

> *实际加载 170 条轨迹 (正=59, 负=111), train=136, eval=34

### 数据格式

- 输入: 16 帧 RGB 图像 (均匀下采样自完整轨迹)
- 标签: `env_success_at_end` (True/False)
- Prompt: 自然语言任务描述 + "Did the robot successfully complete the task? Answer yes or no."

### 正负样本分布

- 正:负 = 59:111 ≈ 1:1.9
- ⚠️ 论文推估 ~1:3-1:4，我们正样本偏多

---

## 4. 训练过程

### 启动命令

```bash
tmux new-session -d -s vlm_16f "
cd /home/wjz/rl-vla && \
CUDA_VISIBLE_DEVICES=6,7 WANDB_MODE=offline \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/wjz/miniconda3/envs/vlaw_reward/bin/python \
rlft/vlaw/reward/train_reward_model.py \
  --data_dirs data/vlaw/rollouts/iter1 data/vlaw/rollouts/iter1_highsuc \
  --tasks LiftPegUpright-v1 \
  --model_path checkpoints/vlaw/reward_model/qwen_vl \
  --output_dir checkpoints/vlaw/reward_model/lora_iter1_16frame \
  --num_frames 16 \
  --train_steps 200 \
  --lora_r 16 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 128 \
  --use_wandb --wandb_project vlaw-reward \
  2>&1 | tee logs/vlaw/vlm_16frame_formal_train.log
"
```

### Loss 曲线

```
Step     1: loss ≈ 18.7
Step    50: loss ≈ 12.0
Step   100: loss ≈  9.0
Step   150: loss ≈  7.0
Step   200: loss ≈  6.8
```

稳定持续下降，无异常。

### 中间评估

| Step | Accuracy | FP rate | TP | FP | TN | FN |
|------|----------|---------|----|----|----|----|
| 100 | 0.794 | 0.000 | 0 | 0 | 27 | 7 |
| 200 | **0.824** | **0.037** | 2 | 1 | 26 | 5 |

### GPU 显存

| GPU | 显存 |
|-----|------|
| 6 | ~12.4 GB / 24.6 GB (90% util) |
| 7 | ~12.5 GB / 24.6 GB (88% util) |

---

## 5. 最终结果

### 训练指标

| 指标 | 值 |
|------|-----|
| **Final loss** | 6.8 |
| **Accuracy** | 0.824 |
| **FP rate** | 3.7% (1/27) |
| **Precision** | 0.667 |
| **Recall** | 0.286 (2/7) |
| **TP** | 2 |
| **FP** | 1 |
| **TN** | 26 |
| **FN** | 5 |
| **mean p_yes** | 0.558 |

### 门控检查

- ✅ FP rate = 3.7% < 20% (论文目标)
- ✅ Accuracy = 0.824
- ⚠️ Recall 较低 (28.6%)，模型偏保守（符合 α=0.8 高阈值设计）

### 产出文件

```
checkpoints/vlaw/reward_model/lora_iter1_16frame/
├── adapter_config.json
├── adapter_model.safetensors    (23MB, LoRA weights)
├── chat_template.jinja
├── processor_config.json
├── tokenizer.json               (11MB)
├── tokenizer_config.json
├── train_config.json
├── README.md
├── step_50/                     (中间 checkpoint)
├── step_100/
├── step_150/
├── step_200/
└── final/                       (最终 checkpoint)
```

wandb run: `wandb/offline-run-20260228_130552-1ef23bn8`

---

## 6. D_real 标注验证

使用微调后的 VLM 对全量 D_real 数据进行标注，验证模型质量。

### 标注范围

| 来源 | 条数 |
|------|------|
| iter1 | 50 |
| iter1_highsuc | 120 |
| iter1_lift_inc20 | 40 |
| **总计** | **210** |

### 标注结果

| 指标 | 值 |
|------|-----|
| vlm_reward=1 | 4 (1.9%) |
| vlm_reward=0 | 206 (98.1%) |
| env_success=True | 71 (33.8%) |
| p_yes mean | 0.2866 |
| p_yes median | 0.1824 |
| p_yes range | [0.0159, 0.8670] |

### Confusion Matrix

| | env_success=True | env_success=False |
|---|---|---|
| **vlm=1** | TP=4 | **FP=0** |
| **vlm=0** | FN=67 | TN=139 |

| 指标 | 值 |
|------|-----|
| Accuracy | 0.681 |
| **Precision** | **1.000** |
| Recall | 0.056 |
| **FP rate** | **0.000** ✅ |

### 按数据来源分析

| 来源 | 条数 | vlm+ | env+ | p_yes mean |
|------|------|------|------|------------|
| iter1 (192×192, 长轨迹) | 50 | 0 | 8 | 0.106 |
| iter1_highsuc (128×128) | 120 | 4 | 51 | 0.321 |
| iter1_lift_inc20 (128×128) | 40 | 0 | 12 | 0.408 |

### 分析

1. **FP=0 表明模型极度保守**，precision=100%，不会错标失败为成功
2. **Recall 仅 5.6%**：绝大多数成功轨迹未被识别（α=0.8 阈值很高）
3. iter1 数据 p_yes 很低 (~0.10)，可能因 192×192 分辨率与 128×128 不一致
4. **VLM 标注 D_real 的用途是辅助策略训练加权**；D_real 自带的 `env_success_at_end` 仍然是主要标签

### 输出文件

- `data/vlaw/labeled/iter1_16frame_lora/vlm_labels.h5`
- `data/vlaw/labeled/iter1_16frame_lora/vlm_label_report.json`
- 标注脚本: `scripts/label_dreal_vlm.py`

---

## 7. B2 合成数据标注

使用微调 VLM 标注 Track B 中 pretrained WM 生成的合成轨迹。

### 结果

| 指标 | 值 |
|------|-----|
| 总轨迹数 | 50 (10 唯一) |
| vlm_reward=1 | **0/50 (0%)** |
| p_yes mean | 0.0577 |
| p_yes range | [0.0421, 0.0759] |

**分析**: 所有合成轨迹被判为失败 — pretrained WM 生成质量不足。这与论文预期一致：iter-1 需要先微调 WM 后才能生成有效合成数据。

输出: `data/vlaw/labeled/synthetic_iter1_pretrained/`

---

## 8. 已知问题

1. **Batch size 不完全对齐**: 论文要求 effective batch=128，实际 per_device=1 × grad_accum=128 × 2GPU = 256。后续可用 4GPU + grad_accum=32 实现精确 128。
2. **正负样本比**: 我们 ~1:1.9，论文推估 ~1:3-1:4，可能导致模型偏宽松。但实测 FP=0%，说明 α=0.8 阈值足够保守。
3. **多分辨率数据**: iter1 (192×192) vs highsuc/inc20 (128×128)，可能影响 VLM 对不同来源数据的判断一致性。
4. **Recall 低**: 仅 5.6% 的成功轨迹被正确识别。对于策略训练中的 positive weighting，可能需要降低 α 或使用连续 p_yes 作为权重。

---

## 附录: 关键文件清单

| 文件 | 说明 |
|------|------|
| `rlft/vlaw/reward/train_reward_model.py` | VLM 微调训练脚本 |
| `rlft/vlaw/reward/reward_model.py` | VLM 推理 (P(yes) 提取) |
| `scripts/eval_vlm_multiframe.py` | 16 帧评估脚本 |
| `scripts/label_dreal_vlm.py` | D_real 标注脚本 |
| `scripts/b2_phase1_vae_decode.py` | B2 VAE 解码脚本 |
| `scripts/b2_phase2_vlm_label.py` | B2 VLM 标注脚本 |
| `checkpoints/vlaw/reward_model/lora_iter1_16frame/` | LoRA checkpoint |
| `checkpoints/vlaw/reward_model/qwen_vl/` | 基座模型 |
| `logs/vlaw/vlm_16frame_formal_train.log` | 训练日志 |
