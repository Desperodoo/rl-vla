# VLAW 论文 vs 我们的实现：VLM 奖励模型微调对比报告

> **创建时间**: 2026-02-27 | **论文**: VLAW (arXiv:2602.12063)
> **目的**: 对比 VLAW 论文 Appendix C 中 VLM 微调方案与我们 `train_reward_model.py` 的实现差异

---

## 1. 论文原文要求 (Section 4.1 + Appendix C)

论文关键段落（原文摘录）：

> *"Each trajectory is **temporally downsampled into a 16-frame video** before being fed to the model. We **fine-tune the Qwen3-VL-4B-Instruct model for 200 steps with batch size 128**."*
>
> *"We instead examine the model-assigned **probability of the 'yes' token** and only label a trajectory as successful when this probability exceeds a **threshold of 0.8**"*
>
> *"we find that the **zero-shot VLM is not accurate enough**, so in the first iteration, we **fine-tune the VLM** with the success labels r_τ in D_real."*

---

## 2. 逐项超参对比

| 维度 | VLAW 论文 (Appendix C) | 我们的实现 (`train_reward_model.py`) | 一致? | 备注 |
|------|----------------------|-------------------------------------|-------|------|
| **基座模型** | Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Instruct (`qwen_vl/`) | ✅ 一致 | — |
| **微调方法** | LoRA（引用 RoboReward） | LoRA (PEFT) | ✅ 一致 | 论文和 RoboReward 均使用 LoRA |
| **训练步数** | **200 steps** | `train_steps=200` | ✅ 一致 | — |
| **Batch size** | **128** | `per_device_batch_size=1 × gradient_accumulation_steps=128` → eff=128 | ✅ 一致 | 通过梯度累积实现等效 batch |
| **输入帧数** | **16 帧**均匀下采样 | `num_frames=16` | ✅ 一致 | — |
| **判定阈值** | α = 0.8 (`P('yes') > 0.8`) | `threshold=0.8` | ✅ 一致 | — |
| **P('yes') 提取** | softmax(yes/no token logits) | `softmax([logits[yes_id], logits[no_id]])[0]` | ✅ 一致 | 我们额外支持大小写变体聚合 |
| **LoRA rank** | **论文未明确指定** | `lora_r=16, lora_alpha=32` | ⚠️ 推测值 | 可能参考 RoboReward |
| **LoRA target modules** | **论文未明确指定** | `["q_proj", "v_proj"]` | ⚠️ 推测值 | 常见 VLM LoRA 配置 |
| **学习率** | **论文未指定** | `lr=2e-5` | ⚠️ 推测值 | VLM LoRA 微调常见值 |
| **Warmup** | **论文未指定** | `warmup_steps=20` (LinearLR 0.1→1.0) | ⚠️ 推测值 | — |
| **Weight decay** | **论文未指定** | `weight_decay=0.01` | ⚠️ 推测值 | AdamW 默认 |
| **梯度裁剪** | **论文未指定** | `clip_grad_norm=1.0` | ⚠️ 推测值 | 标准做法 |
| **损失函数** | **论文未说明**（推测 SFT） | teacher-forcing LM loss (`model(**inputs, labels=input_ids).loss`) | ✅ 合理 | 标准 causal LM SFT |
| **输入格式** | "16-frame video" | 多张图 (images 模式) | ⚠️ 有差异 | 见下方分析 |
| **Gradient checkpointing** | **论文未指定** | ✅ 已启用 | ✅ 合理 | 节省显存 |

---

## 3. 数据量与数据分布对比

| 维度 | VLAW 论文 (DROID) | 我们 (ManiSkill) | 差距 |
|------|-------------------|------------------|------|
| **平台** | DROID 真机 (Franka + 3 相机, 320×192) | ManiSkill3 仿真 (2 相机, 128×128) | 分辨率/视角差异 |
| **任务数** | 5 个 DROID 任务 | 1 个 (LiftPegUpright-v1) | 我们聚焦单任务验证 |
| **每任务 rollout** | **K=50 条/任务** → **~250 条总计** | 50(iter1) + 50(highsuc) = **~100 条** | 单任务看约 2× |
| **推估成功率** | 真机策略 ~20-40% 成功 | iter1:16% + highsuc:70% → 混合 ~43% | 我们偏高 |
| **正样本数** | 推估 ~50-100 条成功 | ~43 条成功 | 数量级相当 |
| **负样本数** | 推估 ~150-200 条失败 | ~57 条失败 | ⚠️ 负样本偏少 |
| **正负比例** | 推估 ~1:3 到 ~1:4 | **~1:1.3** | ⚠️ 类别分布差异较大 |
| **标签来源** | 真机 success 标签 | ManiSkill `env_success_at_end` | ✅ 等价 |

---

## 4. 关键差异详细分析

### 差异 1: 输入格式 — "video" vs "images"

**论文原文**: *"temporally downsampled into a 16-frame video"*

**实际情况**: Qwen3-VL 支持两种多帧输入模式：
- `video` 模式：将帧编码为视频特征，内部可能做时间降采样
- `images` 模式：每帧独立编码为图像特征

**我们的实验结果** (`eval_vlm_multiframe.py`, 170 条轨迹)：

| 配置 | AUC | Recall@FP<20% |
|------|-----|---------------|
| 单帧（最后一帧） | 0.5793 | 18.6% |
| **16帧多图 (images)** | **0.8153** | **67.8%** |
| 16帧视频 (video) | 0.6452 | 44.1% |

**结论**: images 模式 AUC=0.82 远高于 video 模式 AUC=0.65。原因可能是 Qwen3-VL 对 video 输入做内部时间降采样导致关键帧丢失。论文说 "video" 可能只是描述性用语（指"视频序列"而非技术格式）。

**建议**: 保持 images 模式。这是一个合理的工程优化。

### 差异 2: LoRA 具体配置（rank、target modules、lr）

论文仅指定了 **200 steps + batch=128 + 16帧 + LoRA**，未给出 rank、target modules、learning rate 等细节。

我们当前配置的来源：
- `lora_r=16, lora_alpha=32` — 可能参考了 RoboReward (arXiv:2601.00675) 或通用最佳实践
- `target_modules=["q_proj", "v_proj"]` — VLM LoRA 最常见的注入点
- `lr=2e-5` — Qwen 系列 LoRA 微调推荐值

VLAW 引用了 **RoboReward** 作为 VLM 奖励模型的技术基础。如需进一步确认 LoRA 细节，应参考 RoboReward 原文。

### 差异 3: 正负样本比例不平衡

| | VLAW (推估) | 我们 |
|--|------------|------|
| 正:负比 | ~1:3 ~ 1:4 | ~1:1.3 |

我们的 `iter1_highsuc` 数据有 70% 成功率，导致正样本比例偏高。这可能导致微调后的模型**对成功判定更宽松**（FP 率偏高）。

**潜在影响**:
- 模型倾向于输出更高的 P('yes')，α=0.8 阈值可能不够保守
- 合成轨迹标注时 FP 率可能超过预期

**建议**:
1. 训练时对正样本下采样，或对负样本过采样，使正负比接近 1:3
2. 或者在损失函数中加入类别权重 (class weight)
3. 评估时重点关注 FP rate，如超过 20% 则调高阈值

### 差异 4: 数据规模（单任务 vs 多任务）

VLAW 在 5 个 DROID 任务上共 ~250 条轨迹微调 VLM。我们当前只用 LiftPegUpright-v1 的 ~100 条。

**影响**: 单任务数据可能导致 VLM 过拟合到特定视觉模式。但由于 LoRA 参数量很小 (~24MB adapter)，且只训 200 步，过拟合风险有限。

---

## 5. 我们额外做了的（论文未提及）

| 额外实现 | 说明 |
|----------|------|
| Gradient checkpointing | 节省显存，使单卡 4090 可运行 |
| 大小写变体 token 聚合 | `reward_model.py` 聚合 yes/Yes/YES 等变体的 logit |
| 多数据目录支持 | `data_dirs` 参数支持从多个路径加载数据 |
| Accelerate 多卡支持 | 已实现 `multi_gpu=True` 模式（当前未使用） |
| Flash Attention 2 | 自动检测并启用（加速推理 + 训练） |
| 定期评估 + checkpoint | 每 50 步评估一次 + 保存 checkpoint |

---

## 6. 一致性总结

### ✅ 已确认一致（论文明确指定）

- 基座模型 (Qwen3-VL-4B-Instruct)
- 微调方法 (LoRA)
- 训练步数 (200 steps)
- Batch size (128，通过梯度累积)
- 输入帧数 (16 帧均匀下采样)
- 判定阈值 (α=0.8)
- P('yes') 概率提取方式 (softmax over yes/no logits)

### ⚠️ 推测但合理（论文未指定）

- LoRA rank (r=16)
- LoRA alpha (32)
- LoRA target modules (q_proj, v_proj)
- 学习率 (2e-5)
- Warmup (20 steps)
- Weight decay (0.01)
- 梯度裁剪 (1.0)
- 损失函数 (teacher-forcing SFT)

### ❗ 存在差异（需关注）

- **输入格式**: 我们用 images 模式（AUC 更高），论文说 "video"
- **正负样本比例**: 我们 ~1:1.3 vs 论文推估 ~1:3
- **数据规模**: 单任务 ~100 条 vs 论文 5 任务 ~250 条

---

## 7. 改进建议

| 优先级 | 建议 | 理由 |
|--------|------|------|
| **高** | 调整训练数据正负比至 ~1:3 | 对齐论文数据分布，减少 FP |
| 中 | 查阅 RoboReward 论文确认 LoRA 细节 | 补充论文未指定的超参 |
| 低 | 尝试 video 模式 + LoRA 微调后的 AUC | 确认 images vs video 在微调后是否仍有差异 |
| 低 | 添加类别加权损失 | 进一步控制类别不平衡 |

---

## 附录：当前训练命令

```bash
# tmux:vlm_train (GPU 6)
CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
  --gradient_accumulation_steps 128
```

## 附录：相关文件

- 训练脚本: `rlft/vlaw/reward/train_reward_model.py`
- 推理模型: `rlft/vlaw/reward/reward_model.py`
- 多帧评估: `scripts/eval_vlm_multiframe.py`
- 复现计划: `.github/VLAW_REPRODUCTION_PLAN.md` (Section 3.3)
- 论文: VLAW arXiv:2602.12063, Section 4.1 + Appendix C
- 参考: RoboReward arXiv:2601.00675
