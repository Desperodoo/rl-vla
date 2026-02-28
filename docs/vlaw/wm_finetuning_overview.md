# Ctrl-World 世界模型微调技术概览

> 最后更新: 2026-02-28 | 对应代码: `ctrl_world/models/ctrl_world.py`, `ctrl_world/scripts/train_wm.py`

---

## 1. 数据准备

### 原始数据来源
ManiSkill3 仿真环境中的 rollout 轨迹，每条轨迹包含：
- **RGB 图像**: 384×192 (双相机拼接为 384 高度), uint8
- **7D 动作**: (dx, dy, dz, drx, dry, drz, gripper)
- **任务描述**: 自然语言指令 (如 "Lift the peg and insert it upright")

### VAE 编码
RGB 图像通过 SVD (Stable Video Diffusion) 的 VAE encoder 编码为 latent 表示：
- 输入: `(T, 3, 384, 192)` RGB
- 输出: `(T, 4, 48, 24)` float16 latent (空间下采样 8×)
- 存储格式: HDF5，每条轨迹包含 `latent`, `action`, `text` 字段

### 滑窗切分
长轨迹被切成固定长度窗口 `num_history (1) + num_frames (15) = 16 帧`，形成训练 sample：

| 字段 | 形状 | 说明 |
|------|------|------|
| `latent` | `(16, 4, 48, 24)` | 视频 latent 序列 |
| `action` | `(16, 7)` | 对应的机器人动作 |
| `text` | string | 任务指令 |

### 当前数据规模
- 235 条轨迹 (demo 25 + highsuc 50 + inc20 40 + 原始 iter1 rollout 120+)
- 滑窗切分后: ~4378 训练窗口 + ~60 验证窗口
- 路径: `data/vlaw/encoded/reencode_highsuc_inc20/LiftPegUpright-v1/`

---

## 2. 模型架构

Ctrl-World 基于 Stable Video Diffusion (SVD) 架构:

```
┌─────────────────────────────────────────┐
│              CrtlWorld                   │
│                                          │
│  ┌──────────┐  ┌──────────────────────┐ │
│  │ VAE      │  │ UNet (~1.5B params)  │ │
│  │ (冻结)   │  │  - Spatial Attn      │ │
│  │ Encoder  │  │  - Temporal Attn     │ │
│  │ + Decoder│  │  - Cross Attn        │ │
│  └──────────┘  └──────────────────────┘ │
│                                          │
│  ┌──────────┐  ┌──────────────────────┐ │
│  │ CLIP     │  │ Action Encoder       │ │
│  │ Text Enc │  │ (MLP: 7D→1024)      │ │
│  │ (冻结)   │  │ + Text Embedding     │ │
│  └──────────┘  └──────────────────────┘ │
└─────────────────────────────────────────┘
```

### 组件职责
- **VAE**: 编码/解码 RGB ↔ latent，完全冻结
- **CLIP Text Encoder**: 编码任务指令文本，完全冻结
- **UNet**: 扩散模型核心，预测去噪后的视频 latent
- **Action Encoder**: MLP 将 7D 动作 + CLIP text embedding 映射为 1024 维条件向量

### 微调策略
- **Phase-A** (可选): 冻结 UNet 空间层，只训练 temporal attention + action encoder
- **Phase-B** (当前使用): 全量微调 UNet + action encoder

---

## 3. Loss 设计

使用 **EDM (Elucidating Diffusion Models) 风格的 x₀-prediction weighted MSE loss**。

### 扩散前向过程

```python
# 噪声级别采样 (log-normal)
σ ~ exp(N(μ=0.7, σ²=1.6²))

# EDM 参数化
c_skip = 1 / (σ² + 1)
c_out  = -σ / √(σ² + 1)
c_in   = 1 / √(σ² + 1)

# 加噪
noisy_latents = latents + ε · σ    (ε ~ N(0, I))

# History 帧加轻微噪声 (0~0.3) 作为条件
σ_h ~ |N(0, 0.3²)|
noisy_history = (1/√(σ_h²+1)) · (history + σ_h · ε_h)
```

### UNet 预测与 Loss

```python
# UNet 输入: noisy future + noisy history + channel-wise condition
input = concat([noisy_history, c_in · noisy_future], dim=time)
input = concat([input, condition_latent], dim=channel)

# UNet 输出 → x₀ 预测
model_pred = UNet(input, c_noise, action_hidden)
predict_x₀ = c_out · model_pred + c_skip · noisy_latents

# 加权 MSE Loss (只对 future frames)
loss_weight = (σ² + 1) / σ²    # 低噪声 → 高权重
loss = mean((predict_x₀[:, history:] - latents[:, history:])² · loss_weight)
```

### 关键设计点
1. **只对 future frames 计算 loss**: history 帧是条件输入，不需要预测
2. **EDM 权重**: `(σ²+1)/σ²`，低噪声级别的样本获得更高的 loss 权重（因为低噪声更接近最终输出，细节更重要）
3. **History 噪声增强**: 给 history latent 加 0~0.3 的噪声，增强鲁棒性
4. **Current frame 条件化**: 当前帧加 0~0.2 噪声后在 channel 维拼接，作为 UNet 的额外条件

### Action 条件注入
- 7D action 序列 → `Action_encoder2` MLP → action latent `(B, T, 1024)`
- 同时融合 CLIP text embedding（任务指令语义）
- 5% 概率 drop action condition (用零向量替代)，为 classifier-free guidance 训练
- 注入 UNet: 作为 `encoder_hidden_states`（通过 cross-attention）

---

## 4. 训练框架与超参

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 基础模型 | SVD UNet ~1.5B + VAE + CLIP | pretrained checkpoint-10000.pt (DROID) |
| 分布式框架 | HuggingFace Accelerate + DeepSpeed ZeRO-2 | Stage 2, fp16, offload=None |
| GPU | 4 × RTX 4090 (24GB) | ~18GB/卡 使用中 |
| per_device_batch | 1 | 单卡 batch size |
| gradient_accumulation | 8 | 有效 batch size = 4×1×8 = 32 |
| 优化器 | AdamW | lr=1e-5, 无 warmup |
| 梯度裁剪 | max_grad_norm | via Accelerate |
| gradient_checkpointing | ✅ | UNet 启用，降低显存 |
| 总步数 | 2000 global steps | ~4 epochs over 4378 samples |
| Checkpoint | 每 500 steps | .pt 文件 ~4.6GB |
| 验证视频 | 每 validation_steps | 解码 latent→RGB→mp4 |
| 混合精度 | fp16 (DeepSpeed) | 训练和推理全 fp16 |
| DeepSpeed 配置 | `ds_zero2.json` | ZeRO stage=2, allgather bucket=5e8 |

### 关键文件
- 训练脚本: `ctrl_world/scripts/train_wm.py`
- 模型定义: `ctrl_world/models/ctrl_world.py`
- 扩散 pipeline: `ctrl_world/models/pipeline_ctrl_world.py`
- 配置: `ctrl_world/config.py` (ManiSkillWMConfig)
- DeepSpeed: `ctrl_world/ds_zero2.json`

### 训练时间估算
- 每 global step: ~80-90 秒 (4×4090, 含 8 micro-steps 梯度累积)
- 2000 steps: ~44-50 小时
- 从 pretrained DROID checkpoint 出发微调

---

## 5. 监控指标

### 训练中可在线监控

| 指标 | 来源 | 频率 | 健康范围 |
|------|------|------|---------|
| `train_loss` | wandb (via Accelerate) | 每 100 global steps* | 应持续下降，无 NaN/Inf |
| 验证视频 | mp4 文件 | 每 validation_steps | 目视检查生成质量 |
| tqdm postfix loss | stdout | 每 100 steps | 同上 |

> *默认每 100 步记录一次，可修改为逐步记录

### 离线评估

| 指标 | 工具 | 说明 |
|------|------|------|
| PSNR | `scripts/verify_ctrl_world.py` | 目标 > 18 dB |
| SSIM | 同上 | 越高越好 |
| LPIPS | 同上 | 越低越好 |

### 需注意的问题
1. WM 训练脚本默认每 100 步才写一次 wandb，早期步骤看不到 loss
2. 验证视频触发条件是 `global_step % validation_steps == 5`（硬编码偏移 5），`validation_steps ≤ 5` 时永远不触发
3. 训练不自动计算 PSNR/SSIM，需要离线用中间 checkpoint 手动评估
