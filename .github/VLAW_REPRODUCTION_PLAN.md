# VLAW 复现计划 — 精简版

> **论文**: VLAW (arXiv:2602.12063) | **完整版**: [`docs/vlaw/archive/reproduction_plan_full.md`](../docs/vlaw/archive/reproduction_plan_full.md)
> **创建**: 2026-02-24 | **精简**: 2026-02-27（原 816 行 → 精简保留核心参考）

---

## 一、核心替换映射

| VLAW 原版 | 本次复现 | 说明 |
|-----------|---------|------|
| π₀.₅ (VLA, Transformer + FM) | ShortCut Flow (1D U-Net, PlainConv) | 保留 flow matching |
| DROID 真机 (Franka, 3 相机) | ManiSkill3 GPU 仿真 (2 相机, 192×192) | 保留 RGB 观测 |
| Ctrl-World (SVD, 320×192) | Ctrl-World (SVD, 384×192, 2-cam 垂直拼接) | 保留完整视频扩散架构 |
| Qwen3-VL-4B 二分类 | Qwen3-VL-4B-Instruct + LoRA + 16帧 images | 保持一致 |
| DROID 95K 轨迹 | ManiSkill 235 条 / 4378 窗口 | 缩放训练步数匹配 |

---

## 二、GPU 分配

```
GPU 0-3: Ctrl-World WM 训练 (DeepSpeed ZeRO-2, 4 GPU)
GPU 4-5: ManiSkill 数据收集 / VAE 编码 / Imagination
GPU 6-7: VLM 奖励模型 (推理/微调, 可扩至 4,5,7,8)
GPU 8-9: ShortCut Flow 策略训练 + 评估
```

| 模型训练 | 方案 | VRAM/卡 |
|---------|------|---------|
| WM 全量微调 1.5B UNet | DeepSpeed ZeRO-2, fp16, grad_ckpt | ~18GB |
| VLM LoRA 微调 | Accelerate DDP, bf16, grad_ckpt | ~12GB |
| Policy FM | 单卡 | ~8GB |

---

## 三、完整迭代算法 (Algorithm 1 精简)

```
for i = 1 to K_iter (2 轮):
  Step 1: Rollout π_θ → D_real (50条/任务)
  Step 2: VAE encode D_real
  Step 3: VLM 标注 D_real → D_real+ (微调后 VLM, α=0.8, 16帧)
  Step 4: 微调 WM on D_real + λ·D_demo
  Step 5: Imagination (π_θ × WM) → D_syn (200-500条)
  Step 6: VLM 标注 D_syn → D_syn+
  Step 7: 策略更新 π_θ on D_real+ ∪ D_syn+ (Weighted FM, 2K steps)
  Step 8: 评估 π_θ (50ep/task)
```

---

## 四、关键风险与缓解

| # | 风险 | 缓解 |
|---|------|------|
| 1 | WM 在 4090 上 OOM | DeepSpeed ZeRO-2 ✅ 已验证 |
| 2 | ManiSkill 视觉风格迁移差 | pretrained WM 质量已验证 (PSNR=22.35) |
| 3 | VLM 判别不准 | 16帧 images AUC=0.82 ✅; LoRA 微调后预期更高 |
| 4 | Imagination action 发散 | action clipping + latent norm 监控 |
| 5 | 合成数据质量不足 | α=0.8 保守筛选 + real≥50% 占比 |

---

## 五、成功标准

| 指标 | 最低要求 | 目标值 | VLAW 论文 |
|------|---------|--------|----------|
| Base → VLAW success_rate 提升 | > 10% abs | > 20% abs | 39.2% abs |
| WM 合成数据贡献 | > 5% abs | > 10% abs | 11.6% abs |
| WM PSNR (action replay) | > 18 | > 20 | 21.77 |
| VLM reward FP rate | < 20% | < 10% | 5% |
| 完整 2 轮迭代 | ✓ | ✓ | ✓ |

> 重点在于验证 **VLAW 框架有效性** — 迭代改进趋势和合成数据正向贡献。

---

## 六、依赖项速查

**Conda 环境**: `ctrl_world` (WM) / `rlft_ms3` (数据+策略) / `vlaw_reward` (VLM)

**关键包**: diffusers 0.34.0, transformers 5.2.0, peft 0.18.1, accelerate, mani-skill3

**模型权重**: SVD (~7GB) + CLIP (~600MB) + Ctrl-World ckpt (~8.7GB) + Qwen3-VL-4B (~8.3GB) + ShortCut Flow (~40MB)
