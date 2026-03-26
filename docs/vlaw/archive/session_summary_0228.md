# 02-28 会话进展总结

> **时间**: 2026-02-28 03:00 → 16:00
> **覆盖**: V1.1/V1.2 补充验证 → 正式训练 (WM + VLM) → Track B (Imagination + 标注) → WM 评估 → Policy 尝试

---

## 完成事项

### 1. Phase 1.5b 补充验证 ✅

| 验证项 | 结果 |
|--------|------|
| V1.1 WM 验证视频 | ✅ mp4 有效 (384×384, h264, 2fps)，修复了 3 个 Bug |
| V1.2 WM wandb | ✅ `log_every_n_steps` 可配置 |
| V1.2 VLM wandb | ✅ `--use_wandb` 验证通过 |

### 2. WM Iter-1 正式训练 ✅

- 2000 global steps / 2 小时 / GPU 0-3 (DeepSpeed ZeRO-2)
- PSNR: 23.01 (pretrained) → **23.34** (iter1), Δ=+0.33
- LPIPS: 0.1297 → **0.1190**, Δ=-0.01
- 门控 PSNR > 18.0: **PASS**
- 详见 [docs/vlaw/wm_iter1_training.md](docs/vlaw/wm_iter1_training.md)

### 3. VLM 16 帧训练 ✅

- 200 steps / ~50 min / GPU 6-7
- Loss: 18.7 → 6.8, Accuracy: 0.824, FP: 3.7%
- D_real 210 条标注: FP=0%, Precision=100%
- 详见 [docs/vlaw/vlm_iter1_training.md](docs/vlaw/vlm_iter1_training.md)

### 4. Track B: Imagination + 标注 ⚠️

- B1: 50/200 轨迹成功 (190 failed — cuda:0 device error)
- B2: 50 条合成标注, vlm_reward 全=0 (pretrained WM 质量不足, 符合预期)
- D_real 标注 (fine-tuned VLM): 210 条, 4 vlm_reward=1, FP=0%

### 5. Bug 修复 (3 个新 Bug)

| Bug ID | 问题 | 修复文件 |
|--------|------|---------|
| BUG-013 | config.py SVD/CLIP 路径缺 `../` | `ctrl_world/config.py` L189-190 |
| BUG-014 | tmux 中 ffmpeg 不在 PATH | tmux 命令 + train_wm.py L30-34 |
| BUG-015 | 训练循环无 early-stop break | `ctrl_world/scripts/train_wm.py` L194-197 |

---

## 阻塞项

### Policy 架构不匹配 ❌

- ShortCut Flow base checkpoint 使用 **视觉编码器** (PlainConv, global_cond_dim=626)
- VLAWPolicyUpdater 使用 **raw state** (global_cond_dim=50)
- 无法直接加载 base checkpoint 的权重到 state-observation 策略
- **需要**: 适配 VLAWPolicyUpdater 使用视觉 observations，或从 scratch 训练 state-only policy

---

## 新增文档

| 文件 | 内容 |
|------|------|
| [docs/vlaw/wm_iter1_training.md](docs/vlaw/wm_iter1_training.md) | WM Iter-1 完整训练记录 |
| [docs/vlaw/vlm_iter1_training.md](docs/vlaw/vlm_iter1_training.md) | VLM 16 帧训练+标注记录 |
| `.github/knowledge/bugs-and-fixes.md` | 新增 BUG-013/014/015 |
| `.github/vlaw-status.md` | 更新至当前状态 |
| `.github/VLAW_NEXT_STEPS.md` | 标记已完成任务 |

---

## 新增 Checkpoints

| 模型 | 路径 | 大小 |
|------|------|------|
| WM iter1 (4 个) | `checkpoints/vlaw/world_model/iter1/checkpoint-{500,1000,1500,2000}.pt` | 4.4GB each |
| VLM 16 帧 LoRA | `checkpoints/vlaw/reward_model/lora_iter1_16frame/` | 23MB adapter |

---

## 新增数据

| 数据 | 路径 | 条数 |
|------|------|------|
| 合成轨迹 (B1) | `data/vlaw/synthetic/iter1/` | 50 条 |
| D_real VLM 标注 | `data/vlaw/labeled/iter1_16frame_lora/` | 210 条 |
| B2 合成标注 | `data/vlaw/labeled/synthetic_iter1_pretrained/` | 50 条 |

---

## 下一步

1. **Policy 架构适配**: 修改 VLAWPolicyUpdater 支持视觉 observations (global_cond_dim=626)
2. **Track C**: 使用微调 WM (iter1) 生成高质量合成数据 → VLM 标注 → 策略训练
3. **Imagination 修复**: 解决 cuda:0 device error，提高合成轨迹成功率
4. **Iter-2**: WM/VLM 第二轮微调
