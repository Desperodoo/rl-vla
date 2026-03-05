# World Model 深度评估分析

> **日期**: 2026-03-05  
> **数据来源**: `results/vlaw/wm_deep_viz/summary.json`  
> **评估方式**: 单步预测 (GT history 6帧 → predict 5帧 → 与 GT 对比)  
> **评估脚本**: `rlft/vlaw/scripts/eval_wm.py --visualize`  
> **评估集**: `data/vlaw/encoded/eval/` (13 条轨迹, 65 帧)

---

## 1. 整体指标对比

| Model | PSNR (all frames) | SSIM | LPIPS | PSNR (F1-F4 only) | SSIM (F1-F4) |
|-------|--------------------|------|-------|--------------------|--------------|
| pretrained | 22.99 ± 3.81 | 0.7633 ± 0.10 | 0.1917 | **19.29 ± 6.09** | 0.7135 |
| ckpt-400   | 29.36 ± 3.41 | 0.9350 ± 0.03 | 0.0418 | **26.68 ± 4.82** | 0.9226 |
| **Δ (ckpt-400 vs pretrained)** | **+6.37 dB** | **+0.172** | **-0.150** | **+7.39 dB** | **+0.209** |

> **门控结果**: ckpt-400 PSNR=29.36 远超门控阈值 (>18 dB) ✅ 和 VLAW 论文值 (21.77 dB)

---

## 2. ⚠️ 关键发现: Frame 0 严重拉高均值

**Frame 0 是 "当前帧" (current frame)**, 它被直接提供给模型作为 conditioning，因此 PSNR 异常高 (~38-40 dB)，不代表真实预测能力。

### Per-Frame PSNR 分解

| Frame | pretrained PSNR | ckpt-400 PSNR | Δ |
|-------|-----------------|---------------|---|
| **F0 (current)** | 37.80 ± 0.65 | 40.10 ± 1.13 | +2.30 |
| F1 | 22.85 ± 6.92 | 28.97 ± 5.51 | +6.12 |
| F2 | 20.61 ± 4.66 | 26.39 ± 4.41 | +5.78 |
| F3 | 16.17 ± 4.55 | 25.90 ± 4.39 | +9.73 |
| F4 | 17.51 ± 5.54 | 25.44 ± 4.05 | +7.93 |

### Per-Frame SSIM 分解

| Frame | pretrained SSIM | ckpt-400 SSIM | Δ |
|-------|-----------------|---------------|---|
| **F0 (current)** | 0.9625 | 0.9850 | +0.023 |
| F1 | 0.7880 | 0.9365 | +0.149 |
| F2 | 0.7460 | 0.9221 | +0.176 |
| F3 | 0.6308 | 0.9164 | +0.286 |
| F4 | 0.6892 | 0.9153 | +0.226 |

### 结论

- **排除 F0 后**: pretrained 实际 PSNR ≈ 19.3 dB, ckpt-400 ≈ 26.7 dB
- **ckpt-400 的真正优势体现在 F3-F4**: pretrained 在 F3-F4 快速衰减到 ~16-17 dB, 而 ckpt-400 保持在 ~25+ dB
- 微调使模型学会了对 ManiSkill 场景的长程一致性预测

---

## 3. 弱轨迹分析 (traj_0017, 0018, 0019)

这三条轨迹较长 (T=51), pretrained 模型在上面表现极差:

| Traj | pretrained F1-F4 PSNR | ckpt-400 F1-F4 PSNR | Δ |
|------|----------------------|---------------------|---|
| traj_0017 | ~10.7-12.7 dB | ~21.7-22.8 dB | +10+ |
| traj_0018 | ~10.2-20.5 dB | ~20.8-22.7 dB | +10+ |
| traj_0019 | ~10.5-14.1 dB | ~22.3-23.2 dB | +10+ |
| **avg F1-F4** | **12.64 dB** | **22.12 dB** | **+9.48** |

> 弱轨迹上 pretrained 接近随机噪声水平 (~10 dB), ckpt-400 仍保持可用质量 (~22 dB)

---

## 4. 单步 PSNR ≠ 自回归 Imagination 质量

### 两种评估方式的本质区别

| | eval_wm (单步) | Imagination (自回归) |
|---|---|---|
| **输入** | GT history 6帧 | 仅第1帧为真实 VAE latent |
| **预测** | 5帧 | 12轮 × 5帧 = 60帧 |
| **误差传播** | 无 | 每轮 pred 喂入下轮 history |
| **条件质量** | 完美 (GT latent) | 逐轮退化 (pred latent) |
| **PSNR 期望** | ~29 dB (ckpt-400) | 远低于 29 (误差累积) |

### 质量衰减机制

**History buffer 结构** (对齐官方 `history_idx=[0,0,-12,-9,-6,-3]`):
- Pos[0,1]: **始终**指向 `latent_history[0]` = 真实初始帧 (永久锚定)
- Pos[2-5]: 负偏移采样, 初始为真实帧, 逐渐被预测帧替代
- Current frame: 从 round 1 起始终为 predicted (最重要的条件信号)

**逐轮条件帧 real/pred 构成** (6 history + 1 current = 7 条件帧):
- Round 0: 7/7 real (buffer 初始化为 24 份真实帧)
- Round 1-2: 6/7 real (仅 current 变为 predicted)
- Round 3: 5/7 real (Pos[5] 开始变为 pred₀)
- Round 6: 4/7 real
- Round 9: 3/7 real
- Round 11: **2/7 real** (仅 Pos[0,1] 保持真实)

**误差来源**:
1. **Current frame 从 round 1 起就是 predicted** — 这是扩散 conditioning 最重要的输入
2. 4/6 history slots 在后半段全为 predicted, 各轮预测误差通过 history 传播
3. 锚定帧 (Pos[0,1]) 来自 T=0, 对 T≈55 时刻的预测指导价值有限
4. **Viz strip**: 取第1帧/中间帧(~第30帧)/最后帧(~第60帧), 后两者质量自然较差

### 补充因素

- **VAE dtype**: Imagination viz 使用 `float16` VAE decode (`run_imagination.py` L387), eval_wm 使用 `float32` (CrtlWorld 默认). float16 精度损失对弱信号帧影响更大, 但非主因
- **帧裁剪**: Imagination viz 截取 `[:, :192, :, :]` (上半部分 = base_camera view), 合理做法 — WM 输出 384×192 (双相机拼接), 上192行=base_camera, 下192行=hand_camera
- **推理步数**: Imagination 使用 25 步 (效率优先), eval_wm 使用 50 步, 但差异较小

### ⛔ 重要修正 (ADR-034, 2026-03-05)

> **以下旧建议已被 ADR-034 推翻**。经人工视觉审核，Imagination 生成几乎完全不可用，
> 自动化指标 (latent Δ<0.02, L2 drift<1%) 与肉眼观感严重脱节。

1. **eval_WM PSNR 仅供参考, 不作为 WM 质量门控** — 单步 GT history 条件下的 PSNR=29 不能反映自回归 rollout 的实际质量
2. **WM 质量必须通过 Imagination 可视化 + 人工审核确认** — 不能仅依赖 eval_WM 定量指标
3. **Imagination 质量差不再视为"预期行为"** — 需要通过 WM 继续训练来改善
4. **所有下游环节 (策略更新、评估) 阻塞**, 直到 Imagination 人工确认可用

---

## 5. 输出文件索引

| 路径 | 说明 |
|------|------|
| `results/vlaw/wm_deep_viz/summary.json` | 完整 per-traj per-frame 数据 (JSON) |
| `results/vlaw/wm_deep_viz/pretrained/` | pretrained 模型可视化 |
| `results/vlaw/wm_deep_viz/pretrained/all_trajs_grid.png` | 所有轨迹 GT vs Pred 对比大图 |
| `results/vlaw/wm_deep_viz/pretrained/traj_XXXX_compare.png` | 每条轨迹对比图 |
| `results/vlaw/wm_deep_viz/pretrained/traj_XXXX_frames/` | 单帧 GT/Pred/Diff 文件 |
| `results/vlaw/wm_deep_viz/ckpt-400/` | ckpt-400 模型可视化 (同上结构) |
| `results/vlaw/wm_eval/` | eval_wm.py 标准输出 (条形图 + JSON) |
