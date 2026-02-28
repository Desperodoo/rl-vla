# VLAW 基线评估与评测结果汇总

> **最后更新**: 2026-02-28 | **数据**: `results/vlaw/`

本文档整合了所有散落的评测报告，覆盖 WM/VLM/Policy 三个维度的基线与 Iter-1 结果。

---

## 1. 策略基线 (ShortCut Flow)

**评测**: 20 episodes, `results/vlaw/pld_eval_baseline_20ep.json`

| 指标 | 值 |
|------|-----|
| success_once | 95.0% |
| success_at_end | **75.0%** |

> 这是所有后续策略更新的比较基准。

---

## 2. 世界模型 (Ctrl-World)

### 2.1 Pretrained vs Phase-A 基线

**评测脚本**: `scripts/vlaw/eval/eval_wm_horizon.py`  
**报告**: `results/vlaw/wm_baseline_report.md` (原始)  
**可视化**: `results/vlaw/wm_baseline/{pretrained,phase_a_step12000}/`

#### Horizon 分解

| Horizon | pretrained PSNR | phase_a PSNR | pretrained SSIM | phase_a SSIM |
|---------|----------------|--------------|-----------------|--------------|
| 5 帧 | 22.10 | 22.60 | 0.7946 | 0.6373 |
| 10 帧 | 22.50 | 22.37 | 0.7976 | 0.6063 |
| 15 帧 | 22.29 | 21.99 | 0.7917 | 0.5877 |
| 20 帧 | 22.35 | 21.70 | 0.7943 | 0.5754 |

**结论**: Phase-A 在 H5 略好但 H10+ 快速退化 (SSIM -0.22)，决定从 pretrained 开始 Iter-1 微调 (ADR-007)。

### 2.2 Iter-1 微调评估

**评测脚本**: `scripts/vlaw/eval/eval_wm_iter1.py`  
**报告**: `results/vlaw/wm_iter1_eval_report.md` (原始)  
**可视化**: `results/vlaw/wm_iter1_eval/`

| 指标 | pretrained (ckpt-10000) | iter1 (ckpt-2000) | Delta |
|------|------------------------|-------------------|-------|
| **PSNR ↑** | 23.01 ± 5.05 | **23.34 ± 2.72** | **+0.33** |
| SSIM ↑ | 0.8014 ± 0.1182 | 0.7929 ± 0.0770 | -0.0085 |
| **LPIPS ↓** | 0.1297 ± 0.1127 | **0.1190 ± 0.0697** | **-0.0107** |

**门控**: PSNR = 23.34 > 18.0 ✅ PASS

> 详细训练记录见 [wm_iter1_training.md](wm_iter1_training.md)

---

## 3. VLM 奖励模型 (Qwen3-VL-4B)

### 3.1 Zero-shot 单帧基线

**评测**: 160 条轨迹 (success=63, fail=97)  
**报告**: `results/vlaw/vlm_baseline_report.md` (原始)

| 模式 | AUC | p_yes (成功) | p_yes (失败) |
|------|-----|-------------|-------------|
| Zero-shot | 0.585 | 0.083 | 0.068 |
| LoRA (单帧) | 0.617 | 0.009 | 0.007 |

**结论**: 单帧零样本 AUC ≈ 随机，α=0.8 阈值下 TP=0。必须多帧 + LoRA。

### 3.2 多帧评估 (ADR-008 依据)

**评测**: 170 条轨迹 (success=59, fail=111)  
**报告**: `results/vlaw/vlm_multiframe_report.md` (原始)  
**数据**: `results/vlaw/vlm_multiframe_eval.json`

| 配置 | AUC | Recall@FP<20% |
|------|-----|---------------|
| 单帧 images | 0.579 | 18.6% |
| **16帧 images** | **0.815** | **67.8%** |
| 16帧 video | 0.645 | 44.1% |

**结论**: 16帧 images 模式 AUC 最高 (0.815)，video 模式因内部时间降采样丢帧 (ADR-008)。

### 3.3 Iter-1 LoRA 16帧微调

**训练**: 200 steps, 2×GPU, loss 18.7→6.8

| 指标 | 值 |
|------|-----|
| Accuracy | 0.824 |
| FP rate | 3.7% |
| Precision | 0.667 |
| Recall | 0.286 |
| mean p_yes | 0.558 |

**D_real 标注验证 (210 条)**:

| | env_success=True | env_success=False |
|---|---|---|
| vlm=1 | TP=4 | **FP=0** |
| vlm=0 | FN=67 | TN=139 |

FP rate = **0.0%** ✅ (门控 < 20%)

> 详细训练记录见 [vlm_iter1_training.md](vlm_iter1_training.md)  
> 论文对齐分析见 [vlm_finetuning_comparison.md](vlm_finetuning_comparison.md)

---

## 4. 合成数据 (Track B, pretrained WM)

| 指标 | 值 |
|------|-----|
| 生成成功率 | 50/200 (25%) |
| vlm_reward=1 | 0/50 (0%) |
| p_yes mean | 0.058 |

pretrained WM 质量不足，合成数据全部被 VLM 判为失败。符合论文预期——需微调 WM 才能生成有效合成数据。

---

## 5. 指标汇总表

| 维度 | 指标 | 基线 | Iter-1 | 门控 | 状态 |
|------|------|------|--------|------|------|
| WM | PSNR | 23.01 | **23.34** | >18 | ✅ |
| WM | LPIPS | 0.130 | **0.119** | — | ✅ |
| VLM | AUC (zero-shot) | 0.585 | — | — | — |
| VLM | AUC (16帧 images) | 0.815 | — | — | ✅ |
| VLM | FP rate | — | **3.7%** | <20% | ✅ |
| VLM | D_real FP | — | **0.0%** | <20% | ✅ |
| Policy | success_at_end | **75.0%** | ❌ blocked | >75% | ⬜ |
| 合成 | 成功率 (pretrained) | 25% | — | 20-40% | ⚠️ |
