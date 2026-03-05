# VLAW 下一步推进计划

> **最后更新**: 2026-03-05 11:40 | **历史版本**: `docs/vlaw/archive/VLAW_NEXT_STEPS_*.md`
> **状态面板**: [`vlaw-status.md`](vlaw-status.md) | **评测汇总**: [`docs/vlaw/baselines_and_evaluation.md`](../docs/vlaw/baselines_and_evaluation.md)

---

## 已完成阶段 (02-28 ~ 03-05)

- ✅ Phase 1.5: V1-V6 全链路验证 + 转换后重新验证
- ✅ Track A1: WM iter1 2000 steps (PSNR=23.40, SSIM=0.79, LPIPS=0.12)
- ✅ Track A2: VLM 16帧 LoRA 200 steps (AUC=0.808, Acc=84.7%, FP@0.8=0%)
- ✅ Track B1-B2: Imagination 50条 + VLM 标注
- ✅ ctrl_world submodule → 普通目录
- ✅ Pretrained Policy 验证: success_once=74-88% (avg~80%), 修复零动作 padding bug
- ✅ T-POLICY-FIX (ADR-009): UNet参数修正(64,128,256)/64 + 实数据10步验证通过
- ✅ Imagination 3 bug 修复: PlainConv参数 + get_actions API + obs格式, mock 5/5 验证
- ✅ T-EXP-WM-01: WM 4000步消融 — PSNR=24.11 (2k/4k持平), 结论: 2000步已最佳
- ✅ T-EXP-VLM-01-4frame: VLM 4帧消融 — acc=0.706, FP=11.1% (vs 16帧 0.824/3.7%)
- ✅ **T-EXP-VLM-01-8frame: VLM 8帧消融 — acc=0.735, FP=18.5%, ROC-AUC=0.8269**
- ✅ **T-IMAGINATION-001: 200/200 合成轨迹 (iter1 WM + real policy), 0失败, 582min**
- ✅ **T-VLM-LABEL-001: 200条 VLM 标注, vlm=1: 0/200 (p_yes max=0.27 << α=0.8)**
- ✅ **T-POLICY-001: Weighted FM 2000步, loss 1.6→0.279, ckpt: policy/iter1/**
- ✅ **T-POLICY-FIX-EMA: EMA ckpt 保存格式 bug 修复, 重新保存 iter1 ckpt**
- ✅ **T-DIAG-SYN-001~003: D_syn+=0 诊断三步全部完成** (合成帧解码+真实帧对照+VLM交叉验证)
- ✅ **T-EXP-VLM-04: VLM use_video_format A/B 对比** — video AUC=0.83 >> multi-image 0.72, Youden 最优 α=0.40 (TP=70.6%, FP=1.4%), 确认 WM 合成质量为主要瓶颈
- ⚠️ **T-EVAL-ITER1-001: Iter-1 严重退化 (baseline 78.1% → iter-1 17.2%), 需调整策略**
- ✅ **BUG-019 发现并修复**: `load_initial_frames()` 使用 `torch.randn` 随机噪声作为初始 latent → 修复为从 `data/vlaw/encoded/` 加载真实帧 VAE 编码 `latent_concat[0]`。**3/3 验证通过, 旧 200 条合成数据无效需重生成** (T-IMAGINATION-002)
- ✅ **WM 自回归 Rollout 模糊调研** — 详见 [`docs/vlaw/wm_autoregressive_blurring_research.md`](../docs/vlaw/wm_autoregressive_blurring_research.md)
- ✅ **T-EXP-WM-05: WM 最优步数搜索** — 2000步, loss=0.0268, ckpt@500/1000/1500/2000, tmux `wm_optimal`
- ✅ **T-EXP-WM-05-EVAL: WM 最优步数评估 (ADR-018)** — 4个 ckpt 评估: 1000步最优性价比 (PSNR=25.80, SSIM=0.891, LPIPS=0.056), 2000步 PSNR=25.87 仅微弱优势。后续 WM 训练可缩短至 1000 步 (节省 50% 时间)
- ✅ **T-VLM-LABEL-002b: VLM 标注 002b sliding window 200条** — p_yes max=0.531 (旧噪声数据 0.27 →2倍提升), D_syn+(α=0.4)=7, D_syn+(α=0.8)=0。首次产出 D_syn+ > 0！
- ✅ **T-EXP-VLM-02 r=8 完成** — acc=0.794, FP=0%, ROC-AUC 待确认, ckpt: `ablation_lora_r8/`
- ✅ **T-BC-SCALING-V2 全部完成** — 6/6组 20K步从零训练: 25d=0.02, 50d=0.04, 100d=0.10, 200d=0.16, 400d=0.32, 669d=0.40 (success_once). Go/No-Go: 669d=40%>30% ✅. 结果: `results/vlaw/bc_scaling_v2/scaling_results.jsonl`
- ✅ **T-MBRL-ENV 完成** — ImaginationRLEnv 实现 + 40/40 pytest 通过 (`rlft/vlaw/world_model/imagination_rl_env.py`, 37KB). WM+VLM 包装为 Gym 环境接口已就绪
- 🔄 **T-EXP-VLM-03 运行中** — VLM 步数消融, GPU 6, tmux `vlm_steps_ablation`, 当前: 400步训练中 (100步✅)
- ✅ **T-WM-ALIGN-ABLATION 完成** — aligned(6/200) ≈ sliding(7/200) D_syn+, 差异不显著 (Δ=1), 未达 go 标准 (需 ≥20%). ADR-021: 保留当前做法, WM 质量是主瓶颈
- ✅ **D_syn+ 数据准备完成** — 002a(6) + 002b(7) = 13条 D_syn+ (α=0.4) 合并入 demo 数据, 解码帧检查通过
- ✅ **T-BC-FLYWHEEL-A 完成** — 100K步 BC 基线: 100d=0.10, 669d=0.54 (success_once). 新代码: `scripts/vlaw/run_bc_flywheel_a_single.sh`
- 🔄 **T-BC-FLYWHEEL-B 已启动** — 113t(GPU0)+682t(GPU9) 100K步, demo+D_syn+ 合并训练, tmux `bc_flywheel_b`, 启动于 00:34. 新代码: `scripts/vlaw/prepare_dsyn_plus_combined.py`, `scripts/vlaw/run_bc_flywheel_b_single.sh`
- ✅ **T-BC-FLYWHEEL-B 完成** — 113t(100d+13syn)=**0.34** (+240% vs A=0.10), 682t(669d+13syn)=**0.48** (-11% vs A=0.54). 小数据 D_syn+ 价值巨大, 大数据被稀释. 提升 D_syn+ 产出率是最关键杠杆
- ✅ **T-EXP-WM-05v2 训练完成** — 20 个 ckpt (100-2000步), num_frames=5, demos. ⏸️ 评估阻塞 (ADR-024)
- ✅ **T-EXP-VLM-02 全部完成** — r=8✅(acc=0.794,FP=0%) / r=32✅(acc=0.85,FP=0%) / r=64✅(acc=0.85,FP=0%). r=32/r=64 缺 ROC-AUC 评估
- ✅ **T-EXP-VLM-03 全部完成** — 100/400/800步. **⚠️ 严重异常**: AUC=0.31/0.38/0.39, p_yes≈零. 极可能是脚本 bug → T-EXP-VLM-03-BUG
- ~~⏸️ **ADR-024: WM 相关任务全部阻塞**~~ → **已解除 (ADR-025 No-Go + ADR-026 Fresh Start)**
- ✅ **帧率与时间尺度分析 (ADR-023)** — 详见 [`docs/vlaw/frame_rate_timing_analysis.md`](../docs/vlaw/frame_rate_timing_analysis.md)

#### Fresh Start 阶段 (03-04 ~ 03-05)
- ✅ **ADR-026 / BUG-020**: 发现 demo_prep.py rgb_render=rgb_base 全链路污染 → 全部数据/ckpt 归档, Fresh Start
- ✅ **ADR-027 / BUG-024**: selection bias 诊断+解决 (方案 A 大量采集 1200+ 条)
- ✅ **旧数据清理**: v1 污染数据→`_archive/v1/`, v2 frame_skip=3→`_archive/v2_frameskip3/`, AWSC 错误数据→删除, pilot→清理
- ✅ **v3 大规模采集完成 (Phase 0 数据就绪)**:
  - mixed: 1200 条, success_at_end=46.0%, frame_skip=4 (5Hz), T_mean=34.3
  - eval: 20 条, success_at_end=85.0%
  - high_suc: 552 条 (从 mixed 筛选 success_at_end=True), T_mean=15.3
- ✅ **质量报告 GO**: 11/11 检查全通过 (`results/vlaw/data_quality_report_v3.md`)
  - frame_skip=4 ✅, EMA 权重 ✅, T_max=51 ✅, 0 幽灵 ✅, 0 空轨迹 ✅
  - actions ∈ [-1,1] ✅, action std > 0.01 ✅, rgb_base≠rgb_render diff=56.5 ✅
  - success_at_end=46% (无 selection bias) ✅, action_dim=7 ✅
  - 发现 WM 帧率不匹配 (6.67Hz vs DROID 5Hz) + Imagination 严重过长 (9s vs 任务 2.4s, ~73% 无效帧)
  - 推荐: `num_interact` 从 12 缩短至 4, 预期 D_syn+ 从 3.5% 显著提升, 生成速度 ~3×
- ✅ **T-VAE-ENCODE-V3 完成**: VAE 编码 mixed 1.1GB + high_suc 272MB + eval 13MB, latent (T,4,48,24) fp16, top-bot-diff=0.90
- ✅ **stat.json v3 重新生成**: state_01/state_99 percentile 格式, 49618 样本
- ✅ **T-VLM-V3 完成**: LoRA v3 (r=16, 200 steps, 16帧 video, accum=16), FP=0%, Precision=1.0, Recall=42.4%, Accuracy=72.8%. 比 v2 更保守 (零误报但 recall 较低)
- 🔄 **T-WM-V3 训练中**: GPU 0-3, DeepSpeed ZeRO-2, 2000 steps, LR=1e-5, 已运行 ~10h, checkpoint-200.pt 已保存 (07:08)
- ✅ **T-VLM-ABLATION-V3 全部完成**: VLM v3 消融实验 (Steps + Threshold + LoRA rank)
  - Steps 消融: 100步=0% recall, 200步=42.4%, **300步=61.2%** (最佳), 400步=71.8% (+2.1% FP)
  - Threshold 消融 (300步 ckpt): **α=0.5 推荐** (acc=86.7%, recall=85.9%, FP=12.6%)
  - LoRA r=8: recall=1.2% — 容量不足, 不可用
  - **推荐配置: r=16, 300 steps, α=0.5~0.8**
  - 消融报告: `results/vlaw/vlm_ablation_v3_report.md`

---

## ~~🔧 T-TIMING-FIX~~ — 已关闭 (ADR-025 No-Go + ADR-026 Fresh Start)

> **ADR-025**: 方案 A (`num_interact=4`) No-Go — D_syn+=0/50, 20 帧太短
> **ADR-026**: Fresh Start 采用 frame_skip=4 (方案 B), 所有旧数据推倒重来
> **当前状态**: TIMING 问题已通过 Fresh Start + frame_skip=4 彻底解决, 此 section 归档

<details><summary>原计划 (点击展开)</summary>

### 问题总结

| 维度 | DROID (原版) | 当前 ManiSkill | 偏差 |
|------|------------|---------------|------|
| WM 帧率 | 5 Hz (帧间隔 0.2s) | 6.67 Hz (帧间隔 0.15s) | +33% |
| 1 interaction 时长 | 1.0 秒 | 0.75 秒 | -25% |
| Imagination 总时长 | 12-20s ≈ 任务时长 | 9s >> 任务 2.4s | 3.75× 过长 |
| 无效帧比例 | 低 (1:1 匹配) | ~73% | 严重 |

### WM Finetuning 与 TIMING 的关系 (ADR-024 分析)

| 方案 | 对 WM 训练的影响 | 对数据的影响 | 现有 ckpt 是否可复用 |
|------|------------------|-------------|-------------------|
| **方案 A** (num_interact=4) | **零影响** — 仅控制推理时自回归循环次数 | **零影响** | ✅ 全部可复用 |
| **方案 B** (frame_skip=4) | **需推倒重来** — 帧率从 6.67Hz→5Hz, 重新采集+VAE编码+训练+stat.json | **全部重新处理** | ❌ 作废, 但 pretrained 权重更匹配 5Hz |

### 执行计划 (方案 A → 方案 B 顺序推进)

| task_id | 任务 | 方案 | 优先级 | 依赖 | 状态 |
|---------|------|------|--------|------|------|
| **T-TIMING-QUICKTEST** | `num_interact=4` 生成 50 条 → VLM 标注 → 对比 D_syn+ 通过率 | A | **P0 Gating** | GPU 空闲 | ⬜ 待执行 |
| T-TIMING-A-FULL | 方案 A 验证成功后: `num_interact=4` 生成 200 条 → 全流程 | A | P0 | T-TIMING-QUICKTEST Go | ⬜ |
| **T-TIMING-FRAMESKIP** | 方案 B: `frame_skip=4` 对齐 5Hz + 重新采集+编码+训练 WM | B | P1 | A 验证有效后 | ⬜ |
| T-TIMING-FULL-REDESIGN | 方案 C: 全面适配 num_frames/num_interact/num_history | C | P3 | B 完成后 | ⬜ |

### Go/No-Go

- **方案 A 验证 (T-TIMING-QUICKTEST)**: D_syn+ (α=0.4) 通过率从 3.5% (7/200) 提升到 **≥10%** → Go to 方案 A 全量 + 方案 B
- **方案 A 失败**: 通过率无提升 → TIMING 非主因, 从其他方向排查
- **方案 B 进入条件**: 方案 A 确认有效, 且希望进一步提升精度 → 重新采集所有数据 + 重新训练 WM

</details>

---

## ~~🔧 T-WM-ALIGN-HISTORY~~ — v1 归档 (ADR-021: sliding window 保留)

<details><summary>v1 History Buffer 消融 (点击展开)</summary>

> **调研结论**: 我们的 imagination 滑动窗口实现与 Ctrl-World 官方代码存在 3 处关键差异，可能加速图像模糊退化。
> **详细调研报告**: [`docs/vlaw/wm_autoregressive_blurring_research.md`](../docs/vlaw/wm_autoregressive_blurring_research.md)

### 必须对齐的改动

| # | 改动 | 当前 | 目标 (对齐官方) | 涉及文件 |
|---|------|------|----------------|----------|
| 1 | **History 构建方式** | 滑动窗口 `lat_buf[-window_len:]` | 列表式 buffer + 稀疏采样 `history_idx=[0,0,-12,-9,-6,-3]` | `imagination.py`, `imagination_env.py` |
| 2 | **第一帧锚定** | 无 — 初始真实帧被挤出 buffer | **始终保留** — `history_idx[:2]` 永远指向真实初始帧 latent | `imagination.py`, `imagination_env.py` |
| 3 | **num_history** | 4 | **6** (与官方 DROID 配置一致) | `config.py (wm_args_maniskill)`, `imagination.py`, `imagination_env.py` |

### 保持不变的设计

| 项目 | 当前值 | 说明 |
|------|--------|------|
| 总 rollout 长度 | `num_interact=12, pred_step=5` → 60 帧 | 与官方一致，不下降 |
| VLM 自然过滤 | α 阈值 + VLM 质量筛选 | 模糊轨迹被低分自动过滤 |

### 执行计划

| task_id | 任务 | owner | 依赖 |
|---------|------|-------|------|
| **T-WM-ALIGN-HISTORY** | 对齐 history buffer 构建 (list + 稀疏采样 + 第一帧锚定 + num_history=6) | Imagination | BUG-019 fix ✅ |
| **T-IMAGINATION-002a** | 用 **对齐后** 代码生成 200 条合成轨迹 (官方做法) | Imagination | T-WM-ALIGN-HISTORY ✅ |
| **T-IMAGINATION-002b** | 用 **当前滑动窗口** 代码生成 200 条合成轨迹 (对照组) | Imagination | BUG-019 fix ✅ |
| **T-WM-ALIGN-ABLATION** | 消融对比: 对齐 vs 当前做法 (帧质量 PSNR/SSIM + VLM p_yes 分布 + D_syn+ 数量) | Eval | T-IMAGINATION-002a ✅ + T-IMAGINATION-002b ✅ |

> **注意**: T-IMAGINATION-002a 需等 T-WM-ALIGN-HISTORY 完成后再执行；T-IMAGINATION-002b 仅需 BUG-019 修复（使用真实初始 latent），无需等待对齐改动。两组实验可并行在不同 GPU 上执行。

### 消融设计

**目标**: 量化对齐官方 history 构建方式带来的合成质量增益，避免盲改。

| 维度 | 对齐组 (002a) | 对照组 (002b) |
|------|--------------|--------------|
| History 构建 | 列表式 + `[0,0,-12,-9,-6,-3]` | 滑动窗口 `lat_buf[-window_len:]` |
| 第一帧锚定 | ✅ 始终保留 | ❌ 挤出 buffer |
| num_history | 6 | 4 |
| 其他 | 一致 (同 WM ckpt, 同 policy, 同初始 latent) | 一致 |

**评估指标** (逐帧对比):
1. **帧质量退化曲线**: 每 5 帧 (每轮 interact) 解码为 RGB → 计算 PSNR/SSIM vs 真实渲染帧 (如有) 或 vs 第一帧
2. **VLM p_yes 分布**: 对两组各 200 条轨迹做 VLM 标注，比较 p_yes 均值/中位数/D_syn+ 数量
3. **Latent norm 轨迹**: 记录每帧 latent 的 L2 norm，观察是否有发散趋势

**Go/No-Go**: 若对齐组在 D_syn+ 数量或帧质量上有 **显著** 提升 (D_syn+ 多 ≥20% 或末帧 PSNR 高 ≥1dB)，则后续统一使用对齐版本；否则保留当前做法（更简洁）。

</details>

---

## 🚀 Fresh Start 主线: Phase 0 → Phase 1 → Phase 2 → Imagination → Iter-2

> **当前状态**: Phase 0 ✅完成 | Phase 1 WM 🔄训练中 (GPU 0-3, step~200/2000) | Phase 2 VLM ✅完成+消融完成
> **总体路线**: ~~数据准备~~ → ~~VAE 编码 + stat.json~~ → **WM 微调** → ~~VLM 微调~~ → Imagination → 标注 → 策略训练 → 评估

### 待推进任务 (Fresh Start)

| 优先级 | task_id | 任务 | owner | GPU | 依赖 | 状态 |
|--------|---------|------|-------|-----|------|------|
| ~~P0~~ | ~~T-VAE-ENCODE-V3~~ | v3 rollouts VAE 编码 + stat.json 重新生成 | Data | 4-5 | ✅ | ✅ 完成 (mixed 1.1GB + high_suc 272MB + eval 13MB) |
| **P0** | **T-WM-V3** | WM 微调 (pretrained → v3 数据, 2000 步) | WM | 0-3 | T-VAE-ENCODE-V3 ✅ | 🔄 训练中 (step~200/2000, 已运行 ~10h, ckpt-200 已保存) |
| ~~P0~~ | ~~T-VLM-V3~~ | VLM LoRA 微调 (v3 rollout 数据, r=16, 200 步) | Reward | 6-7 | ✅ | ✅ 完成 (FP=0%, recall=42.4%, acc=72.8%) |
| **P1** | **T-IMAGINATION-V3** | Imagination (v3 WM + AWSC policy, 200+ 条) | Imagination | 4-5 | T-WM-V3 ✅ | ⬜ 等待 WM 训练完成 |
| P1 | **T-VLM-LABEL-V3** | VLM 标注合成轨迹 | Reward | 6-7 | T-IMAGINATION-V3 ✅ + T-VLM-V3 ✅ | ⬜ |
| P2 | **T-POLICY-V3** | Weighted FM 策略训练 (D_real + D_syn+) | Policy | 8-9 | T-VLM-LABEL-V3 ✅ | ⬜ |
| P2 | **T-EVAL-V3** | ManiSkill 评估 (对比 baseline 78%) | Eval | 8-9 | T-POLICY-V3 ✅ | ⬜ |
| P3 | T-ITER-ROUND-2 | 第 2 轮迭代 (如 Iter-1 有效) | 全部 | — | T-EVAL-V3 ✅ | ⬜ |

### 关键参数 (Fresh Start, 基于 v1 结论)
- **数据**: frame_skip=4 (5Hz), max_episode_steps=200, LiftPegUpright-v1
- **WM**: pretrained → finetuning 2000 步, num_frames=5, num_history=6, lr=1e-5, DeepSpeed ZeRO-2, grad_accum=8
- **VLM**: LoRA r=16, **300 步** (最佳甘甜点, v3 消融确认), 16 帧, video 模式, accum=16
- **VLM v3 消融结论**: r=8不可用 (1.2% recall), 300步 > 200步 > 400步; α=0.5 最佳平衡 (86.7% acc, 85.9% recall, 12.6% FP); α=0.8 駶守 (零 FP, 61.2% recall)
- **Imagination**: num_interact=12, BUG-019 已修复
- **评估**: rgb_base vs rgb_render diff > 30; success_at_end 比率非 100%/0%

---

## 🔧 WM 等待期并行任务 (B·D·C 三线)

> **背景**: WM v3 训练预计还需 ~30h (GPU 0-3 占满)，GPU 4-9 全部空闲，应充分利用进行管线预验证和代码整固。
> **核心原则**: **先固化稳定版脚本 (T-SCRIPTS-CONSOLIDATE)，再用固化脚本执行 BDC 三线任务。**

### ⚠️ T-SCRIPTS-CONSOLIDATE: Pipeline 脚本固化与冗余清理 (最高前置优先级)

> **问题**: 过去调试反复创建一次性脚本，同功能存在 2-5 个版本，未固化出稳定入口。每次新任务又从头写，增加 bug 风险、浪费时间。
> **目标**: 审查所有 v1/v2 调试脚本，提取已验证的逻辑（含 bug fix），合并为每个 pipeline 阶段的**唯一稳定入口脚本**，冗余版本归档。

#### 需要固化的 6 条管线

| 管线 | 稳定版目标路径 | 当前冗余版本 (需审查/归档) | 关键 bug fix 需合入 |
|------|---------------|--------------------------|-------------------|
| **Imagination** | `rlft/vlaw/scripts/run_imagination.py` | `scripts/vlaw/run/run_imagination_iter1.py`, `run_b1_imagination_200.py`, `run/launch_imagination_003.sh`, `archive/imagination.py`, `archive/imagination_env.py`, `scripts/archive/test_v3_imagination_mini.py` | BUG-019 (真实首帧 latent), BUG-017 (PlainConv+API+obs格式), ADR-021 (sliding window) |
| **WM 评估** | `rlft/vlaw/scripts/eval_wm.py` (新建统一版) | `scripts/vlaw/eval/eval_wm_iter1.py`, `eval_wm_ablation_4000.py`, `eval_wm_horizon.py`, `eval_wm_optimal_steps.py`, `eval_wm_optimal_steps_v2.py`, `eval_wm_standard.py`, `rlft/vlaw/scripts/eval_wm_comparison.py`, `eval_wm_three_models.py` | PSNR/SSIM/LPIPS 标准评估流程 |
| **VLM 标注** | `rlft/vlaw/scripts/label_trajectories.py` (统一 real+synthetic) | `scripts/vlaw/data/label_dreal_vlm.py`, `scripts/vlaw/eval/label_synthetic_trajectories.py`, `label_timing_test.py`, `rlft/vlaw/scripts/label_real_trajectories.py` | BUG-011 (yes/No 大小写), ADR-019 (video 模式) |
| **VLM 评估** | `rlft/vlaw/reward/eval_threshold_ablation.py` (已存在) | `scripts/vlaw/eval/eval_vlm_16frame.py`, `eval_vlm_ablation.py`, `eval_vlm_multiframe.py`, `eval_vlm_v2.py`, `scripts/vlaw/eval_reward_model_v3.py`, `scripts/archive/eval_vlm_baseline.py`, `eval/vlm_crossval_real.py` | ADR-028 (threshold sweep), ADR-029 (eval 集平衡) |
| **Policy 评估** | `rlft/vlaw/scripts/evaluate_policy.py` (已存在) | `scripts/vlaw/eval/eval_pretrained_policy.py` | BUG-016 (零动作 padding) |
| **Policy 训练** | `rlft/vlaw/scripts/run_policy_update.py` (已存在) | `scripts/vlaw/run/run_policy_iter1.py`, `run_b3_policy_train.py` | BUG-018 (EMA ckpt 格式), ADR-012 (灾难性遗忘防范) |

#### 执行方案

1. **审查**: 逐管线对比所有版本，确认哪些 bug fix 已合入 `rlft/vlaw/scripts/` 稳定版，哪些还散落在一次性脚本中
2. **合并**: 将遗漏的 fix 合入稳定版，补充 CLI 参数使其灵活适配不同场景 (iter1/v3/消融)
3. **验证**: 每个稳定版做 dry-run 验证 (无 GPU 训练，只测代码路径)
4. **归档**: 冗余脚本移至 `scripts/vlaw/_archive/` (保留 git 历史，不删除)
5. **记录**: 更新 `knowledge/interfaces.md` 添加"脚本入口汇总"章节

#### 预估工作量

- owner: **Eval-Agent** (负责代码质量和清理)
- GPU: 无需
- 时间: ~2-3h (审查 + 合并 + dry-run)
- **必须在 B·D·C 任务执行前完成 (至少完成 Imagination + WM eval + Policy eval 管线)**

---

### B: Imagination 管线预验证 (GPU 4-5, ~1h) 🔴 高优先级

> **目的**: 用 WM ckpt-200 (质量一般但可用) 做小规模端到端测试，提前暴露管线问题，不等 WM 训练完。

| task_id | 任务 | GPU | 依赖 | 预估 |
|---------|------|-----|------|------|
| **T-IMAGINATION-PRECHECK** | 用 ckpt-200 生成 10-20 条合成轨迹 → VLM 标注 → 检查 D_syn+ 筛选 | 4(Imagination) + 6(VLM) | T-SCRIPTS-CONSOLIDATE (Imagination 管线) | ~1h |

**验证点**: (1) Imagination 代码路径正确 (BUG-019 修复有效); (2) 输出格式与 VLM 标注脚本兼容; (3) D_syn+ 筛选逻辑正常; (4) 如果 ckpt-200 完全不可用，也能确认"WM 需要更多步数"

### D: WM checkpoint-200 质量评估 (GPU 4, ~30min) 🟡 中优先级

| task_id | 任务 | GPU | 依赖 | 预估 |
|---------|------|-----|------|------|
| **T-WM-V3-HEALTHCHECK** | 评估 ckpt-200 的 PSNR/SSIM/LPIPS，确认训练方向正确 | 4 | T-SCRIPTS-CONSOLIDATE (WM eval 管线) | ~30min |

**门控**: PSNR > 15 = 训练方向正确 (pretrained=22.35, 早期 step 预期较低但不应极差)

### C: Policy pipeline dry-run (GPU 8, ~20min) 🟡 中优先级

| task_id | 任务 | GPU | 依赖 | 预估 |
|---------|------|-----|------|------|
| **T-POLICY-DRYRUN** | 用 v3 mixed 数据 + 伪 D_syn (high_suc 冒充) 跑 10 步策略更新, 验证代码通路 | 8 | T-SCRIPTS-CONSOLIDATE (Policy 管线) | ~20min |

**验证点**: (1) 代码能跑通; (2) loss 下降合理; (3) EMA 权重保存格式正确 (BUG-018); (4) Weighted FM 损失在 vlm_reward=0 样本上行为符合预期

### 执行顺序

```
T-SCRIPTS-CONSOLIDATE (无 GPU, ~2-3h)
├─ Imagination 管线固化 → 立即执行 B (GPU 4-5+6)
├─ WM eval 管线固化 → 立即执行 D (GPU 4)
└─ Policy 管线固化 → 立即执行 C (GPU 8)
    └─ 全部完成 → 等 WM v3 训完 → 用固化脚本执行正式 Phase 3-7
```

---

## v1 主线 (归档): Policy 架构适配 → Iter-1 完整迭代

<details><summary>v1 Policy + Iter-1 详情 (点击展开)</summary>

### ADR-009: Policy 架构不匹配 — 已决策

**问题**: ShortCut Flow base checkpoint 使用 **视觉编码器** (PlainConv, `global_cond_dim=626`)，  
当前 `VLAWPolicyUpdater` 使用 **raw state** (`global_cond_dim=50`)，权重无法加载。

**决策**: **方案 1 — 适配 VLAWPolicyUpdater 使用视觉 observations (与 base ckpt 对齐)**

**理由**:
- 保留 base ckpt 的预训练视觉表征，避免从零训练
- 与 VLAW 论文一致（论文使用 image observations，不是 raw state）
- PlainConv encoder 已在 base ckpt 中训练好，直接复用

**具体实施步骤**:

| 步骤 | 描述 | owner |
|------|------|-------|
| 1 | 分析 base ckpt 结构: PlainConv 编码器的输入格式、obs_horizon、image 分辨率 | Policy |
| 2 | 修改 `VLAWPolicyUpdater` + `VLAWSuccessDataset`: 接受 RGB obs → PlainConv 编码 → 626-dim global_cond | Policy |
| 3 | 修改数据加载: D_real/D_syn 需提供 RGB 图像 (已在 HDF5 的 `rgb_base` + `rgb_render` 中) | Policy |
| 4 | 验证: 加载 base ckpt → dry_run → real 数据 10 步训练 → loss 正常 | Policy |
| 5 | 端对端: Weighted FM 训练 2000 步 → ManiSkill 评估 → 对比 baseline 75% | Policy + Eval |

### 任务依赖图

```
Iter-1 完成 (策略退化 78.1% → 17.2%)
  │
  ├──→ 阶段 A: D_syn+ = 0 诊断 (最高优先级)
  │      ├──→ T-DIAG-SYN-001: 解码合成帧 (Imagination)
  │      ├──→ T-DIAG-SYN-002: 解码真实帧对照 (Data)
  │      ├──→ T-DIAG-SYN-003: VLM 交叉验证 (Reward)
  │      └──→ 人工审查 → 判断 WM 问题 or VLM 问题 → 修复
  │
  ├──→ 阶段 B1: BC 数据飞轮验证
  │      ├──→ ~~T-BC-SCALING v1: 5K步, 全≈0% (ADR-020: 重做)~~
  │      ├──→ T-BC-SCALING-V2: 20K步从零训练 scaling curve
  │      ├──→ T-BC-FLYWHEEL-A: 纯 demo 从头训练
  │      └──→ T-BC-FLYWHEEL-B: demo + D_syn+ 从头训练
  │              └──→ Go/No-Go: B > A + 3%?
  │
  ├──→ 阶段 B2: Policy-in-the-Loop Imagination RL
  │      ├──→ T-MBRL-ENV: WM+VLM 包装为 Gym env
  │      └──→ T-MBRL-BC-FINETUNE: RLPD/DSRL/PLD 在想象环境中微调
  │              └──→ Go/No-Go: success_once ≥ 78%?
  │
  └──→ 阶段 B3: 迭代 WM+VLM 共同改进
         ├──→ T-ITER-ROUND-2
         └──→ T-ITER-ROUND-3 (至少 2-3 轮判断收敛性)
```

### ❗ Iter-1 失败分析与修订路线

> **核心发现**: Iter-1 完整迭代已跑通，但策略严重退化 (78.1% → 17.2%)  
> **关键问题**: D_syn+ = 0 (200 条合成轨迹 VLM p_yes max=0.27 << α=0.8，全部被拒绝)  
> **诊断方向**: WM 生成质量不足 vs VLM 对合成图像泛化差 → 需人工判读后决策  
> **策略微调方案 (ADR-014 修订)**: 底层验证优先，不激进跳跃到 AWSC

</details>

---

## 🔬 阶段 A: D_syn+ = 0 诊断 ✅ 已完成 (v1 归档)

**目标**: 判断合成轨迹被 VLM 全部拒绝的根因是 WM 还是 VLM → **结论: WM 合成质量为主要瓶颈**

| 步骤 | task_id | 任务 | owner | GPU | 状态 |
|------|---------|------|-------|-----|------|
| A1 | T-DIAG-SYN-001 | 8 条合成轨迹关键帧解码为 PNG | Imagination | 4 | ✅ 128 PNG |
| A2 | T-DIAG-SYN-002 | 成功/失败各 5 条真实轨迹关键帧对照 | Data | 4 | ✅ 110 PNG |
| A3 | T-DIAG-SYN-003 | VLM 交叉验证 120 条真实轨迹 (multi-image) | Reward | 6-7 | ✅ AUC=0.72 |
| A3b | T-EXP-VLM-04 | VLM use_video_format A/B 对比 | Reward | 6-7 | ✅ video AUC=0.83 |
| A4 | — | **人工审查 + 深入调查**: 确认根因为 BUG-019 (初始 latent = 随机噪声) | 人工+Imagination | — | ✅ BUG-019 已修复, 3/3 验证 OK |

**诊断结论 (最终版, 含 BUG-019 修复)**:
- ✅ **VLM 在 video 模式下判别力良好** (AUC=0.83, Youden α=0.40 时 TP=70.6%, FP=1.4%)
- ✅ **VLAWRewardConfig 默认 `use_video_format=True` 正确**，之前交叉验证脚本误用 multi-image 模式
- ✅ **α=0.8 过于激进**，Youden 最优 α≈0.40
- ✅ **D_syn+=0 的真正根因 = BUG-019** — `load_initial_frames()` 使用 `torch.randn` 随机噪声而非真实帧 VAE 编码，导致所有合成轨迹从纯噪声起步
- ✅ **BUG-019 已修复并验证** — 修复后 3/3 条生成帧质量正常（机械臂+peg 清晰可见）
- ⚠️ **旧 200 条合成数据全部无效，需重新生成** (T-IMAGINATION-002)
- ⚠️ **图像随时间推移仍有模糊趋势** (WM autoregressive rollout 误差累积，属预期行为)

**产出**:
- `results/vlaw/dsyn_diagnosis_frames/` (合成 vs 真实帧 PNG)
- `results/vlaw/dsyn_diagnosis_vlm_crossval.json` (multi-image 120 条)
- `results/vlaw/dsyn_diagnosis_vlm_crossval_video_mode.json` (video 120 条)

---

## 📈 阶段 B: 策略微调验证路线 (等 D_syn+ > 0 后)

> **核心策略 (ADR-014 修订)**: 底层到顶层的验证路线，每步有明确 go/no-go 判据

### B1: 纯 BC 验证数据飞轮 (最小可行实验)

**目标**: 证明 WM+VLM 筛选的合成数据对策略训练有正面增益

| task_id | 任务 | owner | GPU | 依赖 |
|---------|------|-------|-----|------|
| ~~T-BC-SCALING~~ | ~~Demo Scaling Curve (5K步)~~ | ~~Policy~~ | — | ✅ 完成但无信息量 (5K步全≈0%). **ADR-020: 重做20K步** |
| **T-BC-SCALING-V2** | **Demo Scaling Curve (20K步)**: 25/50/100/200/400/669 条 demo 从零训练 ShortCut Flow 20K步, 画 success_once vs demo_count 曲线 | Policy | 8-9 | GPU 释放后执行 |
| T-BC-FLYWHEEL-A | **A 组 (纯 demo)**: 50-100 条真实 demo → 从头训练 ShortCut Flow → 评估 | Policy | 8-9 | T-BC-SCALING-V2 ✅ |
| T-BC-FLYWHEEL-B | **B 组 (demo + D_syn+)**: 50-100 条真实 demo + WM+VLM 筛选的合成 demo → 从头训练 → 评估 | Policy | 8-9 | 阶段 A 完成 + D_syn+ > 0 |
| T-BC-FLYWHEEL-EVAL | 对比 A vs B 的 success_once 差异 → 如果 B > A 则数据飞轮验证通过 | Eval | 8-9 | T-BC-FLYWHEEL-A + B ✅ |

**补充消融** (可选):
- B 组内做多个 α 阈值对比 (0.5 / 0.7 / 0.8) → 确定 VLM 过滤严格度的最优值
- 关注当前模型 (1D U-Net ~1.5M params) 的数据饱和点 → 若 200 条已饱和，合成数据增益有限

**Go/No-Go**: B 组 success_once 比 A 组提升 ≥ 3% → Go to B2

### B2: Policy-in-the-Loop Imagination RL (VLAW 论文核心) ⏸️ BLOCKED (ADR-024)

> **阻塞原因**: 等待 T-TIMING-QUICKTEST + T-TIMING-FRAMESKIP 解决后恢复

**目标**: 在 RLPD/DSRL/PLD 管线中，将 ManiSkill 仿真替换为 WM+VLM (model-based RL)

**架构**: 策略输入当前帧 → 输出动作 → WM 生成下一帧 → VLM 提供奖励 → RL 在想象环境中训练

| task_id | 任务 | owner | GPU | 依赖 |
|---------|------|-------|-----|------|
| T-MBRL-ENV | 实现 WM+VLM 作为 Gym 环境接口 (ImaginationRLEnv): obs→action→WM(next_obs)→VLM(reward) | Imagination | — | ✅ 40/40 pytest, `rlft/vlaw/world_model/imagination_rl_env.py` |
| T-MBRL-BC-FINETUNE | 在 ImaginationRLEnv 中微调 BC 预训练策略 (RLPD/DSRL/PLD 管线) | Policy | 8-9 | T-MBRL-ENV ✅ + B1 go |
| T-MBRL-EVAL | 在 ManiSkill 真实环境中评估 MBRL 微调后的策略 | Eval | 8-9 | T-MBRL-BC-FINETUNE ✅ |

**技术挑战**:
- WM autoregressive 误差累积 → 想象 horizon 不宜过长 (N ≤ 15 帧)
- **已对齐**: 官方 Ctrl-World 通过第一帧锚定 (`history_idx=[0,0,...]`) + `num_history=6` 抑制漂移 → **T-WM-ALIGN-HISTORY ✅ 已完成**
- WM 生成帧无 proprioception → critic 需改为纯视觉 obs，或通过 State Predictor 补充
- VLM 推理慢 → 可能需要轻量 reward proxy (先用 VLM 标注建立数据集，训练小型 reward head)

**Go/No-Go**: MBRL 微调后 success_once ≥ baseline (78%) → Go to B3

### B3: 迭代 WM+VLM 共同改进 ⏸️ BLOCKED (ADR-024)

> **阻塞原因**: 等待 TIMING 解决 + B2 完成

**目标**: 策略提升 → 更好的 rollout → 微调 WM → 更好的合成数据 → 微调 VLM → 循环

| task_id | 任务 | owner | 依赖 |
|---------|------|-------|------|
| T-ITER-LOOP-DESIGN | 设计迭代循环: checkpoint 管理 / 数据混合策略 / 收敛监控指标 | Coordinator | B2 go |
| T-ITER-ROUND-2 | 执行第 2 轮完整迭代 | 全部 | T-ITER-LOOP-DESIGN ✅ |
| T-ITER-ROUND-3 | 执行第 3 轮完整迭代 (至少跑 2-3 轮判断收敛性) | 全部 | T-ITER-ROUND-2 ✅ |

### v1 待推进任务汇总 (大部分已被 Fresh Start 取代)

> **注意**: 以下 v1 任务大部分已失效 (数据/WM/VLM 需基于 v3 数据重做)。
> **当前主线**: 见上方 "Fresh Start 主线" 表格。
> v1 支线消融实验的结论已固化到 knowledge/decisions.md, 新 WM/VLM 消融将基于 v3 数据重新设计。

| 优先级 | task_id | 任务 | 状态 |
|--------|---------|------|------|
| ~~P0~~ | ~~T-TIMING-QUICKTEST~~ | ~~方案 A (num_interact=4)~~ | ❌ No-Go (ADR-025), 已被 Fresh Start 取代 |
| ~~P0~~ | ~~T-EXP-VLM-03-BUG~~ | ~~排查 VLM 步数消融 p_yes≈0~~ | v1 VLM, Fresh Start 后需基于 v3 数据重做 |
| ~~P0~~ | ~~T-EXP-VLM-02-EVAL~~ | ~~r=32/r=64 补齐 AUC~~ | v1 VLM, Fresh Start 后不再需要 |
| ~~P0~~ | ~~T-BC-FLYWHEEL-EVAL~~ | ~~A vs B 正式对比报告~~ | v1 数据, 结论已记录 (ADR: D_syn+ 小数据 +240%) |
| ✅ | T-BC-SCALING-V2 | Demo scaling curve 20K步 | 完成: 25d=0.02→669d=0.40 |
| ✅ | T-BC-FLYWHEEL-A/B | BC 飞轮验证 | A: 100d=0.10/669d=0.54; B: 113t=0.34(+240%)/682t=0.48(-11%) |
| ✅ | T-MBRL-ENV | ImaginationRLEnv | 40/40 pytest, 代码可复用 |
| ~~支线~~ | ~~T-EXP-WM-05v2~~ | ~~WM 步数搜索 v2~~ | v1 数据, 需基于 v3 数据重新消融 |

> **结论**: v1 消融结论 (WM 2000步, VLM r=16 200步 16帧) 作为 Fresh Start 默认配置, 后续在 v3 数据上如需调参再设计新消融。

---

## 支线: v1 扩展实验 (归档)

> **v1 归档**: 以下消融均基于旧数据 (BUG-020 污染/frame_skip 不匹配), 结论仅作为 Fresh Start 默认参数参考。
> ADR-024 阻塞已随 Fresh Start 自然消解。新消融将基于 v3 数据重新设计。

### E1: WM 配置消融 (GPU 0-3) — v1 归档

| task_id | 实验 | 变量 | 对照 (iter1 当前) | GPU | 预估时间 |
|---------|------|------|-------------------|-----|---------|
| ~~T-EXP-WM-01~~ | ~~更长训练~~ | ~~4000 steps~~ | ~~2000 steps~~ | ✅ | PSNR=24.11 (2k=4k), 无额外收益 |
| ~~**T-EXP-WM-05**~~ | ~~最优步数搜索 v1~~ | ~~ckpt@500/1000/1500/2000, reencode+num_frames=15~~ | — | ✅ 完成 | ⚠️ **结论降级**: 配置混淆 (num_frames=15, reencode 数据), ADR-018 仅限特定条件。详见数据审计报告 |
| **T-EXP-WM-05v2** | **最优步数搜索 v2 (修复版)** | 见下方详细设计 | — | ⏸️ 训练完成, 20 ckpt 就绪. **评估阻塞 (ADR-024)** | 修复 v1 的 3 个问题: 统一 num_frames=5, 含 pretrained baseline, 更密 ckpt |
| T-EXP-WM-02 | 更多训练数据 (加入 D_syn) | demos + rollouts + synthetic | demos + rollouts only | 0-3 | ~2h |
| ~~T-EXP-WM-03~~ | ~~WM num_history=4~~ | ~~4帧历史~~ | ~~1帧历史~~ | — | **已决定直接对齐官方 num_history=6 (T-WM-ALIGN-HISTORY)** |
| T-EXP-WM-04 | 学习率消融 | lr=5e-6 / 2e-5 | 1e-5 (当前) | 0-3 | ~2h×2 |
| T-EXP-WM-06 | num_frames 消融 | num_frames=5/10/15 | 5 (标准) | 0-3 | ~2h×3 |

**关注指标**: PSNR / SSIM / LPIPS（与 iter1-2000 对比），以及 Imagination 生成质量

> **T-EXP-WM-05v2 详细设计** (替代原 ADR-013)
>
> **v1 问题**: (1) `num_frames=15` 与 iter1/imagination 推理的 `num_frames=5` 不一致; (2) 训练数据用 reencode 而非 demos+rollouts; (3) 仅 4 个 ckpt 太稀疏; (4) 缺少 pretrained baseline 同条件评估
>
> **v2 方案**:
> - **训练参数**: `num_frames=5, num_history=4, batch_size=1, grad_accum=8, lr=1e-5`
> - **训练数据**: `dataset_names="demos"` (与 iter1 完全一致) 或 `"demos+rollouts_clean"` (扩展版，需数据清理后决定)
> - **训练步数**: 2000 步, `checkpointing_steps=100` (20 个 checkpoint)
> - **评估**: 使用标准评估集 (eval_fixed, 15 trajs, 70 frames), 含 pretrained ckpt-10000 作为 baseline
> - **输出**: step→PSNR/SSIM/LPIPS 曲线图 + 精确拐点定位
> - **依赖**: T-DATA-CLEANUP (数据规整) 的 Phase 1b (建立 eval_fixed) 完成后执行
> - **预估**: GPU 0-3, ~2h 训练 + ~1h 评估 20 个 ckpt

> **数据审计报告**: [`docs/vlaw/data_audit_and_reorganization_proposal.md`](../docs/vlaw/data_audit_and_reorganization_proposal.md) — 详细分析了 7 个 H5 文件的质量、分辨率不一致、reencode 110/160 条无效等问题
> **收益**: 若能证明 500-1000 步已足够，后续迭代 WM 训练时间从 2h 缩短至 0.5-1h，显著加快全 Pipeline 迭代周期。

### E2: VLM 配置消融 (GPU 6-7)

| task_id | 实验 | 变量 | 对照 (16帧 当前) | GPU | 预估时间 |
|---------|------|------|-------------------|-----|---------|
| T-EXP-VLM-01 | 帧数消融 | 4✅/8✅/32⬜ | 16 帧 | 6-7 | 4帧: acc=0.706,FP=11.1%; 8帧: acc=0.735,FP=18.5%,AUC=0.8269 |
| T-EXP-VLM-02 | LoRA rank 消融 | r=8 / r=32 / r=64 | r=16 | 3,4 | r=8 ✅ (acc=0.794, FP=0%), r=32 ✅ (acc=0.85, FP=0%), r=64 ✅ (acc=0.85, FP=0%). **r=32/r=64 缺 AUC**, 待补 T-EXP-VLM-02-EVAL |
| T-EXP-VLM-03 | 训练步数消融 | 100 / 400 / 800 步 | 200 步 | 6-7 | ✅ 全部完成. **⚠️ 严重异常**: 100步 AUC=0.315, 400步 AUC=0.383, 800步 AUC=0.390, p_yes≨10^-4. 极可能是脚本 bug → T-EXP-VLM-03-BUG |
| ~~T-EXP-VLM-04~~ | ~~video 模式 vs images 模式~~ | ~~video input~~ | ~~images (当前)~~ | 6-7 | ✅ **video AUC=0.83 >> images 0.72**, Youden α=0.40, ADR-015 |

**关注指标**: ROC-AUC / Acc@Youden / FP@α=0.8 / FP@Youden（与当前 AUC=0.808 对比）

### E3: 交叉实验 (等主线 Imagination 成功后)

| task_id | 实验 | 描述 |
|---------|------|------|
| T-EXP-CROSS-01 | WM 质量 vs Imagination 成功率 | 用不同 WM ckpt 跑相同 Imagination，对比合成轨迹的 VLM 成功率 |
| T-EXP-CROSS-02 | VLM 灵敏度 vs 策略改进 | 不同 VLM 过滤阈值下的策略提升幅度 |

### 实验执行规则

1. **不阻塞主线**: 主线任务优先使用 GPU，支线实验在 GPU 空闲时执行
2. **统一评估**: 所有实验用同一套测试数据、相同 eval 脚本，结果写入 `results/vlaw/ablation/`
3. **增量执行**: 先跑 E1.01 和 E2.01（最高信息量），根据结果再决定是否继续其他项
4. **结果表格**: 每组实验完成后，生成 markdown 对比表追加到 `docs/vlaw/baselines_and_evaluation.md`

---

## 关键资源

| 资产 | 路径 |
|------|------|
| WM iter1 ckpt | `checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt` (4.4GB) |
| WM pretrained | `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt` |
| VLM LoRA 16帧 | `checkpoints/vlaw/reward_model/lora_iter1_16frame/` (23MB) |
| VLM 基座 | `checkpoints/vlaw/reward_model/qwen_vl/` (8.3GB) |
| Policy base | `checkpoints/il/best_eval_success_once.pt` |
| D_real 编码 | `data/vlaw/encoded/reencode_highsuc_inc20/` (235条, 4378窗口) |

## 参考命令

### WM 训练 (Iter-2 或消融实验)

```bash
tmux new-session -d -s wm "
eval \"\$(conda shell.bash hook)\" && conda activate ctrl_world &&
cd /home/wjz/rl-vla/ctrl_world &&
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline \
accelerate launch --num_processes 4 --use_deepspeed --deepspeed_config_file ds_zero2.json \
  scripts/train_wm.py \
  --ckpt_path ../checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt \
  --dataset_root_path ../data/vlaw/encoded \
  --dataset_meta_info_path ../data/vlaw/meta_info/maniskill \
  --output_dir ../checkpoints/vlaw/world_model/{OUTPUT_NAME} \
  --max_train_steps {STEPS} --validation_steps 500 --checkpointing_steps 500 \
  --gradient_accumulation_steps 8 \
  --task_type maniskill --height 384 --width 192 --action_dim 7 \
  --num_frames 15 --num_history {NUM_HISTORY} \
  2>&1 | tee /home/wjz/rl-vla/logs/vlaw/wm_{NAME}_train.log
"
```

### VLM 训练 (Iter-2 或消融实验)

```bash
tmux new-session -d -s vlm "
cd /home/wjz/rl-vla &&
CUDA_VISIBLE_DEVICES=6,7 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python rlft/vlaw/reward/train_reward_model.py \
  --data_dirs data/vlaw/rollouts/iter1 data/vlaw/rollouts/iter1_highsuc \
  --tasks LiftPegUpright-v1 \
  --model_path checkpoints/vlaw/reward_model/qwen_vl \
  --output_dir checkpoints/vlaw/reward_model/{OUTPUT_NAME} \
  --num_frames {NUM_FRAMES} --train_steps {STEPS} --lora_r {LORA_R} \
  --per_device_batch_size 1 --gradient_accumulation_steps 128 \
  --use_wandb --wandb_project vlaw-reward \
  2>&1 | tee logs/vlaw/vlm_{NAME}_train.log
"
```
