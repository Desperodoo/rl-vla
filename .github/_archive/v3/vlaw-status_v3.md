# VLAW 复现项目 — 状态仪表盘

> **最后更新**: 2026-03-08 18:00 (Phase 0-2 ✅, BDC ✅, **⚠️ Imagination 短时域评估: ckpt-1200/1400 视觉可用但缺操作动态**, WM 继续训练中 step 1525/4000)
> **核心参考**: [`VLAW_REPRODUCTION_PLAN.md`](VLAW_REPRODUCTION_PLAN.md) | [`VLAW_NEXT_STEPS.md`](VLAW_NEXT_STEPS.md) | [`knowledge/`](knowledge/)
> **归档计划**: `_archive/v2/` (Fresh Start Plan, Data Collection Plan V3, NEXT_STEPS 旧版)

---

## Fresh Start 进度

**当前阶段**: **⚠️ WM 继续训练 (Phase 1b)** | 短时域评估部分通过, WM 操作动态不足, 等待更多训练
**阻塞原因**: ADR-034 — WM 短时域视觉质量已达⚠️, 但缺乏操作动态 (peg 不被抓取/提升)
**解除条件**: WM 某 checkpoint 的短时域 Imagination 显示 peg 被操作 (或确认当前质量足以让 VLM 提取有用信号)
**管线可视化 ✅**: 7 脚本均已添加 `--visualize` flag (含 eval_imagination.py)

### Phase 0 — 数据准备 ✅ 完成
- ✅ v3 采集: frame_skip=4 (5Hz), mixed=1200条(46% suc), eval=20条, high_suc=552条
- ✅ VAE 编码: mixed 1.1GB + high_suc 272MB + eval 13MB, latent (T,4,48,24) fp16
- ✅ stat.json 重生成 (state_01/state_99 格式, 49618 样本)
- ✅ 质量报告 GO: 11/11 检查全通过 (`results/vlaw/data_quality_report_v3.md`)
- Bug 修复历史: BUG-022(vec env), BUG-024(selection bias), ADR-026(frame_skip), ADR-027 — 详见 `knowledge/`

### Phase 1 — WM 微调 ⚠️ ckpt-400 eval_WM 合格但 Imagination 不可用
- ✅ 前置: VAE 编码 + stat.json 就绪
- ✅ **训练完成**: GPU 0-3, DeepSpeed ZeRO-2, **2000/2000 steps**, final loss=0.00124
- ⚠️ **ckpt-2000 未保存** — 仅有 checkpoint-200.pt (07:08) + checkpoint-400.pt (12:34)
- ⚠️ **ckpt-400 eval_WM 高分但 Imagination 不通过人工审核** (ADR-034)
  - eval_WM PSNR=29.76 (单步 GT history, 有误导性)
  - Imagination 200 条人工审核：**几乎完全不可用**
- **脚本**: `scripts/vlaw/run/train_wm_v3.sh`
- **输出**: `checkpoints/vlaw/world_model/iter1_v3/`

### Phase 1b — WM 继续训练 🔄 进行中 (ADR-034)
- **目标**: 获取 Imagination 可视化质量通过人工审核的 WM checkpoint
- **方案**: 从 pretrained 重新训练 4000+ 步, 每 200 步保存 ckpt
- **评估指标升级**: eval_WM PSNR 仅供参考, **以 Imagination 可视化 + 人工审核为准**
- **GPU**: 4-7
- **输出**: `checkpoints/vlaw/world_model/iter1_v3_ext/`
- **当前进度**: step ~1525/4000 (38%), loss ~0.27-0.30, ~95s/step
- **已保存 ckpt**: 200, 400, 600, 800, 1000, 1200, 1400
- **阻塞**: Phase 4/5/Iter-2 全部暂停, 直到 Imagination 人工确认可用

#### Imagination 密集评估 (num_interact=4, T=20 短时域)

> **关键发现**: 之前 num_interact=12 (60帧) 评估不公平 — LiftPegUpright 任务仅需 1.5-2.5s (10-17帧)，67% 帧数为 WM 超出任务时长的外推，导致严重幻觉。短时域 num_interact=4 (20帧, ~3s) 匹配实际任务需求。

**长时域评估 (num_interact=12, T=60) — 全部不可用**:

| Checkpoint | Top View | Side View | Overall | 主要问题 |
|-----------|----------|-----------|---------|---------|
| pretrained | ❌ | ❌ | ❌ | 第1帧后彻底崩溃 |
| ckpt-400 | ⚠️ | ❌ | ❌ | 颜色漂移, 物体失去物理一致性 |
| ckpt-800 | ⚠️ | ❌ | ❌ | peg 变小/模糊, 无交互 |
| ckpt-1200 | ⚠️ | ❌ | ❌ | peg 消失/融入桌面, 无交互 |

**短时域评估 (num_interact=4, T=20) — 视觉质量显著改善**:

| Checkpoint | Top View | Side View | Overall | 评估详情 |
|-----------|----------|-----------|---------|---------|
| ckpt-800 | ⚠️ | ❌ | ❌ | peg 仍然消失, 臂部后期变形 |
| ckpt-1200 | ⚠️✅ | ⚠️ | ⚠️ | **最佳** — peg 持续可见, 臂部清晰, 但无操作动态 |
| ckpt-1400 | ⚠️✅ | ⚠️ | ⚠️ | 与1200相当, peg 持久性略好, 无操作动态 |

**核心结论**:
1. 短时域解决了视觉退化问题 — 物体持续可见, 颜色正确, 臂部形状维持
2. 但 WM 不生成操作动态 — 无抓取/提升动作, peg 始终静止在桌面
3. 这是物理/动态建模的根本限制, 非时域长度问题
4. ckpt-1200/1400 为当前最优, 但训练仍在继续 (当前 step 1525/4000)
5. 可视化输出: `data/vlaw/synthetic/wm_eval_step{800,1200,1400}_short/viz_dual/`

### Phase 2 — VLM 微调 ✅ 完成 + 消融全部完成
- ✅ LoRA v3 基线训练完成 (200 steps, r=16, accum=16, 16帧 video)
- 输出: `checkpoints/vlaw/reward_model/lora_v3/`
- **基线指标** (eval 180 样本, α=0.8):
  - FP = 0% ✅ (目标 <20%)
  - Precision = 1.0, Recall = 42.4%
  - Accuracy = 72.8%, mean_p_yes = 0.565

#### VLM v3 消融实验 ✅ 全部完成 (2026-03-05)
- 消融报告: `results/vlaw/vlm_ablation_v3_report.md`

**Steps 消融** (r=16, α=0.8):
| Steps | Acc | Prec | Recall | FP% | 备注 |
|-------|-----|------|--------|-----|------|
| 100 | 52.8% | 0.0% | 0.0% | 0.0% | 未学习 |
| 200 | 72.8% | 100% | 42.4% | 0.0% | =v3基线 |
| **300** | **81.7%** | **100%** | **61.2%** | **0.0%** | **✅ 最佳甜蜜点** |
| 400 | 85.6% | 96.8% | 71.8% | 2.1% | 轻微 FP |

**Threshold 消融** (r=16, 300 steps = 最佳模型):
| α | Acc | Prec | Recall | FP% | 备注 |
|---|-----|------|--------|-----|------|
| **0.5** | **86.7%** | **85.9%** | **85.9%** | **12.6%** | **推荐平衡点** |
| 0.7 | 83.3% | 96.6% | 67.1% | 2.1% | 低 FP |
| 0.8 | 81.7% | 100% | 61.2% | 0.0% | 保守 |

**LoRA Rank 消融**:
| Rank | Acc | Recall | FP% | 备注 |
|------|-----|--------|-----|------|
| r=8 | — | 1.2% | — | ❌ 容量不足, 不可用 |
| **r=16** | **81.7%** | **61.2%** | **0.0%** | **✅ 最优** |

**推荐最佳 VLM 配置**: r=16, 300 steps, α=0.5~0.8 (视用途而定)
- α=0.5: 平衡点 (高 recall 85.9%, FP 12.6%)
- α=0.8: 保守点 (零 FP, recall 61.2%)

### WM 等待期: 脚本固化 + BDC 三线并行 ✅ 全部完成

> **详见**: [`VLAW_NEXT_STEPS.md` §BDC](VLAW_NEXT_STEPS.md)

| task_id | 任务 | 状态 |
|---------|------|------|
| **T-SCRIPTS-CONSOLIDATE** | 管线脚本固化: 6 条稳定入口已就位 + **可视化已添加** (`--visualize` flag), 27 个冗余脚本归档至 `scripts/vlaw/_archive/` | ✅ 完成 |
| **T-IMAGINATION-PRECHECK** | [B] Imagination 管线预验证 (ckpt-400, 15条, GPU 4+6) — **生成 15/15 ✅ + VLM 标注 ✅**: α=0.5 D_syn+=5/15 (33.3%), p_yes mean=0.42 max=0.68; α=0.8 D_syn+=0/15; 通过 traj: 0001,0007,0010,0011,0014; 输出 `data/vlaw/labeled/precheck_ckpt400/` | ✅ 完成 (03-05) |
| **T-WM-V3-HEALTHCHECK** | [D] WM ckpt-400 PSNR/SSIM/LPIPS 评估 (GPU 5) — **✅ PSNR=29.76 >> 15 门控通过**, SSIM=0.9366, LPIPS=0.0431; pretrained baseline: PSNR=22.33, SSIM=0.7546, LPIPS=0.1993; Δ PSNR=+7.43, 训练方向正确 | ✅ 完成 (03-05) |
| **T-POLICY-DRYRUN** | [C] Policy pipeline dry-run 10步 (GPU 8) — loss=2.238, ema_agent ✅, ckpt 94.1MB | ✅ 完成 (03-05) |

### Iter-1 — 迭代管线执行 🔄 进行中

| Step | 任务 | 状态 | 结果 |
|------|------|------|------|
| 3 | D_real VLM 标注 (1200条) | ✅ 完成 | 434 success (36.2%), p_yes mean=0.4446, α=0.5, 143s |
| 4 | WM 微调 | ✅ 已完成 | ckpt-400 PSNR=29.88, SSIM=0.9402, LPIPS=0.0368 |
| 5 | Imagination (200条) | ✅ 完成 | 4-GPU 并行, 96.1MB merged H5, 8 viz PNGs |
| 6 | D_syn VLM 标注 (200条) | ✅ 完成 | **D_syn+ = 122/200 (61.0%)**, p_yes mean=0.5596, α=0.5, 70.7s |
| — | WM 深度验证 | ✅ 完成 | pretrained vs ckpt-400 GT/Pred 对比, per-frame PSNR 分解, Imagination viz 根因分析 |
| — | **Imagination 全面评估** | **✅ 完成** | **自动化指标 OK, 但人工审核不通过 (ADR-034)** |
| — | **⛔ Imagination 人工审核** | **⚠️ 部分通过** | **长时域(60帧)不可用; 短时域(20帧) ckpt-1200/1400 视觉可用但缺操作动态** |
| 1b | **WM 继续训练** | **🔄 进行中** | step 1525/4000, 短时域 ckpt-1200/1400 视觉⚠️, 等更多训练改善动态 |
| 7 | 策略更新 | ⛔ 阻塞 | 等 WM Imagination 人工确认可用 |
| 8 | 评估 | ⛔ 阻塞 | 等 Step 7 |

**⛔ 阻塞**: ADR-034 — Imagination 人工审核不可用, 所有下游暂停
**关键路径**: WM 继续训练 → 关键 ckpt Imagination 可视化 → **人工确认可用** → 恢复 Policy Update
**核心教训**: eval_WM PSNR (单步 GT history) 不能反映 Imagination 实际质量, 以后 WM 质量必须通过 Imagination 可视化 + 人工审核确认

---

## 干净资产

| 资产 | 路径 | 状态 |
|------|------|------|
| 预训练策略 (IL) | `checkpoints/il/best_eval_success_once.pt` | ✅ baseline ~78% success_once |
| AWSC 微调策略 | `runs/fair_comparison/.../awsc/best_s42__1772570560/checkpoints/final.pt` | ✅ eval 验证: success_once=80%, success_at_end=46% |
| WM 预训练 | `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt` | ✅ 8.7GB |
| SVD / CLIP | `checkpoints/vlaw/world_model/pretrained/{svd,clip}/` | ✅ |
| VLM 基座 | `checkpoints/vlaw/reward_model/qwen_vl/` | ✅ 8.3GB |
| State Predictor | `checkpoints/vlaw/state_predictor/` | ✅ |
| 代码库 | `rlft/vlaw/`, `scripts/vlaw/`, `ctrl_world/` | ✅ 管线脚本已固化 (7 入口 in `rlft/vlaw/scripts/`, 27 冗余→`_archive/`, `--visualize` 可视化就绪, eval_wm.py 580行, **eval_imagination.py 1160行 新增**) |
| stat.json | `data/vlaw/meta_info/maniskill/stat.json` | ✅ v3 数据重生成 (state_01/state_99 格式, 49618 样本) |

---

## 数据目录

| 目录 | 状态 | 说明 |
|------|------|------|
| `data/vlaw/rollouts/mixed/` | ✅ **新** | v3 frame_skip=4, 1200条, 46% success, 702MB |
| `data/vlaw/rollouts/eval/` | ✅ **新** | v3 frame_skip=4, 20条, 9MB |
| `data/vlaw/rollouts/high_suc/` | ✅ **新** | 从 mixed 筛选 success_at_end, 552条, 204MB |
| `data/vlaw/encoded/train/` | ✅ **新** | v3 VAE latent, mixed 1.1GB + high_suc 272MB, shape (T,4,48,24) fp16 |
| `data/vlaw/encoded/eval/` | ✅ **新** | v3 VAE latent, 20条, 13MB |
| `data/vlaw/rollouts_awsc/` | 🗑️ 已删除 | frame_skip=5 错误数据 |
| `data/vlaw/encoded_awsc/` | 🗑️ 已删除 | 来自错误数据的编码 |
| `data/vlaw/synthetic/precheck_ckpt400/` | ✅ | Imagination 预验证 15条生成完毕, 7.3MB |
| `data/vlaw/labeled/precheck_ckpt400/` | ✅ | VLM 标注完成: 5/15 通过 (α=0.5, 33.3%) |
| `data/vlaw/labeled/` | ✅ | precheck_ckpt400 标注已完成 |
| `data/vlaw/labeled/iter1_real/` | ✅ **新** | D_real 标注: 1200条, 434 success (36.2%), α=0.5 |
| `data/vlaw/synthetic/iter1/` | ✅ **新** | Imagination: 200条 merged H5 (96.1MB), 4-GPU 并行 |
| `data/vlaw/labeled/iter1_syn/` | ✅ **新** | D_syn 标注: 200条, **D_syn+=122 (61.0%)**, α=0.5 |
| `results/vlaw/imagination_eval/` | ✅ **新** | Imagination 全面评估: report.md + full_results.json + 21 PNG/JSON (5 子目录) |
| `data/vlaw/_archive/` | 📦 | v1 污染数据 + v2 frame_skip=3 数据 |
| `data/vlaw/synthetic/wm_eval_step800_short/` | ✅ **新** | 短时域评估: ckpt-800, num_interact=4, 20条, 19.0min, viz_dual/ |
| `data/vlaw/synthetic/wm_eval_step1200_short/` | ✅ **新** | 短时域评估: ckpt-1200, num_interact=4, 20条, 18.4min, viz_dual/ |
| `data/vlaw/synthetic/wm_eval_step1400_short/` | ✅ **新** | 短时域评估: ckpt-1400, num_interact=4, 20条, 21.5min, viz_dual/ |

---

## Checkpoints

| 目录 | 状态 | 说明 |
|------|------|------|
| `checkpoints/vlaw/world_model/pretrained/` | ✅ | SVD + CLIP + Ctrl-World |
| `checkpoints/vlaw/world_model/iter1_v2/` | 📦 | 基于旧数据训练 (不适用于 v3) |
| `checkpoints/vlaw/world_model/iter1_v3/` | ✅ | v3 训练完成 (2000步, ckpt-200+400 保存, **使用 ckpt-400**: PSNR=29.88) |
| `checkpoints/vlaw/reward_model/qwen_vl/` | ✅ | Qwen3-VL-4B |
| `checkpoints/vlaw/reward_model/lora_v2/` | 📦 | 基于旧数据训练 (不适用于 v3) |
| `checkpoints/vlaw/reward_model/lora_v3/` | ✅ | v3 基线完成, FP=0%, recall=42.4% |
| `checkpoints/vlaw/reward_model/ablation_v3/` | ✅ | v3 消融完成 (steps_400 + lora_r8), 最佳: 300步 r=16 |
| `checkpoints/vlaw/policy/dryrun/` | ✅ | Policy dry-run 10步 (loss=2.238, ema_agent OK) |
| `checkpoints/vlaw/policy/` | 待填 | 待 Iter-1 Step 7 正式策略训练 |
| `checkpoints/vlaw/_archive/v1/` | 📦 | 全部旧 ckpt (~168GB) |

---

## GPU 状态

| GPU | 分配 | 状态 |
|-----|------|------|
| 0-1 | ⚠️ LM Studio 占用 | 🔴 **不可用** |
| 2 | ACP exp_aligned 训练 | 🔴 **进行中** (PID 1890877, 6.5GB) |
| 3 | 空闲 | 🟢 |
| 4-7 | **WM 扩展训练** | 🔴 **Phase 1b — iter1_v3_ext step 1525/4000 (~95s/step)** |
| 8-9 | 空闲 | 🟡 **可用于评估** |

---

### Side Branch: ACP Online Reward for RLPD (ADR-038)

> **状态**: 代码+测试完成, 待设备空闲后上机调试
> **目标**: 用 ACP value model TD-shaped reward 替换 ManiSkill sim dense reward 进行 SAC/AWSC 在线训练

**核心实现**:
- `rlft/envs/acp_reward_wrapper.py` — `DualCameraRewardWrapper` + `ACPRewardComputer`
- `rlft/online/train_rlpd.py` — 新增 `--reward_mode acp|acp_blend` 配置
- `rlft/tests/vlaw/test_acp_reward_wrapper.py` — 26 个测试全部通过

**Running**: `CUDA_VISIBLE_DEVICES=0,1 python -m rlft.online.train_rlpd --reward_mode acp --acp_device cuda:1`
**关键设计**: r(s,s') = (V(s') - V(s)) * scale (PBRS, 保持最优策略不变), ACP model 默认 cuda:1 (~3GB)

---

## v1 关键经验

> 完整记录见 [`knowledge/decisions.md`](knowledge/decisions.md) 和 [`knowledge/bugs-and-fixes.md`](knowledge/bugs-and-fixes.md)。
> 精华摘要见 [`VLAW_NEXT_STEPS.md`](VLAW_NEXT_STEPS.md) "v1→v3 关键结论" 部分。

核心 Bug: BUG-019(随机latent) / BUG-020(rgb坍塌) / BUG-022(vec env) / BUG-024(selection bias)
核心 ADR: frame_skip=4 / WM 2000步 / VLM r=16 300步 video α=0.5~0.8 / num_interact=12 / D_syn+ 小数据+240%

### D_syn+ 产出率追踪

| 版本 | WM ckpt | 数据 | α | D_syn+/total | 产出率 | 备注 |
|------|---------|------|---|-------------|--------|------|
| v1 | iter1 旧 | frame_skip=3 | 0.5 | ~1/28 | 3.5% | BUG-019 未修, 旧数据 |
| v3-precheck | ckpt-400 | frame_skip=4 | 0.5 | 5/15 | 33.3% | BUG-019 已修, v3数据, 小样本预验证 |
| **v3-iter1** | **ckpt-400** | **frame_skip=4** | **0.5** | **122/200** | **61.0%** | **🎯 正式 Iter-1, 产出率远超预期** |

> D_syn+ 产出率: v1 3.5% → v3-precheck 33.3% → **v3-iter1 61.0%** (17.4× vs v1)
> 正式运行 (200条) 产出率 61.0% 远超预验证 (15条) 的 33.3%，可能因为样本量更大更稳定
> 122 条 D_syn+ 对策略训练已具备充分价值
