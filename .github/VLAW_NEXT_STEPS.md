# VLAW 下一步推进计划

> **最后更新**: 2026-03-05 21:00 | **状态面板**: [`vlaw-status.md`](vlaw-status.md) | **高层计划**: [`VLAW_REPRODUCTION_PLAN.md`](VLAW_REPRODUCTION_PLAN.md)
> **⛔ 全局阻塞**: Imagination 人工审核不可用, WM 继续训练中, 所有下游暂停 (ADR-034)
> **v1 归档**: [`docs/vlaw/archive/`](../docs/vlaw/archive/) (v1 实验全部已归档)
> **已归档计划**: `_archive/v2/VLAW_FRESH_START_PLAN.md`, `_archive/v2/VLAW_DATA_COLLECTION_PLAN_V3.md`

---

## 当前进度概览

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 0 数据 | ✅ | v3 frame_skip=4, mixed=1200/eval=20/high_suc=552, VAE 编码 + stat.json 就绪 |
| Phase 1 WM | ✅ | 2000步训练完成, **使用 ckpt-400** (PSNR=29.88, SSIM=0.9402, LPIPS=0.0368) |
| Phase 2 VLM | ✅ | LoRA v3 300步 最佳 (r=16, recall=61.2%, FP=0%), 消融全部完成 |
| WM 等待期 | ✅ | **T-SCRIPTS-CONSOLIDATE✅ + BDC 三线全部完成** (D✅ B✅ C✅) |
| Phase 3 Imagination | ✅ | 200条 merged H5, 4-GPU 并行, 96.1MB |
| Phase 3b D_real | ✅ | 434/1200 success (36.2%), α=0.5 |
| Phase 3c D_syn | ✅ | **122/200 D_syn+ (61.0%)**, α=0.5 |
| WM 深度验证 | ✅ | 完成, 但 eval_WM PSNR 被证明有误导性 (ADR-034) |
| Imagination 全面评估 | ✅ | 自动化指标 OK, **但人工审核不通过** |
| **⛔ Imagination 人工审核** | **❌** | **视觉质量不可用, 阻塞所有下游** |
| **Phase 1b WM 继续训练** | **🔄** | **从 pretrained 训练 4000+步, 以 Imagination 肉眼效果为准** |
| **Phase 4 策略更新** | **⛔ 阻塞** | 等 WM Imagination 人工确认可用 |
| Phase 5 评估 | ⛔ 阻塞 | 等 Phase 4 |

### v1→v3 关键结论 (已固化到 `knowledge/`)

- **数据**: frame_skip=4 (5Hz) 精确匹配 WM, max_episode_steps=200 (ADR-026)
- **WM**: 从 pretrained 微调 2000 步足够 (ADR-010), num_interact=12 (ADR-025)
- **VLM**: r=16 最优, 300步甜蜜点, 16帧 video 模式, accum=16, α=0.5(平衡)/0.8(保守) (ADR-028)
- **BC 飞轮**: 1条 D_syn+ ≈ 28条 demo, 小数据场景 +240% (v1 实验结论)
- **关键 Bug**: BUG-019(随机 latent), BUG-020(rgb 坍塌), BUG-022(vec env no reset), BUG-024(selection bias)

---

## 🚀 主线: Iteration 1 (v3 数据)

> **路线**: ~~Phase 0~~ → ~~Phase 2 VLM~~ → ~~Phase 1 WM~~ → ~~Phase 3 Imagination~~ → **⛔ Phase 1b WM 继续训练** → Phase 4 策略 → Phase 5 评估

| 优先级 | task_id | 任务 | owner | GPU | 状态 |
|--------|---------|------|-------|-----|------|
| ~~P0~~ | ~~**T-WM-V3**~~ | WM 微调 (pretrained→v3, 2000步) | WM | 0-3 | ⚠️ ckpt-400 eval_WM 合格但 Imagination 不可用 |
| **P0** | **T-WM-V3-EXTENDED** | **WM 继续训练 (4000+步, Imagination 可视化评估)** | **WM** | **0-3** | **🔄 进行中** |
| ~~P1~~ | ~~**T-IMAGINATION-V3**~~ | Imagination (v3 WM + AWSC policy, 200条) | Imagination | 0-3 | ✅ 完成 (96.1MB merged) |
| ~~P1~~ | ~~**T-VLM-LABEL-REAL**~~ | D_real VLM 标注 (1200条, α=0.5) | Reward | 6 | ✅ 完成 (434/1200=36.2%) |
| ~~P1~~ | ~~**T-VLM-LABEL-SYN**~~ | D_syn VLM 标注 (200条, α=0.5) | Reward | 6 | ✅ 完成 (**122/200=61.0%**) |
| ~~P1~~ | ~~**T-WM-DEEP-VIZ**~~ | WM 深度可视化验证 + eval_wm 合并 | Eval | — | ✅ 完成 |
| ~~P1~~ | ~~**T-IMAG-EVAL**~~ | Imagination 全面评估 (5维度: latent/decode/action/state/VLM) | Eval | 4 | ✅ 完成 (但人工审核不通过) |
| **P2** | **T-POLICY-V3** | **Weighted FM 策略训练 (D_real+ ∪ D_syn+)** | **Policy** | **8-9** | **⛔ 阻塞 (等 Phase 1b)** |
| P2 | T-EVAL-V3 | ManiSkill 评估 (baseline=78%) | Eval | 8-9 | ⏳ 等策略 |
| P3 | T-ITER-ROUND-2 | 第 2 轮迭代 (如 Iter-1 有效) | 全部 | — | ⏳ |

### 关键参数

- **WM**: pretrained→finetuning 2000步, num_frames=5, num_history=6, lr=1e-5, DeepSpeed ZeRO-2
- **VLM**: LoRA r=16, **300步**, 16帧 video, accum=16
- **Imagination**: num_interact=12, 真实首帧 latent (BUG-019 已修复)
- **门控**: WM PSNR>18 | VLM FP<20% | D_syn+ 产出率>5% | 策略 success > baseline

---

## 🔧 WM 等待期: 脚本固化 + BDC 三线并行 ✅ 全部完成

> GPU 4-9 全部空闲，利用 WM 训练等待期进行管线预验证和代码整固。
> **原则**: 先固化脚本 → 再用固化脚本执行 BDC。
> **状态**: 全部完成 (2026-03-05)

### ~~T-SCRIPTS-CONSOLIDATE~~: 管线脚本固化 ✅ 完成

> 6条稳定入口 in `rlft/vlaw/scripts/`, 27个冗余脚本归档至 `scripts/vlaw/_archive/`

> **问题**: 同功能存在 2-5 个冗余脚本版本，bug fix 散落在一次性脚本中。
> **目标**: 审查→合并→验证→归档，每个管线阶段一个稳定入口。

| 管线 | 稳定版目标路径 | 冗余版本数 | 关键 bug fix 需合入 |
|------|---------------|-----------|-------------------|
| Imagination | `rlft/vlaw/scripts/run_imagination.py` | ~14 | BUG-019, BUG-017, ADR-021 |
| WM 评估 | `rlft/vlaw/scripts/eval_wm.py` (新建) | ~8 | PSNR/SSIM/LPIPS 标准流程 |
| VLM 标注 | `rlft/vlaw/scripts/label_trajectories.py` (新建) | ~4 | BUG-011, ADR-019 |
| VLM 评估 | `rlft/vlaw/reward/eval_threshold_ablation.py` (已有) | ~7 | ADR-028, ADR-029 |
| Policy 评估 | `rlft/vlaw/scripts/evaluate_policy.py` (已有) | ~2 | BUG-016 |
| Policy 训练 | `rlft/vlaw/scripts/run_policy_update.py` (已有) | ~3 | BUG-018, ADR-012 |

- owner: **Eval-Agent** | GPU: 无 | 预估: ~2-3h
- 冗余脚本归档至 `scripts/vlaw/_archive/`

### B·D·C 三线 ✅ 全部完成 (脚本固化后执行)

| task_id | 任务 | GPU | 结果 | 状态 |
|---------|------|-----|------|------|
| ~~**T-IMAGINATION-PRECHECK**~~ [B] | ckpt-400 生成 15条→VLM 标注 | 4+6 | D_syn+=5/15 (33.3% @ α=0.5), p_yes mean=0.42 max=0.68 | ✅ |
| ~~**T-WM-V3-HEALTHCHECK**~~ [D] | ckpt-400 PSNR/SSIM/LPIPS 评估 | 5 | PSNR=29.76, SSIM=0.9366, LPIPS=0.0431 (远超门控>18) | ✅ |
| ~~**T-POLICY-DRYRUN**~~ [C] | v3 mixed + 伪 D_syn 跑 10 步策略更新 | 8 | loss=2.238, ema_agent 格式正确 (BUG-018 修复生效) | ✅ |

```
T-SCRIPTS-CONSOLIDATE (无GPU, ~2-3h) ✅
├→ B: Imagination 预验证 (GPU 4+6) ✅ D_syn+=5/15 (33.3%)
├→ D: WM 健康检查 (GPU 5) ✅ PSNR=29.76
└→ C: Policy dry-run (GPU 8) ✅ loss=2.24
   └→ 全部完成 → 等 WM v3 训完 (~step 440/2000) → 正式 Phase 3-5
```

---

## 未来: B2 Model-Based RL + B3 迭代

> 等 Iter-1 v3 完成后推进。

### B2: Policy-in-the-Loop Imagination RL

- **ImaginationRLEnv** 已实现 (40/40 pytest, `rlft/vlaw/world_model/imagination_rl_env.py`)
- WM+VLM 包装为 Gym 环境 → RLPD/DSRL/PLD 微调
- Go/No-Go: success_once ≥ baseline (78%)

### B3: 迭代 WM+VLM 共同改进

- 策略提升 → 更好 rollout → 微调 WM → 更好合成数据 → 微调 VLM → 循环
- 至少 2-3 轮判断收敛性

---

## 参考

### GPU 分配

```
GPU 0-3: WM 训练 (DeepSpeed ZeRO-2)
GPU 4-5: 数据收集 / VAE / Imagination
GPU 6-7: VLM 推理/微调
GPU 8-9: 策略训练 + 评估
```

### Conda 环境

| 用途 | 环境名 |
|------|--------|
| 数据+策略 | `rlft_ms3` |
| WM 训练 | `ctrl_world` |
| VLM 训练/推理 | `vlaw_reward` |

### 干净资产

| 资产 | 路径 |
|------|------|
| AWSC 策略 | `runs/fair_comparison/.../awsc/best_s42__1772570560/checkpoints/final.pt` |
| IL 基线策略 | `checkpoints/il/best_eval_success_once.pt` |
| WM pretrained | `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt` |
| VLM 基座 | `checkpoints/vlaw/reward_model/qwen_vl/` |
| VLM LoRA v3 | `checkpoints/vlaw/reward_model/lora_v3/` (基线) + `ablation_v3/steps_300/` (最佳) |
| WM v3 (训练中) | `checkpoints/vlaw/world_model/iter1_v3/` |
| v3 编码数据 | `data/vlaw/encoded/train/` + `data/vlaw/encoded/eval/` |
| stat.json | `data/vlaw/meta_info/maniskill/stat.json` |
