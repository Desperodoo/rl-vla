# VLAW 复现项目 — 状态仪表盘

> **最后更新**: 2026-02-28 16:00 (T+5) | 技术知识见 [`knowledge/`](knowledge/) | 历史日志见 `logs/vlaw/work-logs/`

> **当前执行策略（2026-02-26 起）**: 先做 **LiftPegUpright-only** 全链路验证（数据→标注→WM→评估），PickCube/StackCube 延后到最后集中验证。
> **计划调整 (02-27 22:30)**:
> - **Phase 1.5**: ✅ **全部通过 (V1-V6, 02-28 02:00)**  — 正式训练前最小规模全链路验证完成
> - **Phase 2**: 正式训练已就绪 — WM 2000步 / VLM 200步 / 合成数据生成 / 策略迭代
> - **Track B**: WM/VLM 训练期间用 pretrained WM + zero-shot VLM 并行跑完整 Policy Pipeline
> - WM/VLM 正式训练在 Phase 1.5 通过后启动
---

## 阶段状态

| 阶段 | 状态 | 最后更新 | 关键指标/备注 |
|------|------|---------|-------------|
| P0.1 Ctrl-World 环境 | ✅ | 2026-02-25 | env: ctrl_world / VRAM: 13GB / 推理: 3视频 ✅ |
| P0.2 ManiSkill RGB | ✅ | 2026-02-24 | PSNR=27.83dB / Latent (1,4,48,24) ✅ |
| P0.3 VLM 模型 | ✅ | 2026-02-25 | Qwen3-VL-4B-Instruct @ qwen_vl/ / VRAM: 8.88GB ✅ |
| P1.1 Rollout 收集器 | ✅ | 2026-02-25 | data_collector.py / HDF5 / num_envs=64 ✅ |
| P1.2 VAE 编码管线 | ✅ | 2026-02-25 | data_pipeline.py / latent (T,4,48,24) float16 ✅ |
| P1.3 演示数据 | ✅ | 2026-02-25 | 3任务×25条 / 100% 成功 / VAE编码+stat.json ✅ |
| P2.1 WM 代码适配 | ✅ | 2026-02-25 | config/dataset/adapter/train_wm / 5个文件 ✅ |
| **P2.2 WM Phase-A 训练** | ✅ | 2026-02-26 | GPU 0-3 已完成 step 10000 / ckpt 在 `checkpoints/vlaw/world_model/phase_a/` |
| **P2.2b WM Iter-1 微调** | ✅ | 2026-02-28 | GPU 0-3, 2000 steps/2h, PSNR=23.34>18✅, ckpt: `world_model/iter1/checkpoint-2000.pt` (4.4GB) |
| P2.3 WM 验证 | ✅ | 2026-02-26 | **三模型对比完成** / pretrained 23.07dB vs ckpt-8000 22.51dB vs ckpt-10000 22.06dB / PSNR 全部>18 / SSIM: 0.8249→0.6239→0.5873 / LPIPS: 0.1158→0.1593→0.1777 / 对比图→logs/vlaw/wm_comparison_frames/ |
| P3.1 奖励模型实现 | ✅ | 2026-02-24 | reward_model.py / VRAM 17GB ✅ |
| P3.2 VLM Fine-tuning | ✅ | 2026-02-26 | **已完成!** LoRA r=16 / 200步 / GPU6 23.9GB / final→checkpoints/vlaw/reward_model/lora_iter1/final/ |
| **P3.2b VLM 16帧 微调** | ✅ | 2026-02-28 | 200步, 2GPU, loss 18.7→6.8, acc=0.824, FP=3.7%, ckpt: `lora_iter1_16frame/` |
| P3.3 VLM FP率验证 | ✅ | 2026-02-26 | **FP=0.0% ✅ PASS** (<20%) / p_yes_max=0.107 (过于保守,待D_syn改善) / 结果→data/vlaw/labeled/iter1_lora/ |
| P4.1 State Predictor | ✅ | 2026-02-25 | **已训练** / 按任务分模型 / LiftPeg(obs=25,loss=0.0067) PickCube(obs=29,loss=0.0030) StackCube(obs=25,loss=0.0042) / ckpt: checkpoints/vlaw/state_predictor/{task}/state_predictor_iter1.pt |
| P4.2 Imagination 引擎 | ✅ | 2026-02-27 | imagination_env.py 1044行 / env.step() 模式验证通过 (5条轨迹) / 2个device mismatch bug已修复 |
| P4.3 合成数据生成 (B1) | ⚠️ | 2026-02-28 | Track B pretrained WM: 50/200 成功 (190 failed cuda:0 error), vlm_reward 全=0 |
| P5.1 Weighted FM Loss | ✅ | 2026-02-25 | compute_weighted_loss / policy_updater.py ✅ |
| P5.2 策略更新验证 | ❌ | 2026-02-28 | **Architecture mismatch**: ShortCut Flow base ckpt用视觉encoder(dim=626), VLAWPolicyUpdater用raw state(dim=50), 需适配 |
| P6.1 主训练脚本 | ✅ | 2026-02-25 | `rlft/online/train_vlaw.py` / 8步完整循环 / dry_run✅ |
| P6.2 2轮迭代训练 | ⬜ | — | — |
| P7.1-P7.4 评估 | 🔄 | 2026-02-26 | Eval-Agent同源评估(20ep): success_once=95.0%, success_at_end=75.0%; 对比collector 20/20=10%显著提升 |

**图例**: ⬜ 未开始 | 🔄 进行中 | ✅ 已完成 | ❌ 阻塞 | ⚠️ 需修复

---

## 关键 Checkpoints

| 模型 | 路径 | 状态 |
|------|------|------|
| ShortCut Flow (Base) | `checkpoints/il/best_eval_success_once.pt` | ✅ |
| Ctrl-World (pretrained) | `checkpoints/vlaw/world_model/pretrained/` | ✅ 17.2GB |
| Ctrl-World (Iter1) | `checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt` | ✅ 2000步, PSNR=23.34, 4.4GB |
| Qwen3-VL-4B-Instruct | `checkpoints/vlaw/reward_model/qwen_vl/` | ✅ 8.3GB |
| Qwen3-VL-4B LoRA 16帧 | `checkpoints/vlaw/reward_model/lora_iter1_16frame/` | ✅ acc=0.824, FP=3.7%, 23MB |
| State Predictor | `checkpoints/vlaw/state_predictor/` | ✅ LiftPeg/Pick/Stack |
| Policy Iter 1 | `checkpoints/vlaw/policy/iter1/` | ❌ 架构不匹配 |

> Phase-A WM 和旧单帧 LoRA 已移至 `_archive/`。测试 checkpoints 已删除 (~89GB)。

---

## 数据目录

| 数据 | 路径 | 状态 | 数量 |
|------|------|------|------|
| 演示 (原始) | `data/vlaw/demos/{task}/` | ✅ | 3任务×25条 |
| 演示 (VAE编码) | `data/vlaw/encoded/demos/{task}/` | ✅ | 326 train samples |
| 动作统计量 | `data/vlaw/meta_info/maniskill/stat.json` | ✅ | 7D p01/p99 |
| D_real Iter1 (原始) | `data/vlaw/rollouts/iter1/` | ✅ | 3任务×50条 |
| D_real highsuc (原始) | `data/vlaw/rollouts/iter1_highsuc/` | ✅ | 50条, 70%成功 |
| D_real inc20 (原始) | `data/vlaw/rollouts/iter1_lift_inc20/` | ✅ | 40条, 30%成功 |
| D_real 全量 (VAE编码) | `data/vlaw/encoded/reencode_highsuc_inc20/` | ✅ | 235条, 4378窗口 |
| D_real (VLM标注, 16帧LoRA) | `data/vlaw/labeled/iter1_16frame_lora/` | ✅ | 210条, FP=0% |
| D_syn (pretrained WM) | `data/vlaw/synthetic/iter1_pretrained/` | ⚠️ | 50条, vlm=0 |
| 非活跃任务数据 | `data/vlaw/deferred/` | 💤 | PickCube+StackCube |

> 测试数据 (test_*, regression_*, pld_retest*) 已删除 (~127MB)。

---

## GPU 状态

| GPU | 分配 | 状态 |
|-----|------|------|
| 0-3 | WM iter1 | 🟢 空闲 ✅ 训练完成 |
| 4-5 | Imagination | 🟢 空闲 ✅ B1 完成 (50条) |
| 6-7 | VLM 16帧 | 🟢 空闲 ✅ 训练+标注完成 |
| 8-9 | Policy + Eval | 🟢 空闲 ❌ Policy 架构不匹配, 暂停 |

---

## 下一步任务

> **推进计划详见 [VLAW_NEXT_STEPS.md](VLAW_NEXT_STEPS.md)** | **评测汇总见 [`docs/vlaw/baselines_and_evaluation.md`](../docs/vlaw/baselines_and_evaluation.md)**

### 已完成 (02-24 ~ 02-28)

- ✅ Phase 1.5: V1-V6 全链路验证 (02-28 02:00)
- ✅ Phase 1.5b: V1.1 视频验证 + V1.2 wandb 集成
- ✅ WM iter1: 2000 steps, PSNR=23.34 > 18 PASS
- ✅ VLM 16帧: 200 steps, acc=0.824, FP=3.7%
- ✅ Track B1-B2: Imagination 50条 + VLM 标注 (vlm=0; D_real FP=0%)
- ✅ WM iter1 评估: PSNR=23.34, LPIPS=0.119

### 待推进

| task_id | 任务 | owner | 状态 |
|---------|------|-------|------|
| **T-POLICY-FIX** | **解决 Policy 架构不匹配** | Policy | **❌ 阻塞** |
| T-IMAGINATION-001 | Track C: 合成轨迹 (微调 WM, 200-500条) | Imagination | ⬜ |
| T-VLM-LABEL-001 | Track C: VLM 标注合成轨迹 | Reward | ⬜ |
| T-POLICY-001 | Track C: 策略更新 Weighted FM | Policy | ⬜ (等 T-POLICY-FIX) |
| T-EVAL-ITER1-001 | Iter1 策略评估 (对比 baseline 75%) | Eval | ⬜ |
| T-WM-ITER2-001 | Iter2 全流程 | 全部 | ⬜ |

> 详细历史操作见 `logs/vlaw/` 中的 Agent 结果文件。
