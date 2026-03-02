# VLAW 复现项目 — 状态仪表盘

> **最后更新**: 2026-03-02 21:25 (T+7) | 技术知识见 [`knowledge/`](knowledge/) | 任务追溯见 [`TASK_REGISTRY.md`](TASK_REGISTRY.md) | 历史日志见 `logs/vlaw/work-logs/`

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
| P4.2 Imagination 引擎 | ✅ | 2026-03-01 | imagination_env.py / 3 bug修复 (PlainConv参数+API+obs格式) / mock验证5/5通过 |
| P4.3 合成数据生成 (B1) | ⚠️ | 2026-02-28 | Track B pretrained WM: 50/200 成功 (190 failed cuda:0 error), vlm_reward 全=0 |
| P4.3b 合成数据生成 (Iter1) | ⚠️ | 2026-03-02 | **200条数据因 BUG-019 无效** (初始 latent=随机噪声), 需用修复后脚本重新生成。已验证修复: 3/3 OK (`synthetic/iter1_fixtest3/`) |
| P4.4 合成数据 VLM 标注 | ⚠️ | 2026-03-02 | **旧标注因 BUG-019 无效** (源数据为噪声), 需等 P4.3b 重新生成后重新标注。旧结果 p_yes max=0.27 系噪声帧所致 |
| P5.1 Weighted FM Loss | ✅ | 2026-02-25 | compute_weighted_loss / policy_updater.py ✅ |
| P5.2 策略更新验证 | ✅ | 2026-03-01 | **ADR-009 完成**: UNet默认参数修正(64,128,256)/64, 实数据验证✅ (10步 loss=1.642, 0 missing keys, checkpoint保存正常) |
| **P5.3 Iter-1 Weighted FM 训练** | ✅ | 2026-03-01 | 2000步, loss 1.6→0.279, ckpt: `policy/iter1/policy_iter1.pt`, wandb: vlaw_policy_iter1 |
| P6.1 主训练脚本 | ✅ | 2026-02-25 | `rlft/online/train_vlaw.py` / 8步完整循环 / dry_run✅ |
| P6.2 2轮迭代训练 | ⬜ | — | Iter-1 完成但严重退化，需调整策略后重试 |
| **P7.1 Iter-1 策略评估** | ⚠️ | 2026-03-01 | **严重退化**: baseline 78.1% → iter-1 17.2% (-60.9%), EMA bug已修, 详见 `results/vlaw/iter1_eval_report.md` |
| **P7.2 D_syn+ 诊断** | ✅ | 2026-03-02 | **阶段 A 完成 + BUG-019 根因定位**: 合成帧质量差的真正根因是初始 latent 使用随机噪声 (BUG-019)，非 WM 训练不足。修复后 3/3 验证通过，帧质量正常。 |
| **P7.3 History 对齐** | ✅ | 2026-03-02 | **T-WM-ALIGN-HISTORY**: 列表式 latent/action history + 稀疏采样 `[0,0,-12,-9,-6,-3]` + num_history=6, 3文件4处修改 |
| **P7.4 合成数据消融** | 🔄 | 2026-03-02 | **T-IMAGINATION-002b** ✅ 200/200 sliding window + **T-VLM-LABEL-002b** ✅ 标注完成 (p_yes max=0.531, D_syn+(α=0.4)=7); **T-IMAGINATION-002a** 🔄 180/200 aligned, GPU 5, ETA ~1h |
| **P8.1 BC 数据飞轮验证** | ✅ | 2026-03-02 | **T-BC-SCALING-V2 完成**: 6/6组全完成. 25d=0.02, 50d=0.04, 100d=0.10, 200d=0.16, 400d=0.32, 669d=0.40 (success_once, 20K步从零训练) |
| **P8.2 Imagination RL** | 🔄 | 2026-03-02 | **T-MBRL-ENV ✅ 已完成**: ImaginationRLEnv 实现 + 40/40 pytest 通过 (`rlft/vlaw/world_model/imagination_rl_env.py`). 待 T-MBRL-BC-FINETUNE |
| **P8.3 迭代共同改进** | ⬜ | — | 阶段 B3: 2-3 轮完整 Policy↔WM↔VLM 迭代循环 |

**图例**: ⬜ 未开始 | 🔄 进行中 | ✅ 已完成 | ❌ 阻塞 | ⚠️ 需修复

---

## 关键 Checkpoints

| 模型 | 路径 | 状态 |
|------|------|------|
| ShortCut Flow (Base) | `checkpoints/il/best_eval_success_once.pt` | ✅ |
| Ctrl-World (pretrained) | `checkpoints/vlaw/world_model/pretrained/` | ✅ 17.2GB |
| Ctrl-World (Iter1) | `checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt` | ✅ 2000步, PSNR=23.34, 4.4GB |
| Ctrl-World (4000步消融) | `checkpoints/vlaw/world_model/ablation_4000steps/` | ✅ 4000步, loss~0.02, 4×9.3GB |
| Ctrl-World (最优步数消融 v1) | `checkpoints/vlaw/world_model/ablation_optimal_steps/` | ⚠️ v1 结论降级: 配置混淆 (num_frames=15, reencode) 导致不可比。ADR-018 降级 |
| Ctrl-World (最优步数 v2) | `checkpoints/vlaw/world_model/optimal_steps_v2/` | 🔄 训练中, GPU 1,2,7,8, tmux `wm_05v2` (num_frames=5, demos, 20个ckpt) |
| Qwen3-VL-4B-Instruct | `checkpoints/vlaw/reward_model/qwen_vl/` | ✅ 8.3GB |
| Qwen3-VL-4B LoRA 16帧 | `checkpoints/vlaw/reward_model/lora_iter1_16frame/` | ✅ acc=0.824, FP=3.7%, 23MB |
| Qwen3-VL-4B LoRA 4帧 | `checkpoints/vlaw/reward_model/ablation_4frame/` | ✅ acc=0.706, FP=11.1%, 23MB |
| Qwen3-VL-4B LoRA 8帧 | `checkpoints/vlaw/reward_model/ablation_8frame/` | ✅ acc=0.735, FP=18.5%, ROC-AUC=0.8269, 23MB |
| Qwen3-VL-4B LoRA r=8 | `checkpoints/vlaw/reward_model/ablation_lora_r8/` | ✅ acc=0.794, FP=0%, 15MB |
| Qwen3-VL-4B LoRA r=32 | `checkpoints/vlaw/reward_model/ablation_lora_r32/` | 🔄 训练中, GPU 3,4 |
| Qwen3-VL-4B LoRA 100步 | `checkpoints/vlaw/reward_model/ablation_100steps/` | 🔄 训练中 (T-EXP-VLM-03), GPU 6 |
| State Predictor | `checkpoints/vlaw/state_predictor/` | ✅ LiftPeg/Pick/Stack |
| Policy Iter 1 | `checkpoints/vlaw/policy/iter1/policy_iter1.pt` | ✅ 2000步, loss=0.279, EMA修复后含 `ema_agent` 键 |
| Policy Iter 1 (中间ckpt) | `checkpoints/vlaw/policy/iter1/policy_iter1_step{500,1000,1500,2000}.pt` | ✅ 每500步保存 |

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
| D_real 全量 (VAE编码) | `data/vlaw/encoded/reencode_highsuc_inc20/` | ⚠️ DEPRECATED | 原160条, 仅1文件有效, 已被 reencode_valid 替代 |
| D_real 有效 (VAE编码) | `data/vlaw/encoded/reencode_valid/` | ✅ | 50条 (T=17-67, 192×192) |
| D_real 无效 (归档) | `data/vlaw/encoded/_archive/reencode_invalid/` | 📦 | 4文件110条 (T≤8, 128×128, 不可用) |
| D_real (VLM标注, 16帧LoRA) | `data/vlaw/labeled/iter1_16frame_lora/` | ✅ | 210条, FP=0% |
| D_syn (pretrained WM) | `data/vlaw/synthetic/iter1_pretrained/` | ⚠️ | 50条, vlm=0 |
| D_syn (Iter1 WM, real policy) | `data/vlaw/synthetic/iter1_wm_real/` | ⚠️ | 200条 **因 BUG-019 无效**, 需重新生成 |
| D_syn (VLM标注) | `data/vlaw/labeled/synthetic_iter1_wm_real/` | ⚠️ | 200条 **因 BUG-019 无效**, 需重新标注 |
| D_syn fixtest (修复验证) | `data/vlaw/synthetic/iter1_fixtest3/` | ✅ | 3条, BUG-019 修复后验证通过 |
| D_syn 002b (sliding window) | `data/vlaw/synthetic/iter1_002b_sliding/` | ✅ | 200/200条, 98MB H5, 534.6min, 对照组 |
| D_syn 002b VLM 标注 | `data/vlaw/labeled/synthetic_iter1_002b_sliding/` | ✅ | 200条, p_yes max=0.531, D_syn+(α=0.4)=7, D_syn+(α=0.8)=0 |
| D_syn 002a (aligned) | `data/vlaw/synthetic/iter1_002a_aligned/` | 🔄 | 180/200条 (batch100+batch150+batch200 H5), 稀疏采样实验组, ETA ~1h |
| **评估集 (固定)** | `data/vlaw/encoded/eval_fixed/eval_set.h5` | ✅ | **15条** (5 demo + 10 rollout), 永不训练 |
| 非活跃任务数据 | `data/vlaw/deferred/` | 💤 | PickCube+StackCube |

> 测试数据 (test_*, regression_*, pld_retest*) 已删除 (~127MB)。

---

## GPU 状态

| GPU | 分配 | VRAM Used | 状态 |
|-----|------|-----------|------|
| 0 | **空闲** | 33 MiB / 24564 MiB | 🟢 **可用** (BC Scaling V2 已完成) |
| 1 | WM 05v2 (T-EXP-WM-05v2) | 12882 MiB / 24564 MiB | 🔴 训练中 (tmux `wm_05v2`, ckpt-1100/2000) |
| 2 | WM 05v2 | 12802 MiB / 24564 MiB | 🔴 同上 |
| 3 | VLM Rank消融 (T-EXP-VLM-02) | 12090 MiB / 24564 MiB | 🔴 r=32 训练中 (tmux `vlm_ablation`) |
| 4 | VLM Rank消融 | 11962 MiB / 24564 MiB | 🔴 同上 |
| 5 | Imagination (T-IMAGINATION-002a) | 6051 MiB / 24564 MiB | 🔴 运行中 (tmux `imag_002a`, 180/200, ETA ~1h) |
| 6 | VLM 步数消融 (T-EXP-VLM-03) | 11676 MiB / 24564 MiB | 🔴 100步训练中 (tmux `vlm_steps_ablation`) |
| 7 | WM 05v2 | 12776 MiB / 24564 MiB | 🔴 同上 |
| 8 | WM 05v2 | 12776 MiB / 24564 MiB | 🔴 同上 |
| 9 | **空闲** | 16 MiB / 24564 MiB | 🟢 **可用** (BC Scaling V2 已完成) |

> **8/10 GPU 被占用**: WM-05v2 (1,2,7,8) + VLM-02 (3,4) + Imagination (5) + VLM-03 (6). **GPU 0, 9 空闲**. (截至 2026-03-02 21:25)

---

## 下一步任务

> **推进计划详见 [VLAW_NEXT_STEPS.md](VLAW_NEXT_STEPS.md)** | **评测汇总见 [`docs/vlaw/baselines_and_evaluation.md`](../docs/vlaw/baselines_and_evaluation.md)**

### 已完成 (02-24 ~ 02-28)

- ✅ Phase 1.5: V1-V6 全链路验证 (02-28 02:00)
- ✅ Phase 1.5b: V1.1 视频验证 + V1.2 wandb 集成
- ✅ WM iter1: 2000 steps, PSNR=23.34 > 18 PASS
- ✅ VLM 16帧: 200 steps, acc=0.824, FP=3.7%
- ✅ Track B1-B2: Imagination 50条 + VLM 标注 (vlm=0; D_real FP=0%)
- ✅ WM iter1 评估复现: PSNR=23.40 (vs pretrained 23.39), SSIM=0.7913, LPIPS=0.1200, 与训练时报告一致
- ✅ VLM 16帧 LoRA 评估复现: ROC-AUC=0.8084, Acc=84.7%, FP@Youden=1.8%, FP@α=0.8=0.0%
- ✅ ctrl_world 从 submodule 转为普通目录 (commit 5e16d60)
- ✅ 转换后重新验证: WM ManiSkill verify PSNR=18.26 ✅ | V3 Imagination ✅ | V4 VLM labeling ✅ | V5 Policy dry_run ✅ | V6 Eval ✅
- ✅ Pretrained Policy 验证: success_once=74-88% (avg~80%), 修复零动作padding bug (pred_horizon=8+obs_horizon=2→只产出7个有效action, 不应pad零)
- ✅ WM 4000步消融 (T-EXP-WM-01): 4000/4000 steps, final loss~0.02, 4 checkpoints@1000/2000/3000/4000 (各9.3GB), 训练~4h (GPU 0-3)
  - **评估结果**: ablation-2000 PSNR=24.11/SSIM=0.841/LPIPS=0.098; ablation-4000 PSNR=24.11/SSIM=0.818/LPIPS=0.107
  - **结论**: 4000步无额外收益(略有过拟合), 2000步已是最佳。ablation-2k优于iter1-2k(PSNR+0.71)可能因数据混合不同
- ✅ VLM 4帧消融 (T-EXP-VLM-01): 200/200 steps, loss=6.43, **acc=0.706, FP_rate=11.1%** (TP=0,FP=3,TN=24,FN=7) — 对比16帧 acc=0.824/FP=3.7%，4帧明显差（符合预期）
- ✅ Imagination 200条完成 (T-IMAGINATION-001): 200/200, 0失败, 582min total, iter1 WM + real policy
- ✅ VLM 8帧消融 (T-EXP-VLM-01-8F): 200/200 steps, loss=6.76, **acc=0.735, FP=18.5%** (TP=3,FP=5,TN=22,FN=4) — 介于4帧和16帧之间
- ✅ T-VLM-LABEL-001: 200条合成轨迹 VLM 标注, vlm=1: 0/200 (p_yes max=0.27 << α=0.8)
- ✅ T-POLICY-001: Weighted FM 策略更新, 2000步, loss 1.6→0.279, ckpt: policy/iter1/policy_iter1.pt
- ✅ T-POLICY-FIX-EMA: `_save_checkpoint()` 添加 ema_agent 提取逻辑, 重新保存 iter1 ckpt
- ✅ T-EVAL-ITER1-001: **Iter-1 策略严重退化** — success_once=17.2% (baseline 78.1%), -60.9% abs
  - EMA ckpt 保存格式 bug 已修复 (10.9%→17.2%), 但核心问题是微调本身导致退化
  - 可能原因: lr过高(1e-5), 步数过多(2000), D_syn无效(vlm=0), EMA衰减率过快, 缺少demo共训练
  - 详见 results/vlaw/iter1_eval_report.md

- ✅ **T-EXP-VLM-04: VLM use_video_format A/B 对比** — video AUC=0.83 >> multi-image 0.72, Youden 最优 α=0.40 (TP=70.6%, FP=1.4%), 合成轨迹 α=0.40 下仍 D_syn+=0 (p_yes max=0.27)
- ✅ **T-DIAG-SYN-001~003 + A/B 对比 全部完成**: 根因确认为 WM 合成质量 (非 VLM 校准问题)
- ✅ **BUG-019 发现并修复**: `load_initial_frames()` 使用 `torch.randn` 随机噪声作为初始 latent → 修复为真实帧 VAE 编码 `latent_concat[0]`。3/3 验证通过，解码帧质量正常。**旧 200 条合成数据无效需重新生成。**
- ✅ **WM 自回归 Rollout 模糊调研完成**: 官方 Ctrl-World 通过第一帧锚定 (`history_idx=[0,0,-12,-9,-6,-3]`) + `num_history=6` 抑制漂移。我们实现缺少第一帧锚定且 num_history=4 较小，需对齐 (**T-WM-ALIGN-HISTORY**)
  - 详见 [`docs/vlaw/wm_autoregressive_blurring_research.md`](../docs/vlaw/wm_autoregressive_blurring_research.md)
- ✅ **T-WM-ALIGN-HISTORY 完成**: 3 文件 4 处修改，对齐官方 Ctrl-World 列表式稀疏采样 history buffer
  - `ctrl_world/config.py`: `num_history=4→6`
  - `scripts/vlaw/run/run_imagination_iter1.py`: `num_history=4→6`
  - `rlft/vlaw/world_model/imagination_env.py`: 列表式 latent_history/action_history + `history_idx=[0,0,-12,-9,-6,-3]` + 列表式追加更新 + policy 引用更新
- ✅ **T-IMAGINATION-002b 完成**: 200/200 条 sliding window 对照组, 534.6min, 98MB H5, data: `iter1_002b_sliding/`
- 🔄 **T-IMAGINATION-002a 进行中**: 稀疏采样实验组, GPU 5, tmux `imag_002a`, 180/200条, ETA ~1h
- ✅ **T-VLM-LABEL-002b 完成**: 200条 sliding window 合成轨迹 VLM 标注, p_yes max=0.531 (旧噪声=0.27), D_syn+(α=0.4)=7, D_syn+(α=0.8)=0
  - **关键发现**: BUG-019 修复后 p_yes 从 0.27 → 0.531 (2倍提升), 首次在 α=0.4 下产出 D_syn+=7 条
- ✅ **T-EXP-VLM-02 r=8 完成**: acc=0.794, FP=0%, ckpt: `ablation_lora_r8/`
- 🔄 **T-EXP-VLM-02 r=32 训练中**: GPU 3,4, tmux `vlm_ablation`, 启动于 18:13
- ✅ **T-BC-SCALING-V2 全部完成**: 6/6组 20K步从零训练: 25d=0.02, 50d=0.04, 100d=0.10, 200d=0.16, 400d=0.32, 669d=0.40 (success_once). 结果: `results/vlaw/bc_scaling_v2/scaling_results.jsonl`
  - **669 demos 达到 40% success_once** → Go/No-Go 通过 (>30%)，scaling curve 有信息量
  - **scaling 符合预期**: 数据量越多 success_once 越高，但与预训练 baseline 80% 仍有差距 (20K 步仅为基线 1M 步的 2%)
- ✅ **T-MBRL-ENV 完成**: ImaginationRLEnv 实现 + 40/40 pytest 通过 (`rlft/vlaw/world_model/imagination_rl_env.py`, 37KB)
- 🔄 **T-EXP-VLM-03 运行中**: VLM 步数消融 100步, GPU 6, tmux `vlm_steps_ablation`, ckpt: `ablation_100steps/`
- ✅ **T-EXP-WM-05 完成**: WM 最优步数搜索, GPU 0-3, 2000/2000步, loss=0.0268, ckpt@500/1000/1500/2000
- ⚠️ **T-EXP-WM-05-EVAL 完成但 ADR-018 降级**: 配置混淆 (num_frames=15, reencode 数据 vs iter1 的 num_frames=5, demos), 结论不可推广。v2 训练中
- ✅ **T-BC-SCALING 完成**: 6/6组全部跑完, success_once≈0% (25d=1%, 其余=0%). 5000步从零训练完全不够 (基线用1M步). 需重新设计
- ✅ **T-DATA-CLEANUP Phase 1**: eval_fixed (15条固定评估集) + reencode清理 + 标准评估脚本 eval_wm_standard.py
- ✅ **数据审计报告**: docs/vlaw/data_audit_and_reorganization_proposal.md — 发现 reencode 110/160条无效, 3组WM实验不可比
- 🔄 **T-EXP-WM-05v2 训练中**: WM 最优步数修复版, GPU 1,2,7,8, tmux `wm_05v2`, ckpt-1100/2000, num_frames=5, demos, 20个ckpt
- 🔄 **T-EXP-VLM-02 进行中**: LoRA rank 消融 r=8✅/r=32🔄/r=64⬜, GPU 3,4, tmux `vlm_ablation`
- ✅ **T-RL-BASELINE-AWSC/PLD/DSRL**: 跳过，sweep 数据已有 (ADR-017)

### ⚠️ Iter-1 关键发现 (2026-03-01)

1. **D_syn VLM=0**: WM合成200条轨迹中无一通过 VLM 过滤 (p_yes max=0.27 << α=0.8)，说明当前 WM 合成质量不足以产生 VLM 可识别的成功轨迹
2. **策略灾难性遗忘**: 2000步 Weighted FM 微调导致 success_once 从 78.1% → 17.2%，可能因 lr=1e-5 过高 + 无 demo replay 保护
3. **EMA save format bug**: `_save_checkpoint()` 未提取 `ema_agent` 键，修复后 10.9%→17.2%

### 🔍 阶段 A 诊断结论 (2026-03-02 00:15, **最终版**)

**T-DIAG-SYN-001** ✅: 8 条合成轨迹关键帧解码 → `results/vlaw/dsyn_diagnosis_frames/synthetic/` (128 PNG)
**T-DIAG-SYN-002** ✅: 5 成功 + 5 失败真实轨迹关键帧 → `results/vlaw/dsyn_diagnosis_frames/real_{success,failure}/` (110 PNG)
**T-DIAG-SYN-003** ✅: VLM 交叉验证 120 条真实轨迹 (multi-image AUC=0.72) → `results/vlaw/dsyn_diagnosis_vlm_crossval.json`
**T-EXP-VLM-04** ✅: VLM use_video_format A/B 对比 (video AUC=0.83) → `results/vlaw/dsyn_diagnosis_vlm_crossval_video_mode.json`

**⚠️ use_video_format A/B 对比 (ADR-015)**:
之前交叉验证误用 multi-image 模式 (use_video_format=False)，而 VLM 在 video 模式 (True) 下判别力更强：

| 指标 | Video Mode (True) ✅ | Multi-Image (False) |
|------|---------------------|---------------------|
| p_yes(success) | **0.5348** ± 0.2699 | 0.6993 ± 0.0909 |
| p_yes(failure) | **0.1601** ± 0.1114 | 0.4955 ± 0.2407 |
| AUC | **0.8306** | 0.7234 |
| FP@0.8 | **0/69 (0%)** | 3/69 (4.3%) |
| Youden最优 α | **0.40** (J=0.691) | — |
| TP@最优α | **36/51 (70.6%)** | — |
| FP@最优α | **1/69 (1.4%)** | — |

**修正后的诊断结论 (T-EXP-VLM-04 ✅)**:
1. **VLM 在 video 模式下表现良好** (AUC=0.83)，之前的 AUC=0.81 评估结果可靠
2. **α=0.8 过于激进** — Youden 最优 α≈**0.40**，此时 TP=70.6%，FP=1.4%
3. **VLAWRewardConfig 默认 `use_video_format=True` 是正确的**，不需要修改
4. **D_syn+=0 的根因已定位: BUG-019** — 初始 latent 使用 `torch.randn` 随机噪声而非真实帧 VAE 编码，导致所有 200 条合成轨迹从纯噪声起步
5. **BUG-019 已修复并验证** — 修复后 3/3 条轨迹帧质量正常（机械臂+peg 清晰可见），用户目视确认 OK
6. **旧 200 条合成数据无效，需先对齐官方 history 构建方式，再用修复后脚本重新生成** (`T-WM-ALIGN-HISTORY` → `T-IMAGINATION-002`)
7. **之前交叉验证脚本误用 multi-image 模式** (AUC=0.72)，video 模式 AUC=0.83 显著更优

> **结果文件**: `results/vlaw/dsyn_diagnosis_vlm_crossval_video_mode.json` (video 模式 120 条交叉验证)
> **下一步**: 对齐官方 history buffer 构建 (T-WM-ALIGN-HISTORY) → 重新生成 200 条合成轨迹 (T-IMAGINATION-002) → 重新 VLM 标注 → 评估 D_syn+ 数量

### 🔄 策略方案转向 (2026-03-01, ADR-014)

> Weighted FM 在小模型+小数据场景下灾难性遗忘已被验证。**策略微调方案切换至 AWSC + WM 数据增强**。
> 详见 [`docs/vlaw/policy_finetuning_strategy_report.md`](../docs/vlaw/policy_finetuning_strategy_report.md)

**新执行计划**:
1. ~~**T-RL-BASELINE-AWSC**~~: ✅ 已有 sweep 数据 (aggressive=91%), 无需重训 (ADR-017)
2. ~~**T-RL-BASELINE-PLD/DSRL**~~: ✅ 已有 sweep 数据, 无需重训 (ADR-017)
3. **AWSC + D_syn 增强**: D_syn+ 加入 demo buffer，验证 WM 数据增强价值
4. **WM 步数优化 (T-EXP-WM-05, ADR-013)**: 减少步数+增加 eval 频率，缩短迭代周期

### 待推进

#### 主线: Online RL Baselines → WM+VLM 增强

| task_id | 任务 | owner | 依赖 | 状态 |
|---------|------|-------|------|------|
| **T-EXP-VLM-04** | **VLM video vs images A/B 对比** | Reward | — | ✅ video AUC=0.83 >> 0.72 |
| **T-WM-ALIGN-HISTORY** | **对齐官方 history buffer (列表式+稀疏采样+第一帧锚定+num_history=6)** | Imagination | 调研✅ | ✅ 3文件4处修改 |
| **T-IMAGINATION-002b** | **sliding window 对照组 200 条** | Imagination | BUG-019 fix ✅ | ✅ 200/200, 534.6min |
| **T-IMAGINATION-002a** | **对齐后稀疏采样 200 条** | Imagination | T-WM-ALIGN-HISTORY ✅ | 🔄 180/200, GPU 5, ETA ~1h |
| **T-VLM-LABEL-002b** | **VLM 标注 002b (sliding window)** | Reward | T-IMAGINATION-002b ✅ | ✅ p_yes max=0.531, D_syn+(α=0.4)=7 |
| **T-WM-ALIGN-ABLATION** | **消融对比: 对齐 vs 当前 (帧质量+VLM p_yes+D_syn+)** | Eval | 002a+002b ✅ | ⬜ (等 002a 完成后标注) |
| **T-VLM-LABEL-002a** | **VLM 标注 002a (aligned)** | Reward | T-IMAGINATION-002a 🔄 | ⬜ |
| ~~T-RL-BASELINE-AWSC~~ | ~~AWSC baseline~~ | — | — | ✅ sweep 已有: aggressive=91% (ADR-017) |
| ~~T-RL-BASELINE-PLD~~ | ~~PLD-SAC baseline~~ | — | — | ✅ sweep 已有: conservative=81% (ADR-017) |
| ~~T-RL-BASELINE-DSRL~~ | ~~DSRL-SAC baseline~~ | — | — | ✅ sweep 已有 (ADR-017) |
| T-AWSC-WM-AUGMENT | AWSC + D_syn demo buffer 增强 | Policy | D_syn+ > 0 | ⬜ |
| T-MBRL-ENV | WM+VLM 包装为 RL 环境接口 | Imagination | — | ✅ ImaginationRLEnv 40/40 pytest |
| T-WM-ITER2-001 | Iter-2 (AWSC + WM 迭代) | 全部 | D_syn+ > 0 | ⬜ |

#### Iter-1 已完成 (存档)

| task_id | 任务 | owner | 依赖 | 状态 |
|---------|------|-------|------|------|
| T-POLICY-FIX | Policy 视觉 obs 适配 (ADR-009) | Policy | — | ✅ |
| T-IMAGINATION-001 | 合成轨迹 (iter1 WM, 200条) | Imagination | WM ✅ | ✅ 200/200 |
| T-VLM-LABEL-001 | VLM 标注合成轨迹 | Reward | T-IMAGINATION-001 ✅ | ✅ vlm=0/200 |
| T-POLICY-001 | Weighted FM 策略更新 | Policy | — | ✅ (2000步, 但退化) |
| T-EVAL-ITER1-001 | Iter1 策略评估 | Eval | T-POLICY-001 ✅ | ✅ 17.2% (退化) |
| T-POLICY-FIX-EMA | EMA ckpt 保存格式修复 | Eval | — | ✅ |

#### 支线: WM/VLM 配置消融 (空闲 GPU 并行)

| task_id | 实验 | owner | GPU | 状态 |
|---------|------|-------|-----|------|
| T-EXP-WM-01 | WM 4000步 vs 2000步 | WM | 0-3 | ✅ 完成: PSNR=24.11(2k/4k皆同), 4000步无额外收益 |
| **T-EXP-WM-05** | **WM 最优步数搜索 (↓步数+↑Eval频率)** | WM | 0-3 | ✅ 完成: 2000步, loss=0.0268, ckpt@500/1000/1500/2000 |
| **T-EXP-WM-05-EVAL** | **WM 最优步数评估 (ADR-018)** | WM | 0 | ✅ 完成: **1000步最优** PSNR=25.80, SSIM=0.891, LPIPS=0.056 |
| T-EXP-WM-02 | WM 混合 D_syn 训练 | WM | 0-3 | ⬜ |
| T-EXP-WM-03 | WM num_history=4 vs 1 | WM | 0-3 | ⬜ |
| T-EXP-WM-04 | WM 学习率消融 (5e-6/2e-5) | WM | 0-3 | ⬜ |
| T-EXP-VLM-01 | VLM 帧数消融 (4/8/32帧) | Reward | 6-7 | ✅ 4帧/8帧/16帧全完成 |
| T-EXP-VLM-02 | VLM LoRA rank (r=8/32/64) | Reward | 3,4 | � r=8 ✅ (acc=0.794, FP=0%), r=32 训练中 (tmux `vlm_ablation`) |
| T-EXP-VLM-03 | VLM 步数消融 (100/400/800) | Reward | 6 | 🔄 100步训练中 (tmux `vlm_steps_ablation`), ckpt: `ablation_100steps/` |
| T-EXP-VLM-04 | VLM video vs images 模式 | Reward | 6-7 | ✅ video AUC=0.83>>0.72, Youden α=0.40, 确认 WM 为瓶颈 |

> **T-EXP-WM-05 说明 (ADR-013)**: 已知扩大步数无收益，但尚未探索减少步数是否能更早达到最优点。将 validation_steps 从 500 降至 100，绘制完整 step-PSNR 曲线。若 500-1000 步已足够，WM 训练时间从 ~2h 缩短至 ~0.5-1h。

> 详细历史操作见 `logs/vlaw/` 中的 Agent 结果文件。
