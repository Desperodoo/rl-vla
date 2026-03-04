# VLAW 任务注册表 (Task Registry)

> **用途**: task_id → Agent → result 文件 → 关键产物的追溯映射。
> **规则**: Coordinator 在每个任务完成后更新本表。vlaw-status.md 只保留概述。
> **前身**: `docs/vlaw/archive/MEMORY_INDEX.md` (02-26 归档版)

---

## 已完成任务

### Phase 0 — 数据审计与修复 (02-26)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-AUDIT-001 | Eval | `logs/vlaw/Eval-Agent-result-20260226_221642.md` | `logs/vlaw/data_audit_report.{md,json}` | 9文件/340轨迹扫描, 5异常 |
| T-DATA-FIX-001 | Coordinator | — (直接操作) | 3目录5异常文件已清除 | 剩余全部 (4,48,24) 正确 |

### Phase 1 — 基线报告 (02-26)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-EVAL-BASELINE-001 | Eval | `logs/vlaw/Eval-Agent-result-20260226_172239.md` | `results/vlaw/pld_eval_baseline_20ep.json` | success_once=95%, success_at_end=75% |
| T-WM-BASELINE-001 | WM | `logs/vlaw/WM-Agent-result-20260226_223747.md` | `results/vlaw/wm_baseline_report.md` | pretrained H20 PSNR=22.35 |
| T-VLM-BASELINE-001 | Reward | `logs/vlaw/Reward-Agent-result-20260226_225815.md` | `results/vlaw/vlm_baseline_report.md` | ZS AUC=0.59, LoRA AUC=0.62 |

### 数据收集 (02-26)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-DATA-LIFT-001 | Data | `logs/vlaw/Data-Agent-result-20260226_174404.md` | `data/vlaw/rollouts/iter1_highsuc/` + encoded | 50条, success=70% |
| T-DATA-LIFT-002 | Data | `logs/vlaw/Data-Agent-result-20260226_202856.md` | `data/vlaw/rollouts/iter1_lift_inc20/` + encoded | 40条, success=30% |
| T-REWARD-REAL-001 | Reward | `logs/vlaw/Reward-Agent-result-20260226_205054.md` | `data/vlaw/labeled/iter1_lift_only/` | n=160, vlm_succ=0% |
| T-WM-COMP-001 | WM | `logs/vlaw/WM-Agent-result-20260226_165343.md` | `logs/vlaw/wm_comparison_frames/` | pre=23.07 / 8k=22.51 / 10k=22.06 |

### Phase 2 — Iter 1 训练 (02-27 ~ 02-28)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-WM-ITER1-001 | WM | `logs/vlaw/WM-Agent-result-20260228_035423.md` | `checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt` (4.4GB) | PSNR=23.34, 2000步/2h |
| T-VLM-16F-001 | Reward | `logs/vlaw/Reward-Agent-result-20260228_034302.md` | `checkpoints/vlaw/reward_model/lora_iter1_16frame/` (23MB) | acc=0.824, FP=3.7%, loss 18.7→6.8 |

### Track B — Imagination & 标注 (02-28)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-IMAGINATION-B1 | Imagination | `logs/vlaw/Imagination-Agent-result-20260228_125350.md` | `data/vlaw/synthetic/iter1_pretrained/` | 50/200成功, 190 cuda:0 error |
| T-VLM-LABEL-B2 | Reward | `logs/vlaw/Reward-Agent-result-20260228_125352.md` | `data/vlaw/labeled/synthetic_iter1_pretrained/` | vlm_reward 全=0 |
| T-VLM-LABEL-REAL-16F | Reward | `logs/vlaw/Reward-Agent-result-20260228_140547.md` | `data/vlaw/labeled/iter1_16frame_lora/` | 210条, FP=0% |

### 评估复现 (02-28)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-WM-EVAL-REPRODUCE | WM | `logs/vlaw/WM-Agent-result-20260228_180924.md` | `results/vlaw/wm_iter1_eval_report.md` | PSNR=23.40 vs pretrained 23.39 |
| T-VLM-EVAL-REPRODUCE | Reward | `logs/vlaw/Reward-Agent-result-20260228_181745.md` | `results/vlaw/vlm_16frame_comparison.md` | AUC=0.8084, Acc=84.7%, FP@0.8=0% |

### Policy 适配 & Imagination 修复 (03-01)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-POLICY-FIX | Policy | `logs/vlaw/Policy-Agent-result-20260301_*.md` | UNet(64,128,256)/64 + PlainConv 视觉编码器 | 10步 loss=1.642, 0 missing keys |
| T-IMAGINATION-FIX | Imagination | `logs/vlaw/Imagination-Agent-result-20260301_*.md` | 3 bug 修复 (PlainConv+API+obs) | mock 5/5 ✅, real 5/5 ✅ |

### WM/VLM 消融实验 (03-01)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-EXP-WM-01 | WM | `logs/vlaw/WM-Agent-result-20260301_122649.md` | `checkpoints/vlaw/world_model/ablation_4000steps/` (4×9.3GB) | PSNR=24.11 (2k=4k), SSIM=0.84/0.82 |
| T-EXP-WM-01-EVAL | WM | `logs/vlaw/WM-Agent-result-20260301_122649.md` | `results/vlaw/wm_ablation_4000_eval_report.md` | 4000步无额外收益 |
| T-EXP-VLM-01-4F | Reward | `logs/vlaw/Reward-Agent-result-20260301_122648.md` | `checkpoints/vlaw/reward_model/ablation_4frame/` (23MB) | acc=0.706, FP=11.1% |
| T-EXP-VLM-01-8F | Reward | `logs/vlaw/Reward-Agent-result-20260301_141412.md` | `checkpoints/vlaw/reward_model/ablation_8frame/` (23MB) | acc=0.735, FP=18.5%, ROC-AUC=0.8269 |

### Iter-1 完整迭代 (03-01)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-IMAGINATION-001 | Imagination | `logs/vlaw/Imagination-Agent-result-20260301_023612.md` | `data/vlaw/synthetic/iter1_wm_real/` (99MB) | 200/200 完成, 0失败, 582min |
| T-VLM-LABEL-001 | Reward | `logs/vlaw/Reward-Agent-result-20260301_141412.md` | `data/vlaw/labeled/synthetic_iter1_wm_real/` | 200条, vlm=1: 0/200, p_yes max=0.27 |
| T-POLICY-001 | Policy | `logs/vlaw/Policy-Agent-result-20260301_144422.md` | `checkpoints/vlaw/policy/iter1/policy_iter1.pt` | 2000步, loss 1.6→0.279, wandb: vlaw_policy_iter1 |
| T-POLICY-FIX-EMA | Eval | `logs/vlaw/Eval-Agent-result-20260301_154055.md` | `rlft/vlaw/policy/policy_updater.py` 修复 | _save_checkpoint 添加 ema_agent 提取 |
| T-EVAL-ITER1-001 | Eval | `logs/vlaw/Eval-Agent-result-20260301_154055.md` | `results/vlaw/iter1_eval_report.md` | **⚠️ baseline=78.1% → iter-1=17.2% (-60.9%)** |

### D_syn+ = 0 诊断 + VLM A/B 对比 (03-01 ~ 03-02)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-DIAG-SYN-001 | Imagination | `logs/vlaw/Imagination-Agent-result-20260301_203746.md` | `results/vlaw/dsyn_diagnosis_frames/synthetic/` (128 PNG) | 8条合成轨迹×16帧 |
| T-DIAG-SYN-002 | Data | `logs/vlaw/Data-Agent-result-20260301_203747.md` | `results/vlaw/dsyn_diagnosis_frames/real_{success,failure}/` (110 PNG) | 5成功+5失败真实轨迹×11帧 |
| T-DIAG-SYN-003 | Reward | `logs/vlaw/Reward-Agent-result-20260301_204402.md` | `results/vlaw/dsyn_diagnosis_vlm_crossval.json` | multi-image AUC=0.72, p_yes(succ)=0.70 |
| T-EXP-VLM-04 | Reward | `logs/vlaw/Reward-Agent-result-20260301_204402.md` | `results/vlaw/dsyn_diagnosis_vlm_crossval_video_mode.json` | **video AUC=0.83 >> multi-image 0.72**, Youden α=0.40 (TP=70.6%, FP=1.4%), 合成 p_yes max=0.27 < α=0.40 → D_syn+=0 |

> **诊断结论 (ADR-015)**: VLM 在 video 模式下表现良好 (AUC=0.83)，D_syn+=0 的根因确认为 BUG-019（初始 latent 使用随机噪声）而非 WM 训练不足或 VLM 校准问题。

### WM Initial Latent Bug 修复 (03-02)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-DIAG-WM-PIPELINE | Imagination | `logs/vlaw/Imagination-Agent-result-20260302_002833.md` | `scripts/vlaw/run/run_imagination_iter1.py` (BUG-019 修复) | 根因: `load_initial_frames()` 使用 `torch.randn` 而非真实帧 VAE 编码 |
| T-DIAG-WM-PIPELINE-VERIFY | Imagination | `logs/vlaw/Imagination-Agent-result-20260302_002833.md` | `data/vlaw/synthetic/iter1_fixtest3/` (3条), `results/vlaw/dsyn_diagnosis_frames/fixtest/` (解码帧) | 3/3 轨迹生成 OK, 帧质量正常 (机械臂+peg 清晰可见), 用户目视确认 |

### 数据清理 + BC Scaling + 支线实验 (03-02)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-DATA-CLEANUP | Data | `logs/vlaw/Data-Agent-result-20260302_*.md` | `data/vlaw/encoded/eval_fixed/eval_set.h5` (15条), `scripts/vlaw/eval/eval_wm_standard.py`, `data/vlaw/encoded/reencode_valid/` | Phase 1 完成: eval_fixed + reencode清理 + 标准评估脚本 |
| T-BC-SCALING | Policy | `logs/vlaw/bc_scaling_*.log` | `runs/` 下 6 个 wandb 离线运行 | 6/6组完成, 全部 success≈0% (5K步不够, 基线用1M步). **ADR-020: 重做20K步** |
| T-EXP-WM-05-EVAL | WM | `results/vlaw/wm_optimal_steps_eval/report.md` | 4个 ckpt 评估报告 | ❗ **ADR-018 降级**: v1 配置混淆 (num_frames=15, reencode), 结论不可推广 |

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-WM-ALIGN-HISTORY | Imagination | `logs/vlaw/Imagination-Agent-result-20260302_125948.md` | 3文件4处修改: `ctrl_world/config.py`, `scripts/vlaw/run/run_imagination_iter1.py`, `rlft/vlaw/world_model/imagination_env.py` | 列表式 latent/action history + 稀疏采样 `[0,0,-12,-9,-6,-3]` + num_history=4→6 |
| T-IMAGINATION-002b | Imagination | `logs/vlaw/Imagination-Agent-result-20260302_023601.md` | `data/vlaw/synthetic/iter1_002b_sliding/` (98MB H5, 200条) | 200/200条, sliding window 对照组, 534.6min, 0失败 |
| T-EXP-WM-05 | WM | `logs/vlaw/WM-Agent-result-20260302_*.md` | `checkpoints/vlaw/world_model/ablation_optimal_steps/` (ckpt@500/1000/1500/2000) | 2000步完成, loss=0.0268, 4个 checkpoint |
| T-EXP-WM-05-EVAL | WM | `logs/vlaw/WM-Agent-result-20260302_*.md` | `results/vlaw/wm_optimal_steps_eval/report.md` | **1000步最优性价比** PSNR=25.80/SSIM=0.891/LPIPS=0.056, 2000步PSNR=25.87仅微弱优势, ADR-018: 后续 WM 训练可缩短至 1000步 |

### 消融实验 + BC 飞轮验证 (03-02 ~ 03-03)

| task_id | owner | result_md | 关键产物 | 指标 |
|---------|-------|-----------|---------|------|
| T-IMAGINATION-002a | Imagination | `logs/vlaw/Imagination-Agent-result-20260302_*.md` | `data/vlaw/synthetic/iter1_002a_aligned/` (200条) | 200/200 aligned 稀疏采样实验组, GPU 5 |
| T-VLM-LABEL-002a | Reward | `logs/vlaw/Reward-Agent-result-20260303_000337.md` | `data/vlaw/labeled/synthetic_iter1_002a_aligned/` | 200条, p_yes max=0.500, mean=0.179, D_syn+(α=0.4)=6, D_syn+(α=0.8)=0 |
| T-VLM-LABEL-002b | Reward | `logs/vlaw/Reward-Agent-result-20260302_143121.md` | `data/vlaw/labeled/synthetic_iter1_002b_sliding/` | 200条, p_yes max=0.531, mean=0.153, D_syn+(α=0.4)=7, D_syn+(α=0.8)=0. 首次 D_syn+>0 |
| T-WM-ALIGN-ABLATION | Eval | — (直接结论) | ADR-021 | 消融结论: aligned(6)≈sliding(7), 无显著差异 (Δ=1), WM 质量是主瓶颈, 保留 sliding 做法 |
| T-BC-SCALING-V2 | Policy | `logs/vlaw/bc_scaling_v2_*.log` | `runs/bc_scaling_v2_*` (6组), `results/vlaw/bc_scaling_v2/scaling_results.jsonl` | 20K步 6/6组: 25d=0.02, 50d=0.04, 100d=0.10, 200d=0.16, 400d=0.32, 669d=0.40 |
| T-BC-FLYWHEEL-A | Policy | `logs/vlaw/Policy-Agent-result-20260303_002944.md` | `runs/bc_flywheel_a_*` (2组) | 100K步 BC 基线: 100d=0.10, 669d=0.54 (success_once) |
| T-DSYN-PLUS-PREP | Data | `logs/vlaw/Data-Agent-result-20260303_001952.md` | `data/vlaw/combined/flywheel_b_{100,669}demos/`, `results/vlaw/dsyn_plus_decoded_frames/` | 13条 D_syn+ (α=0.4, 002a:6+002b:7) 合并入 demo, 解码帧检查通过 |
| T-MBRL-ENV | Imagination | `logs/vlaw/Imagination-Agent-result-20260302_*.md` | `rlft/vlaw/world_model/imagination_rl_env.py` (37KB) | ImaginationRLEnv 实现 + 40/40 pytest 通过 |
| T-BC-FLYWHEEL-B | Policy | `results/vlaw/bc_flywheel_b/flywheel_b_results.jsonl` | `runs/bc_flywheel_b_{113,682}demos_100000steps_*` | 113t=0.34 (+240% vs A), 682t=0.48 (-11% vs A). D_syn+ 小数据场景极有价值 |
| T-EXP-WM-05v2 | WM | tmux `wm_05v2` (已结束) | `checkpoints/vlaw/world_model/optimal_steps_v2/` (20 ckpt @100-2000步) | 训练完成. 评估阻塞 (ADR-024: WM 任务等待 TIMING 解决) |
| T-EXP-VLM-02-r32 | Reward | tmux `vlm_ablation` (已结束) | `checkpoints/vlaw/reward_model/ablation_lora_r32/` | acc=0.85, FP=0%. 缺 AUC, 待 T-EXP-VLM-02-EVAL |
| T-EXP-VLM-02-r64 | Reward | tmux `vlm_ablation` (已结束) | `checkpoints/vlaw/reward_model/ablation_lora_r64/` | acc=0.85, FP=0%. 缺 AUC, 待 T-EXP-VLM-02-EVAL |
| T-EXP-VLM-03-100 | Reward | `results/vlaw/vlm_steps_ablation/eval_100steps.json` | `checkpoints/vlaw/reward_model/ablation_steps_100/` | ⚠️ 异常: AUC=0.315, p_yes≈10^-4. 极可能脚本 bug → T-EXP-VLM-03-BUG |
| T-EXP-VLM-03-400 | Reward | `results/vlaw/vlm_steps_ablation/eval_400steps.json` | `checkpoints/vlaw/reward_model/ablation_steps_400/` | ⚠️ 异常: AUC=0.383, p_yes≈10^-4. 极可能脚本 bug → T-EXP-VLM-03-BUG |
| T-EXP-VLM-03-800 | Reward | `results/vlaw/vlm_steps_ablation/eval_800steps.json` | `checkpoints/vlaw/reward_model/ablation_steps_800/` | ⚠️ 异常: AUC=0.390, p_yes≈10^-4. 极可能脚本 bug → T-EXP-VLM-03-BUG |

---

## 进行中 / 待推进

### 阶段 A: D_syn+ = 0 诊断 ✅ 已完成

| task_id | owner | 状态 | 备注 |
|---------|-------|------|------|
| **T-DIAG-SYN-001** | Imagination | ✅ | 8 条合成轨迹关键帧解码 → 128 PNG |
| **T-DIAG-SYN-002** | Data | ✅ | 成功/失败各 5 条真实轨迹关键帧 → 110 PNG |
| **T-DIAG-SYN-003** | Reward | ✅ | VLM 交叉验证 120 条 (multi-image AUC=0.72) |
| **T-EXP-VLM-04** | Reward | ✅ | VLM video A/B 对比: AUC=0.83, Youden α=0.40, 确认 WM 为瓶颈 |
| **T-DIAG-SYN-REVIEW** | 人工+Imagination | ✅ | BUG-019 确认: 初始 latent 使用随机噪声, 已修复并验证 3/3 OK |

### 阶段 A 后续: 对齐官方 + 重新生成合成数据

| task_id | owner | 状态 | 备注 |
|---------|-------|------|------|
| **T-WM-ALIGN-HISTORY** | Imagination | ✅ | 3文件4处修改: 列表式+稀疏采样+第一帧锚定+num_history=6 |
| **T-IMAGINATION-002b** | Imagination | ✅ | 200/200条 sliding window 对照组, 534.6min, data: `iter1_002b_sliding/` |
| **T-VLM-LABEL-002b** | Reward | ✅ | 200条 VLM 标注: p_yes max=0.531, mean=0.153, D_syn+(α=0.4)=7, D_syn+(α=0.8)=0, 首次 D_syn+>0! |
| **T-IMAGINATION-002a** | Imagination | ✅ | 200/200条 稀疏采样实验组, GPU 5, data: `iter1_002a_aligned/` |
| **T-VLM-LABEL-002a** | Reward | ✅ | 200条 VLM 标注: p_yes max=0.500, mean=0.179, D_syn+(α=0.4)=6, D_syn+(α=0.8)=0. 消融结论: aligned(6) vs sliding(7) 无显著差异 |
| **T-WM-ALIGN-ABLATION** | Eval | ✅ | 消融结论 (ADR-021): aligned(6)≈sliding(7), Δ=1 未达 go 标准, 保留 sliding 做法, WM 质量是主瓶颈 |

### 阶段 B1: 纯 BC 数据飞轮验证

| task_id | owner | 状态 | 备注 |
|---------|-------|------|------|
| **T-BC-SCALING-V2** | Policy | ✅ | 20K步 6/6组完成: 25d=0.02...669d=0.40. Go/No-Go: 669d=40%>30% ✅ |
| **T-BC-FLYWHEEL-A** | Policy | ✅ | 100K步 BC 基线: 100d=0.10, 669d=0.54 (success_once) |
| **T-BC-FLYWHEEL-B** | Policy | ✅ | 113t=0.34 (+240% vs A), 682t=0.48 (-11% vs A). D_syn+ 小数据场景极有价值 |
| **T-BC-FLYWHEEL-EVAL** | Eval | ⬜ | A vs B 对比评估, Go/No-Go: B > A + 3% |

### 阶段 B2: Policy-in-the-Loop Imagination RL

| task_id | owner | 状态 | 备注 |
|---------|-------|------|------|
| **T-MBRL-ENV** | Imagination | ✅ | ImaginationRLEnv 实现 + 40/40 pytest, `rlft/vlaw/world_model/imagination_rl_env.py` |
| **T-MBRL-BC-FINETUNE** | Policy | ⬜ | 在 ImaginationRLEnv 中用 RLPD/DSRL/PLD 微调 BC 预训练策略 |
| **T-MBRL-EVAL** | Eval | ⬜ | 在 ManiSkill 真实环境评估，Go/No-Go: success_once ≥ 78% |

### 阶段 B3: 迭代 WM+VLM 共同改进

| task_id | owner | 状态 | 备注 |
|---------|-------|------|------|
| **T-ITER-LOOP-DESIGN** | Coordinator | ⬜ | 设计迭代循环: checkpoint 管理 / 数据混合策略 / 收敛监控 |
| **T-ITER-ROUND-2** | 全部 | ⬜ | 第 2 轮完整迭代 |
| **T-ITER-ROUND-3** | 全部 | ⬜ | 第 3 轮完整迭代 (至少 2-3 轮判断收敛性) |

### 支线实验

| task_id | owner | 状态 | 备注 |
|---------|-------|------|------|
| **T-EXP-WM-05** | WM | ✅ | 完成: 2000/2000步, loss=0.0268, ckpt@500/1000/1500/2000. ❗ ADR-018 降级 |
| **T-EXP-WM-05-EVAL** | WM | ✅ | 完成: 4个 ckpt 评估, 1000步最优 PSNR=25.80. ❗ 配置混淆 (num_frames=15, reencode) 导致结论不可推广 |
| **T-EXP-WM-05v2** | WM | ⏸️ | 训练完成 (20 ckpt @100-2000步), 评估阻塞 (ADR-024: WM 任务等待 TIMING 解决) |
| **T-EXP-VLM-02** | Reward | ✅ | r=8 (acc=0.794, FP=0%), r=32 (acc=0.85, FP=0%), r=64 (acc=0.85, FP=0%). r=32/r=64 缺 AUC → T-EXP-VLM-02-EVAL |
| T-EXP-WM-02~04 | WM | ⏸️ | BLOCKED (ADR-024): WM 扩展消融等待 TIMING 解决 |
| T-EXP-VLM-03 | Reward | ✅ | 100/400/800步全完成. ⚠️ 全部异常 (AUC=0.31-0.39, p_yes≈10^-4) → T-EXP-VLM-03-BUG |
| ~~T-EXP-VLM-04~~ | Reward | ✅ | 已完成: video AUC=0.83 >> images 0.72, ADR-015 |

