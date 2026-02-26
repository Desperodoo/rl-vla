# VLAW 复现项目 — 状态仪表盘

> **最后更新**: 2026-02-26 (T+4) | 详细日志见 [`work-logs/`](work-logs/) | 技术知识见 [`knowledge/`](knowledge/)

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
| **P2.2 WM Phase-A 训练** | 🔄 | 2026-02-26 | GPU 0-3 / **step 8000→10000 续训中** / ckpt-8000✅ / 3个bug已修复(stat.json路径/data路径/max_steps) / PID 411804 |
| P2.3 WM 验证 | ✅ | 2026-02-26 | **三模型对比完成** / pretrained 23.07dB vs ckpt-8000 22.51dB vs ckpt-10000 22.06dB / PSNR 全部>18 / SSIM: 0.8249→0.6239→0.5873 / LPIPS: 0.1158→0.1593→0.1777 / 对比图→logs/vlaw/wm_comparison_frames/ |
| P3.1 奖励模型实现 | ✅ | 2026-02-24 | reward_model.py / VRAM 17GB ✅ |
| P3.2 VLM Fine-tuning | ✅ | 2026-02-26 | **已完成!** LoRA r=16 / 200步 / GPU6 23.9GB / final→checkpoints/vlaw/reward_model/lora_iter1/final/ |
| P3.3 VLM FP率验证 | ✅ | 2026-02-26 | **FP=0.0% ✅ PASS** (<20%) / p_yes_max=0.107 (过于保守,待D_syn改善) / 结果→data/vlaw/labeled/iter1_lora/ |
| P4.1 State Predictor | ✅ | 2026-02-25 | **已训练** / 按任务分模型 / LiftPeg(obs=25,loss=0.0067) PickCube(obs=29,loss=0.0030) StackCube(obs=25,loss=0.0042) / ckpt: checkpoints/vlaw/state_predictor/{task}/state_predictor_iter1.pt |
| P4.2 Imagination 引擎 | ✅ | 2026-02-25 | imagination.py / dry_run✅ |
| P4.3 合成数据生成 | ✅ | — | **env.step() 版代码实现完成**（imagination_env.py, 1044行）；尚需 WM checkpoint 才能实际运行 |
| P5.1 Weighted FM Loss | ✅ | 2026-02-25 | compute_weighted_loss / policy_updater.py ✅ |
| P5.2 策略更新验证 | ⬜ | — | 等待 P4.3 完成 |
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
| Ctrl-World (Phase-A) | `checkpoints/vlaw/world_model/phase_a/` | 🔄 step8000/10000, 续训中 |
| Qwen3-VL-4B LoRA Iter1 | `checkpoints/vlaw/reward_model/lora_iter1/final/` | ✅ step200完成 (23.6MB adapter) |
| Qwen3-VL-4B-Instruct | `checkpoints/vlaw/reward_model/qwen_vl/` | ✅ 8.3GB |
| State Predictor | `checkpoints/vlaw/state_predictor/` | ⬜ 待训练 |
| Policy Iter 1 | `checkpoints/vlaw/policy/iter1/` | ⬜ 待训练 |

---

## 数据目录

| 数据 | 路径 | 状态 | 数量 |
|------|------|------|------|
| 演示 (原始) | `data/vlaw/demos/{task}/` | ✅ | 3任务×25条 |
| 演示 (VAE编码) | `data/vlaw/encoded/demos/{task}/` | ✅ | 3任务，326 train samples |
| 动作统计量 | `data/vlaw/meta_info/maniskill/stat.json` | ✅ | 7D p01/p99 |
| D_real Iter 1 (原始) | `data/vlaw/rollouts/iter1/` | ✅ | 3任务×50条，成功率: Lift=16% / Pick=0% / Stack=0% |
| D_real Iter 1 (VAE编码) | `data/vlaw/encoded/rollouts/iter1/` | ✅ | 3任务×50条 / latent(T,4,48,24) float16 |
| D_real Iter 1 (VLM标注 v1) | `data/vlaw/labeled/iter1/` | ✅ | 150条 / vlm_reward全0 / p_yes低 / success_once 旧语义 |
| D_real Iter 1 (VLM标注 v2) | `data/vlaw/labeled/iter1_v2/` | ✅ | 150条 / success_at_end 正确语义 / p_yes_max=0.148 / 连续奖励可用 |
| D_real iter1_highsuc (原始) | `data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1/` | ✅ | 新增正式数据 `LiftPegUpright-v1_real_1772098799.h5`：50条（35成功/15失败，成功率70.0%） |
| D_real iter1_highsuc (VAE编码) | `data/vlaw/encoded/rollouts/iter1_highsuc/LiftPegUpright-v1/` | ✅ | `LiftPegUpright-v1_real_1772098799.h5` 已编码完成（50条，latent `(T,4,32,16)` float16） |
| D_syn Iter 1 | `data/vlaw/synthetic/iter1/` | ⬜ | 目标: 500条/任务 |

---

## GPU 状态

| GPU | 分配 | 状态 |
|-----|------|------|
| 0-3 | WM Phase-A 续训 (step8000→10000) | 🔴 占用 ~23GB×4 |
| 4-5 | 空闲 | 🟢 空闲 |
| 6 | 空闲 (reward训练已完成) | 🟢 空闲 |
| 7 | 空闲 | 🟢 空闲 |
| 8-9 | 空闲 | 🟢 空闲 |

---

## 当前阻塞与下一步

| **待做** | **前置条件** | **可立即开始？** |
|------|---------|-----------|
| WM Phase-A 完成 (step 10000) | 正在运行 | ⏳ 约73min |
| WM 三模型验证 (PSNR/SSIM/LPIPS) | 已完成 | ✅ pretrained/ckpt8000/ckpt10000 = 23.07/22.51/22.06 dB，全部>18；关键帧图 9 张已输出 |
| LiftPegUpright 高成功率采集恢复 | 需进一步对齐 train_pld eval pipeline | ✅ 可立即开始（GPU 4-5），当前 10 条验证 success_rate=0.0% |
| Imagination 合成轨迹 (P4.3) | WM ckpt-10000 + VLM fine-tuned | ⏳ 等待 WM 完成 |
| **策略更新 Iter 1 (P5.2)** | P4.3 D_syn + VLM标注 | ❌ 等待 WM + D_syn |

**Refactor 试运行 (2026-02-26):**
- 已启动“边重构边推进项目”验证：新增 `.github/VLAW_EXECUTION_BOARD.md`、`.github/MEMORY_INDEX.md`、`.github/agents/RESULT_FILE_PROTOCOL.md`。
- 本轮在线验证任务：`T-DATA-PICK-001` 与 `T-DATA-STACK-001`（Data-Agent 并行执行），要求同时产出 `logs/vlaw/*-result-*.md` 与同名 `.json` 摘要，用真实任务验证重构可用性。

**flash_attn 状态**: 已安装（2.8.3）并完成 reward 训练脚本多卡改造验证（Accelerate dry-run 通过）。

**Reward Model 数据问题** (2026-02-26发现):
- iter1 标注数据 150条**全为负样本**（0条成功轨迹），导致模型只学到"拒绝"，FP=0但TP也=0
- 根本原因: iter1 rollout 用低成功率策略（成功率<16%，甚至0%），收集数据中缺少正样本
- 修复计划: 从 demo 数据（25条×3任务，100%成功）提取正样本构建平衡训练集后重新微调

**PLDSACPolicy 支持** (2026-02-26新增):
- `rlft/vlaw/data/collector.py` 已新增 `PLDSACPolicy` 类，支持自动检测并加载 PLD-SAC checkpoint
- 加载逻辑：检测 "agent" key → 自动切换为 PLD 模式，加载 PLDActor + base ShortCutFlow
- 注意：PLD 策略用 192×192 采集时成功率仅6%（训练时用 128×128 + FlattenRGBDObs），待对齐

**已修复 Bug**:
- BUG-013: WM resume 脚本 `data_stat_path` 为相对路径 → 已改为绝对路径
- BUG-014: WM dataset `dataset_root_path` 指向原始数据 → 改为 `data/vlaw/encoded`
- BUG-015: WM resume `max_train_steps=2000` 与 `initial_step=8000` 冲突 → 改为 `max_train_steps=10000`
- BUG-016: `train_ctrl_world.sh` 中 META_INFO 路径继承自 DATA_ROOT → 修复为独立绝对路径

**Agent 系统更新** (2026-02-26):
- `vlaw-coordinator.agent.md`: 完整重写，引入 §T/§D/§R 三规范；顶部明确 `⛔` 禁止 Coordinator 自行接管
- **Race condition 根本原因**: subagent 被要求"只返回 3 行摘要"时，父 Agent 框架在捕获文本前已关闭响应流 → "Agent completed with no output"
- **修复**: 所有 7 个 subagent 文件的"向 Coordinator 返回"规范已从"≤3行"改为"完整文本+文件双轨"
- **日志路径修复**: 所有 RESULT_FILE 从 `/tmp/vlaw-*` 改为 `/home/wjz/rl-vla/logs/vlaw/`（避免 VS Code 写入审批弹窗）
- **⛔ /tmp/ 禁止规则**: 所有 subagent 文件输出规范顶部新增明确禁止向 `/tmp/` 写入任何文件（含 `*_path.txt`、`current_result_file.txt` 等），防止 subagent 自行发挥

**任务恢复状态** (2026-02-26):
- **Task 2 / Data-Agent**: `collector.py` action chunking 修复步骤 1-4 已完成，`Step 5` 的 10 条验证已恢复并完成（参数：LiftPegUpright-v1 / num_envs=10 / obs_horizon=2 / act_steps=8 / 128×128 / GPU4）；首次运行出现 CUDA illegal memory，隔离 `CUDA_VISIBLE_DEVICES=4` 重试后成功，输出 `data/vlaw/rollouts/test_chunking_resume/LiftPegUpright-v1_real_1772096170.h5`（10条，success_rate=0.0%）；当前状态 `✅ 已完成`。
- **Task 2 / Data-Agent（续）**: 按 Coordinator 要求完成 Step3+ 验证：`rlft_ms3` 环境下先跑 10/10 再跑 20/20（参数：LiftPegUpright-v1 / obs_horizon=2 / act_steps=8 / 128×128 / GPU4）；输出 `LiftPegUpright-v1_real_1772096918.h5`（10条，success_rate=0.0%）与 `LiftPegUpright-v1_real_1772096962.h5`（20条，success_rate=0.0%）；当前状态 `✅ 已完成`。
- **Task 2 / Data-Agent（纠偏复测）**: 已按要求强制使用 PLD checkpoint `runs/pld_sweep_v3/pld_sac/ablate_temp_0.5__1771762688/checkpoints/best.pt` 在 `rlft_ms3` 环境复测（`CUDA_VISIBLE_DEVICES=4` + `--gpu_id 0`，参数：LiftPegUpright-v1 / obs_horizon=2 / act_steps=8 / 128×128）；10/10 输出 `LiftPegUpright-v1_real_1772097159.h5`（success_rate=0.0%），20/20 输出 `LiftPegUpright-v1_real_1772097206.h5`（success_rate=10.0%）。日志验真通过：两轮均出现“检测到 PLD-SAC checkpoint”，且均未出现“使用随机策略”；当前状态 `✅ 已完成`。
- **Task 2 / Data-Agent（Step4修复后复测）**: 按要求仅对 `rlft/vlaw/data/collector.py` 做最小修复，覆盖差异 #2/#3/#4（结束判定、success 统计、PLD base flow 加载参数）；随后在 `rlft_ms3` 环境执行 20/20 回归（`CUDA_VISIBLE_DEVICES=4`，同 checkpoint 与同任务配置），日志 `logs/vlaw/collector_post_fix_20of20_20260226_173626.log`，输出 `data/vlaw/rollouts/pld_retest_after_fix/LiftPegUpright-v1/LiftPegUpright-v1_real_1772098598.h5`，**success_rate=65.0% (13/20)**，相较修复前 10.0% 提升 +55.0 个百分点；当前状态 `✅ 已完成`。
- **Task 2 / Data-Agent（正式高成功率数据）**: 在 `rlft_ms3` 环境按正式参数执行 LiftPegUpright-v1 采集与编码（checkpoint=`runs/pld_sweep_v3/pld_sac/ablate_temp_0.5__1771762688/checkpoints/best.pt`，`obs_horizon=2`，`act_steps=8`，128×128，`CUDA_VISIBLE_DEVICES=4` + `--gpu_id 0`）；采集输出 `data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1/LiftPegUpright-v1_real_1772098799.h5`，**50条成功率 70.0% (35/50)**；编码输出 `data/vlaw/encoded/rollouts/iter1_highsuc/LiftPegUpright-v1/LiftPegUpright-v1_real_1772098799.h5`，`latent_concat` 规格 `(T,4,32,16)` `float16`（50条，total_frames=116）；日志分别为 `logs/vlaw/collector_iter1_highsuc_20260226_173946.log` 与 `logs/vlaw/pipeline_iter1_highsuc_single_20260226_174150.log`；当前状态 `✅ 已完成`。
- **Task 2 / Data-Agent（Step4 状态收口）**: 已完成状态文件补录与正式产物锚定：`rollout_success_rate=70.0%`、`num_trajs=50`；原始产物 `data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1/LiftPegUpright-v1_real_1772098799.h5`，编码产物 `data/vlaw/encoded/rollouts/iter1_highsuc/LiftPegUpright-v1/LiftPegUpright-v1_real_1772098799.h5`；当前状态 `✅ 已完成`。
- **Task 1 / WM-Agent**: 详细对比报告（pretrained vs ckpt-8000 vs ckpt-10000）已完成；当前状态 `✅ 完成`，结果文件：`logs/vlaw/WM-Agent-result-20260226_165343.md`。
- **Task 3 / Eval-Agent（同源评估）**: 已在 `rlft_ms3` 环境完成 checkpoint 同源评估（LiftPegUpright-v1, 20 episodes, `CUDA_VISIBLE_DEVICES=9`）；输出 `results/vlaw/pld_eval_baseline_20ep.json`，指标为 success_once=95.0%、success_at_end=75.0%、reward=2.3957、return=239.57。对比 collector 20/20=10%：checkpoint 显著更优，归因主要是 collector 端 rollout/config 与训练态存在不一致，而同源 pipeline 复现出 checkpoint 实际能力；当前状态 `✅ 已完成`。
