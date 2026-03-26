# VLAW 下一步推进计划

> **最后更新**: 2026-02-28 16:00 | **完整历史版**: [`docs/vlaw/archive/VLAW_NEXT_STEPS_full_0227.md`](../docs/vlaw/archive/VLAW_NEXT_STEPS_full_0227.md)
> **目标**: 最小验证 → 正式训练 + 并行 Policy Pipeline → 多轮迭代
> **核心原则**: ①先用最小实验验证全链路无故障；②利用 pretrained 模型并行构建 Policy Pipeline

---

## 当前状态（截至 2026-02-28）

### ✅ 已完成

| 阶段 | 关键结论 |
|------|---------|
| Phase 0: 数据审计 | 异常清除，剩余 4 文件全部 `(4,48,24)` |
| Phase 1A: WM 基线 | pretrained H20 PSNR=22.35/SSIM=0.79; Phase-A 退化 → ADR-007 |
| Phase 1B: VLM 基线 | 单帧 AUC=0.59 → 16帧 images AUC=0.82 → ADR-008 |
| Step 0: 数据扩充 | 235 条 / 4378 窗口 (re-encode highsuc+inc20) |
| Step 1d: env.step() 验证 | 5/5 成功，2 个 device mismatch bug 已修 |
| 基础设施 | DeepSpeed ZeRO-2 + ffmpeg + dtype 全部修复 |
| **Phase 1.5: 全链路验证** | **V1-V6 全部通过 (02-28 02:00)，正式训练已就绪** |

### Phase 1.5 验证详情 (02-28 完成)

| 验证项 | 结果 | 关键数据 |
|--------|------|---------|
| V1: WM mini-train | ✅ | DeepSpeed ZeRO-2, 4GPU, 40 micro-steps, ~18GB/GPU, ckpt 4.6GB |
| V2: VLM LoRA mini-train | ✅ | 4GPU Accelerate, 3 steps, loss 19.87→19.78, ckpt+推理 OK |
| V3: Imagination mini | ✅ | 5 traj, env.step() 模式, HDF5 格式 (latent 15×4×48×24) |
| V4: VLM labeling mini | ✅ | zero-shot 16帧 p_yes=[0.000013, 0.000075], pipeline OK |
| V5: Policy mini-train | ✅ | dry_run loss=2.14, real data loss 1.56→1.25, ckpt round-trip OK |
| V6: Policy mini-eval | ✅ | 5 ep × T=100, inference loop OK, 0% success (预期) |

### 可用资产

| 资产 | 路径 | 规模 |
|------|------|------|
| Demo (编码) | `encoded/demos/LiftPegUpright-v1/` | 25 条 |
| D_real iter1 (编码) | `encoded/rollouts/iter1/LiftPegUpright-v1/` | 50 条, 16% 成功 |
| D_real re-encode | `encoded/reencode_highsuc_inc20/LiftPegUpright-v1/` | 160 条 (5 HDF5) |
| 策略基线 | `results/vlaw/pld_eval_baseline_20ep.json` | success_at_end=75% |

---

## 三轨并行策略

```
██ Phase 1.5: 最小规模全链路验证 (优先级最高, ~2h) ██
  [V1] WM mini: 5 steps + ckpt + validate video
  [V2] VLM mini: 3 steps, 4GPU + ckpt + inference
  [V3] Imagination mini: pretrained WM → 5 条
  [V4] VLM labeling mini: zero-shot → 标注 5 条
  [V5] Policy mini: 10 steps, weighted FM
  [V6] Policy eval mini: 5 episodes
  ↓ 全部通过 → 启动正式训练

██ Track A: 正式训练 (Phase 1.5 后) ██
  [A1] WM 2000 steps, GPU 0-3, ~50h
  [A2] VLM LoRA 200 steps, GPU 4,5,7,8, ~1-2h

██ Track B: pretrained pipeline (与 Track A 并行) ██
  [B1] Imagination: pretrained WM → 200 条, GPU 4-5
  [B2] VLM 标注: zero-shot 16帧, GPU 6
  [B3] Policy 训练, GPU 8-9
  [B4] Policy 评估: 50ep, 对比 baseline 75%

██ Track C: 微调迭代 (Track A 完成后) ██
  微调 WM/VLM → Imagination → 标注 → 策略 → 评估
```

---

## Phase 1.5: 最小规模全链路验证 ✅ (02-28 02:00 完成)

> 6 项全部通过，正式训练已就绪。详细结果见上方表格。

**门控**: ✅ 6 项全通过 → 启动正式训练。

---

## Phase 1.5b: 补充验证 + wandb 集成 ⬜ (Phase 1.5 之后, 正式训练之前)

> Phase 1.5 中 V1 未覆盖验证视频生成和 wandb 日志。正式训练前需补充验证。

### V1.1: WM 验证视频生成测试

**背景**: 训练代码中验证触发条件为 `global_step % validation_steps == 5`（硬编码偏移 5）。
这意味着 `validation_steps ≤ 5` 时**永远不会触发**。V1 mini 用了 `validation_steps=5`，
实际上验证视频从未生成。之前的多次训练崩溃就发生在验证视频环节 (ffmpeg/dtype 问题)。

**方案**: 使用 `--max_train_steps=6 --validation_steps=6 --checkpointing_steps=6`
- Step 5: 触发 `validate_video_generation()` (5%6=5 ✅)
- Step 6: 保存 checkpoint + 训练结束
- 验证视频位置: `{output_dir}/samples/train_steps_5_0.mp4`
- 每个 `video_num` 会单独生成: `train_steps_5_0.mp4`, `train_steps_5_1.mp4`, ...

**命令模板**:
```bash
cd /home/wjz/rl-vla/ctrl_world && \
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline \
/home/wjz/miniconda3/envs/ctrl_world/bin/accelerate launch \
  --num_processes 4 --use_deepspeed --deepspeed_config_file ds_zero2.json \
  scripts/train_wm.py \
  --ckpt_path ../checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt \
  --dataset_root_path ../data/vlaw/encoded \
  --dataset_meta_info_path ../data/vlaw/meta_info/maniskill \
  --output_dir ../checkpoints/vlaw/world_model/v1_video_test \
  --max_train_steps 6 \
  --validation_steps 6 \
  --checkpointing_steps 6 \
  --gradient_accumulation_steps 8 \
  --task_type maniskill --height 384 --width 192 --action_dim 7 \
  --num_frames 15 --num_history 1 \
  --tag v1_video_test
```

**验证项**: 训练 6 步 → 在 step 5 触发验证视频 → 检查输出 mp4 文件是否正常
**CPU/GPU**: GPU 0-3, ~10 min
**产出**: `checkpoints/vlaw/world_model/v1_video_test/samples/train_steps_5_*.mp4`

### V1.2: WM + VLM wandb 逐步日志验证

**背景**: WM 训练默认每 100 步才写一次 wandb（`global_step % 100 == 0`），
mini 测试中完全看不到 loss 记录。VLM 有 `--use_wandb` 开关但之前未启用。
正式训练需要 wandb 在线监控。

**需要的代码修改** (2 处):

1. **WM: `ctrl_world/scripts/train_wm.py` line 171**
   ```python
   # 改前:
   if global_step % 100 == 0:
   # 改后:
   if global_step % args.log_every_n_steps == 0:
   ```
   同时:
   ```python
   # line 173 改前:
   accelerator.log({"train_loss": train_loss/100}, step=global_step)
   # 改后:
   accelerator.log({"train_loss": train_loss/args.log_every_n_steps}, step=global_step)
   ```
   在 config.py 的 `ManiSkillWMConfig` 中添加:
   ```python
   log_every_n_steps: int = 1  # mini-test 用 1, 正式训练用 10
   ```

2. **VLM**: 已支持 `--use_wandb`，无需修改代码，加命令行参数即可

**迷你测试命令** (验证 wandb 集成):

WM: 用改后代码 + `--log_every_n_steps=1 --max_train_steps=3 --validation_steps=99999`，
检查 wandb 面板出现 3 个 loss 数据点。

VLM: 加 `--use_wandb --wandb_project vlaw-reward --train_steps=2 --eval_steps=2 --gradient_accumulation_steps=2`，
检查 wandb 面板出现 train/loss + eval metrics。

**完成标准**: wandb 面板可见 step-by-step loss，VLM eval metrics 正常输出

---

## Iter 1 步骤表

| 步骤 | 内容 | Agent | 依赖 | GPU | 状态 |
|------|------|-------|------|-----|------|
| 0 | 数据扩充 (235 条) | Data | — | 4-5 | ✅ |
| V1-V6 | Phase 1.5 验证 | 各 | Step 0 | 各 | ✅ 02-28 02:00 |
| **V1.1** | **WM 验证视频测试 (6 steps)** | **WM** | **V1** | **0-3** | **✅ mp4有效, config.py修复** |
| **V1.2** | **WM+VLM wandb 逐步日志验证** | **WM+Reward** | **V1.1** | **0-3 / 4-5** | **✅ wandb集成OK** |
| 1a | WM 正式训练 (2000 steps) | WM | V1.1+V1.2 | 0-3 | ✅ 2h, PSNR=23.34 |
| 1c | VLM LoRA 16帧 (200 steps) | Reward | V1.2 | 6-7 | ✅ loss 6.8, acc=0.824, FP=3.7% |
| B1 | Imagination pretrained 200 条 | Imagination | V3 | 4-5 | ⚠️ 50/200 (cuda:0 error) |
| B2 | VLM标注 (fine-tuned VLM) | Reward | B1 | 6 | ✅ 50条, vlm=0; D_real 210条, FP=0% |
| B3 | Policy 训练 pretrained | Policy | B2 | 8-9 | ❌ **架构不匹配, 暂停** |
| B4 | Policy 评估 pretrained | Eval | B3 | 9 | ⬜ 等 B3 |
| 2 | WM 质量验证 (PSNR>18) | Eval | 1a | 0 | ✅ PSNR=23.34 PASS |
| 3 | Imagination 微调 WM 200-500 条 | Imagination | 2 | 4-5 | ⬜ |
| 4 | VLM 标注合成 (微调 VLM) | Reward | 3,1c | 6 | ⬜ |
| 5 | 策略更新 Weighted FM | Policy | 4 | 8-9 | ⬜ |
| 6 | 策略评估 50ep | Eval | 5 | 9 | ⬜ |
| 7 | 新策略 rollout + 编码 + 标注 | Data+Reward | 6 | 4-6 | ⬜ |

**门控**: WM PSNR>18 / 合成成功率 20-40% / 策略 success_at_end>75% / VLM AUC>0.85

---

## VLM 多卡启动命令

```bash
tmux kill-session -t vlm_train
tmux new-session -d -s vlm_train "
cd /home/wjz/rl-vla && \
CUDA_VISIBLE_DEVICES=4,5,7,8 \
/home/wjz/miniconda3/envs/vlaw_reward/bin/accelerate launch \
  --num_processes 4 --multi_gpu \
  rlft/vlaw/reward/train_reward_model.py \
  --data_dirs data/vlaw/rollouts/iter1 data/vlaw/rollouts/iter1_highsuc \
  --tasks LiftPegUpright-v1 \
  --model_path checkpoints/vlaw/reward_model/qwen_vl \
  --output_dir checkpoints/vlaw/reward_model/lora_iter1_16frame \
  --num_frames 16 --train_steps 200 --lora_r 16 \
  --per_device_batch_size 1 --gradient_accumulation_steps 32 --multi_gpu \
  2>&1 | tee logs/vlaw/vlm_lora_16frame_4gpu_train.log
"
```

---

## 时间线 (Day 3+)

```
Day 3 (02-28): Phase 1.5 ✅ 已完成
               → Track A 待启动: WM (GPU 0-3) + VLM (GPU 4,5,7,8)
               → Track B 待启动: pretrained pipeline (GPU 4-9)
Day 3-4:       Track B 完成 (首个端到端结果)
Day 5 (03-01): Track A WM 完成 → Track C
Day 5-6:       Track C + Iter 2
Day 6-7:       Phase 3 最终评估
```

| 阶段 | 耗时 | GPU |
|------|------|-----|
| Phase 1.5 | ~2h | 各 |
| WM 正式训练 | ~50h | 0-3 |
| VLM 正式训练 | ~1-2h | 4,5,7,8 |
| Track B (B1-B4) | ~6-10h | 4-9 |
| Track C | ~4-8h | 4-9 |
| **到首个端到端结果** | **~8-12h** | |
| **到微调结果** | **~55h** | |

---

## 注意事项

1. **WM**: DeepSpeed ZeRO-2, ~88s/step, pretrained 起点, batch=1×grad_accum=8, ~4 epochs
2. **VLM**: 16帧 images > video (AUC 0.82 vs 0.65), 多卡 4×batch1×grad_accum32=有效 batch128, ~12GB/卡
3. **Phase 1.5 是硬性前置**: 先验证再训练
4. **Track B 双重价值**: 验证 pipeline + 产出首个端到端结果
5. **WM 验证视频触发条件**: `global_step % validation_steps == 5` (硬编码偏移)，`validation_steps ≤ 5` 永远不触发
6. **wandb 正式训练配置**: WM 用 `--log_every_n_steps=10`; VLM 用 `--use_wandb --wandb_project vlaw-reward`
7. **WM 微调技术文档**: 见 `docs/vlaw/wm_finetuning_overview.md`
