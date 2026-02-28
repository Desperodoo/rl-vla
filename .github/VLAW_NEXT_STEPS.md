# VLAW 下一步推进计划

> **最后更新**: 2026-02-28 16:30 | **历史版本**: `docs/vlaw/archive/VLAW_NEXT_STEPS_*.md`
> **状态面板**: [`vlaw-status.md`](vlaw-status.md) | **评测汇总**: [`docs/vlaw/baselines_and_evaluation.md`](../docs/vlaw/baselines_and_evaluation.md)

---

## 已完成阶段 (02-28)

- ✅ Phase 1.5: V1-V6 全链路验证
- ✅ Phase 1.5b: V1.1 视频验证 + V1.2 wandb 集成
- ✅ Track A1: WM 2000 steps (PSNR=23.34 > 18 PASS)
- ✅ Track A2: VLM 16帧 200 steps (acc=0.824, FP=3.7%)
- ✅ Track B1: Imagination 50/200 条 (pretrained WM)
- ✅ Track B2: VLM 标注 (合成 vlm=0; D_real FP=0%)
- ✅ WM 评估 (PSNR=23.34, LPIPS=0.119)

---

## ❌ 阻塞项: Policy 架构不匹配

ShortCut Flow base checkpoint 使用 **视觉编码器** (PlainConv, global_cond_dim=626)，
VLAWPolicyUpdater 使用 **raw state** (global_cond_dim=50)。

**解决方案** (需选择一个):
1. 适配 `VLAWPolicyUpdater` 使用视觉 observations (与 base ckpt 对齐)
2. 从 scratch 训练 state-only policy (丢弃 base ckpt)
3. 修改 ShortCut Flow 训练配置, 训练一个 state-based base ckpt

---

## 待推进任务

| 优先级 | 任务 | 依赖 | GPU |
|--------|------|------|-----|
| **P0** | **解决 Policy 架构不匹配** | — | — |
| P1 | Track C: Imagination (微调 WM, 200-500 条) | WM iter1 ✅ | 4-5 |
| P1 | Track C: VLM 标注合成数据 | Imagination + VLM ✅ | 6-7 |
| P2 | Track C: 策略更新 Weighted FM | P0 + VLM 标注 | 8-9 |
| P2 | Track C: 策略评估 (对比 baseline 75%) | 策略更新 | 9 |
| P3 | Iter-2 全流程 | Iter-1 完成 | 全部 |

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

### WM 训练

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
  --output_dir ../checkpoints/vlaw/world_model/iter2 \
  --max_train_steps 2000 --validation_steps 500 --checkpointing_steps 500 \
  --gradient_accumulation_steps 8 \
  --task_type maniskill --height 384 --width 192 --action_dim 7 \
  --num_frames 15 --num_history 1 \
  2>&1 | tee /home/wjz/rl-vla/logs/vlaw/wm_iter2_train.log
"
```

### VLM 训练

```bash
tmux new-session -d -s vlm "
cd /home/wjz/rl-vla &&
CUDA_VISIBLE_DEVICES=6,7 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python rlft/vlaw/reward/train_reward_model.py \
  --data_dirs data/vlaw/rollouts/iter1 data/vlaw/rollouts/iter1_highsuc \
  --tasks LiftPegUpright-v1 \
  --model_path checkpoints/vlaw/reward_model/qwen_vl \
  --output_dir checkpoints/vlaw/reward_model/lora_iter2 \
  --num_frames 16 --train_steps 200 --lora_r 16 \
  --per_device_batch_size 1 --gradient_accumulation_steps 128 \
  --use_wandb --wandb_project vlaw-reward \
  2>&1 | tee logs/vlaw/vlm_iter2_train.log
"
```
