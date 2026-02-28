# Ctrl-World 世界模型 Iter-1 微调记录

> **训练日期**: 2026-02-28 13:02 → 15:03 (约 2 小时)
> **技术概览**: 见 [wm_finetuning_overview.md](wm_finetuning_overview.md)

---

## 1. 调试过程 (V1.1 阶段)

正式训练前经历了 3 轮调试（V1.1: 验证视频生成 6-step mini test），每轮修复一个关键 Bug：

### 崩溃 1: ffmpeg not found

- **症状**: `mediapy._check_ffmpeg() → RuntimeError: ffmpeg not found`
- **根因**: tmux 会话未激活 conda 环境，`ffmpeg` 不在 PATH 中
- **修复**: 
  - 运行时: tmux 命令中显式 `eval "$(conda shell.bash hook)" && conda activate ctrl_world`
  - 代码层: `train_wm.py` L30-34 已有 hack（硬编码 ffmpeg 路径到 PATH）
- **Bug ID**: BUG-014

### 崩溃 2: SVD model path not a valid local path

- **症状**: `OSError: ../checkpoints/... is not a local folder and is not a valid model identifier`
- **根因**: `config.py` 中 `wm_args_maniskill` 的 `svd_model_path` 和 `clip_model_path` 使用了 `checkpoints/...` 相对路径，但训练脚本从 `ctrl_world/` 目录运行，需要 `../checkpoints/...`
- **修复**: `ctrl_world/config.py` L189-190 添加 `../` 前缀
  ```python
  # 修复前:
  svd_model_path: str = "checkpoints/vlaw/world_model/pretrained/svd"
  clip_model_path: str = "checkpoints/vlaw/world_model/pretrained/clip"
  # 修复后:
  svd_model_path: str = "../checkpoints/vlaw/world_model/pretrained/svd"
  clip_model_path: str = "../checkpoints/vlaw/world_model/pretrained/clip"
  ```
- **Bug ID**: BUG-013

### 崩溃 3: 训练不会在 max_train_steps 处停止

- **症状**: 训练会跑完整个 dataloader epoch 而非在指定步数停止
- **根因**: 训练循环内层无 break 条件
- **修复**: `train_wm.py` L194-197 添加 early-stop break：
  ```python
  if global_step >= args.max_train_steps:
      break
  # 外层循环同样添加
  ```
- **Bug ID**: BUG-015

### V1.1 通过

3 轮修复后，6-step mini test 完成，验证视频 `train_steps_5_*.mp4` 正常生成（384×384, h264, 2fps）。

---

## 2. 超参数

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **基础模型** | checkpoint-10000.pt (DROID pretrained) | SVD UNet ~1.5B 参数 |
| **分布式** | DeepSpeed ZeRO-2, 4× RTX 4090 | `ds_zero2.json`, ~14.5GB/卡 |
| **per_device_batch** | 1 | |
| **gradient_accumulation** | 8 | 有效 batch = 4×1×8 = 32 |
| **总步数** | 2000 global steps | ~4 epochs over 4378 windows |
| **每步耗时** | ~3.6-4.0 秒 | |
| **Checkpoint 间隔** | 每 500 steps | .pt 文件 ~4.4GB |
| **验证视频间隔** | 500 steps (offset 5: step 5, 505, 1005, 1505) | |
| **优化器** | AdamW | lr=1e-5, 无 warmup |
| **梯度裁剪** | via Accelerate (max_grad_norm) | |
| **混合精度** | fp16 (DeepSpeed) | |
| **Gradient checkpointing** | ✅ | UNet 启用 |
| **微调策略** | Phase-B: 全量微调 UNet + Action Encoder | |
| **log_every_n_steps** | 500 (per wandb) | 按 500 步记录 loss |
| **任务** | LiftPegUpright-v1 (ManiSkill3) | |
| **latent 尺寸** | (T, 4, 48, 24) — 384×192 图像下采样 8× | |
| **num_frames** | 15 (future) | |
| **num_history** | 1 | 总序列长度 = 16 帧 |
| **action_dim** | 7 | (dx, dy, dz, drx, dry, drz, gripper) |

---

## 3. 数据集

| 来源 | 条数 | 说明 |
|------|------|------|
| demos | 25 | 专家演示, 100% 成功 |
| iter1 rollouts | 50 | 基线策略 rollout, 16% 成功 |
| iter1_highsuc | 50 | 高成功率 rollout, 70% 成功 |
| iter1_lift_inc20 | 40 | 渐进难度 rollout, 30% 成功 |
| 原始 iter1 其他 | 70+ | 混合 rollout |
| **总计** | **235 条轨迹** | |
| **滑窗后** | **~4378 训练窗口 + ~60 验证窗口** | 窗口 = 16 帧 (1 history + 15 future) |

数据路径: `data/vlaw/encoded/reencode_highsuc_inc20/LiftPegUpright-v1/`

---

## 4. 训练过程

### 启动命令

```bash
tmux new-session -d -s wm_iter1 "
eval \"\$(conda shell.bash hook 2>/dev/null)\" && conda activate ctrl_world && \
cd /home/wjz/rl-vla/ctrl_world && \
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline \
/home/wjz/miniconda3/envs/ctrl_world/bin/accelerate launch \
  --num_processes 4 --use_deepspeed --deepspeed_config_file ds_zero2.json \
  scripts/train_wm.py \
  --ckpt_path ../checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt \
  --dataset_root_path ../data/vlaw/encoded \
  --dataset_meta_info_path ../data/vlaw/meta_info/maniskill \
  --output_dir ../checkpoints/vlaw/world_model/iter1 \
  --max_train_steps 2000 \
  --validation_steps 500 \
  --checkpointing_steps 500 \
  --gradient_accumulation_steps 8 \
  --task_type maniskill --height 384 --width 192 --action_dim 7 \
  --num_frames 15 --num_history 1 \
  --tag wm_iter1_formal \
  2>&1 | tee /home/wjz/rl-vla/logs/vlaw/wm_iter1_formal_train.log
"
```

**关键注意**: 
- 必须从 `ctrl_world/` 目录运行（`cd /home/wjz/rl-vla/ctrl_world`）
- tmux 中必须显式激活 conda（见 BUG-014）
- 所有路径相对于 `ctrl_world/`，使用 `../` 前缀指向项目根

### Loss 曲线

```
Step    5:   (初始)
Step  505:   loss = 0.0173
Step 1005:   loss = 0.0086
Step 1505:   loss = 0.00269
Step 2000:   loss = 0.00806 (尾部略回升, 正常)
```

Loss 呈稳定下降趋势，2000 步完成无异常。

### GPU 显存

4 卡均匀分布（ZeRO-2）：

| GPU | 显存 |
|-----|------|
| 0 | ~14.5 GB / 24.6 GB |
| 1 | ~14.6 GB / 24.6 GB |
| 2 | ~14.5 GB / 24.6 GB |
| 3 | ~14.5 GB / 24.6 GB |

### 产出文件

```
checkpoints/vlaw/world_model/iter1/
├── checkpoint-500.pt     (4.4GB, 13:33)
├── checkpoint-1000.pt    (4.4GB, 14:04)
├── checkpoint-1500.pt    (4.4GB, 14:33)
├── checkpoint-2000.pt    (4.4GB, 15:03)  ← 最终 checkpoint
└── samples/              (13 个 mp4 验证视频)
```

日志: `logs/vlaw/wm_iter1_formal_train.log`

---

## 5. 评估结果

评估脚本: `scripts/eval_wm_iter1.py`  
评估报告: `results/vlaw/wm_iter1_eval_report.md`

### 整体指标

| 指标 | pretrained (ckpt-10000) | iter1 (ckpt-2000) | Delta |
|------|------------------------|-------------------|-------|
| **PSNR ↑** | 23.01 ± 5.05 | **23.34 ± 2.72** | **+0.33** |
| SSIM ↑ | 0.8014 ± 0.1182 | 0.7929 ± 0.0770 | -0.0085 |
| **LPIPS ↓** | 0.1297 ± 0.1127 | **0.1190 ± 0.0697** | **-0.0107** |
| #trajs | 14 | 14 | — |
| #frames | 70 | 70 | — |

### 门控检查

- ✅ PSNR = 23.34 > 18.0（门控阈值）
- ✅ LPIPS 改善 -0.01（感知质量提升）
- ⚠️ SSIM 略降 -0.008（在误差范围内）
- **结论**: **PASS** — 微调有效，iter1 WM checkpoint 可用于后续 Imagination

### 逐轨迹 PSNR

| 轨迹 (demo val) | PSNR | SSIM |
|-----------------|------|------|
| traj_0020 | 23.69 | 0.8687 |
| traj_0021 | 24.08 | 0.8606 |
| traj_0022 | 25.80 | 0.8908 |
| traj_0023 | 24.35 | 0.8695 |
| traj_0024 | 28.20 | 0.9347 |

| 轨迹 (rollout val) | PSNR | SSIM |
|--------------------|------|------|
| traj_0040 | 23.46 | 0.7628 |
| traj_0041 | 21.21 | 0.6926 |
| traj_0042 | 20.21 | 0.7284 |
| traj_0043 | 22.35 | 0.7579 |
| traj_0044 | 21.01 | 0.7148 |
| traj_0046 | 23.31 | 0.7393 |
| traj_0047 | 24.00 | 0.7715 |
| traj_0048 | 21.80 | 0.7433 |
| traj_0049 | 23.32 | 0.7651 |

**观察**: Demo 轨迹 PSNR 普遍更高 (24-28)，rollout 轨迹 (21-24) 因动作更多样。

---

## 6. 已知问题与后续

1. **PSNR 提升幅度不大 (+0.33)**：仅 2000 步微调，论文未给出具体目标值。iter2 可在此基础上继续训练。
2. **验证视频触发条件**: `global_step % validation_steps == 5`（硬编码偏移 5），`validation_steps ≤ 5` 时永远不触发。
3. **Imagination 生成**: 使用 pretrained WM 的 Track B 中仅 50/200 轨迹成功（cuda:0 device error），微调 WM 的 Track C 待后续执行。
4. **Policy 架构不匹配**: ShortCut Flow base checkpoint 使用视觉编码器 (global_cond_dim=626)，VLAWPolicyUpdater 使用原始 state (global_cond_dim=50)，需适配。

---

## 附录: 关键文件清单

| 文件 | 说明 |
|------|------|
| `ctrl_world/scripts/train_wm.py` | 训练脚本 |
| `ctrl_world/config.py` | 配置 (ManiSkillWMConfig) |
| `ctrl_world/ds_zero2.json` | DeepSpeed ZeRO-2 配置 |
| `ctrl_world/models/ctrl_world.py` | 模型定义 |
| `scripts/eval_wm_iter1.py` | 评估脚本 |
| `results/vlaw/wm_iter1_eval_report.md` | 评估报告 |
| `logs/vlaw/wm_iter1_formal_train.log` | 训练日志 |
| `checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt` | 最终 checkpoint |
