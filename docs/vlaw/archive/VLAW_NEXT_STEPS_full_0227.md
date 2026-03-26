# VLAW 下一步推进计划

> **创建时间**: 2026-02-26 | **最后更新**: 2026-02-27 22:30
> **目标**: 最小验证 → 正式训练 + 并行 Policy Pipeline → 多轮迭代
> **核心原则**: ①先用最小实验验证全链路无故障，再启动长时间训练；②利用 pretrained 模型并行构建 Policy Pipeline，不空等

---

## 当前进展总结（截至 2026-02-27 下午）

### ✅ 已完成

| 阶段 | 完成内容 | 关键结论 |
|------|---------|---------|
| Phase 0 | 数据审计 + 异常清除 | 9 文件扫描，5 文件异常已删除，剩余 4 文件全部 `(4,48,24)` 正确 |
| Phase 1A | WM 多 horizon 基线 | pretrained H20 PSNR=22.35/SSIM=0.79（衰减小）; Phase-A H20 PSNR=21.70/SSIM=0.58（衰减大） |
| Phase 1B | VLM zero-shot vs LoRA 基线 | ZS AUC=0.59, LoRA AUC=0.62; 单帧评估区分力极弱; α=0.8 完全无效 |
| Step 0 | 数据扩充 | 160 条 re-encode (highsuc+inc20) → 总 235 条 / 4378 训练窗口 |
| Step 1b | VLM 16帧多帧评估 | **16帧多图 AUC=0.8153** (+0.236 vs 单帧); Recall@FP<20%=67.8%; images > video 格式 |
| Step 1d | Imagination env.step() 验证 | **5/5 轨迹成功**；2 个 device mismatch bug 已修复 (L420, L700)；输出 (T,4,48,24) latents, (T,7) actions, (T,25) states |
| 基础设施 | DeepSpeed ZeRO-2 + dtype 适配 + ffmpeg | 5 次崩溃修复 + 2次 WM 重启（SIGHUP + ffmpeg）；WM 全量微调在 4×4090 上可行 |

### 🔄 进行中 / ⚠️ 需恢复

| 任务 | 状态 | GPU | 预计完成 |
|------|------|-----|----------|
| WM iter1 训练 | ❌ **已崩溃** (step ~16/2000, ffmpeg not found, 20:08) | 0-3 (已释放) | 需重启, ~50h |
| VLM LoRA 16帧重做 | ⚠️ 运行 2h 无训练输出 (卡在 JIT 编译/首次前向), 建议 kill → 多卡重启 | 6 (~11.7GB) | 见多卡加速方案 |

### 🔧 诊断详情 (02-27 21:40)

**WM 崩溃分析**:
- tmux:wm_train 会话已丢失，GPU 0-3 完全释放 (17 MiB)
- 根因：验证阶段生成视频时调用 ffmpeg → `RuntimeError: Program 'ffmpeg' is not found`
- 此为 BUG-007 复发（第 3 次因 ffmpeg 崩溃）
- checkpoint 目录 `iter1/` 仅有 swanlog，**无模型权重保存**
- 修复方案：(a) 在 rlft_ms3 env 安装 ffmpeg, 或 (b) 设 `--validation_steps=99999` 禁用验证视频

**VLM 训练分析**:
- PID 1333722 存活，CPU 367% (torch._inductor compile workers)，GPU 6 使用 11.7GB/36%
- 日志最后输出：`use_cache=False` (gradient checkpointing 设置完成)
- **未输出 LoRA trainable parameters 信息、也未输出任何 step/loss**
- 已运行 ~2h 无实质进展，16帧多图输入序列极长，首次 JIT 编译时间不可控
- **建议：kill 当前进程 → 多卡并行重启（见下方分析）**

### 可用资产清单

| 资产 | 路径 | 规模 |
|------|------|------|
| Demo 数据 (编码) | `encoded/demos/LiftPegUpright-v1/` | 25 条, 100% 成功 |
| D_real iter1 (编码) | `encoded/rollouts/iter1/LiftPegUpright-v1/` | 50 条, 16% 成功 |
| D_real re-encode (编码) | `encoded/reencode_highsuc_inc20/LiftPegUpright-v1/` | 160 条 (5 HDF5), shape (4,48,24) ✅ |
| Lift-only 合并 (编码) | `encoded/lift_only_iter1_48x24/LiftPegUpright-v1/` | 75 条 (25 demo + 50 rollout) |
| 策略基线 | `results/vlaw/pld_eval_baseline_20ep.json` | success_once=95%, success_at_end=75% |
| WM 基线报告 | `results/vlaw/wm_baseline_report.md` | 多 horizon PSNR/SSIM/LPIPS |
| VLM 基线报告 | `results/vlaw/vlm_baseline_report.md` | ROC-AUC + confusion matrix |
| VLM 16帧报告 | `results/vlaw/vlm_multiframe_report.md` | 三配置对比 + AUC + CM |

---

## 关键问题诊断（Phase 1 后更新 + 论文对照分析）

### 论文 vs 我们的数据规模差异（根因分析）

| 维度 | VLAW 论文 (DROID) | 我们 (ManiSkill) | 差距 |
|------|-------------------|------------------|------|
| WM 预训练数据 | 95K 轨迹 (DROID) | 25 demo + 50 rollout = **75 条** → 扩充至 **235 条** | 1/1267 → 1/404 |
| WM 总帧数 | 数百万帧 | **2,559 帧** → 扩充至 **~10K 帧** | ~1/1000 → ~1/300 |
| WM 训练窗口数¹ | ~数十万 | **~100 个** → 扩充至 **4,378 个** | ~1/1000 → ~1/50 |
| WM Phase-B 步数 | 50K steps, batch=4 | ~~10K steps~~ → **2000 steps**, batch=1 × grad_accum=8 | 按 ~4 epochs 缩放 |
| WM 数据遍历次数 | 50K×4 / 95K ≈ **2 epochs** | 2000×8 / 4378 ≈ **~4 epochs** | 合理范围 |
| VLM 输入 | **16 帧**序列 | ~~单帧~~ → **已改为 16 帧** ✅ | 已对齐 |
| VLM 微调 | 200 steps, batch=128, 16帧 | 已完成但是**单帧**训练 | 需用 16 帧重做 |

> ¹ 训练窗口：数据扩充后 235 条轨迹，9 帧滑动窗口 → 4,378 个训练窗口。

**核心结论（更新）**：数据扩充后训练窗口从 ~100 增加到 4,378，WM 训练遍历次数从 ~800 epochs 降到 ~4 epochs，过拟合风险大幅降低。VLM 16帧评估 AUC 从 0.58 跃升至 0.82，确认单帧是之前区分力弱的根本原因。

### 问题清单

| 问题 | Phase 1 基线数据 | 影响 & 对策 |
|------|-----------------|-------------|
| ~~Latent shape 不一致~~ | ✅ 已解决：异常数据已清除 | 无 |
| ~~**数据量不足（根因）**~~ | ~~仅 75 条 / ~100 窗口~~ → **已扩至 235 条 / 4378 窗口** | ✅ 过拟合风险从 ~800 epochs 降至 ~4 epochs |
| ~~VLM 区分力弱~~ | ~~ZS AUC=0.59~~ → **16帧多图 AUC=0.8153** | ✅ 确认根因是单帧，16帧后大幅改善 |
| ~~WM Phase-A 退化~~ | pretrained SSIM=0.79 vs Phase-A=0.58 | ✅ Iter1 已从 **pretrained** 开始训练 |
| WM 逐帧周期性 | pretrained 逐帧 PSNR 呈 5帧周期振荡 | SVD temporal conditioning 特性，不影响平均质量 |
| ~~磁盘紧张~~ | ~~63GB 可用~~ | ✅ 已挂载 15TB 磁盘 (13TB free)，data/checkpoints 软链接到 /mnt/disk_2 |
| VLM LoRA 需重做 | 当前 LoRA 基于单帧训练 | 需用 16 帧 images 格式重新 finetune |

---

## Phase 0: 数据质量审计 ✅ 已完成

- **执行**: Eval-Agent 扫描 9 文件/340 轨迹
- **结果**: 4 正常 `(4,48,24)` + 3 错 shape `(4,32,16)` + 2 无 latent
- **处理**: 异常数据直接清除（数据可重新采集），不做修复
- **产出**: `logs/vlaw/data_audit_report.md` + `.json`

---

## Phase 1: WM & VLM 基线报告 ✅ 已完成

### Task 1A: WM 基线 ✅

**核心数据**（按 horizon 分解）：

| Horizon | pretrained PSNR | Phase-A PSNR | pretrained SSIM | Phase-A SSIM |
|---------|----------------|--------------|-----------------|--------------|
| 5 帧 | 22.10 | 22.60 | 0.79 | 0.64 |
| 10 帧 | 22.50 | 22.37 | 0.80 | 0.61 |
| 15 帧 | 22.29 | 21.99 | 0.79 | 0.59 |
| 20 帧 | 22.35 | 21.70 | 0.79 | 0.58 |

**结论**: pretrained 在长 horizon 更稳定；Phase-A SSIM 衰减显著。后续 Iter1 从 pretrained 起训更合理。

### Task 1B: VLM 基线 ✅

**核心数据**：

| 模型 | ROC-AUC | p_yes (success) | p_yes (fail) | @ α=0.8 TP |
|------|---------|-----------------|--------------|-----------|
| Zero-shot | 0.585 | 0.083 ± 0.069 | 0.068 ± 0.073 | 0 |
| LoRA ft | 0.617 | 0.009 ± 0.006 | 0.007 ± 0.007 | 0 |

**结论**: 两种模型的 p_yes 都极低(<0.1)，α=0.8 阈值完全无效。LoRA 微调后 p_yes 反而更低（过于保守）。
**根因**: 当前使用**单帧**评估，论文使用 16 帧序列。后续需改为多帧输入并扩充正样本重新微调。

---

## Phase 2: 多轮迭代（LiftPegUpright-only） — 🔄 进行中

基于 VLAW Algorithm 1，先做 **2 轮**，根据结果决定是否继续。

### 本轮对话完成的工作 (02-27)

#### Step 0: 数据扩充 ✅
- Data-Agent 重新 VAE 编码 160 条轨迹（highsuc 50 + inc20 40 + 交叉来源）
- 总数据：235 条 / 4378 训练窗口（从 75 条 / ~100 窗口扩充 44 倍）
- 路径：`encoded/reencode_highsuc_inc20/LiftPegUpright-v1/` (5 HDF5, shape (4,48,24))

#### Step 1a: WM Iter1 微调 🔄 训练中
- **决策**：全量微调 1.5B UNet（而非仅 Action Encoder），因 Phase-A 仅训练 AE 导致 SSIM 退化
- **DeepSpeed 适配**：全量微调在 DDP 下 OOM (~22.8GB/24GB)，引入 ZeRO-2 将 optimizer states + gradients 分片到 4 卡 → ~18GB/GPU
  - 训练速度从理论 ~50-60s/step (DDP) 变为 ~88s/step (ZeRO-2)，慢 ~50%
  - **但这是可行性前提，不是性能优化**：没有 ZeRO-2，全量微调在 4×4090 24GB 上根本跑不了
  - 替代方案（LoRA 部分微调）不符合 ADR-007 "全量微调"的决策
- **dtype 修复链**：DeepSpeed fp16 模式下经历 5 次崩溃修复
  1. OOM → DeepSpeed ZeRO-2
  2. contiguous tensor → `p.data = p.data.contiguous()`
  3. Float vs Half (训练) → forward 中 latents/actions/UNet 输入 `.to(dtype=dtype)`
  4. Float vs Half (验证) → actions/latents dtype cast in `validate_video_generation()`
  5. Float vs Half (VAE decode) → chunk `.to(dtype=_vae_dtype)` + ffmpeg 安装
- **当前状态**：step 16/2000, ~88s/step, GPU 0-3 ~18GB, ETA ~47h (~3/1 下午)
- **训练配置**：`pretrained` 起点 / 2000 steps / lr=1e-5 / batch=1×grad_accum=8 / ~4 epochs
- **日志**：`logs/vlaw/wm_iter1_train.log` / wandb: `zhuzhulab/vlaw_ctrl_world/runs/t0zchllz`

#### Step 1b: VLM 16帧多帧评估 ✅
- 修改 `reward_model.py` 支持 `use_video_format=True` 和 `fps` 参数
- 创建 `scripts/eval_vlm_multiframe.py` 三配置对比脚本
- **结果**（170 条轨迹：59 成功 / 111 失败, Qwen3-VL-4B zero-shot）:

  | 配置 | AUC | Recall@FP<20% | 备注 |
  |------|-----|---------------|------|
  | 单帧（最后一帧） | 0.5793 | 18.6% | 旧基线，接近随机 |
  | **16帧多图 (images)** | **0.8153** | **67.8%** | **最佳，+0.236** |
  | 16帧视频 (video) | 0.6452 | 44.1% | video 格式不如 images |

- **关键发现**：
  - 多帧 images 模式 AUC=0.82，zero-shot 即有强区分力
  - images > video：可能因 Qwen3-VL 视频处理内部做时间降采样导致帧丢失
  - p_yes 绝对值仍低 (<0.01)，α=0.8 阈值仍无效 → 需 LoRA 微调提高绝对值
- **报告**：`results/vlaw/vlm_multiframe_eval.json` + `results/vlaw/vlm_multiframe_report.md`

### Phase 1 后的调整（已执行）
1. ✅ WM Iter1 从 **pretrained** 权重开始（Phase-A 退化严重）— ADR-007
2. ✅ VLM 评估改为 **16帧多图输入**（AUC 0.58→0.82）— ADR-008
3. ✅ 数据扩充到 235 条 / 4378 窗口（~4 epochs 而非 ~800）
4. 待定: VLM LoRA 用 16 帧重新微调 → 预期 AUC > 0.85 + p_yes 提高到可用阈值

### VLM 多卡并行加速方案分析 (02-27 21:40)

#### 背景
- 当前 VLM LoRA 16帧训练单卡运行 2h 无输出，JIT 编译因 16帧长序列耗时不可控
- 训练脚本 `train_reward_model.py` 已内建完整的多卡支持（Accelerate + DistributedSampler）

#### 当前配置 vs 多卡配置

| 参数 | 当前 (1 GPU) | 建议 (4 GPU) | 说明 |
|------|-------------|-------------|------|
| GPU | GPU 6 | GPU 4,5,7,8 | 避开 WM 的 0-3 |
| `per_device_batch_size` | 1 | 1 | 16帧序列长，单样本就很大 |
| `gradient_accumulation_steps` | 128 | 32 | 除以 GPU 数量 |
| 有效 batch | 1×128 = **128** | 4×1×32 = **128** | **保持不变** |
| 每步微批次数/GPU | 128 | 32 | **每卡减少 4 倍** |
| 预计提速 | — | **~3-4x** | 近线性 (DDP 通信开销小, LoRA 只 24MB) |

#### VRAM 可行性
- Qwen3-VL-4B (bfloat16): ~8.3GB / GPU
- LoRA 可训练参数: ~24MB / GPU
- Activations (gradient checkpointing 启用): ~3-4GB / GPU
- **总计 ~12GB / GPU ≪ 24GB (4090)**  ✅
- 4 卡总 VRAM: 48GB (仅用 ~48GB)，不需要 ZeRO 分片

#### 代码就绪度
- ✅ `--multi_gpu` 参数已实现
- ✅ Accelerate DDP (bf16 mixed precision)
- ✅ DistributedSampler (正确分片)
- ✅ 仅 rank 0 保存 checkpoint + 评估
- ✅ `accelerator.backward()` + `accelerator.wait_for_everyone()`

#### 启动命令
```bash
# Kill 当前单卡进程
tmux kill-session -t vlm_train

# 多卡重启 (4 GPU)
tmux new-session -d -s vlm_train "
cd /home/wjz/rl-vla && \
CUDA_VISIBLE_DEVICES=4,5,7,8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/wjz/miniconda3/envs/vlaw_reward/bin/accelerate launch \
  --num_processes 4 --multi_gpu \
  rlft/vlaw/reward/train_reward_model.py \
  --data_dirs data/vlaw/rollouts/iter1 data/vlaw/rollouts/iter1_highsuc \
  --tasks LiftPegUpright-v1 \
  --model_path checkpoints/vlaw/reward_model/qwen_vl \
  --output_dir checkpoints/vlaw/reward_model/lora_iter1_16frame \
  --num_frames 16 \
  --train_steps 200 \
  --lora_r 16 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --multi_gpu \
  2>&1 | tee logs/vlaw/vlm_lora_16frame_4gpu_train.log
"
```

#### 风险与备选
- **风险 1**: flash_attention_2 在多卡模式下可能有兼容性问题 → 加 `--attn_implementation sdpa` 回退
- **风险 2**: Accelerate + peft 版本兼容 → 当前 peft 0.18.1 应支持
- **备选**: 若 4 卡有问题，先试 2 卡 (grad_accum=64, 2x 加速)

### 下一步推进（调整后的三轨并行策略）

> **核心调整 (02-27 22:30)**:
> 1. 正式训练前，先用最小规模跑通全链路（每个环节 3-5 steps），确认 Pipeline 无故障
> 2. 利用 pretrained WM + zero-shot VLM 并行构建 & 验证 Policy Training Pipeline

```
██ Phase 1.5: 最小规模全链路验证 (优先级最高, ~2-3h) ██
  所有正式训练之前必须通过此验证。
  [V1] WM mini-train: 5 steps → ckpt save/load → validate video (ffmpeg 修复)
  [V2] VLM mini-train: 3 steps, 多卡 → ckpt save/load → inference 测试
  [V3] Imagination mini: pretrained WM → 生成 5 条合成轨迹
  [V4] VLM labeling mini: zero-shot VLM → 标注 V3 的 5 条轨迹
  [V5] Policy mini-train: weighted FM, 10 steps → 确认 loss 下降
  [V6] Policy mini-eval: 5 episodes → 确认 eval 流程正常
  ↓ 全部通过后，启动正式训练

██ Track A: 正式长时间训练 (Phase 1.5 通过后) ██
  [A1] WM 正式训练: 2000 steps, GPU 0-3, ~50h
  [A2] VLM LoRA 正式训练: 200 steps, 4GPU, ~1-2h

██ Track B: Policy Pipeline 预验证 (与 Track A 并行, 用 pretrained 模型) ██
  可在 Track A 训练期间同时进行，不依赖 WM/VLM 训练完成
  [B1] Imagination 合成: pretrained WM → 生成 200 条合成轨迹, GPU 4-5
  [B2] VLM 标注: zero-shot VLM 16帧 (AUC=0.82) → 标注合成数据, GPU 6
  [B3] Policy 训练: Weighted FM (D_demo + D_real+ + D_syn+), GPU 8-9
  [B4] Policy 评估: 50ep, 对比 baseline 75%
  → 这一轮结果本身就有意义：验证 Pipeline + 获得首个端到端指标

██ Track C: 正式迭代 (Track A 完成 + Track B 验证通过后) ██
  [C1-C7] 用训练后的 WM + VLM 重新跑 Imagination → 标注 → 策略更新 → 评估
  预期优于 Track B（WM/VLM 经过微调），但 Track B 已验证全链路无问题
```

**为什么这样调整**:
- WM 训练 ~50h，VLM 训练 ~1-2h，每次崩溃代价极高
- Phase 1.5 用 ~2h 验证全部环节，发现问题的成本极低
- Track B 利用 pretrained WM (PSNR=22.35, SSIM=0.79) + zero-shot VLM (AUC=0.82)，两者质量均可用
- Track B 既是验证，也能产出首个端到端结果（可能已经 work）
- Track A 完成后再跑 Track C，增量仅是替换 WM/VLM checkpoint

---

## Phase 1.5: 最小规模全链路验证 ⬜ 新增

> **目的**: 用最小步数跑通 WM 训练/VLM 训练/Imagination/标注/Policy 训练/评估全部 6 个环节，
> 确认每个环节的 I/O、dtype、checkpoint、依赖都正确，避免正式训练跑几十小时后崩溃。

### V1: WM mini-train (5 steps)
- **配置**: 与正式训练完全相同，仅 `--train_steps=5 --validation_steps=5`
- **验证项**:
  - [ ] DeepSpeed ZeRO-2 启动成功（4 GPU）
  - [ ] 5 steps 训练无报错，loss 有值
  - [ ] checkpoint 正确保存到 `iter1_mini/`
  - [ ] 验证视频生成成功（ffmpeg 已安装）
  - [ ] checkpoint 可被 Imagination 加载
- **预计耗时**: ~15min（启动+5步+验证）
- **修复前置**: 先安装 ffmpeg 到 rlft_ms3 env

### V2: VLM mini-train (3 steps)
- **配置**: 多卡模式，`--train_steps=3 --gradient_accumulation_steps=4 --eval_steps=3`
- **验证项**:
  - [ ] Accelerate 多卡启动成功
  - [ ] 模型加载 + LoRA 附加 + gradient checkpointing 正常
  - [ ] 3 steps 训练输出 loss
  - [ ] step 3 评估 + checkpoint 保存成功
  - [ ] LoRA checkpoint 可被 reward_model.py 推理加载
- **预计耗时**: ~20min（4卡，grad_accum=4 → 每步仅 4 微批次）

### V3: Imagination mini (5 条)
- **输入**: pretrained WM checkpoint
- **验证项**:
  - [ ] ImaginationEngine 初始化成功
  - [ ] env.step() 模式生成 5 条合成轨迹
  - [ ] 输出格式正确: latents (T,4,48,24), actions (T,7), states (T,25)
  - [ ] 轨迹可写入 HDF5 / 可被 VLM 标注读取
- **预计耗时**: ~10min
- **注意**: Step 1d 已验证 env.step()，此处主要验证数据落盘和下游可读

### V4: VLM labeling mini (5 条)
- **输入**: V3 生成的 5 条合成轨迹
- **使用**: zero-shot VLM 16帧 images 模式
- **验证项**:
  - [ ] 合成轨迹可被 reward_model.py 正确读取
  - [ ] VLM 推理输出 p_yes 值
  - [ ] 二分类标签写回正确
- **预计耗时**: ~5min

### V5: Policy mini-train (10 steps)
- **输入**: D_demo (25条) + V4 标注的 D_syn (5条)
- **验证项**:
  - [ ] policy_updater.py 正确加载数据
  - [ ] compute_weighted_loss 计算成功，loss 有值
  - [ ] 10 steps 训练无报错，loss 下降
  - [ ] 策略 checkpoint 保存成功
- **预计耗时**: ~10min

### V6: Policy mini-eval (5 episodes)
- **输入**: V5 保存的策略 checkpoint
- **验证项**:
  - [ ] 策略加载成功
  - [ ] ManiSkill 环境 rollout 正常
  - [ ] success_rate 指标计算正确
- **预计耗时**: ~5min

### Phase 1.5 总计: ~1-2h

**门控**: 6 项全部通过 → 启动正式训练。任何一项失败 → 修复后重跑该项。

---

## Phase 2: 多轮迭代（LiftPegUpright-only） — 🔄 进行中

### Track B: Policy Pipeline 预验证（用 pretrained 模型并行）⬜ 新增

> **核心思路**: WM 正式训练需要 ~50h，但 pretrained WM 质量已经不错 (PSNR=22.35, SSIM=0.79)，
> zero-shot VLM 16帧 AUC=0.82 也有足够的区分力。先用它们跑完整 pipeline，
> 既验证下游所有环节，又能获得首个端到端策略改进结果。

| 步骤 | 内容 | 依赖 | GPU | 预计耗时 |
|------|------|------|-----|----------|
| B1 | Imagination: pretrained WM → 200 条合成轨迹 | Phase 1.5 通过 | 4-5 | 2-4h |
| B2 | VLM 标注: zero-shot 16帧 → 标注 B1 轨迹 | B1 | 6 | 1h |
| B3 | Policy 训练: Weighted FM (D_demo + D_real+ + D_syn+) | B2 | 8-9 | 2-4h |
| B4 | Policy 评估: 50ep, 对比 baseline 75% | B3 | 9 | 30min |

**B1-B4 可在 WM/VLM 正式训练期间并行执行**（使用不同 GPU），不浪费等待时间。

**预期结果**:
- 若 success_at_end > 75%（baseline）: pretrained 模型已足够，正式训练的 WM/VLM 只会更好
- 若 success_at_end ≤ 75%: 至少验证了 pipeline 无 bug，等正式 WM/VLM 完成后重跑 Track C
- 无论哪种情况，Track B 都不浪费 — 它消除了 "训练 50h 后才发现 policy 代码有 bug" 的风险

### Track A: 正式长时间训练（Phase 1.5 通过后启动）

| 步骤 | 内容 | GPU | 预计耗时 |
|------|------|-----|----------|
| A1 | WM iter1 全量微调 (DeepSpeed ZeRO-2, 2000 steps) | 0-3 | ~50h |
| A2 | VLM LoRA 16帧 (Accelerate 4GPU, 200 steps) | 4,5,7,8 | ~1-2h |

A2 完成后释放 GPU 4,5,7,8 → 可用于 Track B。

### Track C: 正式迭代（Track A 完成后）

用训练后的 WM + VLM 替换 pretrained 模型，重跑 Imagination → 标注 → Policy → 评估。
预期优于 Track B 结果（微调模型 > pretrained）。

### Iter 1 完整步骤表（调整后）

| 步骤 | 内容 | Agent | 依赖 | GPU | 状态 |
|------|------|-------|------|-----|------|
| 0 | 数据扩充 (re-encode 160 条) | Data-Agent | — | 4-5 | ✅ 完成 |
| **V1-V6** | **Phase 1.5: 最小规模全链路验证** | **各 Agent** | **Step 0** | **各** | **⬜ 新增，最高优先** |
| 1a | WM iter1 正式训练 (2000 steps) | WM-Agent | V1 通过 | 0-3 | ⬜ 等 Phase 1.5 |
| 1b | VLM 16帧评估 | Reward-Agent | — | 6 | ✅ 完成 |
| 1c | VLM LoRA 16帧正式训练 (200 steps) | Reward-Agent | V2 通过 | 4,5,7,8 | ⬜ 等 Phase 1.5 |
| 1d | Imagination env.step() 验证 | Imagination-Agent | — | 4-5 | ✅ 完成 |
| **B1** | **Imagination: pretrained WM → 200 条** | **Imagination-Agent** | **V3 通过** | **4-5** | **⬜ 与 1a 并行** |
| **B2** | **VLM 标注: zero-shot → 标注 B1** | **Reward-Agent** | **B1** | **6** | **⬜ 与 1a 并行** |
| **B3** | **Policy 训练: pretrained pipeline** | **Policy-Agent** | **B2** | **8-9** | **⬜ 与 1a 并行** |
| **B4** | **Policy 评估: pretrained pipeline** | **Eval-Agent** | **B3** | **9** | **⬜ 与 1a 并行** |
| 2 | WM iter1 质量验证（PSNR > 18 门控） | Eval-Agent | 1a | 0 | ⬜ 等 WM |
| 3 | 合成轨迹生成（微调 WM, 200-500 条） | Imagination-Agent | 2 | 4-5 | ⬜ |
| 4 | VLM 标注合成轨迹（微调 VLM） | Reward-Agent | 3, 1c | 6 | ⬜ |
| 5 | 策略更新（Weighted FM: D_demo + D_real+ + D_syn+） | Policy-Agent | 4 | 8-9 | ⬜ |
| 6 | 策略评估（50ep，对比 B4 + baseline） | Eval-Agent | 5 | 9 | ⬜ |
| 7 | 新策略 rollout 50 条 + VAE 编码 + VLM 标注 | Data-Agent + Reward | 6 | 4-6 | ⬜ |

**Iter 1 门控**:
- WM PSNR > 18 → 继续；否则调参重训
- 合成成功率 20-40% → 继续；否则检查 WM / VLM
- 策略 success_at_end > 基线 75% → 成功
- VLM LoRA 16帧 AUC > 0.85 → 继续；否则扩充数据或调参

### Iter 2

重复 2.1-2.8，数据集累积（D_real_iter1 + D_real_iter2），WM/VLM 继续微调。

---

## Phase 3: 评估与扩展

**派遣**: `Eval-Agent`

1. 生成 Base → Iter1 → Iter2 完整对比表（类似 VLAW Table 2）
2. LiftPeg 成功 → 扩展到 PickCube / StackCube
3. 可选消融（w/o WM grounding, w/o synthetic data, w/ env reward 等）

---

## Checkpoint 与磁盘管理

| 模型 | 大小/ckpt | 保留策略 |
|------|----------|---------|
| WM (Ctrl-World) | ~17GB | 只保留 best + latest（每轮 ~34GB） |
| VLM LoRA | ~24MB | 全部保留（很小） |
| 策略 | ~40MB | 全部保留 |
| 合成数据 | ~1-5GB | 保留当轮，下轮覆盖 |

**磁盘**: data/ 和 checkpoints/ 已软链接到 /mnt/disk_2 (15TB, 13TB free)，无空间压力。

---

## 执行节奏（实际 vs 计划）

```
Day 1 (02-26):  Phase 0 完成（数据审计+清除）+ Phase 1A/1B 完成 ✅
Day 2 (02-27):  Step 0 数据扩充 ✅ + Step 1b VLM 16帧评估 ✅
                Step 1a WM Iter1 训练启动（DeepSpeed 排障 5 次后稳定）
                Step 1a WM 崩溃 3 次（SIGHUP + ffmpeg ×2），tmux:wm_train 丢失
                Step 1c VLM LoRA 16帧重做启动 (单卡, tmux:vlm_train)
                Step 1c VLM 运行 2h 无训练输出（JIT 编译卡住），计划多卡重启
                Step 1d Imagination env.step() 验证完成 ✅
                计划调整：新增 Phase 1.5 最小验证 + Track B pretrained pipeline
Day 3 (02-28):  [上午] Phase 1.5 最小规模全链路验证 (V1-V6, ~2h)
                  - V1: WM mini 5步 + ffmpeg 修复
                  - V2: VLM mini 3步 + 多卡验证
                  - V3-V6: Imagination → VLM标注 → Policy训练 → 评估
                [Phase 1.5 通过后]
                  启动 Track A: WM 正式训练 (GPU 0-3) + VLM 正式训练 (GPU 4,5,7,8)
                  启动 Track B: pretrained pipeline (GPU 4-9, VLM训练完成后)
                Track B: B1 Imagination 合成 200 条 → B2 VLM标注 → B3 Policy训练
Day 3-4:        Track B 完成: 首个端到端策略结果
                Track A: WM 继续训练中 (~500-700/2000 steps)
Day 5 (03-01):  Track A: WM 完成 (~2000 steps)
                Track C: 微调 WM/VLM → Imagination → 标注 → 策略 → 评估
Day 5-6:        Track C 完成 + Iter 2
Day 6-7:        Phase 3 最终评估
```

### 时间预估明细

| 阶段 | 耗时 | GPU | 状态 |
|------|------|-----|------|
| Step 0: 数据扩充 | ~1h | 4-5 | ✅ 完成 |
| **Phase 1.5: 全链路验证** | **~2h** | **各** | **⬜ 最高优先** |
| Step 1a: WM 正式训练 | **~50h** (DeepSpeed ZeRO-2) | 0-3 | ⬜ 等 V1 通过 |
| Step 1b: VLM 16帧评估 | ~1h | 6 | ✅ 完成 |
| [A2] VLM LoRA 16帧正式训练 | **~1-2h** (4 GPU, grad_accum=32) | 4,5,7,8 | ⬜ 等 V2 通过 |
| **[B1] Imagination (pretrained WM)** | **2-4h** | **4-5** | **⬜ 与 WM 并行** |
| **[B2] VLM 标注 (zero-shot)** | **1h** | **6** | **⬜** |
| **[B3] Policy 训练 (pretrained)** | **2-4h** | **8-9** | **⬜** |
| **[B4] Policy 评估 (pretrained)** | **30min** | **9** | **⬜** |
| [C] WM 验证 | 30min | 0 | ⬜ 等 WM |
| [D-G] 微调 pipeline (Track C) | 4-8h | 4-9 | ⬜ 等 WM+VLM |
| **总计 到首个端到端结果 (Track B)** | **~8-12h** (Phase 1.5 + Track B) | | |
| **总计 到微调结果 (Track C)** | **~55h** (WM 占主导) | | |

---

## 关键注意事项

1. ~~**数据一致性**~~ ✅ 已解决
2. ~~**VLM 阈值**: α=0.8 无效~~ → 根因是**单帧**评估；16 帧多图 zero-shot AUC=0.82 ✅ 已确认改善
3. ~~**VLM 多帧输入**~~ ✅ 已实现并验证：16 帧 images 模式最优
4. **WM 起点选择**: ✅ Iter1 已从 pretrained 开始（22.35 dB, SSIM 0.79），而非 Phase-A
5. **WM 训练步数**: ✅ 已按数据量缩放 — 2000 steps × 8 / 4378 ≈ ~4 epochs
6. ~~**数据扩充是前置条件**~~ ✅ 已完成：235 条 / 4378 窗口
7. **并行策略**: Step 1a (WM) 和 Step 1b (VLM) ✅ 已并行执行
8. ~~**磁盘**~~ ✅ 已解决：15TB 挂载
9. **DeepSpeed 速度代价**: ZeRO-2 使 WM 训练慢 ~50% (~88s vs ~50-60s/step)，但这是**唯一能在 4×4090 上全量微调 1.5B 模型的方案**。ETA 从原计划 4-8h 延长到 ~47h
10. **WM 训练完成前的并行利用**: Track B 利用 pretrained 模型并行跑完整 Policy Pipeline，避免 50h 空等
11. **VLM images > video**: Qwen3-VL 对多图模式支持更好，后续标注统一使用 images 格式
12. **Phase 1.5 是硬性前置**: 任何正式训练之前必须通过最小规模全链路验证，避免长时间训练后才发现 bug
13. **Track B 双重价值**: 既验证 Pipeline，又产出首个端到端结果；若 pretrained 已足够，可能不需要等 WM 训练完成
