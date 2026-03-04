# VLAW 数据重置 & 重做计划 (Fresh Start)

> **创建日期**: 2026-03-04
> **触发原因**: BUG-020 — `demo_prep.py` 将 `rgb_base` 复制为 `rgb_render`，导致全部 demo 数据双相机坍塌为单视角。WM/VLM/Imagination/Policy 全链路结果均被污染。
> **决策**: 彻底放弃官方 demos，用已有预训练策略定制数据，从头执行全部实验。

---

## 一、根因与污染范围

### BUG-020: demo_prep.py `rgb_render = rgb_base.copy()`

ManiSkill demo replay 只产出 `base_camera` 一个相机视角，`demo_prep.py` 第 296 行将 `rgb_base` 直接 copy 给 `rgb_render`。导致：

| 受影响资产 | 路径 | 问题 |
|------------|------|------|
| 原始 demo 数据 | `data/vlaw/demos/` | rgb_base ≡ rgb_render，diff=0.00 |
| 编码 demo 数据 | `data/vlaw/encoded/demos/` | latent_concat top-bot diff=0.057（应为 ~0.85）|
| WM iter1/ablation/optimal_steps 训练 | `checkpoints/vlaw/world_model/iter1,ablation_*,optimal_*` | **全部** 用 collapsed demos 训练 |
| WM 评估 | `results/vlaw/wm_*` | 用 demo traj 评估，PSNR/SSIM 不可信 |
| eval_fixed traj 0-4 | `data/vlaw/encoded/eval_fixed/` | demo 来源，collapsed |
| Imagination 初始帧 | timing_test 等 | 回退到 demos 后继承 bug |
| 全部 synthetic 数据 | `data/vlaw/synthetic/` | 基于污染 WM 生成 |
| 全部 labeled 数据 | `data/vlaw/labeled/` | 基于污染 synthetic 标注 |
| combined/flywheel | `data/vlaw/combined/` | 混入 demo + synthetic |

VLM 训练数据（rollouts）本身双相机是正确的（diff=56.68），但 VLM 消融的 gradient_accumulation 设置有独立 bug。

### 唯一干净资产

| 资产 | 路径 | 说明 |
|------|------|------|
| 预训练策略 | `checkpoints/il/best_eval_success_once.pt` | baseline success_rate ≈ 78% |
| WM 预训练权重 | `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt` | 8.7GB |
| WM 依赖模型 | `checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid/` | VAE/CLIP/ImageEncoder |
| VLM 基础模型 | `checkpoints/vlaw/reward_model/qwen_vl/` | Qwen3-VL-4B-Instruct |
| State Predictor | `checkpoints/vlaw/state_predictor/` | rollout 训练，理论干净 |
| 代码库 | `rlft/vlaw/`, `scripts/vlaw/` | 多轮 debug 完成 |

---

## 二、归档方案

所有旧数据/checkpoints/results 移入 `_archive/v1_contaminated/`，保持原子目录结构。

### data/vlaw/ 归档
```
demos/                → _archive/v1_contaminated/demos/
encoded/demos/        → _archive/v1_contaminated/encoded_demos/
encoded/eval_fixed/   → _archive/v1_contaminated/encoded_eval_fixed/
encoded/rollouts/     → _archive/v1_contaminated/encoded_rollouts/
rollouts/iter1/       → _archive/v1_contaminated/rollouts_iter1/
rollouts/iter1_highsuc/ → _archive/v1_contaminated/rollouts_iter1_highsuc/
labeled/              → _archive/v1_contaminated/labeled/
synthetic/            → _archive/v1_contaminated/synthetic/
combined/             → _archive/v1_contaminated/combined/
```

### checkpoints/vlaw/ 归档
```
world_model/iter1/                → _archive/v1/wm_iter1/
world_model/ablation_4000steps/   → _archive/v1/wm_ablation_4000steps/
world_model/ablation_optimal_steps/ → _archive/v1/wm_ablation_optimal_steps/
world_model/optimal_steps_v2/     → _archive/v1/wm_optimal_steps_v2/
reward_model/lora_iter1_16frame/  → _archive/v1/vlm_lora_iter1_16frame/
reward_model/ablation_*/          → _archive/v1/vlm_ablation_*/
reward_model/retrain_100steps_fixed/ → _archive/v1/vlm_retrain_100steps/
policy/iter1/                     → _archive/v1/policy_iter1/
policy/iter1_dryrun*/             → 直接删除
policy/flywheel_*/                → 直接删除（空目录）
```

### results/vlaw/ 归档
```
wm_*                  → _archive/v1/
vlm_steps_ablation/   → _archive/v1/
ablation/             → _archive/v1/
bc_*                  → _archive/v1/
dsyn_*                → _archive/v1/
wm_visual_comparison/ → _archive/v1/
```

---

## 三、重做计划

### Phase 0: 数据准备（~2h, GPU 4-5）

**0.1 用预训练策略收集新数据**

彻底放弃官方 demo，全部用预训练策略收集：

| 批次 | 目录 | 条数 | 策略 | 用途 |
|------|------|------|------|------|
| `rollouts/high_suc/` | 成功轨迹筛选 | ≥100 | `best_eval_success_once.pt` | WM 训正样本, VLM 正样本 |
| `rollouts/mixed/` | 不过滤保留全部 | ≥200 | 同上 | WM 混合训练, VLM 正负样本 |
| `rollouts/eval/` | 永不训练 | 20 | 同上 | 固定评估集 |

- 分辨率 192×192，双相机（`base_camera` + `env.render()`）
- 收集后立即验证 `rgb_base` vs `rgb_render` diff > 30

**0.2 VAE 编码**
- 全部新数据 → `encoded/train/`, `encoded/eval/`
- 生成新 `eval_set.h5`（全部双相机正确）

**0.3 更新 stat.json**
- 从新数据重新计算归一化统计

---

### Phase 1: WM 微调（~3h, GPU 0-3）

**1.1 训练**
- 数据：`encoded/train/`（high_suc + mixed, ≥300 traj）
- 起点：pretrained Ctrl-World checkpoint-10000.pt
- 2000 步，每 200 步保存 checkpoint
- `dataset_root_path = 'data/vlaw/encoded/train'`

**1.2 评估**
- 用 `encoded/eval/eval_set.h5`
- PSNR / SSIM / LPIPS 指标
- 数据量比旧版 25 条大 10 倍+，WM 质量预期显著提升

---

### Phase 2: VLM 微调（~1h, GPU 6-7）

**2.1 标注数据**
- 从 `rollouts/mixed/` 提取 rgb_base 帧序列
- 用 `env_success` 作为 ground truth 标签
- 确保正负样本均衡

**2.2 LoRA 训练**
- Qwen3-VL-4B, LoRA r=16（已确认最优）
- 200 步
- **gradient_accumulation_steps = 8~16**（上次消融发现 accum=128 导致退化）

**2.3 评估**
- 用 `rollouts/eval/` 评估
- 目标：ROC-AUC > 0.80, FP < 20% @ α=0.8

---

### Phase 3: Imagination & Labeling（~12h, GPU 0-3 + 6-7）

**3.1 合成数据生成**
- Phase 1 WM + 预训练策略
- `load_initial_frames` 从 `encoded/train/` 加载（双相机正确）
- 500 条轨迹

**3.2 VLM 标注**
- Phase 2 VLM 对 500 条评分
- D_syn+ 筛选（α=0.4 / 0.8 两档）
- 目标产出率 > 5%（≥25 条）

---

### Phase 4: 策略迭代（~4h, GPU 8-9）

**4.1 BC 飞轮验证**
- 100 条 rollout(高成功率) + D_syn+ → 合并训练
- 对比纯 rollout 基线
- 验证 D_syn+ 数据增强价值

**4.2 Imagination RL**（可选）
- 如 D_syn+ 产出率足够高则尝试

---

### Phase 5: 消融实验（优先级低，Phase 1-4 完成后）

**5.1 WM 步数消融**
- 数据修复后重新扫描 100~2000 步
- 用新 eval_set 评估，结论可能不同

**5.2 WM 帧数消融**
- num_frames = 5 vs 15

**5.3 VLM 步数消融**
- 100 / 200 / 400 / 800 步
- 修正 gradient_accumulation_steps 后重跑

**5.4 VLM LoRA rank 消融**
- r = 8 / 16 / 32 / 64
- 同上修正后重跑

---

## 四、目录结构规范

```
data/vlaw/
├── rollouts/
│   ├── high_suc/LiftPegUpright-v1/     # 高成功率数据
│   ├── mixed/LiftPegUpright-v1/        # 混合成功率数据
│   └── eval/LiftPegUpright-v1/         # 评估专用（永不训练）
├── encoded/
│   ├── train/LiftPegUpright-v1/        # 训练集 VAE 编码
│   └── eval/eval_set.h5               # 评估集 VAE 编码
├── synthetic/
│   └── iter1/                          # Imagination 合成数据
├── labeled/
│   └── iter1/                          # VLM 标注结果
├── combined/
│   └── flywheel/                       # 策略训练合并数据
├── meta_info/
│   └── maniskill/stat.json
└── _archive/                           # 所有旧数据
    └── v1_contaminated/
```

命名规则：
- **不用版本号前缀**（无 v1_/v2_ 等）
- 迭代轮次用 `iter{N}` 标注
- 同类数据按用途分目录（high_suc / mixed / eval）
- 评估集永不修改、永不训练

---

## 五、关键代码修复

| 文件 | 修改 | 优先级 |
|------|------|--------|
| `rlft/vlaw/data/demo_prep.py` | 标记 DEPRECATED，不再使用 | P0 |
| `scripts/vlaw/run/run_imagination_iter1.py` `load_initial_frames()` | 搜索路径改为 `encoded/train/{task}` | P0 |
| WM 训练配置 | `dataset_root_path` → `data/vlaw/encoded/train` | P1 |
| VLM 训练脚本 | `gradient_accumulation_steps` 改为 8~16 | P2 |
| `encoded/eval_fixed/` 引用 | 全部改为 `encoded/eval/` | P1 |
| `data/vlaw/encoded/README.md` | 重写 | P0 |

---

## 六、时间与 GPU 分配

| Phase | 耗时 | GPU | 依赖 |
|-------|------|-----|------|
| 0 数据收集+编码 | ~2h | 4-5 | 无 |
| 1 WM 微调 | ~3h | 0-3 | Phase 0 |
| 2 VLM 微调 | ~1h | 6-7 | Phase 0 |
| 3 Imagination+标注 | ~12h | 0-3 + 6-7 | Phase 1 + 2 |
| 4 策略迭代 | ~4h | 8-9 | Phase 3 |
| 5 消融实验 | ~6h | 灵活 | Phase 0（低优先级）|
| **合计** | **~22h（不含消融）** | | |

Phase 1 和 Phase 2 **可完全并行**（不同 GPU 组）。

---

## 七、验收标准

| 检查点 | 指标 |
|--------|------|
| 数据收集后 | rgb_base vs rgb_render diff > 30 |
| VAE 编码后 | latent top-bot diff > 0.5 |
| WM 微调后 | PSNR > 18 on eval_set |
| VLM 微调后 | ROC-AUC > 0.80, FP < 20% |
| Imagination 后 | D_syn+ 产出率 > 5% |
| 策略迭代后 | success_rate > baseline (78%) |
