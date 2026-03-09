# WM Imagination 采样参数消融实验报告

**日期**：2026-03-08
**WM Checkpoint**：iter1_v3_ext/checkpoint-1200（step 1200）
**Imagination 配置**：短时域 num_interact=4, T=20, 20 条轨迹/组
**对比可视化**：`data/vlaw/synthetic/ablation_comparison/`（原图 1172×1684）、`small/`（缩小版 586×842）

---

## 1. 实验背景

ADR-034 指出 WM 短时域 Imagination 存在"缺乏操作动态"问题：机械臂有运动但无法生成抓取/抬起等关键操作。本次消融实验旨在排查该问题是否可通过采样参数调整改善，测试了 3 组参数变体与 baseline 对比。

## 2. 实验设计

| 实验组 | num_inference_steps | guidance_scale | GPU | 耗时 | 说明 |
|--------|-------------------|----------------|-----|------|------|
| baseline | 25 | ~3.0（adapter 默认） | — | ~18 min | 已有数据（`wm_eval_step1200_short`） |
| E1-steps50 | **50** | ~3.0 | 3 | ~36 min | 去噪步数翻倍，预期质量提升 |
| E2-cfg5 | 25 | **5.0** | 8 | ~18 min | 更强条件引导 |
| E3-cfg1 | 25 | **1.0** | 9 | ~11 min | 无 CFG（跳过 unconditional 分支，速度 ~2x） |

三组实验在 GPU 3/8/9 上并行执行，每组生成 20 条合成轨迹 + 10 条可视化 strip。

## 3. 逐维度对比分析

### 3.1 物体持久性（Peg 可见性）

| 配置 | Top Camera | Side Camera |
|------|-----------|-------------|
| baseline | 前 2-3 帧可见红蓝 peg，后续帧 peg 逐渐消失或扁平化 | peg 在 3-4 帧后缩小/融入桌面 |
| steps50 | 与 baseline 相似，无显著改善 | **略优**，噪声更少，桌面纹理更清晰 |
| cfg5 | peg 色彩对比度略好，保持帧数更长 | peg 退化为深色团块而非完全消失 |
| cfg1 | 前 1-3 帧 peg 红蓝着色**最清晰** | 与 baseline 相当 |

**结论**：cfg1 和 cfg5 在 top camera 上 peg 保持略好，但所有配置在后期帧均出现 peg 退化。Side camera 一致较弱。

### 3.2 机械臂运动

| 配置 | 平滑度 | 运动幅度 | 说明 |
|------|--------|---------|------|
| baseline | 基准 | 正常 | — |
| steps50 | **最佳**，帧间过渡最平滑 | 正常 | 更多去噪步 → 更少噪声 |
| cfg5 | 略差 | 偶尔过度夸张 | 高 CFG 可能增强运动幅度 |
| cfg1 | 自然但约束最弱 | 正常 | 无 CFG 引导→更自由 |

### 3.3 操作动态（核心问题）

**所有 4 组配置均未能生成有意义的操作动态**（抓取、抬起 peg）。

具体表现：
- 机械臂向 peg 移动并下降，但未出现夹爪闭合
- Peg 始终停留于桌面，从未被抬起
- 后期帧中 peg 逐渐消失或与桌面融合

这证实了 **ADR-034 的判断**：操作动态缺失是 WM 模型能力问题，而非采样参数问题。

### 3.4 视觉质量与清晰度

| 配置 | 质量评级 | 说明 |
|------|---------|------|
| steps50 | ⭐⭐⭐⭐ | 最佳——更少噪声伪影，纹理更清晰 |
| cfg5 | ⭐⭐⭐ | 与 baseline 相当 |
| baseline | ⭐⭐⭐ | 基准水平 |
| cfg1 | ⭐⭐½ | 略差——细节缺少约束 |

### 3.5 推理速度

| 配置 | 相对速度 | 原因 |
|------|---------|------|
| cfg1 | **~2x 快** | `guidance_scale=1.0` → `do_classifier_free_guidance=False`，跳过 unconditional 前向传播 |
| baseline | 1x | — |
| cfg5 | ~1x | CFG 需要双倍前向传播，与 baseline 同速 |
| steps50 | **~0.5x 慢** | 去噪步数翻倍 |

## 4. 横向排名

| 评估维度 | 排名（最优→最差） |
|---------|----------------|
| 物体持久性（Top Camera） | cfg1 ≈ cfg5 > steps50 ≈ baseline |
| 物体持久性（Side Camera） | steps50 > baseline ≈ cfg5 ≈ cfg1 |
| 机械臂运动平滑度 | **steps50** > baseline > cfg5 > cfg1 |
| 视觉质量/清晰度 | **steps50** > cfg5 ≈ baseline > cfg1 |
| 操作动态 | **均未改善** |
| 推理速度 | **cfg1** > baseline ≈ cfg5 > steps50 |

## 5. 结论与建议

### 5.1 核心结论

**采样参数调整无法解决操作动态缺失问题。** 这是 WM ckpt-1200 的模型能力限制——需要更多训练或架构改进。

### 5.2 参数推荐

| 场景 | 推荐配置 | 理由 |
|------|---------|------|
| 最终合成数据生成 | **steps50**（num_inference_steps=50） | 最佳视觉质量和运动平滑度 |
| 快速原型/调试 | **cfg1**（guidance_scale=1.0） | 2x 速度，质量可接受 |
| 默认使用 | baseline（steps=25, cfg≈3.0） | 速度与质量平衡 |

### 5.3 后续行动

1. **继续 WM 扩展训练**（当前 step 1637/4000，GPU 4-7）——等 ckpt-2000 保存后，用 `steps50` 配置重新评估 imagination 质量，观察操作动态是否随训练改善
2. **ckpt-2000 评估若仍无改善**，考虑：
   - 训练数据增强（增加成功抓取轨迹比例）
   - 降低学习率以避免灾难性遗忘
   - 评估是否需要更长 num_frames 以捕获完整操作序列
3. **对 steps50 配置更新到 imagination pipeline 默认值**（已在 ADR-036 中为 `--num_inference_steps` 暴露 CLI 参数）

## 6. 文件索引

| 文件/目录 | 内容 |
|----------|------|
| `data/vlaw/synthetic/ablation_cfg1/` | cfg1 实验原始数据（20 轨迹） |
| `data/vlaw/synthetic/ablation_cfg5/` | cfg5 实验原始数据（20 轨迹） |
| `data/vlaw/synthetic/ablation_steps50/` | steps50 实验原始数据（20 轨迹） |
| `data/vlaw/synthetic/ablation_comparison/` | 4-way 对比图（1172×1684） |
| `data/vlaw/synthetic/ablation_comparison/small/` | 缩小版对比图（586×842） |
| `data/vlaw/synthetic/ablation_comparison_3way/` | 3-way 对比图（不含 steps50） |
| `scripts/viz_ablation_comparison.py` | 可视化对比脚本 |
| `scripts/launch_imag_ablation.sh` | 实验启动脚本 |
| `logs/vlaw/ablation_cfg1.log` | cfg1 实验日志 |
| `logs/vlaw/ablation_cfg5.log` | cfg5 实验日志 |
| `logs/vlaw/ablation_steps50.log` | steps50 实验日志 |
