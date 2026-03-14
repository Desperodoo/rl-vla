# 架构决策记录 (ADR)

> 已固化到代码的决策用一行摘要。仍活跃影响决策的保留详情。

---

## 已固化（一行摘要）

| ADR | 决策 | 日期 |
|-----|------|------|
| ADR-001 | VLM: Qwen3-VL-4B-Instruct (8.3GB)，替换 Qwen2.5-VL-7B | 02-25 |
| ADR-002 | 2 相机**垂直拼接** → (384×192) → latent (4,48,24) | 02-24 |
| ADR-003 | Phase-A 仅训练 AE + temporal attn（已废弃，Iter1 改全量微调） | 02-24 |
| ADR-028 | VLM v3 消融: r=16, 300步最优, α=0.5(平衡)/0.8(保守), r=8不可用 | 03-05 |
| ADR-029 | VLM eval 集必须正负平衡 (v2 仅10%正→recall=0 失败), v3 47%正 ✅ | 03-05 |
| ADR-030 | D_syn+ 产出率 v1→v3: 3.5%→33.3% (9.5×), 归因: v3数据+BUG-019修复+ckpt-400 | 03-05 |
| ADR-031 | Imagination viz 质量差=预期行为 (自回归误差累积, 非 bug) | 03-05 |
| ADR-033 | Imagination 5维度评估: latent OK (Δ<0.02), action ry bias 显著 (Δ=-0.516) 需监控 | 03-05 |
| ADR-034 | ⛔ Imagination 人工审核不可用, eval_WM PSNR 有误导性, WM 需继续训练, 阻塞所有下游 | 03-05 |
| ADR-035 | ACP 集成: Pistar06 value model (SigLIP+Gemma, 0.2% trainable) 提供 per-frame 稠密 advantage 权重 | 03-07 |

---

## 活跃决策

### ADR-004: State Predictor — 临时脚手架

- **⚠️ 临时**：仅跑通 Imagination 流程用。ManiSkill `env.step()` 精确可用，P4.3 已替换为 env.step() 版本。
- **当前状态**: `imagination_env.py` 已实现 env.step() 模式 ✅

### ADR-005: VLAWSuccessDataset 三级成功识别

按优先级过滤: ① `vlm_reward==1` → ② `success==True` → ③ `env_success.any()`

### ADR-006: ManiSkill 仿真定位

- `env.step()` = 本项目的"真实环境"
- Imagination + WM 的价值 = 评估 WM 质量 + Model-based RL 扩展
- 可通过 `num_envs=1..64` 系统测试数据效率

### ADR-007: WM Iter1 从 pretrained 开始（而非 Phase-A）

- pretrained H20: PSNR=22.35, SSIM=0.79
- Phase-A H20: PSNR=21.70, **SSIM=0.58** (退化严重, temporal attn 过拟合)
- Iter1 全量微调 1.5B UNet, DeepSpeed ZeRO-2, 2000 steps

### ADR-008: VLM 必须用 16 帧多图输入 → **修订: video 模式更优 (ADR-015)**

- 单帧 AUC=0.58 (接近随机) → 16帧 images AUC=0.82
- ~~images > video (可能因 Qwen3-VL 视频降采样丢帧)~~ **已推翻**: video AUC=0.83 > images 0.72 (见 ADR-015)
- α=0.8 阈值仅在 LoRA 微调后有效，zero-shot p_yes < 0.01

### ADR-009: Policy 架构适配 — 使用视觉 observations【✅ 已完成】

- **问题**: ShortCut Flow base ckpt 用 PlainConv 视觉编码器 (`global_cond_dim=626`)，VLAWPolicyUpdater 用 raw state (`dim=50`)，权重不匹配
- **决策**: 方案 1 — 适配 VLAWPolicyUpdater 使用视觉 obs (与 base ckpt 对齐)
- **实施**: UNet 默认参数修正为 (64,128,256)/dsed=64，导入 PlainConv 视觉编码器，实数据 10 步验证 loss=1.642，0 missing keys
- **日期**: 2026-02-28 决策 / 2026-03-01 完成

### ADR-010: WM 2000步 vs 4000步 — 2000步已是最佳

- **实验**: T-EXP-WM-01, 4000 steps 消融
- **结果**: ablation-2k PSNR=24.11/SSIM=0.841/LPIPS=0.098; ablation-4k PSNR=24.11/SSIM=0.818/LPIPS=0.107
- **结论**: 4000步无额外收益，SSIM/LPIPS 略有退步（过拟合迹象）。2000步是当前数据规模下的最佳步数
- **日期**: 2026-03-01

### ADR-011: VLM 帧数选择 — 16帧显著优于4帧/8帧

- **实验**: T-EXP-VLM-01, 4/8/16帧对比
- **结果**: 
  - 4帧 acc=0.706/FP=11.1%
  - 8帧 acc=0.735/FP=18.5%/ROC-AUC=0.8269
  - 16帧 acc=0.824/FP=3.7%/ROC-AUC=0.8084 (FP 更低)
- **结论**: 帧数对 VLM 判断质量影响显著，16帧是目前最佳。8帧 AUC 略高但 FP 率显著增大，综合考虑 16帧更安全
- **日期**: 2026-03-01

### ADR-012: Iter-1 策略退化分析 — 灾难性遗忘【活跃】

- **实验**: T-EVAL-ITER1-001, Iter-1 完整迭代后策略评估
- **结果**: baseline 78.1% → iter-1 17.2% (-60.9%)
- **背景**:
  - D_syn 200条 VLM 标注全部 vlm=0 (p_yes max=0.27 << α=0.8)
  - Weighted FM 2000步, lr=1e-5, loss 1.6→0.279
  - EMA 与 online 权重差异极小 (max_diff ~0.002), EMA 几乎无效
- **根因假说**:
  1. D_syn vlm=0 → Weighted FM 权重全为 0 或极小 → 实质是在随机噪声加权数据上微调
  2. lr=1e-5 对微调来说过高，2000步累积破坏了预训练表征
  3. 无 demo replay (原始演示数据共训练) → 缺少正则化锚点 → 灾难性遗忘
  4. EMA 衰减率过快 → 无法有效保留 base policy 知识
- **下一步**: 需要系统性消融 (降低 lr, 加入 demo replay, 减少步数, 调 EMA 衰减)
- **VLAW 论文参考**: Section 4.1 建议混合 D_demo + D_real + D_syn
- **日期**: 2026-03-01

### ADR-013: WM 最优步数搜索 — 减少步数 + 增加 Eval 频率

- **背景**: ADR-010 已证明增加步数 (2000→4000) 无额外收益，但未探索减少步数是否能更早达到最优点
- **当前问题**: eval 间隔 500 步过稀疏 (2000 步仅 4 个 eval 点)，无法精确定位 PSNR 拐点
- **方案**: 训练 2000 步但将 `validation_steps` 从 500 降至 100 (共 20 个 eval 点)，绘制 step-PSNR 曲线找最优步数
- **预期收益**: 若证明 500-1000 步已足够，后续 WM 迭代训练时间从 ~2h 缩短至 ~0.5-1h，显著加快 Pipeline 迭代周期
- **Task**: T-EXP-WM-05
- **日期**: 2026-03-01

### ADR-014: 策略微调方案 — 底层验证优先路线（修订版 v2）

- **背景**: Iter-1 Weighted FM 导致灾难性遗忘 (78.1% → 17.2%)，详见 ADR-012。D_syn+ = 0 是核心阻塞点。
- **决策 (v2 修订)**: 放弃激进跳转到 AWSC，改为底层到顶层的验证路线
- **修订原因**: v1 版直接跳转 AWSC 是"挑最强 RL baseline 叠加 WM augmentation"的激进策略，风险是绕过了关键假设验证 — 即"WM+VLM 合成数据到底有没有正面增益"。需要先用最简单的纯 BC 实验隔离验证数据飞轮假说。
- **修订后路线**:
  1. **阶段 A**: 诊断 D_syn+=0 根因 (WM 质量 vs VLM 泛化) → 人工审核合成帧 → 针对性修复
  2. **阶段 B1**: 纯 BC 验证数据飞轮 — 从 667 条官方 demo 中取子集从头训 ShortCut Flow (A 组=纯 demo, B 组=demo+D_syn+)，对比证明合成数据有增益
  3. **阶段 B2**: Policy-in-the-Loop Imagination RL — 将 ManiSkill 仿真替换为 WM+VLM (model-based RL, VLAW 论文核心思路)，在 RLPD/DSRL/PLD 管线中微调 BC 预训练策略
  4. **阶段 B3**: 迭代 WM+VLM 共同改进 — 策略提升→更好 rollout→微调 WM→更好合成数据→微调 VLM→循环
- **Go/No-Go 判据**: B1: B组 > A组 +3%; B2: success_once ≥ baseline 78%; B3: 连续 2-3 轮不退化
- **注**: AWSC/PLD/DSRL 的历史 sweep 数据仍作为参考上界 (AWSC=91% s_once with ManiSkill sim)
- **详细分析**: [`docs/vlaw/policy_finetuning_strategy_report.md`](../../docs/vlaw/policy_finetuning_strategy_report.md)
- **日期**: 2026-03-01 (v1) → 2026-03-01 (v2 修订)

### ADR-016: History Buffer 对齐官方 Ctrl-World — 列表式稀疏采样

- **背景**: WM autoregressive rollout 帧模糊调研发现我们的 imagination 实现与 Ctrl-World 官方代码有 3 处关键差异
- **差异 1**: 滑动窗口 `lat_buf[-window_len:]` vs 官方列表式 buffer + 稀疏采样 `history_idx=[0,0,-12,-9,-6,-3]`
- **差异 2**: 无第一帧锚定 (初始帧被挤出 buffer) vs 官方始终保留初始帧 (`history_idx[:2]` 永远指向真实首帧)
- **差异 3**: `num_history=4` vs 官方 `num_history=6` (DROID 配置)
- **决策**: 全部对齐官方做法 (T-WM-ALIGN-HISTORY)
- **修改范围**: 3 文件 4 处改动
  - `ctrl_world/config.py`: `num_history=4→6`
  - `scripts/vlaw/run/run_imagination_iter1.py`: `num_history=4→6`
  - `rlft/vlaw/world_model/imagination_env.py`: 列表式 latent_history/action_history + `history_idx=[0,0,-12,-9,-6,-3]` + 列表式追加更新 + policy 引用更新
- **验证计划**: T-IMAGINATION-002a (对齐) vs T-IMAGINATION-002b (滑动窗口) 消融对比，Go/No-Go: D_syn+ 多≥20% 或末帧 PSNR 高≥1dB
- **日期**: 2026-03-02

### ADR-017: Online RL Baselines 直接沿用 Sweep 数据，无需重训

- **背景**: T-RL-BASELINE-AWSC/PLD/DSRL 原计划在当前 base policy 上重新训练 online RL baselines
- **决策**: 直接沿用 `results.json` 中 `maniskill_sweep_v3/` 的 sweep 实验结果作为 baselines，不再重训
- **理由**: 这些 baseline 都是基于 ManiSkill 仿真器的 online RL，sweep 已覆盖多种超参组合，结果充分：
  - **AWSC**: best=aggressive 91% s_once / shortcut_weight_1.0 82% / reward_scale_0.5 85%
  - **PLD-SAC**: best=conservative 81% s_once / ema_0.99 81%
  - **DSRL-SAC**: 数据同在 sweep 中 (参考 `runs/dsrl_sweep/`)
  - **Flow Matching (BC)**: pred_horizon_8 71% s_once
  - **ShortCut Flow (BC)**: weights_1.0_1.0 64% s_once
- **影响**: T-RL-BASELINE-AWSC / PLD / DSRL 标记为 ✅ 跳过，T-AWSC-WM-AUGMENT 直接依赖 D_syn+ > 0
- **日期**: 2026-03-02

### ADR-018: WM 训练可缩短至 1000 步 — 节省 50% 训练时间

- **背景**: ADR-013 提出减少 WM 训练步数的可能性，T-EXP-WM-05 + T-EXP-WM-05-EVAL 完成了系统评估
- **实验**: 训练 2000 步 (validation_steps=100)，在 step 500/1000/1500/2000 保存 ckpt，逐个评估
- **结果**:

  | Step | PSNR | SSIM | LPIPS ↓ | Δ PSNR (vs iter1 2000步) |
  |------|------|------|---------|--------------------------|
  | 500 | 25.21 | 0.8814 | 0.0714 | +1.81 |
  | 1000 | **25.80** | **0.8909** | 0.0564 | **+2.40** |
  | 1500 | 25.25 | 0.8784 | 0.0608 | +1.85 |
  | 2000 | 25.87 | 0.8834 | 0.0546 | +2.47 |

- **分析**:
  - 1000 步已达到最佳 PSNR (25.87) 的 0.5 dB 容差内，且 SSIM 最优 (0.8909)
  - 1500 步出现轻微回落 (可能过拟合后恢复)，2000 步仅比 1000 步高 0.07 dB
  - **性价比最优**: 1000 步 = 2000 步质量的 99.7%，但节省 50% 训练时间
- **决策**: ~~后续 WM 微调默认训练 1000 步~~ → **降级为"特定条件下参考值"**
- **⚠️ 降级原因 (03-02 数据审计发现)**:
  - v1 实验使用 `num_frames=15` (iter1/imagination 推理用 `num_frames=5`), 训练数据为 reencode (含 110/160 无效轨迹)
  - 与 iter1 使用的 `num_frames=5, dataset_names="demos"` 完全不同，结论不可推广
  - **T-EXP-WM-05v2 正在重新实验**: num_frames=5, demos, 20 个 ckpt, 含 pretrained baseline
- **影响**: 暂缓采用 1000 步结论，等 v2 结果确认
- **报告**: `results/vlaw/wm_optimal_steps_eval/report.md` (v1), `docs/vlaw/data_audit_and_reorganization_proposal.md` (审计)
- **Task**: T-EXP-WM-05 (v1) + T-EXP-WM-05v2 (修复版)
- **日期**: 2026-03-02 (v1) → 2026-03-02 降级

### ADR-019: VLM video 模式 + BUG-019 为 D_syn+=0 根因

- **背景**: Iter-1 VLM 标注 200 条合成轨迹全部被拒绝 (p_yes max=0.27 << α=0.8)
- **发现 1**: VLM video 模式 (AUC=0.83) 远优于 multi-image (0.72), `use_video_format=True` 已是默认配置
- **发现 2**: BUG-019 — `load_initial_frames()` 使用 `torch.randn` 随机噪声作为初始 latent，导致所有合成轨迹从纯噪声起步
- **决策**: 修复 BUG-019 + 保持 video 模式 + 将 α 从 0.8 降至 ~0.4 (Youden 最优点)
- **验证**: 修复后 3/3 条帧质量正常，机械臂+peg 清晰可见
- **影响**: 旧 200 条合成数据无效，需重新生成 (T-IMAGINATION-002a/b)
- **日期**: 2026-03-02

### ADR-020: BC Scaling 增至 20K 步从零训练

- **背景**: T-BC-SCALING v1 使用 5000 步从零训练 ShortCut Flow, 6 组 demo count (25/50/100/200/400/669) 全部 success≈0%
- **分析**:
  - 预训练 baseline 使用 `total_iters=1_000_000` (1M 步), 5K 步仅为其 0.5%
  - ShortCut Flow 1D U-Net (~1.5M params) 从零训练需要足够迭代数才能收敛
  - 5K 步训练不充分导致结果无信息量
- **决策**: 增加训练步数至 **20K 步** (20× 较 v1), 仍从零训练
  - 20K 步 = 基线 1M 步的 2%, 是在合理时间预算内的最大步数
  - 每组 ~10min (总 ~1h), 远低于 1M 步 (~10h)
  - 若 20K 仍不够，考虑改为从预训练 ckpt 微调 (但与"数据飞轮验证"目标冲突)
- **新任务**: T-BC-SCALING-V2, GPU 8-9, 6 组 (25/50/100/200/400/669 demos)
- **Go/No-Go**: 若至少 669 demos 组能达到 >30% success, 则 scaling curve 有信息量
- **结果**: ✅ Go — 669d=0.40 > 30%. 完整 scaling: 25d=0.02, 50d=0.04, 100d=0.10, 200d=0.16, 400d=0.32, 669d=0.40
- **日期**: 2026-03-02

### ADR-021: History Buffer 对齐消融 — 无显著差异，保留当前做法

- **背景**: ADR-016 提出对齐官方 Ctrl-World 的列表式稀疏采样 history buffer (T-WM-ALIGN-HISTORY)，并设计了 T-IMAGINATION-002a (aligned) vs T-IMAGINATION-002b (sliding) 的消融实验
- **实验设计**:
  - **对齐组 (002a)**: 列表式 buffer + `[0,0,-12,-9,-6,-3]` + 第一帧锚定 + num_history=6, 200 条合成轨迹
  - **对照组 (002b)**: 滑动窗口 `lat_buf[-window_len:]` + num_history=4, 200 条合成轨迹
  - **其他条件一致**: 同 WM ckpt (iter1-2000), 同 policy (pretrained), 同初始 latent (BUG-019 修复后)
- **结果**:

  | 维度 | 对齐组 (002a) | 对照组 (002b) | 差异 |
  |------|--------------|--------------|------|
  | D_syn+ (α=0.4) | 6/200 | 7/200 | Δ=-1, 不显著 |
  | p_yes max | 0.500 | 0.531 | sliding 略高 |
  | p_yes mean | 0.179 | 0.153 | aligned 略高 (+17%) |
  | D_syn+ (α=0.8) | 0 | 0 | 一致 |

- **Go/No-Go 判定**: 未达到 "D_syn+ 多 ≥20% 或末帧 PSNR 高 ≥1dB" 的 go 标准。Δ=1 条差异在统计噪声范围内
- **结论**: 两种 history buffer 策略对 VLM 评分无显著差异。**WM 合成质量仍是主要瓶颈** — 400 条中仅 13 条通过 α=0.4 过滤 (3.25%)。history 对齐无法解决根本问题
- **决策**: 保留当前 sliding window 做法 (更简洁)，不强制切换到列表式。后续若 WM 质量提升使 D_syn+ 数量增大，可重新评估
- **影响**: T-WM-ALIGN-HISTORY 的代码修改保留但不作为默认路径
- **Task**: T-WM-ALIGN-ABLATION
- **日期**: 2026-03-03

### ADR-022: BC 飞轮实验设计 — 100K 步从头训练 + D_syn+ 混合

- **背景**: T-BC-SCALING-V2 证明 20K 步从零训练有信息量 (669d=0.40, Go/No-Go ✅), 但与预训练 baseline ~80% 仍有差距
- **方案**: 增至 100K 步 (10× SCALING-V2), 分 A/B 两组对比验证数据飞轮:
  - **A 组 (纯 demo)**: 100 条 / 669 条真实 demo → 100K 步 ShortCut Flow 从头训练
  - **B 组 (demo + D_syn+)**: A 组数据 + 13 条 VLM 筛选合成轨迹 (α=0.4) → 100K 步从头训练
- **结果 (A 组)**: 100d=0.10, 669d=0.54 (success_once, 100K 步)
- **Go/No-Go**: B 组 success_once 比 A 组提升 ≥ 3% → 证明数据飞轮有正面增益
- **注意**: D_syn+ 仅 13 条 (3.25% 通过率), 增益可能有限。关键是验证方向正确性而非绝对增益
- **新代码**: `scripts/vlaw/prepare_dsyn_plus_combined.py`, `scripts/vlaw/run_bc_flywheel_b_single.sh`
- **日期**: 2026-03-03
### ADR-023: 帧率与时间尺度不匹配 — Imagination 严重过长

- **背景**: 调研 VLAW/Ctrl-World 的 imagination 时间设定后发现, 当前 ManiSkill 复现与 DROID 原版在帧率和 rollout 时长上存在系统性偏差
- **发现 1 — WM 帧率不匹配 (6.67 Hz vs 5 Hz)**:
  - DROID: 15 Hz 原始 → `down_sample=3` → **5 Hz** (WM 预训练帧率, 帧间隔 0.2s)
  - ManiSkill: 20 Hz 原始 → `frame_skip=3` → **6.67 Hz** (帧间隔 0.15s)
  - WM 预期每帧间有 0.2s 运动量, 实际只有 0.15s → 动作幅度被系统性低估 25%
- **发现 2 — Imagination 时长远超任务时长 (最严重)**:
  - LiftPegUpright 中位数完成时间: **2.4 秒** (~16 帧 @ 6.67Hz ≈ 3.2 个 interaction)
  - 当前 Imagination: `12 interactions × 5帧 × 0.15s` = **9 秒** (60 帧)
  - **~73% 的帧是无效外推**, 图像质量持续退化, 拖低 VLM p_yes
  - 对比 DROID: imagination 12-20s ≈ 任务时长 12-20s → **1:1 匹配**
- **发现 3 — D_syn+ 低通过率 (3.5%) 的新解释**:
  - 不完全是 WM 本身质量不够, 而是让 WM 预测了远超必要的帧数, 后期退化帧拖垮整条轨迹的 VLM 评分
- **推荐方案 (优先级排序)**:
  1. **方案 A (最小改动)**: `num_interact` 从 12 缩短至 **4** (无需重新采集数据, 仅改 1 个参数)
  2. **方案 B**: 同时改 `frame_skip=4` 对齐 5Hz (需重新采集数据)
  3. **方案 C**: 全面重新设计 `num_frames/num_interact/num_history`
- **验证方式**: 先用 `num_interact=4` 生成 50 条合成轨迹 → VLM 标注 → 对比 D_syn+ 通过率
- **预期**: D_syn+ 从 3.5% 提升到 >10%, 生成速度提升 ~3×
- **详细报告**: [`docs/vlaw/frame_rate_timing_analysis.md`](../../docs/vlaw/frame_rate_timing_analysis.md)
- **日期**: 2026-03-03

### ADR-024: WM 相关任务全部阻塞, 直到 TIMING 问题解决

- **背景**: ADR-023 发现 Imagination 严重过长 (~73% 无效帧), 是 D_syn+ 低通过率 (3.5%) 的关键杠杆. 提升 D_syn+ 产出率优先于所有 WM 消融/评估/迭代
- **决策**: **所有 WM 相关实验/评估/imagination 生成/迭代循环全部阻塞**, 直到 TIMING 方案验证完成
- **阻塞任务列表**:
  - T-EXP-WM-05v2-EVAL (20 ckpt 评估)
  - T-EXP-WM-02~06 (WM 扩展消融)
  - T-IMAGINATION-003 (新合成数据生成)
  - T-MBRL-BC-FINETUNE (Imagination RL)
  - T-WM-ITER2 (第 2 轮迭代)
- **不受影响的任务**: VLM 消融 (T-EXP-VLM-02-EVAL, T-EXP-VLM-03-BUG), BC 飞轮分析 (T-BC-FLYWHEEL-EVAL)
- **解除条件**: T-TIMING-QUICKTEST 完成 → 无论 Go/No-Go, 根据结果决定后续 WM 任务是否恢复
- **WM finetuning 与 TIMING 关系分析**:
  - **方案 A (num_interact=4)**: 仅影响推理, **不影响 WM 训练**, 现有 WM ckpt 全部可复用
  - **方案 B (frame_skip=4)**: 改变数据帧率 6.67Hz→5Hz, **需要推倒重来**: 重新采集→重新 VAE 编码→重新训练 WM→重新计算 stat.json
  - **方案 B 的好处**: frame_skip=4 对齐 DROID 5Hz, 与 pretrained 权重先验更匹配, 理论上 finetuning 收敛更快
- **执行顺序**: 方案 A → 验证有效 → 方案 B → 数据+WM 推倒重来

### ADR-025: TIMING 方案 A No-Go — 保持 num_interact=12

- **背景**: ADR-023 提出 `num_interact` 从 12 缩短至 4 以减少无效帧（方案 A）
- **实验**: T-TIMING-QUICKTEST — 50 条合成轨迹，`num_interact=4`（20帧/条），VLM 标注
- **结果**: D_syn+(α=0.4) = **0/50** (更差！旧方案 7/200 = 3.5%)
- **分析**: 20 帧太短，LiftPegUpright 完成动作需更多时间。缩短 rollout 导致 VLM 无法观察到成功迹象
- **决策**: Plan A 失败，**保持 `num_interact=12`**。ADR-024 阻塞解除
- **日期**: 2026-03-04

### ADR-026: 数据全面重置（Fresh Start）

- **背景**: BUG-020 发现 `demo_prep.py` `rgb_render = rgb_base.copy()` 导致全部 demo 数据双相机坍塌
- **污染范围**: demos → encoded/demos → WM 全部训练 → imagination → synthetic → labeled → combined → policy → 所有评估结果
- **唯一干净数据**: rollout 数据（`data_collector.py` 使用 `env.render()` 获取独立 render_camera）
- **决策**: 
  1. 彻底放弃官方 demo，用预训练策略收集新双相机数据
  2. 全部旧数据/ckpt/results 归档到 `_archive/v1_contaminated/` 和 `_archive/v1/`
  3. 重新执行全部实验（WM/VLM/Imagination/Policy）
- **VLM 独立 bug**: gradient_accumulation=128 导致 VLM 消融结果异常，与 BUG-020 独立但同时修复
- **预期收益**: 
  - 数据量从 25 条 demo → ≥200 条 rollout（10×增长）
  - 双相机正确 → WM 学到正确的多视角分布
  - 两个 bug 同时修复 → 所有消融实验结论可信
- **详细计划**: [VLAW_FRESH_START_PLAN.md](../VLAW_FRESH_START_PLAN.md)
- **日期**: 2026-03-04
- **日期**: 2026-03-03

### ADR-027: 数据采集方案 A — 大量采集消除 Selection Bias

- **背景**: v3 pilot 50 条数据 success_at_end=100% (BUG-024)，根因是成功 episode 早终止 + 采集数量不够
- **真实成功率**: AWSC checkpoint → success_once=80%, success_at_end(200步)=46% (eval 脚本独立验证)
- **ManiSkill3 环境行为确认**:
  - `BaseEnv.step()` L1054: `terminated = info["success"].clone()` — 成功即终止
  - `@register_env("LiftPegUpright-v1", max_episode_steps=50)` — 官方默认仅 50 步
  - collector 使用 `max_episode_steps=200` 覆盖默认值
  - 成功轨迹 10-120 步完成，失败轨迹 200 步 truncated
- **方案**: 大量采集 num_episodes=1200+
  - 64 env 并行，每轮先完成 ~51 个成功 ep (快速) + ~13 个失败 ep (200步)
  - 多轮后失败轨迹自然混入，最终比率趋近真实分布
  - 预期: ~80% success, ~20% failure (适合 WM/VLM 训练)
- **min_traj_length=5**: 放宽阈值 (frame_skip=4 下部分成功轨迹 T 较短)
- **用户确认**: 视觉检验 pilot 数据 GIF/strip 全部正确 ✅
- **日期**: 2026-03-05

### ADR-028: VLM v3 消融结论 — r=16, 300步, α=0.5~0.8【已固化】

- **背景**: v3 数据 (frame_skip=4, 1200条 mixed) 上完成 VLM 全面消融，包含 Steps 消融、Threshold 消融、LoRA Rank 消融
- **实验**:
  - **Steps 消融** (r=16, α=0.8): 50/100/150/200/300/400 steps
  - **Threshold 消融** (r=16, 300 steps): α=0.3~0.9
  - **LoRA Rank 消融**: r=8 (α_lora=16) vs r=16 (α_lora=32)
- **结果摘要**:

  | 维度 | 最佳配置 | 关键指标 |
  |------|---------|---------|
  | Steps | **300步** | acc=81.7%, prec=100%, recall=61.2%, FP=0% (α=0.8) |
  | Threshold (保守) | **α=0.8** | prec=100%, recall=61.2%, FP=0% |
  | Threshold (平衡) | **α=0.5** | acc=86.7%, prec=85.9%, recall=85.9%, FP=12.6% |
  | LoRA Rank | **r=16 唯一可用** | r=8 recall=1.2% (容量不足) |

- **关键发现**:
  1. **200步是"突变点"**: 模型从不可用 (recall=0%) 跳到可用 (42.4%)
  2. **300步是"甜蜜点"**: recall +18.8pp (42.4→61.2%), FP 仍为 0%
  3. **400步开始过拟合**: recall 71.8% 但引入 2.1% FP
  4. **r=8 完全不可用**: 仅 1.2% recall，容量瓶颈
  5. **α=0.5 是最佳平衡点**: 在 300步模型上 acc=86.7%, 双向 recall/precision ≈86%
- **决策**: 
  - VLM 训练默认 **300步** (替代原 200步)，r=16 不变
  - Imagination 标注推荐 α=0.5 (平衡) 或 α=0.8 (保守)，视 D_syn+ 数量需求而定
  - r=8 排除，不再考虑
- **消融报告**: `results/vlaw/vlm_ablation_v3_report.md`
- **日期**: 2026-03-05

### ADR-029: VLM 评估集正负平衡是训练成功的前提条件【已固化】

- **背景**: VLM v2 (lora_v2) 在旧数据上训练，eval 集仅 12 正 / 108 负 (10% 正样本)。v3 用新 mixed 数据，eval 集 85 正 / 95 负 (47% 正样本)
- **v2 失败模式详解**:
  - 5 个 checkpoint (50/100/150/200/final) **全部** TP=0, FP=0, recall=0%
  - mean_p_yes 最高仅 0.036 (远低于 α=0.8)，模型从未学会预测 "成功"
  - accuracy=90% 是假象 — 全预测 "否" 在 90% 负样本集上就能 90%
  - 根因: 训练集正样本过少 (frame_skip=3 数据 + 旧 eval 分布偏斜) → 模型收敛到 "永远说否" 的局部最优
- **v3 vs v2 对比** (相同训练步数 200 步, r=16):
  - v2: recall=0%, precision=0, mean_p_yes=0.036
  - v3: recall=42.4%, precision=1.0, mean_p_yes=0.565
  - **唯一变量**: 数据质量 (帧率修正 + 正负平衡)
- **决策**: VLM 训练/eval 集正样本比例应在 **30%-60%** 范围内。低于 20% 时模型大概率坍塌到全预测负
- **预防措施**: 数据收集后、VLM 训练前，必须检查 eval 集 success_at_end 比率，若 <20% 需补充正样本或调整 eval split
- **日期**: 2026-03-05

### ADR-030: D_syn+ 产出率 v1→v3 从 3.5% 提升至 33.3% (9.5×)【活跃】

- **背景**: BDC-B (Imagination 预验证) 使用 ckpt-400 生成 15 条合成轨迹 + VLM LoRA 300步 标注
- **v1 基线**: D_syn+ ≈ 1/28 (3.5%) — 使用 frame_skip=3 旧数据 + BUG-019 未修 + 旧 WM ckpt
- **v3 结果**: D_syn+ = 5/15 (33.3%) — 使用 frame_skip=4 v3 数据 + BUG-019 已修 + ckpt-400 (PSNR=29.76)
- **p_yes 分布**: mean=0.42, std=0.17, max=0.68, median=0.44
- **α=0.5 通过的轨迹**: traj_0001(0.68), traj_0007(0.68), traj_0010(0.56), traj_0011(0.62), traj_0014(0.62)
- **提升归因**:
  1. **v3 数据帧率修正** (frame_skip=4 → 5Hz 精确匹配 WM) — 消除 timing mismatch
  2. **BUG-019 修复** — 真实首帧 latent 替代随机 latent
  3. **ckpt-400 质量提升** — PSNR=29.76 vs pretrained 22.33 (+7.43 dB)
- **影响**: 33.3% 产出率意味着正式 Phase 3 生成 200 条可期望 ~66 条 D_syn+，对策略训练已有足够价值
- **注意**: 此结果基于 ckpt-400 (20% 训练)，完整训练后 D_syn+ 产出率可能进一步提升
- **决策**: v3 数据 + BUG-019 修复 + WM 微调的组合方向正确，继续推进 Phase 3-5
- **输出**: `data/vlaw/labeled/precheck_ckpt400/`, `data/vlaw/synthetic/precheck_ckpt400/`
- **日期**: 2026-03-05

### ADR-031: Imagination Viz 质量差是预期行为 (自回归误差累积)【已固化】

- **背景**: 用户质疑 Imagination 生成的视频帧质量差 (模糊、失真)，怀疑存在 bug
- **分析方法**: 对比 eval_wm.py 单步预测 vs run_imagination.py 12 轮自回归生成
- **根因**: 自回归误差累积 (主因) + float16 VAE decode (次因)
  - eval_wm.py 单步: GT history 6 帧 → predict 5 帧 → PSNR=29.88 (高质量)
  - imagination: 每轮 pred[-1] → 下轮 history，12 轮后 history 完全由预测帧构成
  - 误差逐轮放大是 autoregressive video prediction 的固有特性
- **Frame 0 均值膨胀**: Frame 0 (~40 dB) 是 conditioning 帧，非真正预测；排除 F0 后实际 F1-F4 ≈ 26.7 dB
- **latent 稳定性**: F0 std=0.9187, F30 std=0.9211, F59 std=0.9265 — 未发散
- **决策**: Imagination viz 质量差不影响管线 (VLM 在 latent 空间判定, 非 pixel)，继续用 ckpt-400 推进
- **附属变更**: eval_wm_deep_viz.py 功能合并到 eval_wm.py (397→580 行)，deep viz 脚本已删除
- **详细分析**: `knowledge/wm-eval-analysis.md`
- **日期**: 2026-03-05

### ADR-032: D_syn+ 正式产出率 61.0% — 远超预验证 (33.3%) 和门控 (>5%)【活跃】

- **背景**: Iter-1 Step 5-6 正式运行 200 条 Imagination + VLM 标注
- **结果**: D_syn+ = 122/200 (61.0%), p_yes mean=0.5596, max=0.9399
- **对比 precheck**: 15 条样本时为 33.3% → 200 条时为 61.0% (样本量更大更稳定)
- **门控阈值**: D_syn+ 产出率 > 5% ✅ (实际 61.0%, 12.2× 超越门控)
- **对策略训练的意义**: D_real+(434) ∪ D_syn+(122) = 556 条正样本可用于 Weighted FM 训练
- **主要改进来源**: 正式运行时生成质量整体更高 (4-GPU 并行无 GPU 争用, 批量更大)
- **决策**: 数据充分，立即进入 Phase 4 策略更新
- **输出**: `data/vlaw/labeled/iter1_syn/`, `data/vlaw/synthetic/iter1/`
- **日期**: 2026-03-05

### ADR-034: ⛔ Imagination 人工审核不可用 — eval_WM PSNR 有误导性, WM 需继续训练【活跃-阻塞】

- **背景**: Iter-1 完成 Imagination 200 条生成 + 5 维度自动化评估后，用户进行了人工视觉审核
- **核心结论**:
  1. **Imagination 生成几乎完全不可用** — 人工审核的视频质量远低于可用标准，自动化指标 (latent Δ<0.02, L2 drift<1%) 与肉眼观感严重脱节
  2. **eval_WM PSNR=29 具有误导性** — 单步预测在完美 GT history 条件下获得高分，但该指标不能反映 Imagination 自回归 rollout 的实际质量
  3. **WM 需要继续训练** — 即使 eval_WM 指标无明显变化，Imagination 效果也可能因 WM 质量提升而改善，需人工判断
- **决策**:
  - ⛔ **阻塞所有下游环节**: Phase 4 策略更新、Phase 5 评估、Iter-2 全部暂停
  - ✅ **启动 WM 继续训练** (Phase 1b): 从 pretrained 或 ckpt-400 resume, 训练更多步数 (4000+), 保存更多 checkpoint
  - ✅ **WM 评估指标升级**: 不再以 eval_WM PSNR 为唯一门控，必须在每个关键 checkpoint 运行 Imagination + VAE decode 可视化 + 人工审核
  - ✅ **解除条件**: 某个 WM checkpoint 的 Imagination 可视化结果经人工确认为"可用"后，方可恢复下游
- **对 ADR-010 的影响**: "2000 步已是最佳" 基于 eval_WM PSNR，现已证明该指标不可靠。需要重新探索更长训练步数
- **对 ADR-031/033 的影响**: 之前的 "Imagination viz 差是预期行为" 和 "latent 质量 OK" 结论需要修正 — 自动化指标不能替代人工审核
- **WM 训练计划**:
  - GPU 0-3, DeepSpeed ZeRO-2
  - 从 pretrained 开始完整训练 4000 步以上
  - 每 200 步保存 checkpoint (确保不再丢失中间 ckpt)
  - 每 400-800 步在关键 checkpoint 上运行 Imagination 快速评估 (10-20 条) + 可视化
- **Task**: T-WM-V3-EXTENDED
- **日期**: 2026-03-05

### ADR-033: Imagination 全面评估结论 — latent 质量 OK, action ry bias 需监控【活跃】

- **背景**: Iter-1 200 条合成轨迹完成后，用 `eval_imagination.py` (1160 行) 进行 5 维度全面评估
- **评估脚本**: `rlft/vlaw/scripts/eval_imagination.py`
- **评估输出**: `results/vlaw/imagination_eval/` (report.md + full_results.json + 21 PNG/JSON)
- **5 维度评估结果**:

  | 维度 | 关键指标 | 结论 |
  |------|---------|------|
  | **Latent 统计** | Δmean<0.02, Δstd<0.01, L2 drift<1% | ✅ 分布匹配良好 |
  | **VAE Decode 质量** | sharpness 衰减 27% (round 0→11) | ⚠️ 预期行为 (自回归累积) |
  | **Action 分析** | **ry 维度偏移: syn=-0.76 vs real=-0.25 (Δ=-0.516)** | ⚠️ 显著, 需监控 |
  | **State 轨迹** | z 坐标终端偏高 (0.102 vs 0.088) | ℹ️ 轻微 |
  | **VLM 标注分解** | p_yes 集中在 0.5-0.7, 122/200 通过 α=0.5 | ✅ 产出率 61% |

- **关键发现**:
  1. **Latent 质量 OK**: 通道分布与真实数据匹配 (Δ<0.02), L2 drift 随 round 增长但未发散 (<1%)
  2. **Action ry bias 显著**: 合成轨迹 ry 维度 (mean=-0.76) 相比真实 (mean=-0.25) 偏移 Δ=-0.516，可能因 WM 对旋转动作的预测偏差传播到 policy 输出
  3. **11/200 条轨迹有 frozen dim4**: action dim4 方差≈0（固定值），疑似 policy 对特定初始状态的退化响应
  4. **VLM p_yes 集中在 0.5-0.7**: 真正高信心 (>0.8) 的轨迹很少，α=0.5 是合理阈值
  5. **History buffer 非"全为 pred"**: 2/7 条件帧始终保持真实帧锚定 (Pos[0,1]), 修正了此前"全部变为 predicted"的不准确描述
- **决策**:
  - action ry bias 暂不 block Iter-1 (数据充分, 策略训练后评估实际影响)
  - 若策略评估发现旋转动作异常，优先排查 ry bias 来源 (WM vs policy)
  - frozen dim4 比例 5.5% 可接受，后续可通过 action smoothness 过滤
- **影响**: Imagination 数据质量满足 Phase 4 策略训练要求，不阻塞
- **日期**: 2026-03-05

### ADR-035: ACP 集成 — Pistar06 Value Model 稠密 Advantage 权重【活跃】

- **背景**: VLAW 原始方案使用 VLM 二值 filtering（成功/失败），粒度为轨迹级。ACP（Advantage-Conditioned Policy）引入 per-frame 稠密 advantage 权重，将 Pistar06 value model（源自 Evo-RL）移植到 ManiSkill3 环境
- **架构**:
  - **Vision encoder**: SigLIP-so400m-patch14-384（~428M params, 冻结）
  - **Language model**: Gemma-3-270m（~268M params, 冻结）
  - **可训练组件**: image projector + language projector + LayerNorm + distributional value head (201 bins) — 共 ~1.55M params (0.2%)
  - **双相机处理**: rgb_base 和 rgb_render 分别输入 SigLIP（128x128 resize 到 384x384），mean-pool 合并，不做竖拼
  - **Value target**: `target = clip((-remaining_steps - c_fail*(1-success)) / (max_len+c_fail), -1, 0)`
  - **Advantage**: N-step (n=4) advantage + per-task quantile binarization (positive_ratio=0.3) + 归一化为 [0,1] 连续权重
- **代码位置**: `rlft/vlaw/acp/`（7 个源文件）, CLI: `rlft/vlaw/scripts/run_acp_{train,infer}.py`
- **HDF5 产出字段**: `acp_value_target`, `acp_value_pred`, `acp_advantage`, `acp_indicator`, `acp_weight` (per-frame) + 3 group attrs
- **Policy 集成**: `PolicyUpdaterConfig.use_acp_weights=True` 时，`VLAWSuccessDataset` 从 HDF5 读取 `acp_weight` 字段，取 action 窗口内 per-frame 权重的均值作为样本权重，传入 `compute_weighted_loss()`
- **Conda 环境**: 复用 `vlaw_reward`
- **GPU 需求**: 单卡 ~3GB VRAM（4090 可用）
- **验证状态**: 28/28 单元测试通过, GPU dry-run 20 步 MAE=0.271, positive_ratio=0.300
- **success_key 配置**: 支持 `env_success`（仿真 GT, per-frame）和 `vlm_success`（VLM 标注, per-trajectory）两种模式
- **质量门控**: 正式训练 8000 步后 value MAE < 0.05, advantage positive_ratio ~30%
- **日期**: 2026-03-07

### ADR-036: Pipeline 参数优化 — 全链路加速【活跃】

- **背景**: 审计所有 pipeline 默认参数后发现多处低效设置：WM DataLoader worker 不足、ACP frozen backbone 运行在 float32、VLM DataLoader 单进程、Imagination 推理步数不可调节。系统优化以减少 wall-clock 时间且不影响模型质量
- **变更清单**:
  1. **WM 训练** (`scripts/vlaw/run/train_wm_v3_ext.sh`): `--num_workers 4→8` (CPU 余量充足); 添加 GPU 扩展文档 (4→8 GPU 时 `GRAD_ACCUM=4` 保持 eff_batch=32)
  2. **Imagination** (`rlft/vlaw/scripts/run_imagination.py`): 新增 `--num_inference_steps` CLI 参数 (默认 25, 支持 10-15 快速评估); `load_wm()` 和 `generate()` 函数签名扩展
  3. **ACP** (`rlft/vlaw/acp/config.py`, `train_value_model.py`, `value_model.py`): dtype 默认 `float32→bfloat16`; 训练/验证/推理循环添加 `torch.cuda.amp.autocast(dtype=torch.bfloat16)`; projector+value head 保持 float32 精度
  4. **VLM 训练** (`rlft/vlaw/reward/train_reward_model.py`): DataLoader `num_workers=0→2, persistent_workers=True`
  5. **VLM 推理** (`rlft/vlaw/reward/reward_model.py`): `use_flash_attention` 默认 `False→True`
  6. **Policy 训练** (`rlft/vlaw/policy/policy_updater.py`): visual encoder forward 包装 `torch.cuda.amp.autocast(dtype=torch.bfloat16)`, 输出 `.float()` 回转
- **预期加速**: ACP ~1.5-2x, VLM 训练 ~20-30% (IO-bound 改善), VLM 推理 ~15-25% (flash_attn), Imagination eval 可选 ~2x (步数减半)
- **风险评估**: 全部为 frozen backbone 或 IO 层面优化, zero 精度风险。唯一需要人工确认的是 flash_attn 包在 vlaw_reward env 中是否已安装
- **验证计划**: ACP bf16 dry-run 20步 对比 fp32 MAE; Imagination steps=15 vs 25 PSNR; Policy bf16 50步 loss 不发散
- **日期**: 2026-03-08
### ADR-043: Imagination 推理 Action Conditioning 修复 + WM v5 审查【活跃】

- **背景**: WM v5 训练完成（4000 steps, BUG-A/B/C 全修复），20 checkpoint 并行 imagination eval 全部完成。所有 checkpoint 的 peg 动态评分 1/10（几乎完全静止），与 v4 一致。训练验证样本（GT actions）确认 WM 本身具备动态建模能力。
- **根因诊断**:
  1. **BUG-D [CRITICAL]**: `imagination_env.py` 推理时 future 5 帧全部 tile 当前 EE pose（L524-528），等于告诉 WM "臂不动"。训练时 WM 接收每帧不同的真实 EE 位姿
  2. **BUG-E [HIGH]**: V5 latents 与 V4 几乎相同（corr=0.999999），BUG-C 修复对编码端无实质影响
  3. **BUG-F [HIGH]**: ManiSkill 数据无 DROID 风格时间跳跃增强（skip=1 vs DROID skip=1-2）
  4. **BUG-G [HIGH]**: Action Encoder (2.11M) 随机初始化，与 UNet (1524M) 共用 LR=1e-5，无 warmup
  5. **BUG-H [MEDIUM]**: ee_pose_history 初始化仅 1 条（应为 num_history*4=24）
- **修复**:
  - BUG-D: `integrate_delta_to_ee_poses()` 将 policy delta actions 累积积分为绝对 EE pose 序列
  - BUG-H: ee_pose_history 初始化改为 num_history*4 条
  - BUG-F/G: 待 BUG-D 修复验证后决定是否需要重训练 WM v6
- **验证**: 修复后在 step 3400/3800/4000 重跑 imagination eval (各 20 trajs, 10 viz)
- **v5 视觉审查结论**: 场景重建 7/10, 臂动态 5/10, peg 动态 1/10, 不可用于下游 imagination pipeline
- **日期**: 2026-03-14
