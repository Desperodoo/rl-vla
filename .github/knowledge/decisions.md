# 架构决策记录 (ADR)

> 已固化到代码的决策用一行摘要。仍活跃影响决策的保留详情。

---

## 已固化（一行摘要）

| ADR | 决策 | 日期 |
|-----|------|------|
| ADR-001 | VLM: Qwen3-VL-4B-Instruct (8.3GB)，替换 Qwen2.5-VL-7B | 02-25 |
| ADR-002 | 2 相机**垂直拼接** → (384×192) → latent (4,48,24) | 02-24 |
| ADR-003 | Phase-A 仅训练 AE + temporal attn（已废弃，Iter1 改全量微调） | 02-24 |

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