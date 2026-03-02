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
- **日期**: 2026-03-02
