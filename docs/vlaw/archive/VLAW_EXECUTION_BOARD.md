# VLAW 执行看板（Execution Board）

> 目的：仅追踪"当前正在推进"的任务，不记录历史流水。
> 策略：**LiftPegUpright-only** 验证，PickCube/StackCube 延后。
> 推进计划详见 [VLAW_NEXT_STEPS.md](VLAW_NEXT_STEPS.md)。

## 当前阶段
- phase: Phase 2 — Iter 1（串行依赖链）
- owner: VLAW-Coordinator
- 更新规则：任务状态变化时更新；详细过程写入 logs 与 work-logs。

## 任务列表

### Phase 0 — 数据质量审计（前置）

| task_id | 模块 | owner_agent | 依赖 | 验收标准 | 状态 |
|---|---|---|---|---|---|
| T-AUDIT-001 | 全量 HDF5 数据审计 | Eval-Agent | — | latent shape 检查报告 + 异常文件列表 | ✅ 完成 |
| T-DATA-FIX-001 | 清除异常数据 | Coordinator | T-AUDIT-001 | 删除shape异常编码文件，剩余全部(4,48,24) | ✅ 完成 |

### Phase 1 — 微调前基线报告（可并行）

| task_id | 模块 | owner_agent | 依赖 | 验收标准 | 状态 |
|---|---|---|---|---|---|
| T-WM-BASELINE-001 | WM pretrained 基线评估 | WM-Agent | Phase 0 | PSNR/SSIM/LPIPS 多 horizon 报告 | ✅ 完成 |
| T-VLM-BASELINE-001 | VLM zero-shot vs LoRA 基线评估 | Reward-Agent | Phase 0 | ROC-AUC + confusion matrix + 最优阈值 | ✅ 完成 |

### Phase 2 — Iter 1（串行依赖链）

| task_id | 模块 | owner_agent | 依赖 | 验收标准 | 状态 |
|---|---|---|---|---|---|
| T-WM-ITER1-001 | WM iter1 微调 | WM-Agent | Phase 1 | ckpt + PSNR > 18 | 🔄 进行中 |
| T-WM-ITER1-EVAL | WM iter1 质量验证 | Eval-Agent | T-WM-ITER1-001 | PSNR > 18 门控通过 | ⬜ |
| T-IMAGINATION-001 | 合成轨迹生成 (200条) | Imagination-Agent | T-WM-ITER1-EVAL | 可用合成数据 | ⬜ |
| T-REWARD-SYN-001 | VLM 标注合成轨迹 | Reward-Agent | T-IMAGINATION-001 | 标注 + 成功率 20-40% | ⬜ |
| T-POLICY-001 | 策略 Iter1 更新 | Policy-Agent | T-REWARD-SYN-001 | 训练完成 | ⬜ |
| T-EVAL-ITER1-001 | Iter1 策略评估 | Eval-Agent | T-POLICY-001 | success_at_end > 75% 基线 | ⬜ |
| T-DATA-ITER1-002 | 新策略 rollout + VAE 编码 | Data-Agent | T-EVAL-ITER1-001 | 50条 + 编码 | ⬜ |

### Phase 2 — Iter 2（Iter 1 完成后）

| task_id | 模块 | owner_agent | 依赖 | 验收标准 | 状态 |
|---|---|---|---|---|---|
| T-WM-ITER2-001 | WM iter2 微调 | WM-Agent | T-DATA-ITER1-002 | ckpt + PSNR > 18 | ⬜ |
| T-EVAL-ITER2-001 | Iter2 全流程评估 | Eval-Agent | … | 对比 Iter1 | ⬜ |

### Phase 3 — 最终评估与扩展

| task_id | 模块 | owner_agent | 依赖 | 验收标准 | 状态 |
|---|---|---|---|---|---|
| T-EVAL-FINAL | Base→Iter1→Iter2 完整对比 | Eval-Agent | Iter2 完成 | 报告 | ⬜ |
| T-DATA-PICK-001 | PickCube 采集 | Data-Agent | Phase 3 | 50条/≥50%成功率 | ⏸️ 延后 |
| T-DATA-STACK-001 | StackCube 采集 | Data-Agent | Phase 3 | 50条/≥50%成功率 | ⏸️ 延后 |

## 已完成摘要（历史）

| task_id | 关键指标 | 主要产物 |
|---------|---------|---------|
| T-DATA-LIFT-001 | 50条，success_rate=70% | `data/vlaw/rollouts/iter1_highsuc/` + encoded |
| T-DATA-LIFT-002 | 40条，success_rate=30% | `data/vlaw/rollouts/iter1_lift_inc20/` + encoded |
| T-WM-COMP-001 | PSNR: pretrained=23.07/ckpt8k=22.51/ckpt10k=22.06 | `logs/vlaw/wm_comparison_frames/` |
| T-EVAL-BASELINE-001 | success_once=95%, success_at_end=75% | `results/vlaw/pld_eval_baseline_20ep.json` |
| T-REWARD-REAL-001 | n=160, vlm_succ=0.0%, env_succ_end=39.4% | `data/vlaw/labeled/iter1_lift_only/` |
| T-AUDIT-001 | 9文件/340轨迹扫描, 5异常 | `logs/vlaw/data_audit_report.md` + `.json` |
| T-DATA-FIX-001 | 3目录5文件异常编码数据已清除 | 剩余4文件全部 (4,48,24) 正确 |
| T-WM-BASELINE-001 | pretrained H20: PSNR=22.35 SSIM=0.79; Phase-A H20: PSNR=21.70 SSIM=0.58 | `results/vlaw/wm_baseline_report.md` + `wm_baseline/` |
| T-VLM-BASELINE-001 | ZS AUC=0.59, LoRA AUC=0.62; α=0.8 无效; 单帧区分力弱 | `results/vlaw/vlm_baseline_report.md` |

## 状态定义
- ⬜ 未开始
- 🔄 进行中
- ✅ 完成
- ⚠️ 需恢复/需人工确认
- ❌ 阻塞
- ⏸️ 延后（当前阶段不执行）
