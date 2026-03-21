# VLAW 下一步推进计划

> **最后更新**: 2026-03-20 | **状态面板**: [vlaw-status.md](vlaw-status.md) | **高层计划**: [VLAW_REPRODUCTION_PLAN.md](VLAW_REPRODUCTION_PLAN.md)
> **归档**: `_archive/v3/VLAW_NEXT_STEPS_v3.md`（2026-03-05 版本）

---

## 主线阻塞：BUG-D（Imagination action conditioning）

WM 需要 absolute EE pose，Policy 输出 delta action。Fix1（积分）和 Fix2（pd_ee_pose 迁移）均失败。

待评估方向：
- **A**: pd_joint_delta_pos → pd_joint_pos → pd_ee_pose 两步转换（需验证）
- **B**: Motion planner 直接在 pd_ee_pose 下生成 demo（需确认 MP 支持）
- **C**: 1-step sim-in-loop（最可靠，牺牲速度）
- **D**: 训练 delta→ee MLP 转换网络（间接方案）

详细诊断：`results/vlaw/wm_diagnostic/DIAGNOSTIC_REPORT.md`

---

## 主线状态更新：BUG-D 已有 Adapter 修复，进入全量验证

Fix4（Dynamics Adapter V1）已在端到端验证中带来 +0.92 dB（29.59→30.51）。
当前主线任务从“寻找修复方向”切换为“全量验证与产出评估”：
- P0：运行 V1 adapter imagination 全量（50-200 条）
- P1：VLM 标注并对比 adapter vs tiled 的 D_syn+ 产出率

---

## ACP + RLPD 支线

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | v6 结果分析（已完成）与配置固化 | ✅ 完成 |
| P1 | 最终 ACP 配置集成到主 pipeline | ⏳ 进行中 |

AWSC 已归档：最佳 SAE=70%（`awsc_td_clip`），默认参数已写入 pipeline。

---

## 主线恢复后

| 优先级 | 任务 | 依赖 |
|--------|------|------|
| P0 | 选定并实施 BUG-D 修复方向 | — |
| P1 | Phase 4 策略更新（Weighted FM, D_real+ ∪ D_syn+） | BUG-D 修复 |
| P2 | Phase 5 评估（baseline=78%） | Phase 4 |
| P3 | Iter-2 第二轮迭代 | Phase 5 |
