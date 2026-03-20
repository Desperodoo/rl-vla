# VLAW 复现项目 — 状态仪表盘

> **最后更新**: 2026-03-19 | **核心参考**: [VLAW_REPRODUCTION_PLAN.md](VLAW_REPRODUCTION_PLAN.md) | [knowledge/](knowledge/)
> **归档**: `_archive/v3/vlaw-status_v3.md`（2026-03-08 版本）

---

## 当前阶段总览

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 0 数据 | ✅ | v3 mixed=1200, high_suc=552, VAE 编码就绪 |
| Phase 1 WM | ✅ | v5 训练完成（BUG-A/B/C/H 已修复），4000 steps |
| Phase 2 VLM | ✅ | LoRA v3 300步, FP=0%, Recall=61.2% |
| Phase 3 Imagination | ⛔ | **BUG-D 阻塞**：future actions 使用 tiled EE pose，Fix1/Fix2 均失败 |
| Phase 4 策略更新 | ⛔ | 等 BUG-D 解决 |
| Phase 5 评估 | ⛔ | 等 Phase 4 |

---

## 关键阻塞：BUG-D（WM-Policy 动作空间鸿沟）

WM 需要 absolute EE pose，Policy 输出 delta action，两者转换需物理仿真。
- Fix1（delta 积分）❌ 累积误差超出 WM 训练分布
- Fix2（pd_ee_pose 迁移）❌ PD 控制器 1 步无法到达目标，demo 转换不可行

可选方向：A) pd_joint_delta_pos 两步转换 | B) Motion planner 直接生成 | C) 1-step sim-in-loop | D) delta→ee MLP

详细诊断：`results/vlaw/wm_diagnostic/DIAGNOSTIC_REPORT.md`、`BUG_D_EXPLAINED.md`

---

## ACP + RLPD 支线（新设备，10x RTX 4090）

| 版本 | 最佳结果 | 报告 |
|------|---------|------|
| v5 sweep (15 configs) | AWSC SAE=70%, SO=96% | `docs/vlaw/acp_v5_rlpd_report.md` |
| v6 sweep (10 configs) | 🔄 准备启动（grasp bonus） | `scripts/acp_v6_scheduler.py` |

AWSC track 已归档（最佳配置 `awsc_td_clip`）。PLD/DSRL SAE≤8% 瓶颈待 v6 grasp bonus 突破。

---

## GPU 分配

### 原设备
| GPU | 任务 | 状态 |
|-----|------|------|
| 0-1 | LMStudio | 占用 |
| 2-9 | 空闲（BUG-D 待解决） | — |

### 新设备（10x RTX 4090）
| GPU | 任务 | 状态 |
|-----|------|------|
| 0-9 | v6 Grasp Bonus Sweep | ⏳ 待启动 |
