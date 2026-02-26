# VLAW 执行看板（Execution Board）

> 目的：仅追踪“当前迭代正在推进”的任务，不记录历史流水。

## 当前迭代
- iteration: iter1
- owner: VLAW-Coordinator
- 更新规则：任务状态发生变化时更新；详细过程写入 logs 与 work-logs。

## 任务列表

| task_id | 模块 | owner_agent | 依赖 | 验收标准 | 状态 |
|---|---|---|---|---|---|
| T-REF-A-001 | refactor-phaseA | VLAW-Coordinator | 无 | 新增 task_id + json 摘要规范并在真实任务验证 | 🔄 |
| T-DATA-PICK-001 | D_real_iter1_highsuc | Data-Agent | collector修复完成 | PickCube 50条 rollout 成功率≥50%，并完成VAE编码 | ⬜ |
| T-DATA-STACK-001 | D_real_iter1_highsuc | Data-Agent | collector修复完成 | StackCube 50条 rollout 成功率≥50%，并完成VAE编码 | ⬜ |
| T-REWARD-REAL-001 | reward-label-real | Reward-Agent | 3任务 highsuc 数据就绪 | 真实轨迹 VLM 标注完成并回写统计 | ⬜ |
| T-WM-ITER1-001 | wm-finetune-iter1 | WM-Agent | real 标注完成 | 产出可用 checkpoint + 核心指标 | ⬜ |

## 状态定义
- ⬜ 未开始
- 🔄 进行中
- ✅ 完成
- ⚠️ 需恢复/需人工确认
- ❌ 阻塞
