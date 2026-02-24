---
name: vlaw-status-check
description: "检查 VLAW 复现项目当前进度，显示各阶段状态"
agent: agent
tools: ['read', 'search']
---

# VLAW 状态检查

请读取以下文件并给出当前 VLAW 复现项目的进度概要：

1. 读取 `.github/vlaw-status.md` — 获取各阶段状态
2. 读取 `.github/VLAW_REPRODUCTION_PLAN.md` 的第五节 — 了解阶段定义
3. 检查以下关键目录是否存在且有内容:
   - `rlft/vlaw/` — 核心模块
   - `ctrl_world/` — 世界模型代码
   - `data/vlaw/` — 数据目录
   - `checkpoints/vlaw/` — 模型权重

请以表格形式输出每个阶段 (P0-P7) 的完成状态，并指出下一步应执行的任务。
