# RESULT_FILE 协议（共享）

> 单点维护：所有 worker/subagent 结果输出统一遵循本协议。

## 最小硬约束（dispatch 必须显式携带）
1. 只允许写入 `logs/vlaw/`，禁止写入 `/tmp/`。
2. 第一条命令创建 `RESULT_FILE`。
3. 每完成一步：
   - 追加写入 RESULT_FILE
   - 在消息中直接输出该步摘要（不能只写文件）
4. 最终消息必须包含：
   - RESULT_FILE 路径
   - 步骤摘要
   - 总体状态（✅/⚠️/❌）

## 推荐初始化
```bash
mkdir -p /home/wjz/rl-vla/logs/vlaw
export RESULT_FILE="/home/wjz/rl-vla/logs/vlaw/<agent>-result-$(date +%Y%m%d_%H%M%S).md"
echo "# <agent> 任务报告 — $(date)" > "$RESULT_FILE"
echo "## 状态：进行中" >> "$RESULT_FILE"
```

## 可机读摘要（Phase A 新增）

在任务收尾时，额外写入同名 json：
- 路径：`${RESULT_FILE%.md}.json`
- schema（最小字段）：
```json
{
  "task_id": "T-EXAMPLE-001",
  "agent": "Data-Agent",
  "start_time": "2026-02-26T17:00:00+08:00",
  "end_time": "2026-02-26T17:20:00+08:00",
  "status": "completed",
  "steps": ["..."],
  "metrics": {"success_rate": 0.7},
  "artifacts": ["data/...h5"],
  "errors": [],
  "next_action": "..."
}
```

## 截断恢复
- Coordinator 通过 `logs/vlaw/*-result-*.md` 找最近文件恢复。
- 若有 json，优先读 json 聚合指标；md 用于人工核查。
