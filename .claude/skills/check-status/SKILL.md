# check-status

当用户调用 `/check-status` 时，执行以下只读操作快速汇总项目状态。

**不修改任何文件。不执行任何训练/推理命令。**

---

## 执行步骤

### Step 1：读取主状态文件

```bash
cat /home/wjz/rl-vla/.github/vlaw-status.md
```

提取并汇总：
- 当前执行阶段
- 当前阻塞（如有）
- 各 GPU 当前分配

### Step 2：读取任务看板

```bash
cat /home/wjz/rl-vla/.github/VLAW_NEXT_STEPS.md
```

提取：进行中任务、最高优先级待办项。

### Step 3：快速扫描资产状态

```bash
# Checkpoint 时间戳
ls -lt /home/wjz/rl-vla/checkpoints/vlaw/world_model/ 2>/dev/null | head -5
ls -lt /home/wjz/rl-vla/checkpoints/vlaw/reward_model/ 2>/dev/null | head -3
ls -lt /home/wjz/rl-vla/checkpoints/vlaw/policy/ 2>/dev/null | head -3

# 数据量
du -sh /home/wjz/rl-vla/data/vlaw/*/ 2>/dev/null

# 最新 result 文件
ls -lt /home/wjz/rl-vla/logs/vlaw/*-result*.md 2>/dev/null | head -5
```

### Step 4：GPU 状态（可选，如 nvidia-smi 可用）

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null
```

---

## 输出格式

以简洁的 Markdown 表格/列表汇报：

```
## VLAW 项目状态快览 — <时间>

**当前阶段**：Phase 1b — WM Extended Training
**阻塞**：ADR-034（Imagination 视觉质量待人工审查）

### GPU 状态
| GPU | 任务 | 状态 |
|-----|------|------|
| 0-3 | WM ext training | 🔄 运行中 |
| 4-9 | — | 空闲 |

### 最新 Checkpoint
- WM: checkpoints/vlaw/world_model/iter1_v3_ext/ckpt-400（2026-03-06）
- VLM: checkpoints/vlaw/reward_model/ablation_v3/（r=16, 300步）
- Policy: checkpoints/vlaw/policy/dryrun/

### 数据状态
- demos: 25 trajectories（LiftPegUpright-v1）
- rollouts: 50 ep（v3 clean）
- encoded: ✅
- synthetic: 200 trajectories（61.0% D_syn+ yield）

### 下一步
1. 等待 WM ext checkpoint 通过 Imagination 人工审查
2. 审查通过后：Imagination-Agent（500 traj，GPU 0-3）+ Reward-Agent 并行
3. 之后：Policy-Agent（GPU 8，2000步）→ Eval-Agent（GPU 9）
```
