# vlaw-execution

本技能将原先分散的 `VLAW-Coordinator`、`Data-Agent`、`WM-Agent`、`Reward-Agent`、`Imagination-Agent`、`Policy-Agent`、`Eval-Agent` 合并为一份统一执行规范。

目的：
- 保留完整的 VLAW Algorithm 1 阶段职责
- 避免多个 Agent 文档重复维护阶段定义、输入输出和质量门控
- 让调度、执行、handoff 规则集中在一个地方

## 1. 总原则

### 1.1 Coordinator 原则

当以协调者身份工作时：
- 唯一职责是调度和管理 Algorithm 1 迭代
- 不直接执行训练、推理、数据处理业务代码
- 允许读取状态文件、结果文件、更新状态文档、派遣子任务

### 1.2 Worker 原则

当以执行者身份工作时：
- 只接管自己职责域内的阶段
- 产出明确的结果文件、指标与下一步 handoff 建议
- 不越权修改别的阶段核心逻辑

### 1.3 结果文件要求

各阶段执行前都应初始化结果文件，例如：

```bash
mkdir -p logs/vlaw
export RESULT_FILE="logs/vlaw/<agent-name>-result-$(date +%Y%m%d_%H%M%S).md"
```

结果文件至少包含：
- 任务名
- 状态
- 已完成步骤
- 关键指标
- 输出路径

## 2. Algorithm 1 迭代循环

每轮迭代的标准顺序：

| 步骤 | 内容 | 责任域 | 并行性 |
|------|------|--------|--------|
| Step 1 | rollout 采集 | Data | 前置 |
| Step 2 | VAE 编码 / HDF5 格式化 | Data | 前置 |
| Step 3 | VLM 标注 `D_real` | Reward | 可与 Step 4 并行 |
| Step 4 | WM 微调 | WM | 可与 Step 3 并行 |
| Step 5 | imagination 生成 `D_syn` | Imagination | 依赖 Step 4 |
| Step 6 | VLM 标注 `D_syn` | Reward | 依赖 Step 5 |
| Step 7 | policy update | Policy | 依赖 Step 6 |
| Step 8 | 评估与消融 | Eval | 依赖 Step 7 |

## 3. 调度提示格式

调度 prompt 建议不超过 10 行，并包含：
- Step 名称
- GPU 分配
- 先读状态文件
- 输入数据 / checkpoint
- 完成标志

不要在 dispatch prompt 里重复整套背景知识。

## 4. 截断恢复协议

当出现以下情况时，视为可能截断：
- 子任务返回空响应
- 结果文件缺少完成标志
- 未报告关键指标

恢复步骤：

1. 读取最新结果文件

```bash
ls -lt logs/vlaw/*-result*.md | head -5
tail -40 <latest-result-file>
```

2. 在状态文件中标记该任务为 `⚠️ 截断`

3. 重派任务，并明确：
- 先检查最新 result 文件
- 跳过已完成步骤
- 从第一个未完成步骤继续

## 5. 各职责域详细说明

### 5.1 Data

职责：
- rollout 采集
- VAE 编码
- HDF5 格式化
- demo 数据预处理

关键文件：
- `rlft/vlaw/data_collector.py`
- `rlft/vlaw/data_pipeline.py`
- `scripts/demo_prep.py`

关键规格：

```python
obs_mode = "rgbd"
num_envs = 64
cameras = ["base_camera", "hand_camera"]
resolution = (192, 192)
concat_mode = "vertical"
fps_downsample = 3
```

VAE latent 规格：
- 输入：`(T, 384, 192, 3)`
- 输出：`(T, 4, 48, 24)`
- 存储键：`latent_concat`

HDF5 schema 关键字段：
- `rgb_base`
- `rgb_hand`
- `rgb_render`
- `state`
- `obs_agent`
- `actions`
- `env_success`
- `latent_concat`
- `task_instruction`
- `vlm_reward`
- `vlm_prob`
- `source`

完成标准：
- VAE 重建 PSNR > 25
- Ctrl-World DataLoader 可读
- demo 至少 25 条
- rollout 至少 50 条 / task

### 5.2 WM

职责：
- Ctrl-World 适配
- 世界模型训练
- validation 与 imagination 质检

关键文件：
- `ctrl_world/dataset/dataset_maniskill.py`
- `ctrl_world/config.py`
- `rlft/vlaw/ctrl_world_adapter.py`
- `rlft/vlaw/train_world_model.py`

约束：
- 遵守最小修改原则
- `ctrl_world/` 内部改动加 `# VLAW MODIFICATION:`

训练配置：
- Phase A：热身，训练 Action Encoder + temporal attention
- Phase B：全量微调，使用 Deepspeed / fp16 / gradient checkpointing

验收：
- PSNR > 18，目标 > 20
- SSIM > 0.7
- LPIPS < 0.3
- imagination 可视化通过人审

### 5.3 Reward

职责：
- Qwen3-VL 奖励模型封装
- LoRA 微调
- `D_real` / `D_syn` 批量标注

关键文件：
- `rlft/vlaw/reward_model.py`
- `rlft/vlaw/train_reward_model.py`

环境：
- `vlaw_reward`

关键公式：

```text
R(tau) = 1[ P("yes" | tau, I) > alpha ]
```

重要要求：
- 必须使用 video 模式
- 不用逐帧 image 模式
- 推荐 `alpha=0.8` 用于 `D_syn+` 筛选

LoRA 配置基线：
- `r=16`
- `lora_alpha=32`
- `target_modules=["q_proj", "v_proj"]`
- `num_train_steps=300`

验收：
- FP rate < 20%，目标 < 10%
- `D_syn+` yield > 5%

### 5.4 Imagination

职责：
- 在世界模型内部做 policy-in-the-loop rollout
- 生成 `D_syn`

关键文件：
- `rlft/vlaw/state_predictor.py`
- `rlft/vlaw/imagination.py`

前置条件：
- 世界模型 checkpoint 通过 imagination 人审

核心逻辑：
- 用真实第一帧初始化 latent
- policy 基于结构化 obs 产生 action chunk
- 世界模型预测未来 latent
- state predictor 推进状态
- 循环得到合成轨迹

关键禁止项：
- 不得用 `torch.randn` 初始化第一帧 latent
- 不得传 flat latent 替代结构化 obs
- 注意 `get_action()` / `get_actions()` API 区分

过滤：
- LPIPS variance 过滤静止无效轨迹
- 后续由 Reward 阶段做 VLM 复核

### 5.5 Policy

职责：
- 用 `D_real+ ∪ D_syn+` 微调 ShortCut Flow policy

关键文件：
- `rlft/algorithms/il/shortcut_flow.py`
- `rlft/algorithms/il/flow_matching.py`
- `rlft/vlaw/policy_updater.py`

核心思想：
- 本质是 Filtered Behavioral Cloning
- 在现阶段可以退化为标准 FM loss + 可选 sample weight

关键配置：
- `learning_rate=1e-5`
- `demo_replay_ratio=0.2`
- `data_mix_ratio=0.5`
- `use_ema=True`

灾难性遗忘防护：
- 必须保留 demo replay
- 不要回到 `1e-4`
- EMA checkpoint 必须保存 `ema_agent`

验收：
- 训练 loss 稳定
- success_rate 不低于基线
- Iter-2 Go/No-Go：`success_once >= 78%`

### 5.6 Eval

职责：
- ManiSkill 评估
- 消融实验
- pytest / schema spot check / shim 清理

关键命令：

```bash
conda run -n rlft_ms3 python -m pytest rlft/tests/vlaw/ -v --tb=short -q
CUDA_VISIBLE_DEVICES=9 conda run -n rlft_ms3 python rlft/envs/evaluate.py \
  --checkpoint <path> --num_episodes 50
```

主要评估对象：
- Base Policy
- Filtered BC
- PLD-SAC
- DSRL-SAC
- VLAW 当前方法

关键成功标准：
- 相对基线提升 > 10% abs，目标 > 20% abs
- WM PSNR > 18
- VLM FP < 20%

## 6. GPU 分工建议

| 职责域 | 推荐 GPU |
|--------|----------|
| WM | 0-3 |
| Data | 4-5 |
| Reward | 6-7 |
| Policy | 8 |
| Eval | 9 |

## 7. 质量门控

| 门控 | 条件 | 下游 |
|------|------|------|
| WM Phase A | PSNR > 18 | 启动 Phase B |
| WM imagination | 人工审查通过 | 启动 Imagination / Policy |
| VLM fine-tune | FP < 20% | 标注 `D_syn` |
| `D_syn+` yield | > 5% | 启动 Policy |
| Policy Iter-2 | `success_once >= 78%` | 进入下一轮 |

## 8. 相关输出路径

```text
data/vlaw/demos/
data/vlaw/rollouts/
data/vlaw/encoded/
data/vlaw/synthetic/
data/vlaw/rollouts_labeled/
data/vlaw/synthetic_labeled/
checkpoints/vlaw/world_model/
checkpoints/vlaw/reward_model/
checkpoints/vlaw/policy/
logs/vlaw/*-result*.md
```
