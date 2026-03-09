# rl-vla — Claude Code 记忆文件

## 项目概述
复现 VLAW 论文 (arXiv:2602.12063)：ManiSkill3 仿真环境中，ShortCut Flow 策略 + Ctrl-World 视频扩散世界模型 + VLM 奖励模型的迭代共同改进。核心算法：K_iter=2 次迭代，每轮收集真实轨迹 → VLM 标注 → 微调 WM → Imagination 生成合成数据 → 策略更新。

技术栈：ShortCut Flow / Flow Matching（策略）| Ctrl-World SVD-UNet ~1.5B（世界模型）| Qwen3-VL-4B LoRA（奖励模型）| **Pistar06 Value Model（ACP 稠密 advantage）** | ManiSkill3（仿真环境）

---

## 当前项目状态（手动更新）

**阶段**：Phase 2 — WM v4 重训练（BUG-A/BUG-B 修复后从零开始）+ ADR-036/037 Pipeline 优化
**已修复 (2026-03-08)**：
- **BUG-A (ADR-037)**：WM action conditioning 语义错配 — 从 delta pose 改为**绝对 EE 位姿** (对齐 DROID)。影响 8 个文件。stat.json 已重新生成。
- **BUG-B (ADR-037)**：Camera VAE 编码差异 — 从像素空间拼接改为**独立 per-camera VAE 编码后 latent 空间拼接** (对齐 DROID)。影响 pipeline.py。
- iter1_v3_ext 训练已停止（使用了错误 stat.json + 错误 latent 编码），需从零重训练。
**当前阻塞**：WM v4 训练中 (step 0/4000, ~52s/step with 8 GPU, 预计 ~58h)
**下游阻塞**：Imagination、策略更新、评估、Iter-2 全部暂停。

**ACP 状态**：
- Baseline: 代码实现 ✅ → Training 8000步 ✅ (best val MAE=0.1675 @ step 7200) → Inference dry-run ✅ → 训练报告 ✅ (`logs/vlaw/acp_report/`)
- Evo-RL 对齐改进：训练已结束（GPU 2 已释放用于 WM 训练）
- 改进计划详见：`.claude/plans/modular-finding-llama.md`

**GPU 分配（当前）**：

| GPU | 任务 | 状态 |
|-----|------|------|
| 0-1 | LMStudio | 占用 |
| 2-9 | WM v4 训练 iter1_v4 (8 GPU, DeepSpeed ZeRO-2) | 运行中 (~14GB/GPU) |

---

## Conda 环境

| 环境 | 用途 | 激活 |
|------|------|------|
| `rlft_ms3` | 数据采集、策略训练、评估、ManiSkill 环境 | `conda activate rlft_ms3` |
| `ctrl_world` | Ctrl-World WM 训练与推理 | `conda activate ctrl_world` |
| `vlaw_reward` | Qwen3-VL LoRA 训练与批量推理 | `conda activate vlaw_reward` |

---

## 常用命令

```bash
# 网络代理（HuggingFace / GitHub / pip）
export http_proxy=http://10.20.93.149:7890
export https_proxy=http://10.20.93.149:7890

# 单次前缀（不改全局环境）
http_proxy=http://10.20.93.149:7890 https_proxy=http://10.20.93.149:7890 python ...

# WM 训练（ctrl_world 环境，GPU 0-3）
conda activate ctrl_world
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  rlft/vlaw/train_world_model.py

# 数据采集（rlft_ms3，GPU 4-5）
CUDA_VISIBLE_DEVICES=4,5 conda run -n rlft_ms3 python rlft/vlaw/data_collector.py \
  --task LiftPegUpright-v1 --num_envs 64 --num_episodes 50

# VLM 标注（vlaw_reward，GPU 6-7）
CUDA_VISIBLE_DEVICES=6,7 conda run -n vlaw_reward python rlft/vlaw/train_reward_model.py

# 策略训练（rlft_ms3，GPU 8）
CUDA_VISIBLE_DEVICES=8 conda run -n rlft_ms3 python rlft/vlaw/policy_updater.py

# ACP value model 训练（vlaw_reward，GPU 6-7）
CUDA_VISIBLE_DEVICES=6,7 conda run -n vlaw_reward python rlft/vlaw/scripts/run_acp_train.py \
  --num_steps 8000 --batch_size 32

# ACP advantage 标注（vlaw_reward，GPU 6）
CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python rlft/vlaw/scripts/run_acp_infer.py \
  --checkpoint_path checkpoints/vlaw/acp/iter1/best.safetensors

# 评估（rlft_ms3，GPU 9）
CUDA_VISIBLE_DEVICES=9 conda run -n rlft_ms3 python rlft/envs/evaluate.py

# RLPD + ACP reward（rlft_ms3，GPU 0+1，0=RL训练，1=ACP模型）
CUDA_VISIBLE_DEVICES=0,1 conda run -n rlft_ms3 python -m rlft.online.train_rlpd \
  --reward_mode acp --acp_checkpoint checkpoints/vlaw/acp/iter1/best.safetensors \
  --acp_device cuda:1 --total_timesteps 500000

# 测试（无 GPU OK）
conda run -n rlft_ms3 python -m pytest rlft/tests/vlaw/ -v --tb=short -q
```

---

## 代码库结构

```
rlft/
  algorithms/il/         ← 模仿学习（ShortCut Flow, Flow Matching）
    shortcut_flow.py     ← 主策略类（compute_weighted_loss 在此修改）
    flow_matching.py
  algorithms/online_rl/  ← 在线 RL（PLD-SAC, DSRL-SAC）
  buffers/               ← 数据缓冲区
  datasets/              ← 数据集加载（OfflineRLDataset 必须）
  envs/                  ← ManiSkill 封装、evaluate.py、acp_reward_wrapper.py (ACP online reward)
  networks/              ← PlainConv encoder（global_cond_dim=626）
  vlaw/                  ← VLAW 核心模块（见 rlft/vlaw/CLAUDE.md）
    data_collector.py    ← P1.1
    data_pipeline.py     ← P1.2（VAE 编码，concat_cameras）
    ctrl_world_adapter.py← P2.1
    train_world_model.py ← P2.2
    reward_model.py      ← P3.1
    train_reward_model.py← P3.2
    state_predictor.py   ← P4.1
    imagination.py       ← P4.2
    policy_updater.py    ← P5.1
  vlaw/acp/              ← ACP 稠密 advantage 模块（从 Evo-RL Pistar06 移植）
    config.py            ← 所有 ACP config（tyro dataclass）
    value_targets.py     ← per-frame value target（env_success GT）
    value_model.py       ← Pistar06 模型封装（SigLIP+Gemma+value head）
    advantage.py         ← N-step advantage、量化阈值、权重归一化
    hdf5_dataset.py      ← HDF5→Dataset（value 训练/推理）
    train_value_model.py ← Value model 训练循环
    infer_values.py      ← 批量推理+advantage 标注写回 HDF5
  roboreward/            ← RoboReward 模块（arXiv:2601.00675）
  tests/vlaw/            ← 单元/集成测试（无真实 GPU/权重）
  online/                ← 训练入口脚本

ctrl_world/              ← Ctrl-World（外部代码，最小修改原则，见 ctrl_world/CLAUDE.md）
scripts/                 ← 辅助脚本
checkpoints/vlaw/        ← 模型权重（见下方资产路径）
data/vlaw/               ← 数据集
logs/vlaw/               ← 子 Agent RESULT_FILE 输出
.github/agents/          ← VS Code Copilot Agent 定义（frontmatter 路由）
.claude/skills/          ← Claude Code + Copilot 共享 skill（主要内容在此）
```

---

## 关键资产路径

| 资产 | 路径 |
|------|------|
| IL policy（基线） | `checkpoints/il/best_eval_success_once.pt` |
| AWSC fine-tuned policy | `runs/fair_comparison/.../awsc/best_s42__1772570560/checkpoints/final.pt` |
| WM pretrained | `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt` (8.7GB) |
| WM iter1_v3（ckpt-400） | `checkpoints/vlaw/world_model/iter1_v3/` |
| WM iter1_v3_ext（已废弃，BUG-A/B） | `checkpoints/vlaw/world_model/iter1_v3_ext/` |
| WM iter1_v4（当前训练中） | `checkpoints/vlaw/world_model/iter1_v4/` |
| SVD pretrained | `checkpoints/vlaw/world_model/pretrained/svd/` |
| CLIP pretrained | `checkpoints/vlaw/world_model/pretrained/clip/` |
| VLM base（Qwen3-VL-4B） | `checkpoints/vlaw/reward_model/qwen_vl/` (8.3GB) |
| VLM LoRA best | `checkpoints/vlaw/reward_model/ablation_v3/` (r=16, 300步) |
| VLM LoRA baseline | `checkpoints/vlaw/reward_model/lora_v3/` (200步) |
| Policy dry-run | `checkpoints/vlaw/policy/dryrun/` |
| State predictor | `checkpoints/vlaw/state_predictor/` |
| ACP pretrained SigLIP | `checkpoints/vlaw/acp/pretrained/siglip/` (~3.3GB, 428M params) |
| ACP pretrained Gemma | `checkpoints/vlaw/acp/pretrained/gemma/` (~549MB, 268M params) |
| ACP value model iter1 | `checkpoints/vlaw/acp/iter1/` (baseline, 8000步, MAE=0.1675) |
| ACP exp_aligned | `checkpoints/vlaw/acp/exp_aligned/` (Evo-RL 对齐实验, 训练中) |
| ACP dryrun checkpoint | `checkpoints/vlaw/acp/dryrun/` (20步 dry-run, MAE=0.271) |
| ACP 训练报告 | `logs/vlaw/acp_report/ACP_Training_Report.md` (8000步, best MAE=0.1675) |
| ACP 对齐实验 log | `logs/vlaw/acp_exp_aligned.log` |
| ACP 改进计划 | `.claude/plans/modular-finding-llama.md` |

---

## 质量门控阈值

| 指标 | 最低门槛 | 目标值 | 论文值 |
|------|---------|-------|-------|
| WM 预测 PSNR | > 18 | > 20 | 21.77 |
| WM Imagination 视觉质量 | 通过人工审查 | — | — |
| VLM 误报率（FP） | < 20% | < 10% | 5% |
| D_syn+ yield rate | > 5% | — | 当前实测 61.0% |
| Policy success_rate 提升 | > 10% abs | > 20% abs | 39.2% abs |
| Policy Iter-2 基线 | success_once ≥ 78% | — | — |
| BC flywheel Go/No-Go | B > A + 3% | — | — |
| ACP value MAE | < 0.1 | < 0.05 | baseline 0.1675, exp_aligned 训练中 (当前 best 0.1957 @ step 1200) |
| ACP advantage positive_ratio | ~30% | — | 已达标 (dry-run 0.300) |

---

## 关键架构决策（ADR 速查）

| ADR | 决策 | 状态 |
|-----|------|------|
| ADR-002 | 双相机**竖拼** → 分辨率 (384,192)，VAE latent shape **(4,48,24)** | 锁定 |
| ADR-007 | WM Iter1 从 pretrained 开始全量 finetune，DeepSpeed ZeRO-2，**`ctrl_world` env** | 锁定 |
| ADR-009 | Policy 使用**视觉 obs（PlainConv）**，global_cond_dim=626，非 raw state | 锁定 |
| ADR-012 | Iter-1 灾难性遗忘根因：无 demo 回放 + lr=1e-5 过高 + EMA 无效 | 已知缺陷 |
| ADR-019 | VLM 必须用 **`video` 模式**（`use_video_format=True`）；否则 D_syn+=0 | 关键 |
| ADR-026 | **所有 v1/v2 数据因 BUG-020（双相机坍塌）已存档**，当前用 v3 数据 | 历史 |
| ADR-034 | **eval_WM PSNR ≠ Imagination 质量**；人工审查 viz 是强制门控 | 当前阻塞 |
| ADR-035 | **ACP 集成**：Pistar06 value model（SigLIP 428M + Gemma 268M + projector+value head）。双相机分别输入 SigLIP（128x128 → resize 384x384）。支持 `unfreeze_vision_top_n` 部分解冻 SigLIP 顶层（Evo-RL 对齐）。LR scheduler 支持 `lr_min` floor。连续 advantage 权重供 `compute_weighted_loss`。支持 `success_key` 配置切换 env_success/vlm_success。Conda env 复用 `vlaw_reward`。 | ✅ 代码+Evo-RL对齐完成 |
| ADR-036 | **Pipeline 参数优化**：WM num_workers 4→8 + GPU 扩展文档; Imagination 新增 `--num_inference_steps` CLI; ACP dtype float32→bfloat16 + autocast; VLM DataLoader num_workers 0→2; VLM use_flash_attention 默认 True; Policy visual encoder bfloat16 autocast | ✅ 已实施 |
| ADR-037 | **WM action conditioning + VAE 编码对齐 DROID**：(A) Action conditioning 从 delta pose 改为绝对 EE 位姿 [tcp_xyz+euler_xyz+gripper_norm]；stat.json 从 joint angle percentiles 改为 EE pose percentiles。(B) VAE 编码从像素空间拼接改为独立 per-camera 编码+latent 空间拼接。影响：generate_stat_json, dataset_maniskill, ctrl_world_adapter, imagination_env, imagination_rl_env, imagination.py, pipeline.py。iter1_v3/v3_ext 训练数据全部作废，需重新编码+重训练。 | ✅ 代码修复完成 |
| ADR-038 | **ACP Online Reward for RLPD**：用 ACP value model TD-shaped reward `r(s,s')=(V(s')-V(s))*scale` 替换 ManiSkill sim dense reward 进行 SAC/AWSC 在线训练。`DualCameraRewardWrapper` 在 `FlattenRGBDObservationWrapper` 前拦截 sensor_data + env.render() 获取双相机图像。支持三种模式：`sim`（默认不变）、`acp`（纯 ACP reward）、`acp_blend`（加权混合）。ACP model 默认部署到 cuda:1 与 RL 训练分 GPU。新增文件：`rlft/envs/acp_reward_wrapper.py`，修改 `train_rlpd.py` Args。 | ✅ 代码+测试完成 |

完整决策记录：`.github/knowledge/decisions.md`（38 条 ADR）

---

## 知识库索引（`.github/knowledge/`）

| 文件 | 内容 |
|------|------|
| `decisions.md` | 34 条 ADR，全部架构决策 |
| `bugs-and-fixes.md` | 24 个 Bug 记录（BUG-001 ~ BUG-024） |
| `interfaces.md` | 模块间接口规范（obs shape、checkpoint key、API 签名） |
| `env-setup.md` | 三套 conda 环境完整安装步骤 |
| `maniskill-envs.md` | ManiSkill 任务列表、demo 数据路径、replay 命令 |
| `wm-eval-analysis.md` | WM 评估分析（eval_WM vs Imagination 差异根因） |
| `sweep-baselines.md` | PLD-SAC/DSRL-SAC 超参扫描结果 |
| `ADR-026-data-quality-diagnosis.md` | v1 数据污染诊断详情 |

---

## 编码规范

- Python 3.10+，所有函数签名必须有 type hints
- 配置管理：`tyro` dataclass（禁用 argparse）
- 实验日志：`wandb`（禁用 tensorboard）
- 数据格式：HDF5（轨迹）、safetensors（权重首选）
- 训练框架：PyTorch 2.x，HuggingFace Accelerate / DDP
- 路径：用 `pathlib.Path`；import 顺序：stdlib → third-party → local

---

## Agent 系统总览

Claude Code skills（`/skill-name` 调用）与 VS Code Copilot agents（`.github/agents/`）共享 `.claude/skills/` 内容。

| Skill / Agent | GPU | 职责 |
|--------------|-----|------|
| `/vlaw-coordinator` | — | 总调度，Algorithm 1 迭代循环，不执行业务代码 |
| `/data-agent` | 4-5 | ManiSkill 数据采集、VAE 编码、HDF5 格式化 |
| `/wm-agent` | 0-3 | Ctrl-World 适配、训练（Phase A/B）、验证 |
| `/reward-agent` | 6-7 | Qwen3-VL LoRA 微调、批量奖励标注 |
| `/imagination-agent` | 0-3 | Policy-in-Loop 闭环 rollout、合成数据生成 |
| `/policy-agent` | 8 | Weighted FM 损失、D_real+∪D_syn+ 策略更新 |
| `/eval-agent` | 9 | 评估基线/消融、pytest 代码质量、shim 清理 |
| `/progress-agent` | — | 汇总状态、更新 `.github/` 进度文件 |
| `/check-status` | — | 只读快速状态检查 |

---

## RESULT_FILE 防截断协议

**每个 Worker 必须将以下代码作为第一个 Bash 命令执行：**

```bash
mkdir -p /home/wjz/rl-vla/logs/vlaw
export RESULT_FILE="/home/wjz/rl-vla/logs/vlaw/AGENT_NAME-result-$(date +%Y%m%d_%H%M%S).md"
echo "# AGENT_NAME 任务报告 — $(date)" > "$RESULT_FILE"
echo "## 状态：进行中" >> "$RESULT_FILE"
```

**每完成一步后双写**（文件 + 消息正文）：
```bash
echo "- [x] Step N: 描述 ($(date +%H:%M))" >> "$RESULT_FILE"
```

最终消息**必须包含**：RESULT_FILE 完整路径、每步摘要、总体状态（✅/⚠️/❌）。

## 截断恢复三步法（Coordinator 专用）

当子 Agent 返回空响应或缺少 ✅/❌ 时：
1. **T1** 读取 `ls -lt logs/vlaw/*-result*.md | head -5`，找到最新结果文件
2. **T2** 更新 `vlaw-status.md`，标记该任务为 `⚠️ 截断`
3. **T3** 重新派遣，prompt 中明确写"跳过已完成的 Step 1-N，从 Step N+1 开始"

**禁止** Coordinator 自己接管 Worker 的业务任务。

---

## 项目状态追踪文件

| 文件 | 用途 |
|------|------|
| `.github/vlaw-status.md` | 实时状态（阶段/GPU/checkpoint/数据） |
| `.github/VLAW_NEXT_STEPS.md` | 待办任务看板（带优先级） |
| `.github/VLAW_REPRODUCTION_PLAN.md` | Algorithm 1 全流程参考 |
| `.github/VLAW_EXECUTION_BOARD.md` | 执行看板 |
| `logs/vlaw/` | 子 Agent RESULT_FILE 输出 |

> Agent 系统配置详解：`.claude/AGENT_SYSTEM_GUIDE.md`
