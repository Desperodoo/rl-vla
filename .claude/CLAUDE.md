# rl-vla — Claude Code 记忆文件

## 项目概述
复现 VLAW 论文 (arXiv:2602.12063)：ManiSkill3 仿真环境中，ShortCut Flow 策略 + Ctrl-World 视频扩散世界模型 + VLM 奖励模型的迭代共同改进。核心算法：K_iter=2 次迭代，每轮收集真实轨迹 → VLM 标注 → 微调 WM → Imagination 生成合成数据 → 策略更新。

技术栈：ShortCut Flow / Flow Matching（策略）| Ctrl-World SVD-UNet ~1.5B（世界模型）| Qwen3-VL-4B LoRA（奖励模型）| Pistar06 Value Model（ACP 稠密 advantage）| ManiSkill3（仿真环境）

---

## 当前项目状态（2026-03-20）

> 本文件由**两台设备**共享。`原设备`运行 WM 训练；`新设备（10x RTX 4090）`运行 ACP/RLPD 支线。

### 主线进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 0 数据 | ✅ | v3 mixed=1200, high_suc=552, VAE 编码就绪 |
| Phase 1 WM | ✅ | v5 训练完成（BUG-A/B/C/H 已修复），4000 steps |
| Phase 2 VLM | ✅ | LoRA v3 300步, FP=0%, Recall=61.2% |
| Phase 3 Imagination | ⛔ | **BUG-D 阻塞** |
| Phase 4 策略更新 | ⛔ | 等 BUG-D |
| Phase 5 评估 | ⛔ | 等 Phase 4 |

### ⛔ 关键阻塞：BUG-D（WM-Policy 动作空间鸿沟）

Imagination 推理时 future actions 使用 tiled 当前 EE pose（告诉 WM "臂不动"），导致 peg 完全静止。这是 Imagination 质量退化的**唯一显著根因**（-4.5~-8.5 dB）。

根本矛盾：WM 需要 absolute EE pose（训练时从 `state[18:21]` 提取）；Policy 输出 delta action（`pd_ee_delta_pose`）；转换需物理仿真——而这恰是 Imagination 想绕开的。

已尝试修复：
- Fix1（delta 积分）❌ — PD 控制器使 raw delta ≠ 实际位移，5 帧后累积误差超出 WM 分布
- Fix2（pd_ee_pose 迁移）❌ — PD 控制器 1 步无法到达目标，demo 转换不可行（ManiSkill 官方确认不支持 env_states + 控制模式转换）

待评估方向：A) pd_joint_delta_pos 两步转换 | B) Motion planner 直接生成 | C) 1-step sim-in-loop | D) delta→ee MLP

详细诊断：`results/vlaw/wm_diagnostic/DIAGNOSTIC_REPORT.md`、`BUG_D_EXPLAINED.md`
BUG-D Fix2 详情：见 `knowledge/decisions.md` ADR-043/045

### 已修复的关键 Bug（摘要）

| Bug | 根因 | 修复 | ADR |
|-----|------|------|-----|
| BUG-A | WM action conditioning 用 delta pose 而非绝对 EE | 改为绝对 EE 位姿 | ADR-037 |
| BUG-B | Camera VAE 像素空间拼接 | 改为独立 per-camera VAE 编码 + latent 拼接 | ADR-037 |
| BUG-C | pipeline.py 误用 sd-vae-ft-mse 编码 | 改用 SVD VAE（但 BUG-E 证实编码端无实质差异） | ADR-040 |
| BUG-H | ee_pose_history 初始化仅 1 条 | 改为 num_history*4=24 条 | ADR-043 |

完整 Bug 数据库：`knowledge/bugs-and-fixes.md`（27 条）

### ACP + RLPD 支线（新设备）

**ACP 重构状态**：
- ACP 已从 `rlft/vlaw/acp` 提取到 `rlft/acp`
- 新主路径：`rlft.acp.{config,advantage,hdf5_dataset,value_model,value_targets,train_value_model,infer_values}`
- `rlft/vlaw/acp` 仅保留兼容 shim，避免旧入口和历史脚本立刻失效
- `rlft/envs/acp_reward_wrapper.py`、`rlft/online/train_rlpd.py`、ACP 训练/推理脚本、分析脚本、单测都已切到新路径

**验证结果**：
- ACP value 训练 smoke：通过
- ACP value 推理 + HDF5 写回 smoke：通过
- `rlft/tests/vlaw/test_acp.py`：通过（28/28）
- PLD `reward_mode=acp` smoke：通过
- DSRL `reward_mode=acp` smoke：通过
- RLPD `reward_mode=acp` smoke：通过（仅支持 ManiSkill 原生 demo schema）

**RLPD 兼容性结论**：
- 已修复 HDF5 根组 `meta` 导致的轨迹枚举崩溃
- 当前 `rlft.online.train_rlpd` 仅保证 ManiSkill 原生 demo schema 可用
- 对 VLAW rollout schema 暂不兼容，按用户要求不再在 RLPD pipeline 中继续支持它

**v5 sweep 结论**（15 configs 全部完成）：
- AWSC 最佳：SAE=70%, SO=96%（`awsc_td_clip` 配置已归档为 pipeline 默认值）
- PLD/DSRL SAE≤8%（结构性瓶颈：无 BC 锚定 + ACP value 无"hold"语义）
- Q-target clipping 彻底修复了 PLD/DSRL critic 爆炸（loss 800-1900→3-40）
- 详细报告：`docs/vlaw/acp_v5_rlpd_report.md`

**ACP v6 Sweep（2026-03-17~18）— ✅ 全部完成**：
- DSRL SAE 从 8% 提升到 **14%**（`dsrl_long_grasp`, 200K steps）
- PLD SAE 维持在 4% 左右，grasp bonus 对 PLD 无显著帮助
- 关键结论：**长训练比 grasp bonus scale 更重要**
- 详细报告：`docs/vlaw/acp_v6_rlpd_report.md`

v6 关键结果：

| 指标 | v5 最佳 | v6 最佳 | 变化 | 来源 |
|------|---------|---------|------|------|
| **DSRL SAE** | 8% | **14%** | **+6%** | dsrl_long_grasp (200K) |
| PLD SAE | 4% | 4% | ±0% | entropy_grasp, grasp1/2_td |
| DSRL SO | 96% | 92% | -4% | 多个配置 |

**v6 完整结果表**：

| 实验名 | 算法 | Grasp | Best SO | Best SAE | Final SAE |
|--------|------|-------|---------|----------|-----------|
| pld_grasp1_td | PLD | 1.0 | 82% | 4% | 2% |
| pld_grasp2_td | PLD | 2.0 | 82% | 4% | 0% |
| pld_grasp5_td | PLD | 5.0 | 82% | 2% | 0% |
| pld_grasp1_pot | PLD | 1.0 | 84% | 2% | 0% |
| pld_entropy_grasp | PLD | 1.0 | 86% | 4% | 2% |
| dsrl_grasp1_td | DSRL | 1.0 | 92% | 4% | 2% |
| dsrl_grasp2_td | DSRL | 2.0 | 92% | 2% | 0% |
| dsrl_grasp5_td | DSRL | 5.0 | 92% | 6% | 0% |
| dsrl_grasp1_pot | DSRL | 1.0 | 90% | 6% | 0% |
| **dsrl_long_grasp** | DSRL | 1.0 | **92%** | **14%** | 2% |

**v6 核心发现**：
- **DSRL 长训练 (200K) 是关键突破**：SAE 从 8%→14%，训练时长比 grasp bonus scale 更重要
- **PLD grasp bonus 无效**：entropy collapse 是结构性问题，grasp bonus 无法弥补
- **Grasp bonus scale 非单调**：scale=1 和 scale=5 效果相近，scale=2 表现最差

历史实验报告：`docs/vlaw/acp_v3_rlpd_report.md`、`docs/vlaw/acp_v4_rlpd_report.md`、`docs/vlaw/acp_mirror_experiments.md`

### GPU 分配

**原设备**：GPU 0-1 LMStudio 占用，2-9 空闲（BUG-D 待解决）
**新设备**：GPU 0-9 空闲（v6 完成）

---

## Conda 环境

| 环境 | 用途 |
|------|------|
| `rlft_ms3` | 数据采集、策略训练、评估、ManiSkill |
| `ctrl_world` | Ctrl-World WM 训练与推理 |
| `vlaw_reward` | Qwen3-VL LoRA 训练与批量推理 |

---

## 常用命令

```bash
# 网络代理
export http_proxy=http://10.20.93.149:7890
export https_proxy=http://10.20.93.149:7890

# WM 训练（ctrl_world，GPU 0-3）
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 rlft/vlaw/train_world_model.py

# 数据采集（rlft_ms3，GPU 4-5）
CUDA_VISIBLE_DEVICES=4,5 conda run -n rlft_ms3 python rlft/vlaw/data_collector.py \
  --task LiftPegUpright-v1 --num_envs 64 --num_episodes 50

# VLM 标注（vlaw_reward，GPU 6-7）
CUDA_VISIBLE_DEVICES=6,7 conda run -n vlaw_reward python rlft/vlaw/train_reward_model.py

# 策略训练（rlft_ms3，GPU 8）
CUDA_VISIBLE_DEVICES=8 conda run -n rlft_ms3 python rlft/vlaw/policy_updater.py

# RLPD + ACP（rlft_ms3，GPU 0+1）
CUDA_VISIBLE_DEVICES=0,1 conda run -n rlft_ms3 python -m rlft.online.train_rlpd \
  --reward_mode acp --acp_checkpoint checkpoints/vlaw/acp/v3_so/best.safetensors --acp_device cuda:1

# ACP 数据采集
bash scripts/collect_acp_data.sh
# ACP 多版本训练
bash scripts/train_acp_multi.sh

# 评估（rlft_ms3，GPU 9）
CUDA_VISIBLE_DEVICES=9 conda run -n rlft_ms3 python rlft/envs/evaluate.py

# 测试
conda run -n rlft_ms3 python -m pytest rlft/tests/vlaw/ -v --tb=short -q
```

---

## 代码库结构

```
rlft/
  algorithms/il/         ← ShortCut Flow, Flow Matching
  algorithms/online_rl/  ← PLD-SAC, DSRL-SAC
  buffers/               ← 数据缓冲区
  datasets/              ← 数据集加载
  envs/                  ← ManiSkill 封装、evaluate.py、acp_reward_wrapper.py
  networks/              ← PlainConv encoder（global_cond_dim=626）
  online/                ← 训练入口脚本（train_rlpd.py, train_pld.py, train_dsrl.py）
  vlaw/                  ← VLAW 核心模块
    acp/                 ← ACP 稠密 advantage（Pistar06 value model）
    data/                ← 数据采集 + 噪声策略（OU/Gaussian）
  tests/vlaw/            ← 单元/集成测试

ctrl_world/              ← Ctrl-World（外部代码，最小修改原则）
scripts/                 ← 辅助脚本
checkpoints/vlaw/        ← 模型权重
data/vlaw/               ← 数据集
docs/vlaw/               ← 技术文档和实验报告
```

---

## 关键资产路径

| 资产 | 路径 |
|------|------|
| IL policy（基线） | `checkpoints/il/best_eval_success_once.pt` |
| WM iter1_v5（当前） | `checkpoints/vlaw/world_model/iter1_v5/` |
| WM pretrained | `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt` |
| VLM LoRA best | `checkpoints/vlaw/reward_model/ablation_v3/` |
| ACP v3_so（推荐） | `checkpoints/vlaw/acp/v3_so/best.safetensors` |
| ACP v3_sae | `checkpoints/vlaw/acp/v3_sae/best.safetensors` |
| ACP v2_combined | `checkpoints/vlaw/acp/v2_combined/best.safetensors` |
| Pretrained policy | `runs/maniskill_sweep_v3/aw_shortcut_flow/.../checkpoints/best_eval_success_once.pt` |
| WM 诊断报告 | `results/vlaw/wm_diagnostic/DIAGNOSTIC_REPORT.md` |
| ACP v5 报告 | `docs/vlaw/acp_v5_rlpd_report.md` |
| ACP Pipeline 文档 | `docs/vlaw/acp_pipeline.md` |

---

## 质量门控阈值

| 指标 | 最低门槛 | 目标值 |
|------|---------|-------|
| WM 预测 PSNR | > 18 | > 20 |
| VLM 误报率（FP） | < 20% | < 10% |
| D_syn+ yield rate | > 5% | — |
| Policy success_rate 提升 | > 10% abs | > 20% abs |
| ACP value MAE | < 0.1 | < 0.05 |

---

## 关键架构决策（ADR 速查）

> 完整记录：`knowledge/decisions.md`（48 条 ADR）

| ADR | 决策 | 状态 |
|-----|------|------|
| ADR-002 | 双相机竖拼 (384,192)，VAE latent (4,48,24) | 锁定 |
| ADR-009 | Policy 使用视觉 obs（PlainConv），global_cond_dim=626 | 锁定 |
| ADR-019 | VLM 必须用 video 模式 | 关键 |
| ADR-035 | ACP: Pistar06 value model (SigLIP+Gemma) | 锁定 |
| ADR-037 | WM action = 绝对 EE 位姿，per-camera VAE 编码 | 锁定 |
| ADR-043 | BUG-D 诊断：action tiling 是唯一显著根因 | 活跃 |
| ADR-045 | BUG-D Fix2 失败：pd_ee_pose 迁移不可行 | ❌ 失败 |
| ADR-047 | ACP v5: Q-clip + potential reward + reward clip | ✅ AWSC 归档 |
| ADR-048 | ACP v6: grasp bonus + 长训练提升 DSRL SAE 到 14% | ✅ 完成 |

---

## 知识库索引（`.github/knowledge/`）

| 文件 | 内容 |
|------|------|
| `decisions.md` | 48 条 ADR |
| `bugs-and-fixes.md` | 27 个 Bug 记录 |
| `interfaces.md` | 模块间接口规范 |
| `sweep-baselines.md` | PLD/DSRL 超参扫描结果 |
| `wm-eval-analysis.md` | WM 评估分析 |
| `env-setup.md` | Conda 环境安装步骤 |
| `maniskill-envs.md` | ManiSkill 任务列表 |

---

## 编码规范

- Python 3.10+，函数签名必须有 type hints
- 配置：`tyro` dataclass（禁用 argparse）
- 日志：`wandb`（禁用 tensorboard）
- 数据：HDF5（轨迹）、safetensors（权重）
- 训练：PyTorch 2.x，HuggingFace Accelerate / DDP
- 路径：`pathlib.Path`；import 顺序：stdlib → third-party → local

---

## 状态追踪

- `.github/vlaw-status.md` — 实时状态仪表盘
- `.github/VLAW_NEXT_STEPS.md` — 待办任务看板
- `.github/VLAW_REPRODUCTION_PLAN.md` — Algorithm 1 全流程参考
- `docs/vlaw/` — 实验报告归档
