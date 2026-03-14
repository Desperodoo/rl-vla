# rl-vla — Claude Code 记忆文件

## 项目概述
复现 VLAW 论文 (arXiv:2602.12063)：ManiSkill3 仿真环境中，ShortCut Flow 策略 + Ctrl-World 视频扩散世界模型 + VLM 奖励模型的迭代共同改进。核心算法：K_iter=2 次迭代，每轮收集真实轨迹 → VLM 标注 → 微调 WM → Imagination 生成合成数据 → 策略更新。

技术栈：ShortCut Flow / Flow Matching（策略）| Ctrl-World SVD-UNet ~1.5B（世界模型）| Qwen3-VL-4B LoRA（奖励模型）| **Pistar06 Value Model（ACP 稠密 advantage）** | ManiSkill3（仿真环境）

---

## 当前项目状态（手动更新）

> ⚠️ 本文件由**两台设备**共享。`原设备`运行 WM 训练；`新设备（10x RTX 4090）`运行 ACP/RLPD 支线任务。

---

### 原设备状态

**阶段**：Phase 2 — WM v5 重训练（BUG-A/B/C 全部修复后从零开始）
**已修复**：
- **BUG-A (ADR-037, 2026-03-08)**：WM action conditioning 语义错配 — 从 delta pose 改为**绝对 EE 位姿** (对齐 DROID)。stat.json 已重新生成。
- **BUG-B (ADR-037, 2026-03-08)**：Camera VAE 编码差异 — 从像素空间拼接改为**独立 per-camera VAE 编码后 latent 空间拼接**。
- **BUG-C (ADR-040, 2026-03-11)**：VAE 编码器不匹配 — `pipeline.py` 误用 `sd-vae-ft-mse` 的 `AutoencoderKL` 编码，而 Ctrl-World 训练/推理使用 SVD 的 `AutoencoderKLTemporalDecoder`。两者权重不同，latent 分布存在偏差。已改为使用 SVD VAE 编码，数据重编码为 train_v5。⚠️ 后续发现 BUG-E：编码端两者权重几乎相同（corr=0.999999），实质影响仅在解码端。
- **BUG-D (ADR-043, 2026-03-14)**：**[CRITICAL] Imagination 推理 future actions 使用 tiled 当前 EE pose** — 推理时 5 个未来帧全部填充相同的当前 EE 位姿（`np.tile`），而训练时 WM 接收每帧不同的真实 EE pose。Fix1 尝试：`integrate_delta_to_ee_poses()` 将 policy delta actions 积分为绝对 EE pose 序列。⚠️ **Fix1 验证失败**：用户肉眼判断效果比 tiled 更差。可能原因：(1) pd_ee_delta_pose 经 PD 控制器转换，raw delta ≠ 实际 EE 位移；(2) 积分 EE pose 超出 WM 训练分布；(3) 旋转合成顺序或 gripper 语义错误。**需要重新分析修复方案**。
- **BUG-E (ADR-043, 2026-03-14)**：V5 latents 与 V4 几乎相同（corr=0.999999），BUG-C 修复对编码端无实质影响。SVD VAE 时序修改集中在解码器。
- **BUG-H (ADR-043, 2026-03-14)**：ee_pose_history 初始化仅 1 条，应为 num_history*4=24 条（对齐 latent_history 和官方代码）。已修复。
**WM v4 评估结果 (2026-03-11)**：4000 步训练完成，best loss=0.177 (step 3400)。视觉质量尚可但物体动态弱（peg 基本静止），判定为**不可用**。分析发现 BUG-C 是主要根因（后修正为 BUG-D 是真正根因）。
**WM v5 评估结果 (2026-03-14)**：4000 步训练完成，best loss≈0.157 (step ~3900)。20 checkpoint 并行 imagination eval 完成：peg 动态 1/10，与 v4 一致。训练验证样本（GT actions）确认 WM 具备动态建模能力 → **BUG-D 是根因**。
**BUG-D Fix1 验证 (2026-03-14)**：step 3400/3800/4000 三个 checkpoint 完成 fix1 eval（integrate delta→EE pose）。**❌ 效果更差**，用户肉眼验证不如原始 tiled 方案。Fix1 输出：`data/vlaw/synthetic/v5_fix1_step{3400,3800,4000}/viz/`。原始对照：`data/vlaw/synthetic/v5_eval_step*/viz/`。
**当前进度**：BUG-D fix1 失败 → 需重新分析 delta action 到 EE pose 的正确积分方式，或转向训练端修复（BUG-F 时间跳跃 + BUG-G Action Encoder LR）
**下游阻塞**：WM imagination 仍不可用。待定：(1) 修正 fix1 积分逻辑；(2) 若推理端无法解决 → WM v6 训练（BUG-F+G 修复）

**待消融 (Step 2 — 时间跳跃数据增强)**：
- 官方 DROID 训练使用随机时间跳跃 `skip=randint(1,2)`, `skip_his=skip*4` (15% prob→0)
- 我们的 ManiSkill dataset 使用严格连续帧 (`skip_step=1`)（BUG-F）
- 需修改 `ctrl_world/dataset/dataset_maniskill.py` 的 `__getitem__` 加入 DROID 风格随机 skip
- **优先级**：BUG-D 修复后 → 若 peg 动态恢复但不足 → 作为 WM v6 消融项

**待改进 (Step 3 — Action Encoder 训练改进)**：
- Action Encoder (2.11M) 随机初始化，与 UNet (1524M) 共用 LR=1e-5，无 warmup（BUG-G）
- 需为 Action Encoder 设置独立高 LR（如 1e-4）+ 500 步 linear warmup
- **优先级**：BUG-D 修复后 → 若 peg 动态恢复但不足 → 与时间跳跃一起作为 WM v6

**原设备 GPU 分配**：

| GPU | 任务 | 状态 |
|-----|------|------|
| 0-1 | LMStudio | 占用 |
| 2-9 | 空闲（fix1 eval 已完成） | — |

---

### 新设备状态（2026-03-11，10x RTX 4090）

**ACP 在新设备上的情况**：
- 环境：`rlft_ms3` + transformers 5.3.0 已安装 ✅
- 模型权重：SigLIP ✅ (`checkpoints/vlaw/acp/pretrained/siglip/`) | Gemma ✅ (`checkpoints/vlaw/acp/pretrained/gemma/`)
- 训练数据：**仅有 expert demo**（25 条轨迹，510 帧，100% 成功率） → `data/vlaw/rollouts/mixed/LiftPegUpright-v1/`
  - ⚠️ 原设备的 1200 条混合质量数据（46% 成功率）未同步到此设备
- ACP iter1（在此设备训练）：best MAE=0.0021（**严重过拟合**，仅 demo 数据）→ `checkpoints/vlaw/acp/iter1/best.safetensors`
  - 注：原设备 ACP iter1 best MAE=0.1675（1200 条混合数据），更具参考价值
- **预训练策略（新设备本地）**：`runs/fair_comparison/awsc/best_s42__1772570560/checkpoints/best.pt` ✅

**RLPD + ACP 实验状态（2026-03-11）**：

| 实验 | 脚本 | 状态 | 结果 |
|------|------|------|------|
| SAC + ACP iter1（demo-only 过拟合） | `run_rlpd_acp.sh` | ✅ 完成 | 500K steps, best SR 1.56%（❌ 不可用） |
| SAC + ACP v2_combined | `run_rlpd_sac_acp_v2.sh` | ✅ 完成 | 500K steps, best SR 1.56%（❌ SAC 模式不可用） |
| AWSC + 错误 ckpt + ACP v2 | `run_rlpd_awsc_acp.sh` (旧) | ✅ 完成 | best SR 81.25%（⚠️ 使用了 RLPD-finetuned ckpt，无法体现提升） |

**ACP Mirror 实验（2026-03-11）— ✅ 全部完成**：
用 ACP reward 替换 sim reward，对比 `runs/fair_comparison/` 的 sim-reward 结果。
使用正确的 IL-trained pretrained checkpoint（与 compare_data_efficiency 相同）。
入口脚本：`scripts/run_acp_mirror_experiments.sh`

| 实验 | GPU | total_steps | Best SR (once/end) | Final SR (once/end) | 状态 |
|------|-----|-------------|---------------------|---------------------|------|
| AWSC + ACP | 0+1 | 500K | 90%/66% | 62%/56% | ✅ 完成（⚠️ success_once 退化，success_at_end 持平 sim） |
| PLD-SAC + ACP | 2+3 | 71K | 82%/2% | 58%/0% | ✅ 完成（❌ success_at_end=0%） |
| DSRL-SAC + ACP | 4+5 | 71K | 92%/6% | 88%/2% | ✅ 完成（❌ success_at_end≈0%） |

对比 sim-reward 基线 success_at_end（seed 42）：AWSC-sim best=72%, PLD-sim best=86%, DSRL-sim best=60%

Pretrained checkpoint: `runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt`
ACP checkpoint: `checkpoints/vlaw/acp/v2_combined/best.safetensors`
WandB project: `rlpd-acp-mirror`
Runs: `runs/{awsc,pld,dsrl}_acp_mirror_s42__177320867{4,5,5}/`

**关键发现**：
- `train/reward/acp_step_mean` 在 step 5200 后降至 0.0 — **已诊断为日志 bug**，value model 输出正常（std=0.059），wrapper 实际返回非零奖励，`online_cum_reward_mean` 全程非零
- **success_at_end 才是真正核心指标**：仅 AWSC 在 success_at_end 上达到 66%（sim=72%），PLD/DSRL 的 success_at_end 均 ≤6%
- ACP value 目标为 success_once 语义，无法引导 success_at_end 行为（根因）
- 详细报告：`docs/vlaw/acp_mirror_experiments.md`

**ACP 数据多样化（ADR-039）— ✅ 全部完成**：

数据采集结果：

| Type | 分布 | Trajs | Frames | SR% |
|------|------|-------|--------|-----|
| A-demo | expert demos | 50 | 510 | 96.0% |
| B-pretrained | 无噪声 AWSC rollout | 400 | 11,395 | 30.5% |
| C-teleop | OU噪声（σ=0.07，pause 4%） | 400 | 13,040 | 7.0% |
| D-rl_prior | Gaussian噪声（σ=0.25） | 400 | 13,243 | 3.5% |
| E-random | 纯随机（ablation） | 100 | 3,500 | 0.0% |
| **Total** | — | **1,350** | **41,688** | — |

ACP 训练结果（5 版本全部完成，GPU 2-6 并行，`rlft_ms3` env）：

| 版本 | 数据 | Checkpoint |
|------|------|-----------|
| v2_demo_only | A | `checkpoints/vlaw/acp/v2_demo_only/best.safetensors` ✅ |
| v2_pretrained_pol | B | `checkpoints/vlaw/acp/v2_pretrained_pol/best.safetensors` ✅ |
| v2_teleop_sim | C（**真机遥操作分布**） | `checkpoints/vlaw/acp/v2_teleop_sim/best.safetensors` ✅ |
| v2_rl_prior | D（**真机RL微调分布**） | `checkpoints/vlaw/acp/v2_rl_prior/best.safetensors` ✅ |
| v2_combined | A+B+C+D（**推荐**） | `checkpoints/vlaw/acp/v2_combined/best.safetensors` ✅ |

注：新设备无 `vlaw_reward` env，ACP 训练使用 `rlft_ms3`（依赖齐全）。`train_acp_multi.sh` 已修改为使用 `rlft_ms3`。

**ACP v2 训练结果汇总**：

| 版本 | 数据 | Best MAE | Val Loss | 质量门控 |
|------|------|----------|----------|---------|
| v2_demo_only | A (50 traj) | 0.0026 | 1.382 | ⚠️ 过拟合 |
| v2_pretrained_pol | B (400 traj) | 0.1272 | 3.250 | ✅ |
| v2_teleop_sim | C (400 traj) | 0.0739 | 3.361 | ✅ |
| v2_rl_prior | D (400 traj) | 0.0516 | 3.073 | ✅ |
| v2_combined | A+B+C+D (1250 traj) | 0.0837 | 3.209 | ✅ 推荐 |

详细结果+训练曲线图：`docs/vlaw/acp_pipeline.md` §8

数据噪声设计依据：
- **Type C (teleop_sim)**：Ornstein-Uhlenbeck 相关噪声（θ=0.15，σ=0.07）+ 随机暂停（4%/步），模拟人类视觉伺服控制的时空平滑性和停顿特征
- **Type D (rl_prior)**：i.i.d. Gaussian（σ=0.25），模拟高熵SAC早期阶段的宽动作分布
- 实现：`rlft/vlaw/data/noisy_policy.py`（`OUNoisePolicyWrapper` + `GaussianNoisePolicyWrapper`）

**新设备 GPU 分配（当前）**：

| GPU | 任务 | 状态 |
|-----|------|------|
| 0-9 | AWSC+ACP Sweep v2（15 configs, 5并行, PID 1550228） | 运行中 |

**AWSC+ACP Sweep v2（2026-03-12）— 运行中**：

基于 wandb 数据分析（`scripts/sweep_acp/fetch_wandb.py`）对 ACP mirror AWSC 内科诊断后重新设计。
分析发现：online_cum_reward=0.05 vs offline=4.34（87x gap），success_once 后期退化 0.82→0.60。

入口：`bash scripts/sweep_acp/sweep.sh run`
WandB project: `ACP-Sweep`
Log: `logs/vlaw/acp_sweep_awsc_v2.log`

| 组别 | 参数 | Configs |
|------|------|---------|
| baseline | 默认(scale=100,bc=2,or=0.15,γ=0.9) | 1 |
| scale | acp_reward_scale: 500/1000/2000 | 3 |
| bc_weight | awsc_bc_weight: 4.0/8.0 | 2 |
| online_ratio | online_ratio: 0.3/0.5 | 2 |
| gamma | gamma: 0.7/0.5 | 2 |
| combined | 多参数组合(5种) | 5 |
| **Total** | — | **15** |

监控：`bash scripts/sweep_acp/sweep.sh status` / `analyze` / `report`
WandB 分析：`python scripts/sweep_acp/fetch_wandb.py -p ACP-Sweep --save_csv`
已完成分析报告：`logs/vlaw/wandb_analysis/awsc_acp_mirror/analysis_report.md`

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

# ACP 数据多样化采集（rlft_ms3，GPU 2-5，并行）——ADR-039
bash scripts/collect_acp_data.sh          # 采集 Type B/C/D/E 各200/200/200/100条
# 或单独采集某类型：
CUDA_VISIBLE_DEVICES=3 conda run -n rlft_ms3 python scripts/collect_acp_data.py \
  --noise_mode teleop --ou_sigma 0.07 --pause_prob 0.04 \
  --num_episodes 200 --output_dir data/vlaw/rollouts/teleop_sim --gpu_id 3

# ACP 多版本训练（vlaw_reward，GPU 6，顺序）
bash scripts/train_acp_multi.sh           # 训练5个版本（v2_demo_only/teleop/rl_prior/combined等）
bash scripts/train_acp_multi.sh --parallel # 并行，GPU 2-6（数据采集完成后）
bash scripts/train_acp_multi.sh --version v2_combined  # 仅训练 combined

# RLPD SAC + ACP v2（重训，数据修复后）
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_rlpd_sac_acp_v2.sh

# RLPD AWSC + pretrained policy + ACP v2（并行实验）
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_rlpd_awsc_acp.sh

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
    visualize.py         ← ACP 推理诊断可视化（scatter/trajectory/advantage 分布）
    episode_viz.py       ← Episode 级可视化（双相机+value 曲线 → PNG/GIF）
  roboreward/            ← RoboReward 模块（arXiv:2601.00675）
  tests/vlaw/            ← 单元/集成测试（无真实 GPU/权重）
  online/                ← 训练入口脚本
  vlaw/data/
    collector.py         ← P1.1 VLAWDataCollector（生产数据采集）
    noisy_policy.py      ← ACP数据多样化：OUNoisePolicyWrapper(teleop) + GaussianNoisePolicyWrapper(rl_prior)

ctrl_world/              ← Ctrl-World（外部代码，最小修改原则，见 ctrl_world/CLAUDE.md）
scripts/                 ← 辅助脚本
checkpoints/vlaw/        ← 模型权重（见下方资产路径）
data/vlaw/               ← 数据集
docs/vlaw/               ← 技术文档
  acp_pipeline.md        ← ACP 完整 pipeline 文档（图文并茂，含 v2 训练结果）
  gen_acp_figures.py     ← ACP 可视化图表生成脚本（从 wandb 日志解析）
  figures/               ← 生成的图表（9 张 ACP 训练/架构图）
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
| WM iter1_v4（已废弃，BUG-C VAE不匹配） | `checkpoints/vlaw/world_model/iter1_v4/` |
| WM iter1_v5（当前，BUG-A/B/C全修复） | `checkpoints/vlaw/world_model/iter1_v5/` |
| SVD pretrained | `checkpoints/vlaw/world_model/pretrained/svd/` |
| CLIP pretrained | `checkpoints/vlaw/world_model/pretrained/clip/` |
| VLM base（Qwen3-VL-4B） | `checkpoints/vlaw/reward_model/qwen_vl/` (8.3GB) |
| VLM LoRA best | `checkpoints/vlaw/reward_model/ablation_v3/` (r=16, 300步) |
| VLM LoRA baseline | `checkpoints/vlaw/reward_model/lora_v3/` (200步) |
| Policy dry-run | `checkpoints/vlaw/policy/dryrun/` |
| State predictor | `checkpoints/vlaw/state_predictor/` |
| ACP pretrained SigLIP | `checkpoints/vlaw/acp/pretrained/siglip/` (~3.3GB, 428M params) |
| ACP pretrained Gemma | `checkpoints/vlaw/acp/pretrained/gemma/` (~549MB, 268M params) |
| ACP value model iter1 | `checkpoints/vlaw/acp/iter1/` (新设备: demo-only 数据，8000步，MAE=0.0021 过拟合；原设备: 混合1200条，MAE=0.1675) |
| ACP v2_demo_only | `checkpoints/vlaw/acp/v2_demo_only/` (✅ 训练完成，A数据) |
| ACP v2_pretrained_pol | `checkpoints/vlaw/acp/v2_pretrained_pol/` (✅ 训练完成，Type B数据) |
| ACP v2_teleop_sim | `checkpoints/vlaw/acp/v2_teleop_sim/` (✅ 训练完成，**真机遥操作分布**) |
| ACP v2_rl_prior | `checkpoints/vlaw/acp/v2_rl_prior/` (✅ 训练完成，**真机RL微调分布**) |
| ACP v2_combined | `checkpoints/vlaw/acp/v2_combined/` (✅ 训练完成，推荐用于RLPD，A+B+C+D) |
| ACP exp_aligned | `checkpoints/vlaw/acp/exp_aligned/` (Evo-RL 对齐实验, 训练中) |
| ACP dryrun checkpoint | `checkpoints/vlaw/acp/dryrun/` (20步 dry-run, MAE=0.271) |
| ACP 训练报告 | `logs/vlaw/acp_report/ACP_Training_Report.md` (8000步, best MAE=0.1675) |
| ACP 对齐实验 log | `logs/vlaw/acp_exp_aligned.log` |
| ACP 改进计划 | `.claude/plans/modular-finding-llama.md` |
| ACP Pipeline 文档 | `docs/vlaw/acp_pipeline.md`（含 v2 五版本训练结果、9 张图表） |

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
| ACP value MAE | < 0.1 | < 0.05 | 原设备 iter1: 0.1675 (1200条混合数据) / 新设备 iter1 demo-only: 0.0021（过拟合，仅供参考） |
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
| ADR-039 | **ACP 训练数据多样化**：iter1 因 demo-only 数据（25条，MAE=0.0021）严重过拟合。解决方案：采集4种分布 Type B/C/D/E 各100-200条，训练5个ACP版本（v2_demo_only/v2_pretrained_pol/v2_teleop_sim/v2_rl_prior/v2_combined）。Type C 用 OU 噪声（θ=0.15σ=0.07+停顿）模拟**真机遥操作**；Type D 用 i.i.d. Gaussian（σ=0.25）模拟**真机RL微调探索**。实现：`rlft/vlaw/data/noisy_policy.py`（OUNoisePolicyWrapper + GaussianNoisePolicyWrapper）。入口脚本：`scripts/collect_acp_data.sh`→`scripts/train_acp_multi.sh`。 | ✅ 数据采集（1350条）+5版本ACP训练全部完成 |
| ADR-040 | **VAE 编码器不匹配修复 (BUG-C)**：`pipeline.py` 误用 `sd-vae-ft-mse` 的 `AutoencoderKL` 编码训练数据，但 Ctrl-World 训练/推理使用 SVD 的 `AutoencoderKLTemporalDecoder`。两者权重不同导致 latent 分布偏差，WM 在错误 latent 空间上训练。修复：改用 SVD VAE (`AutoencoderKLTemporalDecoder`) 编码，数据重编码为 train_v5。v4 评估: best loss=0.177 但物体动态弱，判定不可用。 | ✅ 代码修复+v5数据重编码完成 |
| ADR-041 | **ACP Mirror Experiments**：用 ACP reward 替换 sim dense reward 运行 AWSC/PLD-SAC/DSRL-SAC 三算法。**核心指标 success_at_end**：仅 AWSC 达到 66%（sim=72%，得益于 BC loss），PLD/DSRL 均 ≤6%（sim=86%/60%）。ACP value 目标为 success_once 语义，无法引导保持行为。`acp_step_mean=0` 已确认为日志 bug。入口：`scripts/run_acp_mirror_experiments.sh`。详细报告：`docs/vlaw/acp_mirror_experiments.md`。 | ✅ 完成 |
| ADR-042 | **AWSC+ACP Sweep v2（数据驱动）**：基于 wandb 分析诊断（`fetch_wandb.py` 拉取 wa52z9ce 训练数据），发现 ACP mirror AWSC 3 个核心问题：(1) online_cum_reward=0.05 vs offline=4.34（87x gap，critic 被 demo 信号主导）；(2) success_once 后期退化 0.82→0.60（BC 锚定不足）；(3) advantage_mean≈0.8 正偏高。扫描 3 轴：A 放大 ACP 信号（scale 500-2000, online_ratio 0.3-0.5），B 防遗忘（bc_weight 4-8），C 缩短信用分配（gamma 0.5-0.7）。15 configs，仅 AWSC（PLD/DSRL 暂停）。入口：`scripts/sweep_acp/sweep.sh`。分析工具：`scripts/sweep_acp/fetch_wandb.py`。 | 🔄 运行中 |

| ADR-043 | **WM v5 审查 + Bug 分析 (BUG-D/E/F/G/H)**：20 checkpoint 视觉审查，peg 静止 1/10。BUG-D=tiled future EE pose（fix1 integrate_delta 失败，效果更差）。BUG-E=v5 latent≈v4（编码端无差异）。BUG-F=ManiSkill 无时间跳跃。BUG-G=Action Encoder 训练不足。BUG-H=history init 不足（已修复）。**Fix1 失败后需重新分析路线**：推理端修正 or 训练端 v6。 | ❌ Fix1 失败 |

完整决策记录：`.github/knowledge/decisions.md`（43 条 ADR）

---

## 知识库索引（`.github/knowledge/`）

| 文件 | 内容 |
|------|------|
| `decisions.md` | 43 条 ADR，全部架构决策 |
| `bugs-and-fixes.md` | 27 个 Bug 记录（BUG-001 ~ BUG-027） |
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
