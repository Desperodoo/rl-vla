# rl-vla — Claude Code 记忆文件

## 项目概述
复现 VLAW 论文 (arXiv:2602.12063)：ManiSkill3 仿真环境中，ShortCut Flow 策略 + Ctrl-World 视频扩散世界模型 + VLM 奖励模型的迭代共同改进。核心算法：K_iter=2 次迭代，每轮收集真实轨迹 → VLM 标注 → 微调 WM → Imagination 生成合成数据 → 策略更新。

技术栈：ShortCut Flow / Flow Matching（策略）| Ctrl-World SVD-UNet ~1.5B（世界模型）| Qwen3-VL-4B LoRA（奖励模型）| **Pistar06 Value Model（ACP 稠密 advantage）** | ManiSkill3（仿真环境）

---

## 当前项目状态（手动更新）

> ⚠️ 本文件由**两台设备**共享。`原设备`运行 WM 训练；`新设备（10x RTX 4090）`运行 ACP/RLPD 支线任务。

---

### 原设备状态

**阶段**：Phase 2.5 — BUG-D Fix2（pd_ee_pose 迁移，WM v5 不重训）
**已修复**：
- **BUG-A (ADR-037, 2026-03-08)**：WM action conditioning 语义错配 — 从 delta pose 改为**绝对 EE 位姿** (对齐 DROID)。stat.json 已重新生成。
- **BUG-B (ADR-037, 2026-03-08)**：Camera VAE 编码差异 — 从像素空间拼接改为**独立 per-camera VAE 编码后 latent 空间拼接**。
- **BUG-C (ADR-040, 2026-03-11)**：VAE 编码器不匹配 — `pipeline.py` 误用 `sd-vae-ft-mse` 的 `AutoencoderKL` 编码，而 Ctrl-World 训练/推理使用 SVD 的 `AutoencoderKLTemporalDecoder`。两者权重不同，latent 分布存在偏差。已改为使用 SVD VAE 编码，数据重编码为 train_v5。⚠️ 后续发现 BUG-E：编码端两者权重几乎相同（corr=0.999999），实质影响仅在解码端。
- **BUG-D (ADR-043+045, 2026-03-14~15)**：**[CRITICAL → 修复中] Imagination 推理 future actions 使用 tiled 当前 EE pose**。详见下方 WM v5 诊断报告。**Fix2 方案**：Policy 从 `pd_ee_delta_pose` 切换到 `pd_ee_pose`（绝对位姿），Imagination 中用 `ee_pose_base_to_world()` 转换 base→world frame（仅加 `[-0.615, 0, 0]`）。WM 不需要重训——其 action conditioning 来自 `state_to_ee_pose_7d()`（world frame），与 control_mode 无关。
- **BUG-E (ADR-043, 2026-03-14)**：V5 latents 与 V4 几乎相同（corr=0.999999），BUG-C 修复对编码端无实质影响。SVD VAE 时序修改集中在解码器。
- **BUG-H (ADR-043, 2026-03-14)**：ee_pose_history 初始化仅 1 条，应为 num_history*4=24 条（对齐 latent_history 和官方代码）。已修复。

**WM v5 Imagination 诊断报告 (2026-03-14~15)**：
通过 7 组受控消融实验（`scripts/vlaw/diagnostic/wm_diagnostic_battery.py`）精确隔离了 Imagination 质量退化的每个因素。完整报告：`results/vlaw/wm_diagnostic/DIAGNOSTIC_REPORT.md`、`results/vlaw/wm_diagnostic/BUG_D_EXPLAINED.md`。

| 排名 | 因素 | PSNR 影响 | 实验组 | 结论 |
|------|------|----------|--------|------|
| **1** | **BUG-D: Action tiling（future EE pose 全相同）** | **-4.5 ~ -8.5 dB** | Group A/C/F2 | **唯一显著根因** |
| 2 | num_inference_steps 50→25 | ~0.8 dB | Group B | 次要，非单调 |
| 3 | 自回归误差累积（6 chunk AR） | ~0 dB | Group D_MC | **完全不是问题** |
| 4 | History 采样策略（sparse vs contiguous） | <0.5 dB | Group E | 无影响 |
| 5 | History latent 噪声敏感度 | PSNR 反而上升 | Group D3 | WM 非常鲁棒 |

**关键实验**：
- **Group C Alpha Sweep**：`future_ee = current_ee + α*(gt_ee - current_ee)`。α=1.0 (GT) 是明确最优点，单调升再单调降。动态大的样本 GT vs tiled 差距高达 **8.5 dB**（34.12 vs 26.60 dB）。
- **Group D_MC (6-chunk AR)**：Oracle history vs predicted history，mean gap = **-0.2 dB**（AR 甚至略好）。无随 chunk 递增的退化趋势 → 自回归误差累积在 30 帧内完全不是问题。
- **Group A**：GT actions=36.85, Tiled=32.35, Zero=29.77, Random=25.61 dB → 证实 WM 确实使用 action conditioning。

**BUG-D 根本矛盾**：WM 需要 absolute EE pose（训练时从 `state[18:21]` 提取）；Policy 输出 delta action（`pd_ee_delta_pose`）；两者之间转换需 PD 控制器 + 物理仿真——而这恰是 Imagination 想绕开的。

**Fix1 验证 (2026-03-14) — ❌ 失败**：`integrate_delta_to_ee_poses()` 将 policy delta 积分为绝对 EE pose。失败原因：PD 控制器使 raw delta ≠ 实际 EE 位移，积分 5 帧后累积误差超出 WM 训练分布。Fix1 输出比 tiled 更差。

**Fix2 方案 (ADR-045, 2026-03-15~16) — ❌ 方案失败，需重新设计**：将 Policy action space 从 `pd_ee_delta_pose` 切换到 `pd_ee_pose`。Policy 直接输出绝对 EE 位姿（robot base frame），Imagination 只需加上 robot root offset `[-0.615, 0, 0]` 即可对齐 WM 的 world frame。WM 不需要重训（action conditioning 来自 state vector，与 control_mode 无关）。

Fix2 进度：

| Step | 内容 | 状态 |
|------|------|------|
| Step 0 | pd_ee_pose demo 转换 | ❌ **Demo 无效**（见下方根因分析） |
| Step 0.5 | 生成带 RGB obs 的 demo + euler_rx unwrap | ✅ 完成（669 traj, 27951 frames, 但基于无效 demo） |
| Step 1 | 训练 ShortCut Flow IL policy | ❌ 两轮训练均 0% 成功率 |
| Step 2 | Imagination loop 代码修改（`ee_pose_base_to_world`） | ✅ 完成 |
| Step 3 | Imagination PSNR 验证 | ⛔ 被 Step 1 阻塞 |

**Step 0 根因分析 (2026-03-16) — ❌ Demo 转换方案根本性缺陷**：

`convert_demos_to_ee_pose.py` 的方案 `action[t] = EE_pose(state[t+1])` **在物理层面上就不可行**：

1. **PD 控制器不会在 1 步内到达目标**：pd_ee_pose 使用 PD 控制器（stiffness=1000, damping=100），给定目标 1cm 远需要 ~5 步才收敛。因此 `target[t] = position[t+1]` 时，实际到达位置 = `position[t] + 0.4*(position[t+1]-position[t])`（约 40% 衰减系数），轨迹严重滞后。
2. **Demo replay 验证**：用 `set_state_dict()` 恢复 traj_0 初始状态后，用 demo actions 重放 — **0/20 轨迹成功**，pos_err 从 0.019 累积到 0.059。
3. **Euler angle 误差更严重**：euler_err 从第 1 步起就高达 6.3（≈ 2π），原因是 `scipy.Rotation.as_euler('xyz')` 返回 [-π, π] 但 unwrap 后变为 [0, 2π]，PD 控制器按最短路径旋转时被误导。
4. **ManiSkill 官方确认**：`replay_trajectory.py` 第 308-310 行明确声明 "Cannot use env states when trying to convert from one control mode to another. This is because control mode conversion causes there to be changes in how many actions are taken to achieve the same states"。

**两轮 IL 训练尝试详情**：

| Run | wandb | Steps | Loss | Success Rate | 失败原因 |
|-----|-------|-------|------|-------------|---------|
| il_shortcut_flow_pd_ee_pose_s42 | u1b7d3fa | 175K | 0.013 | **0%** | euler_rx ±π 双峰分布 + demo 无效 |
| il_shortcut_flow_pd_ee_pose_unwrap_s42 | riv6riu7 | 175K | 0.013 | **0%** | demo 本身无效（见上方根因分析） |

euler_rx 双峰问题（已修复但不影响结论）：653/669 轨迹（97.6%）的 euler_rx 存在 ±π 大跳变（44% 近 -π，56% 近 +π）。Flow Matching 平均两模态 → 预测 euler_rx ≈ 0（完全错误）。修复：负值 += 2π 映射到 [0, 2π] 单峰分布。已 commit (8ad4b2a)。但由于 demo 根本无效，修复 euler 后仍 0% 成功率。

**Step 0.5 详情**：新建 `scripts/generate_rgb_demos_ee_pose.py`，从 `trajectory.none.pd_ee_pose.physx_cpu.h5` 重放 env_states 获取 RGB obs。输出：`trajectory.rgb.pd_ee_pose.physx_cpu.h5`（669 traj, 27951 frames, 1357.6 MB, 73秒生成）。内含 euler_rx unwrap（负→+2π）。但底层 demo 无效，RGB obs 只是原始 delta_pose 轨迹的快照，不对应 pd_ee_pose 的实际动态。

**Step 2 详情**：`imagination_env.py` 新增 `ee_pose_base_to_world()` 函数（`ROBOT_ROOT_POS=[-0.615, 0, 0]`，仅平移位置维度 + euler re-wrap [-π,π]），替换 `_run_rollout_in_env()` 和 `_batch_rollout_in_env()` 中已废弃的 `integrate_delta_to_ee_poses()` 调用。`imagination_rl_env.py` 同步替换 `np.tile(current_ee, ...)` 为 `ee_pose_base_to_world(action_chunk)`。**代码逻辑正确，但依赖 Step 1 提供有效策略。**

**BUG-D Fix2 失败后的可选方向**：

| 方向 | 描述 | 可行性 |
|------|------|--------|
| **A: pd_joint_delta_pos → pd_ee_pose 两步转换** | ManiSkill 支持 `pd_joint_pos → pd_ee_pose`（`from_pd_joint_pos_to_ee`）。现有 `trajectory.none.pd_joint_delta_pos.physx_cuda.h5`（993 traj）。路径：pd_joint_delta_pos → pd_joint_pos → pd_ee_pose。但 pd_joint_delta_pos→pd_joint_pos 转换需验证。 | ⚠️ 需验证 |
| **B: 直接用 motion planning 生成 pd_ee_pose demo** | 用 ManiSkill 的 motion planner 在 pd_ee_pose 控制模式下从头生成 demo，跳过转换。 | ⚠️ 需确认 MP 是否支持 pd_ee_pose |
| **C: 放弃 pd_ee_pose，改善 Imagination 的 action 方案** | 不改变 policy action space，在 Imagination 中用 sim env 跑 1 步 delta pose 获取真实 EE pose（1-step sim-in-loop）。牺牲速度但保证精度。 | ✅ 最可靠 |
| **D: 训练 delta→ee 转换网络** | 训练小 MLP 把 (current_state, delta_action) 映射到 next_ee_pose。需收集训练数据对。 | ⚠️ 间接方案 |

**pd_ee_pose 关键技术细节**：
- 7D: `[target_xyz, euler_xyz, gripper]`，robot base frame（非 world frame）
- `normalize_action=False`，unbounded action space（arm 维度 NaN bounds，gripper [-1,1]）
- Robot root: `[-0.615, 0, 0]`，identity rotation → world_pos = base_pos + root_pos，euler 不变
- PD 控制器特性（stiffness=1000, damping=100）：1cm 目标需 ~5 步收敛，不适合 1-step action 转换
- 训练时用 `ActionNormalizer(minmax)` 映射到 [-1,1]，与 `ShortCutFlow.action_bounds=(-1,1)` 兼容
- `CtrlWorldAdapter.rollout()` 内部用 stat.json 的 world frame percentiles 归一化 → 传入 world frame EE 即可

**待消融 (BUG-F — 时间跳跃数据增强)**：
- 官方 DROID 训练使用随机时间跳跃 `skip=randint(1,2)`, `skip_his=skip*4` (15% prob→0)
- 我们的 ManiSkill dataset 使用严格连续帧 (`skip_step=1`)
- **优先级**：BUG-D Fix2 验证后 → 若 peg 动态恢复但不足 → 作为 WM v6 消融项

**待改进 (BUG-G — Action Encoder 训练)**：
- Action Encoder (2.11M) 随机初始化，与 UNet (1524M) 共用 LR=1e-5，无 warmup
- **优先级**：BUG-D Fix2 验证后 → 若 peg 动态恢复但不足 → 与时间跳跃一起作为 WM v6

**原设备 GPU 分配**：

| GPU | 任务 | 状态 |
|-----|------|------|
| 0-1 | LMStudio | 占用 |
| 2-9 | 空闲（BUG-D Fix2 Step 1 训练待启动） | — |

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

**ACP v3 数据采集（ADR-044, 2026-03-14）— ✅ 完成**：

根因分析：ManiSkill 在 success 时立即 early-terminate episode，导致 v2 数据中 success_once ≈ success_at_end（99.84% 一致），ACP 无法学习"保持"语义。
修复：`ignore_terminations=True` 强制 episode 运行到 max_episode_steps，产生"成功后掉落"轨迹。
策略：PLD-SAC s42（SO=100%, SAE=86%），比之前 AWSC s42（SO=92%, SAE=72%）更强。

v3 数据采集结果（`ignore_terminations=True` + PLD-SAC s42 策略）：

| Type | 分布 | Trajs | Frames | S_once | S_end | Mismatch |
|------|------|-------|--------|--------|-------|----------|
| A-demo | expert demos（原有） | 50 | 510 | 50 (100%) | 48 (96%) | 2 (4%) |
| B-pld_pretrained | PLD-SAC s42 无噪声 | 400 | 13,488 | 234 (58.5%) | 121 (30.2%) | **113 (28.2%)** |
| C-pld_teleop | PLD-SAC s42 + OU噪声 | 400 | 13,488 | 58 (14.5%) | 1 (0.2%) | **57 (14.2%)** |
| D-pld_rl_prior | PLD-SAC s42 + Gaussian噪声 | 400 | 13,488 | 21 (5.2%) | 1 (0.2%) | **20 (5.0%)** |
| E-pld_random | 纯随机 | 100 | 3,500 | 0 | 0 | 0 |
| **Total** | — | **1,350** | **44,474** | **363 (26.9%)** | **171 (12.7%)** | **192 (14.2%)** |

数据路径：`data/vlaw/rollouts/v3_pld_{pretrained,teleop,rl_prior,random}/`
验证脚本：`scripts/validate_v3_data.py`
v2 对比：mismatch 0.0% → **14.2%**（192 条轨迹有"成功后掉落"信号）

代码修改：
- `rlft/vlaw/data/collector.py`：新增 `ignore_terminations` 配置
- `scripts/collect_acp_data.py`：新增 `--ignore_terminations` CLI 参数
- `rlft/vlaw/acp/config.py`：新增 `success_mode` 配置（`success_once`/`success_at_end`）
- `rlft/vlaw/acp/value_targets.py`：支持 `success_mode` 分支
- `rlft/vlaw/acp/hdf5_dataset.py`：支持 `success_mode` 传递

**ACP v3 训练（2026-03-14）— ✅ 完成**：

两版并行训练（GPU 6/7），使用 v3 数据（A+B+C+D combined, 1250 traj, 40974 帧），对比 success_once vs success_at_end 标签：

| 版本 | success_mode | Steps | bs | Best MAE | Inference MAE | Pearson r | Checkpoint |
|------|-------------|-------|-----|----------|--------------|-----------|-----------|
| v3_so | success_once | 12,000 | 128 | 0.0724 | 0.0714 | 0.8851 | `checkpoints/vlaw/acp/v3_so/best.safetensors` ✅ |
| v3_sae | success_at_end | 12,000 | 128 | **0.0463** | **0.0452** | **0.9219** | `checkpoints/vlaw/acp/v3_sae/best.safetensors` ✅ |

v3_sae MAE 优 36%，Pearson r 优 4.2%。success_at_end 标签与视觉终态一致，更容易学习。
详细对比报告：`docs/vlaw/acp_v3_so_vs_sae_report.md`

**ACP v3 RLPD 实验（2026-03-16）— ✅ 全部完成**：

3 算法 × 2 ACP 版本 = 6 组在线 RLPD 实验。入口：`scripts/run_acp_v3_experiments.sh`。WandB project: `rlpd-acp-v3`。

| 实验 | GPU | Steps | Best SO | Best SAE | Final SO | Final SAE | 状态 |
|------|-----|-------|---------|----------|----------|-----------|------|
| AWSC + v3_so | 0+1 | 500K | 90% | **68%** | 42% | 40% | ✅ 完成 |
| AWSC + v3_sae | 2+3 | 500K | **92%** | 66% | 52% | 50% | ✅ 完成 |
| PLD + v3_so | 4+5 | 71K | 80% | 8% | 70% | 0% | ✅ 完成 |
| PLD + v3_sae | 6+7 | 71K | 50% | 16% | 2% | 0% | ✅ 完成（❌ 灾难性崩溃） |
| DSRL + v3_so | 8+9 | 71K | **94%** | 6% | 76% | 4% | ✅ 完成 |
| DSRL + v3_sae | 4+5 | 71K | — | — | — | — | ✅ 完成 |

WandB runs: AWSC+so=7weycepc, AWSC+sae=d6wfjs2f, PLD+so=ynp44qlz, PLD+sae=4hjfih2f, DSRL+so=m4wgw4ku, DSRL+sae=1blrmq2r
内科诊断报告：`docs/vlaw/figures/rlpd_acp_v3_internals/diagnosis_report.md`（由 `scripts/analyze_training_internals.py` 生成）

**核心发现**：
- **v3_sae 未表现出预期优势**：尽管模型质量优 36%，RLPD 效果与 v3_so 基本相同（AWSC SAE: 66% vs 68%）
- **v3 对比 v2 无显著改善**：AWSC SAE 68% vs v2 66%，仅 +2%（统计波动范围内）
- **PLD + v3_sae 灾难性崩溃**：SO 从 82% 降至 2%，无 BC 锚定下 v3_sae reward 信号毁灭性
- **DSRL + v3_so 最高 SO=94%**：DSRL 保守正则化有效，但 SAE 仍 ≤6%
- **根因**：success_at_end 信号在 TD reward 框架中被稀释（仅影响 15.4% mismatch 数据 → TD差异 → critic估计衰减 → 策略梯度微乎其微）
- 详细报告：`docs/vlaw/acp_v3_rlpd_report.md`
- 分析脚本：`scripts/analyze_acp_v3_rlpd_results.py`

**Pipeline 修复（ADR-046, 2026-03-16）— ✅ 已完成**：

基于 v3 内科诊断报告的处方，对三个训练脚本进行增强：

| 文件 | 修改内容 |
|------|---------|
| `rlft/online/train_rlpd.py` | 新增 early stopping（patience/threshold/min_steps Args）+ SAE-aware `best_sae.pt` checkpoint |
| `rlft/online/train_pld.py` | 新增 SAE-aware `best_sae.pt` checkpoint + `best_success_at_end` wandb 日志 |
| `rlft/online/train_dsrl.py` | 新增 SAE-aware `best_sae.pt` checkpoint + `best_success_at_end` wandb 日志 |
| `scripts/analyze_training_internals.py` | 新增通用内科诊断脚本（替代 `analyze_rlpd_internals.py`） |
| `.claude/skills/training-internals/SKILL.md` | 新增通用诊断 skill（替代旧 `rlpd-diagnosis`） |

Early stopping 参数（AWSC 专用）：`--early_stop --early_stop_patience 5 --early_stop_so_threshold 0.8 --early_stop_min_steps 100000`

**ACP v4 RLPD 实验（2026-03-16）— 🔄 运行中**：

基于 v3 内科诊断处方，4 组精调实验。入口：`scripts/run_acp_v4_experiments.sh`。WandB project: `rlpd-acp-v4`。

| 实验 | GPU | 核心变更 | 状态 | Log |
|------|-----|---------|------|-----|
| AWSC + bc=4 + scale=500 | 0+1 | bc_weight 2→4, scale 100→500, early_stop | 🔄 运行中 (PID=3272815) | `logs/vlaw/acp_v4_awsc_bc4_s42.log` |
| AWSC + bc=8 + scale=500 | 2+3 | bc_weight 2→8, scale 100→500, early_stop | 🔄 运行中 (PID=3272816) | `logs/vlaw/acp_v4_awsc_bc8_s42.log` |
| PLD + gamma=0.7 + scale=500 | 4+5 | **gamma 0.99→0.7**, scale 100→500 | 🔄 运行中 (PID=3308470) | `logs/vlaw/acp_v4_pld_gamma07_s42.log` |
| DSRL + gamma=0.7 + scale=500 | 6+7 | **gamma 0.95→0.7**, scale 100→500 | 🔄 运行中 (PID=3308471) | `logs/vlaw/acp_v4_dsrl_gamma07_s42.log` |

处方变更对比：
- AWSC: scale 100→500（+5x），bc_weight 2→4/8（+2x/4x），新增 early stopping，仅用 v3_so
- PLD: **gamma 0.99→0.7**（修复 Q-value 暴涨，v3 Q_range=114-140），scale 100→500
- DSRL: **gamma 0.95→0.7**（修复 Q-value 暴涨，v3 Q_range=76-86），scale 100→500

v3 PLD/DSRL 失败根因澄清：
- 诊断报告误标为 "sim reward drowning" → 实际确认 `DualCameraRewardWrapper` **完全替换**了 sim reward
- **真正根因**：gamma 过高导致 Q-value 暴涨（PLD=114-140, DSRL=76-86 vs AWSC=3.9），critic 不稳定
- v4 修复：降低 gamma 到 0.7，压缩 Q-value scale，稳定 critic 训练

**下一步方向（优先级排序）**：
- **P0 — 监控 v4 实验（4 组）**：AWSC 关注 early stop 触发 + SAE 突破；PLD/DSRL 关注 Q-value 是否稳定 + SAE 改善
- **P1 — 分析 AWSC Sweep v2 结果**：`/training-internals rlpd-acp-v4` 到位后与 sweep 结果交叉验证
- **P2 — 探索其他 reward 设计**：直接 value 作 reward、ACP-guided demo selection

**新设备 GPU 分配（当前）**：

| GPU | 任务 | 状态 |
|-----|------|------|
| 0+1 | AWSC v4 + bc=4 + scale=500 (500K max) | 🔄 运行中 |
| 2+3 | AWSC v4 + bc=8 + scale=500 (500K max) | 🔄 运行中 |
| 4+5 | PLD v4 + gamma=0.7 + scale=500 (71K) | 🔄 运行中 |
| 6+7 | DSRL v4 + gamma=0.7 + scale=500 (71K) | 🔄 运行中 |
| 8-9 | 空闲 | — |

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
| ACP v3_so | `checkpoints/vlaw/acp/v3_so/` (✅ v3 数据 + success_once 标签, 4000步) |
| ACP v3_sae | `checkpoints/vlaw/acp/v3_sae/` (✅ v3 数据 + success_at_end 标签, 4000步) |
| ACP dryrun checkpoint | `checkpoints/vlaw/acp/dryrun/` (20步 dry-run, MAE=0.271) |
| ACP 训练报告 | `logs/vlaw/acp_report/ACP_Training_Report.md` (8000步, best MAE=0.1675) |
| ACP 对齐实验 log | `logs/vlaw/acp_exp_aligned.log` |
| ACP 改进计划 | `.claude/plans/modular-finding-llama.md` |
| BUG-D Fix2 计划 | `.claude/plans/toasty-imagining-moon.md` |
| WM 诊断报告 | `results/vlaw/wm_diagnostic/DIAGNOSTIC_REPORT.md` |
| BUG-D 深度解析 | `results/vlaw/wm_diagnostic/BUG_D_EXPLAINED.md` |
| WM 诊断脚本 | `scripts/vlaw/diagnostic/wm_diagnostic_battery.py` |
| pd_ee_pose Demo 转换脚本 | `scripts/convert_demos_to_ee_pose.py` |
| pd_ee_pose RGB Demo 生成脚本 | `scripts/generate_rgb_demos_ee_pose.py` |
| pd_ee_pose Demo（无 obs，❌ 无效） | `~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.none.pd_ee_pose.physx_cpu.h5` (669 traj, 27951 frames, demo replay 0% 成功) |
| pd_ee_pose Demo（RGB obs，❌ 无效） | `~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_pose.physx_cpu.h5` (669 traj, 27951 frames, 1357.6 MB, 基于无效 demo) |
| pd_joint_delta_pos Demo | `~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5` (993 traj, 可能可用于方向 A 转换) |
| ACP v3 内科诊断报告 | `docs/vlaw/figures/rlpd_acp_v3_internals/diagnosis_report.md`（由 analyze_training_internals.py 生成） |
| ACP v3 分析报告 | `docs/vlaw/acp_v3_at_end_report.md`（v2 vs v3 对比，5 张图表） |
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

| ADR-043 | **WM v5 诊断 + Bug 分析 (BUG-D/E/F/G/H)**：7 组受控消融实验精确定位根因。BUG-D=**唯一显著根因**（-4.5~-8.5 dB），其余因素（AR误差/steps/history采样/history噪声）均<1 dB。Fix1 (integrate_delta) ❌ 失败。Alpha sweep 证实 α=1.0 (GT) 最优。D_MC 实验证实 AR 误差累积在 30 帧内不是问题。报告：`results/vlaw/wm_diagnostic/DIAGNOSTIC_REPORT.md`、`BUG_D_EXPLAINED.md`。诊断脚本：`scripts/vlaw/diagnostic/wm_diagnostic_battery.py`。 | ✅ 诊断完成，Fix1 ❌ |
| ADR-045 | **BUG-D Fix2：pd_ee_pose 迁移（消除 WM-Policy 动作空间鸿沟）— ❌ 方案失败**：根本原因：pd_ee_pose PD 控制器不会在 1 步内到达目标（stiffness=1000, 1cm 需 ~5 步收敛），从 delta_pose 轨迹 env_states 提取 `action[t]=EE(state[t+1])` 的转换方式产生持续滞后；demo replay 0/20 成功。euler_rx 2π 偏移进一步恶化（控制器收到 3.15 但实际需 -3.13）。ManiSkill 官方也声明不支持 env_states + 控制模式转换。两轮 IL 训练均 0% 成功。Imagination loop 代码修改（`ee_pose_base_to_world`）逻辑正确但被阻塞。需要新方案：方向 A（pd_joint_delta_pos → pd_ee_pose 两步转换）/ 方向 B（motion planner 直接生成）/ 方向 C（1-step sim-in-loop）/ 方向 D（delta→ee MLP）。 | ❌ 方案失败，需重新设计 |
| ADR-044 | **ACP v3 数据多样化 + success_at_end 支持**：(1) ManiSkill early-termination 导致 v2 数据 success_once≈success_at_end（0% mismatch），ACP 无法学习"保持"语义。(2) `ignore_terminations=True` 强制 episode 运行到 max_episode_steps。(3) 改用 PLD-SAC s42（SO=100%,SAE=86%）替代 AWSC s42。(4) config 新增 `success_mode` 支持 `success_once`/`success_at_end` 两种标签。v3 数据 mismatch=14.2%（192/1350 条）。v3_so/v3_sae 两版 ACP 训练完成（4000 steps each）。 | ✅ 数据+训练完成 |
| ADR-046 | **ACP v4 Pipeline 修复 + 通用诊断工具**：基于 v3 内科诊断报告处方。(1) `train_rlpd.py` 新增 early stopping（AWSC 专用）+ SAE-aware `best_sae.pt`；(2) `train_pld.py`/`train_dsrl.py` 新增 SAE checkpoint；(3) `scripts/analyze_training_internals.py` — 通用五维诊断脚本（替代 hardcoded `analyze_rlpd_internals.py`）；(4) `/training-internals` skill 替代旧 `/rlpd-diagnosis`；(5) `scripts/run_acp_v4_experiments.sh` — 4 组实验：AWSC(bc=4/8,scale=500,early_stop) + PLD(gamma=0.7,scale=500) + DSRL(gamma=0.7,scale=500)；(6) v3 PLD/DSRL 失败根因澄清：非 sim reward 泄漏，而是 gamma 过高致 Q-value 暴涨。v4 实验已于 2026-03-16 启动。 | 🔄 v4 实验运行中 |

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
