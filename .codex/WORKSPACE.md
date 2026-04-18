# rl-vla Workspace Guide

## 1. 项目定位

本仓库同时包含两条主线：

1. VLAW 复现主线
   - 目标：复现 VLAW 论文（arXiv:2602.12063）
   - 核心组合：ShortCut Flow 策略 + Ctrl-World 视频扩散世界模型 + VLM 奖励模型
   - 典型迭代：真实轨迹收集 -> VLM 标注 -> 世界模型微调 -> imagination 生成合成数据 -> 策略更新 -> 评估

2. CARM 真机主线
   - 目标：CARM 机械臂在 ROS1 环境中的 teleop、inference、离线训练与 second-stage 回流闭环
   - 关键模块：`carm_ros_deploy/`、`arm_control_sdk/`、`rlft/offline/`

这两条主线共享部分训练基础设施，但约束不同：
- VLAW 偏仿真、多阶段 research pipeline
- CARM 偏真机、部署与数据契约稳定性

## 2. 当前仓库重点

### 2.1 VLAW 研究主线

当前阶段性结论：
- Phase 0 数据：已完成
- Phase 1 世界模型：已完成到 `iter1_v5` 训练阶段
- Phase 2 VLM 奖励模型：LoRA v3 已完成，FP=0%，Recall=61.2%
- Phase 3 Imagination：仍围绕 BUG-D 进行验证
- Phase 4 策略更新：等待 Phase 3 稳定
- Phase 5 评估：等待 Phase 4

关键阻塞曾经是 BUG-D：
- 症状：imagination 中 future action 使用 tiled 当前 EE pose，导致 peg 静止
- 根因：世界模型训练依赖 absolute EE pose，而 policy 输出 delta action，二者之间缺少可靠转换
- 已验证失败方案：
  - delta 积分
  - `pd_ee_pose` 迁移
- 已有阶段性有效修复：
  - Dynamics Adapter V1：PSNR 29.59 -> 30.51 dB，恢复约 42% gap

当前 VLAW 工作重点：
- 用 V1 adapter 跑全量 imagination
- 做 VLM 标注并比较 `D_syn+` 产出率
- 再决定下游 policy update

### 2.2 CARM 真机主线

当前主线已经完成的关键闭环：
- teleop / inference 时间戳语义对齐到 `obs_stamp_ros`
- teleop / inference camera 契约对齐
- teleop backend 假超时问题修复
- 主训练链默认启用 inactive teleop 过滤
- inference recorder -> 训练侧 schema 守卫补齐
- inference raw -> staging -> `train_carm` second-stage smoke 路径打通

当前重点已经从“能不能读”转向：
- 哪些 inference episode 可以回流训练
- 怎样做正式 admission 与分桶
- 怎样把 second-stage 训练入口标准化
- 如何用 baseline/redeploy 验证效果

### 2.3 script policy 执行层主线

当前已经新增并持续推进第三条主线：
- `script_runtime/`

它的定位不是训练，而是：
- 基于 `arm_control_sdk` 的 script policy 执行层
- 用统一 skill / task / blackboard / trace 抽象承载真实世界任务执行
- 先在 ManiSkill 中验证执行语义，再迁移真机

当前阶段性状态：
- `script_runtime/` 已具备可执行的 pick-place runtime
- 已有统一 `Skill` / `WorldState` / `TaskBlackboard` / `FailureCode`
- 已有轻量行为树执行器和 `session` 装配入口
- 已有 `arm_control_sdk` 桥接与 ManiSkill 桥接
- 已有 `PickCube-v1` 与 `StackCube-v1` 两条验证任务线

当前最新验证结论：
- `PickCube-v1`：`runtime_successes=3/3`, `sim_successes=3/3`
- `StackCube-v1`：`runtime_successes=2/3`, `sim_successes=2/3`

当前最关键的工程判断：
- 任务树结构、环境成功验收和 release 语义已经基本打通
- 当前剩余主要瓶颈是抓取鲁棒性，不是调度结构

## 3. 仓库结构

```text
arm_control_sdk/                        CARM SDK 与 Python 绑定
carm_ros_deploy/                        ROS1 部署工作区
ctrl_world/                             外部 Ctrl-World 代码，最小修改原则
docs/                                   实验、审计、分析、计划文档
recorded_data/                          CARM teleop 数据
inference_logs/                         CARM inference 日志与 episode
rlft/                                   训练、评估、offline/online 算法实现
scripts/                                构建、分析、转换、实验辅助脚本
.claude/                                历史 Claude 记忆文件
.codex/                                 当前 Codex 工作区文档
script_runtime/                         SDK-first script policy 执行层
```

VLAW 相关关键目录：

```text
rlft/
  algorithms/il/         ShortCut Flow / Flow Matching
  algorithms/online_rl/  PLD-SAC / DSRL-SAC
  envs/                  ManiSkill 封装与 evaluate
  online/                在线 RL 训练入口
  vlaw/                  VLAW 核心模块
docs/vlaw/               VLAW 报告、分析与图表
checkpoints/vlaw/        VLAW 权重
data/vlaw/               VLAW 数据集
```

CARM 相关关键目录：

```text
carm_ros_deploy/src/carm_deploy/
  core/                  机器人环境、安全、执行逻辑
  inference/             推理节点
  data/                  录制、分析、转换
  launch/                ROS launch
rlft/offline/            CARM 离线训练与评估
recorded_data/           teleop 数据集
inference_logs/          inference episode / timeline / run_info
script_runtime/          真机 / 仿真共用 script policy runtime
```

script policy 相关关键目录：

```text
script_runtime/
  adapters/              SDK / ManiSkill / learned 统一桥接
  core/                  blackboard / failure code / result / skill base
  executors/             轻量行为树与 trace recorder
  skills/                motion / gripper / perception / checks / recovery
  tasks/                 pick-place 主任务树
  validation/            rollout / report / 可视化
  configs/tasks/         PickCube / StackCube ManiSkill 配置
  artifacts/             trace / gif / grounding / report
```

## 4. 核心环境

### 4.1 Conda 环境

| 环境 | 用途 |
|------|------|
| `rlft_ms3` | ManiSkill、VLAW 数据/策略训练/评估、CARM 第二阶段训练 |
| `ctrl_world` | Ctrl-World 世界模型训练与推理 |
| `vlaw_reward` | Qwen3-VL LoRA 训练与批量标注 |
| `carm` | CARM 真机本地开发环境（依 README 场景） |

### 4.2 网络代理

VLAW 历史工作流常用：

```bash
export http_proxy=http://10.20.93.149:7890
export https_proxy=http://10.20.93.149:7890
```

CARM 真机链路要注意：
- 对机器人内网地址必须显式避开代理
- `10.42.0.101` 与 `10.42.0.0/16` 应在 `no_proxy/NO_PROXY` 中

## 5. 常用命令

### 5.1 VLAW

```bash
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

# 评估
CUDA_VISIBLE_DEVICES=9 conda run -n rlft_ms3 python rlft/envs/evaluate.py

# 测试
conda run -n rlft_ms3 python -m pytest rlft/tests/vlaw/ -v --tb=short -q
```

### 5.2 CARM

```bash
# 环境初始化
source scripts/setup_carm_env.sh

# 编译 catkin
./scripts/build_catkin.sh

# 相机
roslaunch carm_deploy camera.launch

# 数据录制
roslaunch carm_deploy record.launch output_dir:=~/rl-vla/recorded_data

# 推理
roslaunch carm_deploy full_system.launch pretrain:=/path/to/model.pt
roslaunch carm_deploy inference.launch pretrain:=/path/to/model.pt
rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt

# second-stage baseline 入口（仓库已有命名示例）
conda run -n rlft_ms3 python -m rlft.offline.train_carm_stage2 --help
```

### 5.3 script_runtime

```bash
# 单次 PickCube 验证
conda run -n carm python -m script_runtime.runners.maniskill_pick_cube \
  --config script_runtime/configs/tasks/pick_cube_maniskill.yaml

# 批量 PickCube 验证
conda run -n carm python -m script_runtime.runners.maniskill_validate \
  --config script_runtime/configs/tasks/pick_cube_maniskill.yaml

# 单次 StackCube 验证
conda run -n carm python -m script_runtime.runners.maniskill_pick_cube \
  --config script_runtime/configs/tasks/stack_cube_maniskill.yaml

# 批量 StackCube 验证
conda run -n carm python -m script_runtime.runners.maniskill_validate \
  --config script_runtime/configs/tasks/stack_cube_maniskill.yaml

# 相关测试
conda run -n carm python -m pytest -q \
  script_runtime/tests/test_pick_place_task.py \
  script_runtime/tests/test_session.py \
  script_runtime/tests/test_maniskill_validation.py \
  script_runtime/tests/test_maniskill_oracle_integration.py
```

## 6. 关键资产

### 6.1 VLAW

| 资产 | 路径 |
|------|------|
| IL policy | `checkpoints/il/best_eval_success_once.pt` |
| 当前 WM | `checkpoints/vlaw/world_model/iter1_v5/` |
| WM pretrained | `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt` |
| VLM LoRA best | `checkpoints/vlaw/reward_model/ablation_v3/` |
| ACP v3_so | `checkpoints/vlaw/acp/v3_so/best.safetensors` |
| ACP Pipeline 文档 | `docs/vlaw/acp_pipeline.md` |

### 6.2 CARM

| 资产 | 路径 |
|------|------|
| teleop 数据目录 | `recorded_data/` |
| inference 日志目录 | `inference_logs/` |
| CARM 系统架构文档 | `docs/carm_real_robot_system.md` |
| timeline 分析报告 | `docs/carm_timeline_analysis_report.md` |
| HITL 计划与进度 | `docs/hitl_inference_live_*.md` |

### 6.3 script policy / runtime

| 资产 | 路径 |
|------|------|
| script policy 方案文档 | `.codex/SCRIPT_POLICY_PLATFORM_PLAN.md` |
| runtime README | `script_runtime/README.md` |
| PickCube 汇总 | `script_runtime/artifacts/maniskill_validation_summary.json` |
| PickCube 报告 | `script_runtime/artifacts/maniskill_report/REPORT.md` |
| StackCube 汇总 | `script_runtime/artifacts/stack_cube_validation_summary.json` |
| StackCube 报告 | `script_runtime/artifacts/stack_cube_report/REPORT.md` |
| 单次 rollout GIF | `script_runtime/artifacts/single_run_rollout.gif` |
| 单次 grounding | `script_runtime/artifacts/single_run_grounding.json` |

## 7. 关键门控

### 7.1 VLAW

| 指标 | 最低门槛 | 目标值 |
|------|---------|-------|
| WM 预测 PSNR | > 18 | > 20 |
| VLM FP rate | < 20% | < 10% |
| `D_syn+` yield | > 5% | - |
| Policy success_rate 提升 | > 10% abs | > 20% abs |
| ACP value MAE | < 0.1 | < 0.05 |

### 7.2 CARM second-stage

当前工程门槛优先级：
- 数据契约完整且可审计
- admission policy 明确且写入 metadata
- 训练 run 产物可追溯具体 episode、bucket、policy version
- 训练后可导出 deploy-ready checkpoint 并支持 redeploy smoke

## 8. 关键 ADR / 约束速记

| 决策 | 含义 |
|------|------|
| ADR-002 | 双相机竖拼 `(384, 192)`，对应 latent `(4, 48, 24)` |
| ADR-009 | Policy 使用视觉 obs，`global_cond_dim=626` |
| ADR-019 | VLM 必须使用 video 模式 |
| ADR-035 | ACP 使用 Pistar06 value model |
| ADR-037 | WM action 使用绝对 EE 位姿，VAE 编码按相机独立再拼接 |
| ADR-043 | BUG-D 中 action tiling 是 imagination 退化唯一显著根因 |
| ADR-045 | `pd_ee_pose` 迁移方案失败 |
| ADR-047 | ACP v5：Q-clip + potential reward + reward clip 成为稳定组合 |
| ADR-048 | ACP v6：长训练显著提升 DSRL SAE 到 14% |

## 9. 编码与协作规范

- Python 3.10+，函数签名带 type hints
- 配置优先使用 `tyro` dataclass，不新增 `argparse` 风格入口
- 日志优先 `wandb`
- 权重用 `safetensors`，轨迹/episode 用 HDF5
- 路径使用 `pathlib.Path`
- import 顺序：stdlib -> third-party -> local
- `ctrl_world/` 是外部代码，必须最小修改
- CARM 真机链路中，schema 稳定性和审计可追溯性优先于“一次性跑通”

## 10. 状态与知识位置

当前仓库中值得优先阅读的文档：
- `README.md`
- `.codex/SCRIPT_POLICY_PLATFORM_PLAN.md`
- `script_runtime/README.md`
- `docs/carm_real_robot_system.md`
- `docs/carm_timeline_analysis_report.md`
- `docs/hitl_inference_live_owner_source_plan_2026-04-12.md`
- `docs/hitl_inference_live_progress_2026-04-12.md`
- `docs/human_chunk_online_rollout_plan_2026-04-12.md`
- `docs/vlaw/acp_pipeline.md`
- `docs/vlaw/baselines_and_evaluation.md`
- `docs/vlaw/wm_implementation_gap_analysis.md`

经过对 `docs/` 的二次整理后，以下 `.codex` 文档已经承接了最值得长期保留的内容：
- [CARM_PIPELINE.md](/home/amax/rl-vla/.codex/CARM_PIPELINE.md)
- [CARM_ACTION_SEMANTICS_AND_HITL.md](/home/amax/rl-vla/.codex/CARM_ACTION_SEMANTICS_AND_HITL.md)
- [CARM_INFERENCE_ADMISSION_POLICY.md](/home/amax/rl-vla/.codex/CARM_INFERENCE_ADMISSION_POLICY.md)
- [CTRL_WORLD.md](/home/amax/rl-vla/.codex/CTRL_WORLD.md)
- [VLAW_RESEARCH_NOTES.md](/home/amax/rl-vla/.codex/VLAW_RESEARCH_NOTES.md)
- [SCRIPT_POLICY_PLATFORM_PLAN.md](/home/amax/rl-vla/.codex/SCRIPT_POLICY_PLATFORM_PLAN.md)

历史 `.claude` 里提到但仓库当前不一定都存在的 `.github/` 状态文件，后续若继续采用，建议统一迁入 `.codex` 或 `docs/` 体系，避免多套状态源并存。

## 11. 本地自动化约定

原 `.claude/settings.json` 中有两条对开发体验很重要的约定，这里明确迁移为 Codex 工作习惯：

1. 会话结束提醒
   - 若本轮有重要进展，应同步更新项目记忆与状态文档
   - 在当前 `.codex` 体系下，优先更新：
     - `.codex/WORKSPACE.md`
     - `.codex/CARM_PIPELINE.md`
     - `.codex/CARM_INFERENCE_ADMISSION_POLICY.md`
     - 或仓库中实际维护的状态文件

2. Python 文件编辑后自动格式化的习惯
   - 若修改单个 Python 文件，优先运行：

```bash
conda run -n rlft_ms3 python -m black <file.py> --quiet
```

   - 若目标文件不属于 `rlft_ms3` 环境，改用对应环境执行 `black`
   - 这不是强制绑定到某个钩子系统，而是迁移为仓库级协作约定
