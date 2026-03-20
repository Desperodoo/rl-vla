# rlft/vlaw — VLAW 核心模块上下文

此目录包含 VLAW 论文复现的所有新增代码（Phase P0-P7）。Claude Code 进入此目录时自动加载本文件。

## 模块相位图

| 文件 | 阶段 | 状态 |
|------|------|------|
| `data_collector.py` | P1.1 — ManiSkill rollout 采集 | ✅ |
| `data_pipeline.py` | P1.2 — VAE 编码 / concat_cameras | ✅ |
| `ctrl_world_adapter.py` | P2.1 — Ctrl-World 适配层 | ✅ |
| `train_world_model.py` | P2.2 — WM 训练入口 | ✅（运行中）|
| `reward_model.py` | P3.1 — VLM 奖励模型封装 | ✅ |
| `train_reward_model.py` | P3.2 — LoRA 微调入口 | ✅ |
| `state_predictor.py` | P4.1 — State Predictor MLP | ✅ |
| `imagination.py` | P4.2 — Policy-in-Loop 引擎 | ✅（待 WM 审查）|
| `policy_updater.py` | P5.1 — Weighted FM 策略更新 | 🔜 阻塞 |
| `acp/` | P6 — ACP 稠密 advantage（Pistar06 value model，已迁移到 `rlft/acp`；旧兼容层已移除） | ✅ 代码+GPU验证完成 |

## 子包结构（必须维护）

```
rlft/vlaw/
├── __init__.py          ← 导出所有公共接口
├── data/                ← 数据管线子包
├── world_model/         ← WM 子包
├── reward/              ← 奖励模型子包
├── policy/              ← 策略更新子包
├── acp/                 ← 已移除，ACP 主实现位于 `rlft/acp/`
├── utils/               ← 工具函数
└── scripts/             ← 执行脚本（ACP 入口已迁移到 `rlft/acp/`）
```

根目录 `.py` 文件应 ≤20 行；超过则迁移。

## 编码规范（此目录专属）

- 每个文件顶部必须有 module-level docstring，包含 `Phase: P<N>` 标签
- 所有 public 函数和类必须有 type hints + docstring
- 配置类使用 `@dataclass` + `tyro`，禁止 `argparse`
- 日志：`wandb` 记录指标，`print(f"[VLAW] ...")` 记录进度
- **路径约定**：
  - checkpoint → `checkpoints/vlaw/{module}/`
  - 数据 → `data/vlaw/{type}/`
  - 日志 → `logs/vlaw/`
- GPU：通过 `CUDA_VISIBLE_DEVICES` 控制；不在代码中硬编码 device
- 内存：大规模推理前 `torch.cuda.empty_cache()`
- 数据类型：VAE latent 用 `float16`；RGB 用 `uint8`

## 数据格式规范（HDF5 Schema）

每条轨迹包含：
```
rgb_base      (T, 128, 128, 3)  uint8   ← 底部相机（实测 128x128）
rgb_render    (T, 128, 128, 3)  uint8   ← 渲染相机（实测 128x128）
state         (T, 25)           float32
obs_agent     (T, 25)           float32
actions       (T, 7)            float32  ← delta pose xyz+euler+gripper
env_success   (T,)              bool
latent_concat (T, 4, 48, 24)    float16  ← 双相机竖拼 latent（ADR-002）
task_instruction  str
vlm_reward    (T,)              float32  ← VLM 二值奖励
vlm_prob      (T,)              float32  ← P('yes')
source        str               ← "demo" | "rollout" | "synthetic"
```

Group attrs（VLM 标注写入）：
```
vlm_success   int    ← VLM trajectory-level 二值 label（0/1，scalar attr）
vlm_reward    float  ← VLM reward 值
vlm_yes_prob  float  ← P('yes') 概率
success       bool   ← env_success.any() 的 trajectory-level 摘要
```

### ACP 扩展字段（P6 阶段写入）

ACP 标注由 `ACPAnnotator` 写回到每个 `traj_XXXX/` 组：

```
acp_value_target  (T,)   float32  ← GT value target（env_success 计算）
acp_value_pred    (T,)   float32  ← Pistar06 预测值
acp_advantage     (T,)   float32  ← N-step advantage（连续）
acp_indicator     (T,)   int32    ← 二值 indicator（1=positive, 0=negative）
acp_weight        (T,)   float32  ← 归一化权重，直接供 compute_weighted_loss
```

Group attrs 新增：`acp_positive_ratio`, `acp_advantage_mean`, `acp_threshold`

所有 ACP 字段为可选扩展，不影响已有管线（`policy_updater.py` 通过 `use_acp_weights` flag 决定是否读取）。

## 测试规范

```bash
# 运行全部单元/集成测试（无真实 GPU 要求）
conda run -n rlft_ms3 python -m pytest rlft/tests/vlaw/ -v --tb=short -q
```

- 测试文件命名：`test_{module}.py`
- 函数命名：`test_{function}_{scenario}`
- **禁止**：真实 ManiSkill env、下载模型权重、训练循环、硬编码路径
- 共享 fixtures 放 `conftest.py`，使用 `torch.randn` / `np.random` mock 数据
- shape 测试需同时检查 dtype
- GPU 测试加 `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`

## 重要关联文件

- `rlft/algorithms/il/shortcut_flow.py` — Policy 主类（`compute_weighted_loss` 在此修改）
- `rlft/envs/evaluate.py` — 评估入口
- `rlft/datasets/` — `OfflineRLDataset`（必须用，非直接 HDF5 读取）
- `ctrl_world/` — WM 代码（见 `ctrl_world/CLAUDE.md`，最小修改原则）
- `Evo-RL/` — Evo-RL 项目源码（ACP 参考实现，已去 .git）
  - `src/lerobot/values/pistar06/` — Pistar06 原始实现（移植参考）
  - `src/lerobot/scripts/lerobot_value_infer.py` — N-step advantage 算法参考
  - `Evo-RL/ANALYSIS_REPORT.md` — 项目分析报告

## ACP 集成说明（P6）

**来源**：Evo-RL 项目 (`Evo-RL/src/lerobot/values/pistar06/` + `lerobot_value_infer.py`)
**目的**：替换 VLM 的 per-trajectory 稀疏 binary filtering，提供 per-frame 稠密 advantage weights
**状态**：✅ 代码实现完成 + GPU 端到端验证通过

**验证结果（2026-03-07）**：
- 模型：697M total params, 1.55M trainable (0.2%), SigLIP vision-only 428M + Gemma 268M
- GPU: ~3GB VRAM per batch (single 4090)
- Training dry-run: 20步, 1200 traj (41194 frames), loss 5.31→5.24, MAE=0.271
- Inference dry-run: positive_ratio=0.300（精确命中 30% 目标），weights ∈ [0, 1]
- HDF5 写回验证：5 个 per-frame 字段 + 3 个 group attrs 完整写入
- 测试：28/28 通过（含 4 个 VLM label 模式测试）

**核心流程**：
1. 从 `env_success` GT（或 `vlm_success` 标注）计算 per-frame value target：`target = clip((-remaining_steps - c_fail*(1-success)) / (max_len+c_fail), -1, 0)`
2. 训练 Pistar06 value model（SigLIP+Gemma+201-bin distributional value head）预测每帧 value
3. 用 predicted values 计算 N-step advantage：`A(t) = Σr[t:t+n] + V(t+n) - V(t)`，其中 `r[t] = target[t] - target[t+1]`
4. Per-task quantile 二值化（positive_ratio=0.3）或直接用连续 advantage 归一化为 weights
5. `policy_updater.py` 读取 `acp_weight` 替换均匀 weight=1.0，传入 `compute_weighted_loss`

**关键适配**：
- 双相机分别输入 SigLIP vision-only encoder（128x128 → resize 384x384），**不用竖拼**
- 冻结 SigLIP+Gemma backbone，只训练 projector+value head（~1.55M 参数, 0.2%）
- `success_key` 配置化：`env_success`（仿真 GT per-frame）或 `vlm_success`（VLM scalar attr，自动展开为 per-frame）
- Conda env 复用 `vlaw_reward`

**在 VLAW 迭代循环中的位置**：
```
数据采集 → VLM 标注 → [WM 训练 ‖ ACP value 训练] → [Imagination ‖ ACP 推理标注] → 策略更新(use_acp_weights=True)
```

**待完成**：
- 正式训练 8000 步（目标 MAE < 0.05）
- Phase D: Policy Updater 集成（需 reward/value/advantage 层层验证通过）

**详细计划文件**：`.claude/plans/wiggly-honking-ritchie.md`
