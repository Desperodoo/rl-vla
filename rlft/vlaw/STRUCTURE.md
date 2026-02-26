# rlft/vlaw/ 模块结构

## 目录说明

| 目录 | 文件 | 功能 |
|------|------|------|
| `data/` | `collector.py`, `pipeline.py`, `demo_prep.py` | 数据收集（ManiSkill Rollout）与 VAE 编码管线 |
| `world_model/` | `ctrl_world_adapter.py`, `imagination_env.py` | Ctrl-World 推理封装与 env.step() Imagination 引擎 |
| `reward/` | `reward_model.py`, `train_reward_model.py` | VLM 二分类奖励模型（Qwen-VL）& LoRA 微调训练 |
| `policy/` | `policy_updater.py`, `state_predictor.py` | Weighted Filtered BC 策略更新 & 状态递推 MLP |
| `utils/` | `validate_rgb_data.py`, `imagination.py` | 数据验证工具；旧版 Imagination 引擎（仅供参考） |
| `scripts/` | `label_*.py`, `collect_*.py`, ... | VLAW 训练流程入口脚本 |

## 导入路径

### 推荐（新路径，从子目录直接导入）

```python
from rlft.vlaw.reward.reward_model import VLAWRewardModel, VLAWRewardConfig
from rlft.vlaw.data.collector import CollectorConfig, VLAWDataCollector
from rlft.vlaw.data.pipeline import PipelineConfig, VLAWDataPipeline
from rlft.vlaw.policy.policy_updater import PolicyUpdaterConfig, VLAWPolicyUpdater
from rlft.vlaw.policy.state_predictor import StatePredictorConfig, StatePredictor
from rlft.vlaw.world_model.ctrl_world_adapter import CtrlWorldAdapter
from rlft.vlaw.world_model.imagination_env import ImaginationEnvConfig, ImaginationEnvEngine

# 或通过子包 __init__ 导入
from rlft.vlaw.reward import VLAWRewardModel
from rlft.vlaw.data import CollectorConfig, PipelineConfig

# 或通过顶层 __init__ 直接导入
from rlft.vlaw import VLAWRewardModel, CollectorConfig
```

### 向后兼容（旧路径，仍可用但不推荐）

```python
from rlft.vlaw.reward_model import VLAWRewardConfig, VLAWRewardModel   # 旧扁平路径
from rlft.vlaw.data_collector import CollectorConfig                     # 旧扁平路径
from rlft.vlaw.data_pipeline import PipelineConfig                      # 旧扁平路径
from rlft.vlaw.policy_updater import PolicyUpdaterConfig                 # 旧扁平路径
from rlft.vlaw.state_predictor import StatePredictorConfig               # 旧扁平路径
```

### 重型依赖（按需显式导入，不自动加载）

```python
# ImaginationEngine 依赖 Ctrl-World SVD/VAE，不在顶层 __init__ 自动导入
from rlft.vlaw.imagination import ImaginationConfig, ImaginationEngine, SyntheticTrajectory
```

## 模块对应关系（新旧对照）

| 旧路径（扁平文件） | 新路径（子目录）| 说明 |
|---|---|---|
| `rlft/vlaw/data_collector.py` | `rlft/vlaw/data/collector.py` | ManiSkill Rollout 收集器 (P1.1) |
| `rlft/vlaw/data_pipeline.py` | `rlft/vlaw/data/pipeline.py` | VAE 编码管线 (P1.2) |
| `rlft/vlaw/demo_prep.py` | `rlft/vlaw/data/demo_prep.py` | Demo 格式转换 (P1.3) |
| `rlft/vlaw/ctrl_world_adapter.py` | `rlft/vlaw/world_model/ctrl_world_adapter.py` | Ctrl-World 推理封装 (P2.1) |
| `rlft/vlaw/imagination_env.py` | `rlft/vlaw/world_model/imagination_env.py` | env.step() Imagination (P4.3) |
| `rlft/vlaw/reward_model.py` | `rlft/vlaw/reward/reward_model.py` | VLM 奖励模型 (P3.1) |
| `rlft/vlaw/train_reward_model.py` | `rlft/vlaw/reward/train_reward_model.py` | VLM 微调训练 (P3.2) |
| `rlft/vlaw/policy_updater.py` | `rlft/vlaw/policy/policy_updater.py` | Weighted Filtered BC (P5.1) |
| `rlft/vlaw/state_predictor.py` | `rlft/vlaw/policy/state_predictor.py` | 状态递推 MLP (P4.1) |

## 注意事项

1. **单一来源原则**：新子目录中的文件为权威来源；旧扁平文件保留以确保向后兼容
2. **修改功能代码**时优先修改子目录文件，同步更改旧扁平文件（或考虑将旧文件改为 re-export）
3. **重型依赖隔离**：`ImaginationEngine`（依赖 Ctrl-World SVD/VAE）和 `train_reward_model`（依赖 transformers LoRA）不在任何 `__init__` 中自动导入
4. **测试文件**：`rlft/vlaw/test_reward_model.py` 已迁移至 `rlft/tests/vlaw/test_reward_model_legacy.py`
