"""
rlft.vlaw — VLAW 复现模块

目录结构:
    rlft/vlaw/data/        — 数据相关: collector, pipeline, demo_prep
    rlft/vlaw/world_model/ — 世界模型: ctrl_world_adapter, imagination_env
    rlft/vlaw/reward/      — 奖励模型: reward_model, train_reward_model
    rlft/vlaw/policy/      — 策略更新: policy_updater, state_predictor
    rlft/vlaw/utils/       — 工具脚本: validate_rgb_data, imagination

导入路径:
    from rlft.vlaw.data import CollectorConfig, VLAWDataCollector
    from rlft.vlaw.data import PipelineConfig, VLAWDataPipeline
    from rlft.vlaw.world_model import CtrlWorldAdapter
    from rlft.vlaw.world_model import ImaginationEnvConfig, ImaginationEnvEngine
    from rlft.vlaw.reward import VLAWRewardConfig, VLAWRewardModel
    from rlft.vlaw.policy import PolicyUpdaterConfig, VLAWPolicyUpdater
    from rlft.vlaw.policy import StatePredictorConfig, StatePredictor

公开接口:
    VLAWRewardModel       — VLM 二分类奖励模型 (P3.1)
    VLAWRewardConfig      — 奖励模型配置
    uniform_sample_frames — 轨迹帧均匀采样工具
    VLAWDataCollector     — ManiSkill Rollout 收集器 (P1.1)
    CollectorConfig       — 数据收集配置
    VLAWDataPipeline      — VAE 编码管线 (P1.2)
    PipelineConfig        — VAE 管线配置
    DemoConverter         — ManiSkill Demo 转 VLAW 格式 (P1.3)
    DemoPrepConfig        — Demo 准备配置
    PolicyUpdaterConfig   — 策略更新配置 (P5.1)
    VLAWPolicyUpdater     — Weighted Filtered BC 策略更新器 (P5.1)
    VLAWSuccessDataset    — HDF5 成功轨迹数据集 (P5.1)
    StatePredictorConfig  — State Predictor 配置 (P4.1)
    StatePredictor        — 状态递推 MLP (P4.1)
    StatePredictorTrainer — State Predictor 训练器 (P4.1)
    ImaginationConfig     — Imagination Engine 配置 (P4.2)
    ImaginationEngine     — Policy-in-the-Loop Imagination 引擎 (P4.2)
    SyntheticTrajectory   — 合成轨迹数据容器 (P4.2)
    ImaginationEnvConfig  — env.step() 版 Imagination 配置 (P4.3)
    ImaginationEnvEngine  — env.step() 版 Imagination 引擎 (P4.3)
"""

# ── 导入子包公开接口 ──
from .data.collector import CollectorConfig, VLAWDataCollector
from .data.pipeline import PipelineConfig, VLAWDataPipeline, concat_cameras
from .data.demo_prep import DemoPrepConfig, DemoConverter
from .reward.reward_model import VLAWRewardConfig, VLAWRewardModel, uniform_sample_frames
from .policy.state_predictor import StatePredictorConfig, StatePredictor, StatePredictorTrainer
from .policy.policy_updater import PolicyUpdaterConfig, VLAWPolicyUpdater, VLAWSuccessDataset
# world_model 子包依赖 einops 等重型依赖，在部分环境（如 vlaw_reward）中不可用
# 使用 try/except 实现惰性加载，避免导入失败
try:
    from .world_model.ctrl_world_adapter import CtrlWorldAdapter
    from .world_model.imagination_env import ImaginationEnvConfig, ImaginationEnvEngine
except (ImportError, ModuleNotFoundError) as _world_model_err:
    import warnings as _warnings
    _warnings.warn(
        f"[rlft.vlaw] world_model 子包加载失败（可能缺少 einops 等依赖），"
        f"CtrlWorldAdapter / ImaginationEnvEngine 不可用: {_world_model_err}",
        ImportWarning,
        stacklevel=2,
    )
    CtrlWorldAdapter = None  # type: ignore[assignment]
    ImaginationEnvConfig = None  # type: ignore[assignment]
    ImaginationEnvEngine = None  # type: ignore[assignment]
# NOTE: ImaginationEngine 依赖 Ctrl-World 重型依赖，不在此自动导入
# 需要时请显式: from rlft.vlaw.utils.imagination import ImaginationConfig, ImaginationEngine

__all__ = [
    # data
    "CollectorConfig",
    "VLAWDataCollector",
    "PipelineConfig",
    "VLAWDataPipeline",
    "concat_cameras",
    "DemoPrepConfig",
    "DemoConverter",
    # reward
    "VLAWRewardConfig",
    "VLAWRewardModel",
    "uniform_sample_frames",
    # policy
    "PolicyUpdaterConfig",
    "VLAWPolicyUpdater",
    "VLAWSuccessDataset",
    "StatePredictorConfig",
    "StatePredictor",
    "StatePredictorTrainer",
    # world_model
    "CtrlWorldAdapter",
    "ImaginationEnvConfig",
    "ImaginationEnvEngine",
]
