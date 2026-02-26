"""rlft.vlaw.policy — 策略更新相关模块

包含:
    policy_updater  ← policy_updater.py  (Weighted Filtered BC P5.1)
    state_predictor ← state_predictor.py (状态递推 MLP P4.1，临时脚手架)

导入路径:
    from rlft.vlaw.policy import PolicyUpdater
"""

# 从子目录模块导入（新路径的权威来源）
from .policy_updater import PolicyUpdaterConfig, VLAWPolicyUpdater, VLAWSuccessDataset
from .state_predictor import StatePredictorConfig, StatePredictor, StatePredictorTrainer

__all__ = [
    # policy_updater
    "PolicyUpdaterConfig",
    "VLAWPolicyUpdater",
    "VLAWSuccessDataset",
    # state_predictor
    "StatePredictorConfig",
    "StatePredictor",
    "StatePredictorTrainer",
]
