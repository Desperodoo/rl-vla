"""
DSRL Official Implementation for ManiSkill3

基于官方 ajwagen/dsrl 实现，针对 ManiSkill3 和 ShortCut Flow 策略进行适配。

模块结构:
- wrappers/: 环境包装器 (DiffusionPolicyEnvWrapper 等)
- adapters/: ShortCut Flow 适配层
- algorithms/: DSRL-SAC 和 DSRL-NA 算法
- utils/: 工具函数和回调
- configs/: 配置文件
- tests/: 验证测试

使用方法:
    # DSRL-SAC (环境包装器模式)
    python train_dsrl.py --algorithm dsrl_sac --env_id LiftPegUpright-v1
    
    # DSRL-NA (策略内部采样模式)
    python train_dsrl.py --algorithm dsrl_na --env_id LiftPegUpright-v1
"""

__version__ = "0.1.0"
