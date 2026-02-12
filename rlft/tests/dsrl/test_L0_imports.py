"""
L0: DSRL Pipeline 导入与冒烟测试

验证所有新增 DSRL 模块可以被正确导入，无 import 错误。
不需要 GPU 或 checkpoint 文件。

运行:
    conda activate carm
    cd /home/lizh/rl-vla
    python -m pytest rlft/tests/dsrl/test_L0_imports.py -v
"""

import pytest
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# =====================================================================
# L0-1: 基础导入
# =====================================================================

class TestImports:
    """验证所有 DSRL 相关模块可以正常导入。"""

    def test_import_dsrl_sac_agent(self):
        from rlft.algorithms.online_rl.dsrl_sac import DSRLSACAgent
        assert DSRLSACAgent is not None

    def test_import_dsrl_actor(self):
        from rlft.algorithms.online_rl.dsrl_sac import DSRLActor
        assert DSRLActor is not None

    def test_import_dsrl_critic(self):
        from rlft.algorithms.online_rl.dsrl_sac import DSRLCritic
        assert DSRLCritic is not None

    def test_import_scaled_squashed_normal(self):
        from rlft.algorithms.online_rl.dsrl_sac import ScaledSquashedNormal
        assert ScaledSquashedNormal is not None

    def test_import_dsrl_replay_buffer(self):
        from rlft.buffers.dsrl_buffer import DSRLReplayBuffer
        assert DSRLReplayBuffer is not None

    def test_import_flow_wrapper(self):
        from rlft.utils.flow_wrapper import ShortCutFlowWrapper
        assert ShortCutFlowWrapper is not None

    def test_import_load_shortcut_flow_policy(self):
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy
        assert load_shortcut_flow_policy is not None

    def test_import_maniskill_flow_env_wrapper(self):
        from rlft.envs.dsrl_env import ManiSkillFlowEnvWrapper
        assert ManiSkillFlowEnvWrapper is not None


# =====================================================================
# L0-2: __init__.py 导出检查
# =====================================================================

class TestInitExports:
    """确认各 __init__.py 正确导出新增符号。"""

    def test_online_rl_init(self):
        from rlft.algorithms.online_rl import DSRLSACAgent
        assert DSRLSACAgent is not None

    def test_buffers_init(self):
        from rlft.buffers import DSRLReplayBuffer
        assert DSRLReplayBuffer is not None

    def test_envs_init(self):
        from rlft.envs import ManiSkillFlowEnvWrapper
        assert ManiSkillFlowEnvWrapper is not None

    def test_utils_init(self):
        from rlft.utils import ShortCutFlowWrapper, load_shortcut_flow_policy
        assert ShortCutFlowWrapper is not None
        assert load_shortcut_flow_policy is not None


# =====================================================================
# L0-3: 依赖库检查
# =====================================================================

class TestDependencies:
    """检查运行 DSRL 所需的关键依赖库。"""

    def test_torch_available(self):
        import torch
        assert torch.__version__ is not None

    def test_numpy_available(self):
        import numpy as np
        assert np.__version__ is not None

    def test_gymnasium_available(self):
        import gymnasium
        assert gymnasium.__version__ is not None

    def test_maniskill_available(self):
        import mani_skill
        assert mani_skill is not None

    def test_tyro_available(self):
        import tyro
        assert tyro is not None

    def test_rlft_networks(self):
        from rlft.networks import PlainConv, ShortCutVelocityUNet1D
        assert PlainConv is not None
        assert ShortCutVelocityUNet1D is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
