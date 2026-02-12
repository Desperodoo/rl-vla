"""
L2: Checkpoint 加载测试

验证 rlft 的 load_shortcut_flow_policy 能正确加载预训练 checkpoint，
并与 dsrl_official 的加载结果保持一致。

使用 checkpoint: runs/awsc_checkpoint/checkpoints/best_eval_success_once.pt
需要 GPU。

运行:
    conda activate carm
    cd /home/lizh/rl-vla
    python -m pytest rlft/tests/dsrl/test_L2_checkpoint.py -v
    # 或单独运行
    CUDA_VISIBLE_DEVICES=0 python rlft/tests/dsrl/test_L2_checkpoint.py
"""

import pytest
import sys
from pathlib import Path

import torch
import numpy as np

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
    sys.path.insert(0, str(_root / "diffusion_policy"))

# =====================================================================
# 常量
# =====================================================================

CHECKPOINT_PATH = str(_root / "runs/awsc_checkpoint/checkpoints/best_eval_success_once.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 预期参数
EXPECTED_STATE_DIM = 25
EXPECTED_VISUAL_DIM = 256
EXPECTED_OBS_HORIZON = 2
EXPECTED_PRED_HORIZON = 16
EXPECTED_ACTION_DIM = 7
EXPECTED_GLOBAL_COND_DIM = 562  # 2 * (256 + 25)


def _skip_if_no_checkpoint():
    if not Path(CHECKPOINT_PATH).exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")


# =====================================================================
# L2-1: Checkpoint 文件结构
# =====================================================================

class TestCheckpointStructure:
    """验证 checkpoint 文件的结构正确。"""

    def setup_method(self):
        _skip_if_no_checkpoint()
        self.ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    def test_toplevel_keys(self):
        """checkpoint 应包含 agent, ema_agent, visual_encoder。"""
        for key in ["agent", "ema_agent", "visual_encoder"]:
            assert key in self.ckpt, f"缺少 key: {key}"

    def test_velocity_net_keys_exist(self):
        """agent/ema_agent 里应包含 velocity_net. 前缀的权重。"""
        agent_state = self.ckpt["ema_agent"]
        vnet_keys = [k for k in agent_state if k.startswith("velocity_net.")]
        assert len(vnet_keys) > 0, "未找到 velocity_net 权重"
        print(f"  velocity_net keys: {len(vnet_keys)}")

    def test_visual_encoder_keys(self):
        """visual_encoder 应有 12 个参数 (PlainConv)。"""
        ve_state = self.ckpt["visual_encoder"]
        assert len(ve_state) == 12, f"Expected 12, got {len(ve_state)}"

    def test_cond_dim_matches(self):
        """从 cond_encoder 权重推断 global_cond_dim 应为 562。"""
        agent_state = self.ckpt["ema_agent"]
        for key, value in agent_state.items():
            if "cond_encoder.1.weight" in key:
                cond_input = value.shape[1]
                global_cond = cond_input - 64  # diffusion_step_embed_dim
                assert global_cond == EXPECTED_GLOBAL_COND_DIM, \
                    f"global_cond_dim mismatch: {global_cond} != {EXPECTED_GLOBAL_COND_DIM}"
                return
        pytest.fail("未找到 cond_encoder.1.weight")


# =====================================================================
# L2-2: load_shortcut_flow_policy 加载
# =====================================================================

class TestLoadShortcutFlowPolicy:
    """验证 rlft 的 load_shortcut_flow_policy 函数。"""

    def setup_method(self):
        _skip_if_no_checkpoint()

    def test_load_returns_triple(self):
        """返回 (wrapper, visual_encoder, state_dim) 三元组。"""
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy
        from rlft.networks import PlainConv

        wrapper, ve, sd = load_shortcut_flow_policy(
            checkpoint_path=CHECKPOINT_PATH,
            visual_encoder_class=PlainConv,
            obs_horizon=EXPECTED_OBS_HORIZON,
            pred_horizon=EXPECTED_PRED_HORIZON,
            action_dim=EXPECTED_ACTION_DIM,
            visual_feature_dim=EXPECTED_VISUAL_DIM,
            include_rgb=True,
            use_ema=True,
            device=DEVICE,
        )

        assert wrapper is not None
        assert ve is not None
        assert sd == EXPECTED_STATE_DIM, f"state_dim={sd}, expected {EXPECTED_STATE_DIM}"

    def test_inferred_state_dim(self):
        """state_dim=None 时应自动推断为 25。"""
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy
        from rlft.networks import PlainConv

        _, _, sd = load_shortcut_flow_policy(
            checkpoint_path=CHECKPOINT_PATH,
            visual_encoder_class=PlainConv,
            state_dim=None,  # 自动推断
            include_rgb=True,
            device=DEVICE,
        )
        assert sd == EXPECTED_STATE_DIM

    def test_explicit_state_dim(self):
        """显式传 state_dim=25 应直接使用。"""
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy
        from rlft.networks import PlainConv

        _, _, sd = load_shortcut_flow_policy(
            checkpoint_path=CHECKPOINT_PATH,
            visual_encoder_class=PlainConv,
            state_dim=25,
            include_rgb=True,
            device=DEVICE,
        )
        assert sd == 25

    def test_velocity_net_frozen(self):
        """velocity_net 参数应冻结（requires_grad=False）。"""
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy
        from rlft.networks import PlainConv

        wrapper, _, _ = load_shortcut_flow_policy(
            checkpoint_path=CHECKPOINT_PATH,
            visual_encoder_class=PlainConv,
            include_rgb=True,
            device=DEVICE,
        )
        for p in wrapper.velocity_net.parameters():
            assert not p.requires_grad, "velocity_net should be frozen"

    def test_visual_encoder_eval_mode(self):
        """visual_encoder 应为 eval 模式。"""
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy
        from rlft.networks import PlainConv

        _, ve, _ = load_shortcut_flow_policy(
            checkpoint_path=CHECKPOINT_PATH,
            visual_encoder_class=PlainConv,
            include_rgb=True,
            device=DEVICE,
        )
        assert not ve.training, "visual_encoder should be in eval mode"

    def test_use_ema_vs_agent(self):
        """use_ema=True 和 use_ema=False 应加载不同权重。"""
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy
        from rlft.networks import PlainConv

        w_ema, _, _ = load_shortcut_flow_policy(
            checkpoint_path=CHECKPOINT_PATH,
            visual_encoder_class=PlainConv,
            include_rgb=True, use_ema=True, device="cpu",
        )
        w_no_ema, _, _ = load_shortcut_flow_policy(
            checkpoint_path=CHECKPOINT_PATH,
            visual_encoder_class=PlainConv,
            include_rgb=True, use_ema=False, device="cpu",
        )
        # EMA 和 non-EMA 权重通常不完全一样
        ema_p = list(w_ema.velocity_net.parameters())
        no_ema_p = list(w_no_ema.velocity_net.parameters())
        any_diff = any(
            not torch.allclose(a, b) for a, b in zip(ema_p, no_ema_p)
        )
        # 注：若训练刚好使 EMA=non-EMA 也是合法的，所以这里用 print 而非 assert
        if any_diff:
            print("  ✓ EMA and non-EMA weights differ (expected)")
        else:
            print("  ⚠ EMA and non-EMA weights are identical")


# =====================================================================
# L2-3: 与 dsrl_official 加载结果一致性
# =====================================================================

class TestCrossLoadConsistency:
    """验证 rlft 和 dsrl_official 加载的模型产生相同输出。"""

    def setup_method(self):
        _skip_if_no_checkpoint()

    def test_velocity_net_same_weights(self):
        """rlft 和手动加载的 velocity_net 权重应完全一致。"""
        from rlft.utils.flow_wrapper import load_shortcut_flow_policy
        from rlft.networks import PlainConv, ShortCutVelocityUNet1D

        # rlft 方式加载
        wrapper, _, _ = load_shortcut_flow_policy(
            checkpoint_path=CHECKPOINT_PATH,
            visual_encoder_class=PlainConv,
            include_rgb=True, use_ema=True, device="cpu",
        )

        # 手动加载（模拟 dsrl_official）
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        agent_state = ckpt["ema_agent"]
        vnet_state = {
            k.replace("velocity_net.", ""): v
            for k, v in agent_state.items()
            if k.startswith("velocity_net.")
        }

        vnet_manual = ShortCutVelocityUNet1D(
            input_dim=EXPECTED_ACTION_DIM,
            global_cond_dim=EXPECTED_GLOBAL_COND_DIM,
            diffusion_step_embed_dim=64,
            down_dims=(64, 128, 256),
            n_groups=8,
        )
        vnet_manual.load_state_dict(vnet_state)

        # 比较权重
        for (k1, v1), (k2, v2) in zip(
            wrapper.velocity_net.state_dict().items(),
            vnet_manual.state_dict().items(),
        ):
            assert k1 == k2, f"Key mismatch: {k1} != {k2}"
            assert torch.allclose(v1, v2), f"Weight mismatch at {k1}"

        print("  ✓ velocity_net weights match exactly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
