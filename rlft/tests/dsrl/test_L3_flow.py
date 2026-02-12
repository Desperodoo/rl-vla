"""
L3: Flow Policy 集成测试

验证 ShortCutFlowWrapper 的 ODE 积分在加载真实模型后行为正确：
  - 零噪声输入产生合理动作
  - 输出形状、范围正确
  - ODE 积分一致性（多次调用结果相同）
  - 与 dsrl_official 的 FlowWrapper 输出对齐

需要: checkpoint + GPU

运行:
    conda activate carm
    cd /home/lizh/rl-vla
    CUDA_VISIBLE_DEVICES=0 python -m pytest rlft/tests/dsrl/test_L3_flow.py -v -s
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

CHECKPOINT_PATH = str(_root / "runs/awsc_checkpoint/checkpoints/best_eval_success_once.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_STEPS = 8
ACTION_DIM = 7
VISUAL_DIM = 256
STATE_DIM = 25
OBS_DIM = OBS_HORIZON * (VISUAL_DIM + STATE_DIM)  # 562


def _skip_if_no_checkpoint():
    if not Path(CHECKPOINT_PATH).exists():
        pytest.skip("Checkpoint not found")


@pytest.fixture(scope="module")
def flow_policy():
    """模块级 fixture: 只加载一次 flow policy。"""
    _skip_if_no_checkpoint()
    from rlft.utils.flow_wrapper import load_shortcut_flow_policy
    from rlft.networks import PlainConv

    wrapper, ve, sd = load_shortcut_flow_policy(
        checkpoint_path=CHECKPOINT_PATH,
        visual_encoder_class=PlainConv,
        obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON,
        action_dim=ACTION_DIM,
        visual_feature_dim=VISUAL_DIM,
        include_rgb=True,
        use_ema=True,
        device=DEVICE,
    )
    return wrapper, ve, sd


# =====================================================================
# L3-1: 基础 ODE 积分
# =====================================================================

class TestFlowODE:
    """测试 ShortCutFlowWrapper 的 Euler ODE 积分。"""

    def test_zero_noise_output_shape(self, flow_policy):
        """零噪声输入应输出正确形状的动作。"""
        wrapper, _, _ = flow_policy
        B = 4
        obs = torch.randn(B, OBS_DIM, device=DEVICE)
        noise = torch.zeros(B, ACT_STEPS, ACTION_DIM, device=DEVICE)

        actions = wrapper(obs, noise, return_numpy=False, act_steps=ACT_STEPS)
        assert actions.shape == (B, ACT_STEPS, ACTION_DIM), \
            f"Shape mismatch: {actions.shape}"

    def test_output_in_range(self, flow_policy):
        """ODE 输出应被 clamp 到 [-1, 1]。"""
        wrapper, _, _ = flow_policy
        B = 8
        obs = torch.randn(B, OBS_DIM, device=DEVICE)
        noise = torch.randn(B, ACT_STEPS, ACTION_DIM, device=DEVICE) * 0.5

        actions = wrapper(obs, noise, return_numpy=False, act_steps=ACT_STEPS)
        assert actions.abs().max() <= 1.0 + 1e-5, \
            f"Actions out of [-1,1] range: max={actions.abs().max():.4f}"

    def test_consistency(self, flow_policy):
        """相同输入多次调用输出应完全一致。"""
        wrapper, _, _ = flow_policy
        B = 4
        obs = torch.randn(B, OBS_DIM, device=DEVICE)
        noise = torch.randn(B, ACT_STEPS, ACTION_DIM, device=DEVICE)

        a1 = wrapper(obs, noise, return_numpy=False, act_steps=ACT_STEPS)
        a2 = wrapper(obs, noise, return_numpy=False, act_steps=ACT_STEPS)

        diff = (a1 - a2).abs().max().item()
        assert diff < 1e-5, f"Inconsistent outputs: max_diff={diff}"

    def test_different_noise_different_output(self, flow_policy):
        """不同噪声输入应产生不同动作。"""
        wrapper, _, _ = flow_policy
        B = 4
        obs = torch.randn(B, OBS_DIM, device=DEVICE)
        noise1 = torch.zeros(B, ACT_STEPS, ACTION_DIM, device=DEVICE)
        noise2 = torch.randn(B, ACT_STEPS, ACTION_DIM, device=DEVICE)

        a1 = wrapper(obs, noise1, return_numpy=False, act_steps=ACT_STEPS)
        a2 = wrapper(obs, noise2, return_numpy=False, act_steps=ACT_STEPS)

        diff = (a1 - a2).abs().mean().item()
        assert diff > 1e-3, f"Different noise should give different actions, diff={diff}"

    def test_flat_noise_input(self, flow_policy):
        """支持 flat noise (B, T*D) 输入。"""
        wrapper, _, _ = flow_policy
        B = 4
        obs = torch.randn(B, OBS_DIM, device=DEVICE)
        noise_flat = torch.zeros(B, ACT_STEPS * ACTION_DIM, device=DEVICE)

        actions = wrapper(obs, noise_flat, return_numpy=False, act_steps=ACT_STEPS)
        assert actions.shape == (B, ACT_STEPS, ACTION_DIM)

    def test_numpy_output(self, flow_policy):
        """return_numpy=True 应返回 numpy 数组。"""
        wrapper, _, _ = flow_policy
        B = 4
        obs = torch.randn(B, OBS_DIM, device=DEVICE)
        noise = torch.zeros(B, ACT_STEPS, ACTION_DIM, device=DEVICE)

        actions = wrapper(obs, noise, return_numpy=True, act_steps=ACT_STEPS)
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (B, ACT_STEPS, ACTION_DIM)


# =====================================================================
# L3-2: Visual Encoder 验证
# =====================================================================

class TestVisualEncoder:
    """验证加载的 visual encoder 工作正常。"""

    def test_forward_shape(self, flow_policy):
        """PlainConv 前向传播输出正确维度。"""
        _, ve, _ = flow_policy
        B = 4
        rgb = torch.randn(B, 3, 128, 128, device=DEVICE)  # 标准 ManiSkill RGB
        with torch.no_grad():
            feat = ve(rgb)
        assert feat.shape == (B, VISUAL_DIM), f"Shape {feat.shape} != (B, {VISUAL_DIM})"

    def test_encoder_deterministic(self, flow_policy):
        """eval 模式下输出应确定性。"""
        _, ve, _ = flow_policy
        B = 4
        rgb = torch.randn(B, 3, 128, 128, device=DEVICE)
        with torch.no_grad():
            f1 = ve(rgb)
            f2 = ve(rgb)
        assert torch.allclose(f1, f2), "Visual encoder should be deterministic in eval mode"


# =====================================================================
# L3-3: 与 dsrl_official FlowWrapper 输出对齐
# =====================================================================

class TestCrossFlowConsistency:
    """验证 rlft 和 dsrl_official 的 FlowWrapper 在相同输入下输出一致。"""

    def test_same_output(self, flow_policy):
        """两种实现加载同一 checkpoint 后，相同输入应产生相同输出。"""
        rlft_wrapper, _, _ = flow_policy

        try:
            from dsrl_official.utils import ShortCutFlowWrapper as OfficialWrapper
            from rlft.networks import ShortCutVelocityUNet1D
        except ImportError:
            pytest.skip("dsrl_official not available")

        # 手动创建 official wrapper
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        agent_state = ckpt["ema_agent"]
        vnet_state = {
            k.replace("velocity_net.", ""): v
            for k, v in agent_state.items()
            if k.startswith("velocity_net.")
        }
        vnet = ShortCutVelocityUNet1D(
            input_dim=ACTION_DIM,
            global_cond_dim=OBS_DIM,
            diffusion_step_embed_dim=64,
            down_dims=(64, 128, 256),
            n_groups=8,
        ).to(DEVICE)
        vnet.load_state_dict(vnet_state)
        vnet.eval()

        official = OfficialWrapper(
            velocity_net=vnet,
            visual_encoder=None,
            obs_horizon=OBS_HORIZON,
            pred_horizon=PRED_HORIZON,
            action_dim=ACTION_DIM,
            num_inference_steps=8,
            device=DEVICE,
        )

        # 测试
        B = 4
        obs = torch.randn(B, OBS_DIM, device=DEVICE)
        noise = torch.randn(B, PRED_HORIZON, ACTION_DIM, device=DEVICE) * 0.3

        rlft_out = rlft_wrapper(obs, noise, return_numpy=False)
        official_out = official(obs, noise, return_numpy=False)

        diff = (rlft_out - official_out).abs().max().item()
        print(f"  Max diff between rlft and dsrl_official: {diff:.2e}")
        assert diff < 1e-4, f"Outputs differ: max_diff={diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
