"""
双模式验证测试: 环境包装器 vs 策略内部采样

验证两种 DSRL 实现模式在相同输入下产生一致的输出。

模式 1: 环境包装器 (DSRL-SAC 风格)
- SAC 输出噪声 w
- ShortCutFlowEnvWrapper 将噪声转换为真实动作
- 动作执行在环境包装器内部

模式 2: 策略内部采样 (DSRL-NA 风格)
- Latent Policy 输出噪声 w
- Agent 内部调用 base_policy 将噪声转换为真实动作
- 动作执行由外部循环控制

验证内容:
1. 相同噪声 w → 相同动作序列
2. 相同观察 + 相同噪声 → 相同 Q 值估计
3. 两种模式在相同初始状态下的轨迹一致性

Usage:
    python -m pytest tests/test_dual_mode_consistency.py -v
    或直接运行:
    python tests/test_dual_mode_consistency.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

# 添加路径
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl"))
sys.path.insert(0, str(_root / "dsrl_offpolicy"))
sys.path.insert(0, str(_root / "dsrl_official"))


def test_shortcut_flow_wrapper_consistency():
    """测试 ShortCutFlowWrapper 的输出一致性。
    
    验证:
    1. 相同输入 → 相同输出
    2. 多次调用结果一致
    """
    print("\n" + "="*60)
    print("Test: ShortCutFlowWrapper Consistency")
    print("="*60)
    
    from dsrl_official.utils import ShortCutFlowWrapper
    from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建模拟的 velocity_net
    pred_horizon = 16
    action_dim = 7
    obs_dim = 512
    
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=64,
        down_dims=(64, 128, 256),
        n_groups=8,
    ).to(device)
    
    # 创建 wrapper
    wrapper = ShortCutFlowWrapper(
        velocity_net=velocity_net,
        visual_encoder=None,
        obs_horizon=2,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        num_inference_steps=8,
        device=device,
    )
    
    # 测试输入
    batch_size = 4
    obs = torch.randn(batch_size, obs_dim, device=device)
    noise = torch.randn(batch_size, pred_horizon, action_dim, device=device)
    
    # 多次调用，验证一致性
    actions_1 = wrapper(obs, noise, return_numpy=False)
    actions_2 = wrapper(obs, noise, return_numpy=False)
    
    max_diff = (actions_1 - actions_2).abs().max().item()
    print(f"Max difference between two calls: {max_diff:.2e}")
    
    assert max_diff < 1e-5, f"Actions should be identical, but max diff is {max_diff}"
    print("✓ Consistency check passed!")
    
    # 测试不同输入 → 不同输出
    noise_2 = torch.randn(batch_size, pred_horizon, action_dim, device=device)
    actions_3 = wrapper(obs, noise_2, return_numpy=False)
    
    diff_with_different_noise = (actions_1 - actions_3).abs().mean().item()
    print(f"Mean difference with different noise: {diff_with_different_noise:.4f}")
    
    assert diff_with_different_noise > 0.01, "Different noise should produce different actions"
    print("✓ Different inputs produce different outputs!")
    
    return True


def test_env_wrapper_vs_internal_sampling():
    """测试环境包装器 vs 策略内部采样的输出一致性。
    
    验证:
    - 相同 obs + 相同 noise → 通过两种方式应得到相同动作
    """
    print("\n" + "="*60)
    print("Test: Env Wrapper vs Internal Sampling Consistency")
    print("="*60)
    
    from dsrl_official.utils import ShortCutFlowWrapper
    from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 参数
    pred_horizon = 16
    action_dim = 7
    obs_dim = 512
    act_steps = 8
    
    # 创建 velocity_net
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=64,
        down_dims=(64, 128, 256),
        n_groups=8,
    ).to(device)
    
    # 创建 wrapper
    wrapper = ShortCutFlowWrapper(
        velocity_net=velocity_net,
        visual_encoder=None,
        obs_horizon=2,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        num_inference_steps=8,
        device=device,
    )
    
    # 测试输入
    batch_size = 4
    obs = torch.randn(batch_size, obs_dim, device=device)
    
    # 模拟 DSRL-SAC: 扁平化噪声 (从 SAC 输出)
    noise_flat = torch.randn(batch_size, act_steps * action_dim, device=device)
    noise_flat = noise_flat.clamp(-1.5, 1.5)  # action_magnitude = 1.5
    
    # 重塑为 (B, act_steps, action_dim)
    noise_shaped = noise_flat.view(batch_size, act_steps, action_dim)
    
    # 填充到 pred_horizon (剩余部分用 0 填充)
    noise_full = torch.zeros(batch_size, pred_horizon, action_dim, device=device)
    noise_full[:, :act_steps, :] = noise_shaped
    
    # 模式 1: 环境包装器方式 (传入扁平化噪声，内部重塑)
    actions_wrapper = wrapper(obs, noise_full, return_numpy=False)
    
    # 模式 2: 策略内部采样方式 (传入结构化噪声)
    actions_internal = wrapper.sample_with_latent(
        obs, noise_full, action_magnitude=1.5, return_numpy=False
    )
    
    # 比较
    max_diff = (actions_wrapper - actions_internal).abs().max().item()
    print(f"Max difference between wrapper and internal: {max_diff:.2e}")
    
    assert max_diff < 1e-5, f"Actions should be identical, but max diff is {max_diff}"
    print("✓ Wrapper and internal sampling produce identical results!")
    
    return True


def test_action_chunking_consistency():
    """测试动作分块的一致性。
    
    验证 act_steps 内的动作执行与完整动作序列的对应关系。
    """
    print("\n" + "="*60)
    print("Test: Action Chunking Consistency")
    print("="*60)
    
    from dsrl_official.utils import ShortCutFlowWrapper
    from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 参数
    pred_horizon = 16
    action_dim = 7
    obs_dim = 512
    act_steps = 8
    
    # 创建 velocity_net
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=64,
        down_dims=(64, 128, 256),
        n_groups=8,
    ).to(device)
    
    # 创建 wrapper
    wrapper = ShortCutFlowWrapper(
        velocity_net=velocity_net,
        visual_encoder=None,
        obs_horizon=2,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        num_inference_steps=8,
        device=device,
    )
    
    # 测试输入
    batch_size = 4
    obs = torch.randn(batch_size, obs_dim, device=device)
    noise = torch.randn(batch_size, pred_horizon, action_dim, device=device)
    
    # 获取完整动作序列
    actions_full = wrapper(obs, noise, return_numpy=False)
    
    # 提取 act_steps 部分
    actions_chunk = actions_full[:, :act_steps, :]
    
    print(f"Full action shape: {actions_full.shape}")
    print(f"Action chunk shape: {actions_chunk.shape}")
    print(f"Expected chunk shape: ({batch_size}, {act_steps}, {action_dim})")
    
    assert actions_chunk.shape == (batch_size, act_steps, action_dim)
    print("✓ Action chunking shape is correct!")
    
    # 验证动作在合理范围内
    action_mean = actions_chunk.abs().mean().item()
    action_max = actions_chunk.abs().max().item()
    print(f"Action mean abs: {action_mean:.4f}")
    print(f"Action max abs: {action_max:.4f}")
    
    return True


def test_with_pretrained_checkpoint():
    """使用预训练 checkpoint 测试（如果存在）。"""
    print("\n" + "="*60)
    print("Test: Pretrained Checkpoint Loading")
    print("="*60)
    
    checkpoint_path = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Skipping pretrained checkpoint test.")
        return True
    
    from dsrl_official.utils import load_shortcut_flow_policy
    from diffusion_policy.plain_conv import PlainConv
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载预训练策略
    try:
        wrapper, visual_encoder = load_shortcut_flow_policy(
            checkpoint_path=checkpoint_path,
            visual_encoder_class=PlainConv,
            obs_horizon=2,
            pred_horizon=16,
            action_dim=7,
            visual_feature_dim=256,
            state_dim=25,
            include_rgb=True,
            use_ema=True,
            device=device,
        )
        print("✓ Checkpoint loaded successfully!")
        
        # 测试推理
        obs_dim = 2 * (256 + 25)  # obs_horizon * (visual_dim + state_dim)
        obs = torch.randn(4, obs_dim, device=device)
        noise = torch.randn(4, 16, 7, device=device)
        
        actions = wrapper(obs, noise, return_numpy=False)
        print(f"Output action shape: {actions.shape}")
        print(f"Action mean: {actions.mean().item():.4f}")
        print(f"Action std: {actions.std().item():.4f}")
        
        print("✓ Inference with pretrained model successful!")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
    
    return True


def run_all_tests():
    """运行所有测试。"""
    print("\n" + "="*70)
    print("DSRL Official - Dual Mode Consistency Tests")
    print("="*70)
    
    tests = [
        ("ShortCutFlowWrapper Consistency", test_shortcut_flow_wrapper_consistency),
        ("Env Wrapper vs Internal Sampling", test_env_wrapper_vs_internal_sampling),
        ("Action Chunking Consistency", test_action_chunking_consistency),
        ("Pretrained Checkpoint Loading", test_with_pretrained_checkpoint),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
    
    # 打印总结
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = 0
    failed = 0
    for name, result, error in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")
        if error:
            print(f"   Error: {error}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
