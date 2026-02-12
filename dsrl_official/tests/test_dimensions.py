"""
测试输入输出维度一致性
运行: python -m pytest tests/test_dimensions.py -v
或: python tests/test_dimensions.py
"""

import sys
from pathlib import Path

import torch
import numpy as np

# 确保项目路径在 sys.path 中
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 添加 diffusion_policy 路径
dp_path = project_root / "diffusion_policy"
if str(dp_path) not in sys.path:
    sys.path.insert(0, str(dp_path))


# 固定配置
CONFIG = {
    "env_id": "LiftPegUpright-v1",
    "state_dim": 25,           # qpos(9) + qvel(9) + tcp_pose(7)
    "visual_dim": 256,          # PlainConv output
    "obs_horizon": 2,
    "action_dim": 7,
    "act_horizon": 8,
    "pred_horizon": 16,
    # 计算得出
    "global_cond_dim": 562,     # 2 * (256 + 25)
    "action_input": 56,         # 7 * 8
}


def test_environment_observation_space():
    """测试环境观察空间维度"""
    import gymnasium as gym
    import mani_skill.envs  # 注册环境
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    
    # 创建环境
    env = gym.make(
        CONFIG["env_id"],
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
        sim_backend="cpu",  # CPU 后端用于测试
    )
    
    # 应用 FlattenRGBDObservationWrapper
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    
    obs, _ = env.reset()
    
    # 检查观察空间
    print(f"  obs type: {type(obs)}")
    if isinstance(obs, dict):
        for k, v in obs.items():
            if hasattr(v, 'shape'):
                print(f"  obs['{k}']: shape {v.shape}")
    else:
        print(f"  obs shape: {obs.shape}")
    
    # 获取 state 维度
    if "state" in obs:
        state = obs["state"]
        actual_state_dim = state.shape[-1] if len(state.shape) > 0 else 1
        print(f"  state_dim: 期望 {CONFIG['state_dim']}, 实际 {actual_state_dim}")
        
        # 容忍差异 (env 可能有不同的 state 组成)
        if actual_state_dim != CONFIG["state_dim"]:
            print(f"  ⚠️ state_dim 不匹配! 需要更新配置")
    
    # 获取动作空间
    action_space = env.action_space
    actual_action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else None
    print(f"  action_dim: 期望 {CONFIG['action_dim']}, 实际 {actual_action_dim}")
    
    assert actual_action_dim == CONFIG["action_dim"], \
        f"action_dim 不匹配: 期望 {CONFIG['action_dim']}, 实际 {actual_action_dim}"
    
    env.close()
    
    return True, "环境观察空间 OK"


def test_visual_encoder_dimensions():
    """测试 PlainConv visual_encoder 输入输出维度"""
    from diffusion_policy.plain_conv import PlainConv
    
    # 创建 visual encoder
    visual_encoder = PlainConv(
        in_channels=3,  # RGB
        out_dim=CONFIG["visual_dim"],
    )
    
    # 测试输入
    # 假设图像大小 128x128 (ManiSkill 默认)
    batch_size = 4
    test_image = torch.randn(batch_size, 3, 128, 128)
    
    with torch.no_grad():
        output = visual_encoder(test_image)
    
    print(f"  input shape: {tuple(test_image.shape)}")
    print(f"  output shape: {tuple(output.shape)}")
    
    expected_shape = (batch_size, CONFIG["visual_dim"])
    assert output.shape == expected_shape, \
        f"visual_encoder 输出维度不匹配: 期望 {expected_shape}, 实际 {tuple(output.shape)}"
    
    print(f"✓ visual_encoder: {tuple(test_image.shape)} -> {tuple(output.shape)}")
    
    return True, f"visual_encoder: (B, 3, H, W) -> (B, {CONFIG['visual_dim']})"


def test_velocity_net_dimensions():
    """测试 ShortCutVelocityUNet1D 输入输出维度"""
    from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
    
    # 创建 velocity_net (不使用 predict_noise 参数)
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=CONFIG["action_dim"],
        global_cond_dim=CONFIG["global_cond_dim"],
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        n_groups=8,
        kernel_size=5,
    )
    
    batch_size = 4
    
    # 输入
    sample = torch.randn(batch_size, CONFIG["pred_horizon"], CONFIG["action_dim"])  # (B, T, C)
    timestep = torch.rand(batch_size) * 0.99 + 0.01  # (B,) in (0, 1)
    step_size = torch.ones(batch_size)  # (B,)
    global_cond = torch.randn(batch_size, CONFIG["global_cond_dim"])  # (B, D)
    
    print(f"  sample: {tuple(sample.shape)}")
    print(f"  timestep: {tuple(timestep.shape)}")
    print(f"  step_size: {tuple(step_size.shape)}")
    print(f"  global_cond: {tuple(global_cond.shape)}")
    
    with torch.no_grad():
        output = velocity_net(sample, timestep, step_size, global_cond)
    
    print(f"  output: {tuple(output.shape)}")
    
    expected_shape = (batch_size, CONFIG["pred_horizon"], CONFIG["action_dim"])
    assert output.shape == expected_shape, \
        f"velocity_net 输出维度不匹配: 期望 {expected_shape}, 实际 {tuple(output.shape)}"
    
    print(f"✓ velocity_net: sample {tuple(sample.shape)} + cond {tuple(global_cond.shape)} -> {tuple(output.shape)}")
    
    return True, f"velocity_net: (B, {CONFIG['pred_horizon']}, {CONFIG['action_dim']}) -> same"


def test_critic_dimensions():
    """测试 DoubleQNetwork critic 输入输出维度"""
    from diffusion_policy.algorithms.networks import DoubleQNetwork
    
    # 创建 critic
    # 注意: DoubleQNetwork 使用 action_horizon 参数
    critic = DoubleQNetwork(
        obs_dim=CONFIG["global_cond_dim"],
        action_dim=CONFIG["action_dim"],
        action_horizon=CONFIG["act_horizon"],
    )
    
    batch_size = 4
    
    # 输入 - 注意参数顺序: forward(action_seq, obs_cond)
    obs = torch.randn(batch_size, CONFIG["global_cond_dim"])  # (B, obs_dim)
    action = torch.randn(batch_size, CONFIG["act_horizon"], CONFIG["action_dim"])  # (B, T, C)
    
    print(f"  obs: {tuple(obs.shape)}")
    print(f"  action: {tuple(action.shape)}")
    
    with torch.no_grad():
        # DoubleQNetwork.forward(action_seq, obs_cond)
        q1, q2 = critic(action, obs)
    
    print(f"  q1: {tuple(q1.shape)}")
    print(f"  q2: {tuple(q2.shape)}")
    
    expected_shape = (batch_size, 1)
    assert q1.shape == expected_shape, \
        f"critic q1 输出维度不匹配: 期望 {expected_shape}, 实际 {tuple(q1.shape)}"
    assert q2.shape == expected_shape, \
        f"critic q2 输出维度不匹配: 期望 {expected_shape}, 实际 {tuple(q2.shape)}"
    
    print(f"✓ critic: obs {tuple(obs.shape)} + action {tuple(action.shape)} -> q {tuple(q1.shape)}")
    
    return True, f"critic: (B, {CONFIG['global_cond_dim']}) + (B, {CONFIG['act_horizon']}, {CONFIG['action_dim']}) -> (B, 1)"


def test_global_cond_dim_calculation():
    """验证 global_cond_dim 计算公式"""
    expected = CONFIG["obs_horizon"] * (CONFIG["visual_dim"] + CONFIG["state_dim"])
    actual = CONFIG["global_cond_dim"]
    
    print(f"  公式: obs_horizon * (visual_dim + state_dim)")
    print(f"  计算: {CONFIG['obs_horizon']} * ({CONFIG['visual_dim']} + {CONFIG['state_dim']}) = {expected}")
    print(f"  配置: {actual}")
    
    assert expected == actual, \
        f"global_cond_dim 计算错误: 期望 {expected}, 配置 {actual}"
    
    print(f"✓ global_cond_dim = {expected}")
    
    return True, f"global_cond_dim = {expected}"


def test_action_input_calculation():
    """验证 action_input 计算公式"""
    expected = CONFIG["action_dim"] * CONFIG["act_horizon"]
    actual = CONFIG["action_input"]
    
    print(f"  公式: action_dim * act_horizon")
    print(f"  计算: {CONFIG['action_dim']} * {CONFIG['act_horizon']} = {expected}")
    print(f"  配置: {actual}")
    
    assert expected == actual, \
        f"action_input 计算错误: 期望 {expected}, 配置 {actual}"
    
    print(f"✓ action_input = {expected}")
    
    return True, f"action_input = {expected}"


def run_all_tests():
    """运行所有维度测试"""
    print("=" * 60)
    print("DSRL Official - 维度测试")
    print("=" * 60)
    print("固定配置:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    tests = [
        ("global_cond_dim 计算", test_global_cond_dim_calculation),
        ("action_input 计算", test_action_input_calculation),
        ("Visual Encoder 维度", test_visual_encoder_dimensions),
        ("Velocity Net 维度", test_velocity_net_dimensions),
        ("Critic 维度", test_critic_dimensions),
        ("环境观察空间", test_environment_observation_space),
    ]
    
    results = []
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            success, msg = test_fn()
            results.append((name, True, msg))
        except Exception as e:
            print(f"✗ {name} 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, s, _ in results if s)
    failed = len(results) - passed
    
    for name, success, msg in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}: {msg}")
    
    print(f"\n通过: {passed}/{len(results)}, 失败: {failed}/{len(results)}")
    
    if failed > 0:
        print("\n⚠ 有测试失败，请检查配置")
        sys.exit(1)
    else:
        print("\n✓ 所有维度测试通过!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
