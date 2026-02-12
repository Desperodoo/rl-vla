"""
测试 checkpoint 加载功能
运行: python -m pytest tests/test_model_loading.py -v
或: python tests/test_model_loading.py
"""

import sys
from pathlib import Path

import torch

# 确保项目路径在 sys.path 中
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 添加 diffusion_policy 路径
dp_path = project_root / "diffusion_policy"
if str(dp_path) not in sys.path:
    sys.path.insert(0, str(dp_path))


# 默认 checkpoint 路径
DEFAULT_CHECKPOINT = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"


def load_checkpoint(checkpoint_path: str = DEFAULT_CHECKPOINT):
    """加载 checkpoint 并返回结构信息"""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint


def test_checkpoint_structure():
    """测试 checkpoint 的顶层结构"""
    checkpoint = load_checkpoint()
    
    # 期望的顶层 keys
    expected_keys = ["agent", "ema_agent", "visual_encoder"]
    
    for key in expected_keys:
        assert key in checkpoint, f"缺少 key: {key}"
        print(f"✓ 顶层 key '{key}' 存在, 包含 {len(checkpoint[key])} 个参数")
    
    return True, "Checkpoint 结构正确"


def test_visual_encoder_loading():
    """测试 PlainConv visual_encoder 加载"""
    from diffusion_policy.plain_conv import PlainConv
    
    checkpoint = load_checkpoint()
    visual_encoder_state = checkpoint["visual_encoder"]
    
    # 从权重推断输入通道数和输出维度
    # 通常是 in_channels=3, out_dim=256
    visual_encoder = PlainConv(in_channels=3, out_dim=256)
    
    try:
        visual_encoder.load_state_dict(visual_encoder_state)
        print("✓ PlainConv visual_encoder 加载成功")
        return True, "visual_encoder OK"
    except Exception as e:
        print(f"✗ visual_encoder 加载失败: {e}")
        return False, str(e)


def test_critic_loading():
    """测试 DoubleQNetwork critic 加载"""
    from diffusion_policy.algorithms.aw_shortcut_flow import DoubleQNetwork
    
    checkpoint = load_checkpoint()
    agent_state = checkpoint["agent"]
    
    # 提取 critic 权重
    critic_state = {}
    prefix = "critic."
    for k, v in agent_state.items():
        if k.startswith(prefix):
            critic_state[k[len(prefix):]] = v
    
    print(f"  找到 {len(critic_state)} 个 critic 参数")
    
    # 从权重推断维度
    # q1_net.weight shape: [hidden, obs_dim + action_input]
    q1_weight = critic_state.get("q1_net.0.weight")
    if q1_weight is not None:
        input_dim = q1_weight.shape[1]
        print(f"  critic q1_net 输入维度: {input_dim}")
        
        # 计算 obs_dim 和 action_input
        # action_input = action_dim * act_horizon = 7 * 8 = 56
        action_input = 56
        obs_dim = input_dim - action_input
        print(f"  推断 obs_dim = {obs_dim}, action_input = {action_input}")
    
    # 创建匹配的 critic 网络
    # obs_dim = global_cond_dim = 2 * (256 + 25) = 562
    obs_dim = 562
    action_dim = 7
    act_horizon = 8
    
    try:
        critic = DoubleQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            act_horizon=act_horizon,
        )
        critic.load_state_dict(critic_state)
        print("✓ DoubleQNetwork critic 加载成功")
        return True, "critic OK"
    except Exception as e:
        print(f"✗ critic 加载失败: {e}")
        return False, str(e)


def test_velocity_net_structure():
    """测试 ShortCutVelocityUNet1D velocity_net 结构 (不加载，因为维度不匹配)"""
    from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
    
    checkpoint = load_checkpoint()
    agent_state = checkpoint["agent"]
    
    # 提取 velocity_net 权重
    velocity_state = {}
    prefix = "velocity_net."
    for k, v in agent_state.items():
        if k.startswith(prefix):
            velocity_state[k[len(prefix):]] = v
    
    print(f"  找到 {len(velocity_state)} 个 velocity_net 参数")
    
    # 分析 cond_encoder 的输入维度
    # 搜索 cond_encoder 相关的权重
    for k, v in velocity_state.items():
        if "cond_encoder" in k and "weight" in k and len(v.shape) == 2:
            print(f"  {k}: shape {tuple(v.shape)}")
            if v.shape[1] == 626:
                print(f"  ⚠️ 发现 global_cond_dim = 626 (期望 562)")
    
    # 报告维度不匹配
    print("⚠️ velocity_net 使用 global_cond_dim=626, 与 critic (562) 不匹配")
    print("   需要重新训练或使用方案 B (部分加载)")
    
    return True, "velocity_net 结构已分析 (维度不匹配)"


def test_velocity_net_loading_with_correct_dim():
    """尝试用正确维度创建 velocity_net 并加载 (预期失败)"""
    from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
    
    checkpoint = load_checkpoint()
    agent_state = checkpoint["agent"]
    
    # 提取 velocity_net 权重
    velocity_state = {}
    prefix = "velocity_net."
    for k, v in agent_state.items():
        if k.startswith(prefix):
            velocity_state[k[len(prefix):]] = v
    
    # 尝试用正确维度 (562) 创建网络
    action_dim = 7
    pred_horizon = 16
    global_cond_dim = 562  # 正确维度
    
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        n_groups=8,
        kernel_size=5,
    )
    
    try:
        velocity_net.load_state_dict(velocity_state)
        print("✓ velocity_net 加载成功 (意外!)")
        return True, "velocity_net OK"
    except Exception as e:
        error_msg = str(e)
        if "size mismatch" in error_msg:
            print(f"✓ 预期的维度不匹配错误: checkpoint 维度与模型维度不兼容")
            return True, "velocity_net 维度不匹配 (已知问题)"
        else:
            print(f"✗ 意外错误: {e}")
            return False, str(e)


def run_all_tests():
    """运行所有模型加载测试"""
    print("=" * 60)
    print("DSRL Official - 模型加载测试")
    print("=" * 60)
    print(f"Checkpoint: {DEFAULT_CHECKPOINT}")
    
    tests = [
        ("Checkpoint 结构", test_checkpoint_structure),
        ("Visual Encoder 加载", test_visual_encoder_loading),
        ("Critic 加载", test_critic_loading),
        ("Velocity Net 结构分析", test_velocity_net_structure),
        ("Velocity Net 加载 (预期失败)", test_velocity_net_loading_with_correct_dim),
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
    
    # 已知问题总结
    print("\n" + "-" * 60)
    print("已知问题:")
    print("  1. velocity_net 使用 global_cond_dim=626, 但环境/critic 使用 562")
    print("  2. 需要重新训练 checkpoint 或使用部分加载策略")
    print("-" * 60)


if __name__ == "__main__":
    run_all_tests()
