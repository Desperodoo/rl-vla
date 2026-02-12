"""
测试所有必要的导入
运行: python -m pytest tests/test_imports.py -v
或: python tests/test_imports.py
"""

import sys
from pathlib import Path

# 确保项目路径在 sys.path 中
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_stable_baselines3():
    """测试 stable-baselines3 导入"""
    from stable_baselines3 import SAC
    print("✓ stable_baselines3.SAC 导入成功")
    
    # 检查是否是 DSRL fork 版本
    try:
        from stable_baselines3 import DSRL
        print("✓ stable_baselines3.DSRL (fork) 导入成功")
        return True, "SAC + DSRL (fork 版本)"
    except ImportError:
        print("⚠ stable_baselines3.DSRL 不可用 (标准版本)")
        return True, "SAC only (标准版本)"


def test_maniskill3():
    """测试 ManiSkill3 导入"""
    import mani_skill.envs
    print("✓ mani_skill.envs 导入成功")
    
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    print("✓ FlattenRGBDObservationWrapper 导入成功")
    
    return True, "ManiSkill3 OK"


def test_diffusion_policy():
    """测试 diffusion_policy 模块导入"""
    # 添加 diffusion_policy 路径
    import sys
    from pathlib import Path
    dp_path = Path(__file__).parent.parent.parent / "diffusion_policy"
    if str(dp_path) not in sys.path:
        sys.path.insert(0, str(dp_path))
    
    from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
    print("✓ ShortCutVelocityUNet1D 导入成功")
    
    from diffusion_policy.algorithms.aw_shortcut_flow import AWShortCutFlowAgent
    print("✓ AWShortCutFlowAgent 导入成功")
    
    from diffusion_policy.plain_conv import PlainConv
    print("✓ PlainConv 导入成功")
    
    return True, "diffusion_policy OK"


def test_dsrl_official_modules():
    """测试 dsrl_official 模块导入"""
    from dsrl_official import utils
    print("✓ dsrl_official.utils 导入成功")
    
    from dsrl_official import env_utils
    print("✓ dsrl_official.env_utils 导入成功")
    
    from dsrl_official import callbacks
    print("✓ dsrl_official.callbacks 导入成功")
    
    # 验证关键类存在
    assert hasattr(utils, 'ShortCutFlowWrapper'), "ShortCutFlowWrapper 未定义"
    print("✓ ShortCutFlowWrapper 类存在")
    
    # 注意: 类名是 ShortCutFlowEnvWrapper, 不是 DSRLEnvWrapper
    assert hasattr(env_utils, 'ShortCutFlowEnvWrapper'), "ShortCutFlowEnvWrapper 未定义"
    print("✓ ShortCutFlowEnvWrapper 类存在")
    
    return True, "dsrl_official OK"


def test_torch():
    """测试 PyTorch 导入"""
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA 不可用")
    
    return True, f"PyTorch {torch.__version__}"


def run_all_tests():
    """运行所有导入测试"""
    print("=" * 60)
    print("DSRL Official - 导入测试")
    print("=" * 60)
    
    tests = [
        ("PyTorch", test_torch),
        ("stable-baselines3", test_stable_baselines3),
        ("ManiSkill3", test_maniskill3),
        ("diffusion_policy", test_diffusion_policy),
        ("dsrl_official", test_dsrl_official_modules),
    ]
    
    results = []
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            success, msg = test_fn()
            results.append((name, True, msg))
        except Exception as e:
            print(f"✗ {name} 失败: {e}")
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
        print("\n⚠ 有测试失败，请检查环境配置")
        sys.exit(1)
    else:
        print("\n✓ 所有导入测试通过!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
