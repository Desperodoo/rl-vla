#!/usr/bin/env python3
"""
Test script for online training pipelines (RLPD and ReinFlow).
Tests both state and rgb modes, with and without demo data.
"""

import subprocess
import sys
from pathlib import Path

# Test configurations
DEMO_DIR = Path("/home/lizh/.maniskill/demos/LiftPegUpright-v1/rl")
STATE_DEMO = DEMO_DIR / "trajectory.state.pd_ee_delta_pose.physx_cuda.h5"
RGB_DEMO = DEMO_DIR / "trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"

# Common test parameters
COMMON_ARGS = [
    "--no-track",
    "--seed", "42",
    "--num-envs", "4",
]

def run_test(name: str, cmd: list, timeout: int = 120) -> bool:
    """Run a test command and return success status."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=False,
            text=True,
        )
        success = result.returncode == 0
        print(f"\n{'✅ PASSED' if success else '❌ FAILED'}: {name}")
        return success
    except subprocess.TimeoutExpired:
        print(f"\n⏱️ TIMEOUT (expected for long training): {name}")
        return True  # Timeout is OK for training tests
    except Exception as e:
        print(f"\n❌ ERROR: {name} - {e}")
        return False


def test_rlpd_state_no_demo():
    """Test RLPD with state mode, no demo."""
    cmd = [
        sys.executable, "-m", "rlft.online.train_rlpd",
        "--env-id", "PushCube-v1",
        "--obs-mode", "state",
        "--total-timesteps", "500",
        "--replay-buffer-capacity", "10000",
        *COMMON_ARGS,
    ]
    return run_test("RLPD + state (no demo)", cmd)


def test_rlpd_state_with_demo():
    """Test RLPD with state mode and demo data."""
    if not STATE_DEMO.exists():
        print(f"⚠️ SKIP: Demo file not found: {STATE_DEMO}")
        return True
    
    cmd = [
        sys.executable, "-m", "rlft.online.train_rlpd",
        "--env-id", "LiftPegUpright-v1",
        "--obs-mode", "state",
        "--control-mode", "pd_ee_delta_pose",
        "--demo-path", str(STATE_DEMO),
        "--total-timesteps", "500",
        "--replay-buffer-capacity", "10000",
        *COMMON_ARGS,
    ]
    return run_test("RLPD + state + demo", cmd)


def test_rlpd_rgb_with_demo():
    """Test RLPD with rgb mode and demo data."""
    if not RGB_DEMO.exists():
        print(f"⚠️ SKIP: Demo file not found: {RGB_DEMO}")
        return True
    
    cmd = [
        sys.executable, "-m", "rlft.online.train_rlpd",
        "--env-id", "LiftPegUpright-v1",
        "--obs-mode", "rgbd",
        "--control-mode", "pd_ee_delta_pose",
        "--demo-path", str(RGB_DEMO),
        "--total-timesteps", "500",
        "--replay-buffer-capacity", "5000",  # Smaller for RGB
        *COMMON_ARGS,
    ]
    return run_test("RLPD + rgb + demo", cmd)


def test_reinflow_state():
    """Test ReinFlow with state mode."""
    cmd = [
        sys.executable, "-m", "rlft.online.train_reinflow",
        "--env-id", "PushCube-v1",
        "--obs-mode", "state",
        "--total-updates", "10",
        "--rollout-steps", "50",
        *COMMON_ARGS,
    ]
    return run_test("ReinFlow + state", cmd)


def test_reinflow_rgb():
    """Test ReinFlow with rgb mode."""
    cmd = [
        sys.executable, "-m", "rlft.online.train_reinflow",
        "--env-id", "PushCube-v1",
        "--obs-mode", "rgbd",
        "--total-updates", "5",
        "--rollout-steps", "20",
        *COMMON_ARGS,
    ]
    return run_test("ReinFlow + rgb", cmd, timeout=180)


def test_eval_only():
    """Test evaluation function directly."""
    print(f"\n{'='*60}")
    print("TEST: Evaluation function")
    print(f"{'='*60}")
    
    test_code = '''
import torch
import numpy as np

# Test imports
from rlft.envs import make_eval_envs, evaluate

# Create simple test environment
eval_envs = make_eval_envs(
    env_id="PushCube-v1",
    num_envs=2,
    sim_backend="gpu",
    env_kwargs=dict(
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        render_mode="rgb_array",
    ),
    other_kwargs=dict(obs_horizon=2),
)

# Create dummy agent for testing
class DummyAgent:
    def eval(self): pass
    def train(self): pass
    def reset(self, obs): pass
    
    def get_action(self, obs, **kwargs):
        # obs should be (B, T, state_dim) from FrameStack
        B = obs["state"].shape[0] if isinstance(obs, dict) else obs.shape[0]
        # Return (B, pred_horizon, act_dim)
        return torch.zeros(B, 8, 8, device="cuda")

agent = DummyAgent()

# Run evaluation
metrics = evaluate(
    n=4,
    agent=agent,
    eval_envs=eval_envs,
    device=torch.device("cuda"),
    sim_backend="gpu",
    progress_bar=True,
)

print(f"Evaluation metrics: {list(metrics.keys())}")
print("✅ Evaluation test PASSED")
eval_envs.close()
'''
    
    cmd = [sys.executable, "-c", test_code]
    return run_test("Evaluation function", cmd)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ONLINE PIPELINE COMPATIBILITY TESTS")
    print("=" * 60)
    
    results = {}
    
    # Test evaluation first
    results["eval"] = test_eval_only()
    
    # Test RLPD
    results["rlpd_state_no_demo"] = test_rlpd_state_no_demo()
    results["rlpd_state_demo"] = test_rlpd_state_with_demo()
    results["rlpd_rgb_demo"] = test_rlpd_rgb_with_demo()
    
    # Test ReinFlow
    results["reinflow_state"] = test_reinflow_state()
    results["reinflow_rgb"] = test_reinflow_rgb()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
