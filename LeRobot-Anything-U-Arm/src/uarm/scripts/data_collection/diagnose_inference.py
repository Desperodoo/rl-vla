#!/usr/bin/env python3
"""
Simple step-by-step diagnostic for ARX5 policy inference.

This script tests each component individually to find where the problem is.
"""

import sys
import os
import time
import numpy as np

# Add paths
sys.path.insert(0, "/home/lizh/arx5-sdk/python")
os.chdir("/home/lizh/arx5-sdk/python")

RLFT_PATH = os.path.expanduser("~/rlft/diffusion_policy")
if RLFT_PATH not in sys.path:
    sys.path.insert(0, RLFT_PATH)

def test_robot_connection():
    """Test 1: Check robot connection and state reading."""
    print("\n" + "=" * 60)
    print("Test 1: Robot Connection")
    print("=" * 60)
    
    try:
        import arx5_interface as arx5
        robot = arx5.Arx5JointController("X5", "can0")
        
        state = robot.get_state()
        print(f"✓ Robot connected!")
        print(f"  Joint positions: {np.array(state.pos())}")
        print(f"  Joint velocities: {np.array(state.vel())}")
        print(f"  Gripper position: {state.gripper_pos}")
        
        return robot
    except Exception as e:
        print(f"✗ Robot connection failed: {e}")
        return None


def test_simple_motion(robot, amplitude=0.05):
    """Test 2: Send a simple motion command."""
    print("\n" + "=" * 60)
    print("Test 2: Simple Motion Command")
    print("=" * 60)
    
    if robot is None:
        print("✗ Skipping - no robot")
        return
    
    import arx5_interface as arx5
    
    # Get current state
    state = robot.get_state()
    current_pos = np.array(state.pos())
    
    print(f"Current position: {current_pos}")
    
    # Create target position (small offset on joint 0)
    target_pos = current_pos.copy()
    target_pos[0] += amplitude  # Move joint 0 by amplitude rad
    
    print(f"Target position:  {target_pos}")
    print(f"Delta:            {target_pos - current_pos}")
    
    input("\nPress Enter to execute motion (Ctrl+C to cancel)...")
    
    # Send command
    js = arx5.JointState(6)
    js.pos()[:] = target_pos
    js.gripper_pos = float(state.gripper_pos)
    
    print("Sending command...")
    
    # Execute for 1 second
    start = time.time()
    while time.time() - start < 1.0:
        robot.set_joint_cmd(js)
        robot.send_recv_once()
        time.sleep(0.002)  # 500Hz
    
    # Check final position
    final_state = robot.get_state()
    final_pos = np.array(final_state.pos())
    
    print(f"\nFinal position:   {final_pos}")
    print(f"Achieved delta:   {final_pos - current_pos}")
    
    if np.abs(final_pos[0] - target_pos[0]) < 0.01:
        print("✓ Motion command works!")
    else:
        print("✗ Motion command may have issues")


def test_policy_output():
    """Test 3: Check policy output format."""
    print("\n" + "=" * 60)
    print("Test 3: Policy Output Format")
    print("=" * 60)
    
    # Load a small sample from dataset
    import h5py
    
    dataset_path = os.path.expanduser("~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5")
    
    with h5py.File(dataset_path, 'r') as f:
        traj = f['traj_0']
        obs = traj['obs']
        
        # Get sample
        joint_pos = np.array(obs['joint_pos'][100])
        joint_vel = np.array(obs['joint_vel'][100])
        gripper_pos = np.array(obs['gripper_pos'][100])
        rgb = np.array(obs['images']['wrist']['rgb'][100])
        action = np.array(traj['actions'][100])
        
        print(f"Sample observation at frame 100:")
        print(f"  joint_pos:   {joint_pos}")
        print(f"  joint_vel:   {joint_vel}")
        print(f"  gripper_pos: {gripper_pos}")
        print(f"  rgb shape:   {rgb.shape}")
        print()
        print(f"Sample action at frame 100:")
        print(f"  action:      {action}")
        print()
        
        # Check if action is absolute position or delta
        prev_action = np.array(traj['actions'][99])
        delta = action - prev_action
        print(f"Previous action: {prev_action}")
        print(f"Delta:           {delta}")
        print()
        
        # Compare action to joint_pos
        print(f"Action vs joint_pos:")
        print(f"  action[:6]:     {action[:6]}")
        print(f"  joint_pos:      {joint_pos}")
        print(f"  difference:     {action[:6] - joint_pos}")
        
        if np.allclose(action[:6], joint_pos, atol=0.1):
            print("\n✓ Action appears to be ABSOLUTE position (same as joint_pos)")
        else:
            print("\n? Action format unclear")


def test_action_execution_loop(robot):
    """Test 4: Execute actions from dataset."""
    print("\n" + "=" * 60)
    print("Test 4: Execute Actions from Dataset")
    print("=" * 60)
    
    if robot is None:
        print("✗ Skipping - no robot")
        return
    
    import h5py
    import arx5_interface as arx5
    
    dataset_path = os.path.expanduser("~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5")
    
    with h5py.File(dataset_path, 'r') as f:
        traj = f['traj_0']
        actions = np.array(traj['actions'])
    
    print(f"Loaded {len(actions)} actions from dataset")
    print(f"Action range: [{actions.min():.4f}, {actions.max():.4f}]")
    
    # Get current state
    state = robot.get_state()
    current_pos = np.array(state.pos())
    
    print(f"\nCurrent robot position: {current_pos}")
    print(f"First action in dataset: {actions[0]}")
    
    input("\nPress Enter to replay first 100 actions at 30Hz (Ctrl+C to cancel)...")
    
    # Replay at 30Hz
    for i in range(min(100, len(actions))):
        action = actions[i]
        
        js = arx5.JointState(6)
        js.pos()[:] = action[:6]
        js.gripper_pos = float(action[6])
        
        robot.set_joint_cmd(js)
        robot.send_recv_once()
        
        if i % 10 == 0:
            state = robot.get_state()
            print(f"Step {i}: target={action[:3]}, actual={np.array(state.pos())[:3]}")
        
        time.sleep(1/30)  # 30Hz
    
    print("✓ Replay complete!")


def main():
    print("=" * 60)
    print("ARX5 Policy Inference Diagnostic")
    print("=" * 60)
    
    # Test 1: Robot connection
    robot = test_robot_connection()
    
    # Test 2: Simple motion
    if robot:
        try:
            test_simple_motion(robot, amplitude=0.05)
        except KeyboardInterrupt:
            print("\nSkipped")
    
    # Test 3: Policy output format
    test_policy_output()
    
    # Test 4: Action execution
    if robot:
        try:
            test_action_execution_loop(robot)
        except KeyboardInterrupt:
            print("\nSkipped")
    
    # Reset robot
    if robot:
        print("\nResetting robot to home...")
        robot.reset_to_home()
    
    print("\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
