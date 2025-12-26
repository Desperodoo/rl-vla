#!/usr/bin/env python3
"""
Test script for ARX5 robot motion without policy inference.
This isolates hardware issues from inference issues.

Usage:
    python test_robot_motion.py
"""

import sys
import time
import numpy as np

def test_robot_connection():
    """Test basic robot connection."""
    print("\n" + "=" * 60)
    print("Step 1: Testing Robot Connection")
    print("=" * 60)
    
    try:
        import arx5_interface as arx5
        print("✓ arx5_interface imported")
    except ImportError as e:
        print(f"✗ Failed to import arx5_interface: {e}")
        return None, None
    
    try:
        solver = arx5.Arx5Solver("X5", "OD")
        print("✓ Solver created")
    except Exception as e:
        print(f"✗ Failed to create solver: {e}")
        return None, None
    
    try:
        ctrl = arx5.Arx5ControllerBase(solver, "can0")
        print("✓ Controller created on can0")
        
        # Enable robot
        ctrl.enable_background_send_recv()
        time.sleep(0.5)
        print("✓ Background send/recv enabled")
        
        return solver, ctrl
    except Exception as e:
        print(f"✗ Failed to create controller: {e}")
        return None, None


def get_current_state(ctrl):
    """Get current robot state."""
    js = ctrl.get_joint_state()
    pos = np.array(js.pos())
    vel = np.array(js.vel())
    gripper = js.gripper_pos
    return pos, vel, gripper


def test_read_state(ctrl):
    """Test reading robot state."""
    print("\n" + "=" * 60)
    print("Step 2: Testing State Reading")
    print("=" * 60)
    
    try:
        pos, vel, gripper = get_current_state(ctrl)
        print(f"✓ Current joint position: {pos}")
        print(f"✓ Current joint velocity: {vel}")
        print(f"✓ Current gripper: {gripper}")
        return pos, gripper
    except Exception as e:
        print(f"✗ Failed to read state: {e}")
        return None, None


def test_small_motion(ctrl, initial_pos, initial_gripper):
    """Test small motion command."""
    print("\n" + "=" * 60)
    print("Step 3: Testing Small Motion")
    print("=" * 60)
    
    import arx5_interface as arx5
    
    print(f"Initial position: {initial_pos[:3]}...")
    print(f"Initial gripper: {initial_gripper}")
    
    # Small delta on joint 2 (shoulder)
    delta = 0.05  # 0.05 rad ≈ 3 degrees
    target_pos = initial_pos.copy()
    target_pos[1] += delta  # Move shoulder slightly
    
    print(f"\nTarget position: {target_pos[:3]}...")
    print(f"Delta on joint 2: {delta} rad ({np.degrees(delta):.1f} deg)")
    
    input("\n>>> Press Enter to send motion command (or Ctrl+C to abort)...")
    
    # Send command
    js = arx5.JointState(6)
    js.pos()[:] = target_pos
    js.gripper_pos = initial_gripper
    
    print("\nSending command...")
    
    # Run for a few steps
    for i in range(50):
        ctrl.set_joint_cmd(js)
        ctrl.send_recv_once()
        
        if i % 10 == 0:
            cur_pos, _, cur_gripper = get_current_state(ctrl)
            print(f"  Step {i}: pos[1]={cur_pos[1]:.4f} (target={target_pos[1]:.4f})")
        
        time.sleep(0.01)  # 100 Hz
    
    # Check final position
    final_pos, _, _ = get_current_state(ctrl)
    error = np.abs(final_pos[1] - target_pos[1])
    
    print(f"\nFinal position: {final_pos[:3]}...")
    print(f"Position error on joint 2: {error:.4f} rad")
    
    if error < 0.01:
        print("✓ Motion command executed successfully!")
        return True
    else:
        print("✗ Motion command may not have been executed properly")
        return False


def test_return_home(ctrl, initial_pos, initial_gripper):
    """Return to initial position."""
    print("\n" + "=" * 60)
    print("Step 4: Returning to Initial Position")
    print("=" * 60)
    
    import arx5_interface as arx5
    
    input(">>> Press Enter to return home (or Ctrl+C to abort)...")
    
    js = arx5.JointState(6)
    js.pos()[:] = initial_pos
    js.gripper_pos = initial_gripper
    
    print("Returning...")
    
    for i in range(50):
        ctrl.set_joint_cmd(js)
        ctrl.send_recv_once()
        time.sleep(0.01)
    
    final_pos, _, _ = get_current_state(ctrl)
    error = np.linalg.norm(final_pos - initial_pos)
    
    print(f"Final position error: {error:.4f} rad")
    
    if error < 0.01:
        print("✓ Returned to initial position")
    else:
        print("✗ May not have returned exactly")


def test_interpolated_motion(ctrl, initial_pos, initial_gripper):
    """Test interpolated motion (similar to inference)."""
    print("\n" + "=" * 60)
    print("Step 5: Testing Interpolated Motion (like inference)")
    print("=" * 60)
    
    import arx5_interface as arx5
    
    # Generate a trajectory like the inference would
    n_steps = 100
    delta = 0.1  # Total motion
    
    # Interpolate linearly
    positions = []
    for i in range(n_steps):
        alpha = i / (n_steps - 1)
        pos = initial_pos.copy()
        pos[1] += delta * alpha  # Gradually move shoulder
        positions.append(pos)
    
    print(f"Generated {n_steps} interpolated steps")
    print(f"Total motion on joint 2: {delta} rad")
    
    input("\n>>> Press Enter to execute trajectory (or Ctrl+C to abort)...")
    
    print("\nExecuting trajectory...")
    
    for i, pos in enumerate(positions):
        js = arx5.JointState(6)
        js.pos()[:] = pos
        js.gripper_pos = initial_gripper
        
        ctrl.set_joint_cmd(js)
        ctrl.send_recv_once()
        
        if i % 20 == 0:
            cur_pos, _, _ = get_current_state(ctrl)
            print(f"  Step {i}/{n_steps}: pos[1]={cur_pos[1]:.4f} (target={pos[1]:.4f})")
        
        time.sleep(0.01)  # 100 Hz
    
    final_pos, _, _ = get_current_state(ctrl)
    print(f"\nFinal position: {final_pos[:3]}...")
    print("✓ Trajectory completed")


def main():
    print("=" * 60)
    print("ARX5 Robot Motion Test")
    print("=" * 60)
    print("\nThis script tests basic robot motion without policy inference.")
    print("Use this to verify hardware is working before running inference.\n")
    
    # Step 1: Connect
    solver, ctrl = test_robot_connection()
    if ctrl is None:
        print("\n✗ Cannot proceed without robot connection")
        return 1
    
    try:
        # Step 2: Read state
        initial_pos, initial_gripper = test_read_state(ctrl)
        if initial_pos is None:
            print("\n✗ Cannot proceed without reading state")
            return 1
        
        # Step 3: Small motion
        success = test_small_motion(ctrl, initial_pos, initial_gripper)
        
        # Step 4: Return home
        test_return_home(ctrl, initial_pos, initial_gripper)
        
        # Step 5: Interpolated motion (optional)
        response = input("\n>>> Run interpolated motion test? (y/n): ")
        if response.lower() == 'y':
            # Re-read position
            initial_pos, initial_gripper = get_current_state(ctrl)[:2] + (initial_gripper,)
            initial_pos, _, initial_gripper = get_current_state(ctrl)
            test_interpolated_motion(ctrl, initial_pos, initial_gripper)
            test_return_home(ctrl, initial_pos, initial_gripper)
        
        print("\n" + "=" * 60)
        print("Test Complete")
        print("=" * 60)
        
        if success:
            print("\n✓ Robot motion is working!")
            print("  If inference still doesn't work, the issue is in:")
            print("  - Policy model output")
            print("  - Observation building")
            print("  - Action interpolation")
            print("\n  Run with --verbose flag to debug inference.")
        else:
            print("\n✗ Robot motion test failed!")
            print("  Check:")
            print("  - CAN bus connection")
            print("  - Motor power")
            print("  - Emergency stop")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    finally:
        # Cleanup
        if ctrl is not None:
            try:
                ctrl.disable_background_send_recv()
            except:
                pass


if __name__ == "__main__":
    sys.exit(main())
