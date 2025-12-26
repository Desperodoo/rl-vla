#!/usr/bin/env python3
"""
Minimal test to debug policy inference execution.
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


def main():
    # 1. Load policy
    print("Loading policy...")
    
    from data_collection.arx5_policy_inference import ARX5PolicyInference, InferenceConfig
    
    config = InferenceConfig()
    config.enable_filter = False  # Disable filter for debugging
    
    checkpoint = os.path.expanduser(
        "~/rlft/diffusion_policy/runs/consistency_flow-pick_cube-real__1__1766132160/checkpoints/iter_42000.pt"
    )
    
    runner = ARX5PolicyInference(
        checkpoint_path=checkpoint,
        config=config,
        device="cuda",
        verbose=True
    )
    
    # 2. Setup
    print("\nSetting up...")
    runner.setup()
    
    if runner.robot_ctrl is None:
        print("ERROR: Robot not connected!")
        return
    
    if runner.camera_manager is None:
        print("WARNING: Camera not connected!")
    
    # 3. Get initial state
    print("\n" + "=" * 60)
    print("Initial Robot State")
    print("=" * 60)
    
    state = runner.robot_ctrl.get_state()
    current_pos = np.array(state.pos())
    gripper_pos = state.gripper_pos
    
    print(f"Current joint_pos: {current_pos}")
    print(f"Current gripper:   {gripper_pos}")
    
    # 4. Get camera observation
    print("\n" + "=" * 60)
    print("Camera Observation")
    print("=" * 60)
    
    obs = runner.get_observation()
    print(f"RGB shape: {obs['rgb'].shape}")
    print(f"State:     {obs['state']}")
    
    # 5. Run inference
    print("\n" + "=" * 60)
    print("Running Inference")
    print("=" * 60)
    
    # Need to fill obs buffer
    runner.obs_buffer.append(obs)
    runner.obs_buffer.append(obs)  # obs_horizon = 2
    
    # Predict
    actions = runner.predict_action()
    
    print(f"Predicted actions shape: {actions.shape}")
    print(f"First predicted action:  {actions[0]}")
    print(f"Last predicted action:   {actions[-1]}")
    
    # 6. Compare with current position
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    
    delta = actions[0, :6] - current_pos
    print(f"Delta (action - current): {delta}")
    print(f"Max delta: {np.abs(delta).max():.4f} rad")
    
    if np.abs(delta).max() < 0.001:
        print("\n⚠️  WARNING: Predicted action is almost same as current position!")
        print("   This means robot won't move much.")
    
    # 7. Interpolate actions
    print("\n" + "=" * 60)
    print("Interpolated Actions")
    print("=" * 60)
    
    interp_actions = runner.interpolate_actions(actions)
    print(f"Interpolated shape: {interp_actions.shape}")
    print(f"First interp:       {interp_actions[0]}")
    print(f"Last interp:        {interp_actions[-1]}")
    
    # 8. Check safety limits
    print("\n" + "=" * 60)
    print("Safety Limits Check")
    print("=" * 60)
    
    prev_action = np.concatenate([current_pos, [gripper_pos]])
    safe_action = runner.apply_safety_limits(interp_actions[0], prev_action)
    
    print(f"Original first action:    {interp_actions[0]}")
    print(f"After safety limits:      {safe_action}")
    print(f"Diff from original:       {safe_action - interp_actions[0]}")
    
    clipped = not np.allclose(safe_action, interp_actions[0], atol=1e-6)
    if clipped:
        print("\n⚠️  WARNING: Action was clipped by safety limits!")
    
    # 9. Test execution (dry run first)
    print("\n" + "=" * 60)
    print("Test Execution (Dry Run)")
    print("=" * 60)
    
    print("Executing first 10 interpolated actions...")
    for i in range(min(10, len(interp_actions))):
        action = interp_actions[i]
        safe_action = runner.apply_safety_limits(action, prev_action)
        
        # Print delta
        delta_from_prev = safe_action - prev_action
        print(f"Step {i}: delta={delta_from_prev[:3]} (joints 0-2)")
        
        prev_action = safe_action
    
    # 10. Ask user to execute
    print("\n" + "=" * 60)
    print("Execute on Robot?")
    print("=" * 60)
    
    response = input("Execute predicted actions on robot? (y/n): ").strip().lower()
    
    if response == 'y':
        import arx5_interface as arx5
        
        print("\nExecuting...")
        prev_action = np.concatenate([current_pos, [gripper_pos]])
        
        for i in range(min(100, len(interp_actions))):
            action = interp_actions[i]
            safe_action = runner.apply_safety_limits(action, prev_action)
            
            js = arx5.JointState(6)
            js.pos()[:] = safe_action[:6]
            js.gripper_pos = float(safe_action[6])
            
            runner.robot_ctrl.set_joint_cmd(js)
            runner.robot_ctrl.send_recv_once()
            
            prev_action = safe_action
            
            if i % 20 == 0:
                state = runner.robot_ctrl.get_state()
                actual_pos = np.array(state.pos())
                print(f"Step {i}: target={safe_action[:3]}, actual={actual_pos[:3]}")
            
            time.sleep(runner.controller_dt)
        
        print("Done!")
    
    # Cleanup
    runner.cleanup()


if __name__ == "__main__":
    sys.path.insert(0, "/home/lizh/LeRobot-Anything-U-Arm/src/uarm/scripts")
    main()
