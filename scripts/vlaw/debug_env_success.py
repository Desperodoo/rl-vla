"""Debug: check if LiftPegUpright-v1 early terminates on success,
and if info['success'] is per-step or cumulative.
Also generates video frames for visual inspection."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from pathlib import Path


def test_zero_action_termination():
    """Test with zero actions - should never succeed, run to max_episode_steps."""
    print("=" * 60)
    print("Test 1: Zero actions (should NOT succeed)")
    print("=" * 60)
    env = gym.make(
        "LiftPegUpright-v1",
        obs_mode="rgbd",
        render_mode="rgb_array",
        control_mode="pd_ee_delta_pose",
        max_episode_steps=200,
    )
    obs, info = env.reset(seed=42)
    for step in range(205):
        action = np.zeros(7, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        success = info.get("success", False)
        if step < 3 or step % 50 == 0 or terminated or truncated:
            print(f"  Step {step+1:3d}: term={terminated}, trunc={truncated}, suc={success}")
        if terminated or truncated:
            print(f"  >>> Episode ended at step {step+1} (term={terminated}, trunc={truncated})")
            break
    env.close()
    print()


def test_random_action_episodes():
    """Test several random-action episodes to see termination patterns."""
    print("=" * 60)
    print("Test 2: Random actions (3 episodes)")
    print("=" * 60)
    env = gym.make(
        "LiftPegUpright-v1",
        obs_mode="rgbd",
        render_mode="rgb_array",
        control_mode="pd_ee_delta_pose",
        max_episode_steps=200,
    )
    for ep in range(3):
        obs, info = env.reset(seed=ep * 100)
        success_steps = []
        for step in range(210):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            success = bool(info.get("success", False))
            if success:
                success_steps.append(step + 1)
            if terminated or truncated:
                print(f"  Ep {ep}: ended at step {step+1}, term={terminated}, trunc={truncated}, "
                      f"success_at_end={success}, success_steps={success_steps[:5]}{'...' if len(success_steps) > 5 else ''}")
                break
    env.close()
    print()


def test_policy_with_video():
    """Run AWSC policy and save first/last frames as images for visual inspection."""
    print("=" * 60)
    print("Test 3: AWSC policy - 5 episodes with frame saves")
    print("=" * 60)
    
    # Use GPU vec env (same as collector)
    env = gym.make(
        "LiftPegUpright-v1",
        obs_mode="rgbd",
        render_mode="rgb_array",
        control_mode="pd_ee_delta_pose",
        max_episode_steps=200,
        num_envs=5,
    )
    
    # Load AWSC policy
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from rlft.utils.flow_wrapper import load_shortcut_flow_policy
    
    ckpt_path = "runs/fair_comparison/fair_comparison/awsc/best_s42__1772570560/checkpoints/final.pt"
    if not os.path.exists(ckpt_path):
        print(f"  AWSC checkpoint not found: {ckpt_path}")
        env.close()
        return
    
    device = torch.device("cuda")
    flow_wrapper, visual_encoder, state_dim = load_shortcut_flow_policy(ckpt_path, device)
    flow_wrapper.eval()
    if visual_encoder is not None:
        visual_encoder.eval()
    
    print(f"  Policy loaded, state_dim={state_dim}")
    
    obs, info = env.reset(seed=42)
    N = 5
    
    # Build obs features helper (simplified version)
    obs_horizon = 2
    
    # Initialize history
    if isinstance(obs, dict) and "state" in obs and "rgb" in obs:
        agent_state = obs["state"].cpu().numpy() if isinstance(obs["state"], torch.Tensor) else np.array(obs["state"])
        rgb_np = obs["rgb"].cpu().numpy() if isinstance(obs["rgb"], torch.Tensor) else np.array(obs["rgb"])
    else:
        raise ValueError(f"Unexpected obs format: {type(obs)}")
    
    print(f"  agent_state.shape={agent_state.shape}, rgb.shape={rgb_np.shape}")
    
    # Track per-env info
    episode_success_history = {i: [] for i in range(N)}
    episode_terminated = [False] * N
    episode_frames_first = {}
    episode_frames_last = {}
    episode_step_counts = [0] * N
    
    # Save initial frames  
    out_dir = Path("results/vlaw/debug_env_success")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    from PIL import Image
    
    # Get render frames
    render_frames = env.render()
    if isinstance(render_frames, torch.Tensor):
        render_frames = render_frames.cpu().numpy()
    
    for i in range(N):
        episode_frames_first[i] = render_frames[i] if render_frames.ndim == 4 else render_frames
    
    # Run for max steps
    for step in range(200):
        # Simple: use random actions since we just want to check env behavior
        # Actually let's use the AWSC policy properly
        # For simplicity, let's use random actions first
        actions = torch.from_numpy(env.action_space.sample()).float()
        if actions.dim() == 1:
            actions = actions.unsqueeze(0).expand(N, -1)
        
        obs, reward, terminated, truncated, info = env.step(actions)
        
        if isinstance(terminated, torch.Tensor):
            terminated_np = terminated.cpu().numpy()
        else:
            terminated_np = np.array(terminated)
        if isinstance(truncated, torch.Tensor):
            truncated_np = truncated.cpu().numpy()
        else:
            truncated_np = np.array(truncated)
        done = terminated_np | truncated_np
        
        success = info.get("success", None)
        if success is not None:
            if isinstance(success, torch.Tensor):
                success_np = success.cpu().numpy()
            else:
                success_np = np.array(success)
        else:
            success_np = np.zeros(N, dtype=bool)
        
        for i in range(N):
            if not episode_terminated[i]:
                episode_step_counts[i] += 1
                episode_success_history[i].append(bool(success_np[i]))
                if done[i]:
                    episode_terminated[i] = True
                    render_frames = env.render()
                    if isinstance(render_frames, torch.Tensor):
                        render_frames = render_frames.cpu().numpy()
                    episode_frames_last[i] = render_frames[i] if render_frames.ndim == 4 else render_frames
                    
                    sh = episode_success_history[i]
                    any_suc = any(sh)
                    first_suc = sh.index(True) + 1 if any_suc else -1
                    print(f"  Env {i}: ended at step {step+1}, term={bool(terminated_np[i])}, "
                          f"trunc={bool(truncated_np[i])}, success_at_end={bool(success_np[i])}, "
                          f"any_success={any_suc}, first_success_step={first_suc}, "
                          f"total_steps={episode_step_counts[i]}")
        
        if all(episode_terminated):
            break
    
    # Save frames
    for i in range(N):
        if i in episode_frames_first:
            first_frame = episode_frames_first[i]
            if first_frame.ndim == 3:
                img = Image.fromarray(first_frame.astype(np.uint8))
                img.save(out_dir / f"env{i}_first.png")
        if i in episode_frames_last:
            last_frame = episode_frames_last[i]
            if last_frame.ndim == 3:
                img = Image.fromarray(last_frame.astype(np.uint8))
                img.save(out_dir / f"env{i}_last.png")
    
    print(f"\n  Frames saved to {out_dir}/")
    env.close()
    print()


def test_success_stickiness_gpu_vec():
    """Key test: Check if info['success'] is sticky in GPU vec env."""
    print("=" * 60)
    print("Test 4: GPU vec env - success stickiness test")
    print("=" * 60)
    
    env = gym.make(
        "LiftPegUpright-v1",
        obs_mode="rgbd",
        render_mode="rgb_array",
        control_mode="pd_ee_delta_pose",
        max_episode_steps=200,
        num_envs=64,
    )
    obs, info = env.reset(seed=42)
    
    N = 64
    first_success_step = [-1] * N
    last_success_step = [-1] * N
    end_step = [-1] * N
    end_terminated = [None] * N
    end_truncated = [None] * N
    end_success = [None] * N
    success_count = [0] * N
    ended = [False] * N
    
    for step in range(210):
        actions = torch.from_numpy(env.action_space.sample()).float()
        obs, reward, terminated, truncated, info = env.step(actions)
        
        term_np = terminated.cpu().numpy() if isinstance(terminated, torch.Tensor) else np.array(terminated)
        trunc_np = truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else np.array(truncated)
        done = term_np | trunc_np
        
        suc = info.get("success", torch.zeros(N))
        suc_np = suc.cpu().numpy() if isinstance(suc, torch.Tensor) else np.array(suc)
        
        for i in range(N):
            if ended[i]:
                continue
            if bool(suc_np[i]):
                if first_success_step[i] == -1:
                    first_success_step[i] = step + 1
                last_success_step[i] = step + 1
                success_count[i] += 1
            if done[i]:
                ended[i] = True
                end_step[i] = step + 1
                end_terminated[i] = bool(term_np[i])
                end_truncated[i] = bool(trunc_np[i])
                end_success[i] = bool(suc_np[i])
        
        if all(ended):
            break
    
    # Analysis
    term_count = sum(1 for x in end_terminated if x)
    trunc_count = sum(1 for x in end_truncated if x)
    suc_at_end_count = sum(1 for x in end_success if x)
    any_suc_count = sum(1 for x in first_success_step if x != -1)
    
    print(f"  Terminated: {term_count}/{N}, Truncated: {trunc_count}/{N}")
    print(f"  Success_at_end: {suc_at_end_count}/{N} = {suc_at_end_count/N:.1%}")
    print(f"  Any_success (success_once): {any_suc_count}/{N} = {any_suc_count/N:.1%}")
    
    # Check stickiness: if any env has success going True→False→True etc
    sticky_check = [(first_success_step[i], last_success_step[i], success_count[i], end_step[i]) 
                    for i in range(N) if first_success_step[i] != -1]
    
    print(f"\n  Success envs details (first 10):")
    for i in range(N):
        if first_success_step[i] != -1 and i < 10:
            duration = last_success_step[i] - first_success_step[i] + 1
            print(f"    Env {i}: first_suc={first_success_step[i]}, last_suc={last_success_step[i]}, "
                  f"suc_count={success_count[i]}, total_steps={end_step[i]}, "
                  f"term={end_terminated[i]}, trunc={end_truncated[i]}, "
                  f"suc_at_end={end_success[i]}, "
                  f"{'STICKY' if success_count[i] == end_step[i] - first_success_step[i] + 1 else 'NOT_STICKY'}")
    
    # Check: terminated envs - is success at end always True?
    term_suc = [(i, end_success[i], first_success_step[i]) 
                for i in range(N) if end_terminated[i]]
    if term_suc:
        all_term_suc = all(s for _, s, _ in term_suc)
        print(f"\n  All terminated envs have success_at_end=True? {all_term_suc}")
        if not all_term_suc:
            for i, s, fs in term_suc[:5]:
                if not s:
                    print(f"    Env {i}: terminated but success_at_end=False! first_suc={fs}")
    
    # Check: truncated envs - success at end distribution
    trunc_suc = [(i, end_success[i]) for i in range(N) if end_truncated[i]]
    if trunc_suc:
        trunc_suc_count = sum(1 for _, s in trunc_suc if s)
        print(f"\n  Truncated envs: {trunc_suc_count}/{len(trunc_suc)} have success_at_end=True")
    
    env.close()
    print()


if __name__ == "__main__":
    test_zero_action_termination()
    test_success_stickiness_gpu_vec()
