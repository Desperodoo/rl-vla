"""生成 AWSC policy 的 rollout 视频用于人工检查。

使用环境渲染（非 rgb obs），保存完整 episode 的每一帧为视频。
5 个 episode，每个独立 seed，记录 terminated/truncated/success 时序。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import sys
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def save_video_from_frames(frames: list[np.ndarray], path: str, fps: int = 10):
    """用 PIL 保存为 GIF (不需要 ffmpeg)."""
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=1000 // fps, loop=0)
    print(f"  Saved video: {path} ({len(frames)} frames)")


def save_strip(frames: list[np.ndarray], path: str, max_frames: int = 20):
    """保存等间隔采样的拼接长图."""
    n = len(frames)
    if n <= max_frames:
        selected = frames
    else:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        selected = [frames[i] for i in indices]
    
    strip = np.concatenate(selected, axis=1)  # horizontal concat
    Image.fromarray(strip).save(path)
    print(f"  Saved strip: {path} ({len(selected)} frames, {strip.shape[1]}x{strip.shape[0]})")


def main():
    out_dir = Path("results/vlaw/debug_env_success")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Load AWSC policy ----
    from rlft.utils.flow_wrapper import load_shortcut_flow_policy
    
    ckpt_path = "runs/fair_comparison/fair_comparison/awsc/best_s42__1772570560/checkpoints/final.pt"
    device = torch.device("cuda")
    
    # Get config from checkpoint first
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_config = ckpt.get("config", {})
    obs_horizon = ckpt_config.get("obs_horizon", 2)
    pred_horizon = ckpt_config.get("pred_horizon", 8)
    act_steps = min(ckpt_config.get("act_steps", 8), pred_horizon)
    print(f"Policy config: obs_horizon={obs_horizon}, pred_horizon={pred_horizon}, act_steps={act_steps}")
    del ckpt
    
    flow_wrapper, visual_encoder, state_dim = load_shortcut_flow_policy(
        ckpt_path, device=str(device), pred_horizon=pred_horizon, obs_horizon=obs_horizon,
    )
    # flow_wrapper is not nn.Module; velocity_net already in eval mode
    if visual_encoder is not None:
        visual_encoder.eval()
    print(f"State dim={state_dim}, action_dim={flow_wrapper.action_dim}")
    
    # ---- Run 5 independent episodes (single env, CPU physics for cleaner video) ----
    num_episodes = 10
    max_steps = 200
    
    summary = []
    
    for ep_idx in range(num_episodes):
        seed = ep_idx * 1000 + 42
        
        # Create fresh env each time (single env, CPU)
        env = gym.make(
            "LiftPegUpright-v1",
            obs_mode="rgbd",
            render_mode="rgb_array",
            control_mode="pd_ee_delta_pose",
            max_episode_steps=max_steps,
        )
        obs, info = env.reset(seed=seed)
        
        frames = []
        success_history = []
        
        # Obs history buffers
        H, W = 128, 128
        
        # Extract initial obs
        if isinstance(obs, dict):
            from mani_skill.utils.common import flatten_dict_keys
            flat_obs = flatten_dict_keys(obs)
            # Find rgb
            rgb_key = None
            for k in flat_obs:
                if "base_camera" in k and "rgb" in k:
                    rgb_key = k
                    break
            if rgb_key:
                initial_rgb = flat_obs[rgb_key]
            else:
                initial_rgb = None
            # Find state (agent qpos + qvel)
            state_keys = [k for k in flat_obs if "qpos" in k or "qvel" in k]
            
            # Actually for simplicity, let's use the flattened RGBD obs
            pass
        
        # Use render for video frames
        render_frame = env.render()
        if isinstance(render_frame, torch.Tensor):
            render_frame = render_frame.cpu().numpy()
        if render_frame.ndim == 4:
            render_frame = render_frame[0]  # single env
        frames.append(render_frame.copy())
        
        # Build obs features similar to collector
        # For the policy, we need (B, obs_horizon * feat_dim)
        # This is complex; let's build it properly
        
        # Simplified: extract state and RGB, build obs_cond
        def extract_obs(obs_dict):
            """Extract agent_state and rgb from obs dict."""
            if isinstance(obs_dict, dict):
                if "state" in obs_dict:
                    state = obs_dict["state"]
                    rgb = obs_dict.get("rgb", None)
                elif "extra" in obs_dict:
                    # Nested dict format
                    agent = obs_dict.get("agent", {})
                    qpos = agent.get("qpos", None)
                    qvel = agent.get("qvel", None)
                    if qpos is not None and qvel is not None:
                        if isinstance(qpos, torch.Tensor):
                            state = torch.cat([qpos, qvel], dim=-1)
                        else:
                            state = np.concatenate([qpos, qvel], axis=-1)
                    else:
                        state = None
                    
                    sensor = obs_dict.get("sensor_data", {})
                    cam = sensor.get("base_camera", {})
                    rgb = cam.get("rgb", None)
                else:
                    state = None
                    rgb = None
            else:
                state = obs_dict
                rgb = None
            
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            if state is not None and state.ndim == 1:
                state = state[np.newaxis]  # (1, D)
            if rgb is not None:
                if rgb.ndim == 3:
                    rgb = rgb[np.newaxis]  # (1, H, W, C)
                rgb = rgb.astype(np.uint8)
            return state, rgb
        
        state, rgb = extract_obs(obs)
        if state is None:
            print(f"  Ep {ep_idx}: Cannot extract state from obs, skipping.")
            env.close()
            continue
        
        state_dim_actual = state.shape[-1]
        
        # Build obs history
        state_hist = np.tile(state, (1, obs_horizon))  # (1, obs_horizon * state_dim)
        if rgb is not None:
            rgb_hist = np.tile(rgb, (1, obs_horizon, 1, 1))  # (1, obs_horizon, H, W, C)
        else:
            rgb_hist = None
        
        def build_obs_features(state_hist, rgb_hist, visual_encoder, device):
            """Build obs_cond for the policy."""
            B = state_hist.shape[0]
            state_t = torch.from_numpy(state_hist).float().to(device)
            
            if visual_encoder is not None and rgb_hist is not None:
                # (B, obs_horizon, H, W, C) -> process each timestep
                oh = rgb_hist.shape[1]
                vis_feats = []
                for t in range(oh):
                    img = rgb_hist[:, t]  # (B, H, W, C)
                    img_t = torch.from_numpy(img).float().to(device)
                    img_t = img_t.permute(0, 3, 1, 2) / 255.0  # (B, C, H, W)
                    feat = visual_encoder(img_t)  # (B, feat_dim)
                    vis_feats.append(feat)
                vis_feat = torch.cat(vis_feats, dim=-1)  # (B, obs_horizon * feat_dim)
                obs_cond = torch.cat([vis_feat, state_t], dim=-1)
            else:
                obs_cond = state_t
            
            return obs_cond
        
        # Action buffer for chunk execution  
        action_buffer = None
        action_buffer_idx = 0
        
        terminated_at = None
        truncated_at = None
        success_at_end = False
        
        for step in range(max_steps):
            # Get action from policy (with chunked execution)
            if action_buffer is None or action_buffer_idx >= action_buffer.shape[1]:
                obs_cond = build_obs_features(state_hist, rgb_hist, visual_encoder, device)
                noise = torch.randn(1, pred_horizon, flow_wrapper.action_dim, device=device)
                actions = flow_wrapper(
                    obs=obs_cond,
                    initial_noise=noise,
                    return_numpy=True,
                    act_steps=act_steps,
                )  # (1, act_steps, action_dim)
                action_buffer = actions
                action_buffer_idx = 0
            
            action = action_buffer[0, action_buffer_idx]  # (action_dim,)
            action_buffer_idx += 1
            
            # Step env
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Handle tensors
            if isinstance(terminated, torch.Tensor):
                terminated = terminated.item()
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.item()
            success = info.get("success", False)
            if isinstance(success, torch.Tensor):
                success = success.item()
            
            success_history.append(bool(success))
            
            # Save render frame
            render_frame = env.render()
            if isinstance(render_frame, torch.Tensor):
                render_frame = render_frame.cpu().numpy()
            if render_frame.ndim == 4:
                render_frame = render_frame[0]
            frames.append(render_frame.copy())
            
            # Update obs history
            state, rgb = extract_obs(obs)
            if state is not None:
                state_hist = np.concatenate([
                    state_hist[:, state_dim_actual:],
                    state
                ], axis=-1)
            if rgb is not None and rgb_hist is not None:
                rgb_hist = np.concatenate([
                    rgb_hist[:, 1:],
                    rgb[:, np.newaxis]
                ], axis=1)
            
            if terminated:
                terminated_at = step + 1
                success_at_end = bool(success)
                break
            if truncated:
                truncated_at = step + 1
                success_at_end = bool(success)
                break
        
        env.close()
        
        # Report
        any_suc = any(success_history)
        first_suc = success_history.index(True) + 1 if any_suc else -1
        status = "✅ SUCCESS" if success_at_end else "❌ FAIL"
        end_type = "terminated" if terminated_at else ("truncated" if truncated_at else "timeout")
        end_step = terminated_at or truncated_at or max_steps
        
        print(f"Ep {ep_idx} (seed={seed}): {status} | {end_type} at step {end_step} | "
              f"first_success_step={first_suc} | frames={len(frames)}")
        
        summary.append({
            "ep": ep_idx, "seed": seed, "success": success_at_end,
            "end_type": end_type, "end_step": end_step,
            "first_success": first_suc, "total_frames": len(frames),
        })
        
        # Save video and strip
        if frames:
            save_video_from_frames(frames, str(out_dir / f"ep{ep_idx}_{status.split()[0]}_step{end_step}.gif"), fps=10)
            save_strip(frames, str(out_dir / f"ep{ep_idx}_strip.png"), max_frames=16)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(summary)
    suc_count = sum(1 for s in summary if s["success"])
    print(f"Success: {suc_count}/{total} = {suc_count/total:.1%}")
    for s in summary:
        status = "✅" if s["success"] else "❌"
        print(f"  Ep {s['ep']}: {status} {s['end_type']}@{s['end_step']} first_suc={s['first_success']}")
    
    print(f"\nOutput dir: {out_dir}")
    print("请检查 GIF 视频和 strip 图片确认 peg 是否真的被扶正。")


if __name__ == "__main__":
    main()
