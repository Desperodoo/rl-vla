"""从 pilot HDF5 数据中提取帧，保存为 GIF 视频供人工检查。

对每条轨迹保存：
1. 每帧拼接图 (base_camera | render_camera)
2. GIF 动画
3. 首帧/末帧/中间帧拼接 strip

用于验证 success_at_end=100% 是否真实。
"""
import os
import sys
import h5py
import numpy as np
from pathlib import Path
from PIL import Image


def save_gif(frames: list[np.ndarray], path: str, fps: int = 5):
    """保存为 GIF."""
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=1000 // fps, loop=0)


def save_strip(frames: list[np.ndarray], path: str, max_frames: int = 12):
    """等间隔采样拼接为横向长图."""
    n = len(frames)
    if n <= max_frames:
        selected = frames
    else:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        selected = [frames[i] for i in indices]
    
    strip = np.concatenate(selected, axis=1)
    Image.fromarray(strip).save(path)
    return len(selected)


def main():
    hdf5_path = "data/vlaw/rollouts/pilot/LiftPegUpright-v1/LiftPegUpright-v1_real_1772640761.h5"
    out_dir = Path("results/vlaw/debug_env_success")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(hdf5_path):
        print(f"HDF5 not found: {hdf5_path}")
        return
    
    f = h5py.File(hdf5_path, "r")
    keys = sorted([k for k in f.keys() if k.startswith("traj")])
    print(f"Total trajectories: {len(keys)}")
    print(f"Output dir: {out_dir}\n")
    
    # Process each trajectory
    for idx, k in enumerate(keys):
        grp = f[k]
        rgb_base = np.array(grp["rgb_base"])     # (T, H, W, 3)
        rgb_render = np.array(grp["rgb_render"])  # (T, H, W, 3) 
        env_success = np.array(grp["env_success"])
        actions = np.array(grp["actions"])
        T = len(env_success)
        
        suc_end = bool(env_success[-1])
        any_suc = env_success.any()
        first_suc = int(np.argmax(env_success)) if any_suc else -1
        
        # Camera diff (detect BUG-020 type issues)
        diff = np.abs(rgb_base.astype(float) - rgb_render.astype(float)).mean()
        
        # Action stats
        act_mean = actions.mean(axis=0)
        act_std = actions.std(axis=0)
        
        status = "✅" if suc_end else "❌"
        print(f"{k}: T={T} {status} suc_end={suc_end} first_suc_frame={first_suc} cam_diff={diff:.1f} act_std_mean={act_std.mean():.3f}")
        
        # Build frames: side-by-side base | render
        combined_frames = []
        for t in range(T):
            # Add border between views
            sep = np.ones((rgb_base.shape[1], 2, 3), dtype=np.uint8) * 200
            frame = np.concatenate([rgb_base[t], sep, rgb_render[t]], axis=1)
            
            # Add step number and success indicator
            # (simple: just annotate via a colored bar at top)
            bar = np.zeros((4, frame.shape[1], 3), dtype=np.uint8)
            if env_success[t]:
                bar[:, :, 1] = 255  # green
            else:
                bar[:, :, 0] = 255  # red
            frame = np.concatenate([bar, frame], axis=0)
            combined_frames.append(frame)
        
        # Save every 5th trajectory as GIF + all as strips
        if idx < 10 or idx % 5 == 0:
            gif_path = out_dir / f"{k}_T{T}_{status.strip()}.gif"
            save_gif(combined_frames, str(gif_path), fps=3)
        
        strip_path = out_dir / f"{k}_strip.png"
        n_strip = save_strip(combined_frames, str(strip_path), max_frames=min(T, 12))
    
    f.close()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"已保存 {len(keys)} 条轨迹的 strip 图 + 前 10 条的 GIF 到:")
    print(f"  {out_dir}/")
    print(f"\n检查要点:")
    print(f"  1. 查看首帧: peg 是否平躺在桌面上 (初始状态)")
    print(f"  2. 查看末帧: peg 是否竖直站立 (成功条件)")
    print(f"  3. 顶部颜色条: 红=失败帧, 绿=成功帧")
    print(f"  4. 左侧=base_camera, 右侧=render_camera")
    print(f"  5. 若所有末帧都显示 peg 竖直, 则 100% success 是真实的")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    main()
