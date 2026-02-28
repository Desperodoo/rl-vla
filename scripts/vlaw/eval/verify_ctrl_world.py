#!/usr/bin/env python3
"""
P0.1 验证脚本: 运行 Ctrl-World 推理并记录峰值显存
"""
import subprocess
import sys
import os
import time
import threading

os.chdir("/home/wjz/rl-vla/ctrl_world")

peak_vram = [0]

def monitor_vram():
    while not done_flag[0]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            val = int(result.stdout.strip().split()[0])
            if val > peak_vram[0]:
                peak_vram[0] = val
        time.sleep(1)

done_flag = [False]
monitor_thread = threading.Thread(target=monitor_vram, daemon=True)
monitor_thread.start()

print("=== 运行 Ctrl-World 推理 ===")
t0 = time.time()
proc = subprocess.run(
    [
        "conda", "run", "-n", "ctrl_world", "python",
        "scripts/rollout_replay_traj.py",
        "--dataset_root_path", "dataset_example",
        "--dataset_meta_info_path", "dataset_meta_info",
        "--dataset_names", "droid_subset",
        "--svd_model_path", "../checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid",
        "--clip_model_path", "../checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32",
        "--ckpt_path", "../checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt",
    ],
    capture_output=True, text=True
)
elapsed = time.time() - t0
done_flag[0] = True
monitor_thread.join(timeout=2)

print(f"推理耗时: {elapsed:.1f}s")
print(f"峰值显存: {peak_vram[0]} MiB / 24564 MiB ({peak_vram[0]/24564*100:.1f}%)")
print(f"返回码: {proc.returncode}")
print("\n--- 最后20行 stdout ---")
lines = proc.stdout.strip().split('\n')
print('\n'.join(lines[-20:]))
if proc.stderr:
    print("\n--- 最后10行 stderr ---")
    err_lines = proc.stderr.strip().split('\n')
    print('\n'.join(err_lines[-10:]))

# 检查输出文件
import glob
mp4s = glob.glob("synthetic_traj/**/*.mp4", recursive=True)
print(f"\n生成视频数: {len(mp4s)}")
for f in mp4s[:5]:
    size = os.path.getsize(f)
    print(f"  {f} ({size/1024:.1f} KB)")
