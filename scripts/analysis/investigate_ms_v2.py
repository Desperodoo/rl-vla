"""ManiSkill 环境调查脚本 v2 - 使用绝对路径"""
import h5py
import numpy as np
from pathlib import Path

BASE = Path("/home/wjz/rl-vla")
tasks = ["LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"]
task_stats = {}

print("=" * 60)
print("步骤 1: HDF5 数据结构调查")
print("=" * 60)

for task in tasks:
    base_path = BASE / "data/vlaw/rollouts/iter1" / task
    h5files = sorted(base_path.glob("*.h5"))
    if not h5files:
        print(f"\n{task}: 无 HDF5 文件 (路径={base_path})")
        continue

    print(f"\n{'='*40}")
    print(f"任务: {task}")
    print(f"文件: {h5files[0].name}")

    with h5py.File(h5files[0], "r") as f:
        trajs = sorted([k for k in f if k.startswith("traj_")])
        print(f"轨迹数量: {len(trajs)}")

        g = f[trajs[0]]
        print(f"\n[第一条轨迹字段]")
        for k in sorted(g.keys()):
            arr = g[k]
            if hasattr(arr, "shape"):
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"  {k}: (group)")

        # success 分析
        if "env_success" in g:
            succ0 = g["env_success"][:]
            print(f"\n[Success 分析 - 第一条轨迹]")
            print(f"  env_success dtype: {succ0.dtype}, shape: {succ0.shape}")
            print(f"  值域: [{succ0.min()}, {succ0.max()}]")
            print(f"  success_once: {bool(np.any(succ0))}")
            print(f"  success_at_end: {bool(succ0[-1])}")

        # 全量统计
        once_count = 0
        end_count = 0
        traj_lengths = []
        for t in trajs:
            if "env_success" in f[t]:
                succ = f[t]["env_success"][:]
                traj_lengths.append(len(succ))
                if np.any(succ):
                    once_count += 1
                if succ[-1]:
                    end_count += 1

        if traj_lengths:
            print(f"\n[全量统计 - {len(trajs)} 条轨迹]")
            print(f"  success_once: {once_count}/{len(trajs)}")
            print(f"  success_at_end: {end_count}/{len(trajs)}")
            print(f"  轨迹长度: min={min(traj_lengths)}, max={max(traj_lengths)}, mean={np.mean(traj_lengths):.1f}")

        # state key
        state_key = "state" if "state" in g else ("obs_agent" if "obs_agent" in g else None)
        state_dim = g[state_key].shape[-1] if state_key else None
        act_dim = g["actions"].shape[-1] if "actions" in g else None

        task_stats[task] = {
            "n_trajs": len(trajs),
            "state_key": state_key,
            "state_dim": state_dim,
            "act_dim": act_dim,
            "success_once": once_count,
            "success_at_end": end_count,
            "traj_len_mean": float(np.mean(traj_lengths)) if traj_lengths else 0,
        }

# ===== 步骤 2: 收集 data_collector 相机信息 =====
print("\n" + "=" * 60)
print("步骤 2: data_collector 相机信息")
print("=" * 60)
collector_file = BASE / "rlft/vlaw/data_collector.py"
if collector_file.exists():
    lines = collector_file.read_text().split("\n")
    for i, line in enumerate(lines):
        if "camera" in line.lower() or "rgb" in line.lower():
            print(f"  L{i+1}: {line.rstrip()}")
else:
    print("  data_collector.py 未找到")

# ===== 汇总 =====
print("\n" + "=" * 60)
print("汇总 - task_stats")
print("=" * 60)
for task, stats in task_stats.items():
    print(f"\n{task}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
