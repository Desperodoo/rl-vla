"""ManiSkill 环境调查脚本 - Data-Agent 使用"""
import h5py
import numpy as np
from pathlib import Path

# ===== 步骤 1: HDF5 数据结构调查 =====
print("=" * 60)
print("步骤 1: HDF5 数据结构调查")
print("=" * 60)

tasks = ["LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"]
task_stats = {}

for task in tasks:
    base_path = Path(f"data/vlaw/rollouts/iter1/{task}")
    h5files = sorted(base_path.glob("*.h5"))
    if not h5files:
        print(f"\n{task}: 无 HDF5 文件")
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
                if hasattr(arr, "keys"):
                    for kk in arr.keys():
                        print(f"    {kk}: shape={arr[kk].shape}, dtype={arr[kk].dtype}")

        # success 分析
        succ0 = g["env_success"][:]
        print(f"\n[Success 分析 - 第一条轨迹]")
        print(f"  env_success dtype: {succ0.dtype}, shape: {succ0.shape}")
        print(f"  值域: [{succ0.min()}, {succ0.max()}]")
        print(f"  success_once (任意步): {bool(np.any(succ0))}")
        print(f"  success_at_end (最后步): {bool(succ0[-1])}")

        # 全量统计
        once_count = 0
        end_count = 0
        traj_lengths = []
        for t in trajs:
            succ = f[t]["env_success"][:]
            traj_lengths.append(len(succ))
            if np.any(succ):
                once_count += 1
            if succ[-1]:
                end_count += 1

        print(f"\n[全量统计 - {len(trajs)} 条轨迹]")
        print(f"  success_once: {once_count}/{len(trajs)}")
        print(f"  success_at_end: {end_count}/{len(trajs)}")
        print(f"  轨迹长度: min={min(traj_lengths)}, max={max(traj_lengths)}, mean={np.mean(traj_lengths):.1f}")

        # state_dim 等关键维度
        state_key = "state" if "state" in g else ("obs_agent" if "obs_agent" in g else None)
        state_dim = None
        if state_key:
            state_dim = g[state_key].shape[-1]

        act_dim = g["actions"].shape[-1] if "actions" in g else None

        task_stats[task] = {
            "n_trajs": len(trajs),
            "state_dim": state_dim,
            "state_key": state_key,
            "act_dim": act_dim,
            "success_once": once_count,
            "success_at_end": end_count,
            "traj_len_mean": float(np.mean(traj_lengths)),
        }

# ===== 步骤 2: 尝试启动 ManiSkill env =====
print("\n" + "=" * 60)
print("步骤 2: ManiSkill obs 结构查询")
print("=" * 60)

env_stats = {}
try:
    import gymnasium as gym
    import mani_skill.envs  # noqa

    for task in tasks:
        try:
            env = gym.make(task, obs_mode="state", render_mode=None)
            obs, _ = env.reset()
            print(f"\n{task}:")
            if isinstance(obs, dict):
                print(f"  obs keys: {list(obs.keys())}")
                agent_info = {}
                if "agent" in obs:
                    for k, v in obs["agent"].items():
                        shape = v.shape if hasattr(v, "shape") else "?"
                        print(f"  agent.{k}: {shape}")
                        agent_info[k] = list(shape)
                if "extra" in obs:
                    for k, v in obs["extra"].items():
                        shape = v.shape if hasattr(v, "shape") else "?"
                        print(f"  extra.{k}: {shape}")
            else:
                print(f"  obs shape: {obs.shape}")
            # action space
            act_space = env.action_space
            print(f"  action_space: {act_space}")
            env.close()
            env_stats[task] = {"agent": agent_info}
        except Exception as e:
            print(f"  FAILED: {e}")
except ImportError as e:
    print(f"mani_skill 导入失败: {e}")

# ===== 步骤 3: success 源码查询 =====
print("\n" + "=" * 60)
print("步骤 3: evaluate() 源码")
print("=" * 60)

task_class_map = {
    "LiftPegUpright-v1": ("mani_skill.envs.tasks.tabletop.lift_peg_upright", "LiftPegUprightEnv"),
    "PickCube-v1": ("mani_skill.envs.tasks.tabletop.pick_cube", "PickCubeEnv"),
    "StackCube-v1": ("mani_skill.envs.tasks.tabletop.stack_cube", "StackCubeEnv"),
}

import importlib
import inspect

for task, (module_path, class_name) in task_class_map.items():
    print(f"\n--- {task} ({class_name}.evaluate) ---")
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        src = inspect.getsource(cls.evaluate)
        # 只打印前 40 行
        lines = src.split("\n")[:40]
        print("\n".join(lines))
    except Exception as e:
        print(f"  FAILED: {e}")

# ===== 汇总 =====
print("\n" + "=" * 60)
print("汇总")
print("=" * 60)
for task, stats in task_stats.items():
    print(f"\n{task}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
