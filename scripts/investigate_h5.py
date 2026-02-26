"""调查 HDF5 数据集结构"""
import h5py
import numpy as np
import os

files = {
    'demos/LiftPegUpright-v1': '/home/wjz/rl-vla/data/vlaw/demos/LiftPegUpright-v1/LiftPegUpright-v1_demo_1771951465.h5',
    'demos/PickCube-v1': '/home/wjz/rl-vla/data/vlaw/demos/PickCube-v1/PickCube-v1_demo_1771999122.h5',
    'demos/StackCube-v1': '/home/wjz/rl-vla/data/vlaw/demos/StackCube-v1/StackCube-v1_demo_1771999343.h5',
    'rollouts/iter1/LiftPegUpright-v1': '/home/wjz/rl-vla/data/vlaw/rollouts/iter1/LiftPegUpright-v1/LiftPegUpright-v1_real_1772017887.h5',
    'rollouts/iter1/PickCube-v1': '/home/wjz/rl-vla/data/vlaw/rollouts/iter1/PickCube-v1/PickCube-v1_real_1772017934.h5',
    'rollouts/iter1/StackCube-v1': '/home/wjz/rl-vla/data/vlaw/rollouts/iter1/StackCube-v1/StackCube-v1_real_1772018007.h5',
    'labeled/iter1/LiftPegUpright-v1': '/home/wjz/rl-vla/data/vlaw/labeled/iter1/LiftPegUpright-v1/LiftPegUpright-v1_vlm_rewards.h5',
    'labeled/iter1/PickCube-v1': '/home/wjz/rl-vla/data/vlaw/labeled/iter1/PickCube-v1/PickCube-v1_vlm_rewards.h5',
    'labeled/iter1/StackCube-v1': '/home/wjz/rl-vla/data/vlaw/labeled/iter1/StackCube-v1/StackCube-v1_vlm_rewards.h5',
}

def print_h5_structure(name, f, depth=0):
    prefix = "  " * depth
    if isinstance(f, h5py.Dataset):
        print(f"{prefix}[DS] {name}: shape={f.shape}, dtype={f.dtype}")
    elif isinstance(f, h5py.Group):
        print(f"{prefix}[GRP] {name}/")
        for k in f.keys():
            print_h5_structure(k, f[k], depth + 1)

for label, path in files.items():
    print(f"\n{'='*60}")
    print(f"FILE: {label}")
    print(f"PATH: {path}")
    if not os.path.exists(path):
        print("  FILE NOT FOUND")
        continue
    with h5py.File(path, 'r') as f:
        print(f"Top-level keys: {list(f.keys())}")
        # print meta if exists
        if 'meta' in f:
            print("\nMeta attrs:")
            for k, v in f['meta'].attrs.items():
                print(f"  meta.attrs[{k}] = {v}")
            print("Meta datasets:")
            for k in f['meta'].keys():
                ds = f['meta'][k]
                print(f"  meta[{k}]: shape={ds.shape}, dtype={ds.dtype}, val={ds[:]}")
        
        # find traj groups
        traj_keys = [k for k in f.keys() if k.startswith('traj')]
        print(f"\nTrajectory count: {len(traj_keys)}")
        if traj_keys:
            # show structure of first traj
            traj0 = traj_keys[0]
            print(f"\nStructure of {traj0}:")
            print_h5_structure(traj0, f[traj0])
            # show shapes of all fields
            print(f"\nField shapes in {traj0}:")
            def show_fields(g, prefix=""):
                for k in g.keys():
                    item = g[k]
                    if isinstance(item, h5py.Dataset):
                        print(f"  {prefix}{k}: shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, h5py.Group):
                        show_fields(item, prefix + k + "/")
            show_fields(f[traj0])

print("\nDone!")
