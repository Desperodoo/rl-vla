"""调查 labeled 和 encoded HDF5 文件结构"""
import h5py
import numpy as np

# Check labeled files
for task in ['LiftPegUpright-v1', 'PickCube-v1', 'StackCube-v1']:
    import glob, os
    paths = glob.glob(f'/home/wjz/rl-vla/data/vlaw/labeled/iter1/{task}/*.h5')
    if not paths:
        print(f'{task}: labeled not found')
        continue
    path = paths[0]
    with h5py.File(path, 'r') as f:
        print(f'\nLabeled {task}:')
        print(f'  Top keys: {list(f.keys())[:5]}')
        tkeys = [k for k in f if k.startswith('traj')]
        print(f'  Traj count: {len(tkeys)}')
        if tkeys:
            t = f[tkeys[0]]
            print(f'  traj_0 keys: {list(t.keys())}')
            for k in t.keys():
                ds = t[k]
                arr = ds[:]
                print(f'    {k}: shape={arr.shape}, dtype={arr.dtype}, sample={arr[:3]}')
        if 'meta' in f:
            for k, v in f['meta'].attrs.items():
                print(f'  meta.attrs[{k}] = {v}')

print('\n--- Encoded demos ---')
for task in ['LiftPegUpright-v1', 'PickCube-v1', 'StackCube-v1']:
    import glob
    paths = glob.glob(f'/home/wjz/rl-vla/data/vlaw/encoded/demos/{task}/*.h5')
    if not paths:
        print(f'{task}: encoded demo not found')
        continue
    path = paths[0]
    with h5py.File(path, 'r') as f:
        print(f'\nEncoded demo {task}:')
        tkeys = sorted([k for k in f if k.startswith('traj')])
        print(f'  Traj count: {len(tkeys)}')
        if tkeys:
            t = f[tkeys[0]]
            print(f'  traj_0 keys: {list(t.keys())}')
            for k in t.keys():
                ds = t[k]
                print(f'    {k}: shape={ds.shape}, dtype={ds.dtype}')

print('\nDone!')
