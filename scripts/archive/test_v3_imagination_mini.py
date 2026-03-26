#!/usr/bin/env python3
"""V3: Imagination mini 验证 — 生成 5 条轨迹 + 保存 HDF5 + 验证格式.

Phase 1.5 V3 验证项:
  1. ImaginationEnvEngine 初始化
  2. 5 条轨迹生成 (mock WM + mock policy + real ManiSkill env.step())
  3. save_trajectories 写入 HDF5
  4. HDF5 格式验证(latent, actions, state, attrs)
  5. 帧可提取（模拟 VLM 读取）

Usage:
    CUDA_VISIBLE_DEVICES=9 python scripts/test_v3_imagination_mini.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import h5py
import numpy as np
import torch

from rlft.vlaw.world_model.imagination_env import (
    ImaginationEnvConfig,
    ImaginationEnvEngine,
    _MockCtrlWorldAdapter,
    _MockPolicy,
)

OUTPUT_DIR = "data/vlaw/synthetic/iter1_mini/LiftPegUpright-v1"


def main() -> None:
    task_id = "LiftPegUpright-v1"
    num_trajs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[V3] device={device}, task={task_id}, num_trajs={num_trajs}")

    # === Step 1: 初始化 ===
    print("\n[V3] Step 1: 初始化 ImaginationEnvEngine ...")
    cfg = ImaginationEnvConfig(
        num_envs=1,
        num_interact=3,
        act_steps=5,
        obs_horizon=2,
        task_id=task_id,
        tasks=[task_id],
        decode_for_policy=False,
        dry_run=False,  # 需要 save
        gpu_id=0,
        sim_backend="physx_cuda",
        camera_width=64,
        camera_height=64,
        output_dir=OUTPUT_DIR,
    )

    mock_wm = _MockCtrlWorldAdapter()
    mock_policy = _MockPolicy()
    engine = ImaginationEnvEngine(
        wm_adapter=mock_wm,
        policy=mock_policy,
        config=cfg,
    )
    print("[V3] ✅ Step 1: 初始化成功")

    # === Step 2: 生成 5 条轨迹 ===
    print(f"\n[V3] Step 2: 生成 {num_trajs} 条轨迹 ...")
    trajectories = []
    for i in range(num_trajs):
        t0 = time.time()
        init_lat = torch.randn(4, 48, 24, dtype=torch.float32)
        init_state = np.zeros(25, dtype=np.float32)  # LiftPeg state_dim=25

        traj = engine.rollout_single(
            initial_latent=init_lat,
            initial_state=init_state,
            instruction=f"complete {task_id}",
            task_id=task_id,
        )
        elapsed = time.time() - t0
        if traj is not None:
            trajectories.append(traj)
            print(f"  traj {i}: steps={traj.actions.shape[0]}, "
                  f"latents={traj.latents.shape}, time={elapsed:.1f}s ✅")
        else:
            print(f"  traj {i}: ❌ returned None")

    print(f"[V3] {'✅' if len(trajectories) == num_trajs else '❌'} "
          f"Step 2: {len(trajectories)}/{num_trajs} 条轨迹生成")

    if not trajectories:
        print("[V3] ❌ 无轨迹可保存，退出")
        sys.exit(1)

    # === Step 3: 保存 HDF5 ===
    print(f"\n[V3] Step 3: 保存 HDF5 到 {OUTPUT_DIR} ...")
    out_path = engine.save_trajectories(trajectories, OUTPUT_DIR)
    print(f"[V3] ✅ Step 3: HDF5 已保存: {out_path}")

    # === Step 4: 验证 HDF5 格式 ===
    print(f"\n[V3] Step 4: 验证 HDF5 格式 ...")
    with h5py.File(out_path, "r") as f:
        keys = list(f.keys())
        print(f"  顶层 keys: {keys}")

        # 检查 meta
        if "meta" in f:
            meta = f["meta"]
            print(f"  meta.attrs: {dict(meta.attrs)}")

        # 检查轨迹
        traj_keys = [k for k in keys if k.startswith("traj_")]
        print(f"  轨迹数: {len(traj_keys)}")

        for tk in traj_keys[:2]:  # 检查前 2 条
            grp = f[tk]
            datasets = list(grp.keys())
            attrs_dict = dict(grp.attrs)
            print(f"  {tk}:")
            print(f"    datasets: {datasets}")
            print(f"    attrs: {attrs_dict}")
            for ds_name in datasets:
                ds = grp[ds_name]
                print(f"    {ds_name}: shape={ds.shape}, dtype={ds.dtype}")

        all_ok = (
            len(traj_keys) == num_trajs
            and all("latent" in f[k] for k in traj_keys)
            and all("actions" in f[k] for k in traj_keys)
            and all("state" in f[k] for k in traj_keys)
        )
    print(f"[V3] {'✅' if all_ok else '❌'} Step 4: HDF5 格式{'正确' if all_ok else '异常'}")

    # === Step 5: 模拟 VLM 读取（提取帧信息） ===
    print(f"\n[V3] Step 5: 模拟 VLM 读取 ...")
    with h5py.File(out_path, "r") as f:
        for tk in traj_keys[:1]:
            grp = f[tk]
            latent = grp["latent"][:]
            actions = grp["actions"][:]
            state = grp["state"][:]
            instruction = grp.attrs.get("task_instruction", "N/A")
            print(f"  {tk}: latent={latent.shape}({latent.dtype}), "
                  f"actions={actions.shape}({actions.dtype}), "
                  f"state={state.shape}({state.dtype}), "
                  f"instruction='{instruction}'")
    print("[V3] ✅ Step 5: VLM 读取模拟成功")

    # === 汇总 ===
    print(f"\n{'='*60}")
    print(f"[V3] Phase 1.5 V3 Imagination Mini 验证结果:")
    print(f"  Step 1 初始化:    ✅")
    print(f"  Step 2 轨迹生成:  {'✅' if len(trajectories)==num_trajs else '❌'} ({len(trajectories)}/{num_trajs})")
    print(f"  Step 3 HDF5 保存: ✅ → {out_path}")
    print(f"  Step 4 格式验证:  {'✅' if all_ok else '❌'}")
    print(f"  Step 5 VLM 读取:  ✅")
    print(f"  总体状态: {'✅ 全部通过' if (len(trajectories)==num_trajs and all_ok) else '⚠️ 部分失败'}")


if __name__ == "__main__":
    main()
