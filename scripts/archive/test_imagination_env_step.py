#!/usr/bin/env python3
"""验证 ImaginationEnvEngine 的 env.step() 模式能否独立运行（无需真实 WM）.

使用 _MockCtrlWorldAdapter (随机 latent) + _MockPolicy (零动作) + 真实 ManiSkill env。
运行 5 条轨迹，报告轨迹长度和数据格式。

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/test_imagination_env_step.py
"""

from __future__ import annotations

import os
import sys
import time

# 确保项目根目录在 path 中
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch

from rlft.vlaw.world_model.imagination_env import (
    ImaginationEnvConfig,
    ImaginationEnvEngine,
    _MockCtrlWorldAdapter,
    _MockPolicy,
)


def main() -> None:
    num_trajs = 5
    task_id = "LiftPegUpright-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TEST] device={device}, task={task_id}, num_trajs={num_trajs}")

    # --- 配置 ---
    cfg = ImaginationEnvConfig(
        num_envs=1,          # 逐条测试
        num_interact=3,      # 减少交互轮数加速验证（原 12）
        act_steps=5,
        obs_horizon=2,
        task_id=task_id,
        tasks=[task_id],
        decode_for_policy=False,  # mock WM 解码无意义
        dry_run=True,
        gpu_id=0,  # CUDA_VISIBLE_DEVICES 已重映射，始终用 0
        # 使用 CPU 后端避免 GPU 内存冲突（physx_cuda 需要 GPU 资源）
        sim_backend="physx_cuda",
        camera_width=64,     # 小分辨率加速
        camera_height=64,
    )

    # --- Mock 组件 ---
    mock_wm = _MockCtrlWorldAdapter()
    mock_policy = _MockPolicy()

    engine = ImaginationEnvEngine(
        wm_adapter=mock_wm,   # type: ignore[arg-type]
        policy=mock_policy,
        config=cfg,
    )

    # --- 运行 5 条轨迹 ---
    results = []
    for i in range(num_trajs):
        print(f"\n{'='*60}")
        print(f"[TEST] 轨迹 {i+1}/{num_trajs}")
        t0 = time.time()

        # 随机初始 latent 和 state
        init_lat = torch.randn(4, 48, 24, dtype=torch.float32)
        init_state = np.zeros(29, dtype=np.float32)

        traj = engine.rollout_single(
            initial_latent=init_lat,
            initial_state=init_state,
            instruction=f"complete {task_id}",
            task_id=task_id,
        )
        elapsed = time.time() - t0

        if traj is not None:
            info = {
                "traj_idx": i,
                "latents_shape": traj.latents.shape,
                "latents_dtype": str(traj.latents.dtype),
                "actions_shape": traj.actions.shape,
                "actions_dtype": str(traj.actions.dtype),
                "states_shape": traj.states.shape,
                "states_dtype": str(traj.states.dtype),
                "instruction": traj.instruction,
                "task_id": traj.task_id,
                "time_sec": round(elapsed, 2),
                "num_steps": traj.actions.shape[0],
            }
            results.append(info)
            print(f"[TEST] ✅ traj {i}: steps={info['num_steps']}, "
                  f"latents={info['latents_shape']}, "
                  f"actions={info['actions_shape']}, "
                  f"states={info['states_shape']}, "
                  f"time={info['time_sec']}s")
        else:
            print(f"[TEST] ❌ traj {i}: rollout_single 返回 None")
            results.append({"traj_idx": i, "error": "returned None", "time_sec": round(elapsed, 2)})

    # --- 汇总报告 ---
    print(f"\n{'='*60}")
    print("[TEST] === 汇总报告 ===")
    success_count = sum(1 for r in results if "error" not in r)
    print(f"成功: {success_count}/{num_trajs}")

    for r in results:
        if "error" not in r:
            print(f"  traj {r['traj_idx']}: "
                  f"steps={r['num_steps']}, "
                  f"latents={r['latents_shape']} ({r['latents_dtype']}), "
                  f"actions={r['actions_shape']} ({r['actions_dtype']}), "
                  f"states={r['states_shape']} ({r['states_dtype']}), "
                  f"time={r['time_sec']}s")
        else:
            print(f"  traj {r['traj_idx']}: ❌ {r['error']}")

    # 验证数据格式一致性
    if success_count > 0:
        ref = [r for r in results if "error" not in r][0]
        print(f"\n[TEST] 数据格式验证:")
        print(f"  latents: {ref['latents_shape']} {ref['latents_dtype']}  (期望: (T, 4, 48, 24) float16)")
        print(f"  actions: {ref['actions_shape']} {ref['actions_dtype']}  (期望: (T, 7) float32)")
        print(f"  states:  {ref['states_shape']} {ref['states_dtype']}  (期望: (T, state_dim) float32)")
        expect_steps = cfg.num_interact * cfg.act_steps
        actual_steps = ref['num_steps']
        print(f"  步数:    {actual_steps} (期望 ≤ {expect_steps} = {cfg.num_interact} × {cfg.act_steps})")

    print(f"\n[TEST] env.step() 模式验证{'通过 ✅' if success_count == num_trajs else '部分失败 ⚠️'}")


if __name__ == "__main__":
    main()
