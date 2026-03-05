#!/usr/bin/env python3
"""Track B Step B1 — 使用 pretrained WM + env.step() 生成 200 条合成轨迹.

Usage (tmux):
    CUDA_VISIBLE_DEVICES=4,5 conda run -n rlft_ms3 --no-banner \
        python scripts/run_b1_imagination_200.py \
        --num_trajs 200 \
        --num_interact 12 \
        --gpu_id 0 \
        --output_dir data/vlaw/synthetic/iter1_pretrained/LiftPegUpright-v1
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
# ctrl_world must be importable
_cw = os.path.join(_root, "ctrl_world")
if _cw not in sys.path:
    sys.path.insert(0, _cw)

import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. 加载 Ctrl-World Adapter（真实预训练 WM）
# ---------------------------------------------------------------------------

def load_real_wm(ckpt_path: str, device: str = "cuda:0") -> "CtrlWorldAdapter":
    """加载 CtrlWorldAdapter + pretrained weights."""
    from ctrl_world.config import wm_args_maniskill  # type: ignore
    from rlft.vlaw.world_model.ctrl_world_adapter import CtrlWorldAdapter

    args = wm_args_maniskill()
    # 确保路径为绝对路径
    args.svd_model_path = os.path.join(
        _root, "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"
    )
    args.clip_model_path = os.path.join(
        _root, "checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32"
    )
    args.data_stat_path = os.path.join(
        _root, "data/vlaw/meta_info/maniskill/stat.json"
    )
    # 推理加速
    args.num_inference_steps = 25
    args.num_frames = 5
    args.num_history = 4

    adapter = CtrlWorldAdapter(
        args,
        ckpt_path=ckpt_path,
        device=device,
        dtype=torch.float16,
    )
    return adapter


# ---------------------------------------------------------------------------
# 2. 加载初始帧（从真实 rollout 数据）
# ---------------------------------------------------------------------------

def load_initial_frames_from_rollouts(
    task_id: str,
    n: int,
) -> list[dict]:
    """从真实 rollout 数据加载初始帧信息（state + random latent）.

    Returns:
        list of {"latent": Tensor(4,48,24), "state": ndarray(25,), "instruction": str}
    """
    rollout_dirs = [
        os.path.join(_root, "data/vlaw/rollouts/iter1"),
        os.path.join(_root, "data/vlaw/rollouts/iter1_lift_only"),
    ]
    
    candidates: list[dict] = []
    for rd in rollout_dirs:
        h5_files = sorted(Path(rd).glob(f"**/{task_id}*.h5"))
        for h5f in h5_files:
            try:
                with h5py.File(str(h5f), "r") as f:
                    traj_keys = [k for k in f.keys() if k.startswith("traj_")]
                    for tk in traj_keys:
                        grp = f[tk]
                        # 读取初始状态
                        if "obs_agent" in grp:
                            state = grp["obs_agent"][0].astype(np.float32)
                        elif "state" in grp:
                            state = grp["state"][0].astype(np.float32)
                        else:
                            state = np.zeros(25, dtype=np.float32)
                        instruction = grp.attrs.get(
                            "task_instruction", "Lift the peg upright"
                        )
                        candidates.append({
                            "state": state,
                            "instruction": instruction,
                        })
            except Exception as e:
                print(f"[B1] ⚠️  读取 {h5f} 失败: {e}")

    print(f"[B1] 从真实数据加载了 {len(candidates)} 条初始帧候选")

    if not candidates:
        # 兜底：全部使用零状态
        candidates = [{
            "state": np.zeros(25, dtype=np.float32),
            "instruction": "Lift the peg upright",
        }]

    # 随机采样 n 条（可重复）
    rng = np.random.default_rng(42)
    idxs = rng.integers(0, len(candidates), size=n)
    results = []
    for i, idx in enumerate(idxs):
        c = candidates[idx]
        # 为每条生成随机 initial latent（因为真实数据没有 VAE latent）
        lat = torch.randn(4, 48, 24, dtype=torch.float32)
        results.append({
            "latent": lat,
            "state": c["state"],
            "instruction": c["instruction"],
        })
    return results


# ---------------------------------------------------------------------------
# 3. 主生成逻辑
# ---------------------------------------------------------------------------

def generate_trajectories(
    num_trajs: int,
    num_interact: int,
    act_steps: int,
    gpu_id: int,
    wm_ckpt: str,
    output_dir: str,
    task_id: str,
    save_every: int = 50,
) -> str:
    """生成合成轨迹并保存为 HDF5."""
    device = f"cuda:{gpu_id}"
    print(f"\n{'='*60}")
    print(f"[B1] 开始生成 {num_trajs} 条合成轨迹")
    print(f"  task: {task_id}")
    print(f"  WM ckpt: {wm_ckpt}")
    print(f"  device: {device}")
    print(f"  num_interact: {num_interact}, act_steps: {act_steps}")
    print(f"  output: {output_dir}")
    print(f"{'='*60}\n")

    # ---- 加载 WM ----
    t0 = time.time()
    print("[B1] Step 1: 加载 Ctrl-World Adapter ...")
    wm_adapter = load_real_wm(wm_ckpt, device=device)
    print(f"[B1] ✅ WM 加载完成 ({time.time()-t0:.1f}s)")

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"[B1] 当前 GPU 显存: {mem:.2f} GB")

    # ---- 加载初始帧 ----
    print("[B1] Step 2: 加载初始帧 ...")
    init_frames = load_initial_frames_from_rollouts(task_id, num_trajs)
    print(f"[B1] ✅ 加载 {len(init_frames)} 条初始帧")

    # ---- 初始化 ImaginationEnvEngine ----
    from rlft.vlaw.world_model.imagination_env import (
        ImaginationEnvConfig,
        ImaginationEnvEngine,
        _MockPolicy,
    )

    cfg = ImaginationEnvConfig(
        num_envs=1,
        num_interact=num_interact,
        act_steps=act_steps,
        obs_horizon=2,
        task_id=task_id,
        tasks=[task_id],
        decode_for_policy=False,  # 用 latent 直接喂策略，不解码（节省时间）
        dry_run=False,
        gpu_id=gpu_id,
        sim_backend="physx_cuda",
        camera_width=192,
        camera_height=192,
        output_dir=output_dir,
    )

    # 使用 mock policy（返回零动作）—— 可替换为真实策略
    mock_policy = _MockPolicy()
    engine = ImaginationEnvEngine(
        wm_adapter=wm_adapter,
        policy=mock_policy,
        config=cfg,
    )
    print("[B1] ✅ ImaginationEnvEngine 初始化完成")

    # ---- 逐条生成 ----
    os.makedirs(output_dir, exist_ok=True)
    all_trajectories = []
    total_time = 0.0
    saved_paths = []

    for i in range(num_trajs):
        t_start = time.time()
        frame = init_frames[i]
        try:
            traj = engine.rollout_single(
                initial_latent=frame["latent"],
                initial_state=frame["state"],
                instruction=frame["instruction"],
                task_id=task_id,
            )
        except Exception:
            print(f"[B1] ❌ traj {i}: 异常\n{traceback.format_exc()}")
            traj = None

        elapsed = time.time() - t_start
        total_time += elapsed

        if traj is not None:
            all_trajectories.append(traj)
            steps = traj.actions.shape[0]
            avg_time = total_time / (i + 1)
            eta = avg_time * (num_trajs - i - 1) / 60
            print(
                f"[B1] traj {i+1}/{num_trajs}: steps={steps}, "
                f"time={elapsed:.1f}s, avg={avg_time:.1f}s, "
                f"ETA={eta:.0f}min, success={len(all_trajectories)} ✅"
            )
        else:
            print(f"[B1] traj {i+1}/{num_trajs}: ❌ failed ({elapsed:.1f}s)")

        # 定期保存（防止中断丢失数据）
        if (i + 1) % save_every == 0 and all_trajectories:
            save_path = _save_checkpoint(
                engine, all_trajectories, output_dir, i + 1
            )
            saved_paths.append(save_path)
            print(f"[B1] 💾 中间保存: {save_path} ({len(all_trajectories)} 条)")

        # 清理 GPU 缓存
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ---- 最终保存 ----
    if all_trajectories:
        final_path = engine.save_trajectories(all_trajectories, output_dir)
        saved_paths.append(final_path)
        print(f"\n[B1] ✅ 最终保存: {final_path}")
    else:
        final_path = ""
        print("\n[B1] ❌ 无轨迹生成")

    # ---- 汇总 ----
    summary = {
        "task_id": task_id,
        "num_target": num_trajs,
        "num_generated": len(all_trajectories),
        "success_rate": len(all_trajectories) / max(num_trajs, 1),
        "total_time_min": total_time / 60,
        "avg_time_per_traj_s": total_time / max(len(all_trajectories), 1),
        "output_dir": output_dir,
        "saved_files": saved_paths,
        "final_file": final_path,
    }
    print(f"\n{'='*60}")
    print(f"[B1] 生成完成:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"{'='*60}")

    # 保存 summary
    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)

    return final_path


def _save_checkpoint(engine, trajectories, output_dir, batch_id):
    """保存中间 checkpoint."""
    ts = int(time.time())
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"synthetic_env_batch{batch_id}_{ts}.h5"

    with h5py.File(str(out_path), "w") as f:
        meta = f.create_group("meta")
        meta.attrs["num_trajectories"] = len(trajectories)
        meta.attrs["source"] = "imagination_env_b1"
        meta.attrs["batch_id"] = batch_id
        if trajectories:
            meta.attrs["env_id"] = trajectories[0].task_id
        meta.attrs["step5_method"] = "env.step()"

        for idx, traj in enumerate(trajectories):
            grp = f.create_group(f"traj_{idx:04d}")
            grp.create_dataset(
                "latent", data=traj.latents,
                chunks=True, compression="gzip", compression_opts=1,
            )
            grp.create_dataset(
                "actions", data=traj.actions,
                chunks=True, compression="gzip", compression_opts=1,
            )
            grp.create_dataset(
                "state", data=traj.states,
                chunks=True, compression="gzip", compression_opts=1,
            )
            grp.attrs["task_instruction"] = traj.instruction
            grp.attrs["task_id"] = traj.task_id
            grp.attrs["source"] = "imagination_env_b1"

    return str(out_path)


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="B1: 200 条合成轨迹生成")
    parser.add_argument("--num_trajs", type=int, default=200)
    parser.add_argument("--num_interact", type=int, default=12)
    parser.add_argument("--act_steps", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="CUDA_VISIBLE_DEVICES 映射后的 GPU ID")
    parser.add_argument(
        "--wm_ckpt",
        type=str,
        default=os.path.join(
            _root,
            "checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt",
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            _root,
            "data/vlaw/synthetic/iter1_pretrained/LiftPegUpright-v1",
        ),
    )
    parser.add_argument("--task_id", type=str, default="LiftPegUpright-v1")
    parser.add_argument("--save_every", type=int, default=50,
                        help="每 N 条保存一次中间结果")
    args = parser.parse_args()

    generate_trajectories(
        num_trajs=args.num_trajs,
        num_interact=args.num_interact,
        act_steps=args.act_steps,
        gpu_id=args.gpu_id,
        wm_ckpt=args.wm_ckpt,
        output_dir=args.output_dir,
        task_id=args.task_id,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
