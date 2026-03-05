#!/usr/bin/env python3
"""Iter1 WM + ShortCut Flow 策略 → Imagination 合成轨迹生成.

使用 iter1 微调后的 Ctrl-World WM + 真实策略（ShortCut Flow）进行
Policy-in-the-Loop rollout 生成合成轨迹。

Usage:
    # 小规模验证（20条）
    CUDA_VISIBLE_DEVICES=4 conda run -n rlft_ms3 --no-banner \
        python scripts/vlaw/run/run_imagination_iter1.py \
        --num_trajs 20 --output_dir data/vlaw/synthetic/iter1_test20

    # 全量生成（300条）
    CUDA_VISIBLE_DEVICES=4 conda run -n rlft_ms3 --no-banner \
        python scripts/vlaw/run/run_imagination_iter1.py \
        --num_trajs 300 --output_dir data/vlaw/synthetic/iter1_wm
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

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _root not in sys.path:
    sys.path.insert(0, _root)
_cw = os.path.join(_root, "ctrl_world")
if _cw not in sys.path:
    sys.path.insert(0, _cw)

import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. 加载 Ctrl-World Adapter (iter1 WM)
# ---------------------------------------------------------------------------

def load_real_wm(ckpt_path: str, device: str = "cuda:0"):
    """加载 CtrlWorldAdapter + iter1 weights."""
    from ctrl_world.config import wm_args_maniskill
    from rlft.vlaw.world_model.ctrl_world_adapter import CtrlWorldAdapter

    args = wm_args_maniskill()
    args.svd_model_path = os.path.join(
        _root, "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"
    )
    args.clip_model_path = os.path.join(
        _root, "checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32"
    )
    args.data_stat_path = os.path.join(
        _root, "data/vlaw/meta_info/maniskill/stat.json"
    )
    args.num_inference_steps = 25
    args.num_frames = 5
    args.num_history = 6   # 对齐官方 DROID 配置 (history_idx 需要 6 帧)

    adapter = CtrlWorldAdapter(
        args,
        ckpt_path=ckpt_path,
        device=device,
        dtype=torch.float16,
    )
    return adapter


# ---------------------------------------------------------------------------
# 2. 加载 ShortCut Flow 策略
# ---------------------------------------------------------------------------

def load_policy(ckpt_path: str, device: str = "cuda:0"):
    """加载 ShortCut Flow 策略 (wrapped as MockPolicy-compatible interface).

    ImaginationEnvEngine.rollout_single 调用 policy.get_actions(obs_tensor)，
    其中 obs_tensor 是 (1, obs_horizon * feat_dim) 的 flat tensor。
    但 ShortCutFlowWrapper 需要 (B, obs_horizon*obs_dim) 且 obs_dim = visual_feat + state_dim。

    这里用一个 thin adapter 将接口对齐。

    Bug 1 fix: 必须传 visual_encoder_class=PlainConv 等参数，
    否则 visual_encoder=None, state_dim 计算会出错。
    """
    from rlft.utils.flow_wrapper import load_shortcut_flow_policy
    from rlft.networks import PlainConv

    wrapper, visual_encoder, state_dim = load_shortcut_flow_policy(
        ckpt_path,
        visual_encoder_class=PlainConv,
        obs_horizon=2,
        pred_horizon=8,
        action_dim=7,
        visual_feature_dim=256,
        include_rgb=True,
        use_ema=True,
        device=device,
    )
    print(
        f"[Iter1] 策略加载完成: state_dim={state_dim}, "
        f"visual_encoder={'None' if visual_encoder is None else type(visual_encoder).__name__}, "
        f"device={device}"
    )
    return wrapper, visual_encoder, state_dim


class PolicyAdapter:
    """将 ShortCutFlowWrapper 适配为 ImaginationEnvEngine 需要的接口.

    Bug 2 fix: ShortCutFlowWrapper 没有 get_actions() 方法，
    只有 __call__(obs_cond, initial_noise, return_numpy, act_steps)。

    Bug 3 fix: imagination_env 传来的 obs_tensor 是 flattened VAE latent，
    但 ShortCutFlowWrapper 期望 obs_cond = (B, obs_horizon * (256 + state_dim))。
    需要：decode VAE → RGB → PlainConv → visual_feat + state → obs_cond。

    imagination_env.py 通过 decoded_rgb / agent_state kwargs 传递解码后的数据。
    """

    def __init__(
        self,
        flow_wrapper,
        visual_encoder: torch.nn.Module | None,
        state_dim: int,
        obs_horizon: int = 2,
        act_steps: int = 5,
        device: str = "cuda:0",
    ):
        self.flow_wrapper = flow_wrapper
        self.visual_encoder = visual_encoder
        self.state_dim = state_dim
        self.obs_horizon = obs_horizon
        self.act_steps = act_steps
        self.device = torch.device(device)

        self.visual_feature_dim = (
            visual_encoder.out_dim if visual_encoder is not None else 0
        )
        self.single_obs_dim = self.visual_feature_dim + state_dim
        self._obs_history: list[np.ndarray] = []

        if visual_encoder is not None:
            visual_encoder.eval()
            for p in visual_encoder.parameters():
                p.requires_grad = False

    def reset_history(self) -> None:
        """Reset obs history (call at start of each trajectory)."""
        self._obs_history.clear()

    @torch.no_grad()
    def get_actions(
        self,
        obs_features: torch.Tensor,
        *,
        decoded_rgb: np.ndarray | None = None,
        agent_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate action chunk via ShortCut Flow.

        When *decoded_rgb* and *agent_state* are provided (and visual_encoder
        is available), performs the correct encoding pipeline:
            PlainConv(rgb) -> visual_feat (256,)
            [visual_feat, state] -> per-frame (281,)
            stack obs_horizon frames -> obs_cond (562,)
            ShortCutFlowWrapper(obs_cond, noise) -> (1, act_steps, 7)

        Returns:
            np.ndarray of shape ``(act_steps, action_dim)``.
        """
        if (
            decoded_rgb is not None
            and agent_state is not None
            and self.visual_encoder is not None
        ):
            return self._real_policy_forward(decoded_rgb, agent_state)
        # Fallback: zero actions
        return np.zeros((self.act_steps, 7), dtype=np.float32)

    def _real_policy_forward(
        self, decoded_rgb: np.ndarray, agent_state: np.ndarray,
    ) -> np.ndarray:
        """Encode RGB + state -> obs_cond -> flow_wrapper -> action chunk."""
        # --- 1. Encode RGB through PlainConv ---
        rgb = decoded_rgb
        if rgb.ndim == 3:
            rgb = rgb[np.newaxis]  # (1, H, W, 3)
        # (B, H, W, C) -> (B, C, H, W)
        rgb_t = (
            torch.from_numpy(rgb)
            .float()
            .permute(0, 3, 1, 2)
            .to(self.device)
        )
        if rgb_t.max() > 1.0:
            rgb_t = rgb_t / 255.0
        visual_feat = self.visual_encoder(rgb_t)  # (B, 256)
        visual_feat_np = visual_feat.cpu().numpy()[0]  # (256,)

        # --- 2. Concat visual + state -> per-frame feature ---
        state = agent_state[: self.state_dim].astype(np.float32)
        frame_feat = np.concatenate([visual_feat_np, state])  # (281,)

        # --- 3. Update obs history ---
        self._obs_history.append(frame_feat)
        if len(self._obs_history) > self.obs_horizon:
            self._obs_history = self._obs_history[-self.obs_horizon :]
        while len(self._obs_history) < self.obs_horizon:
            self._obs_history.insert(0, frame_feat.copy())

        # --- 4. Build obs_cond ---
        obs_cond = np.stack(self._obs_history, axis=0).flatten()  # (562,)
        obs_cond_t = (
            torch.from_numpy(obs_cond).float().unsqueeze(0).to(self.device)
        )  # (1, 562)

        # --- 5. ShortCutFlowWrapper call ---
        pred_horizon = self.flow_wrapper.pred_horizon
        action_dim = self.flow_wrapper.action_dim
        noise = torch.zeros(
            1, pred_horizon, action_dim, device=self.device,
        )
        actions = self.flow_wrapper(
            obs_cond_t,
            noise,
            return_numpy=True,
            act_steps=self.act_steps,
        )  # (1, act_steps, 7)

        return actions[0]  # (act_steps, 7)


# ---------------------------------------------------------------------------
# 3. 加载初始帧
# ---------------------------------------------------------------------------

def load_initial_frames(task_id: str, n: int) -> list[dict]:
    """从 VAE-encoded 数据加载真实初始帧 latent + state.

    优先扫描 data/vlaw/encoded/ 目录下带 latent_concat 的 H5 文件，
    取每条轨迹的第 0 帧作为初始帧。若 encoded 数据不存在则 fallback
    到 rollout 目录（此时 latent 为随机噪声，会打印 warning）。
    """
    # --- 优先从 encoded 数据加载（含真实 VAE latent） ---
    encoded_dirs = [
        os.path.join(_root, "data/vlaw/encoded/reencode_highsuc_inc20", task_id),
        os.path.join(_root, "data/vlaw/encoded/demos", task_id),
    ]

    candidates: list[dict] = []
    for ed in encoded_dirs:
        if not Path(ed).exists():
            continue
        h5_files = sorted(Path(ed).glob("*.h5"))
        for h5f in h5_files:
            try:
                with h5py.File(str(h5f), "r") as f:
                    traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
                    for tk in traj_keys:
                        grp = f[tk]
                        # latent_concat 是 VAE 编码的真实帧
                        if "latent_concat" not in grp:
                            continue
                        latent = grp["latent_concat"][0].astype(np.float32)  # (4,48,24)
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
                            "latent": latent,
                            "state": state,
                            "instruction": instruction,
                        })
            except Exception as e:
                print(f"[Iter1] ⚠️  读取 encoded {h5f} 失败: {e}")

    if candidates:
        print(f"[Iter1] ✅ 从 encoded 数据加载了 {len(candidates)} 条初始帧（含真实 VAE latent）")
    else:
        # --- Fallback: 从 rollout 目录加载（无 latent，用随机噪声） ---
        print("[Iter1] ⚠️  未找到 encoded 数据，fallback 到 rollout 目录（latent 为随机噪声！）")
        rollout_dirs = [
            os.path.join(_root, "data/vlaw/rollouts/iter1"),
            os.path.join(_root, "data/vlaw/rollouts/iter1_highsuc"),
            os.path.join(_root, "data/vlaw/rollouts/iter1_lift_inc20"),
        ]
        for rd in rollout_dirs:
            h5_pattern = f"**/{task_id}*.h5"
            h5_files = sorted(Path(rd).glob(h5_pattern)) if Path(rd).exists() else []
            for h5f in h5_files:
                try:
                    with h5py.File(str(h5f), "r") as f:
                        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
                        for tk in traj_keys:
                            grp = f[tk]
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
                                "latent": None,  # 标记：无真实 latent
                                "state": state,
                                "instruction": instruction,
                            })
                except Exception as e:
                    print(f"[Iter1] ⚠️  读取 {h5f} 失败: {e}")

        print(f"[Iter1] 从 rollout 数据加载了 {len(candidates)} 条初始帧候选（无 latent）")

    if not candidates:
        candidates = [{
            "latent": None,
            "state": np.zeros(25, dtype=np.float32),
            "instruction": "Lift the peg upright",
        }]

    rng = np.random.default_rng(42)
    idxs = rng.integers(0, len(candidates), size=n)
    results = []
    for idx in idxs:
        c = candidates[idx]
        if c["latent"] is not None:
            lat = torch.from_numpy(c["latent"])  # 真实 VAE latent
        else:
            lat = torch.randn(4, 48, 24, dtype=torch.float32)  # fallback 随机噪声
            print("[Iter1] ⚠️  使用随机噪声 latent（非真实帧）")
        results.append({
            "latent": lat,
            "state": c["state"],
            "instruction": c["instruction"],
        })
    return results


# ---------------------------------------------------------------------------
# 4. 主生成逻辑
# ---------------------------------------------------------------------------

def generate(
    num_trajs: int,
    num_interact: int,
    act_steps: int,
    gpu_id: int,
    wm_ckpt: str,
    policy_ckpt: str,
    output_dir: str,
    task_id: str,
    use_real_policy: bool,
    save_every: int = 50,
) -> dict:
    """生成合成轨迹并保存为 HDF5."""
    device = f"cuda:{gpu_id}"
    print(f"\n{'='*60}")
    print(f"[Iter1] 开始生成 {num_trajs} 条合成轨迹")
    print(f"  task: {task_id}")
    print(f"  WM ckpt: {wm_ckpt}")
    print(f"  Policy ckpt: {policy_ckpt}")
    print(f"  use_real_policy: {use_real_policy}")
    print(f"  device: {device}")
    print(f"  num_interact: {num_interact}, act_steps: {act_steps}")
    print(f"  output: {output_dir}")
    print(f"{'='*60}\n")

    # ---- Step A: 加载 WM ----
    t0 = time.time()
    print("[Iter1] 加载 Ctrl-World Adapter ...")
    wm_adapter = load_real_wm(wm_ckpt, device=device)
    print(f"[Iter1] ✅ WM 加载完成 ({time.time()-t0:.1f}s)")

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"[Iter1] WM GPU 显存: {mem:.2f} GB")

    # ---- Step B: 加载 Policy ----
    if use_real_policy:
        print("[Iter1] 加载 ShortCut Flow 策略 ...")
        try:
            wrapper, visual_encoder, state_dim = load_policy(policy_ckpt, device=device)
            policy = PolicyAdapter(
                wrapper, visual_encoder, state_dim,
                obs_horizon=2, act_steps=act_steps, device=device,
            )
            print(f"[Iter1] ✅ 真实策略加载完成 (state_dim={state_dim})")
        except Exception as e:
            print(f"[Iter1] ⚠️  真实策略加载失败: {e}, fallback 到 mock policy")
            from rlft.vlaw.world_model.imagination_env import _MockPolicy
            policy = _MockPolicy()
    else:
        from rlft.vlaw.world_model.imagination_env import _MockPolicy
        policy = _MockPolicy()
        print("[Iter1] 使用 Mock Policy（零动作）")

    # ---- Step C: 加载初始帧 ----
    print("[Iter1] 加载初始帧 ...")
    init_frames = load_initial_frames(task_id, num_trajs)
    print(f"[Iter1] ✅ 加载 {len(init_frames)} 条初始帧")

    # ---- Step D: 初始化 ImaginationEnvEngine ----
    from rlft.vlaw.world_model.imagination_env import (
        ImaginationEnvConfig,
        ImaginationEnvEngine,
    )

    cfg = ImaginationEnvConfig(
        num_envs=1,
        num_interact=num_interact,
        act_steps=act_steps,
        obs_horizon=2,
        task_id=task_id,
        tasks=[task_id],
        decode_for_policy=use_real_policy,
        dry_run=False,
        gpu_id=gpu_id,
        sim_backend="physx_cuda",
        camera_width=192,
        camera_height=192,
        output_dir=output_dir,
    )

    engine = ImaginationEnvEngine(
        wm_adapter=wm_adapter,
        policy=policy,
        config=cfg,
    )
    print("[Iter1] ✅ ImaginationEnvEngine 初始化完成")

    # ---- Step E: 逐条生成 ----
    os.makedirs(output_dir, exist_ok=True)
    all_trajectories = []
    total_time = 0.0
    saved_paths = []
    sapien_errors = 0
    other_errors = 0

    for i in range(num_trajs):
        t_start = time.time()
        frame = init_frames[i]
        traj = None
        error_msg = ""
        try:
            traj = engine.rollout_single(
                initial_latent=frame["latent"],
                initial_state=frame["state"],
                instruction=frame["instruction"],
                task_id=task_id,
            )
        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            if "SAPIEN" in tb or "Vulkan" in tb or "physical device" in tb:
                sapien_errors += 1
                print(f"[Iter1] ❌ traj {i}: SAPIEN/Vulkan error")
            else:
                other_errors += 1
                print(f"[Iter1] ❌ traj {i}: {error_msg[:120]}")

        elapsed = time.time() - t_start
        total_time += elapsed

        if traj is not None:
            all_trajectories.append(traj)
            steps = traj.actions.shape[0]
            avg_time = total_time / (i + 1)
            eta = avg_time * (num_trajs - i - 1) / 60
            print(
                f"[Iter1] traj {i+1}/{num_trajs}: steps={steps}, "
                f"time={elapsed:.1f}s, avg={avg_time:.1f}s, "
                f"ETA={eta:.0f}min, ok={len(all_trajectories)}, "
                f"sapien_err={sapien_errors}"
            )
        else:
            if not error_msg:
                print(f"[Iter1] traj {i+1}/{num_trajs}: ❌ returned None ({elapsed:.1f}s)")

        # 定期保存
        if (i + 1) % save_every == 0 and all_trajectories:
            save_path = _save_checkpoint(all_trajectories, output_dir, i + 1)
            saved_paths.append(save_path)
            print(f"[Iter1] 💾 中间保存: {save_path} ({len(all_trajectories)} 条)")

        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ---- 最终保存 ----
    final_path = ""
    if all_trajectories:
        final_path = _save_final(all_trajectories, output_dir)
        saved_paths.append(final_path)
        print(f"\n[Iter1] ✅ 最终保存: {final_path}")
    else:
        print("\n[Iter1] ❌ 无轨迹生成")

    # ---- 汇总 ----
    summary = {
        "task_id": task_id,
        "num_target": num_trajs,
        "num_generated": len(all_trajectories),
        "num_failed": num_trajs - len(all_trajectories),
        "sapien_errors": sapien_errors,
        "other_errors": other_errors,
        "success_rate": len(all_trajectories) / max(num_trajs, 1),
        "total_time_min": round(total_time / 60, 1),
        "avg_time_per_traj_s": round(total_time / max(len(all_trajectories), 1), 1),
        "output_dir": output_dir,
        "final_file": final_path,
        "wm_ckpt": wm_ckpt,
        "policy_ckpt": policy_ckpt,
        "use_real_policy": use_real_policy,
    }
    print(f"\n{'='*60}")
    print(f"[Iter1] 生成完成:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"{'='*60}")

    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)

    # ---- 关闭 ----
    engine.close()

    return summary


def _save_checkpoint(trajectories, output_dir, batch_id):
    """保存中间 checkpoint."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"synthetic_iter1_batch{batch_id}_{ts}.h5"

    with h5py.File(str(out_path), "w") as f:
        meta = f.create_group("meta")
        meta.attrs["num_trajectories"] = len(trajectories)
        meta.attrs["source"] = "imagination_env_iter1"
        meta.attrs["batch_id"] = batch_id
        meta.attrs["step5_method"] = "env.step()"
        if trajectories:
            meta.attrs["env_id"] = trajectories[0].task_id

        for idx, traj in enumerate(trajectories):
            grp = f.create_group(f"traj_{idx:04d}")
            grp.create_dataset("latent", data=traj.latents, chunks=True, compression="gzip", compression_opts=1)
            grp.create_dataset("actions", data=traj.actions, chunks=True, compression="gzip", compression_opts=1)
            grp.create_dataset("state", data=traj.states, chunks=True, compression="gzip", compression_opts=1)
            grp.attrs["task_instruction"] = traj.instruction
            grp.attrs["task_id"] = traj.task_id
            grp.attrs["source"] = "imagination_env_iter1"
    return str(out_path)


def _save_final(trajectories, output_dir):
    """保存最终结果."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"synthetic_iter1_final_{ts}.h5"

    with h5py.File(str(out_path), "w") as f:
        meta = f.create_group("meta")
        meta.attrs["num_trajectories"] = len(trajectories)
        meta.attrs["source"] = "imagination_env_iter1"
        meta.attrs["step5_method"] = "env.step()"
        if trajectories:
            meta.attrs["env_id"] = trajectories[0].task_id

        for idx, traj in enumerate(trajectories):
            grp = f.create_group(f"traj_{idx:04d}")
            grp.create_dataset("latent", data=traj.latents, chunks=True, compression="gzip", compression_opts=1)
            grp.create_dataset("actions", data=traj.actions, chunks=True, compression="gzip", compression_opts=1)
            grp.create_dataset("state", data=traj.states, chunks=True, compression="gzip", compression_opts=1)
            grp.attrs["task_instruction"] = traj.instruction
            grp.attrs["task_id"] = traj.task_id
            grp.attrs["source"] = "imagination_env_iter1"
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Iter1 WM + Policy Imagination 生成")
    parser.add_argument("--num_trajs", type=int, default=20)
    parser.add_argument("--num_interact", type=int, default=12)
    parser.add_argument("--act_steps", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="CUDA_VISIBLE_DEVICES 映射后的 GPU ID")
    parser.add_argument("--wm_ckpt", type=str,
                        default=os.path.join(_root, "checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt"))
    parser.add_argument("--policy_ckpt", type=str,
                        default=os.path.join(_root, "checkpoints/il/best_eval_success_once.pt"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_root, "data/vlaw/synthetic/iter1_test20"))
    parser.add_argument("--task_id", type=str, default="LiftPegUpright-v1")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--use_real_policy", action="store_true", default=True,
                        help="使用真实 ShortCut Flow 策略（默认 True）")
    parser.add_argument("--mock_policy", action="store_true",
                        help="使用 mock 策略（零动作）")
    args = parser.parse_args()

    use_real = args.use_real_policy and not args.mock_policy

    generate(
        num_trajs=args.num_trajs,
        num_interact=args.num_interact,
        act_steps=args.act_steps,
        gpu_id=args.gpu_id,
        wm_ckpt=args.wm_ckpt,
        policy_ckpt=args.policy_ckpt,
        output_dir=args.output_dir,
        task_id=args.task_id,
        use_real_policy=use_real,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
