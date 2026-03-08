#!/usr/bin/env python3
"""Imagination 合成轨迹生成 — 稳定入口脚本.

使用 Ctrl-World WM + ShortCut Flow 策略 + ManiSkill env.step() 生成 D_syn。
合并自 run_imagination_iter1.py / run_b1_imagination_200.py，已包含：
  - BUG-019 修复: 从 encoded 数据加载真实 VAE latent（非随机噪声）
  - BUG-017 修复: PlainConv 视觉编码器 + 正确的策略 API 调用
  - ADR-021: 保留 sliding window history（经消融验证无显著差异）

用法:
    # 小规模验证
    CUDA_VISIBLE_DEVICES=4 conda run -n rlft_ms3 --no-banner python \\
        rlft/vlaw/scripts/run_imagination.py \\
        --wm_ckpt checkpoints/vlaw/world_model/iter1_v3/checkpoint-2000.pt \\
        --num_trajs 20 --output_dir data/vlaw/synthetic/iter1_test20

    # 全量生成
    CUDA_VISIBLE_DEVICES=4 conda run -n rlft_ms3 --no-banner python \\
        rlft/vlaw/scripts/run_imagination.py \\
        --wm_ckpt checkpoints/vlaw/world_model/iter1_v3/checkpoint-2000.pt \\
        --num_trajs 200 --output_dir data/vlaw/synthetic/iter1_v3

    # Dry-run 验证通路
    conda run -n rlft_ms3 python rlft/vlaw/scripts/run_imagination.py --dry_run
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

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "ctrl_world"))

import h5py
import numpy as np
import torch


# ── 1. 加载 Ctrl-World WM ──────────────────────────────────────────────────

def load_wm(ckpt_path: str, device: str = "cuda:0", num_inference_steps: int = 25):
    """加载 CtrlWorldAdapter + finetuned weights."""
    from ctrl_world.config import wm_args_maniskill
    from rlft.vlaw.world_model.ctrl_world_adapter import CtrlWorldAdapter

    args = wm_args_maniskill()
    args.svd_model_path = str(WORKSPACE / "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid")
    args.clip_model_path = str(WORKSPACE / "checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32")
    args.data_stat_path = str(WORKSPACE / "data/vlaw/meta_info/maniskill/stat.json")
    args.num_inference_steps = num_inference_steps
    args.num_frames = 5
    args.num_history = 6  # 对齐官方 DROID 配置

    adapter = CtrlWorldAdapter(
        args, ckpt_path=ckpt_path, device=device, dtype=torch.float16,
    )
    return adapter


# ── 2. 加载 ShortCut Flow 策略 ─────────────────────────────────────────────

def load_policy(ckpt_path: str, device: str = "cuda:0"):
    """加载 ShortCut Flow 策略.

    BUG-017 fix: 必须传 PlainConv 参数, 否则 visual_encoder=None.
    """
    from rlft.utils.flow_wrapper import load_shortcut_flow_policy
    from rlft.networks import PlainConv

    wrapper, visual_encoder, state_dim = load_shortcut_flow_policy(
        ckpt_path,
        visual_encoder_class=PlainConv,
        obs_horizon=2, pred_horizon=8, action_dim=7,
        visual_feature_dim=256, include_rgb=True, use_ema=True, device=device,
    )
    print(
        f"[IMAGINATION] 策略加载完成: state_dim={state_dim}, "
        f"visual_encoder={type(visual_encoder).__name__ if visual_encoder else 'None'}"
    )
    return wrapper, visual_encoder, state_dim


class PolicyAdapter:
    """将 ShortCutFlowWrapper 适配为 ImaginationEnvEngine 的策略接口.

    BUG-017 fixes:
      - Bug 2: 正确调用 flow_wrapper.__call__() 而非不存在的 get_actions()
      - Bug 3: 从 decoded_rgb + agent_state 构建 obs_cond, 而非 flat VAE latent
    """

    def __init__(self, flow_wrapper, visual_encoder: torch.nn.Module | None,
                 state_dim: int, obs_horizon: int = 2, act_steps: int = 5,
                 device: str = "cuda:0"):
        self.flow_wrapper = flow_wrapper
        self.visual_encoder = visual_encoder
        self.state_dim = state_dim
        self.obs_horizon = obs_horizon
        self.act_steps = act_steps
        self.device = torch.device(device)
        self.visual_feature_dim = visual_encoder.out_dim if visual_encoder else 0
        self.single_obs_dim = self.visual_feature_dim + state_dim
        self._obs_history: list[np.ndarray] = []
        if visual_encoder is not None:
            visual_encoder.eval()
            for p in visual_encoder.parameters():
                p.requires_grad = False

    def reset_history(self) -> None:
        self._obs_history.clear()

    @torch.no_grad()
    def get_actions(self, obs_features: torch.Tensor, *,
                    decoded_rgb: np.ndarray | None = None,
                    agent_state: np.ndarray | None = None) -> np.ndarray:
        if decoded_rgb is not None and agent_state is not None and self.visual_encoder is not None:
            return self._real_policy_forward(decoded_rgb, agent_state)
        return np.zeros((self.act_steps, 7), dtype=np.float32)

    def _real_policy_forward(self, decoded_rgb: np.ndarray, agent_state: np.ndarray) -> np.ndarray:
        rgb = decoded_rgb[np.newaxis] if decoded_rgb.ndim == 3 else decoded_rgb
        rgb_t = torch.from_numpy(rgb).float().permute(0, 3, 1, 2).to(self.device)
        if rgb_t.max() > 1.0:
            rgb_t = rgb_t / 255.0
        visual_feat_np = self.visual_encoder(rgb_t).cpu().numpy()[0]

        state = agent_state[:self.state_dim].astype(np.float32)
        frame_feat = np.concatenate([visual_feat_np, state])

        self._obs_history.append(frame_feat)
        if len(self._obs_history) > self.obs_horizon:
            self._obs_history = self._obs_history[-self.obs_horizon:]
        while len(self._obs_history) < self.obs_horizon:
            self._obs_history.insert(0, frame_feat.copy())

        obs_cond = np.stack(self._obs_history, axis=0).flatten()
        obs_cond_t = torch.from_numpy(obs_cond).float().unsqueeze(0).to(self.device)

        noise = torch.zeros(1, self.flow_wrapper.pred_horizon,
                            self.flow_wrapper.action_dim, device=self.device)
        actions = self.flow_wrapper(obs_cond_t, noise, return_numpy=True, act_steps=self.act_steps)
        return actions[0]


# ── 3. 加载初始帧 (BUG-019 修复: 使用真实 VAE latent) ──────────────────────

def load_initial_frames(task_id: str, n: int,
                        encoded_dirs: list[str] | None = None,
                        seed: int = 42) -> list[dict]:
    """从 VAE-encoded 数据加载真实初始帧 latent + state.

    BUG-019 修复: 禁止使用 torch.randn 随机噪声, 必须有真实 VAE latent.
    """
    if encoded_dirs is None:
        encoded_dirs = [str(WORKSPACE / "data/vlaw/encoded/train" / task_id)]

    candidates: list[dict] = []
    for ed in encoded_dirs:
        if not Path(ed).exists():
            continue
        for h5f in sorted(Path(ed).glob("*.h5")):
            try:
                with h5py.File(str(h5f), "r") as f:
                    for tk in sorted(k for k in f.keys() if k.startswith("traj_")):
                        grp = f[tk]
                        if "latent_concat" not in grp:
                            continue
                        latent = grp["latent_concat"][0].astype(np.float32)
                        state = grp.get("obs_agent", grp.get("state", None))
                        state = state[0].astype(np.float32) if state is not None else np.zeros(25, np.float32)
                        instruction = grp.attrs.get("task_instruction", "Lift the peg upright")
                        candidates.append({"latent": latent, "state": state, "instruction": instruction})
            except Exception as e:
                print(f"[IMAGINATION] ⚠️ 读取 {h5f} 失败: {e}")

    if not candidates:
        raise FileNotFoundError(
            f"[IMAGINATION] ❌ 未找到 encoded 数据 (BUG-019 要求真实 VAE latent).\n"
            f"  搜索路径: {encoded_dirs}"
        )
    print(f"[IMAGINATION] ✅ 从 encoded 数据加载 {len(candidates)} 条初始帧（真实 VAE latent）")

    rng = np.random.default_rng(seed)
    return [
        {"latent": torch.from_numpy(candidates[i]["latent"]),
         "state": candidates[i]["state"],
         "instruction": candidates[i]["instruction"]}
        for i in rng.integers(0, len(candidates), size=n)
    ]


def _save_trajectories(trajectories: list, output_dir: str, suffix: str = "final") -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"synthetic_{suffix}_{int(time.time())}.h5"
    with h5py.File(str(out_path), "w") as f:
        meta = f.create_group("meta")
        meta.attrs["num_trajectories"] = len(trajectories)
        meta.attrs["source"] = "imagination_consolidated"
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
    return str(out_path)


# ── 4. 主生成逻辑 ──────────────────────────────────────────────────────────

def generate(*, num_trajs: int, num_interact: int, act_steps: int, gpu_id: int,
             wm_ckpt: str, policy_ckpt: str, output_dir: str, task_id: str,
             use_real_policy: bool, encoded_dir: str | None = None,
             save_every: int = 50, seed: int = 42,
             num_inference_steps: int = 25) -> dict:
    device = f"cuda:{gpu_id}"
    print(f"\n{'='*60}")
    print(f"[IMAGINATION] 生成 {num_trajs} 条 | task={task_id}")
    print(f"  WM={wm_ckpt}  Policy={policy_ckpt}")
    print(f"  num_interact={num_interact}, act_steps={act_steps}")
    print(f"{'='*60}\n")

    t0 = time.time()
    wm_adapter = load_wm(wm_ckpt, device=device, num_inference_steps=num_inference_steps)
    print(f"[IMAGINATION] ✅ WM 加载 ({time.time()-t0:.1f}s)")

    if use_real_policy:
        try:
            wrapper, visual_encoder, state_dim = load_policy(policy_ckpt, device=device)
            policy = PolicyAdapter(wrapper, visual_encoder, state_dim,
                                   obs_horizon=2, act_steps=act_steps, device=device)
        except Exception as e:
            print(f"[IMAGINATION] ⚠️ 策略加载失败: {e}, fallback mock")
            from rlft.vlaw.world_model.imagination_env import _MockPolicy
            policy = _MockPolicy()
    else:
        from rlft.vlaw.world_model.imagination_env import _MockPolicy
        policy = _MockPolicy()

    init_frames = load_initial_frames(task_id, num_trajs,
                                      [encoded_dir] if encoded_dir else None,
                                      seed=seed)

    from rlft.vlaw.world_model.imagination_env import ImaginationEnvConfig, ImaginationEnvEngine
    cfg = ImaginationEnvConfig(
        num_envs=1, num_interact=num_interact, act_steps=act_steps, obs_horizon=2,
        task_id=task_id, tasks=[task_id], decode_for_policy=use_real_policy,
        dry_run=False, gpu_id=gpu_id, sim_backend="physx_cuda",
        camera_width=192, camera_height=192, output_dir=output_dir,
    )
    engine = ImaginationEnvEngine(wm_adapter=wm_adapter, policy=policy, config=cfg)

    os.makedirs(output_dir, exist_ok=True)
    all_trajectories: list = []
    total_time = 0.0
    sapien_errors = other_errors = 0

    for i in range(num_trajs):
        t_start = time.time()
        traj = None
        try:
            traj = engine.rollout_single(
                initial_latent=init_frames[i]["latent"],
                initial_state=init_frames[i]["state"],
                instruction=init_frames[i]["instruction"],
                task_id=task_id,
            )
        except Exception as e:
            tb = traceback.format_exc()
            if "SAPIEN" in tb or "Vulkan" in tb:
                sapien_errors += 1
            else:
                other_errors += 1
                print(f"[IMAGINATION] ❌ traj {i}: {str(e)[:120]}")

        elapsed = time.time() - t_start
        total_time += elapsed
        if traj is not None:
            all_trajectories.append(traj)
            if hasattr(policy, 'reset_history'):
                policy.reset_history()
        if (i + 1) % save_every == 0 and all_trajectories:
            _save_trajectories(all_trajectories, output_dir, f"batch{i+1}")
        if (i + 1) % 10 == 0:
            eta = (total_time / (i + 1)) * (num_trajs - i - 1) / 60
            print(f"[IMAGINATION] {i+1}/{num_trajs}: ok={len(all_trajectories)} err={sapien_errors+other_errors} ETA={eta:.0f}m")
            gc.collect(); torch.cuda.empty_cache()

    final_path = _save_trajectories(all_trajectories, output_dir, "final") if all_trajectories else ""
    if final_path:
        print(f"[IMAGINATION] ✅ 最终保存: {final_path}")

    summary = {
        "task_id": task_id, "num_target": num_trajs,
        "num_generated": len(all_trajectories),
        "sapien_errors": sapien_errors, "other_errors": other_errors,
        "total_time_min": round(total_time / 60, 1),
        "final_file": final_path, "wm_ckpt": wm_ckpt, "policy_ckpt": policy_ckpt,
    }
    with open(os.path.join(output_dir, "generation_summary.json"), "w") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)
    engine.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="VLAW Imagination 合成轨迹生成 (稳定版)")
    parser.add_argument("--wm_ckpt", type=str,
                        default=str(WORKSPACE / "checkpoints/vlaw/world_model/iter1_v3/checkpoint-2000.pt"))
    parser.add_argument("--policy_ckpt", type=str,
                        default=str(WORKSPACE / "checkpoints/il/best_eval_success_once.pt"))
    parser.add_argument("--output_dir", type=str,
                        default=str(WORKSPACE / "data/vlaw/synthetic/iter1_v3"))
    parser.add_argument("--encoded_dir", type=str, default=None,
                        help="初始帧 encoded 目录 (默认: data/vlaw/encoded/train/{task})")
    parser.add_argument("--task_id", type=str, default="LiftPegUpright-v1")
    parser.add_argument("--num_trajs", type=int, default=200)
    parser.add_argument("--num_interact", type=int, default=12)
    parser.add_argument("--act_steps", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (用于多 GPU 并行时选取不同初始帧)")
    parser.add_argument("--use_real_policy", action="store_true", default=True)
    parser.add_argument("--mock_policy", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--visualize", action="store_true",
                        help="VAE 解码关键帧 strip PNG")
    parser.add_argument("--vis_count", type=int, default=5,
                        help="可视化前 N 条合成轨迹")
    parser.add_argument("--num_inference_steps", type=int, default=25,
                        help="WM diffusion denoising steps (25=quality, 10-15=fast)")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Imagination 通路验证")
        print(f"  WM={args.wm_ckpt}  Policy={args.policy_ckpt}  Output={args.output_dir}")
        print("[DRY RUN] ✅ 参数解析正常")
        return

    summary = generate(
        num_trajs=args.num_trajs, num_interact=args.num_interact,
        act_steps=args.act_steps, gpu_id=args.gpu_id,
        wm_ckpt=args.wm_ckpt, policy_ckpt=args.policy_ckpt,
        output_dir=args.output_dir, task_id=args.task_id,
        use_real_policy=args.use_real_policy and not args.mock_policy,
        encoded_dir=args.encoded_dir, save_every=args.save_every,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
    )

    if args.visualize and summary.get("final_file"):
        _visualize_synthetic(
            h5_path=summary["final_file"],
            output_dir=args.output_dir,
            vis_count=args.vis_count,
            device=f"cuda:{args.gpu_id}",
        )


# ── 可视化 ────────────────────────────────────────────────────────────────────

def _visualize_synthetic(h5_path: str, output_dir: str, vis_count: int = 5,
                        device: str = "cuda:0") -> None:
    """VAE 解码合成轨迹关键帧 (第一帧/中间帧/最后帧) 保存为 strip PNG.

    保存到 {output_dir}/viz/.
    """
    from PIL import Image as PILImage
    from diffusers.models import AutoencoderKLTemporalDecoder

    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 加载 VAE
    vae_path = str(WORKSPACE / "checkpoints/vlaw/world_model/pretrained"
                   "/stable-video-diffusion-img2vid/vae")
    if not Path(vae_path).exists():
        vae_path = str(WORKSPACE / "checkpoints/vlaw/world_model/pretrained"
                       "/stable-video-diffusion-img2vid")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        vae_path, torch_dtype=torch.float16).to(device).eval()

    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        n = min(vis_count, len(traj_keys))
        for i in range(n):
            grp = f[traj_keys[i]]
            latents = grp["latent"][:].astype(np.float32)  # (T, 4, 48, 24)
            T = latents.shape[0]
            # 第一帧 / 中间帧 / 最后帧
            idxs = np.array([0, T // 2, T - 1]) if T >= 3 else np.arange(T)
            rgb = _decode_latent_for_viz(vae, latents, idxs, device)
            # 拼 strip
            strip = np.concatenate([rgb[j] for j in range(rgb.shape[0])], axis=1)
            img = PILImage.fromarray(strip)
            save_path = viz_dir / f"{traj_keys[i]}_strip.png"
            img.save(str(save_path))
            print(f"[IMAGINATION] 🖼️ {save_path.name}")

    del vae
    torch.cuda.empty_cache()
    print(f"[IMAGINATION] 📊 可视化保存至 {viz_dir}")


@torch.inference_mode()
def _decode_latent_for_viz(vae, latents: np.ndarray, frame_indices: np.ndarray,
                          device: str, chunk_size: int = 4) -> np.ndarray:
    """解码 VAE latent 为 RGB (base camera 上半部分)."""
    selected = torch.from_numpy(latents[frame_indices]).to(device).to(torch.float16)
    decoded_list = []
    for i in range(0, selected.shape[0], chunk_size):
        chunk = selected[i:i+chunk_size] / vae.config.scaling_factor
        out = vae.decode(chunk, num_frames=chunk.shape[0]).sample
        decoded_list.append(out)
    decoded = torch.cat(decoded_list, dim=0)
    decoded = (decoded / 2.0 + 0.5).clamp(0, 1) * 255
    return decoded.float().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)[:, :192, :, :]


if __name__ == "__main__":
    main()
