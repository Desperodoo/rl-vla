#!/usr/bin/env python3
"""Policy 评估 — 稳定入口脚本.

使用 rlft/envs/evaluate.py 的 TESTED evaluate() 函数评估策略。
合并自: eval_pretrained_policy.py
已包含:
  - BUG-016 修复: 不 pad 零动作, 直接返回 7 个有效 action
  - 使用与 PLD 相同的 make_flow_eval_envs 创建环境

用法:
    # 评估预训练策略
    CUDA_VISIBLE_DEVICES=8 conda run -n rlft_ms3 python \\
        rlft/vlaw/scripts/evaluate_policy.py \\
        --policy_ckpt checkpoints/il/best_eval_success_once.pt

    # 评估 iter-1 策略
    CUDA_VISIBLE_DEVICES=8 conda run -n rlft_ms3 python \\
        rlft/vlaw/scripts/evaluate_policy.py \\
        --policy_ckpt checkpoints/vlaw/policy/iter1/policy_iter1.pt

    # Dry-run
    conda run -n rlft_ms3 python rlft/vlaw/scripts/evaluate_policy.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))


class BaseFlowPolicyAgent(nn.Module):
    """Wrap ShortCutFlowWrapper + PlainConv for evaluate().

    BUG-016 fix: 不 pad 零动作。evaluate() 使用 action_seq.shape[1] 迭代,
    返回 7 个 action（pred_horizon=8, obs_horizon=2, slice from index 1）。
    """

    def __init__(self, base_policy, visual_encoder, state_dim: int,
                 visual_feature_dim: int = 256, obs_horizon: int = 2,
                 act_steps: int = 8, action_dim: int = 7, device: str = "cuda"):
        super().__init__()
        self.base_policy = base_policy
        self.visual_encoder = visual_encoder
        self.state_dim = state_dim
        self.visual_feature_dim = visual_feature_dim
        self.obs_horizon = obs_horizon
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.device = device
        self.single_obs_dim = state_dim + (visual_feature_dim if visual_encoder else 0)

    def _encode_obs(self, obs) -> torch.Tensor:
        parts = []
        B = None
        if isinstance(obs, dict):
            if self.visual_encoder is not None and "rgb" in obs:
                rgb = obs["rgb"]
                if isinstance(rgb, np.ndarray):
                    rgb = torch.from_numpy(rgb).to(self.device)
                rgb = rgb.float()
                B = rgb.shape[0]
                T = rgb.shape[1] if rgb.dim() == 5 else 1
                if rgb.dim() == 5:
                    rgb = rgb.reshape(B * T, *rgb.shape[2:])
                if rgb.dim() == 4 and rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    rgb = rgb.permute(0, 3, 1, 2)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                with torch.no_grad():
                    vfeat = self.visual_encoder(rgb)
                vfeat = vfeat.reshape(B, T, -1) if T > 1 else vfeat.unsqueeze(1)
                parts.append(vfeat)

            state = obs.get("state", obs.get("agent", None))
            if state is not None:
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).to(self.device).float()
                if B is None:
                    B = state.shape[0]
                if state.dim() == 2:
                    state = state.unsqueeze(1)
                if state.shape[-1] > self.state_dim:
                    state = state[..., :self.state_dim]
                parts.append(state)

            combined = torch.cat(parts, dim=-1)
            combined = combined[:, -self.obs_horizon:, :self.single_obs_dim]
            return combined.reshape(B, -1)
        else:
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).to(self.device).float()
            return obs

    @torch.no_grad()
    def get_action(self, obs, **kwargs) -> torch.Tensor:
        """BUG-016: 不 pad 零动作, 直接返回有效 action."""
        obs_cond = self._encode_obs(obs)
        B = obs_cond.shape[0]
        zero_noise = torch.zeros(B, self.act_steps, self.action_dim, device=self.device)
        actions = self.base_policy(
            obs_cond, zero_noise, return_numpy=False, act_steps=self.act_steps,
        )
        return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="VLAW Policy 评估 (稳定版)")
    parser.add_argument("--policy_ckpt", type=str,
                        default=str(WORKSPACE / "checkpoints/il/best_eval_success_once.pt"))
    parser.add_argument("--output_path", type=str,
                        default=str(WORKSPACE / "results/vlaw/policy_eval.json"))
    parser.add_argument("--env_id", type=str, default="LiftPegUpright-v1")
    parser.add_argument("--num_envs", type=int, default=50)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--pred_horizon", type=int, default=8)
    parser.add_argument("--act_steps", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--visualize", action="store_true",
                        help="保存成功/失败 rollout GIF")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Policy 评估通路验证")
        print(f"  ckpt={args.policy_ckpt}")
        results = {args.env_id: {"success_rate": 0.0, "dry_run": True}}
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print("[DRY RUN] ✅ 完成")
        return

    device = f"cuda:{args.gpu_id}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load policy
    from rlft.utils.flow_wrapper import load_shortcut_flow_policy
    from rlft.networks import PlainConv

    base_policy, visual_encoder, state_dim = load_shortcut_flow_policy(
        args.policy_ckpt, visual_encoder_class=PlainConv,
        obs_horizon=args.obs_horizon, pred_horizon=args.pred_horizon,
        action_dim=7, visual_feature_dim=256, include_rgb=True,
        use_ema=True, device=device,
    )
    print(f"[EVAL-POLICY] state_dim={state_dim}")

    # 2. Create eval env (same as PLD)
    from rlft.online._flow_helpers import make_flow_eval_envs
    eval_args = SimpleNamespace(
        obs_mode="rgb", control_mode="pd_ee_delta_pose", reward_mode="dense",
        max_episode_steps=args.max_episode_steps, env_id=args.env_id,
        num_eval_envs=args.num_envs, sim_backend="physx_cuda",
        obs_horizon=args.obs_horizon,
    )
    eval_envs = make_flow_eval_envs(eval_args)

    # 3. Agent
    agent = BaseFlowPolicyAgent(
        base_policy, visual_encoder, state_dim,
        obs_horizon=args.obs_horizon, act_steps=args.act_steps, device=device,
    )

    # 4. Evaluate
    from rlft.envs.evaluate import evaluate
    t0 = time.time()
    eval_metrics = evaluate(n=args.num_episodes, agent=agent, eval_envs=eval_envs,
                            device=device, sim_backend="physx_cuda")
    elapsed = time.time() - t0

    results = {}
    for k, v in eval_metrics.items():
        results[k] = round(float(np.mean(v)), 4)

    s_once = results.get("success_once", 0) * 100
    s_end = results.get("success_at_end", 0) * 100
    print(f"\n[EVAL-POLICY] success_once={s_once:.1f}%, success_at_end={s_end:.1f}% ({elapsed:.1f}s)")

    full = {
        "checkpoint": args.policy_ckpt, "env_id": args.env_id,
        "num_episodes": args.num_episodes, "metrics": results,
        "success_once_pct": s_once, "success_at_end_pct": s_end,
        "time_s": round(elapsed, 1),
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(full, f, indent=2)
    print(f"[EVAL-POLICY] ✅ 保存: {args.output_path}")

    if args.visualize:
        _visualize_rollouts(
            agent=agent, eval_envs=eval_envs, device=device,
            output_dir=str(Path(args.output_path).parent),
            max_steps=args.max_episode_steps,
        )

    eval_envs.close()


# ── 可视化 ──────────────────────────────────────────────────────────────────

def _visualize_rollouts(agent, eval_envs, device: str, output_dir: str,
                        max_steps: int = 100) -> None:
    """保存 1 条成功和 1 条失败 rollout 的 rgb_base 帧为 GIF.

    保存到 {output_dir}/viz/.
    """
    from PIL import Image as PILImage

    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 运行少量 rollout, 收集帧
    rollouts: list[dict] = []  # {"frames": [...], "success": bool}
    try:
        obs, _ = eval_envs.reset()
        frames_buf: list[list[np.ndarray]] = [[] for _ in range(eval_envs.num_envs)]
        success_buf = np.zeros(eval_envs.num_envs, dtype=bool)
        done_buf = np.zeros(eval_envs.num_envs, dtype=bool)

        for step in range(max_steps):
            action_seq = agent.get_action(obs)
            for act_idx in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(
                    action_seq[:, act_idx])
                # 采样帧 (每 5 步采 1 帧)
                if step % 5 == 0:
                    if isinstance(obs, dict) and "rgb" in obs:
                        rgb = obs["rgb"]
                        if isinstance(rgb, torch.Tensor):
                            rgb = rgb.cpu().numpy()
                        for env_i in range(min(eval_envs.num_envs, 10)):
                            if not done_buf[env_i] and rgb.ndim >= 4:
                                frame = rgb[env_i]
                                if frame.ndim == 4:  # (T, H, W, C)
                                    frame = frame[-1]
                                if frame.ndim == 3 and frame.shape[0] in (3, 6):
                                    frame = frame.transpose(1, 2, 0)
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                frames_buf[env_i].append(frame[:, :, :3].astype(np.uint8))

                done = terminated | truncated
                if "success" in info:
                    succ = info["success"]
                    if isinstance(succ, torch.Tensor):
                        succ = succ.cpu().numpy()
                    success_buf = success_buf | succ.astype(bool)
                done_buf = done_buf | (done.cpu().numpy() if isinstance(done, torch.Tensor) else done).astype(bool)

        for env_i in range(min(eval_envs.num_envs, 10)):
            if frames_buf[env_i]:
                rollouts.append({"frames": frames_buf[env_i],
                                 "success": bool(success_buf[env_i])})
    except Exception as e:
        print(f"[EVAL-POLICY] ⚠️ 可视化采集失败: {e}")
        return

    # 保存 GIF: 1 success + 1 fail
    saved = {"success": False, "fail": False}
    for rollout in rollouts:
        label = "success" if rollout["success"] else "fail"
        if saved[label]:
            continue
        frames = rollout["frames"]
        if not frames:
            continue
        pil_frames = [PILImage.fromarray(f) for f in frames]
        gif_path = viz_dir / f"rollout_{label}.gif"
        pil_frames[0].save(
            str(gif_path), save_all=True, append_images=pil_frames[1:],
            duration=200, loop=0)
        print(f"[EVAL-POLICY] 🎬 {gif_path.name} ({len(frames)} frames)")
        saved[label] = True
        if saved["success"] and saved["fail"]:
            break


if __name__ == "__main__":
    main()
