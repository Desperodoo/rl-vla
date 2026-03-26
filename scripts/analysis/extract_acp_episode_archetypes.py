#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from rlft.algorithms.online_rl.dsrl_sac import DSRLSACAgent
from rlft.algorithms.online_rl.pld_sac import PLDSACAgent
from rlft.envs import make_eval_envs
from rlft.envs.acp_reward_wrapper import ACPRewardConfig, DualCameraRewardWrapper
from rlft.networks import PlainConv
from rlft.online.train_dsrl import DSRLEvalAgentWrapper
from rlft.online.train_pld import PLDEvalAgentWrapper
from rlft.utils.flow_wrapper import load_shortcut_flow_policy

PROJECT_ROOT = Path("/home/wjz/rl-vla")
OUT_DIR = PROJECT_ROOT / "docs" / "vlaw" / "figures" / "acp_episode_archetypes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


@dataclass
class CaseConfig:
    name: str
    algo: str
    reward: str
    run_dir: Path
    checkpoint_name: str


CASES = [
    CaseConfig("dsrl_sim_best_sae", "dsrl", "sim", PROJECT_ROOT / "runs/dsrl_v7_reg_qclip0_sim_s42__1774197495", "best_sae.pt"),
    CaseConfig("dsrl_acp_best_so", "dsrl", "acp", PROJECT_ROOT / "runs/dsrl_v7_qclip0_acp_mirror_s42__1774237641", "best.pt"),
    CaseConfig("pld_sim_best_sae", "pld", "sim", PROJECT_ROOT / "runs/pld_v7_reg_qclip0_sim_s42__1774197493", "best_sae.pt"),
    CaseConfig("pld_acp_best_so", "pld", "acp", PROJECT_ROOT / "runs/pld_v7_qclip0_acp_mirror_s42__1774237641", "best.pt"),
]


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _build_env_and_agent(case: CaseConfig):
    cfg = json.loads((case.run_dir / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    include_rgb = True

    base_policy, visual_encoder, inferred_state_dim = load_shortcut_flow_policy(
        checkpoint_path=cfg["checkpoint"],
        visual_encoder_class=PlainConv,
        obs_horizon=cfg["obs_horizon"],
        pred_horizon=cfg["pred_horizon"],
        action_dim=cfg["action_dim"],
        visual_feature_dim=cfg["visual_feature_dim"],
        include_rgb=include_rgb,
        use_ema=cfg.get("use_ema", True),
        device=str(device),
    )
    state_dim = cfg.get("state_dim", 0) or inferred_state_dim
    obs_dim = cfg["obs_horizon"] * (cfg["visual_feature_dim"] + state_dim)

    env_kwargs = dict(
        control_mode=cfg["control_mode"],
        obs_mode="rgbd",
        render_mode="rgb_array",
        reward_mode=cfg["reward_mode"],
        max_episode_steps=cfg["max_episode_steps"],
    )
    wrappers: list = []
    if case.reward == "acp":
        acp_config = ACPRewardConfig(
            checkpoint_path=cfg["acp_checkpoint"],
            task_instruction=cfg["acp_task_instruction"],
            reward_scale=cfg["acp_reward_scale"],
            reward_shaping=cfg["acp_reward_shaping"],
            reward_clip=cfg["acp_reward_clip"],
            grasp_bonus=cfg.get("acp_grasp_bonus", 0.0),
            device=cfg.get("acp_device") or "cuda:1",
        )
        wrappers.append(lambda env: DualCameraRewardWrapper(env, acp_config))
    wrappers.append(FlattenRGBDObservationWrapper)

    eval_envs = make_eval_envs(
        env_id=cfg["env_id"],
        num_envs=1,
        sim_backend=cfg["sim_backend"],
        env_kwargs=env_kwargs,
        other_kwargs=dict(obs_horizon=cfg["obs_horizon"]),
        video_dir=None,
        wrappers=wrappers,
    )

    ckpt = torch.load(case.run_dir / "checkpoints" / case.checkpoint_name, map_location=device)

    if case.algo == "dsrl":
        agent = DSRLSACAgent(
            obs_dim=obs_dim,
            act_steps=cfg["act_steps"],
            action_dim=cfg["action_dim"],
            action_magnitude=cfg["action_magnitude"],
            hidden_dims=[cfg["layer_size"]] * cfg["num_layers"],
            num_qs=cfg["num_qs"],
            gamma=cfg["gamma"],
            tau=cfg["tau"],
            init_temperature=cfg["init_temperature"],
            target_entropy=cfg["target_entropy"],
            log_std_init=cfg["log_std_init"],
            use_layer_norm=cfg["use_layer_norm"],
            q_target_clip=cfg.get("q_target_clip", 0.0),
            device=str(device),
        ).to(device)
        agent.load_state_dict(ckpt["agent"], strict=False)
        wrapper = DSRLEvalAgentWrapper(
            agent=agent,
            base_policy=base_policy,
            visual_encoder=visual_encoder,
            include_rgb=True,
            obs_horizon=cfg["obs_horizon"],
            act_steps=cfg["act_steps"],
            action_dim=cfg["action_dim"],
            device=str(device),
        )
    else:
        agent = PLDSACAgent(
            obs_dim=obs_dim,
            act_steps=cfg["act_steps"],
            action_dim=cfg["action_dim"],
            action_scale=cfg["action_scale"],
            hidden_dims=[cfg["layer_size"]] * cfg["num_layers"],
            num_qs=cfg["num_qs"],
            gamma=cfg["gamma"],
            tau=cfg["tau"],
            init_temperature=cfg["init_temperature"],
            target_entropy=cfg["target_entropy"],
            log_std_init=cfg["log_std_init"],
            use_layer_norm=cfg["use_layer_norm"],
            q_target_clip=cfg.get("q_target_clip", 0.0),
            min_temperature=cfg.get("min_temperature", 0.0),
            entropy_bonus_coef=cfg.get("entropy_bonus_coef", 0.0),
            device=str(device),
        ).to(device)
        agent.load_state_dict(ckpt["agent"], strict=False)
        if visual_encoder is not None and "visual_encoder" in ckpt:
            visual_encoder.load_state_dict(ckpt["visual_encoder"], strict=False)
        wrapper = PLDEvalAgentWrapper(
            agent=agent,
            base_policy=base_policy,
            visual_encoder=visual_encoder,
            include_rgb=True,
            obs_horizon=cfg["obs_horizon"],
            act_steps=cfg["act_steps"],
            action_dim=cfg["action_dim"],
            action_scale=cfg["action_scale"],
            device=str(device),
        )

    wrapper.eval()
    return cfg, eval_envs, wrapper, device


def _extract_rgb(obs: dict) -> np.ndarray:
    rgb = obs["rgb"]
    rgb = _to_np(rgb)
    if rgb.ndim == 5:
        rgb = rgb[0, -1]
    elif rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.shape[0] in [3, 4, 6, 9, 12]:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.max() <= 1.0:
        rgb = (rgb * 255.0).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    return rgb


def rollout_case(case: CaseConfig, max_episodes: int = 12) -> dict:
    cfg, env, eval_wrapper, device = _build_env_and_agent(case)
    episodes: list[dict] = []
    obs, info = env.reset()
    current = {
        "frames": [],
        "sim_reward": [],
        "acp_total_reward": [],
        "acp_base_reward": [],
        "acp_grasp_bonus": [],
        "is_grasping": [],
        "success_step": [],
    }

    completed = 0
    while completed < max_episodes:
        obs_t = common.to_tensor(obs, device)
        action_seq = eval_wrapper.get_action(obs_t, deterministic=True)
        action_seq = _to_np(action_seq)
        for i in range(action_seq.shape[1]):
            current["frames"].append(_extract_rgb(obs))
            obs, rew, terminated, truncated, info = env.step(action_seq[:, i])
            current["sim_reward"].append(float(_to_np(info.get("sim_reward", rew))[0]))
            current["acp_total_reward"].append(float(_to_np(info.get("acp_total_reward", np.array([0.0])))[0]))
            current["acp_base_reward"].append(float(_to_np(info.get("acp_base_reward", np.array([0.0])))[0]))
            current["acp_grasp_bonus"].append(float(_to_np(info.get("acp_grasp_bonus", np.array([0.0])))[0]))
            current["is_grasping"].append(float(_to_np(info.get("is_grasping", np.array([0.0])))[0]))
            current["success_step"].append(float(_to_np(info.get("success", np.array([0.0])))[0]))
            if _to_np(truncated).any():
                final_info = info["final_info"]
                episode_metrics = final_info["episode"] if isinstance(final_info, dict) else final_info[0]["episode"]
                so = float(_to_np(episode_metrics["success_once"])[0])
                sae = float(_to_np(episode_metrics["success_at_end"])[0])
                episodes.append(
                    {
                        "so": so,
                        "sae": sae,
                        "frames": np.stack(current["frames"], axis=0),
                        "sim_reward": np.array(current["sim_reward"], dtype=np.float32),
                        "acp_total_reward": np.array(current["acp_total_reward"], dtype=np.float32),
                        "acp_base_reward": np.array(current["acp_base_reward"], dtype=np.float32),
                        "acp_grasp_bonus": np.array(current["acp_grasp_bonus"], dtype=np.float32),
                        "is_grasping": np.array(current["is_grasping"], dtype=np.float32),
                        "success_step": np.array(current["success_step"], dtype=np.float32),
                    }
                )
                current = {k: [] for k in current}
                completed += 1
                break

    env.close()
    return {"case": case.name, "algo": case.algo, "reward": case.reward, "episodes": episodes}


def pick_archetypes(result: dict) -> dict[str, dict] | None:
    eps = result["episodes"]
    stable = [e for e in eps if e["so"] > 0.5 and e["sae"] > 0.5]
    drop = [e for e in eps if e["so"] > 0.5 and e["sae"] < 0.5]
    if not stable or not drop:
        return None
    stable_ep = stable[0]
    drop_ep = max(drop, key=lambda e: float(np.max(e["is_grasping"])))
    return {"stable": stable_ep, "drop": drop_ep}


def render_archetype(case_name: str, archetypes: dict[str, dict]) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    for col, (label, ep) in enumerate(archetypes.items()):
        T = len(ep["frames"])
        key_ids = sorted(set([0, max(0, T // 3), max(0, 2 * T // 3), T - 1]))
        for j, idx in enumerate(key_ids):
            axes[0, j].imshow(ep["frames"][idx])
            axes[0, j].set_title(f"{label} t={idx}")
            axes[0, j].axis("off")

    for row_idx, (label, ep) in enumerate(archetypes.items(), start=1):
        t = np.arange(len(ep["sim_reward"]))
        ax = axes[row_idx, 0]
        ax.plot(t, ep["success_step"], label="success")
        ax.plot(t, ep["is_grasping"], label="grasping")
        ax.set_title(f"{label}: success/grasp")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)

        ax = axes[row_idx, 1]
        ax.plot(t, ep["sim_reward"], label="sim_reward")
        ax.set_title(f"{label}: sim reward")

        ax = axes[row_idx, 2]
        ax.plot(t, ep["acp_base_reward"], label="acp_base")
        ax.plot(t, ep["acp_grasp_bonus"], label="grasp_bonus")
        ax.plot(t, ep["acp_total_reward"], label="acp_total")
        ax.set_title(f"{label}: acp rewards")
        ax.legend(fontsize=7)

        ax = axes[row_idx, 3]
        cumulative = np.cumsum(ep["acp_total_reward"])
        ax.plot(t, cumulative, label="cum_acp_total")
        ax.set_title(f"{label}: cum acp reward")

        for c in range(4):
            axes[row_idx, c].grid(True, alpha=0.25)

    fig.suptitle(f"Episode archetypes: {case_name}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{case_name}_archetypes.png", bbox_inches="tight")
    plt.close(fig)


def save_hdf5(case_name: str, archetypes: dict[str, dict]) -> None:
    with h5py.File(OUT_DIR / f"{case_name}_archetypes.h5", "w") as f:
        for label, ep in archetypes.items():
            grp = f.create_group(label)
            for key, value in ep.items():
                if isinstance(value, np.ndarray):
                    grp.create_dataset(key, data=value)
                else:
                    grp.attrs[key] = value


def main() -> None:
    summary: dict[str, Any] = {}
    for case in CASES:
        result = rollout_case(case)
        archetypes = pick_archetypes(result)
        if archetypes is None:
            summary[case.name] = {"status": "no_stable_or_drop_pair_found", "num_episodes": len(result["episodes"])}
            continue
        render_archetype(case.name, archetypes)
        save_hdf5(case.name, archetypes)
        summary[case.name] = {
            "status": "ok",
            "num_episodes": len(result["episodes"]),
            "stable": {"so": archetypes["stable"]["so"], "sae": archetypes["stable"]["sae"]},
            "drop": {"so": archetypes["drop"]["so"], "sae": archetypes["drop"]["sae"]},
        }
    (OUT_DIR / "archetype_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[OK] Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
