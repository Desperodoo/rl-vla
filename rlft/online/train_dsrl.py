"""
DSRL-SAC Training Script — integrated into the rlft framework.

DSRL-SAC trains a standard SAC agent **in the noise space** of a frozen
ShortCut Flow policy. The environment wrapper decodes noise → real actions
internally.

Key design choices (tuned via sweep on LiftPegUpright-v1):
    * Actor: 3×2048 MLP + Tanh, log_std_init=−5.
    * Critic: 3×2048 MLP + Tanh, 10 Q-networks.
    * UTD = 60.
    * action_magnitude = 2.5.
    * gamma = 0.95 (matches 100-step episode horizon).
    * target_entropy = −3.5 (balanced exploration in noise space).
    * num_seed_steps = 0 (pretrained policy provides sufficient initial exploration).
    * Buffer: standard MDP (env wrapper handles action chunking).

Usage::

    python -m rlft.online.train_dsrl \\
        --env_id LiftPegUpright-v1 \\
        --checkpoint /path/to/shortcut_flow_best.pt \\
        --total_timesteps 1000000

    # Custom hyperparameters
    python -m rlft.online.train_dsrl \\
        --utd_ratio 80 \\
        --action_magnitude 3.0 \\
        --num_envs 100
"""

ALGO_NAME = "DSRL-SAC"

import os
import random
import time
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import tyro

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# rlft imports
from rlft.networks import PlainConv, ShortCutVelocityUNet1D
from rlft.algorithms.online_rl.dsrl_sac import DSRLSACAgent
from rlft.buffers.dsrl_buffer import DSRLReplayBuffer
from rlft.envs import make_eval_envs, evaluate
from rlft.envs.dsrl_env import ManiSkillFlowEnvWrapper
from rlft.utils.flow_wrapper import ShortCutFlowWrapper, load_shortcut_flow_policy
from rlft.utils.checkpoint import save_checkpoint


# =====================================================================
# Arguments
# =====================================================================

@dataclass
class Args:
    """DSRL-SAC training arguments."""

    # ----- experiment -----
    exp_name: Optional[str] = None
    seed: int = 42
    cuda: bool = True
    track: bool = True
    wandb_project: str = "DSRL-SAC"
    wandb_entity: Optional[str] = None

    # ----- environment -----
    env_id: str = "LiftPegUpright-v1"
    num_envs: int = 50
    num_eval_envs: int = 50
    max_episode_steps: int = 100
    control_mode: str = "pd_ee_delta_pose"
    obs_mode: str = "rgb"
    sim_backend: str = "physx_cuda"
    reward_mode: str = "dense"

    # ----- pretrained checkpoint -----
    checkpoint: str = ""
    """Path to pretrained ShortCut Flow .pt file."""
    use_ema: bool = True

    # ----- model dims -----
    obs_horizon: int = 2
    pred_horizon: int = 8
    act_steps: int = 8
    action_dim: int = 7
    state_dim: int = 0
    """0 = auto-infer from checkpoint."""
    visual_feature_dim: int = 256

    # ----- DSRL-SAC hyper-parameters -----
    action_magnitude: float = 2.5
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.95
    tau: float = 0.005
    utd_ratio: int = 60
    num_seed_steps: int = 0
    """No warmup needed — pretrained policy already provides sufficient initial exploration quality."""
    init_temperature: float = 0.5
    target_entropy: float = -3.5
    log_std_init: float = -5.0
    max_grad_norm: float = 10.0

    # ----- network architecture -----
    num_layers: int = 3
    layer_size: int = 2048
    num_qs: int = 10
    use_layer_norm: bool = True

    # ----- logging / eval / saving -----
    log_freq: int = 100
    eval_freq: int = 10_000
    num_eval_episodes: int = 50
    save_freq: int = 50_000
    capture_video: bool = False


# =====================================================================
# Env helpers
# =====================================================================

def _make_train_envs(args: Args):
    """Create GPU-vectorized training environments.

    **Critical**: uses the same FrameStack + ManiSkillVectorEnv wrapping as
    the eval envs so that obs_history is correctly managed by FrameStack
    (including proper per-env reset on auto-reset).  Without FrameStack the
    wrapper's manual obs_history roll contaminates observations across
    episode boundaries.
    """
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    env_kwargs = dict(
        obs_mode="rgbd" if "rgb" in args.obs_mode else "state",
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    wrappers = [FlattenRGBDObservationWrapper] if "rgb" in args.obs_mode else []

    return make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        env_kwargs=env_kwargs,
        other_kwargs=dict(obs_horizon=args.obs_horizon),
        video_dir=None,
        wrappers=wrappers,
    )


def _make_eval_envs(args: Args):
    """Create eval envs using rlft.envs.make_eval_envs (with FrameStack)."""
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    env_kwargs = dict(
        control_mode=args.control_mode,
        obs_mode="rgbd" if "rgb" in args.obs_mode else "state",
        render_mode="rgb_array",
        reward_mode=args.reward_mode,
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    wrappers = [FlattenRGBDObservationWrapper] if "rgb" in args.obs_mode else []

    return make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        sim_backend=args.sim_backend,
        env_kwargs=env_kwargs,
        other_kwargs=dict(obs_horizon=args.obs_horizon),
        video_dir=None,
        wrappers=wrappers,
    )


# =====================================================================
# Evaluation agent wrapper  (converts DSRLSACAgent → evaluate() API)
# =====================================================================

class DSRLEvalAgentWrapper:
    """Adapts ``DSRLSACAgent`` for ``rlft.envs.evaluate()``.

    ``evaluate()`` calls ``agent.get_action(obs)`` and expects
    ``(B, act_horizon, action_dim)`` *real* actions.

    Here we:
    1. Encode the observation via the same visual encoder.
    2. Sample noise w from the actor (deterministic = 0 noise = pretrained
       policy output).
    3. Decode w → real actions using the flow wrapper.
    """

    def __init__(
        self,
        agent: DSRLSACAgent,
        base_policy: ShortCutFlowWrapper,
        visual_encoder: Optional[nn.Module],
        include_rgb: bool,
        obs_horizon: int,
        act_steps: int,
        action_dim: int,
        device: str,
    ):
        self.agent = agent
        self.base_policy = base_policy
        self.visual_encoder = visual_encoder
        self.include_rgb = include_rgb
        self.obs_horizon = obs_horizon
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.device = device

    def _encode_obs(self, obs) -> torch.Tensor:
        """Encode stacked observation to flat features."""
        from rlft.datasets.data_utils import encode_observations
        return encode_observations(
            obs_seq=obs,
            visual_encoder=self.visual_encoder,
            include_rgb=self.include_rgb,
            device=self.device,
        )

    @torch.no_grad()
    def get_action(self, obs, deterministic=True, **kwargs):
        """Return real actions ``(B, act_steps, action_dim)``."""
        obs_cond = self._encode_obs(obs)
        noise = self.agent.select_action(obs_cond, deterministic=deterministic)
        noise_3d = noise.view(-1, self.act_steps, self.action_dim)
        actions = self.base_policy(obs_cond, noise_3d, return_numpy=False, act_steps=self.act_steps)
        return actions

    def eval(self):
        self.agent.eval()
        return self

    def train(self, mode=True):
        self.agent.train(mode)
        return self


# =====================================================================
# SB3-VecEnv adapter (thin wrapper so wrapped env plays nice with rlft)
# =====================================================================

class _VecEnvAdapter:
    """Minimal adapter: ``ManiSkillFlowEnvWrapper`` → numpy-based VecEnv."""

    def __init__(self, env: ManiSkillFlowEnvWrapper):
        self.env = env
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        obs, info = self.env.reset()
        return self._t2n(obs), info

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        done = term | trunc
        return (
            self._t2n(obs),
            self._t2n(rew),
            self._t2n(done),
            self._t2n(term),
            self._t2n(trunc),
            info,
        )

    @staticmethod
    def _t2n(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    def close(self):
        self.env.close()


# =====================================================================
# Initial rollout (zero-noise = pretrained policy output)
# =====================================================================

def _collect_warmup(env_adapter: _VecEnvAdapter, buffer: DSRLReplayBuffer, n_steps: int):
    """Collect *n_steps* transitions using zero noise (= pretrained policy)."""
    print(f"[Warmup] Collecting {n_steps} transitions with zero noise …")
    obs, _ = env_adapter.reset()
    collected = 0
    ep_rews: list = []
    cur_rew = np.zeros(env_adapter.num_envs)

    while collected < n_steps:
        action = np.zeros((env_adapter.num_envs, env_adapter.action_space.shape[0]), dtype=np.float32)
        next_obs, rew, done, _term, _trunc, _info = env_adapter.step(action)

        buffer.add(obs, action, rew, next_obs, done.astype(np.float32))

        cur_rew += rew
        for i in range(env_adapter.num_envs):
            if done[i]:
                ep_rews.append(cur_rew[i])
                cur_rew[i] = 0.0

        obs = next_obs
        collected += env_adapter.num_envs

    avg = np.mean(ep_rews) if ep_rews else 0.0
    print(f"[Warmup] Done — {collected} transitions, "
          f"{len(ep_rews)} episodes, avg_reward={avg:.2f}")
    return collected


# =====================================================================
# Info extraction helper
# =====================================================================

def _extract_success(info, num_envs: int) -> np.ndarray:
    """Extract per-env success flags from ManiSkill3 info dict.

    Handles both ``ManiSkillVectorEnv`` format (``final_info``) and
    the legacy per-step ``success`` tensor.
    """
    if not isinstance(info, dict):
        return np.zeros(num_envs)

    # 1. ManiSkillVectorEnv format: info["final_info"]["episode"]["success_once"]
    final_info = info.get("final_info")
    if isinstance(final_info, dict) and "episode" in final_info:
        so = final_info["episode"].get("success_once")
        if so is not None:
            if isinstance(so, torch.Tensor):
                return so.float().cpu().numpy()
            return np.asarray(so, dtype=np.float32)

    # 2. Fallback: per-step success tensor
    success = info.get("success")
    if success is not None:
        if isinstance(success, torch.Tensor):
            return success.float().cpu().numpy()
        if isinstance(success, np.ndarray):
            return success.astype(np.float32)

    return np.zeros(num_envs)


# =====================================================================
# Main
# =====================================================================

def main():
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = (
            f"dsrl-sac-{args.env_id}-{args.num_envs}envs-"
            f"utd{args.utd_ratio}-mag{args.action_magnitude}-seed{args.seed}"
        )
    run_name = f"{args.exp_name}__{int(time.time())}"
    log_dir = f"runs/{run_name}"
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)

    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ---- seed ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # ---- wandb ----
    if args.track and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)

    # =================================================================
    # 1. Load base policy
    # =================================================================
    print("[1/5] Loading pretrained ShortCut Flow policy …")
    include_rgb = "rgb" in args.obs_mode

    base_policy, visual_encoder, inferred_state_dim = load_shortcut_flow_policy(
        checkpoint_path=args.checkpoint,
        visual_encoder_class=PlainConv if include_rgb else None,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_dim=args.action_dim,
        visual_feature_dim=args.visual_feature_dim,
        include_rgb=include_rgb,
        use_ema=args.use_ema,
        device=str(device),
    )
    state_dim = args.state_dim if args.state_dim > 0 else inferred_state_dim
    visual_dim = args.visual_feature_dim if include_rgb else 0
    obs_dim = args.obs_horizon * (visual_dim + state_dim)
    print(f"  state_dim={state_dim}, visual_dim={visual_dim}, obs_dim={obs_dim}")

    # =================================================================
    # 2. Create environments
    # =================================================================
    print("[2/5] Creating environments …")
    raw_train_env = _make_train_envs(args)

    wrapped_train_env = ManiSkillFlowEnvWrapper(
        env=raw_train_env,
        base_policy=base_policy,
        visual_encoder=visual_encoder,
        action_magnitude=args.action_magnitude,
        act_steps=args.act_steps,
        action_dim=args.action_dim,
        state_dim=state_dim,
        visual_feature_dim=args.visual_feature_dim,
        obs_horizon=args.obs_horizon,
        include_rgb=include_rgb,
        device=str(device),
    )
    train_adapter = _VecEnvAdapter(wrapped_train_env)

    eval_envs = _make_eval_envs(args)

    print(f"  Train: {args.num_envs} envs | Eval: {args.num_eval_envs} envs")
    print(f"  Action space (noise): {train_adapter.action_space}")
    print(f"  Observation space:    {train_adapter.observation_space}")

    # =================================================================
    # 3. Create agent + optimizers
    # =================================================================
    print("[3/5] Creating DSRLSACAgent …")
    hidden_dims = [args.layer_size] * args.num_layers

    agent = DSRLSACAgent(
        obs_dim=obs_dim,
        act_steps=args.act_steps,
        action_dim=args.action_dim,
        action_magnitude=args.action_magnitude,
        hidden_dims=hidden_dims,
        num_qs=args.num_qs,
        gamma=args.gamma,
        tau=args.tau,
        init_temperature=args.init_temperature,
        target_entropy=args.target_entropy,
        log_std_init=args.log_std_init,
        use_layer_norm=args.use_layer_norm,
        device=str(device),
    ).to(device)

    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.learning_rate)
    temp_optimizer = optim.Adam([agent.log_alpha], lr=args.learning_rate)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Agent parameters: {total_params / 1e6:.2f} M")

    # eval wrapper
    eval_wrapper = DSRLEvalAgentWrapper(
        agent=agent,
        base_policy=base_policy,
        visual_encoder=visual_encoder,
        include_rgb=include_rgb,
        obs_horizon=args.obs_horizon,
        act_steps=args.act_steps,
        action_dim=args.action_dim,
        device=str(device),
    )

    # =================================================================
    # 4. Replay buffer + warmup
    # =================================================================
    noise_dim = args.act_steps * args.action_dim
    buffer = DSRLReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=obs_dim,
        noise_dim=noise_dim,
        device=str(device),
    )

    if args.num_seed_steps > 0:
        _collect_warmup(train_adapter, buffer, args.num_seed_steps)

    # =================================================================
    # 5. Pre-training evaluation (baseline)
    # =================================================================
    print("[4/5] Baseline evaluation …")
    eval_wrapper.eval()
    baseline_metrics = evaluate(
        args.num_eval_episodes, eval_wrapper, eval_envs, device, args.sim_backend,
    )
    for k in baseline_metrics:
        baseline_metrics[k] = np.mean(baseline_metrics[k])
        writer.add_scalar(f"eval/{k}", baseline_metrics[k], 0)
        print(f"  {k}: {baseline_metrics[k]:.4f}")

    if args.track and HAS_WANDB:
        wandb.log({f"baseline/{k}": v for k, v in baseline_metrics.items()}, step=0)

    # =================================================================
    # 6. Training loop
    # =================================================================
    print("[5/5] Training …")
    obs, _ = train_adapter.reset()
    total_steps = 0
    best_success = 0.0
    ep_rews = defaultdict(float)
    ep_successes: list = []
    training_metrics = defaultdict(list)

    pbar = tqdm(total=args.total_timesteps, desc="DSRL-SAC")

    while total_steps < args.total_timesteps:
        # ---- collect ----
        agent.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(device)
            if total_steps < args.num_seed_steps:
                noise = np.zeros(
                    (train_adapter.num_envs, noise_dim), dtype=np.float32,
                )
            else:
                noise = agent.select_action(obs_t, deterministic=False).cpu().numpy()

        next_obs, rew, done, term, trunc, info = train_adapter.step(noise)
        buffer.add(obs, noise, rew, next_obs, done.astype(np.float32))

        # episode stats
        for i in range(train_adapter.num_envs):
            ep_rews[i] += rew[i]
            if done[i]:
                ep_rews[i] = 0.0

        # Track success at episode boundaries (ManiSkillVectorEnv format)
        if done.any():
            success_vals = _extract_success(info, train_adapter.num_envs)
            ep_successes.extend(success_vals.tolist())

        obs = next_obs
        total_steps += train_adapter.num_envs
        pbar.update(train_adapter.num_envs)

        # ---- training updates ----
        if total_steps >= args.num_seed_steps and buffer.size >= args.batch_size:
            agent.train()
            for _ in range(args.utd_ratio):
                batch = buffer.sample(args.batch_size)

                # critic
                critic_optimizer.zero_grad()
                critic_loss, c_met = agent.compute_critic_loss(
                    batch["obs"], batch["actions"], batch["next_obs"],
                    batch["rewards"], batch["dones"],
                )
                critic_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                critic_optimizer.step()

                for k, v in c_met.items():
                    training_metrics[f"critic/{k}"].append(v)

                # actor
                actor_optimizer.zero_grad()
                actor_loss, a_met = agent.compute_actor_loss(batch["obs"])
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                actor_optimizer.step()

                for k, v in a_met.items():
                    training_metrics[f"actor/{k}"].append(v)

                # temperature
                temp_optimizer.zero_grad()
                temp_loss, t_met = agent.compute_temperature_loss(batch["obs"])
                temp_loss.backward()
                temp_optimizer.step()

                for k, v in t_met.items():
                    training_metrics[f"temp/{k}"].append(v)

                # target update
                agent.update_target()

        # ---- logging ----
        if total_steps % args.log_freq == 0 and training_metrics:
            log_dict = {}
            for mk, mv in training_metrics.items():
                val = np.mean(mv)
                writer.add_scalar(f"train/{mk}", val, total_steps)
                log_dict[f"train/{mk}"] = val
            if ep_successes:
                sr = np.mean(ep_successes[-100:])
                writer.add_scalar("train/success_rate", sr, total_steps)
                log_dict["train/success_rate"] = sr
            if args.track and HAS_WANDB:
                wandb.log(log_dict, step=total_steps)
            training_metrics.clear()

        # ---- eval ----
        if total_steps % args.eval_freq < train_adapter.num_envs:
            eval_wrapper.eval()
            eval_met = evaluate(
                args.num_eval_episodes, eval_wrapper, eval_envs, device, args.sim_backend,
            )
            print(f"\n[Step {total_steps}] Eval:")
            eval_log = {}
            for k in eval_met:
                eval_met[k] = np.mean(eval_met[k])
                writer.add_scalar(f"eval/{k}", eval_met[k], total_steps)
                eval_log[f"eval/{k}"] = eval_met[k]
                print(f"  {k}: {eval_met[k]:.4f}")

            if args.track and HAS_WANDB:
                wandb.log(eval_log, step=total_steps)

            sr = eval_met.get("success_once", 0)
            if sr > best_success:
                best_success = sr
                save_checkpoint(
                    path=f"{log_dir}/checkpoints/best.pt",
                    agent=agent,
                    args=args,
                    total_steps=total_steps,
                    save_args_json=False,
                )
                print(f"  New best! ({sr:.2%})")

        # ---- checkpoint ----
        if total_steps % args.save_freq < train_adapter.num_envs:
            save_checkpoint(
                path=f"{log_dir}/checkpoints/step_{total_steps}.pt",
                agent=agent,
                args=args,
                total_steps=total_steps,
                save_args_json=False,
                extra={
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                    "temp_optimizer": temp_optimizer.state_dict(),
                },
            )

        pbar.set_postfix({
            "success": f"{np.mean(ep_successes[-100:]) if ep_successes else 0:.0%}",
            "buf": buffer.size,
        })

    pbar.close()

    # ---- final save ----
    save_checkpoint(
        path=f"{log_dir}/checkpoints/final.pt",
        agent=agent,
        args=args,
        total_steps=total_steps,
        save_args_json=False,
    )

    train_adapter.close()
    eval_envs.close()
    writer.close()

    if args.track and HAS_WANDB:
        wandb.log({"final/best_success_rate": best_success})
        wandb.finish()

    print(f"\nDone. Best success rate: {best_success:.2%}")


if __name__ == "__main__":
    main()
