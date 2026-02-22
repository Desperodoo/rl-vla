"""
PLD-SAC Training Script — Stage 1 (Probe) of the PLD framework.

Trains a lightweight residual SAC agent on top of a frozen ShortCut Flow
base policy.  The RL agent outputs additive residual actions
``a_delta ∈ [-ξ, +ξ]`` which are composed with the base policy's output
inside the ``ManiSkillResidualEnvWrapper``.

Key design points (PLD paper §4.1 + PLD sweep v1/v2 tuning):
    * Actor:  3×1024 MLP + Tanh, log_std_init = -5.0.
    * Critic: 3×1024 MLP + LayerNorm + Tanh, 5 Q-networks.
    * action_scale (ξ) = 0.3.
    * UTD ratio = 60  (DSRL sweep: most impactful parameter).
    * gamma = 0.99    (rewards long-term success retention).
    * target_entropy = -3.5  (avoids over-conservative collapse).
    * init_temperature = 0.1  (near-deterministic start preserves pretrained init).
    * learning_rate = 1e-4  (prevents Q-divergence under high UTD).
    * Pure online replay buffer (online_ratio=1.0).
    * Cal-QL critic pretraining: 1000 steps, alpha=0.0 (minimal bias).
    * Base policy probing at episode start (probing_alpha = 0.6).

Workflow::

    1. Load frozen base policy (ShortCut Flow).
    2. Collect *offline* demonstrations by rolling out base policy.
    3. Pretrain critic using Cal-QL objective on offline data.
    4. Online RL loop:
       a. Probe base policy for ``probe_steps`` at episode start.
       b. Collect residual RL transitions into online buffer.
       c. Sample mixed batches (50/50) for SAC updates.
    5. Periodically evaluate and checkpoint.

Usage::

    python -m rlft.online.train_pld \\
        --env_id LiftPegUpright-v1 \\
        --checkpoint /path/to/shortcut_flow_best.pt \\
        --total_timesteps 500000

    # Custom hyperparameters
    python -m rlft.online.train_pld \\
        --action_scale 0.3 \\
        --calql_pretrain_steps 10000 \\
        --num_envs 50
"""

ALGO_NAME = "PLD-SAC"

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
import torch.nn.functional as F
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
from rlft.algorithms.online_rl.pld_sac import PLDSACAgent
from rlft.buffers.pld_buffer import PLDReplayBuffer
from rlft.envs import make_eval_envs, evaluate
from rlft.envs.pld_env import ManiSkillResidualEnvWrapper
from rlft.utils.flow_wrapper import ShortCutFlowWrapper, load_shortcut_flow_policy
from rlft.utils.checkpoint import save_checkpoint
from rlft.online._flow_helpers import (
    make_flow_train_envs,
    make_flow_eval_envs,
    FlowVecEnvAdapter,
    extract_success,
)


# =====================================================================
# Arguments
# =====================================================================

@dataclass
class Args:
    """PLD-SAC training arguments."""

    # ----- experiment -----
    exp_name: Optional[str] = None
    seed: int = 42
    cuda: bool = True
    track: bool = True
    wandb_project: str = "PLD-SAC"
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

    # ----- PLD-SAC hyper-parameters -----
    action_scale: float = 0.3
    """Residual action bounds [-ξ, +ξ].  PLD sweep: ξ=0.3 optimal (at_end=0.68)."""
    total_timesteps: int = 500_000
    learning_rate: float = 1e-4
    """PLD sweep: lr=1e-4 prevents Q-divergence under high UTD (at_end 0.20→0.82)."""
    online_buffer_size: int = 500_000
    offline_buffer_size: int = 200_000
    batch_size: int = 1024
    gamma: float = 0.99
    """PLD sweep: γ=0.99 rewards long-term success retention (at_end 0.20→0.56)."""
    tau: float = 0.005
    utd_ratio: int = 60
    """DSRL sweep: UTD ratio is the most impactful parameter. 60-80 optimal."""
    init_temperature: float = 0.1
    """PLD sweep: near-deterministic start preserves pretrained init (at_end=0.78)."""
    target_entropy: float = -3.5
    """DSRL sweep: auto (-56 for 56-dim action) is over-conservative;
    -3.5 balances exploration and exploitation."""
    log_std_init: float = -5.0
    """DSRL sweep: conservative initial exploration (std≈0.007)."""
    max_grad_norm: float = 10.0
    online_ratio: float = 1.0
    """PLD sweep: pure online replay (at_end 0.20→0.56). 0.0 = all offline."""

    # ----- network architecture -----
    num_layers: int = 3
    layer_size: int = 1024
    """PLD sweep: 3×1024 reduces overparameterization (at_end 0.20→0.72)."""
    num_qs: int = 5
    """PLD sweep: num_qs=5 balances pessimism vs exploration (at_end 0.20→0.72)."""
    use_layer_norm: bool = True

    # ----- offline data & Cal-QL pretraining -----
    offline_demo_episodes: int = 50
    """PLD sweep: fewer demos reduce offline distribution interference (at_end=0.72)."""
    calql_pretrain_steps: int = 1000
    """PLD sweep: minimal critic warm-up avoids excessive offline bias (at_end=0.64)."""
    calql_alpha: float = 0.0
    """PLD sweep: conservative loss hurts online finetuning (at_end 0.20→0.66)."""

    # ----- base policy probing -----
    probe_steps: int = 5
    """Number of RL steps at episode start using ONLY the base policy."""
    probing_alpha: float = 0.6
    """Probability of accepting base-policy-only probe at episode start.
    If uniform random < probing_alpha, probe; otherwise skip probing."""

    # ----- logging / eval / saving -----
    log_freq: int = 100
    eval_freq: int = 10_000
    num_eval_episodes: int = 50
    save_freq: int = 50_000
    capture_video: bool = False


# =====================================================================
# Env helpers — delegated to rlft.online._flow_helpers
# =====================================================================

_make_train_envs = make_flow_train_envs
_make_eval_envs = make_flow_eval_envs


# =====================================================================
# Evaluation agent wrapper  (converts PLDSACAgent → evaluate() API)
# =====================================================================

class PLDEvalAgentWrapper:
    """Adapts ``PLDSACAgent`` for ``rlft.envs.evaluate()``.

    ``evaluate()`` calls ``agent.get_action(obs)`` and expects
    ``(B, act_horizon, action_dim)`` *real* actions.

    Here we:
    1. Encode the observation via the same visual encoder.
    2. Get residual a_delta from actor (deterministic).
    3. Get base policy output a_base (deterministic, zero noise).
    4. Compose a_bar = clamp(a_base + a_delta, -1, 1).
    5. Return a_bar reshaped to ``(B, act_steps, action_dim)``.
    """

    def __init__(
        self,
        agent: PLDSACAgent,
        base_policy: ShortCutFlowWrapper,
        visual_encoder: Optional[nn.Module],
        include_rgb: bool,
        obs_horizon: int,
        act_steps: int,
        action_dim: int,
        action_scale: float,
        device: str,
    ):
        self.agent = agent
        self.base_policy = base_policy
        self.visual_encoder = visual_encoder
        self.include_rgb = include_rgb
        self.obs_horizon = obs_horizon
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.action_scale = action_scale
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
        # 1. Residual action from actor
        a_delta = self.agent.select_action(obs_cond, deterministic=deterministic)
        a_delta_3d = a_delta.view(-1, self.act_steps, self.action_dim)

        # 2. Base policy output (deterministic = zero noise)
        zero_noise = torch.zeros_like(a_delta_3d)
        a_base = self.base_policy(
            obs_cond, zero_noise, return_numpy=False, act_steps=self.act_steps,
        )

        # 3. Align temporal dims (flow may return fewer steps than act_steps)
        n_actual = a_base.shape[1]
        if n_actual < self.act_steps:
            a_delta_3d = a_delta_3d[:, :n_actual, :]

        # 4. Compose
        a_bar = torch.clamp(a_base + a_delta_3d, -1.0, 1.0)
        return a_bar

    def eval(self):
        self.agent.eval()
        return self

    def train(self, mode=True):
        self.agent.train(mode)
        return self


# =====================================================================
# SB3-VecEnv adapter — delegated to rlft.online._flow_helpers
# =====================================================================

_VecEnvAdapter = FlowVecEnvAdapter


# =====================================================================
# Offline data collection (base policy rollout)
# =====================================================================

def _collect_offline_demos(
    env_adapter: _VecEnvAdapter,
    buffer: PLDReplayBuffer,
    num_episodes: int,
):
    """Roll out the base policy (zero residual) and store transitions as offline data.

    Only stores transitions from **successful** episodes, as PLD-SAC uses
    demonstrations from a pre-trained VLA.  If the base policy has low
    success rate, we store all episodes but prefer successful ones.

    Note on auto-reset: ManiSkill3 auto-resets on done=True, so ``next_obs``
    for the terminal step is already the *new* episode's observation.  This is
    safe because ``done=True`` makes the TD-target ignore ``next_obs``.
    """
    print(f"[Offline Demo] Collecting {num_episodes} episodes with base policy …")
    obs, _ = env_adapter.reset()
    eps_collected = 0
    ep_rews = []
    ep_successes = []
    cur_rew = np.zeros(env_adapter.num_envs)

    # Temporary storage for current episodes
    ep_buffers = [[] for _ in range(env_adapter.num_envs)]

    while eps_collected < num_episodes:
        # Zero residual → base policy only
        zero_action = np.zeros(
            (env_adapter.num_envs, env_adapter.action_space.shape[0]),
            dtype=np.float32,
        )
        next_obs, rew, done, _term, _trunc, info = env_adapter.step(zero_action)

        cur_rew += rew
        for i in range(env_adapter.num_envs):
            ep_buffers[i].append((obs[i].copy(), zero_action[i].copy(),
                                  rew[i], next_obs[i].copy(), float(done[i])))

            if done[i]:
                # Store all transitions from this episode
                for (o, a, r, no, d) in ep_buffers[i]:
                    buffer.add_offline_single(o, a, r, no, d)

                ep_rews.append(cur_rew[i])
                success = _extract_success(info, env_adapter.num_envs)
                ep_successes.append(success[i])
                cur_rew[i] = 0.0
                ep_buffers[i] = []
                eps_collected += 1

                if eps_collected >= num_episodes:
                    break

        obs = next_obs

    avg_rew = np.mean(ep_rews) if ep_rews else 0.0
    avg_sr = np.mean(ep_successes) if ep_successes else 0.0
    n_transitions = buffer.offline_size
    print(f"[Offline Demo] Done — {eps_collected} episodes, "
          f"{n_transitions} transitions, avg_reward={avg_rew:.2f}, "
          f"success_rate={avg_sr:.2%}")
    return eps_collected


# =====================================================================
# Info extraction helper — delegated to rlft.online._flow_helpers
# =====================================================================

_extract_success = extract_success


# =====================================================================
# Inference-only checkpoint (no critic / critic_target)
# =====================================================================

def _save_inference_checkpoint(path, agent, visual_encoder, args, total_steps):
    """Save lightweight checkpoint with only actor weights (no critic).

    For best.pt / final.pt used at inference time. ``critic`` and
    ``critic_target`` are excluded, reducing size from ~777 MB to ~40 MB
    (with default 3×2048 architecture).
    """
    agent_sd = {
        k: v for k, v in agent.state_dict().items()
        if not k.startswith(("critic.", "critic_target."))
    }
    ckpt = {
        "agent": agent_sd,
        "total_steps": total_steps,
        "config": {k: v for k, v in vars(args).items()},
    }
    if visual_encoder is not None:
        ckpt["visual_encoder"] = visual_encoder.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


# =====================================================================
# Main
# =====================================================================

def main():
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = (
            f"pld-sac-{args.env_id}-{args.num_envs}envs-"
            f"utd{args.utd_ratio}-scale{args.action_scale}-seed{args.seed}"
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
    print("[1/7] Loading pretrained ShortCut Flow policy …")
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
    print("[2/7] Creating environments …")
    raw_train_env = _make_train_envs(args)

    wrapped_train_env = ManiSkillResidualEnvWrapper(
        env=raw_train_env,
        base_policy=base_policy,
        visual_encoder=visual_encoder,
        action_scale=args.action_scale,
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
    print(f"  Action space (residual): {train_adapter.action_space}")
    print(f"  Observation space:       {train_adapter.observation_space}")

    # =================================================================
    # 3. Create agent + optimizers
    # =================================================================
    print("[3/7] Creating PLDSACAgent …")
    hidden_dims = [args.layer_size] * args.num_layers
    residual_dim = args.act_steps * args.action_dim

    agent = PLDSACAgent(
        obs_dim=obs_dim,
        act_steps=args.act_steps,
        action_dim=args.action_dim,
        action_scale=args.action_scale,
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
    eval_wrapper = PLDEvalAgentWrapper(
        agent=agent,
        base_policy=base_policy,
        visual_encoder=visual_encoder,
        include_rgb=include_rgb,
        obs_horizon=args.obs_horizon,
        act_steps=args.act_steps,
        action_dim=args.action_dim,
        action_scale=args.action_scale,
        device=str(device),
    )

    # =================================================================
    # 4. Replay buffer + offline demo collection
    # =================================================================
    print("[4/7] Creating replay buffer & collecting offline demos …")
    buffer = PLDReplayBuffer(
        online_capacity=args.online_buffer_size,
        offline_capacity=args.offline_buffer_size,
        obs_dim=obs_dim,
        action_dim=residual_dim,
        device=str(device),
    )

    if args.offline_demo_episodes > 0:
        _collect_offline_demos(train_adapter, buffer, args.offline_demo_episodes)

    # =================================================================
    # 5. Cal-QL critic pretraining
    # =================================================================
    if args.calql_pretrain_steps > 0 and buffer.offline_size > 0:
        print("[5/7] Cal-QL critic pretraining …")
        agent.pretrain_critic_calql(
            buffer=buffer,
            steps=args.calql_pretrain_steps,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            calql_alpha=args.calql_alpha,
            device=str(device),
        )
    else:
        print("[5/7] Skipping Cal-QL pretraining (no offline data).")

    # =================================================================
    # 6. Baseline evaluation (before RL training)
    # =================================================================
    print("[6/7] Baseline evaluation …")
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
    # 7. Online RL Training Loop
    # =================================================================
    print("[7/7] Online RL training …")
    obs, _ = train_adapter.reset()
    total_steps = 0
    probe_steps_total = 0  # track wasted probe steps separately
    best_success = 0.0
    ep_rews = defaultdict(float)
    ep_successes: list = []
    training_metrics = defaultdict(list)

    # Per-env episode step counter (for probing)
    ep_step_counters = np.zeros(train_adapter.num_envs, dtype=np.int32)

    pbar = tqdm(total=args.total_timesteps, desc="PLD-SAC")

    while total_steps < args.total_timesteps:

        # ---- Base policy probing (per-env independent) ----
        # At the start of each episode, optionally run probe_steps using
        # only the base policy (zero residual).  This "probes" the base
        # policy to skip trivial initial states where it already performs
        # well.  The probe transitions are NOT stored in the replay buffer.
        #
        # Per-env: only probe envs whose ep_step_counter < probe_steps
        # AND whose per-env random draw < probing_alpha.
        # Probe steps are NOT counted towards total_steps (effective RL steps).
        in_probe_phase = ep_step_counters < args.probe_steps
        probe_draw = np.random.random(train_adapter.num_envs) < args.probing_alpha
        should_probe = in_probe_phase & probe_draw  # shape: (num_envs,)

        if should_probe.any() and not (~should_probe).any():
            # ALL envs are in probe phase — step base only, don't count
            next_obs, rew, done, term, trunc, info = train_adapter.step_base_only()

            for i in range(train_adapter.num_envs):
                if done[i]:
                    ep_step_counters[i] = 0
                    ep_rews[i] = 0.0
                else:
                    ep_step_counters[i] += 1

            obs = next_obs
            probe_steps_total += train_adapter.num_envs
            # NOTE: probe steps are NOT added to total_steps
            continue  # Don't store probe transitions in buffer

        # ---- Collect RL transitions ----
        agent.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(device)
            residual = agent.select_action(obs_t, deterministic=False).cpu().numpy()

        next_obs, rew, done, term, trunc, info = train_adapter.step(residual)

        # Store in online buffer (using residual actions, not combined)
        buffer.add_online(obs, residual, rew, next_obs, done.astype(np.float32))

        # Episode stats
        for i in range(train_adapter.num_envs):
            ep_rews[i] += rew[i]
            if done[i]:
                ep_rews[i] = 0.0
                ep_step_counters[i] = 0  # reset probe counter

        # Track success at episode boundaries
        if done.any():
            success_vals = _extract_success(info, train_adapter.num_envs)
            ep_successes.extend(success_vals.tolist())

        obs = next_obs
        total_steps += train_adapter.num_envs
        pbar.update(train_adapter.num_envs)

        # Increment step counters for non-done envs
        for i in range(train_adapter.num_envs):
            if not done[i]:
                ep_step_counters[i] += 1

        # ---- Training updates (UTD ratio) ----
        if buffer.total_size >= args.batch_size:
            agent.train()
            for _ in range(args.utd_ratio):
                batch = buffer.sample_mixed(
                    args.batch_size, online_ratio=args.online_ratio,
                )

                # Critic update
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

                # Actor update
                actor_optimizer.zero_grad()
                actor_loss, a_met = agent.compute_actor_loss(batch["obs"])
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                actor_optimizer.step()

                for k, v in a_met.items():
                    training_metrics[f"actor/{k}"].append(v)

                # Temperature update
                temp_optimizer.zero_grad()
                temp_loss, t_met = agent.compute_temperature_loss(batch["obs"])
                temp_loss.backward()
                temp_optimizer.step()

                for k, v in t_met.items():
                    training_metrics[f"temp/{k}"].append(v)

                # Target network update
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

            # Buffer stats
            writer.add_scalar("buffer/online_size", buffer.online_size, total_steps)
            writer.add_scalar("buffer/offline_size", buffer.offline_size, total_steps)
            log_dict["buffer/online_size"] = buffer.online_size
            log_dict["buffer/offline_size"] = buffer.offline_size

            # Probe overhead
            writer.add_scalar("probe/total_probe_steps", probe_steps_total, total_steps)
            log_dict["probe/total_probe_steps"] = probe_steps_total

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
                _save_inference_checkpoint(
                    f"{log_dir}/checkpoints/best.pt",
                    agent, visual_encoder, args, total_steps,
                )
                print(f"  New best! ({sr:.2%})")

        # ---- checkpoint ----
        if total_steps % args.save_freq < train_adapter.num_envs:
            save_checkpoint(
                path=f"{log_dir}/checkpoints/step_{total_steps}.pt",
                agent=agent,
                visual_encoder=visual_encoder,
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
            "online": buffer.online_size,
            "offline": buffer.offline_size,
        })

    pbar.close()

    # ---- final save ----
    _save_inference_checkpoint(
        f"{log_dir}/checkpoints/final.pt",
        agent, visual_encoder, args, total_steps,
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
