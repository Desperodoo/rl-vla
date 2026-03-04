"""Quick smoke test for train_imagination_rl.py with mock env."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["WANDB_DISABLED"] = "true"

import time
import torch
import numpy as np
import torch.optim as optim

from rlft.online.train_imagination_rl import (
    Args, create_imagination_env, evaluate_imagination, SimpleReplayBuffer,
)
from rlft.algorithms.online_rl.pld_sac import PLDSACAgent


def run_smoke_test():
    args = Args()
    args.use_mock = True
    args.agent_gpu = 0
    args.wm_gpu = 0
    args.vlm_gpu = 0

    device = torch.device("cuda:0")
    env, info = create_imagination_env(args)
    print(f"Env created: {info}")
    obs_dim = 4633
    act_dim = 7

    agent = PLDSACAgent(
        obs_dim=obs_dim, act_steps=1, action_dim=act_dim, action_scale=1.0,
        hidden_dims=[512, 512, 512], num_qs=5, gamma=0.95, tau=0.001,
        init_temperature=0.5, target_entropy=-3.5, log_std_init=-3.0,
        use_layer_norm=True, device="cuda:0",
    ).to(device)
    print(f"Agent: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M params")

    buf = SimpleReplayBuffer(10000, obs_dim, act_dim, "cuda:0")
    obs, _ = env.reset(seed=42)
    ep_count = 0
    ep_rew = 0.0

    actor_opt = optim.Adam(agent.actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(agent.critic.parameters(), lr=1e-4)
    temp_opt = optim.Adam([agent.log_alpha], lr=1e-4)

    t0 = time.time()
    for step in range(300):
        if step < 50:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                action = agent.select_action(obs_t, deterministic=False).cpu().numpy().squeeze(0)

        next_obs, reward, term, trunc, info_step = env.step(action)
        done = term or trunc
        buf.add(obs, action, reward, next_obs, done)
        ep_rew += reward
        obs = next_obs

        if done:
            ep_count += 1
            print(f"  Ep {ep_count}: rew={ep_rew:.3f}")
            ep_rew = 0.0
            obs, _ = env.reset()

        if step >= 50 and buf.size >= 64:
            batch = buf.sample(64)
            critic_opt.zero_grad()
            cl, _ = agent.compute_critic_loss(
                batch["obs"], batch["actions"], batch["next_obs"],
                batch["rewards"], batch["dones"],
            )
            cl.backward()
            critic_opt.step()

            actor_opt.zero_grad()
            al, _ = agent.compute_actor_loss(batch["obs"])
            al.backward()
            actor_opt.step()

            temp_opt.zero_grad()
            tl, _ = agent.compute_temperature_loss(batch["obs"])
            tl.backward()
            temp_opt.step()
            agent.update_target()

    elapsed = time.time() - t0
    print(f"300 steps in {elapsed:.1f}s ({300 / elapsed:.0f} steps/s)")

    eval_m = evaluate_imagination(env, agent, 2, "cuda:0")
    print(f"Eval: avg_reward={eval_m['avg_reward']:.3f}, p_yes_max={eval_m['avg_p_yes_max']:.3f}")
    env.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    run_smoke_test()
