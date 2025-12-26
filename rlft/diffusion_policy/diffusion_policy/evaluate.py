from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from mani_skill.utils import common

def evaluate(
    n: int, 
    agent, 
    eval_envs, 
    device, 
    sim_backend: str, 
    progress_bar: bool = True,
    agent_kwargs: dict = None,
):
    """Evaluate agent in environment.
    
    Args:
        n: Number of episodes to evaluate
        agent: Agent with get_action() method
        eval_envs: Evaluation environments
        device: Torch device
        sim_backend: Simulation backend ('physx_cpu' or 'physx_cuda')
        progress_bar: Whether to show progress bar
        agent_kwargs: Additional kwargs to pass to agent.get_action()
            For ReinFlowAgent: {'deterministic': True, 'use_ema': True}
    
    Returns:
        eval_metrics: Dict of metrics from evaluation
    """
    if agent_kwargs is None:
        agent_kwargs = {}
    
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n, desc="Evaluating")
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs, **agent_kwargs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)
    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
