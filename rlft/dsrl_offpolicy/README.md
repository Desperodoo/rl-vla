# DSRL Off-Policy: SAC-based Dual-Stage Reinforcement Learning

This is the **off-policy version** of DSRL, replacing PPO with SAC for improved sample efficiency through higher UTD (Update-To-Data) ratio.

## Key Differences from On-Policy DSRL

| Feature | On-Policy (PPO) | Off-Policy (SAC) |
|---------|-----------------|------------------|
| Algorithm | PPO with GAE | SAC with TD learning |
| Buffer | Rollout Buffer | Replay Buffer |
| Data Reuse | 1x (on-policy) | UTD=10~100x |
| Value Network | V(s) | Q^W(s, w) Double Q |
| Entropy | Coefficient-based | Auto temperature α |
| Sample Efficiency | Low | High |

## Architecture

```
dsrl_offpolicy/
├── agents/
│   └── dsrl_sac_agent.py         # SAC agent with latent steering
├── models/
│   ├── __init__.py
│   └── latent_q_network.py       # Double Q^W(s,w) + target networks
├── buffers/
│   ├── __init__.py
│   └── macro_replay_buffer.py    # SMDP replay buffer
├── train/
│   ├── __init__.py
│   ├── train_stage1_offline.py   # Offline AW-MLE warm start
│   └── train_stage2_online.py    # Online SAC with UTD
├── configs/
│   ├── stage1_offline.yaml
│   └── stage2_online.yaml
└── tests/
    ├── test_components.py        # Unit tests
    ├── test_agent.py             # Agent tests
    └── test_minimal_training.py  # Integration test
```

## Two-Stage Training

### Stage 1: Offline AW-MLE (Same as On-Policy)

Stage 1 is identical between on-policy and off-policy because AW-MLE is inherently off-policy (uses frozen pretrained Q-network for scoring).

```bash
cd /home/lizh/rl-vla/rlft/dsrl_offpolicy/train

python train_stage1_offline.py \
    --env_id LiftPegUpright-v1 \
    --demo_path ~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --awsc_checkpoint ../../dsrl/checkpoints/best_eval_success_once.pt \
    --total_iters 100000 \
    --batch_size 256 \
    --track
```

### Stage 2: Online SAC

```bash
cd /home/lizh/rl-vla/rlft/dsrl_offpolicy/train

python train_stage2_online.py \
    --env_id LiftPegUpright-v1 \
    --awsc_checkpoint ../../dsrl/checkpoints/best_eval_success_once.pt \
    --stage1_checkpoint runs/dsrl_offpolicy_stage1-LiftPegUpright-v1-seed1/checkpoints/best_eval_success_once.pt \
    --total_timesteps 500000 \
    --utd_ratio 20 \
    --batch_size 256 \
    --warmup_steps 5000 \
    --track
```

## Key Hyperparameters

### SAC (Stage 2)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `utd_ratio` | 20 | Gradient steps per env step |
| `batch_size` | 256 | Minibatch size |
| `replay_buffer_size` | 100000 | Buffer capacity (macro-steps) |
| `actor_lr` | 1e-4 | Actor learning rate |
| `critic_lr` | 3e-4 | Critic learning rate |
| `temp_lr` | 1e-4 | Temperature learning rate |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Target network soft update rate |
| `init_temperature` | 0.1 | Initial SAC temperature |
| `target_entropy` | auto | -0.5 * latent_dim |
| `warmup_steps` | 5000 | Steps before starting updates |

### AW-MLE (Stage 1)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_candidates` | 32 | M: latent candidates per obs |
| `kappa` | 1.0 | UCB coefficient |
| `tau` | 5.0 | Soft baseline temperature |
| `beta_latent` | 1.0 | Advantage weighting temperature |
| `kl_coef` | 1e-3 | KL-to-prior regularization |

## Monitoring Metrics

### Stage 2 SAC Metrics

- `td_error`: TD error (should decrease)
- `q_mean`: Mean Q-value (should increase, not explode)
- `alpha`: Temperature (should stabilize, not explode)
- `entropy`: Policy entropy (should not collapse to 0)
- `actor_loss`: Actor loss
- `critic_loss`: Critic loss

### Stage 1 AW-MLE Metrics

- `nll_loss`: Negative log-likelihood
- `kl_loss`: KL-to-prior regularization
- `eff_num`: Effective sample size (higher is better)
- `corr_A_logprob`: Correlation between advantage and log_prob

## Testing

### Run Unit Tests

```bash
cd /home/lizh/rl-vla/rlft/dsrl_offpolicy
pytest tests/ -v
```

### Run Minimal Training Test

```bash
cd /home/lizh/rl-vla/rlft/dsrl_offpolicy/tests
python test_minimal_training.py
```

## Debugging Checklist

### Stage 1 Verification

```python
# After Stage 1, compare:
E[Q^W(s,w)] for w ~ prior
E[Q^W(s,w)] for w ~ π_w_stage1
# Expected: Stage 1 policy should have higher Q-values
```

### Stage 2 Verification

1. **td_error**: Should decrease over training
2. **q_mean**: Should increase but not explode (stay bounded)
3. **alpha**: Should not explode (if exploding, policy is too random)
4. **entropy**: Should not instantly collapse to 0

### Common Issues

1. **Q-values exploding**: Reduce `critic_lr`, increase `tau`
2. **Temperature exploding**: Check `target_entropy` is appropriate
3. **No learning**: Increase `warmup_steps`, check buffer is filling
4. **Unstable training**: Reduce `utd_ratio`, increase `batch_size`

## References

- On-policy DSRL: `../dsrl/`
- SAC: [Haarnoja et al., 2018](https://arxiv.org/abs/1812.05905)
- RLPD (similar off-policy design): `../rlpd/`
