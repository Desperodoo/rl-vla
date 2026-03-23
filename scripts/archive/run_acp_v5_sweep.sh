#!/bin/bash
# ============================================================================
# run_acp_v5_sweep.sh — ACP v5 Sweep: Critic Stabilization + SAE Breakthrough
#
# 15 experiments across AWSC / PLD / DSRL with pure ACP reward (NO sim reward).
#
# Key v5 innovations vs v4:
#   1. Q-target clipping for PLD/DSRL (was missing entirely)
#   2. Reward clipping to prevent outlier TD rewards
#   3. Lower gamma (0.3-0.5) WITHOUT compensating scale increase
#   4. V(s) potential reward (r = V(s') * scale) for SAE breakthrough
#   5. v3_sae checkpoint (success_at_end training objective)
#
# GPU layout: 5 GPU pairs (0+1, 2+3, 4+5, 6+7, 8+9)
#   Wave 1: PLD  #6-10  (5 slots, 71K steps each, ~1.5h)
#   Wave 2: DSRL #11-15 (5 slots, 71K steps each, ~1.5h)
#   Wave 3: AWSC #1-5   (5 slots, 500K steps each, ~9h with early stop)
#
# Usage:
#   bash scripts/run_acp_v5_sweep.sh                # Launch Wave 1 (PLD)
#   bash scripts/run_acp_v5_sweep.sh --wave2        # Launch Wave 2 (DSRL)
#   bash scripts/run_acp_v5_sweep.sh --wave3        # Launch Wave 3 (AWSC)
#   bash scripts/run_acp_v5_sweep.sh --all          # Launch all 15
#   bash scripts/run_acp_v5_sweep.sh --status       # Check status
#
# Environment: rlft_ms3
# WandB project: rlpd-acp-v5
# ============================================================================
set -euo pipefail

cd /home/wjz/rl-vla
export PYTHONPATH=/home/wjz/rl-vla

# ── Common settings ──────────────────────────────────────────────────────
CHECKPOINT="runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
ACP_V3_SO="checkpoints/vlaw/acp/v3_so/best.safetensors"
ACP_V3_SAE="checkpoints/vlaw/acp/v3_sae/best.safetensors"
SEED="${SEED:-42}"
ENV_ID="LiftPegUpright-v1"
NUM_ENVS=50
NUM_EVAL_ENVS=50
MAX_EPISODE_STEPS=100
WANDB_PROJECT="rlpd-acp-v5"

TOTAL_STEPS_AWSC=500000
TOTAL_STEPS_PLD=71000
TOTAL_STEPS_DSRL=71000

# ── Parse arguments ──────────────────────────────────────────────────────
MODE="wave1"
while [[ $# -gt 0 ]]; do
    case $1 in
        --wave1) MODE="wave1"; shift ;;
        --wave2) MODE="wave2"; shift ;;
        --wave3) MODE="wave3"; shift ;;
        --all) MODE="all"; shift ;;
        --status) MODE="status"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Verify prerequisites ─────────────────────────────────────────────────
for f in "$CHECKPOINT" "$ACP_V3_SO" "$ACP_V3_SAE"; do
    if [[ ! -f "$f" ]]; then
        echo "[v5] ERROR: Not found: $f"
        exit 1
    fi
done

mkdir -p logs/vlaw

# ── Status check ─────────────────────────────────────────────────────────
if [[ "$MODE" == "status" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[v5] Status check"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for log in logs/vlaw/acp_v5_*.log; do
        if [[ -f "$log" ]]; then
            name=$(basename "$log" .log)
            last_eval=$(grep -oP '\[Step \d+\] Eval:' "$log" 2>/dev/null | tail -1 || echo "no eval yet")
            last_so=$(grep 'success_once' "$log" 2>/dev/null | tail -1 || echo "")
            echo "  $name: $last_eval  $last_so"
        fi
    done
    echo ""
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    exit 0
fi

PIDS=()

# ══════════════════════════════════════════════════════════════════════════
# Common PLD args (shared across all PLD configs)
# ══════════════════════════════════════════════════════════════════════════
PLD_COMMON=(
    --checkpoint "${CHECKPOINT}"
    --acp_reward
    --acp_device cuda:1
    --env_id "${ENV_ID}"
    --num_envs "${NUM_ENVS}"
    --num_eval_envs "${NUM_EVAL_ENVS}"
    --total_timesteps "${TOTAL_STEPS_PLD}"
    --max_episode_steps "${MAX_EPISODE_STEPS}"
    --action_scale 0.3
    --utd_ratio 60
    --target_entropy -3.5
    --init_temperature 0.5
    --learning_rate 1e-4
    --num_layers 3
    --layer_size 1024
    --num_qs 5
    --calql_pretrain_steps 1000
    --calql_alpha 0.0
    --online_ratio 1.0
    --offline_demo_episodes 50
    --seed "${SEED}"
    --track
    --wandb_project "${WANDB_PROJECT}"
)

# ══════════════════════════════════════════════════════════════════════════
# Common DSRL args (shared across all DSRL configs)
# ══════════════════════════════════════════════════════════════════════════
DSRL_COMMON=(
    --checkpoint "${CHECKPOINT}"
    --acp_reward
    --acp_device cuda:1
    --env_id "${ENV_ID}"
    --num_envs "${NUM_ENVS}"
    --num_eval_envs "${NUM_EVAL_ENVS}"
    --total_timesteps "${TOTAL_STEPS_DSRL}"
    --max_episode_steps "${MAX_EPISODE_STEPS}"
    --action_magnitude 2.5
    --utd_ratio 60
    --target_entropy -3.5
    --log_std_init -5.0
    --learning_rate 3e-4
    --num_layers 3
    --layer_size 2048
    --num_qs 10
    --num_seed_steps 0
    --seed "${SEED}"
    --track
    --wandb_project "${WANDB_PROJECT}"
)

# ══════════════════════════════════════════════════════════════════════════
# Common AWSC args (shared across all AWSC configs)
# ══════════════════════════════════════════════════════════════════════════
AWSC_COMMON=(
    --algorithm awsc
    --pretrain_path "${CHECKPOINT}"
    --reward_mode acp
    --acp_device cuda:1
    --env_id "${ENV_ID}"
    --num_envs "${NUM_ENVS}"
    --num_eval_envs "${NUM_EVAL_ENVS}"
    --total_timesteps "${TOTAL_STEPS_AWSC}"
    --max_episode_steps "${MAX_EPISODE_STEPS}"
    --online_ratio 0.15
    --utd_ratio 20
    --lr_actor 1e-4
    --lr_critic 1e-4
    --num_qs 10
    --num_min_qs 2
    --awsc_beta 50.0
    --awsc_bc_weight 4.0
    --awsc_advantage_mode per_state_v
    --awsc_num_inference_steps 8
    --early_stop
    --early_stop_patience 5
    --early_stop_so_threshold 0.8
    --early_stop_min_steps 100000
    --seed "${SEED}"
    --track
    --wandb_project_name "${WANDB_PROJECT}"
)

# ══════════════════════════════════════════════════════════════════════════
# Wave 1: PLD Experiments (#6-10) — 5 GPU pairs
# ══════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "wave1" || "$MODE" == "all" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[v5] Wave 1: PLD experiments (5 configs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # #6: pld_stable_g05 — TD + gamma=0.5 + q_clip=20 + reward_clip=5 (GPU 0+1)
    echo "[v5] #6 pld_stable_g05 (GPU 0+1) — H1+H2+H5: full protection"
    CUDA_VISIBLE_DEVICES=0,1 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_pld \
        "${PLD_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SO}" \
        --acp_reward_scale 100 \
        --acp_reward_shaping td \
        --acp_reward_clip 5 \
        --q_target_clip 20 \
        --gamma 0.5 \
        --exp_name "pld_v5_stable_g05_s${SEED}" \
        > logs/vlaw/acp_v5_pld_stable_g05.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #7: pld_stable_g03 — TD + gamma=0.3 + q_clip=20 + reward_clip=5 (GPU 2+3)
    echo "[v5] #7 pld_stable_g03 (GPU 2+3) — H2: ultra-low gamma"
    CUDA_VISIBLE_DEVICES=2,3 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_pld \
        "${PLD_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SO}" \
        --acp_reward_scale 100 \
        --acp_reward_shaping td \
        --acp_reward_clip 5 \
        --q_target_clip 20 \
        --gamma 0.3 \
        --exp_name "pld_v5_stable_g03_s${SEED}" \
        > logs/vlaw/acp_v5_pld_stable_g03.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #8: pld_v_reward_g05 — Potential + gamma=0.5 + q_clip=20 (GPU 4+5)
    echo "[v5] #8 pld_v_reward_g05 (GPU 4+5) — H1+H3: V(s) potential reward"
    CUDA_VISIBLE_DEVICES=4,5 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_pld \
        "${PLD_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SO}" \
        --acp_reward_scale 5 \
        --acp_reward_shaping potential \
        --q_target_clip 20 \
        --gamma 0.5 \
        --exp_name "pld_v5_v_reward_g05_s${SEED}" \
        > logs/vlaw/acp_v5_pld_v_reward_g05.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #9: pld_v_reward_sae — Potential + gamma=0.5 + q_clip=20 + v3_sae (GPU 6+7)
    echo "[v5] #9 pld_v_reward_sae (GPU 6+7) — H1+H3+H4: V(s) + sae ckpt"
    CUDA_VISIBLE_DEVICES=6,7 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_pld \
        "${PLD_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SAE}" \
        --acp_reward_scale 5 \
        --acp_reward_shaping potential \
        --q_target_clip 20 \
        --gamma 0.5 \
        --exp_name "pld_v5_v_reward_sae_s${SEED}" \
        > logs/vlaw/acp_v5_pld_v_reward_sae.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #10: pld_baseline_g07 — TD + gamma=0.7 + q_clip=20 (control: v4 gamma + q_clip) (GPU 8+9)
    echo "[v5] #10 pld_baseline_g07 (GPU 8+9) — control: v4 gamma + q_clip only"
    CUDA_VISIBLE_DEVICES=8,9 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_pld \
        "${PLD_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SO}" \
        --acp_reward_scale 100 \
        --acp_reward_shaping td \
        --q_target_clip 20 \
        --gamma 0.7 \
        --exp_name "pld_v5_baseline_g07_s${SEED}" \
        > logs/vlaw/acp_v5_pld_baseline_g07.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"
fi

# ══════════════════════════════════════════════════════════════════════════
# Wave 2: DSRL Experiments (#11-15) — 5 GPU pairs
# ══════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "wave2" || "$MODE" == "all" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[v5] Wave 2: DSRL experiments (5 configs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # #11: dsrl_stable_g05 — TD + gamma=0.5 + q_clip=20 + reward_clip=5 (GPU 0+1)
    echo "[v5] #11 dsrl_stable_g05 (GPU 0+1) — H1+H2+H5: full protection"
    CUDA_VISIBLE_DEVICES=0,1 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_dsrl \
        "${DSRL_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SO}" \
        --acp_reward_scale 100 \
        --acp_reward_shaping td \
        --acp_reward_clip 5 \
        --q_target_clip 20 \
        --gamma 0.5 \
        --exp_name "dsrl_v5_stable_g05_s${SEED}" \
        > logs/vlaw/acp_v5_dsrl_stable_g05.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #12: dsrl_stable_g03 — TD + gamma=0.3 + q_clip=20 + reward_clip=5 (GPU 2+3)
    echo "[v5] #12 dsrl_stable_g03 (GPU 2+3) — H2: ultra-low gamma"
    CUDA_VISIBLE_DEVICES=2,3 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_dsrl \
        "${DSRL_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SO}" \
        --acp_reward_scale 100 \
        --acp_reward_shaping td \
        --acp_reward_clip 5 \
        --q_target_clip 20 \
        --gamma 0.3 \
        --exp_name "dsrl_v5_stable_g03_s${SEED}" \
        > logs/vlaw/acp_v5_dsrl_stable_g03.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #13: dsrl_v_reward_g05 — Potential + gamma=0.5 + q_clip=20 (GPU 4+5)
    echo "[v5] #13 dsrl_v_reward_g05 (GPU 4+5) — H1+H3: V(s) potential reward"
    CUDA_VISIBLE_DEVICES=4,5 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_dsrl \
        "${DSRL_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SO}" \
        --acp_reward_scale 5 \
        --acp_reward_shaping potential \
        --q_target_clip 20 \
        --gamma 0.5 \
        --exp_name "dsrl_v5_v_reward_g05_s${SEED}" \
        > logs/vlaw/acp_v5_dsrl_v_reward_g05.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #14: dsrl_v_reward_sae — Potential + gamma=0.5 + q_clip=20 + v3_sae (GPU 6+7)
    echo "[v5] #14 dsrl_v_reward_sae (GPU 6+7) — H1+H3+H4: V(s) + sae ckpt"
    CUDA_VISIBLE_DEVICES=6,7 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_dsrl \
        "${DSRL_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SAE}" \
        --acp_reward_scale 5 \
        --acp_reward_shaping potential \
        --q_target_clip 20 \
        --gamma 0.5 \
        --exp_name "dsrl_v5_v_reward_sae_s${SEED}" \
        > logs/vlaw/acp_v5_dsrl_v_reward_sae.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #15: dsrl_baseline_g07 — TD + gamma=0.7 + q_clip=20 (control) (GPU 8+9)
    echo "[v5] #15 dsrl_baseline_g07 (GPU 8+9) — control: v4 gamma + q_clip only"
    CUDA_VISIBLE_DEVICES=8,9 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_dsrl \
        "${DSRL_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SO}" \
        --acp_reward_scale 100 \
        --acp_reward_shaping td \
        --q_target_clip 20 \
        --gamma 0.7 \
        --exp_name "dsrl_v5_baseline_g07_s${SEED}" \
        > logs/vlaw/acp_v5_dsrl_baseline_g07.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"
fi

# ══════════════════════════════════════════════════════════════════════════
# Wave 3: AWSC Experiments (#1-5) — 5 GPU pairs
# ══════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "wave3" || "$MODE" == "all" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[v5] Wave 3: AWSC experiments (5 configs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # #1: awsc_v_reward — Potential + scale=5 + v3_so (GPU 0+1)
    echo "[v5] #1 awsc_v_reward (GPU 0+1) — H3: V(s) potential reward"
    CUDA_VISIBLE_DEVICES=0,1 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_rlpd \
        "${AWSC_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SO}" \
        --acp_reward_scale 5 \
        --acp_reward_shaping potential \
        --exp_name "awsc_v5_v_reward_s${SEED}" \
        > logs/vlaw/acp_v5_awsc_v_reward.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #2: awsc_v_reward_sae — Potential + scale=5 + v3_sae (GPU 2+3)
    echo "[v5] #2 awsc_v_reward_sae (GPU 2+3) — H3+H4: V(s) + sae ckpt"
    CUDA_VISIBLE_DEVICES=2,3 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_rlpd \
        "${AWSC_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SAE}" \
        --acp_reward_scale 5 \
        --acp_reward_shaping potential \
        --exp_name "awsc_v5_v_reward_sae_s${SEED}" \
        > logs/vlaw/acp_v5_awsc_v_reward_sae.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #3: awsc_td_sae — TD + scale=100 + v3_sae (GPU 4+5)
    echo "[v5] #3 awsc_td_sae (GPU 4+5) — H4: sae ckpt only"
    CUDA_VISIBLE_DEVICES=4,5 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_rlpd \
        "${AWSC_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SAE}" \
        --acp_reward_scale 100 \
        --acp_reward_shaping td \
        --exp_name "awsc_v5_td_sae_s${SEED}" \
        > logs/vlaw/acp_v5_awsc_td_sae.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #4: awsc_td_clip — TD + scale=100 + v3_so + reward_clip=5 (GPU 6+7)
    echo "[v5] #4 awsc_td_clip (GPU 6+7) — H5: reward clipping"
    CUDA_VISIBLE_DEVICES=6,7 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_rlpd \
        "${AWSC_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SO}" \
        --acp_reward_scale 100 \
        --acp_reward_shaping td \
        --acp_reward_clip 5 \
        --exp_name "awsc_v5_td_clip_s${SEED}" \
        > logs/vlaw/acp_v5_awsc_td_clip.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"

    # #5: awsc_v4_repro — TD + scale=500 + v3_so (v4 bc=4 reproduction) (GPU 8+9)
    echo "[v5] #5 awsc_v4_repro (GPU 8+9) — control: v4 bc=4 reproduction"
    CUDA_VISIBLE_DEVICES=8,9 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_rlpd \
        "${AWSC_COMMON[@]}" \
        --acp_checkpoint "${ACP_V3_SO}" \
        --acp_reward_scale 500 \
        --acp_reward_shaping td \
        --exp_name "awsc_v5_v4repro_s${SEED}" \
        > logs/vlaw/acp_v5_awsc_v4repro.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$!"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[v5] Launched. PIDs: ${PIDS[*]:-none}"
echo ""
echo "Monitor:"
echo "  bash scripts/run_acp_v5_sweep.sh --status"
echo "  nvidia-smi"
echo ""
echo "When done, run diagnosis:"
echo "  python scripts/analyze_training_internals.py --project ${WANDB_PROJECT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
