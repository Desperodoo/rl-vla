#!/usr/bin/env python3
"""
ACP v5 Dynamic GPU Scheduler

Monitors 5 GPU pairs (0+1, 2+3, 4+5, 6+7, 8+9).
As soon as a pair frees up, the next job in the queue starts — no wave boundary.

Job order: PLD #6-10 (Wave 1) → DSRL #11-15 (Wave 2) → AWSC #1-5 (Wave 3)
Wave 1 jobs already running are auto-adopted on startup.

Usage:
    # Run in background (recommended):
    nohup conda run -n rlft_ms3 --no-capture-output \\
        python scripts/acp_v5_scheduler.py \\
        > logs/vlaw/acp_v5_scheduler.log 2>&1 &

    # Status (from another terminal):
    python scripts/acp_v5_scheduler.py --status

    # Dry-run (show detection + queue without launching):
    python scripts/acp_v5_scheduler.py --dry_run
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "vlaw"
STATE_FILE = LOG_DIR / "acp_v5_scheduler_state.json"

GPU_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
FREE_MIB = 500        # GPU is "free" if memory.used < this (MiB)
POLL_SEC  = 30        # seconds between scheduler ticks
CONDA_ENV = "rlft_ms3"

# ── Checkpoint paths ──────────────────────────────────────────────────────────
_CKPT    = str(ROOT / "runs/maniskill_sweep_v3/aw_shortcut_flow"
               "/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt")
_ACP_SO  = str(ROOT / "checkpoints/vlaw/acp/v3_so/best.safetensors")
_ACP_SAE = str(ROOT / "checkpoints/vlaw/acp/v3_sae/best.safetensors")
_SEED    = "42"
_WANDB   = "rlpd-acp-v5"

# ── Shared arg blocks ─────────────────────────────────────────────────────────
_PLD_BASE = [
    "--checkpoint", _CKPT, "--acp_reward", "--acp_device", "cuda:1",
    "--env_id", "LiftPegUpright-v1",
    "--num_envs", "50", "--num_eval_envs", "50",
    "--total_timesteps", "71000", "--max_episode_steps", "100",
    "--action_scale", "0.3", "--utd_ratio", "60",
    "--target_entropy", "-3.5", "--init_temperature", "0.5",
    "--learning_rate", "1e-4", "--num_layers", "3", "--layer_size", "1024",
    "--num_qs", "5", "--calql_pretrain_steps", "1000", "--calql_alpha", "0.0",
    "--online_ratio", "1.0", "--offline_demo_episodes", "50",
    "--seed", _SEED, "--track", "--wandb_project", _WANDB,
]
_DSRL_BASE = [
    "--checkpoint", _CKPT, "--acp_reward", "--acp_device", "cuda:1",
    "--env_id", "LiftPegUpright-v1",
    "--num_envs", "50", "--num_eval_envs", "50",
    "--total_timesteps", "71000", "--max_episode_steps", "100",
    "--action_magnitude", "2.5", "--utd_ratio", "60",
    "--target_entropy", "-3.5", "--log_std_init", "-5.0",
    "--learning_rate", "3e-4", "--num_layers", "3", "--layer_size", "2048",
    "--num_qs", "10", "--num_seed_steps", "0",
    "--seed", _SEED, "--track", "--wandb_project", _WANDB,
]
_AWSC_BASE = [
    "--algorithm", "awsc", "--pretrain_path", _CKPT,
    "--reward_mode", "acp", "--acp_device", "cuda:1",
    "--env_id", "LiftPegUpright-v1",
    "--num_envs", "50", "--num_eval_envs", "50",
    "--total_timesteps", "500000", "--max_episode_steps", "100",
    "--online_ratio", "0.15", "--utd_ratio", "20",
    "--lr_actor", "1e-4", "--lr_critic", "1e-4",
    "--num_qs", "10", "--num_min_qs", "2",
    "--awsc_beta", "50.0", "--awsc_bc_weight", "4.0",
    "--awsc_advantage_mode", "per_state_v", "--awsc_num_inference_steps", "8",
    "--early_stop", "--early_stop_patience", "5",
    "--early_stop_so_threshold", "0.8", "--early_stop_min_steps", "100000",
    "--seed", _SEED, "--track", "--wandb_project_name", _WANDB,
]


# ── Job dataclass ─────────────────────────────────────────────────────────────
@dataclass
class Job:
    job_id:   int
    name:     str         # short name used in log filename
    exp_name: str         # used for pgrep detection & wandb
    wave:     int
    module:   str
    args:     list        # python -m <module> <args...>
    log_file: str
    # runtime state
    status:   str = "pending"   # pending | running | done | failed
    pid:      Optional[int] = None
    slot:     Optional[int] = None
    start_ts: Optional[str] = None
    end_ts:   Optional[str] = None


def _make_jobs() -> list[Job]:
    def pld(jid, short, exp, extra):
        return Job(jid, short, exp, wave=1, module="rlft.online.train_pld",
                   args=_PLD_BASE + extra + ["--exp_name", exp],
                   log_file=str(LOG_DIR / f"acp_v5_{short}.log"))

    def dsrl(jid, short, exp, extra):
        return Job(jid, short, exp, wave=2, module="rlft.online.train_dsrl",
                   args=_DSRL_BASE + extra + ["--exp_name", exp],
                   log_file=str(LOG_DIR / f"acp_v5_{short}.log"))

    def awsc(jid, short, exp, extra):
        return Job(jid, short, exp, wave=3, module="rlft.online.train_rlpd",
                   args=_AWSC_BASE + extra + ["--exp_name", exp],
                   log_file=str(LOG_DIR / f"acp_v5_{short}.log"))

    return [
        # ── Wave 1: PLD ──────────────────────────────────────────────────────
        pld( 6, "pld_stable_g05",    f"pld_v5_stable_g05_s{_SEED}",
             ["--acp_checkpoint", _ACP_SO,  "--acp_reward_scale", "100",
              "--acp_reward_shaping", "td", "--acp_reward_clip", "5",
              "--q_target_clip", "20",      "--gamma", "0.5"]),
        pld( 7, "pld_stable_g03",    f"pld_v5_stable_g03_s{_SEED}",
             ["--acp_checkpoint", _ACP_SO,  "--acp_reward_scale", "100",
              "--acp_reward_shaping", "td", "--acp_reward_clip", "5",
              "--q_target_clip", "20",      "--gamma", "0.3"]),
        pld( 8, "pld_v_reward_g05",  f"pld_v5_v_reward_g05_s{_SEED}",
             ["--acp_checkpoint", _ACP_SO,  "--acp_reward_scale", "5",
              "--acp_reward_shaping", "potential",
              "--q_target_clip", "20",      "--gamma", "0.5"]),
        pld( 9, "pld_v_reward_sae",  f"pld_v5_v_reward_sae_s{_SEED}",
             ["--acp_checkpoint", _ACP_SAE, "--acp_reward_scale", "5",
              "--acp_reward_shaping", "potential",
              "--q_target_clip", "20",      "--gamma", "0.5"]),
        pld(10, "pld_baseline_g07",  f"pld_v5_baseline_g07_s{_SEED}",
             ["--acp_checkpoint", _ACP_SO,  "--acp_reward_scale", "100",
              "--acp_reward_shaping", "td",
              "--q_target_clip", "20",      "--gamma", "0.7"]),
        # ── Wave 2: DSRL ─────────────────────────────────────────────────────
        dsrl(11, "dsrl_stable_g05",   f"dsrl_v5_stable_g05_s{_SEED}",
             ["--acp_checkpoint", _ACP_SO,  "--acp_reward_scale", "100",
              "--acp_reward_shaping", "td", "--acp_reward_clip", "5",
              "--q_target_clip", "20",      "--gamma", "0.5"]),
        dsrl(12, "dsrl_stable_g03",   f"dsrl_v5_stable_g03_s{_SEED}",
             ["--acp_checkpoint", _ACP_SO,  "--acp_reward_scale", "100",
              "--acp_reward_shaping", "td", "--acp_reward_clip", "5",
              "--q_target_clip", "20",      "--gamma", "0.3"]),
        dsrl(13, "dsrl_v_reward_g05", f"dsrl_v5_v_reward_g05_s{_SEED}",
             ["--acp_checkpoint", _ACP_SO,  "--acp_reward_scale", "5",
              "--acp_reward_shaping", "potential",
              "--q_target_clip", "20",      "--gamma", "0.5"]),
        dsrl(14, "dsrl_v_reward_sae", f"dsrl_v5_v_reward_sae_s{_SEED}",
             ["--acp_checkpoint", _ACP_SAE, "--acp_reward_scale", "5",
              "--acp_reward_shaping", "potential",
              "--q_target_clip", "20",      "--gamma", "0.5"]),
        dsrl(15, "dsrl_baseline_g07", f"dsrl_v5_baseline_g07_s{_SEED}",
             ["--acp_checkpoint", _ACP_SO,  "--acp_reward_scale", "100",
              "--acp_reward_shaping", "td",
              "--q_target_clip", "20",      "--gamma", "0.7"]),
        # ── Wave 3: AWSC ─────────────────────────────────────────────────────
        awsc( 1, "awsc_v_reward",     f"awsc_v5_v_reward_s{_SEED}",
              ["--acp_checkpoint", _ACP_SO,  "--acp_reward_scale", "5",
               "--acp_reward_shaping", "potential"]),
        awsc( 2, "awsc_v_reward_sae", f"awsc_v5_v_reward_sae_s{_SEED}",
              ["--acp_checkpoint", _ACP_SAE, "--acp_reward_scale", "5",
               "--acp_reward_shaping", "potential"]),
        awsc( 3, "awsc_td_sae",       f"awsc_v5_td_sae_s{_SEED}",
              ["--acp_checkpoint", _ACP_SAE, "--acp_reward_scale", "100",
               "--acp_reward_shaping", "td"]),
        awsc( 4, "awsc_td_clip",      f"awsc_v5_td_clip_s{_SEED}",
              ["--acp_checkpoint", _ACP_SO,  "--acp_reward_scale", "100",
               "--acp_reward_shaping", "td", "--acp_reward_clip", "5"]),
        awsc( 5, "awsc_v4repro",      f"awsc_v5_v4repro_s{_SEED}",
              ["--acp_checkpoint", _ACP_SO,  "--acp_reward_scale", "500",
               "--acp_reward_shaping", "td"]),
    ]


# ── GPU helpers ───────────────────────────────────────────────────────────────
def gpu_mem_mib() -> dict[int, int]:
    """Returns {gpu_id: used_mib}."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    out = {}
    for line in r.stdout.strip().splitlines():
        parts = line.strip().split(", ")
        if len(parts) == 2:
            out[int(parts[0])] = int(parts[1])
    return out


def free_slots(mem: dict[int, int], occupied: set[int]) -> list[int]:
    """Slots where both GPUs have < FREE_MIB and not marked occupied."""
    result = []
    for slot, (g0, g1) in enumerate(GPU_PAIRS):
        if slot not in occupied:
            if mem.get(g0, 0) < FREE_MIB and mem.get(g1, 0) < FREE_MIB:
                result.append(slot)
    return result


# ── Process helpers ───────────────────────────────────────────────────────────
def pid_alive(pid: int) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def find_pid(exp_name: str) -> Optional[int]:
    """Find PID of a running process matching exp_name."""
    r = subprocess.run(["pgrep", "-f", exp_name], capture_output=True, text=True)
    pids = [int(p) for p in r.stdout.split() if p.isdigit()]
    # Prefer the deeper process (the actual python, not the conda wrapper)
    return pids[-1] if pids else None


# ── Scheduler ─────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _bar() -> str:
    return "━" * 64


def launch(job: Job, slot: int, dry_run: bool) -> Optional[int]:
    g0, g1 = GPU_PAIRS[slot]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": f"{g0},{g1}",
           "PYTHONPATH": str(ROOT)}
    cmd = ["conda", "run", "-n", CONDA_ENV, "--no-capture-output",
           "python", "-m", job.module] + job.args
    print(f"[{_ts()}] LAUNCH #{job.job_id} {job.name} → slot {slot} "
          f"(GPU {g0},{g1})  wave={job.wave}")
    print(f"         log: {job.log_file}")
    if dry_run:
        return -1
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(job.log_file, "a") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=f,
                                start_new_session=True)
    return proc.pid


def save_state(jobs: list[Job]) -> None:
    data = [asdict(j) for j in jobs]
    STATE_FILE.write_text(json.dumps(data, indent=2))


def print_status(jobs: list[Job]) -> None:
    done    = [j for j in jobs if j.status == "done"]
    running = [j for j in jobs if j.status == "running"]
    pending = [j for j in jobs if j.status == "pending"]
    failed  = [j for j in jobs if j.status == "failed"]
    print(_bar())
    print(f"ACP v5 Scheduler — {_ts()}")
    print(_bar())
    if done:
        print(f" DONE  ({len(done)}): " +
              ", ".join(f"#{j.job_id} {j.name}" for j in done))
    if running:
        print(f" RUN   ({len(running)}):")
        for j in running:
            slot_str = f"slot{j.slot} GPU{GPU_PAIRS[j.slot][0]},{GPU_PAIRS[j.slot][1]}" \
                if j.slot is not None else "?"
            print(f"   [{slot_str}] #{j.job_id} {j.name}  pid={j.pid}  "
                  f"since={j.start_ts or '?'}")
    if pending:
        print(f" PEND  ({len(pending)}): " +
              ", ".join(f"#{j.job_id} {j.name}" for j in pending))
    if failed:
        print(f" FAIL  ({len(failed)}): " +
              ", ".join(f"#{j.job_id} {j.name}" for j in failed))
    mem = gpu_mem_mib()
    mem_str = "  ".join(f"GPU{i}:{mem.get(i,0)}MiB" for i in range(10))
    print(f" MEM:  {mem_str}")
    print(_bar())


def run_scheduler(dry_run: bool) -> None:
    jobs = _make_jobs()
    # slot → Job (running)
    slot_job: dict[int, Job] = {}

    # ── Startup: adopt already-running Wave 1 jobs ────────────────────────
    print(_bar())
    print(f"[{_ts()}] ACP v5 Scheduler starting (dry_run={dry_run})")
    print(_bar())

    for job in jobs:
        log = Path(job.log_file)
        if not log.exists() or log.stat().st_size == 0:
            continue  # definitely not started yet

        pid = find_pid(job.exp_name)
        if pid and pid_alive(pid):
            # Still running — adopt it
            job.status = "running"
            job.pid = pid
            job.start_ts = _iso()
            # Infer slot from original Wave 1 launch order (slot == index among W1 jobs)
            w1_jobs = [j for j in jobs if j.wave == 1]
            if job in w1_jobs:
                job.slot = w1_jobs.index(job)
                slot_job[job.slot] = job
            print(f"[ADOPT] #{job.job_id} {job.name}  pid={pid}  "
                  f"slot={job.slot}  GPU {GPU_PAIRS[job.slot] if job.slot is not None else '?'}")
        else:
            # Log exists, process dead → already done
            job.status = "done"
            job.end_ts = _iso()
            print(f"[DONE]  #{job.job_id} {job.name}  (log exists, process gone)")

    pending = [j for j in jobs if j.status == "pending"]
    print(f"[{_ts()}] Queue: {len(pending)} pending, "
          f"{sum(1 for j in jobs if j.status=='running')} running, "
          f"{sum(1 for j in jobs if j.status=='done')} done")
    print(_bar())

    if dry_run:
        # Show what WOULD be launched next, then exit
        mem = gpu_mem_mib()
        occupied = set(slot_job.keys())
        available = free_slots(mem, occupied)
        print(f"[DRY-RUN] Free slots: {available}  |  Next pending: "
              + ", ".join(f"#{j.job_id} {j.name}" for j in pending[:len(available)]))
        print_status(jobs)
        save_state(jobs)
        return

    save_state(jobs)

    # ── Main loop ─────────────────────────────────────────────────────────
    while any(j.status in ("pending", "running") for j in jobs):
        time.sleep(POLL_SEC)
        ts = _ts()

        # Check for finished running jobs
        for slot in list(slot_job.keys()):
            job = slot_job[slot]
            if not pid_alive(job.pid):
                job.status = "done"
                job.end_ts = _iso()
                del slot_job[slot]
                print(f"[{ts}] DONE   #{job.job_id} {job.name}  "
                      f"slot {slot} freed  (ran {job.start_ts} → {job.end_ts})")

        # Find free GPU slots (not occupied by our tracked jobs)
        occupied = set(slot_job.keys())
        mem = gpu_mem_mib()
        available = free_slots(mem, occupied)

        # Launch next pending job on each available slot
        pending_queue = [j for j in jobs if j.status == "pending"]
        for slot in available:
            if not pending_queue:
                break
            job = pending_queue.pop(0)
            pid = launch(job, slot, dry_run)
            if pid:
                job.status = "running"
                job.pid = pid
                job.slot = slot
                job.start_ts = _iso()
                slot_job[slot] = job

        save_state(jobs)

        # Periodic status print every 5 ticks (~2.5 min)
        tick_count = getattr(run_scheduler, "_tick", 0) + 1
        run_scheduler._tick = tick_count
        if tick_count % 5 == 0:
            print_status(jobs)

    # ── All done ──────────────────────────────────────────────────────────
    print(_bar())
    print(f"[{_ts()}] All 15 jobs complete!")
    print_status(jobs)
    save_state(jobs)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ACP v5 GPU Scheduler")
    parser.add_argument("--dry_run", action="store_true",
                        help="Detect + show queue without launching anything")
    parser.add_argument("--status", action="store_true",
                        help="Print current state from state file and GPU memory, then exit")
    args = parser.parse_args()

    if args.status:
        if STATE_FILE.exists():
            import dataclasses
            raw = json.loads(STATE_FILE.read_text())
            jobs = []
            for d in raw:
                j = Job(**{k: v for k, v in d.items()
                           if k in {f.name for f in dataclasses.fields(Job)}})
                jobs.append(j)
            print_status(jobs)
        else:
            print("No state file found. Scheduler not started yet.")
            mem = gpu_mem_mib()
            print("GPU memory:", {i: f"{mem.get(i,0)}MiB" for i in range(10)})
        return

    run_scheduler(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
