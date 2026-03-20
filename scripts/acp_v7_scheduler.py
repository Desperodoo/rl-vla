#!/usr/bin/env python3
"""
ACP v7 Dynamic GPU Scheduler — PLD mechanism sweep + DSRL controls.

Focus:
- PLD mechanism diagnosis: residual capacity, entropy control, prior looseness,
  reward grounding.
- DSRL long-training controls: reward learnability reference.
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

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "vlaw"
STATE_FILE = LOG_DIR / "acp_v7_scheduler_state.json"

GPU_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
FREE_MIB = 500
POLL_SEC = 30
CONDA_ENV = "rlft_ms3"

_CKPT = str(ROOT / "runs/maniskill_sweep_v3/aw_shortcut_flow" / "cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt")
_ACP_SO = str(ROOT / "checkpoints/vlaw/acp/v3_so/best.safetensors")
_WANDB = "rlpd-acp-v7"

_PLD_BASE = [
    "--checkpoint", _CKPT, "--acp_reward", "--acp_device", "cuda:1",
    "--env_id", "LiftPegUpright-v1",
    "--num_envs", "50", "--num_eval_envs", "50",
    "--total_timesteps", "200000", "--max_episode_steps", "100",
    "--utd_ratio", "60", "--learning_rate", "1e-4",
    "--num_layers", "3", "--layer_size", "1024", "--num_qs", "5",
    "--q_target_clip", "20", "--gamma", "0.5",
    "--acp_checkpoint", _ACP_SO, "--acp_reward_scale", "100",
    "--acp_reward_shaping", "td", "--acp_reward_clip", "5",
    "--acp_grasp_bonus", "1.0", "--seed", "42",
    "--track", "--wandb_project", _WANDB,
]

_DSRL_BASE = [
    "--checkpoint", _CKPT, "--acp_reward", "--acp_device", "cuda:1",
    "--env_id", "LiftPegUpright-v1",
    "--num_envs", "50", "--num_eval_envs", "50",
    "--total_timesteps", "300000", "--max_episode_steps", "100",
    "--action_magnitude", "2.5", "--utd_ratio", "60",
    "--target_entropy", "-3.5", "--log_std_init", "-5.0",
    "--learning_rate", "3e-4", "--num_layers", "3", "--layer_size", "2048",
    "--num_qs", "10", "--num_seed_steps", "0",
    "--acp_checkpoint", _ACP_SO, "--acp_reward_scale", "100",
    "--acp_reward_shaping", "td", "--acp_reward_clip", "5",
    "--acp_grasp_bonus", "1.0", "--q_target_clip", "20", "--gamma", "0.5",
    "--track", "--wandb_project", _WANDB,
]


@dataclass
class Job:
    job_id: int
    name: str
    exp_name: str
    module: str
    args: list
    log_file: str
    status: str = "pending"
    pid: Optional[int] = None
    slot: Optional[int] = None
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None


def _make_jobs() -> list[Job]:
    def pld(jid: int, short: str, exp: str, extra: list[str]) -> Job:
        return Job(
            jid, short, exp, "rlft.online.train_pld",
            _PLD_BASE + extra + ["--exp_name", exp],
            str(LOG_DIR / f"acp_v7_{short}.log"),
        )

    def dsrl(jid: int, short: str, exp: str, extra: list[str]) -> Job:
        return Job(
            jid, short, exp, "rlft.online.train_dsrl",
            _DSRL_BASE + extra + ["--exp_name", exp],
            str(LOG_DIR / f"acp_v7_{short}.log"),
        )

    return [
        pld(1, "pld_scale03_base", "pld_v7_scale03_base_s42", [
            "--action_scale", "0.3", "--target_entropy", "-3.5", "--init_temperature", "0.5",
            "--online_ratio", "1.0", "--probe_steps", "5", "--probing_alpha", "0.6",
            "--calql_pretrain_steps", "1000", "--calql_alpha", "0.0",
        ]),
        pld(2, "pld_scale05_base", "pld_v7_scale05_base_s42", [
            "--action_scale", "0.5", "--target_entropy", "-3.5", "--init_temperature", "0.5",
            "--online_ratio", "1.0", "--probe_steps", "5", "--probing_alpha", "0.6",
            "--calql_pretrain_steps", "1000", "--calql_alpha", "0.0",
        ]),
        pld(3, "pld_entropy_floor", "pld_v7_entropy_floor_s42", [
            "--action_scale", "0.3", "--target_entropy", "-2.0", "--init_temperature", "1.0",
            "--min_temperature", "0.05", "--entropy_bonus_coef", "0.02",
            "--online_ratio", "1.0", "--probe_steps", "5", "--probing_alpha", "0.6",
            "--calql_pretrain_steps", "1000", "--calql_alpha", "0.0",
        ]),
        pld(4, "pld_entropy_floor_scale05", "pld_v7_entropy_floor_scale05_s42", [
            "--action_scale", "0.5", "--target_entropy", "-2.0", "--init_temperature", "1.0",
            "--min_temperature", "0.05", "--entropy_bonus_coef", "0.02",
            "--online_ratio", "1.0", "--probe_steps", "5", "--probing_alpha", "0.6",
            "--calql_pretrain_steps", "1000", "--calql_alpha", "0.0",
        ]),
        pld(5, "pld_less_probe", "pld_v7_less_probe_s42", [
            "--action_scale", "0.3", "--target_entropy", "-3.5", "--init_temperature", "0.5",
            "--online_ratio", "1.0", "--probe_steps", "1", "--probing_alpha", "0.2",
            "--calql_pretrain_steps", "1000", "--calql_alpha", "0.0",
        ]),
        pld(6, "pld_online_dominant", "pld_v7_online_dominant_s42", [
            "--action_scale", "0.3", "--target_entropy", "-3.5", "--init_temperature", "0.5",
            "--online_ratio", "0.9", "--probe_steps", "1", "--probing_alpha", "0.2",
            "--offline_demo_episodes", "20", "--calql_pretrain_steps", "200", "--calql_alpha", "0.0",
        ]),
        pld(7, "pld_td_grasp", "pld_v7_td_grasp_s42", [
            "--action_scale", "0.3", "--target_entropy", "-3.5", "--init_temperature", "0.5",
            "--online_ratio", "1.0", "--probe_steps", "5", "--probing_alpha", "0.6",
            "--calql_pretrain_steps", "1000", "--calql_alpha", "0.0",
        ]),
        pld(8, "pld_high_grasp", "pld_v7_high_grasp_s42", [
            "--action_scale", "0.3", "--target_entropy", "-3.5", "--init_temperature", "0.5",
            "--online_ratio", "1.0", "--probe_steps", "5", "--probing_alpha", "0.6",
            "--calql_pretrain_steps", "1000", "--calql_alpha", "0.0",
            "--acp_grasp_bonus", "3.0",
        ]),
        dsrl(9, "dsrl_long_300k_s42", "dsrl_v7_long_300k_s42", ["--seed", "42"]),
        dsrl(10, "dsrl_long_300k_s43", "dsrl_v7_long_300k_s43", ["--seed", "43"]),
        dsrl(11, "dsrl_long_300k_potential", "dsrl_v7_long_300k_potential_s42", [
            "--seed", "42", "--acp_reward_scale", "5", "--acp_reward_shaping", "potential",
        ]),
    ]


def gpu_mem_mib() -> dict[int, int]:
    r = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
    ], capture_output=True, text=True)
    out: dict[int, int] = {}
    for line in r.stdout.strip().splitlines():
        parts = line.strip().split(", ")
        if len(parts) == 2:
            out[int(parts[0])] = int(parts[1])
    return out


def free_slots(mem: dict[int, int], occupied: set[int]) -> list[int]:
    slots = []
    for slot, (g0, g1) in enumerate(GPU_PAIRS):
        if slot not in occupied and mem.get(g0, 0) < FREE_MIB and mem.get(g1, 0) < FREE_MIB:
            slots.append(slot)
    return slots


def pid_alive(pid: int) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _bar() -> str:
    return "━" * 64


def launch(job: Job, slot: int, dry_run: bool) -> Optional[int]:
    g0, g1 = GPU_PAIRS[slot]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": f"{g0},{g1}", "PYTHONPATH": str(ROOT)}
    cmd = ["conda", "run", "-n", CONDA_ENV, "--no-capture-output", "python", "-m", job.module] + job.args
    print(f"[{_ts()}] LAUNCH #{job.job_id} {job.name} → slot {slot} (GPU {g0},{g1})")
    print(f"         log: {job.log_file}")
    if dry_run:
        return -1
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(job.log_file, "a") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=f, start_new_session=True)
    return proc.pid


def save_state(jobs: list[Job]) -> None:
    STATE_FILE.write_text(json.dumps([asdict(j) for j in jobs], indent=2))


def print_status(jobs: list[Job]) -> None:
    done = [j for j in jobs if j.status == "done"]
    running = [j for j in jobs if j.status == "running"]
    pending = [j for j in jobs if j.status == "pending"]
    failed = [j for j in jobs if j.status == "failed"]
    print(_bar())
    print(f"ACP v7 Scheduler — {_ts()}")
    print(_bar())
    if done:
        print(f" DONE  ({len(done)}): " + ", ".join(f"#{j.job_id} {j.name}" for j in done))
    if running:
        print(f" RUN   ({len(running)}):")
        for j in running:
            slot_str = f"slot{j.slot} GPU{GPU_PAIRS[j.slot][0]},{GPU_PAIRS[j.slot][1]}" if j.slot is not None else "?"
            print(f"   [{slot_str}] #{j.job_id} {j.name} pid={j.pid} since={j.start_ts or '?'}")
    if pending:
        print(f" PEND  ({len(pending)}): " + ", ".join(f"#{j.job_id} {j.name}" for j in pending))
    if failed:
        print(f" FAIL  ({len(failed)}): " + ", ".join(f"#{j.job_id} {j.name}" for j in failed))
    mem = gpu_mem_mib()
    print(" MEM:  " + "  ".join(f"GPU{i}:{mem.get(i, 0)}MiB" for i in range(10)))
    print(_bar())


def run_scheduler(dry_run: bool) -> None:
    jobs = _make_jobs()
    slot_job: dict[int, Job] = {}
    print(_bar())
    print(f"[{_ts()}] ACP v7 Scheduler starting (dry_run={dry_run})")
    print(_bar())

    while True:
        for slot, job in list(slot_job.items()):
            if not pid_alive(job.pid or -1):
                job.status = "done"
                job.end_ts = _iso()
                del slot_job[slot]
                print(f"[{_ts()}] DONE  #{job.job_id} {job.name}")

        mem = gpu_mem_mib()
        available = free_slots(mem, set(slot_job.keys()))
        pending = [j for j in jobs if j.status == "pending"]

        for slot, job in zip(available, pending):
            pid = launch(job, slot, dry_run)
            job.status = "running"
            job.pid = pid
            job.slot = slot
            job.start_ts = _iso()
            slot_job[slot] = job

        save_state(jobs)
        print_status(jobs)

        if all(j.status in {"done", "failed"} for j in jobs):
            break
        if dry_run:
            break
        time.sleep(POLL_SEC)


def show_status() -> None:
    if not STATE_FILE.exists():
        print("No state file found.")
        return
    jobs = [Job(**item) for item in json.loads(STATE_FILE.read_text())]
    print_status(jobs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        run_scheduler(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
