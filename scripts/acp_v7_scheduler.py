#!/usr/bin/env python3
"""ACP v7 canonical 6-run scheduler.

Retained experiment surface:
- AWSC sim / acp
- PLD sim / acp (qclip0-controlled mirror)
- DSRL sim / acp (qclip0-controlled mirror)

This file intentionally replaces the earlier multi-mode v7 scheduler. Historical
fair-replay / drift-regression / qclip0 diagnosis stages are preserved in docs
and reports, not as active scheduler entrypoints.
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "vlaw"

GPU_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
FREE_MIB = 500
POLL_SEC = 30
CONDA_ENV = "rlft_ms3"
MODE = "diag_core"

_CKPT = str(
    ROOT
    / "runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
)
_ACP_SO = str(ROOT / "checkpoints/vlaw/acp/v3_so/best.safetensors")
_WANDB_DIAG = "rlpd-acp-v7-diag"


@dataclass
class Job:
    job_id: int
    name: str
    exp_name: str
    module: str
    args: list[str]
    log_file: str
    status: str = "pending"
    pid: Optional[int] = None
    slot: Optional[int] = None
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None


def _state_file() -> Path:
    return LOG_DIR / f"acp_v7_scheduler_state_{MODE}.json"


def _with_tracking(base: list[str], wandb_project: str, arg_name: str = "--wandb_project") -> list[str]:
    return [*base, "--track", arg_name, wandb_project]


def _awsc_base(reward_mode: str) -> list[str]:
    args = [
        "--algorithm",
        "awsc",
        "--pretrain_path",
        _CKPT,
        "--env_id",
        "LiftPegUpright-v1",
        "--num_envs",
        "50",
        "--num_eval_envs",
        "50",
        "--total_timesteps",
        "500000",
        "--max_episode_steps",
        "100",
        "--online_ratio",
        "0.15",
        "--utd_ratio",
        "20",
        "--lr_actor",
        "1e-4",
        "--lr_critic",
        "1e-4",
        "--num_qs",
        "10",
        "--num_min_qs",
        "2",
        "--awsc_beta",
        "50.0",
        "--awsc_bc_weight",
        "4.0",
        "--awsc_advantage_mode",
        "per_state_v",
        "--awsc_num_inference_steps",
        "8",
        "--early_stop",
        "--early_stop_patience",
        "5",
        "--early_stop_so_threshold",
        "0.8",
        "--early_stop_min_steps",
        "100000",
        "--seed",
        "42",
        "--track",
        "--wandb_project_name",
        _WANDB_DIAG,
        "--reward_mode",
        reward_mode,
    ]
    if reward_mode == "acp":
        args.extend(
            [
                "--acp_device",
                "cuda:1",
                "--acp_checkpoint",
                _ACP_SO,
                "--acp_reward_scale",
                "100",
                "--acp_reward_shaping",
                "td",
                "--acp_reward_clip",
                "5",
            ]
        )
    return args


def _pld_canonical_sim() -> list[str]:
    return [
        "--checkpoint",
        _CKPT,
        "--env_id",
        "LiftPegUpright-v1",
        "--num_envs",
        "50",
        "--num_eval_envs",
        "50",
        "--max_episode_steps",
        "100",
        "--control_mode",
        "pd_ee_delta_pose",
        "--obs_mode",
        "rgb",
        "--sim_backend",
        "physx_cuda",
        "--reward_mode",
        "dense",
        "--total_timesteps",
        "71000",
        "--learning_rate",
        "1e-4",
        "--online_buffer_size",
        "500000",
        "--offline_buffer_size",
        "200000",
        "--batch_size",
        "256",
        "--gamma",
        "0.99",
        "--tau",
        "0.005",
        "--utd_ratio",
        "60",
        "--init_temperature",
        "0.1",
        "--target_entropy",
        "-3.5",
        "--log_std_init",
        "-5.0",
        "--max_grad_norm",
        "10.0",
        "--online_ratio",
        "1.0",
        "--num_layers",
        "3",
        "--layer_size",
        "1024",
        "--num_qs",
        "5",
        "--offline_demo_episodes",
        "50",
        "--calql_pretrain_steps",
        "1000",
        "--calql_alpha",
        "0.0",
        "--probe_steps",
        "5",
        "--probing_alpha",
        "0.6",
        "--action_scale",
        "0.3",
        "--q_target_clip",
        "0",
        "--eval_freq",
        "5000",
        "--num_eval_episodes",
        "50",
        "--save_freq",
        "50000000",
        "--seed",
        "42",
    ]


def _pld_canonical_acp() -> list[str]:
    return [
        *_pld_canonical_sim()[:2],
        "--acp_reward",
        "--acp_device",
        "cuda:1",
        *_pld_canonical_sim()[2:],
        "--acp_checkpoint",
        _ACP_SO,
        "--acp_reward_scale",
        "100",
        "--acp_reward_shaping",
        "td",
        "--acp_reward_clip",
        "5",
        "--acp_grasp_bonus",
        "1.0",
    ]


def _dsrl_canonical_sim() -> list[str]:
    return [
        "--checkpoint",
        _CKPT,
        "--env_id",
        "LiftPegUpright-v1",
        "--num_envs",
        "50",
        "--num_eval_envs",
        "50",
        "--max_episode_steps",
        "100",
        "--control_mode",
        "pd_ee_delta_pose",
        "--obs_mode",
        "rgb",
        "--sim_backend",
        "physx_cuda",
        "--reward_mode",
        "dense",
        "--total_timesteps",
        "71000",
        "--learning_rate",
        "3e-4",
        "--buffer_size",
        "1000000",
        "--batch_size",
        "256",
        "--gamma",
        "0.95",
        "--tau",
        "0.005",
        "--utd_ratio",
        "60",
        "--num_seed_steps",
        "0",
        "--init_temperature",
        "0.5",
        "--target_entropy",
        "-3.5",
        "--log_std_init",
        "-5.0",
        "--max_grad_norm",
        "10.0",
        "--num_layers",
        "3",
        "--layer_size",
        "2048",
        "--num_qs",
        "10",
        "--action_magnitude",
        "2.5",
        "--q_target_clip",
        "0",
        "--eval_freq",
        "5000",
        "--num_eval_episodes",
        "50",
        "--save_freq",
        "50000000",
        "--seed",
        "42",
    ]


def _dsrl_canonical_acp() -> list[str]:
    return [
        *_dsrl_canonical_sim()[:2],
        "--acp_reward",
        "--acp_device",
        "cuda:1",
        *_dsrl_canonical_sim()[2:],
        "--acp_checkpoint",
        _ACP_SO,
        "--acp_reward_scale",
        "100",
        "--acp_reward_shaping",
        "td",
        "--acp_reward_clip",
        "5",
        "--acp_grasp_bonus",
        "1.0",
    ]


def _make_jobs() -> list[Job]:
    pld_sim = _with_tracking(_pld_canonical_sim(), _WANDB_DIAG)
    pld_acp = _with_tracking(_pld_canonical_acp(), _WANDB_DIAG)
    dsrl_sim = _with_tracking(_dsrl_canonical_sim(), _WANDB_DIAG)
    dsrl_acp = _with_tracking(_dsrl_canonical_acp(), _WANDB_DIAG)
    awsc_sim = _awsc_base("sim")
    awsc_acp = _awsc_base("acp")
    return [
        Job(
            1,
            "awsc_acp_core",
            "awsc_v7_diag_acp_s42",
            "rlft.online.train_rlpd",
            awsc_acp + ["--exp_name", "awsc_v7_diag_acp_s42"],
            str(LOG_DIR / "acp_v7_diag_awsc_acp.log"),
        ),
        Job(
            2,
            "awsc_sim_core",
            "awsc_v7_diag_sim_s42",
            "rlft.online.train_rlpd",
            awsc_sim + ["--exp_name", "awsc_v7_diag_sim_s42"],
            str(LOG_DIR / "acp_v7_diag_awsc_sim.log"),
        ),
        Job(
            3,
            "pld_acp_qclip0_mirror",
            "pld_v7_qclip0_acp_mirror_s42",
            "rlft.online.train_pld",
            pld_acp + ["--exp_name", "pld_v7_qclip0_acp_mirror_s42"],
            str(LOG_DIR / "acp_v7_q0acp_pld_acp.log"),
        ),
        Job(
            4,
            "pld_sim_qclip0_baseline",
            "pld_v7_qclip0_sim_baseline_s42",
            "rlft.online.train_pld",
            pld_sim + ["--exp_name", "pld_v7_qclip0_sim_baseline_s42"],
            str(LOG_DIR / "acp_v7_q0acp_pld_sim.log"),
        ),
        Job(
            5,
            "dsrl_acp_qclip0_mirror",
            "dsrl_v7_qclip0_acp_mirror_s42",
            "rlft.online.train_dsrl",
            dsrl_acp + ["--exp_name", "dsrl_v7_qclip0_acp_mirror_s42"],
            str(LOG_DIR / "acp_v7_q0acp_dsrl_acp.log"),
        ),
        Job(
            6,
            "dsrl_sim_qclip0_baseline",
            "dsrl_v7_qclip0_sim_baseline_s42",
            "rlft.online.train_dsrl",
            dsrl_sim + ["--exp_name", "dsrl_v7_qclip0_sim_baseline_s42"],
            str(LOG_DIR / "acp_v7_q0acp_dsrl_sim.log"),
        ),
    ]


def gpu_mem_mib() -> dict[int, int]:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
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
    _state_file().write_text(json.dumps([asdict(j) for j in jobs], indent=2))


def print_status(jobs: list[Job]) -> None:
    done = [j for j in jobs if j.status == "done"]
    running = [j for j in jobs if j.status == "running"]
    pending = [j for j in jobs if j.status == "pending"]
    failed = [j for j in jobs if j.status == "failed"]
    print(_bar())
    print(f"ACP v7 Scheduler [{MODE}] — {_ts()}")
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
    print(f"[{_ts()}] ACP v7 Scheduler starting mode={MODE} (dry_run={dry_run})")
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
    state_file = _state_file()
    if not state_file.exists():
        print(f"No state file found for mode={MODE}.")
        return
    jobs = [Job(**item) for item in json.loads(state_file.read_text())]
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
