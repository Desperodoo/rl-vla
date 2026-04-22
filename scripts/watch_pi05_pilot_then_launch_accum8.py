from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _screen_session_exists(name: str) -> bool:
    result = subprocess.run(["screen", "-ls"], capture_output=True, text=True, check=False)
    return name in result.stdout


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _derive_python_bin(accelerate_bin: str) -> str:
    accelerate_path = Path(accelerate_bin).expanduser().resolve()
    python_bin = accelerate_path.parent / "python"
    if python_bin.exists():
        return str(python_bin)
    return sys.executable


def _build_launch_command(
    *,
    repo_root: Path,
    python_bin: str,
    source_cfg: dict,
    output_dir: str,
    launcher_output_dir: str,
    policy_repo_id: str,
    job_name: str,
    main_process_port: int,
    gradient_accumulation_steps: int,
) -> list[str]:
    args = [
        python_bin,
        "-m",
        "rlft.offline.launch_pi05_zero2_full_ft",
        "--dataset_root",
        source_cfg["dataset_root"],
        "--output_dir",
        output_dir,
        "--launcher_output_dir",
        launcher_output_dir,
        "--accelerate_bin",
        source_cfg["accelerate_bin"],
        "--lerobot_train_bin",
        source_cfg["lerobot_train_bin"],
        "--policy_pretrained_path",
        source_cfg["policy_pretrained_path"],
        "--official_openpi_checkpoint_name",
        source_cfg["official_openpi_checkpoint_name"],
        "--policy_repo_id",
        policy_repo_id,
        "--job_name",
        job_name,
        "--gpus",
        source_cfg["gpus"],
        "--num_processes",
        str(source_cfg["num_processes"]),
        "--main_process_port",
        str(main_process_port),
        "--batch_size",
        str(source_cfg["batch_size"]),
        "--steps",
        str(source_cfg["steps"]),
        "--learning_rate",
        str(source_cfg["learning_rate"]),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--zero_stage",
        str(source_cfg["zero_stage"]),
        "--mixed_precision",
        source_cfg["mixed_precision"],
        "--policy_dtype",
        source_cfg["policy_dtype"],
        "--monitor_interval_s",
        str(source_cfg["monitor_interval_s"]),
        "--profile_interval_s",
        str(source_cfg["profile_interval_s"]),
    ]
    if source_cfg.get("save_freq") is not None:
        args.extend(["--save_freq", str(source_cfg["save_freq"])])
    if source_cfg.get("hf_token"):
        args.extend(["--hf_token", source_cfg["hf_token"]])
    return args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default="/home/wjz/rl-vla")
    parser.add_argument("--pilot_screen_name", required=True)
    parser.add_argument("--pilot_log", required=True)
    parser.add_argument("--source_launch_config", required=True)
    parser.add_argument("--watch_log", required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--main_process_port", type=int, default=29728)
    parser.add_argument("--poll_interval_s", type=int, default=60)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    pilot_log = Path(args.pilot_log).expanduser().resolve()
    source_launch_config = Path(args.source_launch_config).expanduser().resolve()
    watch_log = Path(args.watch_log).expanduser().resolve()
    watch_log.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with watch_log.open("a") as f:
            f.write(line + "\n")

    log("Watcher started")
    log(f"Watching screen session: {args.pilot_screen_name}")
    log(f"Watching train log: {pilot_log}")

    while True:
        text = _read_text(pilot_log)
        if "End of training" in text:
            log("Detected successful pilot completion marker: End of training")
            break
        if not _screen_session_exists(args.pilot_screen_name):
            log("Pilot screen session disappeared before success marker; aborting auto-launch")
            return 1
        time.sleep(args.poll_interval_s)

    source_cfg = json.loads(source_launch_config.read_text())
    stamp = _timestamp()
    output_dir = f"/mnt/disk_2/wjz/pi05_runs/pi05_fullft_zero2_6gpu_accum{args.gradient_accumulation_steps}_ctrl_5k_{stamp}"
    launcher_output_dir = f"{output_dir}__launcher"
    job_name = Path(output_dir).name
    policy_repo_id = f"carm/{job_name.replace('_', '-')}"
    python_bin = _derive_python_bin(source_cfg["accelerate_bin"])
    launch_cmd = _build_launch_command(
        repo_root=repo_root,
        python_bin=python_bin,
        source_cfg=source_cfg,
        output_dir=output_dir,
        launcher_output_dir=launcher_output_dir,
        policy_repo_id=policy_repo_id,
        job_name=job_name,
        main_process_port=args.main_process_port,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    screen_name = f"pi05_fullft_accum{args.gradient_accumulation_steps}_{stamp}"
    shell_cmd = f"cd {shlex.quote(str(repo_root))} && " + " ".join(shlex.quote(part) for part in launch_cmd)
    log(f"Launching follow-up screen session: {screen_name}")
    log(f"Launch command: {shell_cmd}")
    subprocess.run(["screen", "-dmS", screen_name, "sh", "-lc", shell_cmd], check=True)
    log(f"Follow-up launched successfully: {screen_name}")
    log(f"Expected launcher dir: {launcher_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
