from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import tyro


@dataclass
class Args:
    dataset_root: str
    output_dir: str
    launcher_output_dir: Optional[str] = None
    policy_pretrained_path: str = "/mnt/disk_2/wjz/openpi/pi05_droid_pytorch"
    policy_repo_id: str = "zhili0818/pi05-full-lora-openpi-droid-v3"
    job_name: str = "pi05-full-lora-openpi-droid-v3"
    gpus: str = "0,1,2,3,4,5"
    num_processes: int = 6
    main_process_port: int = 29672
    batch_size: int = 2
    steps: int = 20000
    learning_rate: float = 5e-5
    peft_r: int = 16
    hf_token: Optional[str] = None
    monitor_interval_s: int = 30


def _resource_snapshot() -> dict:
    result: dict[str, object] = {"time": time.time()}
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = []
        for line in gpu.stdout.strip().splitlines():
            idx, name, mem_used, mem_total, util = [part.strip() for part in line.split(",", 4)]
            rows.append(
                {
                    "index": int(idx),
                    "name": name,
                    "memory_used_mib": int(mem_used),
                    "memory_total_mib": int(mem_total),
                    "utilization_gpu_pct": int(util),
                }
            )
        result["gpus"] = rows
    except Exception as exc:
        result["gpu_error"] = str(exc)
    return result


def _monitor(stop_event: threading.Event, monitor_path: Path, interval_s: int) -> None:
    with monitor_path.open("a") as handle:
        while not stop_event.is_set():
            handle.write(json.dumps(_resource_snapshot()) + "\n")
            handle.flush()
            stop_event.wait(interval_s)


def main() -> None:
    args = tyro.cli(Args)
    training_output_dir = Path(args.output_dir).expanduser().resolve()
    launcher_output_dir = (
        Path(args.launcher_output_dir).expanduser().resolve()
        if args.launcher_output_dir
        else training_output_dir.parent / f"{training_output_dir.name}__launcher"
    )
    launcher_output_dir.mkdir(parents=True, exist_ok=True)

    log_path = launcher_output_dir / "train.log"
    monitor_path = launcher_output_dir / "resource_monitor.jsonl"
    command_path = launcher_output_dir / "launch_command.sh"
    config_path = launcher_output_dir / "launch_config.json"

    command = [
        "/home/wjz/miniconda3/envs/rlft_ms3_lerobot/bin/accelerate",
        "launch",
        "--main_process_port",
        str(args.main_process_port),
        "--num_processes",
        str(args.num_processes),
        "/home/wjz/miniconda3/envs/rlft_ms3_lerobot/bin/lerobot-train",
        "--policy.type=pi05",
        "--dataset.repo_id=carm/pi05_local",
        f"--dataset.root={Path(args.dataset_root).expanduser().resolve()}",
        f"--policy.repo_id={args.policy_repo_id}",
        f"--policy.pretrained_path={Path(args.policy_pretrained_path).expanduser().resolve()}",
        "--policy.push_to_hub=false",
        f"--job_name={args.job_name}",
        f"--output_dir={training_output_dir}",
        "--seed=1",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--optimizer.lr={args.learning_rate}",
        "--policy.gradient_checkpointing=true",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.dtype=bfloat16",
        "--peft.method_type=LORA",
        f"--peft.r={args.peft_r}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUNBUFFERED"] = "1"
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    shell_line = " ".join(shlex.quote(part) for part in command)
    command_path.write_text(shell_line + "\n")
    config_path.write_text(json.dumps(asdict(args), indent=2) + "\n")

    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=_monitor, args=(stop_event, monitor_path, args.monitor_interval_s), daemon=True)
    monitor_thread.start()

    with log_path.open("w") as log_handle:
        log_handle.write("COMMAND: " + shell_line + "\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        returncode = process.wait()

    stop_event.set()
    monitor_thread.join(timeout=5)
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
