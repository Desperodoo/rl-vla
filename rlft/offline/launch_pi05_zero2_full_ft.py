from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import tyro

from rlft.offline.pi05_bridge import build_probe_environment, resolve_default_openpi_pi05_pretrained_path


@dataclass
class Args:
    dataset_root: str
    output_dir: str
    launcher_output_dir: Optional[str] = None
    accelerate_bin: Optional[str] = None
    lerobot_train_bin: Optional[str] = None
    training_module: str = "rlft.offline.patched_lerobot_train"
    policy_pretrained_path: Optional[str] = None
    official_openpi_checkpoint_name: str = "pi05_base"
    policy_repo_id: str = "carm/pi05-full-ft-zero2-smoke"
    job_name: str = "pi05-full-ft-zero2-smoke"
    gpus: str = "0,1,2,3,4,5"
    num_processes: int = 6
    main_process_port: int = 29690
    batch_size: int = 1
    steps: int = 8
    learning_rate: float = 5e-5
    save_freq: Optional[int] = None
    gradient_accumulation_steps: int = 1
    zero_stage: int = 2
    mixed_precision: str = "bf16"
    policy_dtype: str = "bfloat16"
    hf_token: Optional[str] = None
    monitor_interval_s: int = 30
    profile_interval_s: int = 5


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
    repo_root = Path(__file__).resolve().parents[2]
    policy_pretrained_path = args.policy_pretrained_path or resolve_default_openpi_pi05_pretrained_path(
        args.official_openpi_checkpoint_name
    )
    accelerate_bin = args.accelerate_bin or shutil.which("accelerate")
    if not accelerate_bin:
        raise FileNotFoundError("Could not resolve 'accelerate' from PATH. Pass --accelerate_bin explicitly.")
    if not args.training_module and not (args.lerobot_train_bin or shutil.which("lerobot-train")):
        raise FileNotFoundError("Could not resolve 'lerobot-train' from PATH. Pass --lerobot_train_bin explicitly.")

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
    profile_path = launcher_output_dir / "memory_profile.jsonl"

    command = [
        accelerate_bin,
        "launch",
        "--use_deepspeed",
        "--zero_stage",
        str(args.zero_stage),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--mixed_precision",
        args.mixed_precision,
        "--main_process_port",
        str(args.main_process_port),
        "--num_processes",
        str(args.num_processes),
    ]
    if args.training_module:
        command.extend(["-m", args.training_module])
    else:
        lerobot_train_bin = args.lerobot_train_bin or shutil.which("lerobot-train")
        assert lerobot_train_bin is not None
        command.append(lerobot_train_bin)

    command.extend(
        [
            "--policy.type=pi05",
            "--dataset.repo_id=carm/pi05_local",
            f"--dataset.root={Path(args.dataset_root).expanduser().resolve()}",
            f"--policy.repo_id={args.policy_repo_id}",
            f"--policy.pretrained_path={Path(policy_pretrained_path).expanduser().resolve()}",
            "--policy.push_to_hub=false",
            f"--job_name={args.job_name}",
            f"--output_dir={training_output_dir}",
            "--seed=1",
            f"--batch_size={args.batch_size}",
            f"--steps={args.steps}",
            f"--optimizer.lr={args.learning_rate}",
            *([f"--save_freq={args.save_freq}"] if args.save_freq is not None else []),
            "--policy.gradient_checkpointing=true",
            "--policy.freeze_vision_encoder=false",
            "--policy.train_expert_only=false",
            f"--policy.dtype={args.policy_dtype}",
        ]
    )

    env, _ = build_probe_environment(str(repo_root))
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

    last_profile_ts = 0.0
    with log_path.open("w") as log_handle, profile_path.open("w") as profile_handle:
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
            now = time.time()
            if now - last_profile_ts >= args.profile_interval_s:
                profile_handle.write(json.dumps(_resource_snapshot()) + "\n")
                profile_handle.flush()
                last_profile_ts = now
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
