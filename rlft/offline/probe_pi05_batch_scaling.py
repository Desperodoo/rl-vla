from __future__ import annotations

import json
import os
import shlex
import signal
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
    output_root: str
    policy_pretrained_path: str = "/mnt/disk_2/wjz/openpi/pi05_droid_pytorch"
    gpus: str = "0,1,2,3,4,5"
    num_processes: int = 6
    batch_sizes: str = "2,4,6,8"
    smoke_steps: int = 400
    warmup_seconds: int = 180
    learning_rate: float = 5e-5
    peft_r: int = 16
    hf_token: Optional[str] = None


def _snapshot() -> list[dict]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in gpu.stdout.strip().splitlines():
        idx, mem_used, mem_total, util = [part.strip() for part in line.split(",", 3)]
        rows.append(
            {
                "index": int(idx),
                "memory_used_mib": int(mem_used),
                "memory_total_mib": int(mem_total),
                "utilization_gpu_pct": int(util),
            }
        )
    return rows


def main() -> None:
    args = tyro.cli(Args)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batch_sizes = [int(part) for part in args.batch_sizes.split(",") if part.strip()]
    selected_gpu_indices = [int(part) for part in args.gpus.split(",") if part.strip()]
    summary = []

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUNBUFFERED"] = "1"
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    for offset, batch_size in enumerate(batch_sizes):
        run_name = f"batch{batch_size}_smoke"
        train_out = output_root / run_name / "train_output"
        launcher_out = output_root / run_name / "launcher"
        launcher_out.mkdir(parents=True, exist_ok=True)
        command = [
            "/home/wjz/miniconda3/envs/rlft_ms3_lerobot/bin/accelerate",
            "launch",
            "--main_process_port",
            str(29700 + offset),
            "--num_processes",
            str(args.num_processes),
            "/home/wjz/miniconda3/envs/rlft_ms3_lerobot/bin/lerobot-train",
            "--policy.type=pi05",
            "--dataset.repo_id=carm/pi05_local",
            f"--dataset.root={Path(args.dataset_root).expanduser().resolve()}",
            f"--policy.repo_id=zhili0818/{run_name}",
            f"--policy.pretrained_path={Path(args.policy_pretrained_path).expanduser().resolve()}",
            "--policy.push_to_hub=false",
            f"--job_name={run_name}",
            f"--output_dir={train_out}",
            "--seed=1",
            f"--batch_size={batch_size}",
            f"--steps={args.smoke_steps}",
            f"--optimizer.lr={args.learning_rate}",
            "--policy.gradient_checkpointing=true",
            "--policy.freeze_vision_encoder=true",
            "--policy.train_expert_only=true",
            "--policy.dtype=bfloat16",
            "--peft.method_type=LORA",
            f"--peft.r={args.peft_r}",
        ]
        (launcher_out / 'launch_command.sh').write_text(' '.join(shlex.quote(c) for c in command) + '\n')
        (launcher_out / 'launch_config.json').write_text(json.dumps(asdict(args), indent=2) + '\n')

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        log_path = launcher_out / 'train.log'
        start = time.time()
        max_used = None
        warmup_reached = False
        log_handle = log_path.open('w')
        try:
            while True:
                line = process.stdout.readline() if process.stdout else ''
                if line:
                    log_handle.write(line)
                    log_handle.flush()
                if process.poll() is not None:
                    break
                snapshot = _snapshot()
                gpu_map = {row['index']: row for row in snapshot}
                used = {idx: gpu_map[idx]['memory_used_mib'] for idx in selected_gpu_indices if idx in gpu_map}
                if max_used is None:
                    max_used = used
                else:
                    for k, v in used.items():
                        max_used[k] = max(max_used.get(k, 0), v)
                if time.time() - start > args.warmup_seconds:
                    warmup_reached = True
                    process.send_signal(signal.SIGTERM)
                    break
                time.sleep(5)
            returncode = process.wait(timeout=120)
        except Exception:
            process.kill()
            returncode = -9
        finally:
            log_handle.close()

        if warmup_reached and returncode in {0, -15, 143}:  # terminated intentionally after warmup
            final_returncode = 0
        else:
            final_returncode = returncode

        summary.append(
            {
                'batch_size': batch_size,
                'returncode': final_returncode,
                'raw_returncode': returncode,
                'warmup_reached': warmup_reached,
                'max_memory_mib': max_used or {},
                'train_output': str(train_out),
                'launcher_output': str(launcher_out),
            }
        )

    summary_path = output_root / 'batch_sweep_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
