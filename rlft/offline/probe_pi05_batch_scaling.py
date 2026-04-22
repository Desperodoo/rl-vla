from __future__ import annotations

import json
import re
import shlex
import signal
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import tyro

from rlft.offline.pi05_bridge import build_probe_environment, resolve_default_openpi_pi05_pretrained_path


@dataclass
class Args:
    dataset_root: str
    output_root: str
    accelerate_bin: Optional[str] = None
    lerobot_train_bin: Optional[str] = None
    policy_pretrained_path: Optional[str] = None
    official_openpi_checkpoint_name: str = "pi05_base"
    gpus: str = "0,1,2,3,4,5"
    num_processes: int = 6
    batch_sizes: str = "2,4,6,8"
    smoke_steps: int = 400
    warmup_seconds: int = 180
    startup_seconds: int = 60
    steady_seconds: int = 180
    startup_steps: int = 30
    steady_steps: int = 150
    stop_after_steps: Optional[int] = None
    profile_interval_s: int = 5
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


def _estimate_step(line: str) -> Optional[int]:
    line = line.strip()
    if "step:" not in line:
        return None
    token = line.split("step:", 1)[1].split()[0]
    token = token.rstrip(",")
    multiplier = 1
    if token.endswith(("K", "k")):
        multiplier = 1000
        token = token[:-1]
    try:
        return int(float(token) * multiplier)
    except ValueError:
        return None


def _estimate_step_fallback(line: str) -> Optional[int]:
    matches = re.findall(r"(\d+)\/(\d+)\s*\[", line)
    if matches:
        step_token, _ = matches[-1]
        try:
            return int(step_token)
        except ValueError:
            return None
    matches = re.findall(r"\b(\d+)\/(\d+)\b", line)
    if matches:
        step_token, total_token = matches[-1]
        try:
            step = int(step_token)
            total = int(total_token)
            if total > 0:
                return step
        except ValueError:
            return None
    return None


def _profile_record(selected_gpu_indices: list[int], step: Optional[int], phase: str, elapsed_s: float) -> dict:
    snapshot = _snapshot()
    gpu_map = {row["index"]: row for row in snapshot}
    selected = {idx: gpu_map[idx] for idx in selected_gpu_indices if idx in gpu_map}
    return {
        "time": time.time(),
        "elapsed_s": elapsed_s,
        "step": step,
        "phase": phase,
        "gpus": selected,
    }


def _current_phase(elapsed_s: float, step: Optional[int], args: Args) -> str:
    if step is not None:
        if step < args.startup_steps:
            return "startup"
        if step < args.startup_steps + args.steady_steps:
            return "steady"
        return "post_steady"
    if elapsed_s < args.startup_seconds:
        return "startup"
    if elapsed_s < args.startup_seconds + args.steady_seconds:
        return "steady"
    return "post_steady"


def _window_ready(elapsed_s: float, step: Optional[int], args: Args) -> bool:
    if step is not None:
        return step >= args.startup_steps + args.steady_steps
    return elapsed_s >= args.startup_seconds + args.steady_seconds


def _update_max_memory(max_used: Optional[dict[int, int]], record: dict) -> dict[int, int]:
    gpus = record["gpus"]
    if max_used is None:
        return {idx: row["memory_used_mib"] for idx, row in gpus.items()}
    for idx, row in gpus.items():
        max_used[idx] = max(max_used.get(idx, 0), row["memory_used_mib"])
    return max_used


def main() -> None:
    args = tyro.cli(Args)
    policy_pretrained_path = args.policy_pretrained_path or resolve_default_openpi_pi05_pretrained_path(
        args.official_openpi_checkpoint_name
    )
    accelerate_bin = args.accelerate_bin or shutil.which("accelerate")
    lerobot_train_bin = args.lerobot_train_bin or shutil.which("lerobot-train")
    if not accelerate_bin:
        raise FileNotFoundError("Could not resolve 'accelerate' from PATH. Pass --accelerate_bin explicitly.")
    if not lerobot_train_bin:
        raise FileNotFoundError("Could not resolve 'lerobot-train' from PATH. Pass --lerobot_train_bin explicitly.")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batch_sizes = [int(part) for part in args.batch_sizes.split(",") if part.strip()]
    selected_gpu_indices = [int(part) for part in args.gpus.split(",") if part.strip()]
    summary = []

    env, _ = build_probe_environment()
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
            accelerate_bin,
            "launch",
            "--main_process_port",
            str(29700 + offset),
            "--num_processes",
            str(args.num_processes),
            lerobot_train_bin,
            "--policy.type=pi05",
            "--dataset.repo_id=carm/pi05_local",
            f"--dataset.root={Path(args.dataset_root).expanduser().resolve()}",
            f"--policy.repo_id=zhili0818/{run_name}",
            f"--policy.pretrained_path={Path(policy_pretrained_path).expanduser().resolve()}",
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
        (launcher_out / "launch_command.sh").write_text(" ".join(shlex.quote(c) for c in command) + "\n")
        (launcher_out / "launch_config.json").write_text(json.dumps(asdict(args), indent=2) + "\n")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        log_path = launcher_out / "train.log"
        profile_path = launcher_out / "memory_profile.jsonl"
        start = time.time()
        max_used = None
        warmup_reached = False
        startup_reached = False
        steady_reached = False
        profile_ready_reached = False
        step_window_reached = False
        last_profile_ts = 0.0
        last_step: Optional[int] = None
        startup_memory_mib = None
        steady_peak_mib = None
        seen_steps: deque[int] = deque(maxlen=128)
        log_handle = log_path.open("w")
        profile_handle = profile_path.open("w")
        try:
            while True:
                line = process.stdout.readline() if process.stdout else ""
                if line:
                    log_handle.write(line)
                    log_handle.flush()
                    parsed_step = _estimate_step(line)
                    if parsed_step is None:
                        parsed_step = _estimate_step_fallback(line)
                    if parsed_step is not None:
                        if not seen_steps or parsed_step != seen_steps[-1]:
                            seen_steps.append(parsed_step)
                        last_step = parsed_step
                if process.poll() is not None:
                    break

                now = time.time()
                elapsed_s = now - start
                if now - last_profile_ts >= args.profile_interval_s:
                    phase = _current_phase(elapsed_s, last_step, args)
                    record = _profile_record(selected_gpu_indices, last_step, phase, elapsed_s)
                    profile_handle.write(json.dumps(record) + "\n")
                    profile_handle.flush()
                    max_used = _update_max_memory(max_used, record)
                    if phase != "startup" and not startup_reached:
                        startup_reached = True
                        startup_memory_mib = {idx: row["memory_used_mib"] for idx, row in record["gpus"].items()}
                    if phase == "steady":
                        current = {idx: row["memory_used_mib"] for idx, row in record["gpus"].items()}
                        if steady_peak_mib is None:
                            steady_peak_mib = current
                        else:
                            for idx, value in current.items():
                                steady_peak_mib[idx] = max(steady_peak_mib.get(idx, 0), value)
                        steady_reached = True
                    if _window_ready(elapsed_s, last_step, args):
                        profile_ready_reached = True
                    last_profile_ts = now

                if args.stop_after_steps is not None and last_step is not None and last_step >= args.stop_after_steps:
                    step_window_reached = True
                    process.send_signal(signal.SIGTERM)
                    break

                if time.time() - start > args.warmup_seconds:
                    warmup_reached = True
                    process.send_signal(signal.SIGTERM)
                    break
                time.sleep(1)
            returncode = process.wait(timeout=120)
        except Exception:
            process.kill()
            returncode = -9
        finally:
            profile_handle.close()
            log_handle.close()

        if warmup_reached and returncode in {0, -15, 143}:
            final_returncode = 0
        else:
            final_returncode = returncode

        summary.append(
            {
                "batch_size": batch_size,
                "returncode": final_returncode,
                "raw_returncode": returncode,
                "warmup_reached": warmup_reached,
                "startup_reached": startup_reached,
                "steady_reached": steady_reached,
                "profile_ready_reached": profile_ready_reached,
                "step_window_reached": step_window_reached,
                "last_step": last_step,
                "seen_steps_tail": list(seen_steps),
                "startup_memory_mib": startup_memory_mib or {},
                "steady_peak_memory_mib": steady_peak_mib or {},
                "max_memory_mib": max_used or {},
                "train_output": str(train_out),
                "launcher_output": str(launcher_out),
                "memory_profile": str(profile_path),
            }
        )

    summary_path = output_root / "batch_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
