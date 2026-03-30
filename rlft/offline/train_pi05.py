"""LeRobot-first pi0.5 training entrypoint.

This entrypoint validates the current CARM->pi0.5 bridge, writes reproducible
run metadata, and can optionally dispatch into LeRobot's concrete training CLI.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from rlft.datasets import get_carm_data_info, worker_init_fn
from rlft.offline.pi05_bridge import (
    Pi05BridgeContract,
    Pi05ObservationContract,
    build_lerobot_train_command,
    build_pi05_dataset_bridge,
    build_pi05_run_config,
    build_probe_environment,
    DEFAULT_OPENPI_PI05_DROID_PRETRAINED_PATH,
    export_carm_to_lerobot_dataset,
    probe_lerobot_environment,
    validate_bridge_dataset,
    validate_lerobot_dataset_path,
    validate_lerobot_train_command,
)


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "CARM-pi05"
    wandb_entity: Optional[str] = None

    demo_path: str = "~/rl-vla/recorded_data/mix"
    num_demos: Optional[int] = None
    state_mode: Literal["joint_only", "ee_only", "both"] = "joint_only"
    normalize_actions: bool = True
    action_norm_mode: Literal["standard", "minmax"] = "standard"

    obs_horizon: int = 2
    action_horizon: int = 16
    window_stride: int = 1
    batch_size: int = 8
    num_dataload_workers: int = 0
    total_iters: int = 1000
    lr: float = 1e-4

    visual_encoder_type: Literal["plain_conv", "resnet10", "resnet18", "resnet34", "resnet50"] = "resnet18"
    auto_image_size: bool = True

    upstream_family: Literal["lerobot"] = "lerobot"
    policy_type: Literal["pi0.5", "pi05"] = "pi0.5"
    upstream_repo_path: Optional[str] = None
    fail_if_upstream_missing: bool = False
    dispatch_to_lerobot: bool = False
    lerobot_dataset_repo_id: Optional[str] = None
    lerobot_dataset_path: Optional[str] = None
    policy_repo_id: Optional[str] = None
    policy_pretrained_path: Optional[str] = None
    use_official_openpi_checkpoint: bool = False
    policy_push_to_hub: bool = False
    use_peft: bool = False
    peft_method_type: str = "LORA"
    peft_r: int = 16
    peft_target_modules: Optional[str] = None
    peft_full_training_modules: Optional[str] = None

    bridge_smoke_only: bool = False
    bridge_validate_only: bool = False
    strict: bool = False
    auto_export_lerobot_dataset: bool = False
    export_output_dir: Optional[str] = None


def _ensure_run_name(args: Args) -> str:
    if args.exp_name is None:
        args.exp_name = f"pi05-{args.policy_type}-seed{args.seed}"
    return f"{args.exp_name}__{int(time.time())}"


def _build_contract(args: Args) -> Pi05BridgeContract:
    base_contract = Pi05BridgeContract(
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        window_stride=args.window_stride,
    )
    return Pi05BridgeContract(
        obs_horizon=base_contract.obs_horizon,
        action_horizon=base_contract.action_horizon,
        window_stride=base_contract.window_stride,
        observation=Pi05ObservationContract(
            image_key=base_contract.observation.image_key,
            state_key=base_contract.observation.state_key,
            ee_pose_key=base_contract.observation.ee_pose_key,
            image_layout=base_contract.observation.image_layout,
            state_mode=args.state_mode,
            image_size=base_contract.observation.image_size,
            normalize_images=base_contract.observation.normalize_images,
            include_depth=base_contract.observation.include_depth,
        ),
        action=base_contract.action,
    )


def _ensure_unique_dispatch_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    suffix = int(time.time() * 1000)
    return base_dir.parent / f"{base_dir.name}-dispatch-{suffix}"


def _get_distributed_context() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, rank, world_size


def _wait_for_export(export_dir: Path, timeout_s: float = 300.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info_ready = (export_dir / "meta" / "info.json").exists()
        tasks_ready = (export_dir / "meta" / "tasks.parquet").exists()
        episodes_ready = (export_dir / "meta" / "episodes").exists() and any((export_dir / "meta" / "episodes").glob("*/*.parquet"))
        data_ready = (export_dir / "data").exists() and any((export_dir / "data").glob("*/*.parquet"))
        done_ready = (export_dir / ".export_done").exists()
        if done_ready or (info_ready and tasks_ready and episodes_ready and data_ready):
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for exported LeRobot dataset at {export_dir}")


def _validate_peft_args(args: Args) -> None:
    if args.use_official_openpi_checkpoint and not args.policy_pretrained_path:
        args.policy_pretrained_path = DEFAULT_OPENPI_PI05_DROID_PRETRAINED_PATH
    if args.use_peft and not args.policy_pretrained_path:
        raise ValueError(
            "PEFT/LoRA requires policy_pretrained_path. "
            "LeRobot does not support training pi05 from scratch with use_peft=True."
        )


def main() -> None:
    args = tyro.cli(Args)
    _validate_peft_args(args)
    run_name = _ensure_run_name(args)
    local_rank, rank, world_size = _get_distributed_context()
    is_main_process = rank == 0

    if args.track and wandb is not None and is_main_process:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}") if is_main_process else None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    demo_path = str(Path(args.demo_path).expanduser())
    contract = _build_contract(args)
    data_info = get_carm_data_info(demo_path, state_mode=args.state_mode)

    run_config = build_pi05_run_config(args, contract.as_metadata(), data_info)
    run_dir = Path("runs") / run_name
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process:
        with open(checkpoints_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(checkpoints_dir / "pi05_bridge_config.json", "w") as f:
            json.dump(run_config, f, indent=2)

    probe_result = probe_lerobot_environment(upstream_repo_path=args.upstream_repo_path)
    if is_main_process:
        with open(checkpoints_dir / "env_probe.json", "w") as f:
            json.dump(probe_result, f, indent=2)

    bridge_validation = validate_bridge_dataset(
        data_path=demo_path,
        contract=contract,
        num_episodes=args.num_demos,
        normalize_actions=args.normalize_actions,
        action_norm_mode=args.action_norm_mode,
    )
    if is_main_process:
        with open(checkpoints_dir / "bridge_validation.json", "w") as f:
            json.dump(bridge_validation, f, indent=2)

    if args.strict and not bridge_validation["summary"]["ok"]:
        raise RuntimeError("Bridge validation failed in strict mode. See bridge_validation.json.")

    dataset = build_pi05_dataset_bridge(
        data_path=demo_path,
        contract=contract,
        num_episodes=args.num_demos,
        normalize_actions=args.normalize_actions,
        action_norm_mode=args.action_norm_mode,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )

    first_batch = next(iter(dataloader))
    print("Bridge smoke batch:")
    for key, value in first_batch.items():
        if torch.is_tensor(value):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")

    action_stats = dataset.get_action_stats()
    if action_stats is not None and is_main_process:
        with open(checkpoints_dir / "action_normalizer.json", "w") as f:
            json.dump(action_stats, f, indent=2)

    if writer is not None:
        writer.add_text("bridge/upstream_family", args.upstream_family)
        writer.add_text("bridge/policy_type", args.policy_type)
        writer.add_scalar("bridge/num_sequences", len(dataset), 0)
        writer.add_scalar("bridge/state_dim", data_info["state_dim"], 0)
        writer.add_scalar("bridge/action_dim_raw", data_info["action_dim"], 0)

    if args.bridge_smoke_only:
        if writer is not None:
            writer.flush()
            writer.close()
        print("Bridge smoke only completed.")
        print(f"Run metadata written to: {checkpoints_dir}")
        return

    if args.auto_export_lerobot_dataset and not args.lerobot_dataset_repo_id and not args.lerobot_dataset_path:
        export_output_dir = Path(args.export_output_dir or str(run_dir / "lerobot_dataset")).expanduser().resolve()
        args.lerobot_dataset_path = str(export_output_dir)
        if world_size == 1 or is_main_process:
            export_result = export_carm_to_lerobot_dataset(
                demo_path=demo_path,
                output_dir=str(export_output_dir),
                contract=contract,
                num_episodes=args.num_demos,
            )
            args.lerobot_dataset_path = export_result["dataset_path"]
            Path(args.lerobot_dataset_path, ".export_done").write_text("ok\n")
            if is_main_process:
                with open(checkpoints_dir / "lerobot_export.json", "w") as f:
                    json.dump(export_result, f, indent=2)
        else:
            _wait_for_export(export_output_dir)

    elif world_size > 1 and args.lerobot_dataset_path:
        _wait_for_export(Path(args.lerobot_dataset_path).expanduser().resolve())

    lerobot_dataset_validation = None
    if args.lerobot_dataset_path and (world_size == 1 or is_main_process):
        lerobot_dataset_validation = validate_lerobot_dataset_path(args.lerobot_dataset_path)
        if is_main_process:
            with open(checkpoints_dir / "lerobot_dataset_validation.json", "w") as f:
                json.dump(lerobot_dataset_validation, f, indent=2)
        if args.strict and not lerobot_dataset_validation["summary"]["ok"]:
            raise RuntimeError("LeRobot dataset validation failed in strict mode. See lerobot_dataset_validation.json.")

    env, _ = build_probe_environment(args.upstream_repo_path)
    cli_check = next((check for check in probe_result["checks"] if check["name"] == "lerobot_train_cli"), None)
    lerobot_train = None
    if cli_check is not None:
        lerobot_train = cli_check.get("details", {}).get("resolved_path")

    command = None
    command_validation = None
    dispatch_run_dir = _ensure_unique_dispatch_dir(run_dir.resolve()) if args.dispatch_to_lerobot else run_dir.resolve()
    if lerobot_train is not None or args.lerobot_dataset_repo_id or args.lerobot_dataset_path:
        command = build_lerobot_train_command(args, str(dispatch_run_dir))
        if lerobot_train is not None:
            command = [lerobot_train, *command[1:]]
        if is_main_process:
            with open(checkpoints_dir / "lerobot_train_command.json", "w") as f:
                json.dump(command, f, indent=2)
        command_validation = validate_lerobot_train_command(args, command, probe_result)
        if is_main_process:
            with open(checkpoints_dir / "lerobot_train_command_validation.json", "w") as f:
                json.dump(command_validation, f, indent=2)
        if args.strict and not command_validation["summary"]["ok"]:
            raise RuntimeError("LeRobot command validation failed in strict mode. See lerobot_train_command_validation.json.")

    if args.bridge_validate_only:
        if writer is not None:
            writer.flush()
            writer.close()
        print("Bridge validation only completed.")
        print(f"Run metadata written to: {checkpoints_dir}")
        return

    if args.dispatch_to_lerobot:
        if lerobot_train is None or command is None:
            raise FileNotFoundError(
                "Could not find 'lerobot-train'. Install LeRobot or provide upstream_repo_path with the CLI on PATH."
            )
        print("Dispatching to LeRobot:")
        print(" ".join(command))
        subprocess.run(command, check=True, env=env)
    else:
        if command is not None and lerobot_train is not None:
            print("Prepared LeRobot command:")
            print(" ".join(command))
        else:
            print("LeRobot CLI not found; bridge metadata only for now.")
            if args.fail_if_upstream_missing:
                raise FileNotFoundError(
                    "Could not find 'lerobot-train'. Install LeRobot or set upstream_repo_path."
                )

    if writer is not None:
        writer.flush()
        writer.close()

    print("train_pi05 completed successfully.")
    print(f"Run metadata written to: {checkpoints_dir}")


if __name__ == "__main__":
    main()
