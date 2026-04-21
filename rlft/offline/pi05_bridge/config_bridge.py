from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_OPENPI_CHECKPOINT_ROOT = Path(
    os.environ.get("OPENPI_CHECKPOINT_ROOT", "~/openpi")
).expanduser()
DEFAULT_OPENPI_PI05_BASE_PRETRAINED_PATH = str(DEFAULT_OPENPI_CHECKPOINT_ROOT / "pi05_base_pytorch")
DEFAULT_OPENPI_PI05_DROID_PRETRAINED_PATH = str(DEFAULT_OPENPI_CHECKPOINT_ROOT / "pi05_droid_pytorch")
DEFAULT_OPENPI_PI05_LIBERO_PRETRAINED_PATH = str(DEFAULT_OPENPI_CHECKPOINT_ROOT / "pi05_libero_pytorch")

DEFAULT_OPENPI_PI05_PRETRAINED_PATHS = {
    "pi05_base": DEFAULT_OPENPI_PI05_BASE_PRETRAINED_PATH,
    "pi05_droid": DEFAULT_OPENPI_PI05_DROID_PRETRAINED_PATH,
    "pi05_libero": DEFAULT_OPENPI_PI05_LIBERO_PRETRAINED_PATH,
}

LEROBOT_POLICY_NAME_MAP = {
    "pi0.5": "pi05",
    "pi05": "pi05",
}


def resolve_target_image_size(visual_encoder_type: str, auto_image_size: bool = True) -> tuple[int, int]:
    """Mirror train_carm.py image-size defaults for bridge config generation."""
    if not auto_image_size:
        return (128, 128)
    if visual_encoder_type in {"resnet18", "resnet34", "resnet50"}:
        return (224, 224)
    return (128, 128)


def resolve_default_openpi_pi05_pretrained_path(checkpoint_name: str = "pi05_base") -> str:
    try:
        return DEFAULT_OPENPI_PI05_PRETRAINED_PATHS[checkpoint_name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Unsupported OpenPI pi05 checkpoint '{checkpoint_name}'. "
            f"Expected one of: {sorted(DEFAULT_OPENPI_PI05_PRETRAINED_PATHS)}"
        ) from exc


def build_pi05_run_config(args: Any, contract_metadata: dict, data_info: dict) -> dict:
    """Build a stable run config for a future LeRobot/pi0.5 trainer wrapper.

    This does not assume a specific upstream repo yet. It produces a normalized,
    serializable config blob that train/eval/deploy code can all consume.
    """
    target_image_size = resolve_target_image_size(
        getattr(args, "visual_encoder_type", "resnet18"),
        getattr(args, "auto_image_size", True),
    )
    policy_type = getattr(args, "policy_type", "pi0.5")
    lerobot_policy_type = LEROBOT_POLICY_NAME_MAP.get(policy_type, policy_type)

    return {
        "upstream": {
            "family": getattr(args, "upstream_family", "lerobot"),
            "policy_type": policy_type,
            "policy_type_lerobot": lerobot_policy_type,
            "repo_path": getattr(args, "upstream_repo_path", None),
            "dataset_repo_id": getattr(args, "lerobot_dataset_repo_id", None),
            "dataset_path": getattr(args, "lerobot_dataset_path", None),
            "policy_repo_id": getattr(args, "policy_repo_id", f"carm/{lerobot_policy_type}-smoke"),
            "policy_pretrained_path": getattr(args, "policy_pretrained_path", None),
            "official_checkpoint_name": getattr(args, "official_openpi_checkpoint_name", "pi05_base"),
            "push_to_hub": getattr(args, "policy_push_to_hub", False),
            "use_peft": getattr(args, "use_peft", False),
        },
        "data": {
            "demo_path": getattr(args, "demo_path", None),
            "num_demos": getattr(args, "num_demos", None),
            "data_info": data_info,
        },
        "bridge": contract_metadata,
        "training": {
            "exp_name": getattr(args, "exp_name", None),
            "seed": getattr(args, "seed", 1),
            "batch_size": getattr(args, "batch_size", 32),
            "total_iters": getattr(args, "total_iters", 1000),
            "lr": getattr(args, "lr", 1e-4),
            "num_dataload_workers": getattr(args, "num_dataload_workers", 0),
        },
        "observation": {
            "state_mode": getattr(args, "state_mode", "joint_only"),
            "target_image_size": list(target_image_size),
            "visual_encoder_type": getattr(args, "visual_encoder_type", "resnet18"),
        },
        "action": {
            "normalize_actions": getattr(args, "normalize_actions", True),
            "action_norm_mode": getattr(args, "action_norm_mode", "standard"),
        },
    }


def build_lerobot_train_command(args: Any, run_dir: str) -> list[str]:
    """Build a concrete LeRobot training command for the current CLI shape."""
    policy_type = getattr(args, "policy_type", "pi0.5")
    lerobot_policy_type = LEROBOT_POLICY_NAME_MAP.get(policy_type, policy_type)
    policy_repo_id = getattr(args, "policy_repo_id", f"carm/{lerobot_policy_type}-smoke")
    policy_pretrained_path = getattr(args, "policy_pretrained_path", None)
    if getattr(args, "use_official_openpi_checkpoint", False) and not policy_pretrained_path:
        checkpoint_name = getattr(args, "official_openpi_checkpoint_name", "pi05_base")
        policy_pretrained_path = resolve_default_openpi_pi05_pretrained_path(checkpoint_name)
    command = ["lerobot-train", f"--policy.type={lerobot_policy_type}"]

    dataset_repo_id = getattr(args, "lerobot_dataset_repo_id", None)
    dataset_path = getattr(args, "lerobot_dataset_path", None)
    if dataset_repo_id:
        command.append(f"--dataset.repo_id={dataset_repo_id}")
    elif dataset_path:
        command.append(f"--dataset.repo_id=carm/pi05_local")
        command.append(f"--dataset.root={dataset_path}")
    else:
        raise ValueError(
            "LeRobot training requires either lerobot_dataset_repo_id or lerobot_dataset_path."
        )

    command.extend(
        [
            f"--policy.repo_id={policy_repo_id}",
            f"--policy.push_to_hub={str(getattr(args, 'policy_push_to_hub', False)).lower()}",
            f"--job_name={getattr(args, 'exp_name', 'pi05')}",
            f"--output_dir={run_dir}",
            f"--seed={getattr(args, 'seed', 1)}",
            f"--batch_size={getattr(args, 'batch_size', 8)}",
            f"--steps={getattr(args, 'total_iters', 1000)}",
            f"--optimizer.lr={getattr(args, 'lr', 1e-4)}",
            "--policy.gradient_checkpointing=true",
            "--policy.freeze_vision_encoder=true",
            "--policy.train_expert_only=true",
            "--policy.dtype=bfloat16",
        ]
    )

    if policy_pretrained_path:
        command.append(f"--policy.pretrained_path={policy_pretrained_path}")

    if getattr(args, 'use_peft', False):
        if not policy_pretrained_path:
            raise ValueError(
                "PEFT/LoRA requires policy_pretrained_path. "
                "LeRobot does not support training pi05 from scratch with use_peft=True."
            )
        command.append(f"--peft.method_type={getattr(args, 'peft_method_type', 'LORA')}")
        command.append(f"--peft.r={getattr(args, 'peft_r', 16)}")
        peft_target_modules = getattr(args, 'peft_target_modules', None)
        if peft_target_modules:
            command.append(f"--peft.target_modules={peft_target_modules}")
        peft_full_training_modules = getattr(args, 'peft_full_training_modules', None)
        if peft_full_training_modules:
            command.append(f"--peft.full_training_modules={peft_full_training_modules}")

    return command
