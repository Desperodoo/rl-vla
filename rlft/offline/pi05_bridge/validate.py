from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from .contract import Pi05BridgeContract
from .dataset_bridge import build_pi05_dataset_bridge


EXPECTED_IMAGE_RANK = 4
EXPECTED_STATE_RANK = 2
EXPECTED_ACTION_RANK = 2


def _result(name: str, ok: bool, message: str, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return {
        "name": name,
        "ok": ok,
        "message": message,
        "details": details or {},
    }


def _require_lerobot_dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "LeRobotDataset is unavailable. Install Hugging Face LeRobot first, e.g. 'pip install lerobot'."
        ) from exc
    return LeRobotDataset


def _load_lerobot_dataset(local_dataset_dir: Path, repo_id: str = "carm/pi05_local"):
    LeRobotDataset = _require_lerobot_dataset_class()
    constructors = [
        lambda: LeRobotDataset(repo_id=repo_id, root=local_dataset_dir),
        lambda: LeRobotDataset(repo_id=repo_id, root=str(local_dataset_dir)),
        lambda: LeRobotDataset(repo_id=repo_id),
    ]

    last_err: Exception | None = None
    for ctor in constructors:
        try:
            dataset = ctor()
            _ = len(dataset)
            return dataset
        except TypeError as exc:
            last_err = exc
            continue
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Failed to load LeRobotDataset from {local_dataset_dir}: {last_err}")


def validate_bridge_dataset(
    data_path: str,
    contract: Pi05BridgeContract,
    num_episodes: Optional[int] = None,
    normalize_actions: bool = True,
    action_norm_mode: str = "standard",
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    dataset = build_pi05_dataset_bridge(
        data_path=data_path,
        contract=contract,
        num_episodes=num_episodes,
        normalize_actions=normalize_actions,
        action_norm_mode=action_norm_mode,
    )

    checks.append(
        _result(
            "dataset_nonempty",
            len(dataset) > 0,
            "Bridge dataset has samples" if len(dataset) > 0 else "Bridge dataset is empty",
            {"num_sequences": len(dataset)},
        )
    )

    if len(dataset) > 0:
        sample = dataset[0]
        expected_keys = {
            contract.observation.image_key,
            contract.observation.state_key,
            contract.observation.ee_pose_key,
            contract.action.action_key,
            "action_unnormalized",
            "episode_index",
            "start_index",
        }
        checks.append(
            _result(
                "sample_keys",
                expected_keys.issubset(sample.keys()),
                "Sample keys match contract" if expected_keys.issubset(sample.keys()) else "Sample keys missing required entries",
                {
                    "expected_keys": sorted(expected_keys),
                    "actual_keys": sorted(sample.keys()),
                },
            )
        )

        image = sample[contract.observation.image_key]
        state = sample[contract.observation.state_key]
        ee_pose = sample[contract.observation.ee_pose_key]
        action = sample[contract.action.action_key]

        checks.extend(
            [
                _result(
                    "image_tensor",
                    torch.is_tensor(image) and image.ndim == EXPECTED_IMAGE_RANK,
                    "Image tensor shape is valid" if torch.is_tensor(image) and image.ndim == EXPECTED_IMAGE_RANK else "Image tensor rank mismatch",
                    {"shape": tuple(image.shape), "dtype": str(image.dtype)},
                ),
                _result(
                    "state_tensor",
                    torch.is_tensor(state) and state.ndim == EXPECTED_STATE_RANK,
                    "State tensor shape is valid" if torch.is_tensor(state) and state.ndim == EXPECTED_STATE_RANK else "State tensor rank mismatch",
                    {"shape": tuple(state.shape), "dtype": str(state.dtype)},
                ),
                _result(
                    "ee_pose_tensor",
                    torch.is_tensor(ee_pose) and ee_pose.ndim == EXPECTED_STATE_RANK and ee_pose.shape[-1] == 7,
                    "EE pose tensor shape is valid" if torch.is_tensor(ee_pose) and ee_pose.ndim == EXPECTED_STATE_RANK and ee_pose.shape[-1] == 7 else "EE pose tensor shape mismatch",
                    {"shape": tuple(ee_pose.shape), "dtype": str(ee_pose.dtype)},
                ),
                _result(
                    "action_tensor",
                    torch.is_tensor(action) and action.ndim == EXPECTED_ACTION_RANK and action.shape[-1] == contract.action.target_dim,
                    "Action tensor shape is valid" if torch.is_tensor(action) and action.ndim == EXPECTED_ACTION_RANK and action.shape[-1] == contract.action.target_dim else "Action tensor shape mismatch",
                    {"shape": tuple(action.shape), "dtype": str(action.dtype)},
                ),
            ]
        )

    summary_ok = all(check["ok"] for check in checks)
    return {"summary": {"ok": summary_ok}, "checks": checks}


def validate_lerobot_dataset_path(dataset_path: str | Path, repo_id: str = "carm/pi05_local") -> dict[str, Any]:
    path = Path(dataset_path).expanduser().resolve()
    checks: list[dict[str, Any]] = []

    checks.append(_result("path_exists", path.exists(), "Dataset path exists" if path.exists() else "Dataset path does not exist", {"path": str(path)}))
    if path.exists():
        checks.append(_result("path_is_dir", path.is_dir(), "Dataset path is a directory" if path.is_dir() else "Dataset path is not a directory", {"path": str(path)}))
        checks.append(
            _result(
                "meta_info_exists",
                (path / "meta" / "info.json").exists(),
                "Found meta/info.json" if (path / "meta" / "info.json").exists() else "Missing meta/info.json",
                {"path": str(path)},
            )
        )
        checks.append(
            _result(
                "meta_tasks_exists",
                (path / "meta" / "tasks.parquet").exists(),
                "Found meta/tasks.parquet" if (path / "meta" / "tasks.parquet").exists() else "Missing meta/tasks.parquet",
                {"path": str(path)},
            )
        )
        checks.append(
            _result(
                "meta_episodes_exists",
                any((path / "meta" / "episodes").glob("*/*.parquet")) if (path / "meta" / "episodes").exists() else False,
                "Found meta/episodes parquet files" if (path / "meta" / "episodes").exists() and any((path / "meta" / "episodes").glob("*/*.parquet")) else "Missing meta/episodes parquet files",
                {"path": str(path)},
            )
        )
        checks.append(
            _result(
                "data_parquet_exists",
                any((path / "data").glob("*/*.parquet")) if (path / "data").exists() else False,
                "Found data parquet files" if (path / "data").exists() and any((path / "data").glob("*/*.parquet")) else "Missing data parquet files",
                {"path": str(path)},
            )
        )
        try:
            dataset = _load_lerobot_dataset(path, repo_id=repo_id)
            checks.append(_result("lerobot_dataset_load", True, "Loaded LeRobotDataset successfully", {"length": len(dataset)}))
        except Exception as exc:
            checks.append(_result("lerobot_dataset_load", False, f"Failed to load LeRobotDataset: {exc}", {"path": str(path), "repo_id": repo_id}))

    summary_ok = all(check["ok"] for check in checks)
    return {"summary": {"ok": summary_ok}, "checks": checks}


def validate_lerobot_train_command(args: Any, command: list[str], probe_result: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    dataset_repo_id = getattr(args, "lerobot_dataset_repo_id", None)
    dataset_path = getattr(args, "lerobot_dataset_path", None)
    has_exactly_one_dataset = bool(dataset_repo_id) ^ bool(dataset_path)
    checks.append(
        _result(
            "dataset_selector",
            has_exactly_one_dataset,
            "Exactly one dataset selector provided" if has_exactly_one_dataset else "Must provide exactly one of lerobot_dataset_repo_id or lerobot_dataset_path",
            {"repo_id": dataset_repo_id, "dataset_path": dataset_path},
        )
    )

    cli_check = next((check for check in probe_result.get("checks", []) if check.get("name") == "lerobot_train_cli"), None)
    cli_ok = bool(cli_check and cli_check.get("ok"))
    checks.append(_result("cli_available", cli_ok, "LeRobot CLI is available" if cli_ok else "LeRobot CLI unavailable", {"cli_check": cli_check}))

    policy_pretrained_path = getattr(args, "policy_pretrained_path", None)
    peft_requires_pretrained = (not getattr(args, "use_peft", False)) or bool(policy_pretrained_path)
    checks.append(
        _result(
            "peft_pretrained_path",
            peft_requires_pretrained,
            "PEFT pretrained path is configured" if peft_requires_pretrained else "use_peft=True requires policy_pretrained_path",
            {"use_peft": getattr(args, "use_peft", False), "policy_pretrained_path": policy_pretrained_path},
        )
    )

    required_prefixes = ["--policy.type=", "--job_name=", "--output_dir=", "--seed=", "--batch_size=", "--steps=", "--optimizer.lr="]
    present = {prefix: any(token.startswith(prefix) for token in command) for prefix in required_prefixes}
    checks.append(
        _result(
            "command_required_flags",
            all(present.values()),
            "All required command flags are present" if all(present.values()) else "Missing required command flags",
            {"command": command, "required_flags": present},
        )
    )

    summary_ok = all(check["ok"] for check in checks)
    return {"summary": {"ok": summary_ok}, "checks": checks}
