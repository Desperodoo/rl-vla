from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _resolve_openpi_repo() -> Path:
    repo = Path("/tmp/openpi_for_patch").resolve()
    if not repo.exists():
        raise FileNotFoundError(
            "Expected local OpenPI repo at /tmp/openpi_for_patch. "
            "Clone Physical-Intelligence/openpi there first."
        )
    return repo


def _run_python(repo: Path, code: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env.update(env or {})
    merged_env["PYTHONPATH"] = str(repo / "src") + (
        ":" + merged_env["PYTHONPATH"] if "PYTHONPATH" in merged_env and merged_env["PYTHONPATH"] else ""
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo),
        env=merged_env,
        check=True,
        capture_output=True,
        text=True,
    )


def _download_openpi_checkpoint(repo: Path, checkpoint_uri: str, cache_dir: str | None = None, force_download: bool = False) -> Path:
    env: dict[str, str] = {}
    if cache_dir:
        env["OPENPI_DATA_HOME"] = str(Path(cache_dir).expanduser().resolve())
    code = (
        "import datetime\n"
        "if not hasattr(datetime, 'UTC'): datetime.UTC = datetime.timezone.utc\n"
        "from openpi.shared import download\n"
        f"path = download.maybe_download({checkpoint_uri!r}, force_download={force_download!r})\n"
        "print(path)\n"
    )
    result = _run_python(repo, code, env=env)
    path = Path(result.stdout.strip().splitlines()[-1]).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"OpenPI checkpoint was not downloaded to {path}")
    return path


def _infer_config_name(checkpoint_name: str) -> str:
    mapping = {
        "pi05_base": "pi05_droid",
        "pi05_droid": "pi05_droid",
        "pi05_libero": "pi05_libero",
    }
    if checkpoint_name not in mapping:
        raise ValueError(
            f"Unsupported OpenPI pi05 checkpoint '{checkpoint_name}'. "
            f"Expected one of: {sorted(mapping)}"
        )
    return mapping[checkpoint_name]


def _write_lerobot_pi05_config(
    output_dir: Path,
    *,
    action_dim: int,
    action_horizon: int,
    paligemma_variant: str,
    action_expert_variant: str,
    precision: str,
) -> None:
    config = {
        "type": "pi05",
        "n_obs_steps": 1,
        "input_features": {},
        "output_features": {},
        "device": None,
        "use_amp": False,
        "use_peft": False,
        "push_to_hub": False,
        "repo_id": None,
        "private": None,
        "tags": None,
        "license": None,
        "pretrained_path": None,
        "paligemma_variant": paligemma_variant,
        "action_expert_variant": action_expert_variant,
        "dtype": precision,
        "chunk_size": action_horizon,
        "n_action_steps": action_horizon,
        "max_state_dim": 32,
        "max_action_dim": action_dim,
        "num_inference_steps": 10,
        "time_sampling_beta_alpha": 1.5,
        "time_sampling_beta_beta": 1.0,
        "time_sampling_scale": 0.999,
        "time_sampling_offset": 0.001,
        "min_period": 0.004,
        "max_period": 4.0,
        "rtc_config": None,
        "image_resolution": [224, 224],
        "empty_cameras": 0,
        "tokenizer_max_length": 200,
        "normalization_mapping": {
            "VISUAL": "IDENTITY",
            "STATE": "QUANTILES",
            "ACTION": "QUANTILES",
        },
        "gradient_checkpointing": False,
        "compile_model": False,
        "compile_mode": "max-autotune",
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "optimizer_lr": 2.5e-05,
        "optimizer_betas": [0.9, 0.95],
        "optimizer_eps": 1e-08,
        "optimizer_weight_decay": 0.01,
        "optimizer_grad_clip_norm": 1.0,
        "scheduler_warmup_steps": 1000,
        "scheduler_decay_steps": 30000,
        "scheduler_decay_lr": 2.5e-06,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def _convert_jax_checkpoint_to_pytorch(
    repo: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    config_name: str,
    precision: str = "bfloat16",
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "examples/convert_jax_model_to_pytorch.py",
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--config_name",
            config_name,
            "--output_path",
            str(output_dir),
            "--precision",
            precision,
        ],
        cwd=str(repo),
        env={
            **os.environ,
            "PYTHONPATH": (
                str(Path(__file__).resolve().parent)
                + ":"
                + str(repo / "src")
                + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ and os.environ["PYTHONPATH"] else "")
            ),
        },
        check=True,
        text=True,
    )
    return output_dir


def _validate_lerobot_pretrained_dir(pretrained_dir: Path) -> dict[str, Any]:
    pretrained_dir = pretrained_dir.expanduser().resolve()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, message: str, details: dict[str, Any] | None = None) -> None:
        checks.append({"name": name, "ok": ok, "message": message, "details": details or {}})

    add("path_exists", pretrained_dir.exists(), "Pretrained directory exists" if pretrained_dir.exists() else "Pretrained directory missing", {"path": str(pretrained_dir)})
    add("config_json", (pretrained_dir / "config.json").exists(), "Found config.json" if (pretrained_dir / "config.json").exists() else "Missing config.json", {"path": str(pretrained_dir / 'config.json')})
    add("model_safetensors", (pretrained_dir / "model.safetensors").exists(), "Found model.safetensors" if (pretrained_dir / "model.safetensors").exists() else "Missing model.safetensors", {"path": str(pretrained_dir / 'model.safetensors')})

    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        config = PreTrainedConfig.from_pretrained(pretrained_dir)
        policy_cls = get_policy_class(config.type)
        _ = policy_cls.from_pretrained(pretrained_name_or_path=pretrained_dir, config=config, strict=False)
        add("lerobot_load", True, "LeRobot policy load succeeded", {"policy_type": config.type})
    except Exception as exc:
        add("lerobot_load", False, f"LeRobot policy load failed: {exc}", {"path": str(pretrained_dir)})

    return {"summary": {"ok": all(c["ok"] for c in checks)}, "checks": checks}


def prepare_openpi_pi05_checkpoint(
    checkpoint_name: str,
    *,
    cache_dir: str | None = None,
    output_dir: str | None = None,
    force_download: bool = False,
    force_reconvert: bool = False,
    precision: str = "bfloat16",
) -> dict[str, Any]:
    repo = _resolve_openpi_repo()
    checkpoint_uri = f"gs://openpi-assets/checkpoints/{checkpoint_name}"
    config_name = _infer_config_name(checkpoint_name)
    jax_checkpoint_dir = _download_openpi_checkpoint(repo, checkpoint_uri, cache_dir=cache_dir, force_download=force_download)

    if output_dir is None:
        pytorch_dir = jax_checkpoint_dir.parent / f"{jax_checkpoint_dir.name}_pytorch"
    else:
        pytorch_dir = Path(output_dir).expanduser().resolve()

    if pytorch_dir.exists() and force_reconvert:
        shutil.rmtree(pytorch_dir)

    if not (pytorch_dir / "config.json").exists() or not (pytorch_dir / "model.safetensors").exists():
        _convert_jax_checkpoint_to_pytorch(
            repo=repo,
            checkpoint_dir=jax_checkpoint_dir,
            output_dir=pytorch_dir,
            config_name=config_name,
            precision=precision,
        )

    _write_lerobot_pi05_config(
        output_dir=pytorch_dir,
        action_dim=32,
        action_horizon=15,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        precision=precision,
    )

    validation = _validate_lerobot_pretrained_dir(pytorch_dir)
    result = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_uri": checkpoint_uri,
        "config_name": config_name,
        "jax_checkpoint_dir": str(jax_checkpoint_dir),
        "lerobot_pretrained_path": str(pytorch_dir),
        "validation": validation,
    }

    metadata_path = pytorch_dir / "openpi_conversion_metadata.json"
    metadata_path.write_text(json.dumps(result, indent=2) + "\n")
    result["metadata_path"] = str(metadata_path)
    return result
