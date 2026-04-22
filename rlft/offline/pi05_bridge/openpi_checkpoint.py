from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlparse
from urllib.request import urlopen


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


def _resolve_openpi_cache_path(checkpoint_uri: str, cache_dir: str | None = None) -> Path:
    parsed = urlparse(checkpoint_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Only gs:// checkpoint URIs are supported, got: {checkpoint_uri}")
    root = Path(cache_dir or "~/.cache/openpi").expanduser().resolve()
    return root / parsed.netloc / parsed.path.strip("/")


def _list_public_gcs_objects(bucket: str, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    page_token: str | None = None
    base_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token
        url = f"{base_url}?{urlencode(params)}"
        with urlopen(url) as response:
            payload = json.loads(response.read().decode("utf-8"))
        objects.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return objects


def _download_http_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _relative_public_gcs_name(prefix: str, object_name: str) -> str:
    return object_name[len(prefix):].lstrip("/")


def _download_http_file_checked(url: str, destination: Path, expected_size: int, max_retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, max_retries + 1):
        if tmp_path.exists():
            tmp_path.unlink()
        wget_bin = shutil.which("wget")
        if wget_bin is not None:
            subprocess.run(
                [
                    wget_bin,
                    "--quiet",
                    "--tries=3",
                    "--timeout=60",
                    "-O",
                    str(tmp_path),
                    url,
                ],
                check=True,
            )
        else:
            with urlopen(url) as response, tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        actual_size = tmp_path.stat().st_size
        if actual_size == expected_size:
            tmp_path.replace(destination)
            return
        if attempt == max_retries:
            raise IOError(
                f"Downloaded size mismatch for {destination}: expected {expected_size} bytes, got {actual_size}"
            )


def _public_gcs_checkpoint_complete(local_path: Path, prefix: str, objects: list[dict[str, Any]]) -> bool:
    if not local_path.exists():
        return False
    expected: dict[str, int] = {}
    for obj in objects:
        relative_name = _relative_public_gcs_name(prefix, obj["name"])
        if not relative_name:
            continue
        expected[relative_name] = int(obj["size"])
    if not expected:
        return False
    for relative_name, expected_size in expected.items():
        file_path = local_path / relative_name
        if not file_path.is_file() or file_path.stat().st_size != expected_size:
            return False
    return True


def _download_public_openpi_checkpoint_http(checkpoint_uri: str, cache_dir: str | None = None) -> Path:
    parsed = urlparse(checkpoint_uri)
    if parsed.scheme != "gs" or parsed.netloc != "openpi-assets":
        raise ValueError(f"HTTP fallback only supports public openpi-assets checkpoints, got: {checkpoint_uri}")

    local_path = _resolve_openpi_cache_path(checkpoint_uri, cache_dir)
    prefix = parsed.path.strip("/")
    objects = _list_public_gcs_objects(parsed.netloc, prefix)
    if not objects:
        raise FileNotFoundError(f"No public GCS objects found for {checkpoint_uri}")
    if _public_gcs_checkpoint_complete(local_path, prefix, objects):
        return local_path

    scratch_path = local_path.with_suffix(".partial")
    if scratch_path.exists():
        shutil.rmtree(scratch_path)
    scratch_path.mkdir(parents=True, exist_ok=True)

    def download_object(obj: dict[str, Any]) -> None:
        name = obj["name"]
        relative_name = _relative_public_gcs_name(prefix, name)
        if not relative_name:
            return
        media_url = f"https://storage.googleapis.com/{parsed.netloc}/{name}"
        expected_size = int(obj["size"])
        existing_path = local_path / relative_name
        target_path = scratch_path / relative_name
        if existing_path.is_file() and existing_path.stat().st_size == expected_size:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing_path, target_path)
            return
        _download_http_file_checked(media_url, target_path, expected_size)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_object, obj) for obj in objects]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    if local_path.exists():
        shutil.rmtree(local_path)
    shutil.move(str(scratch_path), str(local_path))
    return local_path


def _download_openpi_checkpoint(repo: Path, checkpoint_uri: str, cache_dir: str | None = None, force_download: bool = False) -> Path:
    parsed = urlparse(checkpoint_uri)
    if parsed.scheme == "gs" and parsed.netloc == "openpi-assets":
        local_path = _resolve_openpi_cache_path(checkpoint_uri, cache_dir)
        if local_path.exists() and force_download:
            shutil.rmtree(local_path)
        return _download_public_openpi_checkpoint_http(checkpoint_uri, cache_dir)

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
    input_features = {
        "observation.image": {
            "type": "VISUAL",
            "shape": [3, 224, 224],
        },
        "observation.state": {
            "type": "STATE",
            "shape": [32],
        },
    }
    output_features = {
        "action": {
            "type": "ACTION",
            "shape": [action_dim],
        }
    }
    normalization_mapping = {
        "VISUAL": "IDENTITY",
        "STATE": "QUANTILES",
        "ACTION": "QUANTILES",
    }
    config = {
        "type": "pi05",
        "n_obs_steps": 1,
        "input_features": input_features,
        "output_features": output_features,
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
        "normalization_mapping": normalization_mapping,
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


def _resolve_pi05_tokenizer_name() -> str:
    tokenizer_override = os.environ.get("PI05_TOKENIZER_PATH")
    if tokenizer_override:
        candidate = Path(tokenizer_override).expanduser().resolve()
        if candidate.exists():
            return str(candidate)
    return "google/paligemma-3b-pt-224"


def _write_lerobot_pi05_processors(pretrained_dir: Path, *, action_dim: int) -> None:
    input_features = {
        "observation.image": {
            "type": "VISUAL",
            "shape": [3, 224, 224],
        },
        "observation.state": {
            "type": "STATE",
            "shape": [32],
        },
    }
    output_features = {
        "action": {
            "type": "ACTION",
            "shape": [action_dim],
        }
    }
    normalization_mapping = {
        "VISUAL": "IDENTITY",
        "STATE": "QUANTILES",
        "ACTION": "QUANTILES",
    }

    preprocessor = {
        "name": "policy_preprocessor",
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {}},
            },
            {
                "registry_name": "to_batch_processor",
                "config": {},
            },
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "eps": 1e-08,
                    "features": {**input_features, **output_features},
                    "norm_map": normalization_mapping,
                },
            },
            {
                "registry_name": "pi05_prepare_state_tokenizer_processor_step",
                "config": {"max_state_dim": 32, "task_key": "task"},
            },
            {
                "registry_name": "tokenizer_processor",
                "config": {
                    "tokenizer_name": _resolve_pi05_tokenizer_name(),
                    "max_length": 200,
                    "task_key": "task",
                    "padding_side": "right",
                    "padding": "max_length",
                    "truncation": True,
                },
            },
            {
                "registry_name": "device_processor",
                "config": {"device": "cuda", "float_dtype": None},
            },
        ],
    }
    postprocessor = {
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": "unnormalizer_processor",
                "config": {
                    "eps": 1e-08,
                    "features": output_features,
                    "norm_map": normalization_mapping,
                },
            },
            {
                "registry_name": "device_processor",
                "config": {"device": "cpu", "float_dtype": None},
            },
        ],
    }

    (pretrained_dir / "policy_preprocessor.json").write_text(json.dumps(preprocessor, indent=2) + "\n")
    (pretrained_dir / "policy_postprocessor.json").write_text(json.dumps(postprocessor, indent=2) + "\n")


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
    add("policy_preprocessor", (pretrained_dir / "policy_preprocessor.json").exists(), "Found policy_preprocessor.json" if (pretrained_dir / "policy_preprocessor.json").exists() else "Missing policy_preprocessor.json", {"path": str(pretrained_dir / 'policy_preprocessor.json')})
    add("policy_postprocessor", (pretrained_dir / "policy_postprocessor.json").exists(), "Found policy_postprocessor.json" if (pretrained_dir / "policy_postprocessor.json").exists() else "Missing policy_postprocessor.json", {"path": str(pretrained_dir / 'policy_postprocessor.json')})

    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        config = PreTrainedConfig.from_pretrained(pretrained_dir)
        add(
            "image_feature_configured",
            bool(getattr(config, "image_features", {})),
            "Configured at least one image feature" if getattr(config, "image_features", {}) else "No image features configured in config.json",
            {"image_features": list(getattr(config, "image_features", {}).keys())},
        )
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
    _write_lerobot_pi05_processors(pytorch_dir, action_dim=32)

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
