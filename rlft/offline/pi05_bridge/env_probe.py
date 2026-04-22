from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


NON_FATAL_CHECKS = {"lerobot_train_help"}


def _check_result(name: str, ok: bool, message: str, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return {
        "name": name,
        "ok": ok,
        "message": message,
        "details": details or {},
    }


def _probe_import(module_name: str) -> tuple[bool, str, dict[str, Any]]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False, f"Module '{module_name}' not importable", {"module": module_name}
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, f"Module '{module_name}' import failed: {exc}", {"module": module_name}

    version = getattr(module, "__version__", None)
    return True, f"Module '{module_name}' importable", {"module": module_name, "version": version}


def _run_command(command: list[str], env: dict[str, str], timeout: float = 5.0) -> tuple[bool, str, dict[str, Any]]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except FileNotFoundError:
        return False, f"Command not found: {command[0]}", {"command": command}
    except subprocess.TimeoutExpired:
        return True, f"Command timed out after {timeout}s but appears runnable: {' '.join(command)}", {"command": command, "timed_out": True}
    except Exception as exc:
        return False, f"Command failed to run: {exc}", {"command": command}

    ok = result.returncode == 0
    message = "Command succeeded" if ok else f"Command exited with code {result.returncode}"
    return ok, message, {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def _prepend_env_path(existing: Optional[str], prefix: str) -> str:
    if not existing:
        return prefix
    parts = [part for part in existing.split(os.pathsep) if part]
    if prefix in parts:
        return existing
    return os.pathsep.join([prefix, *parts])


def _is_valid_cuda_root(path: Path) -> bool:
    return (path / "bin" / "nvcc").is_file()


def _iter_cuda_root_candidates(env: dict[str, str]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Optional[Path]) -> None:
        if path is None:
            return
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    for key in ("CUDA_HOME", "CUDA_PATH"):
        value = env.get(key)
        if value:
            add(Path(value))

    nvcc_path = shutil.which("nvcc", path=env.get("PATH"))
    if nvcc_path:
        add(Path(nvcc_path).resolve().parent.parent)

    add(Path("/usr/local/cuda"))
    add(Path("/opt/cuda"))
    add(Path("/usr/lib/cuda"))

    for base_dir in (Path("/usr/local"), Path("/opt"), Path("/usr/lib")):
        if not base_dir.exists():
            continue
        for candidate in sorted(base_dir.glob("cuda-*"), key=lambda path: path.name, reverse=True):
            add(candidate)

    return candidates


def enrich_env_with_cuda_home(env: dict[str, str]) -> dict[str, str]:
    for candidate in _iter_cuda_root_candidates(env):
        if not _is_valid_cuda_root(candidate):
            continue
        cuda_home = str(candidate)
        env["CUDA_HOME"] = cuda_home
        env["CUDA_PATH"] = cuda_home
        env["PATH"] = _prepend_env_path(env.get("PATH"), str(candidate / "bin"))
        for lib_dir_name in ("lib64", "lib"):
            lib_dir = candidate / lib_dir_name
            if lib_dir.is_dir():
                env["LD_LIBRARY_PATH"] = _prepend_env_path(env.get("LD_LIBRARY_PATH"), str(lib_dir))
                break
        break
    return env


def build_probe_environment(upstream_repo_path: Optional[str] = None) -> tuple[dict[str, str], Optional[str]]:
    env = enrich_env_with_cuda_home(os.environ.copy())
    resolved_repo_path = None
    if upstream_repo_path:
        resolved_repo_path = str(Path(upstream_repo_path).expanduser().resolve())
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = resolved_repo_path if not existing else f"{resolved_repo_path}:{existing}"
    return env, resolved_repo_path


def probe_lerobot_environment(upstream_repo_path: Optional[str] = None) -> dict[str, Any]:
    env, resolved_repo_path = build_probe_environment(upstream_repo_path)
    checks: list[dict[str, Any]] = []

    checks.append(
        _check_result(
            "python_runtime",
            True,
            "Captured Python runtime information",
            {
                "executable": sys.executable,
                "version": sys.version,
                "platform": platform.platform(),
                "cwd": os.getcwd(),
                "conda_prefix": os.environ.get("CONDA_PREFIX"),
                "path": env.get("PATH"),
                "pythonpath": env.get("PYTHONPATH"),
                "cuda_home": env.get("CUDA_HOME"),
                "cuda_path": env.get("CUDA_PATH"),
                "ld_library_path": env.get("LD_LIBRARY_PATH"),
                "upstream_repo_path": resolved_repo_path,
            },
        )
    )

    for module_name in ["torch", "numpy", "h5py", "tyro", "lerobot"]:
        ok, message, details = _probe_import(module_name)
        checks.append(_check_result(f"import_{module_name}", ok, message, details))

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_ok else 0
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda_ok else []
        checks.append(
            _check_result(
                "torch_cuda",
                True,
                "Collected torch/CUDA status",
                {
                    "torch_version": torch.__version__,
                    "torch_cuda_version": torch.version.cuda,
                    "cuda_available": cuda_ok,
                    "device_count": device_count,
                    "device_names": device_names,
                },
            )
        )
    except Exception as exc:
        checks.append(_check_result("torch_cuda", False, f"Failed to collect torch/CUDA status: {exc}"))

    lerobot_train = shutil.which("lerobot-train", path=env.get("PATH"))
    checks.append(
        _check_result(
            "lerobot_train_cli",
            lerobot_train is not None,
            "Found lerobot-train CLI" if lerobot_train else "lerobot-train CLI not found on PATH",
            {"resolved_path": lerobot_train},
        )
    )

    if lerobot_train is not None:
        ok, message, details = _run_command([lerobot_train, "--help"], env=env, timeout=15.0)
        checks.append(_check_result("lerobot_train_help", ok, message, details))

    summary_ok = all(check["ok"] for check in checks if check["name"] not in {"python_runtime", *NON_FATAL_CHECKS})
    return {
        "summary": {
            "ok": summary_ok,
            "num_checks": len(checks),
        },
        "checks": checks,
    }


def main() -> None:
    upstream_repo_path = sys.argv[1] if len(sys.argv) > 1 else None
    result = probe_lerobot_environment(upstream_repo_path=upstream_repo_path)
    print(json.dumps(result, indent=2))
    if not result["summary"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
