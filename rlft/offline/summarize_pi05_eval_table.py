from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
    base_batch2_val: str
    base_batch2_test: str
    base_batch64_val: str
    base_batch64_test: str
    libero_batch2_val: str
    libero_batch2_test: str
    libero_batch64_val: str
    libero_batch64_test: str


def _load_metrics(path: str) -> tuple[float, float]:
    data = json.loads(Path(path).read_text())
    return float(data["mean_action_mse"]), float(data["mean_action_mae"])


def _fmt_row(model: str, setup: str, split: str, mse: float, mae: float) -> str:
    return f"| {model} | {setup} | {split} | {mse:.9f} | {mae:.9f} |"


def main() -> None:
    args = tyro.cli(Args)

    rows = [
        ("pi05_base", "batch2", "val", *_load_metrics(args.base_batch2_val)),
        ("pi05_base", "batch2", "test", *_load_metrics(args.base_batch2_test)),
        ("pi05_base", "batch64", "val", *_load_metrics(args.base_batch64_val)),
        ("pi05_base", "batch64", "test", *_load_metrics(args.base_batch64_test)),
        ("pi05_libero", "batch2", "val", *_load_metrics(args.libero_batch2_val)),
        ("pi05_libero", "batch2", "test", *_load_metrics(args.libero_batch2_test)),
        ("pi05_libero", "batch64", "val", *_load_metrics(args.libero_batch64_val)),
        ("pi05_libero", "batch64", "test", *_load_metrics(args.libero_batch64_test)),
    ]

    print("| model_init | train_setup | split | mean_action_mse | mean_action_mae |")
    print("| --- | --- | --- | ---: | ---: |")
    for row in rows:
        print(_fmt_row(*row))


if __name__ == "__main__":
    main()
