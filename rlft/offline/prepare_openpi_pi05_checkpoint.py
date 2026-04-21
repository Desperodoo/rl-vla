from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Optional

import tyro

from rlft.offline.pi05_bridge.openpi_checkpoint import prepare_openpi_pi05_checkpoint


@dataclass
class Args:
    checkpoint_name: Literal["pi05_base", "pi05_droid", "pi05_libero"] = "pi05_base"
    cache_dir: Optional[str] = "~/.cache/openpi"
    output_dir: Optional[str] = None
    force_download: bool = False
    force_reconvert: bool = False
    precision: Literal["float32", "bfloat16"] = "bfloat16"


def main() -> None:
    args = tyro.cli(Args)
    result = prepare_openpi_pi05_checkpoint(
        checkpoint_name=args.checkpoint_name,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        force_download=args.force_download,
        force_reconvert=args.force_reconvert,
        precision=args.precision,
    )
    print(json.dumps(result, indent=2))
    if not result["validation"]["summary"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
