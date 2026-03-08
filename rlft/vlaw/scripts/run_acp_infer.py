"""CLI: ACP advantage 推理与标注。

用法:
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python rlft/vlaw/scripts/run_acp_infer.py \\
        --checkpoint_path checkpoints/vlaw/acp/iter1/best.safetensors
"""

from __future__ import annotations

import tyro

from rlft.vlaw.acp.config import ACPInferConfig
from rlft.vlaw.acp.infer_values import ACPAnnotator


def main() -> None:
    cfg = tyro.cli(ACPInferConfig)
    annotator = ACPAnnotator(cfg, device="cuda:0")
    result = annotator.run()
    print(f"[ACP] 推理标注结果: {result}")


if __name__ == "__main__":
    main()
