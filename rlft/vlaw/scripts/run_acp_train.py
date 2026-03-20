"""CLI: ACP value model 训练。

用法:
    CUDA_VISIBLE_DEVICES=6,7 conda run -n vlaw_reward python -m rlft.acp.train_value_model \\
        --num_steps 8000 --batch_size 32
"""

from __future__ import annotations

import tyro

from rlft.acp.config import ACPTrainConfig
from rlft.acp.train_value_model import ACPValueTrainer


def main() -> None:
    cfg = tyro.cli(ACPTrainConfig)
    trainer = ACPValueTrainer(cfg, device="cuda:0")
    result = trainer.train()
    print(f"[ACP] 训练结果: {result}")


if __name__ == "__main__":
    main()
