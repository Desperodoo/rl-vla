"""训练 State Predictor MLP on D_real rollout data.

每个任务 obs_dim 可能不同（LiftPeg/StackCube=25, PickCube=29），
因此分任务训练——每个任务保存独立 checkpoint。

用法示例:
    CUDA_VISIBLE_DEVICES=8 python rlft/vlaw/policy/train_state_predictor.py \\
        --data_dir data/vlaw/rollouts/iter1 \\
        --output_dir checkpoints/vlaw/state_predictor \\
        --max_steps 5000 \\
        --batch_size 256 \\
        --hidden_dim 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证 rlft 包可找到
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rlft.vlaw.policy.state_predictor import StatePredictorConfig, StatePredictorTrainer


# 任务 → (子目录名, obs_dim)
TASK_CONFIGS = {
    "LiftPegUpright-v1": 25,
    "PickCube-v1": 29,
    "StackCube-v1": 25,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train State Predictor per task")
    parser.add_argument(
        "--data_dir",
        default="data/vlaw/rollouts/iter1",
        help="包含各任务子目录的根路径（每个子目录内有 .h5 文件）",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_CONFIGS.keys()),
        help="要训练的任务目录名列表",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--output_dir",
        default="checkpoints/vlaw/state_predictor",
        help="checkpoint 根目录，每个任务保存至 output_dir/{task}/",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    out_root = Path(args.output_dir)
    results: dict[str, dict] = {}

    for task in args.tasks:
        obs_dim = TASK_CONFIGS.get(task, 25)
        task_data_dir = str(data_root / task)
        task_ckpt_dir = str(out_root / task)

        print(f"\n{'='*60}")
        print(f"[train_state_predictor] Task={task}, obs_dim={obs_dim}")
        print(f"  data_dir={task_data_dir}")
        print(f"  ckpt_dir={task_ckpt_dir}")
        print(f"{'='*60}")

        cfg = StatePredictorConfig(
            state_dim=obs_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            checkpoint_dir=task_ckpt_dir,
            gpu_id=args.gpu_id,
        )

        trainer = StatePredictorTrainer(cfg)
        result = trainer.train(task_data_dir)

        # 将 state_predictor.pt 重命名为 state_predictor_iter1.pt
        src = Path(task_ckpt_dir) / "state_predictor.pt"
        dst = Path(task_ckpt_dir) / "state_predictor_iter1.pt"
        if src.exists():
            src.rename(dst)
            print(f"[train_state_predictor] Renamed → {dst}")
            result["checkpoint_path"] = str(dst)

        results[task] = result
        print(
            f"[train_state_predictor] {task} done: "
            f"final_loss={result['final_loss']:.6f}, ckpt={result['checkpoint_path']}"
        )

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print("[train_state_predictor] ALL DONE")
    for task, res in results.items():
        print(
            f"  {task}: final_loss={res['final_loss']:.6f}, "
            f"ckpt={res['checkpoint_path']}"
        )

    # 为了向后兼容，将第一个任务的 ckpt 软链接到根目录 state_predictor_iter1.pt
    first_task = list(results.keys())[0]
    main_ckpt_src = Path(results[first_task]["checkpoint_path"])
    main_ckpt_link = out_root / "state_predictor_iter1.pt"
    if main_ckpt_src.exists() and not main_ckpt_link.exists():
        import shutil
        shutil.copy2(str(main_ckpt_src), str(main_ckpt_link))
        print(f"[train_state_predictor] Copied first task ckpt → {main_ckpt_link}")

    return results


if __name__ == "__main__":
    main()
