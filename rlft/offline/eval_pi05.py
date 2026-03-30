from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from peft import PeftConfig


@dataclass
class Args:
    dataset_root: str
    policy_pretrained_path: str
    peft_adapter_path: Optional[str] = None
    output_path: Optional[str] = None
    repo_id: str = "carm/pi05_local"
    tokenizer_path_override: Optional[str] = "/home/wjz/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
    max_episodes: Optional[int] = None
    max_frames: Optional[int] = None
    device: str = "cuda"


def _to_float_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def _load_policy_and_processors(args: Args, dataset: LeRobotDataset):
    policy_path = Path(args.peft_adapter_path or args.policy_pretrained_path).expanduser().resolve()
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.device = args.device
    cfg.pretrained_path = policy_path
    cfg.use_peft = (args.peft_adapter_path is not None) or (policy_path / "adapter_config.json").exists()
    if cfg.use_peft and args.peft_adapter_path is None:
        try:
            _ = PeftConfig.from_pretrained(policy_path)
        except Exception:
            cfg.use_peft = False

    policy = make_policy(cfg=cfg, ds_meta=dataset.meta, rename_map={})
    preprocessor_overrides = {
        "rename_observations_processor": {"rename_map": {}},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
    }
    if args.tokenizer_path_override:
        preprocessor_overrides["tokenizer_processor"] = {
            "tokenizer_name": args.tokenizer_path_override,
        }
    postprocessor_overrides = {
        "unnormalizer_processor": {
            "stats": dataset.meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        }
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(policy_path),
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
    return policy, preprocessor, postprocessor


def main() -> None:
    args = tyro.cli(Args)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset = LeRobotDataset(repo_id=args.repo_id, root=dataset_root)
    policy, preprocessor, postprocessor = _load_policy_and_processors(args, dataset)
    policy.eval()

    max_frames = args.max_frames or len(dataset)
    episode_metrics: dict[int, list[float]] = {}
    sq_errors: list[np.ndarray] = []
    abs_errors: list[np.ndarray] = []

    with torch.no_grad():
        iterator = tqdm(range(min(len(dataset), max_frames)), desc="Evaluating pi05")
        for idx in iterator:
            sample = dataset[idx]
            episode_index = int(sample.get("episode_index", -1))
            processed = preprocessor(sample)
            pred = policy.select_action(processed)
            pred = postprocessor(pred)
            pred_arr = _to_float_array(pred).reshape(-1)
            gt_arr = _to_float_array(sample["action"]).reshape(-1)
            dim = min(pred_arr.shape[0], gt_arr.shape[0])
            pred_arr = pred_arr[:dim]
            gt_arr = gt_arr[:dim]
            diff = pred_arr - gt_arr
            sq_errors.append(diff ** 2)
            abs_errors.append(np.abs(diff))
            episode_metrics.setdefault(episode_index, []).append(float(np.mean(np.abs(diff))))

    mse = np.mean(np.stack(sq_errors), axis=0)
    mae = np.mean(np.stack(abs_errors), axis=0)
    results = {
        "dataset_root": str(dataset_root),
        "policy_pretrained_path": str(Path(args.policy_pretrained_path).expanduser().resolve()),
        "peft_adapter_path": str(Path(args.peft_adapter_path).expanduser().resolve()) if args.peft_adapter_path else None,
        "num_frames": len(sq_errors),
        "num_episodes": len(dataset.meta.episodes),
        "mean_action_mse": float(np.mean(mse)),
        "mean_action_mae": float(np.mean(mae)),
        "per_dim_mse": mse.tolist(),
        "per_dim_mae": mae.tolist(),
        "per_episode_mean_mae": {str(k): float(np.mean(v)) for k, v in episode_metrics.items() if k >= 0},
    }

    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else dataset_root / "pi05_eval_results.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
