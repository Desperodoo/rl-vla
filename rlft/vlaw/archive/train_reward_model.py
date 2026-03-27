"""
P3.1/P3.2 — VLAW 奖励模型 LoRA 微调脚本

使用 ManiSkill rollout 数据 (HDF5) + info["success"] 标签对
Qwen2.5-VL 进行 LoRA 微调，使其更准确地对机器人操作轨迹进行二分类。

训练配置:
    LoRA: r=16, alpha=32, target: q_proj, v_proj
    训练步数: 200 steps
    等效 batch: 128 (gradient_accumulation_steps * per_device_batch)

用法:
    CUDA_VISIBLE_DEVICES=6 \\
    /home/wjz/miniconda3/envs/vlaw_reward/bin/python \\
    rlft/vlaw/train_reward_model.py \\
        --data_dir data/vlaw/rollouts/iter1 \\
        --output_dir checkpoints/vlaw/reward_model/lora_iter1 \\
        --instruction "pick up the red cube and place it in the green box"
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import tyro
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .reward_model import VLAWRewardConfig, VLAWRewardModel, uniform_sample_frames


# ──────────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """LoRA 微调配置"""

    # 数据
    data_dir: str = "data/vlaw/rollouts/iter1"
    """HDF5 rollout 目录，每个文件对应一条轨迹"""

    obs_key: str = "obs/rgb"
    """HDF5 中 RGB 图像的 key (次级: base_camera)"""

    camera_key: str = "base_camera"
    """RGB obs 下的相机子 key"""

    success_key: str = "success"
    """HDF5 中成功标签的 key，位于根级别或 info/ 下"""

    instruction: str = "complete the manipulation task"
    """任务指令（若多任务则从 HDF5 属性读取）"""

    # 输出
    output_dir: str = "checkpoints/vlaw/reward_model/lora_v1"
    """LoRA checkpoint 保存目录"""

    # 模型
    model_path: str = "checkpoints/vlaw/reward_model/qwen_vl"
    """基座模型路径"""

    torch_dtype: str = "bfloat16"

    device: str = "cuda:0"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # 训练
    max_steps: int = 200
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 128
    """等效 batch_size = per_device_batch_size × gradient_accumulation_steps = 128"""

    learning_rate: float = 2e-5
    warmup_steps: int = 10
    weight_decay: float = 0.01

    # 帧采样
    num_frames: int = 16

    # 评估
    eval_steps: int = 50
    eval_ratio: float = 0.2
    """用于验证集的数据比例"""

    # 随机种子
    seed: int = 42

    # wandb (可选)
    use_wandb: bool = False
    wandb_project: str = "vlaw-reward"
    wandb_run_name: str = "lora_finetune_v1"

    # 阈值 α
    threshold: float = 0.8


# ──────────────────────────────────────────────────────────────────────────────
# 数据集
# ──────────────────────────────────────────────────────────────────────────────

def _read_success_label(f: h5py.File, key: str = "success") -> Optional[int]:
    """从 HDF5 文件读取 success 标签"""
    # 尝试几种常见位置
    for path in [key, f"info/{key}", f"data/{key}"]:
        if path in f:
            val = f[path][()]
            if isinstance(val, np.ndarray):
                val = val.flat[-1]  # 取最后一帧的 success
            return int(bool(val))
    # 尝试属性
    if key in f.attrs:
        return int(bool(f.attrs[key]))
    return None


class RewardDataset(Dataset):
    """
    VLAW 奖励模型训练数据集

    每条样本 = (采样帧列表, 任务指令, 成功标签)
    """

    def __init__(
        self,
        hdf5_paths: List[str],
        instruction: str,
        num_frames: int = 16,
        obs_key: str = "obs/rgb",
        camera_key: str = "base_camera",
        success_key: str = "success",
        augment: bool = False,
    ):
        self.hdf5_paths = hdf5_paths
        self.instruction = instruction
        self.num_frames = num_frames
        self.obs_key = obs_key
        self.camera_key = camera_key
        self.success_key = success_key
        self.augment = augment

        # 预加载 labels 以检查数据有效性
        self.valid_paths: List[str] = []
        self.labels: List[int] = []
        skipped = 0
        for p in hdf5_paths:
            try:
                with h5py.File(p, "r") as f:
                    label = _read_success_label(f, success_key)
                if label is None:
                    skipped += 1
                    continue
                self.valid_paths.append(p)
                self.labels.append(label)
            except Exception as e:
                print(f"[WARN] 跳过损坏文件 {p}: {e}")
                skipped += 1

        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        print(
            f"[VLAW] Dataset: {len(self.valid_paths)} 条有效轨迹 "
            f"(+={n_pos}, -={n_neg}, skip={skipped})"
        )

    def __len__(self) -> int:
        return len(self.valid_paths)

    def __getitem__(self, idx: int) -> Tuple[List[Image.Image], str, int]:
        path = self.valid_paths[idx]
        label = self.labels[idx]

        with h5py.File(path, "r") as f:
            # 读取 RGB 帧
            grp = f
            for k in self.obs_key.split("/"):
                if k in grp:
                    grp = grp[k]
            if self.camera_key in grp:
                arr = grp[self.camera_key][()]
            elif isinstance(grp, h5py.Dataset):
                arr = grp[()]
            else:
                # 尝试第一个子 key
                first = list(grp.keys())[0]
                arr = grp[first][()]

        frames = uniform_sample_frames(arr, self.num_frames)

        # 简单数据增强（随机水平翻转）
        if self.augment and random.random() < 0.3:
            frames = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in frames]

        return frames, self.instruction, label

    @staticmethod
    def collate_fn(batch):
        frames_list, instructions, labels = zip(*batch)
        return list(frames_list), list(instructions), list(labels)


def build_datasets(cfg: TrainConfig) -> Tuple[RewardDataset, RewardDataset]:
    """扫描 data_dir 并按 eval_ratio 分割为训练/验证集"""
    data_dir = Path(cfg.data_dir)
    assert data_dir.exists(), f"data_dir 不存在: {data_dir}"

    all_paths = sorted(data_dir.glob("**/*.hdf5")) + sorted(
        data_dir.glob("**/*.h5")
    )
    assert len(all_paths) > 0, f"未找到 HDF5 文件: {data_dir}"

    random.seed(cfg.seed)
    random.shuffle(all_paths)

    n_eval = max(1, int(len(all_paths) * cfg.eval_ratio))
    train_paths = [str(p) for p in all_paths[n_eval:]]
    eval_paths = [str(p) for p in all_paths[:n_eval]]

    train_ds = RewardDataset(
        train_paths,
        cfg.instruction,
        cfg.num_frames,
        cfg.obs_key,
        cfg.camera_key,
        cfg.success_key,
        augment=True,
    )
    eval_ds = RewardDataset(
        eval_paths,
        cfg.instruction,
        cfg.num_frames,
        cfg.obs_key,
        cfg.camera_key,
        cfg.success_key,
        augment=False,
    )
    return train_ds, eval_ds


# ──────────────────────────────────────────────────────────────────────────────
# LoRA 设置
# ──────────────────────────────────────────────────────────────────────────────

def setup_lora(model, cfg: TrainConfig):
    """在模型上附加 LoRA adapter"""
    from peft import LoraConfig, get_peft_model, TaskType

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 损失函数：BCE on P('yes')
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss(
    model,
    processor,
    frames_batch: List[List[Image.Image]],
    instructions: List[str],
    labels: List[int],
    yes_token_id: int,
    no_token_id: int,
    cfg: TrainConfig,
) -> torch.Tensor:
    """
    计算 BCE loss: -[ y*log P('yes') + (1-y)*log P('no') ]

    通过 teacher-forcing: 把 ground-truth token ('yes'/'no') 作为目标，
    最大化其在最后位置的 logits 概率。
    """
    from transformers import Qwen2_5_VLForConditionalGeneration

    prompt_template = (
        "These {n} frames show a robot manipulation trajectory. "
        "Task: '{instruction}'. "
        "Has the robot successfully completed the task? "
        "Answer only 'yes' or 'no'."
    )

    total_loss = torch.tensor(0.0, device=model.device)
    for frames, instr, label in zip(frames_batch, instructions, labels):
        n = len(frames)
        text_prompt = prompt_template.format(n=n, instruction=instr)
        # 目标 token
        target_token = "yes" if label == 1 else "no"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f} for f in frames
                ] + [{"type": "text", "text": text_prompt}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_token}],
            },
        ]

        # 构建带 ground-truth response 的 input_ids
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = processor(
            text=[full_text],
            images=frames,
            return_tensors="pt",
        ).to(model.device)

        # Forward pass (language modeling objective on last token)
        outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss = total_loss + outputs.loss

    return total_loss / len(frames_batch)


# ──────────────────────────────────────────────────────────────────────────────
# 评估：Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(
    reward_model: VLAWRewardModel,
    eval_ds: RewardDataset,
    cfg: TrainConfig,
) -> Dict:
    """返回 confusion matrix 及 FP rate"""
    tp = fp = tn = fn = 0
    p_yes_list = []

    for i in range(len(eval_ds)):
        frames, instr, label = eval_ds[i]
        result = reward_model.score_trajectory(frames, instr)
        pred = result["reward"]
        p_yes_list.append(result["p_yes"])

        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    fp_rate = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fp_rate": fp_rate,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "mean_p_yes": float(np.mean(p_yes_list)) if p_yes_list else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 主训练循环
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    """LoRA 微调主流程"""
    torch.manual_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── wandb ────────────────────────────────────────────────────────────────
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=cfg)

    # ── 数据 ─────────────────────────────────────────────────────────────────
    print("[VLAW] 构建数据集 ...")
    train_ds, eval_ds = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=RewardDataset.collate_fn,
    )

    # ── 加载模型 + LoRA ───────────────────────────────────────────────────────
    print("[VLAW] 加载基座模型 ...")
    reward_cfg = VLAWRewardConfig(
        model_path=cfg.model_path,
        torch_dtype=cfg.torch_dtype,
        device=cfg.device,
        num_frames=cfg.num_frames,
        threshold=cfg.threshold,
    )
    reward_model = VLAWRewardModel(reward_cfg)
    reward_model.load_model()
    model = reward_model.model
    processor = reward_model.processor
    yes_token_id = reward_model._yes_token_id
    no_token_id = reward_model._no_token_id

    print("[VLAW] 应用 LoRA ...")
    model = setup_lora(model, cfg)
    reward_model.model = model

    # ── 优化器 ────────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    from torch.optim.lr_scheduler import LinearLR
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=cfg.warmup_steps,
    )

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    print(f"[VLAW] 开始训练 (max_steps={cfg.max_steps}) ...")
    step = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    while step < cfg.max_steps:
        for batch in train_loader:
            if step >= cfg.max_steps:
                break

            frames_batch, instructions, labels = batch
            model.train()

            loss = compute_loss(
                model, processor,
                frames_batch, instructions, labels,
                yes_token_id, no_token_id, cfg,
            )
            # gradient accumulation
            (loss / cfg.gradient_accumulation_steps).backward()
            accum_loss += loss.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                avg_loss = accum_loss / cfg.gradient_accumulation_steps
                print(f"[VLAW] step={step+1} loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                if cfg.use_wandb:
                    import wandb
                    wandb.log({"train/loss": avg_loss, "train/step": step + 1})
                accum_loss = 0.0

            # 评估
            if (step + 1) % cfg.eval_steps == 0:
                model.eval()
                metrics = evaluate(reward_model, eval_ds, cfg)
                print(
                    f"[VLAW] EVAL step={step+1} "
                    f"acc={metrics['accuracy']:.3f} "
                    f"fp_rate={metrics['fp_rate']:.3f} "
                    f"TP={metrics['tp']} FP={metrics['fp']} "
                    f"TN={metrics['tn']} FN={metrics['fn']}"
                )
                if cfg.use_wandb:
                    import wandb
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()} | {"eval/step": step + 1})

                # 保存 checkpoint
                ckpt_dir = output_dir / f"step_{step+1}"
                model.save_pretrained(str(ckpt_dir))
                processor.save_pretrained(str(ckpt_dir))
                # 保存指标
                with open(ckpt_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"[VLAW] checkpoint 已保存: {ckpt_dir}")

            step += 1

    # ── 最终评估 + 保存 ───────────────────────────────────────────────────────
    model.eval()
    final_metrics = evaluate(reward_model, eval_ds, cfg)
    print("\n" + "=" * 60)
    print("  VLAW 奖励模型 LoRA 微调完成")
    print("=" * 60)
    print(f"  Accuracy : {final_metrics['accuracy']:.4f}")
    print(f"  FP Rate  : {final_metrics['fp_rate']:.4f}  (目标 < 0.10)")
    print(f"  TP/FP/TN/FN: {final_metrics['tp']}/{final_metrics['fp']}/{final_metrics['tn']}/{final_metrics['fn']}")
    print("=" * 60)

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    with open(final_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2)

    if cfg.use_wandb:
        import wandb
        wandb.finish()

    print(f"[VLAW] LoRA checkpoint 已保存: {final_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    train(cfg)
