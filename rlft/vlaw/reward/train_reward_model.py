"""
P3.2 — VLAW 奖励模型 LoRA 微调脚本

使用 ManiSkill rollout 数据 (HDF5, 每任务一文件) + env_success 标签对
Qwen3-VL (Qwen2.5-VL) 进行 LoRA 微调，使模型能准确判断机器人操作是否成功。

HDF5 格式:
    <task>.h5
        traj_XXXX/
            rgb_base:    [T, H, W, C] uint8
            env_success: [T] bool/float   <- env_success[-1] = label

训练配置 (VLAW 论文 Appendix C):
    LoRA: r=16, alpha=32, dropout=0.1, target: q_proj, v_proj
    训练步数: 200 steps
    等效 batch: 128 (per_device_batch * gradient_accumulation_steps)
    lr: 2e-5, linear warmup 20 steps

用法 (单卡):
    CUDA_VISIBLE_DEVICES=6 python rlft/vlaw/reward/train_reward_model.py \\
        --data_dir data/vlaw/rollouts/iter1 \\
        --tasks LiftPegUpright-v1 PickCube-v1 StackCube-v1 \\
        --model_path checkpoints/vlaw/reward_model/qwen_vl \\
        --output_dir checkpoints/vlaw/reward_model/lora_iter1

用法 (多卡 - Accelerate):
    CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \\
        --num_processes 4 --multi_gpu \\
        rlft/vlaw/reward/train_reward_model.py \\
        --data_dir data/vlaw/rollouts/iter1 \\
        --tasks LiftPegUpright-v1 PickCube-v1 StackCube-v1 \\
        --model_path checkpoints/vlaw/reward_model/qwen_vl \\
        --output_dir checkpoints/vlaw/reward_model/lora_iter1 \\
        --multi_gpu

用法 (多卡 - torchrun fallback):
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \\
        rlft/vlaw/reward/train_reward_model.py \\
        --data_dir data/vlaw/rollouts/iter1 \\
        --multi_gpu

注意:
    - flash_attn 可用时自动启用 flash_attention_2 (可通过 --attn_implementation eager 关闭)
    - multi_gpu 模式下 per_device_batch_size 为每张卡的 batch size
    - checkpoint 仅在主进程 (rank 0) 保存
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# flash_attn 可用性检测
_FLASH_ATTN_AVAILABLE = False
try:
    import flash_attn  # noqa: F401
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass

# Typing compat
try:
    from typing import Optional  # noqa
except ImportError:
    pass


# ────────────────────────────────────────────────────────────────────────────
# 任务指令映射
# ────────────────────────────────────────────────────────────────────────────

TASK_INSTRUCTIONS: Dict[str, str] = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
    "PickCube-v1":        "Pick up the cube and place it on the target location.",
    "StackCube-v1":       "Stack the red cube on top of the green cube.",
    "LiftPeg":            "Lift the peg and insert it upright into the holder.",
    "PickCube":           "Pick up the cube and place it on the target location.",
    "StackCube":          "Stack the red cube on top of the green cube.",
    "default":            "Complete the manipulation task successfully.",
}


def get_instruction(task_id: str) -> str:
    if task_id in TASK_INSTRUCTIONS:
        return TASK_INSTRUCTIONS[task_id]
    for k, v in TASK_INSTRUCTIONS.items():
        if task_id.startswith(k) or k.startswith(task_id):
            return v
    return TASK_INSTRUCTIONS["default"]


# ────────────────────────────────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """LoRA 微调超参 (VLAW 论文 Appendix C)"""

    data_dir: str = "data/vlaw/rollouts/iter1"
    data_dirs: List[str] = field(default_factory=list)
    tasks: List[str] = field(
        default_factory=lambda: ["LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"]
    )
    output_dir: str = "checkpoints/vlaw/reward_model/lora_iter1"
    model_path: str = "checkpoints/vlaw/reward_model/qwen_vl"
    torch_dtype: str = "bfloat16"
    device: str = "cuda:0"

    # Attention implementation: "flash_attention_2" | "sdpa" | "eager" | "auto"
    # auto: flash_attn 可用 → flash_attention_2, 否则 → eager
    attn_implementation: str = "auto"

    # 多卡训练 (HuggingFace Accelerate / torch DDP)
    multi_gpu: bool = False

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # 训练
    train_steps: int = 200
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 128
    lr: float = 2e-5
    warmup_steps: int = 20
    weight_decay: float = 0.01

    # 兼容旧接口
    learning_rate: float = 0.0
    max_steps: int = 0

    # 采样
    num_frames: int = 16

    # 评估
    eval_steps: int = 50
    eval_ratio: float = 0.2

    seed: int = 42
    threshold: float = 0.8
    use_wandb: bool = False
    wandb_project: str = "vlaw-reward"
    wandb_run_name: str = "lora_iter1"

    # 视频模式: True = Qwen3-VL 原生 video 输入 (ADR-008/015: AUC +0.11)
    use_video_format: bool = True
    video_fps: float = 2.0

    def __post_init__(self) -> None:
        if self.learning_rate > 0:
            self.lr = self.learning_rate
        if self.max_steps > 0:
            self.train_steps = self.max_steps
        # 解析 attn_implementation: auto → 自动检测
        if self.attn_implementation == "auto":
            if _FLASH_ATTN_AVAILABLE:
                self.attn_implementation = "flash_attention_2"
                print("[VLAW] flash_attn 可用 → attn_implementation=flash_attention_2")
            else:
                self.attn_implementation = "eager"
                print("[VLAW] flash_attn 不可用 → attn_implementation=eager")


# ────────────────────────────────────────────────────────────────────────────
# 分布式工具函数
# ────────────────────────────────────────────────────────────────────────────

def _is_main_process(accelerator=None) -> bool:
    """判断当前是否为主进程 (rank 0)。"""
    if accelerator is not None:
        return accelerator.is_main_process
    # torch DDP fallback
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    return rank == 0


def _log(msg: str, accelerator=None) -> None:
    """只在主进程打印。"""
    if _is_main_process(accelerator):
        print(msg)


# ────────────────────────────────────────────────────────────────────────────
# 数据加载：per-task HDF5，episodes 为 key
# ────────────────────────────────────────────────────────────────────────────

def _find_task_h5_files(data_dir: Path, task_id: str) -> List[Path]:
    """在 data_dir/ 下寻找与 task_id 匹配的所有 HDF5 文件。"""
    result: List[Path] = []
    task_dir = data_dir / task_id
    if task_dir.is_dir():
        for ext in ["*.h5", "*.hdf5"]:
            result.extend(sorted(task_dir.glob(ext)))
        if result:
            return result
    # 模糊匹配
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        if task_id in d.name or d.name in task_id:
            for ext in ["*.h5", "*.hdf5"]:
                result.extend(sorted(d.glob(ext)))
            if result:
                return result
    return result


class _Episode:
    """轻量轨迹容器（延迟加载图像）。"""
    __slots__ = ("h5_path", "ep_key", "instruction", "label", "num_frames")

    def __init__(self, h5_path: Path, ep_key: str, instruction: str,
                 label: int, num_frames: int) -> None:
        self.h5_path = h5_path
        self.ep_key = ep_key
        self.instruction = instruction
        self.label = label
        self.num_frames = num_frames


def _load_frames(ep: "_Episode") -> List[Image.Image]:
    """从 HDF5 读取 RGB 帧并均匀采样。"""
    try:
        with h5py.File(str(ep.h5_path), "r") as f:
            grp = f[ep.ep_key]
            arr = None
            for key in ["rgb_base", "rgb", "image", "obs_image"]:
                if key in grp:
                    node = grp[key]
                    arr = node[()] if hasattr(node, "shape") else grp[key][list(grp[key].keys())[0]][()]
                    break
                # check nested obs/rgb
                if "obs" in grp and key in grp["obs"]:
                    arr = grp["obs"][key][()]
                    break

            if arr is None:
                return [Image.fromarray(np.zeros((128, 128, 3), np.uint8))] * ep.num_frames

        T = arr.shape[0]
        n = min(ep.num_frames, T)
        idxs = np.linspace(0, T - 1, n, dtype=int)
        frames: List[Image.Image] = []
        for i in idxs:
            f_arr = arr[i]
            if f_arr.dtype != np.uint8:
                f_arr = np.clip(f_arr * 255, 0, 255).astype(np.uint8)
            if f_arr.ndim == 3 and f_arr.shape[-1] == 4:
                f_arr = f_arr[..., :3]
            frames.append(Image.fromarray(f_arr))
        return frames
    except Exception as exc:
        print(f"[WARN] 读取帧失败 {ep.ep_key}: {exc}")
        return [Image.fromarray(np.zeros((128, 128, 3), np.uint8))] * ep.num_frames


class RewardDataset(Dataset):
    """VLAW 奖励模型训练集 / 验证集。"""

    def __init__(self, episodes: List["_Episode"]) -> None:
        self.episodes = episodes
        n_pos = sum(e.label for e in episodes)
        print(f"[VLAW] Dataset: {len(episodes)} 条  (+={n_pos}, -={len(episodes)-n_pos})")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Tuple[List[Image.Image], str, int]:
        ep = self.episodes[idx]
        return _load_frames(ep), ep.instruction, ep.label

    @staticmethod
    def collate_fn(batch):
        frames_list, instructions, labels = zip(*batch)
        return list(frames_list), list(instructions), list(labels)


def _load_episodes_from_dir(data_dir: Path, task_ids: List[str],
                           num_frames: int) -> List["_Episode"]:
    """从单个 data_dir 加载所有 episodes（支持多个 h5 文件）。"""
    eps: List[_Episode] = []
    for task_id in task_ids:
        h5_paths = _find_task_h5_files(data_dir, task_id)
        if not h5_paths:
            continue
        instruction = get_instruction(task_id)
        for h5_path in h5_paths:
            print(f"[VLAW] {task_id}: {h5_path}  inst='{instruction[:50]}'")
            try:
                with h5py.File(str(h5_path), "r") as f:
                    for ep_key in sorted(f.keys()):
                        grp = f[ep_key]
                        label = None
                        for sk in ["env_success", "success"]:
                            if sk in grp:
                                v = grp[sk][()]
                                if isinstance(v, np.ndarray):
                                    v = v.flat[-1]
                                label = int(bool(v))
                                break
                        if label is None and "success" in grp.attrs:
                            label = int(bool(grp.attrs["success"]))
                        if label is None:
                            continue
                        eps.append(_Episode(h5_path, ep_key, instruction, label, num_frames))
            except Exception as exc:
                print(f"[WARN] 打开 {h5_path} 失败: {exc}")
    return eps


def build_datasets(cfg: TrainConfig) -> Tuple[RewardDataset, RewardDataset]:
    """构建训练/验证数据集。支持多个 data_dir。"""
    # 收集所有要搜索的目录
    dirs: List[Path] = []
    if cfg.data_dirs:
        for d in cfg.data_dirs:
            p = Path(d)
            if p.exists():
                dirs.append(p)
            else:
                print(f"[WARN] data_dirs 中不存在: {d}")
    else:
        dirs.append(Path(cfg.data_dir))

    for d in dirs:
        assert d.exists(), f"data_dir 不存在: {d}"

    task_ids = cfg.tasks
    all_eps: List[_Episode] = []
    for data_dir in dirs:
        print(f"[VLAW] 从 {data_dir} 加载 ...")
        eps = _load_episodes_from_dir(data_dir, task_ids, cfg.num_frames)
        all_eps.extend(eps)
        print(f"[VLAW]   → {len(eps)} 条")

    n_pos = sum(e.label for e in all_eps)
    print(f"[VLAW] 总计 {len(all_eps)} 条轨迹  (+={n_pos}, -={len(all_eps)-n_pos})")
    assert all_eps, "没有有效样本"

    random.seed(cfg.seed)
    random.shuffle(all_eps)
    n_eval = max(1, int(len(all_eps) * cfg.eval_ratio))
    train_eps = all_eps[n_eval:]
    eval_eps  = all_eps[:n_eval]
    print(f"[VLAW] 分割: train={len(train_eps)}, eval={len(eval_eps)}")
    return RewardDataset(train_eps), RewardDataset(eval_eps)


# ────────────────────────────────────────────────────────────────────────────
# LoRA
# ────────────────────────────────────────────────────────────────────────────

def setup_lora(model, cfg: TrainConfig):
    """附加 LoRA adapter。"""
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


# ────────────────────────────────────────────────────────────────────────────
# 损失函数
# ────────────────────────────────────────────────────────────────────────────

def _make_messages(frames: List[Image.Image], instruction: str,
                   label: Optional[int] = None,
                   use_video_format: bool = True,
                   video_fps: float = 2.0) -> list:
    n = len(frames)
    text = (
        f"These {n} frames show a robot manipulation trajectory. "
        f"Task: '{instruction}'. "
        "Has the robot successfully completed the task? "
        "Answer only 'yes' or 'no'."
    )
    if use_video_format and n > 1:
        # Qwen3-VL 原生 video 输入 (带时序 PE, ADR-008/015: AUC +0.11)
        user_content = [
            {"type": "video", "video": frames, "fps": video_fps},
            {"type": "text", "text": text},
        ]
    else:
        # 多图模式 (向后兼容)
        user_content = [{"type": "image", "image": f} for f in frames]
        user_content.append({"type": "text", "text": text})
    msgs = [{"role": "user", "content": user_content}]
    if label is not None:
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": "yes" if label else "no"}]})
    return msgs


def compute_single_loss(model, processor, frames: List[Image.Image],
                        instruction: str, label: int,
                        device: str,
                        use_video_format: bool = True,
                        video_fps: float = 2.0) -> torch.Tensor:
    """单样本 teacher-forcing LM loss (支持 video/multi-image 模式)。"""
    msgs = _make_messages(frames, instruction, label,
                          use_video_format=use_video_format,
                          video_fps=video_fps)
    try:
        # Qwen3-VL 需要关闭 thinking 模式，否则模板会插入 <think> 标签
        try:
            full_text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                enable_thinking=False)
        except TypeError:
            full_text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False)
        # 使用 process_vision_info 正确处理 video/image 内容
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(msgs)
            inputs = processor(
                text=[full_text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
            ).to(device)
        except (ImportError, Exception):
            # 降级: 直接传入 PIL frames
            inputs = processor(text=[full_text], images=frames,
                               return_tensors="pt").to(device)
        out = model(**inputs, labels=inputs["input_ids"])
        return out.loss
    except Exception as exc:
        print(f"[WARN] loss 计算失败: {exc}")
        return torch.tensor(0.0, device=device, requires_grad=True)


# ────────────────────────────────────────────────────────────────────────────
# 评估
# ────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(model, processor, eval_ds: RewardDataset,
             cfg: TrainConfig, yes_id: int, no_id: int,
             device: str = "cuda:0") -> Dict:
    """在 eval_ds 上评估模型指标 (accuracy, FP rate 等)。"""
    tp = fp = tn = fn = 0
    p_yes_list: List[float] = []

    for i in range(len(eval_ds)):
        frames, instr, label = eval_ds[i]
        msgs = _make_messages(frames, instr, label=None,
                              use_video_format=cfg.use_video_format,
                              video_fps=cfg.video_fps)
        try:
            # Qwen3-VL 需要关闭 thinking 模式
            try:
                prompt = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                prompt = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(msgs)
                inp = processor(
                    text=[prompt],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    return_tensors="pt",
                ).to(device)
            except (ImportError, Exception):
                inp = processor(text=[prompt], images=frames,
                                return_tensors="pt").to(device)
            out = model(**inp)
            logits = out.logits[0, -1, :]
            p_yes = float(torch.softmax(
                torch.stack([logits[yes_id], logits[no_id]]), 0
            )[0].cpu())
        except Exception as exc:
            print(f"[WARN] eval 推理失败: {exc}")
            p_yes = 0.0

        p_yes_list.append(p_yes)
        pred = 1 if p_yes >= cfg.threshold else 0
        if label == 1 and pred == 1:   tp += 1
        elif label == 0 and pred == 1: fp += 1
        elif label == 0 and pred == 0: tn += 1
        else:                          fn += 1

    total = max(tp + fp + tn + fn, 1)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fp_rate":   fp  / max(fp + tn, 1),
        "accuracy":  (tp + tn) / total,
        "precision": tp  / max(tp + fp, 1),
        "recall":    tp  / max(tp + fn, 1),
        "mean_p_yes": float(np.mean(p_yes_list)) if p_yes_list else 0.0,
    }


# ────────────────────────────────────────────────────────────────────────────
# 主训练函数
# ────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    """LoRA 微调主流程 — 支持单卡和多卡 (Accelerate)。"""
    torch.manual_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)

    # ── Accelerate 初始化 (多卡) ──────────────────────────────────────────
    accelerator = None
    if cfg.multi_gpu:
        from accelerate import Accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision="bf16" if cfg.torch_dtype == "bfloat16" else "no",
        )
        device = accelerator.device
        _log(f"[VLAW] Accelerate: {accelerator.num_processes} GPUs, "
             f"device={device}", accelerator)
    else:
        device = cfg.device

    if _is_main_process(accelerator):
        output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.use_wandb and _is_main_process(accelerator):
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                   config=vars(cfg))

    # ── 数据 ──────────────────────────────────────────────────────────────
    _log("[VLAW] 构建数据集 ...", accelerator)
    train_ds, eval_ds = build_datasets(cfg)

    # 多卡：DistributedSampler
    sampler = None
    shuffle = True
    if cfg.multi_gpu and accelerator is not None and accelerator.num_processes > 1:
        sampler = DistributedSampler(
            train_ds,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=cfg.seed,
        )
        shuffle = False  # sampler handles shuffling

    train_loader = DataLoader(
        train_ds, batch_size=cfg.per_device_batch_size,
        shuffle=shuffle, sampler=sampler,
        collate_fn=RewardDataset.collate_fn, num_workers=0,
    )

    # ── 模型 ──────────────────────────────────────────────────────────────
    _log(f"[VLAW] 加载模型: {cfg.model_path} (attn={cfg.attn_implementation}) ...",
         accelerator)
    import transformers
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    tdtype = dtype_map.get(cfg.torch_dtype, torch.bfloat16)

    # 多卡模式下不使用 device_map（由 Accelerate 管理设备分配）
    _device_map = None if cfg.multi_gpu else cfg.device

    model = None
    for cls_name in ["Qwen3VLForConditionalGeneration",
                      "Qwen2_5_VLForConditionalGeneration",
                      "Qwen2VLForConditionalGeneration"]:
        try:
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(
                cfg.model_path,
                torch_dtype=tdtype,
                device_map=_device_map,
                attn_implementation=cfg.attn_implementation,
            )
            _log(f"[VLAW] 模型: {cls_name} (attn={cfg.attn_implementation})",
                 accelerator)
            break
        except Exception as e:
            _log(f"[VLAW] {cls_name} 跳过: {e}", accelerator)

    if model is None:
        raise RuntimeError("无法加载 VL 模型")

    processor = None
    for cls_name in ["Qwen3VLProcessor", "Qwen2_5_VLProcessor", "Qwen2VLProcessor", "AutoProcessor"]:
        try:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                from transformers import AutoProcessor as cls
            processor = cls.from_pretrained(cfg.model_path)
            _log(f"[VLAW] 处理器: {cls_name}", accelerator)
            break
        except Exception as e:
            _log(f"[VLAW] {cls_name} 跳过: {e}", accelerator)

    if processor is None:
        raise RuntimeError("无法加载 Processor")

    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[-1]
    no_id  = processor.tokenizer.encode("no",  add_special_tokens=False)[-1]
    _log(f"[VLAW] yes_token_id={yes_id}, no_token_id={no_id}", accelerator)

    # ── Gradient Checkpointing (节省显存) ────────────────────────────────
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        _log("[VLAW] gradient checkpointing 已启用", accelerator)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ── LoRA ──────────────────────────────────────────────────────────────
    _log("[VLAW] 应用 LoRA ...", accelerator)
    model = setup_lora(model, cfg)
    model.train()

    # ── 优化器 ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    from torch.optim.lr_scheduler import LinearLR
    warmup_sched = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=max(cfg.warmup_steps, 1),
    )

    # ── Accelerate prepare ────────────────────────────────────────────────
    if accelerator is not None:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )

    # ── 训练循环 ──────────────────────────────────────────────────────────
    eff_batch = cfg.per_device_batch_size * cfg.gradient_accumulation_steps
    if accelerator is not None:
        eff_batch *= accelerator.num_processes
    _log(f"[VLAW] 训练: steps={cfg.train_steps}, "
         f"per_device={cfg.per_device_batch_size}, "
         f"grad_accum={cfg.gradient_accumulation_steps}, "
         f"eff_batch={eff_batch}, "
         f"video_format={cfg.use_video_format}", accelerator)

    step = 0
    micro = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    while step < cfg.train_steps:
        if sampler is not None:
            sampler.set_epoch(step)  # for proper shuffling

        for batch in train_loader:
            if step >= cfg.train_steps:
                break

            frames_b, instrs, labels = batch
            model.train()

            batch_loss = torch.tensor(0.0, device=device)
            cnt = 0
            for frames, instr, lbl in zip(frames_b, instrs, labels):
                loss = compute_single_loss(
                    model, processor, frames, instr, int(lbl), device,
                    use_video_format=cfg.use_video_format,
                    video_fps=cfg.video_fps)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    batch_loss = batch_loss + loss
                    cnt += 1
            if cnt > 0:
                batch_loss = batch_loss / cnt

            if accelerator is not None:
                # Accelerate 管理梯度累积和 backward
                accelerator.backward(batch_loss / cfg.gradient_accumulation_steps)
            else:
                (batch_loss / cfg.gradient_accumulation_steps).backward()
            accum_loss += float(batch_loss.item())
            micro += 1

            if micro % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step < cfg.warmup_steps:
                    warmup_sched.step()
                optimizer.zero_grad()

                avg_loss = accum_loss / cfg.gradient_accumulation_steps
                lr_now   = optimizer.param_groups[0]["lr"]
                _log(f"[VLAW] step={step+1}/{cfg.train_steps} "
                     f"loss={avg_loss:.4f} lr={lr_now:.2e}", accelerator)
                if cfg.use_wandb and _is_main_process(accelerator):
                    import wandb
                    wandb.log({"train/loss": avg_loss, "train/step": step+1,
                               "train/lr": lr_now})
                accum_loss = 0.0
                step += 1

                # 定期评估 + checkpoint (仅主进程)
                if step % cfg.eval_steps == 0 and _is_main_process(accelerator):
                    # unwrap model for eval
                    eval_model = (accelerator.unwrap_model(model)
                                  if accelerator is not None else model)
                    eval_model.eval()
                    m = evaluate(eval_model, processor, eval_ds, cfg,
                                 yes_id, no_id, device=str(device))
                    print(f"[VLAW] EVAL step={step} "
                          f"acc={m['accuracy']:.3f} fp_rate={m['fp_rate']:.3f} "
                          f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
                    if cfg.use_wandb:
                        import wandb
                        wandb.log({f"eval/{k}": v for k, v in m.items()}
                                  | {"eval/step": step})
                    ckpt_dir = output_dir / f"step_{step}"
                    eval_model.save_pretrained(str(ckpt_dir))
                    processor.save_pretrained(str(ckpt_dir))
                    with open(ckpt_dir / "metrics.json", "w") as fj:
                        json.dump(m, fj, indent=2)
                    print(f"[VLAW] checkpoint → {ckpt_dir}")
                    model.train()

                # 同步所有进程
                if accelerator is not None:
                    accelerator.wait_for_everyone()

                if step >= cfg.train_steps:
                    break

    # ── 最终评估 + 保存 (仅主进程) ─────────────────────────────────────────
    if _is_main_process(accelerator):
        eval_model = (accelerator.unwrap_model(model)
                      if accelerator is not None else model)
        eval_model.eval()
        print("[VLAW] 最终评估 ...")
        final_m = evaluate(eval_model, processor, eval_ds, cfg,
                           yes_id, no_id, device=str(device))

        print("\n" + "=" * 60)
        print("  VLAW 奖励模型 LoRA 微调完成")
        print("=" * 60)
        print(f"  Accuracy : {final_m['accuracy']:.4f}")
        print(f"  FP Rate  : {final_m['fp_rate']:.4f}  (目标 < 0.20)")
        print(f"  TP/FP/TN/FN : {final_m['tp']}/{final_m['fp']}/{final_m['tn']}/{final_m['fn']}")
        print(f"  Attn Impl: {cfg.attn_implementation}")
        if cfg.multi_gpu and accelerator is not None:
            print(f"  GPUs     : {accelerator.num_processes}")
        print("=" * 60)

        # 保存到 output_dir 根目录（adapter + processor）
        eval_model.save_pretrained(str(output_dir))
        processor.save_pretrained(str(output_dir))

        # 同时保存到 final/ 子目录
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        eval_model.save_pretrained(str(final_dir))
        processor.save_pretrained(str(final_dir))

        with open(final_dir / "metrics.json", "w") as fj:
            json.dump(final_m, fj, indent=2)
        with open(output_dir / "train_config.json", "w") as fj:
            json.dump(vars(cfg), fj, indent=2, default=str)

        if cfg.use_wandb:
            import wandb
            wandb.finish()

        print(f"[VLAW] LoRA adapter 已保存: {output_dir}")

    # 确保所有进程同步完成
    if accelerator is not None:
        accelerator.wait_for_everyone()


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import tyro
        cfg = tyro.cli(TrainConfig)
        train(cfg)
        return
    except ImportError:
        pass

    import argparse
    p = argparse.ArgumentParser(description="VLAW VLM LoRA 微调")
    p.add_argument("--data_dir",   default="data/vlaw/rollouts/iter1")
    p.add_argument("--data_dirs",  nargs="*", default=[],
                   help="多个数据目录 (优先于 --data_dir)")
    p.add_argument("--tasks",      nargs="+",
                   default=["LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"])
    p.add_argument("--model_path", default="checkpoints/vlaw/reward_model/qwen_vl")
    p.add_argument("--output_dir", default="checkpoints/vlaw/reward_model/lora_iter1")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--lora_r",     type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--train_steps", type=int, default=200)
    p.add_argument("--per_device_batch_size",    type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--multi_gpu", action="store_true",
                   help="启用多卡训练 (Accelerate)")
    p.add_argument("--attn_implementation", default="auto",
                   choices=["auto", "flash_attention_2", "sdpa", "eager"],
                   help="Attention 实现方式")
    p.add_argument("--use_video_format", action="store_true", default=True,
                   help="使用 Qwen3-VL 原生 video 输入 (ADR-008: AUC +0.11)")
    p.add_argument("--no_video_format", dest="use_video_format", action="store_false",
                   help="使用多图模式")
    p.add_argument("--video_fps", type=float, default=2.0)
    p.add_argument("--eval_ratio", type=float, default=0.2,
                   help="内部评估集比例")
    a = p.parse_args()

    cfg = TrainConfig(
        data_dir=a.data_dir, data_dirs=a.data_dirs, tasks=a.tasks,
        model_path=a.model_path, output_dir=a.output_dir,
        num_frames=a.num_frames, lora_r=a.lora_r, lora_alpha=a.lora_alpha,
        train_steps=a.train_steps,
        per_device_batch_size=a.per_device_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        lr=a.lr, warmup_steps=a.warmup_steps,
        seed=a.seed, device=a.device, use_wandb=a.use_wandb,
        multi_gpu=a.multi_gpu,
        attn_implementation=a.attn_implementation,
        use_video_format=a.use_video_format,
        video_fps=a.video_fps,
        eval_ratio=a.eval_ratio,
    )
    train(cfg)


if __name__ == "__main__":
    main()
