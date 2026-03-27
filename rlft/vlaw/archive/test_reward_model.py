# ⚠️ 已迁移 → rlft/tests/vlaw/test_reward_model_legacy.py
# 此文件保留仅供向后兼容，请使用新位置的测试文件

"""
P0.3 验证脚本: 在 GPU 上加载 Qwen2.5-VL 并测试零样本推理
用法:
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python rlft/vlaw/test_reward_model.py \
        --model_path checkpoints/vlaw/reward_model/qwen_vl
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0.3 VLM reward model inference test")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/vlaw/reward_model/qwen_vl",
        help="Local model path or HuggingFace model ID",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use_flash_attn", action="store_true", default=False)
    return parser.parse_args()


def make_dummy_frame(h: int = 224, w: int = 224) -> Image.Image:
    """生成随机 RGB 帧用于测试"""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def check_vram(device: str) -> tuple[float, float]:
    """返回 (已用 GB, 总 GB)"""
    idx = int(device.split(":")[-1]) if ":" in device else 0
    used = torch.cuda.memory_allocated(idx) / 1e9
    total = torch.cuda.get_device_properties(idx).total_memory / 1e9
    return used, total


def test_inference(model_path: str, device: str, dtype_str: str, use_flash_attn: bool) -> None:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch_dtype = getattr(torch, dtype_str)
    attn_impl = "flash_attention_2" if use_flash_attn else "eager"

    print(f"\n{'='*50}")
    print(f"  模型路径: {model_path}")
    print(f"  设备:     {device}")
    print(f"  数据类型: {dtype_str}")
    print(f"  Attention: {attn_impl}")
    print(f"{'='*50}\n")

    # ── 加载模型 ───────────────────────────────────────────────
    print("[1/3] 加载 processor ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print("[2/3] 加载模型权重 ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0

    used_gb, total_gb = check_vram(device)
    print(f"    加载耗时: {load_time:.1f}s  |  显存: {used_gb:.1f}/{total_gb:.0f} GB")

    # ── 构建二分类 prompt ──────────────────────────────────────
    print("[3/3] 零样本推理测试 ...")
    instruction = "pick up the red cube and place it in the bin"
    num_frames = 4  # 测试使用少量帧

    frames = [make_dummy_frame() for _ in range(num_frames)]

    # Qwen2.5-VL 多图格式
    image_content = [{"type": "image", "image": frame} for frame in frames]
    messages = [
        {
            "role": "user",
            "content": image_content + [
                {
                    "type": "text",
                    "text": (
                        f"These {num_frames} frames show a robot manipulation trajectory. "
                        f"Task: '{instruction}'. "
                        "Has the robot successfully completed the task? "
                        "Answer only 'yes' or 'no'."
                    ),
                }
            ],
        }
    ]

    # 构建输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=frames,
        return_tensors="pt",
    ).to(device)

    # 提取 'yes' / 'no' token logits
    yes_token_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_token_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]

    t1 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # 最后一个 token 的 logits

    infer_time = time.time() - t1

    # 计算 P('yes') 概率
    yes_logit = logits[0, yes_token_id].float()
    no_logit = logits[0, no_token_id].float()
    probs = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)
    p_yes = probs[0].item()

    print(f"\n  推理耗时:  {infer_time*1000:.1f} ms")
    print(f"  P('yes'):   {p_yes:.4f}")
    print(f"  P('no'):    {1-p_yes:.4f}")
    print(f"  判定 (α=0.8): {'SUCCESS' if p_yes > 0.8 else 'FAILURE'}")
    print(f"\n  显存峰值:  {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ── 结论 ───────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  P0.3 验证通过: 模型加载和推理均正常")
    print("="*50 + "\n")


if __name__ == "__main__":
    args = parse_args()
    test_inference(
        model_path=args.model_path,
        device=args.device,
        dtype_str=args.dtype,
        use_flash_attn=args.use_flash_attn,
    )
