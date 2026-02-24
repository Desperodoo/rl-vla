"""
下载 Ctrl-World 预训练权重
使用方法: HF_ENDPOINT=https://hf-mirror.com python scripts/download_weights.py
"""
import os
import sys
import argparse

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from huggingface_hub import snapshot_download

SAVE_DIR = "checkpoints/vlaw/world_model/pretrained"
os.makedirs(SAVE_DIR, exist_ok=True)

# 排除非 PyTorch 格式（保留 .safetensors, .bin, .pt）
IGNORE = ['*.msgpack', 'tf_model.h5', 'tf_*', 'rust_model.ot', 'flax_*']

MODELS = [
    ("openai/clip-vit-base-patch32",                     "clip-vit-base-patch32"),
    ("stabilityai/stable-video-diffusion-img2vid",        "stable-video-diffusion-img2vid"),
    ("yjguo/Ctrl-World",                                  "Ctrl-World"),
]

parser = argparse.ArgumentParser()
parser.add_argument('--only', default=None, help='只下载某个模型: clip | svd | ctrl_world')
args = parser.parse_args()

for repo_id, local_name in MODELS:
    # 过滤
    if args.only == 'clip' and local_name != 'clip-vit-base-patch32':
        continue
    if args.only == 'svd' and local_name != 'stable-video-diffusion-img2vid':
        continue
    if args.only == 'ctrl_world' and local_name != 'Ctrl-World':
        continue

    local_dir = os.path.join(SAVE_DIR, local_name)
    print(f"\n{'='*60}")
    print(f"下载: {repo_id}")
    print(f"保存: {local_dir}")
    print(f"{'='*60}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=IGNORE,
        )
        print(f"[OK] {repo_id} 下载完成")
    except Exception as e:
        print(f"[ERROR] {repo_id} 下载失败: {e}", file=sys.stderr)
        sys.exit(1)

print("\n全部下载完成!")
