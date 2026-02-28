"""
下载 Ctrl-World 预训练权重
使用方法: HF_ENDPOINT=https://hf-mirror.com python scripts/download_weights.py [--only clip|svd|ctrl_world]

预训练权重清单:
  - CLIP:       openai/clip-vit-base-patch32        → clip-vit-base-patch32/
  - SVD:        stabilityai/stable-video-diffusion-img2vid → stable-video-diffusion-img2vid/
  - Ctrl-World: yjguo/Ctrl-World (checkpoint-10000.pt) → Ctrl-World/checkpoint-10000.pt
"""
import os
import sys
import shutil
import argparse

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from huggingface_hub import snapshot_download, hf_hub_download

SAVE_DIR = "checkpoints/vlaw/world_model/pretrained"
os.makedirs(SAVE_DIR, exist_ok=True)

# 排除非 PyTorch 格式（保留 .safetensors, .bin, .pt）
IGNORE = ['*.msgpack', 'tf_model.h5', 'tf_*', 'rust_model.ot', 'flax_*', '*.gif', '*.png', '*.jpg']

parser = argparse.ArgumentParser()
parser.add_argument('--only', default=None, help='只下载某个模型: clip | svd | ctrl_world')
args = parser.parse_args()


def download_snapshot(repo_id: str, local_name: str) -> None:
    local_dir = os.path.join(SAVE_DIR, local_name)
    print(f"\n{'='*60}\n下载: {repo_id}\n保存: {local_dir}\n{'='*60}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, ignore_patterns=IGNORE)
    # 清理 snapshot_download 产生的 blob 缓存（会占用额外空间）
    cache_dir = os.path.join(local_dir, '.cache')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"[INFO] 已清理 .cache 冗余目录")
    print(f"[OK] {repo_id} 下载完成")


def download_file(repo_id: str, filename: str, local_name: str) -> None:
    local_dir = os.path.join(SAVE_DIR, local_name)
    os.makedirs(local_dir, exist_ok=True)
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest):
        size_gb = os.path.getsize(dest) / 1e9
        print(f"[SKIP] {dest} 已存在 ({size_gb:.2f} GB)")
        return
    print(f"\n{'='*60}\n下载: {repo_id}/{filename}\n保存: {dest}\n{'='*60}")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    size_gb = os.path.getsize(path) / 1e9
    print(f"[OK] {repo_id}/{filename} 下载完成 ({size_gb:.2f} GB)")


try:
    # 1. CLIP
    if args.only in (None, 'clip'):
        clip_dir = os.path.join(SAVE_DIR, 'clip-vit-base-patch32')
        if os.path.exists(os.path.join(clip_dir, 'pytorch_model.bin')):
            print("[SKIP] clip-vit-base-patch32 已存在")
        else:
            download_snapshot("openai/clip-vit-base-patch32", "clip-vit-base-patch32")

    # 2. SVD
    if args.only in (None, 'svd'):
        svd_dir = os.path.join(SAVE_DIR, 'stable-video-diffusion-img2vid')
        if os.path.exists(os.path.join(svd_dir, 'model_index.json')):
            print("[SKIP] stable-video-diffusion-img2vid 已存在")
        else:
            download_snapshot("stabilityai/stable-video-diffusion-img2vid", "stable-video-diffusion-img2vid")

    # 3. Ctrl-World — 仅单文件 checkpoint-10000.pt (~8GB)
    if args.only in (None, 'ctrl_world'):
        download_file("yjguo/Ctrl-World", "checkpoint-10000.pt", "Ctrl-World")

except Exception as e:
    print(f"[ERROR] 下载失败: {e}", file=sys.stderr)
    sys.exit(1)

print("\n全部下载完成!")
