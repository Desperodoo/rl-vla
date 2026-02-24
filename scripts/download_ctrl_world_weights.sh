#!/bin/bash
# 下载 Ctrl-World 预训练权重
# 使用 hf-mirror.com 镜像加速

set -e

export HF_ENDPOINT=https://hf-mirror.com
SAVE_DIR="/home/wjz/rl-vla/checkpoints/vlaw/world_model/pretrained"
mkdir -p "$SAVE_DIR"

echo "=== Ctrl-World 预训练权重下载 ==="
echo "保存目录: $SAVE_DIR"
echo "磁盘空间: $(df -h / | tail -1 | awk '{print $4}') 可用"
echo ""

download_model() {
    local repo_id="$1"
    local local_dir="$2"
    echo "[$(date '+%H:%M:%S')] 开始下载: $repo_id → $local_dir"
    conda run -n ctrl_world python -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download(
    repo_id='$repo_id',
    local_dir='$local_dir',
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*', 'rust_model*'],
)
print('下载完成: $repo_id')
"
    echo "[$(date '+%H:%M:%S')] 完成: $repo_id"
}

# 1. CLIP (~600MB)
if [ ! -f "$SAVE_DIR/clip-vit-base-patch32/config.json" ]; then
    download_model "openai/clip-vit-base-patch32" "$SAVE_DIR/clip-vit-base-patch32"
else
    echo "[SKIP] clip-vit-base-patch32 已存在"
fi

# 2. SVD (~8GB)
if [ ! -f "$SAVE_DIR/stable-video-diffusion-img2vid/model_index.json" ]; then
    download_model "stabilityai/stable-video-diffusion-img2vid" "$SAVE_DIR/stable-video-diffusion-img2vid"
else
    echo "[SKIP] stable-video-diffusion-img2vid 已存在"
fi

# 3. Ctrl-World (~8GB)
if [ ! -f "$SAVE_DIR/Ctrl-World/model_index.json" ] && [ ! -f "$SAVE_DIR/Ctrl-World/README.md" ]; then
    download_model "yjguo/Ctrl-World" "$SAVE_DIR/Ctrl-World"
else
    echo "[SKIP] Ctrl-World 已存在"
fi

echo ""
echo "=== 所有权重下载完成 ==="
echo "目录大小: $(du -sh $SAVE_DIR 2>/dev/null | cut -f1)"
