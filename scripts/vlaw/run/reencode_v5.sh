#!/bin/bash
# Re-encode v4 data with correct SVD VAE (BUG-C fix, ADR-040)
#
# BUG-C: pipeline.py 之前使用 sd-vae-ft-mse 的 AutoencoderKL 编码,
#         但 Ctrl-World 训练/推理使用 SVD 的 AutoencoderKLTemporalDecoder.
#         两个 VAE 权重不同, latent 分布存在偏差.
#
# 本脚本: 从 v4 HDF5 读取原始 RGB, 用 SVD VAE 重新编码 latent_concat, 写入 v5.
#
# Usage:
#   cd /home/wjz/rl-vla && CUDA_VISIBLE_DEVICES=2 bash scripts/vlaw/run/reencode_v5.sh
set -euo pipefail

ROOT="/home/wjz/rl-vla"
PYTHON="/home/wjz/miniconda3/envs/ctrl_world/bin/python"

SRC_DIR="${ROOT}/data/vlaw/encoded/train_v4/LiftPegUpright-v1"
DST_DIR="${ROOT}/data/vlaw/encoded/train_v5/LiftPegUpright-v1"
SVD_VAE_PATH="${ROOT}/checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"

mkdir -p "${DST_DIR}"

echo "============================================"
echo "  Re-encode v4 → v5 (SVD VAE, BUG-C fix)"
echo "  src: ${SRC_DIR}"
echo "  dst: ${DST_DIR}"
echo "  VAE: ${SVD_VAE_PATH}/vae"
echo "============================================"

${PYTHON} -c "
import sys, time
sys.path.insert(0, '${ROOT}')
from pathlib import Path
from rlft.vlaw.data.pipeline import PipelineConfig, VLAWDataPipeline

src_dir = Path('${SRC_DIR}')
dst_dir = Path('${DST_DIR}')
svd_vae = '${SVD_VAE_PATH}'

cfg = PipelineConfig(
    input_dir=str(src_dir),
    output_dir=str(dst_dir),
    vae_local_path=svd_vae,
    gpu_id=0,
    batch_size=16,
    verbose=True,
)
pipeline = VLAWDataPipeline(cfg)

h5_files = sorted(src_dir.glob('*.h5'))
print(f'Found {len(h5_files)} HDF5 files to re-encode')

t0 = time.perf_counter()
for h5_path in h5_files:
    out_path = dst_dir / h5_path.name
    print(f'\nEncoding {h5_path.name} → {out_path.name} ...')
    pipeline.encode_single_hdf5(h5_path, out_path)

elapsed = time.perf_counter() - t0
print(f'\n=== All done in {elapsed:.1f}s ===')
print(f'Output: ${DST_DIR}')
"

echo ""
echo "Re-encoding complete. Output: ${DST_DIR}"
