#!/usr/bin/env bash
set -euo pipefail

# One-click benchmark launcher for dual cameras.
# It sweeps FPS levels and prints a recommended stable capture setting.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WRIST_SERIAL="${WRIST_SERIAL:-218622279840}"
THIRD_SERIAL="${THIRD_SERIAL:-037522250003}"
FPS_LIST="${FPS_LIST:-15,30,45,60}"
SYNC_SLOP="${SYNC_SLOP:-0.05}"
RECORD_FREQ="${RECORD_FREQ:-120}"
DURATION="${DURATION:-20}"
SETTLE="${SETTLE:-6}"

cd "${ROOT_DIR}"

echo "[INFO] Dual camera benchmark"
echo "[INFO] wrist_serial=${WRIST_SERIAL}"
echo "[INFO] third_serial=${THIRD_SERIAL}"
echo "[INFO] fps_list=${FPS_LIST}"
echo "[INFO] sync_slop=${SYNC_SLOP}"
echo "[INFO] record_freq(stress)=${RECORD_FREQ}"
echo "[INFO] duration=${DURATION}, settle=${SETTLE}"

python3 scripts/benchmark_dual_camera_fps.py \
  --wrist-serial "${WRIST_SERIAL}" \
  --third-serial "${THIRD_SERIAL}" \
  --fps-list "${FPS_LIST}" \
  --sync-slop "${SYNC_SLOP}" \
  --record-freq "${RECORD_FREQ}" \
  --duration "${DURATION}" \
  --settle "${SETTLE}"
