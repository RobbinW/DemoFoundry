#!/usr/bin/env bash
set -euo pipefail

episode=$(cd "${1:-runs/marvin}" && pwd)
cd "$(dirname "$0")/../.."
clip=$(mktemp -d)
trap 'rm -rf "$clip"' EXIT

ffmpeg -nostdin -loglevel error -i "$episode/proxy/left.mp4" -t 6 \
  -c:v libx264 -crf 16 -threads 2 "$clip/left.mp4"
bash inference/h3/download_weights.sh
torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-4}" -m inference.h3.render \
  --model weights/MiniMax-H3 --adapter weights/DemoFoundry-H3.safetensors \
  --reference "$episode/references/left.png" --proxy "$clip/left.mp4" \
  --size 640 352 --output "$episode/rendered/left-6s.mp4"
