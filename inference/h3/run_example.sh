#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

bash inference/h3/download_weights.sh
(
  cd weights
  example_checksum='7a5b2c98a5037e4db0b3a5d3fe975b0a3b6ddbe596f7c986680b457c53692204  h3-example.tar.gz'
  if ! echo "$example_checksum" | sha256sum --check --status 2>/dev/null; then
    gh release download h3-v0.1 -R RobbinW/DemoFoundry -p h3-example.tar.gz --clobber
  fi
  echo "$example_checksum" | sha256sum -c -
)
tar -xzf weights/h3-example.tar.gz -C weights
exec torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-4}" -m inference.h3.render \
  --model weights/MiniMax-H3 --adapter weights/DemoFoundry-H3.safetensors \
  --reference weights/h3-example/reference.png --proxy weights/h3-example/proxy.mkv \
  --size 640 352 --output runs/example.mp4
