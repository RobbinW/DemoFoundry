#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

hf download RobbinWang123/demofoundry-h3 DemoFoundry-H3.safetensors LICENSE.model NOTICE \
  --revision e7d135ec12c697d0c7da6476aa160b09a367a7e4 --local-dir weights
(
  cd weights
  echo '8838c120bcc728e43a3478b13513a56a431f0336e12d6a969a843b89dc8e8fe7  DemoFoundry-H3.safetensors' | sha256sum -c -
)
hf download MiniMaxAI/MiniMax-H3 --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include 'modular_model_index.json' 'transformer_ref/*' 'text_encoder/*' 'vae/*' \
            'audio_vae/*' 'scheduler/*' 'audio_scheduler/*' 'processor/*' 'tokenizer/*' \
  --local-dir weights/MiniMax-H3
