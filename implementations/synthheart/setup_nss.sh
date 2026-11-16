#!/usr/bin/env bash
# One-time setup: build a separate environment for NVIDIA NeMo Safe Synthesizer (NSS) in .venv-nss.
# It is separate from the SynthHeart .venv because NSS pins its own torch / vLLM builds.
#
# Run on the AI Factory, from implementations/synthheart:
#     bash setup_nss.sh
#
# CUDA=130 bash setup_nss.sh     # CUDA 13 wheels (needs NVIDIA driver 580.65.06+). Default: CUDA 12.9
# NSS_VERSION=0.1.17             # the release this comparison was written against
set -euo pipefail
cd "$(dirname "$0")"
VENV="$PWD/.venv-nss"
NSS_VERSION="${NSS_VERSION:-0.1.17}"

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== Environment: $VENV =="
[ -x "$VENV/bin/python" ] || uv venv --seed --python 3.12 "$VENV"
installer="$(mktemp)"
curl -fsSL -o "$installer" "https://github.com/NVIDIA-NeMo/Safe-Synthesizer/releases/download/v${NSS_VERSION}/install_nss.sh"
# NVIDIA's installer: uv pip install "nemo-safe-synthesizer[engine,cu129]==<version>" plus the CUDA wheel pages.
# ISOLATED keeps it from reading this repo's pyproject.toml / uv settings.
NSS_INSTALLER_ISOLATED=1 UV_PROJECT_ENVIRONMENT="$VENV" CUDA="${CUDA:-129}" bash "$installer"
rm -f "$installer"

echo "== Check: versions, GPU visible, base model downloaded =="
"$VENV/bin/python" - <<'EOF'
import importlib.metadata as md

import torch
from huggingface_hub import snapshot_download

print("nemo-safe-synthesizer", md.version("nemo-safe-synthesizer"), "| torch", torch.__version__)
assert torch.cuda.is_available(), "torch cannot see the GPU. Try CUDA=130 bash setup_nss.sh, or check the driver."
print("gpu:", torch.cuda.get_device_name(0))
# NSS fine-tunes this model. Downloading it now (about 6 GB) shows straight away whether Hugging Face is reachable.
path = snapshot_download("HuggingFaceTB/SmolLM3-3B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
print("base model cached at", path)
EOF
"$VENV/bin/safe-synthesizer" --help | head -n 3
echo "NSS environment ready. Next: the smoke run in NSS_COMPARISON.md"
