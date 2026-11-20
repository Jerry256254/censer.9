#!/usr/bin/env bash
# Creates the virtualenv and installs PyTorch for the detected accelerator.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_VERSION="3.12"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

uv venv --python "$PYTHON_VERSION" .venv

if [ -e /dev/kfd ]; then
    echo "AMD GPU detected (ROCm) - installing torch+rocm7.1"
    INDEX="https://download.pytorch.org/whl/rocm7.1"
    TORCH_PKG="torch"
elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected - installing torch+cu126"
    INDEX="https://download.pytorch.org/whl/cu126"
    TORCH_PKG="torch"
else
    echo "No GPU detected - installing CPU torch"
    INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_PKG="torch"
fi

uv pip install --python .venv/bin/python --index-url "$INDEX" "$TORCH_PKG"
uv pip install --python .venv/bin/python numpy datasets tokenizers pytest

echo
echo "Verifying torch:"
.venv/bin/python -c "
import torch
print('  torch', torch.__version__)
print('  accelerator available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  device:', torch.cuda.get_device_name(0))
"
echo
echo "Done. Next: source .venv/bin/activate && make data && make train"
