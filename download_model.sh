#!/bin/bash
# ============================================================
# Download do modelo Llama 3.1 8B GGUF para llama-cpp
# ============================================================

MODELS_DIR="$HOME/projetos/movie-bot/models"
mkdir -p "$MODELS_DIR"

MODEL_FILE="$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

if [ -f "$MODEL_FILE" ]; then
    echo "Modelo ja existe: $MODEL_FILE"
    ls -lh "$MODEL_FILE"
    exit 0
fi

echo "A fazer download do Llama 3.1 8B Instruct Q4_K_M (~4.7GB)..."
echo "Pode demorar varios minutos dependendo da ligacao."
echo ""

source "$HOME/projetos/movie-bot/backend/venv/bin/activate"

python -c "
from huggingface_hub import hf_hub_download
import os

path = hf_hub_download(
    repo_id='bartowski/Meta-Llama-3.1-8B-Instruct-GGUF',
    filename='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
    local_dir='$MODELS_DIR',
    local_dir_use_symlinks=False,
)
print(f'Download completo: {path}')
print(f'Tamanho: {os.path.getsize(path) / 1e9:.1f} GB')
"
