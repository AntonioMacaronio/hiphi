#!/usr/bin/env bash
# Source this file before dataset, viewer, or Python commands on this node.
export HIPHI_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_STORAGE_ROOT=/data/sky/projects/hiphi
export SHARED_CACHE_ROOT=/data/sky/cache
export TMPDIR="$PROJECT_STORAGE_ROOT/tmp"
export XDG_CACHE_HOME="$SHARED_CACHE_ROOT"
export PIP_CACHE_DIR="$SHARED_CACHE_ROOT/pip"
export UV_CACHE_DIR="$SHARED_CACHE_ROOT/uv"
export HF_HOME="$SHARED_CACHE_ROOT/huggingface"
export HF_XET_CACHE="$HF_HOME/xet"
export TORCH_HOME="$SHARED_CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$PROJECT_STORAGE_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$PROJECT_STORAGE_ROOT/cache/triton"
export WANDB_DIR="$PROJECT_STORAGE_ROOT/runs/wandb"
export WANDB_CACHE_DIR="$SHARED_CACHE_ROOT/wandb"
export CONDA_PKGS_DIRS="$SHARED_CACHE_ROOT/conda-pkgs"
export PYTHONPYCACHEPREFIX="$PROJECT_STORAGE_ROOT/cache/pycache"
export HIPHI_PYTHON=/home/sky/miniconda3/envs/ei_world_model/bin/python
# Reuse an existing login without copying credentials or printing their contents.
if [[ -z "${HF_TOKEN_PATH:-}" && ! -f "$HF_HOME/token" && -f /home/sky/.cache/huggingface/token ]]; then
    export HF_TOKEN_PATH=/home/sky/.cache/huggingface/token
fi
mkdir -p "$TMPDIR" "$PROJECT_STORAGE_ROOT/datasets" "$PROJECT_STORAGE_ROOT/runs" \
    "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$HF_XET_CACHE" "$TORCH_HOME" \
    "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
    "$CONDA_PKGS_DIRS" "$PYTHONPYCACHEPREFIX"
printf 'HiPHI checkout: %s\nHiPHI data and outputs: %s\n' "$HIPHI_REPO_ROOT" "$PROJECT_STORAGE_ROOT"
