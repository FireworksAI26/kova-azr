#!/bin/bash
set -euo pipefail
trap 'echo "ERROR: line ${LINENO}: command failed: ${BASH_COMMAND} (exit $?)" >&2' ERR
export PS4='+(line ${LINENO}): '
set -x
: "${HF_TOKEN:?HF_TOKEN env var is required}"
export HF_TOKEN="$HF_TOKEN"
export KOVA_ROOT="$HOME/kova"
export BASE_MODEL_ID='Qwen/Qwen3.5-4B'
export KOVA_CKPT_ROOT="$KOVA_ROOT/checkpoints"
export PYTHONPATH="$KOVA_ROOT/src:${PYTHONPATH:-}"
mkdir -p "$KOVA_ROOT/scripts" "$KOVA_ROOT/checkpoints" "$KOVA_ROOT/logs"
cd "$KOVA_ROOT"
apt-get update -qq && apt-get install -y -qq build-essential git curl wget sqlite3 openjdk-17-jdk rustc cargo nodejs npm gcc g++ golang-go
python -m pip install -q --upgrade pip
python -m pip install -q transformers datasets accelerate peft 'trl==0.15.2' mergekit bitsandbytes sentencepiece safetensors sympy wandb playwright pytest huggingface_hub lm-eval tensorboard numpy
python -m playwright install chromium
npm install --silent tsx three
curl -fsSL https://raw.githubusercontent.com/FireworksAI26/kova-azr/refs/heads/main/run_uvr.py -o scripts/run_uvr.py
python scripts/run_uvr.py --max-hours 17 --batch-size 4
hf upload "KovaUser/kova-uvr-qwen4b" "${KOVA_CKPT_ROOT}/uvr/"
