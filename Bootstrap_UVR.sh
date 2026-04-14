#!/bin/bash
set -e
export KOVA_ROOT="$HOME/kova"
export BASE_MODEL_ID='Qwen/Qwen3.5-4B'
export KOVA_CKPT_ROOT="$KOVA_ROOT/checkpoints"
export PYTHONPATH="$KOVA_ROOT/src:$PYTHONPATH"
mkdir -p "$KOVA_ROOT/scripts" "$KOVA_ROOT/checkpoints" "$KOVA_ROOT/logs"
cd "$KOVA_ROOT"
apt-get update -qq && apt-get install -y -qq build-essential git curl wget sqlite3 openjdk-17-jdk rustc cargo nodejs npm gcc g++ golang-go
pip install -q transformers datasets accelerate peft 'trl>=0.15,<0.18' bitsandbytes sentencepiece safetensors sympy wandb playwright pytest huggingface_hub lm-eval tensorboard numpy
python -m playwright install chromium
npm install --silent tsx three
hf auth login --token $HF_TOKEN
curl -sSL https://raw.githubusercontent.com/FireworksAI26/kova-azr/refs/heads/main/run_uvr.py -o scripts/run_uvr.py
python scripts/run_uvr.py --max-hours 17 --batch-size 4
hf upload KovaUser/kova-uvr-qwen4b $KOVA_CKPT_ROOT/uvr/
