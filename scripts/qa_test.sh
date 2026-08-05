#!/bin/bash
# QA smoke test for pulsar-ds4flash + DeepSeek-V4-Flash-0731 UD-Q2_K_XL
# Usage: qa_test.sh <model-dir> [--ctx N]
set -u
MODEL_DIR="${1:-/home/neron/models}"
SHARD1="$MODEL_DIR/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf"
BIN=/home/neron/projects/pulsar-ds4flash/target/release/pulsar-cli
CTX="${2:-4096}"

if [ ! -f "$SHARD1" ]; then echo "missing $SHARD1"; exit 1; fi
if [ ! -f "$BIN" ]; then echo "missing $BIN"; exit 1; fi

echo "=== model shards ==="
ls -la "$MODEL_DIR" | grep gguf
echo "=== GPU state ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

declare -a QUESTIONS=(
  "What is the capital of France? Answer with only one word."
  "What is 2+2? Answer with only the number."
  "What is the capital of Germany? Answer with only one word."
  "What color is the sky on a clear day? Answer with one word."
  "How many legs does a dog have? Answer with only the number."
)

for i in "${!QUESTIONS[@]}"; do
  Q="${QUESTIONS[$i]}"
  echo ""
  echo "=== Q$((i+1)): $Q ==="
  # -n 24 tokens, temp 0 for determinism
  timeout 300 env PULSAR_CPU=1 PULSAR_DEV_CACHE_GB=4 "$BIN" -m "$SHARD1" -p "$Q" -n 24 --ctx "$CTX" --temp 0 2>&1 | tail -20
  echo "exit: $?"
done
