#!/bin/bash
# Sustained-decode benchmark for pulsar-ds4flash + DeepSeek-V4-Flash-0731
# Warm-run protocol per the README: first run fills the census, second run measures.
set -u
MODEL_DIR="${1:-/home/neron/models/deepseek-v4-0731/UD-Q2_K_XL}"
SHARD1="$MODEL_DIR/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf"
BIN=/home/neron/projects/pulsar-ds4flash/target/release/pulsar-cli
CTX="${2:-8192}"

echo "=== benchmark: pulsar-ds4flash + DS-V4-Flash-0731 UD-Q2_K_XL ==="
echo "model: $SHARD1"
echo "ctx:   $CTX"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# Warm run (fills census, hot cache, tiers) - short
echo ""
echo "--- warm run (n=32) ---"
timeout 300 env PULSAR_CPU=1 PULSAR_DEV_CACHE_GB=4 "$BIN" -m "$SHARD1" -p "The capital of France is" -n 32 --ctx "$CTX" --temp 0 2>&1 | tail -3

# Sustained benchmark: n=64 (README standard), repeat 3x for variance
echo ""
echo "--- sustained runs (n=64) ---"
for run in 1 2 3; do
  echo "run $run:"
  timeout 600 env PULSAR_CPU=1 PULSAR_DEV_CACHE_GB=4 "$BIN" -m "$SHARD1" -p "The quick brown fox jumps over the lazy dog. The capital of France is" -n 64 --ctx "$CTX" --temp 0 2>&1 | grep -E "^pulsar:|^Paris|tok/s"
done
