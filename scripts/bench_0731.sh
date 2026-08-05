#!/bin/bash
# README-style benchmark: pulsar-ds4flash + DS-V4-Flash-0731 UD-Q2_K_XL
# Protocol per README: n=32 vs n=64, warm census (run once before measuring),
# CPU lane on, report variance across runs.
set -u
MOD="/home/neron/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
BIN=/home/neron/projects/pulsar-ds4flash/target/release/pulsar-cli
CTX="${1:-8192}"
PROMPT="The quick brown fox jumps over the lazy dog. The capital of France is"

echo "=== benchmark: DS-V4-Flash-0731 UD-Q2_K_XL (ctx $CTX) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

echo ""
echo "--- warm run (n=32, fills census/tiers) ---"
timeout 300 env PULSAR_CPU=1 PULSAR_DEV_CACHE_GB=4 "$BIN" -m "$MOD" -p "$PROMPT" -n 32 --ctx "$CTX" --temp 0 2>&1 | grep -E "tokens in|resident|auto budget" | tail -3

for n in 32 64; do
  echo ""
  echo "--- n=$n ---"
  for run in 1 2 3; do
    timeout 400 env PULSAR_CPU=1 PULSAR_DEV_CACHE_GB=4 "$BIN" -m "$MOD" -p "$PROMPT" -n $n --ctx "$CTX" --temp 0 2>&1 | grep -E "tokens in" | tail -1 | sed "s/^/run $run: /"
  done
done
