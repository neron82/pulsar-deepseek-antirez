#!/bin/bash
# QA via chat template (stdin pipe). Reasoning models need the template
# to suppress thinking-token leaks in plain completion mode.
set -u
MOD="/home/neron/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
BIN=/home/neron/projects/pulsar-ds4flash/target/release/pulsar-cli
CTX=4096

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
  printf '%s\n' "$Q" | timeout 240 env PULSAR_CPU=1 PULSAR_DEV_CACHE_GB=4 "$BIN" -m "$MOD" --chat -p "hello" -n 48 --ctx "$CTX" --temp 0 2>/tmp/qa_err_$i.txt | grep -vE "^>|pulsar:|^$" | tail -4
  echo "exit: $?"
done
