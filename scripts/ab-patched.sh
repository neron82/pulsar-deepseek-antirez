#!/bin/bash
# Patched-tree legs (sync removals + on-device embed broadcast + qwen38
# MTP prefill cap). Stops and restarts the 3090 inference server.
set -e
~/models/start_qwen38abl.sh stop
sleep 3
cd ~/projects/pulsar-ds4flash
M=~/models/qwen38flash/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf
SC=~/models/qwen38flash/UD-Q3_K_XL/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
P="Write a 200 word essay about the tradeoffs of event-driven architecture in embedded systems."
LOG=/tmp/ab3090
mkdir -p "$LOG"
for mode in trunk mtp; do
  for pass in 1 2; do
    tag="patched_${mode}_p${pass}"
    if [ "$mode" = mtp ]; then
      CUDA_VISIBLE_DEVICES=0 PULSAR_MTP=1 PULSAR_MTP_SIDECAR="$SC" \
        target/release/pulsar-cli -m "$M" -p "$P" -n 256 --ctx 4096 > "$LOG/$tag.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES=0 \
        target/release/pulsar-cli -m "$M" -p "$P" -n 256 --ctx 4096 > "$LOG/$tag.log" 2>&1
    fi
    echo "=== $tag rc=$? ==="
    grep -E 'tokens in|prefill|drafts accepted|panicked' "$LOG/$tag.log" || true
  done
done
~/models/start_qwen38abl.sh start
