#!/bin/bash
# 3090 speed A/B: baseline (pre sync-removal) vs current tree
# (sync removals + on-device embed broadcast).
#
# Prereqs:
#   - the llama.cpp engine on the 3090 is STOPPED (GPU 0 must be free;
#     CUDA device 0 = 3090, 1 = 3060 Ti)
#   - /tmp/pulsar-cli.baseline exists (pre-patch build saved 2026-08-28);
#     rebuild it from the pre-patch tree if missing
# Each leg runs twice: the first pass warms the page cache + host expert
# cache, the second is the number to compare. Raw logs in /tmp/ab3090/.
set -e
cd ~/projects/pulsar-ds4flash
M=~/models/qwen38flash/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf
SC=~/models/qwen38flash/UD-Q3_K_XL/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
P="Write a 200 word essay about the tradeoffs of event-driven architecture in embedded systems."
LOG=/tmp/ab3090
mkdir -p "$LOG"

for BIN in /tmp/pulsar-cli.baseline target/release/pulsar-cli; do
    N=$(basename "$BIN")
    for mode in trunk mtp; do
        for pass in 1 2; do
            tag="${N}_${mode}_p${pass}"
            log="$LOG/$tag.log"
            rc=0
            if [ "$mode" = mtp ]; then
                CUDA_VISIBLE_DEVICES=0 PULSAR_MTP=1 PULSAR_MTP_SIDECAR="$SC" \
                    "$BIN" -m "$M" -p "$P" -n 256 --ctx 4096 \
                    > "$log" 2>&1 || rc=$?
            else
                CUDA_VISIBLE_DEVICES=0 "$BIN" -m "$M" -p "$P" -n 256 --ctx 4096 \
                    > "$log" 2>&1 || rc=$?
            fi
            if [ "$rc" -ne 0 ] || ! grep -q 'tokens in' "$log"; then
                echo "FAIL $tag (rc=$rc):" >&2
                tail -5 "$log" >&2
                exit 1
            fi
            echo "=== $tag ==="
            grep -E 'tokens in|prefill' "$log"
        done
    done
done
