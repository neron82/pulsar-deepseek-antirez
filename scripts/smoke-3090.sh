#!/bin/bash
# 3090 smoke test for the MTP sidecar expert-fetch fix (2026-08-29).
#
# Prereq: the 3090 inference engine is STOPPED, i.e.
#     ~/models/start_qwen38abl.sh stop
# so GPU 1 (3090, 24 GB) is free. GPU 0 (3060 Ti, 8 GB) is untouched.
#
# Legs (all greedy, one pass each - this is a smoke, not the A/B):
#   1 trunk    - baseline, no MTP
#   2 mtp      - MTP depth 1 via the sidecar, timing on
#   3 mtp-dbg  - PULSAR_MTP_DEBUG: moe_dbg / mtp-draft lines
#               (the sidecar expert fetch itself - the fix under test)
#   4 tier     - MTP with a 2 GiB expert pool -> tier path must stay clean
set -e
cd ~/projects/pulsar-ds4flash
M=~/models/qwen38flash/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf
SC=~/models/qwen38flash/UD-Q3_K_XL/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
P="Explain how a memory-mapped file works in a virtual memory system. Be concise but technical, covering page tables, demand paging, and copy-on-write semantics for a short, self-contained essay-driven architecture in embedded systems."
LOG=/tmp/smoke3090
mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=1

echo "=== 1 trunk (baseline, no MTP) ==="
./target/release/pulsar-cli -m "$M" -p "$P" -n 256 --ctx 4096 > "$LOG/1_trunk.log" 2>&1
grep -E 'tokens in|prefill' "$LOG/1_trunk.log"

echo "=== 2 mtp (depth 1, sidecar, timing) ==="
PULSAR_MTP=1 PULSAR_MTP_SIDECAR="$SC" PULSAR_MTP_TIMING=1 \
  ./target/release/pulsar-cli -m "$M" -p "$P" -n 256 --ctx 4096 > "$LOG/2_mtp.log" 2>&1
grep -E 'tokens in|prefill|drafts accepted|MTP' "$LOG/2_mtp.log"

echo "=== 3 mtp-dbg (sidecar expert fetch debug) ==="
PULSAR_MTP=1 PULSAR_MTP_SIDECAR="$SC" PULSAR_MTP_DEBUG=1 \
  ./target/release/pulsar-cli -m "$M" -p "$P" -n 8 --ctx 4096 > "$LOG/3_mtp_dbg.log" 2>&1
grep -E 'moe_dbg|mtp-draft|MTP expert|budget|resident' "$LOG/3_mtp_dbg.log" | head -20

echo "=== 4 tier (2 GiB expert pool -> tiers) ==="
PULSAR_MTP=1 PULSAR_MTP_SIDECAR="$SC" PULSAR_DEV_CACHE_GB=2 \
  ./target/release/pulsar-cli -m "$M" -p "$P" -n 128 --ctx 4096 > "$LOG/4_tier.log" 2>&1
grep -E 'tokens in|prefill|tier|drafts accepted' "$LOG/4_tier.log"

echo "=== done: logs in $LOG ==="
