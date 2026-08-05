# Pulsar DS-V4-Flash 128k Performance Optimization Ledger

Status: ACTIVE (2026-08-04). Model: DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf (~86.7 GB).
Hardware: RTX 3090 (CUDA 0, primary) + RTX 3060 Ti (CUDA 1, tier). PULSAR_CACHE_GB=85 (whole model in RAM).

Every entry = one experiment with a verdict. A/B discipline: same-binary batch-vs-host for
correctness (cross-build codegen variance at the razor mask boundary makes cross-build dump
comparisons unreliable — see "Methodology lessons" below).

## Summary of wins (cumulative on the 128k target)

| Change | Prefill (117k tok) | Decode @128k | Correctness |
|---|---|---|---|
| Baseline (bench128.log) | 14007.5s (8.36 tok/s eff) | 3.80 tok/s | reference |
| GPU indexer + batched mask path + partial top-k (bench128-partial.log) | 10493.9s (11.16 tok/s eff) | 5.21 tok/s | ids byte-identical @128k |
| + ballot-compacted attention value loop (bench128-all.log) | 8489.2s (13.80 tok/s eff) | 6.01 tok/s | ids byte-identical @128k |
| + barrier reduction / parity double-buffering (bench128-barrier.log) | 7968.1s (14.70 tok/s eff) | 6.23 tok/s | ids byte-identical @128k |

Net: **+43.1% prefill, +63.9% decode** on the 128k scenario, output ids byte-identical
at every stage. Barrier reduction added -6.1% prefill (8489 -> 7968s) and +3.7% decode
on top of the ballot build.

## What worked

### 1. GPU indexer scorer + batched mask path (the big one)
- Problem: host indexer_allowed scoring was 2.5ms/call, quadratic in context (81.9s of 177s
  prefill at 3.2k ctx; ~2328s of the 10466s 128k prefill). Plus one blocking D2H per masked
  token (~3630 stream drains per chunk).
- Fix: prep query on host (rope+QAT, bit-exact math), score on device via
  dsv4_idx_scores kernel, defer attention launches, ONE bulk D2H readback of all scores,
  host top-k per token, ONE packed mask H2D, deferred attention launches in original i order.
- Ring-snapshot requirement: the live SWA ring is mutated in-loop by later tokens, so a
  deferred attention at end-of-chunk would read FUTURE tokens (causal violation). Each
  masked token's window is D2D-copied at record time.
- Verdict: WIN. Prefill idx_host collapsed from 5191s to 2328s; decode +61% at 3.3k ctx.

### 2. Warp-shuffle max reduce in attention
- Replaced the 6-barrier shared-memory max tree with __shfl_xor_sync (0 barriers).
- First attempt was NONDETERMINISTIC (leaders writing red[] raced with red[tid]=s writes) —
  fixed with a separate wmax[4] array.
- Verdict: WIN. +5.6% decode at 9.7k ctx (17.0 vs 16.1 tok/s). Bit-exact.

### 3. Partial-selection top-k (select_nth_unstable_by)
- Replaced full sort_unstable_by with O(n) partition. The comparator (score desc, idx asc)
  is a total order, so the top-k SET is identical to a full sort's first k.
- Verdict: WIN. Included in the bench128-partial numbers above. ids identical @128k.

### 4. Ballot-compacted attention value loop (2026-08-04, in flight)
- Problem: the value-accumulation loop walks all 128 rows per chunk even though the mask
  hides ~97% of them at 128k (top_k=512 over ~29k rows -> ~2 visible rows per 128-chunk).
- Fix: __ballot_sync per warp to find visible rows, compact their indices into shared
  memory in ascending order, accumulate ONLY those. BIT-EXACT BY CONSTRUCTION: the original
  loop `continue`s on w==0.0f doing zero float ops, so dropping hidden rows changes nothing
  (same rows, same order, same ops). Chunk boundaries are untouched — unlike the earlier
  killed dense-buffer compaction which shifted rows across chunk boundaries and changed the
  float reduction order.
- A/B: batch==host, 77616 masks 0 diffs, dumps identical (1329907 bytes). GREEN.
- 60k speed run (bench60-compact.log vs bench60.log baseline): prefill 58572 tok in
  3147.35s vs 3662.96s = **-14.1%**; decode 7.38 vs 6.85 tok/s = **+7.7%**; ids identical.
  Depth-scaling: per-chunk -9% @16k, -14.7% @27.6k, -17% @38.4k, -19.5% @47.6k (the win
  grows with context; the added barriers dominate below ~8k where most rows are visible).
  Interesting: attn-kernels timer went UP (+34s, the 2 added barriers) but route_read
  dropped 525s — the attention finishing earlier shortens the host drain at the router
  readback. Net effect on the 128k target should be larger than the 60k number.

### 5. Bulk idx_q_prep H2D — REVERTED (regression, +2.4%)
- Problem hypothesis: the batched mask path did ONE blocking idx_q_prep.write per masked
  token (~768/chunk at 128k), each serializing against the previous score kernel on the
  default stream.
- Fix: host-prep ALL queries first, ONE bulk write, then queue the score kernels
  back-to-back. Same bytes, same order -> bit-identical (A/B GREEN: batch==host, 77616
  masks 0 diffs, dumps identical 1325731 bytes).
- Speed verdict: 60k prefill 3223.50s vs 3147.35s compaction baseline = **+76s (+2.4%
  SLOWER)** at every depth (+1s/chunk consistently); decode unchanged (7.44 vs 7.38,
  noise). REVERTED.
- Why: the old per-token flow hid the host prep under GPU kernel execution (launches are
  async, only the small write blocks). The bulk flow serializes ALL host prep first with
  the GPU idle, losing more overlap than it saves in write syncs. Lesson: at these
  transfer sizes (few KB per query), sync overhead < lost overlap; do NOT bulk-transfer
  small per-token buffers on the default stream — keep the interleave.

### 6. Barrier reduction in attention (parity double-buffering) (2026-08-05)
- Problem: the ballot-compacted value loop still took 5 __syncthreads per chunk (red
  leftover, wmax, s_w, crow, end-of-loop). At 128k that's ~230 chunks x 5 barriers x
  64 heads per token — the barriers are a per-chunk CONSTANT tax that dominated the
  shallow zone (the old ballot build was +7% SLOWER below ~8k).
- Fix: double-buffer s_w/wmax/crow by chunk parity ((base>>7)&1), drop the dead red[]
  write (shuffle reduce never read it), drop the s_w barrier (each thread only reads its
  own s_w at the vis ballot), drop the end-of-loop barrier (parity prevents write/read
  collision between a slow reader of chunk k and a fast writer of chunk k+1). 5 barriers
  -> 2 per chunk. NO float op changed -> bit-exact by construction.
- A/B: batch==host, 77616 masks 0 diffs, dumps identical (1329245 bytes). GREEN.
- 60k speed run (bench60-barrier.log): prefill 58572 tok in 2888.80s vs 3147.35s
  compaction baseline = **-8.2%**; vs original baseline -21.1%; decode 7.93 vs 7.38
  tok/s = **+7.5%**; ids identical. The win is a flat per-chunk constant (-3.5s/chunk
  at EVERY depth, shallow AND deep) — the barrier tax is gone. The ballot build's
  shallow-zone penalty also disappeared (now faster than baseline even at 3.8k).
- Lesson: parity double-buffering removes loop-carried barriers with zero float change;
  the shuffle-reduce comment about "leaders write a SEPARATE array" was the only thing
  keeping the dead red[] write alive.

### 7. Host top-k parallelization (scoped threads) (2026-08-05)
- Problem: per chunk, the host scores up to `cap` compressed rows (cap grows with
  depth: ~2-4k @4k, ~10k @40k, ~29k @128k) and selects top_k via sequential
  select_nth_unstable_by — 768 tokens per chunk, one after another. At depth the
  idx host segment is ~90% of the attention-token loop (573s of 593s @58k).
- Fix: chunk the masked-token top-k work over scoped threads (max 8 via
  available_parallelism). Each thread does EXACTLY the same select_nth_unstable_by
  on its own score slice and writes its own mask region — bit-exact by
  construction (same comparator, same partition algorithm, no float ops).
- A/B: batch==host, 77616 masks 0 diffs, dumps identical (1329344 bytes). GREEN.
- 60k speed run (bench60-topk-final.log vs bench60-barrier.log, same prompt/model/
  env): prefill 58572 tok in 2832.18s vs 2888.80s = **-1.96%**; the win is
  depth-scaling (bucket Δ%: -0.52 @0-8k, -1.13 @8-16k, -1.50 @16-24k, -1.90
  @24-32k, -2.16 @32-40k, -2.46 @40-49k, -2.73 @49-58k) — shallow zone is
  overhead-dominated, deep zone amortizes; last chunks @58k save ~1.4-1.5s each.
  Decode 7.92 vs 7.93 tok/s (noise); ids identical.
- Extrapolation: at 128k (cap ~29k) the deep-zone curve suggests ~-3.5-4% in the
  final third. Cheap, bit-exact, keep.

## What did NOT work (honest failures)

### A. Dense-buffer compaction of attention rows — KILLED (not bit-exact)
- Removing hidden rows shifts visible rows across 128-row chunk boundaries -> different
  float summation order -> rounding drift in the post-attention heads hash. Same drift
  class as grouped-MoE. The ballot compaction (item 4) avoids this by keeping chunk
  boundaries intact.

### B. Grouped MoE — KILLED (not bit-exact)
- Float reordering in the expert path changed output. Hard blocker per Basti's rules.

### C. Chunk-skip (all-masked attention chunks) — REVERTED (net loss at 60k)
- ORIGINAL (killed): in-kernel phase-0 ballot pass over all chunks to find empty
  ones. Killed by data: at A/B depth (~3.9k) only ~8% of chunks are empty, and the
  phase-0 walk costs more than the skip saves.
- DATA that revived it: PULSAR_MASK_LOG across 1.67M masks to 80k depth shows the
  empty-chunk fraction SCALES with depth: 0.3% @0-8k, 16.3% @32-40k, 25.9% @49-57k,
  35.8% @65-80k (and growing). At the 128k target it will exceed 50%.
- NEW DESIGN (landed, then killed by measurement): host builds a 1-byte per-128-row
  bitmap at top-k time, kernel skips 0-chunks block-uniform. BIT-EXACT (A/B GREEN:
  77616 masks 0 diffs, dumps identical 1329344 bytes).
- FINAL VERDICT (60k A/B, chunkskip vs barrier baseline): NET LOSS at every depth —
  +1.5s/chunk @13-16.5k, +2.0s @21-24k, widening with depth. An empty chunk is
  memory-bound and cheap; the skip tax (extra __syncthreads + uniform branch +
  bitmap read per chunk) costs more than skipping saves. REVERTED bit-exact
  (kernel/FFI/wrappers/host restored; 60k run killed after the trend was clear).
- Do NOT revisit without a 128k-only measurement — but note the loss curve was
  still widening, and the tax grows with chunk count (i.e. with the same depth
  that creates the empties).

### D. Layer-crossing H2D expert prefetch — REVERTED (prediction incorrect)
- 'Layer N's experts == layer N+1's' is false per-token; MoE routing differs between
  layers, so the drain accepted wrong-layer weights and ids came out all zeros.
  MLA's prefetch works only because decode predicts the next token (MTP); prefill has no
  such source. Queue block kept behind `if false && flag` for analysis.

### E. Forcing expert triples into the dev cache — REVERTED (churn cost)
- maybe_insert_triple_force made moe-kernels 5.1s -> 17.3s (coldest-scan victim); even
  O(1) ring_next -> 19.5s. With a 6GB cache against a ~4500-triple working set every
  admission is evicted before reuse.

## Methodology lessons (critical)

1. Cross-build codegen variance at the razor mask boundary: three builds of identical
   source produced three different outputs (1312614 / 1314171 / 1314486 bytes), each
   internally consistent. The mask boundary n=513-517 (1-5 rows past top_k) flips excluded
   rows on last-ulp differences. Rule: A/B only within the same binary/build epoch.
2. The host path (PULSAR_NO_GPU_IDX) is the independent oracle and MUST keep original
   semantics (in-loop masked attention). A restructure that silently skipped masked
   attention in host mode made the "reference" garbage — divergence moved @2051 -> @2055
   after restoring it.
3. Deferred attention reading the live ring = causal violation that corrupts hidden state
   and cascades. Ring snapshot at record time fixed batch==host.
4. route_read (D2H of router logits, ~64% of 128k prefill wall) is NOT bandwidth-bound:
   it's the host draining the GPU pipeline at each layer. Fixing it needs stream/async
   infra, not transfer batching alone.

## Remaining levers (queued, not started — no stacking on an unproven base)

1. Bulk idx_q_prep H2D (768 per-token blocking writes -> 1 write). Implemented, built,
   awaiting A/B after the ballot run finishes. Same bytes, same order -> bit-identical.
2. Attention score-loop compaction (the ballot fix only removes dead VALUE iterations;
   the score dot-product still walks every row). Bigger fish at depth, but the mask is
   DERIVED from the scores, so it needs a two-pass or streaming approach — bit-exactness
   risk is higher.
3. route_read async: overlap the router D2H with the tail work (hc_post, rope, grouped
   out projections) via a side stream. Requires the stream infra that doesn't exist yet.
4. Both-GPU decode overlap / expert-tier residency policy: tier on 3060 Ti held only
   866/11008 triples (7.9%) at 128k; vram cache 25-32% hits. The user explicitly allowed
   using both GPUs. The tier sizing and cache policy is a config/design lever, not a
   kernel change.

## Build note
CUDA 13 dropped Pascal: the default arch list "61,75,80,86,89" FAILS. Build with:
    PULSAR_CUDA_ARCH=86 cargo build --release
(both GPUs are sm_86; keeps the cc>=8 GEMM gate).
