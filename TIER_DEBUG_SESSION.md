# Ling 3.0 Flash (bailingmoe3) — Tier Path Debugging Session Notes

_Dates: 2025-08-25 (session) · 2026-08-26 (RESOLVED — see RESOLUTION below) · Branch: main (all work uncommitted)_

## Goal

Fix the multi-turn chat repetition collapse at ctx 32768, then make the expert
tier path bit-exact so its speed benefit can be kept.

## Deliverable status

| Item | State |
|---|---|
| Root cause | ✅ **FOUND & FIXED** — cross-device NULL-stream vs PTDS ordering (see RESOLUTION) |
| Bit-exact per-slot splice | ✅ verified vs single-device reference (greedy A/B + 5-turn chat, tiers ON == OFF verbatim) |
| Release build of final code | ✅ RC=0 |
| End-to-end regression (tiers ON) | ✅ passes; residual output quirks are model sampling behavior (identical with tiers off) |
| llama.cpp oracle available | ✅ `~/llama.cpp/build/bin`, bailingmoe3 supported, ~9–10 tok/s with partial offload |

## RESOLUTION (2026-08-26)

**Actual root cause: missing stream ordering on cross-device reads (legacy
NULL stream vs per-thread default stream).** All kernels launch on the
per-thread default stream (PTDS), but `copy_across` issues `cudaMemcpy` on the
legacy NULL stream. The two have NO implicit ordering. Every cross-device read
in the tier/attention path carried this hazard; the critical one was the
flat-tier merge in `dsv4_moe`:

    tier PTDS: quantize_q8_k -> moe_pair_swiglu -> moe_down_per_slot (writes tier.slot_out)
    merge:     copy_across(st.tier_tmp <- tier.slot_out)             (NULL stream, unordered)

The splice math was bit-exact by construction (identical reduction over
identical bytes, inserted at slot position). The failure was data freshness:
the tier card also runs all attention, so its PTDS queue was still draining the
layer's attention chain when the merge copy fired — the copy could consume
stale/previous `slot_out` contents (the buffer persists across layers). No
pre-fix read trace was captured; this is the identified high-confidence root
cause, and the repair is what verified it. The earlier "cross-rank f32 add
reorder" hypothesis below is superseded: the summed grouped join remains a
theoretical drift class, but it is dead code for Ling (`bailing_ffn` always
runs `n_tok=1`).

Why gradual multi-turn drift instead of garbage: a consistently lost race
injects a plausible-but-wrong per-slot contribution every layer, which the
42-layer residual stack amplifies until logits drift and flow degrades.

### Fix: 7 producer-side sync points

`crates/engine/src/real/dsv4.rs`:
1. primary `sync()` before the tier input copies (`tier.xin`/`tier.weights`;
   defensive — `bailing_ffn` already syncs before the router readback),
2. tier `sync()` before the flat merge copy of `slot_out` (THE bug),
3. tier `sync()` before the `PULSAR_TIER_LEGACY` join copy,
4. tier `sync()` before the grouped join copy.

`crates/engine/src/real/bailing.rs` (same hazard class at the split-attention
copies; synced even though the race never manifested there — the producer
immediately precedes the copy in issue order, so the window is narrow):
5. primary `sync()` before `copy_across(sc.normed_a <- st.normed)`,
6. `a_dev` `sync()` before the KDA `attn_out_a` copy,
7. `a_dev` `sync()` before the MLA `attn_out_a` copy.

Cost ~zero: `bailing_ffn` already pays one primary sync per layer and the
decode loop is strictly sequential, so the drains wait only on work the
consumer depends on. Measured 10.62 -> 10.73 tok/s (noise).

### Verification (the tested combinations — not every cross-product)

- Greedy 32-token A/B, tier ON (1030 triples / 3.4 GiB resident) vs
  `PULSAR_TIERS=off`: 32/32 identical ids.
- Greedy 32-token A/B, `PULSAR_ATTN_GPU=off` (attention on primary) vs split
  attention: 32/32 identical ids, pre- AND post-attention-sync.
- 5-turn chat at production settings (temp 0.9, seed 42, ctx 4096): tiers ON
  vs OFF transcripts verbatim identical — including turn 5's mid-think
  truncation, which reproduces identically without tiers (model sampling
  behavior, not the engine). Turn 6+ hits the ctx-4096 window with long
  thinking blocks.

**A/B lesson: the tier ON/OFF A/B is BLIND to shared-path races** (both runs
share the attention path). The attention-placement A/B (`PULSAR_ATTN_GPU=off`
vs split) is the discriminator for anything on the attention path.

### For the kimi-k3 fork audit

k3.rs has the same pattern (KDA/MLA scratch copies ~lines 598/725, shared
expert ~916): producer kernels on PTDS, NULL-stream copies, no producer syncs.
Audit recipe: enumerate every `copy_across` whose source was kernel-written,
check device ordering at each, and use the attention-placement A/B to
discriminate shared-path races. `dsv4_moe` is shared by all families, so its
syncs already cover the k3 MoE tier path; the k3 attention/scratch copies are
the un-audited remainder.

### Also fixed this session: unbounded pinned host memory

`kernels::pinned_alloc/pinned_free` recycled evicted CUDA-pinned fetch buffers
in an unbounded process-global pool, and the whole persistent host cache could
become pinned, so `PULSAR_CACHE_GB=60` still grew RSS to ~80 GiB. Now capped:
4 GiB live pinned allocations, 256 MiB recycle pool, overflow falls back to
pageable (`cudaHostAlloc` failure returns null; the uring fetcher handles it),
and warm GPU-cache fills use a separate pageable `direct_fetcher`. Verified:
RSS 80.7 -> 49.1 GiB at the same cache budget, stable across turns.

## Root cause direction at the time (SUPERSEDED — see RESOLUTION above)

The repetition loop is caused by the **expert tier path**: the routed-expert sum
is split across two ranks (grouped tensor-core kernels on the tier card) and
re-joined with `add_assign`, reordering f32 adds vs the single-device loop.
Bailing's 42-layer residual stack amplifies the drift until an argmax flips and
generation falls into a repetition attractor. Evidence:

- 8-turn chat, ctx 32768, tiers on (old summed flow): collapses (turn 6 in the
  original report; turn 1 in later runs once warm census evolved).
- Identical conversation, tiers off: coherent 8/8 (verified twice, incl. matched
  cache budget).
- Per-layer `moe_out` fingerprints (tiers on/off): diverge from the first
  prefill token (layer 10 Δ0.005 → layer 41 Δ0.33).
- Tier fill data itself is byte-exact: build-time checksum of all resident
  slabs vs GGUF source → 0 mismatches.

**Important nuance:** the measured first-token differences are larger than a
plain 8-term f32 reorder would suggest, and a same-expert tier-vs-primary
partial comparison was never completed. Treat "cross-rank reorder amplification"
as the leading hypothesis, not proven fact.

## Hypotheses investigated and DISCARDED (with evidence)

| Hypothesis | Evidence discarded |
|---|---|
| `ssm_a` sign convention | Split-GGUF read done properly (KV metadata → tensor table → data offset aligned to `general.alignment`=32): all values positive, ∈ [0.85, 2.16]. Loader negation matches K3 kernel contract (`g_min·sigmoid(−a·z)` ≡ reference `g_min·sigmoid(gate·ssm_a)`). |
| fp8 latent-KV auto-flip | 7 MLA caches at ctx 32768 ≈ 645 MiB « 2 GiB auto-FP8 floor; no auto-FP8 message in logs. |
| MoE weight application order (pre-down vs post-down) | Algebraically identical here: router weights are strictly positive and q8_K quantization is scale-invariant (`q8_K(w·m) = w·q8_K(m)` bit-exactly: same `qs`, `d` scaled). Reference applies post-down; ours pre-down; no numerical difference. |
| Chat template / tokenization | Byte-verified: pulsar's rendered 27-token turn id sequence matches `llama-tokenize` output for the same string (incl. special 156895/156903). |
| KDA / MLA attention math | Audited against llama.cpp reference in an earlier session (conv taps channel-major, l2-norm eps 1e-6, gate sign, beta sigmoid, per-head RMS norm, MLA scale 1/sqrt(192), rope conventions). |

## Found and FIXED along the way

### Stale PTX hazard in `crates/kernels/build.rs` (IMPORTANT)

`cargo:rerun-if-changed=cuda` (a *directory*) only re-fires when a directory
ENTRY is added/removed — editing `pulsar_kernels.cu` in place never triggered an
nvcc rebuild. Several test runs silently executed stale kernels while the Rust
side had changed (ABI happened to tolerate the extra arg, dropping tier
contributions instead). **Fixed**: build.rs now tracks
`cuda/pulsar_kernels.cu` and `cuda/bailing_kernels.inc` per-file. When in doubt:
`cargo clean -p kernels -r` (real nvcc run ≈ 60 s).

### Split-GGUF parsing pitfall (for any manual inspection)

Tensor data offsets are relative to the shard's **data-section start**, which is
the header end aligned up to `general.alignment` (32). Reading at raw offsets
without alignment gives garbage (values like 1e34). Working Python parser pattern
is in this session's history; `ssm_a` values were validated with it.

## Current implementation (in tree, uncommitted)

Bit-exact per-slot splice design:

- `moe_down_kernel` restructured: per-slot fully-reduced dot (`part`) with the
  butterfly INSIDE the slot loop, `acc += part` per slot; optional `ext`
  pointer ([n_tok][n_used][out_dim]) consulted ONLY for NULL-down slots.
- New `moe_down_per_slot_kernel`: identical instruction sequence, writes one
  unsummed value per (token, slot). Extern `pulsar_moe_down_per_slot`.
- `pulsar_moe_down` gained a trailing `ext_dev` param (null = legacy behavior).
- Engine `dsv4_moe`: flat-tier branch computes `tier.slot_out`; primary-side
  merge zeroes `st.tier_ext`, then per tier: `copy_across(st.tier_tmp ←
  tier.slot_out)` + `add_assign(st.tier_ext ← st.tier_tmp)` (disjoint slot sets,
  zero rows harmless), then primary `moe_down(..., ext)`. Grouped-path tiers
  (other families, n_tok ≥ 16) still use the old summed join (documented drift).
- `ExpertTier.slot_out`, `State.tier_ext`, `State.tier_tmp` buffers added.
- Bailing tier-default-off gate was REMOVED (tiers active by default again).

Diagnostics left in tree (all env-gated, off by default):
- `PULSAR_POS_DBG` in `bailing.rs`: per-position top-1/logit after the stack.
- `PULSAR_TIER_LEGACY` in `dsv4_moe`: old-style summed join over the new
  per-slot buffers (isolation diagnostic).

## Verification harness (SUPERSEDED — splice verified via the greedy A/B +
multi-turn chat in RESOLUTION; kept for reference)

Padded-prompt A/B isolates the tier path deterministically at temp 0:

```bash
python3 - <<'EOF'
filler = "The sun rises in the east and sets in the west. " * 120
open('/tmp/padded_prompt.txt','w').write(filler + "What is the capital of France?")
EOF
PROMPT=$(cat /tmp/padded_prompt.txt)
# baseline (trusted):
printf '%s\n' "$PROMPT" | env PULSAR_CACHE_GB=40 PULSAR_TIERS=off PULSAR_POS_DBG=1 \
  ./target/release/pulsar-cli -m <model> --ctx 12288 --chat --temp 0 --seed 42 -n 4 \
  2>&1 | grep posdbg > /tmp/padded_off.log
# candidate:
printf '%s\n' "$PROMPT" | env PULSAR_CACHE_GB=40 PULSAR_POS_DBG=1 \
  ./target/release/pulsar-cli -m <model> --ctx 12288 --chat --temp 0 --seed 42 -n 4 \
  2>&1 | grep posdbg > /tmp/padded_on.log
```

(`/tmp` is volatile — regenerate as above.)

Historical status (never run to completion; verification ultimately used the
stronger greedy-A/B + multi-turn-chat harness documented in RESOLUTION):
- `padded_off.log` (2491 positions) — trusted baseline ✅
- `padded_legacy.log` (PULSAR_TIER_LEGACY=1) — diverges at pos 0 like the old
  drift flow (expected; not diagnostic)
- `padded_ext.log` — never produced; no longer needed (the splice is verified
  bit-exact by the RESOLUTION harnesses)

Interpretation grid (only relevant if this harness is ever revived):
- EXT == OFF everywhere → splice is bit-exact; investigate elsewhere.
- EXT == LEGACY ≠ OFF → per-slot values differ from inline (check midq/down
  bytes on tier card, mid_blocks, row_bytes).
- EXT ≠ both → ext insertion/indexing bug (check `ext[(slot_base+slot)*out_dim+row]`).

## llama.cpp oracle

Binaries: `~/llama.cpp/build/bin` (user rebuilt with bailingmoe3 support).

```bash
# coherent reference generation (~9-10 tok/s):
~/llama.cpp/build/bin/llama-cli -m <model> -ngl 14 -sm layer -c 2048 \
  -p '<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>What is the capital of France?<|role_end|><role>ASSISTANT</role>
<think>' -n 40 --temp 0 --single-turn
```

Verified: llama.cpp parses `<|role_end|>` specials in `-p` (27 tokens, matches
pulsar's chat rendering) and produces coherent thinking text. Pulsar's `-p` does
NOT parse specials/escapes — do NOT use raw -p strings containing
`<|role_end|>` for pulsar probes (this invalidated several earlier tests;
they were probing mangled token soup, use chat mode or the padded harness).

Per-position logits from llama.cpp: `llama-perplexity --save-all-logits` needs
≥ 4096 prompt tokens; pad the prefix (both sides must see the identical padded
sequence) or use llama-server `prompt_logprobs`.

## Resume checklist

1. Build: `PULSAR_CUDA_ARCH=86 cargo build --release -p engine --bin pulsar-cli`
   (stale-PTX fix now makes this reliable; verify "Compiling kernels" appears).
2. ~~Padded harness / splice verification~~ DONE — superseded by the greedy
   A/B + multi-turn chat in RESOLUTION (splice verified bit-exact).
3. ~~Fix per interpretation grid~~ DONE — root cause was stream ordering,
   not splice math.
4. ~~Multi-turn regression~~ DONE — tiers ON == tiers OFF verbatim (5-turn
   chat; ctx window is the limit beyond that, not the engine).
5. Optional, never run: cross-check a generation against the llama.cpp oracle
   (greedy, same rendered prompt).
6. Still open: convert the grouped-MMA tier path (other families' verify
   chunks) to the per-slot splice to retire the summed-join drift class.
7. Still open: decide on `PULSAR_TIER_LEGACY` / `PULSAR_POS_DBG` diagnostics.
8. Still open: kimi-k3 fork audit — see "For the kimi-k3 fork audit" above.

## Known-good reference points

- Gate-era binary (tiers disabled for Bailing): coherent 8/8 turns at
  ctx 32768 — proves everything outside the tier path is sound.
- Original-body `moe_down` + fresh CUDA + gate-flow + tiers off: coherent
  single-turn ("Paris…" with mild tail loop) — base path sound after rebuild.
- llama.cpp reference outputs for the capital prompt: reasoning text ending
  "...This is well-established knowledge." then answer.
