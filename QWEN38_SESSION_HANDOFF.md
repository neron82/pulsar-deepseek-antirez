# Qwen 3.8 Flash Next — session handoff (2026-08-28)

Status: **All tracked items done** (second pass 2026-08-28: redundant
PLE/embed syncs removed + on-device embed broadcast — bit-exact-verified
by greedy A/B, see §5.1–5.2). All work uncommitted on `main`.
The +32 GB RAM upgrade landed (~120 GB total); host expert-cache
coverage is unmeasured — see §5.4.

## 1. Task

Investigate and fix Qwen 3.8-Flash-Next (`qwen4exp`) performance in pulsar,
using the llama.cpp port at `~/llama.cpp-qwen4` (branch with MTP/nextn support,
`src/models/qwen4exp.cpp`) as the reference. The implementation was working
but slow; also fix the active compile warnings.

## 2. Environment / assets

- GPUs: RTX 3090 (24 GB) + RTX 3060 Ti (sm_86 both) → build with
  `PULSAR_CUDA_ARCH=86 cargo build --release` (the default arch list
  `61,75,80,86,89` FAILS on CUDA 13 per the perf ledger).
- The 3090 is normally occupied by the assistant's own inference engine, so
  most agent-side testing ran on the 3060 Ti only (expert cache too small
  there to be representative — treat those numbers as smoke-grade).
- Model: `~/models/qwen38flash/UD-Q3_K_XL/`
  - trunk: `Qwen3.8-Flash-Next-UD-Q3_K_XL-0000{1,2,3}-of-00003.gguf` (84 GB,
    48 layers, n_embd 2560, vocab 248320, ctx 262144)
  - MTP sidecars: `Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf` (~2.6 GB),
    `mtp-Qwen3.8-Flash-Next-BF16.gguf` (~7.7 GB), plus a Q8_0 variant
  - `.warm` file (warm start), `log3.txt` / `log3_filtered.txt` = llama.cpp
    oracle runs
- Oracle numbers (llama.cpp-qwen4, dual GPU, from `log3.txt`):
  **12.75 tok/s eval, MTP draft acceptance 0.54 (1746/3210), mean draft len
  2.63.** That is the target pulsar should approach on the same box.

## 3. Findings (what was limiting performance)

1. **Missing MTP (nextn) speculative head — the big one.** "Next" in
   Qwen 3.8 Flash Next *is* the MTP block (blk.48, `nextn.*` tensors): own
   `embed_tokens`, `eh_proj`, `hnorm`, `ennorm`, `shared_head_head/norm`.
   Pulsar ran only the 48-layer trunk; the llama.cpp oracle runs MTP
   (acceptance 0.54 ≈ 1.5 tokens/step free). ~2–2.5× of the oracle speed
   was unclaimed.
2. **MTP expert-read bug:** the MTP block's expert tensors live in the
   *sidecar* file (abs offsets 1.4–3.2 GB into it), but the stream fetcher
   only maps the trunk shards, so the MTP MoE (`dsv4_moe`) read the wrong
   region → all-NaN. Fixed via sidecar-aware loading/fetching (see §4).
3. **Redundant decode-time syncs** in the PLE (layer-1 host IQ4NL hash
   gather) and embedding eval path — removed (second pass, §4).
4. **Compile warnings** — all fixed (see §4).

## 4. What was changed (uncommitted)

`git status`: modified `engine/src/{lib.rs (±877), real/dsv4.rs (±174),
real/{bailing,k3,qwen35}.rs, bin/pulsar-cli.rs}`, `kernels/{build.rs,
cuda/pulsar_kernels.cu, src/lib.rs}`, `quant/src/iq.rs`,
`tokenizer/src/lib.rs`; new: `engine/src/real/qwen38.rs` (+ its kernels
`kernels/cuda/qwen38_kernels.inc`), `gguf/src/bin/dump_tensors.rs`,
`docs/qwen38-port-notes.md`.

- **MTP port (qwen38.rs + lib.rs):** sidecar GGUF loading for blk.48 +
  nextn tensors + its own embedding/head; `mtp_eval_layer` arm for the
  draft block (HC layer graph, own KV/experts); `mtp_body`/`mtp_draft`
  (wide hnorm, draft embedding, HC head — note: no `output_norm`, the head
  is a fourth HC mixer); trunk `res_hc` capture into `mtp_hidden` with
  prefill fill; verify-loop recurrent-state handling with `Qwen38Rt`
  snapshot/restore so draft tokens that get rejected roll state back.
- **Warnings fixed:** `quant/src/iq.rs` unused mut; `dsv4.rs` unused
  inits/mut/fields; `engine/src/lib.rs` unused vars, dead fields,
  visibility; `qwen35.rs` unused field + a `dbg`-style method.
  Verified behavior-preserving via greedy A/B (§7).
- **Redundant sync removal + on-device embed broadcast (second pass,
  2026-08-28):** `eval_ple` dropped 6 of 9 `cudaDeviceSynchronize`
  calls (kept: before the gated→ple_val_v D2D, before the step-5 D2Ds,
  and the caller-side trailing sync that drains for the next token's
  H2D); the PLE row gather uses one reusable 90B read buffer per chunk
  instead of 16 Vec allocations per token. The chunk-start embed
  broadcast (D2H + host copy + H2D) is now the `pulsar_qwen38_broadcast_hc`
  PTDS kernel (`qwen38_kernels.inc` + FFI in `kernels/src/lib.rs`), used
  at the three staging sites: chunk start (per-chunk head sync now
  drains the prior chunk's st.tok readers + prior call's MTP re-anchor
  D2D), the MTP draft, and the MTP prefill fill.
  **Verified bit-exact:** greedy A/B, 52-token aligned context via
  `--tokens` + `-n 32`, baseline binary vs patched: trunk IDENTICAL and
  PULSAR_MTP=1 IDENTICAL (3060 Ti, smoke-grade).

## 5. Left to do

1. **[DONE]** Remove redundant PLE/embed eval syncs + batch PLE host reads.
2. **[DONE]** Rebuild + greedy A/B (trunk and MTP legs, bit-exact).
3. **User-run 3090 speed A/B** (the only remaining validation):
   `scripts/ab-3090.sh` — stops-on-you: it needs the llama.cpp engine on
   the 3090 stopped (GPU 0). Runs baseline vs current, trunk + MTP, 2
   passes (pass 2 = warm numbers), 256 tokens, ctx 4096; raw logs in
   `/tmp/ab3090/`. Compare pass-2 tok/s against the 12.75 tok/s oracle.
4. **[OPEN, low priority]** Host expert cache after the RAM upgrade: the
   64 GiB Qwen38 cap was left unchanged; the smoke runs showed the tested
   workload hitting the host cache 100% of the time after the VRAM tier.
   Exact expert tensor bytes are not yet measured (bit-count estimate
   ≈ 60 GiB, 512 × 48 × 3 × 2560 × 640 weights, Q3_K_XL). If the 3090
   speed run looks SSD-bound, measure real coverage (dump_tensors or
   serve stats) before raising the cap / `PULSAR_CACHE_GB`.
5. **Consider committing** in logical chunks (qwen38 port / MTP /
   warnings / sync-removal) once the 3090 numbers are in.

## 6. Commands

```sh
# build (both cards are sm_86; the default arch list fails on CUDA 13)
PULSAR_CUDA_ARCH=86 cargo build --release

# CUDA device indexes: 0 = 3090 (the assistant's engine sits there),
# 1 = 3060 Ti. nvidia-smi disagrees (its 0 is the 3060 Ti).

# one-shot / A/B (token-aligned for bit-exact comparison).
# -m must be SHARD 1 of the split trunk, not the directory:
M=~/models/qwen38flash/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf
SC=~/models/qwen38flash/UD-Q3_K_XL/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
target/release/pulsar-cli -m "$M" --tokens <ids> -n 32 --ctx 2048
target/release/pulsar-cli -m "$M" -p "text" -n 32
# MTP: PULSAR_MTP=1 + explicit sidecar (auto-discovery picks the BF16
# sidecar, whose expert tensors the loader rejects):
PULSAR_MTP=1 PULSAR_MTP_SIDECAR="$SC" target/release/pulsar-cli -m "$M" -p "text" -n 32

# serve (OpenAI-compatible, webui at /)
PULSAR_MTP=1 PULSAR_CUDA_ARCH=86 target/release/pulsar-serve \
  -m ~/models/qwen38flash/UD-Q3_K_XL/ --port 11435 --ctx 8192
# env: PULSAR_MTP=1 (spec decode on) · PULSAR_MTP_SIDECAR=<gguf>
#      PULSAR_CACHE_GB=<n> (expert cache) · PULSAR_KV=auto|f32|fp8
#      PULSAR_CPU=1 (AVX2 CPU expert lane)

# 3090 speed A/B (needs the llama.cpp engine stopped on GPU 0)
scripts/ab-3090.sh
```

## 7. Invariants / gotchas

- **QSA dense window (deliberate v1 limit):** dense attention only while
  `indexer_top_k + compress_ratio - 1 = 2051` visible cells remain; beyond
  that, chat gets a clean session termination, not a CUDA core dump.
  Block-sparse QSA is the next feature for longer contexts.
- Greedy A/B methodology: feed exact ids via `--tokens` (aligned with the
  pre-patch binary's `--dump-tokens`-style output) — sampling A/Bs drift.
- 3060 Ti runs: small expert cache ⇒ acceptance/speed are smoke-grade only;
  real evaluation needs the 3090 + tier run.
- The trunk shards are a SPLIT file (virtual = concatenation); the sidecar
  is a SEPARATE file — any fetcher routing on abs offsets must know which
  file owns the offset (that was the NaN bug).
- **MTP draft acceptance 0/32 on the Q8_0 sidecar** (3060 Ti smoke run,
  both baseline and patched — pre-existing, not a regression). The oracle
  hit 0.54 with its own quant; either the Q8_0 sidecar degrades the
  drafter or the draft path is subtly off — check with
  `PULSAR_MTP_DEBUG=1` (nan counts per stage) before trusting MTP speed
  numbers.

## 8. References

- `docs/qwen38-port-notes.md` — arch shapes, verified from the real header
- `docs/perf-optimization-ledger.md` — ds4-era perf methodology & wins
- `TIER_DEBUG_SESSION.md` — cross-device stream-ordering bug history
  (NULL stream vs PTDS) — still relevant if tier path changes
- `~/llama.cpp-qwen4` — reference port: `src/models/qwen4exp.cpp` (+MTP),
  llama-arch/model/graph changes; prebuilt oracle binaries there
- `~/models/qwen38flash/UD-Q3_K_XL/log3.txt` — oracle run log (12.75 tok/s)
