# qwen3.8 (qwen4exp) port notes

Task: add the Qwen 3.8-Flash-Next model line (`qwen4exp`) to pulsar.

References:
- llama.cpp unsloth fork `src/models/qwen4exp.cpp` (+ `llama-memory-hybrid-idx`
  QSA block), oracle binary prebuilt in `~/llama.cpp-unsloth/build/bin`.
- Model: `~/models/qwen38flash/UD-Q3_K_XL/` (84 GB, 3-shard split GGUF).

## Shape (all verified from the real header)

- arch `qwen4exp`, 48 layers, n_embd 2560, vocab 248320, ctx 262144
- HYBRID residual: **hyper-connections** everywhere. hc=4 streams
  (`hyper_connection.count`), low-rank mixer (`low_rank`=320): grouped RMS
  per stream + full-width gamma `hc_{attn,ffn}_{norm,down,up}` + an F32
  inject `[hc_dim -> 4]` producing per-stream scatter weights; combine is
  `res[c] += blk * 2·sigmoid(inj[c]/hc)`. The final "norm" is a fourth HC
  mixer (`output_hc_{norm,down,up}`) — **no output_norm.weight exists**
  (engine's Model.output_norm became Option).
- GDN on every layer where `(il+1) % full_attention_interval(4) != 0`:
  key_dim 2048 (16 k-heads x 128), value_dim 6144 (**48 v-heads**),
  conv_dim 10240, conv K=4, alpha/beta are FULL [2560->48] projections,
  output gate after the per-v-head RMS norm is **SIGMOID** (qwen35 uses
  silu). Shared conv/l2/split/gdn_batch kernels reused unchanged.
- Full attention every 4th layer (12 layers): fused per-head `[q|gate]`
  projection (24 heads x 256 + gate), kv heads 2, partial neox rope rot 64
  sections [11,11,10,0] — text-only collapses to plain neox, same as qwen35.
  QSA indexer tensors exist (`indexer.q/k_proj` BF16 -> f32 at load):
  compress ratio 4/block, top_k 2048.
- MoE EVERY layer: 512 experts top-10 softmax-renormalized router, ff 640,
  shared expert with scalar SIGMOID gate (ffn_gate_inp_shexp).
- PLE n-gram hash embedding on layer 1 ONLY: ngram 3, 8 heads/ngram level
  (16 heads total), head_dim 160, hash multipliers ~2^45 (host int64), conv
  kernel 4 with DILATION = ngram = 3 over the wide stream.
- Tokenizer pre `qwen35` (already supported); eos <|im_end|> 248046;
  ple.eos_token_id 248044 used for hash-EOS only.

## Engine integration

- New `Family::Qwen38`; Shape fields added: `hc_low_rank`, `qsa_ratio`,
  `ple_*` block (incl `ple_head_dim` from embedding_length_per_layer_input).
- Loader arms in lib.rs: Attn::Qwen38(Box<Qwen38W>) with hc_attn/hc_ffn mixers,
  optional indexer projections (bf16 -> f32 via read_f16_as_f32), shexp_gate
  (f32 via upload_as_f32), PLE weights; `output_hc: Option<Qwen38Hc>` on Model
  replaces output_norm (now Option; all five consumers assert).
- Runtime in `real/qwen38.rs`: chunked T_MAX forward, token-major
  res_hc [T][hc][n_embd]; hc_norm / down→silu(self-gate trick) / up→sigmoid-
  gate-logits → xn *= sigmoid(logits) → stream mean via new kernels in
  `cuda/qwen38_kernels.inc`: `pulsar_qwen38_hc_norm` (grouped rms + gamma,
  gamma indexed `s % hc`), `hc_mean`, `hc_combine`, `ple_score`,
  `ple_apply`, `ple_conv` (taps `kern[c*K+j]`, zero-pad before sequence
  start), `residual_add`. silu implemented as self-gated sigmoid via
  existing `qwen35_sigmoid_gate`. GDN path reuses ALL qwen35 batch kernels;
  output gate swapped to sigmoid. PLE table reads ROWS straight from the
  shard files (`PleTable` VFile handle; 16 rows x 90 B IQ4NL per token)
  dequantized host-side through the ggml kvalues codebook — the 26.8 GiB
  table never loads into RAM/VRAM.
- Checkpoints: RecurrentCkpt::Qwen38 (S+conv+PLE-hist triples);
  interval/cap share the qwen35 lane.
- ChatML generation now opens the literal `<think>` block required by this
  GGUF. The generated `</think>` delimiter transitions to visible answer
  text and is hidden by the CLI formatter.

## Bugs found & fixed during bring-up

1. `matmul_q8_0` takes f32 activations and quantizes internally — several
   call sites were passing pre-q8_K buffers (produced 1e38 garbage).
2. HC projections operate on the flattened token row `[hc*n_embd]`, not on
   each stream independently. The original port passed `t*hc` rows with
   width `hc*n_embd`, causing strided reads and output corruption.
3. HC stream mean used the wrong stride and mixed different token rows.
4. HC combine must scale the injection by `1/hc` before `2*sigmoid`.
5. The final HC head uses `output_hc_norm.weight`; it is not weightless.
6. The FFN combine must consume the total shared+routed FFN output. The
   original path combined a never-written routed-output scratch buffer and
   silently discarded the shared expert.
7. The F32 injection weights go through `matmul_f32`, and the dsv4_moe
   contract reads `State.xq`/`State.normed` and writes `State.moe_out`.
8. Qwen38 down experts with `Q8_0` or `IQ4_NL` require q8_0 activations;
   gate/up experts remain q8_K. Added a dedicated packed q8_0 down path.
9. PLE uses the exact IQ4_NL codebook
   `[-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113]`.
10. PLE hash predecessors persist across forward calls, as well as across
    T_MAX prefill chunks; missing predecessors are EOS-padded.
11. The PLE table is intentionally disk/host resident and is marked as
    consumed by the loader before the unconsumed-tensor check.
12. PLE convolution uses `[K,C]` taps-fastest storage (`kern[c*K+j]`) and
    explicit zero padding before the sequence start.
13. QSA remains dense-only in v1 and refuses positions beyond
    `indexer_top_k + compress_ratio - 1` rather than silently changing the
    model's attention semantics.

## Status

Validated end-to-end in the release Pulsar binary:

- The 84 GB split GGUF loads and runs on the two-GPU workstation.
- `--ctx 32768 --chat` produces readable text with fp8 KV enabled.
- Deterministic raw prompt `The capital of France is` produces the first
  token `Paris` (token 11751), matching the llama.cpp oracle's top token.
- A short chat turn produces readable reasoning text; no `probe:` diagnostics
  are emitted in normal operation.
- The loader no longer warns about `per_layer_token_embd.weight`.

Qwen38 expert tiers are enabled by default on a spare CUDA device. Their
Q8_0/IQ4_NL down-expert path now uses the audited per-slot splice and matches
the single-device host-store path in deterministic 16-token A/B generation.
`PULSAR_TIERS=off` remains available for the single-device comparison.
The Qwen38 default host expert-cache budget is 64 GiB, capped against
`MemAvailable` with a 6 GiB reserve; `PULSAR_CACHE_GB` still overrides it.
On the reference dual-GPU run, the secondary card held 5.6 GiB of resident
expert triples and the host cache reported 35.0/64.0 GiB after warm start.

The dense-only QSA boundary remains a deliberate v1 limitation:
`indexer_top_k + compress_ratio - 1 = 2051` visible cells. When reached,
chat reports a clean session termination rather than a CUDA core dump: the
implementation rejects positions beyond the provably dense window and does
not attempt an invalid sparse fallback. Block-sparse QSA (block pooling,
scored selection, and masked attention) is the next feature needed for
longer contexts.
