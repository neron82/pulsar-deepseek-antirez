# DeepSeek-V4-Flash correctness and reasoning repairs

This note records a correctness repair to the optimized DeepSeek-V4-Flash
path, plus the related tokenizer/UI repair that restores visible reasoning.
It is deliberately a correctness record, not a performance claim: the defects
below could make a fast run generate corrupted or misleading output.

## Scope

Validated models:

- `DeepSeek-V4-Flash-0731-UD-Q8_K_XL`
- `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731`

Validated configuration included chat mode, `--ctx 131268`, temperature 1,
top-p 1, min-p 0.05, repeat penalty 1.1, and seed 42. CUDA builds used
`PULSAR_CUDA_ARCH=75,86`.

## Attention-kernel repairs

### Block-wide visible-row denominator

The ballot-compaction path builds a per-warp visibility mask for each
128-row attention chunk. The compacted rank needs each warp's prefix count,
but the softmax/value loop needs the **block-wide** visible-row total.

The old kernel reused the warp-local prefix for `n_vis`. The four warps could
therefore process different counts (for example 32, 64, 96, and 128 rows) for
the same head/chunk. That corrupts the attention denominator and value sum.

The repaired kernel sums all four `warp_vis` ballots after the shared state is
ready. The prefix is still used only for each row's compacted rank.

### Parity-buffer lifetime race

The optimized kernel parity-double-buffers some shared arrays. That protects
chunk `k` from a writer in `k+1`, but it does not protect `k` from a fast warp
reaching `k+2`, which reuses the same parity while another warp is still
reading `k`.

An end-of-chunk `__syncthreads()` now closes the block-wide read phase before
any warp can write the next chunk's shared state. This is a correctness barrier,
not an attempt to recover the old barrier count blindly.

### Per-thread default-stream ordering

CUDA kernels are built with `--default-stream=per-thread`. The asynchronous
copy path previously recorded/waited on legacy NULL stream, which has no
ordering edge to that per-thread compute stream. A compute kernel could read a
partially copied expert slab.

Copy-stream gates and waits now use `STREAM_PER_THREAD`, matching the actual
compute stream.

## Sampling repair

Repeat penalty now follows llama.cpp semantics:

- apply it once per distinct token in the recent generated window;
- divide positive logits by the penalty;
- multiply negative logits by the penalty.

Dividing a negative logit makes it less negative and can accidentally promote
a repeated token. Focused sampler tests cover both negative logits and repeated
occurrences of the same token.

## Reasoning/template repair

DeepSeek V4 Flash is a reasoning model, and its default assistant prefix opens
`<think>`. The prior repair path accidentally defaulted the markers to
thinking-off; CLI and API callers that omitted an override therefore never
received reasoning.

The tokenizer now:

1. keeps the native `<think>` / `</think>` controls as the preferred path;
2. preserves a fallback for alternate GGUF exports that encode the same
   controls as ` thinking` / ` response` or byte-BPE `Ġthinking` /
   `Ġresponse`;
3. defaults DeepSeek V4 Flash reasoning on, while honoring
   `enable_thinking: false`;
4. closes replayed assistant history with the model EOS and does not replay
   prior private reasoning.

The server stream splitter recognizes the configured DeepSeek closing token by
ID, so byte-BPE fallback exports can transition from `reasoning_content` to
ordinary `content` correctly.

The bundled WebUI now always sends its current `enable_thinking` checkbox
state. Previously it sent an explicit value only for false, allowing a checked
control to disagree with the server-side default.

## Verification evidence

### Deterministic tests

- CUDA kernel self-tests: 11 passed, including the dsv4 self-test; the dsv4
  test was repeated 40 additional times after the barrier repair.
- Tokenizer tests: 12 unit tests and 2 parity tests passed.
- Serve tests: 3 passed.
- Engine sampler tests: 2 new focused tests passed; the engine suite was green.
- `git diff --check` was clean.

### Real-model checks

- UD-Q8 CLI: coherent multi-turn technical output after the attention repairs;
  the thinking-enabled run emitted `</think>` and then a normal final answer.
- UD-Q8 OpenAI-compatible SSE: emitted `reasoning_content` with
  `chat_template_kwargs.enable_thinking=true`.
- IQ2XXS CLI: loaded and opened the native `<think>` control token in the
  real chat prompt.

The real-model logs were retained under `/tmp/pulsar-ds4flash-*-thinking-*`
and `/tmp/pulsar-ds4flash-fixed-attention-*` during validation.

## Operational note

Any already-running `pulsar-serve` instance must be restarted after building
these sources, otherwise it will continue serving the old binary and its old
embedded WebUI.
