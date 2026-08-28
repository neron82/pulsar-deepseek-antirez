/* Qwen3.8 (qwen4exp) HC-mixer + PLE kernels.
 *
 * Included by pulsar_kernels.cu after qwen35_kernels.inc (reuses its
 * conv/split/l2/gdn kernels). The hyper-connection residual replaces
 * every layer norm: [n_tok][hc][n_embd] streams carry grouped RMS
 * (per-stream reduction, full-width gamma), a low-rank down/silu/up
 * gate, and a per-stream scatter weight from an inject projection.
 *
 * Reference: llama.cpp (unsloth fork) src/models/qwen4exp.cpp
 * build_hc_mix / build_hc_combine / build_ple.
 */

/* Grouped RMS over each residual stream + gamma scale.
 *
 * x: [rows][hc][n_embd] f32; out may alias x. w: gamma [hc*n_embd]
 * (NULL = weightless). One BLOCK per stream row (grid rows*hc), all
 * threads of the block reduce that one stream: no cross-warp reads,
 * no divergent barriers. eps follows ggml rms_norm semantics.
 */
__global__ static void qwen38_hc_norm_kernel(
        float *out, const float *x, const float *w,
        uint32_t n_embd, uint32_t hc, float eps) {
    const uint64_t s = blockIdx.x;      /* stream slot = row*hc + c */
    const float *xs = x + s * n_embd;
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        acc += xs[i] * xs[i];
    }
    /* warp reduce then one shared round across the block's warps */
    __shared__ float warp_sums[32];
    for (uint32_t o = 16u; o > 0; o >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    }
    const uint32_t warp = threadIdx.x / 32;
    if ((threadIdx.x & 31u) == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        /* threads < blockDim.x/32 hold partials in lane order */
        float v = (threadIdx.x < blockDim.x / 32u &&
                   threadIdx.x < 32u)
                      ? warp_sums[threadIdx.x]
                      : 0.0f;
        for (uint32_t o = 16u; o > 0; o >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, o);
        }
        if (threadIdx.x == 0) warp_sums[0] = v;
    }
    __syncthreads();

    const float inv = rsqrtf(warp_sums[0] / (float)n_embd + eps);
    float *os = out + s * n_embd;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float g = w ? w[((uint64_t)(s % hc)) * n_embd + i] : 1.0f;
        os[i] = xs[i] * inv * g;
    }
}

extern "C" int pulsar_qwen38_hc_norm(
        void *out_dev, const void *x_dev, const void *w_dev,
        uint32_t rows, uint32_t n_embd, uint32_t hc, float eps) {
    if (!out_dev || !x_dev || rows == 0 || n_embd == 0 || hc == 0) return 0;
    qwen38_hc_norm_kernel<<<rows * hc, 128>>>(
            (float *)out_dev, (const float *)x_dev, (const float *)w_dev,
            n_embd, hc, eps);
    return cuda_ok(cudaGetLastError(), "qwen38 hc norm launch");
}

/* Stream-mean collapse: mixed[t][e] = mean_c xn[t][c][e].
 * One thread per output element; hc is tiny so the read fan-out is fine.
 */
__global__ static void qwen38_hc_mean_kernel(
        float *out, const float *xn, uint32_t n_tok,
        uint32_t n_embd, uint32_t hc) {
    const uint64_t idx =
            (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tok * n_embd;
    if (idx >= total) return;
    const uint64_t token = idx / n_embd;
    const uint32_t e = (uint32_t)(idx % n_embd);
    float acc = 0.0f;
    for (uint32_t c = 0; c < hc; c++) {
        /* xn is token-major: [token][stream][embedding]. */
        acc += xn[((token * hc + c) * n_embd) + e];
    }
    out[idx] = acc / (float)hc;
}

extern "C" int pulsar_qwen38_hc_mean(
        void *out_dev, const void *xn_dev, uint32_t n_tok,
        uint32_t n_embd, uint32_t hc) {
    if (!out_dev || !xn_dev || n_tok == 0 || n_embd == 0 || hc == 0)
        return 0;
    const uint64_t total = (uint64_t)n_tok * n_embd;
    qwen38_hc_mean_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)out_dev, (const float *)xn_dev, n_tok, n_embd, hc);
    return cuda_ok(cudaGetLastError(), "qwen38 hc mean launch");
}

/* Embed broadcast: out[t][c][e] = src[t][e] - replicate each n_embd-wide
 * embedding row across all hc streams (token-major [t][c][e]). Replaces
 * the D2H + host copy + H2D roundtrip at chunk/draft/fill starts.
 * One thread per float4; n_embd must be a multiple of 4.
 */
__global__ static void qwen38_broadcast_hc_kernel(
        float4 *out, const float4 *src, uint32_t n_tok,
        uint32_t n4, uint32_t hc) {
    const uint64_t idx =
            (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tok * n4 * hc;
    if (idx >= total) return;
    const uint32_t k = (uint32_t)(idx % n4);
    const uint64_t token = idx / n4 / hc;
    out[idx] = src[token * n4 + k];
}

extern "C" int pulsar_qwen38_broadcast_hc(
        void *out_dev, const void *src_dev, uint32_t n_tok,
        uint32_t n_embd, uint32_t hc) {
    if (!out_dev || !src_dev || n_tok == 0 || hc == 0) return 0;
    if (n_embd % 4 != 0) return -1;
    const uint32_t n4 = n_embd / 4;
    const uint64_t total = (uint64_t)n_tok * n4 * hc;
    qwen38_broadcast_hc_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float4 *)out_dev, (const float4 *)src_dev, n_tok, n4, hc);
    return cuda_ok(cudaGetLastError(), "qwen38 broadcast hc launch");
}

/* HC combine: res_hc[s][e] += block_out[row_b(s)][e] * 2*sigmoid(inj[s]/hc).
 * s = row*hc + c, row_b(s) = s / hc. One thread per (stream, element).
 * The 1/hc scaling is part of the reference graph's scatter weight.
 */
__global__ static void qwen38_hc_combine_kernel(
        float *res, const float *blk_out, const float *inj,
        uint32_t n_tok, uint32_t n_embd, uint32_t hc) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tok) return;
    const uint32_t c = blockIdx.y;
    float *dst = res + ((uint64_t)t * hc + c) * n_embd;
    const float w = 2.0f / (1.0f + expf(-inj[(uint64_t)t * hc + c] / (float)hc));
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) {
        dst[e] += blk_out[(uint64_t)t * n_embd + e] * w;
    }
}

extern "C" int pulsar_qwen38_hc_combine(
        void *res_dev, const void *blk_out_dev, const void *inj_dev,
        uint32_t n_tok, uint32_t n_embd, uint32_t hc) {
    if (!res_dev || !blk_out_dev || !inj_dev || n_tok == 0 || n_embd == 0 ||
        hc == 0)
        return 0;
    dim3 grid(n_tok, hc);
    qwen38_hc_combine_kernel<<<grid, 256>>>(
            (float *)res_dev, (const float *)blk_out_dev,
            (const float *)inj_dev, n_tok, n_embd, hc);
    return cuda_ok(cudaGetLastError(), "qwen38 hc combine launch");
}

/* PLE gate on the host boundary: gate[s] = sigmoid(sgn*sqrt(|s|)) where
 * |s| clamped at >=1e-6 happens here. One thread per token. Applied via
 * the generic qwen35_row_sigmoid_scale afterwards. */
__global__ static void qwen38_ple_signed_mag_kernel(
        float *gate, const float *score_sum, uint32_t n_rows) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    const float s = score_sum[i];
    const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
    gate[i] = copysignf(mag, s);
}

extern "C" int pulsar_qwen38_ple_gate(
        void *gate_dev, const void *score_dev, uint32_t n_rows) {
    if (!gate_dev || !score_dev || n_rows == 0) return 0;
    qwen38_ple_signed_mag_kernel<<<(n_rows + 255u) / 256u, 256>>>(
            (float *)gate_dev, (const float *)score_dev, n_rows);
    return cuda_ok(cudaGetLastError(), "qwen38 ple gate launch");
}

/* PLE dilated depthwise conv + silu over the hidden streams:
 *   out[c] = silu(sum_k kern[k*n_dim+c] * x[c][hist - (K-1-k)*dil])
 * x: [padded][dim] f32 with history prepended (history rows FIRST).
 * Output layout [n_tok][dim]. One block per channel. */
__global__ static void qwen38_ple_conv_kernel(
        float *out, const float *x, const float *kern,
        uint32_t n_tok, uint32_t dim, uint32_t padded, uint32_t k,
        uint32_t dil) {
    const uint32_t c = blockIdx.x;
    if (c >= dim) return;
    extern __shared__ float sh[]; /* [k] tap weights for this channel */
    /* gguf ple_conv1d ne = [K, C]: taps fastest, so element (j, c) sits at
     * c*K + j - channel-major rows of K taps */
    for (uint32_t j = threadIdx.x; j < k; j += blockDim.x) {
        sh[j] = kern[(uint64_t)c * k + j];
    }
    __syncthreads();
    const uint32_t hist = padded - n_tok; /* history rows prepended */
    for (uint32_t t = threadIdx.x; t < n_tok; t += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < k; j++) {
            /* tap j reads (K-1-j)*dil positions before token t's row;
             * reads BEFORE the sequence start are zero (nn.Conv1d
             * zero-padding at a fresh sequence) */
            const int64_t back = (int64_t)(k - 1 - j) * dil;
            const int64_t src = (int64_t)(hist + t) - back;
            if (src < 0) continue;
            acc += sh[j] * x[(uint64_t)src * dim + c];
        }
        out[(uint64_t)t * dim + c] = acc / (1.0f + expf(-acc));
    }
}

extern "C" int pulsar_qwen38_ple_conv(
        void *out_dev, const void *x_dev, const void *kern_dev,
        uint32_t n_tok, uint32_t dim, uint32_t padded, uint32_t k,
        uint32_t dil) {
    if (!out_dev || !x_dev || !kern_dev || n_tok == 0 || dim == 0 ||
        padded < n_tok || k == 0 || dil == 0)
        return 0;
    qwen38_ple_conv_kernel<<<dim, 256, k * sizeof(float)>>>(
            (float *)out_dev, (const float *)x_dev, (const float *)kern_dev,
            n_tok, dim, padded, k, dil);
    return cuda_ok(cudaGetLastError(), "qwen38 ple conv launch");
}

/* PLE signed-score reduce + magnitude gate:
 *   s[t][c]  = sum_e key[t*hc+c][e] * query[t*hc+c][e]
 *   g[t][c]  = sigmoid(sgn(s)*sqrt(max(|s|/sqrt(scale_div),1e-6)))
 * One block per (token, stream); all threads reduce that stream.
 * key/query are [T*hc][n_embd]. scale multiplies AFTER the sum
 * (reference scales s by 1/sqrt(n_embd) before the magnitude). */
__global__ static void qwen38_ple_score_kernel(
        float *gate, const float *key, const float *query,
        uint32_t n_tok, uint32_t n_embd, uint32_t hc, float scale) {
    const uint32_t row = blockIdx.x;   /* t*hc + c */
    const float *ks = key + (uint64_t)row * n_embd;
    const float *qs = query + (uint64_t)row * n_embd;
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        acc += ks[i] * qs[i];
    }
    __shared__ float warp_sums[32];
    for (uint32_t o = 16u; o > 0; o >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    }
    const uint32_t warp = threadIdx.x / 32;
    if ((threadIdx.x & 31u) == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (threadIdx.x < blockDim.x / 32u && threadIdx.x < 32u)
                      ? warp_sums[threadIdx.x]
                      : 0.0f;
        for (uint32_t o = 16u; o > 0; o >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, o);
        }
        if (threadIdx.x == 0) warp_sums[0] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float s = warp_sums[0] * scale;
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        /* NOTE: copysign BEFORE sigmoid, matching ggml_sgn * ggml_sqrt */
        gate[row] = copysignf(mag, s);
    }
}

extern "C" int pulsar_qwen38_ple_score(
        void *gate_dev, const void *key_dev, const void *query_dev,
        uint32_t n_tok, uint32_t n_embd, uint32_t hc, float scale) {
    if (!gate_dev || !key_dev || !query_dev || n_tok == 0 || n_embd == 0 ||
        hc == 0)
        return 0;
    qwen38_ple_score_kernel<<<n_tok * hc, 128>>>(
            (float *)gate_dev, (const float *)key_dev,
            (const float *)query_dev, n_tok, n_embd, hc, scale);
    return cuda_ok(cudaGetLastError(), "qwen38 ple score launch");
}

/* PLE value broadcast + gate application:
 *   out[t*hc+c][e] = value[t][e] * gate[t*hc+c]
 * value is [T][n_embd]; every stream sees the SAME value scaled by its
 * own gate scalar (the reference's repeat_4d then mul). */
__global__ static void qwen38_ple_apply_kernel(
        float *out, const float *value, const float *gate,
        uint32_t n_tok, uint32_t n_embd, uint32_t hc) {
    const uint32_t t = blockIdx.x;
    const uint32_t c = blockIdx.y;
    const float g = 1.0f / (1.0f + expf(-gate[(uint64_t)t * hc + c]));
    const float *v = value + (uint64_t)t * n_embd;
    float *o = out + ((uint64_t)t * hc + c) * n_embd;
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) {
        o[e] = v[e] * g;
    }
}

extern "C" int pulsar_qwen38_ple_apply(
        void *out_dev, const void *value_dev, const void *gate_dev,
        uint32_t n_tok, uint32_t n_embd, uint32_t hc) {
    if (!out_dev || !value_dev || !gate_dev || n_tok == 0 || n_embd == 0 ||
        hc == 0)
        return 0;
    dim3 grid(n_tok, hc);
    qwen38_ple_apply_kernel<<<grid, 256>>>(
            (float *)out_dev, (const float *)value_dev,
            (const float *)gate_dev, n_tok, n_embd, hc);
    return cuda_ok(cudaGetLastError(), "qwen38 ple apply launch");
}

/* PLE/HC residual add: res[t][c][e] += tile[t][c*dim+e].
 * Adds a [T][hc*dim] tile across all streams of the wide residual.
 * dim here is n_embd (tile layout token-major, stream-fastest). */
__global__ static void qwen38_residual_add_kernel(
        float *res, const float *tile, uint32_t n_tok,
        uint32_t n_embd, uint32_t hc) {
    const uint32_t t = blockIdx.x;
    const uint32_t c = blockIdx.y;
    float *r = res + ((uint64_t)t * hc + c) * n_embd;
    const float *v = tile + ((uint64_t)t * hc + c) * n_embd;
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) {
        r[e] += v[e];
    }
}

extern "C" int pulsar_qwen38_residual_add(
        void *res_dev, const void *tile_dev, uint32_t n_tok,
        uint32_t n_embd, uint32_t hc) {
    if (!res_dev || !tile_dev || n_tok == 0 || n_embd == 0 || hc == 0)
        return 0;
    dim3 grid(n_tok, hc);
    qwen38_residual_add_kernel<<<grid, 256>>>(
            (float *)res_dev, (const float *)tile_dev, n_tok, n_embd, hc);
    return cuda_ok(cudaGetLastError(), "qwen38 residual add launch");
}
