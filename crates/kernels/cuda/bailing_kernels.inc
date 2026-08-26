/* Ling 3.0 Flash (bailingmoe3) kernels.
 *
 * Included by pulsar_kernels.cu after k3_kernels.inc, which supplies the
 * KDA pieces Bailing shares with K3: the depthwise conv step, l2-norm, the
 * elementwise sigmoid gate, and the per-channel-decay delta-rule step
 * (k3_kda_coeffs / k3_kda_step). Bailing's KDA differs from K3 only in the
 * decay projection: ssm_f_a is a DIRECT hidden->d_inner projection (no
 * f_a->f_b rank bottleneck), and ssm_a stores +exp(A_log) which the loader
 * negates so the shared coeff kernel is reused unchanged.
 *
 * The one genuinely new piece is the noaux_tc grouped router: experts are
 * split into n_expert_groups groups, the top n_group_used groups by summed
 * top-2 score are kept, then top-k selection runs inside the survivors.
 * Reference: llama.cpp build_moe_ffn (n_expert_groups > 1 branch) and the
 * DeepSeek-V3 modeling_deepseek.py noaux_tc path.
 */

/* Grouped sigmoid router, one block per token. 256 threads compute the
 * sigmoid+bias scores; thread 0 then does the group selection and top-k
 * (k=8 over 512 experts is ~5k comparisons, negligible next to the MoE).
 *
 *   score[e] = sigmoid(logits[e]) + bias[e]
 *   group_score[g] = sum of the top-2 scores in group g
 *   keep the top n_group_used groups, mask the rest to -inf
 *   select top k_used experts from the masked scores
 *   weights = sigmoid(logits[sel]) / sum * weight_scale   (unbiased, normed)
 */
__global__ static void bailing_router_select_kernel(
        int32_t *selected,         /* [n_tok][k_used] */
        float *weights,            /* [n_tok][k_used] */
        const float *logits,       /* [n_tok][n_expert] */
        const float *bias,         /* [n_expert] */
        uint32_t n_expert,
        uint32_t k_used,
        uint32_t n_groups,
        uint32_t n_group_used,
        float weight_scale,
        uint32_t n_tok) {
    const uint32_t token = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t n_exp_per_group = n_expert / n_groups;

    __shared__ float scores[512];
    __shared__ float group_score[64];
    __shared__ float probs[512];

    const float *log = logits + (uint64_t)token * n_expert;
    for (uint32_t e = tid; e < n_expert; e += blockDim.x) {
        const float raw = log[e];
        const float p = 1.0f / (1.0f + expf(-raw));
        probs[e] = p;
        scores[e] = p + bias[e];
    }
    __syncthreads();

    /* top-2 sum per group */
    for (uint32_t g = tid; g < n_groups; g += blockDim.x) {
        float m1 = -INFINITY, m2 = -INFINITY;
        for (uint32_t i = 0; i < n_exp_per_group; i++) {
            const float s = scores[g * n_exp_per_group + i];
            if (s > m1) { m2 = m1; m1 = s; }
            else if (s > m2) { m2 = s; }
        }
        group_score[g] = m1 + m2;
    }
    __syncthreads();

    if (tid == 0) {
        /* select top n_group_used groups (selection sort, tiny) */
        uint32_t chosen[64];
        for (uint32_t g = 0; g < n_groups; g++) chosen[g] = g;
        for (uint32_t i = 0; i < n_group_used; i++) {
            uint32_t best = i;
            for (uint32_t j = i + 1; j < n_groups; j++) {
                if (group_score[chosen[j]] > group_score[chosen[best]]) best = j;
            }
            const uint32_t t = chosen[i]; chosen[i] = chosen[best]; chosen[best] = t;
        }
        /* mask non-chosen groups */
        for (uint32_t g = 0; g < n_groups; g++) {
            bool keep = false;
            for (uint32_t i = 0; i < n_group_used; i++) {
                if (chosen[i] == g) { keep = true; break; }
            }
            if (!keep) {
                for (uint32_t i = 0; i < n_exp_per_group; i++) {
                    scores[g * n_exp_per_group + i] = -INFINITY;
                }
            }
        }
        /* top-k over the masked scores */
        int32_t sel[64];
        float wsum = 0.0f;
        for (uint32_t k = 0; k < k_used; k++) {
            float best = -INFINITY;
            uint32_t best_e = 0;
            for (uint32_t e = 0; e < n_expert; e++) {
                if (scores[e] > best) { best = scores[e]; best_e = e; }
            }
            scores[best_e] = -INFINITY;
            sel[k] = (int32_t)best_e;
            wsum += probs[best_e];
        }
        wsum = fmaxf(wsum, 6.103515625e-5f);
        for (uint32_t k = 0; k < k_used; k++) {
            selected[(uint64_t)token * k_used + k] = sel[k];
            weights[(uint64_t)token * k_used + k] = probs[sel[k]] / wsum * weight_scale;
        }
    }
}

extern "C" int pulsar_bailing_router_select(
        void *selected_dev, void *weights_dev,
        const void *logits_dev, const void *bias_dev,
        uint32_t n_expert, uint32_t k_used,
        uint32_t n_groups, uint32_t n_group_used,
        float weight_scale, uint32_t n_tok) {
    if (!selected_dev || !weights_dev || !logits_dev || !bias_dev ||
        n_expert == 0 || k_used == 0 || n_groups == 0 || n_group_used == 0 ||
        n_tok == 0 || n_expert % n_groups != 0 || n_expert > 512u) {
        return 0;
    }
    bailing_router_select_kernel<<<n_tok, 256>>>(
            (int32_t *)selected_dev, (float *)weights_dev,
            (const float *)logits_dev, (const float *)bias_dev,
            n_expert, k_used, n_groups, n_group_used, weight_scale, n_tok);
    return cuda_ok(cudaGetLastError(), "bailing router select launch");
}
