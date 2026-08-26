//! Ling 3.0 Flash (bailingmoe3) forward path.
//!
//! Reference: llama.cpp PR #26608 src/models/bailingmoe3.cpp. Each block is
//! 5 KDA layers then 1 gated MLA layer (35 KDA + 7 MLA over 42 layers), a
//! 512-expert sigmoid router with noaux_tc group selection, and a per-layer
//! SwiGLU clamp. KDA is K3's Gated DeltaNet but with a DIRECT hidden->d_inner
//! decay projection (ssm_f_a, no f_a->f_b bottleneck); MLA applies real RoPE
//! and a per-head sigmoid output gate.
//!
//! Decode-shaped: one token per pass, prefill loops tokens. The KDA
//! recurrence (conv window + delta state) is sequential, so a batched pass
//! would have to loop inside the kernels anyway.

use super::{Attn, Ffn, LayerW, MatW, Model, Result, State};
use kernels::DeviceBuf;

/// Matmul over either weight encoding (mirrors k3::matw).
fn matw(
    out: &mut DeviceBuf,
    w: &MatW,
    x: &DeviceBuf,
    xq: &DeviceBuf,
    in_dim: u32,
    out_dim: u32,
    t: u32,
) -> Result {
    match w {
        MatW::Q8(b) => kernels::matmul_q8_0(out, b, x, in_dim, out_dim, t)?,
        MatW::Kq(k) => kernels::matmul_kq(out, &k.w, xq, in_dim, out_dim, t, k.row_bytes, k.quant)?,
    }
    Ok(())
}

/// Bytes a q8_K quantization of `n` elements occupies (one row).
fn q8k_bytes(n: u32) -> usize {
    (n as usize).div_ceil(kernels::Q8_K_BLOCK_ELEMS) * kernels::Q8_K_BLOCK_BYTES
}

/// Per-KDA-layer recurrent state, on that layer's owner card.
struct KdaState {
    /// delta-rule state [n_head][head_dim][head_dim]
    s: DeviceBuf,
    /// one conv window per stream, [conv_k - 1][d_inner] each.
    conv_q: DeviceBuf,
    conv_k: DeviceBuf,
    conv_v: DeviceBuf,
    dev: i32,
}

/// Scratch for whichever card a layer lands on.
struct BailingScratch {
    dev: i32,
    /// hop buffers for the residual in / attention out
    normed_a: DeviceBuf,
    attn_out_a: DeviceBuf,
    /// q8_K of the widest vector ever quantized into this buffer
    xq: DeviceBuf,
    // KDA
    q: DeviceBuf,
    k: DeviceBuf,
    v: DeviceBuf,
    cq: DeviceBuf,
    ck: DeviceBuf,
    cv: DeviceBuf,
    g: DeviceBuf,
    beta: DeviceBuf,
    o: DeviceBuf,
    gate: DeviceBuf,
    // MLA
    mq: DeviceBuf,
    kv_raw: DeviceBuf,
    kv_norm: DeviceBuf,
    qk_low: DeviceBuf,
    heads: DeviceBuf,
    selected: DeviceBuf,
}

impl BailingScratch {
    fn new(m: &Model, dev: i32, ctx: u32) -> Result<BailingScratch> {
        let s = m.shape;
        kernels::set_device(dev)?;
        let f32s = |n: usize| DeviceBuf::alloc(n * 4);
        let d_inner = (s.n_head * s.kda_head_dim) as usize;
        Ok(BailingScratch {
            dev,
            normed_a: f32s(s.n_embd as usize)?,
            attn_out_a: f32s(s.n_embd as usize)?,
            xq: DeviceBuf::alloc(q8k_bytes(
                s.n_embd
                    .max(s.n_head * s.kda_head_dim)
                    .max(s.n_head * s.value_mla)
                    .max(s.n_ff_exp),
            ))?,
            q: f32s(d_inner)?,
            k: f32s(d_inner)?,
            v: f32s(d_inner)?,
            cq: f32s(d_inner)?,
            ck: f32s(d_inner)?,
            cv: f32s(d_inner)?,
            g: f32s(d_inner)?,
            beta: f32s(s.n_head as usize)?,
            o: f32s(d_inner)?,
            gate: f32s(d_inner)?,
            mq: f32s((s.n_head * s.qk_dim()) as usize)?,
            kv_raw: f32s((s.n_kv_lora + s.qk_rope) as usize)?,
            kv_norm: f32s(s.n_kv_lora as usize)?,
            qk_low: f32s((s.n_head * s.n_kv_lora) as usize)?,
            heads: f32s((s.n_head * s.value_mla) as usize)?,
            selected: DeviceBuf::alloc(ctx as usize * 4)?,
        })
    }
}

pub(super) struct BailingRt {
    primary: i32,
    states: Vec<Option<KdaState>>,
    scratch: Vec<BailingScratch>,
}

impl BailingRt {
    pub fn new(m: &Model, ctx: u32) -> Result<BailingRt> {
        let s = m.shape;
        let primary = kernels::get_device();
        let d_inner = (s.n_head * s.kda_head_dim) as usize;
        let hd = s.kda_head_dim as usize;

        let mut states = Vec::with_capacity(s.n_exec_layer as usize);
        for il in 0..s.n_exec_layer as usize {
            let is_kda = matches!(&m.layers[il].attn, Attn::Bailing(w) if w.kda.is_some());
            if !is_kda {
                states.push(None);
                continue;
            }
            let dev = m.attn_layer_dev.get(il).copied().unwrap_or(primary);
            kernels::set_device(dev)?;
            let sbytes = s.n_head as usize * hd * hd * 4;
            let cbytes = (s.ssm_conv_k as usize - 1) * d_inner * 4;
            let mut st = KdaState {
                s: DeviceBuf::alloc(sbytes)?,
                conv_q: DeviceBuf::alloc(cbytes)?,
                conv_k: DeviceBuf::alloc(cbytes)?,
                conv_v: DeviceBuf::alloc(cbytes)?,
                dev,
            };
            kernels::zero(&mut st.s, sbytes)?;
            kernels::zero(&mut st.conv_q, cbytes)?;
            kernels::zero(&mut st.conv_k, cbytes)?;
            kernels::zero(&mut st.conv_v, cbytes)?;
            states.push(Some(st));
        }

        let mut devs: Vec<i32> = m.attn_layer_dev.iter().copied().collect();
        devs.push(primary);
        devs.sort_unstable();
        devs.dedup();
        let mut scratch = Vec::with_capacity(devs.len());
        for d in devs {
            scratch.push(BailingScratch::new(m, d, ctx)?);
        }
        kernels::set_device(primary)?;
        Ok(BailingRt {
            primary,
            states,
            scratch,
        })
    }

    fn reset(&mut self) -> Result {
        let primary = kernels::get_device();
        for st in self.states.iter_mut().flatten() {
            kernels::set_device(st.dev)?;
            for b in [&mut st.s, &mut st.conv_q, &mut st.conv_k, &mut st.conv_v] {
                let n = b.bytes();
                kernels::zero(b, n)?;
            }
        }
        kernels::set_device(primary)?;
        Ok(())
    }

    fn sc(&mut self, dev: i32) -> Result<&mut BailingScratch> {
        self.scratch
            .iter_mut()
            .find(|x| x.dev == dev)
            .ok_or_else(|| "bailing: no scratch for owner device".into())
    }
}

impl Model {
    pub(super) fn forward_bailing(
        &self,
        st: &mut State,
        tokens: &[u32],
        pos0: u32,
        rows: u32,
    ) -> Result<Option<Vec<f32>>> {
        if tokens.is_empty() {
            return Err("empty batch".into());
        }
        if pos0 + tokens.len() as u32 > st.ctx {
            return Err("position exceeds context".into());
        }
        let mut rt = st.bailing.take().ok_or("bailing state missing")?;
        let r = self.forward_bailing_inner(st, &mut rt, tokens, pos0, rows);
        st.bailing = Some(rt);
        r
    }

    fn forward_bailing_inner(
        &self,
        st: &mut State,
        rt: &mut BailingRt,
        tokens: &[u32],
        pos0: u32,
        rows: u32,
    ) -> Result<Option<Vec<f32>>> {
        let s = self.shape;
        if pos0 == 0 {
            rt.reset()?;
        }
        let primary = rt.primary;
        kernels::set_device(primary)?;
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = pos0 + i as u32;
            let ids = [tok as i32];
            st.tok.write(0, kernels::as_bytes(&ids))?;
            kernels::embed_q8_0(
                &mut st.cur,
                &self.token_embd,
                &st.tok,
                s.n_embd,
                s.n_vocab,
                1,
            )?;
            for il in 0..s.n_exec_layer as usize {
                self.eval_bailing_layer(st, rt, il, &self.layers[il], pos)?;
            }
            kernels::set_device(primary)?;
            // PULSAR_POS_DBG: per-position top-1 after the full stack -
            // teacher-forced sequence to diff against the llama.cpp oracle
            if std::env::var_os("PULSAR_POS_DBG").is_some() {
                kernels::rms_norm(
                    &mut st.normed,
                    &st.cur,
                    &self.output_norm,
                    s.n_embd,
                    1,
                    s.rms_eps,
                )?;
                self.head_logits(st, 1)?;
                kernels::sync()?;
                let lg = st.logits.read_f32(s.n_vocab as usize)?;
                let (best, bv) =
                    lg.iter()
                        .enumerate()
                        .fold(
                            (0usize, f32::MIN),
                            |(bi, bv), (i2, &v)| if v > bv { (i2, v) } else { (bi, bv) },
                        );
                eprintln!("posdbg pos={} tok={} next={} p={:.4}", pos, tok, best, bv);
            }
        }
        if rows == 0 {
            return Ok(None);
        }
        if rows != 1 {
            return Err("bailing: decode path emits one row per pass".into());
        }
        kernels::rms_norm(
            &mut st.normed,
            &st.cur,
            &self.output_norm,
            s.n_embd,
            1,
            s.rms_eps,
        )?;
        self.head_logits(st, 1)?;
        kernels::sync()?;
        Ok(Some(st.logits.read_f32(s.n_vocab as usize)?))
    }

    pub(super) fn eval_bailing_layer(
        &self,
        st: &mut State,
        rt: &mut BailingRt,
        il: usize,
        l: &LayerW,
        pos: u32,
    ) -> Result {
        let s = self.shape;
        let eps = s.rms_eps;
        let primary = rt.primary;
        let Attn::Bailing(w) = &l.attn else {
            return Err("bailing layer without Bailing attn weights".into());
        };
        let d_inner = s.n_head * s.kda_head_dim;
        let n_embd = s.n_embd;

        kernels::rms_norm(&mut st.normed, &st.cur, &l.attn_norm, n_embd, 1, eps)?;

        // ---- attention half, on this layer's owner card
        let a_dev = self.attn_layer_dev.get(il).copied().unwrap_or(primary);
        if a_dev != primary {
            let bytes = n_embd as usize * 4;
            let sc = rt.sc(a_dev)?;
            // The D2D copy below runs on the legacy NULL stream, which has
            // no ordering against the PTDS rms_norm that just wrote
            // st.normed. Flush the producer device first.
            kernels::sync()?;
            kernels::copy_across(&mut sc.normed_a, &st.normed, bytes)?;
            kernels::set_device(a_dev)?;
        }

        if let Some(kda) = &w.kda {
            let sc = rt
                .scratch
                .iter_mut()
                .find(|x| x.dev == a_dev)
                .ok_or("bailing scratch")?;
            let xin: *const DeviceBuf = if a_dev != primary {
                &sc.normed_a
            } else {
                &st.normed
            };
            let xin = unsafe { &*xin };
            if matches!(kda.wq, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut sc.xq, xin, n_embd, 1)?;
            }
            matw(&mut sc.q, &kda.wq, xin, &sc.xq, n_embd, d_inner, 1)?;
            matw(&mut sc.k, &kda.wk, xin, &sc.xq, n_embd, d_inner, 1)?;
            matw(&mut sc.v, &kda.wv, xin, &sc.xq, n_embd, d_inner, 1)?;
            // DIRECT decay projection: f_a is [n_embd -> d_inner], no f_b.
            matw(&mut sc.g, &kda.f_a, xin, &sc.xq, n_embd, d_inner, 1)?;
            matw(&mut sc.beta, &kda.beta_w, xin, &sc.xq, n_embd, s.n_head, 1)?;
            // a holds -exp(A_log) (negated at load); the shared K3 coeff
            // kernel computes g_min * sigmoid(exp(A_log) * (z + dt_bias)).
            kernels::k3_kda_coeffs(
                &mut sc.g,
                &mut sc.beta,
                &kda.a,
                &kda.dt_bias,
                1,
                s.n_head,
                s.kda_head_dim,
                s.kda_gate_lb,
            )?;

            let ks = rt.states[il].as_mut().ok_or("bailing kda state missing")?;
            kernels::qwen35_conv_step(
                &mut sc.cq,
                &sc.q,
                &kda.conv_q,
                &mut ks.conv_q,
                d_inner,
                s.ssm_conv_k,
            )?;
            kernels::qwen35_conv_step(
                &mut sc.ck,
                &sc.k,
                &kda.conv_k,
                &mut ks.conv_k,
                d_inner,
                s.ssm_conv_k,
            )?;
            kernels::qwen35_conv_step(
                &mut sc.cv,
                &sc.v,
                &kda.conv_v,
                &mut ks.conv_v,
                d_inner,
                s.ssm_conv_k,
            )?;
            kernels::qwen35_l2_norm(&mut sc.cq, s.n_head, s.kda_head_dim, eps)?;
            kernels::qwen35_l2_norm(&mut sc.ck, s.n_head, s.kda_head_dim, eps)?;
            kernels::k3_kda_step(
                &mut sc.o,
                &mut ks.s,
                &sc.cq,
                &sc.ck,
                &sc.cv,
                &sc.g,
                &sc.beta,
                s.n_head,
                s.kda_head_dim,
            )?;
            // per-head rms norm, then the full-rank sigmoid output gate
            kernels::gqa_head_rms_norm(
                &mut sc.o,
                Some(&kda.ssm_norm),
                s.n_head,
                s.kda_head_dim,
                eps,
            )?;
            matw(&mut sc.gate, &kda.wg, xin, &sc.xq, n_embd, d_inner, 1)?;
            kernels::qwen35_sigmoid_gate(&mut sc.o, &sc.gate, d_inner)?;
            if matches!(kda.out, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut sc.xq, &sc.o, d_inner, 1)?;
            }
            if a_dev != primary {
                matw(
                    &mut sc.attn_out_a,
                    &kda.out,
                    &sc.o,
                    &sc.xq,
                    d_inner,
                    n_embd,
                    1,
                )?;
                // Producer-side ordering: matw wrote attn_out_a on this
                // device's per-thread stream, and the NULL-stream copy does
                // not order against it. Sync before reading.
                kernels::sync()?;
                kernels::copy_across(&mut st.attn_out, &sc.attn_out_a, n_embd as usize * 4)?;
                kernels::set_device(primary)?;
            } else {
                matw(
                    &mut st.attn_out,
                    &kda.out,
                    &sc.o,
                    &sc.xq,
                    d_inner,
                    n_embd,
                    1,
                )?;
            }
        } else if let Some(mla) = &w.mla {
            let rope = s.rope_cfg();
            let kv_raw_dim = s.n_kv_lora + s.qk_rope;
            let sc = rt
                .scratch
                .iter_mut()
                .find(|x| x.dev == a_dev)
                .ok_or("bailing scratch")?;
            let xin: *const DeviceBuf = if a_dev != primary {
                &sc.normed_a
            } else {
                &st.normed
            };
            let xin = unsafe { &*xin };
            // DIRECT q projection (no q_a/q_b lora pair), then real RoPE.
            kernels::matmul_q8_0(&mut sc.mq, &mla.q, xin, n_embd, s.n_head * s.qk_dim(), 1)?;
            kernels::mla_rope_tail(&mut sc.mq, 1, s.n_head, s.qk_dim(), s.qk_rope, pos, &rope)?;
            kernels::matmul_q8_0(&mut sc.kv_raw, &mla.kv_a_mqa, xin, n_embd, kv_raw_dim, 1)?;
            kernels::mla_kv_lora_rms_norm(
                &mut sc.kv_norm,
                &sc.kv_raw,
                &mla.kv_a_norm,
                1,
                kv_raw_dim,
                s.n_kv_lora,
                eps,
            )?;
            kernels::mla_store_compact_kv(
                &mut st.kcache[il],
                &mut st.vcache[il],
                &sc.kv_norm,
                &sc.kv_raw,
                pos,
                1,
                st.ctx,
                kv_raw_dim,
                s.n_kv_lora,
                s.qk_rope,
                st.kvq_lat,
            )?;
            let visible = pos + 1;
            kernels::mla_fill_selected_range(&mut sc.selected, 1, pos, visible, st.ctx)?;
            kernels::mla_qk_lowrank(
                &mut sc.qk_low,
                &sc.mq,
                &mla.k_b,
                1,
                s.n_head,
                s.n_kv_lora,
                s.qk_nope,
                s.qk_dim(),
            )?;
            kernels::mla_attention(
                &mut sc.heads,
                &sc.mq,
                &sc.qk_low,
                &st.kcache[il],
                &st.vcache[il],
                &mla.v_b,
                &sc.selected,
                1,
                visible,
                st.ctx,
                s.n_head,
                s.n_kv_lora,
                s.qk_nope,
                s.qk_rope,
                s.value_mla,
                &rope,
                st.kvq_lat,
            )?;
            // per-head sigmoid output gate, broadcast across the value dims.
            // Reads the NORMED layer input, not the attention result.
            if matches!(mla.gate, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut sc.xq, xin, n_embd, 1)?;
            }
            matw(&mut sc.gate, &mla.gate, xin, &sc.xq, n_embd, s.n_head, 1)?;
            kernels::qwen35_row_sigmoid_scale(&mut sc.heads, &sc.gate, s.n_head, s.value_mla)?;
            if matches!(mla.out, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut sc.xq, &sc.heads, s.n_head * s.value_mla, 1)?;
            }
            if a_dev != primary {
                matw(
                    &mut sc.attn_out_a,
                    &mla.out,
                    &sc.heads,
                    &sc.xq,
                    s.n_head * s.value_mla,
                    n_embd,
                    1,
                )?;
                // Producer-side ordering, same hazard as the KDA branch:
                // sync the producer device before the NULL-stream copy.
                kernels::sync()?;
                kernels::copy_across(&mut st.attn_out, &sc.attn_out_a, n_embd as usize * 4)?;
                kernels::set_device(primary)?;
            } else {
                matw(
                    &mut st.attn_out,
                    &mla.out,
                    &sc.heads,
                    &sc.xq,
                    s.n_head * s.value_mla,
                    n_embd,
                    1,
                )?;
            }
        } else {
            return Err("bailing layer with neither kda nor mla".into());
        }

        kernels::add(&mut st.after_attn, &st.cur, &st.attn_out, n_embd)?;

        // ---- FFN half
        kernels::rms_norm(&mut st.normed, &st.after_attn, &l.ffn_norm, n_embd, 1, eps)?;
        self.bailing_ffn(st, il, l)?;
        kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, n_embd)?;
        Ok(())
    }

    /// Dense FFN on the leading layers, grouped-router MoE everywhere else.
    fn bailing_ffn(&self, st: &mut State, il: usize, l: &LayerW) -> Result {
        let s = self.shape;
        let act = s.moe_act_op;
        if let Ffn::Dense { gate, up, down } = &l.ffn {
            kernels::matmul_q8_0(
                &mut st.gate_act,
                gate,
                &st.normed,
                s.n_embd,
                s.n_ff_dense,
                1,
            )?;
            kernels::matmul_q8_0(&mut st.up_act, up, &st.normed, s.n_embd, s.n_ff_dense, 1)?;
            kernels::swiglu(
                &mut st.ffn_mid,
                &st.gate_act,
                &st.up_act,
                s.n_ff_dense,
                0.0,
                1.0,
                act,
            )?;
            kernels::matmul_q8_0(
                &mut st.ffn_out,
                down,
                &st.ffn_mid,
                s.n_ff_dense,
                s.n_embd,
                1,
            )?;
            return Ok(());
        }
        let Ffn::Moe {
            gate_inp,
            probs_b,
            shexp,
            gate_exps,
            up_exps,
            down_exps,
            ..
        } = &l.ffn
        else {
            return Err("bailing layer without a dense or MoE ffn".into());
        };
        // grouped noaux_tc router: top n_group_used groups, then top-k.
        kernels::matmul_f32(
            &mut st.router_logits,
            gate_inp,
            &st.normed,
            s.n_embd,
            s.n_expert,
            1,
        )?;
        kernels::bailing_router_select(
            &mut st.router_selected,
            &mut st.router_weights,
            &st.router_logits,
            probs_b,
            s.n_expert,
            s.n_expert_used,
            s.n_expert_groups,
            s.n_group_used,
            s.expert_weight_scale,
            1,
        )?;
        // shared expert, with its per-layer clamp.
        let clamp_sh = self.clamp_shexp_l.get(il).copied().unwrap_or(0.0);
        if let Some((sg, su, sd)) = shexp {
            kernels::matmul_q8_0(&mut st.gate_act, sg, &st.normed, s.n_embd, s.n_ff_shexp, 1)?;
            kernels::matmul_q8_0(&mut st.up_act, su, &st.normed, s.n_embd, s.n_ff_shexp, 1)?;
            kernels::swiglu(
                &mut st.ffn_mid,
                &st.gate_act,
                &st.up_act,
                s.n_ff_shexp,
                clamp_sh,
                1.0,
                act,
            )?;
            kernels::matmul_q8_0(
                &mut st.shared_out,
                sd,
                &st.ffn_mid,
                s.n_ff_shexp,
                s.n_embd,
                1,
            )?;
        } else {
            kernels::zero(&mut st.shared_out, s.n_embd as usize * 4)?;
        }
        kernels::quantize_q8_k(&mut st.xq, &st.normed, s.n_embd, 1)?;
        kernels::sync()?;
        let selected = st.router_selected.read_i32(s.n_expert_used as usize)?;
        self.dsv4_moe(
            st, il, &selected, gate_exps, up_exps, down_exps, act, 1, s.n_embd,
        )?;
        kernels::add(&mut st.ffn_out, &st.moe_out, &st.shared_out, s.n_embd)?;
        Ok(())
    }
}
