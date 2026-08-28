//! Qwen3.8-Flash-Next (qwen4exp) forward graph over the pulsar CUDA
//! kernels. Port of llama.cpp `src/models/qwen4exp.cpp` (unsloth fork),
//! decode-first like the qwen35 sibling: prefill loops T_MAX-token chunks
//! because the GDN recurrences and the PLE conv are sequential.
//!
//! Layer skeleton (all layers):
//!   res_hc [T][hc][n_embd] wide residual stream (token-major)
//!     -> hc_mix(attn) -> {GDN | gated attention} -> hc_combine
//!     -> hc_mix(ffn)  -> shared+routed MoE        -> hc_combine
//! The head is a final hc_mix: there is NO output_norm in this arch.
//!
//! GDN kernels (conv/split/l2/delta/coeffs) are shared with qwen35
//! verbatim; only the output gate differs (SIGMOID not silu). Attention
//! keeps qwen35's per-head interleaved [q|gate] fused q projection +
//! partial neox rope. QSA degenerates to dense attention below
//! indexer_top_k + compress_ratio - 1 visible cells; v1 serves that window
//! densely and errors beyond it.
//!
//! PLE (layer 1): host int64 hash -> per-row IQ4NL reads straight off the
//! mmap'd shard files (16 rows x 90B per token - the 26.8 GiB table never
//! loads), dequant on CPU through the ggml kvalues codebook, then the
//! standard grouped-norm / signed-gate / dilated-conv device path.

use super::{Attn, Ffn, MatW, Model, Result, Shape, State};
use kernels::DeviceBuf;

/// Verify/prefill chunk width (matches the other recurrent families).
pub(super) const T_MAX: usize = 16;

/// ggml kvalues_iq4nl codebook (the IQ4_NL table, not IQ4_XS).
const KVALUES_IQ4NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0,
    25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
];

/// Per-GDN-layer device state.
pub(super) struct GdnState {
    /// delta-rule state [v_heads][state][state] on owner device
    pub s: DeviceBuf,
    /// conv window [conv_k-1][conv_dim]
    pub conv: DeviceBuf,
    /// PLE dilated conv history [(conv_k-1)*ngram][hc_dim]; only on the
    /// layer hosting both the PLE block and a GDN mixer
    pub ple_hist: Option<DeviceBuf>,
    pub dev: i32,
}

fn ple_hist_cols(s: &Shape) -> usize {
    ((s.ple_conv_k as usize).max(1) - 1) * s.ple_ngram.max(1) as usize
}

fn ple_heads(s: &Shape) -> usize {
    ((s.ple_ngram as usize).saturating_sub(1)) * s.ple_heads_per_ngram as usize
}

fn ple_row_bytes(s: &Shape) -> u64 {
    // IQ4NL: 18 bytes per 32 values; row width = ple_head_dim
    ((s.ple_head_dim as u64) / 32) * 18
}

/// Everything needed to read PLE table rows straight from disk.
struct PleTable {
    file: super::VFile,
    /// byte offset of the tensor data start within the virtual shard file
    base: u64,
    row_bytes: u64,
    rows: u64,
    head_dim: usize,
}

/// qwen38 runtime: recurrent states + scratch sized for T_MAX chunks.
pub(super) struct Qwen38Rt {
    pub states: Vec<Option<GdnState>>,
    ple_table: Option<PleTable>,
    /// Host-side PLE token history for the single active sequence. The
    /// device history stores only convolution rows; hash predecessors are
    /// token IDs and must remain available across forward calls.
    ple_tokens: Vec<u32>,
    ple_next_pos: u32,
    // wide HC residual [T][hc][n_embd] (token-major)
    res_hc: DeviceBuf,
    // grouped-normed streams [T*hc][hc_dim]
    xn: DeviceBuf,
    lo: DeviceBuf,        // [T*hc][low_rank]
    gate_full: DeviceBuf, // [T*hc][low_rank]
    gated: DeviceBuf,     // [T*hc][hc_dim]
    inj: DeviceBuf,       // [T*hc]
    cur: DeviceBuf,       // [T][n_embd]
    attn_out: DeviceBuf,  // [T][n_embd]
    ffn_mid: DeviceBuf,   // [T][max_glu]
    xq: DeviceBuf,        // q8_K scratch over hc_dim-wide rows
    xq2: DeviceBuf,       // q8_K scratch over low_rank/n_embd rows
    qfull: DeviceBuf,
    gate_a: DeviceBuf,
    q: DeviceBuf,
    k: DeviceBuf,
    v: DeviceBuf,
    heads: DeviceBuf,
    // GDN scratch
    gdn_qkv: DeviceBuf,
    gdn_conv_out: DeviceBuf, // [T][conv_dim] conv+silu output
    gdn_z: DeviceBuf,
    gdn_gq: DeviceBuf,
    gdn_gk: DeviceBuf,
    gdn_gv: DeviceBuf,
    gdn_g: DeviceBuf,
    gdn_beta: DeviceBuf,
    gdn_o: DeviceBuf,
    // MoE / router
    router_logits: DeviceBuf,
    gate_act: DeviceBuf,
    up_act: DeviceBuf,
    shg: DeviceBuf,
    shared_out: DeviceBuf,
    midq: DeviceBuf,
    // PLE staging
    ple_emb_host: Vec<f32>, // gathered+dequantized rows [T][heads*head_dim]
    ple_emb_dev: DeviceBuf, // same, staged H2D
    ple_key_v: DeviceBuf,   // [T*hc][hc_dim]
    ple_val_v: DeviceBuf,   // [T][n_embd]
    ple_query: DeviceBuf,   // [T*hc][hc_dim] normed residual streams
    ple_gate: DeviceBuf,    // [T*hc]
    ple_norm: DeviceBuf,    // [T*hc][hc_dim] normalized gated value
    ple_pad: DeviceBuf,     // [(hist+T)][hc_dim]
    ple_out: DeviceBuf,     // [T][hc_dim]
}

impl Qwen38Rt {
    pub(super) fn new(m: &Model) -> Result<Qwen38Rt> {
        let s = m.shape;
        let primary = kernels::get_device();
        let key_dim = (s.ssm_k_heads * s.ssm_state) as usize; // 2048
        let value_dim = (s.ssm_v_heads * s.ssm_state) as usize; // 6144
        let conv_dim = 2 * key_dim + value_dim; // 10240
        let hc = s.n_hc as usize; // 4
        let hc_dim = hc * s.n_embd as usize; // 10240
        let lr = s.hc_low_rank as usize; // 320
        let mb = T_MAX;
        let heads = ple_heads(&s); // 16

        let mut states: Vec<Option<GdnState>> =
            Vec::with_capacity(s.n_exec_layer as usize);
        for il in 0..s.n_exec_layer {
            if (il + 1) % s.full_attn_interval == 0 {
                states.push(None); // full-attention layer: KV only
                continue;
            }
            let dev = m.layer_dev(il as usize);
            kernels::set_device(dev)?;
            let sbytes =
                s.ssm_v_heads as usize * s.ssm_state as usize * s.ssm_state as usize * 4;
            let cbytes = (s.ssm_conv_k as usize - 1) * conv_dim * 4;
            let ple_hist = if il as i32 == s.ple_layer {
                Some(DeviceBuf::alloc(ple_hist_cols(&s) * hc_dim * 4)?)
            } else {
                None
            };
            let mut gs = GdnState {
                s: DeviceBuf::alloc(sbytes)?,
                conv: DeviceBuf::alloc(cbytes)?,
                ple_hist,
                dev,
            };
            kernels::zero(&mut gs.s, sbytes)?;
            kernels::zero(&mut gs.conv, cbytes)?;
            if let Some(ph) = &mut gs.ple_hist {
                kernels::zero(ph, ph.bytes())?;
            }
            states.push(Some(gs));
        }
        kernels::set_device(primary)?;

        // PLE table reader: located through the merged gguf, reads go
        // straight to the shard files (mmap-like page cache behavior).
        let ple_table = if s.ple_layer >= 0 {
            let ti = m
                .gguf
                .tensor("per_layer_token_embd.weight")
                .ok_or("qwen38: missing per_layer_token_embd table")?;
            Some(PleTable {
                file: super::VFile::open(&m.shards)?,
                base: m.gguf.data_offset + ti.offset,
                row_bytes: ple_row_bytes(&s),
                rows: ti.dims[1],
                head_dim: s.ple_head_dim as usize,
            })
        } else {
            None
        };

        let f32s = |n: usize| DeviceBuf::alloc(n * 4);
        let max_glu = s.n_ff_exp.max(s.n_ff_shexp).max(1) as usize;
        Ok(Qwen38Rt {
            states,
            ple_table,
            ple_tokens: Vec::new(),
            ple_next_pos: 0,
            res_hc: f32s(mb * hc * s.n_embd as usize)?,
            xn: f32s(mb * hc * s.n_embd as usize)?,
            lo: f32s(mb * hc * lr)?,
            gate_full: f32s(mb * hc * hc * s.n_embd as usize)?,
            gated: f32s(mb * hc * s.n_embd as usize)?,
            inj: f32s(mb * hc)?,
            cur: f32s(mb * s.n_embd as usize)?,
            attn_out: f32s(mb * s.n_embd as usize)?,
            ffn_mid: f32s(mb * max_glu)?,
            xq: DeviceBuf::alloc(
                mb * hc * s.n_embd as usize / kernels::Q8_K_BLOCK_ELEMS
                    * kernels::Q8_K_BLOCK_BYTES,
            )?,
            // widest contraction through this scratch: hc_dim (10240) in
            // the PLE projections and attn/gdn q8_K rows
            xq2: DeviceBuf::alloc(
                mb * (hc * s.n_embd.max(lr as u32) as usize)
                    / kernels::Q8_K_BLOCK_ELEMS
                    * kernels::Q8_K_BLOCK_BYTES,
            )?,
            qfull: f32s(mb * 2 * (s.n_head * s.head_dim) as usize)?,
            gate_a: f32s(mb * (s.n_head * s.head_dim) as usize)?,
            q: f32s(mb * (s.n_head * s.head_dim) as usize)?,
            k: f32s(mb * (s.n_head_kv * s.head_dim) as usize)?,
            v: f32s(mb * (s.n_head_kv * s.head_dim) as usize)?,
            heads: f32s(mb * (s.n_head * s.head_dim) as usize)?,
            gdn_qkv: f32s(mb * conv_dim)?,
            gdn_conv_out: f32s(mb * conv_dim)?,
            gdn_z: f32s(mb * value_dim)?,
            gdn_gq: f32s(mb * key_dim)?,
            gdn_gk: f32s(mb * key_dim)?,
            gdn_gv: f32s(mb * value_dim)?,
            gdn_g: f32s(mb * s.ssm_v_heads as usize)?,
            gdn_beta: f32s(mb * s.ssm_v_heads as usize)?,
            gdn_o: f32s(mb * value_dim)?,
            router_logits: f32s(mb * s.n_expert as usize)?,
            gate_act: f32s(mb * max_glu)?,
            up_act: f32s(mb * max_glu)?,
            shg: f32s(mb)?,
            shared_out: f32s(mb * s.n_embd as usize)?,
            midq: DeviceBuf::alloc(
                mb * max_glu / kernels::Q8_K_BLOCK_ELEMS * kernels::Q8_K_BLOCK_BYTES,
            )?,
            ple_emb_host: vec![0f32; mb * heads * s.ple_head_dim as usize],
            ple_emb_dev: f32s(mb * heads * s.ple_head_dim as usize)?,
            ple_key_v: f32s(mb * hc_dim)?,
            ple_val_v: f32s(mb * hc * s.n_embd as usize)?,
            ple_query: f32s(mb * hc_dim)?,
            ple_gate: f32s(mb * hc)?,
            ple_norm: f32s(mb * hc_dim)?,
            ple_pad: f32s((mb + ple_hist_cols(&s)) * hc_dim)?,
            ple_out: f32s(mb * hc_dim)?,
        })
    }

    fn rt_reset(&mut self) -> Result {
        self.ple_tokens.clear();
        self.ple_next_pos = 0;
        let primary = kernels::get_device();
        for g in self.states.iter_mut().flatten() {
            kernels::set_device(g.dev)?;
            let (sb, cb) = (g.s.bytes(), g.conv.bytes());
            kernels::zero(&mut g.s, sb)?;
            kernels::zero(&mut g.conv, cb)?;
            if let Some(ph) = &mut g.ple_hist {
                kernels::zero(ph, ph.bytes())?;
            }
        }
        kernels::set_device(primary)?;
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn ckpt(
        &self,
    ) -> Result<Vec<Option<(DeviceBuf, DeviceBuf, Option<DeviceBuf>)>>> {
        let primary = kernels::get_device();
        let mut out = Vec::with_capacity(self.states.len());
        for gs in &self.states {
            out.push(match gs {
                Some(g) => {
                    kernels::set_device(g.dev)?;
                    let mut s2 = DeviceBuf::alloc(g.s.bytes())?;
                    kernels::copy_d2d(&mut s2, 0, &g.s, 0, g.s.bytes())?;
                    let mut c2 = DeviceBuf::alloc(g.conv.bytes())?;
                    kernels::copy_d2d(&mut c2, 0, &g.conv, 0, g.conv.bytes())?;
                    let p2 = match &g.ple_hist {
                        Some(ph) => {
                            let mut b = DeviceBuf::alloc(ph.bytes())?;
                            kernels::copy_d2d(&mut b, 0, ph, 0, ph.bytes())?;
                            Some(b)
                        }
                        None => None,
                    };
                    Some((s2, c2, p2))
                }
                None => None,
            });
        }
        kernels::set_device(primary)?;
        Ok(out)
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn ckpt_restore(
        &mut self,
        ck: &[Option<(DeviceBuf, DeviceBuf, Option<DeviceBuf>)>],
    ) -> Result {
        let primary = kernels::get_device();
        for (gs, c) in self.states.iter_mut().zip(ck) {
            if let (Some(g), Some((s2, c2, p2))) = (gs, c) {
                kernels::set_device(g.dev)?;
                kernels::copy_d2d(&mut g.s, 0, s2, 0, s2.bytes())?;
                kernels::copy_d2d(&mut g.conv, 0, c2, 0, c2.bytes())?;
                if let (Some(dst), Some(src)) = (&mut g.ple_hist, p2) {
                    kernels::copy_d2d(dst, 0, &src, 0, src.bytes())?;
                }
            }
        }
        kernels::set_device(primary)?;
        Ok(())
    }
}


/* ---- forward -------------------------------------------------------------- */

impl Model {

    pub(super) fn forward_qwen38(
        &self,
        st: &mut State,
        tokens: &[u32],
        pos0: u32,
        rows: u32,
    ) -> Result<Option<Vec<f32>>> {
        if tokens.is_empty() {
            return Err("empty batch".into());
        }
        if rows as usize > T_MAX {
            return Err("qwen38: rows exceeds the verify chunk".into());
        }
        if pos0 + tokens.len() as u32 > st.ctx {
            return Err("position exceeds context".into());
        }
        let mut rt = st.qwen38.take().ok_or("qwen38 state missing")?;
        let r = self.forward_qwen38_inner(st, &mut rt, tokens, pos0, rows);
        st.qwen38 = Some(rt);
        r
    }

    fn forward_qwen38_inner(
        &self,
        st: &mut State,
        rt: &mut Qwen38Rt,
        tokens: &[u32],
        pos0: u32,
        rows: u32,
    ) -> Result<Option<Vec<f32>>> {
        let s = self.shape;
        if pos0 == 0 {
            rt.rt_reset()?;
        }
        let n_embd = s.n_embd;

        let mut consumed = 0usize;
        for chunk in tokens.chunks(T_MAX) {
            let t = chunk.len() as u32;
            // Drain before staging: the previous chunk's PTDS tail (embed
            // reads st.tok) and any prior call's legacy-stream tail (an MTP
            // re-anchor D2D reads res_hc) must finish before the staging
            // below reuses those buffers (TIER_DEBUG_SESSION lesson).
            kernels::sync()?;
            // embed -> wide residual: hc identical copies of every token
            // row (TOKEN-MAJOR [t][c][e] layout matching the HC kernels),
            // broadcast on device - no D2H+host+H2D roundtrip
            let ids: Vec<i32> = chunk.iter().map(|&x| x as i32).collect();
            st.tok.write(0, kernels::as_bytes(&ids))?;
            kernels::embed_q8_0(
                &mut rt.cur,
                &self.token_embd,
                &st.tok,
                n_embd,
                s.n_vocab,
                t,
            )?;
            kernels::qwen38_broadcast_hc(&mut rt.res_hc, &rt.cur, t, n_embd, s.n_hc)?;
            self.eval_qwen38_span(st, rt, chunk, pos0 + consumed as u32, t)?;
            consumed += chunk.len();
        }

        if rows == 0 {
            return Ok(None);
        }

        // ---- head HC mix. It is a per-token op (no cross-token state), so
        // run it on the LAST CHUNK only: all scratch is sized T_MAX, while
        // `consumed` covers the whole call (a chat turn can exceed one
        // chunk). res_hc already holds exactly that final chunk's streams.
        let last_chunk = tokens.chunks(T_MAX).last().map(|c| c.len()).unwrap_or(0);
        if rows as usize > last_chunk {
            return Err("qwen38: rows exceeds the final chunk".into());
        }
        let t_all = last_chunk as u32;
        let head_hc = self
            .output_hc
            .as_ref()
            .ok_or("qwen38: model missing output_hc mixer")?;
        self.hc_norm(rt, &head_hc.norm, t_all, true)?;
        self.hc_mix_core(rt, head_hc, t_all, false)?;
        let off = (last_chunk - rows as usize) * n_embd as usize * 4;
        // head_logits contracts over State.normed: publish the HC-mixed
        // final streams there first (no further norm - the HC mix IS it)
        kernels::copy_d2d(
            &mut st.normed,
            0,
            &rt.cur,
            off,
            rows as usize * n_embd as usize * 4,
        )?;
        kernels::copy_d2d(
            &mut st.last_row,
            0,
            &rt.cur,
            off,
            rows as usize * n_embd as usize * 4,
        )?;
        self.head_logits(st, rows)?;
        kernels::sync()?;
        let lg = st.logits.read_f32(rows as usize * s.n_vocab as usize)?;
        Ok(Some(lg))
    }

    fn eval_qwen38_span(
        &self,
        st: &mut State,
        rt: &mut Qwen38Rt,
        chunk: &[u32],
        pos0: u32,
        t: u32,
    ) -> Result {
        let s = self.shape;
        for il in 0..self.layers.len() {
            let l = &self.layers[il];
            let Attn::Qwen38(w) = &l.attn else {
                return Err("qwen38 layer without Qwen38 weights".into());
            };

            if il as i32 == s.ple_layer {
                if std::env::var_os("PULSAR_QWEN38_SKIP_PLE").is_some() {
                    eprintln!("pulsar: qwen38 PULSAR_QWEN38_SKIP_PLE=1 - layer {il} runs WITHOUT the PLE block (output will not match the oracle)");
                } else {
                    self.eval_ple(st, rt, chunk, pos0, t, il)?;
                    kernels::sync()?;
                }
            }

            // ---- attn-side HC mix
            self.hc_norm(rt, &w.hc_attn.norm, t, true)?;
            self.hc_mix_core(rt, &w.hc_attn, t, true)?;
            // ---- token mixer produces rt.attn_out from rt.cur
            if let Some(gdn) = &w.gdn {
                self.eval_gdn_layer(rt, il, gdn, t)?;
            } else if let Some(attn) = &w.attn {
                self.eval_attn_layer(st, rt, il, pos0, t, attn)?;
            } else {
                return Err("qwen38 layer with neither attn nor gdn".into());
            }
            kernels::qwen38_hc_combine(
                &mut rt.res_hc,
                &rt.attn_out,
                &rt.inj,
                t,
                s.n_embd,
                s.n_hc,
            )?;

            // ---- ffn-side HC mix + MoE
            self.hc_norm(rt, &w.hc_ffn.norm, t, true)?;
            self.hc_mix_core(rt, &w.hc_ffn, t, true)?;
            self.eval_moe(st, rt, il, l, w, t)?;
            kernels::qwen38_hc_combine(
                &mut rt.res_hc,
                &rt.cur,
                &rt.inj,
                t,
                s.n_embd,
                s.n_hc,
            )?;
        }
        Ok(())
    }

    /* ---- MTP (nextn) draft forward --------------------------------------
     * The trunk's final wide res_hc (st.mtp_hidden, hc_dim-wide) feeds the
     * MTP block: per-stream RMS + hnorm gamma + stream-mean collapse to
     * n_embd, concatenated with the enormed token embedding, projected by
     * eh_proj to the MTP layer's input, broadcast to the HC streams, run
     * through the MTP block's full-attention + MoE, then the shared
     * output_hc mixer produces the draft logits.
     */
    pub(super) fn forward_qwen38_mtp_draft(
        &self,
        st: &mut State,
        mtp_layer: &super::LayerW,
        token: u32,
        pos: u32,
    ) -> Result<u32> {
        let s = self.shape;
        let mtp = self.mtp.as_ref().expect("qwen38 mtp without a layer");
        let hc = s.n_hc as usize;
        let n_embd = s.n_embd as usize;
        let hc_dim = hc * n_embd;
        let row = n_embd * 4;
        let mut rt = st.qwen38.take().ok_or("qwen38 state missing")?;

        // 1. embed + enorm (n_embd)
        st.tok.write(0, kernels::as_bytes(&[token as i32]))?;
        kernels::embed_q8_0(
            &mut st.mtp_e_raw,
            mtp.tok_embd.as_ref().unwrap_or(&self.token_embd),
            &st.tok,
            s.n_embd,
            s.n_vocab,
            1,
        )?;
        kernels::rms_norm(
            &mut st.mtp_e,
            &st.mtp_e_raw,
            &mtp.enorm,
            s.n_embd,
            1,
            s.rms_eps,
        )?;
        // 2. hidden: st.mtp_hidden (wide) -> rms + hnorm -> stream-mean collapse
        kernels::qwen38_hc_norm(
            &mut rt.xn,
            &st.mtp_hidden,
            Some(&mtp.hnorm),
            1,
            s.n_embd,
            s.n_hc,
            s.rms_eps,
        )?;
        kernels::qwen38_hc_mean(&mut st.mtp_h, &rt.xn, 1, s.n_embd, s.n_hc)?;
        // 3. concat [e; h] + eh_proj -> st.cur (n_embd)
        kernels::copy_d2d(&mut st.mtp_x, 0, &st.mtp_e, 0, row)?;
        kernels::copy_d2d(&mut st.mtp_x, row, &st.mtp_h, 0, row)?;
        kernels::matmul_q8_0(
            &mut st.cur,
            &mtp.eh_proj,
            &st.mtp_x,
            2 * s.n_embd,
            s.n_embd,
            1,
        )?;
        if std::env::var_os("PULSAR_MTP_DEBUG").is_some() {
            let cnt = |v: &Vec<f32>| v.iter().filter(|x| !x.is_finite()).count();
            let e = st.mtp_e.read_f32(n_embd)?;
            let h = st.mtp_h.read_f32(n_embd)?;
            let c = st.cur.read_f32(n_embd)?;
            let mh = st.mtp_hidden.read_f32(hc_dim)?;
            eprintln!(
                "mtp: dbg2 e_nan={} h_nan={} cur_nan={} mtp_hidden_nan={}",
                cnt(&e),
                cnt(&h),
                cnt(&c),
                cnt(&mh)
            );
        }
        // 4. broadcast st.cur to the HC streams (rt.res_hc, wide) on
        //    device; the chain below is on-stream after it
        kernels::qwen38_broadcast_hc(&mut rt.res_hc, &st.cur, 1, s.n_embd, s.n_hc)?;
        // 5. run the MTP block's full-attention + MoE
        self.eval_qwen38_mtp_layer(st, &mut rt, mtp_layer, pos)?;
        // 6. head: the shared output_hc mixer (no separate output_norm)
        let head = self
            .output_hc
            .as_ref()
            .ok_or("qwen38: model missing output_hc mixer")?;
        self.hc_norm(&mut rt, &head.norm, 1, true)?;
        self.hc_mix_core(&mut rt, head, 1, false)?;
        kernels::copy_d2d(&mut st.normed, 0, &rt.cur, 0, row)?;
        if let Some(hw) = &mtp.output_w {
            kernels::matmul_q8_0(&mut st.logits, hw, &st.normed, s.n_embd, s.n_vocab, 1)?;
            if self.logit_softcap > 0.0 {
                kernels::softcap(&mut st.logits, s.n_vocab, self.logit_softcap)?;
            }
            if self.logit_scale != 1.0 {
                kernels::scale(&mut st.logits, s.n_vocab, self.logit_scale)?;
            }
            if self.n_vocab_out < s.n_vocab {
                kernels::fill_row_tail(
                    &mut st.logits,
                    1,
                    s.n_vocab,
                    self.n_vocab_out,
                    f32::NEG_INFINITY,
                )?;
            }
        } else {
            self.head_logits(st, 1)?;
        }
        kernels::sync()?;
        let lg = st.logits.read_f32(s.n_vocab as usize)?;
        if std::env::var_os("PULSAR_MTP_DEBUG").is_some() {
            let wide = rt.res_hc.read_f32(hc_dim)?;
            let wnan = wide.iter().filter(|x| !x.is_finite()).count();
            let wnorm: f32 = wide.iter().map(|x| x * x).sum::<f32>().sqrt();
            let lnan = lg.iter().filter(|x| !x.is_finite()).count();
            let mut top: Vec<(usize, f32)> =
                lg.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            let top5: Vec<String> = top
                .iter()
                .take(5)
                .map(|(i, v)| format!("{}={:.2}", i, v))
                .collect();
            eprintln!(
                "mtp: dbg res_hc nan={} norm={:.3} logits nan={} top5=[{}]",
                wnan, wnorm, lnan, top5.join(" ")
            );
        }
        let draft = super::argmax(&lg);
        // 7. re-anchor st.mtp_hidden (wide) on the MTP block's res_hc
        kernels::copy_d2d(&mut st.mtp_hidden, 0, &rt.res_hc, 0, hc_dim * 4)?;
        st.qwen38 = Some(rt);
        Ok(draft)
    }

    /// Run the MTP block's single full-attention + MoE layer on rt.res_hc.
    /// Mirrors one trunk full-attention layer, but on the MTP weights and
    /// the MTP KV slot (il = n_exec_layer).
    fn eval_qwen38_mtp_layer(
        &self,
        st: &mut State,
        rt: &mut Qwen38Rt,
        mtp_layer: &super::LayerW,
        pos: u32,
    ) -> Result {
        let s = self.shape;
        let Attn::Qwen38(w) = &mtp_layer.attn else {
            return Err("qwen38 mtp layer without Qwen38 weights".into());
        };
        let il = s.n_exec_layer as usize;
        let t = 1u32;
        // attn-side HC mix
        self.hc_norm(rt, &w.hc_attn.norm, t, true)?;
        self.hc_mix_core(rt, &w.hc_attn, t, true)?;
        let attn = w.attn.as_ref().ok_or("qwen38 mtp layer without attention")?;
        self.eval_attn_layer(st, rt, il, pos, t, attn)?;
        kernels::qwen38_hc_combine(
            &mut rt.res_hc,
            &rt.attn_out,
            &rt.inj,
            t,
            s.n_embd,
            s.n_hc,
        )?;
        // ffn-side HC mix + MoE
        self.hc_norm(rt, &w.hc_ffn.norm, t, true)?;
        self.hc_mix_core(rt, &w.hc_ffn, t, true)?;
        self.eval_moe(st, rt, il, mtp_layer, w, t)?;
        kernels::qwen38_hc_combine(
            &mut rt.res_hc,
            &rt.cur,
            &rt.inj,
            t,
            s.n_embd,
            s.n_hc,
        )?;
        if std::env::var_os("PULSAR_MTP_DEBUG").is_some() {
            let cnt = |v: &Vec<f32>| v.iter().filter(|x| !x.is_finite()).count();
            let hcd = s.n_hc as usize * s.n_embd as usize;
            let a = rt.attn_out.read_f32(s.n_embd as usize)?;
            let c = rt.cur.read_f32(s.n_embd as usize)?;
            let r = rt.res_hc.read_f32(hcd)?;
            eprintln!(
                "mtp: dbg3 attn_nan={} moe_nan={} res_nan={}",
                cnt(&a),
                cnt(&c),
                cnt(&r)
            );
        }
        Ok(())
    }

    /// Fill the MTP block's KV over a prefill/verify chunk. Row i's hidden
    /// input is the trunk's res_hc at position (chunk_start - 1 + i): row 0
    /// uses st.mtp_hidden (the position before the chunk), rows 1..n use the
    /// trunk's res_hc for the previous rows. The wide hidden is per-stream
    /// normed (hnorm), stream-collapsed to n_embd, concatenated with the
    /// enormed embedding, projected by eh_proj, broadcast to the HC streams,
    /// and the MTP block's full-attention + MoE runs to fill its KV.
    pub(super) fn mtp_prefill_fill_qwen38(
        &self,
        st: &mut State,
        n_tok: u32,
        pos0: u32,
    ) -> Result {
        let s = self.shape;
        let mtp = self.mtp.as_ref().expect("qwen38 mtp without a layer");
        let hc = s.n_hc as usize;
        let n_embd = s.n_embd as usize;
        let hc_dim = hc * n_embd;
        let row = n_embd * 4;
        let hrow = hc_dim * 4;
        let n = n_tok as usize;
        let mut rt = st.qwen38.take().ok_or("qwen38 state missing")?;

        // 1. wide hidden inputs into xn: row 0 = st.mtp_hidden, rows 1..n = res_hc[0..n-1]
        kernels::copy_d2d(&mut rt.xn, 0, &st.mtp_hidden, 0, hrow)?;
        if n > 1 {
            kernels::copy_d2d(
                &mut rt.xn,
                hrow,
                &rt.res_hc,
                0,
                (n - 1) * hrow,
            )?;
        }
        // 2. hnorm + per-stream RMS -> stream-mean collapse to n_embd
        kernels::qwen38_hc_norm(
            &mut rt.gated,
            &rt.xn,
            Some(&mtp.hnorm),
            n_tok,
            s.n_embd,
            s.n_hc,
            s.rms_eps,
        )?;
        kernels::qwen38_hc_mean(&mut st.mtp_h, &rt.gated, n_tok, s.n_embd, s.n_hc)?;
        // 3. enormed embeddings
        kernels::embed_q8_0(
            &mut st.mtp_e_raw,
            mtp.tok_embd.as_ref().unwrap_or(&self.token_embd),
            &st.tok,
            s.n_embd,
            s.n_vocab,
            n_tok,
        )?;
        kernels::rms_norm(
            &mut st.mtp_e,
            &st.mtp_e_raw,
            &mtp.enorm,
            s.n_embd,
            n_tok,
            s.rms_eps,
        )?;
        // 4. interleave [e; h] per row into mtp_x
        for i in 0..n {
            kernels::copy_d2d(
                &mut st.mtp_x,
                i * 2 * row,
                &st.mtp_e,
                i * row,
                row,
            )?;
            kernels::copy_d2d(
                &mut st.mtp_x,
                i * 2 * row + row,
                &st.mtp_h,
                i * row,
                row,
            )?;
        }
        // 5. eh_proj -> st.cur (n rows, n_embd)
        kernels::matmul_q8_0(
            &mut st.cur,
            &mtp.eh_proj,
            &st.mtp_x,
            2 * s.n_embd,
            s.n_embd,
            n_tok,
        )?;
        // 6. re-anchor st.mtp_hidden (wide) on the trunk's last-row res_hc
        kernels::copy_d2d(
            &mut st.mtp_hidden,
            0,
            &rt.res_hc,
            (n - 1) * hrow,
            hrow,
        )?;
        // 7. broadcast st.cur to the HC streams (res_hc, wide, n rows) on
        //    device. Sync first: the re-anchor D2D above (and the previous
        //    draft's) still reads res_hc on the legacy stream, and the
        //    broadcast writes it on PTDS.
        kernels::sync()?;
        kernels::qwen38_broadcast_hc(&mut rt.res_hc, &st.cur, n_tok, s.n_embd, s.n_hc)?;
        // 8. run the MTP block's full-attention + MoE over the chunk
        let Attn::Qwen38(w) = &mtp.layer.attn else {
            st.qwen38 = Some(rt);
            return Err("qwen38 mtp layer without Qwen38 weights".into());
        };
        let il = s.n_exec_layer as usize;
        self.hc_norm(&mut rt, &w.hc_attn.norm, n_tok, true)?;
        self.hc_mix_core(&mut rt, &w.hc_attn, n_tok, true)?;
        let attn = w.attn.as_ref().ok_or("qwen38 mtp layer without attention")?;
        self.eval_attn_layer(st, &mut rt, il, pos0, n_tok, attn)?;
        kernels::qwen38_hc_combine(
            &mut rt.res_hc,
            &rt.attn_out,
            &rt.inj,
            n_tok,
            s.n_embd,
            s.n_hc,
        )?;
        self.hc_norm(&mut rt, &w.hc_ffn.norm, n_tok, true)?;
        self.hc_mix_core(&mut rt, &w.hc_ffn, n_tok, true)?;
        self.eval_moe(st, &mut rt, il, &mtp.layer, w, n_tok)?;
        kernels::qwen38_hc_combine(
            &mut rt.res_hc,
            &rt.cur,
            &rt.inj,
            n_tok,
            s.n_embd,
            s.n_hc,
        )?;
        st.qwen38 = Some(rt);
        Ok(())
    }

    /// Step 1 of an HC mix: grouped rms (per stream) + full-width gamma.
    /// With gamma=None (head mixer) falls back to weightless.
    #[allow(clippy::too_many_arguments)]
    fn hc_norm(
        &self,
        rt: &mut Qwen38Rt,
        w: &DeviceBuf,
        t: u32,
        gamma: bool,
    ) -> Result {
        let s = self.shape;
        kernels::qwen38_hc_norm(
            &mut rt.xn,
            &rt.res_hc,
            if gamma { Some(w) } else { None },
            t,
            s.n_embd,
            s.n_hc,
            s.rms_eps,
        )?;
        Ok(())
    }

    /// Step 2 of an HC mix (xn already filled):
    ///   lo = silu(xn @ down / hc)      (silu via self-gated sigmoid trick)
    ///   gate logits = lo @ up          [rows][hc_dim], kept UN-sigmoided
    ///   inj = xn @ inject              (when requested)
    /// then xn *= sigmoid(gate logits) pointwise (qwen35_sigmoid_gate),
    /// and finally stream-mean collapse into rt.cur.
    ///
    /// `xn` is physically [token][stream][embedding], which is also
    /// [token][hc*n_embd] for the two projections. The projections therefore
    /// run once per token, not once per stream.
    fn hc_mix_core(
        &self,
        rt: &mut Qwen38Rt,
        w: &super::Qwen38Hc,
        t: u32,
        inject: bool,
    ) -> Result {
        let s = self.shape;
        let hc = s.n_hc;
        let dim = hc * s.n_embd;
        let lr = s.hc_low_rank;

        match &w.down {
            MatW::Q8(b) => {
                kernels::matmul_q8_0(&mut rt.lo, b, &rt.xn, dim, lr, t)?
            }
            MatW::Kq(k) => {
                kernels::quantize_q8_k(&mut rt.xq, &rt.xn, dim, t)?;
                kernels::matmul_kq(
                    &mut rt.lo,
                    &k.w,
                    &rt.xq,
                    dim,
                    lr,
                    t,
                    k.row_bytes,
                    k.quant,
                )?
            }
        }
        // Reference order: scale by 1/hc before SiLU.
        kernels::scale(&mut rt.lo, t * lr, 1.0 / hc as f32)?;
        kernels::copy_d2d(&mut rt.gate_full, 0, &rt.lo, 0, t as usize * lr as usize * 4)?;
        kernels::qwen35_sigmoid_gate(&mut rt.lo, &rt.gate_full, t * lr)?;

        // Up projection produces the full-width gate logits [T][hc*n_embd].
        match &w.up {
            MatW::Q8(b) => {
                kernels::matmul_q8_0(&mut rt.gated, b, &rt.lo, lr, dim, t)?
            }
            MatW::Kq(k) => {
                kernels::quantize_q8_k(&mut rt.xq2, &rt.lo, lr, t)?;
                kernels::matmul_kq(
                    &mut rt.gated,
                    &k.w,
                    &rt.xq2,
                    lr,
                    dim,
                    t,
                    k.row_bytes,
                    k.quant,
                )?
            }
        }

        // Inject is a token-level projection from all HC streams to hc
        // scatter weights: [T][hc*n_embd] -> [T][hc].
        if inject && w.inject.bytes() > 4 {
            kernels::matmul_f32(&mut rt.inj, &w.inject, &rt.xn, dim, hc, t)?;
        }

        kernels::qwen35_sigmoid_gate(&mut rt.xn, &rt.gated, t * dim)?;
        kernels::qwen38_hc_mean(&mut rt.cur, &rt.xn, t, s.n_embd, hc)?;
        Ok(())
    }

    /* ---- token mixers ---------------------------------------------------- */

    /// Gated DeltaNet layer: recurrences shared with qwen35; SIGMOID output
    /// gate applied after the per-v-head RMS norm (the one math delta).
    fn eval_gdn_layer(
        &self,
        rt: &mut Qwen38Rt,
        il: usize,
        gdn: &super::Qwen38Gdn,
        t: u32,
    ) -> Result {
        let s = self.shape;
        let key_dim = s.ssm_k_heads * s.ssm_state;
        let value_dim = s.ssm_v_heads * s.ssm_state;
        let conv_dim = 2 * key_dim + value_dim;
        let eps = s.rms_eps;

        // all three projections contract over rt.cur ([T][n_embd])
        match &gdn.wqkv {
            MatW::Q8(b) => {
                kernels::matmul_q8_0(&mut rt.gdn_qkv, b, &rt.cur, s.n_embd, conv_dim, t)?
            }
            MatW::Kq(k) => kernels::matmul_kq(
                &mut rt.gdn_qkv, &k.w, &rt.xq2, s.n_embd, conv_dim, t, k.row_bytes, k.quant,
            )?,
        }
        match &gdn.wz {
            MatW::Q8(b) => {
                kernels::matmul_q8_0(&mut rt.gdn_z, b, &rt.cur, s.n_embd, value_dim, t)?
            }
            MatW::Kq(k) => kernels::matmul_kq(
                &mut rt.gdn_z, &k.w, &rt.xq2, s.n_embd, value_dim, t, k.row_bytes, k.quant,
            )?,
        }
        kernels::matmul_f32(&mut rt.gdn_g, &gdn.alpha_w, &rt.cur, s.n_embd, s.ssm_v_heads, t)?;
        kernels::matmul_f32(&mut rt.gdn_beta, &gdn.beta_w, &rt.cur, s.n_embd, s.ssm_v_heads, t)?;
        kernels::qwen35_gdn_coeffs(
            &mut rt.gdn_g,
            &mut rt.gdn_beta,
            &gdn.a,
            &gdn.dt_bias,
            t,
            s.ssm_v_heads,
        )?;

        let gs = rt.states[il].as_mut().ok_or("qwen38 gdn state missing")?;
        kernels::qwen35_conv_batch(
            &mut rt.gdn_conv_out, &rt.gdn_qkv, &gdn.conv, &mut gs.conv, conv_dim,
            s.ssm_conv_k, t,
        )?;
        kernels::qwen35_split_qkv(
            &mut rt.gdn_gq, &mut rt.gdn_gk, &mut rt.gdn_gv, &rt.gdn_conv_out, t,
            key_dim, value_dim,
        )?;
        kernels::qwen35_l2_norm(&mut rt.gdn_gq, t * s.ssm_k_heads, s.ssm_state, eps)?;
        kernels::qwen35_l2_norm(&mut rt.gdn_gk, t * s.ssm_k_heads, s.ssm_state, eps)?;
        kernels::qwen35_gdn_batch(
            &mut rt.gdn_o, &mut gs.s, &rt.gdn_gq, &rt.gdn_gk, &rt.gdn_gv, &rt.gdn_g,
            &rt.gdn_beta, s.ssm_v_heads, s.ssm_k_heads, s.ssm_state, t,
        )?;
        // gated per-head rms norm THEN sigmoid(z) output gate
        kernels::gqa_head_rms_norm(
            &mut rt.gdn_o,
            Some(&gdn.ssm_norm),
            t * s.ssm_v_heads,
            s.ssm_state,
            eps,
        )?;
        kernels::qwen35_sigmoid_gate(&mut rt.gdn_o, &rt.gdn_z, t * value_dim)?;
        // output projection into rt.attn_out
        match &gdn.ssm_out {
            MatW::Q8(b) => {
                kernels::matmul_q8_0(&mut rt.attn_out, b, &rt.gdn_o, value_dim, s.n_embd, t)?
            }
            MatW::Kq(k) => {
                kernels::quantize_q8_k(&mut rt.midq, &rt.gdn_o, value_dim, t)?;
                kernels::matmul_kq(
                    &mut rt.attn_out, &k.w, &rt.midq, value_dim, s.n_embd, t, k.row_bytes, k.quant,
                )?
            }
        }
        Ok(())
    }

    /// Sigmoid-gated full attention, per-head interleaved [q|gate] fusion,
    /// partial neox rope over rot_dim of head_dim, dense while inside the
    /// provably-dense QSA window.
    fn eval_attn_layer(
        &self,
        st: &mut State,
        rt: &mut Qwen38Rt,
        il: usize,
        pos0: u32,
        t: u32,
        attn: &super::Qwen38Attn,
    ) -> Result {
        let s = self.shape;
        let hd = s.head_dim;
        let eps = s.rms_eps;
        let dense_bound = s.n_idx_topk + s.qsa_ratio.saturating_sub(1) as u32;
        if attn.has_indexer && pos0 + t > dense_bound && s.qsa_ratio > 0 {
            return Err(format!(
                "qwen38: position {} exceeds the provably-dense QSA window \
                 indexer_top_k({}) + ratio({}) - 1; block-sparse selection is \
                 the declared perf pass",
                pos0 + t, s.n_idx_topk, s.qsa_ratio
            )
            .into());
        }
        match &attn.wq {
            MatW::Q8(b) => {
                kernels::matmul_q8_0(
                    &mut rt.qfull, b, &rt.cur, s.n_embd, 2 * s.n_head * hd, t,
                )?
            }
            MatW::Kq(k) => kernels::matmul_kq(
                &mut rt.qfull, &k.w, &rt.xq2, s.n_embd, 2 * s.n_head * hd, t,
                k.row_bytes, k.quant,
            )?,
        }
        // flat (token, head) split of [q | gate]
        kernels::qwen35_split_gate(&mut rt.q, &mut rt.gate_a, &rt.qfull, t * s.n_head, hd)?;
        match (&attn.wk, &attn.wv) {
            (MatW::Q8(bk), MatW::Q8(bv)) => {
                kernels::matmul_q8_0(&mut rt.k, bk, &rt.cur, s.n_embd, s.n_head_kv * hd, t)?;
                kernels::matmul_q8_0(&mut rt.v, bv, &rt.cur, s.n_embd, s.n_head_kv * hd, t)?;
            }
            _ => return Err("qwen38 attn k/v must be q8_0 (file ships q8_0)".into()),
        }
        kernels::gqa_head_rms_norm(&mut rt.q, Some(&attn.q_norm), t * s.n_head, hd, eps)?;
        kernels::gqa_head_rms_norm(&mut rt.k, Some(&attn.k_norm), t * s.n_head_kv, hd, eps)?;
        kernels::gqa_rope(&mut rt.q, t, s.n_head, hd, s.rot_dim, pos0, s.rope_freq_base, None)?;
        kernels::gqa_rope(&mut rt.k, t, s.n_head_kv, hd, s.rot_dim, pos0, s.rope_freq_base, None)?;
        kernels::gqa_kv_append(
            &mut st.kcache[il], &rt.k, t, s.n_head_kv, hd, st.ctx, pos0, st.kvq,
        )?;
        kernels::gqa_kv_append(
            &mut st.vcache[il], &rt.v, t, s.n_head_kv, hd, st.ctx, pos0, st.kvq,
        )?;
        kernels::gqa_attention_rel(
            &mut rt.heads, &rt.q, &st.kcache[il], &st.vcache[il], t, s.n_head,
            s.n_head_kv, hd, st.ctx, pos0, 1.0 / (hd as f32).sqrt(), 0, None, 0,
            st.kvq, None,
        )?;
        // sigmoid output gate over every head dim (raw fused-gate logits)
        kernels::qwen35_sigmoid_gate(&mut rt.heads, &rt.gate_a, t * s.n_head * hd)?;
        match &attn.out {
            MatW::Q8(b) => {
                kernels::matmul_q8_0(
                    &mut rt.attn_out, b, &rt.heads, s.n_head * hd, s.n_embd, t,
                )?
            }
            MatW::Kq(k) => kernels::matmul_kq(
                &mut rt.attn_out, &k.w, &rt.heads, s.n_head * hd, s.n_embd, t,
                k.row_bytes, k.quant,
            )?,
        }
        Ok(())
    }

    /// Shared+routed MoE on rt.cur. Softmax router over SELECTED top-k with
    /// renormalized weights (reference LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
    /// norm_w=true) + scalar-sigmoid-gated shared expert.
    #[allow(clippy::too_many_arguments)]
    fn eval_moe(
        &self,
        st: &mut State,
        rt: &mut Qwen38Rt,
        il: usize,
        l: &super::LayerW,
        wq: &super::Qwen38W,
        t: u32,
    ) -> Result {
        let s = self.shape;
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
            return Err("qwen38 layer without MoE ffn".into());
        };
        // dsv4_moe's tier path reads State.normed (the qwen35 contract:
        // "st.normed still holds the FFN input"), so publish rt.cur there.
        kernels::copy_d2d(
            &mut st.normed,
            0,
            &rt.cur,
            0,
            t as usize * s.n_embd as usize * 4,
        )?;
        kernels::matmul_f32(&mut rt.router_logits, gate_inp, &rt.cur, s.n_embd, s.n_expert, t)?;
        kernels::router_select(
            &mut st.router_selected,
            &mut st.router_weights,
            &rt.router_logits,
            probs_b,
            s.n_expert,
            s.n_expert_used,
            s.expert_weight_scale,
            t,
            1, // softmax mode
            0,
        )?;
        if let Some((sg, su, sd)) = shexp {
            kernels::matmul_q8_0(&mut rt.gate_act, sg, &rt.cur, s.n_embd, s.n_ff_exp, t)?;
            kernels::matmul_q8_0(&mut rt.up_act, su, &rt.cur, s.n_embd, s.n_ff_exp, t)?;
            kernels::swiglu(
                &mut rt.ffn_mid, &rt.gate_act, &rt.up_act, t * s.n_ff_exp, 0.0, 1.0, 0,
            )?;
            kernels::matmul_q8_0(&mut rt.shared_out, sd, &rt.ffn_mid, s.n_ff_exp, s.n_embd, t)?;
            // scalar sigmoid gate ffn_gate_inp_shexp -> per-row logit
            kernels::matmul_f32(&mut rt.shg, &wq.shexp_gate, &rt.cur, s.n_embd, 1, t)?;
            kernels::qwen35_row_sigmoid_scale(&mut rt.shared_out, &rt.shg, t, s.n_embd)?;
        } else {
            kernels::zero(&mut rt.shared_out, (t * s.n_embd) as usize * 4)?;
        }
        // dsv4_moe contracts the gate/up experts against State.xq (q8_K of
        // the FFN input, [n_tok][n_embd]); it builds moe_mid/midq itself.
        // rt.cur must already be published into st.normed (tier copies).
        kernels::quantize_q8_k(&mut st.xq, &rt.cur, s.n_embd, t)?;
        kernels::sync()?;
        let selected = st.router_selected.read_i32((t * s.n_expert_used) as usize)?;
        if std::env::var_os("PULSAR_MTP_DEBUG").is_some()
            && il >= s.n_exec_layer as usize
        {
            let rl = rt
                .router_logits
                .read_f32(t as usize * s.n_expert as usize)?;
            eprintln!(
                "mtp: moe_dbg il={il} router_nan={} sel={selected:?}",
                rl.iter().filter(|x| !x.is_finite()).count()
            );
        }
        self.dsv4_moe(
            st, il, &selected, gate_exps, up_exps, down_exps, 0, t, s.n_embd,
        )?;
        if std::env::var_os("PULSAR_MTP_DEBUG").is_some()
            && il >= s.n_exec_layer as usize
        {
            let n = t as usize * s.n_embd as usize;
            let v = st.moe_out.read_f32(n)?;
            let sh = rt.shared_out.read_f32(n)?;
            eprintln!(
                "mtp: moe_dbg il={il} t={t} moe_out nan={} zero={} first8=[{}] shared_first8=[{}]",
                v.iter().filter(|x| !x.is_finite()).count(),
                v.iter().filter(|x| **x == 0.0).count(),
                v.iter().take(8).map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(" "),
                sh.iter().take(8).map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(" ")
            );
        }
        // dsv4_moe writes into State.moe_out: combine shared + routed there
        // and publish to rt.cur for the HC combine step. add() with the same
        // dst/src is allowed (out aliases a).
        kernels::add(&mut st.ffn_mid, &st.moe_out, &rt.shared_out, t * s.n_embd)?;
        kernels::copy_d2d(
            &mut rt.cur,
            0,
            &st.ffn_mid,
            0,
            t as usize * s.n_embd as usize * 4,
        )?;
        Ok(())
    }

    /* ---- PLE ------------------------------------------------------------- */

    /// PLE n-gram hash embedding block (build_ple): host int64 hash ->
    /// per-row IQ4NL gather from the disk-resident table -> key/value
    /// projections -> grouped norms -> signed-magnitude gate -> dilated
    /// depthwise conv with rolling state -> residual add into res_hc.
    fn eval_ple(
        &self,
        st: &mut State,
        rt: &mut Qwen38Rt,
        chunk: &[u32],
        pos0: u32,
        t: u32,
        il: usize,
    ) -> Result {
        let s = self.shape;
        let hc = s.n_hc;
        let dim = hc * s.n_embd;
        let heads = ple_heads(&s);
        let hd = s.ple_head_dim as usize;
        let table = rt
            .ple_table
            .as_ref()
            .ok_or("qwen38: PLE layer but table reader missing")?;
        let Attn::Qwen38(w) = &self.layers[il].attn else {
            return Err("qwen38 layer without Qwen38 weights".into());
        };
        let Some(ple) = &w.ple else {
            return Err("qwen38: PLE weights missing".into());
        };

        // 1. Host-side n-gram hash. The reference keeps the predecessor
        // history per sequence; Pulsar has one active sequence, so retain
        // the last ngram-1 token IDs in the runtime across forward calls.
        if pos0 != rt.ple_next_pos {
            rt.ple_tokens.clear();
        }
        let history = rt.ple_tokens.clone();
        let ngram = s.ple_ngram as usize;
        let eos = s.ple_eos_tok as i64;
        let img = if s.ple_img_tok != 0 {
            s.ple_img_tok as i64
        } else {
            eos
        };
        let tok_of = |j: usize| -> i64 {
            let x = chunk[j] as i64;
            if s.ple_img_tok != 0 && x == img {
                x
            } else {
                x
            }
        };
        // one reusable row buffer for the whole chunk (the gather used to
        // allocate per row: 16 x 90B Vecs per token)
        let mut raw = vec![0u8; table.row_bytes as usize];
        let mut next_history = history.clone();
        for tt in 0..t as usize {
            let mut ctx = vec![eos; ngram];
            ctx[0] = tok_of(tt);
            for j in 1..ngram {
                ctx[j] = if tt >= j {
                    tok_of(tt - j)
                } else {
                    let back = j - tt;
                    if back <= history.len() {
                        history[history.len() - back] as i64
                    } else {
                        eos
                    }
                };
            }
            let mut cut = false;
            let mut cv = vec![eos; ngram];
            cv[0] = ctx[0];
            for si in 1..ngram {
                cv[si] = if cut { eos } else { ctx[si] };
                if ctx[si] == eos {
                    cut = true;
                }
            }
            for n in 2..=ngram {
                let mut mixed = (cv[0] as u64).wrapping_mul(s.ple_mult[0]);
                for j in 1..n {
                    mixed ^= (cv[j] as u64).wrapping_mul(s.ple_mult[j]);
                }
                let base = (n - 2) * s.ple_heads_per_ngram as usize;
                for g in 0..s.ple_heads_per_ngram as usize {
                    let h_i = base + g;
                    let idx = mixed % s.ple_vocab[h_i] + s.ple_offs[h_i];
                    Self::gather_ple_rows(
                        table,
                        &mut raw,
                        idx as i64,
                        &mut rt.ple_emb_host[(tt * heads + h_i) * hd..][..hd],
                    )?;
                }
            }
            next_history.push(tok_of(tt) as u32);
            if next_history.len() > ngram.saturating_sub(1) {
                let drop = next_history.len() - (ngram - 1);
                next_history.drain(..drop);
            }
        }
        rt.ple_tokens = next_history;
        rt.ple_next_pos = pos0 + t;

        // Stream discipline: the kernels above run on the per-thread
        // default stream, and the only legacy-stream (plain cudaMemcpy)
        // ops are the H2D stage below and the step-4/5 D2Ds. A
        // whole-device sync is needed ONLY between a PTDS kernel's write
        // and a legacy copy touching the same memory: before the
        // gated->ple_val_v copy and before the step-5 copies. The caller
        // syncs after eval_ple, draining this call's PTDS tail so the
        // next token's H2D stage cannot land on buffers the matmuls
        // above still read.
        // 2. stage rows H2D; projections flatten head-major into the row:
        // emb[t] = [row_0 | ... | row_{heads-1}], contraction width n_embd.
        // key/value both carry [n_embd -> dim] and [n_embd -> n_embd].
        let width_t = t as usize * heads * hd;
        rt.ple_emb_dev.write(0, kernels::as_bytes(&rt.ple_emb_host[..width_t]))?;

        // q8_0 buffers (converted at load): key [in -> hc_dim] with
        // in = head_dim*heads = 2560 (= n_embd for this model); value
        // [in -> n_embd]. The contraction width is the FLATTENED head
        // axis of the gathered rows.
        let in_w = (heads * hd) as u32;
        kernels::matmul_q8_0(&mut rt.ple_key_v, &ple.ple_key, &rt.ple_emb_dev, in_w, dim, t)?;
        kernels::matmul_q8_0(&mut rt.ple_val_v, &ple.ple_value, &rt.ple_emb_dev, in_w, s.n_embd, t)?;

        // 3. grouped norms: key normalized IN PLACE over projections;
        // query normalized from the current residual streams.
        kernels::qwen38_hc_norm_inplace(&mut rt.ple_key_v, Some(&ple.norm_key), t, s.n_embd, hc, s.rms_eps)?;
        kernels::qwen38_hc_norm(
            &mut rt.ple_query, &rt.res_hc, Some(&ple.norm_query), t, s.n_embd, hc, s.rms_eps,
        )?;

        // 4. signed score -> magnitude gate -> value broadcast per stream
        //    gated[t][c][e] = value[t][e] * sigmoid(gate_logit)
        kernels::qwen38_ple_score(
            &mut rt.ple_gate, &rt.ple_key_v, &rt.ple_query, t, s.n_embd, hc,
            1.0 / (s.n_embd as f32).sqrt(),
        )?;
        // keep a copy of value*sigmoid(gate) PRE norm_conv for the residual:
        // ple_apply writes into ple_norm; snapshot via copy (bounded).
        kernels::qwen38_ple_apply(
            &mut rt.gated, &rt.ple_val_v, &rt.ple_gate, t, s.n_embd, hc,
        )?;
        kernels::sync()?;
        kernels::copy_d2d(
            &mut rt.ple_val_v, /* pre-norm staged over val buffer */
            0,
            &rt.gated,
            0,
            t as usize * hc as usize * s.n_embd as usize * 4,
        )?;
        kernels::qwen38_hc_norm(
            &mut rt.ple_norm, &rt.gated, Some(&ple.norm_conv), t, s.n_embd, hc, s.rms_eps,
        )?;
        kernels::sync()?;
        // 5. dilated depthwise conv over TIME with rolling history; the
        // padded input is [hist | T] x dim. Kernel emits silu(...) taps.
        let hist_cols = ple_hist_cols(&s);
        let gs = rt.states[il].as_mut().ok_or("qwen38: ple state missing")?;
        let hist_buf = gs.ple_hist.as_mut().ok_or("qwen38: ple hist missing")?;
        kernels::copy_d2d(&mut rt.ple_pad, 0, hist_buf, 0, hist_cols * dim as usize * 4)?;
        kernels::copy_d2d(
            &mut rt.ple_pad,
            hist_cols * dim as usize * 4,
            &rt.ple_norm,
            0,
            t as usize * dim as usize * 4,
        )?;
        kernels::qwen38_ple_conv(
            &mut rt.ple_out,
            &rt.ple_pad,
            &ple.conv1d,
            t,
            dim,
            (hist_cols + t as usize) as u32,
            s.ple_conv_k,
            s.ple_ngram,
        )?;
        // roll history forward by `keep` newest rows of the padded input
        let keep = hist_cols.min(t as usize);
        if keep > 0 {
            kernels::copy_d2d(
                hist_buf,
                0,
                &rt.ple_pad,
                (hist_cols + t as usize - keep) * dim as usize * 4,
                keep * dim as usize * 4,
            )?;
        }

        // 6. residual: res_hc += gated(pre-norm) per stream, then += conv
        // result collapsed identically. Both pieces are [T][dim]/[T][dim]
        // tiles spanning every stream exactly once, so strided adds work.
        kernels::qwen38_residual_add(&mut rt.res_hc, &rt.ple_val_v, t, s.n_embd, hc)?;
        kernels::qwen38_residual_add(&mut rt.res_hc, &rt.ple_out, t, s.n_embd, hc)?;
        let _ = st;
        Ok(())
    }
    /// Read + dequantize one IQ4NL row straight from the shard files,
    /// reusing `raw` as the read buffer (no per-row allocation).
    fn gather_ple_rows(
        table: &PleTable,
        raw: &mut Vec<u8>,
        row: i64,
        out: &mut [f32],
    ) -> Result<()> {
        if row < 0 || row as u64 >= table.rows {
            return Err(format!("PLE row {row} out of range").into());
        }
        let off = table.base + row as u64 * table.row_bytes;
        table
            .file
            .read_exact_at(raw, off)
            .map_err(|e| format!("PLE row read: {e}"))?;
        // per 32-value block: f16 scale + 16 nibble bytes
        for b in 0..(table.head_dim / 32) {
            let bp = &raw[b * 18..b * 18 + 18];
            let d = super::requant::f16_to_f32(u16::from_le_bytes([bp[0], bp[1]]));
            let qs = &bp[2..18];
            let ob = &mut out[b * 32..][..32];
            for j in 0..16 {
                let lo = (qs[j] & 0xf) as usize;
                let hi = (qs[j] >> 4) as usize;
                ob[j] = d * KVALUES_IQ4NL[lo];
                ob[j + 16] = d * KVALUES_IQ4NL[hi];
            }
        }
        Ok(())
    }
}
