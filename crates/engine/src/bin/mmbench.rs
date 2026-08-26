// Quick kernel microbenchmark: matmul throughput at prefill dimensions.
// Build: cargo run --release -p engine --bin mmbench
use std::time::Instant;

fn main() {
    let cases: Vec<(u32, u32, u32, &str, bool)> = vec![
        // in_dim, out_dim, n_tok, label, is_q8
        (4096, 4096, 512, "q8 out_proj 4096x4096x512", true),
        (4096, 256, 512, "q8 q_b 4096x256x512", true),
        (4096, 512, 512, "q8 comp_kv 4096x512x512", true),
        (4096, 128, 512, "q8 kv 4096x128x512", true),
        (4096, 1024, 512, "q8 out_a 4096x1024x512", true),
        (1024, 4096, 512, "q8 attn_out 1024x4096x512", true),
        (4096, 256, 512, "f32 router 4096x256x512", false),
        (4096, 64, 512, "q8 shexp 4096x64x512", true),
        (4096, 64, 512, "f32 hc_fn 4096x64x512", false),
    ];
    let in_dim = 4096usize;
    let out_dim = 4096usize;
    let n_tok = 512usize;
    let mut x = kernels::DeviceBuf::alloc(in_dim * n_tok * 4).unwrap();
    let mut w = kernels::DeviceBuf::alloc(out_dim * in_dim * 4).unwrap();
    let mut out = kernels::DeviceBuf::alloc(out_dim * n_tok * 4).unwrap();
    // fill with deterministic data
    let xh: Vec<f32> = (0..in_dim * n_tok)
        .map(|i| ((i * 2654435761) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    let wh: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i * 40503) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    x.write(0, kernels::as_bytes(&xh)).unwrap();
    w.write(0, kernels::as_bytes(&wh)).unwrap();

    for (id, od, nt, label, is_q8) in cases {
        let mut best = f64::MAX;
        let mut best_gflop = 0.0f64;
        for _ in 0..5 {
            let t = Instant::now();
            if is_q8 {
                kernels::matmul_q8_0(&mut out, &w, &x, id, od, nt).unwrap();
            } else {
                kernels::matmul_f32(&mut out, &w, &x, id, od, nt).unwrap();
            }
            kernels::sync().unwrap();
            let dt = t.elapsed().as_secs_f64();
            if dt < best {
                best = dt;
            }
            best_gflop = 2.0 * id as f64 * od as f64 * nt as f64 / 1e9;
        }
        let topts = best_gflop / best / 1000.0; // GFLOP/s -> TFLOP/s
        println!("{label:34} {best:8.4} sec  {topts:7.2} TFLOP/s");
    }
    // sanity: matmul_f32 tiny
    let mut small = kernels::DeviceBuf::alloc(128 * 8 * 4).unwrap();
    let mut sw = kernels::DeviceBuf::alloc(128 * 8 * 4).unwrap();
    let mut so = kernels::DeviceBuf::alloc(128 * 8 * 4).unwrap();
    let sh: Vec<f32> = (0..128 * 8).map(|i| i as f32 * 0.001).collect();
    small.write(0, kernels::as_bytes(&sh)).unwrap();
    sw.write(0, kernels::as_bytes(&sh)).unwrap();
    kernels::matmul_f32(&mut so, &sw, &small, 128, 8, 8).unwrap();
    kernels::sync().unwrap();
    let vals = so.read_f32(64).unwrap();
    println!(
        "sanity matmul_f32[0]={:.4} (expected ~{:.4})",
        vals[0],
        (0..128)
            .map(|i| i as f32 * 0.001 * i as f32 * 0.001)
            .sum::<f32>()
            * 8.0
            / 128.0
    );
}
