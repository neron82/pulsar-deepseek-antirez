// Offset probe: data_offset + expert tensor offsets for sidecar vs trunk
fn main() {
    let path = std::env::args().nth(1).expect("path");
    let shards = gguf::split_shards(std::path::Path::new(&path))
        .unwrap_or_else(|| vec![std::path::PathBuf::from(&path)]);
    let heads: Vec<Vec<u8>> = shards
        .iter()
        .map(|p| {
            let mut f = std::fs::File::open(p).expect("open shard");
            let mut h = vec![0u8; 16 << 20];
            use std::io::Read;
            let n = f.read(&mut h).expect("read head");
            h.truncate(n);
            h
        })
        .collect();
    let bases: Vec<u64> = heads
        .iter()
        .scan(0u64, |acc, h| {
            let g = gguf::Gguf::parse(h).expect("parse head");
            let b = *acc;
            *acc += g
                .tensors
                .iter()
                .map(|t| t.byte_size().unwrap_or(0))
                .sum::<u64>();
            Some(b)
        })
        .collect();
    let g = gguf::Gguf::merge_split(
        heads
            .iter()
            .map(|h| gguf::Gguf::parse(h).expect("parse"))
            .collect(),
        &bases,
    );
    let data_off = g.data_offset;
    let mut by_off: Vec<_> = g.tensors.iter().collect();
    by_off.sort_by_key(|t| t.offset);
    println!("data_offset = {data_off:#x} ({data_off})");
    for t in by_off.iter().take(8) {
        println!(
            "  abs={:#12x} off={:#12x} {:?} {} bytes",
            data_off + t.offset,
            t.offset,
            t.name,
            t.byte_size().unwrap_or(0)
        );
    }
    for t in &g.tensors {
        if t.name.contains("exps") || t.name.contains("ffn_gate_inp") {
            println!(
                "  ABS={:#14x} (rel {:#x}) {} {:?} {} bytes",
                data_off + t.offset,
                t.offset,
                t.name,
                t.ty,
                t.byte_size().unwrap_or(0)
            );
        }
    }
}
