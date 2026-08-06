// Quick tensor-type dump via the workspace gguf crate
// Usage: cargo run -p gguf --bin list_tensors -- <model.gguf>
fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let shards = gguf::split_shards(std::path::Path::new(&path)).unwrap_or_else(|| vec![std::path::PathBuf::from(&path)]);
    // Read first 16 MB of each shard header and merge
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
            *acc += g.tensors.iter().map(|t| t.byte_size().unwrap_or(0)).sum::<u64>();
            Some(b)
        })
        .collect();
    let g = gguf::Gguf::merge_split(heads.iter().map(|h| gguf::Gguf::parse(h).expect("parse")).collect(), &bases);
    use std::collections::BTreeMap;
    let mut by_type: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for t in &g.tensors {
        by_type
            .entry(format!("{:?}", t.ty))
            .or_default()
            .push(t.name.clone());
    }
    for (ty, names) in &by_type {
        println!("== {} ({} tensors) ==", ty, names.len());
        for n in names {
            if n.contains("compressor") || n.contains("indexer") || n.contains("gate") || n.contains("router") {
                println!("  {}", n);
            }
        }
    }
}
