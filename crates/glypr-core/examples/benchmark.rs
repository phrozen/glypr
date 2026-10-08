use anyhow::Result;
use candle_core::{DType, Device};
use glypr_core::GlyprEmbedder;
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn get_snapshot_dir() -> PathBuf {
    glypr_core::hub::resolve_google_weights()
        .expect("Failed to resolve Google weights")
        .parent()
        .expect("Snapshot dir has no parent")
        .to_path_buf()
}

fn bench_latencies(embedder: &GlyprEmbedder, name: &str) -> Result<serde_json::Value> {
    println!("\n--- Benchmarking Latencies on {} ---", name);

    let test_corpora = [
        ("short_query (~10 words)", "What is the capital and largest city of France?", true),
        (
            "medium_passage (~50 words)",
            "Photosynthesis is the biological process used by plants, algae, and certain bacteria to harness energy from sunlight and turn it into chemical energy. Photosynthesis in green plants involves the chlorophyll pigment absorbing light energy.",
            false,
        ),
        (
            "long_passage (~200 words)",
            "The quick brown fox jumps over the lazy dog in an iterative search paradigm. The quick brown fox jumps over the lazy dog in an iterative search paradigm. The quick brown fox jumps over the lazy dog in an iterative search paradigm. The quick brown fox jumps over the lazy dog in an iterative search paradigm. The quick brown fox jumps over the lazy dog in an iterative search paradigm. The quick brown fox jumps over the lazy dog in an iterative search paradigm. The quick brown fox jumps over the lazy dog in an iterative search paradigm. The quick brown fox jumps over the lazy dog in an iterative search paradigm. The quick brown fox jumps over the lazy dog in an iterative search paradigm. The quick brown fox jumps over the lazy dog in an iterative search paradigm.",
            false,
        ),
    ];

    // Warmup
    for _ in 0..3 {
        let _ = embedder.embed_query("Warmup query", "SearchQuery", Some(768))?;
    }

    let mut results = serde_json::Map::new();

    for (label, text, is_query) in test_corpora {
        let mut times_ms = Vec::with_capacity(30);
        for _ in 0..30 {
            let t0 = Instant::now();
            if is_query {
                let _ = embedder.embed_query(text, "SearchQuery", Some(768))?;
            } else {
                let _ = embedder.embed_document(text, None, Some(768))?;
            }
            let el = t0.elapsed().as_secs_f64() * 1000.0;
            times_ms.push(el);
        }

        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times_ms[times_ms.len() / 2];
        let p95 = times_ms[(times_ms.len() as f64 * 0.95) as usize];
        let min = times_ms[0];

        println!(
            "  {:<30}: Median = {:6.2} ms | P95 = {:6.2} ms | Min = {:6.2} ms",
            label, median, p95, min
        );

        results.insert(
            label.to_string(),
            json!({
                "median_ms": median,
                "p95_ms": p95,
                "min_ms": min,
            }),
        );
    }

    Ok(serde_json::Value::Object(results))
}

fn main() -> Result<()> {
    let snapshot_dir = get_snapshot_dir();
    println!("============================================================");
    println!("  glypr v0.1.0 - Native Rust Performance Benchmark");
    println!("============================================================");

    let mut all_metrics = serde_json::Map::new();

    #[cfg(feature = "metal")]
    {
        println!("\nInitializing Apple Metal (BF16)...");
        let t0 = Instant::now();
        let device = Device::new_metal(0)?;
        let embedder = GlyprEmbedder::load(&snapshot_dir, &device, DType::BF16)?;
        let load_sec = t0.elapsed().as_secs_f64();
        println!("Metal Model Load Time: {:.3} s", load_sec);

        let latencies = bench_latencies(&embedder, "Apple Metal (BF16)")?;
        all_metrics.insert(
            "metal_bf16".to_string(),
            json!({
                "load_time_sec": load_sec,
                "latencies_ms": latencies,
            }),
        );
    }

    println!("\nInitializing CPU (F32)...");
    let t0 = Instant::now();
    let device = Device::Cpu;
    let embedder = GlyprEmbedder::load(&snapshot_dir, &device, DType::F32)?;
    let load_sec = t0.elapsed().as_secs_f64();
    println!("CPU Model Load Time: {:.3} s", load_sec);

    let latencies = bench_latencies(&embedder, "CPU (F32)")?;
    all_metrics.insert(
        "cpu_f32".to_string(),
        json!({
            "load_time_sec": load_sec,
            "latencies_ms": latencies,
        }),
    );

    // Save metrics
    let out_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/golden_vectors")
        .join("rust_metrics.json");
    fs::write(&out_path, serde_json::to_string_pretty(&all_metrics)?)?;
    println!("\n✓ Saved Rust benchmark metrics to {:?}", out_path);

    Ok(())
}
