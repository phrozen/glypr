use anyhow::Result;
use candle_core::{DType, Device};
use glypr_core::{GlyprEmbedder, PipelineProfile};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
struct LatencyStats {
    median_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    tok_per_sec: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct BucketProfileResult {
    seq_len: usize,
    unprofiled_stats: LatencyStats,
    instrumented_profile: PipelineProfile,
}

#[derive(Debug, Serialize, Deserialize)]
struct BatchThroughputResult {
    batch_size: usize,
    seq_len: usize,
    batch_latency_ms: f64,
    latency_per_item_ms: f64,
    tokens_per_sec: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct DeepProfileReport {
    device: String,
    dtype: String,
    model_load_sec: f64,
    sequence_length_scaling: Vec<BucketProfileResult>,
    batch_throughput_256t: Vec<BatchThroughputResult>,
    batch_throughput_512t: Vec<BatchThroughputResult>,
}

fn get_snapshot_dir() -> PathBuf {
    glypr_core::hub::resolve_google_weights()
        .expect("Failed to resolve Google weights")
        .parent()
        .expect("Snapshot dir has no parent")
        .to_path_buf()
}

fn get_golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/golden_vectors")
}

fn main() -> Result<()> {
    let snapshot_dir = get_snapshot_dir();
    let golden_dir = get_golden_dir();
    let corpus_json_path = golden_dir.join("corpus_8k.json");

    println!("============================================================");
    println!("  glypr Deep Profiler & Layer Bottleneck Analysis");
    println!("============================================================");

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    let dtype = if device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };

    let dev_name = if device.is_metal() {
        "Apple Metal GPU"
    } else {
        "CPU"
    };
    println!("Target Device: {} | Precision: {:?}", dev_name, dtype);

    let t_load = Instant::now();
    let embedder = GlyprEmbedder::load(&snapshot_dir, &device, dtype)?;
    let model_load_sec = t_load.elapsed().as_secs_f64();
    println!("Model Loaded in: {:.3} s\n", model_load_sec);

    // Warmup
    print!("Warming up GPU kernels... ");
    for _ in 0..3 {
        let _ = embedder.embed_query(
            "Warmup query for kernel compilation",
            "SearchQuery",
            Some(768),
        )?;
    }
    println!("Done.\n");

    // Load corpus texts for exact token sizes
    let corpus_bytes = fs::read(&corpus_json_path)?;
    let corpus: serde_json::Value = serde_json::from_slice(&corpus_bytes)?;
    let entries = corpus["entries"].as_array().expect("entries array");

    // 1. Single Sequence Scaling across token lengths: [64, 256, 512, 1024, 2048, 4096, 8192]
    let token_buckets = [64, 256, 512, 1024, 2048, 4096, 8192];
    let mut seq_results = Vec::new();

    println!(
        "------------------------------------------------------------------------------------------------------------------"
    );
    println!("  PART 1: SEQUENCE LENGTH SCALING & OPERATOR BOTTLENECK BREAKDOWN");
    println!(
        "------------------------------------------------------------------------------------------------------------------"
    );
    println!(
        "{:<6} | {:<10} | {:<10} | {:<9} | {:<9} | {:<9} | {:<9} | {:<9} | {:<9} | {:<9}",
        "Tokens",
        "Total Med",
        "Tok/sec",
        "Tokenize",
        "Embed+PLE",
        "RoPE+Mask",
        "Attn QKV",
        "Attn Core",
        "MLP GeGLU",
        "PLE Blk"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------"
    );

    for &b in &token_buckets {
        // Find document entry for this bucket
        let entry = entries
            .iter()
            .find(|e| e["bucket_tokens"].as_u64() == Some(b as u64) && e["type"] == "document")
            .expect("document entry for bucket");
        let content = entry["content"].as_str().unwrap();

        // Measure unprofiled baseline (10 runs, or 3 for 8k)
        let runs = if b >= 4096 { 3 } else { 7 };
        let mut times = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t0 = Instant::now();
            let _ = embedder.embed_document(content, None, Some(768))?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let median = times[times.len() / 2];
        let p95 = times[((times.len() as f64) * 0.95) as usize];
        let min = times[0];
        let tok_per_sec = (b as f64) / (median / 1000.0);

        let unprofiled_stats = LatencyStats {
            median_ms: median,
            p95_ms: p95,
            min_ms: min,
            tok_per_sec,
        };

        // Measure fine-grained instrumented profile
        let formatted = embedder.format_document(content, None);
        let (_, inst_pipe) = embedder.profile_encode_batch(&[&formatted], Some(768))?;

        // Aggregate stage sums across all 24 layers
        let sum_qkv: f64 = inst_pipe.forward.layers.iter().map(|l| l.qkv_proj_ms).sum();
        let sum_attn_core: f64 = inst_pipe
            .forward
            .layers
            .iter()
            .map(|l| l.attn_core_ms)
            .sum();
        let sum_mlp: f64 = inst_pipe.forward.layers.iter().map(|l| l.mlp_ms).sum();
        let sum_ple_blk: f64 = inst_pipe
            .forward
            .layers
            .iter()
            .map(|l| l.ple_block_ms)
            .sum();

        let embed_ple = inst_pipe.forward.embed_tokens_ms + inst_pipe.forward.ple_input_ms;
        let rope_mask = inst_pipe.forward.rope_calc_ms + inst_pipe.forward.mask_build_ms;

        println!(
            "{:<6} | {:7.2} ms | {:7.0} t/s | {:6.2} ms | {:6.2} ms | {:6.2} ms | {:6.2} ms | {:6.2} ms | {:6.2} ms | {:6.2} ms",
            b,
            median,
            tok_per_sec,
            inst_pipe.tokenizer_ms,
            embed_ple,
            rope_mask,
            sum_qkv,
            sum_attn_core,
            sum_mlp,
            sum_ple_blk
        );

        seq_results.push(BucketProfileResult {
            seq_len: b,
            unprofiled_stats,
            instrumented_profile: inst_pipe,
        });
    }

    println!(
        "------------------------------------------------------------------------------------------------------------------\n"
    );

    // 2. Batched Throughput Analysis for typical RAG chunks (256 tokens)
    println!(
        "--------------------------------------------------------------------------------------------------"
    );
    println!("  PART 2: BATCHED INFERENCE THROUGHPUT (Sequence Length = 256 tokens)");
    println!(
        "--------------------------------------------------------------------------------------------------"
    );
    println!(
        "{:<10} | {:<12} | {:<14} | {:<14} | {:<12}",
        "Batch Size", "Total Tokens", "Batch Latency", "Per-Item Lat", "Throughput"
    );
    println!(
        "--------------------------------------------------------------------------------------------------"
    );

    let entry_256 = entries
        .iter()
        .find(|e| e["bucket_tokens"].as_u64() == Some(256) && e["type"] == "document")
        .unwrap();
    let text_256 = entry_256["content"].as_str().unwrap();

    let batch_sizes_256 = [1, 4, 8, 16, 32];
    let mut batch_results_256 = Vec::new();

    for &bs in &batch_sizes_256 {
        let texts: Vec<&str> = vec![text_256; bs];
        let runs = if bs >= 16 { 3 } else { 5 };
        let mut times = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t0 = Instant::now();
            let _ = embedder.embed_batch_documents(&texts, None, Some(768))?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let median_batch = times[times.len() / 2];
        let per_item = median_batch / (bs as f64);
        let total_tok = bs * 256;
        let tps = (total_tok as f64) / (median_batch / 1000.0);

        println!(
            "{:<10} | {:<12} | {:8.2} ms    | {:8.2} ms    | {:7.0} tok/s",
            bs, total_tok, median_batch, per_item, tps
        );

        batch_results_256.push(BatchThroughputResult {
            batch_size: bs,
            seq_len: 256,
            batch_latency_ms: median_batch,
            latency_per_item_ms: per_item,
            tokens_per_sec: tps,
        });
    }

    println!(
        "--------------------------------------------------------------------------------------------------\n"
    );

    // 3. Batched Throughput Analysis for 512 tokens
    println!(
        "--------------------------------------------------------------------------------------------------"
    );
    println!("  PART 3: BATCHED INFERENCE THROUGHPUT (Sequence Length = 512 tokens)");
    println!(
        "--------------------------------------------------------------------------------------------------"
    );
    println!(
        "{:<10} | {:<12} | {:<14} | {:<14} | {:<12}",
        "Batch Size", "Total Tokens", "Batch Latency", "Per-Item Lat", "Throughput"
    );
    println!(
        "--------------------------------------------------------------------------------------------------"
    );

    let entry_512 = entries
        .iter()
        .find(|e| e["bucket_tokens"].as_u64() == Some(512) && e["type"] == "document")
        .unwrap();
    let text_512 = entry_512["content"].as_str().unwrap();

    let batch_sizes_512 = [1, 4, 8, 16];
    let mut batch_results_512 = Vec::new();

    for &bs in &batch_sizes_512 {
        let texts: Vec<&str> = vec![text_512; bs];
        let runs = if bs >= 16 { 2 } else { 4 };
        let mut times = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t0 = Instant::now();
            let _ = embedder.embed_batch_documents(&texts, None, Some(768))?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let median_batch = times[times.len() / 2];
        let per_item = median_batch / (bs as f64);
        let total_tok = bs * 512;
        let tps = (total_tok as f64) / (median_batch / 1000.0);

        println!(
            "{:<10} | {:<12} | {:8.2} ms    | {:8.2} ms    | {:7.0} tok/s",
            bs, total_tok, median_batch, per_item, tps
        );

        batch_results_512.push(BatchThroughputResult {
            batch_size: bs,
            seq_len: 512,
            batch_latency_ms: median_batch,
            latency_per_item_ms: per_item,
            tokens_per_sec: tps,
        });
    }

    println!(
        "--------------------------------------------------------------------------------------------------\n"
    );

    // Save report to JSON
    let report = DeepProfileReport {
        device: dev_name.to_string(),
        dtype: format!("{:?}", dtype),
        model_load_sec,
        sequence_length_scaling: seq_results,
        batch_throughput_256t: batch_results_256,
        batch_throughput_512t: batch_results_512,
    };

    let report_path = golden_dir.join("profile_deep_report.json");
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    println!("✓ Deep profile report saved to {:?}", report_path);

    Ok(())
}
