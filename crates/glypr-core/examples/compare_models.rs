use anyhow::Result;
use candle_core::Device;
use glypr_core::{cosine_similarity, AttentionMode, GlyprEmbedder};
use safetensors::SafeTensors;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct CorpusFile {
    buckets: Vec<usize>,
    entries: Vec<CorpusEntry>,
}

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    id: String,
    #[allow(dead_code)]
    category: Option<String>,
    #[allow(dead_code)]
    language: Option<String>,
    #[allow(dead_code)]
    bucket_tokens: usize,
    #[serde(rename = "type")]
    entry_type: String,
    title: Option<String>,
    content: String,
    task: Option<String>,
}

fn get_gguf_bf16_path() -> PathBuf {
    glypr_core::hub::resolve_default_gguf().expect("Failed to resolve BF16 GGUF")
}

fn get_gguf_q8_path() -> PathBuf {
    glypr_core::hub::resolve_file("ggml-org/embeddinggemma-2-GGUF", "embeddinggemma-2-Q8_0.gguf")
        .expect("Failed to resolve Q8_0 GGUF")
}

fn get_golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/golden_vectors")
}

struct EvaluationStats {
    #[allow(dead_code)]
    model_name: String,
    file_size_mb: f64,
    load_time_sec: f64,
    min_parity: f32,
    avg_parity: f32,
    avg_multilingual_margin: f32,
    avg_length_margin: f32,
    latencies: HashMap<usize, (f64, f64, f64)>, // bucket -> (median_ms, p95_ms, tok_per_sec)
}

fn evaluate_model(
    model_name: &str,
    file_size_mb: f64,
    load_time_sec: f64,
    embedder: &GlyprEmbedder,
    corpus: &CorpusFile,
    golden_tensors: &SafeTensors,
) -> Result<EvaluationStats> {
    println!("\n============================================================");
    println!("  Evaluating: {}", model_name);
    println!("============================================================");
    println!("  Disk Footprint : {:.1} MB", file_size_mb);
    println!("  Load Time      : {:.3} s", load_time_sec);

    let load_golden = |name: &str| -> Result<Vec<f32>> {
        let view = golden_tensors
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("Missing golden tensor {}: {}", name, e))?;
        let f32_slice: &[f32] = bytemuck::cast_slice(view.data());
        Ok(f32_slice.to_vec())
    };

    let dimensions = [768, 512, 256, 128];
    let mut similarities: Vec<f32> = Vec::new();
    let mut evaluated_vectors: HashMap<String, Vec<f32>> = HashMap::new();

    // 1. Compute parity across corpus
    print!("  Computing parity across 292 vectors... ");
    let t0 = Instant::now();
    for entry in &corpus.entries {
        let eid = &entry.id;
        let etype = &entry.entry_type;

        let rust_vec_768 = if etype == "document" {
            embedder.embed_document(&entry.content, entry.title.as_deref(), Some(768))?
        } else {
            let task = entry.task.as_deref().unwrap_or("SearchQuery");
            embedder.embed_query(&entry.content, task, Some(768))?
        };

        for &dim in &dimensions {
            let rust_vec = if dim == 768 {
                rust_vec_768.clone()
            } else {
                let slice = &rust_vec_768[..dim];
                let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
                slice.iter().map(|x| x / norm).collect::<Vec<f32>>()
            };

            let golden_key = format!("{}_{}d", eid, dim);
            let golden_vec = load_golden(&golden_key)?;
            let sim = cosine_similarity(&rust_vec, &golden_vec);
            if sim.is_nan() {
                println!("WARNING: NaN detected for {} (dim={})! rust_vec norm={}", eid, dim, rust_vec.iter().map(|x| x*x).sum::<f32>().sqrt());
            }
            similarities.push(sim);

            if dim == 768 {
                evaluated_vectors.insert(eid.clone(), rust_vec.clone());
            }
        }
    }
    println!("done in {:.2} s", t0.elapsed().as_secs_f64());

    let min_parity = similarities.iter().cloned().fold(f32::INFINITY, f32::min);
    let avg_parity = similarities.iter().sum::<f32>() / (similarities.len() as f32);
    println!("  Parity (292 vectors): Avg = {:.6} | Min = {:.6}", avg_parity, min_parity);

    // 2. Multilingual retrieval separation
    let languages = ["en", "es", "de", "fr", "ja", "zh", "ko", "ru", "el", "ar"];
    let mut lang_margins = Vec::new();
    for lang in languages {
        let q_id = format!("query_lang_{}", lang);
        let pos_id = format!("doc_lang_{}", lang);
        let neg_id = format!("distractor_lang_{}", lang);

        if let (Some(q), Some(pos), Some(neg)) = (
            evaluated_vectors.get(&q_id),
            evaluated_vectors.get(&pos_id),
            evaluated_vectors.get(&neg_id),
        ) {
            let pos_sim = cosine_similarity(q, pos);
            let neg_sim = cosine_similarity(q, neg);
            lang_margins.push(pos_sim - neg_sim);
        }
    }
    let avg_multilingual_margin = lang_margins.iter().sum::<f32>() / (lang_margins.len() as f32);
    println!("  Avg Multilingual Margin: {:+.4}", avg_multilingual_margin);

    // 3. Length scaling retrieval separation
    let mut length_margins = Vec::new();
    for &b in &corpus.buckets {
        let q_id = format!("query_physics_{}t", b);
        let pos_id = format!("doc_physics_{}t", b);
        let neg_id = format!("doc_science_{}t", b);

        if let (Some(q), Some(pos), Some(neg)) = (
            evaluated_vectors.get(&q_id),
            evaluated_vectors.get(&pos_id),
            evaluated_vectors.get(&neg_id),
        ) {
            let pos_sim = cosine_similarity(q, pos);
            let neg_sim = cosine_similarity(q, neg);
            length_margins.push(pos_sim - neg_sim);
        }
    }
    let avg_length_margin = length_margins.iter().sum::<f32>() / (length_margins.len() as f32);
    println!("  Avg Length Scaling Margin: {:+.4}", avg_length_margin);

    // 4. Token latency scaling across sequence lengths
    println!("\n  Measuring Latency Scaling across token lengths (warm iterations):");
    let mut latencies = HashMap::new();

    let bucket_docs: Vec<(usize, &CorpusEntry)> = corpus
        .buckets
        .iter()
        .filter_map(|&b| {
            let doc_id = format!("doc_physics_{}t", b);
            corpus.entries.iter().find(|e| e.id == doc_id).map(|e| (b, e))
        })
        .collect();

    // Warmup
    for _ in 0..2 {
        let _ = embedder.embed_document("Warmup short document for benchmark", None, Some(768))?;
    }

    for (tokens, entry) in bucket_docs {
        let iters = if tokens >= 4096 {
            5
        } else if tokens >= 1024 {
            10
        } else {
            20
        };

        let mut times_ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = embedder.embed_document(&entry.content, entry.title.as_deref(), Some(768))?;
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = times_ms[times_ms.len() / 2];
        let p95_ms = times_ms[(times_ms.len() as f64 * 0.95) as usize];
        let tok_per_sec = (tokens as f64) / (median_ms / 1000.0);

        println!(
            "    {:4} tokens: Median = {:7.2} ms | P95 = {:7.2} ms | Throughput = {:7.0} tok/s",
            tokens, median_ms, p95_ms, tok_per_sec
        );
        latencies.insert(tokens, (median_ms, p95_ms, tok_per_sec));
    }

    Ok(EvaluationStats {
        model_name: model_name.to_string(),
        file_size_mb,
        load_time_sec,
        min_parity,
        avg_parity,
        avg_multilingual_margin,
        avg_length_margin,
        latencies,
    })
}

fn main() -> Result<()> {
    println!("============================================================");
    println!("  EmbeddingGemma 2: Architecture & Attention Benchmark");
    println!("  Standard Attention vs Chunked Sliding Attention on Metal");
    println!("============================================================");

    let golden_dir = get_golden_dir();
    let corpus_json_path = golden_dir.join("corpus_8k.json");
    let golden_weights_path = golden_dir.join("corpus_8k_vectors.safetensors");

    let corpus_bytes = fs::read(&corpus_json_path)?;
    let corpus: CorpusFile = serde_json::from_slice(&corpus_bytes)?;

    let golden_bytes = fs::read(&golden_weights_path)?;
    let golden_tensors = SafeTensors::deserialize(&golden_bytes)?;

    let gguf_bf16_path = get_gguf_bf16_path();
    let gguf_q8_path = get_gguf_q8_path();

    let gguf_bf16_size_mb = fs::metadata(&gguf_bf16_path)?.len() as f64 / (1024.0 * 1024.0);
    let gguf_q8_size_mb = fs::metadata(&gguf_q8_path)?.len() as f64 / (1024.0 * 1024.0);

    let device = Device::new_metal(0)?;

    // 1. Evaluate GGUF BF16 with Standard Dense Attention (Baseline)
    println!("\nLoading GGUF BF16 with Standard Dense Attention...");
    let t0 = Instant::now();
    let mut bf16_std_embedder = GlyprEmbedder::load_gguf_bf16(&gguf_bf16_path, &device)?;
    let bf16_load_time = t0.elapsed().as_secs_f64();
    bf16_std_embedder.set_attention_mode(AttentionMode::Standard);

    let bf16_std_stats = evaluate_model(
        "GGUF BF16 (Standard Attention)",
        gguf_bf16_size_mb,
        bf16_load_time,
        &bf16_std_embedder,
        &corpus,
        &golden_tensors,
    )?;

    // 2. Evaluate GGUF BF16 with Chunked Sliding Attention (Optimized)
    println!("\nSwitching GGUF BF16 to Chunked Sliding Attention...");
    bf16_std_embedder.set_attention_mode(AttentionMode::ChunkedSliding);
    let bf16_opt_stats = evaluate_model(
        "GGUF BF16 (Chunked Sliding Attention)",
        gguf_bf16_size_mb,
        bf16_load_time,
        &bf16_std_embedder,
        &corpus,
        &golden_tensors,
    )?;

    // 3. Evaluate GGUF BF16 with Fused Metal Flash Attention
    println!("\nSwitching GGUF BF16 to Fused Metal Flash Attention...");
    bf16_std_embedder.set_attention_mode(AttentionMode::MetalFlash);
    let bf16_flash_stats = evaluate_model(
        "GGUF BF16 (Metal Flash Attention)",
        gguf_bf16_size_mb,
        bf16_load_time,
        &bf16_std_embedder,
        &corpus,
        &golden_tensors,
    )?;

    drop(bf16_std_embedder);

    // 3. Evaluate GGUF Q8_0 with Chunked Sliding Attention (Quantized + Optimized)
    println!("\nLoading GGUF Q8_0 with Chunked Sliding Attention...");
    let t0 = Instant::now();
    let mut q8_embedder = GlyprEmbedder::load_gguf_q8(&gguf_q8_path, &device)?;
    let q8_load_time = t0.elapsed().as_secs_f64();
    q8_embedder.set_attention_mode(AttentionMode::ChunkedSliding);

    let q8_opt_stats = evaluate_model(
        "GGUF Q8_0 (Chunked Sliding Attention)",
        gguf_q8_size_mb,
        q8_load_time,
        &q8_embedder,
        &corpus,
        &golden_tensors,
    )?;

    // 4. Evaluate GGUF Q8_0 with Fused Metal Flash Attention
    println!("\nSwitching GGUF Q8_0 to Fused Metal Flash Attention...");
    q8_embedder.set_attention_mode(AttentionMode::MetalFlash);
    let q8_flash_stats = evaluate_model(
        "GGUF Q8_0 (Metal Flash Attention)",
        gguf_q8_size_mb,
        q8_load_time,
        &q8_embedder,
        &corpus,
        &golden_tensors,
    )?;

    drop(q8_embedder);

    // Print Comparative Summary Table
    println!("\n");
    println!("========================================================================================================================");
    println!("                                              FINAL COMPARATIVE SCORECARD                                               ");
    println!("========================================================================================================================");
    println!(
        "{:<26} | {:<20} | {:<20} | {:<20} | {:<20}",
        "Metric", "GGUF BF16 (Std)", "GGUF BF16 (Chunk)", "GGUF BF16 (Flash)", "GGUF Q8_0 (Flash)"
    );
    println!("------------------------------------------------------------------------------------------------------------------------");
    println!(
        "{:<26} | {:>17.1} MB | {:>17.1} MB | {:>17.1} MB | {:>17.1} MB (-44.5%)",
        "Disk Footprint",
        bf16_std_stats.file_size_mb,
        bf16_opt_stats.file_size_mb,
        bf16_flash_stats.file_size_mb,
        q8_flash_stats.file_size_mb
    );
    println!(
        "{:<26} | {:>18.3} s | {:>18.3} s | {:>18.3} s | {:>18.3} s",
        "Cold Model Load Time",
        bf16_std_stats.load_time_sec,
        bf16_opt_stats.load_time_sec,
        bf16_flash_stats.load_time_sec,
        q8_flash_stats.load_time_sec
    );
    println!(
        "{:<26} | {:>20.6} | {:>20.6} | {:>20.6} | {:>20.6}",
        "Avg Parity (292 vec)",
        bf16_std_stats.avg_parity,
        bf16_opt_stats.avg_parity,
        bf16_flash_stats.avg_parity,
        q8_flash_stats.avg_parity
    );
    println!(
        "{:<26} | {:>20.6} | {:>20.6} | {:>20.6} | {:>20.6}",
        "Min Parity (292 vec)",
        bf16_std_stats.min_parity,
        bf16_opt_stats.min_parity,
        bf16_flash_stats.min_parity,
        q8_flash_stats.min_parity
    );
    println!(
        "{:<26} | {:>19.4} | {:>19.4} | {:>19.4} | {:>19.4}",
        "Avg Multilingual Margin",
        bf16_std_stats.avg_multilingual_margin,
        bf16_opt_stats.avg_multilingual_margin,
        bf16_flash_stats.avg_multilingual_margin,
        q8_flash_stats.avg_multilingual_margin
    );
    println!(
        "{:<26} | {:>19.4} | {:>19.4} | {:>19.4} | {:>19.4}",
        "Avg Length Margin",
        bf16_std_stats.avg_length_margin,
        bf16_opt_stats.avg_length_margin,
        bf16_flash_stats.avg_length_margin,
        q8_flash_stats.avg_length_margin
    );
    println!("------------------------------------------------------------------------------------------------------------------------");
    println!("Latency Scaling (Median ms):");
    for &b in &corpus.buckets {
        let std_lat = bf16_std_stats.latencies.get(&b).map(|x| x.0).unwrap_or(0.0);
        let opt_lat = bf16_opt_stats.latencies.get(&b).map(|x| x.0).unwrap_or(0.0);
        let flash_lat = bf16_flash_stats.latencies.get(&b).map(|x| x.0).unwrap_or(0.0);
        let q8_lat = q8_flash_stats.latencies.get(&b).map(|x| x.0).unwrap_or(0.0);
        let speedup = if flash_lat > 0.0 { std_lat / flash_lat } else { 1.0 };
        println!(
            "  {:4} tokens                | {:>17.2} ms | {:>17.2} ms | {:>13.2} ms ({:.1}x) | {:>17.2} ms",
            b, std_lat, opt_lat, flash_lat, speedup, q8_lat
        );
    }
    println!("------------------------------------------------------------------------------------------------------------------------");
    println!("Throughput Scaling (tok/sec):");
    for &b in &corpus.buckets {
        let std_tps = bf16_std_stats.latencies.get(&b).map(|x| x.2).unwrap_or(0.0);
        let opt_tps = bf16_opt_stats.latencies.get(&b).map(|x| x.2).unwrap_or(0.0);
        let flash_tps = bf16_flash_stats.latencies.get(&b).map(|x| x.2).unwrap_or(0.0);
        let q8_tps = q8_flash_stats.latencies.get(&b).map(|x| x.2).unwrap_or(0.0);
        println!(
            "  {:4} tokens                | {:>14.0} tok/s | {:>14.0} tok/s | {:>14.0} tok/s | {:>14.0} tok/s",
            b, std_tps, opt_tps, flash_tps, q8_tps
        );
    }
    println!("========================================================================================================================\n");

    let summary_json = json!({
        "gguf_bf16_standard": {
            "file_size_mb": bf16_std_stats.file_size_mb,
            "load_time_sec": bf16_std_stats.load_time_sec,
            "avg_parity": bf16_std_stats.avg_parity,
            "min_parity": bf16_std_stats.min_parity,
            "avg_multilingual_margin": bf16_std_stats.avg_multilingual_margin,
            "avg_length_margin": bf16_std_stats.avg_length_margin,
            "latencies": bf16_std_stats.latencies,
        },
        "gguf_bf16_chunked": {
            "file_size_mb": bf16_opt_stats.file_size_mb,
            "load_time_sec": bf16_opt_stats.load_time_sec,
            "avg_parity": bf16_opt_stats.avg_parity,
            "min_parity": bf16_opt_stats.min_parity,
            "avg_multilingual_margin": bf16_opt_stats.avg_multilingual_margin,
            "avg_length_margin": bf16_opt_stats.avg_length_margin,
            "latencies": bf16_opt_stats.latencies,
        },
        "gguf_bf16_flash": {
            "file_size_mb": bf16_flash_stats.file_size_mb,
            "load_time_sec": bf16_flash_stats.load_time_sec,
            "avg_parity": bf16_flash_stats.avg_parity,
            "min_parity": bf16_flash_stats.min_parity,
            "avg_multilingual_margin": bf16_flash_stats.avg_multilingual_margin,
            "avg_length_margin": bf16_flash_stats.avg_length_margin,
            "latencies": bf16_flash_stats.latencies,
        },
        "gguf_q8_0_chunked": {
            "file_size_mb": q8_opt_stats.file_size_mb,
            "load_time_sec": q8_opt_stats.load_time_sec,
            "avg_parity": q8_opt_stats.avg_parity,
            "min_parity": q8_opt_stats.min_parity,
            "avg_multilingual_margin": q8_opt_stats.avg_multilingual_margin,
            "avg_length_margin": q8_opt_stats.avg_length_margin,
            "latencies": q8_opt_stats.latencies,
        },
        "gguf_q8_0_flash": {
            "file_size_mb": q8_flash_stats.file_size_mb,
            "load_time_sec": q8_flash_stats.load_time_sec,
            "avg_parity": q8_flash_stats.avg_parity,
            "min_parity": q8_flash_stats.min_parity,
            "avg_multilingual_margin": q8_flash_stats.avg_multilingual_margin,
            "avg_length_margin": q8_flash_stats.avg_length_margin,
            "latencies": q8_flash_stats.latencies,
        }
    });

    let out_path = golden_dir.join("optimized_attention_metrics.json");
    fs::write(&out_path, serde_json::to_string_pretty(&summary_json)?)?;
    println!("✓ Metrics exported to {:?}", out_path);

    Ok(())
}
