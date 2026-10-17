use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use glypr_core::{cosine_similarity, GlyprEmbedder};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct CorpusFile {
    buckets: Vec<usize>,
    entries: Vec<CorpusEntry>,
}

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    id: String,
    category: Option<String>,
    #[allow(dead_code)]
    language: Option<String>,
    bucket_tokens: usize,
    #[allow(dead_code)]
    actual_tokens: Option<usize>,
    #[allow(dead_code)]
    domain: String,
    #[serde(rename = "type")]
    entry_type: String,
    title: Option<String>,
    content: String,
    task: Option<String>,
}

fn allow_missing_weights() -> bool {
    std::env::var("GLYPR_ALLOW_MISSING_WEIGHTS").as_deref() == Ok("1")
}

fn get_snapshot_dir() -> Result<Option<PathBuf>> {
    match glypr_core::hub::resolve_google_weights() {
        Ok(p) => Ok(p.parent().map(|d| d.to_path_buf())),
        Err(e) if allow_missing_weights() => {
            eprintln!("Skipping test (GLYPR_ALLOW_MISSING_WEIGHTS=1): {e}");
            Ok(None)
        }
        Err(e) => Err(anyhow::anyhow!(
            "Failed to resolve official Google Safetensors weights: {e}"
        )),
    }
}

#[allow(dead_code)]
fn get_gguf_bf16_path() -> Result<Option<PathBuf>> {
    match glypr_core::hub::resolve_default_gguf() {
        Ok(p) => Ok(Some(p)),
        Err(e) if allow_missing_weights() => {
            eprintln!("Skipping test (GLYPR_ALLOW_MISSING_WEIGHTS=1): {e}");
            Ok(None)
        }
        Err(e) => Err(anyhow::anyhow!("Failed to resolve BF16 GGUF: {e}")),
    }
}

fn get_gguf_q8_path() -> Result<Option<PathBuf>> {
    match glypr_core::hub::resolve_file("ggml-org/embeddinggemma-2-GGUF", "embeddinggemma-2-Q8_0.gguf") {
        Ok(p) => Ok(Some(p)),
        Err(e) if allow_missing_weights() => {
            eprintln!("Skipping test (GLYPR_ALLOW_MISSING_WEIGHTS=1): {e}");
            Ok(None)
        }
        Err(e) => Err(anyhow::anyhow!("Failed to resolve Q8_0 GGUF: {e}")),
    }
}

fn get_golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/golden_vectors")
}

fn run_corpus_validation(
    embedder: &GlyprEmbedder,
    model_name: &str,
    min_threshold: f32,
    avg_threshold: f32,
    max_bucket: Option<usize>,
) -> Result<()> {
    let golden_dir = get_golden_dir();
    let corpus_json_path = golden_dir.join("corpus_8k.json");
    let golden_weights_path = golden_dir.join("corpus_8k_vectors.safetensors");

    assert!(
        corpus_json_path.exists() && golden_weights_path.exists(),
        "Golden corpus files missing in {:?}",
        golden_dir
    );

    let corpus_bytes = fs::read(&corpus_json_path)?;
    let corpus: CorpusFile = serde_json::from_slice(&corpus_bytes)?;

    let golden_bytes = fs::read(&golden_weights_path)?;
    let golden_tensors = SafeTensors::deserialize(&golden_bytes)?;

    let load_golden = |name: &str| -> Result<Vec<f32>> {
        let view = golden_tensors
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("Missing golden tensor {}: {}", name, e))?;
        let f32_slice: &[f32] = bytemuck::cast_slice(view.data());
        Ok(f32_slice.to_vec())
    };

    println!("\n============================================================");
    println!(
        "  Validating Parity: {} (Min >= {:.4}, Avg >= {:.5})",
        model_name, min_threshold, avg_threshold
    );
    println!("============================================================");

    let dimensions = [768, 512, 256, 128];
    let mut total_evaluated = 0usize;
    let mut sum_sim = 0.0f64;
    let mut category_min_sim: HashMap<String, f32> = HashMap::new();
    let mut bucket_min_sim: HashMap<usize, f32> = HashMap::new();
    let mut evaluated_vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for entry in &corpus.entries {
        let bucket = entry.bucket_tokens;
        if let Some(max) = max_bucket {
            // Always include at least one 1024-token entry so sliding window (> 512) is exercised!
            if bucket > max && entry.id != "query_physics_1024t" {
                continue;
            }
        }

        let eid = &entry.id;
        let etype = &entry.entry_type;
        let cat = entry.category.as_deref().unwrap_or("general").to_string();

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
                glypr_core::truncate_mrl(&rust_vec_768, dim)?
            };

            let golden_key = format!("{}_{}d", eid, dim);
            let golden_vec = load_golden(&golden_key)?;

            let sim = cosine_similarity(&rust_vec, &golden_vec);
            sum_sim += sim as f64;

            let min_c = category_min_sim.entry(cat.clone()).or_insert(1.0);
            if sim < *min_c {
                *min_c = sim;
            }

            let min_b = bucket_min_sim.entry(bucket).or_insert(1.0);
            if sim < *min_b {
                *min_b = sim;
            }

            if dim == 768 {
                evaluated_vectors.insert(eid.clone(), rust_vec.clone());
            }

            assert!(
                sim >= min_threshold,
                "Parity failure for {} at {}d on {}: cosine sim = {:.6} (expected >= {:.4})",
                eid,
                dim,
                model_name,
                sim,
                min_threshold
            );
            total_evaluated += 1;
        }
    }

    let avg_sim = (sum_sim / (total_evaluated as f64)) as f32;
    println!(
        "  Evaluated {} vectors (Avg Parity = {:.6}) across categories:",
        total_evaluated, avg_sim
    );
    for (cat, min_s) in &category_min_sim {
        println!("    Category {:<16}: Min Parity = {:.6}", cat, min_s);
    }

    assert!(
        avg_sim >= avg_threshold,
        "Average parity failure on {}: avg_sim = {:.6} (expected >= {:.5})",
        model_name,
        avg_sim,
        avg_threshold
    );

    // Pairwise Ranking Separation 1: Length scaling buckets
    if max_bucket.is_none() {
        println!("\n  Retrieval Ranking Separation across 8k Buckets (768d):");
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
                let margin = pos_sim - neg_sim;
                println!("    Bucket {:4}t: Pos = {:.4} | Neg = {:.4} | Margin = {:+.4}", b, pos_sim, neg_sim, margin);
                assert!(
                    margin > 0.08,
                    "Ranking margin too small at {}t on {}: margin = {:.4}",
                    b,
                    model_name,
                    margin
                );
            }
        }

        // Pairwise Ranking Separation 2: Multilingual 10 Languages
        println!("\n  Retrieval Ranking Separation across 10 Languages (768d):");
        let languages = ["en", "es", "de", "fr", "ja", "zh", "ko", "ru", "el", "ar"];
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
                let margin = pos_sim - neg_sim;
                println!("    Lang [{:<2}]: Pos = {:.4} | Neg = {:.4} | Margin = {:+.4}", lang, pos_sim, neg_sim, margin);
                assert!(
                    margin > 0.15,
                    "Multilingual margin failure for {} on {}: margin = {:.4}",
                    lang,
                    model_name,
                    margin
                );
            }
        }
    }

    println!("\n✓ All {} parity checks passed on {}!\n", total_evaluated, model_name);
    Ok(())
}

#[test]
fn test_corpus_parity_cpu() -> Result<()> {
    let snapshot_dir = match get_snapshot_dir()? {
        Some(d) => d,
        None => return Ok(()),
    };
    let embedder = GlyprEmbedder::load(&snapshot_dir, &Device::Cpu, DType::F32)?;
    run_corpus_validation(&embedder, "Safetensors FP32 (CPU)", 0.9999, 0.99995, Some(64))
}

#[cfg(feature = "metal")]
#[test]
fn test_corpus_parity_safetensors_metal() -> Result<()> {
    let snapshot_dir = match get_snapshot_dir()? {
        Some(d) => d,
        None => return Ok(()),
    };
    let device = Device::new_metal(0)?;
    let mut embedder = GlyprEmbedder::load(&snapshot_dir, &device, DType::BF16)?;
    embedder.set_attention_mode(glypr_core::AttentionMode::ChunkedSliding);
    run_corpus_validation(&embedder, "Safetensors BF16 (Metal ChunkedSliding)", 0.9998, 0.9999, None)?;

    #[cfg(feature = "metal-flash-attn")]
    {
        embedder.set_attention_mode(glypr_core::AttentionMode::MetalFlash);
        run_corpus_validation(&embedder, "Safetensors BF16 (Metal FlashAttention)", 0.9998, 0.9999, None)?;
    }
    Ok(())
}

#[test]
fn test_corpus_parity_gguf_q8_cpu() -> Result<()> {
    let gguf_path = match get_gguf_q8_path()? {
        Some(p) => p,
        None => return Ok(()),
    };
    let device = Device::Cpu;
    let mut embedder = GlyprEmbedder::load_gguf_q8(&gguf_path, &device)?;
    embedder.set_attention_mode(glypr_core::AttentionMode::ChunkedSliding);
    // Full 292 vectors across all buckets up to 8192 tokens
    run_corpus_validation(&embedder, "GGUF Q8_0 (CPU Chunked)", 0.9994, 0.9998, None)
}

#[cfg(feature = "metal")]
#[test]
fn test_corpus_parity_gguf_bf16_metal() -> Result<()> {
    let gguf_path = match get_gguf_bf16_path()? {
        Some(p) => p,
        None => return Ok(()),
    };
    let device = Device::new_metal(0)?;
    let mut embedder = GlyprEmbedder::load_gguf_bf16(&gguf_path, &device)?;
    embedder.set_attention_mode(glypr_core::AttentionMode::ChunkedSliding);
    run_corpus_validation(&embedder, "GGUF BF16 (Metal ChunkedSliding)", 0.9998, 0.9999, None)?;

    #[cfg(feature = "metal-flash-attn")]
    {
        embedder.set_attention_mode(glypr_core::AttentionMode::MetalFlash);
        run_corpus_validation(&embedder, "GGUF BF16 (Metal FlashAttention)", 0.9998, 0.9999, None)?;
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_corpus_parity_gguf_q8_metal() -> Result<()> {
    let gguf_path = match get_gguf_q8_path()? {
        Some(p) => p,
        None => return Ok(()),
    };
    let device = Device::new_metal(0)?;
    let mut embedder = GlyprEmbedder::load_gguf_q8(&gguf_path, &device)?;
    embedder.set_attention_mode(glypr_core::AttentionMode::ChunkedSliding);
    run_corpus_validation(&embedder, "GGUF Q8_0 (Metal ChunkedSliding)", 0.9996, 0.9998, None)?;

    #[cfg(feature = "metal-flash-attn")]
    {
        embedder.set_attention_mode(glypr_core::AttentionMode::MetalFlash);
        run_corpus_validation(&embedder, "GGUF Q8_0 (Metal FlashAttention)", 0.9996, 0.9998, None)?;
    }
    Ok(())
}

#[test]
fn test_long_context_pooling_precision() -> Result<()> {
    let gguf_path = match get_gguf_bf16_path()? {
        Some(p) => p,
        None => return Ok(()),
    };
    let device = if candle_core::utils::metal_is_available() {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    let embedder = GlyprEmbedder::load_gguf_bf16(&gguf_path, &device)?;

    // Test across long contexts: 1024, 2048, 4096, 8192 tokens
    for &seq_len in &[1024, 2048, 4096, 8192] {
        let hidden_dim = 768;
        // Generate reproducible pseudo-random numbers with small non-zero mean and high variance
        let mut raw_vec = Vec::with_capacity(seq_len * hidden_dim);
        for t in 0..seq_len {
            for c in 0..hidden_dim {
                let v = ((t * 7919 + c * 104729) % 10000) as f64 / 5000.0 - 1.0;
                raw_vec.push(v as f32);
            }
        }

        // Compute high-precision F64 ground-truth reference
        let mut ref_mean = vec![0.0f64; hidden_dim];
        for t in 0..seq_len {
            for c in 0..hidden_dim {
                ref_mean[c] += raw_vec[t * hidden_dim + c] as f64;
            }
        }
        for c in 0..hidden_dim {
            ref_mean[c] /= seq_len as f64;
        }
        let norm_f64 = ref_mean.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        let ref_l2: Vec<f32> = ref_mean.iter().map(|v| (v / norm_f64) as f32).collect();

        // Create token embeddings tensor in BF16
        let t_tensor = Tensor::from_vec(raw_vec, (1, seq_len, hidden_dim), &device)?
            .to_dtype(DType::BF16)?;

        // Pool and normalize
        let pooled = embedder.model.pool_and_normalize(&t_tensor, None, None)?;
        let pooled_vec = pooled.to_dtype(DType::F32)?.squeeze(0)?.to_vec1::<f32>()?;

        // Compute cosine similarity vs F64 ground truth
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..hidden_dim {
            dot += pooled_vec[i] * ref_l2[i];
            norm_a += pooled_vec[i] * pooled_vec[i];
            norm_b += ref_l2[i] * ref_l2[i];
        }
        let cos_sim = dot / (norm_a.sqrt() * norm_b.sqrt());
        println!("Long-context seq_len={} pooling cosine vs F64 ground truth: {:.7}", seq_len, cos_sim);
        // Phase 0 baseline characterization: BF16 pooling yields ~0.9992 at 1K, ~0.9993 at 2K, ~0.987 at 4K, and ~0.95 at 8K due to 7-bit mantissa swamping.
        // In Phase 2, casting to FP32 before mean/sum will elevate all of these to > 0.99999.
        assert!(
            cos_sim >= 0.950,
            "Pooling precision failure at seq_len={}: cos_sim={:.7}",
            seq_len,
            cos_sim
        );
    }
    Ok(())
}

