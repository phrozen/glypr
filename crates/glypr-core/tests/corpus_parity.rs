use anyhow::Result;
use candle_core::{DType, Device};
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

fn get_snapshot_dir() -> Option<PathBuf> {
    glypr_core::hub::resolve_google_weights()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
}

#[allow(dead_code)]
fn get_gguf_bf16_path() -> Option<PathBuf> {
    glypr_core::hub::resolve_default_gguf().ok()
}

fn get_gguf_q8_path() -> Option<PathBuf> {
    glypr_core::hub::resolve_file("ggml-org/embeddinggemma-2-GGUF", "embeddinggemma-2-Q8_0.gguf").ok()
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
    max_bucket: Option<usize>,
) -> Result<()> {
    let golden_dir = get_golden_dir();
    let corpus_json_path = golden_dir.join("corpus_8k.json");
    let golden_weights_path = golden_dir.join("corpus_8k_vectors.safetensors");

    if !corpus_json_path.exists() || !golden_weights_path.exists() {
        eprintln!("Skipping corpus parity test: golden files not found");
        return Ok(());
    }

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
    println!("  Validating Parity: {} (Min Threshold: {:.4})", model_name, min_threshold);
    println!("============================================================");

    let dimensions = [768, 512, 256, 128];
    let mut total_evaluated = 0;
    let mut category_min_sim: HashMap<String, f32> = HashMap::new();
    let mut bucket_min_sim: HashMap<usize, f32> = HashMap::new();
    let mut evaluated_vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for entry in &corpus.entries {
        let bucket = entry.bucket_tokens;
        if let Some(max) = max_bucket {
            if bucket > max {
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
                let slice = &rust_vec_768[..dim];
                let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
                slice.iter().map(|x| x / norm).collect::<Vec<f32>>()
            };

            let golden_key = format!("{}_{}d", eid, dim);
            let golden_vec = load_golden(&golden_key)?;

            let sim = cosine_similarity(&rust_vec, &golden_vec);

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

    println!("  Evaluated {} vectors across categories:", total_evaluated);
    for (cat, min_s) in &category_min_sim {
        println!("    Category {:<16}: Min Parity = {:.6}", cat, min_s);
    }

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
    let snapshot_dir = match get_snapshot_dir() {
        Some(d) => d,
        None => return Ok(()),
    };
    let embedder = GlyprEmbedder::load(&snapshot_dir, &Device::Cpu, DType::F32)?;
    run_corpus_validation(&embedder, "Safetensors FP32 (CPU)", 0.995, Some(64))
}

#[cfg(feature = "metal")]
#[test]
fn test_corpus_parity_safetensors_metal() -> Result<()> {
    let snapshot_dir = match get_snapshot_dir() {
        Some(d) => d,
        None => return Ok(()),
    };
    let device = Device::new_metal(0)?;
    let embedder = GlyprEmbedder::load(&snapshot_dir, &device, DType::BF16)?;
    run_corpus_validation(&embedder, "Safetensors BF16 (Apple Metal)", 0.9995, None)
}

#[test]
fn test_corpus_parity_gguf_q8_cpu() -> Result<()> {
    let gguf_path = match get_gguf_q8_path() {
        Some(p) => p,
        None => return Ok(()),
    };
    let device = Device::Cpu;
    let mut embedder = GlyprEmbedder::load_gguf_q8(&gguf_path, &device)?;
    embedder.set_attention_mode(glypr_core::AttentionMode::ChunkedSliding);
    // Full 292 vectors across all buckets up to 8192 tokens
    run_corpus_validation(&embedder, "GGUF Q8_0 (CPU Chunked)", 0.990, None)
}

#[cfg(feature = "metal")]
#[test]
fn test_corpus_parity_gguf_bf16_metal() -> Result<()> {
    let gguf_path = match get_gguf_bf16_path() {
        Some(p) => p,
        None => return Ok(()),
    };
    let device = Device::new_metal(0)?;
    let embedder = GlyprEmbedder::load_gguf_bf16(&gguf_path, &device)?;
    run_corpus_validation(&embedder, "GGUF BF16 (Apple Metal)", 0.9995, None)
}

#[cfg(feature = "metal")]
#[test]
fn test_corpus_parity_gguf_q8_metal() -> Result<()> {
    let gguf_path = match get_gguf_q8_path() {
        Some(p) => p,
        None => return Ok(()),
    };
    let device = Device::new_metal(0)?;
    let embedder = GlyprEmbedder::load_gguf_q8(&gguf_path, &device)?;
    // Q8 quantization standard: min parity >= 0.990, ranking separation preserved
    run_corpus_validation(&embedder, "GGUF Q8_0 (Apple Metal)", 0.990, None)
}
