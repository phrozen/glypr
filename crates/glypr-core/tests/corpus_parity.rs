mod common;

use anyhow::Result;
use candle_core::{DType, Tensor};
use glypr_core::{AttentionMode, GlyprEmbedder, Task, cosine_similarity};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

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

const SMOKE_SUBSET_IDS: &[&str] = &[
    // 64t / 512t / 1024t / 8192t length & sliding-window coverage
    "query_physics_64t",
    "doc_physics_64t",
    "doc_science_512t",
    "doc_physics_1024t",
    "query_code_1024t",
    "doc_physics_8192t",
    // Diverse task prompts
    "query_task_classification",
    "query_task_clustering",
    "query_task_qa",
    "query_task_factchecking",
    "query_task_coderetrieval",
    "query_task_similarity",
    "doc_task_titled",
    // Multilingual scripts (Latin en, CJK ja, RTL ar)
    "query_lang_en",
    "doc_lang_ja",
    "doc_lang_ar",
];

fn run_corpus_validation(
    embedder: &GlyprEmbedder,
    model_name: &str,
    min_threshold: f32,
    avg_threshold: f32,
    id_filter: Option<&[&str]>,
) -> Result<()> {
    let golden_dir = common::golden_dir();
    let corpus_json_path = golden_dir.join("corpus_8k.json");
    let golden_weights_path = golden_dir.join("corpus_8k_vectors.safetensors");

    assert!(
        corpus_json_path.exists() && golden_weights_path.exists(),
        "Golden corpus files missing in {:?}",
        golden_dir
    );

    let corpus: CorpusFile = serde_json::from_slice(&fs::read(&corpus_json_path)?)?;
    let golden_bytes = fs::read(&golden_weights_path)?;
    let golden_tensors = SafeTensors::deserialize(&golden_bytes)?;

    let load_golden = |name: &str| -> Result<Vec<f32>> {
        let view = golden_tensors
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("Missing golden tensor {name}: {e}"))?;
        Ok(bytemuck::cast_slice::<u8, f32>(view.data()).to_vec())
    };

    let dimensions = [768usize, 512, 256, 128];
    let mut total_evaluated = 0usize;
    let mut sum_sim = 0.0f64;
    let mut category_min_sim: HashMap<String, f32> = HashMap::new();
    let mut evaluated_vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for entry in &corpus.entries {
        if let Some(allowed) = id_filter
            && !allowed.contains(&entry.id.as_str())
        {
            continue;
        }

        let eid = &entry.id;
        let etype = &entry.entry_type;
        let cat = entry.category.as_deref().unwrap_or("general").to_string();

        let rust_vec_768 = if etype == "document" {
            embedder.embed_text_with_dim(
                &entry.content,
                Task::Document {
                    title: entry.title.as_deref(),
                },
                Some(768),
            )?
        } else {
            let task = Task::from_name_and_title(
                entry.task.as_deref().unwrap_or("SearchQuery"),
                entry.title.as_deref(),
            );
            embedder.embed_text_with_dim(&entry.content, task, Some(768))?
        };

        for &dim in &dimensions {
            let rust_vec = if dim == 768 {
                rust_vec_768.clone()
            } else {
                glypr_core::truncate_mrl(&rust_vec_768, dim)?
            };

            let golden_key = format!("{eid}_{dim}d");
            let golden_vec = load_golden(&golden_key)?;

            let sim = cosine_similarity(&rust_vec, &golden_vec);
            sum_sim += sim as f64;

            let min_c = category_min_sim.entry(cat.clone()).or_insert(1.0);
            if sim < *min_c {
                *min_c = sim;
            }

            if dim == 768 {
                evaluated_vectors.insert(eid.clone(), rust_vec.clone());
            }

            assert!(
                sim >= min_threshold,
                "Parity failure for {eid} (bucket={}) at {dim}d on {model_name}: cosine sim = {sim:.6} (expected >= {min_threshold:.4})",
                entry.bucket_tokens
            );
            total_evaluated += 1;
        }
    }

    if let Some(allowed) = id_filter {
        assert_eq!(
            total_evaluated,
            allowed.len() * dimensions.len(),
            "Expected all {} smoke entries across 4 MRL dims",
            allowed.len()
        );
    }

    let avg_sim = (sum_sim / (total_evaluated as f64)) as f32;
    assert!(
        avg_sim >= avg_threshold,
        "Average parity failure on {model_name}: avg_sim = {avg_sim:.6} (expected >= {avg_threshold:.5})"
    );

    if id_filter.is_none() {
        for &b in &corpus.buckets {
            let q_id = format!("query_physics_{b}t");
            let pos_id = format!("doc_physics_{b}t");
            let neg_id = format!("doc_science_{b}t");

            if let (Some(q), Some(pos), Some(neg)) = (
                evaluated_vectors.get(&q_id),
                evaluated_vectors.get(&pos_id),
                evaluated_vectors.get(&neg_id),
            ) {
                let margin = cosine_similarity(q, pos) - cosine_similarity(q, neg);
                assert!(
                    margin > 0.08,
                    "Ranking margin too small at {b}t on {model_name}: margin = {margin:.4}"
                );
            }
        }

        for lang in ["en", "es", "de", "fr", "ja", "zh", "ko", "ru", "el", "ar"] {
            let q_id = format!("query_lang_{lang}");
            let pos_id = format!("doc_lang_{lang}");
            let neg_id = format!("distractor_lang_{lang}");

            if let (Some(q), Some(pos), Some(neg)) = (
                evaluated_vectors.get(&q_id),
                evaluated_vectors.get(&pos_id),
                evaluated_vectors.get(&neg_id),
            ) {
                let margin = cosine_similarity(q, pos) - cosine_similarity(q, neg);
                assert!(
                    margin > 0.15,
                    "Multilingual margin failure for {lang} on {model_name}: margin = {margin:.4}"
                );
            }
        }
    }

    Ok(())
}

#[test]
fn test_corpus_parity_smoke() -> Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    #[allow(unused_mut)]
    let mut mode = AttentionMode::ChunkedSliding;
    #[cfg(feature = "metal-flash-attn")]
    if embedder.device().is_metal() {
        mode = AttentionMode::MetalFlash;
    }
    embedder.set_attention_mode(mode);

    run_corpus_validation(
        &embedder,
        "Tier 1 Corpus Smoke (16 entries x 4 MRL dims)",
        0.9998,
        0.9999,
        Some(SMOKE_SUBSET_IDS),
    )
}

#[test]
#[ignore = "Tier 2: full 292-vector 8k parity sweep"]
fn test_corpus_parity_cpu() -> Result<()> {
    let Some(shared) = common::shared_safetensors_cpu()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    embedder.set_attention_mode(AttentionMode::ChunkedSliding);
    run_corpus_validation(&embedder, "Safetensors FP32 (CPU)", 0.9999, 0.99995, None)
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Tier 2: full 292-vector 8k parity sweep"]
fn test_corpus_parity_safetensors_metal() -> Result<()> {
    let Some(shared) = common::shared_safetensors_metal()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    if !embedder.device().is_metal() {
        return Ok(());
    }

    embedder.set_attention_mode(AttentionMode::ChunkedSliding);
    run_corpus_validation(
        &embedder,
        "Safetensors BF16 (Metal ChunkedSliding)",
        0.9998,
        0.9999,
        None,
    )?;

    #[cfg(feature = "metal-flash-attn")]
    {
        embedder.set_attention_mode(AttentionMode::MetalFlash);
        run_corpus_validation(
            &embedder,
            "Safetensors BF16 (Metal FlashAttention)",
            0.9998,
            0.9999,
            None,
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "Tier 2: full 292-vector 8k parity sweep"]
fn test_corpus_parity_gguf_q8_cpu_full_8k() -> Result<()> {
    let Some(shared) = common::shared_gguf_q8_cpu()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    embedder.set_attention_mode(AttentionMode::ChunkedSliding);
    run_corpus_validation(&embedder, "GGUF Q8_0 (CPU Chunked)", 0.9994, 0.9998, None)
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Tier 2: full 292-vector 8k parity sweep"]
fn test_corpus_parity_gguf_bf16_metal() -> Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    if !embedder.device().is_metal() {
        return Ok(());
    }

    embedder.set_attention_mode(AttentionMode::ChunkedSliding);
    run_corpus_validation(
        &embedder,
        "GGUF BF16 (Metal ChunkedSliding)",
        0.9998,
        0.9999,
        None,
    )?;

    #[cfg(feature = "metal-flash-attn")]
    {
        embedder.set_attention_mode(AttentionMode::MetalFlash);
        run_corpus_validation(
            &embedder,
            "GGUF BF16 (Metal FlashAttention)",
            0.9998,
            0.9999,
            None,
        )?;
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Tier 2: full 292-vector 8k parity sweep"]
fn test_corpus_parity_gguf_q8_metal() -> Result<()> {
    let Some(shared) = common::shared_gguf_q8_metal()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    if !embedder.device().is_metal() {
        return Ok(());
    }

    embedder.set_attention_mode(AttentionMode::ChunkedSliding);
    run_corpus_validation(
        &embedder,
        "GGUF Q8_0 (Metal ChunkedSliding)",
        0.9996,
        0.9998,
        None,
    )?;

    #[cfg(feature = "metal-flash-attn")]
    {
        embedder.set_attention_mode(AttentionMode::MetalFlash);
        run_corpus_validation(
            &embedder,
            "GGUF Q8_0 (Metal FlashAttention)",
            0.9996,
            0.9998,
            None,
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "Tier 2: model-backed 8K pooling precision check (4K BF16 covered in Tier 0 synthetic_model)"]
fn test_long_context_pooling_precision() -> Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    let device = embedder.device();

    for &seq_len in &[1024usize, 2048, 4096, 8192] {
        let hidden_dim = 768;
        let mut raw_vec = Vec::with_capacity(seq_len * hidden_dim);
        for t in 0..seq_len {
            for c in 0..hidden_dim {
                let v = ((t * 7919 + c * 104729) % 10000) as f64 / 5000.0 - 1.0;
                raw_vec.push(v as f32);
            }
        }

        let mut ref_mean = vec![0.0f64; hidden_dim];
        for t in 0..seq_len {
            for c in 0..hidden_dim {
                ref_mean[c] += raw_vec[t * hidden_dim + c] as f64;
            }
        }
        for val in &mut ref_mean {
            *val /= seq_len as f64;
        }
        let norm_f64 = ref_mean
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(1e-12);
        let ref_l2: Vec<f32> = ref_mean.iter().map(|v| (v / norm_f64) as f32).collect();

        let t_tensor =
            Tensor::from_vec(raw_vec, (1, seq_len, hidden_dim), device)?.to_dtype(DType::BF16)?;
        let pooled = embedder.model().pool_and_normalize(&t_tensor, None, None)?;
        let pooled_vec = pooled.to_dtype(DType::F32)?.squeeze(0)?.to_vec1::<f32>()?;
        let cos_sim = cosine_similarity(&pooled_vec, &ref_l2);
        assert!(
            cos_sim >= 0.9999,
            "Pooling precision failure at seq_len={seq_len}: cos_sim={cos_sim:.7}"
        );
    }
    Ok(())
}
