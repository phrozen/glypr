mod common;

use anyhow::Result;
use glypr_core::{AttentionMode, GlyprEmbedder, Task, cosine_similarity};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Deserialize)]
struct ManifestFile {
    supported_dims: Vec<usize>,
    test_cases: HashMap<String, ManifestCase>,
    similarities: HashMap<String, f32>,
}

#[derive(Debug, Deserialize)]
struct ManifestCase {
    #[serde(rename = "type")]
    case_type: String,
    task: Option<String>,
    title: Option<String>,
    text: String,
}

fn assert_full_golden_manifest(
    embedder: &GlyprEmbedder,
    label: &str,
    min_vec_cos: f32,
    max_pair_delta: f32,
) -> Result<()> {
    let golden_dir = common::golden_dir();
    let golden_weights_path = golden_dir.join("baseline_vectors.safetensors");
    let manifest_path = golden_dir.join("manifest.json");

    assert!(
        golden_weights_path.exists() && manifest_path.exists(),
        "Golden vector fixtures missing in {:?}",
        golden_dir
    );

    let manifest: ManifestFile = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    assert_eq!(manifest.test_cases.len(), 7, "expected 7 manifest cases");
    assert_eq!(
        manifest.supported_dims,
        vec![768, 512, 256, 128],
        "expected 4 MRL dims"
    );
    assert_eq!(
        manifest.similarities.len(),
        4,
        "expected 4 pairwise similarities"
    );

    let golden_bytes = fs::read(&golden_weights_path)?;
    let golden_tensors = SafeTensors::deserialize(&golden_bytes)?;
    let load_golden = |name: &str| -> Result<Vec<f32>> {
        let view = golden_tensors
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("Missing golden tensor {name}: {e}"))?;
        Ok(bytemuck::cast_slice::<u8, f32>(view.data()).to_vec())
    };

    let mut embeddings: HashMap<(String, usize), Vec<f32>> = HashMap::new();
    let mut case_ids: Vec<&String> = manifest.test_cases.keys().collect();
    case_ids.sort();

    for case_id in case_ids {
        let case = &manifest.test_cases[case_id];
        let task = if case.case_type == "document" {
            Task::Document {
                title: case.title.as_deref(),
            }
        } else {
            let task_name = case.task.as_deref().unwrap_or("SearchQuery");
            Task::from_name_and_title(task_name, case.title.as_deref())
        };

        for &dim in &manifest.supported_dims {
            let trunc = if dim == 768 { None } else { Some(dim) };
            let rust_vec = embedder.embed_text_with_dim(&case.text, task, trunc)?;
            assert_eq!(rust_vec.len(), dim);

            let key = format!("{case_id}_{dim}d");
            let golden_vec = load_golden(&key)?;
            let sim = cosine_similarity(&rust_vec, &golden_vec);
            assert!(
                sim >= min_vec_cos,
                "[{label}] {key} cosine similarity {sim:.6} < {min_vec_cos:.4}"
            );
            embeddings.insert((case_id.clone(), dim), rust_vec);
        }
    }

    let pair_checks = [
        (
            "search_query_1_vs_doc_1_titled_768d",
            ("search_query_1", 768usize),
            ("search_doc_1_titled", 768usize),
        ),
        (
            "search_query_1_vs_doc_1_titled_512d",
            ("search_query_1", 512usize),
            ("search_doc_1_titled", 512usize),
        ),
        (
            "code_query_1_vs_code_doc_1_768d",
            ("code_query_1", 768usize),
            ("code_doc_1", 768usize),
        ),
        (
            "sim_sentence_1_vs_2_768d",
            ("sim_sentence_1", 768usize),
            ("sim_sentence_2", 768usize),
        ),
    ];

    for (pair_key, (lhs_id, lhs_dim), (rhs_id, rhs_dim)) in pair_checks {
        let expected = manifest.similarities[pair_key];
        let lhs = &embeddings[&(lhs_id.to_string(), lhs_dim)];
        let rhs = &embeddings[&(rhs_id.to_string(), rhs_dim)];
        let actual = cosine_similarity(lhs, rhs);
        let delta = (actual - expected).abs();
        assert!(
            delta <= max_pair_delta,
            "[{label}] pair '{pair_key}': actual={actual:.6}, expected={expected:.6}, delta={delta:.6} > {max_pair_delta}"
        );
    }

    Ok(())
}

#[test]
#[ignore = "Tier 2: CPU Safetensors golden parity"]
fn test_golden_parity_queries_and_docs() -> Result<()> {
    let Some(shared) = common::shared_safetensors_cpu()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    embedder.set_attention_mode(AttentionMode::ChunkedSliding);
    assert_full_golden_manifest(&embedder, "Safetensors F32 (CPU)", 0.9999, 0.005)
}

#[cfg(feature = "metal")]
#[test]
fn test_golden_parity_metal() -> Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    if !embedder.device().is_metal() {
        return Ok(());
    }

    embedder.set_attention_mode(AttentionMode::ChunkedSliding);
    assert_full_golden_manifest(&embedder, "GGUF BF16 (Metal ChunkedSliding)", 0.9998, 0.005)?;

    #[cfg(feature = "metal-flash-attn")]
    {
        embedder.set_attention_mode(AttentionMode::MetalFlash);
        assert_full_golden_manifest(&embedder, "GGUF BF16 (Metal MetalFlash)", 0.9998, 0.005)?;
    }

    Ok(())
}
