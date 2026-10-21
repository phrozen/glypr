use anyhow::Result;
use candle_core::{DType, Device};
use glypr_core::{GlyprEmbedder, cosine_similarity};
use safetensors::SafeTensors;
use std::fs;
use std::path::PathBuf;

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
            "Failed to resolve official Google Safetensors weights for golden parity test: {e}"
        )),
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

#[test]
fn test_golden_parity_queries_and_docs() -> Result<()> {
    let snapshot_dir = match get_snapshot_dir()? {
        Some(dir) => dir,
        None => return Ok(()),
    };

    let golden_dir = get_golden_dir();
    let golden_weights_path = golden_dir.join("baseline_vectors.safetensors");
    let manifest_path = golden_dir.join("manifest.json");

    assert!(
        golden_weights_path.exists() && manifest_path.exists(),
        "Golden vector fixtures missing in {:?}",
        golden_dir
    );

    // Load golden reference tensors
    let golden_bytes = fs::read(&golden_weights_path)?;
    let golden_tensors = SafeTensors::deserialize(&golden_bytes)?;

    let load_golden = |name: &str| -> Result<Vec<f32>> {
        let view = golden_tensors
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("Missing golden tensor {}: {}", name, e))?;
        let f32_slice: &[f32] = bytemuck::cast_slice(view.data());
        Ok(f32_slice.to_vec())
    };

    println!(
        "Initializing GlyprEmbedder on CPU (F32) from {:?}",
        snapshot_dir
    );
    let device = Device::Cpu;
    let embedder = GlyprEmbedder::load(&snapshot_dir, &device, DType::F32)?;

    // Test Case 1: search_query_1
    println!("Testing search_query_1...");
    let query_text = "What causes the northern lights?";
    let rust_768 = embedder.embed_query(query_text, "SearchQuery", Some(768))?;
    let golden_768 = load_golden("search_query_1_768d")?;

    let sim_to_golden = cosine_similarity(&rust_768, &golden_768);
    println!(
        "  search_query_1 (768d) Cosine similarity to Golden: {:.6}",
        sim_to_golden
    );

    let max_abs_diff = rust_768
        .iter()
        .zip(golden_768.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  search_query_1 (768d) Max absolute diff: {:.6}",
        max_abs_diff
    );

    // Tightened four-9s parity assertion: cosine similarity with Python golden reference >= 0.9999
    assert!(
        sim_to_golden >= 0.9999,
        "search_query_1 cosine similarity to golden ({:.6}) below four-9s threshold 0.9999",
        sim_to_golden
    );

    // Test Case 2: search_doc_1_titled
    println!("Testing search_doc_1_titled...");
    let doc_text = "The northern lights are caused by charged solar wind particles colliding with Earth atmospheric gases.";
    let doc_title = "Aurora Borealis Explained";
    let rust_doc_768 = embedder.embed_document(doc_text, Some(doc_title), Some(768))?;
    let golden_doc_768 = load_golden("search_doc_1_titled_768d")?;

    let doc_sim_to_golden = cosine_similarity(&rust_doc_768, &golden_doc_768);
    println!(
        "  search_doc_1_titled (768d) Cosine similarity to Golden: {:.6}",
        doc_sim_to_golden
    );
    assert!(
        doc_sim_to_golden >= 0.9999,
        "search_doc_1_titled cosine similarity to golden ({:.6}) below four-9s threshold 0.9999",
        doc_sim_to_golden
    );

    // Test Case 3: Matryoshka Truncation (512d, 256d, 128d)
    for dim in [512, 256, 128] {
        let rust_mrl = embedder.embed_query(query_text, "SearchQuery", Some(dim))?;
        let golden_mrl = load_golden(&format!("search_query_1_{}d", dim))?;
        assert_eq!(rust_mrl.len(), dim);
        let mrl_sim = cosine_similarity(&rust_mrl, &golden_mrl);
        println!(
            "  search_query_1 ({}d) Cosine similarity to Golden: {:.6}",
            dim, mrl_sim
        );
        assert!(
            mrl_sim >= 0.9999,
            "search_query_1 ({}d) cosine similarity ({:.6}) below four-9s threshold 0.9999",
            dim,
            mrl_sim
        );
    }

    // Test Case 4: Semantic Retrieval Cosine Similarity Separation
    let query_doc_sim = cosine_similarity(&rust_768, &rust_doc_768);
    println!(
        "  Query vs Titled Doc semantic similarity: {:.6}",
        query_doc_sim
    );
    assert!(
        query_doc_sim > 0.80,
        "Expected high semantic similarity between query and relevant document (> 0.80), got {:.4}",
        query_doc_sim
    );

    println!("\n✓ All golden parity tests passed with four-9s (>= 0.9999) fidelity!");
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_golden_parity_metal() -> Result<()> {
    let snapshot_dir = match get_snapshot_dir()? {
        Some(dir) => dir,
        None => return Ok(()),
    };
    let golden_dir = get_golden_dir();
    let golden_weights_path = golden_dir.join("baseline_vectors.safetensors");
    assert!(
        golden_weights_path.exists(),
        "baseline_vectors.safetensors missing"
    );

    let golden_bytes = fs::read(&golden_weights_path)?;
    let golden_tensors = SafeTensors::deserialize(&golden_bytes)?;
    let view = golden_tensors.tensor("search_query_1_768d")?;
    let golden_768: Vec<f32> = bytemuck::cast_slice(view.data()).to_vec();

    println!("Initializing GlyprEmbedder on Apple Metal (BF16)...");
    let device = Device::new_metal(0)?;
    let mut embedder = GlyprEmbedder::load(&snapshot_dir, &device, DType::BF16)?;

    let query_text = "What causes the northern lights?";

    embedder.set_attention_mode(glypr_core::AttentionMode::ChunkedSliding);
    let rust_chunked = embedder.embed_query(query_text, "SearchQuery", Some(768))?;
    let sim_chunked = cosine_similarity(&rust_chunked, &golden_768);
    println!(
        "  Metal (BF16, ChunkedSliding) Cosine similarity to Golden: {:.6}",
        sim_chunked
    );
    assert!(
        sim_chunked >= 0.9999,
        "Metal ChunkedSliding cosine similarity to golden ({:.6}) below 0.9999",
        sim_chunked
    );

    #[cfg(feature = "metal-flash-attn")]
    {
        embedder.set_attention_mode(glypr_core::AttentionMode::MetalFlash);
        let rust_flash = embedder.embed_query(query_text, "SearchQuery", Some(768))?;
        let sim_flash = cosine_similarity(&rust_flash, &golden_768);
        println!(
            "  Metal (BF16, MetalFlash)     Cosine similarity to Golden: {:.6}",
            sim_flash
        );
        assert!(
            sim_flash >= 0.9999,
            "MetalFlash cosine similarity to golden ({:.6}) below 0.9999",
            sim_flash
        );
    }

    println!("✓ Metal golden parity passed (>= 0.9999)!");
    Ok(())
}
