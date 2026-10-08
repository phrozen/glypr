use anyhow::Result;
use candle_core::{DType, Device};
use glypr_core::{cosine_similarity, GlyprEmbedder};
use std::path::PathBuf;

fn get_snapshot_dir() -> Option<PathBuf> {
    glypr_core::hub::resolve_google_weights()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
}

#[test]
fn test_batch_single_equivalence() -> Result<()> {
    let snapshot_dir = match get_snapshot_dir() {
        Some(dir) => dir,
        None => return Ok(()),
    };

    println!("Initializing GlyprEmbedder on CPU (F32)...");
    let device = Device::Cpu;
    let embedder = GlyprEmbedder::load(&snapshot_dir, &device, DType::F32)?;

    let query1 = "What causes the aurora borealis?";
    let query2 = "How to write a fast hash table in Rust with zero allocations?";

    // 1. Encode individually
    let single1 = embedder.embed_query(query1, "SearchQuery", Some(768))?;
    let single2 = embedder.embed_query(query2, "CodeRetrieval", Some(768))?;

    // 2. Encode as a batch
    // Note: since query1 and query2 have different tasks, encode formatted texts together:
    let batch_res = embedder.encode_batch_texts(&[
        &embedder.format_query(query1, "SearchQuery"),
        &embedder.format_query(query2, "CodeRetrieval"),
    ], Some(768))?;

    assert_eq!(batch_res.len(), 2);
    let batch1 = &batch_res[0];
    let batch2 = &batch_res[1];

    let sim1 = cosine_similarity(&single1, batch1);
    let sim2 = cosine_similarity(&single2, batch2);

    println!("  Query 1 batch vs single cosine similarity: {:.6}", sim1);
    println!("  Query 2 batch vs single cosine similarity: {:.6}", sim2);

    // Padding mask should preserve numerical equality
    assert!(sim1 > 0.9999, "Batch equivalence for query 1 ({:.6}) below 0.9999", sim1);
    assert!(sim2 > 0.9999, "Batch equivalence for query 2 ({:.6}) below 0.9999", sim2);

    println!("✓ Batch equivalence test passed!");
    Ok(())
}
