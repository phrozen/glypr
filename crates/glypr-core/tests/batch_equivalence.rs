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

#[test]
fn test_multimodal_batch_equivalence() -> Result<()> {
    let gguf_path = match glypr_core::hub::resolve_default_gguf() {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };
    let mmproj_path = match glypr_core::hub::resolve_canonical_mmproj(false) {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    println!("Initializing GlyprEmbedder with GGUF BF16 on {:?}...", device);
    let mut embedder = GlyprEmbedder::load_gguf_bf16(&gguf_path, &device)?;
    embedder.attach_multimodal_projector(&mmproj_path)?;

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf();
    let img_path = manifest_dir.join("tests/golden_vectors/multimodal_test_image.png");
    let audio_path = manifest_dir.join("tests/golden_vectors/multimodal_test_audio.wav");

    if !img_path.exists() || !audio_path.exists() {
        println!("Skipping multimodal batch test (fixtures missing)");
        return Ok(());
    }

    let img = image::open(&img_path)?;
    let mut reader = hound::WavReader::open(&audio_path)?;
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    // 1. Single individual embeddings
    let single_img = embedder.embed_image_with_budget(&img, None, 140)?;
    let single_text = embedder.embed_interleaved(
        &[glypr_core::ModalityPart::Text("A high fantasy trading card")],
        glypr_core::Task::SearchQuery,
    )?;
    let single_audio = embedder.embed_audio(&samples, None)?;


    // 2. Homogeneous Image Batch (2 copies of image, budget 140)
    println!("\nTesting Homogeneous Image Batch...");
    let homo_batch = vec![
        glypr_core::MultimodalInput::image(&img, Some(140), glypr_core::Task::Raw),
        glypr_core::MultimodalInput::image(&img, Some(140), glypr_core::Task::Raw),
    ];
    let homo_res = embedder.embed_batch(&homo_batch, None)?;
    assert_eq!(homo_res.len(), 2);
    let sim_img0 = cosine_similarity(&single_img, &homo_res[0]);
    let sim_img1 = cosine_similarity(&single_img, &homo_res[1]);
    println!("  Image 0 vs single sim: {:.6}", sim_img0);
    println!("  Image 1 vs single sim: {:.6}", sim_img1);
    assert!(sim_img0 > 0.9999, "Homogeneous image batch sim ({:.6}) < 0.9999", sim_img0);
    assert!(sim_img1 > 0.9999, "Homogeneous image batch sim ({:.6}) < 0.9999", sim_img1);

    // 3. Heterogeneous Multimodal Batch (Image + Text Query + Audio)
    println!("\nTesting Heterogeneous Multimodal Batch (Image + Text + Audio)...");
    let hetero_batch = vec![
        glypr_core::MultimodalInput::image(&img, Some(140), glypr_core::Task::Raw),
        glypr_core::MultimodalInput::text("A high fantasy trading card", glypr_core::Task::SearchQuery),
        glypr_core::MultimodalInput::audio(&samples, glypr_core::Task::Raw),
    ];
    let hetero_res = embedder.embed_batch(&hetero_batch, None)?;
    assert_eq!(hetero_res.len(), 3);

    let sim_het_img = cosine_similarity(&single_img, &hetero_res[0]);
    let sim_het_txt = cosine_similarity(&single_text, &hetero_res[1]);
    let sim_het_aud = cosine_similarity(&single_audio, &hetero_res[2]);

    println!("  Heterogeneous Image vs single sim: {:.6}", sim_het_img);
    println!("  Heterogeneous Text vs single sim:  {:.6}", sim_het_txt);
    println!("  Heterogeneous Audio vs single sim: {:.6}", sim_het_aud);

    assert!(sim_het_img > 0.9995, "Heterogeneous image sim ({:.6}) < 0.9995", sim_het_img);
    assert!(sim_het_txt > 0.9995, "Heterogeneous text sim ({:.6}) < 0.9995", sim_het_txt);
    assert!(sim_het_aud > 0.9995, "Heterogeneous audio sim ({:.6}) < 0.9995", sim_het_aud);

    println!("✓ All multimodal batch equivalence checks passed!");
    Ok(())
}

