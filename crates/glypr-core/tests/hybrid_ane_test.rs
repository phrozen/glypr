#![cfg(feature = "coreml")]

mod common;

use glypr_core::{MediaItem, VisionBackendPolicy, coreml::CoreMLComputeUnits, hub};
use image::DynamicImage;
use std::path::PathBuf;

fn find_coreml_dir() -> Option<PathBuf> {
    if let Some(dir) = hub::resolve_coreml_dir(None)
        && dir.is_dir()
    {
        return Some(dir);
    }
    for cand in ["models/coreml", "../../models/coreml"] {
        let p = PathBuf::from(cand);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

#[test]
fn test_hybrid_ane_vision_with_gpu_backbone() -> anyhow::Result<()> {
    let Some(coreml_dir) = find_coreml_dir() else {
        eprintln!("Skipping test: CoreML vision directory not found");
        return Ok(());
    };

    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());

    if embedder.coreml_vision().is_none()
        && let Err(e) =
            embedder.attach_coreml_vision_dir(&coreml_dir, CoreMLComputeUnits::CpuAndNeuralEngine)
    {
        eprintln!("Skipping test: failed to load CoreML vision models ({e})");
        return Ok(());
    }

    // Create a 480x680 portrait image
    let img = DynamicImage::new_rgb8(480, 680);

    // 1. Run with VisionBackendPolicy::Auto (routes to ANE for 140 budget = 1134 patches)
    embedder.set_vision_backend(VisionBackendPolicy::Auto);
    let items = vec![MediaItem::Image {
        image: &img,
        budget: Some(140),
    }];

    let emb_hybrid = embedder.embed_composite(Some("Describe the image"), &items, None)?;

    assert_eq!(emb_hybrid.len(), 768);
    let norm: f32 = emb_hybrid.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "Embedding norm should be ~1.0 in BF16, got {norm}"
    );
    assert!(
        emb_hybrid.iter().all(|x| x.is_finite()),
        "Hybrid embedding contains non-finite values"
    );

    // 2. If GGUF mmproj is attached, compare cosine similarity between ANE and GPU mmproj
    if embedder.multimodal().is_some() {
        embedder.set_vision_backend(VisionBackendPolicy::Metal);
        let emb_metal = embedder.embed_composite(Some("Describe the image"), &items, None)?;
        embedder.set_vision_backend(VisionBackendPolicy::Auto);

        let sim = glypr_core::cosine_similarity(&emb_hybrid, &emb_metal);
        assert!(
            sim > 0.995,
            "Hybrid ANE vs Metal parity failed: {sim} <= 0.995"
        );
    }

    Ok(())
}
