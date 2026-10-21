#![cfg(feature = "coreml")]

use candle_core::{DType, Device};
use glypr_core::{GlyprEmbedder, MediaItem, VisionBackendPolicy, coreml::CoreMLComputeUnits, hub};
use image::DynamicImage;
use std::path::Path;

fn setup_embedder() -> anyhow::Result<GlyprEmbedder> {
    let device = if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    let dtype = if device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };

    let base_gguf = hub::resolve_default_gguf()?;
    let mut embedder = GlyprEmbedder::load_gguf(&base_gguf, &device, dtype)?;

    // Attach GGUF multimodal projector if available
    if let Ok(mmproj) = hub::resolve_default_mmproj()
        && mmproj.exists()
    {
        let _ = embedder.attach_multimodal_projector(&mmproj);
    }

    // Attach CoreML Vision Tower
    let candidates = ["models/coreml", "../../models/coreml"];
    for cand in candidates {
        if Path::new(cand).is_dir() {
            let _ = embedder.attach_coreml_vision_dir(cand, CoreMLComputeUnits::CpuAndNeuralEngine);
            break;
        }
    }

    Ok(embedder)
}

#[test]
fn test_hybrid_ane_vision_with_gpu_backbone() -> anyhow::Result<()> {
    let mut embedder = match setup_embedder() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Skipping test: setup failed ({})", e);
            return Ok(());
        }
    };

    if embedder.coreml_vision.is_none() {
        eprintln!("Skipping test: CoreML vision models not found");
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

    println!(
        "Running hybrid multimodal embedding (Vision on ANE, Backbone on {:?})...",
        embedder.device
    );
    let emb_hybrid = embedder.embed_composite(Some("Describe the image"), &items, None)?;

    assert_eq!(emb_hybrid.len(), 768);
    let norm: f32 = emb_hybrid.iter().map(|&x| x * x).sum::<f32>().sqrt();
    println!(
        "✓ Hybrid ANE embedding generated: length {}, L2 norm {:.6}",
        emb_hybrid.len(),
        norm
    );
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "Embedding norm should be ~1.0 in BF16, got {}",
        norm
    );

    let nan_count = emb_hybrid.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, 0, "Hybrid embedding contains NaNs");

    // 2. If GGUF mmproj is attached, compare cosine similarity between ANE and GPU mmproj
    if embedder.multimodal.is_some() {
        embedder.set_vision_backend(VisionBackendPolicy::Metal);
        println!("Running pure Metal/CPU multimodal embedding for parity check...");
        let emb_metal = embedder.embed_composite(Some("Describe the image"), &items, None)?;

        let sim = glypr_core::cosine_similarity(&emb_hybrid, &emb_metal);
        println!(
            "✓ Parity between Hybrid ANE and Pure Metal/CPU: {:.6} Cosine Similarity",
            sim
        );
        assert!(
            sim > 0.995,
            "Hybrid ANE vs Metal parity failed: {} <= 0.995",
            sim
        );
    }

    Ok(())
}
