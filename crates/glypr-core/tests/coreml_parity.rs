#![cfg(feature = "coreml")]

use glypr_core::coreml::{CoreMLComputeUnits, CoreMLTextBackbone, CoreMLVisionTower};
use glypr_core::multimodal::ProcessedImage;
use std::path::Path;

#[test]
fn test_coreml_vision_tower_ane() -> anyhow::Result<()> {
    let model_path = "models/coreml/gemma2_vision_tower_p1134_t126_w8a16.mlmodelc";
    if !Path::new(model_path).exists() {
        eprintln!("Skipping test: {} not found", model_path);
        return Ok(());
    }

    let tower = CoreMLVisionTower::from_path(model_path, CoreMLComputeUnits::CpuAndNeuralEngine)?;
    assert!(tower.supports_patches(1134));
    assert_eq!(tower.num_soft_tokens_for_patches(1134), Some(126));

    let dummy_image = ProcessedImage {
        pixel_values: vec![0.0f32; 1134 * 768],
        position_ids: vec![0u32; 1134 * 2],
        num_patches: 1134,
        num_soft_tokens: 126,
        patch_width: 27,
        patch_height: 42,
        target_width: 432,
        target_height: 672,
    };

    let (buf, shape) = tower.encode_raw(&dummy_image)?;
    assert_eq!(shape, vec![1, 126, 512]);
    assert_eq!(buf.len(), 1 * 126 * 512);

    let nan_count = buf.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, 0, "Vision tower output contains NaNs");

    // Candle Tensor conversion
    let device = candle_core::Device::Cpu;
    let tensor = tower.encode_to_tensor(&dummy_image, &device)?;
    assert_eq!(tensor.dims(), &[126, 512]);

    Ok(())
}

#[test]
fn test_coreml_text_backbone_ane() -> anyhow::Result<()> {
    let model_path = "models/coreml/gemma2_text_backbone_s512_w8a16.mlmodelc";
    if !Path::new(model_path).exists() {
        eprintln!("Skipping test: {} not found", model_path);
        return Ok(());
    }

    let backbone = CoreMLTextBackbone::from_path(model_path, CoreMLComputeUnits::CpuAndNeuralEngine)?;
    assert_eq!(backbone.seq_len(), 512);

    let sample_tokens = [2i32, 603, 235292, 1146, 235284, 1489, 235292, 235248, 1841, 6496, 573, 11956, 5028, 235336];
    let embedding = backbone.embed_token_ids(&sample_tokens)?;
    assert_eq!(embedding.len(), 768);

    let nan_count = embedding.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, 0, "Text backbone output contains NaNs");

    let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "Embedding norm {} is not 1.0", norm);

    Ok(())
}
