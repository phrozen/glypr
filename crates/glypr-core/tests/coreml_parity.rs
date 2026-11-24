#![cfg(feature = "coreml")]

mod common;

use candle_core::DType;
use glypr_core::coreml::{CoreMLComputeUnits, CoreMLTextBackbone, CoreMLVisionTower};
use glypr_core::multimodal::{ProcessedImage, preprocess_image};
use glypr_core::{GlyprError, Task, cosine_similarity};
use safetensors::SafeTensors;
use std::path::PathBuf;

fn resolve_model_path(name: &str) -> Option<PathBuf> {
    if let Some(dir) = glypr_core::hub::resolve_coreml_dir(None) {
        let p = dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    for base in &["models/coreml", "../../models/coreml"] {
        let p = PathBuf::from(base).join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

#[test]
fn test_coreml_vision_tower_ane() -> anyhow::Result<()> {
    let p1134_name = "gemma2_vision_tower_p1134_t126_w8a16.mlmodelc";
    let p2394_name = "gemma2_vision_tower_p2394_t266_w8a16.mlmodelc";

    let Some(p1134_path) = resolve_model_path(p1134_name) else {
        eprintln!("Skipping CoreML vision parity: {p1134_name} not found");
        return Ok(());
    };
    let Some(p2394_path) = resolve_model_path(p2394_name) else {
        eprintln!("Skipping CoreML vision parity: {p2394_name} not found");
        return Ok(());
    };

    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mm) = embedder.multimodal() else {
        if common::allow_missing_weights() {
            eprintln!("Skipping CoreML vision parity: mmproj not cached");
            return Ok(());
        }
        anyhow::bail!("mmproj must be attached to compare CoreMLVisionTower vs GGUF VisionTower");
    };
    let Some(ref gguf_vt) = mm.vision_tower else {
        anyhow::bail!("Loaded mmproj does not contain a VisionTower");
    };

    let portrait_path = common::golden_dir().join("portrait_test.png");
    let portrait_img = image::open(&portrait_path)?;

    let tower_140 =
        CoreMLVisionTower::from_path(&p1134_path, CoreMLComputeUnits::CpuAndNeuralEngine)?;
    let tower_280 =
        CoreMLVisionTower::from_path(&p2394_path, CoreMLComputeUnits::CpuAndNeuralEngine)?;

    assert!(tower_140.supports_patches(1134));
    assert_eq!(tower_140.num_soft_tokens_for_patches(1134), Some(126));
    assert!(tower_280.supports_patches(2394));
    assert_eq!(tower_280.num_soft_tokens_for_patches(2394), Some(266));

    // 1. Test both p1134_t126 (Budget 140) and p2394_t266 (Budget 280) against GGUF VisionTower
    for (budget, expected_patches, expected_tokens, tower) in [
        (140usize, 1134usize, 126usize, &tower_140),
        (280usize, 2394usize, 266usize, &tower_280),
    ] {
        let processed = preprocess_image(&portrait_img, budget, 16, 3)?;
        assert_eq!(processed.num_patches, expected_patches);
        assert_eq!(processed.num_soft_tokens, expected_tokens);

        let (ane_buf, ane_shape) = tower.encode_raw(&processed)?;
        assert_eq!(ane_shape, vec![1, expected_tokens, 512]);
        assert_eq!(ane_buf.len(), expected_tokens * 512);
        assert!(
            ane_buf.iter().all(|x| x.is_finite()),
            "CoreML vision output contains non-finite values at budget={budget}"
        );

        let gguf_tensor = gguf_vt.encode(&processed)?.to_dtype(DType::F32)?;
        let gguf_buf = gguf_tensor.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(gguf_buf.len(), ane_buf.len());

        let cos = cosine_similarity(&ane_buf, &gguf_buf);
        assert!(
            cos >= 0.999,
            "CoreMLVisionTower budget={budget} ({expected_patches}p) cosine={cos:.6} < 0.999 vs GGUF VisionTower"
        );
    }

    // 2. Orientation rejection check: landscape (27x42) must be rejected by portrait (42x27) bucket
    let landscape_same_patches = ProcessedImage {
        pixel_values: vec![0.0f32; 1134 * 768],
        position_ids: vec![0u32; 1134 * 2],
        num_patches: 1134,
        num_soft_tokens: 126,
        patch_height: 27,
        patch_width: 42,
        target_height: 27 * 16,
        target_width: 42 * 16,
    };
    let orient_err = tower_140
        .encode_raw(&landscape_same_patches)
        .expect_err("Portrait 42x27 bucket must reject 27x42 landscape grid");
    assert!(
        matches!(
            orient_err,
            GlyprError::UnsupportedCoreMLPatchGrid {
                patch_height: 27,
                patch_width: 42,
                num_patches: 1134,
                ..
            }
        ),
        "unexpected orientation error: {orient_err:?}"
    );

    Ok(())
}

#[test]
fn test_coreml_text_backbone_ane() -> anyhow::Result<()> {
    let name = "gemma2_text_backbone_s512_w8a16.mlmodelc";
    let Some(model_path) = resolve_model_path(name) else {
        eprintln!("Skipping CoreML text parity: {name} not found");
        return Ok(());
    };

    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    let tokenizer = embedder.tokenizer();

    let backbone =
        CoreMLTextBackbone::from_path(&model_path, CoreMLComputeUnits::CpuAndNeuralEngine)?;
    assert_eq!(backbone.seq_len(), 512);

    let golden_bytes = std::fs::read(common::golden_dir().join("baseline_vectors.safetensors"))?;
    let golden_tensors = SafeTensors::deserialize(&golden_bytes)?;

    for (tensor_key, text, task) in [
        (
            "search_query_1_768d",
            "What causes the northern lights?",
            Task::SearchQuery,
        ),
        (
            "code_query_1_768d",
            "Fast inverse square root implementation in C",
            Task::CodeRetrieval,
        ),
    ] {
        let embedding = backbone.embed_text(text, tokenizer, Some(task))?;
        assert_eq!(embedding.len(), 768);
        assert!(
            embedding.iter().all(|x| x.is_finite()),
            "CoreML text output for {tensor_key} contains non-finite values"
        );
        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Embedding norm {norm} is not 1.0 for {tensor_key}"
        );

        let view = golden_tensors.tensor(tensor_key)?;
        let golden_vec: &[f32] = bytemuck::cast_slice(view.data());
        let cos = cosine_similarity(&embedding, golden_vec);
        assert!(
            cos >= 0.995,
            "CoreMLTextBackbone {tensor_key} cosine={cos:.6} < 0.995 vs baseline_vectors"
        );
    }

    Ok(())
}
