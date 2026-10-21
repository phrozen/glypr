use candle_core::{DType, Device, Tensor};
use glypr_core::{DEFAULT_MAX_SEQ_LEN, GlyprEmbedder, MediaItem};
use std::time::Instant;

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

    let base_gguf = glypr_core::hub::resolve_default_gguf()?;
    let mut embedder = GlyprEmbedder::load_gguf(&base_gguf, &device, dtype)?;

    if let Ok(mmproj_gguf) = glypr_core::hub::resolve_default_mmproj()
        && mmproj_gguf.exists()
    {
        embedder.attach_multimodal_projector(&mmproj_gguf)?;
    }

    Ok(embedder)
}

#[test]
fn test_text_exceeds_max_context() -> anyhow::Result<()> {
    let embedder = match setup_embedder() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: model weights not resolved");
            return Ok(());
        }
    };

    // Construct a long text with > 8,192 tokens (~9,000 words)
    let long_text = "word ".repeat(9500);

    let start = Instant::now();
    let res = embedder.embed_query(&long_text, "SearchQuery", None);
    let elapsed = start.elapsed();

    assert!(res.is_err(), "Expected error for oversized text query");
    let err_str = res.unwrap_err().to_string();
    println!("✓ Text exceeds context error: {}", err_str);
    assert!(err_str.contains("exceeds maximum context limit"));
    assert!(
        elapsed.as_millis() < 500,
        "Should fail fast before running transformer backbone"
    );

    Ok(())
}

#[test]
fn test_batch_text_exceeds_max_context() -> anyhow::Result<()> {
    let embedder = match setup_embedder() {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    let normal_text = "This is a standard query.";
    let oversized_text = "context ".repeat(9500);

    let res = embedder.embed_batch_queries(&[normal_text, &oversized_text], "SearchQuery", None);
    assert!(
        res.is_err(),
        "Expected batch to fail due to oversized element"
    );
    let err_str = res.unwrap_err().to_string();
    println!("✓ Batch item exceeds context error: {}", err_str);
    assert!(err_str.contains("Batch item 1"));
    assert!(err_str.contains("exceeds maximum context limit"));

    Ok(())
}

#[test]
fn test_multimodal_multiple_images_burst_exceeds_budget() -> anyhow::Result<()> {
    let embedder = match setup_embedder() {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    if embedder.multimodal.is_none() {
        eprintln!("Skipping multimodal test: mmproj not attached");
        return Ok(());
    }

    // 8 images at 1,120 token budget = 8 * (1120 + 2) = 8,976 tokens (> 8,192)
    let img = image::DynamicImage::new_rgb8(224, 224);
    let items = vec![
        MediaItem::Image {
            image: &img,
            budget: Some(1120),
        },
        MediaItem::Image {
            image: &img,
            budget: Some(1120),
        },
        MediaItem::Image {
            image: &img,
            budget: Some(1120),
        },
        MediaItem::Image {
            image: &img,
            budget: Some(1120),
        },
        MediaItem::Image {
            image: &img,
            budget: Some(1120),
        },
        MediaItem::Image {
            image: &img,
            budget: Some(1120),
        },
        MediaItem::Image {
            image: &img,
            budget: Some(1120),
        },
        MediaItem::Image {
            image: &img,
            budget: Some(1120),
        },
    ];

    let start = Instant::now();
    let res = embedder.embed_composite(Some("Describe all 8 images"), &items, None);
    let elapsed = start.elapsed();

    assert!(
        res.is_err(),
        "Expected error for multimodal burst exceeding context limit"
    );
    let err_str = res.unwrap_err().to_string();
    println!(
        "✓ Multimodal multi-image error: {} (took {:?})",
        err_str, elapsed
    );
    assert!(err_str.contains("Multimodal sequence length"));
    assert!(err_str.contains("exceeds maximum context limit"));

    // Critical assertion: verify preflight stopped execution BEFORE running 8 ViT forwards!
    assert!(
        elapsed.as_millis() < 50,
        "Preflight check MUST return in < 50ms without executing heavy Vision Tower forward passes (took {:?})",
        elapsed
    );

    Ok(())
}

#[test]
fn test_multimodal_audio_plus_long_text_exceeds_budget() -> anyhow::Result<()> {
    let embedder = match setup_embedder() {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    if embedder.multimodal.is_none() {
        return Ok(());
    }

    // 1.0 second audio (27 tokens: 25 soft + 2 framing)
    let audio_samples = vec![0.0f32; 16000];
    // Long prompt with ~8,200 tokens
    let long_prompt = "audio description ".repeat(4500);

    let items = vec![MediaItem::Audio {
        samples: &audio_samples,
    }];

    let start = Instant::now();
    let res = embedder.embed_composite(Some(&long_prompt), &items, None);
    let elapsed = start.elapsed();

    assert!(
        res.is_err(),
        "Expected error for audio + oversized text prompt"
    );
    let err_str = res.unwrap_err().to_string();
    println!(
        "✓ Multimodal audio + text error: {} (took {:?})",
        err_str, elapsed
    );
    assert!(err_str.contains("Multimodal sequence length"));
    assert!(err_str.contains("exceeds maximum context limit"));

    // Critical assertion: verify preflight stopped execution BEFORE running Conformer!
    let max_allowed_ms = if cfg!(debug_assertions) { 500 } else { 50 };
    assert!(
        elapsed.as_millis() < max_allowed_ms,
        "Preflight check MUST return fast without executing Conformer forward pass (took {:?})",
        elapsed
    );

    Ok(())
}

#[test]
fn test_configurable_max_seq_len() -> anyhow::Result<()> {
    let mut embedder = match setup_embedder() {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    let prompt = "This is a query testing dynamic max_seq_len adjustment.";
    let token_count = embedder.tokenizer.encode(prompt)?.len();
    assert!(token_count > 5, "Prompt should produce more than 5 tokens");

    // Restrict max sequence length to 5 tokens
    embedder.set_max_seq_len(5);
    let res_fail = embedder.embed_query(prompt, "SearchQuery", None);
    assert!(res_fail.is_err(), "Should fail when seq_len > 5");
    let err_str = res_fail.unwrap_err().to_string();
    assert!(err_str.contains("exceeds maximum context limit (5 tokens)"));

    // Restore to standard 8,192 limit
    embedder.set_max_seq_len(DEFAULT_MAX_SEQ_LEN);
    let res_ok = embedder.embed_query(prompt, "SearchQuery", None);
    assert!(
        res_ok.is_ok(),
        "Should succeed after restoring context limit"
    );

    Ok(())
}

#[test]
fn test_model_level_defense_in_depth() -> anyhow::Result<()> {
    let embedder = match setup_embedder() {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    // If max_position_embeddings is 262,144 (or 8,192 depending on config), test that passing > max_position_embeddings fails
    let max_pos = embedder.model.config.max_position_embeddings;
    let oversized_len = max_pos + 1;
    let dummy_ids = Tensor::zeros((1, oversized_len), DType::U32, &embedder.device)?;

    let res = embedder.model.forward(&dummy_ids, None);
    assert!(
        res.is_err(),
        "Direct model forward must reject inputs larger than max_position_embeddings"
    );
    let err_str = res.unwrap_err().to_string();
    println!("✓ Model defense-in-depth error: {}", err_str);
    assert!(err_str.contains("exceeds model max_position_embeddings"));

    Ok(())
}
