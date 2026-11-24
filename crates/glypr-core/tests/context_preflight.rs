mod common;

use candle_core::{DType, Tensor};
use glypr_core::{DEFAULT_MAX_SEQ_LEN, GlyprError, MediaItem};
use std::time::Instant;

#[test]
fn test_text_exceeds_max_context() -> anyhow::Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let embedder = shared.lock().unwrap_or_else(|e| e.into_inner());

    // Construct a long text with > 8,192 tokens (~9,000 words)
    let long_text = "word ".repeat(9500);

    let start = Instant::now();
    let res = embedder.embed_query(&long_text, "SearchQuery", None);
    let elapsed = start.elapsed();

    assert!(res.is_err(), "Expected error for oversized text query");
    let err = res.unwrap_err();
    assert!(matches!(err, GlyprError::ContextLimitExceeded { .. }));
    assert!(err.to_string().contains("exceeds maximum context limit"));
    assert!(
        elapsed.as_millis() < 500,
        "Should fail fast before running transformer backbone"
    );

    Ok(())
}

#[test]
fn test_batch_text_exceeds_max_context() -> anyhow::Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let embedder = shared.lock().unwrap_or_else(|e| e.into_inner());

    let normal_text = "This is a standard query.";
    let oversized_text = "context ".repeat(9500);

    let res = embedder.embed_batch_queries(&[normal_text, &oversized_text], "SearchQuery", None);
    assert!(
        res.is_err(),
        "Expected batch to fail due to oversized element"
    );
    let err = res.unwrap_err();
    assert!(matches!(err, GlyprError::ContextLimitExceeded { .. }));
    assert!(err.to_string().contains("exceeds maximum context limit"));

    Ok(())
}

#[test]
fn test_multimodal_multiple_images_burst_exceeds_budget() -> anyhow::Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let embedder = shared.lock().unwrap_or_else(|e| e.into_inner());

    if embedder.multimodal().is_none() {
        eprintln!("Skipping multimodal test: mmproj not attached");
        return Ok(());
    }

    // 8 images at 1,120 token budget = 8 * (1120 + 2) = 8,976 tokens (> 8,192)
    let img = image::DynamicImage::new_rgb8(224, 224);
    let items = vec![
        MediaItem::Image {
            image: &img,
            budget: Some(1120),
        };
        8
    ];

    let start = Instant::now();
    let res = embedder.embed_composite(Some("Describe all 8 images"), &items, None);
    let elapsed = start.elapsed();

    assert!(
        res.is_err(),
        "Expected error for multimodal burst exceeding context limit"
    );
    let err = res.unwrap_err();
    assert!(matches!(err, GlyprError::ContextLimitExceeded { .. }));
    assert!(err.to_string().contains("exceeds maximum context limit"));
    assert!(
        elapsed.as_millis() < 50,
        "Preflight check MUST return in < 50ms without executing heavy Vision Tower forward passes (took {elapsed:?})"
    );

    Ok(())
}

#[test]
fn test_multimodal_audio_plus_long_text_exceeds_budget() -> anyhow::Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let embedder = shared.lock().unwrap_or_else(|e| e.into_inner());

    if embedder.multimodal().is_none() {
        return Ok(());
    }

    let audio_samples = vec![0.0f32; 16000];
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
    let err = res.unwrap_err();
    assert!(matches!(err, GlyprError::ContextLimitExceeded { .. }));
    assert!(err.to_string().contains("exceeds maximum context limit"));

    let max_allowed_ms = if cfg!(debug_assertions) { 500 } else { 50 };
    assert!(
        elapsed.as_millis() < max_allowed_ms,
        "Preflight check MUST return fast without executing Conformer forward pass (took {elapsed:?})"
    );

    Ok(())
}

#[test]
fn test_configurable_max_seq_len() -> anyhow::Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());

    let prompt = "This is a query testing dynamic max_seq_len adjustment.";
    let token_count = embedder.tokenizer().encode(prompt)?.len();
    assert!(token_count > 5, "Prompt should produce more than 5 tokens");

    embedder.set_max_seq_len(5);
    let res_fail = embedder.embed_query(prompt, "SearchQuery", None);
    // Restore immediately before asserting so shared fixture stays clean on panic
    embedder.set_max_seq_len(DEFAULT_MAX_SEQ_LEN);

    let err = res_fail.expect_err("Should fail when seq_len > 5");
    assert!(
        err.to_string()
            .contains("exceeds maximum context limit (5 tokens)"),
        "unexpected error: {err}"
    );

    let res_ok = embedder.embed_query(prompt, "SearchQuery", None);
    assert!(
        res_ok.is_ok(),
        "Should succeed after restoring context limit"
    );

    Ok(())
}

#[test]
fn test_model_level_defense_in_depth() -> anyhow::Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let embedder = shared.lock().unwrap_or_else(|e| e.into_inner());

    let max_pos = embedder.model().config.max_position_embeddings;
    let oversized_len = max_pos + 1;
    let dummy_ids = Tensor::zeros((1, oversized_len), DType::U32, embedder.device())?;

    let res = embedder.model().forward(&dummy_ids, None);
    assert!(
        res.is_err(),
        "Direct model forward must reject inputs larger than max_position_embeddings"
    );
    assert!(
        res.unwrap_err()
            .to_string()
            .contains("exceeds model max_position_embeddings")
    );

    Ok(())
}
