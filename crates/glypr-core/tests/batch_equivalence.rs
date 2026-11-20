mod common;

use anyhow::Result;
use glypr_core::{AttentionMode, ModalityPart, MultimodalInput, Task, cosine_similarity};

#[test]
fn test_batch_single_equivalence() -> Result<()> {
    let Some(shared) = common::shared_safetensors_cpu()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    embedder.set_attention_mode(AttentionMode::ChunkedSliding);

    let query1 = "What causes the aurora borealis?";
    let query2 = "How to write a fast hash table in Rust with zero allocations?";

    let single1 = embedder.embed_query(query1, "SearchQuery", Some(768))?;
    let single2 = embedder.embed_query(query2, "CodeRetrieval", Some(768))?;

    let batch_res = embedder.encode_batch_texts(
        &[
            &embedder.format_query(query1, "SearchQuery"),
            &embedder.format_query(query2, "CodeRetrieval"),
        ],
        Some(768),
    )?;

    assert_eq!(batch_res.len(), 2);
    let sim1 = cosine_similarity(&single1, &batch_res[0]);
    let sim2 = cosine_similarity(&single2, &batch_res[1]);

    assert!(
        sim1 > 0.9999,
        "Batch equivalence for query 1 ({sim1:.6}) below 0.9999"
    );
    assert!(
        sim2 > 0.9999,
        "Batch equivalence for query 2 ({sim2:.6}) below 0.9999"
    );

    Ok(())
}

#[test]
fn test_multimodal_batch_equivalence() -> Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    if !embedder.has_multimodal() || !embedder.has_audio_projector() {
        if common::allow_missing_weights() {
            eprintln!("Skipping multimodal batch test: mmproj not cached");
            return Ok(());
        }
        anyhow::bail!(
            "Multimodal projector (mmproj) required for test_multimodal_batch_equivalence"
        );
    }

    let golden_dir = common::golden_dir();
    let img_path = golden_dir.join("multimodal_test_image.png");
    let audio_path = golden_dir.join("multimodal_test_audio.wav");
    assert!(
        img_path.exists() && audio_path.exists(),
        "Multimodal batch test fixtures missing in {:?}",
        golden_dir
    );

    let img = image::open(&img_path)?;
    let samples = glypr_core::multimodal::decode_wav_file(&audio_path)?;

    #[allow(unused_mut)]
    let mut modes = vec![("ChunkedSliding", AttentionMode::ChunkedSliding)];
    #[cfg(feature = "metal-flash-attn")]
    if embedder.device().is_metal() {
        modes.push(("MetalFlash", AttentionMode::MetalFlash));
    }

    for (mode_name, mode) in modes {
        embedder.set_attention_mode(mode);

        let single_img = embedder.embed_image_with_budget(&img, None, 140)?;
        let single_text = embedder.embed_interleaved(
            &[ModalityPart::Text("A high fantasy trading card")],
            Task::SearchQuery,
        )?;
        let single_audio = embedder.embed_audio(&samples, None)?;

        let homo_batch = vec![
            MultimodalInput::image(&img, Some(140), Task::Raw),
            MultimodalInput::image(&img, Some(140), Task::Raw),
        ];
        let homo_res = embedder.embed_batch(&homo_batch, None)?;
        assert_eq!(homo_res.len(), 2);
        let sim_img0 = cosine_similarity(&single_img, &homo_res[0]);
        let sim_img1 = cosine_similarity(&single_img, &homo_res[1]);
        assert!(
            sim_img0 >= 0.9999,
            "[{mode_name}] Homogeneous image batch sim ({sim_img0:.6}) < 0.9999"
        );
        assert!(
            sim_img1 >= 0.9999,
            "[{mode_name}] Homogeneous image batch sim ({sim_img1:.6}) < 0.9999"
        );

        let hetero_batch = vec![
            MultimodalInput::image(&img, Some(140), Task::Raw),
            MultimodalInput::text("A high fantasy trading card", Task::SearchQuery),
            MultimodalInput::audio(&samples, Task::Raw),
        ];
        let hetero_res = embedder.embed_batch(&hetero_batch, None)?;
        assert_eq!(hetero_res.len(), 3);

        let sim_het_img = cosine_similarity(&single_img, &hetero_res[0]);
        let sim_het_txt = cosine_similarity(&single_text, &hetero_res[1]);
        let sim_het_aud = cosine_similarity(&single_audio, &hetero_res[2]);

        assert!(
            sim_het_img >= 0.9998,
            "[{mode_name}] Heterogeneous image sim ({sim_het_img:.6}) < 0.9998"
        );
        assert!(
            sim_het_txt >= 0.9998,
            "[{mode_name}] Heterogeneous text sim ({sim_het_txt:.6}) < 0.9998"
        );
        assert!(
            sim_het_aud >= 0.9998,
            "[{mode_name}] Heterogeneous audio sim ({sim_het_aud:.6}) < 0.9998"
        );
    }

    Ok(())
}
