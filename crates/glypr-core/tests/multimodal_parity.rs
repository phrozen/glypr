mod common;

use glypr_core::{AttentionMode, GlyprEmbedder, MediaItem, ModalityPart, Task, cosine_similarity};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct GoldenCase {
    #[allow(dead_code)]
    prompt: Option<String>,
    #[allow(dead_code)]
    query: Option<String>,
    #[allow(dead_code)]
    budget: Option<usize>,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct GoldenData {
    #[serde(default)]
    synthetic_image_file: Option<String>,
    #[serde(default)]
    synthetic_audio_file: Option<String>,
    image_file: String,
    audio_file: String,
    cases: HashMap<String, GoldenCase>,
}

struct MultimodalThresholds {
    /// Audio, Baboon, Budget 560, Interleaved & Composite, Text Queries
    pub high_precision: f32,
    /// Resized Vision >= 140t and music.wav
    pub standard_media: f32,
    /// Extreme 70-token PIL vs image-crate downscale
    pub budget_70: f32,
    /// Minimum average across all 24 golden comparisons
    pub min_average: f32,
}

fn load_golden_data() -> anyhow::Result<(std::path::PathBuf, GoldenData)> {
    let golden_dir = common::golden_dir();
    let golden_path = golden_dir.join("multimodal_golden.json");
    let content = std::fs::read_to_string(&golden_path)?;
    let golden: GoldenData = serde_json::from_str(&content)?;
    Ok((golden_dir, golden))
}

fn run_multimodal_parity_suite(
    embedder: &GlyprEmbedder,
    suite_label: &str,
    thresh: &MultimodalThresholds,
) -> anyhow::Result<()> {
    let (golden_dir, golden) = load_golden_data()?;
    let mut sims = Vec::new();

    // [1-4] Synthetic image & audio
    let synth_img_file = golden
        .synthetic_image_file
        .as_deref()
        .unwrap_or(&golden.image_file);
    let synth_img = image::open(golden_dir.join(synth_img_file))?;

    let synth_img_emb = embedder.embed_image(&synth_img, None)?;
    let sim_synth_img = cosine_similarity(&synth_img_emb, &golden.cases["image_only"].embedding);
    assert!(
        sim_synth_img >= thresh.standard_media,
        "[{suite_label}] sim_synth_img={sim_synth_img}"
    );
    sims.push(sim_synth_img);

    let synth_img_p_emb = embedder.embed_image(&synth_img, Some("Describe this picture"))?;
    let sim_synth_img_p =
        cosine_similarity(&synth_img_p_emb, &golden.cases["image_prompt"].embedding);
    assert!(
        sim_synth_img_p >= thresh.standard_media,
        "[{suite_label}] sim_synth_img_p={sim_synth_img_p}"
    );
    sims.push(sim_synth_img_p);

    let synth_audio_file = golden
        .synthetic_audio_file
        .as_deref()
        .unwrap_or(&golden.audio_file);
    let synth_samples = glypr_core::multimodal::decode_wav_file(golden_dir.join(synth_audio_file))?;

    let synth_aud_emb = embedder.embed_audio(&synth_samples, None)?;
    let sim_synth_aud = cosine_similarity(&synth_aud_emb, &golden.cases["audio_only"].embedding);
    assert!(
        sim_synth_aud >= thresh.standard_media,
        "[{suite_label}] sim_synth_aud={sim_synth_aud}"
    );
    sims.push(sim_synth_aud);

    let synth_aud_p_emb = embedder.embed_audio(&synth_samples, Some("A musical sound"))?;
    let sim_synth_aud_p =
        cosine_similarity(&synth_aud_p_emb, &golden.cases["audio_prompt"].embedding);
    assert!(
        sim_synth_aud_p >= thresh.high_precision,
        "[{suite_label}] sim_synth_aud_p={sim_synth_aud_p}"
    );
    sims.push(sim_synth_aud_p);

    // [5-8] Real image & audio
    let real_img = image::open(golden_dir.join("lenna.png"))?;
    let real_img_emb = embedder.embed_image(&real_img, None)?;
    let sim_real_img = cosine_similarity(&real_img_emb, &golden.cases["real_image_only"].embedding);
    assert!(
        sim_real_img >= thresh.standard_media,
        "[{suite_label}] sim_real_img={sim_real_img}"
    );
    sims.push(sim_real_img);

    let real_img_p_emb = embedder.embed_image(
        &real_img,
        Some("A classic portrait photograph of a woman wearing a hat"),
    )?;
    let sim_real_img_p = cosine_similarity(
        &real_img_p_emb,
        &golden.cases["real_image_prompt"].embedding,
    );
    assert!(
        sim_real_img_p >= thresh.standard_media,
        "[{suite_label}] sim_real_img_p={sim_real_img_p}"
    );
    sims.push(sim_real_img_p);

    let lena_samples = glypr_core::multimodal::decode_wav_file(golden_dir.join("lena_16k.wav"))?;
    let real_aud_emb = embedder.embed_audio(&lena_samples, None)?;
    let sim_real_aud = cosine_similarity(&real_aud_emb, &golden.cases["real_audio_only"].embedding);
    assert!(
        sim_real_aud >= thresh.high_precision,
        "[{suite_label}] sim_real_aud={sim_real_aud}"
    );
    sims.push(sim_real_aud);

    let real_aud_p_emb = embedder.embed_audio(
        &lena_samples,
        Some("A woman speaking English into a microphone"),
    )?;
    let sim_real_aud_p = cosine_similarity(
        &real_aud_p_emb,
        &golden.cases["real_audio_prompt"].embedding,
    );
    assert!(
        sim_real_aud_p >= thresh.high_precision,
        "[{suite_label}] sim_real_aud_p={sim_real_aud_p}"
    );
    sims.push(sim_real_aud_p);

    // [9-12] Extended images
    let baboon_img = image::open(golden_dir.join("baboon.png"))?;
    let baboon_emb = embedder.embed_image(&baboon_img, None)?;
    let sim_baboon = cosine_similarity(&baboon_emb, &golden.cases["baboon_only"].embedding);
    assert!(
        sim_baboon >= thresh.high_precision,
        "[{suite_label}] sim_baboon={sim_baboon}"
    );
    sims.push(sim_baboon);

    let pepper_img = image::open(golden_dir.join("pepper.png"))?;
    let pepper_emb = embedder.embed_image(&pepper_img, None)?;
    let sim_pepper = cosine_similarity(&pepper_emb, &golden.cases["pepper_only"].embedding);
    assert!(
        sim_pepper >= thresh.standard_media,
        "[{suite_label}] sim_pepper={sim_pepper}"
    );
    sims.push(sim_pepper);

    let countryside_img = image::open(golden_dir.join("countryside.png"))?;
    let countryside_emb = embedder.embed_image(&countryside_img, None)?;
    let sim_countryside = cosine_similarity(
        &countryside_emb,
        &golden.cases["countryside_only"].embedding,
    );
    assert!(
        sim_countryside >= thresh.standard_media,
        "[{suite_label}] sim_countryside={sim_countryside}"
    );
    sims.push(sim_countryside);

    let dog_img = image::open(golden_dir.join("dog.png"))?;
    let dog_emb = embedder.embed_image(&dog_img, None)?;
    let sim_dog = cosine_similarity(&dog_emb, &golden.cases["dog_only"].embedding);
    assert!(
        sim_dog >= thresh.standard_media,
        "[{suite_label}] sim_dog={sim_dog}"
    );
    sims.push(sim_dog);

    // [13-14] Extended audio
    let female_samples =
        glypr_core::multimodal::decode_wav_file(golden_dir.join("female_voice_16k.wav"))?;
    let female_emb = embedder.embed_audio(&female_samples, None)?;
    let sim_female = cosine_similarity(&female_emb, &golden.cases["female_voice_only"].embedding);
    assert!(
        sim_female >= thresh.standard_media,
        "[{suite_label}] sim_female={sim_female}"
    );
    sims.push(sim_female);

    let music_samples = glypr_core::multimodal::decode_wav_file(golden_dir.join("music_16k.wav"))?;
    let music_emb = embedder.embed_audio(&music_samples, None)?;
    let sim_music = cosine_similarity(&music_emb, &golden.cases["music_only"].embedding);
    assert!(
        sim_music >= thresh.standard_media,
        "[{suite_label}] sim_music={sim_music}"
    );
    sims.push(sim_music);

    // [15-18] Dynamic vision token budget sweeps
    for budget in [70usize, 140, 280, 560] {
        let b_emb = embedder.embed_image_with_budget(&pepper_img, None, budget)?;
        let case_key = format!("pepper_budget_{budget}");
        let sim = cosine_similarity(&b_emb, &golden.cases[&case_key].embedding);
        let min_req = if budget == 70 {
            thresh.budget_70
        } else if budget == 560 {
            thresh.high_precision
        } else {
            thresh.standard_media
        };
        assert!(
            sim >= min_req,
            "[{suite_label}] Budget {budget} sim={sim} < {min_req}"
        );
        sims.push(sim);
    }

    // [19] Composite multi-image
    let comp1_emb = embedder.embed_composite(
        Some("A comparison of two pictures. First: <|image|> Second: <|image|>"),
        &[
            MediaItem::Image {
                image: &baboon_img,
                budget: None,
            },
            MediaItem::Image {
                image: &pepper_img,
                budget: None,
            },
        ],
        None,
    )?;
    let sim_comp1 = cosine_similarity(&comp1_emb, &golden.cases["composite_multi_image"].embedding);
    assert!(
        sim_comp1 >= thresh.standard_media,
        "[{suite_label}] Composite multi-image sim={sim_comp1}"
    );
    sims.push(sim_comp1);

    // [20] Interleaved multimodal (embed_composite + embed_interleaved)
    let comp2_emb = embedder.embed_composite(
        Some("A picture and some audio: <|image|> <|audio|>"),
        &[
            MediaItem::Image {
                image: &baboon_img,
                budget: None,
            },
            MediaItem::Audio {
                samples: &music_samples,
            },
        ],
        None,
    )?;
    let sim_comp2 = cosine_similarity(&comp2_emb, &golden.cases["composite_interleaved"].embedding);
    assert!(
        sim_comp2 >= thresh.high_precision,
        "[{suite_label}] Interleaved composite sim={sim_comp2}"
    );
    sims.push(sim_comp2);

    let interleaved_parts = [
        ModalityPart::Text("A picture and some audio: "),
        ModalityPart::Image {
            image: &baboon_img,
            budget: None,
        },
        ModalityPart::Text(" "),
        ModalityPart::Audio {
            samples: &music_samples,
        },
    ];
    let interleaved_emb = embedder.embed_interleaved(&interleaved_parts, Task::Raw)?;
    let sim_interleaved = cosine_similarity(
        &interleaved_emb,
        &golden.cases["composite_interleaved"].embedding,
    );
    assert!(
        sim_interleaved >= thresh.high_precision,
        "[{suite_label}] embed_interleaved vs Python Golden sim={sim_interleaved}"
    );
    assert!(
        cosine_similarity(&comp2_emb, &interleaved_emb) > 0.9999,
        "[{suite_label}] Cross-API consistency failed"
    );

    // [21-24] All 4 cross-modal queries in multimodal_golden.json
    let query_dog_emb = embedder.embed_query(
        "A photograph of a dog sitting in grass",
        "SearchQuery",
        None,
    )?;
    let sim_q_dog = cosine_similarity(&query_dog_emb, &golden.cases["query_dog"].embedding);
    assert!(
        sim_q_dog >= thresh.high_precision,
        "[{suite_label}] query_dog sim={sim_q_dog}"
    );
    sims.push(sim_q_dog);
    assert!(
        cosine_similarity(&query_dog_emb, &dog_emb)
            > cosine_similarity(&query_dog_emb, &baboon_emb) + 0.03
    );

    let query_countryside_emb = embedder.embed_query(
        "A wide landscape view of rolling hills and trees in the countryside",
        "SearchQuery",
        None,
    )?;
    let sim_q_countryside = cosine_similarity(
        &query_countryside_emb,
        &golden.cases["query_countryside"].embedding,
    );
    assert!(
        sim_q_countryside >= thresh.high_precision,
        "[{suite_label}] query_countryside sim={sim_q_countryside}"
    );
    sims.push(sim_q_countryside);

    let query_music_emb =
        embedder.embed_query("Acoustic guitar instrumental music", "SearchQuery", None)?;
    let sim_q_music = cosine_similarity(&query_music_emb, &golden.cases["query_music"].embedding);
    assert!(
        sim_q_music >= thresh.high_precision,
        "[{suite_label}] query_music sim={sim_q_music}"
    );
    sims.push(sim_q_music);
    assert!(
        cosine_similarity(&query_music_emb, &music_emb)
            > cosine_similarity(&query_music_emb, &female_emb) + 0.15
    );

    let query_female_emb =
        embedder.embed_query("A woman speaking English clearly", "SearchQuery", None)?;
    let sim_q_female = cosine_similarity(
        &query_female_emb,
        &golden.cases["query_female_voice"].embedding,
    );
    assert!(
        sim_q_female >= thresh.high_precision,
        "[{suite_label}] query_female_voice sim={sim_q_female}"
    );
    sims.push(sim_q_female);

    assert_eq!(sims.len(), 24, "Expected all 24 golden cases asserted");
    let avg_sim: f32 = sims.iter().sum::<f32>() / (sims.len() as f32);
    assert!(
        avg_sim >= thresh.min_average,
        "[{suite_label}] Average multimodal parity {avg_sim:.6} below threshold {:.6}",
        thresh.min_average
    );

    Ok(())
}

fn require_multimodal(embedder: &GlyprEmbedder) -> anyhow::Result<bool> {
    if embedder.has_multimodal() && embedder.has_audio_projector() {
        return Ok(true);
    }
    if common::allow_missing_weights() {
        eprintln!(
            "Skipping multimodal parity (mmproj not cached and GLYPR_ALLOW_MISSING_WEIGHTS=1)"
        );
        return Ok(false);
    }
    anyhow::bail!("Multimodal projector (mmproj) must be attached for multimodal_parity tests")
}

#[test]
fn test_multimodal_smoke_parity() -> anyhow::Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    if !require_multimodal(&embedder)? {
        return Ok(());
    }

    #[allow(unused_mut)]
    let mut mode = AttentionMode::ChunkedSliding;
    #[cfg(feature = "metal-flash-attn")]
    if embedder.device().is_metal() {
        mode = AttentionMode::MetalFlash;
    }
    embedder.set_attention_mode(mode);

    let (golden_dir, golden) = load_golden_data()?;

    // 1. Synthetic image
    let synth_img_file = golden
        .synthetic_image_file
        .as_deref()
        .unwrap_or(&golden.image_file);
    let synth_img = image::open(golden_dir.join(synth_img_file))?;
    let synth_img_emb = embedder.embed_image(&synth_img, None)?;
    assert!(
        cosine_similarity(&synth_img_emb, &golden.cases["image_only"].embedding) >= 0.99965,
        "smoke synthetic image parity failed"
    );

    // 2. Real image (dog.png)
    let dog_img = image::open(golden_dir.join("dog.png"))?;
    let dog_emb = embedder.embed_image(&dog_img, None)?;
    assert!(
        cosine_similarity(&dog_emb, &golden.cases["dog_only"].embedding) >= 0.99965,
        "smoke dog.png parity failed"
    );

    // 3. Real audio (music_16k.wav)
    let music_samples = glypr_core::multimodal::decode_wav_file(golden_dir.join("music_16k.wav"))?;
    let music_emb = embedder.embed_audio(&music_samples, None)?;
    assert!(
        cosine_similarity(&music_emb, &golden.cases["music_only"].embedding) >= 0.99965,
        "smoke music_16k.wav parity failed"
    );

    // 4. Multi-image composite (baboon + pepper)
    let baboon_img = image::open(golden_dir.join("baboon.png"))?;
    let pepper_img = image::open(golden_dir.join("pepper.png"))?;
    let comp_multi = embedder.embed_composite(
        Some("A comparison of two pictures. First: <|image|> Second: <|image|>"),
        &[
            MediaItem::Image {
                image: &baboon_img,
                budget: None,
            },
            MediaItem::Image {
                image: &pepper_img,
                budget: None,
            },
        ],
        None,
    )?;
    assert!(
        cosine_similarity(
            &comp_multi,
            &golden.cases["composite_multi_image"].embedding
        ) >= 0.99965,
        "smoke composite_multi_image parity failed"
    );

    // 5. Interleaved (embed_composite vs embed_interleaved + golden)
    let comp_inter = embedder.embed_composite(
        Some("A picture and some audio: <|image|> <|audio|>"),
        &[
            MediaItem::Image {
                image: &baboon_img,
                budget: None,
            },
            MediaItem::Audio {
                samples: &music_samples,
            },
        ],
        None,
    )?;
    let inter = embedder.embed_interleaved(
        &[
            ModalityPart::Text("A picture and some audio: "),
            ModalityPart::Image {
                image: &baboon_img,
                budget: None,
            },
            ModalityPart::Text(" "),
            ModalityPart::Audio {
                samples: &music_samples,
            },
        ],
        Task::Raw,
    )?;
    assert!(
        cosine_similarity(
            &comp_inter,
            &golden.cases["composite_interleaved"].embedding
        ) >= 0.9999
    );
    assert!(cosine_similarity(&comp_inter, &inter) > 0.9999);

    // 6. VideoFrames consistency check
    let video_comp = embedder.embed_composite(
        Some("Video: <|video|>"),
        &[MediaItem::VideoFrames {
            frames: &[&synth_img, &baboon_img],
            budget_per_frame: Some(70),
        }],
        None,
    )?;
    let video_inter = embedder.embed_interleaved(
        &[
            ModalityPart::Text("Video: "),
            ModalityPart::VideoFrame {
                frame: &synth_img,
                budget: Some(70),
            },
            ModalityPart::VideoFrame {
                frame: &baboon_img,
                budget: Some(70),
            },
        ],
        Task::Raw,
    )?;
    assert!(
        cosine_similarity(&video_comp, &video_inter) > 0.9999,
        "VideoFrames vs VideoFrame consistency failed"
    );

    // 7. Previously unasserted query_* keys in multimodal_golden.json
    let q_countryside = embedder.embed_query(
        "A wide landscape view of rolling hills and trees in the countryside",
        "SearchQuery",
        None,
    )?;
    assert!(
        cosine_similarity(&q_countryside, &golden.cases["query_countryside"].embedding) >= 0.9999,
        "smoke query_countryside parity failed"
    );
    let q_female = embedder.embed_query("A woman speaking English clearly", "SearchQuery", None)?;
    assert!(
        cosine_similarity(&q_female, &golden.cases["query_female_voice"].embedding) >= 0.9999,
        "smoke query_female_voice parity failed"
    );

    Ok(())
}

#[test]
#[ignore = "Tier 2: full 24-case multimodal parity"]
fn test_multimodal_golden_parity() -> anyhow::Result<()> {
    let Some(shared) = common::shared_gguf_bf16_metal()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    if !require_multimodal(&embedder)? {
        return Ok(());
    }

    let bf16_thresholds = MultimodalThresholds {
        high_precision: 0.9999,
        standard_media: 0.99965,
        budget_70: 0.9994,
        min_average: 0.9998,
    };

    embedder.set_attention_mode(AttentionMode::ChunkedSliding);
    run_multimodal_parity_suite(
        &embedder,
        "BF16 GGUF + BF16 mmproj (ChunkedSliding)",
        &bf16_thresholds,
    )?;

    #[cfg(feature = "metal-flash-attn")]
    if embedder.device().is_metal() {
        embedder.set_attention_mode(AttentionMode::MetalFlash);
        run_multimodal_parity_suite(
            &embedder,
            "BF16 GGUF + BF16 mmproj (MetalFlash)",
            &bf16_thresholds,
        )?;
    }

    Ok(())
}

#[test]
#[ignore = "Tier 2: full 24-case multimodal parity"]
fn test_multimodal_golden_parity_q8() -> anyhow::Result<()> {
    let Some(shared) = common::shared_gguf_q8_metal()? else {
        return Ok(());
    };
    let mut embedder = shared.lock().unwrap_or_else(|e| e.into_inner());
    if !require_multimodal(&embedder)? {
        return Ok(());
    }

    let q8_thresholds = MultimodalThresholds {
        high_precision: 0.9994,
        standard_media: 0.9984,
        budget_70: 0.9984,
        min_average: 0.9995,
    };

    embedder.set_attention_mode(AttentionMode::ChunkedSliding);
    run_multimodal_parity_suite(
        &embedder,
        "Q8_0 GGUF + Q8_0 mmproj (ChunkedSliding)",
        &q8_thresholds,
    )?;

    #[cfg(feature = "metal-flash-attn")]
    if embedder.device().is_metal() {
        embedder.set_attention_mode(AttentionMode::MetalFlash);
        run_multimodal_parity_suite(
            &embedder,
            "Q8_0 GGUF + Q8_0 mmproj (MetalFlash)",
            &q8_thresholds,
        )?;
    }

    Ok(())
}
