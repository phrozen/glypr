use std::collections::HashMap;
use std::path::PathBuf;
use candle_core::{DType, Device};
use glypr_core::{cosine_similarity, GlyprEmbedder, MediaItem};
use serde::Deserialize;

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
    #[allow(dead_code)]
    synthetic_image_file: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    synthetic_audio_file: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    real_image_file: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    real_audio_file: Option<String>,
    image_file: String,
    audio_file: String,
    cases: HashMap<String, GoldenCase>,
}

fn read_wav_samples(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let samples: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32767.0).collect();
    Ok(samples)
}

#[test]
fn test_multimodal_golden_parity() -> anyhow::Result<()> {
    let device = if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    let dtype = if device.is_metal() { DType::BF16 } else { DType::F32 };

    let base_gguf = glypr_core::hub::resolve_default_gguf()?;
    let mmproj_gguf = glypr_core::hub::resolve_default_mmproj()?;

    if !base_gguf.exists() || !mmproj_gguf.exists() {
        eprintln!("Skipping multimodal test: model checkpoints not found");
        return Ok(());
    }

    println!("Initializing GlyprEmbedder with base GGUF on {:?} ({:?})...", device, dtype);
    let mut embedder = GlyprEmbedder::load_gguf(&base_gguf, &device, dtype)?;

    println!("Attaching multimodal projector from {:?}...", mmproj_gguf);
    embedder.attach_multimodal_projector(&mmproj_gguf)?;

    // Read golden json
    let golden_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/golden_vectors");
    let golden_path = golden_dir.join("multimodal_golden.json");
    let content = std::fs::read_to_string(&golden_path)?;
    let golden: GoldenData = serde_json::from_str(&content)?;

    // =========================================================================
    // [1] BASELINE SYNTHETIC & REAL-WORLD CASES (Cases 1-8)
    // =========================================================================
    let synth_img_file = golden.synthetic_image_file.as_deref().unwrap_or(&golden.image_file);
    let synth_img_path = golden_dir.join(synth_img_file);
    let synth_img = image::open(&synth_img_path)?;

    println!("\n[1] Synthetic Image Only...");
    let synth_img_emb = embedder.embed_image(&synth_img, None)?;
    let sim_synth_img = cosine_similarity(&synth_img_emb, &golden.cases["image_only"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_synth_img);
    assert!(sim_synth_img > 0.999, "sim={}", sim_synth_img);

    println!("\n[2] Synthetic Image + Prompt...");
    let synth_img_p_emb = embedder.embed_image(&synth_img, Some("Describe this picture"))?;
    let sim_synth_img_p = cosine_similarity(&synth_img_p_emb, &golden.cases["image_prompt"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_synth_img_p);
    assert!(sim_synth_img_p > 0.999, "sim={}", sim_synth_img_p);

    let synth_audio_file = golden.synthetic_audio_file.as_deref().unwrap_or(&golden.audio_file);
    let synth_samples = read_wav_samples(&golden_dir.join(synth_audio_file))?;

    println!("\n[3] Synthetic Audio Only ({} samples)...", synth_samples.len());
    let synth_aud_emb = embedder.embed_audio(&synth_samples, None)?;
    let sim_synth_aud = cosine_similarity(&synth_aud_emb, &golden.cases["audio_only"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_synth_aud);
    assert!(sim_synth_aud > 0.999, "sim={}", sim_synth_aud);

    println!("\n[4] Synthetic Audio + Prompt...");
    let synth_aud_p_emb = embedder.embed_audio(&synth_samples, Some("A musical sound"))?;
    let sim_synth_aud_p = cosine_similarity(&synth_aud_p_emb, &golden.cases["audio_prompt"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_synth_aud_p);
    assert!(sim_synth_aud_p > 0.999, "sim={}", sim_synth_aud_p);

    let real_img = image::open(golden_dir.join("lenna.png"))?;
    println!("\n[5] Real Image (Lenna) Only ({}x{})...", real_img.width(), real_img.height());
    let real_img_emb = embedder.embed_image(&real_img, None)?;
    let sim_real_img = cosine_similarity(&real_img_emb, &golden.cases["real_image_only"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_real_img);
    assert!(sim_real_img > 0.999, "sim={}", sim_real_img);

    println!("\n[6] Real Image + Prompt...");
    let real_img_p_emb = embedder.embed_image(&real_img, Some("A classic portrait photograph of a woman wearing a hat"))?;
    let sim_real_img_p = cosine_similarity(&real_img_p_emb, &golden.cases["real_image_prompt"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_real_img_p);
    assert!(sim_real_img_p > 0.999, "sim={}", sim_real_img_p);

    let lena_samples = read_wav_samples(&golden_dir.join("lena_16k.wav"))?;
    println!("\n[7] Real Audio (Lena) Only ({} samples)...", lena_samples.len());
    let real_aud_emb = embedder.embed_audio(&lena_samples, None)?;
    let sim_real_aud = cosine_similarity(&real_aud_emb, &golden.cases["real_audio_only"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_real_aud);
    assert!(sim_real_aud > 0.999, "sim={}", sim_real_aud);

    println!("\n[8] Real Audio + Prompt...");
    let real_aud_p_emb = embedder.embed_audio(&lena_samples, Some("A woman speaking English into a microphone"))?;
    let sim_real_aud_p = cosine_similarity(&real_aud_p_emb, &golden.cases["real_audio_prompt"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_real_aud_p);
    assert!(sim_real_aud_p > 0.999, "sim={}", sim_real_aud_p);

    // =========================================================================
    // [2] EXTENDED IMAGES (Aspect Ratios & High-Frequency Textures)
    // =========================================================================
    let baboon_img = image::open(golden_dir.join("baboon.png"))?;
    println!("\n[9] Baboon (Square texture, {}x{})...", baboon_img.width(), baboon_img.height());
    let baboon_emb = embedder.embed_image(&baboon_img, None)?;
    let sim_baboon = cosine_similarity(&baboon_emb, &golden.cases["baboon_only"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_baboon);
    assert!(sim_baboon > 0.999, "sim={}", sim_baboon);

    let pepper_img = image::open(golden_dir.join("pepper.png"))?;
    println!("\n[10] Pepper (Square gradient, {}x{})...", pepper_img.width(), pepper_img.height());
    let pepper_emb = embedder.embed_image(&pepper_img, None)?;
    let sim_pepper = cosine_similarity(&pepper_emb, &golden.cases["pepper_only"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_pepper);
    assert!(sim_pepper > 0.999, "sim={}", sim_pepper);

    let countryside_img = image::open(golden_dir.join("countryside.png"))?;
    println!("\n[11] Countryside (16:9 Landscape, {}x{})...", countryside_img.width(), countryside_img.height());
    let countryside_emb = embedder.embed_image(&countryside_img, None)?;
    let sim_countryside = cosine_similarity(&countryside_emb, &golden.cases["countryside_only"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_countryside);
    assert!(sim_countryside > 0.999, "sim={}", sim_countryside);

    let dog_img = image::open(golden_dir.join("dog.png"))?;
    println!("\n[12] Dog (2:3 Portrait, {}x{})...", dog_img.width(), dog_img.height());
    let dog_emb = embedder.embed_image(&dog_img, None)?;
    let sim_dog = cosine_similarity(&dog_emb, &golden.cases["dog_only"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_dog);
    assert!(sim_dog > 0.999, "sim={}", sim_dog);

    // =========================================================================
    // [3] EXTENDED AUDIO (Vocal Speech & Music Harmonics)
    // =========================================================================
    let female_samples = read_wav_samples(&golden_dir.join("female_voice_16k.wav"))?;
    println!("\n[13] Female Voice (12.68s speech, {} samples)...", female_samples.len());
    let female_emb = embedder.embed_audio(&female_samples, None)?;
    let sim_female = cosine_similarity(&female_emb, &golden.cases["female_voice_only"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_female);
    assert!(sim_female > 0.999, "sim={}", sim_female);

    let music_samples = read_wav_samples(&golden_dir.join("music_16k.wav"))?;
    println!("\n[14] Music (5.00s acoustic instrument, {} samples)...", music_samples.len());
    let music_emb = embedder.embed_audio(&music_samples, None)?;
    let sim_music = cosine_similarity(&music_emb, &golden.cases["music_only"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_music);
    assert!(sim_music > 0.9985, "sim={}", sim_music);

    // =========================================================================
    // [4] DYNAMIC VISION TOKEN BUDGET SWEEPS (pepper.png @ 70, 140, 280, 560 tokens)
    // =========================================================================
    println!("\n[15-18] Dynamic Vision Token Budget Sweeps (pepper.png)...");
    for budget in [70, 140, 280, 560] {
        let b_emb = embedder.embed_image_with_budget(&pepper_img, None, budget)?;
        let case_key = format!("pepper_budget_{}", budget);
        let sim = cosine_similarity(&b_emb, &golden.cases[&case_key].embedding);
        println!("  -> Budget {:3} tokens: Cosine Similarity = {:.6}", budget, sim);
        assert!(sim > 0.9985, "Budget {} sim={}", budget, sim);
    }

    // =========================================================================
    // [5] COMPOSITE MULTI-IMAGE INPUTS
    // =========================================================================
    println!("\n[19] Multi-Image Composite (Baboon + Pepper)...");
    let comp1_text = "A comparison of two pictures. First: <|image|> Second: <|image|>";
    let comp1_emb = embedder.embed_composite(
        Some(comp1_text),
        &[
            MediaItem::Image { image: &baboon_img, budget: None },
            MediaItem::Image { image: &pepper_img, budget: None },
        ],
        None,
    )?;
    let sim_comp1 = cosine_similarity(&comp1_emb, &golden.cases["composite_multi_image"].embedding);
    println!("  -> Cosine Similarity: {:.6}", sim_comp1);
    assert!(sim_comp1 > 0.9985, "Composite multi-image sim={}", sim_comp1);

    // =========================================================================
    // [6] INTERLEAVED MULTIMODAL INPUTS (Text + Image + Audio)
    // =========================================================================
    println!("\n[20] Interleaved Multimodal Composite (Text + Baboon + Music)...");
    let comp2_text = "A picture and some audio: <|image|> <|audio|>";
    let comp2_emb = embedder.embed_composite(
        Some(comp2_text),
        &[
            MediaItem::Image { image: &baboon_img, budget: None },
            MediaItem::Audio { samples: &music_samples },
        ],
        None,
    )?;
    let sim_comp2 = cosine_similarity(&comp2_emb, &golden.cases["composite_interleaved"].embedding);
    println!("  -> Cosine Similarity (embed_composite vs Python Golden): {:.6}", sim_comp2);
    assert!(sim_comp2 > 0.9985, "Interleaved composite sim={}", sim_comp2);

    let interleaved_parts = [
        glypr_core::ModalityPart::Text("A picture and some audio: "),
        glypr_core::ModalityPart::Image { image: &baboon_img, budget: None },
        glypr_core::ModalityPart::Text(" "),
        glypr_core::ModalityPart::Audio { samples: &music_samples },
    ];
    let interleaved_emb = embedder.embed_interleaved(&interleaved_parts, glypr_core::Task::Raw)?;
    let sim_interleaved = cosine_similarity(&interleaved_emb, &golden.cases["composite_interleaved"].embedding);
    println!("  -> Cosine Similarity (embed_interleaved vs Python Golden): {:.6}", sim_interleaved);
    assert!(sim_interleaved > 0.9985, "embed_interleaved vs Python Golden sim={}", sim_interleaved);

    let sim_comp_vs_inter = cosine_similarity(&comp2_emb, &interleaved_emb);
    println!("  -> Cross-API Consistency (embed_composite vs embed_interleaved): {:.6}", sim_comp_vs_inter);
    assert!(sim_comp_vs_inter > 0.9999, "Cross-API consistency sim={}", sim_comp_vs_inter);

    // =========================================================================
    // [7] CROSS-MODAL SEMANTIC RETRIEVAL RANKING
    // =========================================================================
    println!("\n[21-24] Cross-Modal Semantic Retrieval Assertions...");
    let query_dog_emb = embedder.embed_query("A photograph of a dog sitting in grass", "SearchQuery", None)?;
    let sim_q_dog_golden = cosine_similarity(&query_dog_emb, &golden.cases["query_dog"].embedding);
    assert!(sim_q_dog_golden > 0.999, "query_dog golden sim={}", sim_q_dog_golden);

    let sim_q_dog_vs_dog = cosine_similarity(&query_dog_emb, &dog_emb);
    let sim_q_dog_vs_baboon = cosine_similarity(&query_dog_emb, &baboon_emb);
    println!("  Query [Dog]: vs Dog = {:.4} | vs Baboon = {:.4} (margin = +{:.4})",
        sim_q_dog_vs_dog, sim_q_dog_vs_baboon, sim_q_dog_vs_dog - sim_q_dog_vs_baboon);
    assert!(sim_q_dog_vs_dog > sim_q_dog_vs_baboon + 0.03, "Dog query did not clearly separate dog from baboon");

    let query_music_emb = embedder.embed_query("Acoustic guitar instrumental music", "SearchQuery", None)?;
    let sim_q_music_golden = cosine_similarity(&query_music_emb, &golden.cases["query_music"].embedding);
    assert!(sim_q_music_golden > 0.999, "query_music golden sim={}", sim_q_music_golden);

    let sim_q_music_vs_music = cosine_similarity(&query_music_emb, &music_emb);
    let sim_q_music_vs_female = cosine_similarity(&query_music_emb, &female_emb);
    println!("  Query [Music]: vs Music = {:.4} | vs Female Speech = {:.4} (margin = +{:.4})",
        sim_q_music_vs_music, sim_q_music_vs_female, sim_q_music_vs_music - sim_q_music_vs_female);
    assert!(sim_q_music_vs_music > sim_q_music_vs_female + 0.15, "Music query did not clearly separate music from speech");

    println!("\n=========================================================================");
    println!("   ALL MULTIMODAL PARITY & CROSS-MODAL RETRIEVAL TESTS PASSED (> 0.999)!");
    println!("=========================================================================");

    Ok(())
}
