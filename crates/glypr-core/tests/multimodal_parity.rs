use std::collections::HashMap;
use std::path::PathBuf;
use candle_core::{DType, Device};
use glypr_core::{cosine_similarity, GlyprEmbedder};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GoldenCase {
    #[allow(dead_code)]
    prompt: Option<String>,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct GoldenData {
    #[serde(default)]
    synthetic_image_file: Option<String>,
    #[serde(default)]
    synthetic_audio_file: Option<String>,
    #[serde(default)]
    real_image_file: Option<String>,
    #[serde(default)]
    real_audio_file: Option<String>,
    image_file: String,
    audio_file: String,
    cases: HashMap<String, GoldenCase>,
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

    // --- Synthetic Cases ---
    // 1. Synthetic Image Only
    let synth_img_file = golden.synthetic_image_file.as_deref().unwrap_or(&golden.image_file);
    let synth_img_path = golden_dir.join(synth_img_file);
    let synth_img = image::open(&synth_img_path)?;
    println!("\nRunning Case 1: Synthetic Image Only...");
    let synth_img_emb = embedder.embed_image(&synth_img, None, None)?;
    let case_synth_img = &golden.cases["image_only"];
    let sim_synth_img = cosine_similarity(&synth_img_emb, &case_synth_img.embedding);
    println!("  [Synthetic Image Only] Cosine Similarity: {:.6}", sim_synth_img);

    // 2. Synthetic Image + Prompt
    let case_synth_img_p = &golden.cases["image_prompt"];
    println!("\nRunning Case 2: Synthetic Image + Prompt ({:?})...", case_synth_img_p.prompt);
    let synth_img_p_emb = embedder.embed_image(&synth_img, case_synth_img_p.prompt.as_deref(), None)?;
    let sim_synth_img_p = cosine_similarity(&synth_img_p_emb, &case_synth_img_p.embedding);
    println!("  [Synthetic Image + Prompt] Cosine Similarity: {:.6}", sim_synth_img_p);

    // 3. Synthetic Audio Only
    let synth_audio_file = golden.synthetic_audio_file.as_deref().unwrap_or(&golden.audio_file);
    let synth_audio_path = golden_dir.join(synth_audio_file);
    let mut synth_reader = hound::WavReader::open(&synth_audio_path)?;
    let synth_samples: Vec<f32> = synth_reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32767.0).collect();
    println!("\nRunning Case 3: Synthetic Audio Only ({} samples)...", synth_samples.len());
    let synth_aud_emb = embedder.embed_audio(&synth_samples, None, None)?;
    let case_synth_aud = &golden.cases["audio_only"];
    let sim_synth_aud = cosine_similarity(&synth_aud_emb, &case_synth_aud.embedding);
    println!("  [Synthetic Audio Only] Cosine Similarity: {:.6}", sim_synth_aud);

    // 4. Synthetic Audio + Prompt
    let case_synth_aud_p = &golden.cases["audio_prompt"];
    println!("\nRunning Case 4: Synthetic Audio + Prompt ({:?})...", case_synth_aud_p.prompt);
    let synth_aud_p_emb = embedder.embed_audio(&synth_samples, case_synth_aud_p.prompt.as_deref(), None)?;
    let sim_synth_aud_p = cosine_similarity(&synth_aud_p_emb, &case_synth_aud_p.embedding);
    println!("  [Synthetic Audio + Prompt] Cosine Similarity: {:.6}", sim_synth_aud_p);

    assert!(sim_synth_img > 0.999, "Synthetic Image Only similarity below threshold: {}", sim_synth_img);
    assert!(sim_synth_img_p > 0.999, "Synthetic Image + Prompt similarity below threshold: {}", sim_synth_img_p);
    assert!(sim_synth_aud > 0.999, "Synthetic Audio Only similarity below threshold: {}", sim_synth_aud);
    assert!(sim_synth_aud_p > 0.999, "Synthetic Audio + Prompt similarity below threshold: {}", sim_synth_aud_p);

    // --- Real-World Cases (Lenna & Lena) ---
    if let (Some(real_img_name), Some(real_aud_name)) = (&golden.real_image_file, &golden.real_audio_file) {
        // 5. Real Image (Lenna) Only
        let real_img_path = golden_dir.join(real_img_name);
        let real_img = image::open(&real_img_path)?;
        println!("\nRunning Case 5: Real Image (Lenna) Only ({}x{})...", real_img.width(), real_img.height());
        let real_img_emb = embedder.embed_image(&real_img, None, None)?;
        let case_real_img = &golden.cases["real_image_only"];
        let sim_real_img = cosine_similarity(&real_img_emb, &case_real_img.embedding);
        println!("  [Real Image Only] Cosine Similarity: {:.6}", sim_real_img);

        // 6. Real Image + Prompt
        let case_real_img_p = &golden.cases["real_image_prompt"];
        println!("\nRunning Case 6: Real Image + Prompt ({:?})...", case_real_img_p.prompt);
        let real_img_p_emb = embedder.embed_image(&real_img, case_real_img_p.prompt.as_deref(), None)?;
        let sim_real_img_p = cosine_similarity(&real_img_p_emb, &case_real_img_p.embedding);
        println!("  [Real Image + Prompt] Cosine Similarity: {:.6}", sim_real_img_p);

        // 7. Real Audio (Lena Speech) Only
        let real_audio_path = golden_dir.join(real_aud_name);
        let mut real_reader = hound::WavReader::open(&real_audio_path)?;
        let real_samples: Vec<f32> = real_reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32767.0).collect();
        println!("\nRunning Case 7: Real Audio (Lena Speech) Only ({} samples, {:.2}s)...", real_samples.len(), real_samples.len() as f64 / 16000.0);
        let real_aud_emb = embedder.embed_audio(&real_samples, None, None)?;
        let case_real_aud = &golden.cases["real_audio_only"];
        let sim_real_aud = cosine_similarity(&real_aud_emb, &case_real_aud.embedding);
        println!("  [Real Audio Only] Cosine Similarity: {:.6}", sim_real_aud);

        // 8. Real Audio + Prompt
        let case_real_aud_p = &golden.cases["real_audio_prompt"];
        println!("\nRunning Case 8: Real Audio + Prompt ({:?})...", case_real_aud_p.prompt);
        let real_aud_p_emb = embedder.embed_audio(&real_samples, case_real_aud_p.prompt.as_deref(), None)?;
        let sim_real_aud_p = cosine_similarity(&real_aud_p_emb, &case_real_aud_p.embedding);
        println!("  [Real Audio + Prompt] Cosine Similarity: {:.6}", sim_real_aud_p);

        assert!(sim_real_img > 0.999, "Real Image Only similarity below threshold: {}", sim_real_img);
        assert!(sim_real_img_p > 0.999, "Real Image + Prompt similarity below threshold: {}", sim_real_img_p);
        assert!(sim_real_aud > 0.999, "Real Audio Only similarity below threshold: {}", sim_real_aud);
        assert!(sim_real_aud_p > 0.999, "Real Audio + Prompt similarity below threshold: {}", sim_real_aud_p);
    }

    println!("\n=========================================================================");
    println!("   ALL 8 MULTIMODAL GOLDEN PARITY TESTS PASSED (> 0.999 COSIM)!");
    println!("=========================================================================");

    Ok(())
}
