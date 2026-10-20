use candle_core::{DType, Device};
use glypr_core::GlyprEmbedder;
use std::path::PathBuf;
use std::time::Instant;

fn bench_device(device: Device, dtype: DType, device_label: &str) -> anyhow::Result<()> {
    println!("\n================================================================================");
    println!("  MULTIMODAL BENCHMARK: {} ({:?})", device_label, dtype);
    println!("================================================================================");

    let base_gguf = glypr_core::hub::resolve_default_gguf()?;
    let mmproj_gguf = glypr_core::hub::resolve_default_mmproj()?;

    if !base_gguf.exists() || !mmproj_gguf.exists() {
        println!("Skipping: GGUF models not found in cache.");
        return Ok(());
    }

    let mut embedder = GlyprEmbedder::load_gguf(&base_gguf, &device, dtype)?;
    embedder.attach_multimodal_projector(&mmproj_gguf)?;

    let test_img_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/golden_vectors/multimodal_test_image.png");

    let img = image::open(&test_img_path)?;
    // Create an exact 480x680 card image to mirror standard portrait document dimensions
    let card_img = img.resize_exact(480, 680, image::imageops::FilterType::Nearest);

    // Warmup run
    println!("Warming up engine...");
    let _ = embedder.embed_image_with_budget(&card_img, Some("Card art query"), 140)?;

    for budget in [140, 280] {
        println!("\n--- Benchmarking 480x680 Card (Budget: {}) ---", budget);
        let iterations = 5;
        let mut total_duration = std::time::Duration::ZERO;
        let mut total_vision = std::time::Duration::ZERO;
        let mut total_backbone = std::time::Duration::ZERO;

        let proj = embedder.multimodal.as_ref().unwrap();
        let t_vis = Instant::now();
        let _ = proj.encode_image_with_budget(&card_img, budget)?;
        let vision_dur = t_vis.elapsed();

        for i in 1..=iterations {
            let t1 = Instant::now();
            let _ = embedder.embed_image_with_budget(&card_img, Some("Card art query"), budget)?;
            let total_dur = t1.elapsed();

            let backbone_dur = total_dur.saturating_sub(vision_dur);

            total_duration += total_dur;
            total_vision += vision_dur;
            total_backbone += backbone_dur;

            println!(
                "  Iteration {}: End-to-End: {:.2} ms (Vision Tower: {:.2} ms | Splicing & Backbone: {:.2} ms)",
                i,
                total_dur.as_secs_f64() * 1000.0,
                vision_dur.as_secs_f64() * 1000.0,
                backbone_dur.as_secs_f64() * 1000.0
            );
        }

        let avg_ms = (total_duration.as_secs_f64() * 1000.0) / (iterations as f64);
        let avg_vis = (total_vision.as_secs_f64() * 1000.0) / (iterations as f64);
        let avg_bb = (total_backbone.as_secs_f64() * 1000.0) / (iterations as f64);
        let throughput = 1000.0 / avg_ms;
        println!(
            "  -> Avg Total: {:.2} ms (Vision: {:.2} ms, Backbone: {:.2} ms) | Throughput: {:.2} cards/sec",
            avg_ms, avg_vis, avg_bb, throughput
        );
    }

    // =========================================================================
    // AUDIO BENCHMARKS
    // =========================================================================
    let golden_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/golden_vectors");

    let audio_test_cases = [
        ("Short (1.0s Synthetic)", golden_dir.join("multimodal_test_audio.wav")),
        ("Medium (5.0s Music)", golden_dir.join("music_16k.wav")),
        ("Long (12.3s Speech)", golden_dir.join("lena_16k.wav")),
    ];

    for (label, wav_path) in &audio_test_cases {
        if !wav_path.exists() {
            println!("Skipping audio {}: file not found at {:?}", label, wav_path);
            continue;
        }

        let samples = glypr_core::multimodal::decode_wav_file(wav_path)?;
        let audio_dur_secs = samples.len() as f64 / 16000.0;
        let num_soft_tokens = glypr_core::multimodal::compute_audio_soft_token_count(samples.len());

        println!(
            "\n--- Benchmarking Audio: {} ({:.2}s, {} samples -> {} soft tokens) ---",
            label, audio_dur_secs, samples.len(), num_soft_tokens
        );

        let proj = embedder.multimodal.as_ref().unwrap();

        // Warmup
        let _ = embedder.embed_audio(&samples, Some("Audio query"))?;

        let t_prep_0 = Instant::now();
        let _ = glypr_core::multimodal::preprocess_audio(&samples)?;
        let prep_dur = t_prep_0.elapsed();

        let t_tower_0 = Instant::now();
        let _ = proj.encode_audio(&samples)?;
        let tower_dur = t_tower_0.elapsed();

        let iterations = 5;
        let mut total_duration = std::time::Duration::ZERO;
        let mut total_preprocessor = std::time::Duration::ZERO;
        let mut total_tower = std::time::Duration::ZERO;
        let mut total_backbone = std::time::Duration::ZERO;

        for i in 1..=iterations {
            let t_e2e_0 = Instant::now();
            let _ = embedder.embed_audio(&samples, Some("Audio query"))?;
            let total_dur = t_e2e_0.elapsed();

            let backbone_dur = total_dur.saturating_sub(tower_dur);

            total_duration += total_dur;
            total_preprocessor += prep_dur;
            total_tower += tower_dur;
            total_backbone += backbone_dur;

            println!(
                "  Iteration {}: End-to-End: {:.2} ms (Preprocessor: {:.2} ms | Audio Tower: {:.2} ms | Backbone: {:.2} ms)",
                i,
                total_dur.as_secs_f64() * 1000.0,
                prep_dur.as_secs_f64() * 1000.0,
                tower_dur.as_secs_f64() * 1000.0,
                backbone_dur.as_secs_f64() * 1000.0
            );
        }

        let avg_total_ms = (total_duration.as_secs_f64() * 1000.0) / (iterations as f64);
        let avg_prep_ms = (total_preprocessor.as_secs_f64() * 1000.0) / (iterations as f64);
        let avg_tower_ms = (total_tower.as_secs_f64() * 1000.0) / (iterations as f64);
        let avg_bb_ms = (total_backbone.as_secs_f64() * 1000.0) / (iterations as f64);
        let audio_speedup = audio_dur_secs / (avg_total_ms / 1000.0);
        let rtf = (avg_total_ms / 1000.0) / audio_dur_secs;

        println!(
            "  -> Avg: {:.2} ms (Prep: {:.2} ms, Tower: {:.2} ms, Backbone: {:.2} ms) | RTF: {:.3}x | Throughput: {:.1}x real-time ({:.1} audio-sec/s)",
            avg_total_ms, avg_prep_ms, avg_tower_ms, avg_bb_ms, rtf, audio_speedup, audio_speedup
        );
    }

    Ok(())
}

#[test]
fn bench_multimodal_pipeline() -> anyhow::Result<()> {
    #[cfg(feature = "metal")]
    if candle_core::utils::metal_is_available() {
        if let Ok(metal_dev) = Device::new_metal(0) {
            bench_device(metal_dev, DType::BF16, "Apple Silicon Metal GPU")?;
        }
    }

    // Also benchmark CPU track if explicit or on non-metal runs
    if std::env::var("BENCH_CPU").is_ok() {
        bench_device(Device::Cpu, DType::F32, "Universal CPU SIMD")?;
    }

    Ok(())
}
