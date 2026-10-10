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
    // Create an exact 480x680 card image to mirror MTG card dimensions
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

        for i in 1..=iterations {
            let t0 = Instant::now();
            let _ = proj.encode_image_with_budget(&card_img, budget)?;
            let vision_dur = t0.elapsed();

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
