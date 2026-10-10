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

        for i in 1..=iterations {
            let start = Instant::now();
            let vec = embedder.embed_image_with_budget(&card_img, Some("Card art query"), budget)?;
            let elapsed = start.elapsed();
            total_duration += elapsed;
            println!("  Iteration {}: {:.2} ms (norm: {:.4})", i, elapsed.as_secs_f64() * 1000.0, vec.iter().map(|x| x*x).sum::<f32>().sqrt());
        }

        let avg_ms = (total_duration.as_secs_f64() * 1000.0) / (iterations as f64);
        let throughput = 1000.0 / avg_ms;
        println!("  -> Average Latency: {:.2} ms | Throughput: {:.2} cards/sec", avg_ms, throughput);
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
