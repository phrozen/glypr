use candle_core::{DType, Device};
use glypr_core::coreml::{CoreMLComputeUnits, CoreMLVisionTower};
use glypr_core::multimodal::image_preprocessor::preprocess_image;
use glypr_core::multimodal::projector::MultimodalProjector;
use image::DynamicImage;
use std::path::Path;
use std::time::Instant;

fn bench_budget(budget: usize, model_file: &str) -> anyhow::Result<()> {
    println!("------------------------------------------------------------");
    println!("  BENCHMARK: Single Image (Budget {}, portrait)", budget);
    println!("------------------------------------------------------------");

    let img = DynamicImage::new_rgb8(480, 680);
    let processed = preprocess_image(&img, budget, 16, 3)?;
    println!(
        "Preprocessed image: {} patches -> {} soft tokens\n",
        processed.num_patches, processed.num_soft_tokens
    );

    let coreml_model_path = format!("models/coreml/{}", model_file);
    if Path::new(&coreml_model_path).exists() {
        let ane_tower = CoreMLVisionTower::from_path(
            &coreml_model_path,
            CoreMLComputeUnits::CpuAndNeuralEngine,
        )?;
        for _ in 0..3 {
            let _ = ane_tower.encode_raw(&processed)?;
        }

        let iters = 20;
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = ane_tower.encode_raw(&processed)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mean = times.iter().sum::<f64>() / iters as f64;
        println!(
            "1. Pure ANE (CpuAndNeuralEngine, INT8):  {:6.2} ms  ({:4.1} img/s)",
            mean,
            1000.0 / mean
        );

        let all_tower = CoreMLVisionTower::from_path(&coreml_model_path, CoreMLComputeUnits::All)?;
        for _ in 0..3 {
            let _ = all_tower.encode_raw(&processed)?;
        }

        let mut times_all = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = all_tower.encode_raw(&processed)?;
            times_all.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mean_all = times_all.iter().sum::<f64>() / iters as f64;
        println!(
            "2. CoreML ALL (ANE + GPU, INT8):          {:6.2} ms  ({:4.1} img/s)",
            mean_all,
            1000.0 / mean_all
        );
    }

    if let Ok(mmproj_path) = glypr_core::hub::resolve_default_mmproj()
        && mmproj_path.exists()
        && candle_core::utils::metal_is_available()
    {
        let device = Device::new_metal(0)?;
        let projector = MultimodalProjector::from_gguf_file(&mmproj_path, &device, DType::BF16)?;
        if let Some(tower) = &projector.vision_tower {
            for _ in 0..3 {
                let _ = tower.encode(&processed)?;
            }
            let iters = 20;
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = tower.encode(&processed)?;
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let mean = times.iter().sum::<f64>() / iters as f64;
            println!(
                "3. Candle Metal ViT (GGUF BF16 on GPU):   {:6.2} ms  ({:4.1} img/s)",
                mean,
                1000.0 / mean
            );
        }
    }

    if let Ok(q8_mmproj_path) = glypr_core::hub::resolve_canonical_mmproj(true)
        && q8_mmproj_path.exists()
        && candle_core::utils::metal_is_available()
    {
        let device = Device::new_metal(0)?;
        let projector = MultimodalProjector::from_gguf_file(&q8_mmproj_path, &device, DType::BF16)?;
        if let Some(tower) = &projector.vision_tower {
            for _ in 0..3 {
                let _ = tower.encode(&processed)?;
            }
            let iters = 20;
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = tower.encode(&processed)?;
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let mean = times.iter().sum::<f64>() / iters as f64;
            println!(
                "4. Candle Metal ViT (GGUF Q8_0 on GPU):   {:6.2} ms  ({:4.1} img/s)",
                mean,
                1000.0 / mean
            );
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    bench_budget(140, "gemma2_vision_tower_p1134_t126_w8a16.mlmodelc")?;
    println!();
    bench_budget(280, "gemma2_vision_tower_p2394_t266_w8a16.mlmodelc")?;
    Ok(())
}
