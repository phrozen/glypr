use candle_core::{DType, Device};
use glypr_core::coreml::{CoreMLComputeUnits, CoreMLTextBackbone};
use glypr_core::{hub, GlyprEmbedder, Task};
use std::path::Path;
use std::time::Instant;

fn bench_text_ane_vs_metal(seq_len: usize) -> anyhow::Result<()> {
    println!("------------------------------------------------------------");
    println!("  BENCHMARK: Text Embedding ({} tokens)", seq_len);
    println!("------------------------------------------------------------");

    // Construct a sample text of approximately `seq_len` tokens
    let sample_word = "neural ";
    // roughly 1.3 words per token
    let word_count = (seq_len as f64 * 0.8).max(1.0) as usize;
    let text = sample_word.repeat(word_count);

    // 1. CoreML ANE Text Backbone (models/coreml/gemma2_text_backbone_s512_w8a16.mlmodelc)
    let coreml_text_path = "models/coreml/gemma2_text_backbone_s512_w8a16.mlmodelc";
    if Path::new(coreml_text_path).exists() {
        let ane_backbone = CoreMLTextBackbone::from_path(coreml_text_path, CoreMLComputeUnits::CpuAndNeuralEngine)?;
        let sample_ids: Vec<i32> = (0..seq_len).map(|i| ((i * 37 + 100) % 250000) as i32).collect();

        // Warmup
        for _ in 0..5 {
            let _ = ane_backbone.embed_token_ids(&sample_ids)?;
        }

        let iters = 30;
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = ane_backbone.embed_token_ids(&sample_ids)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mean = times.iter().sum::<f64>() / iters as f64;
        println!("1. Pure ANE (CpuAndNeuralEngine, INT8):  {:6.2} ms  ({:5.1} queries/s)", mean, 1000.0 / mean);

        let all_backbone = CoreMLTextBackbone::from_path(coreml_text_path, CoreMLComputeUnits::All)?;
        for _ in 0..5 {
            let _ = all_backbone.embed_token_ids(&sample_ids)?;
        }
        let mut times_all = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = all_backbone.embed_token_ids(&sample_ids)?;
            times_all.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mean_all = times_all.iter().sum::<f64>() / iters as f64;
        println!("2. CoreML ALL (ANE + GPU, INT8):          {:6.2} ms  ({:5.1} queries/s)", mean_all, 1000.0 / mean_all);
    }

    // 2. Candle Metal (BF16 GGUF base)
    if candle_core::utils::metal_is_available() {
        if let Ok(bf16_path) = hub::resolve_canonical_model(false) {
            let device = Device::new_metal(0)?;
            let embedder_bf16 = GlyprEmbedder::load_gguf(&bf16_path, &device, DType::BF16)?;

            // Warmup
            for _ in 0..5 {
                let _ = embedder_bf16.embed_text(&text, Task::Raw)?;
            }

            let iters = 30;
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = embedder_bf16.embed_text(&text, Task::Raw)?;
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let mean = times.iter().sum::<f64>() / iters as f64;
            println!("3. Candle Metal (GGUF BF16 on GPU):       {:6.2} ms  ({:5.1} queries/s)", mean, 1000.0 / mean);
        }

        // 3. Candle Metal (Q8_0 GGUF base)
        if let Ok(q8_path) = hub::resolve_canonical_model(true) {
            let device = Device::new_metal(0)?;
            let embedder_q8 = GlyprEmbedder::load_gguf(&q8_path, &device, DType::BF16)?;

            // Warmup
            for _ in 0..5 {
                let _ = embedder_q8.embed_text(&text, Task::Raw)?;
            }

            let iters = 30;
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = embedder_q8.embed_text(&text, Task::Raw)?;
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let mean = times.iter().sum::<f64>() / iters as f64;
            println!("4. Candle Metal (GGUF Q8_0 on GPU):       {:6.2} ms  ({:5.1} queries/s)", mean, 1000.0 / mean);
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("============================================================");
    println!("  BENCHMARK: Text Embedding (ANE vs Candle Metal)");
    println!("============================================================");

    bench_text_ane_vs_metal(16)?;
    println!();
    bench_text_ane_vs_metal(128)?;
    println!();
    bench_text_ane_vs_metal(512)?;

    Ok(())
}
