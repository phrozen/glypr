//! Automated Glypr batch benchmark harness (Batch Size = 8).

use anyhow::Result;
use candle_core::Device;
use glypr_core::multimodal::ModalityPart;
use glypr_core::{AttentionMode, GlyprEmbedder, MultimodalInput, Task, VisionBackendPolicy, hub};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize, Default)]
struct BatchEntry {
    latency_ms: f64,
    per_item_ms: f64,
    throughput_items_per_sec: f64,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct BatchResultsRust {
    batch_size: usize,
    measurements: std::collections::HashMap<String, BatchEntry>,
}

#[derive(Deserialize)]
struct CorpusItem {
    id: String,
    content: String,
}

#[derive(Deserialize)]
struct CorpusFile {
    entries: Vec<CorpusItem>,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<()> {
    println!("=================================================================");
    println!("  GLYPR METAL 4 GPU BATCH BENCHMARK (Batch Size = 8)");
    println!("=================================================================");

    let mut results = BatchResultsRust {
        batch_size: 8,
        measurements: std::collections::HashMap::new(),
    };

    // 1. Locate Models & Corpus
    let gguf_path = hub::resolve_default_gguf()
        .unwrap_or_else(|_| PathBuf::from("models/embeddinggemma-2-BF16.gguf"));
    let mmproj_path = hub::resolve_default_mmproj()
        .unwrap_or_else(|_| PathBuf::from("models/mmproj-embeddinggemma-2-BF16.gguf"));

    let corpus_candidates = [
        "tests/golden_vectors/corpus_8k.json",
        "../../tests/golden_vectors/corpus_8k.json",
    ];
    let mut corpus_content = String::new();
    for p in &corpus_candidates {
        if let Ok(c) = fs::read_to_string(p) {
            corpus_content = c;
            break;
        }
    }
    let corpus: CorpusFile =
        serde_json::from_str(&corpus_content).unwrap_or_else(|_| CorpusFile { entries: vec![] });
    let corpus_map: std::collections::HashMap<String, String> = corpus
        .entries
        .into_iter()
        .map(|e| (e.id, e.content))
        .collect();

    let text_512 = corpus_map
        .get("doc_physics_512t")
        .cloned()
        .unwrap_or_else(|| "quantum information theory ".repeat(50));
    let text_2048 = corpus_map
        .get("doc_physics_2048t")
        .cloned()
        .unwrap_or_else(|| "physics and general relativity ".repeat(180));
    let text_4096 = corpus_map
        .get("doc_physics_4096t")
        .cloned()
        .unwrap_or_else(|| "thermodynamics and statistical physics ".repeat(360));
    let text_8192 = corpus_map
        .get("doc_physics_8192t")
        .cloned()
        .unwrap_or_else(|| "spacetime curvature and quantum mechanics ".repeat(400));

    let img_portrait = image::open("tests/golden_vectors/portrait_test.png")
        .or_else(|_| image::open("../../tests/golden_vectors/portrait_test.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgb8(480, 680));
    let img_baboon = image::open("tests/golden_vectors/baboon.png")
        .or_else(|_| image::open("../../tests/golden_vectors/baboon.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgb8(267, 266));
    let img_pepper = image::open("tests/golden_vectors/pepper.png")
        .or_else(|_| image::open("../../tests/golden_vectors/pepper.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgb8(512, 512));

    let audio_samples = vec![0.0f32; 16000];

    let dev_metal = Device::new_metal(0)?;
    println!("\nLoading GlyprEmbedder (Metal GPU BF16)...");
    let mut embedder = GlyprEmbedder::load_gguf_bf16(&gguf_path, &dev_metal)?;
    embedder.attach_multimodal_projector(&mmproj_path)?;
    embedder.set_attention_mode(AttentionMode::MetalFlash);
    embedder.set_vision_backend(VisionBackendPolicy::Metal);

    // -------------------------------------------------------------------------
    // 1. Text Batch Benchmarks (B = 8)
    // -------------------------------------------------------------------------
    println!("\n[1/3] Benchmarking Batched Text (B = 8)...");

    let text_cases = [
        ("text_512_b8", &text_512),
        ("text_2048_b8", &text_2048),
        ("text_4096_b8", &text_4096),
        ("text_8192_b8", &text_8192),
    ];

    for (name, txt) in text_cases {
        let batch = vec![txt.as_str(); 8];
        // Warmup
        for _ in 0..1 {
            let _ = embedder.embed_batch_documents(&batch, None, None)?;
        }
        let mut times = vec![];
        for _ in 0..5 {
            let t0 = Instant::now();
            let _ = embedder.embed_batch_documents(&batch, None, None)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        let per_item = med / 8.0;
        let tput = 8.0 / (med / 1000.0);
        println!(
            "  ✓ {:20}: {:7.2} ms total ({:6.2} ms/item, {:6.1} items/s)",
            name, med, per_item, tput
        );
        results.measurements.insert(
            name.to_string(),
            BatchEntry {
                latency_ms: (med * 100.0).round() / 100.0,
                per_item_ms: (per_item * 100.0).round() / 100.0,
                throughput_items_per_sec: (tput * 10.0).round() / 10.0,
            },
        );
    }

    // -------------------------------------------------------------------------
    // 2. Images Batch Across 4 Budgets (B = 8)
    // -------------------------------------------------------------------------
    println!("\n[2/3] Benchmarking Batched Images Across 4 Budgets (B = 8)...");

    let vision_cases: [(&str, &image::DynamicImage, usize); 4] = [
        ("vision_70_b8", &img_pepper, 70),
        ("vision_140_b8", &img_portrait, 140),
        ("vision_280_b8", &img_baboon, 280),
        ("vision_560_b8", &img_pepper, 560),
    ];

    for (name, img, budget) in vision_cases {
        let batch: Vec<MultimodalInput> = (0..8)
            .map(|_| MultimodalInput::image(img, Some(budget), Task::Raw))
            .collect();

        // Warmup
        for _ in 0..1 {
            let _ = embedder.embed_batch(&batch, None)?;
        }
        let mut times = vec![];
        for _ in 0..5 {
            let t0 = Instant::now();
            let _ = embedder.embed_batch(&batch, None)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        let per_item = med / 8.0;
        let tput = 8.0 / (med / 1000.0);
        println!(
            "  ✓ {:20}: {:7.2} ms total ({:6.2} ms/item, {:6.1} items/s)",
            name, med, per_item, tput
        );
        results.measurements.insert(
            name.to_string(),
            BatchEntry {
                latency_ms: (med * 100.0).round() / 100.0,
                per_item_ms: (per_item * 100.0).round() / 100.0,
                throughput_items_per_sec: (tput * 10.0).round() / 10.0,
            },
        );
    }

    // -------------------------------------------------------------------------
    // 3. Hybrid Multimodal Batch (B = 8)
    // -------------------------------------------------------------------------
    println!("\n[3/3] Benchmarking Batched Hybrid Multimodal (B = 8)...");

    // A. Text + Vision
    {
        let parts_tv = [
            ModalityPart::Text("A photograph of an artifact in a museum: "),
            ModalityPart::Image {
                image: &img_portrait,
                budget: Some(140),
            },
        ];
        let batch: Vec<MultimodalInput> = (0..8)
            .map(|_| MultimodalInput::new(&parts_tv, Task::Raw))
            .collect();

        for _ in 0..1 {
            let _ = embedder.embed_batch(&batch, None)?;
        }
        let mut times = vec![];
        for _ in 0..5 {
            let t0 = Instant::now();
            let _ = embedder.embed_batch(&batch, None)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        let per_item = med / 8.0;
        let tput = 8.0 / (med / 1000.0);
        println!(
            "  ✓ {:20}: {:7.2} ms total ({:6.2} ms/item, {:6.1} items/s)",
            "hybrid_text_vision_b8", med, per_item, tput
        );
        results.measurements.insert(
            "hybrid_text_vision_b8".to_string(),
            BatchEntry {
                latency_ms: (med * 100.0).round() / 100.0,
                per_item_ms: (per_item * 100.0).round() / 100.0,
                throughput_items_per_sec: (tput * 10.0).round() / 10.0,
            },
        );
    }

    // B. Text + Vision + Audio
    {
        let parts_tva = [
            ModalityPart::Text("A video clip with background music: "),
            ModalityPart::Image {
                image: &img_portrait,
                budget: Some(140),
            },
            ModalityPart::Audio {
                samples: &audio_samples,
            },
        ];
        let batch: Vec<MultimodalInput> = (0..8)
            .map(|_| MultimodalInput::new(&parts_tva, Task::Raw))
            .collect();

        for _ in 0..1 {
            let _ = embedder.embed_batch(&batch, None)?;
        }
        let mut times = vec![];
        for _ in 0..5 {
            let t0 = Instant::now();
            let _ = embedder.embed_batch(&batch, None)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        let per_item = med / 8.0;
        let tput = 8.0 / (med / 1000.0);
        println!(
            "  ✓ {:20}: {:7.2} ms total ({:6.2} ms/item, {:6.1} items/s)",
            "hybrid_text_vision_audio_b8", med, per_item, tput
        );
        results.measurements.insert(
            "hybrid_text_vision_audio_b8".to_string(),
            BatchEntry {
                latency_ms: (med * 100.0).round() / 100.0,
                per_item_ms: (per_item * 100.0).round() / 100.0,
                throughput_items_per_sec: (tput * 10.0).round() / 10.0,
            },
        );
    }

    let out_path = Path::new("target/bench_batch_rust.json");
    if let Some(p) = out_path.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(out_path, serde_json::to_string_pretty(&results)?)?;
    println!("\n✓ Rust batch benchmarks saved to: {:?}", out_path);

    Ok(())
}
