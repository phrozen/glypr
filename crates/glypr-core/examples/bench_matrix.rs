use anyhow::Result;
use candle_core::{DType, Device};
use glypr_core::coreml::{CoreMLComputeUnits, CoreMLTextBackbone, CoreMLVisionTower};
use glypr_core::multimodal::ModalityPart;
use glypr_core::multimodal::image_preprocessor::preprocess_image;
use glypr_core::multimodal::projector::MultimodalProjector;
use glypr_core::{AttentionMode, GlyprEmbedder, Task, VisionBackendPolicy, hub};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct CorpusFile {
    entries: Vec<CorpusEntry>,
}

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    id: String,
    content: String,
}

#[cfg(target_os = "macos")]
#[allow(clippy::duplicated_attributes)]
#[link(name = "Metal", kind = "framework")]
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut std::ffi::c_void;
    fn sel_registerName(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn objc_msgSend(receiver: *mut std::ffi::c_void, selector: *mut std::ffi::c_void) -> usize;
}

#[cfg(target_os = "macos")]
fn get_metal_allocated_bytes() -> u64 {
    unsafe {
        let dev = MTLCreateSystemDefaultDevice();
        if dev.is_null() {
            return 0;
        }
        let sel = sel_registerName(c"currentAllocatedSize".as_ptr());
        let size = objc_msgSend(dev, sel);
        size as u64
    }
}

#[cfg(not(target_os = "macos"))]
fn get_metal_allocated_bytes() -> u64 {
    0
}

fn get_process_rss_bytes() -> u64 {
    unsafe {
        let mut rusage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut rusage);
        #[cfg(target_os = "macos")]
        {
            rusage.ru_maxrss as u64
        }
        #[cfg(not(target_os = "macos"))]
        {
            (rusage.ru_maxrss as u64) * 1024
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

#[derive(Debug, Serialize, Default)]
struct BenchmarkMatrixRust {
    metal: EngineMeasurements,
    ane: EngineMeasurements,
    cpu: EngineMeasurements,
}

#[derive(Debug, Serialize, Default)]
struct EngineMeasurements {
    text_512_ms: Option<f64>,
    vision_140_ms: Option<f64>,
    vision_280_ms: Option<f64>,
    hybrid_512_ms: Option<f64>,
    text_8192_ms: Option<f64>,
    vision_560_ms: Option<f64>,
    hybrid_long_ms: Option<f64>,
    peak_memory_8k_mb: Option<f64>,
}

fn main() -> Result<()> {
    println!("============================================================");
    println!("  GLYPR NATIVE RUST BENCHMARK HARNESS (Metal / ANE / CPU)   ");
    println!("============================================================");

    let mut matrix = BenchmarkMatrixRust::default();

    // 1. Locate Models & Corpus
    let gguf_default = hub::resolve_default_gguf()
        .unwrap_or_else(|_| PathBuf::from("models/embeddinggemma-2-BF16.gguf"));
    let mmproj_default = hub::resolve_default_mmproj()
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
    let text_8192 = corpus_map
        .get("doc_physics_8192t")
        .cloned()
        .unwrap_or_else(|| "spacetime curvature and quantum mechanics ".repeat(400));

    let img_portrait = image::open("tests/golden_vectors/portrait_test.png")
        .or_else(|_| image::open("../../tests/golden_vectors/portrait_test.png"))
        .unwrap_or_else(|_| DynamicImage::new_rgb8(480, 680));
    let img_pepper = image::open("tests/golden_vectors/pepper.png")
        .or_else(|_| image::open("../../tests/golden_vectors/pepper.png"))
        .unwrap_or_else(|_| DynamicImage::new_rgb8(512, 512));

    let p140 = preprocess_image(&img_portrait, 140, 16, 3)?;
    let p280 = preprocess_image(&img_portrait, 280, 16, 3)?;
    let p560 = preprocess_image(&img_pepper, 560, 16, 3)?;

    // =========================================================================
    // SECTION A: Apple Neural Engine (ANE) Benchmarks
    // =========================================================================
    println!("\n>>> BENCHMARKING APPLE NEURAL ENGINE (ANE)...");
    let ane_vision_140_path = "models/coreml/gemma2_vision_tower_p1134_t126_w8a16.mlmodelc";
    let ane_vision_280_path = "models/coreml/gemma2_vision_tower_p2394_t266_w8a16.mlmodelc";
    let ane_text_path = "models/coreml/gemma2_text_backbone_s512_w8a16.mlmodelc";

    if Path::new(ane_text_path).exists() {
        let ane_text =
            CoreMLTextBackbone::from_path(ane_text_path, CoreMLComputeUnits::CpuAndNeuralEngine)?;
        let sample_ids: Vec<i32> = (0..512).map(|i| (i * 37 + 100) % 250000).collect();
        // Warmup
        for _ in 0..3 {
            let _ = ane_text.embed_token_ids(&sample_ids)?;
        }
        let mut times = vec![];
        for _ in 0..10 {
            let t0 = Instant::now();
            let _ = ane_text.embed_token_ids(&sample_ids)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        matrix.ane.text_512_ms = Some((med * 100.0).round() / 100.0);
        println!("  ✓ ANE Text @ 512t:       {:6.2} ms", med);
    }

    if Path::new(ane_vision_140_path).exists() {
        let ane_v140 = CoreMLVisionTower::from_path(
            ane_vision_140_path,
            CoreMLComputeUnits::CpuAndNeuralEngine,
        )?;
        for _ in 0..3 {
            let _ = ane_v140.encode_raw(&p140)?;
        }
        let mut times = vec![];
        for _ in 0..10 {
            let t0 = Instant::now();
            let _ = ane_v140.encode_raw(&p140)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        matrix.ane.vision_140_ms = Some((med * 100.0).round() / 100.0);
        println!("  ✓ ANE Vision @ 140b:     {:6.2} ms", med);
    }

    if Path::new(ane_vision_280_path).exists() {
        let ane_v280 = CoreMLVisionTower::from_path(
            ane_vision_280_path,
            CoreMLComputeUnits::CpuAndNeuralEngine,
        )?;
        for _ in 0..3 {
            let _ = ane_v280.encode_raw(&p280)?;
        }
        let mut times = vec![];
        for _ in 0..10 {
            let t0 = Instant::now();
            let _ = ane_v280.encode_raw(&p280)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        matrix.ane.vision_280_ms = Some((med * 100.0).round() / 100.0);
        println!("  ✓ ANE Vision @ 280b:     {:6.2} ms", med);
    }

    // =========================================================================
    // SECTION B: Apple Silicon Metal GPU Benchmarks
    // =========================================================================
    println!("\n>>> BENCHMARKING APPLE SILICON METAL GPU...");
    if gguf_default.exists() {
        let dev_metal = Device::new_metal(0)?;
        let mut embedder_metal = GlyprEmbedder::load_gguf(&gguf_default, &dev_metal, DType::BF16)?;
        embedder_metal.set_attention_mode(AttentionMode::MetalFlash);
        if mmproj_default.exists() {
            let _ = embedder_metal.attach_multimodal_projector(&mmproj_default);
        }
        let _ = embedder_metal
            .attach_coreml_vision_dir("models/coreml", CoreMLComputeUnits::CpuAndNeuralEngine);

        // 1. Text 512t
        for _ in 0..2 {
            let _ = embedder_metal.embed_text(&text_512, Task::Raw)?;
        }
        let mut times = vec![];
        for _ in 0..10 {
            let t0 = Instant::now();
            let _ = embedder_metal.embed_text(&text_512, Task::Raw)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        matrix.metal.text_512_ms = Some((med * 100.0).round() / 100.0);
        println!("  ✓ Metal Text @ 512t:     {:6.2} ms", med);

        // 2. Multimodal Vision on Metal ViT
        if mmproj_default.exists() {
            let proj_metal =
                MultimodalProjector::from_gguf_file(&mmproj_default, &dev_metal, DType::BF16)?;
            if let Some(tower) = &proj_metal.vision_tower {
                for _ in 0..2 {
                    let _ = tower.encode(&p140)?;
                }
                let mut times = vec![];
                for _ in 0..10 {
                    let t0 = Instant::now();
                    let _ = tower.encode(&p140)?;
                    times.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
                let med = median(times);
                matrix.metal.vision_140_ms = Some((med * 100.0).round() / 100.0);
                println!("  ✓ Metal Vision @ 140b:   {:6.2} ms", med);

                let mut times = vec![];
                for _ in 0..10 {
                    let t0 = Instant::now();
                    let _ = tower.encode(&p280)?;
                    times.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
                let med = median(times);
                matrix.metal.vision_280_ms = Some((med * 100.0).round() / 100.0);
                println!("  ✓ Metal Vision @ 280b:   {:6.2} ms", med);

                let mut times = vec![];
                for _ in 0..5 {
                    let t0 = Instant::now();
                    let _ = tower.encode(&p560)?;
                    times.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
                let med = median(times);
                matrix.metal.vision_560_ms = Some((med * 100.0).round() / 100.0);
                println!("  ✓ Metal Vision @ 560b:   {:6.2} ms", med);
            }
        }

        // 3. Hybrid <= 512t
        embedder_metal.set_vision_backend(VisionBackendPolicy::Metal);
        let mut times = vec![];
        for _ in 0..5 {
            let t0 = Instant::now();
            let _ = embedder_metal.embed_interleaved(
                &[
                    ModalityPart::Text("Card description: "),
                    ModalityPart::Image {
                        image: &img_portrait,
                        budget: Some(140),
                    },
                ],
                Task::Raw,
            )?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        matrix.metal.hybrid_512_ms = Some((med * 100.0).round() / 100.0);
        println!("  ✓ Metal Hybrid @ <=512t: {:6.2} ms", med);

        // ANE Hybrid <= 512t
        embedder_metal.set_vision_backend(VisionBackendPolicy::AneStrict);
        let mut times = vec![];
        for _ in 0..5 {
            let t0 = Instant::now();
            let _ = embedder_metal.embed_interleaved(
                &[
                    ModalityPart::Text("Card description: "),
                    ModalityPart::Image {
                        image: &img_portrait,
                        budget: Some(140),
                    },
                ],
                Task::Raw,
            )?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        matrix.ane.hybrid_512_ms = Some((med * 100.0).round() / 100.0);
        println!("  ✓ ANE Hybrid @ <=512t:   {:6.2} ms", med);

        // 4. Hybrid > 512t (2K text + image)
        embedder_metal.set_vision_backend(VisionBackendPolicy::Metal);
        let mut times = vec![];
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = embedder_metal.embed_interleaved(
                &[
                    ModalityPart::Text(&text_2048),
                    ModalityPart::Image {
                        image: &img_portrait,
                        budget: Some(140),
                    },
                ],
                Task::Raw,
            )?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        matrix.metal.hybrid_long_ms = Some((med * 100.0).round() / 100.0);
        println!("  ✓ Metal Hybrid @ >512t:  {:6.2} ms", med);

        embedder_metal.set_vision_backend(VisionBackendPolicy::AneStrict);
        let mut times = vec![];
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = embedder_metal.embed_interleaved(
                &[
                    ModalityPart::Text(&text_2048),
                    ModalityPart::Image {
                        image: &img_portrait,
                        budget: Some(140),
                    },
                ],
                Task::Raw,
            )?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        matrix.ane.hybrid_long_ms = Some((med * 100.0).round() / 100.0);
        println!("  ✓ ANE Hybrid @ >512t:    {:6.2} ms (Hybrid Routing)", med);

        // 5. Text @ 8192t
        embedder_metal.set_vision_backend(VisionBackendPolicy::Auto);
        for _ in 0..1 {
            let _ = embedder_metal.embed_text(&text_8192, Task::Raw)?;
        }
        let mut times = vec![];
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = embedder_metal.embed_text(&text_8192, Task::Raw)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        let metal_mem_mb = (get_metal_allocated_bytes() as f64) / (1024.0 * 1024.0);
        matrix.metal.text_8192_ms = Some((med * 100.0).round() / 100.0);
        matrix.metal.peak_memory_8k_mb = Some((metal_mem_mb * 10.0).round() / 10.0);
        println!(
            "  ✓ Metal Text @ 8192t:    {:6.2} ms (Peak Metal VRAM: {:6.1} MB)",
            med, metal_mem_mb
        );
    }

    // =========================================================================
    // SECTION C: Universal CPU SIMD Benchmarks
    // =========================================================================
    println!("\n>>> BENCHMARKING UNIVERSAL CPU SIMD...");
    if gguf_default.exists() {
        let dev_cpu = Device::Cpu;
        let mut embedder_cpu = GlyprEmbedder::load_gguf(&gguf_default, &dev_cpu, DType::F32)?;
        embedder_cpu.set_attention_mode(AttentionMode::ChunkedSliding);
        if mmproj_default.exists() {
            let _ = embedder_cpu.attach_multimodal_projector(&mmproj_default);
        }

        // Text 512t
        for _ in 0..1 {
            let _ = embedder_cpu.embed_text(&text_512, Task::Raw)?;
        }
        let mut times = vec![];
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = embedder_cpu.embed_text(&text_512, Task::Raw)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(times);
        matrix.cpu.text_512_ms = Some((med * 100.0).round() / 100.0);
        println!("  ✓ CPU Text @ 512t:       {:6.2} ms", med);

        if mmproj_default.exists() {
            let proj_cpu =
                MultimodalProjector::from_gguf_file(&mmproj_default, &dev_cpu, DType::F32)?;
            if let Some(tower) = &proj_cpu.vision_tower {
                let t0 = Instant::now();
                let _ = tower.encode(&p140)?;
                let med = t0.elapsed().as_secs_f64() * 1000.0;
                matrix.cpu.vision_140_ms = Some((med * 100.0).round() / 100.0);
                println!("  ✓ CPU Vision @ 140b:     {:6.2} ms", med);

                let t0 = Instant::now();
                let _ = tower.encode(&p280)?;
                let med = t0.elapsed().as_secs_f64() * 1000.0;
                matrix.cpu.vision_280_ms = Some((med * 100.0).round() / 100.0);
                println!("  ✓ CPU Vision @ 280b:     {:6.2} ms", med);
            }
        }

        let rss_mb = (get_process_rss_bytes() as f64) / (1024.0 * 1024.0);
        matrix.cpu.peak_memory_8k_mb = Some((rss_mb * 10.0).round() / 10.0);
    }

    let out_path = Path::new("target/bench_matrix_rust.json");
    let json_data = serde_json::to_string_pretty(&matrix)?;
    fs::write(out_path, json_data)?;
    println!("\n✓ Rust benchmark matrix saved to {:?}", out_path);

    Ok(())
}
