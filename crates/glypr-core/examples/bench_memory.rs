use anyhow::Result;
use candle_core::Device;
use glypr_core::{AttentionMode, GlyprEmbedder};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Deserialize)]
struct CorpusFile {
    entries: Vec<CorpusEntry>,
}

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    id: String,
    content: String,
    title: Option<String>,
}

#[cfg(target_os = "macos")]
#[link(name = "Metal", kind = "framework")]
#[link(name = "Foundation", kind = "framework")]
extern "C" {
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
        let sel = sel_registerName(b"currentAllocatedSize\0".as_ptr() as *const _);
        let size = objc_msgSend(dev, sel);
        size as u64
    }
}

#[cfg(not(target_os = "macos"))]
fn get_metal_allocated_bytes() -> u64 {
    0
}

fn get_peak_rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        // On macOS, ru_maxrss is in bytes
        usage.ru_maxrss as f64 / (1024.0 * 1024.0)
    }
    #[cfg(not(target_os = "macos"))]
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        // On Linux, ru_maxrss is in kilobytes
        usage.ru_maxrss as f64 / 1024.0
    }
}

fn get_gguf_bf16_path() -> PathBuf {
    glypr_core::hub::resolve_default_gguf().expect("Failed to resolve BF16 GGUF")
}

fn get_gguf_q8_path() -> PathBuf {
    glypr_core::hub::resolve_file("ggml-org/embeddinggemma-2-GGUF", "embeddinggemma-2-Q8_0.gguf")
        .expect("Failed to resolve Q8_0 GGUF")
}

fn run_single_measurement(model_type: &str, mode: &str, token_len: usize) -> Result<()> {
    let base_metal_mb = get_metal_allocated_bytes() as f64 / (1024.0 * 1024.0);

    let is_cpu = mode == "cpu";
    let device = if is_cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };

    let mut embedder = if model_type == "bf16" {
        let path = get_gguf_bf16_path();
        GlyprEmbedder::load_gguf_bf16(&path, &device)?
    } else {
        let path = get_gguf_q8_path();
        GlyprEmbedder::load_gguf_q8(&path, &device)?
    };

    let attn_mode = match mode {
        "standard" => AttentionMode::Standard,
        "chunked" => AttentionMode::ChunkedSliding,
        "flash" => AttentionMode::MetalFlash,
        _ => AttentionMode::ChunkedSliding,
    };
    embedder.set_attention_mode(attn_mode);

    let loaded_metal_mb = get_metal_allocated_bytes() as f64 / (1024.0 * 1024.0);
    let model_metal_ram = if is_cpu { 0.0 } else { loaded_metal_mb - base_metal_mb };

    // Load document from corpus
    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/golden_vectors")
        .join("corpus_8k.json");
    let corpus_bytes = fs::read(&corpus_path)?;
    let corpus: CorpusFile = serde_json::from_slice(&corpus_bytes)?;

    let doc_id = format!("doc_physics_{}t", token_len);
    let entry = corpus.entries.iter().find(|e| e.id == doc_id)
        .ok_or_else(|| anyhow::anyhow!("Doc {} not found", doc_id))?;

    // Background thread sampling Metal VRAM every 1ms
    let running = Arc::new(AtomicBool::new(true));
    let peak_sampled = Arc::new(AtomicU64::new(get_metal_allocated_bytes()));

    let running_clone = running.clone();
    let peak_clone = peak_sampled.clone();

    let sampler_thread = thread::spawn(move || {
        while running_clone.load(Ordering::Relaxed) {
            let current = get_metal_allocated_bytes();
            peak_clone.fetch_max(current, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(1));
        }
    });

    // Embed target document
    let t0 = Instant::now();
    let _ = embedder.embed_document(&entry.content, entry.title.as_deref(), Some(768))?;
    let duration_ms = t0.elapsed().as_secs_f64() * 1000.0;

    running.store(false, Ordering::Relaxed);
    sampler_thread.join().unwrap();

    let peak_metal_mb = if is_cpu { 0.0 } else { peak_sampled.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0) };
    let end_metal_mb = if is_cpu { 0.0 } else { get_metal_allocated_bytes() as f64 / (1024.0 * 1024.0) };
    let transient_metal_mb = if is_cpu { 0.0 } else { peak_metal_mb - loaded_metal_mb };
    let peak_rss_mb = get_peak_rss_mb();

    // Output machine readable line
    println!(
        "RESULT|{}|{}|{}|{:.1}|{:.1}|{:.1}|{:.1}|{:.1}|{:.2}|{:.1}",
        model_type, mode, token_len, model_metal_ram, loaded_metal_mb, peak_metal_mb, transient_metal_mb, end_metal_mb, duration_ms, peak_rss_mb
    );

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 4 {
        let model = &args[1];
        let mode = &args[2];
        let tokens: usize = args[3].parse()?;
        return run_single_measurement(model, mode, tokens);
    }

    println!("========================================================================================================================");
    println!("                     EmbeddingGemma 2: Hardware Memory & Scaling Benchmark (Apple Silicon)                              ");
    println!("========================================================================================================================\n");

    let metal_configurations = [
        ("bf16", "standard", 1024, "GGUF BF16 with Standard Attention @ 1,024 tokens"),
        ("bf16", "chunked",  1024, "GGUF BF16 with Chunked Attention  @ 1,024 tokens"),
        ("bf16", "flash",    1024, "GGUF BF16 with Metal Flash        @ 1,024 tokens"),
        ("q8",   "flash",    1024, "GGUF Q8_0 with Metal Flash        @ 1,024 tokens"),
        ("bf16", "standard", 2048, "GGUF BF16 with Standard Attention @ 2,048 tokens"),
        ("bf16", "chunked",  2048, "GGUF BF16 with Chunked Attention  @ 2,048 tokens"),
        ("bf16", "flash",    2048, "GGUF BF16 with Metal Flash        @ 2,048 tokens"),
        ("q8",   "flash",    2048, "GGUF Q8_0 with Metal Flash        @ 2,048 tokens"),
        ("bf16", "standard", 4096, "GGUF BF16 with Standard Attention @ 4,096 tokens"),
        ("bf16", "chunked",  4096, "GGUF BF16 with Chunked Attention  @ 4,096 tokens"),
        ("bf16", "flash",    4096, "GGUF BF16 with Metal Flash        @ 4,096 tokens"),
        ("q8",   "flash",    4096, "GGUF Q8_0 with Metal Flash        @ 4,096 tokens"),
        ("bf16", "standard", 8192, "GGUF BF16 with Standard Attention @ 8,192 tokens"),
        ("bf16", "chunked",  8192, "GGUF BF16 with Chunked Attention  @ 8,192 tokens"),
        ("bf16", "flash",    8192, "GGUF BF16 with Metal Flash        @ 8,192 tokens"),
        ("q8",   "flash",    8192, "GGUF Q8_0 with Metal Flash        @ 8,192 tokens"),
    ];

    println!("--- 1. Apple Silicon Metal GPU Memory & Latency Scaling ---");
    println!("{:<8} | {:<12} | {:<8} | {:<12} | {:<14} | {:<14} | {:<10} | {:<12}",
        "Model", "Attention", "Tokens", "Model VRAM", "Peak VRAM", "Transient VRAM", "Latency", "Process RSS"
    );
    println!("------------------------------------------------------------------------------------------------------------------------");

    for (model, mode, tokens, _label) in metal_configurations {
        let output = std::process::Command::new(std::env::current_exe()?)
            .arg(model)
            .arg(mode)
            .arg(tokens.to_string())
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.starts_with("RESULT|") {
                let parts: Vec<&str> = line.split('|').collect();
                if parts.len() >= 11 {
                    let m_type = parts[1].to_uppercase();
                    let a_mode = match parts[2] {
                        "standard" => "Standard",
                        "chunked" => "Chunked",
                        "flash" => "Metal Flash",
                        _ => parts[2],
                    };
                    let t_len = format!("{}t", parts[3]);
                    let m_ram = format!("{} MB", parts[4]);
                    let p_ram = format!("{} MB", parts[6]);
                    let tr_ram = format!("{} MB", parts[7]);
                    let lat = format!("{} ms", parts[9]);
                    let rss = format!("{} MB", parts[10]);

                    println!("{:<8} | {:<12} | {:>8} | {:>12} | {:>14} | {:>14} | {:>10} | {:>12}",
                        m_type, a_mode, t_len, m_ram, p_ram, tr_ram, lat, rss
                    );
                }
            }
        }
    }

    println!("\n--- 2. CPU SIMD (pulp NEON/AVX) Latency, Throughput & Memory Scaling ---");
    let cpu_tokens = [64, 256, 512, 1024, 2048, 4096];
    println!("{:<8} | {:<12} | {:<8} | {:<12} | {:<16} | {:<12}",
        "Model", "Backend", "Tokens", "Latency", "Throughput", "Process RSS"
    );
    println!("------------------------------------------------------------------------------------------------------------------------");

    for tokens in cpu_tokens {
        for model in ["q8", "bf16"] {
            let output = std::process::Command::new(std::env::current_exe()?)
                .arg(model)
                .arg("cpu")
                .arg(tokens.to_string())
                .output()?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.starts_with("RESULT|") {
                    let parts: Vec<&str> = line.split('|').collect();
                    if parts.len() >= 11 {
                        let m_type = parts[1].to_uppercase();
                        let t_len = format!("{}t", parts[3]);
                        let lat_f: f64 = parts[9].parse().unwrap_or(0.0);
                        let tok_per_sec = (tokens as f64 / (lat_f / 1000.0)).round();
                        let lat = format!("{:.1} ms", lat_f);
                        let tps = format!("{:.0} tok/s", tok_per_sec);
                        let rss = format!("{} MB", parts[10]);

                        println!("{:<8} | {:<12} | {:>8} | {:>12} | {:>16} | {:>12}",
                            m_type, "CPU SIMD", t_len, lat, tps, rss
                        );
                    }
                }
            }
        }
    }

    println!("========================================================================================================================\n");

    Ok(())
}
