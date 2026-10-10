use anyhow::Result;
use candle_core::Device;
use glypr_core::{multimodal::VISION_BUDGET_MEDIUM, GlyprEmbedder};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(target_os = "macos")]
#[allow(clippy::duplicated_attributes)]
#[link(name = "Metal", kind = "framework")]
#[link(name = "Foundation", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut std::ffi::c_void;
    fn sel_registerName(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn objc_msgSend(receiver: *mut std::ffi::c_void, selector: *mut std::ffi::c_void) -> usize;
}

#[cfg(target_os = "macos")]
fn get_metal_allocated_mb() -> f64 {
    unsafe {
        let dev = MTLCreateSystemDefaultDevice();
        if dev.is_null() {
            return 0.0;
        }
        let sel = sel_registerName(c"currentAllocatedSize".as_ptr());
        let size = objc_msgSend(dev, sel);
        size as f64 / (1024.0 * 1024.0)
    }
}

fn get_peak_rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage.ru_maxrss as f64 / (1024.0 * 1024.0)
    }
    #[cfg(not(target_os = "macos"))]
    0.0
}

fn main() -> Result<()> {
    println!("============================================================");
    println!("  MTG CARDS BATCH MEMORY & LATENCY PROFILER");
    println!("============================================================");

    let gguf_path = glypr_core::hub::resolve_default_gguf()?;
    let mmproj_path = glypr_core::hub::resolve_canonical_mmproj(false)?;

    let device = Device::new_metal(0)?;
    println!("Loading GGUF BF16 base model...");
    let mut embedder = GlyprEmbedder::load_gguf_bf16(&gguf_path, &device)?;
    println!("Attaching BF16 multimodal projector...");
    embedder.attach_multimodal_projector(&mmproj_path)?;

    println!("Initial State:");
    println!("  Metal VRAM: {:.2} MB", get_metal_allocated_mb());
    println!("  Peak RSS:   {:.2} MB", get_peak_rss_mb());

    let cards_dir = PathBuf::from("tests/cards");
    let mut card_paths = Vec::new();
    for entry in std::fs::read_dir(&cards_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) == Some("jpg") {
            card_paths.push(path);
        }
    }
    card_paths.sort();
    println!("\nFound {} MTG cards in {:?}", card_paths.len(), cards_dir);

    // Warm up with first card
    let img_warmup = image::open(&card_paths[0])?;
    let _ = embedder.embed_image_with_budget(&img_warmup, None, VISION_BUDGET_MEDIUM)?;
    println!("\nWarmed up with card 0. Post-warmup:");
    println!("  Metal VRAM: {:.2} MB", get_metal_allocated_mb());
    println!("  Peak RSS:   {:.2} MB", get_peak_rss_mb());

    println!("\n--- Running Sequential Embedding on all {} cards (Budget 140) ---", card_paths.len());
    let t_start = Instant::now();
    let mut latencies = Vec::with_capacity(card_paths.len());

    for (i, p) in card_paths.iter().enumerate() {
        let t_card = Instant::now();
        let img = image::open(p)?;
        let _vec = embedder.embed_image_with_budget(&img, None, VISION_BUDGET_MEDIUM)?;
        let dur = t_card.elapsed().as_secs_f64() * 1000.0;
        latencies.push(dur);

        if (i + 1) % 8 == 0 || i + 1 == card_paths.len() {
            println!(
                "  [{:2}/{}] Card {:<35} -> {:6.2} ms | Metal VRAM: {:6.2} MB | Peak RSS: {:6.2} MB",
                i + 1,
                card_paths.len(),
                p.file_stem().unwrap().to_str().unwrap(),
                dur,
                get_metal_allocated_mb(),
                get_peak_rss_mb()
            );
        }
    }

    let total_time_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let throughput = card_paths.len() as f64 / (total_time_ms / 1000.0);

    println!("\n============================================================");
    println!("  SEQUENTIAL PROFILE RESULTS");
    println!("============================================================");
    println!("  Total Wallclock:  {:.2} ms ({:.2} s)", total_time_ms, total_time_ms / 1000.0);
    println!("  Average Latency:  {:.2} ms / card", avg_latency);
    println!("  Throughput:       {:.2} cards / sec", throughput);
    println!("  Final Metal VRAM: {:.2} MB", get_metal_allocated_mb());
    println!("  Final Peak RSS:   {:.2} MB", get_peak_rss_mb());
    println!("============================================================");

    Ok(())
}
