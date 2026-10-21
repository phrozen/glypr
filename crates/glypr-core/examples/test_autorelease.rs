#[allow(clippy::duplicated_attributes)]
#[link(name = "Metal", kind = "framework")]
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut std::ffi::c_void;
    fn sel_registerName(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn objc_msgSend(receiver: *mut std::ffi::c_void, selector: *mut std::ffi::c_void) -> usize;
    fn objc_autoreleasePoolPush() -> *mut std::ffi::c_void;
    fn objc_autoreleasePoolPop(pool: *mut std::ffi::c_void);
}

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

use anyhow::Result;
use candle_core::Device;
use glypr_core::{AttentionMode, GlyprEmbedder};
use std::fs;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("Testing autorelease pool effect on Metal memory...");

    let gguf_path = glypr_core::hub::resolve_default_gguf()?;
    let device = Device::new_metal(0)?;

    let mut embedder = GlyprEmbedder::load_gguf_bf16(&gguf_path, &device)?;
    embedder.set_attention_mode(AttentionMode::ChunkedSliding);

    println!(
        "Model Loaded. Metal VRAM: {:.2} MB",
        get_metal_allocated_bytes() as f64 / (1024.0 * 1024.0)
    );

    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/golden_vectors")
        .join("corpus_8k.json");
    let corpus_bytes = fs::read(&corpus_path)?;
    let corpus: serde_json::Value = serde_json::from_slice(&corpus_bytes)?;
    let entry = corpus["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["id"] == "doc_physics_8192t")
        .unwrap();
    let content = entry["content"].as_str().unwrap();

    let pool = unsafe { objc_autoreleasePoolPush() };

    let t0 = std::time::Instant::now();
    let _ = embedder.embed_document(content, None, Some(768))?;
    println!(
        "Embedded 8k doc in {:.2} ms. Metal VRAM before pool pop: {:.2} MB",
        t0.elapsed().as_secs_f64() * 1000.0,
        get_metal_allocated_bytes() as f64 / (1024.0 * 1024.0)
    );

    unsafe { objc_autoreleasePoolPop(pool) };
    println!(
        "Metal VRAM after pool pop: {:.2} MB",
        get_metal_allocated_bytes() as f64 / (1024.0 * 1024.0)
    );

    Ok(())
}
