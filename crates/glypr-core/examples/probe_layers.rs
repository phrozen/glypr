use anyhow::Result;
use candle_core::{Device, Module};
use glypr_core::{AttentionMode, GlyprEmbedder};
use std::fs;
use std::path::PathBuf;

#[cfg(target_os = "macos")]
#[link(name = "Metal", kind = "framework")]
#[link(name = "Foundation", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut std::ffi::c_void;
    fn sel_registerName(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn objc_msgSend(receiver: *mut std::ffi::c_void, selector: *mut std::ffi::c_void) -> usize;
}

#[cfg(target_os = "macos")]
fn get_metal_mb() -> f64 {
    unsafe {
        let dev = MTLCreateSystemDefaultDevice();
        if dev.is_null() {
            return 0.0;
        }
        let sel = sel_registerName(b"currentAllocatedSize\0".as_ptr() as *const _);
        let size = objc_msgSend(dev, sel);
        size as f64 / (1024.0 * 1024.0)
    }
}

fn main() -> Result<()> {
    let gguf_path = glypr_core::hub::resolve_default_gguf()?;
    let device = Device::new_metal(0)?;

    println!("Initial Metal: {:.1} MB", get_metal_mb());
    let mut embedder = GlyprEmbedder::load_gguf_bf16(&gguf_path, &device)?;
    println!("Model loaded: {:.1} MB", get_metal_mb());

    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join("tests/golden_vectors").join("corpus_8k.json");
    let corpus_bytes = fs::read(&corpus_path)?;
    let corpus: serde_json::Value = serde_json::from_slice(&corpus_bytes)?;
    let entry = corpus["entries"].as_array().unwrap().iter().find(|e| e["id"] == "doc_physics_8192t").unwrap();
    let content = entry["content"].as_str().unwrap();

    let token_ids = embedder.tokenizer.encode(content)?;
    let input_ids = candle_core::Tensor::new(&token_ids[..8192], &device)?.unsqueeze(0)?;

    for mode in [AttentionMode::ChunkedSliding] {
        embedder.set_attention_mode(mode);
        let mode_str = match mode {
            AttentionMode::Standard => "Standard",
            AttentionMode::ChunkedSliding => "Chunked",
            #[allow(unreachable_patterns)]
            _ => "Optimized",
        };
        println!("\n=== Testing {} Attention ===", mode_str);
        println!("Before forward: {:.1} MB", get_metal_mb());

        // Forward through backbone step-by-step
        let model = &embedder.model;
        let inputs_embeds = model.embed_tokens.forward(&input_ids)?;
        let inputs_embeds = (inputs_embeds * model.embed_scale)?;
        println!("After embed_tokens: {:.1} MB", get_metal_mb());

        let ple_p1 = model.ple.projection.forward(&inputs_embeds)?;
        println!("After PLE linear projection: {:.1} MB", get_metal_mb());
        let ple_p2 = (ple_p1 * model.ple.scale)?;
        println!("After PLE scale: {:.1} MB", get_metal_mb());
        let per_layer_projections = ple_p2.reshape((1, 8192, model.ple.num_layers, model.ple.hidden_size_per_layer_input))?;
        println!("After PLE reshape: {:.1} MB", get_metal_mb());

        let (cos_sliding, sin_sliding) = model.rope_sliding.get_cos_sin(8192, &device, model.dtype)?;
        let (cos_full, sin_full) = model.rope_full.get_cos_sin(8192, &device, model.dtype)?;

        let is_chunked = mode == AttentionMode::ChunkedSliding;
        let sliding_window_for_mask = if is_chunked { None } else { Some(model.config.sliding_window) };
        let sliding_mask = model.build_attention_mask(1, 8192, sliding_window_for_mask, None)?;
        println!("After building sliding_mask: {:.1} MB", get_metal_mb());
        let full_mask = model.build_attention_mask(1, 8192, None, None)?;
        println!("After building full_mask: {:.1} MB", get_metal_mb());

        println!("--- Layer 00 detailed breakdown ---");
        let mut hidden_states = inputs_embeds;
        let l0 = &model.layers[0];
        let p0_raw = per_layer_projections.narrow(2, 0, 1)?.squeeze(2)?;
        let p0 = model.ple.norm.forward(&p0_raw)?;
        println!("L0 start (after sliced PLE norm): {:.1} MB", get_metal_mb());
        let norm0 = l0.input_layernorm.forward(&hidden_states)?;
        println!("L0 after input_layernorm: {:.1} MB", get_metal_mb());
        let attn0 = l0.self_attn.forward(&norm0, sliding_mask.as_ref(), &cos_sliding, &sin_sliding)?;
        println!("L0 after self_attn: {:.1} MB", get_metal_mb());
        let post_attn0 = l0.post_attention_layernorm.forward(&attn0)?;
        println!("L0 after post_attn_ln: {:.1} MB", get_metal_mb());
        let h0 = hidden_states.add(&post_attn0)?;
        println!("L0 after residual 1: {:.1} MB", get_metal_mb());
        let pre_ffn0 = l0.pre_feedforward_layernorm.forward(&h0)?;
        println!("L0 after pre_ffn_ln: {:.1} MB", get_metal_mb());
        let mlp0 = l0.mlp.forward(&pre_ffn0)?;
        println!("L0 after mlp: {:.1} MB", get_metal_mb());
        let post_ffn0 = l0.post_feedforward_layernorm.forward(&mlp0)?;
        println!("L0 after post_ffn_ln: {:.1} MB", get_metal_mb());
        let h0 = h0.add(&post_ffn0)?;
        println!("L0 after residual 2: {:.1} MB", get_metal_mb());
        let h0 = l0.ple_block.forward(&h0, &p0)?;
        println!("L0 after ple_block: {:.1} MB", get_metal_mb());
        let scalar0 = l0.layer_scalar.to_dtype(h0.dtype())?;
        hidden_states = h0.broadcast_mul(&scalar0)?;
        println!("L0 end: {:.1} MB", get_metal_mb());

        for i in 1..model.config.num_hidden_layers {
            let layer = &model.layers[i];
            let layer_type = model.config.layer_types[i];
            let p_raw = per_layer_projections.narrow(2, i, 1)?.squeeze(2)?;
            let per_layer_slice = model.ple.norm.forward(&p_raw)?;
            let (cos, sin, mask) = match layer_type {
                glypr_core::config::LayerType::SlidingAttention => (&cos_sliding, &sin_sliding, sliding_mask.as_ref()),
                glypr_core::config::LayerType::FullAttention => (&cos_full, &sin_full, full_mask.as_ref()),
            };

            // Sample peak during layer
            use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
            use std::sync::Arc;
            use std::thread;
            let running = Arc::new(AtomicBool::new(true));
            let peak = Arc::new(AtomicU64::new(get_metal_mb() as u64));
            let r_c = running.clone();
            let p_c = peak.clone();
            let handle = thread::spawn(move || {
                while r_c.load(Ordering::Relaxed) {
                    let m = get_metal_mb() as u64;
                    p_c.fetch_max(m, Ordering::Relaxed);
                    thread::sleep(std::time::Duration::from_millis(1));
                }
            });

            hidden_states = layer.forward(&hidden_states, &per_layer_slice, mask, cos, sin)?;

            running.store(false, Ordering::Relaxed);
            handle.join().unwrap();
            let peak_during = peak.load(Ordering::Relaxed);

            println!("Layer {:02} ({:?}) -> End: {:.1} MB | Peak during: {} MB", 
                i, layer_type, get_metal_mb(), peak_during);
        }

        let normed = model.norm.forward(&hidden_states)?;
        let _ = model.embedding_projection.forward(&normed)?;
        println!("End step-by-step forward: {:.1} MB", get_metal_mb());
    }

    Ok(())
}
