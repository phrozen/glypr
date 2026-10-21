use candle_core::{D, DType, Device, Result, Tensor};

/// Standard full dense attention with sliding window mask
fn standard_attention(
    q: &Tensor, // [b, h, s, d]
    k: &Tensor, // [b, h, s, d]
    v: &Tensor, // [b, h, s, d]
    window: usize,
) -> Result<Tensor> {
    let (_b, _h, s, _d) = q.dims4()?;
    let device = q.device();

    // Build dense mask: [1, 1, s, s]
    let mut mask_vec = vec![0.0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            let diff = i.abs_diff(j);
            if diff > window {
                mask_vec[i * s + j] = -1e9f32;
            }
        }
    }
    let mask = Tensor::from_vec(mask_vec, (1, 1, s, s), device)?.to_dtype(q.dtype())?;

    let k_t = k.transpose(2, 3)?.contiguous()?;
    let mut attn_weights = q.matmul(&k_t)?;
    attn_weights = attn_weights.broadcast_add(&mask)?;

    let orig_dtype = attn_weights.dtype();
    let attn_weights_f32 = if orig_dtype != DType::F32 {
        attn_weights.to_dtype(DType::F32)?
    } else {
        attn_weights
    };
    let attn_probs_f32 = candle_nn::ops::softmax(&attn_weights_f32, D::Minus1)?;
    let attn_probs = if orig_dtype != DType::F32 {
        attn_probs_f32.to_dtype(orig_dtype)?
    } else {
        attn_probs_f32
    };

    attn_probs.matmul(v)
}

/// Chunked sliding window attention
fn chunked_sliding_attention(
    q: &Tensor, // [b, h, s, d]
    k: &Tensor, // [b, h, s, d]
    v: &Tensor, // [b, h, s, d]
    window: usize,
) -> Result<Tensor> {
    let (_b, _h, s, _d) = q.dims4()?;
    let device = q.device();
    let orig_dtype = q.dtype();

    // If sequence length is within window, chunking is not needed
    if s <= window {
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn_weights = q.matmul(&k_t)?;
        let attn_weights_f32 = attn_weights.to_dtype(DType::F32)?;
        let attn_probs_f32 = candle_nn::ops::softmax(&attn_weights_f32, D::Minus1)?;
        let attn_probs = attn_probs_f32.to_dtype(orig_dtype)?;
        return attn_probs.matmul(v);
    }

    let chunk_size = window;
    let num_chunks = s.div_ceil(chunk_size);

    let mut out_chunks = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let q_start = chunk_idx * chunk_size;
        let q_len = (s - q_start).min(chunk_size);

        // Query chunk: [b, h, q_len, d]
        let q_chunk = q.narrow(2, q_start, q_len)?;

        // Key/Value window: chunk_idx - 1 to chunk_idx + 1
        let kv_start = if chunk_idx > 0 {
            (chunk_idx - 1) * chunk_size
        } else {
            0
        };
        let kv_end = ((chunk_idx + 2) * chunk_size).min(s);
        let kv_len = kv_end - kv_start;

        let k_chunk = k.narrow(2, kv_start, kv_len)?;
        let v_chunk = v.narrow(2, kv_start, kv_len)?;

        // Attention weights for this chunk: [b, h, q_len, kv_len]
        let k_chunk_t = k_chunk.transpose(2, 3)?.contiguous()?;
        let mut attn_weights = q_chunk.matmul(&k_chunk_t)?;

        // Build small local mask of shape [1, 1, q_len, kv_len]
        let mut local_mask = vec![0.0f32; q_len * kv_len];
        let mut has_masked = false;
        for i in 0..q_len {
            let global_q = q_start + i;
            for j in 0..kv_len {
                let global_k = kv_start + j;
                let diff = global_q.abs_diff(global_k);
                if diff > window {
                    local_mask[i * kv_len + j] = -1e9f32;
                    has_masked = true;
                }
            }
        }

        if has_masked {
            let mask_t = Tensor::from_vec(local_mask, (1, 1, q_len, kv_len), device)?
                .to_dtype(orig_dtype)?;
            attn_weights = attn_weights.broadcast_add(&mask_t)?;
        }

        let attn_weights_f32 = if orig_dtype != DType::F32 {
            attn_weights.to_dtype(DType::F32)?
        } else {
            attn_weights
        };
        let attn_probs_f32 = candle_nn::ops::softmax(&attn_weights_f32, D::Minus1)?;
        let attn_probs = if orig_dtype != DType::F32 {
            attn_probs_f32.to_dtype(orig_dtype)?
        } else {
            attn_probs_f32
        };

        // Context for chunk: [b, h, q_len, d]
        let out_chunk = attn_probs.matmul(&v_chunk)?;
        out_chunks.push(out_chunk);
    }

    // Concatenate chunks along sequence dimension: [b, h, s, d]
    Tensor::cat(&out_chunks, 2)
}

fn main() -> anyhow::Result<()> {
    println!("============================================================");
    println!("  Validating Chunked Sliding Attention vs Standard Attention");
    println!("============================================================");

    #[cfg(feature = "metal")]
    let (device, dtype) = (Device::new_metal(0)?, DType::BF16);
    #[cfg(not(feature = "metal"))]
    let (device, dtype) = (Device::Cpu, DType::F32);

    println!("Running on Device: {:?} (DType: {:?})", device, dtype);

    let b = 1;
    let h = 4;
    let d = 256;
    let window = 512;

    for &s in &[600, 1024, 2048, 4096, 8192] {
        println!("\nTesting sequence length s = {} (window = {}):", s, window);

        let q = Tensor::randn(0.0f32, 1.0f32, (b, h, s, d), &device)?.to_dtype(dtype)?;
        let k = Tensor::randn(0.0f32, 1.0f32, (b, h, s, d), &device)?.to_dtype(dtype)?;
        let v = Tensor::randn(0.0f32, 1.0f32, (b, h, s, d), &device)?.to_dtype(dtype)?;

        let t0 = std::time::Instant::now();
        let standard_out = standard_attention(&q, &k, &v, window)?;
        let standard_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = std::time::Instant::now();
        let chunked_out = chunked_sliding_attention(&q, &k, &v, window)?;
        let chunked_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Verify shape
        assert_eq!(standard_out.dims4()?, chunked_out.dims4()?);

        // Compare values
        let diff =
            (&standard_out.to_dtype(DType::F32)? - &chunked_out.to_dtype(DType::F32)?)?.abs()?;
        let max_diff: f32 = diff.max_all()?.to_scalar()?;
        let mean_diff: f32 = diff.mean_all()?.to_scalar()?;

        println!("  Standard Time : {:7.2} ms", standard_ms);
        println!(
            "  Chunked Time  : {:7.2} ms ({:.2}x speedup)",
            chunked_ms,
            standard_ms / chunked_ms
        );
        println!("  Max Diff      : {:e}", max_diff);
        println!("  Mean Diff     : {:e}", mean_diff);

        if max_diff < 1e-3 {
            println!("  ✓ EXACT PARITY VERIFIED!");
        } else {
            println!("  ✗ PARITY FAILURE!");
        }
    }

    Ok(())
}
