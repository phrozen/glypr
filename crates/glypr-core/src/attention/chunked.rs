use candle_core::{DType, Device, Result, Tensor, D};
use super::traits::AttentionBackend;
use super::standard::StandardAttention;

/// Precompute reusable static chunk masks for sliding window attention
fn get_static_chunk_masks(
    window: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, Tensor)> {
    // 1. First chunk mask: shape [1, 1, W, 2W]
    // global_q = i (0..W), global_k = j (0..2W) -> diff = j - i
    let mut mask_first_vec = vec![0.0f32; window * 2 * window];
    for i in 0..window {
        for j in 0..(2 * window) {
            let diff = if j >= i { j - i } else { i - j };
            if diff > window {
                mask_first_vec[i * 2 * window + j] = -1e9f32;
            }
        }
    }
    let mask_first = Tensor::from_vec(mask_first_vec, (1, 1, window, 2 * window), device)?
        .to_dtype(dtype)?;

    // 2. Middle chunk mask: shape [1, 1, W, 3W]
    // global_q = W + i, global_k = j (0..3W) -> diff = |W + i - j|
    let mut mask_middle_vec = vec![0.0f32; window * 3 * window];
    for i in 0..window {
        for j in 0..(3 * window) {
            let diff = if (window + i) >= j { (window + i) - j } else { j - (window + i) };
            if diff > window {
                mask_middle_vec[i * 3 * window + j] = -1e9f32;
            }
        }
    }
    let mask_middle = Tensor::from_vec(mask_middle_vec, (1, 1, window, 3 * window), device)?
        .to_dtype(dtype)?;

    // 3. Last chunk mask: shape [1, 1, W, 2W]
    // global_q = W + i, global_k = j (0..2W) -> diff = (W + i) - j
    let mut mask_last_vec = vec![0.0f32; window * 2 * window];
    for i in 0..window {
        for j in 0..(2 * window) {
            let diff = if (window + i) >= j { (window + i) - j } else { j - (window + i) };
            if diff > window {
                mask_last_vec[i * 2 * window + j] = -1e9f32;
            }
        }
    }
    let mask_last = Tensor::from_vec(mask_last_vec, (1, 1, window, 2 * window), device)?
        .to_dtype(dtype)?;

    Ok((mask_first, mask_middle, mask_last))
}

/// Chunked sliding-window attention backend.
/// Breaks query tokens into chunks of size `sliding_window` and only attends to
/// keys/values in the previous, current, and next chunks (3W span).
/// Reuses precomputed static mask buffers to avoid dynamic allocations during execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkedSlidingAttention;

impl AttentionBackend for ChunkedSlidingAttention {
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        let (_b, _h, s, _d) = q.dims4()?;

        let window = match sliding_window {
            Some(w) if s > w => w,
            _ => {
                return StandardAttention.compute_attention(q, k, v, attention_mask, sliding_window);
            }
        };

        let device = q.device();
        let orig_dtype = q.dtype();
        let chunk_size = window;
        let num_chunks = (s + chunk_size - 1) / chunk_size;

        // Precompute static mask tensors once outside the chunk loop
        let static_masks = if attention_mask.is_none() {
            Some(get_static_chunk_masks(window, device, orig_dtype)?)
        } else {
            None
        };

        let mut out_chunks = Vec::with_capacity(num_chunks);

        for chunk_idx in 0..num_chunks {
            let q_start = chunk_idx * chunk_size;
            let q_len = (s - q_start).min(chunk_size);

            let q_chunk = q.narrow(2, q_start, q_len)?;

            let kv_start = if chunk_idx > 0 { (chunk_idx - 1) * chunk_size } else { 0 };
            let kv_end = ((chunk_idx + 2) * chunk_size).min(s);
            let kv_len = kv_end - kv_start;

            let k_chunk = k.narrow(2, kv_start, kv_len)?;
            let v_chunk = v.narrow(2, kv_start, kv_len)?;

            let k_chunk_t = k_chunk.transpose(2, 3)?.contiguous()?;
            let mut attn_weights = q_chunk.matmul(&k_chunk_t)?;

            // Apply attention mask
            if let Some(mask) = attention_mask {
                // Slice provided global/padding mask
                let mask_chunk = mask.narrow(2, q_start, q_len)?.narrow(3, kv_start, kv_len)?;
                attn_weights = attn_weights.broadcast_add(&mask_chunk)?;
            } else if let Some((ref mask_first, ref mask_middle, ref mask_last)) = static_masks {
                // Zero-allocation path using precomputed static masks
                if chunk_idx == 0 {
                    let m = if q_len == window && kv_len == 2 * window {
                        mask_first
                    } else {
                        &mask_first.narrow(2, 0, q_len)?.narrow(3, 0, kv_len)?
                    };
                    attn_weights = attn_weights.broadcast_add(m)?;
                } else if chunk_idx == num_chunks - 1 {
                    let m = if q_len == window && kv_len == 2 * window {
                        mask_last
                    } else {
                        &mask_last.narrow(2, 0, q_len)?.narrow(3, 0, kv_len)?
                    };
                    attn_weights = attn_weights.broadcast_add(m)?;
                } else {
                    let m = if q_len == window && kv_len == 3 * window {
                        mask_middle
                    } else {
                        &mask_middle.narrow(2, 0, q_len)?.narrow(3, 0, kv_len)?
                    };
                    attn_weights = attn_weights.broadcast_add(m)?;
                }
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

            let out_chunk = attn_probs.matmul(&v_chunk)?;
            out_chunks.push(out_chunk);
        }

        Tensor::cat(&out_chunks, 2)
    }
}
