use candle_core::{DType, Result, Tensor, D};
use super::traits::AttentionBackend;

/// Standard full dense attention (computes full NxN matrix).
/// Used for short sequences, full-attention layers, and as the reference ground truth.
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardAttention;

impl AttentionBackend for StandardAttention {
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        _sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        let (_b, _h, s, _d) = q.dims4()?;

        // If sequence length is large (> 512), chunk queries to eliminate the gigabyte-scale
        // [batch, heads, seq, seq] matrix during float32 softmax.
        // Row-wise softmax along dim=-1 is mathematically identical per query token.
        let chunk_size = 512;
        if s > chunk_size {
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let num_chunks = s.div_ceil(chunk_size);
            let mut out_chunks = Vec::with_capacity(num_chunks);
            let orig_dtype = q.dtype();

            for chunk_idx in 0..num_chunks {
                let q_start = chunk_idx * chunk_size;
                let q_len = (s - q_start).min(chunk_size);
                let q_chunk = q.narrow(2, q_start, q_len)?;

                let mut attn_weights = q_chunk.matmul(&k_t)?;

                if let Some(mask) = attention_mask {
                    let mask_chunk = mask.narrow(2, q_start, q_len)?;
                    attn_weights = attn_weights.broadcast_add(&mask_chunk)?;
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

                let out_chunk = attn_probs.matmul(v)?;
                out_chunks.push(out_chunk);
            }

            return Tensor::cat(&out_chunks, 2);
        }

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let mut attn_weights = q.matmul(&k_t)?;

        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        // Softmax in float32 for numerical stability
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
}
