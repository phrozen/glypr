use candle_core::{Result, Tensor, Device};
use super::traits::AttentionBackend;
use super::chunked::ChunkedSlidingAttention;

/// Fused Metal scaled dot-product attention backend.
/// Uses Apple Silicon hardware-accelerated Steel Attention kernels via candle-nn.
///
/// Features:
/// - Unmasked execution for global full-attention layers with zero memory overhead.
/// - Zero-copy broadcasted additive mask support for sliding-window and padded layers.
/// - Automatic fallback to ChunkedSlidingAttention when invoked on CPU.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalSdpaAttention;

impl AttentionBackend for MetalSdpaAttention {
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        match q.device() {
            Device::Metal(_) => {}
            _ => {
                // Seamless CPU fallback
                return ChunkedSlidingAttention.compute_attention(q, k, v, attention_mask, sliding_window);
            }
        }

        let (b, h, s, _d) = q.dims4()?;

        // If an attention mask is provided, ensure its dtype matches q and broadcast across heads
        let mask_broadcast = if let Some(mask) = attention_mask {
            let mask = if mask.dtype() != q.dtype() {
                mask.to_dtype(q.dtype())?
            } else {
                mask.clone()
            };

            if mask.dims() == &[b, h, s, s] {
                Some(mask)
            } else {
                // Zero-copy strided view across attention heads: [b, 1, s, s] -> [b, h, s, s]
                Some(mask.broadcast_as((b, h, s, s))?)
            }
        } else {
            None
        };

        // Gemma 2 attention: scale is 1.0 (pre-scaled in projections/RoPE), no causal mask, softcapping disabled (1.0)
        candle_nn::ops::sdpa(q, k, v, mask_broadcast.as_ref(), false, 1.0, 1.0)
    }
}
