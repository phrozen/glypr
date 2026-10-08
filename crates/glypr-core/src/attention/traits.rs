use candle_core::{Result, Tensor};

/// Supported attention execution modes in glypr-core
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMode {
    /// Standard full dense attention (computes full NxN attention matrix).
    Standard,
    /// Chunked sliding-window attention (computes NxW banded chunks).
    /// Highly optimized, reduces peak RAM and compute by up to 16x on long contexts.
    /// Seamlessly falls back to Standard when seq_len <= window_size or on full attention layers.
    ChunkedSliding,
    /// Fused Metal SDPA using Apple Silicon Steel Attention with zero-copy broadcasted mask.
    #[cfg(any(feature = "metal", feature = "metal-flash-attn"))]
    MetalSdpa,
    /// Dedicated Metal FlashAttention with hardware tile-skipping (0 MB mask overhead).
    #[cfg(any(feature = "metal", feature = "metal-flash-attn"))]
    MetalFlash,
    #[cfg(feature = "cuda-flash-attn")]
    CudaFlash,
}

impl Default for AttentionMode {
    fn default() -> Self {
        AttentionMode::ChunkedSliding
    }
}

/// Pluggable attention backend trait
pub trait AttentionBackend: std::fmt::Debug + Send + Sync {
    fn compute_attention(
        &self,
        q: &Tensor, // [batch, num_heads, seq_len, head_dim]
        k: &Tensor, // [batch, num_heads, seq_len, head_dim]
        v: &Tensor, // [batch, num_heads, seq_len, head_dim]
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
    ) -> Result<Tensor>;
}
