pub mod traits;
pub mod standard;
pub mod chunked;

pub use traits::{AttentionBackend, AttentionMode};
pub use standard::StandardAttention;
pub use chunked::ChunkedSlidingAttention;
#[cfg(any(feature = "metal", feature = "metal-flash-attn"))]
pub mod metal_sdpa;
#[cfg(any(feature = "metal", feature = "metal-flash-attn"))]
pub use metal_sdpa::MetalSdpaAttention;
#[cfg(any(feature = "metal", feature = "metal-flash-attn"))]
pub mod metal_flash;
#[cfg(any(feature = "metal", feature = "metal-flash-attn"))]
pub use metal_flash::MetalFlashAttention;

use candle_core::{Result, Tensor};
use crate::linear::LinearOp;
use crate::norm::RMSNorm;
use crate::rope::apply_rotary_pos_emb;

pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, n_kv, n_rep, s, d))?
        .contiguous()?
        .reshape((b, n_kv * n_rep, s, d))
}

#[derive(Debug, Clone)]
pub struct SelfAttention {
    pub q_proj: LinearOp,
    pub k_proj: LinearOp,
    pub v_proj: LinearOp,
    pub o_proj: LinearOp,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub v_norm: RMSNorm,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub sliding_window: Option<usize>,
    pub attention_mode: AttentionMode,
}

impl SelfAttention {
    pub fn new(
        q_proj: LinearOp,
        k_proj: LinearOp,
        v_proj: LinearOp,
        o_proj: LinearOp,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        v_norm: RMSNorm,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            sliding_window,
            attention_mode: AttentionMode::default(),
        }
    }

    pub fn set_attention_mode(&mut self, mode: AttentionMode) {
        self.attention_mode = mode;
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden_size) = hidden_states.dims3()?;

        // Projections
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Fused Q/K RMSNorm + RoPE + Transpose and V RMSNorm + Transpose on Metal BF16
        #[cfg(feature = "metal")]
        let (q, k, v) = if q.device().is_metal()
            && q.dtype() == candle_core::DType::BF16
            && self.q_norm.weight().is_some()
            && self.k_norm.weight().is_some()
        {
            let q = crate::metal_kernels::fused_metal_qk_rmsnorm_rope_transpose(
                &q,
                self.q_norm.weight().unwrap(),
                cos,
                sin,
                batch,
                seq_len,
                self.num_attention_heads,
                self.head_dim,
                self.q_norm.eps(),
            )?;
            let k = crate::metal_kernels::fused_metal_qk_rmsnorm_rope_transpose(
                &k,
                self.k_norm.weight().unwrap(),
                cos,
                sin,
                batch,
                seq_len,
                self.num_key_value_heads,
                self.head_dim,
                self.k_norm.eps(),
            )?;
            let v = crate::metal_kernels::fused_metal_v_rmsnorm_transpose(
                &v,
                self.v_norm.weight(),
                batch,
                seq_len,
                self.num_key_value_heads,
                self.head_dim,
                self.v_norm.eps(),
            )?;
            (q, k, v)
        } else {
            // Normalize while [batch, seq_len, num_heads, head_dim] is still contiguous, then transpose
            let q = self
                .q_norm
                .forward(&q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?)?
                .transpose(1, 2)?
                .contiguous()?;
            let k = self
                .k_norm
                .forward(&k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?)?
                .transpose(1, 2)?
                .contiguous()?;
            let v = self
                .v_norm
                .forward(&v.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?)?
                .transpose(1, 2)?
                .contiguous()?;
            let q = apply_rotary_pos_emb(&q, cos, sin)?;
            let k = apply_rotary_pos_emb(&k, cos, sin)?;
            (q, k, v)
        };

        #[cfg(not(feature = "metal"))]
        let (q, k, v) = {
            let q = self
                .q_norm
                .forward(&q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?)?
                .transpose(1, 2)?
                .contiguous()?;
            let k = self
                .k_norm
                .forward(&k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?)?
                .transpose(1, 2)?
                .contiguous()?;
            let v = self
                .v_norm
                .forward(&v.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?)?
                .transpose(1, 2)?
                .contiguous()?;
            let q = apply_rotary_pos_emb(&q, cos, sin)?;
            let k = apply_rotary_pos_emb(&k, cos, sin)?;
            (q, k, v)
        };

        // Pluggable Attention Computation
        let context = match self.attention_mode {
            #[cfg(any(feature = "metal", feature = "metal-flash-attn"))]
            AttentionMode::MetalFlash => {
                // Native GQA: zero memory copies or KV head duplication!
                MetalFlashAttention.compute_attention(&q, &k, &v, attention_mask, self.sliding_window)?
            }
            _ => {
                let num_groups = self.num_attention_heads / self.num_key_value_heads;
                let k_rep = repeat_kv(&k, num_groups)?;
                let v_rep = repeat_kv(&v, num_groups)?;
                match self.attention_mode {
                    AttentionMode::Standard => {
                        StandardAttention.compute_attention(&q, &k_rep, &v_rep, attention_mask, self.sliding_window)?
                    }
                    AttentionMode::ChunkedSliding => {
                        ChunkedSlidingAttention.compute_attention(&q, &k_rep, &v_rep, attention_mask, self.sliding_window)?
                    }
                    #[cfg(any(feature = "metal", feature = "metal-flash-attn"))]
                    AttentionMode::MetalSdpa => {
                        MetalSdpaAttention.compute_attention(&q, &k_rep, &v_rep, attention_mask, self.sliding_window)?
                    }
                    #[cfg(feature = "cuda-flash-attn")]
                    AttentionMode::CudaFlash => {
                        ChunkedSlidingAttention.compute_attention(&q, &k_rep, &v_rep, attention_mask, self.sliding_window)?
                    }
                    #[allow(unreachable_patterns)]
                    _ => unreachable!(),
                }
            }
        };

        // Transpose back to [batch, seq_len, num_heads, head_dim] -> reshape [batch, seq_len, num_heads * head_dim]
        let context = context
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;

        self.o_proj.forward(&context)
    }
}
