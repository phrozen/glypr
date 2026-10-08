use candle_core::{Result, Tensor};
use crate::attention::SelfAttention;
use crate::mlp::MLP;
use crate::norm::RMSNorm;
use crate::ple::TextPLEBlock;

#[derive(Debug, Clone)]
pub struct EncoderLayer {
    pub self_attn: SelfAttention,
    pub mlp: MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub pre_feedforward_layernorm: RMSNorm,
    pub post_feedforward_layernorm: RMSNorm,
    pub ple_block: TextPLEBlock,
    pub layer_scalar: Tensor, // [1]
}

impl EncoderLayer {
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        per_layer_input: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        // 1. Attention Block
        let residual = hidden_states;
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_out = self.self_attn.forward(&normed, attention_mask, cos, sin)?;
        let post_attn = self.post_attention_layernorm.forward(&attn_out)?;
        let hidden_states = residual.add(&post_attn)?;

        // 2. Feedforward Block
        let residual = &hidden_states;
        let pre_ffn = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_out = self.mlp.forward(&pre_ffn)?;
        let post_ffn = self.post_feedforward_layernorm.forward(&mlp_out)?;
        let hidden_states = residual.add(&post_ffn)?;

        // 3. Per-Layer Embeddings (PLE) Block
        let hidden_states = self.ple_block.forward(&hidden_states, per_layer_input)?;

        // 4. Layer scalar multiplication
        let scalar = self.layer_scalar.to_dtype(hidden_states.dtype())?;
        hidden_states.broadcast_mul(&scalar)
    }
}
