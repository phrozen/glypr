use crate::attention::SelfAttention;
use crate::mlp::MLP;
use crate::norm::RMSNorm;
use crate::ple::TextPLEBlock;
use candle_core::{Result, Tensor};

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
        let hidden_states = self
            .post_attention_layernorm
            .forward_residual_add(&attn_out, residual, None)?;

        // 2. Feedforward Block
        let residual = &hidden_states;
        let pre_ffn = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_out = self.mlp.forward(&pre_ffn)?;
        let hidden_states = self
            .post_feedforward_layernorm
            .forward_residual_add(&mlp_out, residual, None)?;

        // 3. Per-Layer Embeddings (PLE) Block + Layer scalar multiplication
        self.ple_block.forward_with_scalar(
            &hidden_states,
            per_layer_input,
            Some(&self.layer_scalar),
        )
    }
}
