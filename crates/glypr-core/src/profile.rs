use candle_core::{Device, Module, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::attention::repeat_kv;
use crate::config::LayerType;
use crate::model::EmbeddingGemma2Model;
use crate::rope::apply_rotary_pos_emb;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerProfile {
    pub layer_idx: usize,
    pub is_sliding: bool,
    pub input_norm_ms: f64,
    pub qkv_proj_ms: f64,
    pub qk_norm_rope_ms: f64,
    pub attn_core_ms: f64,
    pub out_proj_ms: f64,
    pub post_attn_norm_ms: f64,
    pub ffn_pre_norm_ms: f64,
    pub mlp_ms: f64,
    pub post_ffn_norm_ms: f64,
    pub ple_block_ms: f64,
    pub total_layer_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardProfile {
    pub batch_size: usize,
    pub seq_len: usize,
    pub embed_tokens_ms: f64,
    pub ple_input_ms: f64,
    pub rope_calc_ms: f64,
    pub mask_build_ms: f64,
    pub layers: Vec<LayerProfile>,
    pub total_layers_ms: f64,
    pub final_norm_proj_ms: f64,
    pub total_forward_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineProfile {
    pub batch_size: usize,
    pub seq_len: usize,
    pub tokenizer_ms: f64,
    pub forward: ForwardProfile,
    pub pooling_ms: f64,
    pub mrl_l2norm_ms: f64,
    pub total_pipeline_ms: f64,
    pub tokens_per_sec: f64,
}

fn sync(t: &Tensor) -> Result<()> {
    let _ = t.flatten_all()?.narrow(0, 0, 1)?.to_device(&Device::Cpu)?;
    Ok(())
}

fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

impl EmbeddingGemma2Model {
    pub fn forward_profiled(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, ForwardProfile)> {
        let (batch, seq_len) = input_ids.dims2()?;
        let t_start_total = Instant::now();

        // 1. Embed tokens with scaling
        let t0 = Instant::now();
        let inputs_embeds = self.embed_tokens.forward(input_ids)?;
        let inputs_embeds = (inputs_embeds * self.embed_scale)?;
        sync(&inputs_embeds)?;
        let embed_tokens_ms = elapsed_ms(t0);

        // 2. Per-layer embeddings: [24, batch, seq_len, 512]
        let t0 = Instant::now();
        let per_layer_inputs = self.ple.project_and_norm_per_layer(&inputs_embeds)?;
        sync(&per_layer_inputs)?;
        let ple_input_ms = elapsed_ms(t0);

        // 3. Precompute RoPE (cos, sin)
        let t0 = Instant::now();
        let (cos_sliding, sin_sliding) =
            self.rope_sliding
                .get_cos_sin(seq_len, &self.device, self.dtype)?;
        let (cos_full, sin_full) = self
            .rope_full
            .get_cos_sin(seq_len, &self.device, self.dtype)?;
        sync(&cos_sliding)?;
        let rope_calc_ms = elapsed_ms(t0);

        // 4. Attention masks
        let t0 = Instant::now();
        let is_chunked_or_flash = !self.layers.is_empty()
            && match self.layers[0].self_attn.attention_mode {
                crate::attention::AttentionMode::ChunkedSliding => true,
                #[cfg(any(feature = "metal", feature = "metal-flash-attn"))]
                crate::attention::AttentionMode::MetalFlash => true,
                _ => false,
            };
        let sliding_window_for_mask = if is_chunked_or_flash {
            None
        } else {
            Some(self.config.sliding_window)
        };
        let sliding_mask =
            self.build_attention_mask(batch, seq_len, sliding_window_for_mask, attention_mask)?;
        let full_mask = self.build_attention_mask(batch, seq_len, None, attention_mask)?;
        if let Some(ref m) = sliding_mask {
            sync(m)?;
        }
        let mask_build_ms = elapsed_ms(t0);

        // 5. Execute 24 layers with detailed timings
        let mut hidden_states = inputs_embeds;
        let mut layer_profiles = Vec::with_capacity(self.config.num_hidden_layers);

        for i in 0..self.config.num_hidden_layers {
            let t_layer_start = Instant::now();
            let layer = &self.layers[i];
            let layer_type = self.config.layer_types[i];
            let is_sliding = layer_type == LayerType::SlidingAttention;

            let per_layer_slice = per_layer_inputs.get(i)?;
            let (cos, sin, mask) = match layer_type {
                LayerType::SlidingAttention => (&cos_sliding, &sin_sliding, sliding_mask.as_ref()),
                LayerType::FullAttention => (&cos_full, &sin_full, full_mask.as_ref()),
            };

            // Attention block input norm
            let t0 = Instant::now();
            let residual = hidden_states.clone();
            let normed = layer.input_layernorm.forward(&hidden_states)?;
            sync(&normed)?;
            let input_norm_ms = elapsed_ms(t0);

            // Self attention QKV proj
            let t0 = Instant::now();
            let q = layer.self_attn.q_proj.forward(&normed)?;
            let k = layer.self_attn.k_proj.forward(&normed)?;
            let v = layer.self_attn.v_proj.forward(&normed)?;
            sync(&v)?;
            let qkv_proj_ms = elapsed_ms(t0);

            // QK Norm + RoPE + Transpose
            let t0 = Instant::now();
            #[cfg(feature = "metal")]
            let (q, k, v) = if q.device().is_metal()
                && q.dtype() == candle_core::DType::BF16
                && layer.self_attn.q_norm.weight().is_some()
                && layer.self_attn.k_norm.weight().is_some()
            {
                let q = crate::metal_kernels::fused_metal_qk_rmsnorm_rope_transpose(
                    &q,
                    layer.self_attn.q_norm.weight().unwrap(),
                    cos,
                    sin,
                    batch,
                    seq_len,
                    layer.self_attn.num_attention_heads,
                    layer.self_attn.head_dim,
                    layer.self_attn.q_norm.eps(),
                )?;
                let k = crate::metal_kernels::fused_metal_qk_rmsnorm_rope_transpose(
                    &k,
                    layer.self_attn.k_norm.weight().unwrap(),
                    cos,
                    sin,
                    batch,
                    seq_len,
                    layer.self_attn.num_key_value_heads,
                    layer.self_attn.head_dim,
                    layer.self_attn.k_norm.eps(),
                )?;
                let v = crate::metal_kernels::fused_metal_v_rmsnorm_transpose(
                    &v,
                    layer.self_attn.v_norm.weight(),
                    batch,
                    seq_len,
                    layer.self_attn.num_key_value_heads,
                    layer.self_attn.head_dim,
                    layer.self_attn.v_norm.eps(),
                )?;
                (q, k, v)
            } else {
                let q = layer
                    .self_attn
                    .q_norm
                    .forward(&q.reshape((
                        batch,
                        seq_len,
                        layer.self_attn.num_attention_heads,
                        layer.self_attn.head_dim,
                    ))?)?
                    .transpose(1, 2)?
                    .contiguous()?;
                let k = layer
                    .self_attn
                    .k_norm
                    .forward(&k.reshape((
                        batch,
                        seq_len,
                        layer.self_attn.num_key_value_heads,
                        layer.self_attn.head_dim,
                    ))?)?
                    .transpose(1, 2)?
                    .contiguous()?;
                let v = layer
                    .self_attn
                    .v_norm
                    .forward(&v.reshape((
                        batch,
                        seq_len,
                        layer.self_attn.num_key_value_heads,
                        layer.self_attn.head_dim,
                    ))?)?
                    .transpose(1, 2)?
                    .contiguous()?;
                let q = apply_rotary_pos_emb(&q, cos, sin)?;
                let k = apply_rotary_pos_emb(&k, cos, sin)?;
                (q, k, v)
            };
            #[cfg(not(feature = "metal"))]
            let (q, k, v) = {
                let q = layer
                    .self_attn
                    .q_norm
                    .forward(&q.reshape((
                        batch,
                        seq_len,
                        layer.self_attn.num_attention_heads,
                        layer.self_attn.head_dim,
                    ))?)?
                    .transpose(1, 2)?
                    .contiguous()?;
                let k = layer
                    .self_attn
                    .k_norm
                    .forward(&k.reshape((
                        batch,
                        seq_len,
                        layer.self_attn.num_key_value_heads,
                        layer.self_attn.head_dim,
                    ))?)?
                    .transpose(1, 2)?
                    .contiguous()?;
                let v = layer
                    .self_attn
                    .v_norm
                    .forward(&v.reshape((
                        batch,
                        seq_len,
                        layer.self_attn.num_key_value_heads,
                        layer.self_attn.head_dim,
                    ))?)?
                    .transpose(1, 2)?
                    .contiguous()?;
                let q = apply_rotary_pos_emb(&q, cos, sin)?;
                let k = apply_rotary_pos_emb(&k, cos, sin)?;
                (q, k, v)
            };
            sync(&k)?;
            let qk_norm_rope_ms = elapsed_ms(t0);

            // Attention Core (dispatched via configured AttentionBackend)
            let t0 = Instant::now();
            let context = match layer.self_attn.attention_mode {
                #[cfg(any(feature = "metal", feature = "metal-flash-attn"))]
                crate::attention::AttentionMode::MetalFlash => {
                    use crate::attention::AttentionBackend;
                    crate::attention::MetalFlashAttention.compute_attention(
                        &q,
                        &k,
                        &v,
                        mask,
                        layer.self_attn.sliding_window,
                    )?
                }
                _ => {
                    use crate::attention::AttentionBackend;
                    let num_groups =
                        layer.self_attn.num_attention_heads / layer.self_attn.num_key_value_heads;
                    let k_rep = repeat_kv(&k, num_groups)?;
                    let v_rep = repeat_kv(&v, num_groups)?;
                    match layer.self_attn.attention_mode {
                        crate::attention::AttentionMode::Standard => {
                            crate::attention::StandardAttention.compute_attention(
                                &q,
                                &k_rep,
                                &v_rep,
                                mask,
                                layer.self_attn.sliding_window,
                            )?
                        }
                        crate::attention::AttentionMode::ChunkedSliding => {
                            crate::attention::ChunkedSlidingAttention.compute_attention(
                                &q,
                                &k_rep,
                                &v_rep,
                                mask,
                                layer.self_attn.sliding_window,
                            )?
                        }
                        #[cfg(any(feature = "metal", feature = "metal-flash-attn"))]
                        crate::attention::AttentionMode::MetalSdpa => {
                            crate::attention::MetalSdpaAttention.compute_attention(
                                &q,
                                &k_rep,
                                &v_rep,
                                mask,
                                layer.self_attn.sliding_window,
                            )?
                        }
                        #[cfg(feature = "cuda-flash-attn")]
                        crate::attention::AttentionMode::CudaFlash => {
                            crate::attention::ChunkedSlidingAttention.compute_attention(
                                &q,
                                &k_rep,
                                &v_rep,
                                mask,
                                layer.self_attn.sliding_window,
                            )?
                        }
                        #[allow(unreachable_patterns)]
                        _ => unreachable!(),
                    }
                }
            };
            let context = context.transpose(1, 2)?.contiguous()?.reshape((
                batch,
                seq_len,
                layer.self_attn.num_attention_heads * layer.self_attn.head_dim,
            ))?;
            sync(&context)?;
            let attn_core_ms = elapsed_ms(t0);

            // Output projection
            let t0 = Instant::now();
            let attn_out = layer.self_attn.o_proj.forward(&context)?;
            sync(&attn_out)?;
            let out_proj_ms = elapsed_ms(t0);

            // Post-attn norm + residual add
            let t0 = Instant::now();
            let hidden_states_next = layer
                .post_attention_layernorm
                .forward_residual_add(&attn_out, &residual, None)?;
            sync(&hidden_states_next)?;
            let post_attn_norm_ms = elapsed_ms(t0);

            // FFN pre norm
            let t0 = Instant::now();
            let residual = hidden_states_next.clone();
            let pre_ffn = layer
                .pre_feedforward_layernorm
                .forward(&hidden_states_next)?;
            sync(&pre_ffn)?;
            let ffn_pre_norm_ms = elapsed_ms(t0);

            // MLP
            let t0 = Instant::now();
            let mlp_out = layer.mlp.forward(&pre_ffn)?;
            sync(&mlp_out)?;
            let mlp_ms = elapsed_ms(t0);

            // Post FFN norm + residual add
            let t0 = Instant::now();
            let hidden_states_next = layer
                .post_feedforward_layernorm
                .forward_residual_add(&mlp_out, &residual, None)?;
            sync(&hidden_states_next)?;
            let post_ffn_norm_ms = elapsed_ms(t0);

            // PLE Block + Layer scalar
            let t0 = Instant::now();
            hidden_states = layer.ple_block.forward_with_scalar(
                &hidden_states_next,
                &per_layer_slice,
                Some(&layer.layer_scalar),
            )?;
            sync(&hidden_states)?;
            let ple_block_ms = elapsed_ms(t0);

            let total_layer_ms = elapsed_ms(t_layer_start);
            layer_profiles.push(LayerProfile {
                layer_idx: i,
                is_sliding,
                input_norm_ms,
                qkv_proj_ms,
                qk_norm_rope_ms,
                attn_core_ms,
                out_proj_ms,
                post_attn_norm_ms,
                ffn_pre_norm_ms,
                mlp_ms,
                post_ffn_norm_ms,
                ple_block_ms,
                total_layer_ms,
            });
        }

        // 6. Final norm & projection
        let t0 = Instant::now();
        let normed = self.norm.forward(&hidden_states)?;
        let output = self.embedding_projection.forward(&normed)?;
        sync(&output)?;
        let final_norm_proj_ms = elapsed_ms(t0);

        let total_forward_ms = elapsed_ms(t_start_total);
        let total_layers_ms = layer_profiles.iter().map(|l| l.total_layer_ms).sum();

        let profile = ForwardProfile {
            batch_size: batch,
            seq_len,
            embed_tokens_ms,
            ple_input_ms,
            rope_calc_ms,
            mask_build_ms,
            layers: layer_profiles,
            total_layers_ms,
            final_norm_proj_ms,
            total_forward_ms,
        };

        Ok((output, profile))
    }
}
