use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::LayerNorm;

use crate::config::AudioConfig;
use crate::linear::LinearOp;
use crate::multimodal::audio_preprocessor::ProcessedAudio;
use crate::norm::RMSNorm;

fn load_gguf_norm_tensor<R: std::io::Read + std::io::Seek>(
    content: &candle_core::quantized::gguf_file::Content,
    reader: &mut R,
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let qt = content.tensor(reader, name, device)?;
    let t = qt.dequantize(device)?;
    t.to_dtype(dtype)
}

fn load_gguf_linear_op<R: std::io::Read + std::io::Seek>(
    content: &candle_core::quantized::gguf_file::Content,
    reader: &mut R,
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<LinearOp> {
    LinearOp::from_gguf_tensor(content, reader, name, device, dtype)
}
fn load_optional_f32_scalar<R: std::io::Read + std::io::Seek>(
    content: &candle_core::quantized::gguf_file::Content,
    reader: &mut R,
    name: &str,
) -> Option<f32> {
    if let Ok(qt) = content.tensor(reader, name, &Device::Cpu)
        && let Ok(t) = qt.dequantize(&Device::Cpu)
            && let Ok(vec) = t.to_vec1::<f32>() {
                return vec.first().copied();
            }
    None
}

/// Clippable Linear Operator with activation bounds from GGUF (input_min, input_max, output_min, output_max)
#[derive(Debug, Clone)]
pub struct ClippableLinearOp {
    pub linear: LinearOp,
    pub input_min: Option<f32>,
    pub input_max: Option<f32>,
    pub output_min: Option<f32>,
    pub output_max: Option<f32>,
}

impl ClippableLinearOp {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_clamped = match (self.input_min, self.input_max) {
            (Some(min), Some(max)) => x.clamp(min, max)?,
            (Some(min), None) => x.clamp(min, f32::INFINITY)?,
            (None, Some(max)) => x.clamp(f32::NEG_INFINITY, max)?,
            (None, None) => x.clone(),
        };

        let out = self.linear.forward(&x_clamped)?;

        let out_clamped = match (self.output_min, self.output_max) {
            (Some(min), Some(max)) => out.clamp(min, max)?,
            (Some(min), None) => out.clamp(min, f32::INFINITY)?,
            (None, Some(max)) => out.clamp(f32::NEG_INFINITY, max)?,
            (None, None) => out,
        };

        Ok(out_clamped)
    }
}

fn load_gguf_clippable_linear_op<R: std::io::Read + std::io::Seek>(
    content: &candle_core::quantized::gguf_file::Content,
    reader: &mut R,
    base_name: &str,
    device: &Device,
    dtype: DType,
) -> Result<ClippableLinearOp> {
    let weight_name = format!("{}.weight", base_name);
    let linear = load_gguf_linear_op(content, reader, &weight_name, device, dtype)?;
    let input_min = load_optional_f32_scalar(content, reader, &format!("{}.input_min", base_name));
    let input_max = load_optional_f32_scalar(content, reader, &format!("{}.input_max", base_name));
    let output_min = load_optional_f32_scalar(content, reader, &format!("{}.output_min", base_name));
    let output_max = load_optional_f32_scalar(content, reader, &format!("{}.output_max", base_name));

    Ok(ClippableLinearOp {
        linear,
        input_min,
        input_max,
        output_min,
        output_max,
    })
}

/// Subsample Conv Layer: Conv2D(3x3, stride=2, pad=1) + LayerNorm + ReLU
pub struct AudioSubsampleLayer {
    pub conv_weight: Tensor, // [out_channels, in_channels, 3, 3]
    pub norm_weight: Tensor, // [out_channels]
    pub out_channels: usize,
    pub eps: f64,
}

impl AudioSubsampleLayer {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, in_channels, seq_len, mel_bins]
        let conv_out = x.conv2d(&self.conv_weight, 1, 2, 1, 1)?; // padding=1, stride=2, dilation=1

        // LayerNorm over channel dimension after permuting to [batch, seq_len, mel_bins, out_channels]
        let permuted = conv_out.permute((0, 2, 3, 1))?;
        let ln = LayerNorm::new_no_bias(self.norm_weight.clone(), self.eps);
        let normed = ln.forward(&permuted)?;
        let relu_out = normed.relu()?;
        relu_out.permute((0, 3, 1, 2))?.contiguous()
    }
}

/// 2-Stage Audio Subsampling Convolution Projection (downsamples time 4x)
pub struct AudioSubsampleConv {
    pub layer0: AudioSubsampleLayer,
    pub layer1: AudioSubsampleLayer,
    pub input_proj_linear: LinearOp, // [1024, 1024]
}

impl AudioSubsampleConv {
    pub fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        // input_features: [batch, seq_len, 128] -> [batch, 1, seq_len, 128]
        let x = input_features.unsqueeze(1)?;
        let x = self.layer0.forward(&x)?;
        let x = self.layer1.forward(&x)?;

        let (batch, _channels, seq_len, _bins) = x.dims4()?;
        // Permute to [batch, seq_len, bins, channels] and flatten to [batch, seq_len, channels * bins]
        let x = x.permute((0, 2, 3, 1))?.contiguous()?.reshape((batch, seq_len, ()))?;
        self.input_proj_linear.forward(&x)
    }
}

/// Sinusoidal Relative Positional Encoding for Audio Conformer
pub struct AudioRelPositionalEncoding {
    pub context_size: usize,
    pub hidden_size: usize,
    pub inv_timescales: Vec<f32>,
}

impl AudioRelPositionalEncoding {
    pub fn new(config: &AudioConfig) -> Self {
        let context_size = config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right;
        let num_timescales = config.hidden_size / 2;
        let min_timescale = 1.0f64;
        let max_timescale = 10000.0f64;
        let log_increment = (max_timescale / min_timescale).ln() / ((num_timescales - 1).max(1) as f64);

        let mut inv_timescales = Vec::with_capacity(num_timescales);
        for i in 0..num_timescales {
            let val = min_timescale * (-(i as f64) * log_increment).exp();
            inv_timescales.push(val as f32);
        }

        Self {
            context_size,
            hidden_size: config.hidden_size,
            inv_timescales,
        }
    }

    pub fn get_embeddings(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let max_pos = (self.context_size / 2) as i32;
        let num_pos = (max_pos + 1) as usize;
        let num_timescales = self.inv_timescales.len();

        let mut pos_vec = vec![0.0f32; num_pos * self.hidden_size];

        for (i, p) in (0..=max_pos).rev().enumerate() {
            let p_f32 = p as f32;
            let offset = i * self.hidden_size;

            for t in 0..num_timescales {
                let scaled = p_f32 * self.inv_timescales[t];
                pos_vec[offset + t] = scaled.sin();
                pos_vec[offset + num_timescales + t] = scaled.cos();
            }
        }

        Tensor::from_vec(pos_vec, (num_pos, self.hidden_size), device)?.to_dtype(dtype)
    }
}

/// Conformer Attention with Local Chunking, Softcap, and Pre-computed per_dim_scale
pub struct AudioAttention {
    pub q_proj: ClippableLinearOp,
    pub k_proj: ClippableLinearOp,
    pub v_proj: ClippableLinearOp,
    pub post: ClippableLinearOp,
    pub precomputed_rel_k: Tensor, // [1, num_heads, head_dim, 13] in F32
    pub per_dim_scale: Tensor,     // [head_dim], pre-computed softplus in F32
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_past_horizon: usize,
    pub softcap: f64,
    pub q_scale: f64,
    pub k_scale: f64,
}

impl AudioAttention {
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        mask_tensor: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden_size) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?.to_dtype(DType::F32)?;
        let k = self.k_proj.forward(hidden_states)?.to_dtype(DType::F32)?;
        let v = self.v_proj.forward(hidden_states)?.to_dtype(DType::F32)?;

        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_heads, self.head_dim))?;

        // Apply pre-computed per_dim_scale & q_scale / k_scale
        let q_scaled = (q.broadcast_mul(&self.per_dim_scale)? * self.q_scale)?;
        let k_scaled = (k * self.k_scale)?;

        let chunk_size = 12;
        let num_blocks = seq_len.div_ceil(chunk_size);
        let pad_amount = num_blocks * chunk_size - seq_len;

        // 1. Blocked queries: [batch, num_heads, num_blocks, chunk_size, head_dim]
        let q_padded = if pad_amount > 0 {
            q_scaled.pad_with_zeros(1, 0, pad_amount)?
        } else {
            q_scaled
        };
        let q_blocks = q_padded
            .reshape((batch, num_blocks, chunk_size, self.num_heads, self.head_dim))?
            .permute((0, 3, 1, 2, 4))?
            .contiguous()?;

        // 2. Context keys and values: pad 12 on left, 11 on right along sequence dimension
        let k_padded = k_scaled.pad_with_zeros(1, 12, 11)?;
        let v_padded = v.pad_with_zeros(1, 12, 11)?;

        let mut k_blocks_vec = Vec::with_capacity(num_blocks);
        let mut v_blocks_vec = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            k_blocks_vec.push(k_padded.narrow(1, b * 12, 24)?);
            v_blocks_vec.push(v_padded.narrow(1, b * 12, 24)?);
        }
        let k_ctx = Tensor::stack(&k_blocks_vec, 1)?; // [batch, num_blocks, 24, num_heads, head_dim]
        let v_ctx = Tensor::stack(&v_blocks_vec, 1)?; // [batch, num_blocks, 24, num_heads, head_dim]

        // 3. matrix_ac
        let k_perm = k_ctx.permute((0, 3, 1, 4, 2))?.contiguous()?;
        let b_h_n = batch * self.num_heads * num_blocks;
        let q_flat_ac = q_blocks.reshape((b_h_n, chunk_size, self.head_dim))?;
        let k_flat_ac = k_perm.reshape((b_h_n, self.head_dim, 24))?;
        let matrix_ac = q_flat_ac.matmul(&k_flat_ac)?.reshape((batch, self.num_heads, num_blocks, chunk_size, 24))?;

        // 4. matrix_bd (relative position bias using precomputed static rel_k)
        let q_flat = q_blocks.reshape((batch, self.num_heads, num_blocks * chunk_size, self.head_dim))?.contiguous()?;
        let bd_flat = q_flat.matmul(&self.precomputed_rel_k)?; // [batch, num_heads, num_blocks * 12, 13]
        let bd = bd_flat.reshape((batch, self.num_heads, num_blocks, chunk_size, 13))?;

        // 5. _rel_shift: pad right by 12, flatten last two dims, slice to 12 * 24, reshape to (12, 24)
        let bd_padded = bd.pad_with_zeros(4, 0, 12)?; // [batch, num_heads, num_blocks, 12, 25]
        let bd_flat2 = bd_padded.reshape((batch, self.num_heads, num_blocks, chunk_size * 25))?;
        let matrix_bd = bd_flat2.narrow(3, 0, chunk_size * 24)?.reshape((batch, self.num_heads, num_blocks, chunk_size, 24))?;

        // 6. Combine and softcap
        let mut attn_weights = (matrix_ac + matrix_bd)?;
        if self.softcap > 0.0 {
            let sc = self.softcap;
            attn_weights = ((attn_weights / sc)?.tanh()? * sc)?;
        }

        // 7. Causal / sliding window attention mask (cached across layers)
        attn_weights = attn_weights.broadcast_add(mask_tensor)?;

        // 8. Softmax
        let attn_probs = candle_nn::ops::softmax(&attn_weights, D::Minus1)?.contiguous()?;

        // 9. Multiply values
        let v_perm = v_ctx.permute((0, 3, 1, 2, 4))?.contiguous()?; // [batch, num_heads, num_blocks, 24, head_dim]
        let probs_flat = attn_probs.reshape((b_h_n, chunk_size, 24))?;
        let v_flat = v_perm.reshape((b_h_n, 24, self.head_dim))?;
        let attn_out = probs_flat.matmul(&v_flat)?.reshape((batch, self.num_heads, num_blocks, chunk_size, self.head_dim))?;

        // 10. Permute back, flatten blocks, slice [:seq_len], project out
        let attn_out = attn_out
            .permute((0, 2, 3, 1, 4))? // [batch, num_blocks, 12, num_heads, head_dim]
            .contiguous()?
            .reshape((batch, num_blocks * chunk_size, self.num_heads * self.head_dim))?
            .narrow(1, 0, seq_len)?
            .to_dtype(hidden_states.dtype())?;

        self.post.forward(&attn_out)
    }
}

/// Conformer Local 1D Convolution block (Pointwise -> GLU -> Causal Depthwise -> SiLU -> Pointwise)
pub struct AudioLightConv1d {
    pub pre_layer_norm: RMSNorm, // GGUF conv_norm.weight
    pub linear_start: ClippableLinearOp,  // conv_pw1
    pub depthwise_conv_weight: Tensor, // [1024, 1, 5]
    pub conv_norm: RMSNorm,      // GGUF norm_conv.weight
    pub linear_end: ClippableLinearOp,    // conv_pw2
    pub hidden_size: usize,
}

impl AudioLightConv1d {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        let normed = self.pre_layer_norm.forward(x)?;
        let pw1 = self.linear_start.forward(&normed)?;

        #[cfg(feature = "metal")]
        let dw_out = if pw1.device().is_metal() && pw1.dtype() == DType::BF16 {
            crate::metal_kernels::fused_metal_audio_glu_causal_dwconv5(
                &pw1,
                &self.depthwise_conv_weight,
            )?
        } else {
            let half0 = pw1.narrow(D::Minus1, 0, self.hidden_size)?;
            let half1 = pw1.narrow(D::Minus1, self.hidden_size, self.hidden_size)?;
            let sig = candle_nn::ops::sigmoid(&half1)?;
            let glu_out = (half0 * sig)?;

            let dw_in = glu_out.transpose(1, 2)?.contiguous()?;
            let dw_in = dw_in.pad_with_zeros(D::Minus1, 4, 0)?;
            let seq_len = glu_out.dim(1)?;

            let mut dw_out = dw_in
                .narrow(2, 0, seq_len)?
                .broadcast_mul(&self.depthwise_conv_weight.narrow(2, 0, 1)?)?;
            for k in 1..5 {
                let term = dw_in
                    .narrow(2, k, seq_len)?
                    .broadcast_mul(&self.depthwise_conv_weight.narrow(2, k, 1)?)?;
                dw_out = (dw_out + term)?;
            }
            dw_out.transpose(1, 2)?.contiguous()?
        };

        #[cfg(not(feature = "metal"))]
        let dw_out = {
            let half0 = pw1.narrow(D::Minus1, 0, self.hidden_size)?;
            let half1 = pw1.narrow(D::Minus1, self.hidden_size, self.hidden_size)?;
            let sig = candle_nn::ops::sigmoid(&half1)?;
            let glu_out = (half0 * sig)?;

            let dw_in = glu_out.transpose(1, 2)?.contiguous()?;
            let dw_in = dw_in.pad_with_zeros(D::Minus1, 4, 0)?;
            let seq_len = glu_out.dim(1)?;

            let mut dw_out = dw_in
                .narrow(2, 0, seq_len)?
                .broadcast_mul(&self.depthwise_conv_weight.narrow(2, 0, 1)?)?;
            for k in 1..5 {
                let term = dw_in
                    .narrow(2, k, seq_len)?
                    .broadcast_mul(&self.depthwise_conv_weight.narrow(2, k, 1)?)?;
                dw_out = (dw_out + term)?;
            }
            dw_out.transpose(1, 2)?.contiguous()?
        };

        let conv_normed = self.conv_norm.forward(&dw_out)?;
        let silu_out = candle_nn::ops::silu(&conv_normed)?;
        let pw2 = self.linear_end.forward(&silu_out)?;

        residual + pw2
    }
}

/// Conformer Macaron-style Feed-Forward Network
pub struct AudioFeedForward {
    pub pre_layer_norm: RMSNorm,
    pub ffw_layer_1: ClippableLinearOp, // [4096, 1024]
    pub ffw_layer_2: ClippableLinearOp, // [1024, 4096]
    pub post_layer_norm: RMSNorm,
    pub residual_weight: f64, // 0.5
}

impl AudioFeedForward {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        let normed = self.pre_layer_norm.forward(x)?;
        let up = self.ffw_layer_1.forward(&normed)?;
        let act = candle_nn::ops::silu(&up)?;
        let down = self.ffw_layer_2.forward(&act)?;

        let post_normed = self.post_layer_norm.forward(&down)?;
        let scaled = (post_normed * self.residual_weight)?;
        residual + scaled
    }
}

/// 12 Conformer Encoder Layers
pub struct AudioConformerLayer {
    pub feed_forward1: AudioFeedForward,
    pub self_attn: AudioAttention,
    pub lconv1d: AudioLightConv1d,
    pub feed_forward2: AudioFeedForward,
    pub norm_pre_attn: RMSNorm,
    pub norm_post_attn: RMSNorm,
    pub norm_out: RMSNorm,
}

impl AudioConformerLayer {
    pub fn forward(&self, x: &Tensor, mask_tensor: &Tensor) -> Result<Tensor> {
        // Macaron FFN 1
        let x = self.feed_forward1.forward(x)?;

        // Self-Attention + Fused Post-Norm Residual Add
        let pre_attn = self.norm_pre_attn.forward(&x)?;
        let attn_out = self.self_attn.forward(&pre_attn, mask_tensor)?;
        let x = self.norm_post_attn.forward_residual_add(&attn_out, &x, None)?;

        // Light Conv 1D
        let x = self.lconv1d.forward(&x)?;

        // Macaron FFN 2
        let x = self.feed_forward2.forward(&x)?;

        // Output norm
        self.norm_out.forward(&x)
    }
}

/// Complete Audio Tower Subsystem
pub struct AudioTower {
    pub subsampler: AudioSubsampleConv,
    pub rel_pos_enc: AudioRelPositionalEncoding,
    pub layers: Vec<AudioConformerLayer>,
    pub output_proj: LinearOp, // a.pre_encode.out [1536, 1024]
    pub output_bias: Option<Tensor>, // [1536]
    pub projector_norm: RMSNorm, // unscaled RMSNorm 1536
    pub projector: LinearOp, // mm.a.input_projection.weight [512, 1536]
    pub config: AudioConfig,
    pub device: Device,
    pub dtype: DType,
}

impl AudioTower {
    /// Encode a processed audio into soft tokens [num_soft_tokens, 512]
    pub fn encode(&self, processed: &ProcessedAudio) -> Result<Tensor> {
        let num_frames = processed.num_frames;
        let num_mels = processed.num_mels;
        let input_features = Tensor::from_slice(
            &processed.features,
            (1, num_frames, num_mels),
            &self.device,
        )?
        .to_dtype(self.dtype)?;

        // 1. Subsampling convolution down to [1, num_frames / 4, 1024]
        let mut hidden_states = self.subsampler.forward(&input_features)?;

        // 2. Precompute causal sliding window attention mask ONCE for all 12 Conformer layers
        let seq_len = hidden_states.dim(1)?;
        let chunk_size = 12;
        let num_blocks = seq_len.div_ceil(chunk_size);

        let mut mask_vec = vec![-1e9f32; num_blocks * chunk_size * 24];
        for b in 0..num_blocks {
            for q_idx in 0..chunk_size {
                let t_q = b * chunk_size + q_idx;
                if t_q < seq_len {
                    for c in 0..24 {
                        let t_k = (b as i32) * 12 - 12 + (c as i32);
                        if t_k >= 0 && (t_k as usize) < seq_len && (t_k as usize) <= t_q && (t_q - (t_k as usize)) <= 11 {
                            mask_vec[b * chunk_size * 24 + q_idx * 24 + c] = 0.0f32;
                        }
                    }
                }
            }
        }
        let mask_tensor = Tensor::from_vec(
            mask_vec,
            (1, 1, num_blocks, chunk_size, 24),
            &self.device,
        )?;

        // 3. 12 Conformer layers with cached mask
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &mask_tensor)?;
        }

        // 4. Output projection to 1536
        let mut out = self.output_proj.forward(&hidden_states)?;
        if let Some(ref bias) = self.output_bias {
            out = out.broadcast_add(bias)?;
        }

        // 5. Project into 512-dim embedding space
        let normed = self.projector_norm.forward(&out)?;
        let soft_tokens = self.projector.forward(&normed)?;
        soft_tokens.squeeze(0) // [num_soft_tokens, 512]
    }

    /// Load Audio Tower directly from GGUF file
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        content: &candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        config: AudioConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // 1. Subsampler
        let c0_w = content.tensor(reader, "a.conv1d.0.weight", device)?.dequantize(device)?.to_dtype(dtype)?;
        let c0_norm = load_gguf_norm_tensor(content, reader, "a.conv1d.0.norm.weight", device, dtype)?;
        let layer0 = AudioSubsampleLayer {
            conv_weight: c0_w,
            norm_weight: c0_norm,
            out_channels: 128,
            eps: config.rms_norm_eps,
        };

        let c1_w = content.tensor(reader, "a.conv1d.1.weight", device)?.dequantize(device)?.to_dtype(dtype)?;
        let c1_norm = load_gguf_norm_tensor(content, reader, "a.conv1d.1.norm.weight", device, dtype)?;
        let layer1 = AudioSubsampleLayer {
            conv_weight: c1_w,
            norm_weight: c1_norm,
            out_channels: 32,
            eps: config.rms_norm_eps,
        };

        let input_proj_linear = load_gguf_linear_op(content, reader, "a.input_projection.weight", device, dtype)?;
        let subsampler = AudioSubsampleConv {
            layer0,
            layer1,
            input_proj_linear,
        };

        // 2. Relative position encoding
        let rel_pos_enc = AudioRelPositionalEncoding::new(&config);
        let rel_pos_emb = rel_pos_enc.get_embeddings(device, dtype)?;

        // 3. 12 Conformer layers
        let head_dim = config.hidden_size / config.num_attention_heads;
        let q_scale = (head_dim as f64).powf(-0.5) / 2.0f64.ln();
        let k_scale = (1.0 + std::f64::consts::E).ln() / 2.0f64.ln();

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let prefix = format!("a.blk.{}", l);

            // FeedForward 1
            let ffn1_pre_w = load_gguf_norm_tensor(content, reader, &format!("{}.ffn_norm.weight", prefix), device, dtype)?;
            let ffn1_post_w = load_gguf_norm_tensor(content, reader, &format!("{}.ffn_post_norm.weight", prefix), device, dtype)?;
            let ffn1_up = load_gguf_clippable_linear_op(content, reader, &format!("{}.ffn_up", prefix), device, dtype)?;
            let ffn1_down = load_gguf_clippable_linear_op(content, reader, &format!("{}.ffn_down", prefix), device, dtype)?;
            let feed_forward1 = AudioFeedForward {
                pre_layer_norm: RMSNorm::new(Some(ffn1_pre_w), config.rms_norm_eps),
                ffw_layer_1: ffn1_up,
                ffw_layer_2: ffn1_down,
                post_layer_norm: RMSNorm::new(Some(ffn1_post_w), config.rms_norm_eps),
                residual_weight: 0.5,
            };

            // Attention
            let q_proj = load_gguf_clippable_linear_op(content, reader, &format!("{}.attn_q", prefix), device, dtype)?;
            let k_proj = load_gguf_clippable_linear_op(content, reader, &format!("{}.attn_k", prefix), device, dtype)?;
            let v_proj = load_gguf_clippable_linear_op(content, reader, &format!("{}.attn_v", prefix), device, dtype)?;
            let post = load_gguf_clippable_linear_op(content, reader, &format!("{}.attn_out", prefix), device, dtype)?;
            let relative_k_proj = load_gguf_linear_op(content, reader, &format!("{}.attn_k_rel.weight", prefix), device, dtype)?;
            let per_dim_scale = load_gguf_norm_tensor(content, reader, &format!("{}.per_dim_scale.weight", prefix), device, dtype)?;

            // Precompute rel_k and per_dim_scale in F32
            let rel_k = relative_k_proj.forward(&rel_pos_emb)?.to_dtype(DType::F32)?;
            let precomputed_rel_k = rel_k
                .reshape((13, config.num_attention_heads, head_dim))?
                .permute((1, 2, 0))?
                .unsqueeze(0)?
                .contiguous()?;
            let per_dim_scale_f32 = per_dim_scale.to_dtype(DType::F32)?;

            let self_attn = AudioAttention {
                q_proj,
                k_proj,
                v_proj,
                post,
                precomputed_rel_k,
                per_dim_scale: per_dim_scale_f32,
                num_heads: config.num_attention_heads,
                head_dim,
                max_past_horizon: config.attention_context_left.saturating_sub(1),
                softcap: 50.0,
                q_scale,
                k_scale,
            };

            // Norms around attention
            let norm_pre_attn_w = load_gguf_norm_tensor(content, reader, &format!("{}.attn_pre_norm.weight", prefix), device, dtype)?;
            let norm_post_attn_w = load_gguf_norm_tensor(content, reader, &format!("{}.attn_post_norm.weight", prefix), device, dtype)?;

            // Light Conv 1D
            // Remember GGUF norm naming swap: conv_norm is pre-layer-norm, norm_conv is conv-norm
            let dw_pre_ln_w = load_gguf_norm_tensor(content, reader, &format!("{}.conv_norm.weight", prefix), device, dtype)?;
            let dw_pw1 = load_gguf_clippable_linear_op(content, reader, &format!("{}.conv_pw1", prefix), device, dtype)?;
            let dw_w = content.tensor(reader, &format!("{}.conv_dw.weight", prefix), device)?
                .dequantize(device)?.to_dtype(dtype)?.unsqueeze(0)?; // [1, 1024, 5]
            let dw_conv_norm_w = load_gguf_norm_tensor(content, reader, &format!("{}.norm_conv.weight", prefix), device, dtype)?;
            let dw_pw2 = load_gguf_clippable_linear_op(content, reader, &format!("{}.conv_pw2", prefix), device, dtype)?;

            let lconv1d = AudioLightConv1d {
                pre_layer_norm: RMSNorm::new(Some(dw_pre_ln_w), config.rms_norm_eps),
                linear_start: dw_pw1,
                depthwise_conv_weight: dw_w,
                conv_norm: RMSNorm::new(Some(dw_conv_norm_w), config.rms_norm_eps),
                linear_end: dw_pw2,
                hidden_size: config.hidden_size,
            };

            // FeedForward 2
            let ffn2_pre_w = load_gguf_norm_tensor(content, reader, &format!("{}.ffn_norm_1.weight", prefix), device, dtype)?;
            let ffn2_post_w = load_gguf_norm_tensor(content, reader, &format!("{}.ffn_post_norm_1.weight", prefix), device, dtype)?;
            let ffn2_up = load_gguf_clippable_linear_op(content, reader, &format!("{}.ffn_up_1", prefix), device, dtype)?;
            let ffn2_down = load_gguf_clippable_linear_op(content, reader, &format!("{}.ffn_down_1", prefix), device, dtype)?;
            let feed_forward2 = AudioFeedForward {
                pre_layer_norm: RMSNorm::new(Some(ffn2_pre_w), config.rms_norm_eps),
                ffw_layer_1: ffn2_up,
                ffw_layer_2: ffn2_down,
                post_layer_norm: RMSNorm::new(Some(ffn2_post_w), config.rms_norm_eps),
                residual_weight: 0.5,
            };

            let norm_out_w = load_gguf_norm_tensor(content, reader, &format!("{}.ln2.weight", prefix), device, dtype)?;

            layers.push(AudioConformerLayer {
                feed_forward1,
                self_attn,
                lconv1d,
                feed_forward2,
                norm_pre_attn: RMSNorm::new(Some(norm_pre_attn_w), config.rms_norm_eps),
                norm_post_attn: RMSNorm::new(Some(norm_post_attn_w), config.rms_norm_eps),
                norm_out: RMSNorm::new(Some(norm_out_w), config.rms_norm_eps),
            });
        }

        // 4. Output projection
        let output_proj = load_gguf_linear_op(content, reader, "a.pre_encode.out.weight", device, dtype)?;
        let output_bias = if content.tensor_infos.contains_key("a.pre_encode.out.bias") {
            Some(load_gguf_norm_tensor(content, reader, "a.pre_encode.out.bias", device, dtype)?)
        } else {
            None
        };

        // 5. Audio Projector (unscaled RMSNorm 1536 + Linear 1536 -> 512)
        let projector_norm = RMSNorm::new(None, config.rms_norm_eps);
        let projector = load_gguf_linear_op(content, reader, "mm.a.input_projection.weight", device, dtype)?;

        Ok(Self {
            subsampler,
            rel_pos_enc,
            layers,
            output_proj,
            output_bias,
            projector_norm,
            projector,
            config,
            device: device.clone(),
            dtype,
        })
    }
}
