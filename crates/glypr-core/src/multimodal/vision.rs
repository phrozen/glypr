use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Embedding, Linear};

use crate::config::VisionConfig;
use crate::linear::LinearOp;
use crate::mlp::MLP;
use crate::multimodal::image_preprocessor::ProcessedImage;
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
    let qt = content.tensor(reader, name, device)?;
    let info = content.tensor_infos.get(name).ok_or_else(|| {
        candle_core::Error::Msg(format!("Missing tensor info for {}", name))
    })?;
    match info.ggml_dtype {
        candle_core::quantized::GgmlDType::F32
        | candle_core::quantized::GgmlDType::F16
        | candle_core::quantized::GgmlDType::BF16 => {
            let t = qt.dequantize(device)?.to_dtype(dtype)?;
            Ok(LinearOp::new_standard(Linear::new(t, None)))
        }
        _ => {
            let qm = candle_core::quantized::QMatMul::from_qtensor(qt)?;
            Ok(LinearOp::new_quantized(qm))
        }
    }
}

/// 2D Patch Embedder: Linear projection + 2D table coordinate embedding lookup
pub struct VisionPatchEmbedder {
    pub input_proj: LinearOp,
    pub position_embedding_table: Tensor,
    pub hidden_size: usize,
}

impl VisionPatchEmbedder {
    pub fn forward(&self, pixel_values: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // 1. Project patches: [1, num_patches, 768]
        let hidden_states = self.input_proj.forward(pixel_values)?;

        // 2. Position lookup: position_ids is [num_patches, 2] (px, py)
        let x_pos = position_ids.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;
        let y_pos = position_ids.narrow(1, 1, 1)?.squeeze(1)?.contiguous()?;

        let t0 = self.position_embedding_table.get(0)?; // [10240, 768]
        let t1 = self.position_embedding_table.get(1)?; // [10240, 768]

        let x_emb = Embedding::new(t0, self.hidden_size).forward(&x_pos)?;
        let y_emb = Embedding::new(t1, self.hidden_size).forward(&y_pos)?;

        let pos_emb = (x_emb + y_emb)?.unsqueeze(0)?; // [1, num_patches, 768]
        hidden_states.broadcast_add(&pos_emb)
    }
}

/// 2D Axial Rotary Position Embedding (RoPE) for Vision Transformer
#[derive(Clone)]
pub struct VisionRotaryEmbedding {
    pub head_dim: usize,
    pub rope_theta: f64,
}

impl VisionRotaryEmbedding {
    pub fn new(head_dim: usize, rope_theta: f64) -> Self {
        Self { head_dim, rope_theta }
    }

    /// Precompute (cos, sin) tensors once for all patches: [1, 1, num_patches, head_dim]
    pub fn precompute(
        &self,
        position_ids: &Tensor,
        num_patches: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let spatial_dim = self.head_dim / 2;
        let num_freqs = spatial_dim / 2;

        let mut inv_freq = Vec::with_capacity(num_freqs);
        for i in 0..num_freqs {
            let exponent = (2 * i) as f64 / (spatial_dim as f64);
            inv_freq.push(1.0f32 / (self.rope_theta.powf(exponent) as f32));
        }

        let pos_cpu = position_ids.to_device(&Device::Cpu)?.to_dtype(DType::U32)?;
        let pos_data: Vec<u32> = pos_cpu.flatten_all()?.to_vec1()?;

        let mut cos_vec = vec![0.0f32; num_patches * self.head_dim];
        let mut sin_vec = vec![0.0f32; num_patches * self.head_dim];

        for p in 0..num_patches {
            let px = pos_data[p * 2] as f32;
            let py = pos_data[p * 2 + 1] as f32;
            let offset = p * self.head_dim;

            for i in 0..num_freqs {
                let theta_x = px * inv_freq[i];
                let (sx, cx) = (theta_x.sin(), theta_x.cos());
                cos_vec[offset + i] = cx;
                cos_vec[offset + i + num_freqs] = cx;
                sin_vec[offset + i] = sx;
                sin_vec[offset + i + num_freqs] = sx;

                let theta_y = py * inv_freq[i];
                let (sy, cy) = (theta_y.sin(), theta_y.cos());
                cos_vec[offset + spatial_dim + i] = cy;
                cos_vec[offset + spatial_dim + i + num_freqs] = cy;
                sin_vec[offset + spatial_dim + i] = sy;
                sin_vec[offset + spatial_dim + i + num_freqs] = sy;
            }
        }

        let cos = Tensor::from_vec(cos_vec, (1, 1, num_patches, self.head_dim), device)?.to_dtype(dtype)?;
        let sin = Tensor::from_vec(sin_vec, (1, 1, num_patches, self.head_dim), device)?.to_dtype(dtype)?;
        Ok((cos, sin))
    }

    /// Fast GPU/CPU rotation using precomputed (cos, sin) tensors
    pub fn apply_precomputed(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x_h0 = x.narrow(D::Minus1, 0, 16)?.contiguous()?;
        let x_h1 = x.narrow(D::Minus1, 16, 16)?.contiguous()?;
        let rot_h = Tensor::cat(&[&x_h1.neg()?, &x_h0], D::Minus1)?;

        let x_w0 = x.narrow(D::Minus1, 32, 16)?.contiguous()?;
        let x_w1 = x.narrow(D::Minus1, 48, 16)?.contiguous()?;
        let rot_w = Tensor::cat(&[&x_w1.neg()?, &x_w0], D::Minus1)?;
        let rot_x = Tensor::cat(&[&rot_h, &rot_w], D::Minus1)?;

        x.broadcast_mul(cos)? + rot_x.broadcast_mul(sin)?
    }

    /// Legacy on-the-fly RoPE calculation (kept for standalone compatibility)
    pub fn apply_rope(
        &self,
        x: &Tensor,
        position_ids: &Tensor,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let (_batch, _heads, num_patches, _head_dim) = x.dims4()?;
        let (cos, sin) = self.precompute(position_ids, num_patches, dtype, device)?;
        self.apply_precomputed(x, &cos, &sin)
    }
}

/// Multi-head Self-Attention with Q/K RMSNorm and 2D Axial RoPE
pub struct VisionAttention {
    pub q_proj: LinearOp,
    pub k_proj: LinearOp,
    pub v_proj: LinearOp,
    pub o_proj: LinearOp,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub v_norm: RMSNorm,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl VisionAttention {
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &VisionRotaryEmbedding,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden_size) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_heads, self.head_dim))?;

        // Apply Q, K, V RMSNorm
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let v = self.v_norm.forward(&v)?;

        // Permute to [batch, num_heads, seq_len, head_dim]
        let q = q.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;

        // Apply 2D Axial RoPE to Q and K with precomputed tensors
        let q = rope.apply_precomputed(&q, cos, sin)?;
        let k = rope.apply_precomputed(&k, cos, sin)?;

        // Non-causal Bidirectional Attention with scaling = 1.0
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn_weights = q.matmul(&k_t)?; // [batch, num_heads, seq_len, seq_len]

        // Softmax in Float32
        let attn_weights_f32 = attn_weights.to_dtype(DType::F32)?;
        let attn_probs_f32 = candle_nn::ops::softmax(&attn_weights_f32, D::Minus1)?;
        let dtype = hidden_states.dtype();
        let attn_probs = attn_probs_f32.to_dtype(dtype)?;

        // [batch, num_heads, seq_len, head_dim]
        let attn_output = attn_probs.matmul(&v)?;

        // Reshape back to [batch, seq_len, hidden_size]
        let attn_output = attn_output.permute((0, 2, 1, 3))?.reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }
}

/// Vision Transformer Encoder Layer (16 layers total)
pub struct VisionEncoderLayer {
    pub ln1: RMSNorm,
    pub self_attn: VisionAttention,
    pub attn_post_norm: RMSNorm,
    pub ln2: RMSNorm,
    pub mlp: MLP,
    pub ffn_post_norm: RMSNorm,
}

impl VisionEncoderLayer {
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &VisionRotaryEmbedding,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        // Pre-attention layernorm
        let normed = self.ln1.forward(hidden_states)?;
        let attn_out = self.self_attn.forward(&normed, rope, cos, sin)?;
        let attn_out = self.attn_post_norm.forward(&attn_out)?;
        let hidden_states = (hidden_states + attn_out)?;

        // Pre-feedforward layernorm
        let normed = self.ln2.forward(&hidden_states)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let mlp_out = self.ffn_post_norm.forward(&mlp_out)?;
        hidden_states + mlp_out
    }
}

/// 2D Spatial Grid Average Pooler (3x3 grid averaging + sqrt(hidden_size) scaling)
pub struct VisionPooler {
    pub pooling_kernel_size: usize,
    pub root_hidden_size: f64,
}

impl VisionPooler {
    pub fn new(pooling_kernel_size: usize, hidden_size: usize) -> Self {
        Self {
            pooling_kernel_size,
            root_hidden_size: (hidden_size as f64).sqrt(),
        }
    }

    /// Spatial 2D average pooling over 3x3 patch blocks
    pub fn pool(
        &self,
        hidden_states: &Tensor, // [1, num_patches, 768]
        patch_width: usize,
        patch_height: usize,
    ) -> Result<Tensor> {
        let k = self.pooling_kernel_size;
        let out_w = patch_width / k;
        let out_h = patch_height / k;
        let hidden_size = hidden_states.dim(D::Minus1)?;

        let orig_dtype = hidden_states.dtype();
        // Stage 1: Height pooling (Rank 3: [out_h, k, patch_width * hidden_size] -> [out_h, patch_width * hidden_size])
        // Candle Metal kernels natively support rank <= 4 with exact hardware striding.
        let h_blocks = hidden_states
            .to_dtype(DType::F32)?
            .reshape((out_h, k, patch_width * hidden_size))?;
        let h_sum = h_blocks.sum(1)?;

        // Stage 2: Width pooling (Rank 4: [out_h, out_w, k, hidden_size] -> [out_h, out_w, hidden_size])
        let w_blocks = h_sum.reshape((out_h, out_w, k, hidden_size))?;
        let hw_sum = w_blocks.sum(2)?;

        // Scale by root_hidden_size / (k * k) and reshape to [1, out_tokens, hidden_size]
        let scale = self.root_hidden_size / ((k * k) as f64);
        let scaled = (&hw_sum * scale)?;

        scaled
            .reshape((1, out_h * out_w, hidden_size))?
            .to_dtype(orig_dtype)
    }
}

/// Vision Projector: Unscaled RMSNorm + Linear Projection (768 -> 512)
pub struct VisionProjector {
    pub norm: RMSNorm,
    pub projection: LinearOp,
}

impl VisionProjector {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(x)?;
        self.projection.forward(&normed)
    }
}

/// Complete Vision Tower Subsystem
pub struct VisionTower {
    pub patch_embedder: VisionPatchEmbedder,
    pub rotary_emb: VisionRotaryEmbedding,
    pub layers: Vec<VisionEncoderLayer>,
    pub pooler: VisionPooler,
    pub projector: VisionProjector,
    pub config: VisionConfig,
    pub device: Device,
    pub dtype: DType,
}

impl VisionTower {
    /// Encode a processed image into soft tokens [num_soft_tokens, 512]
    pub fn encode(&self, processed: &ProcessedImage) -> Result<Tensor> {
        let num_patches = processed.num_patches;
        let patch_pixels = 768;

        let pixel_tensor = Tensor::from_vec(
            processed.pixel_values.clone(),
            (1, num_patches, patch_pixels),
            &self.device,
        )?
        .to_dtype(self.dtype)?;

        let pos_tensor = Tensor::from_vec(
            processed.position_ids.clone(),
            (num_patches, 2),
            &self.device,
        )?
        .to_dtype(DType::U32)?;

        // 1. Patch projection + 2D position embeddings
        let mut hidden_states = self.patch_embedder.forward(&pixel_tensor, &pos_tensor)?;

        // Precompute 2D Axial RoPE (cos, sin) once across all 16 layers
        let (cos, sin) = self.rotary_emb.precompute(&pos_tensor, num_patches, self.dtype, &self.device)?;

        // 2. 16 Transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &self.rotary_emb, &cos, &sin)?;
        }

        // 3. 2D Spatial pooling
        let pooled = self.pooler.pool(
            &hidden_states,
            processed.patch_width,
            processed.patch_height,
        )?;

        // 4. Project into 512-dim embedding space
        let soft_tokens = self.projector.forward(&pooled)?;
        soft_tokens.squeeze(0) // [num_soft_tokens, 512]
    }

    /// Load Vision Tower directly from GGUF file
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        content: &candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        config: VisionConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // 1. Patch Embedder: permute GGUF [768, 3, 16, 16] -> [768, 16, 16, 3] -> [768, 768]
        let patch_raw = content.tensor(reader, "v.patch_embd.weight", device)?.dequantize(device)?;
        let patch_linear_w = patch_raw.permute((0, 2, 3, 1))?.reshape((768, 768))?.to_dtype(dtype)?;
        let input_proj = LinearOp::new_standard(Linear::new(patch_linear_w, None));

        let pos_table = content.tensor(reader, "v.position_embd.weight", device)?.dequantize(device)?.to_dtype(dtype)?;
        let patch_embedder = VisionPatchEmbedder {
            input_proj,
            position_embedding_table: pos_table,
            hidden_size: config.hidden_size,
        };

        // 2. Rotary embedding
        let rotary_emb = VisionRotaryEmbedding::new(config.head_dim, config.rope_theta);

        // 3. 16 Transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let prefix = format!("v.blk.{}", l);

            let ln1_w = load_gguf_norm_tensor(content, reader, &format!("{}.ln1.weight", prefix), device, dtype)?;
            let ln1 = RMSNorm::new(Some(ln1_w), config.rms_norm_eps);

            let q_proj = load_gguf_linear_op(content, reader, &format!("{}.attn_q.weight", prefix), device, dtype)?;
            let k_proj = load_gguf_linear_op(content, reader, &format!("{}.attn_k.weight", prefix), device, dtype)?;
            let v_proj = load_gguf_linear_op(content, reader, &format!("{}.attn_v.weight", prefix), device, dtype)?;
            let o_proj = load_gguf_linear_op(content, reader, &format!("{}.attn_out.weight", prefix), device, dtype)?;

            let q_norm_w = load_gguf_norm_tensor(content, reader, &format!("{}.attn_q_norm.weight", prefix), device, dtype)?;
            let k_norm_w = load_gguf_norm_tensor(content, reader, &format!("{}.attn_k_norm.weight", prefix), device, dtype)?;

            let q_norm = RMSNorm::new(Some(q_norm_w), config.rms_norm_eps);
            let k_norm = RMSNorm::new(Some(k_norm_w), config.rms_norm_eps);
            let v_norm = RMSNorm::new(None, config.rms_norm_eps); // unscaled

            let self_attn = VisionAttention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                v_norm,
                num_heads: config.num_attention_heads,
                head_dim: config.head_dim,
            };

            let attn_post_norm_w = load_gguf_norm_tensor(content, reader, &format!("{}.attn_post_norm.weight", prefix), device, dtype)?;
            let attn_post_norm = RMSNorm::new(Some(attn_post_norm_w), config.rms_norm_eps);

            let ln2_w = load_gguf_norm_tensor(content, reader, &format!("{}.ln2.weight", prefix), device, dtype)?;
            let ln2 = RMSNorm::new(Some(ln2_w), config.rms_norm_eps);

            let gate_proj = load_gguf_linear_op(content, reader, &format!("{}.ffn_gate.weight", prefix), device, dtype)?;
            let up_proj = load_gguf_linear_op(content, reader, &format!("{}.ffn_up.weight", prefix), device, dtype)?;
            let down_proj = load_gguf_linear_op(content, reader, &format!("{}.ffn_down.weight", prefix), device, dtype)?;
            let mlp = MLP::new(gate_proj, up_proj, down_proj);

            let ffn_post_norm_w = load_gguf_norm_tensor(content, reader, &format!("{}.ffn_post_norm.weight", prefix), device, dtype)?;
            let ffn_post_norm = RMSNorm::new(Some(ffn_post_norm_w), config.rms_norm_eps);

            layers.push(VisionEncoderLayer {
                ln1,
                self_attn,
                attn_post_norm,
                ln2,
                mlp,
                ffn_post_norm,
            });
        }

        // 4. Pooler
        let pooler = VisionPooler::new(config.pooling_kernel_size, config.hidden_size);

        // 5. Projector (unscaled RMSNorm + Linear 768 -> 512)
        let proj_norm = RMSNorm::new(None, config.rms_norm_eps); // unscaled
        let proj_linear = load_gguf_linear_op(content, reader, "mm.input_projection.weight", device, dtype)?;
        let projector = VisionProjector {
            norm: proj_norm,
            projection: proj_linear,
        };

        Ok(Self {
            patch_embedder,
            rotary_emb,
            layers,
            pooler,
            projector,
            config,
            device: device.clone(),
            dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pooler_shapes() -> Result<()> {
        let pooler = VisionPooler::new(3, 768);
        let patch_width = 18;
        let patch_height = 27;
        let num_patches = patch_width * patch_height;
        let device = if candle_core::utils::metal_is_available() {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };
        let t = Tensor::randn(0.0f32, 1.0f32, (1, num_patches, 768), &device)?;
        let k = 3;
        let out_w = patch_width / k;
        let out_h = patch_height / k;
        let out_tokens = out_w * out_h;
        let hidden_size = 768;
        let hs_f32 = t.squeeze(0)?.to_dtype(DType::F32)?;
        let hs_vec: Vec<f32> = hs_f32.flatten_all()?.to_vec1()?;
        let mut old_pooled = vec![0.0f32; out_tokens * hidden_size];
        let k_squared = (k * k) as f32;
        for oy in 0..out_h {
            for ox in 0..out_w {
                let out_idx = oy * out_w + ox;
                let out_offset = out_idx * hidden_size;
                for ky in 0..k {
                    let py = oy * k + ky;
                    for kx in 0..k {
                        let px = ox * k + kx;
                        let patch_idx = py * patch_width + px;
                        let in_offset = patch_idx * hidden_size;
                        for d in 0..hidden_size {
                            old_pooled[out_offset + d] += hs_vec[in_offset + d];
                        }
                    }
                }
                let scale = (pooler.root_hidden_size as f32) / k_squared;
                for d in 0..hidden_size {
                    old_pooled[out_offset + d] *= scale;
                }
            }
        }

        let new_t = pooler.pool(&t, patch_width, patch_height)?;
        let new_vec: Vec<f32> = new_t.squeeze(0)?.flatten_all()?.to_vec1()?;

        for i in 0..old_pooled.len() {
            let diff = (old_pooled[i] - new_vec[i]).abs();
            if diff > 1e-4 {
                panic!("Mismatch at idx {}: old={}, new={}, diff={}", i, old_pooled[i], new_vec[i], diff);
            }
        }
        println!("Test passed successfully!");
        Ok(())
    }
}
