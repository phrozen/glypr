pub mod attention;
pub mod config;
pub mod layer;
pub mod linear;
pub mod mlp;
pub mod model;
pub mod norm;
pub mod ple;
pub mod profile;
pub mod rope;
pub mod tokenizer;
pub mod cpu_simd;
pub mod multimodal;
pub mod hub;
#[cfg(feature = "metal")]
pub mod metal_kernels;

pub use attention::AttentionMode;
pub use linear::LinearOp;
pub use model::EmbeddingGemma2Model;
pub use multimodal::MultimodalProjector;
pub use profile::{ForwardProfile, LayerProfile, PipelineProfile};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use config::TextConfig;
use std::path::Path;
use tokenizer::GlyprTokenizer;

pub struct GlyprEmbedder {
    pub model: EmbeddingGemma2Model,
    pub tokenizer: GlyprTokenizer,
    pub device: Device,
    #[allow(dead_code)]
    pub dtype: DType,
    pub multimodal: Option<MultimodalProjector>,
}

impl GlyprEmbedder {
    pub fn load<P: AsRef<Path>>(model_dir: P, device: &Device, dtype: DType) -> Result<Self> {
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        let weights_path = dir.join("model.safetensors");

        let config = TextConfig::from_json_file(&config_path)
            .with_context(|| format!("Failed to load config from {:?}", config_path))?;
        let tokenizer = GlyprTokenizer::from_file(&tokenizer_path)
            .with_context(|| format!("Failed to load tokenizer from {:?}", tokenizer_path))?;

        let weights_bytes = std::fs::read(&weights_path)
            .with_context(|| format!("Failed to read weights from {:?}", weights_path))?;
        let safetensors = safetensors::SafeTensors::deserialize(&weights_bytes)
            .map_err(|e| anyhow::anyhow!("Safetensors deserialization failed: {}", e))?;

        let model = EmbeddingGemma2Model::from_safetensors(&safetensors, config, device, dtype)
            .map_err(|e| anyhow::anyhow!("Model initialization failed: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
            multimodal: None,
        })
    }

    /// Load a 100% self-contained GGUF file (extracts model weights, architecture config, and BPE tokenizer).
    /// Zero external file or HuggingFace snapshot dependencies!
    pub fn load_gguf<P: AsRef<Path>>(
        gguf_path: P,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(gguf_path.as_ref())
            .with_context(|| format!("Failed to open GGUF file {:?}", gguf_path.as_ref()))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to read GGUF container from {:?}", gguf_path.as_ref()))?;

        let config = TextConfig::from_gguf_metadata(&content.metadata)
            .context("Failed to parse TextConfig from GGUF metadata")?;

        let tokenizer = GlyprTokenizer::from_gguf_metadata(&content.metadata)
            .context("Failed to parse GlyprTokenizer from GGUF metadata")?;

        let model = EmbeddingGemma2Model::from_gguf_content(&content, &mut file, config, device, dtype)
            .map_err(|e| anyhow::anyhow!("GGUF Model initialization failed: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
            multimodal: None,
        })
    }

    /// Load self-contained BF16 GGUF file (e.g., `embeddinggemma-2-BF16.gguf`)
    pub fn load_gguf_bf16<P: AsRef<Path>>(
        gguf_path: P,
        device: &Device,
    ) -> Result<Self> {
        let dtype = if device.is_metal() { DType::BF16 } else { DType::F32 };
        Self::load_gguf(gguf_path, device, dtype)
    }

    /// Load self-contained Q8_0 quantized GGUF file (e.g., `embeddinggemma-2-Q8_0.gguf`)
    pub fn load_gguf_q8<P: AsRef<Path>>(
        gguf_path: P,
        device: &Device,
    ) -> Result<Self> {
        let dtype = if device.is_metal() { DType::BF16 } else { DType::F32 };
        Self::load_gguf(gguf_path, device, dtype)
    }

    /// Dynamically attach an optional multimodal projector GGUF (`mmproj-*.gguf`)
    pub fn attach_multimodal_projector<P: AsRef<Path>>(&mut self, mmproj_path: P) -> Result<()> {
        let projector = MultimodalProjector::from_gguf_file(mmproj_path.as_ref(), &self.device, self.dtype)?;
        self.multimodal = Some(projector);
        Ok(())
    }

    /// Encode an image (with optional prompt) into a dense normalized embedding vector
    pub fn embed_image(
        &self,
        image: &image::DynamicImage,
        prompt: Option<&str>,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<f32>> {
        let projector = self.multimodal.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Multimodal projector not attached. Call embedder.attach_multimodal_projector() first.")
        })?;

        // 1. Encode image through Vision Tower
        let (soft_tokens, num_soft_tokens) = projector.encode_image(image)?;

        // 2. Tokenize prompt with special multimodal tokens: <bos> <boi> <image>*N <eoi> [prompt] <eos>
        let token_ids = self.tokenizer.encode_image_query(prompt, num_soft_tokens)?;
        let seq_len = token_ids.len();
        let input_tensor = Tensor::from_vec(token_ids, (1, seq_len), &self.device)?;

        // 3. Embed tokens with scaling
        let text_embeds = (self.model.embed_tokens.forward(&input_tensor)? * self.model.embed_scale)?;

        // 4. Splice soft tokens into <image> slots: [0..2] text, [2..2+num_soft_tokens] soft tokens, [2+num_soft_tokens..] prompt + eos
        let pre = text_embeds.narrow(1, 0, 2)?;
        let soft = soft_tokens.unsqueeze(0)?;
        let post = text_embeds.narrow(1, 2 + num_soft_tokens, seq_len - (2 + num_soft_tokens))?;
        let merged_embeds = Tensor::cat(&[&pre, &soft, &post], 1)?;

        // 5. Forward through backbone and pool
        let token_embeds = self.model.forward_embeds(&merged_embeds, None)?;
        let pooled = self.model.pool_and_normalize(&token_embeds, None, truncate_dim)?;
        let pooled_f32 = pooled.to_dtype(DType::F32)?;
        let vec = pooled_f32.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vec)
    }

    pub fn embed_audio(
        &self,
        samples: &[f32],
        prompt: Option<&str>,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<f32>> {
        let projector = self.multimodal.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Multimodal projector not attached. Call embedder.attach_multimodal_projector() first.")
        })?;

        // 1. Encode audio through Audio Tower
        let (soft_tokens, num_soft_tokens) = projector.encode_audio(samples)?;

        // 2. Tokenize prompt with special multimodal tokens: <bos> <boa> <audio>*N <eoa> [prompt] <eos>
        let token_ids = self.tokenizer.encode_audio_query(prompt, num_soft_tokens)?;
        let seq_len = token_ids.len();
        let input_tensor = Tensor::from_vec(token_ids, (1, seq_len), &self.device)?;

        // 3. Embed tokens with scaling
        let text_embeds = (self.model.embed_tokens.forward(&input_tensor)? * self.model.embed_scale)?;

        // 4. Splice soft tokens into <audio> slots: [0..2] text, [2..2+num_soft_tokens] soft tokens, [2+num_soft_tokens..] prompt + eos
        let pre = text_embeds.narrow(1, 0, 2)?;
        let soft = soft_tokens.unsqueeze(0)?;
        let post = text_embeds.narrow(1, 2 + num_soft_tokens, seq_len - (2 + num_soft_tokens))?;
        let merged_embeds = Tensor::cat(&[&pre, &soft, &post], 1)?;

        // 5. Forward through backbone and pool
        let token_embeds = self.model.forward_embeds(&merged_embeds, None)?;
        let pooled = self.model.pool_and_normalize(&token_embeds, None, truncate_dim)?;
        let pooled_f32 = pooled.to_dtype(DType::F32)?;
        let vec = pooled_f32.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vec)
    }

    pub fn set_attention_mode(&mut self, mode: AttentionMode) {
        self.model.set_attention_mode(mode);
    }

    pub fn format_query(&self, query: &str, task: &str) -> String {
        self.tokenizer.format_query(query, task)
    }

    pub fn format_document(&self, content: &str, title: Option<&str>) -> String {
        self.tokenizer.format_document(content, title)
    }

    pub fn embed_query(&self, query: &str, task: &str, truncate_dim: Option<usize>) -> Result<Vec<f32>> {
        let formatted = self.tokenizer.format_query(query, task);
        self.encode_single_text(&formatted, truncate_dim)
    }

    pub fn embed_document(&self, content: &str, title: Option<&str>, truncate_dim: Option<usize>) -> Result<Vec<f32>> {
        let formatted = self.tokenizer.format_document(content, title);
        self.encode_single_text(&formatted, truncate_dim)
    }

    pub fn embed_symmetric(&self, text: &str, task: &str, truncate_dim: Option<usize>) -> Result<Vec<f32>> {
        let formatted = self.tokenizer.format_query(text, task);
        self.encode_single_text(&formatted, truncate_dim)
    }

    pub fn embed_batch_queries(
        &self,
        queries: &[&str],
        task: &str,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let formatted: Vec<String> = queries.iter().map(|q| self.tokenizer.format_query(q, task)).collect();
        let formatted_refs: Vec<&str> = formatted.iter().map(|s| s.as_str()).collect();
        self.encode_batch_texts(&formatted_refs, truncate_dim)
    }

    pub fn embed_batch_documents(
        &self,
        contents: &[&str],
        titles: Option<&[&str]>,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let formatted: Vec<String> = match titles {
            Some(t_list) => contents
                .iter()
                .zip(t_list.iter())
                .map(|(c, t)| self.tokenizer.format_document(c, Some(t)))
                .collect(),
            None => contents
                .iter()
                .map(|c| self.tokenizer.format_document(c, None))
                .collect(),
        };
        let formatted_refs: Vec<&str> = formatted.iter().map(|s| s.as_str()).collect();
        self.encode_batch_texts(&formatted_refs, truncate_dim)
    }

    pub fn encode_batch_texts(
        &self,
        texts: &[&str],
        truncate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let batch_size = texts.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        if batch_size == 1 {
            let single = self.encode_single_text(texts[0], truncate_dim)?;
            return Ok(vec![single]);
        }

        let (input_ids, attention_mask) = self.tokenizer.encode_batch_padded(texts, &self.device)?;
        let token_embeds = self.model.forward(&input_ids, Some(&attention_mask))?;
        let pooled = self.model.pool_and_normalize(&token_embeds, Some(&attention_mask), truncate_dim)?;
        let pooled_f32 = pooled.to_dtype(DType::F32)?;

        let mut result = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let vec = pooled_f32.get(b)?.to_vec1::<f32>()?;
            result.push(vec);
        }
        Ok(result)
    }

    fn encode_single_text(&self, text: &str, truncate_dim: Option<usize>) -> Result<Vec<f32>> {
        let token_ids = self.tokenizer.encode(text)?;
        let seq_len = token_ids.len();
        let input_tensor = Tensor::from_vec(token_ids, (1, seq_len), &self.device)?;

        // Forward through backbone
        let token_embeds = self.model.forward(&input_tensor, None)?;

        // Pool and normalize
        let pooled = self.model.pool_and_normalize(&token_embeds, None, truncate_dim)?;

        // Cast to F32 vec
        let pooled_f32 = pooled.to_dtype(DType::F32)?;
        let vec = pooled_f32.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vec)
    }

    pub fn profile_encode_batch(
        &self,
        texts: &[&str],
        truncate_dim: Option<usize>,
    ) -> Result<(Vec<Vec<f32>>, PipelineProfile)> {
        let t_total = std::time::Instant::now();
        let batch_size = texts.len();

        let t_tok = std::time::Instant::now();
        let (input_ids, attention_mask) = if batch_size == 1 {
            let ids = self.tokenizer.encode(texts[0])?;
            let seq_len = ids.len();
            let t = Tensor::from_vec(ids, (1, seq_len), &self.device)?;
            (t, None)
        } else {
            let (ids, mask) = self.tokenizer.encode_batch_padded(texts, &self.device)?;
            (ids, Some(mask))
        };
        let tokenizer_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
        let seq_len = input_ids.dim(1)?;

        let (token_embeds, forward_profile) = self.model.forward_profiled(&input_ids, attention_mask.as_ref())?;

        let t_pool = std::time::Instant::now();
        let pooled = self.model.pool_and_normalize(&token_embeds, attention_mask.as_ref(), truncate_dim)?;
        let pooled_f32 = pooled.to_dtype(DType::F32)?;
        let pooling_ms = t_pool.elapsed().as_secs_f64() * 1000.0;

        let t_mrl = std::time::Instant::now();
        let mut result = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let vec = pooled_f32.get(b)?.to_vec1::<f32>()?;
            result.push(vec);
        }
        let mrl_l2norm_ms = t_mrl.elapsed().as_secs_f64() * 1000.0;
        let total_pipeline_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        let total_tokens = batch_size * seq_len;
        let tokens_per_sec = if total_pipeline_ms > 0.0 {
            (total_tokens as f64) / (total_pipeline_ms / 1000.0)
        } else {
            0.0
        };

        let pipe_profile = PipelineProfile {
            batch_size,
            seq_len,
            tokenizer_ms,
            forward: forward_profile,
            pooling_ms,
            mrl_l2norm_ms,
            total_pipeline_ms,
            tokens_per_sec,
        };

        Ok((result, pipe_profile))
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}
