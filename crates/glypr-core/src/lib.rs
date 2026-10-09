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
pub mod task;
pub mod util;
#[cfg(feature = "metal")]
pub mod metal_kernels;

pub use attention::AttentionMode;
pub use linear::LinearOp;
pub use model::EmbeddingGemma2Model;
pub use multimodal::{
    ModalityPart, MultimodalProjector, VISION_BUDGET_DEFAULT, VISION_BUDGET_HIGH,
    VISION_BUDGET_LOW, VISION_BUDGET_MAX, VISION_BUDGET_MEDIUM, VISION_BUDGET_MIN,
    VISION_BUDGET_ULTRA, validate_vision_budget,
};
pub use profile::{ForwardProfile, LayerProfile, PipelineProfile};
pub use task::Task;
pub use util::{cosine_similarity, l2_normalize, truncate_mrl};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use config::TextConfig;
use std::path::Path;
use tokenizer::GlyprTokenizer;

// 8,192 content tokens + 128 tokens headroom for task prefixes, document titles, and special tokens
pub const DEFAULT_MAX_SEQ_LEN: usize = 8192 + 128; // 8,320 tokens


pub struct GlyprEmbedder {
    pub model: EmbeddingGemma2Model,
    pub tokenizer: GlyprTokenizer,
    pub device: Device,
    #[allow(dead_code)]
    pub dtype: DType,
    pub multimodal: Option<MultimodalProjector>,
    pub max_seq_len: usize,
}

#[derive(Debug, Clone)]
pub enum MediaItem<'a> {
    Image {
        image: &'a image::DynamicImage,
        budget: Option<usize>,
    },
    Audio {
        samples: &'a [f32],
    },
    VideoFrames {
        frames: &'a [&'a image::DynamicImage],
        budget_per_frame: Option<usize>,
    },
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
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
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
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
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

    pub fn set_max_seq_len(&mut self, len: usize) {
        self.max_seq_len = len;
    }

    pub fn with_max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Encode an arbitrary composite sequence: text + multiple images + audio + video frames
    pub fn embed_composite(
        &self,
        prompt: Option<&str>,
        items: &[MediaItem],
        truncate_dim: Option<usize>,
    ) -> Result<Vec<f32>> {
        let projector = self.multimodal.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Multimodal projector not attached. Call embedder.attach_multimodal_projector() first.")
        })?;

        // =========================================================================
        // PHASE 1: Zero-Overhead Preflight Check (< 10 µs, 0 neural forwards)
        // =========================================================================
        let mut placeholders = Vec::with_capacity(items.len());
        for item in items {
            match item {
                MediaItem::Image { image, budget } => {
                    let b = budget.unwrap_or(280);
                    let n_tokens = crate::multimodal::compute_soft_token_count(
                        image.width() as usize,
                        image.height() as usize,
                        b,
                        16,
                        3,
                    )?;
                    placeholders.push(crate::tokenizer::MediaPlaceholder::Image {
                        num_soft_tokens: n_tokens,
                    });
                }
                MediaItem::Audio { samples } => {
                    let n_tokens = crate::multimodal::compute_audio_soft_token_count(samples.len());
                    placeholders.push(crate::tokenizer::MediaPlaceholder::Audio {
                        num_soft_tokens: n_tokens,
                    });
                }
                MediaItem::VideoFrames { frames, budget_per_frame } => {
                    let b = budget_per_frame.unwrap_or(140);
                    let n_tokens = if frames.is_empty() {
                        0
                    } else {
                        crate::multimodal::compute_soft_token_count(
                            frames[0].width() as usize,
                            frames[0].height() as usize,
                            b,
                            16,
                            3,
                        )?
                    };
                    placeholders.push(crate::tokenizer::MediaPlaceholder::Video {
                        num_frames: frames.len(),
                        soft_tokens_per_frame: n_tokens,
                    });
                }
            }
        }

        // 1. Tokenize prompt with special multimodal tokens: <bos> ... [media blocks] ... [prompt] <eos>
        let (token_ids, slots) = self.tokenizer.encode_interleaved(prompt, &placeholders)?;
        let seq_len = token_ids.len();

        // 2. Preflight guard: fail fast if sequence length exceeds maximum context window
        if seq_len > self.max_seq_len {
            anyhow::bail!(
                "Multimodal sequence length ({} tokens) exceeds maximum context limit ({} tokens). \
                 Preflight check failed before neural encoding (media items: {}).",
                seq_len,
                self.max_seq_len,
                items.len()
            );
        }

        // =========================================================================
        // PHASE 2: Neural Tower Execution (ViT & Conformer)
        // =========================================================================
        let mut soft_tensors = Vec::with_capacity(items.len());
        for item in items {
            match item {
                MediaItem::Image { image, budget } => {
                    let b = budget.unwrap_or(280);
                    let (soft_toks, _) = projector.encode_image_with_budget(image, b)?;
                    soft_tensors.push(soft_toks);
                }
                MediaItem::Audio { samples } => {
                    let (soft_toks, _) = projector.encode_audio(samples)?;
                    soft_tensors.push(soft_toks);
                }
                MediaItem::VideoFrames { frames, budget_per_frame } => {
                    let b = budget_per_frame.unwrap_or(140);
                    for frame in frames.iter() {
                        let (soft_toks, _) = projector.encode_image_with_budget(frame, b)?;
                        soft_tensors.push(soft_toks);
                    }
                }
            }
        }

        // =========================================================================
        // PHASE 3: Splicing & Backbone Forward Pass
        // =========================================================================
        let input_tensor = Tensor::from_vec(token_ids, (1, seq_len), &self.device)?;

        // 1. Embed tokens with scaling
        let text_embeds = (self.model.embed_tokens.forward(&input_tensor)? * self.model.embed_scale)?;

        // 2. Splice soft tokens into their respective slot ranges
        let mut chunks = Vec::new();
        let mut curr = 0;

        for (slot, soft_tensor) in slots.iter().zip(soft_tensors.iter()) {
            if slot.start_idx > curr {
                chunks.push(text_embeds.narrow(1, curr, slot.start_idx - curr)?);
            }
            chunks.push(soft_tensor.unsqueeze(0)?);
            curr = slot.start_idx + slot.length;
        }

        if curr < seq_len {
            chunks.push(text_embeds.narrow(1, curr, seq_len - curr)?);
        }

        let chunks_refs: Vec<&Tensor> = chunks.iter().collect();
        let merged_embeds = Tensor::cat(&chunks_refs, 1)?;

        // 3. Forward through backbone and pool
        let token_embeds = self.model.forward_embeds(&merged_embeds, None)?;
        let pooled = self.model.pool_and_normalize(&token_embeds, None, truncate_dim)?;
        let pooled_f32 = pooled.to_dtype(DType::F32)?;
        let vec = pooled_f32.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vec)
    }

    /// Encode a single text input formatted according to the specified Task.
    /// Returns a native 768-dimensional L2-normalized embedding vector.
    pub fn embed_text(&self, text: &str, task: Task) -> Result<Vec<f32>> {
        let formatted = task.format_text(text);
        self.encode_single_text(&formatted, None)
    }

    /// Encode a batch of texts formatted according to the specified Task.
    /// Returns native 768-dimensional L2-normalized embedding vectors.
    pub fn embed_texts(&self, texts: &[&str], task: Task) -> Result<Vec<Vec<f32>>> {
        let formatted: Vec<String> = texts.iter().map(|t| task.format_text(t)).collect();
        let formatted_refs: Vec<&str> = formatted.iter().map(|s| s.as_str()).collect();
        self.encode_batch_texts(&formatted_refs, None)
    }

    /// Encode an ordered sequence of multimodal parts (interleaved text, images, audio, video frames)
    /// with an optional Task prompt.
    ///
    /// Matches Python `sentence-transformers` list format:
    /// `embedder.embed_interleaved(&[ModalityPart::Text("..."), ModalityPart::Image { ... }], Task::SearchQuery)`
    /// Returns native 768-dimensional L2-normalized embedding vector.
    pub fn embed_interleaved(
        &self,
        parts: &[ModalityPart],
        task: Task,
    ) -> Result<Vec<f32>> {
        let has_media = parts.iter().any(|p| !matches!(p, ModalityPart::Text(_)));
        if has_media && self.multimodal.is_none() {
            anyhow::bail!("Multimodal projector not attached. Call embedder.attach_multimodal_projector() first.");
        }

        // 1. Tokenize sequence and validate context window preflight (< 10 µs, 0 neural forwards)
        let (token_ids, slots) = self.tokenizer.encode_modality_sequence(parts, task)?;
        let seq_len = token_ids.len();

        if seq_len > self.max_seq_len {
            anyhow::bail!(
                "Interleaved multimodal sequence length ({} tokens) exceeds maximum context limit ({} tokens). \
                 Preflight check failed before neural encoding (parts: {}).",
                seq_len,
                self.max_seq_len,
                parts.len()
            );
        }

        // 2. Neural tower execution for media parts
        let mut soft_tensors = Vec::new();
        if let Some(projector) = self.multimodal.as_ref() {
            for part in parts {
                match part {
                    ModalityPart::Text(_) => {}
                    ModalityPart::Image { image, budget } => {
                        let b = budget.unwrap_or(multimodal::VISION_BUDGET_DEFAULT);
                        let (soft_toks, _) = projector.encode_image_with_budget(image, b)?;
                        soft_tensors.push(soft_toks);
                    }
                    ModalityPart::Audio { samples } => {
                        let (soft_toks, _) = projector.encode_audio(samples)?;
                        soft_tensors.push(soft_toks);
                    }
                    ModalityPart::VideoFrame { frame, budget } => {
                        let b = budget.unwrap_or(multimodal::VISION_BUDGET_MEDIUM);
                        let (soft_toks, _) = projector.encode_image_with_budget(frame, b)?;
                        soft_tensors.push(soft_toks);
                    }
                }
            }
        }

        // 3. Splicing & Backbone Forward Pass
        let input_tensor = Tensor::from_vec(token_ids, (1, seq_len), &self.device)?;
        let text_embeds = (self.model.embed_tokens.forward(&input_tensor)? * self.model.embed_scale)?;

        let mut chunks = Vec::new();
        let mut curr = 0;

        for (slot, soft_tensor) in slots.iter().zip(soft_tensors.iter()) {
            if slot.start_idx > curr {
                chunks.push(text_embeds.narrow(1, curr, slot.start_idx - curr)?);
            }
            chunks.push(soft_tensor.unsqueeze(0)?);
            curr = slot.start_idx + slot.length;
        }

        if curr < seq_len {
            chunks.push(text_embeds.narrow(1, curr, seq_len - curr)?);
        }

        let chunks_refs: Vec<&Tensor> = chunks.iter().collect();
        let merged_embeds = Tensor::cat(&chunks_refs, 1)?;

        let token_embeds = self.model.forward_embeds(&merged_embeds, None)?;
        let pooled = self.model.pool_and_normalize(&token_embeds, None, None)?;
        let pooled_f32 = pooled.to_dtype(DType::F32)?;
        let vec = pooled_f32.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vec)
    }

    /// Encode an image (with optional prompt) into a dense normalized embedding vector using default budget (280).
    /// Returns native 768-dimensional normalized embedding vector.
    pub fn embed_image(
        &self,
        image: &image::DynamicImage,
        prompt: Option<&str>,
    ) -> Result<Vec<f32>> {
        self.embed_image_with_budget(image, prompt, multimodal::VISION_BUDGET_DEFAULT)
    }

    /// Encode an image with a specific token budget (e.g. 70, 140, 280, 560, 1120).
    /// Returns native 768-dimensional normalized embedding vector.
    pub fn embed_image_with_budget(
        &self,
        image: &image::DynamicImage,
        prompt: Option<&str>,
        budget: usize,
    ) -> Result<Vec<f32>> {
        multimodal::validate_vision_budget(budget)?;
        self.embed_composite(
            prompt,
            &[MediaItem::Image {
                image,
                budget: Some(budget),
            }],
            None,
        )
    }

    /// Encode an audio waveform (with optional prompt) into a dense normalized embedding vector.
    /// Returns native 768-dimensional normalized embedding vector.
    pub fn embed_audio(
        &self,
        samples: &[f32],
        prompt: Option<&str>,
    ) -> Result<Vec<f32>> {
        self.embed_composite(
            prompt,
            &[MediaItem::Audio { samples }],
            None,
        )
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

        let (input_ids, attention_mask) = self.tokenizer.encode_batch_padded_with_limit(texts, self.max_seq_len, &self.device)?;
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
        if seq_len > self.max_seq_len {
            anyhow::bail!(
                "Text sequence length ({} tokens) exceeds maximum context limit ({} tokens)",
                seq_len,
                self.max_seq_len
            );
        }
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
            if seq_len > self.max_seq_len {
                anyhow::bail!(
                    "Text sequence length ({} tokens) exceeds maximum context limit ({} tokens)",
                    seq_len,
                    self.max_seq_len
                );
            }
            let t = Tensor::from_vec(ids, (1, seq_len), &self.device)?;
            (t, None)
        } else {
            let (ids, mask) = self.tokenizer.encode_batch_padded_with_limit(texts, self.max_seq_len, &self.device)?;
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
