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
#[cfg(feature = "coreml")]
pub mod coreml;

pub use attention::AttentionMode;
#[cfg(feature = "coreml")]
pub use coreml::{CoreMLComputeUnits, CoreMLTextBackbone, CoreMLVisionTower};
pub use linear::LinearOp;
pub use model::EmbeddingGemma2Model;
pub use multimodal::{
    ModalityPart, MultimodalProjector, VISION_BUDGET_DEFAULT, VISION_BUDGET_HIGH,
    VISION_BUDGET_LOW, VISION_BUDGET_MAX, VISION_BUDGET_MEDIUM, VISION_BUDGET_MIN,
    VISION_BUDGET_ULTRA, validate_vision_budget,
};
pub use profile::{ForwardProfile, LayerProfile, PipelineProfile};
pub use task::Task;
pub use util::{
    binary_cosine_similarity, cosine_similarity, hamming_distance, int8_cosine_similarity,
    int8_dot_product, l2_normalize, quantize_binary, quantize_int8, truncate_mrl,
    with_autorelease_pool, AutoreleaseScope,
};


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

/// An input item for unified multimodal batch embedding.
/// Can represent plain text, a single image/audio, or an arbitrary interleaved sequence of parts,
/// accompanied by an embedding Task.
#[derive(Debug, Clone)]
pub struct MultimodalInput<'a> {
    pub parts: std::borrow::Cow<'a, [ModalityPart<'a>]>,
    pub task: Task<'a>,
}

impl<'a> MultimodalInput<'a> {
    pub fn new(parts: &'a [ModalityPart<'a>], task: Task<'a>) -> Self {
        Self {
            parts: std::borrow::Cow::Borrowed(parts),
            task,
        }
    }

    pub fn text(text: &'a str, task: Task<'a>) -> Self {
        Self {
            parts: std::borrow::Cow::Owned(vec![ModalityPart::Text(text)]),
            task,
        }
    }

    pub fn image(image: &'a image::DynamicImage, budget: Option<usize>, task: Task<'a>) -> Self {
        Self {
            parts: std::borrow::Cow::Owned(vec![ModalityPart::Image { image, budget }]),
            task,
        }
    }

    pub fn image_with_prompt(
        image: &'a image::DynamicImage,
        prompt: &'a str,
        budget: Option<usize>,
        task: Task<'a>,
    ) -> Self {
        Self {
            parts: std::borrow::Cow::Owned(vec![
                ModalityPart::Text(prompt),
                ModalityPart::Image { image, budget },
            ]),
            task,
        }
    }

    pub fn audio(samples: &'a [f32], task: Task<'a>) -> Self {
        Self {
            parts: std::borrow::Cow::Owned(vec![ModalityPart::Audio { samples }]),
            task,
        }
    }

    pub fn audio_with_prompt(samples: &'a [f32], prompt: &'a str, task: Task<'a>) -> Self {
        Self {
            parts: std::borrow::Cow::Owned(vec![
                ModalityPart::Text(prompt),
                ModalityPart::Audio { samples },
            ]),
            task,
        }
    }
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
        with_autorelease_pool(|| self.embed_composite_inner(prompt, items, truncate_dim))
    }

    fn embed_composite_inner(
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
        with_autorelease_pool(|| self.embed_interleaved_inner(parts, task))
    }

    fn embed_interleaved_inner(
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

    /// Unified batch embedding across arbitrary modalities (text, images, audio, interleaved).
    /// Dynamically optimizes homogeneous sequences into unpadded zero-mask passes,
    /// and safely vectorizes heterogeneous sequences with padding attention masks.
    pub fn embed_batch(
        &self,
        inputs: &[MultimodalInput],
        truncate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let b = inputs.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        if b == 1 {
            let single = self.embed_interleaved(&inputs[0].parts, inputs[0].task)?;
            let vec = match truncate_dim {
                Some(dim) if dim < 768 => truncate_mrl(&single, dim)?,
                _ => single,
            };
            return Ok(vec![vec]);
        }

        with_autorelease_pool(|| self.embed_batch_inner(inputs, truncate_dim))
    }

    fn embed_batch_inner(
        &self,
        inputs: &[MultimodalInput],
        truncate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let b = inputs.len();

        // 1. Check multimodal projector requirement if media parts exist
        let has_any_media = inputs.iter().any(|inp| {
            inp.parts.iter().any(|p| !matches!(p, ModalityPart::Text(_)))
        });
        if has_any_media && self.multimodal.is_none() {
            anyhow::bail!("Multimodal input received in batch, but multimodal projector is not attached.");
        }

        // 2. Pre-extract and batch-encode all images across all items
        let mut image_requests = Vec::new();
        for (item_idx, inp) in inputs.iter().enumerate() {
            for (part_idx, part) in inp.parts.iter().enumerate() {
                if let ModalityPart::Image { image, budget } = part {
                    let b = budget.unwrap_or(multimodal::VISION_BUDGET_DEFAULT);
                    image_requests.push((item_idx, part_idx, *image, b));
                }
            }
        }

        let encoded_images: Vec<(Tensor, usize)> = if !image_requests.is_empty() {
            let projector = self.multimodal.as_ref().unwrap();
            let images_and_budgets: Vec<(&image::DynamicImage, usize)> = image_requests
                .iter()
                .map(|(_, _, img, b)| (*img, *b))
                .collect();
            projector.encode_images_batch(&images_and_budgets)?
        } else {
            Vec::new()
        };

        let mut image_iter = encoded_images.into_iter();

        // 3. Assemble spliced token representations for each item
        let mut item_embeds = Vec::with_capacity(b);
        let mut seq_lens = Vec::with_capacity(b);


        for (item_idx, inp) in inputs.iter().enumerate() {
            let (token_ids, slots) = self.tokenizer.encode_modality_sequence(&inp.parts, inp.task)?;
            let seq_len = token_ids.len();
            if seq_len > self.max_seq_len {
                anyhow::bail!(
                    "Batch item {} sequence length ({}) exceeds maximum context limit ({})",
                    item_idx,
                    seq_len,
                    self.max_seq_len
                );
            }
            seq_lens.push(seq_len);

            let mut soft_tensors = Vec::new();
            if let Some(projector) = self.multimodal.as_ref() {
                for part in inp.parts.iter() {
                    match part {
                        ModalityPart::Text(_) => {}
                        ModalityPart::Image { .. } => {
                            let (soft_toks, _) = image_iter.next().expect("Image count mismatch");
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
            let merged = if chunks_refs.len() == 1 {
                chunks_refs[0].clone()
            } else {
                Tensor::cat(&chunks_refs, 1)?
            };
            item_embeds.push(merged);
        }

        // 4. Batch stacking & Attention Mask construction

        let max_len = *seq_lens.iter().max().unwrap();
        let is_homogeneous = seq_lens.iter().all(|&l| l == max_len);

        let (batched_embeds, pad_mask) = if is_homogeneous {
            let refs: Vec<&Tensor> = item_embeds.iter().collect();
            let stacked = Tensor::cat(&refs, 0)?; // [b, max_len, 512]
            (stacked, None)
        } else {
            // Heterogeneous batch: pad shorter items to max_len with zeros
            let mut padded_items = Vec::with_capacity(b);
            let mut mask_vec = vec![0.0f32; b * max_len];

            for (i, (item, &len)) in item_embeds.into_iter().zip(seq_lens.iter()).enumerate() {
                for j in 0..len {
                    mask_vec[i * max_len + j] = 1.0;
                }

                if len < max_len {
                    let pad_len = max_len - len;
                    let pad_tensor = Tensor::zeros((1, pad_len, 512), self.dtype, &self.device)?;
                    let padded = Tensor::cat(&[&item, &pad_tensor], 1)?;
                    padded_items.push(padded);
                } else {
                    padded_items.push(item);
                }
            }

            let refs: Vec<&Tensor> = padded_items.iter().collect();
            let stacked = Tensor::cat(&refs, 0)?; // [b, max_len, 512]
            let mask_tensor = Tensor::from_vec(mask_vec, (b, max_len), &self.device)?
                .to_dtype(self.dtype)?;
            (stacked, Some(mask_tensor))
        };

        // 5. Forward through language model backbone and pool
        let token_embeds = self.model.forward_embeds(&batched_embeds, pad_mask.as_ref())?;
        let pooled = self.model.pool_and_normalize(&token_embeds, pad_mask.as_ref(), truncate_dim)?;
        let pooled_f32 = pooled.to_dtype(DType::F32)?;
        let out_dim = pooled_f32.dim(1)?;
        let all_floats = pooled_f32.flatten_all()?.to_vec1::<f32>()?;

        let mut results = Vec::with_capacity(b);
        for chunk in all_floats.chunks_exact(out_dim) {
            results.push(chunk.to_vec());
        }

        Ok(results)
    }


    pub fn encode_batch_texts(
        &self,
        texts: &[&str],
        truncate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<MultimodalInput> = texts
            .iter()
            .map(|t| MultimodalInput::text(t, Task::Raw))
            .collect();
        self.embed_batch(&inputs, truncate_dim)
    }


    fn encode_single_text(&self, text: &str, truncate_dim: Option<usize>) -> Result<Vec<f32>> {
        with_autorelease_pool(|| {
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
        })
    }

    pub fn profile_encode_batch(
        &self,
        texts: &[&str],
        truncate_dim: Option<usize>,
    ) -> Result<(Vec<Vec<f32>>, PipelineProfile)> {
        with_autorelease_pool(|| {
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
        })
    }
}
