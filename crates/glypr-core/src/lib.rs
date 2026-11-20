pub mod attention;
pub mod config;
#[cfg(feature = "coreml")]
pub mod coreml;
pub mod cpu_simd;
pub mod hub;
pub mod layer;
pub mod linear;
#[cfg(feature = "metal")]
pub mod metal_kernels;
pub mod mlp;
pub mod model;
pub mod multimodal;
pub mod norm;
pub mod ple;
pub mod profile;
pub mod rope;
pub mod task;
pub mod tokenizer;
pub mod util;

pub use attention::AttentionMode;
#[cfg(feature = "coreml")]
pub use coreml::{CoreMLComputeUnits, CoreMLTextBackbone, CoreMLVisionTower};
pub use linear::LinearOp;
pub use model::EmbeddingGemma2Model;
pub use multimodal::{
    ModalityPart, MultimodalProjector, VISION_BUDGET_DEFAULT, VISION_BUDGET_HIGH,
    VISION_BUDGET_LOW, VISION_BUDGET_MAX, VISION_BUDGET_MEDIUM, VISION_BUDGET_MIN,
    VISION_BUDGET_ULTRA, VisionBackendPolicy, validate_vision_budget,
};
pub use profile::{ForwardProfile, LayerProfile, PipelineProfile};
pub use task::{OwnedTask, Task};
pub use util::{
    AutoreleaseScope, binary_cosine_similarity, cosine_similarity, hamming_distance,
    int8_cosine_similarity, int8_dot_product, l2_normalize, quantize_binary, quantize_int8,
    truncate_mrl, with_autorelease_pool,
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
    #[cfg(feature = "coreml")]
    pub coreml_vision: Option<CoreMLVisionTower>,
    pub vision_backend: VisionBackendPolicy,
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
            #[cfg(feature = "coreml")]
            coreml_vision: None,
            vision_backend: VisionBackendPolicy::Auto,
        })
    }

    /// Load a 100% self-contained GGUF file (extracts model weights, architecture config, and BPE tokenizer).
    /// Zero external file or HuggingFace snapshot dependencies!
    pub fn load_gguf<P: AsRef<Path>>(gguf_path: P, device: &Device, dtype: DType) -> Result<Self> {
        let mut file = std::fs::File::open(gguf_path.as_ref())
            .with_context(|| format!("Failed to open GGUF file {:?}", gguf_path.as_ref()))?;

        let content =
            candle_core::quantized::gguf_file::Content::read(&mut file).with_context(|| {
                format!(
                    "Failed to read GGUF container from {:?}",
                    gguf_path.as_ref()
                )
            })?;

        let config = TextConfig::from_gguf_metadata(&content.metadata)
            .context("Failed to parse TextConfig from GGUF metadata")?;

        let tokenizer = GlyprTokenizer::from_gguf_metadata(&content.metadata)
            .context("Failed to parse GlyprTokenizer from GGUF metadata")?;

        let model =
            EmbeddingGemma2Model::from_gguf_content(&content, &mut file, config, device, dtype)
                .map_err(|e| anyhow::anyhow!("GGUF Model initialization failed: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
            multimodal: None,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            #[cfg(feature = "coreml")]
            coreml_vision: None,
            vision_backend: VisionBackendPolicy::Auto,
        })
    }

    /// Load self-contained BF16 GGUF file (e.g., `embeddinggemma-2-BF16.gguf`)
    pub fn load_gguf_bf16<P: AsRef<Path>>(gguf_path: P, device: &Device) -> Result<Self> {
        let dtype = if device.is_metal() {
            DType::BF16
        } else {
            DType::F32
        };
        Self::load_gguf(gguf_path, device, dtype)
    }

    /// Load self-contained Q8_0 quantized GGUF file (e.g., `embeddinggemma-2-Q8_0.gguf`)
    pub fn load_gguf_q8<P: AsRef<Path>>(gguf_path: P, device: &Device) -> Result<Self> {
        let dtype = if device.is_metal() {
            DType::BF16
        } else {
            DType::F32
        };
        Self::load_gguf(gguf_path, device, dtype)
    }

    /// Dynamically attach an optional multimodal projector GGUF (`mmproj-*.gguf`)
    pub fn attach_multimodal_projector<P: AsRef<Path>>(&mut self, mmproj_path: P) -> Result<()> {
        let projector =
            MultimodalProjector::from_gguf_file(mmproj_path.as_ref(), &self.device, self.dtype)?;
        self.multimodal = Some(projector);
        Ok(())
    }

    #[cfg(feature = "coreml")]
    /// Dynamically attach a compiled CoreML Vision Tower (`.mlmodelc`) for Apple Neural Engine (ANE) acceleration.
    pub fn attach_coreml_vision(&mut self, vision_tower: CoreMLVisionTower) {
        self.coreml_vision = Some(vision_tower);
    }

    #[cfg(feature = "coreml")]
    /// Dynamically attach CoreML Vision Tower models discovered in a directory (e.g. `models/coreml`).
    pub fn attach_coreml_vision_dir<P: AsRef<Path>>(
        &mut self,
        dir: P,
        compute_units: coreml::CoreMLComputeUnits,
    ) -> Result<()> {
        let tower = CoreMLVisionTower::from_dir(dir, compute_units)?;
        self.coreml_vision = Some(tower);
        Ok(())
    }

    /// Set routing policy for vision execution (Auto, AneStrict, Metal).
    pub fn set_vision_backend(&mut self, policy: VisionBackendPolicy) {
        self.vision_backend = policy;
    }

    pub fn with_vision_backend(mut self, policy: VisionBackendPolicy) -> Self {
        self.vision_backend = policy;
        self
    }

    pub fn set_max_seq_len(&mut self, len: usize) {
        self.max_seq_len = len;
    }

    pub fn with_max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Whether this embedder has any multimodal capabilities (GGUF mmproj or CoreML vision).
    pub fn has_multimodal(&self) -> bool {
        self.multimodal.is_some() || self.has_coreml_vision()
    }

    /// Whether CoreML Vision models are attached.
    pub fn has_coreml_vision(&self) -> bool {
        #[cfg(feature = "coreml")]
        {
            self.coreml_vision.is_some()
        }
        #[cfg(not(feature = "coreml"))]
        {
            false
        }
    }

    /// Validate and normalize an optional Matryoshka (MRL) truncation dimension.
    ///
    /// Rejects `Some(0)` and `Some(d)` where `d > self.model.config.embedding_dim`,
    /// and normalizes `Some(embedding_dim)` to `None`.
    pub fn validate_truncate_dim(&self, truncate_dim: Option<usize>) -> Result<Option<usize>> {
        let embedding_dim = self.model.config.embedding_dim;
        match truncate_dim {
            None => Ok(None),
            Some(0) => anyhow::bail!("truncate_dim cannot be 0"),
            Some(d) if d > embedding_dim => anyhow::bail!(
                "truncate_dim ({}) exceeds model embedding_dim ({})",
                d,
                embedding_dim
            ),
            Some(d) if d == embedding_dim => Ok(None),
            Some(d) => Ok(Some(d)),
        }
    }

    /// Encode a single image item using the routed vision engine (ANE CoreML or Metal/CPU GGUF).
    pub fn encode_image_part(&self, image: &image::DynamicImage, budget: usize) -> Result<Tensor> {
        let processed = crate::multimodal::preprocess_image(image, budget, 16, 3)?;

        #[cfg(feature = "coreml")]
        {
            let use_coreml = match self.vision_backend {
                VisionBackendPolicy::Auto => self.coreml_vision.as_ref().is_some_and(|cv| {
                    cv.supports_grid(processed.patch_height, processed.patch_width)
                }),
                VisionBackendPolicy::AneStrict => {
                    let cv = self.coreml_vision.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "AneStrict vision backend requested, but no CoreML vision tower attached"
                        )
                    })?;
                    if !cv.supports_grid(processed.patch_height, processed.patch_width) {
                        anyhow::bail!(
                            "AneStrict vision routing failed: image produced {}x{} patches ({} total), but loaded CoreML models only support grids {:?}",
                            processed.patch_height,
                            processed.patch_width,
                            processed.num_patches,
                            cv.supported_grids()
                        );
                    }
                    true
                }
                VisionBackendPolicy::Metal => false,
            };

            if use_coreml && let Some(cv) = &self.coreml_vision {
                return cv.encode_to_tensor_dtype(&processed, &self.device, self.dtype);
            }
        }

        // Standard Candle / GGUF path
        let projector = self.multimodal.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No vision engine available for image (CoreML model does not support {}x{} patches ({} total) and no GGUF mmproj attached)",
                processed.patch_height,
                processed.patch_width,
                processed.num_patches
            )
        })?;
        let tower = projector.vision_tower.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Vision Tower is not available in attached multimodal projector")
        })?;
        Ok(tower.encode(&processed)?)
    }

    /// Encode an arbitrary composite sequence: text + multiple images + audio + video frames
    pub fn embed_composite(
        &self,
        prompt: Option<&str>,
        items: &[MediaItem],
        truncate_dim: Option<usize>,
    ) -> Result<Vec<f32>> {
        let truncate_dim = self.validate_truncate_dim(truncate_dim)?;
        with_autorelease_pool(|| self.embed_composite_inner(prompt, items, truncate_dim))
    }

    fn embed_composite_inner(
        &self,
        prompt: Option<&str>,
        items: &[MediaItem],
        truncate_dim: Option<usize>,
    ) -> Result<Vec<f32>> {
        let has_coreml_vision = {
            #[cfg(feature = "coreml")]
            {
                self.coreml_vision.is_some()
            }
            #[cfg(not(feature = "coreml"))]
            {
                false
            }
        };

        if self.multimodal.is_none() && !has_coreml_vision {
            anyhow::bail!(
                "Multimodal engine not attached. Call embedder.attach_multimodal_projector() or embedder.attach_coreml_vision() first."
            );
        }

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
                    if samples.is_empty() {
                        anyhow::bail!("Input audio cannot be empty");
                    }
                    let n_tokens = crate::multimodal::compute_audio_soft_token_count(samples.len());
                    placeholders.push(crate::tokenizer::MediaPlaceholder::Audio {
                        num_soft_tokens: n_tokens,
                    });
                }
                MediaItem::VideoFrames {
                    frames,
                    budget_per_frame,
                } => {
                    if frames.is_empty() {
                        anyhow::bail!("VideoFrames input cannot be empty");
                    }
                    let b = budget_per_frame.unwrap_or(140);
                    let mut frame_soft_tokens = Vec::with_capacity(frames.len());
                    for frame in frames.iter() {
                        let n_tokens = crate::multimodal::compute_soft_token_count(
                            frame.width() as usize,
                            frame.height() as usize,
                            b,
                            16,
                            3,
                        )?;
                        frame_soft_tokens.push(n_tokens);
                    }
                    placeholders
                        .push(crate::tokenizer::MediaPlaceholder::Video { frame_soft_tokens });
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
                    let soft_toks = self.encode_image_part(image, b)?;
                    soft_tensors.push(soft_toks);
                }
                MediaItem::Audio { samples } => {
                    let projector = self.multimodal.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("Audio encoding requires attached GGUF mmproj")
                    })?;
                    let (soft_toks, _) = projector.encode_audio(samples)?;
                    soft_tensors.push(soft_toks);
                }
                MediaItem::VideoFrames {
                    frames,
                    budget_per_frame,
                } => {
                    let b = budget_per_frame.unwrap_or(140);
                    for frame in frames.iter() {
                        let soft_toks = self.encode_image_part(frame, b)?;
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
        let text_embeds =
            (self.model.embed_tokens.forward(&input_tensor)? * self.model.embed_scale)?;

        // 2. Splice soft tokens into their respective slot ranges
        let merged_embeds = splice_soft_tokens(&text_embeds, seq_len, &slots, &soft_tensors)?;

        // 3. Forward through backbone and pool
        let token_embeds = self.model.forward_embeds(&merged_embeds, None)?;
        let pooled = self
            .model
            .pool_and_normalize(&token_embeds, None, truncate_dim)?;
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
    pub fn embed_interleaved(&self, parts: &[ModalityPart], task: Task) -> Result<Vec<f32>> {
        with_autorelease_pool(|| self.embed_interleaved_inner(parts, task))
    }

    fn embed_interleaved_inner(&self, parts: &[ModalityPart], task: Task) -> Result<Vec<f32>> {
        let has_media = parts.iter().any(|p| !matches!(p, ModalityPart::Text(_)));
        let has_coreml_vision = {
            #[cfg(feature = "coreml")]
            {
                self.coreml_vision.is_some()
            }
            #[cfg(not(feature = "coreml"))]
            {
                false
            }
        };
        if has_media && self.multimodal.is_none() && !has_coreml_vision {
            anyhow::bail!(
                "Multimodal engine not attached. Call embedder.attach_multimodal_projector() or embedder.attach_coreml_vision() first."
            );
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
        for part in parts {
            match part {
                ModalityPart::Text(_) => {}
                ModalityPart::Image { image, budget } => {
                    let b = budget.unwrap_or(multimodal::VISION_BUDGET_DEFAULT);
                    let soft_toks = self.encode_image_part(image, b)?;
                    soft_tensors.push(soft_toks);
                }
                ModalityPart::Audio { samples } => {
                    let projector = self.multimodal.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("Audio encoding requires attached GGUF mmproj")
                    })?;
                    let (soft_toks, _) = projector.encode_audio(samples)?;
                    soft_tensors.push(soft_toks);
                }
                ModalityPart::VideoFrame { frame, budget } => {
                    let b = budget.unwrap_or(multimodal::VISION_BUDGET_MEDIUM);
                    let soft_toks = self.encode_image_part(frame, b)?;
                    soft_tensors.push(soft_toks);
                }
            }
        }

        // 3. Splicing & Backbone Forward Pass
        let input_tensor = Tensor::from_vec(token_ids, (1, seq_len), &self.device)?;
        let text_embeds =
            (self.model.embed_tokens.forward(&input_tensor)? * self.model.embed_scale)?;
        let merged_embeds = splice_soft_tokens(&text_embeds, seq_len, &slots, &soft_tensors)?;

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
    pub fn embed_audio(&self, samples: &[f32], prompt: Option<&str>) -> Result<Vec<f32>> {
        self.embed_composite(prompt, &[MediaItem::Audio { samples }], None)
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

    pub fn embed_query(
        &self,
        query: &str,
        task: &str,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<f32>> {
        let formatted = self.tokenizer.format_query(query, task);
        self.encode_single_text(&formatted, truncate_dim)
    }

    pub fn embed_document(
        &self,
        content: &str,
        title: Option<&str>,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<f32>> {
        let formatted = self.tokenizer.format_document(content, title);
        self.encode_single_text(&formatted, truncate_dim)
    }

    pub fn embed_symmetric(
        &self,
        text: &str,
        task: &str,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<f32>> {
        let formatted = self.tokenizer.format_query(text, task);
        self.encode_single_text(&formatted, truncate_dim)
    }

    pub fn embed_batch_queries(
        &self,
        queries: &[&str],
        task: &str,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let formatted: Vec<String> = queries
            .iter()
            .map(|q| self.tokenizer.format_query(q, task))
            .collect();
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
        let truncate_dim = self.validate_truncate_dim(truncate_dim)?;
        let b = inputs.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        if b == 1 {
            let single = self.embed_interleaved(&inputs[0].parts, inputs[0].task)?;
            let vec = match truncate_dim {
                Some(dim) => truncate_mrl(&single, dim)?,
                None => single,
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
            inp.parts
                .iter()
                .any(|p| !matches!(p, ModalityPart::Text(_)))
        });
        let has_coreml_vision = {
            #[cfg(feature = "coreml")]
            {
                self.coreml_vision.is_some()
            }
            #[cfg(not(feature = "coreml"))]
            {
                false
            }
        };
        if has_any_media && self.multimodal.is_none() && !has_coreml_vision {
            anyhow::bail!(
                "Multimodal input received in batch, but multimodal engine is not attached."
            );
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

        let encoded_images: Vec<Tensor> = if !image_requests.is_empty() {
            let use_coreml = {
                #[cfg(feature = "coreml")]
                {
                    self.coreml_vision.is_some()
                        && self.vision_backend != VisionBackendPolicy::Metal
                }
                #[cfg(not(feature = "coreml"))]
                {
                    false
                }
            };

            if use_coreml {
                let mut res = Vec::with_capacity(image_requests.len());
                for (_, _, img, b) in &image_requests {
                    res.push(self.encode_image_part(img, *b)?);
                }
                res
            } else {
                let projector = self
                    .multimodal
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Multimodal engine not attached."))?;
                let images_and_budgets: Vec<(&image::DynamicImage, usize)> = image_requests
                    .iter()
                    .map(|(_, _, img, b)| (*img, *b))
                    .collect();
                let batch_out = projector.encode_images_batch(&images_and_budgets)?;
                batch_out.into_iter().map(|(t, _)| t).collect()
            }
        } else {
            Vec::new()
        };

        let mut image_iter = encoded_images.into_iter();

        // 3. Assemble spliced token representations for each item
        let mut item_embeds = Vec::with_capacity(b);
        let mut seq_lens = Vec::with_capacity(b);

        for (item_idx, inp) in inputs.iter().enumerate() {
            let (token_ids, slots) = self
                .tokenizer
                .encode_modality_sequence(&inp.parts, inp.task)?;
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
            for part in inp.parts.iter() {
                match part {
                    ModalityPart::Text(_) => {}
                    ModalityPart::Image { .. } => {
                        let soft_toks = image_iter.next().expect("Image count mismatch");
                        soft_tensors.push(soft_toks);
                    }
                    ModalityPart::Audio { samples } => {
                        let projector = self.multimodal.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("Audio encoding requires attached GGUF mmproj")
                        })?;
                        let (soft_toks, _) = projector.encode_audio(samples)?;
                        soft_tensors.push(soft_toks);
                    }
                    ModalityPart::VideoFrame { frame, budget } => {
                        let b = budget.unwrap_or(multimodal::VISION_BUDGET_MEDIUM);
                        let soft_toks = self.encode_image_part(frame, b)?;
                        soft_tensors.push(soft_toks);
                    }
                }
            }

            let input_tensor = Tensor::from_vec(token_ids, (1, seq_len), &self.device)?;
            let text_embeds =
                (self.model.embed_tokens.forward(&input_tensor)? * self.model.embed_scale)?;
            let merged = splice_soft_tokens(&text_embeds, seq_len, &slots, &soft_tensors)?;
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
                    let pad_tensor = Tensor::zeros(
                        (1, pad_len, self.model.config.hidden_size),
                        self.dtype,
                        &self.device,
                    )?;
                    let padded = Tensor::cat(&[&item, &pad_tensor], 1)?;
                    padded_items.push(padded);
                } else {
                    padded_items.push(item);
                }
            }

            let refs: Vec<&Tensor> = padded_items.iter().collect();
            let stacked = Tensor::cat(&refs, 0)?; // [b, max_len, 512]
            let mask_tensor =
                Tensor::from_vec(mask_vec, (b, max_len), &self.device)?.to_dtype(self.dtype)?;
            (stacked, Some(mask_tensor))
        };

        // 5. Forward through language model backbone and pool
        let token_embeds = self
            .model
            .forward_embeds(&batched_embeds, pad_mask.as_ref())?;
        let pooled =
            self.model
                .pool_and_normalize(&token_embeds, pad_mask.as_ref(), truncate_dim)?;
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
        let truncate_dim = self.validate_truncate_dim(truncate_dim)?;
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
            let pooled = self
                .model
                .pool_and_normalize(&token_embeds, None, truncate_dim)?;

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
        let truncate_dim = self.validate_truncate_dim(truncate_dim)?;
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
                let (ids, mask) = self.tokenizer.encode_batch_padded_with_limit(
                    texts,
                    self.max_seq_len,
                    &self.device,
                )?;
                (ids, Some(mask))
            };
            let tokenizer_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
            let seq_len = input_ids.dim(1)?;

            let (token_embeds, forward_profile) = self
                .model
                .forward_profiled(&input_ids, attention_mask.as_ref())?;

            let t_pool = std::time::Instant::now();
            let pooled = self.model.pool_and_normalize(
                &token_embeds,
                attention_mask.as_ref(),
                truncate_dim,
            )?;
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

fn splice_soft_tokens(
    text_embeds: &Tensor,
    seq_len: usize,
    slots: &[tokenizer::MediaSlot],
    soft_tensors: &[Tensor],
) -> Result<Tensor> {
    if slots.len() != soft_tensors.len() {
        anyhow::bail!(
            "Multimodal splicing slot count ({}) does not match encoded media tensor count ({})",
            slots.len(),
            soft_tensors.len()
        );
    }

    let mut chunks = Vec::new();
    let mut curr = 0;

    for (slot, soft_tensor) in slots.iter().zip(soft_tensors.iter()) {
        let actual_len = soft_tensor.dim(0)?;
        if actual_len != slot.length {
            anyhow::bail!(
                "Multimodal splicing length mismatch at token {}: slot expects {} soft tokens, got {}",
                slot.start_idx,
                slot.length,
                actual_len
            );
        }
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
    if chunks_refs.len() == 1 {
        Ok(chunks_refs[0].clone())
    } else {
        Ok(Tensor::cat(&chunks_refs, 1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn build_tiny_embedder() -> GlyprEmbedder {
        use candle_core::quantized::gguf_file::Value;
        use candle_nn::{Embedding, Linear};
        use config::{LayerType, RoPEParameters};

        let hidden_size = 8usize;
        let intermediate_size = 16usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 4usize;
        let embedding_dim = 16usize;
        let hidden_size_per_layer_input = 8usize;
        let vocab_size = 16usize;
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut rope_parameters = HashMap::new();
        rope_parameters.insert(
            "sliding_attention".to_string(),
            RoPEParameters {
                rope_theta: 10_000.0,
                rope_type: "default".to_string(),
            },
        );
        rope_parameters.insert(
            "full_attention".to_string(),
            RoPEParameters {
                rope_theta: 1_000_000.0,
                rope_type: "default".to_string(),
            },
        );

        let config = TextConfig {
            hidden_size,
            intermediate_size,
            num_hidden_layers: 1,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            vocab_size,
            embedding_dim,
            max_position_embeddings: 128,
            sliding_window: 512,
            rms_norm_eps: 1e-6,
            hidden_size_per_layer_input,
            layer_types: vec![LayerType::SlidingAttention],
            per_layer_config: HashMap::new(),
            rope_parameters,
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 1,
        };

        let ones = |shape: (usize, usize)| -> LinearOp {
            let w = Tensor::ones(shape, dtype, &device).unwrap();
            LinearOp::new_standard(Linear::new(w, None))
        };
        let norm_w = |len: usize| -> norm::RMSNorm {
            norm::RMSNorm::new(Some(Tensor::ones((len,), dtype, &device).unwrap()), 1e-6)
        };

        let embed_weight = Tensor::arange(0u32, (vocab_size * hidden_size) as u32, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
            .reshape((vocab_size, hidden_size))
            .unwrap();
        let embed_tokens = Embedding::new(embed_weight, hidden_size);

        let ple = ple::TextPLE::new(
            ones((hidden_size_per_layer_input, hidden_size)),
            (hidden_size as f64).powf(-0.5),
            norm_w(hidden_size_per_layer_input),
            1,
            hidden_size_per_layer_input,
        );

        let layer = layer::EncoderLayer {
            self_attn: attention::SelfAttention::new(
                ones((num_heads * head_dim, hidden_size)),
                ones((num_kv_heads * head_dim, hidden_size)),
                ones((num_kv_heads * head_dim, hidden_size)),
                ones((hidden_size, num_heads * head_dim)),
                norm_w(head_dim),
                norm_w(head_dim),
                norm::RMSNorm::new(None, 1e-6),
                num_heads,
                num_kv_heads,
                head_dim,
                Some(config.sliding_window),
            ),
            mlp: mlp::MLP::new(
                ones((intermediate_size, hidden_size)),
                ones((intermediate_size, hidden_size)),
                ones((hidden_size, intermediate_size)),
            ),
            input_layernorm: norm_w(hidden_size),
            post_attention_layernorm: norm_w(hidden_size),
            pre_feedforward_layernorm: norm_w(hidden_size),
            post_feedforward_layernorm: norm_w(hidden_size),
            ple_block: ple::TextPLEBlock::new(
                ones((hidden_size_per_layer_input, hidden_size)),
                ones((hidden_size, hidden_size_per_layer_input)),
                norm_w(hidden_size),
            ),
            layer_scalar: Tensor::ones((1,), dtype, &device).unwrap(),
        };

        let model = EmbeddingGemma2Model {
            config,
            embed_tokens,
            embed_scale: (hidden_size as f64).sqrt(),
            ple,
            layers: vec![layer],
            norm: norm_w(hidden_size),
            embedding_projection: ones((embedding_dim, hidden_size)),
            rope_sliding: rope::RotaryEmbedding::new(head_dim, 10_000.0, &device).unwrap(),
            rope_full: rope::RotaryEmbedding::new(head_dim, 1_000_000.0, &device).unwrap(),
            device: device.clone(),
            dtype,
        };

        let vocab = vec!["<unk>", "</s>", "<s>", "hello", "\u{2581}world"];
        let mut meta = HashMap::new();
        meta.insert(
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(
                vocab
                    .into_iter()
                    .map(|s| Value::String(s.to_string()))
                    .collect(),
            ),
        );
        meta.insert(
            "tokenizer.ggml.merges".to_string(),
            Value::Array(Vec::new()),
        );
        let tokenizer = GlyprTokenizer::from_gguf_metadata(&meta).unwrap();

        GlyprEmbedder {
            model,
            tokenizer,
            device,
            dtype,
            multimodal: None,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            #[cfg(feature = "coreml")]
            coreml_vision: None,
            vision_backend: VisionBackendPolicy::Auto,
        }
    }

    #[test]
    fn test_truncate_dim_validation_and_non_hardcoded_dims() {
        let embedder = build_tiny_embedder(); // hidden_size=8, embedding_dim=16

        assert_eq!(embedder.validate_truncate_dim(None).unwrap(), None);
        assert_eq!(embedder.validate_truncate_dim(Some(16)).unwrap(), None);
        assert_eq!(embedder.validate_truncate_dim(Some(8)).unwrap(), Some(8));
        assert!(embedder.validate_truncate_dim(Some(0)).is_err());
        assert!(embedder.validate_truncate_dim(Some(17)).is_err());
        assert!(embedder.validate_truncate_dim(Some(1024)).is_err());

        let inputs = [
            MultimodalInput::text("hello", Task::Raw),
            MultimodalInput::text("hello world", Task::Raw),
        ];

        // Invalid truncate_dim must fail cleanly (no chunks_exact(0) panic, no silent ignore)
        assert!(embedder.embed_batch(&inputs, Some(0)).is_err());
        assert!(embedder.embed_batch(&inputs, Some(1024)).is_err());
        assert!(embedder.encode_single_text("hello", Some(0)).is_err());
        assert!(embedder.encode_single_text("hello", Some(1024)).is_err());
        assert!(
            embedder
                .profile_encode_batch(&["hello", "hello world"], Some(0))
                .is_err()
        );
        assert!(
            embedder
                .profile_encode_batch(&["hello", "hello world"], Some(1024))
                .is_err()
        );

        // Heterogeneous batch with hidden_size=8 (not 512) and embedding_dim=16 (not 768)
        let full = embedder.embed_batch(&inputs, None).unwrap();
        assert_eq!(full.len(), 2);
        assert_eq!(full[0].len(), 16);
        assert_eq!(full[1].len(), 16);

        let exact_dim = embedder.embed_batch(&inputs, Some(16)).unwrap();
        assert_eq!(exact_dim[0].len(), 16);

        let trunc = embedder.embed_batch(&inputs, Some(8)).unwrap();
        assert_eq!(trunc.len(), 2);
        assert_eq!(trunc[0].len(), 8);
        assert_eq!(trunc[1].len(), 8);
    }

    #[test]
    fn test_embed_composite_rejects_empty_video_and_audio_preflight() {
        let mut embedder = build_tiny_embedder();
        embedder.multimodal = Some(MultimodalProjector {
            vision_tower: None,
            audio_tower: None,
            config: config::MultimodalConfig::from_gguf_metadata(&HashMap::new()),
            device: Device::Cpu,
            dtype: DType::F32,
        });

        let empty_video_err = embedder
            .embed_composite(
                None,
                &[MediaItem::VideoFrames {
                    frames: &[],
                    budget_per_frame: None,
                }],
                None,
            )
            .unwrap_err();
        assert!(empty_video_err.to_string().contains("empty"));

        let empty_audio_err = embedder
            .embed_composite(None, &[MediaItem::Audio { samples: &[] }], None)
            .unwrap_err();
        assert!(empty_audio_err.to_string().contains("empty"));
    }
}
