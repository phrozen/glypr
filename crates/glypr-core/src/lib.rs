//! # `glypr-core`: Native Rust Inference Engine for EmbeddingGemma 2
//!
//! `glypr-core` provides a self-contained, hardware-accelerated embedding engine supporting
//! text, images, audio, and interleaved multimodal inputs from a single `.gguf` container
//! (or Hugging Face Safetensors directory).
//!
//! ## Backends & Cargo Features
//! - **Default (`[]`)**: Portable CPU SIMD (`pulp` AVX2/AVX-512/NEON) with zero external runtime dependencies.
//! - **`metal`**: Apple Silicon Metal GPU (`BF16`) with compiled shader caching.
//! - **`metal-flash-attn`**: Hardware-fused Metal FlashAttention kernels (`mpp::tensor_ops` on M5, `simdgroup_matrix` on M1–M4).
//! - **`coreml` / `ane`**: Apple Neural Engine (ANE) acceleration via compiled CoreML `.mlmodelc` vision and text models.
//! - **`accelerate`**: Apple Accelerate BLAS backend for CPU execution.
//!
//! ## Quick Start
//! ```no_run
//! use glypr_core::{AttentionMode, GlyprEmbedder, GlyprResult, Task};
//!
//! # fn main() -> GlyprResult<()> {
//! let mut embedder = GlyprEmbedder::builder()
//!     .gguf_canonical(false)
//!     .attention_mode(AttentionMode::ChunkedSliding)
//!     .build()?;
//!
//! let query = embedder.embed_text("What causes gravitational lensing?", Task::SearchQuery)?;
//! let truncated = embedder.embed_text_with_dim(
//!     "What causes gravitational lensing?",
//!     Task::SearchQuery,
//!     Some(256),
//! )?;
//! assert_eq!(query.len(), 768);
//! assert_eq!(truncated.len(), 256);
//! # Ok(())
//! # }
//! ```

pub mod attention;
pub mod config;
#[cfg(feature = "coreml")]
pub mod coreml;
pub mod cpu_simd;
pub mod error;
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
#[doc(hidden)]
pub use candle_nn as _candle_nn;
#[cfg(feature = "coreml")]
pub use coreml::{CoreMLTextBackbone, CoreMLVisionTower, VisionGridInfo};
pub use error::{GlyprError, GlyprResult};
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

use anyhow::Context;
use candle_core::{DType, Device, Module, Tensor};
use config::TextConfig;
use std::path::{Path, PathBuf};
use tokenizer::GlyprTokenizer;

// 8,192 content tokens + 128 tokens headroom for task prefixes, document titles, and special tokens
pub const DEFAULT_MAX_SEQ_LEN: usize = 8192 + 128; // 8,320 tokens

/// Compute unit selection for CoreML / Apple Neural Engine execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMLComputeUnits {
    /// Route to CPU and Apple Neural Engine (recommended for ANE efficiency).
    #[default]
    CpuAndNeuralEngine,
    /// Allow CoreML to schedule across CPU, GPU, and Neural Engine.
    All,
    /// Restrict CoreML execution to CPU and GPU.
    CpuAndGpu,
    /// Restrict CoreML execution to CPU only.
    CpuOnly,
}

impl CoreMLComputeUnits {
    /// Canonical CLI/display string (`"ane"`, `"all"`, `"gpu"`, `"cpu"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CpuAndNeuralEngine => "ane",
            Self::All => "all",
            Self::CpuAndGpu => "gpu",
            Self::CpuOnly => "cpu",
        }
    }
}

impl std::fmt::Display for CoreMLComputeUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for CoreMLComputeUnits {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ane" | "cpu_and_ne" | "cpu_and_neural_engine" | "neural_engine" => {
                Ok(Self::CpuAndNeuralEngine)
            }
            "all" => Ok(Self::All),
            "gpu" | "cpu_and_gpu" => Ok(Self::CpuAndGpu),
            "cpu" | "cpu_only" => Ok(Self::CpuOnly),
            other => anyhow::bail!(
                "Invalid CoreML compute units '{other}'; expected one of: ane, all, gpu, cpu"
            ),
        }
    }
}

/// Routing policy for pure-text execution when a CoreML text backbone is attached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextBackendPolicy {
    /// Route pure text sequences in the ANE sweet spot (`64..=seq_len` tokens, `batch_size <= 4`)
    /// to CoreML ANE when attached; keep `<64` tokens, `>seq_len`, `batch_size > 4`, and multimodal on Metal/CPU.
    #[default]
    Auto,
    /// Require ANE CoreML text backbone; error if no CoreML text model is attached or `seq_len` exceeds limit.
    AneStrict,
    /// Force Metal/CPU text backbone even when a CoreML text model is attached.
    Metal,
}

impl TextBackendPolicy {
    /// Canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::AneStrict => "ane_strict",
            Self::Metal => "metal",
        }
    }
}

impl std::fmt::Display for TextBackendPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for TextBackendPolicy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "ane" | "ane_strict" | "strict" => Ok(Self::AneStrict),
            "metal" | "gpu" | "cpu" => Ok(Self::Metal),
            other => anyhow::bail!(
                "Invalid text backend policy '{other}'; expected one of: auto, ane, metal"
            ),
        }
    }
}

/// Summary of CoreML vision and text models attached from a directory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoreMLAttachReport {
    /// `(patch_height, patch_width)` grids loaded for ANE vision.
    pub vision_grids: Vec<(usize, usize)>,
    /// Static `seq_len` buckets loaded for ANE text.
    pub text_seq_lens: Vec<usize>,
    /// Non-fatal diagnostics if one modality was absent in the directory.
    pub warnings: Vec<String>,
}

pub struct GlyprEmbedder {
    model: EmbeddingGemma2Model,
    tokenizer: GlyprTokenizer,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
    multimodal: Option<MultimodalProjector>,
    max_seq_len: usize,
    #[cfg(feature = "coreml")]
    coreml_vision: Option<CoreMLVisionTower>,
    #[cfg(feature = "coreml")]
    coreml_text: Option<CoreMLTextBackbone>,
    vision_backend: VisionBackendPolicy,
    text_backend: TextBackendPolicy,
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

#[derive(Debug, Clone)]
enum BuilderModelSource {
    Canonical { use_q8: bool },
    GgufFile(PathBuf),
    SafetensorsDir(PathBuf),
}

#[derive(Debug, Clone)]
enum BuilderMmprojSource {
    Canonical { use_q8: bool },
    File(PathBuf),
}

/// Fluent builder for configuring and loading a [`GlyprEmbedder`].
///
/// Automatically detects Apple Metal availability when `device` is unset, picks
/// `DType::BF16` on Metal or `DType::F32` on CPU, and wires optional `mmproj` and
/// CoreML (ANE) vision + text models.
pub struct GlyprBuilder {
    model_source: BuilderModelSource,
    mmproj_source: Option<BuilderMmprojSource>,
    device: Option<Device>,
    dtype: Option<DType>,
    attention_mode: Option<AttentionMode>,
    vision_backend: VisionBackendPolicy,
    text_backend: TextBackendPolicy,
    coreml_dir: Option<PathBuf>,
    coreml_compute_units: CoreMLComputeUnits,
    max_seq_len: usize,
    offline: bool,
}

impl Default for GlyprBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GlyprBuilder {
    /// Create a new builder configured for canonical BF16 GGUF weights and auto device detection.
    pub fn new() -> Self {
        Self {
            model_source: BuilderModelSource::Canonical { use_q8: false },
            mmproj_source: None,
            device: None,
            dtype: None,
            attention_mode: None,
            vision_backend: VisionBackendPolicy::Auto,
            text_backend: TextBackendPolicy::Auto,
            coreml_dir: None,
            coreml_compute_units: CoreMLComputeUnits::CpuAndNeuralEngine,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            offline: false,
        }
    }

    /// Use canonical Hugging Face GGUF weights (`use_q8 = false` for BF16, `true` for Q8_0).
    pub fn gguf_canonical(mut self, use_q8: bool) -> Self {
        self.model_source = BuilderModelSource::Canonical { use_q8 };
        self
    }

    /// Load base model weights from a local `.gguf` file.
    pub fn gguf_file<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.model_source = BuilderModelSource::GgufFile(path.as_ref().to_path_buf());
        self
    }

    /// Load base model weights from a Hugging Face Safetensors directory (`config.json`, `tokenizer.json`, `model.safetensors`).
    pub fn safetensors_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.model_source = BuilderModelSource::SafetensorsDir(dir.as_ref().to_path_buf());
        self
    }

    /// Attach canonical Hugging Face GGUF multimodal projector (`use_q8 = false` for BF16, `true` for Q8_0).
    pub fn mmproj_canonical(mut self, use_q8: bool) -> Self {
        self.mmproj_source = Some(BuilderMmprojSource::Canonical { use_q8 });
        self
    }

    /// Attach a multimodal projector from a local `mmproj-*.gguf` file.
    pub fn mmproj_file<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.mmproj_source = Some(BuilderMmprojSource::File(path.as_ref().to_path_buf()));
        self
    }

    /// Override the compute device (defaults to auto-detecting safe Metal availability, else CPU).
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Override the tensor dtype (defaults to `BF16` on Metal and `F32` on CPU).
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Set the attention kernel mode (`Auto` picks `MetalFlash` on Metal or `ChunkedSliding` on CPU).
    pub fn attention_mode(mut self, mode: AttentionMode) -> Self {
        self.attention_mode = Some(mode);
        self
    }

    /// Set routing policy for vision execution (`Auto`, `AneStrict`, `Metal`).
    pub fn vision_backend(mut self, policy: VisionBackendPolicy) -> Self {
        self.vision_backend = policy;
        self
    }

    /// Set routing policy for pure-text ANE execution (`Auto`, `AneStrict`, `Metal`).
    pub fn text_backend(mut self, policy: TextBackendPolicy) -> Self {
        self.text_backend = policy;
        self
    }

    /// Attach compiled CoreML `.mlmodelc` vision and text models from `dir`.
    pub fn coreml_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.coreml_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Set CoreML compute units (defaults to `CpuAndNeuralEngine`).
    pub fn coreml_compute_units(mut self, units: CoreMLComputeUnits) -> Self {
        self.coreml_compute_units = units;
        self
    }

    /// Override maximum supported token context length (default `8,320`).
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Enable or disable strict offline Hub resolution (never hits network).
    pub fn offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    /// Construct and initialize the [`GlyprEmbedder`].
    pub fn build(self) -> GlyprResult<GlyprEmbedder> {
        let device = match self.device {
            Some(d) => d,
            None => {
                #[cfg(feature = "metal")]
                {
                    if metal_kernels::is_metal_device_available() {
                        Device::new_metal(0)?
                    } else {
                        Device::Cpu
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    Device::Cpu
                }
            }
        };

        let dtype = self
            .dtype
            .unwrap_or_else(|| GlyprEmbedder::default_dtype_for_device(&device));

        let mut embedder = match self.model_source {
            BuilderModelSource::Canonical { use_q8 } => {
                let path = hub::resolve_canonical_model_with_options(use_q8, self.offline)?;
                GlyprEmbedder::load_gguf(path, &device, dtype)?
            }
            BuilderModelSource::GgufFile(path) => GlyprEmbedder::load_gguf(path, &device, dtype)?,
            BuilderModelSource::SafetensorsDir(dir) => GlyprEmbedder::load(dir, &device, dtype)?,
        };

        embedder.set_max_seq_len(self.max_seq_len);
        embedder.set_vision_backend(self.vision_backend);
        embedder.set_text_backend(self.text_backend);

        if let Some(mode) = self.attention_mode {
            embedder.set_attention_mode(mode);
        }

        if let Some(mmproj) = self.mmproj_source {
            match mmproj {
                BuilderMmprojSource::Canonical { use_q8 } => {
                    let path = hub::resolve_canonical_mmproj_with_options(use_q8, self.offline)?;
                    embedder.attach_multimodal_projector(path)?;
                }
                BuilderMmprojSource::File(path) => {
                    embedder.attach_multimodal_projector(path)?;
                }
            }
        }

        if let Some(dir) = self.coreml_dir {
            #[cfg(feature = "coreml")]
            {
                let _ = embedder.attach_coreml_dir(&dir, self.coreml_compute_units)?;
            }
            #[cfg(not(feature = "coreml"))]
            {
                let _ = (dir, self.coreml_compute_units);
                return Err(GlyprError::InvalidInput(
                    "CoreML directory specified, but glypr-core was built without --features coreml / ane".to_string(),
                ));
            }
        }

        Ok(embedder)
    }
}

impl GlyprEmbedder {
    /// Create a fluent [`GlyprBuilder`] for configuring and loading a `GlyprEmbedder`.
    pub fn builder() -> GlyprBuilder {
        GlyprBuilder::new()
    }

    /// Construct a [`GlyprEmbedder`] from pre-built components (e.g., for custom loading or test harnesses).
    pub fn from_parts(
        model: EmbeddingGemma2Model,
        tokenizer: GlyprTokenizer,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            model,
            tokenizer,
            device,
            dtype,
            multimodal: None,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            #[cfg(feature = "coreml")]
            coreml_vision: None,
            #[cfg(feature = "coreml")]
            coreml_text: None,
            vision_backend: VisionBackendPolicy::Auto,
            text_backend: TextBackendPolicy::Auto,
        }
    }

    fn default_dtype_for_device(device: &Device) -> DType {
        if device.is_metal() {
            DType::BF16
        } else {
            DType::F32
        }
    }

    pub fn load<P: AsRef<Path>>(model_dir: P, device: &Device, dtype: DType) -> GlyprResult<Self> {
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        let weights_path = dir.join("model.safetensors");

        let config = TextConfig::from_json_file(&config_path)
            .with_context(|| format!("Failed to load config from {:?}", config_path))?;
        let tokenizer = GlyprTokenizer::from_file(&tokenizer_path)
            .with_context(|| format!("Failed to load tokenizer from {:?}", tokenizer_path))?;

        let file = std::fs::File::open(&weights_path)?;
        // SAFETY: Model weights are read-only during deserialization and not mutated concurrently.
        let weights_mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap weights from {:?}", weights_path))?;
        let safetensors = safetensors::SafeTensors::deserialize(&weights_mmap).map_err(|e| {
            GlyprError::ModelLoad(format!("Safetensors deserialization failed: {}", e))
        })?;

        let model = EmbeddingGemma2Model::from_safetensors(&safetensors, config, device, dtype)
            .map_err(|e| GlyprError::ModelLoad(format!("Model initialization failed: {}", e)))?;

        let embedder = Self::from_parts(model, tokenizer, device.clone(), dtype);
        embedder.warmup_metal()?;
        Ok(embedder)
    }

    /// Load a 100% self-contained GGUF file (extracts model weights, architecture config, and BPE tokenizer).
    /// Zero external file or HuggingFace snapshot dependencies!
    pub fn load_gguf<P: AsRef<Path>>(
        gguf_path: P,
        device: &Device,
        dtype: DType,
    ) -> GlyprResult<Self> {
        let file = std::fs::File::open(gguf_path.as_ref())?;
        let mut reader = std::io::BufReader::with_capacity(8 * 1024 * 1024, file);

        let content =
            candle_core::quantized::gguf_file::Content::read(&mut reader).with_context(|| {
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
            EmbeddingGemma2Model::from_gguf_content(&content, &mut reader, config, device, dtype)
                .map_err(|e| {
                GlyprError::ModelLoad(format!("GGUF Model initialization failed: {}", e))
            })?;

        let embedder = Self::from_parts(model, tokenizer, device.clone(), dtype);
        embedder.warmup_metal()?;
        Ok(embedder)
    }

    /// Load self-contained BF16 GGUF file (e.g., `embeddinggemma-2-BF16.gguf`)
    pub fn load_gguf_bf16<P: AsRef<Path>>(gguf_path: P, device: &Device) -> GlyprResult<Self> {
        Self::load_gguf(gguf_path, device, Self::default_dtype_for_device(device))
    }

    /// Load self-contained Q8_0 quantized GGUF file (e.g., `embeddinggemma-2-Q8_0.gguf`)
    pub fn load_gguf_q8<P: AsRef<Path>>(gguf_path: P, device: &Device) -> GlyprResult<Self> {
        Self::load_gguf(gguf_path, device, Self::default_dtype_for_device(device))
    }

    /// Read-only reference to the underlying Gemma 2 backbone model.
    pub fn model(&self) -> &EmbeddingGemma2Model {
        &self.model
    }

    /// Mutable reference to the underlying Gemma 2 backbone model.
    pub fn model_mut(&mut self) -> &mut EmbeddingGemma2Model {
        &mut self.model
    }

    /// Read-only reference to the reconstructed BPE tokenizer.
    pub fn tokenizer(&self) -> &GlyprTokenizer {
        &self.tokenizer
    }

    /// Active execution device (Metal GPU or CPU).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Active tensor dtype (`BF16` on Metal, `F32` on CPU).
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Attached GGUF multimodal projector, if any.
    pub fn multimodal(&self) -> Option<&MultimodalProjector> {
        self.multimodal.as_ref()
    }

    /// Mutable reference to the attached GGUF multimodal projector, if any.
    pub fn multimodal_mut(&mut self) -> Option<&mut MultimodalProjector> {
        self.multimodal.as_mut()
    }

    /// Set a pre-constructed multimodal projector on this embedder.
    pub fn set_multimodal_projector(&mut self, projector: MultimodalProjector) {
        self.multimodal = Some(projector);
    }

    /// Builder-style helper to attach a pre-constructed multimodal projector.
    pub fn with_multimodal_projector(mut self, projector: MultimodalProjector) -> Self {
        self.multimodal = Some(projector);
        self
    }

    /// Maximum supported input sequence length in tokens.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Native embedding dimensionality of the loaded model (typically `768`).
    pub fn embedding_dim(&self) -> usize {
        self.model.config.embedding_dim
    }

    /// Configured vision backend routing policy.
    pub fn vision_backend(&self) -> VisionBackendPolicy {
        self.vision_backend
    }

    /// Configured pure-text ANE routing policy.
    pub fn text_backend(&self) -> TextBackendPolicy {
        self.text_backend
    }

    #[cfg(feature = "coreml")]
    /// Attached CoreML Vision Tower, if any.
    pub fn coreml_vision(&self) -> Option<&CoreMLVisionTower> {
        self.coreml_vision.as_ref()
    }

    #[cfg(feature = "coreml")]
    /// Attached CoreML Text Backbone, if any.
    pub fn coreml_text(&self) -> Option<&CoreMLTextBackbone> {
        self.coreml_text.as_ref()
    }

    /// Dynamically attach an optional multimodal projector GGUF (`mmproj-*.gguf`)
    pub fn attach_multimodal_projector<P: AsRef<Path>>(
        &mut self,
        mmproj_path: P,
    ) -> GlyprResult<()> {
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
        compute_units: CoreMLComputeUnits,
    ) -> GlyprResult<()> {
        let tower = CoreMLVisionTower::from_dir(dir, compute_units)?;
        self.coreml_vision = Some(tower);
        Ok(())
    }

    #[cfg(feature = "coreml")]
    /// Dynamically attach a compiled CoreML Text Backbone (`.mlmodelc`) for ANE text acceleration.
    pub fn attach_coreml_text(&mut self, text_backbone: CoreMLTextBackbone) {
        self.coreml_text = Some(text_backbone);
    }

    #[cfg(feature = "coreml")]
    /// Dynamically attach CoreML Text Backbone models discovered in a directory (e.g. `models/coreml`).
    pub fn attach_coreml_text_dir<P: AsRef<Path>>(
        &mut self,
        dir: P,
        compute_units: CoreMLComputeUnits,
    ) -> GlyprResult<()> {
        let backbone = CoreMLTextBackbone::from_dir(dir, compute_units)?;
        self.coreml_text = Some(backbone);
        Ok(())
    }

    #[cfg(feature = "coreml")]
    /// Discover and attach both CoreML Vision and CoreML Text models found in `dir`.
    ///
    /// Succeeds as long as at least one modality (vision or text) loads cleanly from `dir`;
    /// fails only if neither vision nor text `.mlmodelc` bundles could be loaded.
    pub fn attach_coreml_dir<P: AsRef<Path>>(
        &mut self,
        dir: P,
        compute_units: CoreMLComputeUnits,
    ) -> GlyprResult<CoreMLAttachReport> {
        let dir_ref = dir.as_ref();
        if !dir_ref.is_dir() {
            return Err(GlyprError::CoreMLNotFound {
                path: dir_ref.to_path_buf(),
            });
        }

        let mut report = CoreMLAttachReport::default();

        match CoreMLVisionTower::from_dir(dir_ref, compute_units) {
            Ok(tower) => {
                report.vision_grids = tower.supported_grids();
                self.coreml_vision = Some(tower);
            }
            Err(e) => {
                report
                    .warnings
                    .push(format!("CoreML vision not loaded: {e}"));
            }
        }

        match CoreMLTextBackbone::from_dir(dir_ref, compute_units) {
            Ok(backbone) => {
                report.text_seq_lens = backbone.supported_seq_lens();
                self.coreml_text = Some(backbone);
            }
            Err(e) => {
                report.warnings.push(format!("CoreML text not loaded: {e}"));
            }
        }

        if report.vision_grids.is_empty() && report.text_seq_lens.is_empty() {
            return Err(GlyprError::ModelLoad(format!(
                "No usable CoreML vision or text models found in {}: {}",
                dir_ref.display(),
                report.warnings.join(" | ")
            )));
        }

        Ok(report)
    }

    #[cfg(feature = "coreml")]
    /// Pre-warm all attached CoreML vision and text models so the first inference request
    /// does not pay cold ANE graph compilation latency.
    pub fn warmup_coreml(&self) -> GlyprResult<()> {
        if let Some(cv) = &self.coreml_vision {
            cv.warmup()?;
        }
        if let Some(ct) = &self.coreml_text {
            ct.warmup()?;
        }
        Ok(())
    }

    /// Pre-populate RoPE cos/sin tables up to `max_position_embeddings.min(DEFAULT_MAX_SEQ_LEN)`
    /// and, on Metal devices, pre-compile hot Metal shader pipelines so the first request incurs
    /// zero cold-start compilation or RoPE table-construction jitter. No-op beyond RoPE on CPU.
    pub fn warmup_metal(&self) -> GlyprResult<()> {
        let precompute_len = self
            .model
            .config
            .max_position_embeddings
            .clamp(1, DEFAULT_MAX_SEQ_LEN);
        self.model
            .rope_sliding
            .precompute(precompute_len, &self.device, self.dtype)?;
        self.model
            .rope_full
            .precompute(precompute_len, &self.device, self.dtype)?;

        #[cfg(feature = "metal")]
        if let Device::Metal(mdev) = &self.device {
            crate::metal_kernels::warmup_pipelines(mdev.metal_device())?;
            crate::attention::MetalFlashAttention::warmup_pipelines(mdev.metal_device())?;
        }

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

    /// Set routing policy for pure-text execution (Auto, AneStrict, Metal).
    pub fn set_text_backend(&mut self, policy: TextBackendPolicy) {
        self.text_backend = policy;
    }

    pub fn with_text_backend(mut self, policy: TextBackendPolicy) -> Self {
        self.text_backend = policy;
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

    /// Whether a CoreML Text Backbone is attached.
    pub fn has_coreml_text(&self) -> bool {
        #[cfg(feature = "coreml")]
        {
            self.coreml_text.is_some()
        }
        #[cfg(not(feature = "coreml"))]
        {
            false
        }
    }

    /// Whether an audio projector is attached (`mmproj` audio tower).
    pub fn has_audio_projector(&self) -> bool {
        self.multimodal
            .as_ref()
            .is_some_and(|m| m.audio_tower.is_some())
    }

    /// Validate and normalize an optional Matryoshka (MRL) truncation dimension.
    ///
    /// Rejects `Some(0)` and `Some(d)` where `d > self.model.config.embedding_dim`,
    /// and normalizes `Some(embedding_dim)` to `None`.
    pub fn validate_truncate_dim(&self, truncate_dim: Option<usize>) -> GlyprResult<Option<usize>> {
        let embedding_dim = self.model.config.embedding_dim;
        match truncate_dim {
            None => Ok(None),
            Some(0) => Err(GlyprError::InvalidMrlDimension {
                dim: 0,
                max: embedding_dim,
            }),
            Some(d) if d > embedding_dim => Err(GlyprError::InvalidMrlDimension {
                dim: d,
                max: embedding_dim,
            }),
            Some(d) if d == embedding_dim => Ok(None),
            Some(d) => Ok(Some(d)),
        }
    }

    /// Encode a single image item using the routed vision engine (ANE CoreML or Metal/CPU GGUF).
    pub fn encode_image_part(
        &self,
        image: &image::DynamicImage,
        budget: usize,
    ) -> GlyprResult<Tensor> {
        multimodal::validate_vision_budget(budget)?;
        with_autorelease_pool(|| {
            let processed = crate::multimodal::preprocess_image(image, budget, 16, 3)?;

            #[cfg(feature = "coreml")]
            {
                let debug_routing = std::env::var_os("GLYPR_DEBUG").is_some();
                let use_coreml = match self.vision_backend {
                    VisionBackendPolicy::Auto => self.coreml_vision.as_ref().is_some_and(|cv| {
                        let matched = cv.supports_grid(processed.patch_height, processed.patch_width);
                        if debug_routing && !matched {
                            eprintln!(
                                "[glypr:ane] Vision grid {}x{} ({} patches, {}x{} px) missed CoreML buckets {:?}; falling back to Metal/CPU mmproj",
                                processed.patch_height,
                                processed.patch_width,
                                processed.num_patches,
                                processed.target_width,
                                processed.target_height,
                                cv.supported_grids()
                            );
                        }
                        matched
                    }),
                    VisionBackendPolicy::AneStrict => {
                        let cv = self
                            .coreml_vision
                            .as_ref()
                            .ok_or(GlyprError::MissingMultimodalEngine)?;
                        if !cv.supports_grid(processed.patch_height, processed.patch_width) {
                            return Err(GlyprError::UnsupportedCoreMLPatchGrid {
                                patch_height: processed.patch_height,
                                patch_width: processed.patch_width,
                                num_patches: processed.num_patches,
                                supported_grids: cv.supported_grids(),
                            });
                        }
                        true
                    }
                    VisionBackendPolicy::Metal => false,
                };

                if use_coreml && let Some(cv) = &self.coreml_vision {
                    if debug_routing {
                        eprintln!(
                            "[glypr:ane] Vision hit: {}x{} patches ({} tokens) -> CoreML ANE",
                            processed.patch_height,
                            processed.patch_width,
                            processed.num_soft_tokens
                        );
                    }
                    return cv.encode_to_tensor_dtype(&processed, &self.device, self.dtype);
                }
            }

            // Standard Candle / GGUF path
            let projector = self
                .multimodal
                .as_ref()
                .ok_or(GlyprError::MissingMultimodalEngine)?;
            let tower = projector
                .vision_tower
                .as_ref()
                .ok_or(GlyprError::MissingMultimodalEngine)?;
            Ok(tower.encode(&processed)?)
        })
    }

    /// Encode an arbitrary composite sequence: text + multiple images + audio + video frames
    pub fn embed_composite(
        &self,
        prompt: Option<&str>,
        items: &[MediaItem],
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<f32>> {
        let truncate_dim = self.validate_truncate_dim(truncate_dim)?;
        with_autorelease_pool(|| self.embed_composite_inner(prompt, items, truncate_dim))
    }

    fn embed_composite_inner(
        &self,
        prompt: Option<&str>,
        items: &[MediaItem],
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<f32>> {
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
            return Err(GlyprError::MissingMultimodalEngine);
        }

        // =========================================================================
        // PHASE 1: Zero-Overhead Preflight Check (< 10 µs, 0 neural forwards)
        // =========================================================================
        let mut placeholders = Vec::with_capacity(items.len());
        for item in items {
            match item {
                MediaItem::Image { image, budget } => {
                    let b = budget.unwrap_or(280);
                    multimodal::validate_vision_budget(b)?;
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
                        return Err(GlyprError::InvalidInput(
                            "Input audio cannot be empty".to_string(),
                        ));
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
                        return Err(GlyprError::InvalidInput(
                            "VideoFrames input cannot be empty".to_string(),
                        ));
                    }
                    let b = budget_per_frame.unwrap_or(140);
                    multimodal::validate_vision_budget(b)?;
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
            return Err(GlyprError::ContextLimitExceeded {
                actual: seq_len,
                max: self.max_seq_len,
            });
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
                    let projector = self
                        .multimodal
                        .as_ref()
                        .ok_or(GlyprError::MissingAudioProjector)?;
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
    pub fn embed_text(&self, text: &str, task: Task) -> GlyprResult<Vec<f32>> {
        self.embed_text_with_dim(text, task, None)
    }

    /// Encode a single text input formatted according to `task` with optional MRL dimension truncation.
    pub fn embed_text_with_dim(
        &self,
        text: &str,
        task: Task,
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<f32>> {
        let formatted = task.format_text(text);
        self.encode_single_text(&formatted, truncate_dim)
    }

    /// Encode a batch of texts formatted according to the specified Task.
    /// Returns native 768-dimensional L2-normalized embedding vectors.
    pub fn embed_texts(&self, texts: &[&str], task: Task) -> GlyprResult<Vec<Vec<f32>>> {
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
    pub fn embed_interleaved(&self, parts: &[ModalityPart], task: Task) -> GlyprResult<Vec<f32>> {
        self.embed_interleaved_with_dim(parts, task, None)
    }

    /// Encode an ordered sequence of multimodal parts with optional MRL dimension truncation.
    pub fn embed_interleaved_with_dim(
        &self,
        parts: &[ModalityPart],
        task: Task,
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<f32>> {
        let truncate_dim = self.validate_truncate_dim(truncate_dim)?;
        with_autorelease_pool(|| self.embed_interleaved_inner(parts, task, truncate_dim))
    }

    /// Attempt to route a pure-text token sequence to the attached `CoreMLTextBackbone` according to
    /// [`TextBackendPolicy`]. Returns `Ok(Some(vec))` if handled on ANE, or `Ok(None)` to fall
    /// through to Metal/CPU.
    fn try_route_pure_text_ane(
        &self,
        token_ids: &[u32],
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Option<Vec<f32>>> {
        #[cfg(not(feature = "coreml"))]
        {
            let _ = (token_ids, truncate_dim);
            if self.text_backend == TextBackendPolicy::AneStrict {
                return Err(GlyprError::InvalidInput(
                    "AneStrict text backend requested, but glypr-core was built without --features coreml / ane".to_string(),
                ));
            }
            Ok(None)
        }

        #[cfg(feature = "coreml")]
        {
            let debug_routing = std::env::var_os("GLYPR_DEBUG").is_some();
            match self.text_backend {
                TextBackendPolicy::Metal => Ok(None),
                TextBackendPolicy::AneStrict => {
                    let ct = self.coreml_text.as_ref().ok_or_else(|| {
                        GlyprError::InvalidInput(
                            "AneStrict text backend requested, but no CoreML text backbone is attached".to_string(),
                        )
                    })?;
                    let ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
                    if debug_routing {
                        eprintln!(
                            "[glypr:ane] Text strict route: {} tokens -> CoreML ANE (max {})",
                            token_ids.len(),
                            ct.seq_len()
                        );
                    }
                    Ok(Some(ct.embed_token_ids_with_dim(&ids_i32, truncate_dim)?))
                }
                TextBackendPolicy::Auto => {
                    let Some(ct) = &self.coreml_text else {
                        return Ok(None);
                    };
                    if (64..=ct.seq_len()).contains(&token_ids.len()) {
                        if debug_routing {
                            eprintln!(
                                "[glypr:ane] Text sweet-spot hit: {} tokens in [64..={}] -> CoreML ANE",
                                token_ids.len(),
                                ct.seq_len()
                            );
                        }
                        let ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
                        Ok(Some(ct.embed_token_ids_with_dim(&ids_i32, truncate_dim)?))
                    } else {
                        if debug_routing {
                            eprintln!(
                                "[glypr:ane] Text {} tokens outside ANE sweet spot [64..={}]; routing to Metal/CPU",
                                token_ids.len(),
                                ct.seq_len()
                            );
                        }
                        Ok(None)
                    }
                }
            }
        }
    }

    fn embed_interleaved_inner(
        &self,
        parts: &[ModalityPart],
        task: Task,
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<f32>> {
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
            return Err(GlyprError::MissingMultimodalEngine);
        }

        // Validate vision budgets and audio presence upfront before tokenization
        for part in parts {
            match part {
                ModalityPart::Image { budget, .. } => {
                    multimodal::validate_vision_budget(
                        budget.unwrap_or(multimodal::VISION_BUDGET_DEFAULT),
                    )?;
                }
                ModalityPart::VideoFrame { budget, .. } => {
                    multimodal::validate_vision_budget(
                        budget.unwrap_or(multimodal::VISION_BUDGET_MEDIUM),
                    )?;
                }
                ModalityPart::Audio { samples } => {
                    if samples.is_empty() {
                        return Err(GlyprError::InvalidInput(
                            "Input audio cannot be empty".to_string(),
                        ));
                    }
                    if self.multimodal.is_none() {
                        return Err(GlyprError::MissingAudioProjector);
                    }
                }
                ModalityPart::Text(_) => {}
            }
        }

        // 1. Tokenize sequence and validate context window preflight (< 10 µs, 0 neural forwards)
        let (token_ids, slots) = self.tokenizer.encode_modality_sequence(parts, task)?;
        let seq_len = token_ids.len();

        if seq_len > self.max_seq_len {
            return Err(GlyprError::ContextLimitExceeded {
                actual: seq_len,
                max: self.max_seq_len,
            });
        }

        // 1b. Pure-text ANE routing (when sequence contains no media parts)
        if !has_media
            && let Some(ane_vec) = self.try_route_pure_text_ane(&token_ids, truncate_dim)?
        {
            return Ok(ane_vec);
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
                    let projector = self
                        .multimodal
                        .as_ref()
                        .ok_or(GlyprError::MissingAudioProjector)?;
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
        let pooled = self
            .model
            .pool_and_normalize(&token_embeds, None, truncate_dim)?;
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
    ) -> GlyprResult<Vec<f32>> {
        self.embed_image_with_budget(image, prompt, multimodal::VISION_BUDGET_DEFAULT)
    }

    /// Encode an image with a specific token budget (e.g. 70, 140, 280, 560, 1120).
    /// Returns native 768-dimensional normalized embedding vector.
    pub fn embed_image_with_budget(
        &self,
        image: &image::DynamicImage,
        prompt: Option<&str>,
        budget: usize,
    ) -> GlyprResult<Vec<f32>> {
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
    pub fn embed_audio(&self, samples: &[f32], prompt: Option<&str>) -> GlyprResult<Vec<f32>> {
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
    ) -> GlyprResult<Vec<f32>> {
        let formatted = self.tokenizer.format_query(query, task);
        self.encode_single_text(&formatted, truncate_dim)
    }

    pub fn embed_document(
        &self,
        content: &str,
        title: Option<&str>,
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<f32>> {
        let formatted = self.tokenizer.format_document(content, title);
        self.encode_single_text(&formatted, truncate_dim)
    }

    pub fn embed_symmetric(
        &self,
        text: &str,
        task: &str,
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<f32>> {
        let formatted = self.tokenizer.format_query(text, task);
        self.encode_single_text(&formatted, truncate_dim)
    }

    pub fn embed_batch_queries(
        &self,
        queries: &[&str],
        task: &str,
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<Vec<f32>>> {
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
    ) -> GlyprResult<Vec<Vec<f32>>> {
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

    /// Embed pre-tokenized pure-text sequences in a single vectorized `[B, max_len]` forward pass
    /// (preserving `AneStrict` / `Auto` ANE routing when eligible).
    pub fn embed_tokenized_texts(
        &self,
        tokenized: &[Vec<u32>],
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<Vec<f32>>> {
        let truncate_dim = self.validate_truncate_dim(truncate_dim)?;
        if tokenized.is_empty() {
            return Ok(Vec::new());
        }
        with_autorelease_pool(|| self.embed_tokenized_texts_inner(tokenized, truncate_dim))
    }

    fn embed_tokenized_texts_inner(
        &self,
        tokenized: &[Vec<u32>],
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<Vec<f32>>> {
        let b = tokenized.len();
        if b == 0 {
            return Ok(Vec::new());
        }

        for (item_idx, ids) in tokenized.iter().enumerate() {
            if ids.is_empty() {
                return Err(GlyprError::InvalidInput(format!(
                    "Batch item {item_idx} has empty token sequence"
                )));
            }
            if ids.len() > self.max_seq_len {
                return Err(GlyprError::ContextLimitExceeded {
                    actual: ids.len(),
                    max: self.max_seq_len,
                });
            }
        }

        // Pure-text ANE routing check (isolated cleanly before Metal/CPU vectorized batching)
        #[cfg(not(feature = "coreml"))]
        if self.text_backend == TextBackendPolicy::AneStrict {
            return Err(GlyprError::InvalidInput(
                "AneStrict text backend requested, but glypr-core was built without --features coreml / ane".to_string(),
            ));
        }

        #[cfg(feature = "coreml")]
        if self.text_backend != TextBackendPolicy::Metal {
            let try_ane = match self.text_backend {
                TextBackendPolicy::AneStrict => true,
                TextBackendPolicy::Auto => self.coreml_text.is_some() && b <= 4,
                TextBackendPolicy::Metal => false,
            };
            if try_ane {
                let ct = self.coreml_text.as_ref().ok_or_else(|| {
                    GlyprError::InvalidInput(
                        "AneStrict text backend requested, but no CoreML text backbone is attached"
                            .to_string(),
                    )
                })?;

                let all_fit_ane = match self.text_backend {
                    TextBackendPolicy::AneStrict => true,
                    TextBackendPolicy::Auto => tokenized
                        .iter()
                        .all(|ids| (64..=ct.seq_len()).contains(&ids.len())),
                    TextBackendPolicy::Metal => false,
                };

                if all_fit_ane {
                    let mut results = Vec::with_capacity(b);
                    for ids in tokenized {
                        let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
                        results.push(ct.embed_token_ids_with_dim(&ids_i32, truncate_dim)?);
                    }
                    return Ok(results);
                }
            }
        }

        let mut min_len = usize::MAX;
        let mut max_len = 0usize;
        for ids in tokenized {
            min_len = min_len.min(ids.len());
            max_len = max_len.max(ids.len());
        }

        let pad_id = self.model.config.pad_token_id as u32;
        let (input_ids, pad_mask) = if min_len == max_len {
            let mut flat_ids = Vec::with_capacity(b * max_len);
            for ids in tokenized {
                flat_ids.extend_from_slice(ids);
            }
            let ids_tensor = Tensor::from_vec(flat_ids, (b, max_len), &self.device)?;
            (ids_tensor, None)
        } else {
            let mut flat_ids = vec![pad_id; b * max_len];
            let mut mask_vec = vec![0.0f32; b * max_len];
            for (row_idx, ids) in tokenized.iter().enumerate() {
                let start = row_idx * max_len;
                flat_ids[start..start + ids.len()].copy_from_slice(ids);
                mask_vec[start..start + ids.len()].fill(1.0);
            }
            let ids_tensor = Tensor::from_vec(flat_ids, (b, max_len), &self.device)?;
            let mask_tensor = Tensor::from_vec(mask_vec, (b, max_len), &self.device)?;
            (ids_tensor, Some(mask_tensor))
        };

        let token_embeds = self.model.forward(&input_ids, pad_mask.as_ref())?;
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

    /// Unified batch embedding across arbitrary modalities (text, images, audio, interleaved).
    /// Dynamically optimizes homogeneous sequences into unpadded zero-mask passes,
    /// and safely vectorizes heterogeneous sequences with padding attention masks.
    pub fn embed_batch(
        &self,
        inputs: &[MultimodalInput],
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<Vec<f32>>> {
        let truncate_dim = self.validate_truncate_dim(truncate_dim)?;
        let b = inputs.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        if b == 1 {
            let single =
                self.embed_interleaved_with_dim(&inputs[0].parts, inputs[0].task, truncate_dim)?;
            return Ok(vec![single]);
        }

        with_autorelease_pool(|| self.embed_batch_inner(inputs, truncate_dim))
    }

    fn embed_batch_inner(
        &self,
        inputs: &[MultimodalInput],
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<Vec<f32>>> {
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
            return Err(GlyprError::MissingMultimodalEngine);
        }

        // 1b. Pure-text fast path: parallel Rayon tokenization + vectorized [B, max_len] forward
        if !has_any_media {
            use rayon::prelude::*;
            let tokenized: Vec<Vec<u32>> = inputs
                .par_iter()
                .map(|inp| {
                    let (ids, _slots) = self
                        .tokenizer
                        .encode_modality_sequence(&inp.parts, inp.task)?;
                    Ok(ids)
                })
                .collect::<GlyprResult<Vec<_>>>()?;
            return self.embed_tokenized_texts_inner(&tokenized, truncate_dim);
        }

        // 2. Pre-extract and batch-encode all images across all items
        let mut image_requests = Vec::new();
        for (item_idx, inp) in inputs.iter().enumerate() {
            for (part_idx, part) in inp.parts.iter().enumerate() {
                if let ModalityPart::Image { image, budget } = part {
                    let b = budget.unwrap_or(multimodal::VISION_BUDGET_DEFAULT);
                    multimodal::validate_vision_budget(b)?;
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
                    .ok_or(GlyprError::MissingMultimodalEngine)?;
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

        for inp in inputs.iter() {
            let (token_ids, slots) = self
                .tokenizer
                .encode_modality_sequence(&inp.parts, inp.task)?;
            let seq_len = token_ids.len();
            if seq_len > self.max_seq_len {
                return Err(GlyprError::ContextLimitExceeded {
                    actual: seq_len,
                    max: self.max_seq_len,
                });
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
                        let projector = self
                            .multimodal
                            .as_ref()
                            .ok_or(GlyprError::MissingAudioProjector)?;
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
    ) -> GlyprResult<Vec<Vec<f32>>> {
        let inputs: Vec<MultimodalInput> = texts
            .iter()
            .map(|t| MultimodalInput::text(t, Task::Raw))
            .collect();
        self.embed_batch(&inputs, truncate_dim)
    }

    fn encode_single_text(&self, text: &str, truncate_dim: Option<usize>) -> GlyprResult<Vec<f32>> {
        let truncate_dim = self.validate_truncate_dim(truncate_dim)?;
        with_autorelease_pool(|| {
            let token_ids = self.tokenizer.encode(text)?;
            let seq_len = token_ids.len();
            if seq_len > self.max_seq_len {
                return Err(GlyprError::ContextLimitExceeded {
                    actual: seq_len,
                    max: self.max_seq_len,
                });
            }

            if let Some(ane_vec) = self.try_route_pure_text_ane(&token_ids, truncate_dim)? {
                return Ok(ane_vec);
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
    ) -> GlyprResult<(Vec<Vec<f32>>, PipelineProfile)> {
        let truncate_dim = self.validate_truncate_dim(truncate_dim)?;
        with_autorelease_pool(|| {
            let t_total = std::time::Instant::now();
            let batch_size = texts.len();

            let t_tok = std::time::Instant::now();
            let (input_ids, attention_mask) = if batch_size == 1 {
                let ids = self.tokenizer.encode(texts[0])?;
                let seq_len = ids.len();
                if seq_len > self.max_seq_len {
                    return Err(GlyprError::ContextLimitExceeded {
                        actual: seq_len,
                        max: self.max_seq_len,
                    });
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
) -> GlyprResult<Tensor> {
    if slots.len() != soft_tensors.len() {
        return Err(GlyprError::InvalidInput(format!(
            "Multimodal splicing slot count ({}) does not match encoded media tensor count ({})",
            slots.len(),
            soft_tensors.len()
        )));
    }

    let mut chunks = Vec::new();
    let mut curr = 0;

    for (slot, soft_tensor) in slots.iter().zip(soft_tensors.iter()) {
        let actual_len = soft_tensor.dim(0)?;
        if actual_len != slot.length {
            return Err(GlyprError::InvalidInput(format!(
                "Multimodal splicing length mismatch at token {}: slot expects {} soft tokens, got {}",
                slot.start_idx, slot.length, actual_len
            )));
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

        GlyprEmbedder::from_parts(model, tokenizer, device, dtype)
    }

    #[test]
    fn test_backend_policy_and_compute_units_parsing() {
        assert_eq!(
            "auto".parse::<TextBackendPolicy>().unwrap(),
            TextBackendPolicy::Auto
        );
        assert_eq!(
            "ane".parse::<TextBackendPolicy>().unwrap(),
            TextBackendPolicy::AneStrict
        );
        assert_eq!(
            "metal".parse::<TextBackendPolicy>().unwrap(),
            TextBackendPolicy::Metal
        );
        assert!("unknown".parse::<TextBackendPolicy>().is_err());

        assert_eq!(
            "ane".parse::<CoreMLComputeUnits>().unwrap(),
            CoreMLComputeUnits::CpuAndNeuralEngine
        );
        assert_eq!(
            "all".parse::<CoreMLComputeUnits>().unwrap(),
            CoreMLComputeUnits::All
        );
        assert_eq!(
            "gpu".parse::<CoreMLComputeUnits>().unwrap(),
            CoreMLComputeUnits::CpuAndGpu
        );
        assert_eq!(
            "cpu".parse::<CoreMLComputeUnits>().unwrap(),
            CoreMLComputeUnits::CpuOnly
        );
        assert!("tpu".parse::<CoreMLComputeUnits>().is_err());
    }

    #[test]
    fn test_accessors_and_ane_strict_text_rejection() {
        let mut embedder = build_tiny_embedder();
        assert_eq!(embedder.embedding_dim(), 16);
        assert_eq!(embedder.max_seq_len(), DEFAULT_MAX_SEQ_LEN);
        assert!(!embedder.has_coreml_text());
        assert!(!embedder.has_audio_projector());
        assert_eq!(embedder.text_backend(), TextBackendPolicy::Auto);

        let v = embedder
            .embed_text_with_dim("hello", Task::Raw, Some(8))
            .unwrap();
        assert_eq!(v.len(), 8);

        let v_interleaved = embedder
            .embed_interleaved_with_dim(&[ModalityPart::Text("hello")], Task::Raw, Some(8))
            .unwrap();
        assert_eq!(v_interleaved.len(), 8);

        embedder.set_text_backend(TextBackendPolicy::AneStrict);
        assert!(matches!(
            embedder.embed_text("hello", Task::Raw).unwrap_err(),
            GlyprError::InvalidInput(_)
        ));
        assert!(matches!(
            embedder
                .embed_interleaved(&[ModalityPart::Text("hello")], Task::Raw)
                .unwrap_err(),
            GlyprError::InvalidInput(_)
        ));
        assert!(matches!(
            embedder
                .embed_batch(
                    &[
                        MultimodalInput::text("hello", Task::Raw),
                        MultimodalInput::text("hello world", Task::Raw),
                    ],
                    None
                )
                .unwrap_err(),
            GlyprError::InvalidInput(_)
        ));
    }

    #[test]
    fn test_truncate_dim_validation_and_non_hardcoded_dims() {
        let embedder = build_tiny_embedder(); // hidden_size=8, embedding_dim=16

        assert_eq!(embedder.validate_truncate_dim(None).unwrap(), None);
        assert_eq!(embedder.validate_truncate_dim(Some(16)).unwrap(), None);
        assert_eq!(embedder.validate_truncate_dim(Some(8)).unwrap(), Some(8));
        assert!(matches!(
            embedder.validate_truncate_dim(Some(0)).unwrap_err(),
            GlyprError::InvalidMrlDimension { dim: 0, max: 16 }
        ));
        assert!(matches!(
            embedder.validate_truncate_dim(Some(17)).unwrap_err(),
            GlyprError::InvalidMrlDimension { dim: 17, max: 16 }
        ));
        assert!(matches!(
            embedder.validate_truncate_dim(Some(1024)).unwrap_err(),
            GlyprError::InvalidMrlDimension { dim: 1024, max: 16 }
        ));

        let inputs = [
            MultimodalInput::text("hello", Task::Raw),
            MultimodalInput::text("hello world", Task::Raw),
        ];

        // Invalid truncate_dim must fail cleanly (no chunks_exact(0) panic, no silent ignore)
        assert!(matches!(
            embedder.embed_batch(&inputs, Some(0)).unwrap_err(),
            GlyprError::InvalidMrlDimension { dim: 0, max: 16 }
        ));
        assert!(matches!(
            embedder.embed_batch(&inputs, Some(1024)).unwrap_err(),
            GlyprError::InvalidMrlDimension { dim: 1024, max: 16 }
        ));
        assert!(matches!(
            embedder.encode_single_text("hello", Some(0)).unwrap_err(),
            GlyprError::InvalidMrlDimension { dim: 0, max: 16 }
        ));
        assert!(matches!(
            embedder
                .encode_single_text("hello", Some(1024))
                .unwrap_err(),
            GlyprError::InvalidMrlDimension { dim: 1024, max: 16 }
        ));
        assert!(matches!(
            embedder
                .profile_encode_batch(&["hello", "hello world"], Some(0))
                .unwrap_err(),
            GlyprError::InvalidMrlDimension { dim: 0, max: 16 }
        ));
        assert!(matches!(
            embedder
                .profile_encode_batch(&["hello", "hello world"], Some(1024))
                .unwrap_err(),
            GlyprError::InvalidMrlDimension { dim: 1024, max: 16 }
        ));

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
        let embedder = build_tiny_embedder().with_multimodal_projector(MultimodalProjector {
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
        assert!(matches!(empty_video_err, GlyprError::InvalidInput(_)));
        assert!(empty_video_err.to_string().contains("empty"));

        let empty_audio_err = embedder
            .embed_composite(None, &[MediaItem::Audio { samples: &[] }], None)
            .unwrap_err();
        assert!(matches!(empty_audio_err, GlyprError::InvalidInput(_)));
        assert!(empty_audio_err.to_string().contains("empty"));
    }

    #[test]
    fn test_embed_tokenized_texts_and_warmup_metal() {
        let embedder = build_tiny_embedder();
        embedder.warmup_metal().unwrap();

        // Empty slice -> empty Vec
        assert!(
            embedder
                .embed_tokenized_texts(&[], None)
                .unwrap()
                .is_empty()
        );

        let texts = ["hello", "hello world", "world"];
        let tokenized: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| embedder.tokenizer().encode(t).unwrap())
            .collect();

        let from_tokens = embedder.embed_tokenized_texts(&tokenized, None).unwrap();
        let from_texts = embedder.encode_batch_texts(&texts, None).unwrap();
        assert_eq!(from_tokens.len(), from_texts.len());
        for (a, b) in from_tokens.iter().zip(from_texts.iter()) {
            let max_diff = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 1e-5, "tokenized batch mismatch: {max_diff}");
        }

        // Single-item tokenized batch + MRL truncation
        let single_tok = embedder
            .embed_tokenized_texts(&tokenized[1..2], Some(8))
            .unwrap();
        assert_eq!(single_tok.len(), 1);
        assert_eq!(single_tok[0].len(), 8);
    }
}
