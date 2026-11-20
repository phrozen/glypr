use std::path::PathBuf;
use thiserror::Error;

/// Convenience result alias for fallible `glypr-core` operations.
pub type GlyprResult<T> = std::result::Result<T, GlyprError>;

fn format_grids(grids: &[(usize, usize)]) -> String {
    grids
        .iter()
        .map(|(h, w)| format!("{h}x{w}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Structured error type for `glypr-core` operations.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum GlyprError {
    /// Input token sequence exceeds the embedder's configured `max_seq_len` or ANE static limit.
    #[error("Sequence length ({actual} tokens) exceeds maximum context limit ({max} tokens)")]
    ContextLimitExceeded {
        /// Actual sequence length in tokens.
        actual: usize,
        /// Configured or hardware maximum sequence length in tokens.
        max: usize,
    },

    /// Multimodal input was supplied without any attached GGUF mmproj or CoreML vision tower.
    #[error("Multimodal input requires an attached mmproj or CoreML vision tower")]
    MissingMultimodalEngine,

    /// Audio input was supplied without an attached GGUF multimodal projector (`mmproj`).
    #[error("Audio input requires an attached GGUF multimodal projector (mmproj)")]
    MissingAudioProjector,

    /// Vision soft-token budget is outside the supported `[16..=2048]` range.
    #[error("Vision budget {budget} out of supported range [{min}..={max}]")]
    InvalidVisionBudget {
        /// Requested soft-token budget.
        budget: usize,
        /// Minimum allowed budget.
        min: usize,
        /// Maximum allowed budget.
        max: usize,
    },

    /// No loaded CoreML vision model matches the image's patch count.
    #[error("Unsupported CoreML image patch count {actual}; loaded models support {supported:?}")]
    UnsupportedCoreMLPatchCount {
        /// Preprocessed patch count of the input image.
        actual: usize,
        /// Patch counts supported by currently loaded CoreML vision models.
        supported: Vec<usize>,
    },

    /// No loaded CoreML vision model matches the image's exact `(patch_height, patch_width)` grid.
    #[error(
        "Unsupported CoreML image patch grid {patch_height}x{patch_width} ({num_patches} patches); loaded grids: [{}]. Adjust --image-token-budget, pass --bf16-mmproj for GPU fallback, or export this aspect ratio with 'uv run python scripts/export_coreml.py --vision-preset all'.",
        format_grids(.supported_grids)
    )]
    UnsupportedCoreMLPatchGrid {
        /// Image grid height in patches.
        patch_height: usize,
        /// Image grid width in patches.
        patch_width: usize,
        /// Total patches (`patch_height * patch_width`).
        num_patches: usize,
        /// `(patch_height, patch_width)` grids supported by currently loaded CoreML vision models.
        supported_grids: Vec<(usize, usize)>,
    },

    /// Expected CoreML directory or `.mlmodelc` artifact was not found.
    #[error("CoreML directory or artifact not found at {path}")]
    CoreMLNotFound {
        /// Missing CoreML directory/bundle path.
        path: PathBuf,
    },

    /// Offline mode is active and the requested Hugging Face Hub file is not cached locally.
    #[error(
        "File '{filename}' from '{repo_id}' is not in local HF cache ({cache_dir}) and offline mode is enabled. \
         Run 'glypr pull' online or pass --model-file <PATH>."
    )]
    OfflineCacheMiss {
        /// Hugging Face repository ID.
        repo_id: String,
        /// Requested filename within the repository.
        filename: String,
        /// Local Hugging Face Hub cache directory that was searched.
        cache_dir: PathBuf,
    },

    /// Requested Matryoshka (MRL) truncation dimension is `0` or exceeds the model embedding dimension.
    #[error("Invalid MRL dimension {dim} (must be in 1..={max})")]
    InvalidMrlDimension {
        /// Requested output dimension.
        dim: usize,
        /// Maximum native embedding dimension of the model.
        max: usize,
    },

    /// Invalid or malformed caller input.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Failed to load or initialize model weights, configuration, or CoreML bundles.
    #[error("Model load error: {0}")]
    ModelLoad(String),

    /// Underlying filesystem or I/O failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Underlying Candle tensor or device execution failure.
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    /// Catch-all error propagated from lower-level helpers.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
