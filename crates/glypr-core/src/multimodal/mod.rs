pub mod audio;
pub mod audio_preprocessor;
pub mod image_preprocessor;
pub mod projector;
pub mod vision;

pub use audio::AudioTower;
pub use audio_preprocessor::{
    compute_audio_soft_token_count, compute_audio_total_tokens, decode_wav_bytes, decode_wav_file,
    decode_wav_reader, preprocess_audio, ProcessedAudio,
};
pub use image_preprocessor::{
    compute_image_total_tokens, compute_soft_token_count, preprocess_image, validate_vision_budget,
    ProcessedImage, VISION_BUDGET_DEFAULT, VISION_BUDGET_HIGH, VISION_BUDGET_LOW,
    VISION_BUDGET_MAX, VISION_BUDGET_MEDIUM, VISION_BUDGET_MIN, VISION_BUDGET_ULTRA,
};
pub use projector::MultimodalProjector;
pub use vision::VisionTower;

/// An individual part in an ordered multimodal sequence (matching Python sentence-transformers list format).
#[derive(Debug, Clone)]
pub enum ModalityPart<'a> {
    Text(&'a str),
    Image {
        image: &'a image::DynamicImage,
        budget: Option<usize>,
    },
    Audio {
        samples: &'a [f32],
    },
    VideoFrame {
        frame: &'a image::DynamicImage,
        budget: Option<usize>,
    },
}

/// Routing policy for multimodal vision execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionBackendPolicy {
    /// Prefer ANE via CoreML when a matching model is loaded for the image patches, fallback to GPU/CPU.
    #[default]
    Auto,
    /// Strictly require execution on Apple Neural Engine via CoreML. Fail if no matching model is loaded.
    AneStrict,
    /// Force execution on Candle Metal / CPU mmproj, bypassing CoreML even if models are present.
    Metal,
}


