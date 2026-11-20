use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use glypr_core::{
    GlyprEmbedder, GlyprError, MultimodalInput, TextBackendPolicy, truncate_mrl,
    validate_vision_budget, with_autorelease_pool,
};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

pub use glypr_core::OwnedTask;

use crate::schema::InputItem;

/// Structured inference/engine error separating client input errors (`400`) from internal failures (`500`).
#[derive(Debug, Clone, thiserror::Error)]
pub enum EngineError {
    /// Client supplied invalid input, exceeded limits, or failed preflight checks (`400 Bad Request`).
    #[error("{0}")]
    InvalidInput(String),
    /// Internal runtime, worker, or device failure (`500 Internal Server Error`).
    #[error("{0}")]
    Internal(String),
}

fn classify_core_error(err: &anyhow::Error) -> EngineError {
    if let Some(ge) = err.downcast_ref::<GlyprError>() {
        return match ge {
            GlyprError::ContextLimitExceeded { .. }
            | GlyprError::MissingMultimodalEngine
            | GlyprError::MissingAudioProjector
            | GlyprError::InvalidVisionBudget { .. }
            | GlyprError::UnsupportedCoreMLPatchCount { .. }
            | GlyprError::UnsupportedCoreMLPatchGrid { .. }
            | GlyprError::InvalidMrlDimension { .. }
            | GlyprError::InvalidInput(_) => EngineError::InvalidInput(err.to_string()),
            GlyprError::CoreMLNotFound { .. }
            | GlyprError::OfflineCacheMiss { .. }
            | GlyprError::Io(_)
            | GlyprError::Candle(_)
            | GlyprError::Other(_) => EngineError::Internal(err.to_string()),
        };
    }

    let msg = err.to_string();
    let lower = msg.to_ascii_lowercase();
    if lower.contains("exceeds maximum context")
        || lower.contains("exceeds coreml")
        || lower.contains("exceeds model embedding_dim")
        || lower.contains("cannot be empty")
        || lower.contains("cannot embed an empty")
        || lower.contains("cannot be 0")
        || lower.contains("not attached")
        || lower.contains("requires attached")
        || lower.contains("placeholder")
        || lower.contains("vision token budget")
        || lower.contains("anestrict")
        || lower.contains("no coreml vision model loaded")
    {
        EngineError::InvalidInput(msg)
    } else {
        EngineError::Internal(msg)
    }
}

/// An item preprocessed and decoded on CPU (base64 decoded, image opened/decompressed, audio parsed).
#[derive(Clone)]
pub enum PreparedItem {
    Text(String),
    Image {
        image: Arc<image::DynamicImage>,
        text: Option<String>,
        budget: Option<usize>,
    },
    Audio {
        samples: Arc<Vec<f32>>,
        text: Option<String>,
    },
}

/// Decode and prepare an input item on a blocking CPU thread before submitting to the GPU worker queue.
///
/// When `allow_local_files` is `false`, filesystem path probing (`Path::exists`) is completely skipped
/// so HTTP callers can neither read arbitrary files nor probe local path existence.
pub fn prepare_input_item(
    item: InputItem,
    allow_local_files: bool,
) -> std::result::Result<PreparedItem, EngineError> {
    match item {
        InputItem::Text(s) => Ok(PreparedItem::Text(s)),
        InputItem::Media(media) => {
            if media.image.is_some() && media.audio.is_some() {
                return Err(EngineError::InvalidInput(
                    "Media input item cannot specify both 'image' and 'audio' simultaneously"
                        .to_string(),
                ));
            }
            if let Some(b) = media.budget {
                validate_vision_budget(b).map_err(|e| EngineError::InvalidInput(e.to_string()))?;
            }
            if let Some(ref img_src) = media.image {
                let img = decode_image(img_src, allow_local_files)?;
                Ok(PreparedItem::Image {
                    image: Arc::new(img),
                    text: media.text,
                    budget: media.budget,
                })
            } else if let Some(ref aud_src) = media.audio {
                let samples = decode_audio(aud_src, allow_local_files)?;
                Ok(PreparedItem::Audio {
                    samples: Arc::new(samples),
                    text: media.text,
                })
            } else if let Some(text) = media.text {
                Ok(PreparedItem::Text(text))
            } else {
                Err(EngineError::InvalidInput(
                    "Media input item has no text, image, or audio specified".to_string(),
                ))
            }
        }
    }
}

/// An individual item queued for inference.
pub struct InferenceRequest {
    pub item: PreparedItem,
    pub task: OwnedTask,
    pub truncate_dim: Option<usize>,
    pub sender: oneshot::Sender<std::result::Result<(Vec<f32>, usize), EngineError>>,
}

/// Metadata describing the loaded model and runtime capabilities.
#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub id: String,
    pub device: String,
    pub has_multimodal: bool,
    pub has_audio_projector: bool,
    pub vision_backend: String,
    pub text_backend: String,
    pub ane_enabled: bool,
    pub ane_supported_patches: Vec<usize>,
    pub ane_text_enabled: bool,
    pub ane_text_seq_len: Option<usize>,
    pub embedding_dim: usize,
    pub max_seq_len: usize,
    pub created_at: u64,
}

/// Cloneable handle to submit inference requests to the worker loop.
#[derive(Clone)]
pub struct EngineHandle {
    sender: mpsc::Sender<InferenceRequest>,
    pub meta: Arc<ModelMeta>,
    pub allow_local_files: bool,
    pub max_batch_items: usize,
    pub request_timeout_secs: u64,
}

impl EngineHandle {
    /// Returns `true` when the background inference worker channel is open and ready to accept requests.
    pub fn is_ready(&self) -> bool {
        !self.sender.is_closed()
    }

    /// Submit a single request and await its embedding result and prompt token count.
    pub async fn infer(
        &self,
        item: PreparedItem,
        task: OwnedTask,
        truncate_dim: Option<usize>,
    ) -> std::result::Result<(Vec<f32>, usize), EngineError> {
        let (tx, rx) = oneshot::channel();
        let req = InferenceRequest {
            item,
            task,
            truncate_dim,
            sender: tx,
        };

        self.sender
            .send(req)
            .await
            .map_err(|_| EngineError::Internal("Inference worker is not running".to_string()))?;

        rx.await
            .map_err(|_| EngineError::Internal("Inference worker dropped request".to_string()))?
    }
}

/// Spawn the background inference engine on a dedicated OS thread with default server limits.
pub fn spawn_inference_engine(
    embedder: GlyprEmbedder,
    max_batch_size: usize,
    model_id: String,
) -> EngineHandle {
    let config = crate::ServerConfig {
        max_batch_size,
        model_id,
        ..Default::default()
    };
    spawn_inference_engine_with_config(embedder, &config)
}

/// Spawn the background inference engine on a dedicated OS thread configured from [`crate::ServerConfig`].
pub fn spawn_inference_engine_with_config(
    embedder: GlyprEmbedder,
    config: &crate::ServerConfig,
) -> EngineHandle {
    let (tx, mut rx) = mpsc::channel::<InferenceRequest>(1024);

    let device_name = match &embedder.device {
        candle_core::Device::Metal(_) => "metal".to_string(),
        candle_core::Device::Cpu => "cpu".to_string(),
        candle_core::Device::Cuda(_) => "cuda".to_string(),
    };

    let (ane_enabled, ane_patches) = {
        #[cfg(any(feature = "coreml", feature = "ane"))]
        {
            if let Some(cv) = &embedder.coreml_vision {
                (true, cv.supported_patch_counts())
            } else {
                (false, Vec::new())
            }
        }
        #[cfg(not(any(feature = "coreml", feature = "ane")))]
        {
            (false, Vec::new())
        }
    };

    let (ane_text_enabled, ane_text_seq_len) = {
        #[cfg(any(feature = "coreml", feature = "ane"))]
        {
            if let Some(ct) = &embedder.coreml_text {
                (true, Some(ct.max_seq_len()))
            } else {
                (false, None)
            }
        }
        #[cfg(not(any(feature = "coreml", feature = "ane")))]
        {
            (false, None)
        }
    };

    let vision_backend = match embedder.vision_backend {
        glypr_core::VisionBackendPolicy::Auto => {
            if ane_enabled && embedder.multimodal.is_some() {
                "hybrid (ane + metal)"
            } else if ane_enabled {
                "ane"
            } else if embedder.multimodal.is_some() {
                "metal"
            } else {
                "none"
            }
        }
        glypr_core::VisionBackendPolicy::AneStrict => "ane_strict",
        glypr_core::VisionBackendPolicy::Metal => "metal",
    }
    .to_string();

    let text_backend = match embedder.text_backend {
        TextBackendPolicy::Auto => {
            if ane_text_enabled {
                "hybrid (ane + metal)"
            } else {
                device_name.as_str()
            }
        }
        TextBackendPolicy::AneStrict => "ane_strict",
        TextBackendPolicy::Metal => device_name.as_str(),
    }
    .to_string();

    let has_multimodal = embedder.has_multimodal();
    let has_audio_projector = embedder.has_audio_projector();
    let embedding_dim = embedder.embedding_dim();
    let max_seq_len = embedder.max_seq_len();

    let meta = Arc::new(ModelMeta {
        id: config.model_id.clone(),
        device: device_name,
        has_multimodal,
        has_audio_projector,
        vision_backend,
        text_backend,
        ane_enabled,
        ane_supported_patches: ane_patches,
        ane_text_enabled,
        ane_text_seq_len,
        embedding_dim,
        max_seq_len,
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    });

    let max_batch = config.max_batch_size.max(1);

    std::thread::Builder::new()
        .name("glypr-infer-worker".into())
        .spawn(move || {
            tracing::info!("Inference worker started with max_batch_size={}", max_batch);

            while let Some(first_req) = rx.blocking_recv() {
                let mut batch = Vec::with_capacity(max_batch);
                batch.push(first_req);

                // Drain pending requests up to max_batch without blocking
                while batch.len() < max_batch {
                    match rx.try_recv() {
                        Ok(req) => batch.push(req),
                        Err(_) => break,
                    }
                }

                // If under capacity, yield briefly to coalesce concurrently arriving requests
                if batch.len() < max_batch {
                    std::thread::yield_now();
                    while batch.len() < max_batch {
                        match rx.try_recv() {
                            Ok(req) => batch.push(req),
                            Err(_) => break,
                        }
                    }
                }

                let mut work = Vec::with_capacity(batch.len());
                let mut senders = Vec::with_capacity(batch.len());
                for req in batch {
                    work.push((req.item, req.task, req.truncate_dim));
                    senders.push(req.sender);
                }

                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    with_autorelease_pool(|| process_batch_results(&embedder, &work))
                }));

                match outcome {
                    Ok(results) => {
                        for (sender, res) in senders.into_iter().zip(results) {
                            let _ = sender.send(res);
                        }
                    }
                    Err(panic_payload) => {
                        let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                            (*s).to_string()
                        } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "unknown panic".to_string()
                        };
                        tracing::error!("Inference worker caught panic in batch: {}", panic_msg);
                        for sender in senders {
                            let _ = sender.send(Err(EngineError::Internal(format!(
                                "Inference worker internal panic: {}",
                                panic_msg
                            ))));
                        }
                    }
                }
            }

            tracing::info!("Inference worker shut down");
        })
        .expect("Failed to spawn inference worker thread");

    EngineHandle {
        sender: tx,
        meta,
        allow_local_files: config.allow_local_files,
        max_batch_items: config.max_batch_items.max(1),
        request_timeout_secs: config.request_timeout_secs.max(1),
    }
}

fn build_multimodal_input<'a>(item: &'a PreparedItem, task: &'a OwnedTask) -> MultimodalInput<'a> {
    let t = task.as_task();
    match item {
        PreparedItem::Text(s) => MultimodalInput::text(s.as_str(), t),
        PreparedItem::Image {
            image,
            text,
            budget,
        } => {
            if let Some(prompt) = text {
                MultimodalInput::image_with_prompt(image.as_ref(), prompt.as_str(), *budget, t)
            } else {
                MultimodalInput::image(image.as_ref(), *budget, t)
            }
        }
        PreparedItem::Audio { samples, text } => {
            if let Some(prompt) = text {
                MultimodalInput::audio_with_prompt(samples.as_slice(), prompt.as_str(), t)
            } else {
                MultimodalInput::audio(samples.as_slice(), t)
            }
        }
    }
}

/// Per-item inference output `(embedding_vector, prompt_token_count)` or [`EngineError`].
pub type InferResult = std::result::Result<(Vec<f32>, usize), EngineError>;

/// Process a collected batch of inference requests with per-item preflight validation
/// and per-item fallback isolation if a batched forward fails.
fn process_batch_results(
    embedder: &GlyprEmbedder,
    work: &[(PreparedItem, OwnedTask, Option<usize>)],
) -> Vec<InferResult> {
    let inputs: Vec<MultimodalInput> = work
        .iter()
        .map(|(item, task, _)| build_multimodal_input(item, task))
        .collect();

    let mut results: Vec<Option<InferResult>> = vec![None; work.len()];
    let mut valid_indices = Vec::with_capacity(work.len());
    let mut valid_inputs = Vec::with_capacity(work.len());
    let mut valid_token_counts = Vec::with_capacity(work.len());

    // 1. Per-item preflight validation (isolates bad items before GPU/ANE execution)
    for (idx, ((item, _, truncate_dim), inp)) in work.iter().zip(inputs.iter()).enumerate() {
        if let Err(e) = embedder.validate_truncate_dim(*truncate_dim) {
            results[idx] = Some(Err(EngineError::InvalidInput(e.to_string())));
            continue;
        }

        if let PreparedItem::Audio { samples, .. } = item {
            if samples.is_empty() {
                results[idx] = Some(Err(EngineError::InvalidInput(
                    "Input audio cannot be empty".to_string(),
                )));
                continue;
            }
            if !embedder.has_audio_projector() {
                results[idx] = Some(Err(EngineError::InvalidInput(
                    "Audio encoding requires an attached GGUF mmproj audio projector".to_string(),
                )));
                continue;
            }
        }

        match embedder
            .tokenizer
            .encode_modality_sequence(&inp.parts, inp.task)
        {
            Ok((ids, _)) => {
                let tok_count = ids.len();
                if tok_count > embedder.max_seq_len {
                    results[idx] = Some(Err(EngineError::InvalidInput(format!(
                        "Sequence length ({} tokens) exceeds maximum context limit ({} tokens)",
                        tok_count, embedder.max_seq_len
                    ))));
                    continue;
                }
                let is_pure_text = matches!(item, PreparedItem::Text(_));
                if is_pure_text && embedder.text_backend == TextBackendPolicy::AneStrict {
                    #[cfg(any(feature = "coreml", feature = "ane"))]
                    {
                        match &embedder.coreml_text {
                            Some(ct) if tok_count > ct.max_seq_len() => {
                                results[idx] = Some(Err(EngineError::InvalidInput(format!(
                                    "Sequence length ({} tokens) exceeds CoreML AneStrict limit ({} tokens)",
                                    tok_count,
                                    ct.max_seq_len()
                                ))));
                                continue;
                            }
                            None => {
                                results[idx] = Some(Err(EngineError::InvalidInput(
                                    "AneStrict text backend requested, but no CoreML text backbone is attached"
                                        .to_string(),
                                )));
                                continue;
                            }
                            _ => {}
                        }
                    }
                    #[cfg(not(any(feature = "coreml", feature = "ane")))]
                    {
                        results[idx] = Some(Err(EngineError::InvalidInput(
                            "AneStrict text backend requested, but binary was compiled without coreml"
                                .to_string(),
                        )));
                        continue;
                    }
                }
                valid_indices.push(idx);
                valid_inputs.push(inp.clone());
                valid_token_counts.push(tok_count);
            }
            Err(e) => {
                results[idx] = Some(Err(EngineError::InvalidInput(e.to_string())));
            }
        }
    }

    let finalize_item = |vec: Vec<f32>, truncate_dim: Option<usize>, tok_count: usize| {
        let final_vec = match truncate_dim {
            Some(dim) if dim < embedder.embedding_dim() => {
                truncate_mrl(&vec, dim).map_err(|e| EngineError::InvalidInput(e.to_string()))
            }
            _ => Ok(vec),
        };
        final_vec.map(|v| (v, tok_count))
    };

    // 2. Execute valid items in a single coalesced batch; isolate per-item on failure
    if !valid_inputs.is_empty() {
        match embedder.embed_batch(&valid_inputs, None) {
            Ok(embeddings) => {
                for ((idx, tok_count), vec) in valid_indices
                    .into_iter()
                    .zip(valid_token_counts)
                    .zip(embeddings)
                {
                    let truncate_dim = work[idx].2;
                    results[idx] = Some(finalize_item(vec, truncate_dim, tok_count));
                }
            }
            Err(batch_err) => {
                if valid_inputs.len() == 1 {
                    let idx = valid_indices[0];
                    results[idx] = Some(Err(classify_core_error(&batch_err)));
                } else {
                    tracing::warn!(
                        "Coalesced batch of {} items failed ({}); falling back to per-item isolation",
                        valid_inputs.len(),
                        batch_err
                    );
                    for ((idx, tok_count), single_inp) in valid_indices
                        .into_iter()
                        .zip(valid_token_counts)
                        .zip(valid_inputs)
                    {
                        let truncate_dim = work[idx].2;
                        let item_res = match embedder.embed_batch(&[single_inp], None) {
                            Ok(mut vecs) => match vecs.pop() {
                                Some(vec) => finalize_item(vec, truncate_dim, tok_count),
                                None => Err(EngineError::Internal(
                                    "Empty embedding output from single-item fallback".to_string(),
                                )),
                            },
                            Err(e) => Err(classify_core_error(&e)),
                        };
                        results[idx] = Some(item_res);
                    }
                }
            }
        }
    }

    results
        .into_iter()
        .map(|opt| {
            opt.unwrap_or_else(|| {
                Err(EngineError::Internal(
                    "Missing batch inference result".to_string(),
                ))
            })
        })
        .collect()
}

/// Helper to decode an image from base64 data URI, raw base64, or (when explicitly opted in) local path.
fn decode_image(
    input: &str,
    allow_local_files: bool,
) -> std::result::Result<image::DynamicImage, EngineError> {
    let bytes = if let Some(stripped) = input.strip_prefix("data:") {
        if let Some((_, b64)) = stripped.split_once(";base64,") {
            BASE64.decode(b64.trim()).map_err(|e| {
                EngineError::InvalidInput(format!("Failed to base64-decode image data URI: {e}"))
            })?
        } else {
            return Err(EngineError::InvalidInput(
                "Invalid data URI format: missing ';base64,' marker".to_string(),
            ));
        }
    } else if allow_local_files && std::path::Path::new(input).exists() {
        return image::open(input).map_err(|e| {
            EngineError::InvalidInput(format!("Failed to open local image file: {e}"))
        });
    } else {
        BASE64.decode(input.trim()).map_err(|e| {
            if !allow_local_files {
                EngineError::InvalidInput(format!(
                    "Failed to base64-decode image input ({e}). Local file paths are disabled over HTTP by default; \
                     pass a base64 data URI (data:image/png;base64,...) or start the server with --allow-local-files."
                ))
            } else {
                EngineError::InvalidInput(format!("Failed to base64-decode image string: {e}"))
            }
        })?
    };

    image::load_from_memory(&bytes)
        .map_err(|e| EngineError::InvalidInput(format!("Failed to decode image bytes: {e}")))
}

/// Helper to decode audio samples from base64 data URI, raw base64 WAV, or (when explicitly opted in) local path.
fn decode_audio(
    input: &str,
    allow_local_files: bool,
) -> std::result::Result<Vec<f32>, EngineError> {
    if let Some(stripped) = input.strip_prefix("data:") {
        if let Some((_, b64)) = stripped.split_once(";base64,") {
            let bytes = BASE64.decode(b64.trim()).map_err(|e| {
                EngineError::InvalidInput(format!("Failed to base64-decode audio data URI: {e}"))
            })?;
            glypr_core::multimodal::decode_wav_bytes(&bytes)
                .map_err(|e| EngineError::InvalidInput(format!("Failed to decode WAV bytes: {e}")))
        } else {
            Err(EngineError::InvalidInput(
                "Invalid data URI format: missing ';base64,' marker".to_string(),
            ))
        }
    } else if allow_local_files && std::path::Path::new(input).exists() {
        glypr_core::multimodal::decode_wav_file(input)
            .map_err(|e| EngineError::InvalidInput(format!("Failed to decode local WAV file: {e}")))
    } else {
        let bytes = BASE64.decode(input.trim()).map_err(|e| {
            if !allow_local_files {
                EngineError::InvalidInput(format!(
                    "Failed to base64-decode audio input ({e}). Local file paths are disabled over HTTP by default; \
                     pass a base64 data URI (data:audio/wav;base64,...) or start the server with --allow-local-files."
                ))
            } else {
                EngineError::InvalidInput(format!("Failed to base64-decode audio string: {e}"))
            }
        })?;
        glypr_core::multimodal::decode_wav_bytes(&bytes)
            .map_err(|e| EngineError::InvalidInput(format!("Failed to decode WAV bytes: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::MediaInputItem;

    #[test]
    fn test_prepare_input_item_blocks_local_paths_by_default() {
        let tmp = std::env::temp_dir().join(format!(
            "glypr_server_local_path_test_{}.png",
            std::process::id()
        ));
        let img = image::DynamicImage::new_rgb8(16, 16);
        img.save(&tmp).unwrap();

        let item = InputItem::Media(MediaInputItem {
            text: None,
            image: Some(tmp.to_string_lossy().to_string()),
            audio: None,
            budget: None,
        });

        let err = prepare_input_item(item.clone(), false)
            .err()
            .expect("local file path must be rejected when allow_local_files=false");
        assert!(matches!(err, EngineError::InvalidInput(_)));
        assert!(err.to_string().contains("--allow-local-files"));

        let ok = prepare_input_item(item, true);
        assert!(ok.is_ok(), "local file path should work when opted in");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_prepare_input_item_rejects_image_and_audio_conflict_and_bad_budget() {
        let both = InputItem::Media(MediaInputItem {
            text: None,
            image: Some("abc".to_string()),
            audio: Some("def".to_string()),
            budget: None,
        });
        let err = prepare_input_item(both, false).err().unwrap();
        assert!(err.to_string().contains("both 'image' and 'audio'"));

        let bad_budget = InputItem::Media(MediaInputItem {
            text: Some("hi".to_string()),
            image: None,
            audio: None,
            budget: Some(5),
        });
        let err = prepare_input_item(bad_budget, false).err().unwrap();
        assert!(matches!(err, EngineError::InvalidInput(_)));
    }
}
