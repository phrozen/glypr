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

/// A whole request batch queued in one channel message.
pub struct BatchInferenceRequest {
    pub items: Vec<PreparedItem>,
    pub task: OwnedTask,
    pub truncate_dim: Option<usize>,
    pub sender: oneshot::Sender<Vec<InferResult>>,
}

enum WorkerMessage {
    Single(InferenceRequest),
    Batch(BatchInferenceRequest),
}

enum ResponseTarget {
    Single(oneshot::Sender<InferResult>),
    Batch {
        sender: oneshot::Sender<Vec<InferResult>>,
        len: usize,
    },
}

fn push_worker_message(
    msg: WorkerMessage,
    work: &mut Vec<(PreparedItem, OwnedTask, Option<usize>)>,
    targets: &mut Vec<ResponseTarget>,
) {
    match msg {
        WorkerMessage::Single(req) => {
            work.push((req.item, req.task, req.truncate_dim));
            targets.push(ResponseTarget::Single(req.sender));
        }
        WorkerMessage::Batch(batch_req) => {
            let len = batch_req.items.len();
            for item in batch_req.items {
                work.push((item, batch_req.task.clone(), batch_req.truncate_dim));
            }
            targets.push(ResponseTarget::Batch {
                sender: batch_req.sender,
                len,
            });
        }
    }
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
    sender: mpsc::Sender<WorkerMessage>,
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
            .send(WorkerMessage::Single(req))
            .await
            .map_err(|_| EngineError::Internal("Inference worker is not running".to_string()))?;

        rx.await
            .map_err(|_| EngineError::Internal("Inference worker dropped request".to_string()))?
    }

    /// Submit a batch of prepared items in one channel message and await all results in order.
    pub async fn infer_batch(
        &self,
        items: Vec<PreparedItem>,
        task: OwnedTask,
        truncate_dim: Option<usize>,
    ) -> std::result::Result<Vec<(Vec<f32>, usize)>, EngineError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let (tx, rx) = oneshot::channel();
        let req = BatchInferenceRequest {
            items,
            task,
            truncate_dim,
            sender: tx,
        };

        self.sender
            .send(WorkerMessage::Batch(req))
            .await
            .map_err(|_| EngineError::Internal("Inference worker is not running".to_string()))?;

        let results = rx.await.map_err(|_| {
            EngineError::Internal("Inference worker dropped batch request".to_string())
        })?;

        results.into_iter().collect()
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
    let (tx, mut rx) = mpsc::channel::<WorkerMessage>(1024);

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
    let batch_wait = if config.batch_wait_us > 0 {
        Some(std::time::Duration::from_micros(config.batch_wait_us))
    } else {
        None
    };

    std::thread::Builder::new()
        .name("glypr-infer-worker".into())
        .spawn(move || {
            tracing::info!(
                "Inference worker started with max_batch_size={}, batch_wait_us={}",
                max_batch,
                batch_wait.map(|d| d.as_micros() as u64).unwrap_or(0)
            );

            while let Some(first_msg) = rx.blocking_recv() {
                let mut work = Vec::with_capacity(max_batch);
                let mut targets = Vec::new();
                push_worker_message(first_msg, &mut work, &mut targets);

                // Drain pending requests up to max_batch without blocking
                while work.len() < max_batch {
                    match rx.try_recv() {
                        Ok(msg) => push_worker_message(msg, &mut work, &mut targets),
                        Err(_) => break,
                    }
                }

                // Optional micro-batch coalescing window when batch_wait_us > 0
                if work.len() < max_batch
                    && let Some(wait_dur) = batch_wait
                {
                    let deadline = std::time::Instant::now() + wait_dur;
                    while work.len() < max_batch {
                        let now = std::time::Instant::now();
                        if now >= deadline {
                            break;
                        }
                        match rx.try_recv() {
                            Ok(msg) => push_worker_message(msg, &mut work, &mut targets),
                            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                                let remaining = deadline.saturating_duration_since(now);
                                std::thread::sleep(
                                    remaining.min(std::time::Duration::from_micros(50)),
                                );
                            }
                            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
                        }
                    }
                }

                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    with_autorelease_pool(|| process_batch_results(&embedder, &work, max_batch))
                }));

                match outcome {
                    Ok(results) => {
                        let mut res_iter = results.into_iter();
                        for target in targets {
                            match target {
                                ResponseTarget::Single(sender) => {
                                    let res = res_iter.next().unwrap_or_else(|| {
                                        Err(EngineError::Internal(
                                            "Missing inference result".to_string(),
                                        ))
                                    });
                                    let _ = sender.send(res);
                                }
                                ResponseTarget::Batch { sender, len } => {
                                    let chunk: Vec<InferResult> =
                                        res_iter.by_ref().take(len).collect();
                                    let _ = sender.send(chunk);
                                }
                            }
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
                        for target in targets {
                            match target {
                                ResponseTarget::Single(sender) => {
                                    let _ = sender.send(Err(EngineError::Internal(format!(
                                        "Inference worker internal panic: {}",
                                        panic_msg
                                    ))));
                                }
                                ResponseTarget::Batch { sender, len } => {
                                    let errs = (0..len)
                                        .map(|_| {
                                            Err(EngineError::Internal(format!(
                                                "Inference worker internal panic: {}",
                                                panic_msg
                                            )))
                                        })
                                        .collect();
                                    let _ = sender.send(errs);
                                }
                            }
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

enum PreflightItem<'a> {
    Invalid(EngineError),
    ValidText {
        idx: usize,
        token_ids: Vec<u32>,
        tok_count: usize,
        truncate_dim: Option<usize>,
    },
    ValidMultimodal {
        idx: usize,
        input: MultimodalInput<'a>,
        tok_count: usize,
        truncate_dim: Option<usize>,
    },
}

fn finalize_item(
    embedder: &GlyprEmbedder,
    vec: Vec<f32>,
    truncate_dim: Option<usize>,
    tok_count: usize,
) -> InferResult {
    let final_vec = match truncate_dim {
        Some(dim) if dim < embedder.embedding_dim() => {
            truncate_mrl(&vec, dim).map_err(|e| EngineError::InvalidInput(e.to_string()))
        }
        _ => Ok(vec),
    };
    final_vec.map(|v| (v, tok_count))
}

/// Process a collected batch of inference requests with parallel preflight/tokenization,
/// length/modality bucketing, exact order preservation, and per-item fallback isolation.
fn process_batch_results(
    embedder: &GlyprEmbedder,
    work: &[(PreparedItem, OwnedTask, Option<usize>)],
    max_batch: usize,
) -> Vec<InferResult> {
    use rayon::prelude::*;

    let preflight: Vec<PreflightItem<'_>> = work
        .par_iter()
        .enumerate()
        .map(|(idx, (item, task, truncate_dim))| {
            if let Err(e) = embedder.validate_truncate_dim(*truncate_dim) {
                return PreflightItem::Invalid(EngineError::InvalidInput(e.to_string()));
            }

            if let PreparedItem::Audio { samples, .. } = item {
                if samples.is_empty() {
                    return PreflightItem::Invalid(EngineError::InvalidInput(
                        "Input audio cannot be empty".to_string(),
                    ));
                }
                if !embedder.has_audio_projector() {
                    return PreflightItem::Invalid(EngineError::InvalidInput(
                        "Audio encoding requires an attached GGUF mmproj audio projector"
                            .to_string(),
                    ));
                }
            }

            let inp = build_multimodal_input(item, task);
            match embedder
                .tokenizer
                .encode_modality_sequence(&inp.parts, inp.task)
            {
                Ok((ids, _)) => {
                    let tok_count = ids.len();
                    if tok_count > embedder.max_seq_len {
                        return PreflightItem::Invalid(EngineError::InvalidInput(format!(
                            "Sequence length ({} tokens) exceeds maximum context limit ({} tokens)",
                            tok_count, embedder.max_seq_len
                        )));
                    }
                    let is_pure_text = matches!(item, PreparedItem::Text(_));
                    if is_pure_text && embedder.text_backend == TextBackendPolicy::AneStrict {
                        #[cfg(any(feature = "coreml", feature = "ane"))]
                        {
                            match &embedder.coreml_text {
                                Some(ct) if tok_count > ct.max_seq_len() => {
                                    return PreflightItem::Invalid(EngineError::InvalidInput(
                                        format!(
                                            "Sequence length ({} tokens) exceeds CoreML AneStrict limit ({} tokens)",
                                            tok_count,
                                            ct.max_seq_len()
                                        ),
                                    ));
                                }
                                None => {
                                    return PreflightItem::Invalid(EngineError::InvalidInput(
                                        "AneStrict text backend requested, but no CoreML text backbone is attached"
                                            .to_string(),
                                    ));
                                }
                                _ => {}
                            }
                        }
                        #[cfg(not(any(feature = "coreml", feature = "ane")))]
                        {
                            return PreflightItem::Invalid(EngineError::InvalidInput(
                                "AneStrict text backend requested, but binary was compiled without coreml"
                                    .to_string(),
                            ));
                        }
                    }

                    if is_pure_text {
                        PreflightItem::ValidText {
                            idx,
                            token_ids: ids,
                            tok_count,
                            truncate_dim: *truncate_dim,
                        }
                    } else {
                        PreflightItem::ValidMultimodal {
                            idx,
                            input: inp,
                            tok_count,
                            truncate_dim: *truncate_dim,
                        }
                    }
                }
                Err(e) => PreflightItem::Invalid(EngineError::InvalidInput(e.to_string())),
            }
        })
        .collect();

    struct TextBucketEntry {
        orig_idx: usize,
        token_ids: Vec<u32>,
        tok_count: usize,
        truncate_dim: Option<usize>,
    }

    let mut results: Vec<Option<InferResult>> = (0..work.len()).map(|_| None).collect();
    // 4 pure-text length buckets: [<= 128, <= 512, <= 2048, > 2048]
    let mut text_buckets: [Vec<TextBucketEntry>; 4] =
        [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut multimodal_bucket: Vec<(usize, MultimodalInput<'_>, usize, Option<usize>)> = Vec::new();

    for (idx, item) in preflight.into_iter().enumerate() {
        match item {
            PreflightItem::Invalid(err) => {
                results[idx] = Some(Err(err));
            }
            PreflightItem::ValidText {
                idx,
                token_ids,
                tok_count,
                truncate_dim,
            } => {
                let bucket_idx = if tok_count <= 128 {
                    0
                } else if tok_count <= 512 {
                    1
                } else if tok_count <= 2048 {
                    2
                } else {
                    3
                };
                text_buckets[bucket_idx].push(TextBucketEntry {
                    orig_idx: idx,
                    token_ids,
                    tok_count,
                    truncate_dim,
                });
            }
            PreflightItem::ValidMultimodal {
                idx,
                input,
                tok_count,
                truncate_dim,
            } => {
                multimodal_bucket.push((idx, input, tok_count, truncate_dim));
            }
        }
    }

    let chunk_limit = max_batch.max(1);

    // Execute pure-text length buckets via pre-tokenized vectorized [B, max_len] path
    for mut bucket in text_buckets {
        while !bucket.is_empty() {
            let take = bucket.len().min(chunk_limit);
            let mut bucket_meta = Vec::with_capacity(take);
            let mut bucket_ids = Vec::with_capacity(take);
            for entry in bucket.drain(..take) {
                bucket_meta.push((entry.orig_idx, entry.tok_count, entry.truncate_dim));
                bucket_ids.push(entry.token_ids);
            }

            match embedder.embed_tokenized_texts(&bucket_ids, None) {
                Ok(embeddings) => {
                    for ((orig_idx, tok_count, truncate_dim), vec) in
                        bucket_meta.into_iter().zip(embeddings)
                    {
                        results[orig_idx] =
                            Some(finalize_item(embedder, vec, truncate_dim, tok_count));
                    }
                }
                Err(batch_err) => {
                    if bucket_ids.len() == 1 {
                        let orig_idx = bucket_meta[0].0;
                        results[orig_idx] = Some(Err(classify_core_error(&batch_err)));
                    } else {
                        tracing::warn!(
                            "Text bucket of {} items failed ({}); falling back to per-item isolation",
                            bucket_ids.len(),
                            batch_err
                        );
                        for ((orig_idx, tok_count, truncate_dim), single_ids) in
                            bucket_meta.into_iter().zip(bucket_ids)
                        {
                            let item_res = match embedder
                                .embed_tokenized_texts(std::slice::from_ref(&single_ids), None)
                            {
                                Ok(mut vecs) => match vecs.pop() {
                                    Some(vec) => {
                                        finalize_item(embedder, vec, truncate_dim, tok_count)
                                    }
                                    None => Err(EngineError::Internal(
                                        "Empty embedding output from single-item fallback"
                                            .to_string(),
                                    )),
                                },
                                Err(e) => Err(classify_core_error(&e)),
                            };
                            results[orig_idx] = Some(item_res);
                        }
                    }
                }
            }
        }
    }

    // Execute multimodal bucket via embed_batch
    while !multimodal_bucket.is_empty() {
        let take = multimodal_bucket.len().min(chunk_limit);
        let mut mm_meta = Vec::with_capacity(take);
        let mut mm_inputs = Vec::with_capacity(take);
        for (orig_idx, inp, tok_count, truncate_dim) in multimodal_bucket.drain(..take) {
            mm_meta.push((orig_idx, tok_count, truncate_dim));
            mm_inputs.push(inp);
        }

        match embedder.embed_batch(&mm_inputs, None) {
            Ok(embeddings) => {
                for ((orig_idx, tok_count, truncate_dim), vec) in
                    mm_meta.into_iter().zip(embeddings)
                {
                    results[orig_idx] = Some(finalize_item(embedder, vec, truncate_dim, tok_count));
                }
            }
            Err(batch_err) => {
                if mm_inputs.len() == 1 {
                    let orig_idx = mm_meta[0].0;
                    results[orig_idx] = Some(Err(classify_core_error(&batch_err)));
                } else {
                    tracing::warn!(
                        "Multimodal bucket of {} items failed ({}); falling back to per-item isolation",
                        mm_inputs.len(),
                        batch_err
                    );
                    for ((orig_idx, tok_count, truncate_dim), single_inp) in
                        mm_meta.into_iter().zip(mm_inputs)
                    {
                        let item_res = match embedder.embed_batch(&[single_inp], None) {
                            Ok(mut vecs) => match vecs.pop() {
                                Some(vec) => finalize_item(embedder, vec, truncate_dim, tok_count),
                                None => Err(EngineError::Internal(
                                    "Empty embedding output from single-item fallback".to_string(),
                                )),
                            },
                            Err(e) => Err(classify_core_error(&e)),
                        };
                        results[orig_idx] = Some(item_res);
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

    fn build_tiny_engine_test_embedder(max_seq_len: usize) -> GlyprEmbedder {
        use candle_core::quantized::gguf_file::Value;
        use candle_core::{DType, Device, Tensor};
        use glypr_core::_candle_nn::{Embedding, Linear};
        use glypr_core::{
            EmbeddingGemma2Model, LinearOp, TextBackendPolicy, VisionBackendPolicy, attention,
            config, layer, mlp, norm, ple, rope, tokenizer::GlyprTokenizer,
        };
        use std::collections::HashMap;

        let hidden_size = 8usize;
        let intermediate_size = 16usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 4usize;
        let embedding_dim = 64usize;
        let hidden_size_per_layer_input = 8usize;
        let vocab_size = 16usize;
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut rope_parameters = HashMap::new();
        rope_parameters.insert(
            "sliding_attention".to_string(),
            config::RoPEParameters {
                rope_theta: 10_000.0,
                rope_type: "default".to_string(),
            },
        );
        rope_parameters.insert(
            "full_attention".to_string(),
            config::RoPEParameters {
                rope_theta: 1_000_000.0,
                rope_type: "default".to_string(),
            },
        );

        let text_config = config::TextConfig {
            hidden_size,
            intermediate_size,
            num_hidden_layers: 1,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            vocab_size,
            embedding_dim,
            max_position_embeddings: 4096,
            sliding_window: 512,
            rms_norm_eps: 1e-6,
            hidden_size_per_layer_input,
            layer_types: vec![config::LayerType::SlidingAttention],
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

        let embed_weight =
            candle_core::Tensor::arange(0u32, (vocab_size * hidden_size) as u32, &device)
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
                Some(text_config.sliding_window),
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
            layer_scalar: candle_core::Tensor::ones((1,), dtype, &device).unwrap(),
        };

        let model = EmbeddingGemma2Model {
            config: text_config,
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

        let vocab = vec!["<unk>", "</s>", "<s>", "a", "b", "\u{2581}"];
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
            max_seq_len,
            #[cfg(any(feature = "coreml", feature = "ane"))]
            coreml_vision: None,
            #[cfg(any(feature = "coreml", feature = "ane"))]
            coreml_text: None,
            vision_backend: VisionBackendPolicy::Auto,
            text_backend: TextBackendPolicy::Auto,
        }
    }

    #[test]
    fn test_server_config_batch_wait_us_defaults_to_zero() {
        assert_eq!(crate::DEFAULT_BATCH_WAIT_US, 0);
        assert_eq!(crate::ServerConfig::default().batch_wait_us, 0);
    }

    #[test]
    fn test_process_batch_results_length_bucketing_preserves_order_and_truncate_dim() {
        let embedder = build_tiny_engine_test_embedder(3000);

        let text_b2 = "a ".repeat(300); // ~600 tokens -> bucket <= 2048
        let text_b0_a = "a".to_string(); // <= 128 -> bucket <= 128
        let text_b1 = "a ".repeat(100); // ~200 tokens -> bucket <= 512
        let text_b3 = "a ".repeat(1100); // ~2200 tokens -> bucket > 2048
        let text_oversized = "a ".repeat(2000); // > max_seq_len (3000) -> InvalidInput
        let text_b0_b = "a b".to_string(); // <= 128 -> bucket <= 128

        let work = vec![
            (
                PreparedItem::Text(text_b2.clone()),
                OwnedTask::Raw,
                Some(32),
            ),
            (PreparedItem::Text(text_b0_a.clone()), OwnedTask::Raw, None),
            (
                PreparedItem::Text(text_b1.clone()),
                OwnedTask::Raw,
                Some(16),
            ),
            (
                PreparedItem::Text(text_b3.clone()),
                OwnedTask::Raw,
                Some(32),
            ),
            (PreparedItem::Text(text_oversized), OwnedTask::Raw, None),
            (
                PreparedItem::Text(text_b0_b.clone()),
                OwnedTask::Raw,
                Some(8),
            ),
        ];

        let results = process_batch_results(&embedder, &work, 32);
        assert_eq!(results.len(), work.len());

        // Oversized item at index 4 must be isolated without failing any other bucket
        assert!(matches!(results[4], Err(EngineError::InvalidInput(_))));

        let expected = [
            (0usize, text_b2, Some(32), 32usize),
            (1usize, text_b0_a, None, 64usize),
            (2usize, text_b1, Some(16), 16usize),
            (3usize, text_b3, Some(32), 32usize),
            (5usize, text_b0_b, Some(8), 8usize),
        ];

        for (idx, text, trunc, expected_dim) in expected {
            let (vec, tok_count) = results[idx]
                .as_ref()
                .unwrap_or_else(|e| panic!("item {idx} failed unexpectedly: {e}"));
            assert!(*tok_count > 0);
            assert_eq!(vec.len(), expected_dim);

            let ref_vec = embedder
                .embed_text_with_dim(&text, glypr_core::Task::Raw, trunc)
                .unwrap();
            let max_diff = vec
                .iter()
                .zip(ref_vec.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-4,
                "item {idx} diverged from single-item reference: max_diff={max_diff}"
            );
        }
    }
}
