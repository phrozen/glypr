use std::sync::Arc;
use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use glypr_core::{truncate_mrl, GlyprEmbedder, Task};
use tokio::sync::{mpsc, oneshot};

use crate::schema::{InputItem, MediaInputItem};

/// Owned representation of Task to allow moving across thread boundaries.
#[derive(Debug, Clone)]
pub enum OwnedTask {
    SearchQuery,
    Document { title: Option<String> },
    CodeRetrieval,
    Classification,
    Clustering,
    QuestionAnswering,
    FactChecking,
    SentenceSimilarity,
    Custom(String),
    Raw,
}

impl OwnedTask {
    pub fn as_task(&self) -> Task<'_> {
        match self {
            Self::SearchQuery => Task::SearchQuery,
            Self::Document { title } => Task::Document {
                title: title.as_deref(),
            },
            Self::CodeRetrieval => Task::CodeRetrieval,
            Self::Classification => Task::Classification,
            Self::Clustering => Task::Clustering,
            Self::QuestionAnswering => Task::QuestionAnswering,
            Self::FactChecking => Task::FactChecking,
            Self::SentenceSimilarity => Task::SentenceSimilarity,
            Self::Custom(s) => Task::Custom(s),
            Self::Raw => Task::Raw,
        }
    }

    pub fn from_name_and_title(name: &str, title: Option<String>) -> Self {
        match name.to_lowercase().replace(['-', '_', ' '], "").as_str() {
            "searchquery" | "retrieval" | "retrievalquery" | "query" => Self::SearchQuery,
            "document" | "retrievaldocument" => Self::Document { title },
            "coderetrieval" | "code" => Self::CodeRetrieval,
            "classification" | "multilabelclassification" => Self::Classification,
            "clustering" => Self::Clustering,
            "questionanswering" | "qa" => Self::QuestionAnswering,
            "factchecking" => Self::FactChecking,
            "sentencesimilarity" | "sts" | "pairclassification" | "summarization" => {
                Self::SentenceSimilarity
            }
            "raw" | "none" => Self::Raw,
            _ => Self::Custom(name.to_string()),
        }
    }
}

/// An individual item queued for inference.
pub struct InferenceRequest {
    pub item: InputItem,
    pub task: OwnedTask,
    pub truncate_dim: Option<usize>,
    pub sender: oneshot::Sender<Result<Vec<f32>>>,
}

/// Metadata describing the loaded model.
#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub id: String,
    pub device: String,
    pub has_multimodal: bool,
    pub created_at: u64,
}

/// Cloneable handle to submit inference requests to the worker loop.
#[derive(Clone)]
pub struct EngineHandle {
    sender: mpsc::Sender<InferenceRequest>,
    pub meta: Arc<ModelMeta>,
}

impl EngineHandle {
    /// Submit a single request and await its embedding result.
    pub async fn infer(
        &self,
        item: InputItem,
        task: OwnedTask,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<f32>> {
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
            .map_err(|_| anyhow::anyhow!("Inference worker is not running"))?;

        rx.await
            .map_err(|_| anyhow::anyhow!("Inference worker dropped request"))?
    }
}

/// Spawn the background inference engine on a dedicated OS thread.
pub fn spawn_inference_engine(
    embedder: GlyprEmbedder,
    max_batch_size: usize,
    model_id: String,
) -> EngineHandle {
    let (tx, mut rx) = mpsc::channel::<InferenceRequest>(1024);

    let device_name = format!("{:?}", embedder.device);
    let has_multimodal = embedder.multimodal.is_some();
    let meta = Arc::new(ModelMeta {
        id: model_id,
        device: device_name,
        has_multimodal,
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    });

    let max_batch = max_batch_size.max(1);

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

                process_batch(&embedder, batch);
            }

            tracing::info!("Inference worker shut down");
        })
        .expect("Failed to spawn inference worker thread");

    EngineHandle { sender: tx, meta }
}

/// Process a collected batch of inference requests.
fn process_batch(embedder: &GlyprEmbedder, batch: Vec<InferenceRequest>) {
    // Separate plain text items from multimodal items
    let mut text_items: Vec<(usize, String, Option<usize>)> = Vec::new();
    let mut media_items: Vec<(usize, MediaInputItem, OwnedTask, Option<usize>)> = Vec::new();

    // Map each batch index to its sender
    let mut senders = Vec::with_capacity(batch.len());

    for (idx, req) in batch.into_iter().enumerate() {
        senders.push(req.sender);
        match req.item {
            InputItem::Text(text) => {
                let formatted = req.task.as_task().format_text(&text);
                text_items.push((idx, formatted, req.truncate_dim));
            }
            InputItem::Media(media) => {
                media_items.push((idx, media, req.task, req.truncate_dim));
            }
        }
    }

    // Process batched plain texts
    if !text_items.is_empty() {
        let texts_ref: Vec<&str> = text_items.iter().map(|(_, t, _)| t.as_str()).collect();

        match embedder.encode_batch_texts(&texts_ref, None) {
            Ok(embeddings) => {
                for ((idx, _, truncate_dim), vec) in text_items.into_iter().zip(embeddings) {
                    let final_vec = match truncate_dim {
                        Some(dim) if dim < 768 => truncate_mrl(&vec, dim),
                        _ => Ok(vec),
                    };
                    let sender = &mut senders[idx];
                    // Replace with empty sender to take ownership
                    let taken = std::mem::replace(sender, oneshot::channel().0);
                    let _ = taken.send(final_vec);
                }
            }
            Err(e) => {
                let err_msg = e.to_string();
                for (idx, _, _) in text_items {
                    let sender = &mut senders[idx];
                    let taken = std::mem::replace(sender, oneshot::channel().0);
                    let _ = taken.send(Err(anyhow::anyhow!("Batch encoding error: {}", err_msg)));
                }
            }
        }
    }

    // Process media items individually
    for (idx, media, task, truncate_dim) in media_items {
        let res = process_media_item(embedder, &media, &task, truncate_dim);
        let sender = &mut senders[idx];
        let taken = std::mem::replace(sender, oneshot::channel().0);
        let _ = taken.send(res);
    }
}

/// Process a single multimodal input item.
fn process_media_item(
    embedder: &GlyprEmbedder,
    media: &MediaInputItem,
    task: &OwnedTask,
    truncate_dim: Option<usize>,
) -> Result<Vec<f32>> {
    if embedder.multimodal.is_none() {
        anyhow::bail!(
            "Multimodal input received, but server was started without a multimodal projector. \
            Restart the server with --bf16-mmproj, --q8-mmproj, or --mmproj-file."
        );
    }

    let prompt = media.text.as_deref();

    let vec = if let Some(ref img_src) = media.image {
        let img = decode_image(img_src).context("Failed to decode image input")?;
        match media.budget {
            Some(budget) => embedder.embed_image_with_budget(&img, prompt, budget)?,
            None => embedder.embed_image(&img, prompt)?,
        }
    } else if let Some(ref aud_src) = media.audio {
        let samples = decode_audio(aud_src).context("Failed to decode audio input")?;
        embedder.embed_audio(&samples, prompt)?
    } else if let Some(ref txt) = media.text {
        // Text-only media item
        let formatted = task.as_task().format_text(txt);
        embedder.embed_text(&formatted, task.as_task())?
    } else {
        anyhow::bail!("Media input item has no text, image, or audio specified");
    };

    match truncate_dim {
        Some(dim) if dim < 768 => truncate_mrl(&vec, dim),
        _ => Ok(vec),
    }
}

/// Helper to decode an image from base64 data URI, raw base64, or local path.
fn decode_image(input: &str) -> Result<image::DynamicImage> {
    let bytes = if let Some(stripped) = input.strip_prefix("data:") {
        if let Some((_, b64)) = stripped.split_once(";base64,") {
            BASE64
                .decode(b64.trim())
                .context("Failed to base64-decode image data URI")?
        } else {
            anyhow::bail!("Invalid data URI format: missing ';base64,' marker");
        }
    } else if std::path::Path::new(input).exists() {
        return image::open(input).context("Failed to open local image file");
    } else {
        BASE64
            .decode(input.trim())
            .context("Failed to base64-decode image string")?
    };

    image::load_from_memory(&bytes).context("Failed to load image from decoded bytes")
}

/// Helper to decode audio samples from base64 data URI, raw base64 WAV, or local path.
fn decode_audio(input: &str) -> Result<Vec<f32>> {
    let cursor: std::io::Cursor<Vec<u8>> = if let Some(stripped) = input.strip_prefix("data:") {
        if let Some((_, b64)) = stripped.split_once(";base64,") {
            let bytes = BASE64
                .decode(b64.trim())
                .context("Failed to base64-decode audio data URI")?;
            std::io::Cursor::new(bytes)
        } else {
            anyhow::bail!("Invalid data URI format: missing ';base64,' marker");
        }
    } else if std::path::Path::new(input).exists() {
        let mut reader = hound::WavReader::open(input).context("Failed to open local WAV file")?;
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap_or(0) as f32 / 32767.0)
            .collect();
        return Ok(samples);
    } else {
        let bytes = BASE64
            .decode(input.trim())
            .context("Failed to base64-decode audio string")?;
        std::io::Cursor::new(bytes)
    };

    let mut reader = hound::WavReader::new(cursor).context("Failed to parse WAV audio bytes")?;
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap_or(0) as f32 / 32767.0)
        .collect();
    Ok(samples)
}
