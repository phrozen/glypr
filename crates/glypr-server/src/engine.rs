use std::sync::Arc;
use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use glypr_core::{truncate_mrl, with_autorelease_pool, GlyprEmbedder, MultimodalInput, Task};
use tokio::sync::{mpsc, oneshot};

use crate::schema::InputItem;

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
pub fn prepare_input_item(item: InputItem) -> Result<PreparedItem> {
    match item {
        InputItem::Text(s) => Ok(PreparedItem::Text(s)),
        InputItem::Media(media) => {
            if let Some(ref img_src) = media.image {
                let img = decode_image(img_src).context("Failed to decode image input")?;
                Ok(PreparedItem::Image {
                    image: Arc::new(img),
                    text: media.text,
                    budget: media.budget,
                })
            } else if let Some(ref aud_src) = media.audio {
                let samples = decode_audio(aud_src).context("Failed to decode audio input")?;
                Ok(PreparedItem::Audio {
                    samples: Arc::new(samples),
                    text: media.text,
                })
            } else if let Some(text) = media.text {
                Ok(PreparedItem::Text(text))
            } else {
                anyhow::bail!("Media input item has no text, image, or audio specified");
            }
        }
    }
}

/// An individual item queued for inference.
pub struct InferenceRequest {
    pub item: PreparedItem,
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
        item: PreparedItem,
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

                with_autorelease_pool(|| {
                    process_batch(&embedder, batch);
                });
            }

            tracing::info!("Inference worker shut down");
        })
        .expect("Failed to spawn inference worker thread");

    EngineHandle { sender: tx, meta }
}

/// Process a collected batch of inference requests in a single unified multimodal batch.
fn process_batch(embedder: &GlyprEmbedder, batch: Vec<InferenceRequest>) {
    let inputs: Vec<MultimodalInput> = batch
        .iter()
        .map(|req| {
            let task = req.task.as_task();
            match &req.item {
                PreparedItem::Text(s) => MultimodalInput::text(s.as_str(), task),
                PreparedItem::Image { image, text, budget } => {
                    if let Some(prompt) = text {
                        MultimodalInput::image_with_prompt(
                            image.as_ref(),
                            prompt.as_str(),
                            *budget,
                            task,
                        )
                    } else {
                        MultimodalInput::image(image.as_ref(), *budget, task)
                    }
                }
                PreparedItem::Audio { samples, text } => {
                    if let Some(prompt) = text {
                        MultimodalInput::audio_with_prompt(
                            samples.as_slice(),
                            prompt.as_str(),
                            task,
                        )
                    } else {
                        MultimodalInput::audio(samples.as_slice(), task)
                    }
                }
            }
        })
        .collect();

    match embedder.embed_batch(&inputs, None) {
        Ok(embeddings) => {
            for (req, vec) in batch.into_iter().zip(embeddings) {
                let final_vec = match req.truncate_dim {
                    Some(dim) if dim < 768 => truncate_mrl(&vec, dim),
                    _ => Ok(vec),
                };
                let _ = req.sender.send(final_vec);
            }
        }
        Err(e) => {
            let err_msg = e.to_string();
            for req in batch {
                let _ = req.sender.send(Err(anyhow::anyhow!("Batch inference error: {}", err_msg)));
            }
        }
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
