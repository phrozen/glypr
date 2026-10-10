use serde::{Deserialize, Serialize};

/// Supported embedding encoding formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    #[default]
    Float,
    Base64,
    #[serde(alias = "int")]
    Int8,
    Binary,
}

/// An individual media input object (image, audio, or accompanied text).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MediaInputItem {
    pub text: Option<String>,
    pub image: Option<String>,
    pub audio: Option<String>,
    pub budget: Option<usize>,
}

/// An individual input element (either raw text or media object).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputItem {
    Text(String),
    Media(MediaInputItem),
}

/// The `input` field of an embedding request: single item or array of items.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputType {
    Single(InputItem),
    Batch(Vec<InputItem>),
}

impl InputType {
    /// Normalize into a list of `InputItem`s.
    pub fn into_items(self) -> Vec<InputItem> {
        match self {
            InputType::Single(item) => vec![item],
            InputType::Batch(items) => items,
        }
    }
}

/// OpenAI-compatible `POST /v1/embeddings` request body.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingRequest {
    /// The input text or media to embed.
    pub input: InputType,

    /// Model name (optional, defaults to "embeddinggemma-2").
    #[serde(default)]
    pub model: Option<String>,

    /// Matryoshka output dimension truncation (optional, defaults to 768).
    #[serde(default)]
    pub dimensions: Option<usize>,

    /// Output encoding format (optional, defaults to "float").
    #[serde(default)]
    pub encoding_format: Option<EncodingFormat>,

    /// Task instruction prompt (optional, defaults to "Raw").
    #[serde(default)]
    pub task: Option<String>,

    /// Document title (optional, used when task is Document).
    #[serde(default)]
    pub title: Option<String>,

    /// Optional user identifier for OpenAI compatibility.
    #[serde(default)]
    pub user: Option<String>,
}

/// The serialized embedding vector output in the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingOutput {
    Float(Vec<f32>),
    Base64(String),
    Int8(Vec<i8>),
    Binary(String),
}

/// An individual embedding object in the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub index: usize,
    pub embedding: EmbeddingOutput,
}

/// Token usage metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// OpenAI-compatible `POST /v1/embeddings` response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: Usage,
}

/// Model descriptor for `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// OpenAI-compatible `GET /v1/models` response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

/// Health check response for `GET /health`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub device: String,
    pub multimodal: bool,
}
