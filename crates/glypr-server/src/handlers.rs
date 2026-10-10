use std::sync::Arc;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use serde::Serialize;

use crate::engine::{EngineHandle, OwnedTask};
use crate::schema::{
    EmbeddingObject, EmbeddingOutput, EmbeddingRequest, EmbeddingResponse, EncodingFormat,
    HealthResponse, InputItem, ModelListResponse, ModelObject, Usage,
};

/// Shared application state for Axum handlers.
#[derive(Clone)]
pub struct AppState {
    pub engine: EngineHandle,
}

/// OpenAI error response payload.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

fn bad_request(message: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.into(),
                error_type: "invalid_request_error",
                param: None,
                code: None,
            },
        }),
    )
}

fn internal_error(message: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.into(),
                error_type: "internal_server_error",
                param: None,
                code: None,
            },
        }),
    )
}

/// Handler for `POST /v1/embeddings`.
#[allow(clippy::result_large_err)]
pub async fn handle_embeddings(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, Response> {
    let model_id = payload
        .model
        .unwrap_or_else(|| state.engine.meta.id.clone());

    let dimensions = payload.dimensions.unwrap_or(768);
    if dimensions == 0 || dimensions > 768 {
        return Err(bad_request(format!(
            "Invalid dimensions: {}. Must be between 1 and 768.",
            dimensions
        ))
        .into_response());
    }

    let encoding_format = payload.encoding_format.unwrap_or(EncodingFormat::Float);

    let task = payload
        .task
        .as_deref()
        .map(|t| OwnedTask::from_name_and_title(t, payload.title.clone()))
        .unwrap_or(OwnedTask::Raw);

    let items = payload.input.into_items();
    if items.is_empty() {
        return Err(bad_request("Input array cannot be empty").into_response());
    }

    // Reject multimodal inputs immediately if server has no multimodal projector
    if !state.engine.meta.has_multimodal {
        let has_media = items.iter().any(|item| match item {
            InputItem::Media(m) => m.image.is_some() || m.audio.is_some(),
            InputItem::Text(_) => false,
        });
        if has_media {
            return Err(bad_request(
                "Multimodal input received, but server was started without a multimodal projector. \
                Restart the server with --bf16-mmproj, --q8-mmproj, or --mmproj-file.",
            )
            .into_response());
        }
    }

    let truncate_dim = if dimensions < 768 {
        Some(dimensions)
    } else {
        None
    };

    // Dispatch inference tasks concurrently
    let mut handles = Vec::with_capacity(items.len());
    for item in items {
        let engine = state.engine.clone();
        let task_clone = task.clone();
        handles.push(tokio::spawn(async move {
            engine.infer(item, task_clone, truncate_dim).await
        }));
    }

    let mut vectors = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(Ok(vec)) => vectors.push(vec),
            Ok(Err(e)) => return Err(bad_request(e.to_string()).into_response()),
            Err(e) => return Err(internal_error(e.to_string()).into_response()),
        }
    }

    // Post-process vectors into requested output encoding format
    let mut embedding_objects = Vec::with_capacity(vectors.len());
    let mut total_tokens = 0;

    for (index, vec) in vectors.into_iter().enumerate() {
        // Approximate token count based on embedding size
        total_tokens += vec.len() / 8 + 4;

        let output = match encoding_format {
            EncodingFormat::Float => EmbeddingOutput::Float(vec),
            EncodingFormat::Base64 => {
                let bytes: &[u8] = bytemuck::cast_slice(&vec);
                EmbeddingOutput::Base64(BASE64.encode(bytes))
            }
            EncodingFormat::Int8 => {
                let quantized = glypr_core::quantize_int8(&vec);
                EmbeddingOutput::Int8(quantized)
            }
            EncodingFormat::Binary => {
                let quantized = glypr_core::quantize_binary(&vec).map_err(|e| {
                    bad_request(format!("Binary quantization error: {}", e)).into_response()
                })?;
                let bytes: &[u8] = bytemuck::cast_slice(&quantized);
                EmbeddingOutput::Binary(BASE64.encode(bytes))
            }
        };

        embedding_objects.push(EmbeddingObject {
            object: "embedding".to_string(),
            index,
            embedding: output,
        });
    }

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data: embedding_objects,
        model: model_id,
        usage: Usage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

/// Handler for `GET /v1/models`.
pub async fn handle_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.engine.meta.id.clone(),
            object: "model".to_string(),
            created: state.engine.meta.created_at,
            owned_by: "glypr".to_string(),
        }],
    })
}

/// Handler for `GET /health`.
pub async fn handle_health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.engine.meta.id.clone(),
        device: state.engine.meta.device.clone(),
        multimodal: state.engine.meta.has_multimodal,
    })
}
