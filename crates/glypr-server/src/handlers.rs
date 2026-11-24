use axum::{
    Json,
    extract::{FromRequest, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Serialize, de::DeserializeOwned};
use std::sync::Arc;
use std::time::Duration;

use crate::engine::{EngineError, EngineHandle, OwnedTask, PreparedItem};
use crate::schema::{
    EmbeddingObject, EmbeddingOutput, EmbeddingRequest, EmbeddingResponse, EncodingFormat,
    HealthResponse, InputItem, ModelListResponse, ModelObject, ReadyResponse, Usage,
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

fn gateway_timeout(message: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::GATEWAY_TIMEOUT,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.into(),
                error_type: "timeout_error",
                param: None,
                code: None,
            },
        }),
    )
}

fn engine_error_response(err: EngineError) -> Response {
    match err {
        EngineError::InvalidInput(msg) => bad_request(msg).into_response(),
        EngineError::Internal(msg) => internal_error(msg).into_response(),
    }
}

/// JSON body extractor that converts Axum `422` / malformed-JSON rejections into OpenAI-style
/// `400 Bad Request` JSON responses.
pub struct OpenAiJson<T>(pub T);

impl<S, T> FromRequest<S> for OpenAiJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(val)) => Ok(Self(val)),
            Err(rejection) => Err(bad_request(rejection.body_text()).into_response()),
        }
    }
}

/// Handler for `POST /v1/embeddings`.
#[allow(clippy::result_large_err)]
pub async fn handle_embeddings(
    State(state): State<Arc<AppState>>,
    OpenAiJson(payload): OpenAiJson<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, Response> {
    let model_id = payload
        .model
        .unwrap_or_else(|| state.engine.meta.id.clone());

    let max_dim = state.engine.meta.embedding_dim;
    let dimensions = payload.dimensions.unwrap_or(max_dim);
    if dimensions == 0 || dimensions > max_dim {
        return Err(bad_request(format!(
            "Invalid dimensions: {}. Must be between 1 and {}.",
            dimensions, max_dim
        ))
        .into_response());
    }

    let encoding_format = payload.encoding_format.unwrap_or(EncodingFormat::Float);
    if encoding_format == EncodingFormat::Binary && !dimensions.is_multiple_of(64) {
        return Err(bad_request(format!(
            "Binary encoding requires dimensions to be a multiple of 64, got {}",
            dimensions
        ))
        .into_response());
    }

    let task = payload
        .task
        .as_deref()
        .map(|t| OwnedTask::from_name_and_title(t, payload.title.clone()))
        .unwrap_or(OwnedTask::Raw);

    let items = payload.input.into_items();
    if items.is_empty() {
        return Err(bad_request("Input array cannot be empty").into_response());
    }
    if items.len() > state.engine.max_batch_items {
        return Err(bad_request(format!(
            "Input batch size {} exceeds max_batch_items ({})",
            items.len(),
            state.engine.max_batch_items
        ))
        .into_response());
    }

    // Upfront media & capability validation before decoding/inference
    for item in &items {
        if let InputItem::Media(m) = item {
            if m.image.is_some() && m.audio.is_some() {
                return Err(bad_request(
                    "Media input item cannot specify both 'image' and 'audio' simultaneously",
                )
                .into_response());
            }
            if let Some(budget) = m.budget
                && let Err(e) = glypr_core::validate_vision_budget(budget)
            {
                return Err(bad_request(e.to_string()).into_response());
            }
            if m.audio.is_some() && !state.engine.meta.has_audio_projector {
                return Err(bad_request(
                    "Audio input received, but server was started without a GGUF audio projector (mmproj). \
                     Restart the server with --bf16-mmproj, --q8-mmproj, or --mmproj-file.",
                )
                .into_response());
            }
            if m.image.is_some() && !state.engine.meta.has_multimodal {
                return Err(bad_request(
                    "Multimodal input received, but server was started without a multimodal projector or CoreML vision model. \
                     Restart the server with --ane, --bf16-mmproj, --q8-mmproj, or --mmproj-file.",
                )
                .into_response());
            }
        }
    }

    let truncate_dim = if dimensions < max_dim {
        Some(dimensions)
    } else {
        None
    };

    // 1. Skip Tokio blocking pool completely for pure-text batches; offload image/audio decoding
    let all_text = items.iter().all(|item| matches!(item, InputItem::Text(_)));
    let prepared_items = if all_text {
        items
            .into_iter()
            .map(|item| match item {
                InputItem::Text(s) => PreparedItem::Text(s),
                InputItem::Media(_) => unreachable!(),
            })
            .collect()
    } else {
        let allow_local_files = state.engine.allow_local_files;
        let mut decode_handles = Vec::with_capacity(items.len());
        for item in items {
            decode_handles.push(tokio::task::spawn_blocking(move || {
                crate::engine::prepare_input_item(item, allow_local_files)
            }));
        }

        let mut prepared = Vec::with_capacity(decode_handles.len());
        for handle in decode_handles {
            match handle.await {
                Ok(Ok(item)) => prepared.push(item),
                Ok(Err(e)) => return Err(engine_error_response(e)),
                Err(e) => return Err(internal_error(e.to_string()).into_response()),
            }
        }
        prepared
    };

    // 2. Dispatch the whole batch to the engine in one message under request timeout
    let timeout_dur = Duration::from_secs(state.engine.request_timeout_secs);
    let vectors = match tokio::time::timeout(
        timeout_dur,
        state.engine.infer_batch(prepared_items, task, truncate_dim),
    )
    .await
    {
        Ok(Ok(vecs)) => vecs,
        Ok(Err(e)) => return Err(engine_error_response(e)),
        Err(_elapsed) => {
            return Err(gateway_timeout(format!(
                "Inference request exceeded {}s timeout",
                state.engine.request_timeout_secs
            ))
            .into_response());
        }
    };

    // Post-process vectors into requested output encoding format
    let mut embedding_objects = Vec::with_capacity(vectors.len());
    let mut total_tokens = 0;

    for (index, (vec, prompt_tokens)) in vectors.into_iter().enumerate() {
        total_tokens += prompt_tokens;

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
    let m = &state.engine.meta;
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: m.id.clone(),
            object: "model".to_string(),
            created: m.created_at,
            owned_by: "glypr".to_string(),
            device: Some(m.device.clone()),
            vision_backend: Some(m.vision_backend.clone()),
            text_backend: Some(m.text_backend.clone()),
            ane_enabled: Some(m.ane_enabled),
            ane_text_enabled: Some(m.ane_text_enabled),
            ane_text_seq_len: m.ane_text_seq_len,
            has_multimodal: Some(m.has_multimodal),
            has_audio_projector: Some(m.has_audio_projector),
            max_context: Some(m.max_seq_len),
            embedding_dim: Some(m.embedding_dim),
        }],
    })
}

/// Handler for `GET /health`.
pub async fn handle_health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let m = &state.engine.meta;
    Json(HealthResponse {
        status: "ok".to_string(),
        model: m.id.clone(),
        device: m.device.clone(),
        multimodal: m.has_multimodal,
        has_audio_projector: m.has_audio_projector,
        vision_backend: m.vision_backend.clone(),
        text_backend: m.text_backend.clone(),
        ane_enabled: m.ane_enabled,
        ane_supported_patches: m.ane_supported_patches.clone(),
        ane_text_enabled: m.ane_text_enabled,
        ane_text_seq_len: m.ane_text_seq_len,
    })
}

/// Handler for `GET /ready`.
pub async fn handle_ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.engine.is_ready() {
        (
            StatusCode::OK,
            Json(ReadyResponse {
                status: "ready".to_string(),
                model: state.engine.meta.id.clone(),
            }),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadyResponse {
                status: "unavailable".to_string(),
                model: state.engine.meta.id.clone(),
            }),
        )
    }
}
