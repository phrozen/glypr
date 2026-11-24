use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::ServerConfig;
use crate::handlers::{AppState, handle_embeddings, handle_health, handle_models, handle_ready};

/// Construct the Axum router with default 32 MiB body limit, CORS, tracing, and API endpoints.
pub fn build_router(state: Arc<AppState>) -> Router {
    build_router_with_body_limit(state, 32 * 1024 * 1024)
}

/// Construct the Axum router configured from [`ServerConfig`] (including `max_body_bytes`).
pub fn build_router_with_config(state: Arc<AppState>, config: &ServerConfig) -> Router {
    build_router_with_body_limit(state, config.max_body_bytes)
}

fn build_router_with_body_limit(state: Arc<AppState>, max_body_bytes: usize) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/v1/embeddings", post(handle_embeddings))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        .route("/ready", get(handle_ready))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
