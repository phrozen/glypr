use std::sync::Arc;
use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::handlers::{handle_embeddings, handle_health, handle_models, AppState};

/// Construct the Axum router configured with CORS, Tracing, and API endpoints.
pub fn build_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/v1/embeddings", post(handle_embeddings))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
