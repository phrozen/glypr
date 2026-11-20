pub mod engine;
pub mod handlers;
pub mod routes;
pub mod schema;

use anyhow::Result;
use glypr_core::GlyprEmbedder;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

pub use engine::{
    EngineError, EngineHandle, ModelMeta, OwnedTask, spawn_inference_engine,
    spawn_inference_engine_with_config,
};
pub use handlers::AppState;
pub use routes::{build_router, build_router_with_config};
pub use schema::*;

/// Default maximum HTTP request body size (32 MiB, supporting base64 multimodal batches).
pub const DEFAULT_MAX_BODY_BYTES: usize = 32 * 1024 * 1024;
/// Default maximum items allowed in a single `/v1/embeddings` request array.
pub const DEFAULT_MAX_BATCH_ITEMS: usize = 256;
/// Default per-request inference timeout in seconds.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;

/// Server configuration options.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_batch_size: usize,
    pub model_id: String,
    pub max_body_bytes: usize,
    pub max_batch_items: usize,
    pub allow_local_files: bool,
    pub request_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_batch_size: 32,
            model_id: "embeddinggemma-2".to_string(),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_batch_items: DEFAULT_MAX_BATCH_ITEMS,
            allow_local_files: false,
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
        }
    }
}

/// Initialize structured `tracing` logging to `stderr`, respecting `RUST_LOG` and falling back
/// to `default_level`. Safe to call multiple times (subsequent calls are no-ops).
pub fn init_tracing(default_level: &str) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sigterm) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sigterm.recv().await;
        } else {
            std::future::pending::<()>().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            tracing::info!("Received SIGINT (Ctrl+C), shutting down server gracefully...");
        }
        () = terminate => {
            tracing::info!("Received SIGTERM, shutting down server gracefully...");
        }
    }
}

/// Launch the HTTP server and listen for incoming requests until `SIGINT` or `SIGTERM`.
pub async fn run_server(config: ServerConfig, embedder: GlyprEmbedder) -> Result<()> {
    init_tracing("info");

    let engine = spawn_inference_engine_with_config(embedder, &config);
    let meta = engine.meta.clone();
    let state = Arc::new(AppState { engine });
    let app = build_router_with_config(state, &config);

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid host/port configuration: {}", e))?;

    tracing::info!(
        host = %config.host,
        port = config.port,
        model_id = %config.model_id,
        device = %meta.device,
        multimodal = meta.has_multimodal,
        ane_vision = meta.ane_enabled,
        ane_text = meta.ane_text_enabled,
        allow_local_files = config.allow_local_files,
        max_body_bytes = config.max_body_bytes,
        max_batch_items = config.max_batch_items,
        "Starting glypr embedding server on http://{}",
        addr
    );
    tracing::info!(
        "  POST http://{}/v1/embeddings (OpenAI Embeddings API)",
        addr
    );
    tracing::info!("  GET  http://{}/v1/models     (Model metadata)", addr);
    tracing::info!("  GET  http://{}/health        (Health check)", addr);
    tracing::info!("  GET  http://{}/ready         (Readiness probe)", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}
