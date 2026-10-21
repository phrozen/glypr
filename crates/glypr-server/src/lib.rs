pub mod engine;
pub mod handlers;
pub mod routes;
pub mod schema;

use anyhow::Result;
use glypr_core::GlyprEmbedder;
use std::net::SocketAddr;
use std::sync::Arc;

pub use engine::{EngineHandle, ModelMeta, OwnedTask, spawn_inference_engine};
pub use handlers::AppState;
pub use routes::build_router;
pub use schema::*;

/// Server configuration options.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_batch_size: usize,
    pub model_id: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_batch_size: 32,
            model_id: "embeddinggemma-2".to_string(),
        }
    }
}

/// Launch the HTTP server and listen for incoming requests.
pub async fn run_server(config: ServerConfig, embedder: GlyprEmbedder) -> Result<()> {
    let engine = spawn_inference_engine(embedder, config.max_batch_size, config.model_id.clone());
    let state = Arc::new(AppState { engine });
    let app = build_router(state);

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid host/port configuration: {}", e))?;

    println!("🚀 Starting glypr embedding server on http://{}", addr);
    println!(
        "   - POST http://{}/v1/embeddings (OpenAI Embeddings API)",
        addr
    );
    println!("   - GET  http://{}/v1/models     (Model metadata)", addr);
    println!("   - GET  http://{}/health        (Health check)", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
