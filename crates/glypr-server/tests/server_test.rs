use anyhow::Result;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use candle_core::{DType, Device};
use glypr_core::GlyprEmbedder;
use glypr_server::{
    build_router, spawn_inference_engine, AppState, HealthResponse, ModelListResponse,
};
use serde_json::json;
use std::sync::Arc;

async fn start_test_server() -> Result<(String, tokio::task::JoinHandle<()>)> {
    let device = if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    let dtype = if device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };

    let model_path = glypr_core::hub::resolve_default_gguf()?;
    let mut embedder = GlyprEmbedder::load_gguf(&model_path, &device, dtype)?;
    embedder.set_attention_mode(glypr_core::AttentionMode::ChunkedSliding);

    let engine = spawn_inference_engine(embedder, 32, "embeddinggemma-2".to_string());
    let state = Arc::new(AppState { engine });
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let base_url = format!("http://{}", addr);

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    Ok((base_url, handle))
}

#[tokio::test]
async fn test_server_health_and_models() -> Result<()> {
    let (base_url, _server) = match start_test_server().await {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Skipping test: model not available ({})", e);
            return Ok(());
        }
    };

    let client = reqwest::Client::new();

    // 1. Test /health
    let resp = client.get(format!("{}/health", base_url)).send().await?;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let health: HealthResponse = resp.json().await?;
    assert_eq!(health.status, "ok");
    assert_eq!(health.model, "embeddinggemma-2");

    // 2. Test /v1/models
    let resp = client.get(format!("{}/v1/models", base_url)).send().await?;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let models: ModelListResponse = resp.json().await?;
    assert_eq!(models.object, "list");
    assert_eq!(models.data.len(), 1);
    assert_eq!(models.data[0].id, "embeddinggemma-2");

    Ok(())
}

#[tokio::test]
async fn test_server_embeddings_minimal_request() -> Result<()> {
    let (base_url, _server) = match start_test_server().await {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Skipping test: model not available ({})", e);
            return Ok(());
        }
    };

    let client = reqwest::Client::new();

    // Minimal request: only input specified, all other fields omitted
    let payload = json!({
        "input": "Deep learning embedding retrieval"
    });

    let resp = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&payload)
        .send()
        .await?;

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await?;

    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "embeddinggemma-2");
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert_eq!(body["data"][0]["index"], 0);

    let embedding = body["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(embedding.len(), 768);

    Ok(())
}

#[tokio::test]
async fn test_server_embeddings_batch_and_mrl() -> Result<()> {
    let (base_url, _server) = match start_test_server().await {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Skipping test: model not available ({})", e);
            return Ok(());
        }
    };

    let client = reqwest::Client::new();

    // Batch input with MRL dimension 256
    let payload = json!({
        "input": ["First sentence for test", "Second sentence for test"],
        "dimensions": 256,
        "task": "SearchQuery"
    });

    let resp = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&payload)
        .send()
        .await?;

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await?;

    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);

    let emb0 = data[0]["embedding"].as_array().unwrap();
    let emb1 = data[1]["embedding"].as_array().unwrap();
    assert_eq!(emb0.len(), 256);
    assert_eq!(emb1.len(), 256);

    Ok(())
}

#[tokio::test]
async fn test_server_embeddings_quantization_formats() -> Result<()> {
    let (base_url, _server) = match start_test_server().await {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Skipping test: model not available ({})", e);
            return Ok(());
        }
    };

    let client = reqwest::Client::new();

    // 1. Base64 format
    let payload_b64 = json!({
        "input": "Base64 encoding test",
        "encoding_format": "base64"
    });
    let resp = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&payload_b64)
        .send()
        .await?;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await?;
    let b64_str = body["data"][0]["embedding"].as_str().unwrap();
    let decoded_bytes = BASE64.decode(b64_str)?;
    assert_eq!(decoded_bytes.len(), 768 * 4); // 768 f32 floats = 3072 bytes

    // 2. Int8 format
    let payload_int8 = json!({
        "input": "Int8 encoding test",
        "encoding_format": "int8"
    });
    let resp = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&payload_int8)
        .send()
        .await?;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await?;
    let ints = body["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(ints.len(), 768);
    for val in ints {
        let i = val.as_i64().unwrap();
        assert!(i >= -127 && i <= 127);
    }

    // 3. Binary format (packed u64 bit bytes as base64 string)
    let payload_bin = json!({
        "input": "Binary encoding test",
        "encoding_format": "binary"
    });
    let resp = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&payload_bin)
        .send()
        .await?;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await?;
    let bin_b64 = body["data"][0]["embedding"].as_str().unwrap();
    let bin_bytes = BASE64.decode(bin_b64)?;
    assert_eq!(bin_bytes.len(), 768 / 8); // 768 bits = 96 bytes

    Ok(())
}

#[tokio::test]
async fn test_server_concurrent_requests_batching() -> Result<()> {
    let (base_url, _server) = match start_test_server().await {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Skipping test: model not available ({})", e);
            return Ok(());
        }
    };

    let client = reqwest::Client::new();

    // Fire 12 concurrent requests simultaneously to stress adaptive batching
    let mut tasks = Vec::new();
    for i in 0..12 {
        let c = client.clone();
        let u = base_url.clone();
        tasks.push(tokio::spawn(async move {
            let payload = json!({
                "input": format!("Concurrent query item number {}", i),
                "dimensions": 128
            });
            let resp = c
                .post(format!("{}/v1/embeddings", u))
                .json(&payload)
                .send()
                .await?;
            assert_eq!(resp.status(), reqwest::StatusCode::OK);
            let body: serde_json::Value = resp.json().await?;
            let emb = body["data"][0]["embedding"].as_array().unwrap();
            assert_eq!(emb.len(), 128);
            Ok::<(), anyhow::Error>(())
        }));
    }

    for task in tasks {
        task.await??;
    }

    Ok(())
}

#[tokio::test]
async fn test_server_error_handling() -> Result<()> {
    let (base_url, _server) = match start_test_server().await {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Skipping test: model not available ({})", e);
            return Ok(());
        }
    };

    let client = reqwest::Client::new();

    // 1. Invalid dimensions
    let payload_dim = json!({
        "input": "Error test",
        "dimensions": 9999
    });
    let resp = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&payload_dim)
        .send()
        .await?;
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: serde_json::Value = resp.json().await?;
    assert_eq!(body["error"]["type"], "invalid_request_error");

    // 2. Empty input array
    let payload_empty = json!({
        "input": []
    });
    let resp = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&payload_empty)
        .send()
        .await?;
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);

    Ok(())
}

#[tokio::test]
async fn test_server_embeddings_multimodal_request() -> Result<()> {
    let device = if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    let dtype = if device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };

    let base_gguf = match glypr_core::hub::resolve_default_gguf() {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };
    let mmproj_gguf = match glypr_core::hub::resolve_default_mmproj() {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };

    if !base_gguf.exists() || !mmproj_gguf.exists() {
        eprintln!("Checkpoints not found, skipping multimodal test");
        return Ok(());
    }

    let mut embedder = GlyprEmbedder::load_gguf(&base_gguf, &device, dtype)?;
    embedder.set_attention_mode(glypr_core::AttentionMode::ChunkedSliding);
    embedder.attach_multimodal_projector(&mmproj_gguf)?;

    let engine = spawn_inference_engine(embedder, 32, "embeddinggemma-2".to_string());
    let state = Arc::new(AppState { engine });
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let base_url = format!("http://{}", addr);

    let _handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();

    let img_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/golden_vectors/multimodal_test_image.png");
    if !img_path.exists() {
        eprintln!("multimodal_test_image.png not found, skipping");
        return Ok(());
    }

    let img_bytes = std::fs::read(&img_path)?;
    let img_b64 = format!("data:image/png;base64,{}", BASE64.encode(&img_bytes));

    let payload = json!({
        "input": [
            {
                "text": "Describe this test picture",
                "image": img_b64,
                "budget": 70
            }
        ]
    });

    let resp = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&payload)
        .send()
        .await?;

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await?;
    let emb = body["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(emb.len(), 768);

    Ok(())
}
