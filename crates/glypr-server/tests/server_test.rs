use anyhow::Result;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use candle_core::{DType, Device};
use glypr_core::GlyprEmbedder;
use glypr_server::{
    AppState, HealthResponse, ModelListResponse, build_router, spawn_inference_engine,
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

    // 2. Int8 format (testing both "int8" and "int")
    for format_name in ["int8", "int"] {
        let payload_int = json!({
            "input": "Int8 encoding test",
            "encoding_format": format_name
        });
        let resp = client
            .post(format!("{}/v1/embeddings", base_url))
            .json(&payload_int)
            .send()
            .await?;
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        let body: serde_json::Value = resp.json().await?;
        let ints = body["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(ints.len(), 768);
        for val in ints {
            let i = val.as_i64().unwrap();
            assert!((-127..=127).contains(&i));
        }
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

    // Verify task conditioning differentiation for multimodal inputs
    // 1. Raw / unprompted image
    let raw_payload = json!({
        "task": "Raw",
        "input": [{ "image": &img_b64, "budget": 70 }]
    });
    let raw_resp: serde_json::Value = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&raw_payload)
        .send()
        .await?
        .json()
        .await?;
    let emb_raw: Vec<f64> = raw_resp["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    // 2. Query task alias
    let query_payload = json!({
        "task": "Query",
        "input": [{ "image": &img_b64, "budget": 70 }]
    });
    let query_resp: serde_json::Value = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&query_payload)
        .send()
        .await?
        .json()
        .await?;
    let emb_query: Vec<f64> = query_resp["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    // 3. Document task with title
    let doc_payload = json!({
        "task": "Document",
        "title": "Black Lotus",
        "input": [{ "image": &img_b64, "budget": 70 }]
    });
    let doc_resp: serde_json::Value = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&doc_payload)
        .send()
        .await?
        .json()
        .await?;
    let emb_doc: Vec<f64> = doc_resp["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    let sim_raw_query = dot(&emb_raw, &emb_query);
    let sim_raw_doc = dot(&emb_raw, &emb_doc);
    let sim_query_doc = dot(&emb_query, &emb_doc);

    // Verify task conditioning produces distinct embeddings (not identical vectors)
    assert!(
        sim_raw_query < 0.99,
        "Raw vs Query similarity too high: {}",
        sim_raw_query
    );
    assert!(
        sim_raw_doc < 0.99,
        "Raw vs Document similarity too high: {}",
        sim_raw_doc
    );
    assert!(
        sim_query_doc < 0.99,
        "Query vs Document similarity too high: {}",
        sim_query_doc
    );

    Ok(())
}

#[tokio::test]
async fn test_server_multimodal_rejected_without_mmproj() -> Result<()> {
    // start_test_server starts a server WITHOUT mmproj
    let (base_url, _server) = match start_test_server().await {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Skipping test: model not available ({})", e);
            return Ok(());
        }
    };

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
                "image": img_b64
            }
        ]
    });

    let resp = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&payload)
        .send()
        .await?;

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: serde_json::Value = resp.json().await?;
    let error_msg = body["error"]["message"].as_str().unwrap();
    assert!(error_msg.contains("started without a multimodal projector"));
    assert!(error_msg.contains("--bf16-mmproj"));

    Ok(())
}

#[tokio::test]
async fn test_server_concurrent_multimodal_batching() -> Result<()> {
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
    let mmproj_gguf = match glypr_core::hub::resolve_canonical_mmproj(false) {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };

    if !base_gguf.exists() || !mmproj_gguf.exists() {
        eprintln!("Checkpoints not found, skipping concurrent multimodal test");
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

    let req_image1 = json!({
        "task": "Raw",
        "input": [{ "image": &img_b64, "budget": 70 }]
    });
    let req_image2 = json!({
        "task": "Query",
        "input": [{ "text": "Describe picture", "image": &img_b64, "budget": 70 }]
    });
    let req_text1 = json!({
        "task": "SearchQuery",
        "input": "High fantasy trading card"
    });
    let req_text2 = json!({
        "task": "Document",
        "title": "Artifact",
        "input": "Black Lotus card rules"
    });

    let payloads = vec![req_image1, req_image2, req_text1, req_text2];
    let mut handles = Vec::new();

    for p in payloads {
        let c = client.clone();
        let u = format!("{}/v1/embeddings", base_url);
        handles.push(tokio::spawn(async move { c.post(u).json(&p).send().await }));
    }

    for handle in handles {
        let resp = handle.await??;
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        let body: serde_json::Value = resp.json().await?;
        let emb = body["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(emb.len(), 768);
    }

    Ok(())
}

fn build_tiny_test_embedder(max_seq_len: usize, with_dummy_mmproj: bool) -> GlyprEmbedder {
    use candle_core::Tensor;
    use candle_core::quantized::gguf_file::Value;
    use glypr_core::_candle_nn::{Embedding, Linear};
    use glypr_core::{
        EmbeddingGemma2Model, LinearOp, MultimodalProjector, TextBackendPolicy,
        VisionBackendPolicy, attention, config, layer, mlp, norm, ple, rope,
        tokenizer::GlyprTokenizer,
    };
    use std::collections::HashMap;

    let hidden_size = 8usize;
    let intermediate_size = 16usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 4usize;
    let embedding_dim = 64usize;
    let hidden_size_per_layer_input = 8usize;
    let vocab_size = 16usize;
    let device = Device::Cpu;
    let dtype = DType::F32;

    let mut rope_parameters = HashMap::new();
    rope_parameters.insert(
        "sliding_attention".to_string(),
        config::RoPEParameters {
            rope_theta: 10_000.0,
            rope_type: "default".to_string(),
        },
    );
    rope_parameters.insert(
        "full_attention".to_string(),
        config::RoPEParameters {
            rope_theta: 1_000_000.0,
            rope_type: "default".to_string(),
        },
    );

    let text_config = config::TextConfig {
        hidden_size,
        intermediate_size,
        num_hidden_layers: 1,
        num_attention_heads: num_heads,
        num_key_value_heads: num_kv_heads,
        head_dim,
        vocab_size,
        embedding_dim,
        max_position_embeddings: 128,
        sliding_window: 512,
        rms_norm_eps: 1e-6,
        hidden_size_per_layer_input,
        layer_types: vec![config::LayerType::SlidingAttention],
        per_layer_config: HashMap::new(),
        rope_parameters,
        pad_token_id: 0,
        bos_token_id: 2,
        eos_token_id: 1,
    };

    let ones = |shape: (usize, usize)| -> LinearOp {
        let w = Tensor::ones(shape, dtype, &device).unwrap();
        LinearOp::new_standard(Linear::new(w, None))
    };
    let norm_w = |len: usize| -> norm::RMSNorm {
        norm::RMSNorm::new(Some(Tensor::ones((len,), dtype, &device).unwrap()), 1e-6)
    };

    let embed_weight =
        candle_core::Tensor::arange(0u32, (vocab_size * hidden_size) as u32, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
            .reshape((vocab_size, hidden_size))
            .unwrap();
    let embed_tokens = Embedding::new(embed_weight, hidden_size);

    let ple = ple::TextPLE::new(
        ones((hidden_size_per_layer_input, hidden_size)),
        (hidden_size as f64).powf(-0.5),
        norm_w(hidden_size_per_layer_input),
        1,
        hidden_size_per_layer_input,
    );

    let layer = layer::EncoderLayer {
        self_attn: attention::SelfAttention::new(
            ones((num_heads * head_dim, hidden_size)),
            ones((num_kv_heads * head_dim, hidden_size)),
            ones((num_kv_heads * head_dim, hidden_size)),
            ones((hidden_size, num_heads * head_dim)),
            norm_w(head_dim),
            norm_w(head_dim),
            norm::RMSNorm::new(None, 1e-6),
            num_heads,
            num_kv_heads,
            head_dim,
            Some(text_config.sliding_window),
        ),
        mlp: mlp::MLP::new(
            ones((intermediate_size, hidden_size)),
            ones((intermediate_size, hidden_size)),
            ones((hidden_size, intermediate_size)),
        ),
        input_layernorm: norm_w(hidden_size),
        post_attention_layernorm: norm_w(hidden_size),
        pre_feedforward_layernorm: norm_w(hidden_size),
        post_feedforward_layernorm: norm_w(hidden_size),
        ple_block: ple::TextPLEBlock::new(
            ones((hidden_size_per_layer_input, hidden_size)),
            ones((hidden_size, hidden_size_per_layer_input)),
            norm_w(hidden_size),
        ),
        layer_scalar: candle_core::Tensor::ones((1,), dtype, &device).unwrap(),
    };

    let model = EmbeddingGemma2Model {
        config: text_config,
        embed_tokens,
        embed_scale: (hidden_size as f64).sqrt(),
        ple,
        layers: vec![layer],
        norm: norm_w(hidden_size),
        embedding_projection: ones((embedding_dim, hidden_size)),
        rope_sliding: rope::RotaryEmbedding::new(head_dim, 10_000.0, &device).unwrap(),
        rope_full: rope::RotaryEmbedding::new(head_dim, 1_000_000.0, &device).unwrap(),
        device: device.clone(),
        dtype,
    };

    let vocab = vec!["<unk>", "</s>", "<s>", "a", "b", "\u{2581}"];
    let mut meta = HashMap::new();
    meta.insert(
        "tokenizer.ggml.tokens".to_string(),
        Value::Array(
            vocab
                .into_iter()
                .map(|s| Value::String(s.to_string()))
                .collect(),
        ),
    );
    meta.insert(
        "tokenizer.ggml.merges".to_string(),
        Value::Array(Vec::new()),
    );
    let tokenizer = GlyprTokenizer::from_gguf_metadata(&meta).unwrap();

    let multimodal = if with_dummy_mmproj {
        Some(MultimodalProjector {
            vision_tower: None,
            audio_tower: None,
            config: config::MultimodalConfig::from_gguf_metadata(&HashMap::new()),
            device: Device::Cpu,
            dtype: DType::F32,
        })
    } else {
        None
    };

    GlyprEmbedder {
        model,
        tokenizer,
        device,
        dtype,
        multimodal,
        max_seq_len,
        #[cfg(any(feature = "coreml", feature = "ane"))]
        coreml_vision: None,
        #[cfg(any(feature = "coreml", feature = "ane"))]
        coreml_text: None,
        vision_backend: VisionBackendPolicy::Auto,
        text_backend: TextBackendPolicy::Auto,
    }
}

fn build_tiny_test_router(
    max_seq_len: usize,
    with_dummy_mmproj: bool,
    max_batch_items: usize,
) -> axum::Router {
    use glypr_server::{
        ServerConfig, build_router_with_config, spawn_inference_engine_with_config,
    };

    let embedder = build_tiny_test_embedder(max_seq_len, with_dummy_mmproj);
    let config = ServerConfig {
        max_batch_size: 32,
        max_batch_items,
        allow_local_files: false,
        ..Default::default()
    };
    let engine = spawn_inference_engine_with_config(embedder, &config);
    let state = Arc::new(AppState { engine });
    build_router_with_config(state, &config)
}

async fn send_router_request(
    app: axum::Router,
    method: &str,
    uri: &str,
    body_str: Option<String>,
) -> Result<(axum::http::StatusCode, serde_json::Value)> {
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let mut builder = Request::builder().method(method).uri(uri);
    let body = if let Some(s) = body_str {
        builder = builder.header("content-type", "application/json");
        Body::from(s)
    } else {
        Body::empty()
    };
    let req = builder.body(body)?;
    let resp = app.oneshot(req).await?;
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await?;
    let val: serde_json::Value = serde_json::from_slice(&bytes)?;
    Ok((status, val))
}

#[tokio::test]
async fn test_server_contract_validation_ready_and_error_json() -> Result<()> {
    let app = build_tiny_test_router(64, true, 4);

    // 1. /ready and /health and /v1/models
    let (status, ready_body) = send_router_request(app.clone(), "GET", "/ready", None).await?;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(ready_body["status"], "ready");

    let (status, health_body) = send_router_request(app.clone(), "GET", "/health", None).await?;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(health_body["status"], "ok");
    assert_eq!(health_body["multimodal"], true);
    assert_eq!(health_body["has_audio_projector"], false);
    assert_eq!(health_body["ane_text_enabled"], false);

    let (status, models_body) = send_router_request(app.clone(), "GET", "/v1/models", None).await?;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(models_body["data"][0]["embedding_dim"], 64);

    // 2. Malformed JSON and invalid encoding_format return 400 OpenAI JSON (not 422)
    let (status, bad_json_body) = send_router_request(
        app.clone(),
        "POST",
        "/v1/embeddings",
        Some("{not valid json".to_string()),
    )
    .await?;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(bad_json_body["error"]["type"], "invalid_request_error");

    let (status, bad_enum_body) = send_router_request(
        app.clone(),
        "POST",
        "/v1/embeddings",
        Some(json!({"input": "hello", "encoding_format": "bogus"}).to_string()),
    )
    .await?;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(bad_enum_body["error"]["type"], "invalid_request_error");

    // 3. Binary with non-multiple-of-64 dimensions rejected upfront (400)
    let (status, _) = send_router_request(
        app.clone(),
        "POST",
        "/v1/embeddings",
        Some(json!({"input": "hello", "encoding_format": "binary", "dimensions": 32}).to_string()),
    )
    .await?;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);

    // 4. MediaInputItem setting both image and audio rejected (400)
    let (status, _) = send_router_request(
        app.clone(),
        "POST",
        "/v1/embeddings",
        Some(json!({"input": {"image": "abc", "audio": "def"}}).to_string()),
    )
    .await?;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);

    // 5. Audio without audio projector rejected (400)
    let (status, _) = send_router_request(
        app.clone(),
        "POST",
        "/v1/embeddings",
        Some(json!({"input": {"audio": "AAAA"}}).to_string()),
    )
    .await?;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);

    // 6. Local file path rejected when allow_local_files == false (400)
    let (status, local_body) = send_router_request(
        app.clone(),
        "POST",
        "/v1/embeddings",
        Some(json!({"input": {"image": "/etc/passwd"}}).to_string()),
    )
    .await?;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert!(
        local_body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("--allow-local-files")
    );

    // 7. max_batch_items enforced (400)
    let (status, _) = send_router_request(
        app,
        "POST",
        "/v1/embeddings",
        Some(json!({"input": ["hello", "hello", "hello", "hello", "hello"]}).to_string()),
    )
    .await?;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);

    Ok(())
}

#[tokio::test]
async fn test_server_batch_error_isolation() -> Result<()> {
    // max_seq_len = 5 tokens
    let app = build_tiny_test_router(5, false, 32);

    let valid_fut = send_router_request(
        app.clone(),
        "POST",
        "/v1/embeddings",
        Some(json!({"input": "a"}).to_string()),
    );

    // > 5 tokens ("abababab" = 10 tokens with BOS/EOS) fails preflight/isolation
    let oversized_fut = send_router_request(
        app,
        "POST",
        "/v1/embeddings",
        Some(json!({"input": "abababab"}).to_string()),
    );

    let (valid_res, oversized_res) = tokio::join!(valid_fut, oversized_fut);
    let (valid_status, valid_body) = valid_res?;
    let (oversized_status, _) = oversized_res?;

    assert_eq!(
        valid_status,
        axum::http::StatusCode::OK,
        "Healthy concurrent request must not be poisoned by oversized sibling"
    );
    assert_eq!(
        valid_body["data"][0]["embedding"].as_array().unwrap().len(),
        64
    );
    assert_eq!(oversized_status, axum::http::StatusCode::BAD_REQUEST);

    Ok(())
}
