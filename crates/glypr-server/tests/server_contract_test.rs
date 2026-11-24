use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use glypr_core::l2_normalize;
use glypr_server::{
    AppState, EngineHandle, ModelMeta, OwnedTask, ServerConfig, build_router,
    build_router_with_config, engine::PreparedItem,
};
use serde_json::{Value, json};
use std::io::Cursor;
use std::sync::Arc;
use tower::ServiceExt;

fn make_meta(has_multimodal: bool, has_audio_projector: bool, patches: Vec<usize>) -> ModelMeta {
    ModelMeta {
        id: "embeddinggemma-2-mock".to_string(),
        device: "cpu".to_string(),
        has_multimodal,
        has_audio_projector,
        vision_backend: if has_multimodal {
            "metal".to_string()
        } else {
            "none".to_string()
        },
        text_backend: "cpu".to_string(),
        ane_enabled: !patches.is_empty(),
        ane_supported_patches: patches,
        ane_text_enabled: false,
        ane_text_seq_len: None,
        embedding_dim: 768,
        max_seq_len: 8320,
        created_at: 1_700_000_000,
    }
}

fn deterministic_vec(seed: u64, truncate_dim: Option<usize>) -> Vec<f32> {
    let dim = truncate_dim.unwrap_or(768);
    let raw: Vec<f32> = (0..dim)
        .map(|i| (((seed as f32) * 0.13 + (i as f32) * 0.07).sin()) + 0.01)
        .collect();
    l2_normalize(&raw)
}

fn hash_label(s: &str) -> u64 {
    s.bytes().fold(1469598103934665603u64, |acc, b| {
        acc.wrapping_mul(1099511628211).wrapping_add(b as u64)
    })
}

fn mock_engine(
    has_multimodal: bool,
    has_audio_projector: bool,
    allow_local_files: bool,
    patches: Vec<usize>,
) -> EngineHandle {
    let meta = make_meta(has_multimodal, has_audio_projector, patches);
    EngineHandle::mock(meta, allow_local_files, 64, |item, task, truncate_dim| {
        let (tag, tok_count) = match item {
            PreparedItem::Text(s) => (
                format!("txt:{task:?}:{s}"),
                s.split_whitespace().count().max(1),
            ),
            PreparedItem::Image { text, budget, .. } => (
                format!(
                    "img:{task:?}:{}:{}",
                    text.as_deref().unwrap_or(""),
                    budget.unwrap_or(280)
                ),
                128,
            ),
            PreparedItem::Audio { samples, text } => (
                format!(
                    "aud:{task:?}:{}:{}",
                    text.as_deref().unwrap_or(""),
                    samples.len()
                ),
                32,
            ),
        };
        let vec = deterministic_vec(hash_label(&tag), truncate_dim);
        Ok((vec, tok_count))
    })
}

async fn send_json(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let builder = Request::builder().method(method).uri(uri);
    let req = match body {
        Some(v) => builder
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&v).unwrap()))
            .unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    };
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    (status, json)
}

fn tiny_png_base64() -> String {
    let img = image::RgbImage::from_pixel(16, 16, image::Rgb([12, 34, 56]));
    let mut cursor = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .unwrap();
    BASE64.encode(cursor.into_inner())
}

fn tiny_wav_base64() -> String {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Vec::new();
    {
        let mut writer = hound::WavWriter::new(Cursor::new(&mut buf), spec).unwrap();
        for i in 0..400 {
            writer.write_sample(((i % 50) * 300) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }
    BASE64.encode(buf)
}

#[tokio::test]
async fn test_health_ready_and_models_schema() {
    // 1. Empty ane_supported_patches must be skipped from /health JSON
    let app_no_ane = build_router(Arc::new(AppState {
        engine: mock_engine(false, false, false, vec![]),
    }));

    let (status, health) = send_json(&app_no_ane, "GET", "/health", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["status"], "ok");
    assert_eq!(health["ane_text_enabled"], false);
    assert!(
        health.get("ane_supported_patches").is_none(),
        "empty ane_supported_patches should be skipped: {health}"
    );
    assert!(health.get("ane_text_seq_len").is_none());

    let (status, ready) = send_json(&app_no_ane, "GET", "/ready", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ready["status"], "ready");
    assert_eq!(ready["model"], "embeddinggemma-2-mock");

    let (status, models) = send_json(&app_no_ane, "GET", "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["object"], "list");
    let m0 = &models["data"][0];
    assert_eq!(m0["id"], "embeddinggemma-2-mock");
    assert_eq!(m0["ane_text_enabled"], false);
    assert_eq!(m0["embedding_dim"], 768);
    assert_eq!(m0["max_context"], 8320);

    // 2. Non-empty ane_supported_patches must appear in /health JSON
    let app_ane = build_router(Arc::new(AppState {
        engine: mock_engine(true, false, false, vec![1134, 2394]),
    }));
    let (status, health_ane) = send_json(&app_ane, "GET", "/health", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health_ane["ane_enabled"], true);
    assert_eq!(health_ane["ane_supported_patches"], json!([1134, 2394]));
}

#[tokio::test]
async fn test_embeddings_formats_mrl_tasks_and_batch_ordering() {
    let cfg = ServerConfig {
        max_batch_items: 64,
        ..Default::default()
    };
    let app = build_router_with_config(
        Arc::new(AppState {
            engine: mock_engine(true, true, false, vec![]),
        }),
        &cfg,
    );

    // 1. Single string + MRL 768/256/128 + model override + task Document with title
    for dim in [768usize, 256, 128] {
        let (status, body) = send_json(
            &app,
            "POST",
            "/v1/embeddings",
            Some(json!({
                "input": "gravitational lensing",
                "model": "custom-model-alias",
                "dimensions": dim,
                "task": "Document",
                "title": "Astrophysics"
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["model"], "custom-model-alias");
        assert_eq!(body["usage"]["prompt_tokens"], 2);
        assert_eq!(body["usage"]["total_tokens"], 2);
        let arr = body["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(arr.len(), dim);
    }

    // 2. Encoding formats: float, base64, int8, int (alias), binary
    let (status, b64_resp) = send_json(
        &app,
        "POST",
        "/v1/embeddings",
        Some(json!({
            "input": "hello world",
            "dimensions": 128,
            "encoding_format": "base64",
            "task": "SearchQuery"
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let b64_bytes = BASE64
        .decode(b64_resp["data"][0]["embedding"].as_str().unwrap())
        .unwrap();
    assert_eq!(b64_bytes.len(), 128 * 4);

    for fmt in ["int8", "int"] {
        let (status, int8_resp) = send_json(
            &app,
            "POST",
            "/v1/embeddings",
            Some(json!({
                "input": "hello world",
                "dimensions": 128,
                "encoding_format": fmt,
                "task": "Raw"
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let q = int8_resp["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(q.len(), 128);
        assert!(q.iter().all(|v| {
            let n = v.as_i64().unwrap();
            (-127..=127).contains(&n)
        }));
    }

    let (status, bin_resp) = send_json(
        &app,
        "POST",
        "/v1/embeddings",
        Some(json!({
            "input": "hello world",
            "dimensions": 256,
            "encoding_format": "binary"
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let bin_bytes = BASE64
        .decode(bin_resp["data"][0]["embedding"].as_str().unwrap())
        .unwrap();
    assert_eq!(bin_bytes.len(), (256 / 64) * 8);

    // 3. Batch array index ordering & usage sum + multimodal base64 image & audio
    let png_uri = format!("data:image/png;base64,{}", tiny_png_base64());
    let wav_uri = format!("data:audio/wav;base64,{}", tiny_wav_base64());
    let (status, batch_resp) = send_json(
        &app,
        "POST",
        "/v1/embeddings",
        Some(json!({
            "input": [
                "first query",
                "second query with more words",
                { "text": "describe image", "image": png_uri, "budget": 70 },
                { "audio": wav_uri }
            ],
            "dimensions": 128
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let data = batch_resp["data"].as_array().unwrap();
    assert_eq!(data.len(), 4);
    for (idx, item) in data.iter().enumerate() {
        assert_eq!(item["index"], idx);
        assert_eq!(item["embedding"].as_array().unwrap().len(), 128);
    }
    assert_eq!(batch_resp["usage"]["prompt_tokens"], 2 + 5 + 128 + 32);
    assert_eq!(batch_resp["usage"]["total_tokens"], 2 + 5 + 128 + 32);
}

#[tokio::test]
async fn test_embeddings_400_bad_request_matrix() {
    let app_text_only = build_router(Arc::new(AppState {
        engine: mock_engine(false, false, false, vec![]),
    }));
    let app_vision_only = build_router(Arc::new(AppState {
        engine: mock_engine(true, false, false, vec![1134]),
    }));
    let app_full = build_router(Arc::new(AppState {
        engine: mock_engine(true, true, false, vec![]),
    }));

    let png_uri = format!("data:image/png;base64,{}", tiny_png_base64());
    let wav_uri = format!("data:audio/wav;base64,{}", tiny_wav_base64());

    let cases: Vec<(&Router, Value, &str)> = vec![
        (
            &app_full,
            json!({"input": "x", "dimensions": 0}),
            "dimensions 0",
        ),
        (
            &app_full,
            json!({"input": "x", "dimensions": 769}),
            "dimensions 769",
        ),
        (&app_full, json!({"input": []}), "empty input array"),
        (
            &app_full,
            json!({"input": [{"text": null, "image": null, "audio": null}]}),
            "empty media object",
        ),
        (
            &app_full,
            json!({"input": [{"image": png_uri, "audio": wav_uri}]}),
            "image+audio conflict",
        ),
        (
            &app_text_only,
            json!({"input": [{"image": png_uri}]}),
            "multimodal rejected when has_multimodal == false",
        ),
        (
            &app_vision_only,
            json!({"input": [{"audio": wav_uri}]}),
            "audio rejected when has_audio_projector == false",
        ),
        (
            &app_full,
            json!({"input": [{"image": "data:image/png,not-base64"}]}),
            "invalid data URI without ;base64,",
        ),
        (
            &app_full,
            json!({"input": [{"image": "tests/golden_vectors/dog.png"}]}),
            "local path blocked when allow_local_files == false",
        ),
        (
            &app_full,
            json!({"input": [{"image": format!("data:image/png;base64,{}", BASE64.encode(b"not a png"))}]}),
            "corrupt base64 image",
        ),
        (
            &app_full,
            json!({"input": [{"audio": format!("data:audio/wav;base64,{}", BASE64.encode(b"not a wav"))}]}),
            "corrupt base64 WAV",
        ),
        (
            &app_full,
            json!({"input": "x", "encoding_format": "binary", "dimensions": 100}),
            "binary with non-multiple-of-64 dimensions",
        ),
    ];

    for (router, payload, label) in cases {
        let (status, err_json) = send_json(router, "POST", "/v1/embeddings", Some(payload)).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "case '{label}' got {err_json}"
        );
        assert_eq!(
            err_json["error"]["type"], "invalid_request_error",
            "case '{label}' got {err_json}"
        );
    }

    // Malformed JSON must also map to 400 invalid_request_error via OpenAiJson
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from("{not valid json"))
        .unwrap();
    let resp = app_full.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let err_json: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), 1024 * 64).await.unwrap()).unwrap();
    assert_eq!(err_json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn test_32_way_concurrent_burst_preserves_caller_indices() {
    let app = Arc::new(build_router(Arc::new(AppState {
        engine: mock_engine(false, false, false, vec![]),
    })));

    let mut handles = Vec::with_capacity(32);
    for idx in 0..32usize {
        let router = Arc::clone(&app);
        handles.push(tokio::spawn(async move {
            let input_text = format!("concurrent-caller-{idx}");
            let (status, body) = send_json(
                &router,
                "POST",
                "/v1/embeddings",
                Some(json!({
                    "input": input_text,
                    "dimensions": 128,
                    "task": "SearchQuery"
                })),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            let got: Vec<f32> = body["data"][0]["embedding"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let expected = deterministic_vec(
                hash_label(&format!("txt:{:?}:{}", OwnedTask::SearchQuery, input_text)),
                Some(128),
            );
            assert_eq!(got.len(), expected.len());
            for (a, b) in got.iter().zip(expected.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "caller {idx} received mismatched embedding"
                );
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }
}
