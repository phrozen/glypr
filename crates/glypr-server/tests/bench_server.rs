use anyhow::Result;
use candle_core::{DType, Device};
use glypr_core::GlyprEmbedder;
use glypr_server::{build_router, spawn_inference_engine, AppState};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

#[tokio::test]
async fn bench_http_server_latencies() -> Result<()> {
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

    let _server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();

    // Warmup request
    let warmup_payload = json!({ "input": "Warmup request" });
    let _ = client
        .post(format!("{}/v1/embeddings", base_url))
        .json(&warmup_payload)
        .send()
        .await?;

    println!("\n================================================================================");
    println!("  GLYPR HTTP SERVER BENCHMARK (Metal BF16 / Apple Silicon)");
    println!("================================================================================\n");

    // 1. Single Query Latency (Short text: ~8 tokens)
    let iters = 20;
    let mut latencies_single = Vec::with_capacity(iters);
    let payload_single = json!({ "input": "What is the capital of France?" });

    for _ in 0..iters {
        let t0 = Instant::now();
        let resp = client
            .post(format!("{}/v1/embeddings", base_url))
            .json(&payload_single)
            .send()
            .await?;
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        latencies_single.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    latencies_single.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_single = latencies_single[iters / 2];
    let p95_single = latencies_single[(iters as f64 * 0.95) as usize];
    let min_single = latencies_single[0];

    println!("1. End-to-End Single Query Latency (Short query, 768d float):");
    println!("   - Min:    {:.2} ms", min_single);
    println!("   - Median: {:.2} ms", p50_single);
    println!("   - P95:    {:.2} ms\n", p95_single);

    // 2. Encoding Formats Comparison (Payload serialization & transfer)
    println!("2. Encoding Format End-to-End Latency Comparison (768d, single query):");
    for (name, fmt) in [
        ("float  (JSON array)", "float"),
        ("base64 (IEEE-754 bytes)", "base64"),
        ("int8   (JSON [-127,127])", "int8"),
        ("binary (base64 packed bits)", "binary"),
    ] {
        let payload = json!({
            "input": "Vector database nearest neighbor search",
            "encoding_format": fmt
        });
        let mut fmt_latencies = Vec::new();
        for _ in 0..10 {
            let t0 = Instant::now();
            let resp = client
                .post(format!("{}/v1/embeddings", base_url))
                .json(&payload)
                .send()
                .await?;
            assert_eq!(resp.status(), reqwest::StatusCode::OK);
            fmt_latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        fmt_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("   - {:<28}: Median {:.2} ms (Min {:.2} ms)", name, fmt_latencies[5], fmt_latencies[0]);
    }
    println!();

    // 3. Matryoshka Representation Learning (MRL) Dimensions
    println!("3. MRL Truncation End-to-End Latency (single query):");
    for dim in [768, 512, 256, 128] {
        let payload = json!({
            "input": "Benchmarking embedding dimensions with MRL",
            "dimensions": dim
        });
        let mut dim_latencies = Vec::new();
        for _ in 0..10 {
            let t0 = Instant::now();
            let resp = client
                .post(format!("{}/v1/embeddings", base_url))
                .json(&payload)
                .send()
                .await?;
            assert_eq!(resp.status(), reqwest::StatusCode::OK);
            dim_latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        dim_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("   - {:<4}d: Median {:.2} ms (Min {:.2} ms)", dim, dim_latencies[5], dim_latencies[0]);
    }
    println!();

    // 4. Batch Size Scaling
    println!("4. In-Request Batch Size Scaling:");
    for batch_size in [1, 4, 8, 16, 32] {
        let texts: Vec<String> = (0..batch_size)
            .map(|i| format!("Sample text item number {} for batch evaluation", i))
            .collect();
        let payload = json!({ "input": texts });

        let mut batch_times = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            let resp = client
                .post(format!("{}/v1/embeddings", base_url))
                .json(&payload)
                .send()
                .await?;
            assert_eq!(resp.status(), reqwest::StatusCode::OK);
            batch_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        batch_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = batch_times[2];
        let per_vec = med / (batch_size as f64);
        let throughput = (batch_size as f64) / (med / 1000.0);
        println!(
            "   - Batch {:>2}: {:>6.2} ms total | {:>5.2} ms/vec | {:>6.1} vec/s",
            batch_size, med, per_vec, throughput
        );
    }
    println!();

    // 5. Concurrent Clients (Adaptive Dynamic Batching)
    println!("5. Concurrent Client Throughput (Adaptive Dynamic Batching):");
    for num_clients in [4, 8, 16, 32] {
        let t0 = Instant::now();
        let mut handles = Vec::new();

        for i in 0..num_clients {
            let c = client.clone();
            let u = base_url.clone();
            handles.push(tokio::spawn(async move {
                let payload = json!({
                    "input": format!("Concurrent client request {}", i)
                });
                let req_t0 = Instant::now();
                let resp = c
                    .post(format!("{}/v1/embeddings", u))
                    .json(&payload)
                    .send()
                    .await?;
                assert_eq!(resp.status(), reqwest::StatusCode::OK);
                Ok::<f64, anyhow::Error>(req_t0.elapsed().as_secs_f64() * 1000.0)
            }));
        }

        let mut client_latencies = Vec::new();
        for h in handles {
            client_latencies.push(h.await??);
        }
        let total_wall = t0.elapsed().as_secs_f64();
        client_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_lat = client_latencies[num_clients / 2];
        let throughput = (num_clients as f64) / total_wall;

        println!(
            "   - {:>2} concurrent clients: Wall {:>5.2}s | Median Latency: {:>5.2} ms | Throughput: {:>5.1} req/s",
            num_clients, total_wall, median_lat, throughput
        );
    }

    println!("\n================================================================================\n");

    Ok(())
}
