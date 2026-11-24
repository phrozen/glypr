# Glypr (`glypr` / `glypr-core`)

[![CI](https://github.com/phrozen/glypr/actions/workflows/ci.yml/badge.svg)](https://github.com/phrozen/glypr/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/glypr-core.svg)](https://crates.io/crates/glypr-core)
[![Docs.rs](https://img.shields.io/docsrs/glypr-core)](https://docs.rs/glypr-core)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org)

**High-performance, self-contained inference engine in native Rust for Google DeepMind's EmbeddingGemma 2.**

`glypr` embeds **text, code, images, and audio** into a unified 768-dimensional vector space from a single self-contained `.gguf` file—no Python, PyTorch, `config.json`, or `tokenizer.json` required at runtime. It runs across **Apple Silicon (Metal 4 GPU & Apple Neural Engine)** and **Linux/macOS CPUs (NEON, AVX2, AVX-512)**.

---

## Key Features

- **Single-File GGUF Deployment:** Reconstructs model hyperparameters and the 262,144-token Gemma 4 BPE tokenizer directly from `embeddinggemma-2-BF16.gguf` (`532.1 MB`) or `embeddinggemma-2-Q8_0.gguf` (`295.5 MB`) — see [Deployment & Runtimes](docs/deployment/quantization-and-runtimes.md).
- **Apple Metal 4 & FlashAttention:** Hardware-fused Metal kernels with tile-skipping sliding windows, zero-copy GQA, native `Q8_0` dequantization, and pipeline caching (`66.7 ms` cold start) — see [Hardware Acceleration](docs/architecture/hardware-acceleration.md).
- **Apple Neural Engine (ANE) Offload:** Routes compiled multi-aspect Core ML vision grids (`140-*`, `280-*`) and `[64..=seq_len]` pure-text buckets (`batch <= 4`) to ANE via `--features metal,ane` — see [Core ML Exporter](scripts/README.md).
- **Portable Cross-Platform CPU SIMD:** Runtime-dispatched `pulp` kernels (ARM64 NEON, x86_64 AVX2/AVX-512) with Rayon parallelism and optional Apple Accelerate (`--features accelerate`), delivering **8.26× faster GeGLU** (`0.676 ms` vs Candle `5.583 ms`) — see [Specifications](docs/architecture/specifications.md).
- **On-Demand Multimodal & Vector Compression:** Attach `mmproj-*.gguf` dynamically for `.png`/`.jpg`/`.webp` images (budgets `16–2048`) and WAV audio (anti-aliased FIR resampling), plus Matryoshka (`MRL`) truncation (`512d`, `256d`, `128d`) and static `INT8` / `1-bit binary` quantization — see [Inputs & Budgeting](docs/multimodal/inputs-and-budgeting.md) & [Vector Quantization](docs/vector-optimization/quantization.md).
- **OpenAI-Compatible Server & Fluent Rust SDK:** Dynamic length-bucketed batching, `/health` and `/ready` probes, graceful shutdown, `GlyprEmbedder::builder()`, and structured `GlyprError` / `GlyprResult` APIs — see [HTTP API Reference](docs/api/embeddings-schema.md).

---

## Performance at a Glance

Measured in `v0.4.0` release builds on Apple Silicon (`compare_models`, `bench_memory`, `bench_multimodal`) against the PyTorch MPS (`bfloat16`) reference:

| Workload | PyTorch MPS *(ref)* | Glypr Metal 4 (`Q8_0` / `BF16`) | Highlight |
| :--- | :---: | :---: | :--- |
| **Text @ 512t** | 24.65 ms | **22.41 ms** / 27.03 ms | `22,851 tok/s` (`Q8_0`) · ANE ref `8.27 ms` |
| **Text @ 8,192t** | 845.83 ms | **564.91 ms** / 620.81 ms | **1.50× faster** vs MPS (`14,502 tok/s`) |
| **Peak Metal VRAM @ 8K** | 6,592.8 MB | **1,102.8 MB** / 1,177.5 MB | **> 5.4 GB saved** (`662.8 MB` RSS on Q8_0) |
| **CPU SIMD @ 512t** *(`--features accelerate`)* | — | **284.1 ms** BF16 / 539.4 ms Q8_0 | `1,802 tok/s` on CPU (Rayon + `pulp`) |
| **Vision (140b) & Audio (12.3s)** | 74.66 ms *(vision)* | **5.51 ms** Vision tower / **8.33 ms** Audio tower | `17.65 cards/s` / **`237.2×` real-time** audio |
| **Golden Parity (292-vec vs FP32)** | 1.000000 *(Oracle)* | **0.999967** avg (BF16) / **0.999896** avg (Q8_0) | Min `0.999897` (BF16) · `0.999741` (Q8_0) |

> See [Leaderboards & Hardware Benchmarks](docs/benchmarks/leaderboards.md) and [Memory & VRAM Architecture](docs/benchmarks/memory-and-vram.md) for full `1K–8K` sweeps, `B=8` multimodal batching, HTTP server throughput (**`1,226.9 req/s`** at 32 clients), and quantization benchmarks.

---

## Installation

```bash
# Install CLI + server from crates.io (Apple Silicon Metal + ANE)
cargo install glypr --features metal,ane

# Or install with universal CPU SIMD (Linux / macOS)
cargo install glypr

# Or build & install from a local checkout
cargo install --path crates/glypr --features metal,ane
```

| Cargo Feature | Capability | Target |
| :--- | :--- | :--- |
| *(default)* | Portable `pulp` CPU SIMD (NEON / AVX2 / AVX-512) + Rayon | Linux, macOS |
| `metal` | Metal GPU kernels, NAX FlashAttention, zero-copy GQA | Apple Silicon |
| `ane` / `coreml` | Apple Neural Engine execution via `coreml-native` (`ane` enables `coreml`) | Apple Silicon |
| `accelerate` | Apple Accelerate framework BLAS & `vvtanhf` | macOS |

---

## Quickstart

### CLI (`glypr`)

```bash
# Pre-fetch canonical BF16 & Q8_0 GGUF weights from Hugging Face
glypr pull

# Single query, Q8_0 MRL truncation, batch STDIN, and JSONL streaming
glypr embed "What is quantum mechanics?" --task SearchQuery
glypr --q8 embed "What is quantum mechanics?" --dim 256 -o json
printf "First query\nSecond query\n" | glypr embed --batch -o json
printf '{"text":"What causes aurora?","task":"SearchQuery"}\n' | glypr embed --jsonl

# Multimodal (opt-in mmproj) and Apple Neural Engine (--ane)
glypr --bf16-mmproj embed "Card illustration:" ./scene.png --image-token-budget 140
glypr --ane embed ./portrait.png --image-token-budget 140
```

See [`scripts/README.md`](scripts/README.md) to export Core ML `.mlmodelc` vision/text bundles for `--ane`.

### OpenAI-Compatible HTTP Server (`glypr serve`)

```bash
# Start server on port 8080 (add --ane or --bf16-mmproj as needed)
glypr serve --port 8080 --max-batch-size 32

curl -s http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Vector search engine", "dimensions": 256, "encoding_format": "int8"}'
```

Local filesystem paths over HTTP are disabled by default (`--allow-local-files` opts in). Full endpoints and payloads are documented in [HTTP API Reference](docs/api/embeddings-schema.md).

### Rust SDK (`glypr-core`)

```toml
[dependencies]
glypr-core = { version = "0.4.0", features = ["metal"] }
```

```rust
use glypr_core::{
    AttentionMode, GlyprEmbedder, GlyprResult, Task, cosine_similarity, truncate_mrl,
};

fn main() -> GlyprResult<()> {
    let embedder = GlyprEmbedder::builder()
        .gguf_canonical(false) // false = BF16, true = Q8_0
        .attention_mode(AttentionMode::MetalFlash)
        .build()?;

    let query = embedder.embed_text("What causes gravitational lensing?", Task::SearchQuery)?;
    let doc = embedder.embed_text(
        "Massive celestial objects bend spacetime, curving light.",
        Task::Document { title: Some("General Relativity") },
    )?;

    let q_256 = truncate_mrl(&query, 256)?;
    let d_256 = truncate_mrl(&doc, 256)?;
    println!("Cosine similarity (256d): {:.4}", cosine_similarity(&q_256, &d_256));
    Ok(())
}
```

See [docs.rs/glypr-core](https://docs.rs/glypr-core) for API docs and [Vector Quantization](docs/vector-optimization/quantization.md) for `quantize_int8` / `quantize_binary` utilities.

---

## Documentation

Full guides and developer documentation live in [`docs/`](docs/README.md):

| Area | Guides |
| :--- | :--- |
| **Overview & API** | [Documentation Index](docs/README.md) · [HTTP Embeddings API Schema](docs/api/embeddings-schema.md) |
| **Architecture** | [Specifications](docs/architecture/specifications.md) · [Hardware Acceleration (Metal / ANE / SIMD)](docs/architecture/hardware-acceleration.md) · [Modularity](docs/architecture/modularity.md) |
| **Benchmarks** | [Leaderboards & Scaling](docs/benchmarks/leaderboards.md) · [Memory, VRAM & Pareto Frontier](docs/benchmarks/memory-and-vram.md) |
| **Deployment & ANE** | [Quantization & Runtimes](docs/deployment/quantization-and-runtimes.md) · [Core ML / ANE Exporter](scripts/README.md) |
| **Multimodal & Prompts** | [Projector Architecture (`mmproj`)](docs/multimodal/mmproj-architecture.md) · [Inputs & Token Budgeting](docs/multimodal/inputs-and-budgeting.md) · [GGUF Tensor Mapping](docs/multimodal/gguf-nuances-and-conversions.md) · [Task Instructions](docs/prompts/task-instructions.md) |
| **Vector Optimization** | [Matryoshka Representation Learning (MRL)](docs/vector-optimization/matryoshka-mrl.md) · [Static INT8 & Binary Quantization](docs/vector-optimization/quantization.md) |

---

## Contributing, Changelog & License

- **Contributing & QA Suite:** See [CONTRIBUTING.md](CONTRIBUTING.md) for formatting, Clippy, Rustdoc, and the 3-tier parity test suite.
- **Release History:** See [CHANGELOG.md](CHANGELOG.md).
- **Security:** See [SECURITY.md](SECURITY.md).
- **License:** [Apache License 2.0](LICENSE).
