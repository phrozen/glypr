# Glypr (`glypr` / `glypr-core`)

[![Crates.io](https://img.shields.io/crates/v/glypr-core.svg)](https://crates.io/crates/glypr-core)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20%7C%20Linux-lightgrey.svg)]()
[![Acceleration](https://img.shields.io/badge/acceleration-Metal%204%20%7C%20ANE%20%7C%20CPU%20SIMD-success.svg)]()

> **High-Performance, 100% Self-Contained Inference Engine in Native Rust for Google DeepMind's EmbeddingGemma 2**

`glypr` is a blazingly fast, hardware-optimized inference engine and embedder for **EmbeddingGemma 2** (270M text, 170M vision, 300M audio). It delivers state-of-the-art throughput across Apple Silicon (Metal 4 GPU & Apple Neural Engine) and x86_64 / ARM64 Linux/macOS via portable CPU SIMD.

Unlike typical runtime wrappers, `glypr` deploys from a **single self-contained `.gguf` file**—extracting all model hyperparameters, layer topologies, and the full 262,144-token Gemma 4 BPE tokenizer directly from GGUF metadata without external Python, PyTorch, `config.json`, or `tokenizer.json` files.

---

## Architectural Highlights

```
                       ┌───────────────────────────────┐
                       │   Single Self-Contained GGUF  │
                       │   (Model + 262k BPE Vocab)    │
                       └──────────────┬────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│ Apple Silicon    │        │ Apple Neural     │        │ Universal        │
│ Metal 4 GPU      │        │ Engine (ANE)     │        │ CPU SIMD         │
├──────────────────┤        ├──────────────────┤        ├──────────────────┤
│ • NAX FlashAttn  │        │ • 11.17ms Vision │        │ • Pulp AVX2/512  │
│ • Zero-copy GQA  │        │ • 8.24ms 512t RAG│        │ • ARM64 NEON     │
│ • 1.42x vs MPS   │        │ • Zero GPU Watts │        │ • Zero Heap Churn│
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

- **100% Self-Contained Single-File Deployment:** Point `glypr` to a single `.gguf` file (`embeddinggemma-2-BF16.gguf` or `embeddinggemma-2-Q8_0.gguf`). Architecture hyperparameters and the complete 262k-token BPE tokenizer are reconstructed in-memory in milliseconds.
- **Apple Neural Engine (ANE) Acceleration:** Transparently offloads multi-aspect vision presets (`140-portrait`, `140-square`, `140-landscape`, `280-*`) and `[64..=seq_len]` pure-text buckets (`batch <= 4`) directly to the Apple Neural Engine via `coreml-native` (`--features metal,ane`). Prior ANE reference runs clocked Budget 140 vision at **11.07–11.17 ms** (**~2.2× faster than Metal GPU ViT**, **6.74× faster than PyTorch MPS**) and 512-token text at **8.24–8.27 ms** (**~2.98× faster than PyTorch MPS**); `v0.4.0` adds verified `140-landscape` (`0.999955` ANE vs Metal `mmproj`), `140-square` (`0.999836`), and `s1024` sliding-window text (`0.999700`) exports.
- **Apple Metal 4 Neural Accelerators:** Runtime hardware detection (`MTLGPUFamilyMetal4` / `Apple10`) activating cooperative `mpp::tensor_ops::matmul2d` FlashAttention with SIMD-group `reduce_rows`, native `bfloat` ALU, and M5 INT8 GEMM.
- **Outperforming PyTorch MPS at 8K Context (`v0.4.0`):** Drops 8,192-token latency to **564.91 ms** (`14,502 tok/s`, Q8_0) / **620.81 ms** (`13,196 tok/s`, BF16)—outperforming the PyTorch MPS reference run (**845.83–875.00 ms**) by **1.36×–1.55×** while slashing peak Metal VRAM to **1,102.8 MB** (Q8_0, **662.9 MB** RSS) / **1,177.5 MB** (BF16) vs **6,592.8 MB** PyTorch MPS (**> 5.4 GB saved**).
- **Universal Cross-Platform CPU SIMD:** Powered by `pulp` and Rayon with dynamic runtime dispatch (ARM64 NEON & x86_64 AVX2/AVX-512) plus optional Apple Accelerate (`--features accelerate`), delivering an **8.26× kernel speedup** on GeGLU (`0.676 ms` vs Candle `5.583 ms` in `v0.4.0`) with zero intermediate tensor allocations.
- **Instant Cold Startup & Warmup:** Metal shaders and compute pipelines are compiled once per device and cached on disk, reducing cold-start latency 100x from **7,060 ms $\to$ 66.7 ms**, paired with `mmap` Safetensors loading and `8 MiB` `BufReader` GGUF reads.
- **Dynamic Multimodal Projectors (`mmproj`):** Text embeddings remain lightweight and self-contained; vision and audio projector towers (`mmproj-embeddinggemma-2-*.gguf`) can be dynamically attached at runtime on demand.
- **Matryoshka Representation Learning (MRL):** Native support for dynamic vector truncation ($768\text{d} \to 512\text{d}, 256\text{d}, 128\text{d}$) with automatic $L_2$ re-normalization.

---

## Performance Highlights

### Single-Sequence Real-Time Latency (`v0.4.0` Metal 4 GPU + Reference Baselines)

> [!NOTE]
> Glypr Metal 4 GPU figures (`BF16` / `Q8_0`) below reflect **`v0.4.0`** release measurements (`compare_models`, `bench_memory`, `bench_multimodal`, `bench_cpu_ops`, `bench_server`). PyTorch MPS, Apple Neural Engine (ANE), and full-model CPU SIMD reference columns reflect prior hardware reference captures on the same Apple Silicon baseline and are labeled accordingly.

| Workload / Modality | Target Input / Dimensions | Python Reference *(PyTorch MPS, prior ref)* | Glypr Metal 4 GPU (`v0.4.0`) | Glypr ANE *(prior ref)* | Glypr CPU SIMD *(prior ref)* | Highlight (`v0.4.0` / Ref) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Text @ 512 Tokens** | Standard RAG passage chunk | 24.65 ms | **22.41 ms** (Q8_0) / **27.03 ms** (BF16) | **8.27 ms** 🚀 | 497.61 ms | **22,851 tok/s** (Metal Q8_0) / **2.98×** (ANE ref) |
| **Text @ 2,048 Tokens** | 2K document context | — | **76.91 ms** (Q8_0) / **95.29 ms** (BF16) | — | — | **26,627 tok/s** (Metal Q8_0) |
| **Text @ 8,192 Tokens** | Full context window (8K) | 845.83 ms | **564.91 ms** (Q8_0) / **620.81 ms** (BF16) 🏆 | — *(Exceeds 1024t ANE)* | 6,222.30 ms | **1.50× faster** vs MPS ref (`14,502 tok/s`) |
| **Vision @ 140 Budget** | 480×680 card (Tower / Total) | 74.66 ms | **5.51 ms** Tower / **56.65 ms** avg (warm `50.34–51.16 ms`) | **11.07 ms** 🚀 | 1,276.56 ms | **17.65 cards/sec** on Metal BF16 |
| **Vision @ 280 Budget** | 480×680 card (Tower / Total) | 73.06 ms | **8.72 ms** Tower / **119.98 ms** avg (warm `105.40–106.10 ms`) | **23.36 ms** 🚀 | 3,757.40 ms | **8.33 cards/sec** on Metal BF16 |
| **Audio @ 12.3s Speech** | 196,348 samples (307 tokens) | — | **8.33 ms** Tower / **51.74 ms** avg | — | — | **237.2× real-time** on Metal BF16 |
| **Peak Metal VRAM @ 8K** | `[MTLDevice currentAllocatedSize]` | 6,592.8 MB *(MPS ref)* | **1,102.8 MB** (Q8_0, **662.9 MB** RSS) / **1,177.5 MB** (BF16) | — | 1,135 MB *(RSS ref)* | **> 5.4 GB saved** (`496–512 MB` transient) |
| **Golden Parity (Cosine)** | 292-vector suite vs Oracle | 1.000000 *(Oracle)* | **0.999967** (BF16) / **0.999896** (Q8_0) | **0.999928** *(Hybrid ref)* | **0.999943** *(ref)* | **Four-9s Parity** (`3.22 s` Q8_0 / `3.61 s` BF16) |

### High-Throughput Batch Inference Highlights (`v0.4.0` Metal BF16, Batch Size = 8)

| Workload / Modality | Batch Input ($B = 8$) | Python Reference *(PyTorch MPS, prior ref)* | Glypr Metal 4 GPU (`v0.4.0` BF16) | Throughput (`v0.4.0`) | Comparison vs. Python Ref |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Text @ 512 Tokens** | 8 × 512t passages (4,096 tokens) | 125.30 ms | **132.93 ms** (`16.62 ms/item`) | **60.2 items/s** | Near-native parity |
| **Text @ 2,048 Tokens** | 8 × 2Kt documents (16,384 tokens) | 730.06 ms | **687.90 ms** (`85.99 ms/item`) 🏆 | **11.6 items/s** | **1.06× faster on Metal** |
| **Text @ 4,096 Tokens** | 8 × 4Kt documents (32,768 tokens) | 2,110.59 ms | **1,759.32 ms** (`219.91 ms/item`) 🏆 | **4.5 items/s** | **1.20× faster on Metal** |
| **Text @ 8,192 Tokens** | 8 × 8Kt full contexts (65,536 tokens) | 6,860.00 ms | **5,072.81 ms** (`634.10 ms/item`) 🏆 | **1.6 items/s** | **1.35× faster on Metal** |
| **Vision @ 70 Budget** | 8 × compact images (4,536 patches) | 508.53 ms *(Clamped to 280t)* | **152.76 ms** (`19.10 ms/item`) 🚀 | **52.4 items/s** | **3.33× faster on Metal** |
| **Vision @ 140 Budget** | 8 × portrait docs (9,072 patches) | 506.73 ms *(Clamped to 280t)* | **315.96 ms** (`39.50 ms/item`) 🚀 | **25.3 items/s** | **1.60× faster on Metal** |
| **Vision @ 280 Budget** | 8 × standard images (19,152 patches) | 494.74 ms | **703.62 ms** (`87.95 ms/item`) | **11.4 items/s** | Full-resolution extraction |
| **Vision @ 560 Budget** | 8 × high-res images (38,304 patches) | 497.63 ms *(Clamped to 280t)* | **1,789.17 ms** (`223.65 ms/item`) | **4.5 items/s** | **Native Dynamic Budget** |
| **Hybrid: Text + Vision** | 8 × [Query text + 140b image] | 508.55 ms | **315.76 ms** (`39.47 ms/item`) 🚀 | **25.3 items/s** | **1.61× faster on Metal** |
| **Hybrid: Text + Vision + Audio** | 8 × [Text + 140b image + 1.0s audio] | 540.90 ms | **423.20 ms** (`52.90 ms/item`) 🚀 | **18.9 items/s** | **1.28× faster on Metal** |

> 📖 **Full Benchmark Reports:** See [`docs/benchmarks/leaderboards.md`](docs/benchmarks/leaderboards.md) and [`docs/benchmarks/memory-and-vram.md`](docs/benchmarks/memory-and-vram.md) for context scaling sweeps, HTTP server throughput (`1,226.9 req/s` @ 32 clients), CPU SIMD microbenchmarks, and the 24-case multimodal parity scorecard.

---

## Quickstart CLI

### Installation

```bash
# 1. Install globally with Metal GPU + Apple Neural Engine (ANE) acceleration (recommended for macOS):
cargo install --path crates/glypr --features metal,ane

# Or install with Metal GPU only:
cargo install --path crates/glypr --features metal

# Or install with universal CPU SIMD (macOS / Linux):
cargo install --path crates/glypr

# 2. Or build locally in-tree:
cargo build --release -p glypr --features metal,ane
```

The CLI defaults to `--device auto` (Metal GPU on Apple Silicon, falling back to CPU SIMD), **BF16** precision, and `--attention auto` (automatically utilizing fused **Metal Flash Attention** on Apple Silicon GPU, falling back to **Chunked Sliding Attention** on CPU SIMD). Models are fetched and cached automatically from `ggml-org/embeddinggemma-2-GGUF` respecting standard Hugging Face cache directories (`$HF_HOME`, `$HF_HUB_CACHE`).

### 1. Pre-Fetch / Cache Canonical GGUFs (`glypr pull`)

```bash
# Pre-fetch all 4 canonical models (BF16 & Q8_0 base models + multimodal projectors)
glypr pull

# Or selectively cache only Q8_0 weights (or Safetensors for CoreML export):
glypr pull --q8 --q8-mmproj
glypr pull --safetensors
```

### 2. Inspect Cache, Hardware & Model Artifacts (`glypr info` / `glypr inspect`)

```bash
glypr info
glypr inspect models/coreml/gemma2_text_backbone_s512_w8a16.mlmodelc
```

### 3. Generate Embeddings (`glypr embed`)

#### Text Embeddings
```bash
# Default BF16 + Auto Metal Flash Attention, 768-dimensional normalized vector preview
glypr embed "What is quantum mechanics?"

# Opt into Q8_0 precision with Matryoshka dimension truncation (256d) and JSON output
glypr --q8 embed "What is quantum mechanics?" --dim 256 -o json

# Strongly-typed asymmetric task prompt
glypr embed "fn binary_search(arr: &[i32], target: i32)" --task CodeRetrieval --dim 512
```

#### Batch & Streaming JSONL (`--batch` / `--jsonl`) & Explicit File Flags
```bash
# Pipe text from stdin
cat query.txt | glypr embed -o json

# Batch independent lines from STDIN in a single model load
printf "First query\nSecond query\n" | glypr embed --batch -o json

# Stream JSONL objects with per-line task/dim/image metadata
printf '{"text":"What causes aurora?","task":"SearchQuery"}\n{"text":"Aurora borealis","task":"Document","title":"Physics"}\n' | glypr embed --jsonl

# Explicit media/text file flags (avoids path-vs-literal-text ambiguity)
glypr --bf16-mmproj embed --text-file notes.md --image ./scene.png
```

#### Apple Neural Engine (ANE) Acceleration (`--ane`)
Requires CoreML `.mlmodelc` bundles exported via `scripts/export_coreml.py` (discovered automatically from `--coreml-dir`, `$GLYPR_COREML_DIR`, `./models/coreml`, or `~/Library/Application Support/glypr/coreml`). Matching vision presets (`140-portrait`, `280-portrait`, `280-square`, `280-landscape`) accelerate images, while `--text-backend auto` routes pure-text queries in the `[64..=seq_len]` sweet spot (`batch <= 4`) to ANE:
```bash
# 1. Export multi-aspect vision + 512-token text models (intermediate .mlpackage cleaned by default; pass --keep-mlpackage to retain)
uv run python scripts/export_coreml.py --target all --vision-preset default --output-dir models/coreml

# 2. Hybrid Mode: Vision Tower on ANE + Text Backbone on Metal/ANE
glypr --ane embed "Card illustration:" ./portrait.png "and description."

# 3. Zero-mmproj Standalone Image Embedding (runs directly on ANE without GGUF mmproj!)
glypr --ane embed ./portrait.png --image-token-budget 140

# 4. Strict ANE routing for vision or pure text (--text-backend auto|ane|metal)
glypr --ane --vision-backend ane embed ./portrait.png --image-token-budget 140
glypr --ane --text-backend ane embed "RAG passage within 512 tokens"
```

#### Interleaved Multimodal (Text + Images + Audio)
Multimodal projector loading is opt-in (defaulting to text-only with 0 extra memory). Pass `--bf16-mmproj` or `--q8-mmproj` to attach a projector. Positional arguments ending in `.png`, `.jpg`, `.jpeg`, `.webp`, or `.wav` are loaded as media files (and error immediately if missing), while non-media strings such as `README.md` embed as literal text:
```bash
# Interleaved Text + Image
glypr --bf16-mmproj embed "Illustration:" ./scene.png "and description."

# Audio query
glypr --bf16-mmproj embed "Query sound:" ./speech.wav --task SearchQuery

# Adjust vision token budget (70 to 1120 tokens; 140 for compact, 280 default, 560 high)
glypr --bf16-mmproj embed ./scene.png --image-token-budget 140
```

#### Vector Encodings & Raw Binary Streaming
```bash
# Zero-metadata INT8 quantization (values in [-127, 127]) formatted as JSON array
glypr embed "Database indexing" --encoding int -o json

# Binary 1-bit quantization streamed directly as raw bytes to stdout (768d = 96 raw bytes)
glypr embed "High-throughput search" --encoding binary -o raw > vector.bin
```

### 4. OpenAI-Compatible HTTP Serving Engine (`glypr serve`)

Launch a production-ready HTTP server with dynamic batching, MRL truncation, quantization, `/ready` probe, graceful shutdown, and structured `tracing` logs:
```bash
# Start text-only server listening on port 8080 (defaults to auto Metal/CPU device and BF16)
glypr serve --port 8080 --max-batch-size 32

# Or serve with Apple Neural Engine (ANE) vision + pure-text acceleration (warms up ANE on boot):
glypr --ane serve --port 8080

# Or serve with multimodal support (local path inputs over HTTP are disabled by default; pass --allow-local-files to opt in):
glypr --bf16-mmproj serve --port 8080
```

Query with standard OpenAI API clients or `curl`:
```bash
# Standard embedding request
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Deep learning embeddings"}'

# MRL truncation (256d) and fast zero-cost INT8 quantization:
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Vector search engine", "dimensions": 256, "encoding_format": "int8"}'

# Health check:
curl http://127.0.0.1:8080/health
```

---

## Rust SDK API

Add `glypr-core` to your `Cargo.toml`:

```toml
[dependencies]
glypr-core = { version = "0.4.0", features = ["metal"] }
```

### Fluent Builder & Self-Contained GGUF Loading API

```rust
use candle_core::Device;
use glypr_core::{
    cosine_similarity, truncate_mrl, AttentionMode, GlyprEmbedder, GlyprResult, Task,
    TextBackendPolicy, VisionBackendPolicy,
};

fn main() -> GlyprResult<()> {
    // Option A: Fluent builder with safe Metal auto-detection and canonical HF cache resolution
    let embedder = GlyprEmbedder::builder()
        .gguf_canonical(false) // false = BF16, true = Q8_0
        .attention_mode(AttentionMode::MetalFlash)
        .vision_backend(VisionBackendPolicy::Auto)
        .text_backend(TextBackendPolicy::Auto)
        .build()?;

    // Option B: Load directly from a local single GGUF file (no config.json or tokenizer.json required!)
    // let device = Device::new_metal(0)?;
    // let mut embedder = GlyprEmbedder::load_gguf_bf16("embeddinggemma-2-BF16.gguf", &device)?;
    // embedder.set_attention_mode(AttentionMode::MetalFlash); // 620.81 ms BF16 / 564.91 ms Q8_0 @ 8K

    // 1. Strongly-typed text embedding via Task enum (returns native 768d L2-normalized vector)
    let query_vec = embedder.embed_text(
        "What causes gravitational lensing?",
        Task::SearchQuery,
    )?;

    // 2. Document embedding with title conditioning
    let doc_vec = embedder.embed_text(
        "Massive celestial objects bend spacetime, curving the paths of passing light rays.",
        Task::Document { title: Some("General Relativity") },
    )?;

    // 3. Matryoshka Representation Learning (MRL): Slice and re-normalize in O(K)
    let query_256 = truncate_mrl(&query_vec, 256)?;
    let doc_256 = truncate_mrl(&doc_vec, 256)?;

    // 4. Compute semantic similarity
    let score = cosine_similarity(&query_256, &doc_256);
    println!("Cosine similarity (256d): {:.4}", score);

    // 5. Optional: Dynamically attach multimodal projector or CoreML ANE bundles when needed
    // embedder.attach_multimodal_projector("mmproj-embeddinggemma-2-BF16.gguf")?;

    Ok(())
}
```

### Vector Quantization Utilities (`glypr_core::util`)

```rust
use glypr_core::util::{
    quantize_int8, int8_cosine_similarity,
    quantize_binary, hamming_distance, binary_cosine_similarity,
};

// 1. Static INT8 (4x compression, 16.1x faster SIMD dot product)
let q_int8 = quantize_int8(&query_vec); // Vec<i8> (768 bytes for 768d, mmap-ready)
let sim_int8 = int8_cosine_similarity(&q_int8, &doc_int8); // 52.6 ns SIMD dot product

// 2. Binary 1-bit (32x compression, 470x faster hardware POPCNT)
let q_bin = quantize_binary(&query_vec)?; // Vec<u64> (96 bytes for 768d)
let dist = hamming_distance(&q_bin, &doc_bin); // 1.80 ns POPCNT distance
let sim_bin = binary_cosine_similarity(&q_bin, &doc_bin); // Charikar angular cosine
```

| Scheme | Dimensions | Memory / 1M Vectors | Compute Latency | Throughput | Speedup vs Float32 | Accuracy Loss ($\Delta$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Float32 Baseline** | 768d | 3,072 MB | 847.8 ns | 1.2 Mop/s | 1.00× | 0.0000 |
| **Static INT8 Dot (SIMD)** | 768d | **768 MB** (4× comp) | **52.6 ns** | **19.0 Mop/s** | **16.10×** | $\le 0.0043$ |
| **Binary POPCNT Hamming** | 768d | **96 MB** (32× comp) | **1.80 ns** | **555.4 Mop/s** | **470.87×** | $\le 0.1500$ |

---

## Cargo Feature Flags

| Feature | Description | Default Target |
| :--- | :--- | :--- |
| `metal` | Compiles native Metal 4 GPU shaders, NAX FlashAttention, and zero-copy GQA. | macOS (Apple Silicon) |
| `ane` | Activates Apple Neural Engine Core ML acceleration for vision and resident text. | macOS (Apple Silicon) |
| `coreml` | Core ML runtime foundation bindings. | macOS (Apple Silicon) |
| `accelerate` | Links Apple Accelerate framework for hardware BLAS. | macOS |
| *default* | Pure portable CPU SIMD (`pulp` NEON / AVX2 / AVX-512) with zero heap churn. | Universal (macOS, Linux) |

---

## Testing & Parity Verification

Glypr maintains a three-tier verification pyramid covering weightless CI checks, fast local suites, and full live Rust-vs-Python differential parity:

```bash
# Tier 0 (Weightless CI): synthetic tiny-model, EngineHandle::mock server contract, and CLI validation tests
GLYPR_ALLOW_MISSING_WEIGHTS=1 cargo test --locked --workspace

# Tier 1 (~36s): golden integrity check + Apple Silicon workspace check + 132 fast tests (9 Tier 2 ignored)
./tests/scripts/run_qa_suite.sh --tier1

# Tier 2 (~106s): Tier 1 + --release heavy parity tests (9/9) + pytest (4/4) + 26/26 live differential parity checks
./tests/scripts/run_qa_suite.sh --tier2

# Standalone strict FP32 golden fixture integrity check (schema_version: 2 provenance) or live server parity
uv run python tests/scripts/verify_goldens.py --strict-fp32
uv run python tests/scripts/differential_parity.py
```

---

## Documentation Navigation

Detailed engineering specifications, benchmarks, and guides are organized in [`docs/`](docs/):

- **Architecture:**
  - [`docs/architecture/specifications.md`](docs/architecture/specifications.md): Hyperparameters, hybrid 5:1 sliding/global attention, Per-Layer Embeddings (PLE).
  - [`docs/architecture/hardware-acceleration.md`](docs/architecture/hardware-acceleration.md): Dual-track Metal GPU fusion, zero-copy GQA, and `pulp` CPU SIMD.
  - [`docs/architecture/modularity.md`](docs/architecture/modularity.md): Dynamic encoder towers (270M text, 170M vision, 300M audio).
- **Multimodal (`mmproj`):**
  - [`docs/multimodal/mmproj-architecture.md`](docs/multimodal/mmproj-architecture.md): SigLIP Vision & Conformer Audio Tower architecture, Gemma 4 Clippable Linears, and soft token splicing.
  - [`docs/multimodal/gguf-nuances-and-conversions.md`](docs/multimodal/gguf-nuances-and-conversions.md): Complete GGUF quirks, 963-tensor mapping dictionary, Clippable Linears, naming swaps, and quantization artifacts.
  - [`docs/multimodal/inputs-and-budgeting.md`](docs/multimodal/inputs-and-budgeting.md): Vision token budgeting (70–1120 tokens) and audio feature processing.
- **Benchmarks & Memory:**
  - [`docs/benchmarks/leaderboards.md`](docs/benchmarks/leaderboards.md): MTEB/MMEB leaderboard rankings, PyTorch golden parity scorecard, latency scaling.
  - [`docs/benchmarks/memory-and-vram.md`](docs/benchmarks/memory-and-vram.md): Apple Silicon unified memory mechanics, transient allocation analysis, Chunked vs Flash Pareto frontier.
- **Deployment & Optimization:**
  - [`docs/deployment/quantization-and-runtimes.md`](docs/deployment/quantization-and-runtimes.md): Precision guidelines, QAT INT4/INT8, single-file GGUF self-containment, runtimes.
  - [`docs/vector-optimization/matryoshka-mrl.md`](docs/vector-optimization/matryoshka-mrl.md): Matryoshka dimension truncation and L2 re-normalization.
  - [`docs/vector-optimization/quantization.md`](docs/vector-optimization/quantization.md): Zero-cost static INT8 and 1-bit binary vector quantization.
  - [`docs/prompts/task-instructions.md`](docs/prompts/task-instructions.md): Asymmetric and symmetric task instructions and title conditioning.

---

## Contributing

Contributions are welcome! Please review [`CONTRIBUTING.md`](CONTRIBUTING.md) for local setup, testing standards, and development guidelines.

---

## License

Glypr is licensed under the [Apache License, Version 2.0](LICENSE).
