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
- **Apple Neural Engine (ANE) Acceleration:** Transparently offloads portrait vision encoding (1,134 / 2,394 patches) directly to the Apple Neural Engine via `coreml-native` (`--features metal,ane`), running in **11.17 ms (89.5 img/s)**—a **2.2× to 2.4× speedup over Metal GPU ViT**—with **`0.999928`** cosine parity. Standalone resident text backbone executes 512-token RAG queries in **8.24 ms on pure ANE (2.6× faster than GPU)** with zero GPU power consumption.
- **Apple Metal 4 Neural Accelerators:** Runtime hardware detection (`MTLGPUFamilyMetal4` / `Apple10`) activating cooperative `mpp::tensor_ops::matmul2d` FlashAttention with SIMD-group `reduce_rows`, native `bfloat` ALU, and M5 INT8 GEMM.
- **Outperforming PyTorch MPS at 8K Context:** Drops 8,192-token latency to **615.30 ms** (Q8_0) / **670.79 ms** (BF16)—outperforming PyTorch MPS's **875.00 ms** by up to 260 ms (**1.42× faster**) while saving **over 5.0 GB of VRAM** (1.47 GB vs 6.59 GB).
- **Universal Cross-Platform CPU SIMD:** Powered by `pulp` and Rayon with dynamic runtime dispatch (ARM64 NEON & x86_64 AVX2/AVX-512), delivering a **10.49× kernel speedup** on GeGLU and eliminating ~30 GB of intermediate heap churn.
- **Instant Cold Startup:** Metal shaders and compute pipelines are compiled once per device and cached on disk, reducing cold-start latency 100x from **7,060 ms $\to$ 66.7 ms**.
- **Dynamic Multimodal Projectors (`mmproj`):** Text embeddings remain lightweight and self-contained; vision and audio projector towers (`mmproj-embeddinggemma-2-*.gguf`) can be dynamically attached at runtime on demand.
- **Matryoshka Representation Learning (MRL):** Native support for dynamic vector truncation ($768\text{d} \to 512\text{d}, 256\text{d}, 128\text{d}$) with automatic $L_2$ re-normalization.

---

## Performance Highlights

### Single-Sequence Real-Time Latency (Ground-Truth Hardware Telemetry)

| Workload / Modality | Target Input / Dimensions | Python Reference (PyTorch MPS) | Glypr (Metal 4 GPU) | Glypr (Apple Neural Engine) | Glypr (CPU SIMD) | Speedup vs. Python Reference |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Text @ 512 Tokens** | Standard RAG passage chunk | 24.65 ms | 25.72 ms | **8.27 ms** 🚀 | 497.61 ms | **2.98× faster on ANE** |
| **Vision @ 140 Budget** | Portrait doc (1,134 patches) | 74.66 ms | 24.03 ms | **11.07 ms** 🚀 | 1,276.56 ms | **6.74× faster on ANE** |
| **Vision @ 280 Budget** | Standard image (2,394 patches) | 73.06 ms | 36.54 ms | **23.36 ms** 🚀 | 3,757.40 ms | **3.13× faster on ANE** |
| **Hybrid @ $\le 512$ Tokens** | Query text + 140b image | 74.38 ms | 50.79 ms | **24.67 ms** 🚀 | — | **3.01× faster on ANE** |
| **Text @ 8,192 Tokens** | Full context window (8K) | 845.83 ms | **644.36 ms** 🏆 | — *(Exceeds 512t limit)* | 6,222.30 ms | **1.31× faster on Metal GPU** |
| **Vision @ 560 Budget** | High-res inspection (4,788 patches)| 73.84 ms *(Clamped to 280t)* | **86.29 ms** | — *(Fallback to GPU)* | — | **Native Dynamic Budget** |
| **Hybrid @ $> 512$ Tokens** | 2K doc + 140b image | 165.20 ms | 145.28 ms | **122.76 ms (Hybrid)** ⚡ | — | **1.35× faster on Hybrid Routing** |
| **Peak Memory @ 8K** | Physical unified memory (RAM+VRAM)| 6,592 MB (6.44 GB) | **1,473 MB (1.44 GB)** | — | **1,135 MB (1.11 GB)** | **5.1 GB saved (-77%)** |
| **Golden Parity (Cosine)** | 292-vector suite vs Oracle | 1.000000 *(Oracle)* | **0.999951 *(Avg)*** | **0.999928 *(Avg)*** | **0.999943 *(Avg)*** | **Bitwise / Four-9s Parity** |

### High-Throughput Batch Inference Highlights (Batch Size = 8)

| Workload / Modality | Batch Input ($B = 8$) | Python Reference (PyTorch MPS) | Glypr (Metal 4 GPU) | Throughput (Glypr) | Speedup vs. Python Reference |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Text @ 512 Tokens** | 8 × 512t passages (4,096 tokens) | 125.30 ms | **161.05 ms** (20.1 ms/item) | **49.7 items/s** | Near-native parity |
| **Text @ 2,048 Tokens** | 8 × 2Kt documents (16,384 tokens) | 730.06 ms | **839.33 ms** (104.9 ms/item) | **9.5 items/s** | Near-native parity |
| **Text @ 4,096 Tokens** | 8 × 4Kt documents (32,768 tokens) | 2,110.59 ms | **2,161.63 ms** (270.2 ms/item) | **3.7 items/s** | Near-native parity |
| **Text @ 8,192 Tokens** | 8 × 8Kt full contexts (65,536 tokens) | 6,860.00 ms | **6,340.92 ms** (792.6 ms/item) 🏆 | **1.3 items/s** | **1.08× faster on Metal** |
| **Vision @ 70 Budget** | 8 × compact images (4,536 patches) | 508.53 ms *(Clamped to 280t)* | **153.83 ms** (19.2 ms/item) 🚀 | **52.0 items/s** | **3.31× faster on Metal** |
| **Vision @ 140 Budget** | 8 × portrait docs (9,072 patches) | 506.73 ms *(Clamped to 280t)* | **321.53 ms** (40.2 ms/item) 🚀 | **24.9 items/s** | **1.58× faster on Metal** |
| **Vision @ 280 Budget** | 8 × standard images (19,152 patches) | 494.74 ms | **718.34 ms** (89.8 ms/item) | **11.1 items/s** | Full-resolution extraction |
| **Vision @ 560 Budget** | 8 × high-res images (38,304 patches) | 497.63 ms *(Clamped to 280t)* | **1,837.45 ms** (229.7 ms/item) | **4.4 items/s** | **Native Dynamic Budget** |
| **Hybrid: Text + Vision** | 8 × [Query text + 140b image] | 508.55 ms | **322.60 ms** (40.3 ms/item) 🚀 | **24.8 items/s** | **1.58× faster on Metal** |
| **Hybrid: Text + Vision + Audio** | 8 × [Text + 140b image + 1.0s audio] | 540.90 ms | **447.08 ms** (55.9 ms/item) 🚀 | **17.9 items/s** | **1.21× faster on Metal** |

> 📖 **Full Benchmark Reports:** See [`docs/benchmarks/leaderboards.md`](docs/benchmarks/leaderboards.md) for context scaling sweeps, MTEB/MMEB comparisons, and the 24-case multimodal parity scorecard.

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

# Or selectively cache only Q8_0 weights:
glypr pull --q8 --q8-mmproj
```

### 2. Inspect Cache & Device Backend (`glypr info`)

```bash
glypr info
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

#### Pipe via Standard Input (STDIN)
```bash
# Pipe text from stdin
cat query.txt | glypr embed -o json

# High-throughput streaming
echo "Fast stream test" | glypr embed --encoding int8 -o json
```

#### Apple Neural Engine (ANE) Acceleration (`--ane`)
```bash
# 1. Hybrid Mode: Vision Tower on ANE + Text Backbone on Metal GPU
glypr --ane embed "Card illustration:" ./portrait.png "and description."

# 2. Zero-mmproj Standalone Image Embedding (runs directly on ANE without GGUF mmproj!)
glypr --ane embed ./portrait.png

# 3. Explicit Vision Backend Routing (--vision-backend auto|ane|metal)
glypr --ane --vision-backend ane embed ./portrait.png
```

#### Interleaved Multimodal (Text + Images + Audio)
Multimodal projector loading is opt-in (defaulting to text-only with 0 extra memory). Pass `--bf16-mmproj` or `--q8-mmproj` to attach a projector. If an argument points to an image (`.png`, `.jpg`, `.webp`) or 16 kHz `.wav` audio file on disk, `glypr` streams it to the multimodal projector:
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

Launch a production-ready HTTP server with dynamic batching, MRL truncation, and quantization support:
```bash
# Start text-only server listening on port 8080 (defaults to auto Metal/CPU device and BF16)
glypr serve --port 8080 --max-batch-size 32

# Or serve with Apple Neural Engine (ANE) hardware acceleration:
glypr --ane serve --port 8080

# Or serve with multimodal support:
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
glypr-core = { version = "0.3.0", features = ["metal"] }
```

### Self-Contained GGUF Loading & Unified Embedding API

```rust
use anyhow::Result;
use candle_core::Device;
use glypr_core::{
    cosine_similarity, truncate_mrl, AttentionMode, GlyprEmbedder, ModalityPart, Task,
    VISION_BUDGET_DEFAULT, VISION_BUDGET_HIGH,
};

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;

    // 1. Load directly from a single GGUF file (no external config or tokenizer files required!)
    let mut embedder = GlyprEmbedder::load_gguf_bf16("embeddinggemma-2-BF16.gguf", &device)?;

    // 2. Enable hardware-fused Metal FlashAttention (zero-copy GQA, 615 ms @ 8k tokens)
    embedder.set_attention_mode(AttentionMode::MetalFlash);

    // 3. Strongly-typed text embedding via Task enum (returns native 768d L2-normalized vector)
    let query_vec = embedder.embed_text(
        "What causes gravitational lensing?",
        Task::SearchQuery,
    )?;

    // 4. Document embedding with title conditioning
    let doc_vec = embedder.embed_text(
        "Massive celestial objects bend spacetime, curving the paths of passing light rays.",
        Task::Document { title: Some("General Relativity") },
    )?;

    // 5. Matryoshka Representation Learning (MRL): Slice and re-normalize in O(K)
    let query_256 = truncate_mrl(&query_vec, 256)?;
    let doc_256 = truncate_mrl(&doc_vec, 256)?;

    // 6. Compute semantic similarity
    let score = cosine_similarity(&query_256, &doc_256);
    println!("Cosine similarity (256d): {:.4}", score);

    // 7. Optional: Dynamically attach multimodal projector when needed
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
