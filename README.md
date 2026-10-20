# Glypr (`glypr` / `glypr-core`)

> **High-Performance, 100% Self-Contained Rust Engine for Google DeepMind's EmbeddingGemma 2**

`glypr` is a native Rust inference engine for **EmbeddingGemma 2**, providing hardware-optimized text and multimodal embeddings on Apple Silicon (Metal GPU) and Linux/macOS CPU.

It deploys from a **single self-contained `.gguf` file**—extracting all model hyperparameters, layer configurations, and the Gemma 4 BPE tokenizer directly from the GGUF container without needing external `config.json` or `tokenizer.json` files.

---

## Key Highlights

- **100% Self-Contained Single-File Deployment:** Point `glypr` to a single `.gguf` file (`embeddinggemma-2-BF16.gguf` or `embeddinggemma-2-Q8_0.gguf`). Model architecture and the complete 262,144-token BPE tokenizer are reconstructed in-memory in milliseconds.
- **Apple Neural Engine (ANE) Acceleration & Hybrid Execution:** Native integration of Core ML via `coreml-native` (`--features metal,ane`). Transparently routes vision tower execution for portrait documents (1,134 patches / 2,394 patches) directly to the Apple Neural Engine, achieving **11.17 ms (89.5 img/s)**—a **2.2× to 2.4× speedup over GPU Metal ViT**—while splicing output soft tokens directly into the GPU text backbone with **`0.999928`** cosine parity. Standalone resident text backbone executes 512-token RAG queries in **8.24 ms on pure ANE (2.6× faster than GPU Metal)** with zero GPU power consumption.
- **Apple M5 & Metal 4 Neural Accelerators:** Runtime detection (`MTLGPUFamilyMetal4` / `Apple10`) activating cooperative `mpp::tensor_ops::matmul2d` FlashAttention with SIMD-group hardware `reduce_rows`, native `bfloat` ALU, and M5 INT8 GEMM.
- **Beating PyTorch MPS at 8K Tokens:** Drops 8,192-token latency to **615.30 ms** (Q8_0) / **670.79 ms** (BF16) (outperforming PyTorch MPS's **875.00 ms** by up to 260 ms / 1.42x) with peak throughput reaching **23,359 tok/s** at 2K context.
- **Ultra-Efficient Metal VRAM Footprint:** Through zero-copy GQA, native $d_{\text{head}}=512$ full-context kernels, fused backbone MSL operations, and zero-overhead weight concatenation views, peak Metal VRAM at 8,192 tokens dropped from **14.59 GB $\to$ 1.47 GB (BF16) / 1.58 GB (Q8_0)**, slashing transient VRAM by **86.6%** down to **688 MB** and saving **over $5.1\text{ GB}$ vs PyTorch MPS (6.59 GB)**.
- **Instant Cold Startup via MTLLibrary Caching:** Shaders and compute pipelines are compiled once per Metal device and cached, cutting first-inference cold-start latency 100x from **7,060 ms $\to$ 66.7 ms**.
- **FP32 Long-Context Pooling & Direct 64-Bit STFT Twiddles:** Accumulation in FP32 eliminates BF16 7-bit mantissa swamping across up to 8,192 tokens (raising long-context cosine fidelity from `0.9873` to **`0.99975`**). Direct 64-bit trigonometry cuts audio FFT twiddle drift 74x down to **$5.96 \times 10^{-8}$**.
- **2D GeGLU Grid Dispatch & Zero-Alloc Image Preprocessor:** Replaced 1D thread indexing with 2D grid dispatch in `geglu_gate_up_bf16`, eliminating $>245\text{M}$ GPU integer division and modulo operations per 8K sequence. Image preprocessing uses a compile-time `const [f32; 256]` LUT and parallel Rayon chunk writing directly into target buffers without intermediate heap churn.
- **Universal Cross-Platform CPU SIMD:** Powered by `pulp` and Rayon with runtime dynamic architecture dispatch (ARM64 NEON & x86_64 AVX2/AVX-512), delivering a **10.49x isolated kernel speedup** on GeGLU and eliminating ~30 GB of intermediate heap allocations.
- **Harden Four-9s Parity Suite:** Evaluates the complete 292-vector golden test suite across context lengths from 64 to 8,192 tokens with strict **0.9998** (min) / **0.9999** (avg) cosine similarity thresholds across both `ChunkedSliding` and `MetalFlash`, plus complete multimodal coverage across BF16 and Q8_0 mmproj projectors.
- **Dynamic Multimodal Projector (`mmproj`):** Text embeddings remain lightweight and self-contained; vision and audio projector towers (`mmproj-embeddinggemma-2-*.gguf`) can be dynamically attached at runtime on demand.
- **Matryoshka Representation Learning (MRL):** Native support for dynamic vector truncation ($768\text{d} \to 512\text{d}, 256\text{d}, 128\text{d}$) with automatic L2 re-normalization.

---

## Quickstart CLI

### Installation / Build
```bash
# 1. Install globally to ~/.cargo/bin with Metal GPU + Apple Neural Engine (ANE) acceleration (recommended for macOS):
cargo install --path crates/glypr --features metal,ane

# Or install with Metal GPU only:
cargo install --path crates/glypr --features metal

# Or install with universal CPU SIMD:
cargo install --path crates/glypr

# 2. Or build locally in-tree:
cargo build --release -p glypr --features metal,ane
```

The CLI defaults to `--device auto` (Metal GPU on Apple Silicon, falling back to CPU SIMD), **BF16** precision, and `--attention auto` (automatically utilizing fused **Metal Flash Attention** on Apple Silicon GPU, falling back to **Chunked Sliding Attention** on CPU SIMD). Canonical models are resolved and cached automatically from `ggml-org/embeddinggemma-2-GGUF` respecting standard Hugging Face cache paths (`$HF_HOME`, `$HF_HUB_CACHE`).

### 1. Pre-Fetch / Cache Canonical GGUFs (`glypr pull`)
```bash
# Pre-fetch all 4 canonical models (BF16 & Q8_0 base models + multimodal projectors)
./target/release/glypr pull

# Or selectively cache only Q8_0 weights:
./target/release/glypr pull --q8 --q8-mmproj
```

### 2. Inspect Cache & Device Backend (`glypr info`)
```bash
./target/release/glypr info
```

### 3. Generate Embeddings (`glypr embed`)

#### Text Embeddings
```bash
# Default BF16 + Auto Metal Flash Attention, 768-dimensional normalized vector preview
./target/release/glypr embed "What is quantum mechanics?"

# Explicitly choose attention kernel: auto (default), flash, chunked, or standard
./target/release/glypr --attention flash embed "Quantum mechanics"

# Opt into Q8_0 precision with Matryoshka dimension truncation (256d) and JSON output
./target/release/glypr --q8 embed "What is quantum mechanics?" --dim 256 -o json

# Strongly-typed asymmetric task prompt
./target/release/glypr embed "fn binary_search(arr: &[i32], target: i32)" --task CodeRetrieval --dim 512
```

#### Pipe via Standard Input (STDIN)
```bash
# Pipe text from stdin
cat query.txt | ./target/release/glypr embed -o json

echo "Fast stream test" | ./target/release/glypr embed --encoding int8 -o json
```

#### Interleaved Multimodal (Text + Images + Audio)
Multimodal projector loading is explicit and opt-in (defaulting to text-only with 0 extra memory or downloads). Pass `--bf16-mmproj`, `--q8-mmproj`, or `--mmproj-file <FILE>` to attach a projector. If an argument points to an existing image (`.png`, `.jpg`, `.jpeg`, `.webp`) or 16 kHz `.wav` audio file on disk, `glypr` streams it to the multimodal projector; otherwise, it is treated as text:
```bash
# Interleaved Text + PNG Card Image (with BF16 projector)
./target/release/glypr --bf16-mmproj embed "Card illustration:" ./card.png "and card description."

# Audio query
./target/release/glypr --bf16-mmproj embed "Query sound:" ./speech.wav --task SearchQuery

# Adjust vision token budget (e.g. 140 for 480x680 card verbatim, 280 default, 560 high)
./target/release/glypr --bf16-mmproj embed ./card.png --image-token-budget 140
```

#### Apple Neural Engine (ANE) Acceleration (`--ane`)
Enable the Apple Neural Engine to offload vision encoding and achieve **2.2×–2.4× faster throughput** with near-zero GPU power consumption:
```bash
# 1. Hybrid Mode: Vision Tower on ANE + Text Backbone on Metal GPU
./target/release/glypr --ane embed "Card illustration:" ./portrait.png "and description."

# 2. Zero-mmproj Standalone Image Embedding
# When --ane is enabled, portrait images can be embedded directly without downloading or attaching a GGUF mmproj!
./target/release/glypr --ane embed ./portrait.png

# 3. Explicit Vision Backend Routing (--vision-backend auto|ane|metal)
# auto (default): Dispatches matching portrait patch counts to ANE; falls back to Metal mmproj for other budgets
# ane: Strict ANE execution; fails fast if patch count is uncompiled
# metal: Dispatches all image encoding to GPU Metal mmproj
./target/release/glypr --ane --vision-backend ane embed ./portrait.png

# 4. Custom Core ML models directory override
./target/release/glypr --ane --coreml-dir ./models/coreml embed ./portrait.png
```

#### Vector Encodings & Raw Binary Streaming
```bash
# Zero-metadata INT8 quantization (values in [-127, 127]) formatted as JSON array
./target/release/glypr embed "Database indexing" --encoding int -o json

# Binary 1-bit quantization streamed directly as raw bytes to stdout (e.g. 768d = 96 raw bytes)
./target/release/glypr embed "High-throughput search" --encoding binary -o raw > vector.bin
```

### 4. Custom Model & Projector Overrides
To run local models or fine-tunes without contacting Hugging Face:
```bash
./target/release/glypr --model-file /path/to/custom-model.gguf --mmproj-file /path/to/custom-mmproj.gguf embed "Custom weights query"
```

### 5. OpenAI-Compatible HTTP Serving Engine (`glypr serve`)
Launch a high-performance HTTP server with adaptive dynamic batching, MRL truncation, and quantization support. Like `embed`, the server defaults to text-only mode unless started with an mmproj flag:
```bash
# Start text-only server listening on port 8080 (defaults to auto Metal/CPU device and BF16)
./target/release/glypr serve --port 8080 --max-batch-size 32

# Or serve with multimodal support:
./target/release/glypr --bf16-mmproj serve --port 8080

# Or serve with Q8_0 base weights and Q8_0 multimodal projector:
./target/release/glypr --q8 --q8-mmproj serve --port 8080

# Or serve with Apple Neural Engine (ANE) hardware vision acceleration:
./target/release/glypr --ane serve --port 8080
```

Query with standard OpenAI API clients or `curl`:
```bash
# Minimal request (defaults to model="embeddinggemma-2", dim=768, encoding_format="float", task="Raw")
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Deep learning embeddings"}'

# MRL truncation (256d) and fast zero-cost INT8 quantization:
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Vector search engine", "dimensions": 256, "encoding_format": "int8"}'

# Inspect models & health:
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/health
```

---

## Rust SDK API

Add `glypr-core` to your `Cargo.toml`:

```toml
[dependencies]
glypr-core = { path = "crates/glypr-core", features = ["metal"] }
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

    // 2. Enable hardware-fused Metal FlashAttention (zero-copy GQA, 868 ms @ 8k tokens)
    embedder.set_attention_mode(AttentionMode::MetalFlash);

    // 3. Optional: Dynamically attach multimodal projector when needed
    // embedder.attach_multimodal_projector("mmproj-embeddinggemma-2-BF16.gguf")?;

    // 4. Strongly-typed text embedding via Task enum (returns native 768d L2-normalized vector)
    let query_vec = embedder.embed_text(
        "What causes gravitational lensing?",
        Task::SearchQuery,
    )?;

    // 5. Document embedding with title conditioning
    let doc_vec = embedder.embed_text(
        "Massive celestial objects bend spacetime, curving the paths of passing light rays.",
        Task::Document { title: Some("General Relativity") },
    )?;

    // 6. Matryoshka Representation Learning (MRL): Slice and re-normalize in O(K)
    let query_256 = truncate_mrl(&query_vec, 256)?;
    let doc_256 = truncate_mrl(&doc_vec, 256)?;

    // 7. Compute semantic similarity
    let score = cosine_similarity(&query_256, &doc_256);
    println!("Cosine similarity (256d): {:.4}", score);

    // 8. Interleaved Multimodal Sequence (Text + Images + Audio)
    // let img = image::open("diagram.png")?;
    // let interleaved_vec = embedder.embed_interleaved(&[
    //     ModalityPart::Text("A comparison of the diagram: "),
    //     ModalityPart::Image { image: &img, budget: Some(VISION_BUDGET_HIGH) },
    //     ModalityPart::Text(" and recorded observation: "),
    //     ModalityPart::Audio { samples: &audio_samples },
    // ], Task::SearchQuery)?;

    Ok(())
}
```

### Vector Quantization & High-Throughput Search (`glypr::util`)

`glypr` provides zero-cost functional quantization utilities over raw primitives (`&[f32]`, `&[u64]`, `&[i8]`):

```rust
use glypr_core::util::{
    quantize_int8, int8_dot_product, int8_cosine_similarity,
    quantize_binary, hamming_distance, binary_cosine_similarity,
};

// 1. Static INT8 (Zero-metadata, exactly D bytes/vec, 16.1x faster SIMD dot product)
let q_int8 = quantize_int8(&query_vec); // Vec<i8> (768 bytes for 768d, mmap flat file ready)
let sim_int8 = int8_cosine_similarity(&q_int8, &doc_int8); // 52.6 ns SIMD dot product

// 2. Binary 1-bit (32x compression, 470x faster hardware POPCNT)
let q_bin = quantize_binary(&query_vec)?; // Vec<u64> (96 bytes for 768d)
let dist = hamming_distance(&q_bin, &doc_bin); // 1.80 ns POPCNT distance
let sim_bin = binary_cosine_similarity(&q_bin, &doc_bin); // Charikar angular cosine
```

| Scheme | Dimension | Memory / 1M Vecs | Compute Latency | Throughput | Speedup vs Float32 | Accuracy ($\Delta$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Float32 Baseline** | 768d | 3,072 MB | 847.8 ns | 1.2 Mop/s | 1.00x | 0.0000 |
| **Static INT8 Dot (SIMD)** | 768d | **768 MB** (4x comp) | **52.6 ns** | **19.0 Mop/s** | **16.10x** | $\le 0.0043$ |
| **Binary POPCNT Hamming** | 768d | **96 MB** (32x comp) | **1.80 ns** | **555.4 Mop/s** | **470.87x** | $\le 0.1500$ |

> 📖 **Deep Dive:** See [`docs/vector-optimization/quantization.md`](docs/vector-optimization/quantization.md) for 2-pass hierarchical KNN benchmarks and Turso's Block-Scaled Q8 (BSQ8) technical blueprint.

---

## Empirical Benchmark & Hardware Memory Scorecard

Tested on Apple Silicon Metal GPU across sequence lengths up to 8,192 tokens:

| Context Length | Framework / Variant | Attention Engine | Base Model RAM | Peak Metal VRAM | Transient VRAM | Max Process RSS | Latency | Peak Throughput |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024t** | `glypr-core` BF16 | Standard Dense | $781.0\text{ MB}$ | $1,045.7\text{ MB}$ | $264.6\text{ MB}$ | $2,037.3\text{ MB}$ | $95.58\text{ ms}$ | $10,714\text{ tok/s}$ |
| **1,024t** | `glypr-core` BF16 | Metal Flash (Fused) | $785.0\text{ MB}$ | **$986.2\text{ MB}$** | **$201.2\text{ MB}$** | $2,038.5\text{ MB}$ | **$52.28\text{ ms}$** | **$19,587\text{ tok/s}$** |
| **1,024t** | `glypr-core` Q8_0 | Metal Flash (Fused) | **$606.3\text{ MB}$** | **$754.1\text{ MB}$** | **$147.8\text{ MB}$** | **$1,678.5\text{ MB}$** | **$43.93\text{ ms}$** | **$23,311\text{ tok/s}$** 🚀 |
| | | | | | | | | |
| **2,048t** | `glypr-core` BF16 | Metal Flash (Fused) | $785.0\text{ MB}$ | **$1,120.4\text{ MB}$** | **$335.4\text{ MB}$** | $2,038.5\text{ MB}$ | **$107.19\text{ ms}$** | **$19,106\text{ tok/s}$** |
| **2,048t** | `glypr-core` Q8_0 | Metal Flash (Fused) | **$606.3\text{ MB}$** | **$935.2\text{ MB}$** | **$328.9\text{ MB}$** | **$1,678.5\text{ MB}$** | **$87.68\text{ ms}$** | **$23,359\text{ tok/s}$** 🚀 |
| | | | | | | | | |
| **4,096t** | `glypr-core` BF16 | Metal Flash (Fused) | $785.0\text{ MB}$ | **$1,177.4\text{ MB}$** | **$392.4\text{ MB}$** | $2,038.6\text{ MB}$ | **$255.29\text{ ms}$** | **$16,045\text{ tok/s}$** |
| **4,096t** | `glypr-core` Q8_0 | Metal Flash (Fused) | **$606.3\text{ MB}$** | **$1,086.7\text{ MB}$** | **$480.4\text{ MB}$** | **$1,678.6\text{ MB}$** | **$225.10\text{ ms}$** | **$18,197\text{ tok/s}$** |
| | | | | | | | | |
| **8,192t** | **PyTorch MPS (Golden)** | **MPSGraph SDPA** | **$1,472.5\text{ MB}$** | **$6,592.8\text{ MB}$** | $5,120.3\text{ MB}$ | **$1,211.1\text{ MB}$** | **$875.00\text{ ms}$** | **$9,360\text{ tok/s}$** |
| **8,192t** | `glypr-core` BF16 (Unopt) | Standard Dense | $781.0\text{ MB}$ | $4,608.2\text{ MB}$ | $3,827.2\text{ MB}$ | $2,037.2\text{ MB}$ | $2,937.26\text{ ms}$ | $2,789\text{ tok/s}$ |
| **8,192t** | `glypr-core` BF16 (Opt) | Chunked Sliding | $785.0\text{ MB}$ | **$2,880.0\text{ MB}$** | **$2,095.0\text{ MB}$** | $2,038.5\text{ MB}$ | $1,118.50\text{ ms}$ | $7,324\text{ tok/s}$ |
| **8,192t** | **`glypr-core` BF16 (Flash)** | **Metal Flash (Fused)** | **$785.0\text{ MB}$** | **$1,473.5\text{ MB}$** | **$688.4\text{ MB}$** | $2,038.5\text{ MB}$ | **$670.79\text{ ms}$** | **$12,212\text{ tok/s}$** |
| **8,192t** | **`glypr-core` Q8_0 (Flash)** | **Metal Flash (Fused)** | **$606.3\text{ MB}$** | **$1,582.7\text{ MB}$** | **$976.4\text{ MB}$** | **$1,678.5\text{ MB}$** | **$615.30\text{ ms}$** 🏆 | **$13,314\text{ tok/s}$** |

### Apple Neural Engine (ANE) vs Metal GPU Scorecard

Empirical benchmarks measured on Apple Silicon in native release profile:

#### 1. Vision Tower Execution (Single Image)
| Modality / Budget | Input Patches | Soft Tokens | Apple Neural Engine (ANE) | Apple Silicon Metal GPU (ViT) | ANE Speedup | Parity vs GPU mmproj |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Budget 140 Portrait** | 1,134 patches | 126 tokens | **11.17 ms (89.5 img/s)** | 24.03 ms (41.6 img/s) | **2.2× faster** | **`0.999928` Cosine Sim** |
| **Budget 280 Portrait** | 2,394 patches | 266 tokens | **23.19 ms (43.1 img/s)** | 55.30 ms (18.1 img/s) | **2.4× faster** | **`0.999928` Cosine Sim** |

#### 2. Resident Text Backbone (Static 512 Tokens)
| Sequence Length | Hardware Engine | Compute Units | Latency | Throughput | Relative Speedup |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **16 tokens** | Apple Silicon Metal GPU | GPU | $4.21\text{ ms}$ | $237.5\text{ q/s}$ | 1.96× |
| **16 tokens** | Apple Neural Engine (ANE) | ANE (W8A16) | $8.24\text{ ms}$ | $121.3\text{ q/s}$ | 1.00× (ref) |
| **128 tokens** | Apple Silicon Metal GPU | GPU | $7.85\text{ ms}$ | $127.4\text{ q/s}$ | 1.05× |
| **128 tokens** | Apple Neural Engine (ANE) | ANE (W8A16) | **$8.24\text{ ms}$** | **$121.3\text{ q/s}$** | 1.00× |
| **512 tokens (RAG chunk)** | Apple Silicon Metal GPU | GPU | $21.81\text{ ms}$ | $45.8\text{ q/s}$ | 1.00× (ref) |
| **512 tokens (RAG chunk)** | **Apple Neural Engine (ANE)** | **ANE (W8A16)** | **$8.24\text{ ms}$** 🚀 | **$121.3\text{ q/s}$** | **2.6× faster** |

---

## Golden Parity Across 292 Vectors Scorecard

Validation against the official PyTorch MPS Golden Oracle across 10 languages, code, and 4 Matryoshka dimensions (768d, 512d, 256d, 128d):

| Model & Precision | Attention Engine | Full Suite Time | Evaluation Rate | Avg Parity | Min Parity | Speedup vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **GGUF Q8_0** | **Metal Flash (Fused)** | **$3.50\text{ s}$** | **$83.43\text{ vec/s}$** | **0.999873** | **0.999668** | **4.03x** |
| 🥈 **GGUF BF16** | **Metal Flash (Fused)** | **$4.10\text{ s}$** | **$71.22\text{ vec/s}$** | **0.999951** | **0.999860** | **3.44x** |
| 🥉 **PyTorch MPS (Ref)**| **MPSGraph SDPA** | **$5.73\text{ s}$** | **$50.96\text{ vec/s}$** | 1.000000 | 1.000000 | **2.46x** |
| **GGUF BF16** | **Chunked Sliding** | **$6.35\text{ s}$** | **$45.98\text{ vec/s}$** | **0.999943** | **0.999839** | **2.22x** |
| **GGUF Q8_0** | **Chunked Sliding** | **$6.45\text{ s}$** | **$45.27\text{ vec/s}$** | **0.999860** | **0.999620** | **2.19x** |
| **GGUF BF16** | **Standard Dense** | **$14.12\text{ s}$** | **$20.68\text{ vec/s}$** | **0.999943** | **0.999839** | **1.00x** (ref) |

---

## Universal Cross-Platform CPU SIMD

`glypr-core` integrates portable runtime-dispatched SIMD via `pulp` with zero intermediate heap allocations:
- **ARM64 NEON:** Accelerated on Apple Silicon (M1–M4) and AWS Graviton 3/4.
- **x86_64 AVX2 / AVX-512:** Automatic runtime detection on Intel and AMD processors.
- **Microbenchmarks:** GeGLU delivers **10.49x speedup** ($21.5\text{ ms} \to 2.05\text{ ms}$); RMSNorm delivers **1.55x speedup** ($0.34\text{ ms} \to 0.22\text{ ms}$).
- **Heap Churn Elimination:** Intermediate allocations reduced from ~30 GB per 8K sequence to **0**, keeping CPU RSS flat at **1.0–1.2 GB** (Q8_0).

---

---

## Multimodal (`mmproj`) Golden Parity Scorecard

Validated against official Python `SentenceTransformer("google/embeddinggemma-2")` on the multimodal golden benchmark with exact numerical alignment across all 24 test cases:

| Modality & Case | Input Data | Prompt Query | Soft Tokens | Total Seq Len | Target | Measured Parity | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🖼️ **Image Only** | 224x224 RGB image | *None* | 280 | 284 | $> 0.999$ | **0.999829** | ✅ **PASSED** |
| 🖼️ **Image + Prompt** | 224x224 RGB image | `"Describe this picture"` | 280 | 288 | $> 0.999$ | **0.999719** | ✅ **PASSED** |
| 🎙️ **Audio Only** | 1.0s 16 kHz WAV | *None* | 25 | 29 | $> 0.999$ | **0.999644** | ✅ **PASSED** |
| 🎙️ **Audio + Prompt** | 1.0s 16 kHz WAV | `"A musical sound"` | 25 | 32 | $> 0.999$ | **0.999852** | ✅ **PASSED** |
| 🖼️ **Lenna Portrait** | 512x512 RGB | Natural portrait prompt | 280 | 293 | $> 0.999$ | **0.999672** | ✅ **PASSED** |
| 🎙️ **Lena Speech** | 12.27s 16 kHz speech | English speech prompt | 306 | 316 | $> 0.999$ | **0.999961** | ✅ **PASSED** |
| 🖼️ **Baboon (Square Texture)** | 267x266 RGB | *None* | 280 | 284 | $> 0.999$ | **0.999947** | ✅ **PASSED** |
| 🖼️ **Pepper (Square Gradient)**| 512x512 RGB | *None* | 280 | 284 | $> 0.999$ | **0.999876** | ✅ **PASSED** |
| 🖼️ **Countryside (16:9 Landscape)**| 1920x1080 RGB | *None* | 270 | 274 | $> 0.999$ | **0.999695** | ✅ **PASSED** |
| 🖼️ **Dog (2:3 Portrait)** | 1280x1920 RGB | *None* | 288 | 292 | $> 0.999$ | **0.999776** | ✅ **PASSED** |
| 🎙️ **Female Speech** | 12.68s 16 kHz audio | *None* | 317 | 321 | $> 0.999$ | **0.999900** | ✅ **PASSED** |
| 🎙️ **Acoustic Music** | 5.00s 16 kHz audio | *None* | 125 | 129 | $> 0.998$ | **0.999748** | ✅ **PASSED** |
| 🎚️ **Budget Sweep: 70 Tokens** | 512x512 Pepper | *None* | 70 | 74 | $> 0.998$ | **0.999377** | ✅ **PASSED** |
| 🎚️ **Budget Sweep: 140 Tokens**| 512x512 Pepper | *None* | 140 | 144 | $> 0.999$ | **0.999864** | ✅ **PASSED** |
| 🎚️ **Budget Sweep: 280 Tokens**| 512x512 Pepper | *None* | 280 | 284 | $> 0.999$ | **0.999876** | ✅ **PASSED** |
| 🎚️ **Budget Sweep: 560 Tokens**| 512x512 Pepper | *None* | 560 | 564 | $> 0.999$ | **0.999912** | ✅ **PASSED** |
| 🔀 **Multi-Image Composite** | Baboon + Pepper | Dual `<\|image\|>` prompt | 560 | 582 | $> 0.998$ | **0.999883** | ✅ **PASSED** |
| 🔀 **Interleaved Multimodal** | Baboon + Music | Text + Image + Audio | 405 | 422 | $> 0.998$ | **0.999904** | ✅ **PASSED** |
| 🎯 **Cross-Modal Retrieval** | Query vs Image / Audio | Semantic ranking | — | — | Separation | **Margin > +0.05** | ✅ **PASSED** |

Zero-overhead context window preflight check guarantees sequence length $< 8,320$ tokens before initiating neural tower forward passes ($< 50\,\mu\text{s}$ rejection). Zero memory overhead when `mmproj` is unattached.

---

## Documentation Navigation

Comprehensive technical documentation is organized in [`docs/`](docs/):

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
  - [`docs/prompts/task-instructions.md`](docs/prompts/task-instructions.md): Asymmetric and symmetric task instructions and title conditioning.

---

## License

Apache 2.0.
