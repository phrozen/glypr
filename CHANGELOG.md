# Changelog

All notable changes to the Glypr project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.1] - 2026-10-17

### Performance & Architectural Refinements
- **Metal Library Pipeline Caching:** Introduced a thread-safe `MTLLibrary` cache in `metal_kernels/mod.rs` keyed by physical device pointer (`dev_ptr`), compiling the 818-line MSL fused source once per device rather than 15 separate times per pipeline function. Drops iteration 0 cold-start latency 100x from **7,060 ms $\to$ 66.7 ms**.
- **2D Grid Dispatch for GeGLU (`geglu_gate_up_bf16`):** Replaced 1D thread indexing with 2D grid dispatch ($\text{grid} = (\frac{D + 255}{256}, \text{num\_rows}, 1)$, $\text{group} = (256, 1, 1)$). Direct register indexing via `id.x` and `id.y` completely eliminates GPU integer division (`/ D`) and modulo (`% D`), saving over 245 million integer division cycles per 8K sequence across the 26 decoder layers.
- **Zero-Allocation Image Preprocessor:** Replaced dynamic runtime allocation and multi-vector accumulation with a compile-time `const [f32; 256]` pixel normalization table and parallel chunked buffer writing via Rayon `par_chunks_exact_mut`, writing directly into target buffers without intermediate heap churn.
- **M5 NAX Hardware Tile Alignment:** Enforced static $16 \times 32$ cooperative tensor tile alignment ($s \pmod{64} == 0$), cleanly routing unaligned sequence lengths to `steel_attention_sliding` and aligned power-of-two lengths (64, 128, ..., 8192) to M5 NAX to prevent hardware out-of-bounds reads.

### Numerical Precision & Correctness Hardening
- **FP32 Long-Context Mean Pooling:** Converted token embeddings to FP32 before summation in `model.rs:pool_and_normalize`, eliminating BF16 7-bit mantissa swamping across sequences up to 8,192 tokens. Boosted 8,192-token cosine parity from `0.9873` to **`0.99975`** vs double-precision CPU ground truth.
- **Direct 64-Bit STFT Trigonometry:** Hardened `get_fft_512_twiddles` in `audio_preprocessor.rs` with direct 64-bit trigonometry, dropping iterative twiddle magnitude drift error 74x from $4.41 \times 10^{-6}$ down to **$5.96 \times 10^{-8}$** ($< 10^{-6}$ bound).
- **Online Softmax Fast-Math Guard:** Added `isinf` guard in `steel_attention_m5.metal` and `metal_flash.rs` preventing `-inf - (-inf) = NaN` under `-ffast-math`.
- **Memory Safety Invariant:** Reverted speculative uninitialized buffer allocation (`Tensor::empty`) in favor of safe zero-initialized buffers (`Tensor::zeros`) across all custom Metal kernels, ensuring zero dirty memory or memory-pressure bugs.

### Testing & Verification Harness
- **Phase 0 Regression Tripwires:** Added automated twiddle magnitude drift assertion, long-context synthetic pooling precision verification vs F64 ground truth, and cold vs warm latency metrics in `bench_memory.rs`.
- **Full Parity Validation:** Verified four-9s golden parity across all 292 text vectors and 24 multimodal test cases (`0.999844` BF16, `0.999679` Q8_0).

---

## [0.2.0] - 2026-10-17

### Added
- **Apple M5 & Metal 4 Neural Accelerators (NAX):** Runtime hardware capability detection (`MTLGPUFamilyMetal4` / `Apple10`) activating cooperative `mpp::tensor_ops::matmul2d` FlashAttention, SIMD-group hardware `reduce_rows`, native `bfloat` ALU, and M5 INT8 GEMM.
- **Rust 2024 Edition Migration:** Upgraded entire workspace (`glypr-core`, `glypr-server`, `glypr`) and dependencies to Rust 2024 edition.
- **Native Metal Q8_0 Dequantization GEMM/GEMV:** Direct block-quantized INT8 matrix-vector and matrix-matrix multiplications in Metal without F32 intermediate casts.
- **Full Backbone Kernel Fusion:** Single-pass MSL fused kernels for QKV RMSNorm + RoPE + Transpose, 4D PLE Scaling + RMSNorm, PLE GeGLU, and fused residual addition with logit soft-capping.
- **Unified Multimodal Batching Engine:** Batch processing for mixed text, image, and audio inputs across CLI and HTTP server, backed by Objective-C autorelease pool scoping.
- **Native Conformer Audio Acceleration:** 5-tap fused causal depthwise conv1d, Rayon-parallelized STFT preprocessor, cached filterbanks, and precomputed relative positional embeddings.
- **Native SigLIP Vision Tower Acceleration:** Hardware SDPA attention on Metal, 2-stage spatial grid pooling reduction, precomputed 2D axial RoPE, and Rayon parallel patch extraction.

### Performance
- **Beating PyTorch MPS at 8K Context:** 8,192-token inference latency dropped to **615.30 ms** (Q8_0) / **670.79 ms** (BF16), outperforming PyTorch MPS (875.00 ms) by up to 260 ms (1.42x speedup) with peak throughput reaching 23,359 tok/s at 2K context.
- **Ultra-Efficient Metal VRAM Footprint:** Through zero-copy GQA, native $d_{\text{head}}=512$ full-context kernels, fused backbone MSL operations, and zero-overhead weight concatenation views, peak Metal VRAM at 8,192 tokens dropped from **14.59 GB $\to$ 1.47 GB (BF16) / 1.58 GB (Q8_0)**, slashing transient VRAM by **86.6%** down to **688 MB** (saving over 5.1 GB vs PyTorch MPS).
- **Universal CPU SIMD:** Portable runtime architecture dispatch via `pulp` (ARM64 NEON & x86_64 AVX2/AVX-512) delivering a 10.49x isolated kernel speedup on GeGLU and eliminating ~30 GB of intermediate heap churn.
- **Server Dynamic Batching Throughput:** Achieved **1,303.9 vec/s** batch scaling and **1,202.6 req/s** concurrent dynamic batching on Apple Silicon.

### Changed
- Upgraded dependencies: `tokenizers` to 0.22, `tower-http` to 0.7, `base64` to 0.23.
- Enforced strict four-9s golden parity thresholds ($\ge 0.99980$ BF16, $\ge 0.99960$ Q8_0) across 292 golden text vectors and 24 multimodal test cases.

---

## [0.1.0] - 2026-10-09

### Added
- **Native Rust EmbeddingGemma 2 Inference Engine (`glypr-core`):**
  - Self-contained single-file GGUF deployment (`embeddinggemma-2-BF16.gguf` & `embeddinggemma-2-Q8_0.gguf`).
  - In-memory Gemma 4 262,144-token BPE tokenizer parser and reconstruction directly from GGUF metadata without external JSON files.
  - 24-layer transformer backbone supporting 5:1 hybrid attention (20 sliding-window + 4 global layers) and Per-Layer Embeddings (PLE).
  - Dynamic Multimodal Projector (`mmproj`) loading for SigLIP Vision Tower (170M) and Conformer Audio Tower (300M).
  - Matryoshka Representation Learning (MRL) dimension slicing ($768\text{d} \to 512\text{d}, 256\text{d}, 128\text{d}$) with automatic L2 re-normalization.
  - Static INT8 and 1-bit binary quantization with bit-packing and SIMD dot products.
  - Zero-overhead context window preflight validation ($< 8,320$ tokens).
- **Standalone CLI Binary (`glypr`):**
  - Subcommands: `glypr embed`, `glypr pull`, and `glypr info`.
  - Canonical Hugging Face model resolution and caching with `$HF_HOME` / `$HF_HUB_CACHE` awareness.
  - Multi-source input support: arguments, stdin pipes, raw binary streaming (`-o raw`), and formatted JSON arrays.
  - Multi-format media ingestion: `.png`, `.jpg`, `.jpeg`, `.webp` image formats, and 16 kHz `.wav` audio.
  - Configurable vision token budgets (70, 140, 280, 560, 1120 tokens).
- **OpenAI-Compatible HTTP Embedding Server (`glypr-server`):**
  - `/v1/embeddings` endpoint with adaptive request batching and concurrent client coalescing.
  - Support for text and multimodal base64 / URL payloads.
  - Health and model inspection endpoints (`/health`, `/v1/models`).
- **Documentation & Verification Suite:**
  - Full developer documentation, mathematical formulations, hardware specs, and task prompt guides in `docs/`.
  - 292-vector golden reference oracle and 24-case multimodal parity test suite.
