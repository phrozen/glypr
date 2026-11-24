# Changelog

All notable changes to the Glypr project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] - 2026-11-23

### Breaking Changes
- **Encapsulated `GlyprEmbedder` Fields (`UX-10`):** Privatized `GlyprEmbedder` internal fields (`model`, `tokenizer`, `device`, `dtype`, `multimodal`, `max_seq_len`, `coreml_vision`, `coreml_text`, `vision_backend`, `text_backend`). Use the read/write accessors (`model()`, `model_mut()`, `tokenizer()`, `device()`, `dtype()`, `multimodal()`, `multimodal_mut()`, `max_seq_len()`, `set_max_seq_len()`, `with_max_seq_len()`, `embedding_dim()`, `vision_backend()`, `text_backend()`, `coreml_vision()`, `coreml_text()`, `has_multimodal()`, `has_audio_projector()`, `has_coreml_vision()`, `has_coreml_text()`), `GlyprEmbedder::builder()`, or `GlyprEmbedder::from_parts(...)` + `set_multimodal_projector(...)` / `with_multimodal_projector(...)`.
- **Structured `GlyprError` & `GlyprResult<T>` API Migration (`UX-10`):** Marked `GlyprError` as `#[non_exhaustive]`, exported `GlyprResult<T>`, and migrated public fallible methods on `GlyprEmbedder`, `GlyprBuilder`, `CoreMLVisionTower`, `CoreMLTextBackbone`, `glypr_core::hub`, `truncate_mrl`, and `validate_vision_budget` from `anyhow::Result<T>` to `GlyprResult<T>`.
- **Strict Validation & Preflight Guards:** Enforced strict MRL dimension bounds (`1..=embedding_dim`), interleaved/composite media & splicing checks, and CLI flag/pre-load validation (`--bf16` vs `--q8` conflict, `--ane` error without `coreml` feature, missing media path rejection, `--dim` bounds).
- **Default-Off Server Local File Reads:** `glypr-server` disables local filesystem media path reads by default across all bind addresses (`allow_local_files: false`); pass `--allow-local-files` to opt in.

### Added & Improved
- **M5 Metal 4 FlashAttention & Kernel Fusion:** Added unaligned sequence-length (`s`) FlashAttention support on M5 NAX, Q8_0 GEMM/GEMV tuning, Vision/Audio kernel fusion, and hybrid PLE optimizations.
- **Fast Loading & Startup Warmup:** Switched Safetensors loading to `mmap`, GGUF loading to an `8 MiB` `BufReader`, and added RoPE, Metal pipeline, and CoreML ANE startup warmup passes.
- **ANE Text Routing & CoreML Discovery (`UX-01`, `UX-02`, `UX-09`):** Added `CoreMLTextBackbone` multi-bucket discovery and `TextBackendPolicy::Auto` `[64..=512]`-token pure-text ANE routing (`batch_size <= 4`), `(patch_height, patch_width)` vision grid keying, deterministic `resolve_coreml_dir` lookup, and `scripts/export_coreml.py` square/landscape presets + `manifest.json`.
- **Builder & Offline Hub Ergonomics (`UX-05`, `UX-10`):** Added fluent `GlyprEmbedder::builder()` (`GlyprBuilder`) with safe Metal auto-detection and `--offline` / `HF_HUB_OFFLINE` cache-miss diagnostics.
- **CLI Batch/JSONL & Inspection (`UX-03`, `UX-12`):** Added `glypr embed --batch` / `--jsonl`, explicit `--image` / `--audio` / `--text-file` flags, and `glypr inspect <PATH>` for `.gguf` and `.mlmodelc` bundles.
- **Hardened `glypr-server` (`UX-04`):** Added `tracing_subscriber` initialization, `32 MiB` default body limit, `/ready` probe, graceful `SIGINT`/`SIGTERM` shutdown, worker panic isolation, per-item batch error isolation, and length-bucketed micro-batching.
- **CPU SIMD & Accelerate:** Added Rayon-parallelized BF16 CPU SIMD kernels and optional Apple Accelerate (`vvtanhf`) support.
- **Testing Pyramid & FP32 Golden Provenance:** Added weightless Tier 0 synthetic/mock/CLI tests (`GLYPR_ALLOW_MISSING_WEIGHTS=1`, `EngineHandle::mock`), Tier 1 (`./tests/scripts/run_qa_suite.sh --tier1`, 132 passed / 0 failed in 36s), and Tier 2 (`--tier2`, 106s including `9/9` `--release` tests, `pytest` `4/4`, and `26/26` live `differential_parity.py` checks). Regenerated all golden fixtures in FP32 CPU with `schema_version: 2` provenance (`verify_goldens.py --strict-fp32`).

### Fixed
- **GGUF Sliding-Window Radius:** Halved `gemma-embedding2.attention.sliding_window` (`1024 -> 512`) so GGUF long-context masking matches Safetensors/HuggingFace.
- **Audio Anti-Aliased Resampling:** Replaced naive linear resampling (`48 kHz -> 16 kHz`) with a zero-dependency windowed-sinc FIR low-pass resampler.
- **Metal BF16 NaN Guard & Kernel Safety:** Added BF16 NaN guards (`MLX_BF16`), sliding-window fallback fixes, and strict shape/dtype guards across Metal/CPU kernels.
- **Multimodal Prompt Ordering (`MultimodalInput`):** Fixed `MultimodalInput::image_with_prompt` and `MultimodalInput::audio_with_prompt` to emit `[Media, Text]` ordering matching `embed_composite`, `embed_interleaved`, and the Python baseline.
- **Audio STFT & Hann Precision Parity:** Updated audio preprocessor Hann window and FFT twiddle arithmetic to `f64` for exact parity against the PyTorch FP32 reference.
- **Padding Mask Broadcasting (`[B, 1, 1, S]`):** Fixed padded-batch `[B, 1, 1, S]` attention-mask slicing in `attention/chunked.rs` and `attention/standard.rs` (caught via Tier 0 synthetic tiny-model tests).

### Performance
- **Text Backbone Scaling (Metal 4 FlashAttention, `compare_models`):**
  - **GGUF BF16 (Metal Flash):** `64t = 9.94 ms` (`6,439 tok/s`), `256t = 17.81 ms` (`14,372 tok/s`), `512t = 27.03 ms` (`18,940 tok/s`), `1024t = 47.37 ms` (`21,616 tok/s`), `2048t = 95.29 ms` (`21,493 tok/s`), `4096t = 228.92 ms` (`17,892 tok/s`), `8192t = 620.81 ms` (`13,196 tok/s`); 292-vec parity Avg **`0.999967`** / Min **`0.999897`** (suite `3.61 s`).
  - **GGUF Q8_0 (Metal Flash):** `64t = 9.25 ms` (`6,922 tok/s`), `256t = 14.59 ms` (`17,551 tok/s`), `512t = 22.41 ms` (`22,851 tok/s`), `1024t = 39.05 ms` (`26,221 tok/s`), `2048t = 76.91 ms` (`26,627 tok/s`), `4096t = 199.54 ms` (`20,527 tok/s`), `8192t = 564.91 ms` (`14,502 tok/s`); 292-vec parity Avg **`0.999896`** / Min **`0.999741`** (suite `3.22 s`).
- **8K Metal Memory Footprint (`bench_memory`, `flash`):**
  - **BF16 Flash @ 8192t:** Peak Metal VRAM **`1,177.5 MB`** (down from `1,473.5 MB` in v0.3.0), Transient **`512.0 MB`** (down from `688.4 MB`), Model VRAM `665.4 MB`, End `590.3 MB`, Cold `628.55 ms`, Warm `621.14 ms`, Max RSS `2,046.4 MB`.
  - **Q8_0 Flash @ 8192t:** Peak Metal VRAM **`1,102.8 MB`** (down from `1,582.7 MB` in v0.3.0), Transient **`496.0 MB`** (down from `976.4 MB`), Max RSS **`662.9 MB`** (down from `1,678.5 MB`), Model VRAM `606.7 MB`, End `534.8 MB`, Cold `572.26 ms`, Warm `560.64 ms`.
- **Batched Execution (`bench_batch`, `B = 8`, Metal BF16):** `text_512_b8 = 132.93 ms` (`60.2 items/s`), `text_2048_b8 = 687.90 ms` (`11.6 items/s`), `text_4096_b8 = 1,759.32 ms` (`4.5 items/s`), `text_8192_b8 = 5,072.81 ms` (`1.6 items/s`), `vision_70_b8 = 152.76 ms` (`52.4 items/s`), `vision_140_b8 = 315.96 ms` (`25.3 items/s`), `vision_280_b8 = 703.62 ms` (`11.4 items/s`), `vision_560_b8 = 1,789.17 ms` (`4.5 items/s`), `hybrid_text_vision_b8 = 315.76 ms` (`25.3 items/s`), `hybrid_text_vision_audio_b8 = 423.20 ms` (`18.9 items/s`).
- **Multimodal Towers (`bench_multimodal`, Metal BF16):** 480×680 Vision Budget 140 averages `56.65 ms` (warm `50.34–51.16 ms`, Vision Tower `5.51 ms`, `17.65 cards/sec`); Budget 280 averages `119.98 ms` (warm `105.40–106.10 ms`, Vision Tower `8.72 ms`, `8.33 cards/sec`). Audio 1.0s averages `21.91 ms` (`45.6x` real-time), 5.0s averages `29.46 ms` (`169.7x` real-time), and 12.3s speech averages `51.74 ms` (Tower **`8.33 ms`**, **`237.2x`** real-time).
- **CPU SIMD (`--features accelerate`) & HTTP Server (`--features metal`):** CPU RMSNorm (`512x2304`) `0.237 ms` vs Candle `0.312 ms` (`1.32x`, 0 intermediate allocs); CPU GeGLU (`512x9216`) `0.676 ms` vs Candle `5.583 ms` (`8.26x`, 0 intermediate allocs). `glypr-server` achieves `7.82 ms` median single-query latency (`7.72 ms` min, `8.58 ms` P95), `24.23 ms` Batch 32 (**`1,320.6 vec/s`**), and **`1,226.9 req/s`** at 32 concurrent clients (`25.84 ms` median).
- **Live CoreML / ANE Export Parity (`scripts/export_coreml.py`):** `140-landscape` (`27x42`, `_h27_w42`) **`0.999955`** ANE vs Metal `mmproj`; `140-square` (`33x33`, `_h33_w33`) `0.999267` CoreML vs PyTorch FP32 / **`0.999836`** ANE vs Metal `mmproj`; `s1024` text backbone `0.999684` CoreML vs PyTorch FP32, **`0.999601`** ANE vs FP32 golden on `doc_science_512t`, and **`0.999700`** ANE vs Metal on `~770t` sliding-window text.

---

## [0.3.0] - 2026-10-20

### Added
- **Apple Neural Engine (ANE) Acceleration Subsystem:**
  - Integrated native Apple Neural Engine execution for EmbeddingGemma 2 via `coreml-native` (under optional `coreml` / `ane` Cargo features), reusing the existing `objc2 v0.3.2` ecosystem with zero conflicting C++ bindings.
  - **CoreML Vision Tower (`CoreMLVisionTower`):** Automatically discovers and introspects compiled `.mlmodelc` vision models, routes matching patch counts (e.g. 1,134 patches for Budget 140, 2,394 patches for Budget 280), feeds inputs zero-copy via `BorrowedTensor`, and emits embeddings directly to Candle `Tensor` with automatic data type alignment (FP32 $\to$ BF16) for seamless prompt splicing.
  - **Standalone CoreML Text Backbone (`CoreMLTextBackbone`):** Ultra-low-power, resident 512-token document and search query embedding engine executing in ~8.2 ms on pure ANE.
  - **Transparent Vision Backend Routing Policy (`VisionBackendPolicy`):**
    - `Auto`: Transparently dispatches matching portrait image dimensions to the Apple Neural Engine while gracefully falling back to GPU Metal `mmproj` for uncompiled dimensions or budgets.
    - `AneStrict`: Enforces strict execution on the Apple Neural Engine, failing fast with descriptive diagnostics if an image dimension is unsupported.
    - `Metal`: Enforces execution on the GPU Metal `mmproj` projector.
  - **Zero-mmproj Standalone Image Embedding:** Allows running image embeddings on Apple Silicon without downloading or attaching any GGUF `mmproj` file when using `--ane`.
  - **Offline PyTorch-to-CoreML Exporter (`scripts/export_coreml.py`):** Converts PyTorch safetensors into clean Core ML models with W8A16 linear symmetric INT8 weight quantization, strict macOS 14+ deployment target, and verified $>0.998$ cosine parity against PyTorch FP32 ground truth.
- **CLI & Server Flags:**
  - Added `--ane` flag to `glypr` CLI to enable Apple Neural Engine acceleration.
  - Added `--vision-backend <auto|ane|metal>` flag for explicit routing policy control.
  - Added `--coreml-dir <DIR>` to specify custom Core ML model search paths.
- **Clean Server Health & Model Introspection:**
  - Updated `/v1/models` and `/health` endpoints to sanitize device representations (e.g. clean canonical `"metal"` instead of debug string) and expose ANE metadata (`ane_enabled`, `vision_backend`, `ane_supported_patches`).
- **Mathematical Parity Verification:**
  - Rigorous end-to-end integration tests (`hybrid_ane_test.rs` and `coreml_parity.rs`) confirming **0.999928** cosine similarity between ANE Vision + Metal Text Backbone hybrid execution and pure GPU Metal GGUF reference inference.

### Performance
- **Vision Tower (Single Image):**
  - Budget 140 portrait (1,134 patches $\to$ 126 soft tokens): **11.17 ms (89.5 img/s)** on ANE vs **24.03 ms (41.6 img/s)** on GPU Metal ViT (**2.2× faster on ANE**).
  - Budget 280 portrait (2,394 patches $\to$ 266 soft tokens): **23.19 ms (43.1 img/s)** on ANE vs **55.30 ms (18.1 img/s)** on GPU Metal ViT (**2.4× faster on ANE**).
- **Text Backbone:**
  - Flat 8.24 ms execution curve (121.3 q/s) on ANE across 16 to 512 tokens.
  - At 512 tokens (standard RAG chunk): **8.24 ms** on ANE vs **21.81 ms** on GPU Metal (**2.6× faster on ANE**).

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
