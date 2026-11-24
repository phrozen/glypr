# Quantization, Numerical Precision & Runtime Deployment

Deploying EmbeddingGemma 2 in production requires understanding its numerical sensitivity, Quantization-Aware Training (QAT) guarantees, supported runtime ecosystems, and hardware memory dynamics.

---

## 1. Numerical Precision Guidelines

### ⚠️ Critical Warning: Do NOT Use FP16 (`torch.float16`)
EmbeddingGemma 2’s activation values throughout its intermediate attention layers exceed the dynamic range of standard IEEE 754 half-precision (`float16`):
* In `float16`, the model produces **NaNs** or silently corrupted embeddings without raising an explicit error.
* **Safe Formats:** Run inference exclusively in **`bfloat16`**, **`float32`**, or **INT4/INT8 (QAT / GGUF Q8_0)**.

```python
import torch
from sentence_transformers import SentenceTransformer

# Recommended programmatic dtype selection
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    dtype = torch.bfloat16  # Apple Silicon supports BF16
else:
    dtype = torch.float32

model = SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": dtype})
```

---

## 2. Quantization-Aware Training (QAT) at INT4 & INT8

Google DeepMind trained official **INT4** and **INT8** checkpoints using **Quantization-Aware Training (QAT)** rather than naive Post-Training Quantization (PTQ):

### Why QAT is essential for embeddings:
* **The PTQ Problem:** In dense retrieval, 4-bit PTQ distorts vector angular distances, scattering cosine rankings and causing 3–6% drops on MTEB leaderboards.
* **The QAT Advantage:** During QAT, weights adapt to the 4-bit rounding grid throughout training. The resulting INT4 model achieves **<1.5% degradation vs. raw BF16**.
* **Memory & Storage:** Slashes model memory by up to 4x, allowing edge devices with strict RAM envelopes to maintain server-grade retrieval accuracy.

### On-Device Memory Footprint (Google Pixel 11 Pro Benchmarks)

| Model Slice | Precision | Active RAM Footprint | Device Target |
| :--- | :--- | :--- | :--- |
| **Text-Only (270M)** | **INT4 QAT** | **~191 MB** | Background keyboard suggestions, on-device notes search, edge RAG |
| **Text-Only (270M)** | **BF16** | **~540 MB** | Standard desktop / mobile background processes |
| **Full Multimodal (740M)** | **INT4 QAT** | **~567 MB** | Local gallery search, video moments finder, voice-search assistants |
| **Full Multimodal (740M)** | **BF16** | **~1.48 GB** | Server GPUs, local workstations, high-end laptops |

---

## 3. Supported Deployment Runtimes

### A. LiteRT (Formerly TensorFlow Lite)
* **Flagship Google Edge Runtime:** Distributed as pre-compiled `.litertlm` bundles via the LiteRT Community on Hugging Face.
* **Hardware Acceleration:** Native delegates for mobile NPUs, Qualcomm Snapdragon, Google Tensor, and mobile GPUs.
* **Use Case:** Native Android/iOS applications, edge appliances, embedded robotics.

### B. SentenceTransformers (Python)
* Primary library for Python workflows, LangChain, LlamaIndex, and local scripts:
```bash
pip install -U "sentence-transformers[image,audio,video]" transformers
```

### C. llama.cpp & Ollama (GGUF)
* Supported via GGUF format with `Q4_K_M`, `Q4_0`, and `Q8_0` quantizations.
* **Crucial Implementation Note:** When using GGUF for INT4/Q4 inference, ensure the GGUF was converted from the official **QAT checkpoint** rather than applying naive PTQ (`quantize`) to raw BF16 weights.

```bash
ollama run embeddinggemma-2
```

### D. vLLM & SGLang (High-Throughput Server Serving)
* Supported for high-concurrency microservice architectures serving embeddings over OpenAI-compatible `/v1/embeddings` endpoints.
* Enables continuous batching and PagedAttention for maximum token throughput.

### E. Apple Silicon Native (MLX & LMStudio)
* Native support on macOS via Apple's MLX framework, leveraging unified memory across CPU and Metal GPU.

### F. Native Rust Engine (`glypr-core` with Candle & Metal)
* Compiled Rust engine designed for zero-overhead deployment on Apple Silicon and Linux servers.
* Employs pluggable attention dispatch (`Standard`, `ChunkedSliding`, `MetalFlash`), custom MSL shaders, and universal cross-platform CPU SIMD (`pulp`).

### G. Domain Adaptation & Fine-Tuning (Unsloth)
* EmbeddingGemma 2 supports parameter-efficient fine-tuning (LoRA / QLoRA) via Unsloth for enterprise domain adaptation (e.g., medical, legal, specialized codebases).

---

## 4. Hardware Memory Profiling: The Apple Silicon Unified Memory Trap

On Apple Silicon (M-series chips), CPU and GPU share physical unified memory. However, macOS manages virtual memory page tables and IOGPU Metal allocations through separate kernel accounting subsystems.

### The Misleading Metric: CPU Resident Set Size (RSS)
Querying standard POSIX process memory (`ps`, `getrusage`, or `MACH_TASK_BASIC_INFO`) **only reports CPU page table resident memory**. Metal command buffer allocations and GPU unified memory buffers are managed directly by the IOGPU driver and **do not reliably reflect in process RSS**:
* **The Trap:** During long-context inference (8,192 tokens), process RSS might report a deceptively low $\sim 1\text{ GB}$, while the Metal driver is secretly allocating **$14.5\text{ GB}$** of unified memory.
* **The Solution:** True hardware allocations must be queried directly from the Metal device descriptor via `[MTLDevice currentAllocatedSize]`.

```rust
// Native Rust query via Metal / Objective-C runtime
#[cfg(target_os = "macos")]
#[link(name = "Metal", kind = "framework")]
#[link(name = "Foundation", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut std::ffi::c_void;
    fn sel_registerName(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn objc_msgSend(receiver: *mut std::ffi::c_void, selector: *mut std::ffi::c_void) -> usize;
}

pub fn get_metal_allocated_bytes() -> u64 {
    unsafe {
        let dev = MTLCreateSystemDefaultDevice();
        let sel = sel_registerName(b"currentAllocatedSize\0".as_ptr() as *const _);
        objc_msgSend(dev, sel) as u64
    }
}
```

---

## 5. Inference Memory Optimization Blueprint & Empirical Benchmarks

In deep-dive profiling on Apple Silicon Metal GPU, unoptimized transformer inference on 8,192 tokens consumed **$14.59\text{ GB}$**. PyTorch MPS consumed **$6.59\text{ GB}$**.

By eliminating intermediate tensor allocations, unifying FlashAttention across all 24 layers, and deploying custom MSL shaders with M5 Neural Accelerators, `glypr-core` (`v0.4.0`) cuts peak Metal VRAM at 8,192 tokens down to **$1,177.5\text{ MB}$ (BF16)** and **$1,102.8\text{ MB}$ (Q8_0)** (**$662.9\text{ MB}$ Max RSS**)—delivering both the lowest memory footprint and lowest latency (**$564.91\text{ ms}$** Q8_0 / **$620.81\text{ ms}$** BF16).

### The Two Root-Cause Bottlenecks:

1. **Un-sliced 4D PLE RMSNorm (+2,013 MB):**
   * *Problem:* `TextPLE` projected embeddings into $[1, 8192, 24, 512]$ and evaluated `RMSNorm` across the entire 4D tensor in float32. This created 5 intermediate 4D tensors ($402.6\text{ MB} \times 5 = \mathbf{2,013\text{ MB}}$) before Layer 0 even executed.
   * *Solution:* Normalization operates strictly along the hidden dimension ($\text{dim}=-1$, size 512) and is independent across layers. Normalizing each layer's slice $[1, 8192, 512]$ on-demand reduced active norm allocations to $\mathbf{16.7\text{ MB}}$ with **exact $0.0$ numerical difference**.

2. **Dense Full Attention Softmax on Layers 5, 11, 17, 23 (+5,304 MB):**
   * *Problem:* The 4 global attention layers materialized $[1, 4, 8192, 8192]$ matrices. In float32 softmax, element-wise `sub`, `exp`, and `div` operations allocated over **$5.3\text{ GB}$** of transient buffers.
   * *Solution:* Query-chunked block attention. Chunking $Q$ into blocks of $512$ tokens attending to all $8,192$ keys reduced the active matrix from $8192 \times 8192$ ($1.07\text{ GB}$) to $512 \times 8192$ ($67\text{ MB}$). Because row-wise softmax normalizes across keys independently for each query token, this yields **$100\%$ bit-for-bit mathematical identity**.

### Empirical Hardware Metal VRAM Summary (8,192 Tokens, Apple Silicon)

Captured in `v0.4.0` using `bench_memory --features metal,accelerate` (peak IOGPU via `[MTLDevice currentAllocatedSize]`, warm latency) alongside 20-iteration median latency from `compare_models`:

| Precision / Backend | Model VRAM | Peak Metal VRAM | Transient VRAM | Max Process RSS | 8K Warm / Med Latency | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch MPS *(prior ref)*** | $1,472.5\text{ MB}$ | $6,592.8\text{ MB}$ | $5,120.3\text{ MB}$ | $1,211.1\text{ MB}$ | $875.00\text{ ms}$ *(ref)* | $9,360\text{ tok/s}$ |
| **`glypr-core` BF16 (Standard)** | $665.4\text{ MB}$ | $3,858.1\text{ MB}$ | $3,192.6\text{ MB}$ | $2,046.2\text{ MB}$ | $2,970.26 / 2,940.54\text{ ms}$ | $2,786\text{ tok/s}$ |
| **`glypr-core` BF16 (Chunked)** | $665.4\text{ MB}$ | $2,241.6\text{ MB}$ | $1,576.2\text{ MB}$ | $2,046.4\text{ MB}$ | $1,127.60 / 1,113.36\text{ ms}$ | $7,358\text{ tok/s}$ |
| **`glypr-core` BF16 (`MetalFlash`)** | $665.4\text{ MB}$ | **$1,177.5\text{ MB}$** | **$512.0\text{ MB}$** | $2,046.5\text{ MB}$ | **$635.60 / 620.81\text{ ms}$** | **$13,196\text{ tok/s}$** |
| **`glypr-core` Q8_0 (`MetalFlash`)** | **$606.7\text{ MB}$** | **$1,102.8\text{ MB}$** | **$496.0\text{ MB}$** | **$662.8\text{ MB}$** | **$578.34 / 564.91\text{ ms}$** | **$14,502\text{ tok/s}$** 🏆 |

> See [Empirical Memory, VRAM & Transient Allocation Architecture](../benchmarks/memory-and-vram.md#3-empirical-hardware-memory-scorecard-apple-silicon) for the complete `1,024t–8,192t` Metal GPU sweep and `64t–4,096t` CPU SIMD scaling matrix.

---

## 6. Metal FlashAttention: Custom MSL Tile-Skipping Kernel Architecture

To eliminate the $134.2\text{ MB}$ attention mask tensor and surpass the latency of naive banded attention, `glypr-core` implements a dedicated Apple Silicon GPU Metal Shading Language kernel (`steel_attention_sliding.metal`):

### Kernel Design Highlights:
1. **Dynamic Tile-Skipping Range Calculation:**
   Query tokens are processed in tiles of size $B_Q = 32$, and key/value tokens in tiles of $B_K = 16$. For each query tile threadgroup, the active range of key tiles is computed analytically:
   ```metal
   int q_min = tid.x * BQ;
   int q_max = q_min + BQ - 1;
   int kb_start = (params->window_size > 0 && q_min > params->window_size)
       ? (q_min - params->window_size) / BK : 0;
   int kb_lim = (params->window_size > 0)
       ? metal::min(params->NK, (q_max + params->window_size + BK - 1) / BK)
       : params->NK;
   ```
   At 8,192 tokens with sliding window $W = 512$, each query block touches only $\sim 65$ key tiles out of $512$ total tiles, skipping **$>87\%$ of all matrix multiplication tiles**.
2. **In-Register Boundary Clamping (Zero Mask Allocation):**
   Instead of loading mask values from an external $8192 \times 8192$ memory buffer, boundary tiles evaluate distance directly within GPU registers:
   ```metal
   if (params->window_size > 0) {
       int row_pos = tid.x * BQ + (simd_gid * 8) + r;
       int col_pos = kb * BK + c;
       if (abs(row_pos - col_pos) > params->window_size) {
           acc = -INFINITY;
       }
   }
   ```
   This completely frees $134.2\text{ MB}$ of unified memory per inference call.
3. **Dispatch Geometry:**
   Dispatched with an explicit grid: `MTLSize { width: nq, height: h, depth: b }` and threadgroup: `MTLSize { width: 32, height: wm, depth: wn }` (where $w_m = 4, w_n = 1$, corresponding to 128 threads divided into 4 simdgroups of 32 threads).

---

## 7. GGUF Quantization in Production (`BF16` vs `Q8_0`)

* **Bit-for-Bit Weight Verification (`BF16.gguf`):** Every single weight tensor in `embeddinggemma-2-BF16.gguf` matches `model.safetensors` bit-for-bit ($\text{max\_diff} = 0.0\text{e}0$), achieving **`0.999967` avg / `0.999897` min** 292-vector parity under `MetalFlash` in `v0.4.0`.
* **Storage Footprint (`Q8_0.gguf`):** Reduces model disk storage by **$-44.5\%$** ($532.1\text{ MB} \longrightarrow \mathbf{295.5\text{ MB}}$), loading in **$0.543\text{ s}$** cold (`v0.4.0` `8 MiB` `BufReader`).
* **Numerical Parity:** Across the 292-vector golden corpus (`v0.4.0`), GGUF Q8_0 (`MetalFlash`) achieves **`0.999896` average cosine parity** and **`0.999741` minimum parity**.

---

## 8. Single-File Self-Contained GGUF Architecture & Dynamic Multimodal Attachment

In traditional transformer deployments, inference engines require an entire directory of accompanying artifacts:
* `config.json` (~4.5 KB) for layer counts, head dimensions, and RoPE frequencies.
* `tokenizer.json` (~32 MB) or `tokenizer.model` for BPE vocabularies, merges, and normalization rules.
* Model weights (`model.safetensors` or `.bin`).

`glypr-core` eliminates this fragmentation, turning `embeddinggemma-2-BF16.gguf` and `embeddinggemma-2-Q8_0.gguf` into **100% self-contained single-file deployables**.

### How Self-Containment Works:
1. **Embedded Architecture Metadata Extraction:**
   The GGUF container header encapsulates the complete model hyperparameter configuration:
   * `gemma-embedding2.block_count` $\to$ 24 layers.
   * `gemma-embedding2.embedding_length` $\to$ 512 hidden dimension.
   * `gemma-embedding2.attention.sliding_window_pattern` $\to$ 24-element boolean array defining the exact 5:1 sliding/global layer schedule.
   * `gemma-embedding2.attention.head_count_kv` $\to$ per-layer GQA (4:2) and MQA (4:1) key/value head allocations.
   * `gemma-embedding2.rope.freq_base_swa` & `freq_base` $\to$ $10,000$ (sliding) and $1,000,000$ (global) rotary base frequencies.
2. **Embedded BPE Tokenizer Construction:**
   Instead of bundling an external 32 MB `tokenizer.json`, `glypr-core` reconstructs the official Gemma 4 BPE tokenizer in memory directly from GGUF metadata:
   * Vocab table: 262,144 tokens extracted from `tokenizer.ggml.tokens`.
   * Merges table: 514,906 subword pairs extracted from `tokenizer.ggml.merges`.
   * Normalizer & Pre-tokenizer: Automatically applies `Replace(" ", "\u{2581}")` and `Split(" ", MergedWithPrevious)`.
   * **Verification:** Produces 100% token-for-token identical ID sequences across English, code, 10+ foreign languages, and all 292 benchmark corpus entries.
3. **Optional Dynamic Multimodal Projector (`mmproj`):**
   When multimodal capabilities (Vision & Audio) are needed, an optional projector GGUF (`mmproj-embeddinggemma-2-BF16.gguf` or `mmproj-embeddinggemma-2-Q8_0.gguf`) can be dynamically attached at runtime without affecting text-only memory overhead:

```rust
// 1. Single-file self-contained text embedding load (or GlyprEmbedder::builder().gguf_canonical(true).build()?)
let mut embedder = GlyprEmbedder::load_gguf_q8("embeddinggemma-2-Q8_0.gguf", &device)?;

// 2. Optionally attach multimodal projector when image/audio inputs are detected
embedder.attach_multimodal_projector("mmproj-embeddinggemma-2-Q8_0.gguf")?;
```

```bash
# 1. Automatic resolution from Hugging Face cache (defaults to auto Metal GPU & BF16):
glypr embed "Single-file text embedding"

# 2. Opt into Q8_0 weights with Matryoshka dimension truncation and JSON output:
glypr --q8 embed "Single-file text embedding" --dim 256 -o json

# 3. Interleaved multimodal with explicit projector opt-in:
glypr --bf16-mmproj embed "Multimodal query:" ./image.png

# 4. Explicit local file overrides:
glypr --model-file ./custom.gguf --mmproj-file ./custom-mmproj.gguf embed "Local model query"

# 5. Launch high-throughput OpenAI-compatible HTTP embedding server (local file paths off by default):
glypr serve --port 8080 --max-batch-size 32

# 6. Launch server with multimodal projector attached (add --allow-local-files only if trusted local paths are needed):
glypr --bf16-mmproj serve --port 8080
```

---

## 9. Universal Cross-Platform CPU SIMD Deployment

For server deployments without discrete GPUs or edge machines running Linux/macOS, `glypr-core` ships with compile-time and runtime auto-dispatched SIMD via `pulp` (plus optional `--features accelerate` Apple Accelerate support):

* **ARM64 NEON:** Automatically enabled on Apple Silicon (M1–M5) and AWS Graviton 3/4 processors.
* **x86_64 AVX2 / AVX-512:** Dynamically detected at process launch on Intel Xeon and AMD EPYC servers.
* **Rayon Multi-Threading:** Batches sequence chunks across available CPU hardware cores (`0.676 ms` GeGLU / `0.237 ms` RMSNorm in `v0.4.0` with `--features accelerate`).
* **Memory Flatness:** Slashes intermediate CPU heap allocations down to zero, maintaining deterministic process RSS across sequence lengths.

---

## 10. Apple Neural Engine (ANE) & Core ML Runtime Deployment

Introduced in **v0.3.0** and expanded in **v0.4.0**, Glypr provides native Apple Neural Engine (ANE) acceleration via `coreml-native` (features: `coreml`, `ane`).

### A. Core ML Model Compilation Pipeline
The offline export tool [`scripts/export_coreml.py`](../../scripts/export_coreml.py) converts raw Hugging Face safetensors into compiled `.mlmodelc` bundles with **W8A16 linear symmetric INT8 quantization**, writing `manifest.json` and removing intermediate `.mlpackage` bundles by default (`--keep-mlpackage` retains them):

```bash
# 1. Export multi-aspect vision presets + 512-token text backbone
uv run python scripts/export_coreml.py --target all --vision-preset default --output-dir models/coreml

# 2. Export specific vision presets (e.g., portrait, square, landscape)
uv run python scripts/export_coreml.py --target vision --vision-preset 140-portrait,140-square,140-landscape --output-dir models/coreml

# 3. Export multi-bucket Text Backbones (s1024 uses exact |i - j| <= 512 sliding-window masking)
uv run python scripts/export_coreml.py --target text --text-seq-lens 128,512,1024 --output-dir models/coreml
```

### B. Compiled Models, Grid Keying & `v0.4.0` Parity
Vision bundles are tagged `_h{H}_w{W}` and keyed by `(patch_height, patch_width)` so portrait and landscape orientations never collide; text bundles support multi-bucket discovery (`s128`, `s512`, `s1024`) routed via `TextBackendPolicy::Auto` (`[64..=seq_len]`, `batch_size <= 4`) and discovered via `resolve_coreml_dir` (`--coreml-dir` $\to$ `$GLYPR_COREML_DIR` $\to$ `./models/coreml` $\to$ user data/cache dir):

| Compiled Bundle | Compute Target | Input Dimensions | `v0.4.0` Parity / Prior Ref Latency |
| :--- | :---: | :---: | :--- |
| `gemma2_vision_tower_p1134_t126_h42_w27_w8a16.mlmodelc` (`140-portrait`) | ANE (W8A16) | `pixel_values: [1, 1134, 768]` | **11.17 ms (89.5 img/s)** *(v0.3.0 ref; `0.999928` hybrid parity)* |
| `gemma2_vision_tower_p1134_t126_h27_w42_w8a16.mlmodelc` (`140-landscape`) | ANE (W8A16) | `pixel_values: [1, 1134, 768]` | **`0.999955`** ANE vs Metal `mmproj` (`v0.4.0`) |
| `gemma2_vision_tower_p1089_t121_h33_w33_w8a16.mlmodelc` (`140-square`) | ANE (W8A16) | `pixel_values: [1, 1089, 768]` | **`0.999267`** vs PyTorch FP32 / **`0.999836`** ANE vs Metal (`v0.4.0`) |
| `gemma2_vision_tower_p2394_t266_h57_w42_w8a16.mlmodelc` (`280-portrait`) | ANE (W8A16) | `pixel_values: [1, 2394, 768]` | **23.19 ms (43.1 img/s)** *(v0.3.0 ref)* |
| `gemma2_text_backbone_s512_w8a16.mlmodelc` (`512t`) | ANE (W8A16) | `input_ids: [1, 512]` | **8.24 ms (121.3 q/s)** *(v0.3.0 ref)* |
| `gemma2_text_backbone_s1024_w8a16.mlmodelc` (`1024t`, sliding window) | ANE (W8A16) | `input_ids: [1, 1024]` | **`0.999684`** vs PyTorch FP32 / **`0.999601`** (`doc_science_512t`) / **`0.999700`** (`~770t` vs Metal, `v0.4.0`) |

### C. CLI & Server Deployment
```bash
# Install binary with Metal + ANE acceleration
cargo install glypr --features metal,ane
# Or from source: cargo install --path crates/glypr --features metal,ane --force

# Inspect a compiled CoreML bundle or GGUF header
glypr inspect models/coreml/gemma2_text_backbone_s1024_w8a16.mlmodelc

# Run hybrid embedding (Vision on ANE, Text on Metal GPU / ANE sweet spot)
glypr --ane embed ./portrait.png
glypr --ane --text-backend auto embed "Query within 1024 tokens"

# Run HTTP server with ANE acceleration (local filesystem paths disabled by default)
glypr --ane serve --port 8080
```
