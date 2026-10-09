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

By eliminating intermediate tensor allocations across two critical architectural stages and deploying custom MSL shaders, `glypr-core` cut peak Metal VRAM down to **$3.09\text{ GB}$ (Chunked)** and **$3.95\text{ GB}$ (Metal Flash)**—delivering both the lowest memory footprint and lowest latency (**$868.00\text{ ms}$**, beating PyTorch MPS).

### The Two Root-Cause Bottlenecks:

1. **Un-sliced 4D PLE RMSNorm (+2,013 MB):**
   * *Problem:* `TextPLE` projected embeddings into $[1, 8192, 24, 512]$ and evaluated `RMSNorm` across the entire 4D tensor in float32. This created 5 intermediate 4D tensors ($402.6\text{ MB} \times 5 = \mathbf{2,013\text{ MB}}$) before Layer 0 even executed.
   * *Solution:* Normalization operates strictly along the hidden dimension ($\text{dim}=-1$, size 512) and is independent across layers. Normalizing each layer's slice $[1, 8192, 512]$ on-demand reduced active norm allocations to $\mathbf{16.7\text{ MB}}$ with **exact $0.0$ numerical difference**.

2. **Dense Full Attention Softmax on Layers 5, 11, 17, 23 (+5,304 MB):**
   * *Problem:* The 4 global attention layers materialized $[1, 4, 8192, 8192]$ matrices. In float32 softmax, element-wise `sub`, `exp`, and `div` operations allocated over **$5.3\text{ GB}$** of transient buffers.
   * *Solution:* Query-chunked block attention. Chunking $Q$ into blocks of $512$ tokens attending to all $8,192$ keys reduced the active matrix from $8192 \times 8192$ ($1.07\text{ GB}$) to $512 \times 8192$ ($67\text{ MB}$). Because row-wise softmax normalizes across keys independently for each query token, this yields **$100\%$ bit-for-bit mathematical identity**.

### Empirical Hardware Metal VRAM Scorecard (Apple Silicon)

| Context Length | Framework / Variant | Attention Engine | Base Model RAM | Peak Metal VRAM | Transient VRAM | Max Process RSS | Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1,024t** | `glypr-core` BF16 | Standard Dense | $781.0\text{ MB}$ | $1,045.7\text{ MB}$ | $264.6\text{ MB}$ | $2,037.3\text{ MB}$ | $131.90\text{ ms}$ |
| **1,024t** | `glypr-core` BF16 | Chunked Sliding | $781.0\text{ MB}$ | $1,180.3\text{ MB}$ | $399.2\text{ MB}$ | $2,037.4\text{ MB}$ | $121.27\text{ ms}$ |
| **1,024t** | `glypr-core` BF16 | Metal Flash (Fused) | $781.0\text{ MB}$ | **$1,015.3\text{ MB}$** | **$234.2\text{ MB}$** | $2,039.2\text{ MB}$ | **$81.96\text{ ms}$** |
| **1,024t** | `glypr-core` Q8_0 | Metal Flash (Fused) | **$540.9\text{ MB}$** | **$779.2\text{ MB}$** | $238.2\text{ MB}$ | **$1,677.3\text{ MB}$** | **$84.11\text{ ms}$** |
| | | | | | | | |
| **2,048t** | `glypr-core` BF16 | Standard Dense | $781.0\text{ MB}$ | $1,370.0\text{ MB}$ | $589.0\text{ MB}$ | $2,037.2\text{ MB}$ | $316.72\text{ ms}$ |
| **2,048t** | `glypr-core` BF16 | Chunked Sliding | $781.0\text{ MB}$ | $1,430.3\text{ MB}$ | $649.2\text{ MB}$ | $2,037.3\text{ MB}$ | $247.34\text{ ms}$ |
| **2,048t** | `glypr-core` BF16 | Metal Flash (Fused) | $781.0\text{ MB}$ | **$1,253.3\text{ MB}$** | **$472.2\text{ MB}$** | $2,037.2\text{ MB}$ | **$162.69\text{ ms}$** |
| **2,048t** | `glypr-core` Q8_0 | Metal Flash (Fused) | **$540.9\text{ MB}$** | **$1,017.2\text{ MB}$** | $476.2\text{ MB}$ | **$1,677.3\text{ MB}$** | **$162.12\text{ ms}$** |
| | | | | | | | |
| **4,096t** | `glypr-core` BF16 | Standard Dense | $781.0\text{ MB}$ | $2,118.8\text{ MB}$ | $1,337.7\text{ MB}$ | $2,037.3\text{ MB}$ | $928.30\text{ ms}$ |
| **4,096t** | `glypr-core` BF16 | Chunked Sliding | $781.0\text{ MB}$ | $2,006.5\text{ MB}$ | $1,225.5\text{ MB}$ | $2,039.4\text{ MB}$ | $555.57\text{ ms}$ |
| **4,096t** | `glypr-core` BF16 | Metal Flash (Fused) | $781.0\text{ MB}$ | **$1,854.4\text{ MB}$** | **$1,073.3\text{ MB}$** | $2,037.2\text{ MB}$ | **$350.59\text{ ms}$** |
| **4,096t** | `glypr-core` Q8_0 | Metal Flash (Fused) | **$540.9\text{ MB}$** | **$1,622.3\text{ MB}$** | $1,081.3\text{ MB}$ | **$1,677.3\text{ MB}$** | **$360.44\text{ ms}$** |
| | | | | | | | |
| **8,192t** | **PyTorch MPS (Golden)**| **MPSGraph SDPA** | **$1,472.5\text{ MB}$** | **$6,592.8\text{ MB}$** | $5,120.3\text{ MB}$ | **$1,211.1\text{ MB}$** | **$875.50\text{ ms}$** |
| **8,192t** | `glypr-core` BF16 | Standard Dense | $781.0\text{ MB}$ | $4,608.2\text{ MB}$ | $3,827.2\text{ MB}$ | $2,037.2\text{ MB}$ | $3,133.98\text{ ms}$ |
| **8,192t** | `glypr-core` BF16 | Chunked Sliding | $781.0\text{ MB}$ | **$3,094.6\text{ MB}$** | **$2,313.5\text{ MB}$** | $2,037.2\text{ MB}$ | $1,288.27\text{ ms}$ |
| **8,192t** | **`glypr-core` BF16 (Flash)** | **Metal Flash (Fused)** | **$781.0\text{ MB}$** | **$3,951.5\text{ MB}$** | $3,170.4\text{ MB}$ | $2,037.2\text{ MB}$ | **$868.00\text{ ms}$ 🏆** |
| **8,192t** | **`glypr-core` Q8_0 (Flash)** | **Metal Flash (Fused)** | **$540.9\text{ MB}$** | **$3,727.4\text{ MB}$** | $3,186.4\text{ MB}$ | **$1,677.3\text{ MB}$** | **$894.78\text{ ms}$** |

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

* **Bit-for-Bit Weight Verification (`BF16.gguf`):** Every single weight tensor in `embeddinggemma-2-BF16.gguf` matches `model.safetensors` bit-for-bit ($\text{max\_diff} = 0.0\text{e}0$).
* **Storage Footprint (`Q8_0.gguf`):** Reduces model disk storage by **$-44.5\%$** ($532.1\text{ MB} \longrightarrow \mathbf{295.5\text{ MB}}$), loading in just $1.05\text{ s}$.
* **Numerical Parity:** Across the 292-vector golden corpus, GGUF Q8_0 achieves **$0.999864$ average cosine parity** and **$0.999602$ minimum parity**, maintaining identical retrieval margins ($+0.2952$ multilingual, $+0.1498$ length scaling).

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
// 1. Single-file self-contained text embedding load (no config.json or tokenizer.json needed!)
let mut embedder = GlyprEmbedder::load_gguf_q8("embeddinggemma-2-Q8_0.gguf", &device)?;

// 2. Optionally attach multimodal projector when image/audio inputs are detected
embedder.attach_multimodal_projector("mmproj-embeddinggemma-2-Q8_0.gguf")?;
```

```bash
# 1. Automatic resolution from Hugging Face cache (defaults to auto Metal GPU & BF16):
glypr embed "Single-file text embedding"

# 2. Opt into Q8_0 weights with Matryoshka dimension truncation and JSON output:
glypr --q8 embed "Single-file text embedding" --dim 256 -o json

# 3. Interleaved multimodal with automatic mmproj resolution and attachment:
glypr embed "Multimodal query:" ./image.png

# 4. Explicit local file overrides (disables auto-downloading):
glypr --model ./custom.gguf --mmproj ./custom-mmproj.gguf embed "Local model query"

# 5. Launch high-throughput OpenAI-compatible HTTP embedding server:
glypr serve --port 8080 --max-batch-size 32
```

---

## 9. Universal Cross-Platform CPU SIMD Deployment

For server deployments without discrete GPUs or edge machines running Linux/macOS, `glypr-core` ships with compile-time and runtime auto-dispatched SIMD via `pulp`:

* **ARM64 NEON:** Automatically enabled on Apple Silicon (M1–M4) and AWS Graviton 3/4 processors.
* **x86_64 AVX2 / AVX-512:** Dynamically detected at process launch on Intel Xeon and AMD EPYC servers.
* **Rayon Multi-Threading:** Batches sequence chunks across available CPU hardware cores.
* **Memory Flatness:** Slashes intermediate CPU heap allocations from ~30 GB per forward pass down to zero, maintaining process RSS at **1.0–1.2 GB** across sequence lengths.
