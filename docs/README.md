# EmbeddingGemma 2: Technical Documentation & Developer Knowledge Base

> **Model:** `google/embeddinggemma-2`  
> **Release Date:** October 6, 2026  
> **Author:** Google DeepMind (Ian Ballantyne, Maarten Grootendorst, et al.)  
> **License:** Apache 2.0  
> **Base Architecture:** Gemma 4 Decoder with Modular Vision & Audio Towers  
> **Native Precision:** BF16 / FP32 (⚠️ **Strictly avoid FP16**)  

---

## Overview

**EmbeddingGemma 2** is a compact, open-weight multimodal embedding model that unifies **text (including code), images, video, and audio** into a single, shared **768-dimensional vector space**. 

Built upon the **Gemma 4** architecture, it replaces fragile multi-model retrieval pipelines with a single modular network featuring a **270M parameter text base** and optional **170M vision** and **300M audio** encoders (up to **740M total parameters**). All configurations project into the exact same latent space, allowing cross-modal search (e.g., text querying audio or image) without separate alignment models.

---

## Documentation Navigation

This documentation repository is organized into targeted, fact-checked modules:

```
docs/
├── README.md                              # Master index, executive summary & quickstart
├── architecture/
│   ├── specifications.md                  # Layer dimensions, heads, GQA, pooling, projection
│   ├── modularity.md                      # Selective encoder loading (270M, 440M, 570M, 740M)
│   └── hardware-acceleration.md           # [NEW] Metal GPU fusion, zero-copy GQA, and pulp CPU SIMD
├── deployment/
│   └── quantization-and-runtimes.md       # QAT INT4/INT8, LiteRT, GGUF/llama.cpp, vLLM, precision rules
├── benchmarks/
│   ├── leaderboards.md                    # MTEB/MMEB scores, competitor comparison, golden parity scorecard
│   └── memory-and-vram.md                 # [NEW] Metal VRAM profiling, transient churn, Pareto trade-offs
├── vector-optimization/
│   ├── matryoshka-mrl.md                  # MRL dimension truncation (768d, 512d, 256d, 128d), L2 normalization
│   └── quantization.md                    # [NEW] Static INT8, Binary 1-bit, 2-pass KNN, Turso BSQ8 blueprint
├── prompts/
│   └── task-instructions.md               # Asymmetric & symmetric task prefixes, title formatting
└── multimodal/
    ├── inputs-and-budgeting.md            # Vision token budget (70-1120), audio (16kHz), interleaving
    ├── mmproj-architecture.md             # SigLIP ViT and Conformer Audio dynamic projector towers
    └── gguf-nuances-and-conversions.md    # GGUF tensor naming, weight conversions, and alignment
```

---

## Core Specification Snapshot

| Parameter / Feature | Value | Engineering Significance |
| :--- | :--- | :--- |
| **Total Parameter Count** | **740M** | Modular: 270M (Text/Code) + 170M (Vision) + 300M (Audio) |
| **Base Architecture** | Adapted Gemma 4 | 24 layers, $d_{\text{model}} = 512$, $d_{\text{hidden}} = 2048$, GQA/MQA |
| **Context Window** | **8,192 tokens** | Shared globally across all modalities in a single input |
| **Native Output Dimension** | **768** | Linear projection ($512 \to 768$) followed by L2 normalization |
| **MRL Truncation Dims** | **512, 256, 128** | Dynamic slicing with near-lossless performance at 512d/256d |
| **Vocab Size** | **262,144** | Shared Gemma 4 vocabulary |
| **Attention Mechanism** | Sliding Window (512) | 5:1 hybrid (20 sliding GQA 4:2 + 4 global MQA 4:1) |
| **Pooling Type** | Mean Pooling | Computed over active token sequence before projection |
| **Precision Support** | **BF16, FP32, QAT INT4/INT8, GGUF Q8_0** | ❌ **Do NOT use FP16** (activations exceed FP16 dynamic range) |
| **License** | **Apache 2.0** | Full commercial and open-source use |

---

## Quickstart

### Installation

Requires `sentence-transformers` v6.1.0+ and `transformers`:

```bash
pip install -U "sentence-transformers[image,audio,video]" transformers
```

### Basic Text & Code Embedding

```python
import torch
from sentence_transformers import SentenceTransformer

# Select safe precision: BF16 on GPU, FP32 on CPU
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

# Load text-only slice (270M parameters)
model = SentenceTransformer(
    "google/embeddinggemma-2",
    model_kwargs={"torch_dtype": dtype},
    config_kwargs={"vision_config": None, "audio_config": None},
)

# Asymmetric retrieval encoding
query = "Fast inverse square root algorithm"
doc = "title: Q_rsqrt.c | text: float Q_rsqrt(float number) { long i; ... }"

query_emb = model.encode(query, prompt_name="CodeRetrieval")
doc_emb = model.encode(doc, prompt_name="Document")

similarity = model.similarity(query_emb, doc_emb)
print(f"Cosine Similarity: {similarity.item():.4f}")
```

### Cross-Modal Retrieval

```python
# Load full multimodal model (740M parameters)
model = SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": dtype})

# Encode raw image and audio files directly (no task prefix)
image_emb = model.encode({"image": "mountain_pass.jpg"})
audio_emb = model.encode({"audio": "thunderstorm.wav"})

# Encode query using SearchQuery task instruction
text_query_emb = model.encode("hiking under thunder and lightning", prompt_name="SearchQuery")

# Direct cross-modal cosine similarity
print("Text to Image:", model.similarity(text_query_emb, image_emb).item())
print("Text to Audio:", model.similarity(text_query_emb, audio_emb).item())
```

### Matryoshka Dimension Truncation (512d)

```python
# Encode directly with 512d truncation and mandatory L2 normalization
truncated_emb = model.encode(
    "Deep learning system optimization",
    prompt_name="SearchQuery",
    truncate_dim=512,
    normalize_embeddings=True,
)
print("Vector dimension:", truncated_emb.shape[-1])  # 512
```

### Native Rust Engine (`glypr-core`) Quickstart

High-performance, zero-overhead Rust inference on Apple Silicon Metal and Linux/macOS CPU:

```rust
use candle_core::Device;
use glypr_core::{
    cosine_similarity, truncate_mrl, AttentionMode, GlyprEmbedder, ModalityPart, Task,
    VISION_BUDGET_DEFAULT,
};

// Initialize Metal GPU device (or Device::Cpu with pulp SIMD acceleration)
let device = Device::new_metal(0)?;

// Load quantized GGUF (295.5 MB footprint) with embedded tokenizer
let mut embedder = GlyprEmbedder::load_gguf_q8("embeddinggemma-2-Q8_0.gguf", &device)?;

// Enable hardware-fused Metal FlashAttention (zero-copy GQA, 868 ms @ 8k tokens, beats PyTorch MPS)
embedder.set_attention_mode(AttentionMode::MetalFlash);

// 1. Embed query with strongly-typed Task conditioning (returns native 768d vector)
let query_vec = embedder.embed_text(
    "Fast inverse square root algorithm",
    Task::CodeRetrieval,
)?;

// 2. Embed document passage with optional title conditioning
let doc_vec = embedder.embed_text(
    "float Q_rsqrt(float number) { long i; ... }",
    Task::Document { title: Some("Q_rsqrt.c") },
)?;

// 3. Truncate to 512d with O(K) L2 re-normalization via dedicated util module
let query_512 = truncate_mrl(&query_vec, 512)?;
let doc_512 = truncate_mrl(&doc_vec, 512)?;

let sim = cosine_similarity(&query_512, &doc_512);
println!("Cosine Similarity (512d): {:.4}", sim);
```

### Native Rust CLI & OpenAI-Compatible HTTP Serving Engine (`glypr`)

```bash
# 1. Pre-fetch canonical models into local Hugging Face cache
glypr pull

# 2. Inspect active device, cache, and attention backend
glypr info

# 3. Embed text or multimodal sequences (defaults to auto Metal GPU, BF16, and Metal Flash Attention)
glypr embed "What is quantum mechanics?"
glypr embed "Card illustration:" ./card.png --image-token-budget 140

# 4. Launch high-throughput OpenAI-compatible HTTP server with dynamic batching
glypr serve --port 8080 --max-batch-size 32
```

---

## Critical Caveats & Engineering Lessons

1. **Precision Rule:** Never run in FP16 (`torch.float16`). EmbeddingGemma 2's activation magnitude triggers FP16 underflow/overflow, causing silent embedding corruption or NaNs. Always use `bfloat16`, `float32`, or official QAT INT4/INT8 / GGUF Q8_0.
2. **Unified Memory Profiling Rule:** On Apple Silicon, do **not** rely on standard CPU resident memory (`ps`, `getrusage`, or `MACH_TASK_BASIC_INFO`). Metal unified memory allocations are managed by the IOGPU driver and can quietly allocate $14+\text{ GB}$ while CPU RSS reports $1\text{ GB}$. Always query `[MTLDevice currentAllocatedSize]`.
3. **PLE Normalization Rule:** Never normalize Per-Layer Embeddings (PLE) across the entire 4D $[B, N, 24, 512]$ tensor in float32 all at once. Because RMSNorm operates along the last dimension (512), normalize each layer slice on-demand inside the layer loop to save over $2\text{ GB}$ of transient GPU memory with exact $0.0$ numerical difference.
4. **Attention Chunking Rule:** Naive full attention across 8,192 tokens materializes an $8192 \times 8192$ matrix, consuming $>5.3\text{ GB}$ during float32 softmax. In sliding layers, use a $3W$ window scratchpad; in global layers, chunk queries into blocks of 512 tokens to keep peak Metal VRAM under $3.1\text{ GB}$ with $100\%$ bit-for-bit mathematical identity.
5. **Hardware Tile-Skipping & Zero Mask Rule:** Naive attention masks allocate an $8192 \times 8192$ tensor ($134.2\text{ MB}$). By computing sliding window boundaries in-register in custom Metal Shading Language (`steel_attention_sliding.metal`) with tile-skipping ($|row - col| > 512$), memory overhead drops to 0 MB, and 8K token layers accelerate by $18.5\times$ ($249.5\text{ ms} \to 13.45\text{ ms}$).
6. **Zero-Copy GQA Head Indexing Rule:** Avoid calling `repeat_kv` prior to attention. Indexing key and value heads dynamically in GPU registers ($gqa\_factor = 2$) cuts sliding attention layer latency in half ($13.45\text{ ms} \to 6.76\text{ ms}$) with zero temporary tensor materialization.
7. **Activation Clamping Rule:** In deep Gemma-2 MLP layers, pre-activation logits can reach $g < -14.0$, causing cubic polynomial overflow in the PyTorch tanh approximation and yielding `NaN`s on Apple Silicon Metal. Clamping pre-activations to $[-10.0, 10.0]$ completely eliminates `NaN`s while preserving bit-for-bit golden parity.
8. **CPU SIMD Vectorization & Heap Elimination Rule:** On CPU, default framework operations allocate dozens of intermediate tensors for RMSNorm and GeGLU (~30 GB of allocation churn at 8K). By deploying `pulp` vectorized SIMD kernels (`simd_rmsnorm` and `simd_geglu`), GeGLU accelerates by $10.49\times$, RMSNorm by $1.55\times$, and heap allocation churn is reduced to zero.
9. **Task Instruction Rule:** Task prefixes (`task: ... | query: ...`) are **text-only**. Passing media (images, video, audio) with task instruction strings will corrupt multimodal representations.
10. **MRL Normalization Rule:** If slicing vectors manually (`vec[:512]`), you **must** apply L2 normalization afterwards. Slicing breaks unit length and silently degrades cosine similarity rankings.
11. **Document Title Rule:** `prompt_name="Document"` automatically prepends `title: none | text: {content}`. If your document has an actual title or filename, format it manually: `title: {title} | text: {content}`.
