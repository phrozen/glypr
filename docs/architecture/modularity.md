# Modular Encoder Loading & Memory Architecture

A signature architectural feature of **EmbeddingGemma 2** is its **modular encoder architecture**. Rather than forcing deployers to load a monolithic 740M-parameter multimodal model, the model architecture decouples the text backbone, vision tower, and audio tower into selectively loadable modules that **project into the exact same 768-dimensional latent space**.

---

## 1. Modular Configurations & Memory Footprints

When instantiating the model via `sentence-transformers`, unused modality encoders can be omitted at initialization by setting `vision_config` or `audio_config` to `None` in `config_kwargs`. 

Disabled modules are **never loaded from disk or allocated in RAM/VRAM**.

| Setup Name | Active Encoders | Parameter Count | `config_kwargs` Configuration | Target Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Text & Code Base** | Text Backbone only | **270M** | `{"vision_config": None, "audio_config": None}` | Codebase indexing, pure text RAG, document search, edge query engines (~191 MB RAM) |
| **Text + Vision** | Text Backbone + Vision Encoder | **440M** | `{"audio_config": None}` | Visual document search (ViDoRe), photo album search, eCommerce product listings |
| **Text + Audio** | Text Backbone + Audio Encoder | **570M** | `{"vision_config": None}` | Podcast/call-recording retrieval, voice-query search, acoustic event detection |
| **Full Multimodal** | Text + Vision + Audio | **740M** | `{}` | Complete omni-modal search, interleaved rich media indexing (~567 MB RAM quantized) |

---

## 2. Implementation in Python

### Text-Only Deployment (270M)

```python
from sentence_transformers import SentenceTransformer
import torch

MODEL_ID = "google/embeddinggemma-2"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

# Load only the 270M text backbone
text_model = SentenceTransformer(
    MODEL_ID,
    model_kwargs={"torch_dtype": dtype},
    config_kwargs={
        "vision_config": None,
        "audio_config": None,
    },
)

# Text and code queries execute with zero vision/audio memory overhead
query_emb = text_model.encode("async fn stream_response()", prompt_name="CodeRetrieval")
```

### Text + Vision Deployment (440M)

```python
# Text + Image & Video support (omitting 300M audio encoder)
vision_model = SentenceTransformer(
    MODEL_ID,
    model_kwargs={"torch_dtype": dtype},
    config_kwargs={
        "audio_config": None,
    },
)

# Can encode text, images, and video
image_emb = vision_model.encode({"image": "system_diagram.png"})
```

### Full Multimodal Deployment (740M)

```python
# All modalities active
omni_model = SentenceTransformer(
    MODEL_ID,
    model_kwargs={"torch_dtype": dtype},
)

# Interleaved or audio inputs
audio_emb = omni_model.encode({"audio": "speech_sample.wav"})
```

---

## 3. The Shared Latent Space: Zero Re-indexing

Because the vision and audio encoders were trained to align directly into the adapted Gemma 4 text representation space:

$$\mathcal{S}_{\text{text}} \equiv \mathcal{S}_{\text{vision}} \equiv \mathcal{S}_{\text{audio}} \subset \mathbb{R}^{768}$$

### What this unlocks for system design:
1. **Asymmetric Client-Server Architectures:**
   * **Server / Ingestion Pipeline:** Runs the **740M full multimodal model** to index complex documents containing text, diagrams, audio recordings, and video clips into a vector database (e.g., Qdrant, Milvus, pgvector).
   * **Client / Mobile Device:** Runs only the **270M text-only model** locally. The user types a text query, the local device embeds it into the 768d space, and queries the remote vector database directly.
   * **Result:** Perfect semantic similarity matching without needing vision or audio encoders on the client device.
2. **Evolutionary Indexing Without Re-computation:**
   * If an enterprise begins by indexing 10 million text documents with the 270M model, and later decides to ingest 1 million product photos or training videos, **no existing vectors need to be re-computed or re-indexed**. The new media is simply embedded using the vision encoder and inserted into the existing database.

---

## 4. Shared Weights Collocation with Gemma 4

When deploying an on-device agentic system (e.g., an edge device running local RAG with Gemma 4):
* **Shared Tokenizer:** EmbeddingGemma 2 and Gemma 4 use the exact same SentencePiece 262k vocabulary and token mapping. Tokens do not need re-tokenization between embedding retrieval and LLM generation.
* **Shared Audio Architecture:** EmbeddingGemma 2's audio front-end shares architectural components with Gemma 4 audio models, allowing unified execution graph caching on runtimes like **LiteRT** and **MLX**.

---

## 5. Native Rust Engine Modularity (`glypr-core`)

In the native Rust implementation (`glypr-core`), modularity is extended into the runtime execution graph itself, allowing zero-cost swaps of weight formats and attention execution backends.

```
                                  ┌─────────────────────────────┐
                                  │      GlyprEmbedder API      │
                                  └──────────────┬──────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
      ┌─────────────────────────┐                                 ┌─────────────────────────┐
      │   Weights & Formats     │                                 │   Attention Subsystem   │
      ├─────────────────────────┤                                 ├─────────────────────────┤
      │ • Safetensors (BF16)    │                                 │ Trait: AttentionBackend │
      │ • GGUF BF16 (Zero-diff) │                                 │ • Standard Dense        │
      │ • GGUF Q8_0 (295 MB)    │                                 │ • Chunked Sliding (3W)  │
      │ • Embedded Tokenizer    │                                 │ • [Hook] MetalFlash     │
      └─────────────────────────┘                                 │ • [Hook] CudaFlash      │
                                                                  └─────────────────────────┘
```

### A. Pluggable Attention Backend Trait
Attention computation is decoupled from model layers through the `AttentionBackend` trait:

```rust
pub trait AttentionBackend: Send + Sync {
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
    ) -> Result<Tensor>;
}
```

Deployers can switch attention backends dynamically at runtime without reloading weights:

```rust
use glypr_core::{GlyprEmbedder, AttentionMode};

let mut embedder = GlyprEmbedder::load_gguf_q8("embeddinggemma-2-Q8_0.gguf", &device)?;

// Select attention backend:
// - AttentionMode::Standard       (Dense baseline)
// - AttentionMode::ChunkedSliding  (Universal CPU/GPU 3W banded attention)
// - AttentionMode::MetalSdpa       (Apple Silicon fused SDPA with broadcasted mask)
// - AttentionMode::MetalFlash      (Dedicated tile-skipping MSL FlashAttention, 0 MB mask)
embedder.set_attention_mode(AttentionMode::MetalFlash);
```

### B. Single-Artifact Deployment (GGUF Containerization)
Rather than managing multi-file directories containing separate `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `model.safetensors` files, `glypr-core` supports single-file self-contained GGUF deployments:
* **Embedded Vocab & Merges:** All 262,144 token IDs, scores, and merge rules are embedded directly inside the GGUF header and reconstructed into an in-memory BPE tokenizer in milliseconds.
* **Quantized PLE Projections:** Both standard attention matrices and the $512 \to 12,288$ per-layer embedding projections are loaded directly from quantized GGUF tensors, preserving 100% numerical parity ($\text{Avg Parity} = 0.999865$ on Q8_0).

### C. Dynamic Multimodal Projector (`mmproj`) Decoupling
To keep text-only deployments minimal (only 295 MB for Q8_0), vision and audio projector towers are decoupled into optional `mmproj-*.gguf` files:

```rust
// 1. Text embedding runs standalone from a 295 MB Q8_0 container
let mut embedder = GlyprEmbedder::load_gguf_q8("embeddinggemma-2-Q8_0.gguf", &device)?;

// 2. Attach vision/audio projector on demand when multimodal search is needed
embedder.attach_multimodal_projector("mmproj-embeddinggemma-2-Q8_0.gguf")?;
```
This guarantees zero memory overhead when processing text/code, while enabling seamless runtime upgrading to full multimodal search.
