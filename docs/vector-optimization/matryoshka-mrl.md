# Matryoshka Representation Learning (MRL) & Vector Optimization

EmbeddingGemma 2 was trained natively with **Matryoshka Representation Learning (MRL)**. During training, the loss function penalizes error not just across the full 768-dimensional output, but across prefix sub-vectors of dimensions **512**, **256**, and **128**.

This forces the model to pack the highest-variance, most discriminative semantic features into the earliest dimensions of the embedding vector.

---

## 1. Benchmark Results Across Truncation Dimensions

The table below reports official benchmark results across all supported MRL dimensions:

| Output Dimension | Compression Ratio | Memory / 1M Vectors (BF16) | MTEB Multi v2 *(Mean Task)* | MTEB Eng v2 *(Mean Task)* | MTEB Code v1 *(NDCG@10)* | MIEB Lite *(Mean Type)* | MMEB v2 Overall | MSEB Retrieval *(MRR@10)* | MAEB *(Mean Task)* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **768d (Full)** | **1:1** | **~1.5 GB** | **61.36** | **68.46** | **78.68** | **64.64** | **59.01** | **69.54** | **49.39** |
| **512d** | **1:1.5** | **~1.0 GB** | **61.17** | **68.41** | **77.24** | **64.32** | **58.38** | **69.18** | **49.21** |
| **256d** | **1:3** | **~500 MB** | **60.41** | **67.78** | **76.18** | **63.13** | **56.24** | **66.76** | **48.91** |
| **128d** | **1:6** | **~250 MB** | **57.89** | **65.68** | **71.41** | **59.06** | **45.65** | **56.71** | **46.92** |

---

## 2. Dimension Selection Strategy

### 512 Dimensions (The Production Sweet Spot)
* **Quality Delta:** Drop of only **-0.19 on Multilingual**, **-0.05 on English**, and **-0.63 on MMEB v2**.
* **Storage Win:** Immediate **33% reduction** in memory and disk storage across vector stores (e.g., Qdrant, LanceDB, pgvector).
* **Speed Win:** 33% fewer floating-point operations per pairwise dot product or cosine distance calculation.
* **Hardware Alignment:** 512 dimensions align cleanly with AVX-512 vector registers and ARM NEON SIMD blocks.

### 256 Dimensions (Storage-Constrained Systems)
* **Quality Delta:** Retains **~99% of text/code accuracy** and **~95% of multimodal accuracy**.
* **Storage Win:** Cuts storage by **66.7% (3x compression)** down to ~500 MB per million vectors.
* **Best For:** Mobile devices, embedded appliances, and large-scale web indexing where index RAM budget is strictly limited.

### 128 Dimensions (Text-Only & Two-Stage Filtering)
* **Quality Delta:** Text holds ~90% accuracy, but **multimodal quality drops steeply** (MMEB drops from 59.01 to 45.65; audio/sound drops by ~13 points).
* **Warning:** **Never deploy 128d for multimodal (image/video/audio) search**. Use 128d strictly for pure text/code candidate shortlisting or coarse indexing.

---

## 3. Mandatory Engineering Rules

### ⚠️ Rule 1: Mandatory L2 Re-Normalization
When you slice a unit-length 768-dimensional vector down to 512d, 256d, or 128d, **the slice is no longer of unit length**:
$$\|\mathbf{v}_{1:512}\|_2 < \|\mathbf{v}_{1:768}\|_2 = 1.0$$
If you compute cosine similarity or dot product without re-normalizing, ranking quality degrades silently without raising an error.

```python
# Slicing manually: MUST re-normalize
import numpy as np

raw_vec = model.encode("sample text")  # 768d
sliced_vec = raw_vec[:512]             # 512d
norm_vec = sliced_vec / np.linalg.norm(sliced_vec)  # RE-NORMALIZED

# In SentenceTransformers (Python): Pass truncate_dim and normalize_embeddings=True
vec = model.encode(
    "sample text",
    truncate_dim=512,
    normalize_embeddings=True,  # Handles L2 normalization automatically
)

// In glypr-core (Rust): Engine returns native 768d unit vectors.
// Slice and re-normalize in O(K) using the dedicated glypr_core::util module:
use glypr_core::{truncate_mrl, Task};

let raw_768 = embedder.embed_text("sample text", Task::SearchQuery)?;
let vec_512 = truncate_mrl(&raw_768, 512)?;
assert_eq!(vec_512.len(), 512);
```

### ⚠️ Rule 2: Dimension Matching
Queries and documents **must share the exact same dimension**. You cannot score a 768d query vector against a 512d or 128d corpus vector.

---

## 4. The Matryoshka Two-Stage Retrieval Pattern

MRL allows you to implement ultra-fast hierarchical search across millions of records without running a secondary re-ranking model:

```
[Query]
   │
   ▼
Stage 1: Fast Coarse Filter (128d / 256d)
   │  Scan 1,000,000 vectors using vec[:128]
   │  Compute dot product in milliseconds
   ▼
[Top 100 Candidates]
   │
   ▼
Stage 2: Fine Re-ranking (512d / 768d)
   │  Re-rank the top 100 using vec[:512]
   ▼
[Final Top 10 Results]
```

### Python Implementation:

```python
import numpy as np

# Simulate indexed corpus with stored 512d vectors
corpus_embeddings = model.encode(corpus_docs, truncate_dim=512, normalize_embeddings=True)

# Query encoded at 512d
query_emb = model.encode(query, prompt_name="SearchQuery", truncate_dim=512, normalize_embeddings=True)

# 1. Coarse Stage: Slice both to 128d and re-normalize on the fly
q_128 = query_emb[:128] / np.linalg.norm(query_emb[:128])
c_128 = corpus_embeddings[:, :128] / np.linalg.norm(corpus_embeddings[:, :128], axis=1, keepdims=True)

coarse_scores = np.dot(c_128, q_128)
top_100_idx = np.argsort(coarse_scores)[-100:]

# 2. Fine Stage: Score the top 100 with full 512d precision
top_100_vectors = corpus_embeddings[top_100_idx]
fine_scores = np.dot(top_100_vectors, query_emb)
final_top_10 = top_100_idx[np.argsort(fine_scores)[-10:]]
```

### Rust Implementation (`glypr-core`):

```rust
use anyhow::Result;
use glypr_core::{cosine_similarity, truncate_mrl, GlyprEmbedder, Task};

fn two_stage_search(
    query_768: &[f32],
    corpus_768: &[Vec<f32>],
    top_k: usize,
) -> Result<Vec<(usize, f32)>> {
    // 1. Slice query to coarse 128d
    let q_128 = truncate_mrl(query_768, 128)?;

    // 2. Coarse Stage: Scan corpus with 128d dot product
    let mut coarse_scored: Vec<(usize, f32)> = corpus_768
        .iter()
        .enumerate()
        .map(|(idx, doc)| {
            let doc_128 = truncate_mrl(doc, 128).unwrap();
            let score = cosine_similarity(&q_128, &doc_128);
            (idx, score)
        })
        .collect();

    coarse_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let candidates = &coarse_scored[..coarse_scored.len().min(100)];

    // 3. Fine Stage: Re-rank top 100 with full 768d (or 512d) precision
    let mut fine_scored: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&(idx, _)| {
            let score = cosine_similarity(query_768, &corpus_768[idx]);
            (idx, score)
        })
        .collect();

    fine_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    fine_scored.truncate(top_k);
    Ok(fine_scored)
}
```

