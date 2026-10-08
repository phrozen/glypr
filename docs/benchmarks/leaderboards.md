# Benchmark Evaluations & Leaderboard Comparisons

This document provides a comprehensive, fact-checked comparison of **EmbeddingGemma 2** against contemporary sub-1B embedding architectures: **IBM Granite-Embedding-Multilingual-R2**, **Voyage-4-Nano**, and **Jina-v5-Omni-Nano**, followed by ground-truth hardware execution benchmarks of Glypr (`glypr-core`) against the PyTorch MPS Golden Oracle.

---

## 1. EmbeddingGemma 2 Official Benchmark Results

All scores reflect the full-precision (BF16) 768-dimensional model:

| Modality | Benchmark Suite | Evaluation Metric | EmbeddingGemma 2 Score | EmbeddingGemma 1 Score | Relative Gain |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Code** | **MTEB (Code, v1)** | Mean(Task), NDCG@10 | **78.68** 🏆 | 68.76 | **+9.92 (+14.4%)** |
| **Text** | **MTEB (Multilingual, v2)** | Mean(Task), Multiple | **61.36** | 61.15 | +0.21 |
| **Text** | **MTEB (English, v2)** | Mean(Task), Multiple | **68.46** | — | Native |
| **Image** | **MIEB (Lite)** | Mean(TaskType), Multiple | **64.64** 🏆 | — | New Modality |
| **Image** | **MMEB v2 (Image)** | Mean(Task), Hit@1 | **57.28** 🏆 | — | New Modality |
| **VisDoc** | **MMEB v2 (Visual Documents)**| Mean(Task), NDCG@5 | **67.84** | — | New Modality |
| **Video** | **MMEB v2 (Video)** | Mean(Task), Hit@1 | **50.67** 🏆 | — | New Modality |
| **Audio** | **MSEB (Sound Retrieval)** | Mean(Task), MRR@10 | **69.54** 🏆 | — | New Modality |
| **Audio** | **MAEB (Speech & Audio)** | Mean(Task), Multiple | **49.39** 🏆 | — | New Modality |

---

## 2. Head-to-Head Comparison with Key Competitors

| Metric / Capability | **EmbeddingGemma 2** | **Granite-311M-Multilingual-R2** | **Voyage-4-Nano** | **Jina-v5-Omni-Nano** |
| :--- | :--- | :--- | :--- | :--- |
| **Developer** | Google DeepMind | IBM Research | Voyage AI | Jina AI |
| **Release Date** | October 2026 | 2026 | January 2026 | Mid 2026 |
| **Parameter Count** | **740M** (270M text base) | 311M | ~346M | ~0.95B–1.04B |
| **Supported Modalities** | **Text, Code, Image, Video, Audio** | Text, Code | Text only | Text, Image, Video, Audio |
| **Default Dimensions** | 768 (MRL: 512, 256, 128) | 768 (MRL to 128) | 2,048 (MRL to 256) | 768 / 1,024 |
| **Context Window** | 8,192 tokens | **32,768 tokens** 🏆 | **32,000 tokens** 🏆 | 8,192 tokens |
| **License** | **Apache 2.0** | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **MTEB (Code)** | **78.68** 🏆 | ~62–64 | ~68–71 | ~66 |
| **MTEB (Multilingual v2)**| 61.36 | **65.20** 🏆 | ~60–62 | ~62–64 |
| **MMEB v2 (Image)** | **57.28** 🏆 | N/A | N/A | ~47.87 |
| **MIEB (Lite)** | **64.64** 🏆 | N/A | N/A | ~44.36 |
| **ViDoRe (Visual Doc)** | 67.84 | N/A | N/A | **70.05–79.25** 🏆 |
| **Video Retrieval** | **50.67** 🏆 | N/A | N/A | 26.87 |

---

## 3. In-Depth Comparative Analysis

### 1. Does it beat Granite-Embedding-Multilingual-R2 on text?
* **Against Granite-311M-R2:** **No on pure multilingual retrieval.** IBM's Granite-Embedding-311M-Multilingual-R2 remains the category leader for open multilingual text search with **65.2** on MTEB Multilingual Retrieval vs. Gemma 2's **61.36**. Granite also provides a 32K context window based on ModernBERT.
* **Against Granite-97M-R2:** **Yes.** EmbeddingGemma 2 comfortably beats the distilled 97M variant (61.36 vs. 60.3).
* **Where Gemma 2 Wins:** **Code retrieval.** Gemma 2 scores **78.68 on MTEB Code**, far outstripping Granite (~62–64). Furthermore, Granite is strictly text-only.

### 2. Does it beat Voyage-4-Nano on text?
* **On Pure Document Retrieval (RTEB):** **Voyage-4-Nano retains an edge.** Voyage AI's model was narrowly trained for enterprise search and document retrieval, benefiting from a larger 2,048 default dimension and 32k context window (averaging ~0.64–0.66 on retrieval subsets).
* **Where Gemma 2 Wins:** **Efficiency and Code.** Gemma 2 achieves higher code search performance (78.68), requires 62.5% less vector storage by default (768d vs. 2048d), has a smaller text backbone (270M vs. 346M), and supports native non-text modalities.

### 3. Does it beat Jina-v5-Omni-Nano on images?
* **On Overall Image Retrieval (MMEB / MIEB):** **Yes, decisively.** EmbeddingGemma 2 achieves **57.28 on MMEB v2 (Image)** and **64.64 on MIEB (Lite)**, easily beating Jina-v5-Omni-Nano's scores (~44–48).
* **On Video Retrieval:** **Yes.** Gemma 2 scores **50.67** vs. Jina's **26.87** (+23.8 points).
* **Where Jina Holds an Edge:** **ViDoRe Visual Document Retrieval.** Jina's specialized visual document fine-tuning scores ~70–79 on ViDoRe subsets vs. Gemma 2's 67.84.
* **Model Size Context:** EmbeddingGemma 2 achieves these image wins at **less than half the parameter footprint** (440M text+vision vs. ~1.0B total parameters for Jina).

---

## 4. Architectural Decision Guide: Which Model to Pick?

| System Requirement | Recommended Model | Rationale |
| :--- | :--- | :--- |
| **Local / Edge Omni-Modal Search** | **EmbeddingGemma 2** 🏆 | Runs at ~191 MB RAM (INT4 text) or ~567 MB (INT4 multimodal) with Apache 2.0 license. |
| **Developer Tools & Codebase Search** | **EmbeddingGemma 2** 🏆 | Class-leading 78.68 MTEB Code score; indexes functions, symbols, and files with title-prefix steering. |
| **Cross-Modal E-Commerce & Media RAG** | **EmbeddingGemma 2** 🏆 | Matches text queries directly against images, video frames, and audio recordings in a unified 768d space. |
| **High-Volume Multilingual Text RAG** | **Granite-311M-R2** | Highest multilingual score (65.2) and 32k context for long international documents. |
| **Massive Enterprise Legal / Doc Search** | **Voyage-4-Nano** | High-capacity 2048d vector space with 32k context window tuned for enterprise RTEB. |
| **Specialized Scanned PDF / Document OCR** | **Jina-v5-Omni-Nano** *(or Gemma 2 at 1120 tokens)* | Jina retains high marks on ViDoRe document benchmarks, though Gemma 2 at 1,120 vision budget narrows the gap. |

---

## 5. Hardware Runtime Benchmarks: PyTorch MPS Golden vs. Native Rust (`glypr-core`)

To establish ground-truth runtime metrics, `glypr-core` was evaluated against the official PyTorch MPS Golden Oracle on Apple Silicon unified memory using direct Metal hardware queries (`[MTLDevice currentAllocatedSize]`) and POSIX RSS telemetry.

### Evaluation Corpus:
* **292 Golden Vectors:** Spans 10 languages (`en`, `de`, `fr`, `es`, `it`, `pt`, `ja`, `zh`, `ko`, `ru`), symmetric tasks, code retrieval, and context lengths from 64 tokens up to 8,192 tokens (`doc_physics_8192t`) across 4 Matryoshka dimensions (768d, 512d, 256d, 128d).
* **Golden Oracle Reference:** HuggingFace `google/embeddinggemma-2` running on PyTorch MPS (`torch.bfloat16`).

---

### Head-to-Head Hardware Performance (8,192 Tokens):

| Metric | PyTorch MPS (Golden) | `glypr-core` BF16 (Standard) | `glypr-core` BF16 (Chunked) | `glypr-core` BF16 (Metal Flash) | `glypr-core` Q8_0 (Metal Flash) | Delta (BF16 Flash vs PyTorch) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Execution Latency** | $875.00\text{ ms}$ | $3,086.73\text{ ms}$ | $1,267.90\text{ ms}$ | **$868.00\text{ ms}$** 🏆 | **$894.78\text{ ms}$** | **$-7.0\text{ ms}$ (Faster!)** |
| **Peak Throughput** | $9,360\text{ tok/s}$ | $2,654\text{ tok/s}$ | $6,461\text{ tok/s}$ | **$9,438\text{ tok/s}$** 🏆 | **$9,155\text{ tok/s}$** | **$+78\text{ tok/s}$** |
| **Peak Hardware Metal VRAM** | $6,592.8\text{ MB} (6.44\text{ GB})$ | $4,608.2\text{ MB}$ | **$3,094.6\text{ MB}$** | **$3,951.5\text{ MB}$** | **$3,727.4\text{ MB}$** | **$-40.1\%$ ($2.64\text{ GB}$ saved)** |
| **Transient VRAM Churn** | $5,120.3\text{ MB}$ | $3,827.2\text{ MB}$ | **$2,313.5\text{ MB}$** | $3,170.4\text{ MB}$ | $3,186.4\text{ MB}$ | **$-38.1\%$** |
| **Active Mask Tensor Overhead** | $134.2\text{ MB}$ | $134.2\text{ MB}$ | $134.2\text{ MB}$ | **$0.0\text{ MB}$ (Tile-Skipping)** | **$0.0\text{ MB}$ (Tile-Skipping)** | **$-100\%$ (Eliminated)** |
| **Model Disk Size** | $1,420.0\text{ MB}$ | $532.1\text{ MB}$ | $532.1\text{ MB}$ | $532.1\text{ MB}$ | **$295.5\text{ MB}$** | **$-79.2\%$** |
| **Avg Cosine Parity (292 vec)** | $1.000000$ (Ref) | **$0.999943$** | **$0.999943$** | **$0.999949$** | **$0.999864$** | $-0.000051$ |
| **Min Cosine Parity (292 vec)** | $1.000000$ (Ref) | **$0.999839$** | **$0.999839$** | **$0.999870$** | **$0.999602$** | $-0.000130$ |
| **Multilingual Margin** | $+0.2950$ | $+0.2953$ | $+0.2953$ | **$+0.2949$** | **$+0.2952$** | $-0.0001$ |
| **Length Scaling Margin** | $+0.1493$ | $+0.1497$ | $+0.1497$ | **$+0.1498$** | **$+0.1498$** | $+0.0005$ |

---

### Total Execution Time Scorecard (292 Golden Parity Vectors):

| Model & Attention Engine | Total Parity Suite Time | Evaluation Rate | Avg Parity | Min Parity | Speedup vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Glypr GGUF BF16 (Fused Metal Flash)** | **$5.09\text{ s}$** | **$57.37\text{ vec/s}$** | **$0.999949$** | **$0.999870$** | **$2.77\times$** |
| 🥈 **Glypr GGUF Q8_0 (Fused Metal Flash)** | **$5.29\text{ s}$** | **$55.20\text{ vec/s}$** | **$0.999864$** | **$0.999602$** | **$2.67\times$** |
| 🥉 **PyTorch MPS (`bfloat16` Golden Reference)** | **$5.73\text{ s}$** | **$50.96\text{ vec/s}$** | $1.000000$ | $1.000000$ | **$2.46\times$** |
| **Glypr GGUF BF16 (Chunked Sliding)** | **$7.37\text{ s}$** | **$39.62\text{ vec/s}$** | **$0.999943$** | **$0.999839$** | **$1.92\times$** |
| **Glypr GGUF Q8_0 (Chunked Sliding)** | **$7.58\text{ s}$** | **$38.52\text{ vec/s}$** | **$0.999860$** | **$0.999620$** | **$1.86\times$** |
| **Glypr GGUF BF16 (Standard Dense Attention)** | **$14.12\text{ s}$** | **$20.68\text{ vec/s}$** | **$0.999943$** | **$0.999839$** | **$1.00\times$ (ref)** |

---

### Token Latency and Throughput Scaling (Apple Silicon Metal GPU):

| Context Length | Standard Attention | Chunked Sliding | Metal Flash (BF16) | Metal Flash (Q8_0) | Speedup vs Standard | Peak Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64 tokens** | $14.28\text{ ms}$ | $14.29\text{ ms}$ | **$12.99\text{ ms}$** | $14.72\text{ ms}$ | **$1.10\times$** | $4,927\text{ tok/s}$ |
| **256 tokens** | $24.81\text{ ms}$ | $24.87\text{ ms}$ | **$21.98\text{ ms}$** | $23.93\text{ ms}$ | **$1.13\times$** | $11,648\text{ tok/s}$ |
| **512 tokens** | $47.15\text{ ms}$ | $49.87\text{ ms}$ | **$36.32\text{ ms}$** | $37.67\text{ ms}$ | **$1.30\times$** | $14,098\text{ tok/s}$ |
| **1,024 tokens** | $107.65\text{ ms}$ | $111.63\text{ ms}$ | **$69.79\text{ ms}$** | $72.75\text{ ms}$ | **$1.54\times$** | **$14,673\text{ tok/s}$** 🚀 |
| **2,048 tokens** | $291.72\text{ ms}$ | $235.65\text{ ms}$ | **$147.90\text{ ms}$** | $150.09\text{ ms}$ | **$1.97\times$** | $13,847\text{ tok/s}$ |
| **4,096 tokens** | $899.87\text{ ms}$ | $531.37\text{ ms}$ | **$338.57\text{ ms}$** | $349.26\text{ ms}$ | **$2.66\times$** | $12,098\text{ tok/s}$ |
| **8,192 tokens** | $3,086.73\text{ ms}$ | $1,267.90\text{ ms}$ | **$868.00\text{ ms}$ 🏆** | **$894.78\text{ ms}$** | **$3.56\times$** | **$9,438\text{ tok/s}$** |

---

### Universal Cross-Platform CPU SIMD Scaling (`pulp` + Rayon):

Evaluated on multi-core CPU using auto-dispatched SIMD (ARM NEON / x86 AVX2/AVX-512):

| Kernel Operation | Candle CPU Baseline | Pulp SIMD (Parallel Rayon) | Speedup | Intermediate Tensor Allocations |
| :--- | :---: | :---: | :---: | :---: |
| **RMSNorm** ($512 \times 2,304$) | $0.340\text{ ms}$ | **$0.219\text{ ms}$** | **$1.55\times$** | **0** (vs 5 in baseline) |
| **GeGLU** ($512 \times 9,216$) | $21.509\text{ ms}$ | **$2.050\text{ ms}$** | **$10.49\times$ 🚀** | **0** (vs 10 in baseline) |

| Context Length | Model Precision | CPU SIMD Latency | CPU SIMD Throughput | Peak Process RSS (RAM) |
| :---: | :---: | :---: | :---: | :---: |
| **64 tokens** | GGUF Q8_0 | **$135.0\text{ ms}$** | $474\text{ tok/s}$ | **$1,040.9\text{ MB}$** |
| **64 tokens** | GGUF BF16 | $159.1\text{ ms}$ | $402\text{ tok/s}$ | $1,738.7\text{ MB}$ |
| **256 tokens** | GGUF Q8_0 | **$319.7\text{ ms}$** | $801\text{ tok/s}$ | **$1,134.7\text{ MB}$** |
| **256 tokens** | GGUF BF16 | $317.4\text{ ms}$ | $807\text{ tok/s}$ | $1,738.6\text{ MB}$ |
| **512 tokens** | GGUF Q8_0 | $656.4\text{ ms}$ | $780\text{ tok/s}$ | **$1,134.6\text{ MB}$** |
| **512 tokens** | GGUF BF16 | **$607.7\text{ ms}$** | **$843\text{ tok/s}$** | $1,738.7\text{ MB}$ |
| **1,024 tokens** | GGUF Q8_0 | $1,509.2\text{ ms}$ | $679\text{ tok/s}$ | **$1,251.0\text{ MB}$** |
| **1,024 tokens** | GGUF BF16 | **$1,256.7\text{ ms}$** | **$815\text{ tok/s}$** | $1,738.7\text{ MB}$ |
| **2,048 tokens** | GGUF Q8_0 | $3,140.2\text{ ms}$ | $652\text{ tok/s}$ | **$1,406.2\text{ MB}$** |
| **2,048 tokens** | GGUF BF16 | **$2,815.3\text{ ms}$** | **$727\text{ tok/s}$** | $1,763.6\text{ MB}$ |
| **4,096 tokens** | GGUF Q8_0 | $7,086.9\text{ ms}$ | $578\text{ tok/s}$ | **$2,054.6\text{ MB}$** |
| **4,096 tokens** | GGUF BF16 | **$6,222.3\text{ ms}$** | **$658\text{ tok/s}$** | $2,273.1\text{ MB}$ |
