# Architectural Specifications

This document details the low-level model architecture, tensor dimensions, attention mechanics, and parameter distributions of **EmbeddingGemma 2** (`google/embeddinggemma-2`).

---

## 1. Parameter Distribution Breakdown

EmbeddingGemma 2 totals **744,371,512 parameters (~740M)** across three primary subsystems:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   EmbeddingGemma 2 Architecture (740M)                 │
├────────────────────────────────┬───────────────────────────────────────┤
│ Text & Code Backbone (270M)    │ • Embedder (Token Embeddings): 140M   │
│                                │ • Gemma 4 Decoder Transformer: 130M   │
├────────────────────────────────┼───────────────────────────────────────┤
│ Vision Encoder (170M)          │ • Spatial & Temporal Frame Projection │
├────────────────────────────────┼───────────────────────────────────────┤
│ Audio Encoder (300M)           │ • Raw Spectrogram / Waveform Processor│
└────────────────────────────────┴───────────────────────────────────────┘
```

| Subsystem Component | Parameter Count | Architectural Role |
| :--- | :--- | :--- |
| **Token Embeddings** | **140M** | Vocabulary projection for 262,144 tokens into 512 dimensions |
| **Transformer Backbone** | **130M** | 24-layer adapted Gemma 4 decoder with local/global attention |
| **Text-Only Base Total** | **270M** | Standalone footprint when vision and audio are disabled |
| **Vision Encoder** | **170M** | Patch-based encoder for single images, visual docs, video frames |
| **Audio Encoder** | **300M** | Direct audio encoder for speech and acoustic event streams |
| **Linear Projection Head**| **~0.4M** | Linear transformation mapping backbone $512d \to 768d$ |
| **Full Multimodal Total** | **~740M** | Complete omni-modal network |

---

## 2. Transformer Backbone Configuration

The core backbone is adapted from the **Gemma 4** architecture:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Layers ($N_{\text{layers}}$)** | `24` | Depth of the core transformer decoder |
| **Model Dimension ($d_{\text{model}}$)** | `512` | Working hidden size across all attention blocks |
| **Intermediate Hidden Dimension** | `2,048` | Feed-forward network (FFN) expansion ($4 \times d_{\text{model}}$) |
| **Query Attention Heads ($H_{Q}$)** | `4` | Multi-head attention query projections |
| **Local Key/Value Heads ($H_{KV,\text{local}}$)** | `2` | Grouped Query Attention (GQA), `head_dim: 256`, `rope_theta: 10,000` |
| **Global Key/Value Heads ($H_{KV,\text{global}}$)** | `1` | Multi-Query Attention (MQA), `head_dim: 512`, `rope_theta: 1,000,000` (Layers 5, 11, 17, 23) |
| **Local-to-Global Layer Ratio** | `5:1` | 5 local sliding window layers followed by 1 global layer |
| **Sliding Window Size** | `512` tokens | Window length for local attention blocks (`sliding_window: 512`) |
| **Vocabulary Size ($V$)** | `262,144` | Full multilingual Gemma 4 vocabulary |
| **FFN Activation Function** | `gelu_pytorch_tanh` | GeGLU gating in feed-forward networks |
| **Positional Embeddings** | Dual RoPE | $\theta = 10^4$ (sliding) / $\theta = 10^6$ (full), scaled for up to 262k positions |
| **Context Window ($L_{\text{max}}$)** | `8,192` tokens | Operational window for text, image, video, and audio |

---

---

## 3. Attention Mechanism & Efficiency: 5:1 Hybrid GQA/MQA

To achieve ultra-low latency and minimal memory allocation on edge hardware (e.g., Apple Silicon, Google Tensor, laptops), EmbeddingGemma 2 employs an asymmetric **5:1 Local:Global attention pattern**:

```
Layer 0-4  : Local Sliding Window Attention (W = 512) | GQA (4 Q, 2 KV heads, head_dim: 256, rope_theta: 1e4)
Layer 5    : Global Full Attention across context      | MQA (4 Q, 1 KV head,  head_dim: 512, rope_theta: 1e6)
Layer 6-10 : Local Sliding Window Attention (W = 512) | GQA (4 Q, 2 KV heads, head_dim: 256, rope_theta: 1e4)
Layer 11   : Global Full Attention across context      | MQA (4 Q, 1 KV head,  head_dim: 512, rope_theta: 1e6)
Layer 12-16: Local Sliding Window Attention (W = 512) | GQA (4 Q, 2 KV heads, head_dim: 256, rope_theta: 1e4)
Layer 17   : Global Full Attention across context      | MQA (4 Q, 1 KV head,  head_dim: 512, rope_theta: 1e6)
Layer 18-22: Local Sliding Window Attention (W = 512) | GQA (4 Q, 2 KV heads, head_dim: 256, rope_theta: 1e4)
Layer 23   : Global Full Attention across context      | MQA (4 Q, 1 KV head,  head_dim: 512, rope_theta: 1e6)
```

### Layer Type Specifications:

| Parameter | 20 Sliding Window Layers | 4 Global Attention Layers (5, 11, 17, 23) |
| :--- | :--- | :--- |
| **Attention Type** | Sliding Window ($W = 512$ tokens) | Full Dense Global Attention |
| **Query Heads ($H_Q$)** | `4` heads | `4` heads |
| **Key/Value Heads ($H_{KV}$)** | `2` heads (Grouped Query Attention, 2:1) | `1` head (Multi-Query Attention, 4:1) |
| **Head Dimension ($d_{\text{head}}$)** | `256` ($4 \times 256 = 1,024$ projected) | `512` ($4 \times 512 = 2,048$ projected) |
| **RoPE Base Frequency ($\theta$)** | `10,000.0` | `1,000,000.0` (scaled for long context) |
| **Attention Scope** | Token $i$ attends to $[i - 512, i + 512]$ | Token $i$ attends to all $T \in [0, N-1]$ tokens |

### Execution Strategies & Pluggable Attention Engines:

To satisfy diverse deployment environments (from edge CPU to Apple Silicon Metal GPUs), the architecture provides a 4-tier pluggable attention backend system (`AttentionBackend` trait):

1. **Standard Dense Attention (`AttentionMode::Standard`):**
   - Materializes full $[B, H, N, N]$ attention matrices across all layers.
   - At $N = 8,192$, a single full-attention layer in float32 softmax allocates over **$5.3\text{ GB}$** of transient buffers, reaching $14.59\text{ GB}$ peak VRAM across 24 layers. Useful primarily as an unoptimized baseline or for small sequence lengths ($N \le 512$).

2. **Chunked Sliding Attention (`AttentionMode::ChunkedSliding`):**
   - **Sliding Layers (20):** Queries are processed in blocks of size $W = 512$. Each query chunk attends strictly to keys and values in the previous, current, and next chunks ($3W = 1,536$ key span). With precomputed static masks (`mask_first`, `mask_middle`, `mask_last`), intermediate attention matrices stay bounded to a lean $\sim 1\text{--}3\text{ MB}$ footprint.
   - **Global Layers (4):** Queries are processed in blocks of $512$ tokens attending to all $N = 8,192$ keys. Because row-wise softmax along $\text{dim}=-1$ is mathematically independent per query token, this produces **$100\%$ bit-for-bit identical results** while slashing the intermediate matrix footprint from $8192 \times 8192$ ($1.07\text{ GB}$) to $512 \times 8192$ ($67\text{ MB}$).
   - **Hardware Footprint:** Reduces peak Metal VRAM from $14.59\text{ GB} \to 2.88\text{ GB}$ ($2.2\times$ smaller than PyTorch MPS). Operates universally on CPU and Metal GPU without external dependencies.

3. **Hardware-Fused Metal SDPA (`AttentionMode::MetalSdpa`):**
   - Directly executes Apple Silicon hardware-accelerated Steel Attention SDPA kernels via `candle_nn::ops::sdpa`.
   - **Global Layers:** Executes completely unmasked with zero mask memory allocation.
   - **Sliding Layers:** Leverages zero-copy broadcasted additive mask $[B, 1, N, N] \to [B, H, N, N]$ across attention heads.
   - **Fallback:** Seamlessly delegates to `ChunkedSlidingAttention` when invoked on CPU.

4. **Dedicated Sliding-Window & Full-Context Metal FlashAttention (`AttentionMode::MetalFlash`):**
   - Custom Metal Shading Language (`steel_attention_sliding.metal` & `fused_ops.metal`) kernels compiling directly to Apple Silicon GPU machine code.
   - **Default Execution on Metal:** As of Phase 1, `AttentionMode::default()` automatically selects `AttentionMode::MetalFlash` when running on Apple Silicon GPU with the `metal` / `metal-flash-attn` feature, while seamlessly falling back to `ChunkedSlidingAttention` on CPU or non-BF16 activations.
   - **Unified Coverage Across All 24 Layers (`BD=256` & `BD=512`):**
     - Sliding Window Layers (20 layers, $d_{\text{head}} = 256$): Dispatches `steel_attention_bfloat16_bq32_bk16_bd256_wm4_wn1_maskbfloat16` with $B_Q = 32, B_K = 16, W_M = 4, W_N = 1$.
     - Global Full-Attention Layers (Layers 5, 11, 17, 23, $d_{\text{head}} = 512$): Dispatches `steel_attention_bfloat16_bq16_bk16_bd512_wm2_wn1_maskbfloat16` ($B_Q = 16, B_K = 16, W_M = 2, W_N = 1$), eliminating `repeat_kv` memory duplication and $O(S^2)$ dense attention fallback on global layers.
   - **Zero-Copy GQA:** Bypasses KV head replication in memory by passing $gqa\_factor$ directly into thread registers ($G = 2$ on sliding layers, $G = 4$ on global layers).
   - **Native Hardware Tile-Skipping & $O(B \cdot S)$ Stride-Broadcasted Masks:**
     - For unpadded sequences: $0.0\text{ MB}$ mask tensor overhead up to 8,192 tokens.
     - For heterogeneous batches: Replaces $O(B \cdot S^2)$ CPU mask generation with GPU-constructed $O(B \cdot S)$ additive masks (`has_mask = true`) using 0-stride broadcasting (`M_strides = [S, 0, 0]`), preserving $1.000000$ batch equivalence across multimodal batches.
   - **Transient VRAM Elimination:** Drops 8,192-token transient VRAM from $3,298.4\text{ MB} \to 688.4\text{ MB}$ (**$-79.1\%$**) and peak Metal VRAM to **$1,473.5\text{ MB}$** (BF16) / **$1,582.7\text{ MB}$** (Q8_0).

5. **Apple Silicon M5 Neural Accelerators (`steel_attention_m5.metal`):**
   - Automatically engaged when running on Apple M5 hardware (`MTLGPUFamily::Metal4` / `MTLGPUFamily::Apple10` or forced via `GLYPR_METAL_TIER=m5`).
   - Dispatches cooperative `metal::mpp::tensor_ops::matmul2d` contractions with single-SIMDgroup scoped hardware `reduce_rows` (eliminating threadgroup barrier stalls) and dual 256-column tile slicing on global layers ($d_{\text{head}}=512$).
   - Reduces 8,192-token latency to **$615.30\text{ ms}$** (Q8_0) / **$670.79\text{ ms}$** (BF16) with peak throughput reaching **$23,359\text{ tok/s}$** at 2K tokens.

---

## 4. Per-Layer Embeddings (PLE) Architecture

EmbeddingGemma 2 employs **projection-only Per-Layer Embeddings (PLE)** to inject token-identity conditioning into every single decoder layer without inflating vocabulary parameters.

```
                  ┌────────────────────────────────────────┐
                  │    Token Embeddings (E) [B, N, 512]    │
                  └───────────────────┬────────────────────┘
                                      │
                 Linear Projection W_ple (512 -> 24 * 512)
                        Scale Factor = 512^(-0.5)
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Un-normalized PLE [B, N, 24, 512]   │
                  └───────────────────┬────────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            │ Slice per layer l on-demand: p_l [B, N, 512]      │
            │ Apply RMSNorm(p_l)                                │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                            TextPLEBlock (Layer l)                      │
  │                                                                        │
  │   Hidden States (h) ───► Linear Gate ──► GeLU ──┐                      │
  │                                                 ├──► Hadamard (⊙)      │
  │   Layer PLE Slice (p_l) ────────────────────────┘        │             │
  │                                                          ▼             │
  │                                                    Linear Proj         │
  │                                                          │             │
  │                                                          ▼             │
  │                                                       RMSNorm          │
  │                                                          │             │
  │   Hidden States (h) ───────────────────────────────► Add (+)           │
  │                                                          │             │
  │   Layer Scalar (s_l) ──────────────────────────────► Scale (*)         │
  └────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Formulation:
1. **Projection Phase:**
   $$\mathbf{P}_{\text{raw}} = \left( \mathbf{E} \mathbf{W}_{\text{ple}} \right) \cdot d_{\text{model}}^{-0.5}, \quad \mathbf{W}_{\text{ple}} \in \mathbb{R}^{512 \times (24 \cdot 512)}$$
   $$\mathbf{P} = \text{reshape}(\mathbf{P}_{\text{raw}}, [B, N, 24, 512])$$

2. **On-Demand Layer Normalization:**
   To avoid allocating gigabytes of intermediate float32 tensors, the slice for layer $l$ is normalized on demand:
   $$\mathbf{p}_l = \text{RMSNorm}\left( \mathbf{P}_{[:, :, l, :]} \right) \in \mathbb{R}^{B \times N \times 512}$$
   Because RMSNorm operates along the last dimension ($512$), normalizing per layer is **bit-for-bit identical** to normalizing the full 4D tensor ($0.0$ difference).

3. **TextPLEBlock Gating:**
   $$\mathbf{g} = \text{gelu\_tanh}\left( \mathbf{h} \mathbf{W}_{\text{gate}} \right)$$
   $$\mathbf{z} = \mathbf{g} \odot \mathbf{p}_l$$
   $$\mathbf{h}_{\text{ple}} = \text{RMSNorm}\left( \mathbf{z} \mathbf{W}_{\text{proj}} \right)$$
   $$\mathbf{h}_{\text{out}} = \left( \mathbf{h} + \mathbf{h}_{\text{ple}} \right) \cdot s_l$$
   where $s_l \in \mathbb{R}^1$ is a trained scalar output weight for layer $l$.

---

## 5. Pooling Strategy & 512 → 768 Projection

Unlike generative LLMs that extract embeddings from the final causal token ($<eos>$), EmbeddingGemma 2 employs **bidirectional sequence context** paired with **Mean Pooling**:

1. **Mean Pooling Across Valid Tokens:**
   $$\mathbf{h}_{\text{pooled}} = \frac{1}{\sum_{i=1}^T m_i} \sum_{i=1}^T m_i \cdot \mathbf{h}_i$$
   where $m_i \in \{0, 1\}$ is the attention mask, and $\mathbf{h}_i \in \mathbb{R}^{512}$ is the layer 24 output representation for token $i$.

2. **Projection Layer ($512 \to 768$):**
   The 512-dimensional pooled vector is passed through a trained linear projection layer to expand the feature space into the standard 768-dimensional embedding space:
   $$\mathbf{e}_{\text{raw}} = \mathbf{W}_{\text{proj}} \mathbf{h}_{\text{pooled}} + \mathbf{b}_{\text{proj}}, \quad \mathbf{W}_{\text{proj}} \in \mathbb{R}^{768 \times 512}$$

3. **L2 Normalization:**
   $$\mathbf{e}_{\text{final}} = \frac{\mathbf{e}_{\text{raw}}}{\|\mathbf{e}_{\text{raw}}\|_2}$$
   The resulting vector has unit length ($\|\mathbf{e}\|_2 = 1.0$), making inner product ($\mathbf{a} \cdot \mathbf{b}$) mathematically equivalent to cosine similarity.

---

## 6. Tokenizer & Vocabulary Specifications

* **Tokenizer Name:** Gemma 4 Tokenizer
* **Vocab Size:** 262,144 tokens
* **Subword Algorithm:** SentencePiece Byte-Fallback BPE
* **Multilingual Coverage:** 140+ languages natively tokenized with balanced compression rates.
* **Special Tokens:**
  * `<bos>`: Beginning of sequence (`token_id: 2`)
  * `<eos>`: End of sequence (`token_id: 1`)
  * `<pad>`: Padding token (`token_id: 0`)
  * `<unk>`: Unknown token (`token_id: 3`)
  * `boi_token_id`: Beginning of Image sequence (`token_id: 255999`)
  * `boa_token_id`: Beginning of Audio sequence (`token_id: 256000`)
  * `image_token_id`: Image soft-token placeholder (`token_id: 258880`)
  * `audio_token_id`: Audio soft-token placeholder (`token_id: 258881`)
  * `eoi_token_id`: End of Image sequence (`token_id: 258882`)
  * `eoa_token_index`: End of Audio sequence (`token_id: 258883`)
  * `video_token_id`: Video soft-token placeholder (`token_id: 258884`)

---

## 7. Vision Encoder (`gemma4_vision`) Specifications

The visual tower generates patch representations that project into the text decoder's latent space:

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Model Type** | `gemma4_vision` | Patch-based vision transformer |
| **Hidden Size ($d_{\text{vision}}$)** | `768` | Native hidden state dimension |
| **Intermediate Size** | `3,072` | Feed-forward dimension ($4 \times 768$) |
| **Hidden Layers** | `16` | Vision encoder depth |
| **Attention Heads** | `12` | Query and Key/Value heads |
| **Head Dimension** | `64` | Per-head dimension ($12 \times 64 = 768$) |
| **Patch Size** | `16` | $16 \times 16$ pixel spatial patches |
| **Pooling Kernel Size** | `3` | Spatial 2D pooling kernel |
| **Default Soft Tokens** | `280` | Soft tokens emitted per image |
| **Positional Embeddings** | Axial RoPE | `rope_theta: 100.0`, `rope_type: axial` |
| **Activation** | `gelu_pytorch_tanh` | GeGLU / tanh-approximated GELU |
| **Max Positions** | `131,072` | Max vision position embeddings |

---

## 8. Audio Encoder (`gemma4_audio`) Specifications

The audio tower ingests raw 16kHz spectrogram / waveform features:

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Model Type** | `gemma4_audio` | Subsampled convolutional + transformer audio tower |
| **Hidden Size ($d_{\text{audio}}$)** | `1,024` | Native audio hidden state |
| **Hidden Layers** | `12` | Transformer audio depth |
| **Attention Heads** | `8` | Query and Key/Value heads |
| **Output Projection Dim** | `1,536` | Output projection dimension before projection to decoder |
| **Subsampling Channels** | `[128, 32]` | 2-stage convolutional subsampling |
| **Conv Kernel Size** | `5` | Subsampling filter width |
| **Chunk Size / Context** | Chunk: 12, Left: 13, Right: 0 | Causal / left-context-only chunked attention |
| **Activation** | `silu` | Swish / SiLU activation |
| **Logit Capping** | `50.0` | Stabilizes attention softmax dynamic range |
| **Residual Weight** | `0.5` | Weighted residual skip connections |

---

## 9. Universal Cross-Platform CPU SIMD Engine (`pulp`)

For CPU execution on macOS and Linux x86_64/ARM64 servers, `glypr-core` replaces Candle's default heap-allocating CPU operations with custom vectorized kernels powered by `pulp`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Universal CPU SIMD Dispatch                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│     CustomOp1: simd_rmsnorm     │   │      CustomOp2: simd_geglu      │
├─────────────────────────────────┤   ├─────────────────────────────────┤
│ • #[pulp::with_simd] dynamic    │   │ • In-place streaming activation │
│ • Fused mul_add_f32s accumulator│   │ • Bounded tanh polynomial       │
│ • Rayon parallel chunking       │   │ • Zero intermediate allocations │
│ • Zero intermediate allocations │   │ • 10.49x isolated kernel speedup│
└─────────────────────────────────┘   └─────────────────────────────────┘
```

1. **`simd_rmsnorm` (CustomOp1):** Accumulates sum of squares across vector lanes (`mul_add_f32s`) and scales directly into the output buffer, eliminating 5 intermediate allocations per call.
2. **`simd_geglu` (CustomOp2):** Streams gate and up vectors concurrently, applying clamped tanh-approximated GELU directly into registers, eliminating 10 intermediate tensor allocations per call and delivering a **$10.49\times$ kernel speedup** ($21.5\text{ ms} \to 2.05\text{ ms}$).
3. **Parity Preservation:** Validated to within machine epsilon ($\le 7.15 \times 10^{-7}$ max absolute diff on RMSNorm, $0.0$ bit-for-bit match on GeGLU).

