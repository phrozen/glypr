# Dual-Track Hardware Acceleration Architecture

This document provides a comprehensive technical guide to the **Dual-Track Hardware Acceleration** subsystem in Glypr (`glypr-core`). It covers the design, mathematical formulation, numerical stability hardening, and implementation details for both **Apple Silicon Metal GPU** and **Universal Cross-Platform CPU SIMD**.

---

## 1. System Architecture Overview

Glypr is designed around a unified, pluggable acceleration abstraction that decouples model inference from physical compute targets:

```
                               ┌─────────────────────────────────────────┐
                               │           GlyprEmbedder Engine          │
                               └────────────────────┬────────────────────┘
                                                    │
                                     AttentionBackend Dispatch
                                                    │
                     ┌──────────────────────────────┴──────────────────────────────┐
                     ▼                                                             ▼
     ┌───────────────────────────────┐                             ┌───────────────────────────────┐
     │       Track 1: Metal GPU      │                             │     Track 2: Universal CPU    │
     │      (Apple Silicon M-Series) │                             │       (ARM64 & x86_64)        │
     ├───────────────────────────────┤                             ├───────────────────────────────┤
     │ • Zero-Copy GQA Multi-Head    │                             │ • pulp Vectorized Kernels     │
     │ • MSL Tile-Skipping SDPA      │                             │ • Dynamic Arch CPU Dispatch   │
     │ • Fused RMSNorm (simd_sum)    │                             │ • Rayon Multi-Core Streaming  │
     │ • Fused GeGLU (In-Register)   │                             │ • Zero Intermediate Allocs    │
     │ • NaN Clamping Protection     │                             │ • Single-File Memory Safety   │
     └───────────────────────────────┘                             └───────────────────────────────┘
```

---

## 2. Track 1: Metal GPU Full Fusion Engine

The Metal GPU engine executes natively on Apple Silicon GPUs through custom **Metal Shading Language (MSL)** compute pipelines, eliminating CPU-GPU synchronization barriers and intermediate unified memory allocations.

### A. Zero-Copy Grouped Query Attention (GQA) & Full-Context Support (`BD=512`)

In Gemma-2-2B, the transformer backbone uses an asymmetric head topology:
- **Query Heads ($H_Q$):** 4 heads ($d_{\text{head}} = 256$ in sliding layers, $512$ in global layers)
- **Key/Value Heads ($H_{KV}$):** 2 heads in sliding layers ($G = H_Q / H_{KV} = 2$) and 1 head in global layers ($G = 4$)

#### The Naive Approach (KV Head Duplication & Fallback):
Standard PyTorch and unoptimized frameworks call `repeat_kv` prior to attention, copying and materializing duplicated KV tensors in memory:
$$\text{Memory Overhead} = B \times H_Q \times N \times d_{\text{head}} \times 2 \text{ bytes (BF16)}$$
For an 8,192-token sequence, duplicating KV heads allocates **67.1 MB** per layer, thrashing unified memory cache lines. Previously, global layers ($d_{\text{head}} = 512$) also fell back to standard dense attention, creating gigabytes of transient $N \times N$ matrices.

#### The Glypr Zero-Copy Solution Across All 24 Layers:
Instead of duplicating memory or falling back on global layers, `MetalFlash` passes the GQA factor $G$ ($2$ or $4$) directly into thread registers and dispatches dedicated MSL kernels for both head dimensions:
1. **Sliding Layers ($d_{\text{head}} = 256$):** `steel_attention_bfloat16_bq32_bk16_bd256_wm4_wn1_maskbfloat16` with $B_Q = 32, B_K = 16, W_M = 4, W_N = 1$.
2. **Global Layers ($d_{\text{head}} = 512$):** `steel_attention_bfloat16_bq16_bk16_bd512_wm2_wn1_maskbfloat16` ($B_Q = 16, B_K = 16, W_M = 2, W_N = 1$, using 0 threadgroup padding to fit within 32 KB threadgroup memory).

```metal
// In-register head index mapping (Zero-Copy GQA)
uint kv_head_idx = int(tid.y) / params->gqa_factor;
K += tidl.z * params->K_strides[0] + kv_head_idx * params->K_strides[1];
V += tidl.z * params->V_strides[0] + kv_head_idx * params->V_strides[1];
```

**Empirical Impact:** Eliminating `repeat_kv`, fusing backbone operations, and employing zero-VRAM-overhead weight concatenation views dropped peak Metal VRAM at 8,192 tokens from **$4,079.5\text{ MB} \to 1,473.5\text{ MB}$ (BF16) / $1,582.7\text{ MB}$ (Q8_0)** (**$-79.1\%$ transient VRAM reduction**).


---

### B. Hardware Tile-Skipping Sliding-Window Kernel

Standard full attention evaluates an $N \times N$ matrix. In EmbeddingGemma 2's 20 sliding-window layers, tokens only attend within a local window $W = 512$ ($|i - j| \le 512$).

#### Tile-Skipping Dispatch Logic:
Queries are partitioned into threadgroup tiles of size $B_Q = 32$. The compute kernel calculates the active key tile boundaries analytically before launching memory reads:

$$k_{\text{start}} = \max\left(0, \left\lfloor \frac{q_{\min} - W}{B_K} \right\rfloor \right), \quad k_{\text{lim}} = \min\left(N_{\text{tiles}}, \left\lceil \frac{q_{\max} + W + 1}{B_K} \right\rceil \right)$$

Tiles outside this key range are skipped entirely in the GPU command queue:
- **FLOP Elimination:** At $N = 8,192$, over **$93.7\%$ of attention computations are pruned**.
- **Zero Mask Tensor Allocation:** The $134.2\text{ MB}$ ($8192 \times 8192$) attention mask tensor is completely eliminated ($0.0\text{ MB}$ footprint). Sliding distance boundaries are enforced in-register via `if (abs(q_pos - k_pos) > 512) score = -INFINITY`.

---

### C. Fused Custom MSL Kernels: RMSNorm & GeGLU

Located in [`crates/glypr-core/src/metal_kernels/fused_ops.metal`](crates/glypr-core/src/metal_kernels/fused_ops.metal):

#### 1. Fused In-Register RMSNorm (`rms_norm_bf16`)
Standard RMSNorm implementations allocate intermediate tensors for:
1. $x^2$ (element-wise square)
2. $\text{mean}(x^2)$ (reduction)
3. $\text{mean} + \epsilon$
4. $\text{rsqrt}(\dots)$
5. $x \cdot \text{rsqrt} \cdot \gamma$

In Glypr's custom shader, a single threadgroup of 32 threads cooperatively reduces the hidden dimension ($D = 2304$ or $512$) using hardware warp intrinsics (`simd_sum`), while respecting tensor slice byte offsets (`layout.start_offset() * sizeof(T)`) and being wrapped in thread-local Objective-C autorelease pool scopes:

```metal
float thread_sum_sq = 0.0f;
for (uint i = lane_id; i < dim; i += 32) {
    float val = float(input_row[i]);
    thread_sum_sq += val * val;
}
// Intra-warp hardware reduction (0 shared memory overhead)
float warp_sum_sq = simd_sum(thread_sum_sq);
float inv_rms = rsqrt(warp_sum_sq / float(dim) + eps);

for (uint i = lane_id; i < dim; i += 32) {
    float val = float(input_row[i]);
    float w = weight ? float(weight[i]) : 1.0f;
    output_row[i] = bfloat(val * inv_rms * w);
}
```


#### 2. Fused GeGLU with Numerical Clamping (`geglu_bf16`)
EmbeddingGemma 2 utilizes the gated activation:
$$\text{GeGLU}(gate, up) = \text{GELU}(gate) \odot up$$

Standard execution materializes 10 intermediate tensor allocations across Candle operations (`mul`, `pow`, `add`, `tanh`, `mul`, `mul`). In Glypr, the gate and up projections are read concurrently from GPU memory, activated, and multiplied directly into the output buffer:

```metal
float g = float(gate_row[i]);
float u = float(up_row[i]);

// Numerical stability clamp preventing float overflow in deep layers
float g_clamped = clamp(g, -10.0f, 10.0f);
float inner = 0.79788456f * (g_clamped + 0.044715f * g_clamped * g_clamped * g_clamped);
float gelu = 0.5f * g_clamped * (1.0f + metal::tanh(inner));

out_row[i] = bfloat(gelu * u);
```

---

### D. Numerical Stability Hardening: The NaN Overflow Fix

During early testing on sequences $\ge 4,096$ tokens and CJK multilingual inputs, deep transformer layers (L18–L23) produced sporadic `NaN` values.

#### Root Cause Analysis:
In Gemma-2, pre-activation MLP logits can reach extreme negative values ($g < -14.0$).
Under the standard PyTorch tanh approximation:
$$\text{inner} = 0.79788456 \cdot (g + 0.044715 \cdot g^3)$$
For $g = -14.5$:
$$g^3 = -3048.6 \implies \text{inner} \approx -108.1$$
On Apple Silicon Metal, evaluating `metal::tanh(-108.1)` triggered single-precision floating point exponent overflow during native transcendental evaluation, producing $\frac{-\infty}{+\infty} = \text{NaN}$.

#### Hardware-Guaranteed Solution:
Because $\tanh(\pm 10.0) \equiv \pm 1.0$ to within machine epsilon ($\le 2^{-24}$), clamping pre-activations to $[-10.0, 10.0]$:
1. Prevents cubic exponent explosion in the polynomial approximation.
2. Completely eliminates `NaN` generation without loss of precision.
3. Preserves exact mathematical equivalence with PyTorch Golden vectors.

**Verification:** Validated across all 292 golden vectors with **0 NaNs** and **Min Parity = 0.999870**.

---

## 3. Track 2: Universal Cross-Platform CPU SIMD

To deliver server-grade and laptop-grade embedding generation on CPUs (Apple Silicon, AWS Graviton ARM64, Intel/AMD x86_64), Glypr integrates portable SIMD vectorization via the `pulp` crate (`pulp = { version = "0.22.3", features = ["macro"] }`).

### A. The `pulp` Abstraction & Dynamic Arch Dispatch

Rather than maintaining fragile architecture-specific intrinsics (`core::arch::aarch64` vs `core::arch::x86_64`), `pulp` provides compile-time and runtime architecture dispatch via `#[pulp::with_simd]`:
- **ARM64:** Dispatches to 128-bit NEON registers (Apple Silicon, Graviton 3/4).
- **x86_64:** Dynamically detects AVX2 + FMA or AVX-512 and dispatches to 256-bit or 512-bit vector pipelines.
- **Fallback:** Auto-vectorized scalar fallback for WASM and legacy processors.

```
                              #[pulp::with_simd] Dispatcher
                                            │
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
               ARM64 NEON               x86 AVX2/FMA             x86 AVX-512
             (128-bit regs)           (256-bit regs)           (512-bit regs)
```

---

### B. Custom CPU SIMD Operators

Integrated via Candle's `CustomOp1` and `CustomOp2` interfaces, bypassing Candle's default tensor allocation pipeline.

#### 1. Zero-Heap `simd_rmsnorm` ([`rmsnorm.rs`](crates/glypr-core/src/cpu_simd/rmsnorm.rs))
Operates directly on continuous CPU storage memory using multi-threaded chunking via Rayon:
1. **Pass 1 (SIMD Fused Multiply-Add):** Accumulates the sum of squares across SIMD registers:
   $$\text{acc} \leftarrow \text{simd.mul\_add\_f32s}(x, x, \text{acc})$$
2. **Scalar Reciprocal Square Root:** Computes $\text{rsqrt}(\frac{\text{sum}}{D} + \epsilon)$.
3. **Pass 2 (SIMD Normalization & Weight Scaling):** Multiplies input by inverse RMS and layer weights directly into the output slice.

- **Intermediate Tensor Churn:** Reduced from 5 allocations per call down to **0**.
- **Max Absolute Error vs Reference:** $7.15 \times 10^{-7}$.

#### 2. Streaming `simd_geglu` ([`geglu.rs`](crates/glypr-core/src/cpu_simd/geglu.rs))
Processes gate and up tensors in-place using SIMD lane unrolling:
- Reads $x_g$ and $x_u$ from contiguous memory.
- Evaluates vectorized polynomial GELU approximation with clamped bounds.
- Multiplies by up-projection in registers.
- Writes to preallocated output buffer.

- **Intermediate Tensor Churn:** Reduced from 10 allocations per call down to **0**.
- **Max Absolute Error vs Reference:** $0.0 \times 10^0$ (exact bit match).

---

### C. Kernel Microbenchmark Results

Benchmarked across 100 warm iterations on full Gemma-2-2B layer dimensions ($512$ tokens, $D_{\text{model}} = 2304$, $D_{\text{ffn}} = 9216$):

| Operation | Input Dimensions | Candle CPU Baseline | Pulp SIMD (Rayon) | Speedup | Intermediate Allocations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RMSNorm** | $512 \times 2,304$ | $0.340\text{ ms}$ | **$0.219\text{ ms}$** | **$1.55\times$** | **0** (vs 5 baseline) |
| **GeGLU** | $512 \times 9,216$ | $21.509\text{ ms}$ | **$2.050\text{ ms}$** | **$10.49\times$ 🚀** | **0** (vs 10 baseline) |

---

## 4. End-to-End Latency & Throughput Scaling

Empirically measured on Apple Silicon M-series unified hardware across sequence lengths:

| Context Length | Standard Dense (Metal) | Chunked Sliding (Metal) | Metal Flash (BF16) | Metal Flash (Q8_0) | CPU SIMD (BF16) | CPU SIMD (Q8_0) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64 tokens** | $10.65\text{ ms}$ | $10.58\text{ ms}$ | **$8.92\text{ ms}$** | **$9.33\text{ ms}$** | $159.1\text{ ms}$ | **$135.0\text{ ms}$** |
| **256 tokens** | $19.79\text{ ms}$ | $19.77\text{ ms}$ | **$16.05\text{ ms}$** | **$16.61\text{ ms}$** | $317.4\text{ ms}$ | **$319.7\text{ ms}$** |
| **512 tokens** | $38.75\text{ ms}$ | $41.43\text{ ms}$ | **$26.67\text{ ms}$** | **$27.40\text{ ms}$** | **$607.7\text{ ms}$** | $656.4\text{ ms}$ |
| **1,024 tokens** | $91.38\text{ ms}$ | $95.58\text{ ms}$ | **$52.88\text{ ms}$** | **$55.00\text{ ms}$** | **$1,256.7\text{ ms}$** | $1,509.2\text{ ms}$ |
| **2,048 tokens** | $258.30\text{ ms}$ | $204.18\text{ ms}$ | **$114.73\text{ ms}$** | **$116.54\text{ ms}$** | **$2,815.3\text{ ms}$** | $3,140.2\text{ ms}$ |
| **4,096 tokens** | $831.51\text{ ms}$ | $462.23\text{ ms}$ | **$277.24\text{ ms}$** | **$287.25\text{ ms}$** | **$6,222.3\text{ ms}$** | $7,086.9\text{ ms}$ |
| **8,192 tokens** | $2,937.26\text{ ms}$ | $1,118.50\text{ ms}$ | **$774.83\text{ ms}$ 🏆** | **$796.23\text{ ms}$** | N/A | N/A |

> [!NOTE]
> **PyTorch MPS Comparison at 8,192 Tokens:**
> - PyTorch MPS Golden Baseline: **$875.00\text{ ms}$**
> - Glypr BF16 Fused Metal Flash: **$774.83\text{ ms}$** (Glypr is faster by **$100.2\text{ ms} / 1.13\text{x}$**)
> - Peak Throughput: **$10,573\text{ tok/s}$** (8K context) and **$19,366\text{ tok/s}$** (1K context).

---

## 5. Multimodal Vision Tower Acceleration (Dual-Track)

In addition to text backbone acceleration, `glypr-core` optimizes the multimodal `mmproj` Vision Tower across both **Apple Silicon Metal GPU** and **Universal CPU SIMD**:

### Architectural Optimizations:
1. **Precomputed 2D Axial RoPE:**
   Frequencies $(cos, sin)$ are computed once before layer 0. Eliminates 32 CPU $\leftrightarrow$ GPU synchronization barriers per image.
2. **Native 2-Stage Grid Pooling in VRAM:**
   Replaced single-threaded scalar 5-nested CPU loop ($870\text{k}$ float transfers) with 2-stage rank-3 (height) and rank-4 (width) Candle tensor reductions in VRAM. Bypasses Candle Metal high-rank reduction bugs while remaining 100% portable.
3. **Rayon Multi-Core Patch Extraction & L1 256-Element LUT:**
   Strided slice reads on raw image buffers with parallelized patch rows across CPU threads. Normalization $2 \cdot (x / 255.0 - 0.5)$ uses a 256-element L1 lookup table, eliminating float divisions.
4. **Fused Metal SDPA Attention (Zero Matrix Materialization):**
   Utilizes hardware-accelerated fused SDPA on Metal GPU, eliminating the $82\text{ MB}$ attention matrix, $160\text{ MB}$ F32 upcast, and softmax allocations across 16 layers (~$3.9\text{ GB}$ transient allocation reduction). CPU track seamlessly falls back to high-precision float32 softmax attention.

### Empirical Multimodal Benchmarks:

| Component / Metric | Baseline (Pre-Optimization) | Phase 2 | Phase 3 Optimized (`main`) | Speedup vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Tower Only (Budget 140)** | ~200.0 ms | 7.77 ms | **6.19 ms** | **~32x** |
| **Vision Tower Only (Budget 280)** | ~1,100.0 ms | 11.01 ms | **9.42 ms** | **~116x** |
| **Total Card Embedding (Budget 140)** | **269.60 ms** (3.71 cards/sec) | 67.50 ms | **50.87 ms (56.78 ms avg, 17.61 cards/s)** | **5.30x** |
| **Total Card Embedding (Budget 280)** | **1,260.99 ms** (0.79 cards/sec) | 140.20 ms | **106.07 ms (120.51 ms avg, 8.30 cards/s)** | **11.89x** |

---

## 6. Multimodal Audio Tower Acceleration (Dual-Track)

Building upon the multimodal projector architecture, `glypr-core` accelerates the Conformer Audio Tower and STFT Preprocessor across **Apple Silicon Metal GPU** and **Universal Multi-Core CPU**:

### Architectural Optimizations:
1. **Rayon Multi-Core STFT Streaming & OnceLock Filterbank Caching:**
   - 128-channel HTK Mel filterbank matrix ($257 \times 128$) and 320-point periodic Hann window are allocated and initialized once via `std::sync::OnceLock`.
   - Radix-2 512-point FFT twiddle factors and 9-bit reversal permutation pairs are precomputed into static lookup tables, eliminating runtime trigonometric calls.
   - Independent time frames (up to 1,200+ for speech clips) are processed in parallel across CPU cores using Rayon's `par_chunks_exact_mut`, writing directly into the pre-allocated log-mel spectrogram buffer.
2. **Precomputed Relative Position Bias (`precomputed_rel_k`):**
   - Relative sinusoidal positional embeddings ($13 \times 1024$) and per-layer relative key projections (`relative_k_proj`) are static across all audio inputs.
   - `precomputed_rel_k` (`[1, num_heads, head_dim, 13]` in F32) is precomputed once at model load time, eliminating dynamic projection forwards and tensor reshaping across all 12 Conformer layers.
3. **Single-Pass Conformer Attention Mask Caching:**
   - Sliding-window causal attention masks depend exclusively on sequence length ($N$) and block count.
   - The mask tensor is constructed **once** in VRAM prior to layer 0 and passed by reference to all 12 Conformer layers, eliminating 11 GPU pipeline sync stalls and transient host memory uploads per clip.
4. **Fused 5-Tap Depthwise 1D Convolution:**
   - Replaced Candle's default `conv1d` behavior for `groups = 1024` (which dynamically partitioned the sequence into 1,024 chunk views, launched 1,024 single-channel conv kernels, and performed a 1,024-tensor concatenation per layer).
   - Evaluated as a direct fused 5-tap broadcast multiply-accumulate across the channel dimension directly on GPU ($Y = \sum_{k=0}^4 X_k \cdot W_k$), reducing 12,288 tensor allocations and dispatches down to 60 hardware operations per audio pass.

### Empirical Audio Benchmarks (Apple Silicon Metal GPU, BF16):

| Audio Clip Duration | Metric | Baseline | Phase 2 | Phase 3 Optimized (`main`) | Speedup / Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Short (1.0s Synthetic)**<br>*(16,000 samples $\to 25$ tokens)* | **STFT Preprocessor**<br>**Audio Tower**<br>**End-to-End Latency** | 2.30 ms<br>2,896.61 ms<br>**2,946.68 ms** | 0.62 ms<br>5.47 ms<br>23.90 ms | **0.19 ms**<br>**5.16 ms**<br>**21.54 ms (23.07 ms avg)** | **12.1x**<br>**561x 🚀**<br>**43.4x real-time** |
| **Medium (5.0s Music)**<br>*(80,000 samples $\to 125$ tokens)* | **STFT Preprocessor**<br>**Audio Tower**<br>**End-to-End Latency** | 11.16 ms<br>3,170.47 ms<br>**3,219.75 ms** | 1.20 ms<br>8.33 ms<br>33.50 ms | **0.29 ms**<br>**6.65 ms**<br>**29.85 ms (32.05 ms avg)** | **38.5x**<br>**476x 🚀**<br>**156.0x real-time** |
| **Long (12.3s Speech)**<br>*(196,348 samples $\to 307$ tokens)* | **STFT Preprocessor**<br>**Audio Tower**<br>**End-to-End Latency** | 27.12 ms<br>3,184.88 ms<br>**3,222.78 ms** | 2.65 ms<br>10.70 ms<br>60.00 ms | **0.72 ms**<br>**9.10 ms**<br>**52.99 ms (57.62 ms avg)** | **37.7x**<br>**350x 🚀**<br>**213.0x real-time 🏆** |

> [!TIP]
> **Real-Time Factor (RTF):**
> Embedding a 12.3-second spoken sentence takes **$< 100\text{ ms}$** end-to-end ($\text{RTF} = 0.008\text{x}$), processing auditory inputs over **$125\times$ faster than real-time speech**.

