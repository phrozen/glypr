# Tri-Track Hardware Acceleration Architecture

This document provides a comprehensive technical guide to the **Hardware Acceleration Subsystems** in Glypr (`glypr-core`). It covers the design, mathematical formulation, numerical stability hardening, and implementation details for **Apple Silicon Metal GPU**, **Apple Neural Engine (ANE)**, and **Universal Cross-Platform CPU SIMD**.

---

## 1. System Architecture Overview

Glypr is designed around a unified, pluggable acceleration abstraction that decouples model inference from physical compute targets:

```
                               ┌─────────────────────────────────────────┐
                               │           GlyprEmbedder Engine          │
                               └────────────────────┬────────────────────┘
                                                    │
                                     Hardware Routing & Policy Dispatch
                                                    │
                 ┌──────────────────────────────────┼──────────────────────────────────┐
                 ▼                                  ▼                                  ▼
 ┌───────────────────────────────┐  ┌───────────────────────────────┐  ┌───────────────────────────────┐
 │       Track 1: Metal GPU      │  │ Track 2: Neural Engine (ANE)  │  │     Track 3: Universal CPU    │
 │    (Apple Silicon M-Series)   │  │    (CoreML Native Runtime)    │  │       (ARM64 & x86_64)        │
 ├───────────────────────────────┤  ├───────────────────────────────┤  ├───────────────────────────────┤
 │ • Zero-Copy GQA Multi-Head    │  │ • Native CoreML In-Memory     │  │ • pulp Vectorized Kernels     │
 │ • MSL Tile-Skipping SDPA      │  │ • W8A16 Linear Quantization   │  │ • Dynamic Arch CPU Dispatch   │
 │ • Fused RMSNorm (simd_sum)    │  │ • Zero-Copy BorrowedTensor    │  │ • Rayon Multi-Core Streaming  │
 │ • Fused GeGLU (In-Register)   │  │ • 11.17 ms Vision Tower (2.2x)│  │ • Zero Intermediate Allocs    │
 │ • NaN Clamping Protection     │  │ • 8.24 ms Text Backbone (2.6x)│  │ • Single-File Memory Safety   │
 └───────────────────────────────┘  └───────────────────────────────┘  └───────────────────────────────┘
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


#### 2. Fused GeGLU with 2D Grid Dispatch & Numerical Clamping (`geglu_gate_up_bf16`)
EmbeddingGemma 2 utilizes the gated activation:
$$\text{GeGLU}(gate, up) = \text{GELU}(gate) \odot up$$

Standard execution materializes 10 intermediate tensor allocations across Candle operations (`mul`, `pow`, `add`, `tanh`, `mul`, `mul`). In Glypr, the gate and up projections are read concurrently from GPU memory, activated, and multiplied directly into the output buffer:

```metal
float g = float(gate_up[base]);
float u = float(gate_up[base + size_t(D)]);

// Numerical stability clamp preventing float overflow in deep layers
float g_clamped = clamp(g, -10.0f, 10.0f);
float inner = 0.79788456f * (g_clamped + 0.044715f * g_clamped * g_clamped * g_clamped);
float gelu = 0.5f * g_clamped * (1.0f + metal::tanh(inner));

out[out_idx] = bfloat(gelu * u);
```

**2D Grid Dispatch (v0.2.1):** Rather than dispatching a flattened 1D thread grid that forces every thread to compute `row = id / D` and `col = id % D` (16–32 ALU cycles per thread), `geglu_gate_up_bf16` is dispatched as a 2D grid:
$$\text{grid} = \left(\frac{D + 255}{256}, \text{num\_rows}, 1\right), \quad \text{group} = (256, 1, 1)$$
Threads map directly to `col = id.x` and `row = id.y` in hardware registers, eliminating over 245 million integer division and modulo operations per 8K sequence.

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

### E. Apple Silicon M5 Neural Accelerators (Metal 4 / `mpp::tensor_ops`)

Apple's M5 architecture equips every GPU shader core with dedicated Neural Accelerators (NAX), programmed via Metal 4 (`MTLGPUFamilyMetal4` / `MTLGPUFamilyApple10`, MSL 4.0) and Metal Performance Primitives (`metal::mpp::tensor_ops`). Glypr introduces a specialized M5 acceleration subsystem:

#### 1. Zero-Cost Runtime Capability Detection (`MetalCapabilities`)
On startup, `glypr-core` queries the default `MTLDevice` for `MTLGPUFamily(1004)` and `MTLGPUFamily(1010)`, combined with a one-time runtime MSL compiler probe for `#include <metal_tensor>` and `metal::mpp::tensor_ops::matmul2d`.
- **Seamless Portability:** If run on M1–M4 hardware or older macOS versions, the system defaults to the Phase 1–3 `simdgroup_matrix` pipeline with zero runtime overhead.
- **Profiling Toggle:** Accepts `GLYPR_METAL_TIER=auto|m5|legacy` to directly benchmark M5 tensor ops against legacy paths.

#### 2. Native Hardware `bfloat` ALU in Fused MSL Kernels
By gating MSL definitions behind `#if defined(__HAVE_BFLOAT__) typedef bfloat MLX_BF16;`, all 11 fused kernels (norms, RoPE, GeGLU, Conformer convs) leverage M3/M4/M5 native hardware brain-float execution units, eliminating software float-conversion emulation.

#### 3. M5 Block-Quantized INT8 (`Q8_0`) GEMM (`q8_0_mm_m5_bf16`)
For batch tokens ($M > 1$), `crates/glypr-core/src/metal_kernels/m5_ops.metal` executes register-streamed block-quantized INT8 matrix multiplications directly into `bfloat` registers.

#### 4. Hardware Cooperative FlashAttention (`steel_attention_m5.metal`)
- **Single SIMD-Group Scoped Reductions:** Metal 4's `mpp::tensor_ops::reduce_rows` requires `execution_simdgroup`. Launching `WM=4` SIMD groups per threadgroup and partitioning query rows into `BQ_SG = 16` gives each SIMD group an isolated row slice, executing online softmax max/sum reductions directly in hardware without threadgroup memory barriers.
- **Left-Input Cooperative Tensors:** Chaining $S = Q \times K^T$ into $O += P \times V$ utilizes `pv_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>(sT_bf16)` directly in registers.
- **Dual-Tile Column Slicing for $d_{\text{head}}=512$:** To satisfy Metal 4's $N \le 256$ cooperative tensor limit, the kernel splits $P \times V$ into dual 256-column tiles (`oT0` and `oT1`), preserving four-9s cosine parity ($> 0.99986$) without clamping.
- **Hardware Tile Alignment & NaN Fast-Math Hardening (v0.2.1):** M5 NAX operations require static $16 \times 32$ cooperative tensor tiles. Inputs with unaligned sequence lengths ($s \pmod{64} \neq 0$) are cleanly routed to `steel_attention_sliding` while power-of-two lengths (64, 128, ..., 8192) execute on NAX, preventing out-of-bounds cooperative tensor loads. Fast-math evaluation in online softmax subtraction is protected with `isinf` guards against `-inf - (-inf) = NaN`.

**M5 Empirical Impact:** 8,192-token Q8_0 latency dropped from **$814.97\text{ ms} \to 627.20\text{ ms}$** (**$1.30\times$ speedup**), while full 292-vector suite time dropped to **$3.50\text{ s}$** (**$1.64\times$ faster than Python PyTorch MPS**).

---

### F. Device-Level MTLLibrary Caching & Startup Hygiene (v0.2.1)

In v0.2.1, `glypr-core` caches compiled `metal::Library` instances keyed by device pointer in a thread-safe `LIBRARY_CACHE`. Instead of compiling the 818-line fused kernel source 15 times on cache misses for each pipeline function, compilation occurs exactly once per device. Subsequent pipeline requests query `library.get_function(...)` directly, dropping iteration 0 cold-start latency 100x from **7,060 ms $\to$ 66.7 ms**. All device capability probing is additionally wrapped in `with_autorelease_pool` to guarantee clean release of Objective-C reflection metadata.

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
| **64 tokens** | $10.47\text{ ms}$ | $10.47\text{ ms}$ | **$10.30\text{ ms}$** | **$9.47\text{ ms}$** | $165.9\text{ ms}$ | **$152.9\text{ ms}$** |
| **256 tokens** | $19.42\text{ ms}$ | $19.41\text{ ms}$ | **$19.09\text{ ms}$** | **$15.88\text{ ms}$** | $316.1\text{ ms}$ | **$326.0\text{ ms}$** |
| **512 tokens** | $37.59\text{ ms}$ | $40.86\text{ ms}$ | **$28.77\text{ ms}$** | **$24.34\text{ ms}$** | **$591.2\text{ ms}$** | $642.1\text{ ms}$ |
| **1,024 tokens** | $90.59\text{ ms}$ | $94.74\text{ ms}$ | **$52.28\text{ ms}$** | **$43.93\text{ ms}$** | **$1,231.0\text{ ms}$** | $1,383.0\text{ ms}$ |
| **2,048 tokens** | $257.64\text{ ms}$ | $203.50\text{ ms}$ | **$107.19\text{ ms}$** | **$87.68\text{ ms}$** | **$2,604.4\text{ ms}$** | $2,977.9\text{ ms}$ |
| **4,096 tokens** | $830.42\text{ ms}$ | $461.29\text{ ms}$ | **$255.29\text{ ms}$** | **$225.10\text{ ms}$** | **$5,878.8\text{ ms}$** | $6,797.6\text{ ms}$ |
| **8,192 tokens** | $2,946.44\text{ ms}$ | $1,121.50\text{ ms}$ | **$670.79\text{ ms}$** | **$615.30\text{ ms}$ 🏆** | N/A | N/A |

> [!NOTE]
> **PyTorch MPS Comparison at 8,192 Tokens:**
> - PyTorch MPS Golden Baseline: **$875.00\text{ ms}$**
> - Glypr Q8_0 M5 Neural Flash: **$615.30\text{ ms}$** (Glypr is faster by **$259.7\text{ ms} / 1.42\text{x}$**)
> - Glypr BF16 M5 Neural Flash: **$670.79\text{ ms}$** (Glypr is faster by **$204.2\text{ ms} / 1.30\text{x}$**)
> - Peak Throughput: **$13,314\text{ tok/s}$** (8K context) and **$23,359\text{ tok/s}$** (2K context).

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

---

## 7. Track 3: Apple Neural Engine (ANE) Acceleration Subsystem

Introduced in **v0.3.0**, Glypr integrates native Apple Neural Engine execution for both the 170M SigLIP Vision Tower and the 270M resident Text Backbone via `coreml-native` (under optional `coreml` / `ane` Cargo features).

### A. Architectural Rationale: Why ANE?
While Apple Silicon Metal GPUs deliver massive throughput at long sequence lengths (e.g. 23,000+ tok/s at 2K tokens), GPU invocation incurs non-trivial dispatch latency and unified memory power overhead. For visual feature extraction (e.g., fixed-grid image patches) and resident low-latency text embeddings (e.g. 512-token RAG retrieval queries), the Apple Neural Engine provides:
1. **Ultra-Low Latency:** Dedicated fixed-function systolic array ALUs operating independently from the GPU.
2. **Zero Unified Memory Contention:** Keeps GPU execution pipelines completely free for concurrent LLM generation or large text context processing.
3. **Thermal & Energy Efficiency:** Enables persistent background embedding pipelines in local search services and mobile applications without degrading GPU thermal envelopes.

---

### B. Clean Module Exporter & W8A16 Quantization (`scripts/export_coreml.py`)
Standard HuggingFace classes fail Core ML conversion due to dynamic TorchScript graphs. Glypr resolves this with pure PyTorch clean modules:
- **Direct Safetensors Extraction:** Model parameters are extracted directly from `model.safetensors`.
- **Static Shape Guarantee:** Compiles models for deterministic hardware shapes ($1,134$ patches for Budget 140 portrait, $2,394$ patches for Budget 280 portrait, $512$ tokens for Text Backbone).
- **Hardware Pooling & Trigonometry:** Replaces runtime loops with native Core ML `AvgPool2d` layers and precomputed 2D/1D RoPE buffers.
- **W8A16 Linear Symmetric Quantization:** Weights are quantized to INT8 with per-tensor scales while compute activations remain in native Apple Silicon FP16. This strictly avoids BF16 (which is unsupported by ANE ALUs).

---

### C. Native Rust Core ML Runtime (`crates/glypr-core/src/coreml/`)
1. **Zero C++ Interop / Pure Objective-C Runtime:**
   Implemented via `coreml-native = "0.2.0"`, binding directly into Apple's Objective-C runtime via `objc2 v0.3.2`. This eliminates third-party C++ libraries, static toolchain dependencies, and linking hazards.
2. **Zero-Copy MultiArray Interop:**
   Input tensors are wrapped into `BorrowedTensor` directly from raw pixel slices (`image.pixel_values`), avoiding intermediate heap allocations.
3. **Automatic DType Alignment for Splicing:**
   Core ML outputs FP32 float buffers. `CoreMLVisionTower::encode_to_tensor_dtype` automatically casts the tensor to the active model `DType` (e.g. BF16 on Metal) before concatenation into the multimodal token stream.

---

### D. Transparent Vision Routing Policy (`VisionBackendPolicy`)
The embedder manages vision execution through an explicit gating policy:
- **`Auto` (default):** Automatically routes matching portrait image patch counts (e.g., $1,134$ and $2,394$) to the Apple Neural Engine. If an image produces an unsupported patch count (such as custom budgets or landscape aspect ratios), it seamlessly falls back to the GPU Metal `mmproj` projector.
- **`AneStrict`:** Enforces execution strictly on ANE, failing fast with descriptive diagnostics if an image dimension cannot run on the Neural Engine.
- **`Metal`:** Enforces all image encoding to execute on the GPU Metal `mmproj` projector.
- **Zero-`mmproj` Standalone Image Embedding:** When `--ane` is enabled, images matching compiled dimensions can be embedded directly without downloading or attaching any GGUF `mmproj` file.

---

### E. Empirical Benchmarks: Apple Neural Engine vs Metal GPU

Tested in native release mode on Apple Silicon:

#### 1. Vision Tower Execution (Single Image)
| Modality / Budget | Input Patches | Soft Tokens | Apple Neural Engine (ANE) | Apple Silicon Metal GPU (ViT) | ANE Speedup | Parity vs GPU mmproj |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Budget 140 Portrait** | 1,134 patches | 126 tokens | **11.17 ms (89.5 img/s)** | 24.03 ms (41.6 img/s) | **2.2× faster** | **`0.999928` Cosine Sim** |
| **Budget 280 Portrait** | 2,394 patches | 266 tokens | **23.19 ms (43.1 img/s)** | 55.30 ms (18.1 img/s) | **2.4× faster** | **`0.999928` Cosine Sim** |

#### 2. Resident Text Backbone (Static 512 Tokens)
| Sequence Length | Hardware Engine | Compute Units | Latency | Throughput | Relative Speedup |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **16 tokens** | Apple Silicon Metal GPU | GPU | $4.21\text{ ms}$ | $237.5\text{ q/s}$ | 1.96× |
| **16 tokens** | Apple Neural Engine (ANE) | ANE (W8A16) | $8.24\text{ ms}$ | $121.3\text{ q/s}$ | 1.00× (ref) |
| **128 tokens** | Apple Silicon Metal GPU | GPU | $7.85\text{ ms}$ | $127.4\text{ q/s}$ | 1.05× |
| **128 tokens** | Apple Neural Engine (ANE) | ANE (W8A16) | **$8.24\text{ ms}$** | **$121.3\text{ q/s}$** | 1.00× |
| **512 tokens (RAG chunk)** | Apple Silicon Metal GPU | GPU | $21.81\text{ ms}$ | $45.8\text{ q/s}$ | 1.00× (ref) |
| **512 tokens (RAG chunk)** | **Apple Neural Engine (ANE)** | **ANE (W8A16)** | **$8.24\text{ ms}$** 🚀 | **$121.3\text{ q/s}$** | **2.6× faster** |

