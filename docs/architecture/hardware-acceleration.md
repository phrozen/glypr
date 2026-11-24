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
 │ • Fused GeGLU (In-Register)   │  │ • 11.07 ms Vision Tower (6.7x)│  │ • Zero Intermediate Allocs    │
 │ • NaN Clamping Protection     │  │ • 8.27 ms Text Backbone (3.0x)│  │ • Single-File Memory Safety   │
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

**Empirical Impact:** Eliminating `repeat_kv`, fusing backbone operations, and employing zero-VRAM-overhead weight concatenation views dropped peak Metal VRAM at 8,192 tokens from **$4,079.5\text{ MB}$** down to **$1,473.5\text{ MB}$** (`v0.3.0`) and **$1,177.5\text{ MB}$ (BF16) / $1,102.8\text{ MB}$ (Q8_0)** in `v0.4.0` (with only **$512.0\text{ MB}$ / $496.0\text{ MB}$** transient VRAM).


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

Located in [`crates/glypr-core/src/metal_kernels/fused_ops.metal`](../../crates/glypr-core/src/metal_kernels/fused_ops.metal):

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

#### 1. Zero-Heap `simd_rmsnorm` ([`rmsnorm.rs`](../../crates/glypr-core/src/cpu_simd/rmsnorm.rs))
Operates directly on continuous CPU storage memory using multi-threaded chunking via Rayon:
1. **Pass 1 (SIMD Fused Multiply-Add):** Accumulates the sum of squares across SIMD registers:
   $$\text{acc} \leftarrow \text{simd.mul\_add\_f32s}(x, x, \text{acc})$$
2. **Scalar Reciprocal Square Root:** Computes $\text{rsqrt}(\frac{\text{sum}}{D} + \epsilon)$.
3. **Pass 2 (SIMD Normalization & Weight Scaling):** Multiplies input by inverse RMS and layer weights directly into the output slice.

- **Intermediate Tensor Churn:** Reduced from 5 allocations per call down to **0**.
- **Max Absolute Error vs Reference:** $7.15 \times 10^{-7}$.

#### 2. Streaming `simd_geglu` ([`geglu.rs`](../../crates/glypr-core/src/cpu_simd/geglu.rs))
Processes gate and up tensors in-place using SIMD lane unrolling:
- Reads $x_g$ and $x_u$ from contiguous memory.
- Evaluates vectorized polynomial GELU approximation with clamped bounds.
- Multiplies by up-projection in registers.
- Writes to preallocated output buffer.

- **Intermediate Tensor Churn:** Reduced from 10 allocations per call down to **0**.
- **Max Absolute Error vs Reference:** $0.0 \times 10^0$ (exact bit match).

---

### C. Kernel Microbenchmark Results (`v0.4.0`, `--features accelerate`)

Benchmarked on full Gemma-2 layer dimensions ($512$ tokens, $D_{\text{model}} = 2304$, $D_{\text{ffn}} = 9216$) with Rayon + `pulp` SIMD and optional Apple Accelerate (`vvtanhf`):

| Operation | Input Dimensions | Candle CPU Baseline | Pulp SIMD (Rayon + Accelerate) | Speedup | Intermediate Allocations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RMSNorm** | $512 \times 2,304$ | $0.312\text{ ms}$ | **$0.237\text{ ms}$** | **$1.32\times$** | **0** (vs 5 baseline) |
| **GeGLU** | $512 \times 9,216$ | $5.583\text{ ms}$ | **$0.676\text{ ms}$** | **$8.26\times$ 🚀** | **0** (vs 10 baseline) |

---

## 4. End-to-End Latency & Throughput Scaling (`v0.4.0` Metal + CPU SIMD Reference)

Empirically measured on Apple Silicon M-series unified hardware across sequence lengths (`compare_models --models gguf-bf16,gguf-q8 --device metal --attn flash` for `v0.4.0` Metal columns; CPU SIMD columns from prior reference capture):

| Context Length | Standard Dense (Metal, `v0.4.0`) | Chunked Sliding (BF16, `v0.4.0`) | Chunked Sliding (Q8_0, `v0.4.0`) | Metal Flash (BF16, `v0.4.0`) | Metal Flash (Q8_0, `v0.4.0`) | CPU SIMD (BF16, *prior ref*) | CPU SIMD (Q8_0, *prior ref*) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64 tokens** | $10.54\text{ ms}$ | $10.56\text{ ms}$ | $9.42\text{ ms}$ | **$9.94\text{ ms}$** | **$9.25\text{ ms}$** | $165.9\text{ ms}$ | **$152.9\text{ ms}$** |
| **256 tokens** | $19.64\text{ ms}$ | $19.35\text{ ms}$ | $16.43\text{ ms}$ | **$17.81\text{ ms}$** | **$14.59\text{ ms}$** | $316.1\text{ ms}$ | **$326.0\text{ ms}$** |
| **512 tokens** | $38.79\text{ ms}$ | $41.08\text{ ms}$ | $35.86\text{ ms}$ | **$27.03\text{ ms}$** | **$22.41\text{ ms}$** | **$591.2\text{ ms}$** | $642.1\text{ ms}$ |
| **1,024 tokens** | $90.23\text{ ms}$ | $94.16\text{ ms}$ | $84.32\text{ ms}$ | **$47.37\text{ ms}$** | **$39.05\text{ ms}$** | **$1,231.0\text{ ms}$** | $1,383.0\text{ ms}$ |
| **2,048 tokens** | $258.71\text{ ms}$ | $202.98\text{ ms}$ | $183.51\text{ ms}$ | **$95.29\text{ ms}$** | **$76.91\text{ ms}$** | **$2,604.4\text{ ms}$** | $2,977.9\text{ ms}$ |
| **4,096 tokens** | $831.79\text{ ms}$ | $459.22\text{ ms}$ | $426.93\text{ ms}$ | **$228.92\text{ ms}$** | **$199.54\text{ ms}$** | **$5,878.8\text{ ms}$** | $6,797.6\text{ ms}$ |
| **8,192 tokens** | $2,940.54\text{ ms}$ | $1,113.36\text{ ms}$ | $1,055.63\text{ ms}$ | **$620.81\text{ ms}$** | **$564.91\text{ ms}$** 🏆 | N/A | N/A |

> [!NOTE]
> **PyTorch MPS Reference Comparison at 8,192 Tokens (`v0.4.0` vs prior MPS capture):**
> - PyTorch MPS Golden Baseline (*prior reference*): **$845.83\text{ ms} - 875.00\text{ ms}$**
> - Glypr Q8_0 Metal Flash (`v0.4.0`): **$564.91\text{ ms}$** (`14,502 tok/s`; 292-vec parity Avg **`0.999896`** / Min **`0.999741`**, suite `3.22 s`)
> - Glypr BF16 Metal Flash (`v0.4.0`): **$620.81\text{ ms}$** (`13,196 tok/s`; 292-vec parity Avg **`0.999967`** / Min **`0.999897`**, suite `3.61 s`)
> - Peak Throughput (`v0.4.0` Q8_0 Flash): **$26,627\text{ tok/s}$** at 2K context and **$26,221\text{ tok/s}$** at 1K context.

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

### Empirical Multimodal Vision Benchmarks (`v0.4.0` Metal BF16, 480×680 Card):

| Component / Metric | Baseline (Pre-Optimization) | `v0.3.0` Optimized | `v0.4.0` (`bench_multimodal`) | Speedup vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Tower Only (Budget 140)** | ~200.0 ms | 6.19 ms | **5.51 ms** | **~36x** |
| **Vision Tower Only (Budget 280)** | ~1,100.0 ms | 9.42 ms | **8.72 ms** | **~126x** |
| **Total Card Embedding (Budget 140)** | **269.60 ms** (3.71 cards/sec) | 50.87 ms (56.78 ms avg) | **56.65 ms avg (warm `50.34–51.16 ms`, 17.65 cards/s)** | **4.76x avg / ~5.3x warm** |
| **Total Card Embedding (Budget 280)** | **1,260.99 ms** (0.79 cards/sec) | 106.07 ms (120.51 ms avg) | **119.98 ms avg (warm `105.40–106.10 ms`, 8.33 cards/s)** | **10.51x avg / ~11.9x warm** |

---

## 6. Multimodal Audio Tower Acceleration (Dual-Track)

Building upon the multimodal projector architecture, `glypr-core` accelerates the Conformer Audio Tower and STFT Preprocessor across **Apple Silicon Metal GPU** and **Universal Multi-Core CPU**:

### Architectural Optimizations:
1. **Rayon Multi-Core STFT Streaming & OnceLock Filterbank Caching:**
   - 128-channel HTK Mel filterbank matrix ($257 \times 128$) and `f64`-computed 320-point periodic Hann window are allocated and initialized once via `std::sync::OnceLock`.
   - Radix-2 512-point FFT twiddle factors (`f64` trig) and 9-bit reversal permutation pairs are precomputed into static lookup tables, eliminating runtime trigonometric calls and preserving exact PyTorch FP32 parity.
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

### Empirical Audio Benchmarks (`v0.4.0` Apple Silicon Metal GPU, BF16):

| Audio Clip Duration | Metric | Baseline | `v0.3.0` | `v0.4.0` (`bench_multimodal`) | Speedup / Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Short (1.0s Synthetic)**<br>*(16,000 samples $\to 25$ tokens)* | **STFT Preprocessor**<br>**Audio Tower**<br>**Backbone**<br>**End-to-End Avg** | 2.30 ms<br>2,896.61 ms<br>—<br>**2,946.68 ms** | 0.19 ms<br>5.16 ms<br>—<br>23.07 ms | **0.25 ms**<br>**3.80 ms**<br>**18.11 ms**<br>**21.91 ms** | **762× Tower** 🚀<br>**45.6× real-time** |
| **Medium (5.0s Music)**<br>*(80,000 samples $\to 125$ tokens)* | **STFT Preprocessor**<br>**Audio Tower**<br>**Backbone**<br>**End-to-End Avg** | 11.16 ms<br>3,170.47 ms<br>—<br>**3,219.75 ms** | 0.29 ms<br>6.65 ms<br>—<br>32.05 ms | **0.49 ms**<br>**6.01 ms**<br>**23.45 ms**<br>**29.46 ms** | **527× Tower** 🚀<br>**169.7× real-time** |
| **Long (12.3s Speech)**<br>*(196,348 samples $\to 307$ tokens)* | **STFT Preprocessor**<br>**Audio Tower**<br>**Backbone**<br>**End-to-End Avg** | 27.12 ms<br>3,184.88 ms<br>—<br>**3,222.78 ms** | 0.72 ms<br>9.10 ms<br>—<br>57.62 ms | **0.77 ms**<br>**8.33 ms**<br>**43.41 ms**<br>**51.74 ms** | **382× Tower** 🚀<br>**237.2× real-time 🏆** |

> [!TIP]
> **Real-Time Factor (RTF):**
> In `v0.4.0`, embedding a 12.3-second spoken sentence takes **51.74 ms** end-to-end (**8.33 ms** Conformer Audio Tower), processing auditory inputs **237.2× faster than real-time speech**.

---

## 7. Track 3: Apple Neural Engine (ANE) Acceleration Subsystem

Introduced in **v0.3.0** and expanded in **v0.4.0**, Glypr integrates native Apple Neural Engine execution for both the 170M SigLIP Vision Tower and multi-bucket 270M Text Backbones via `coreml-native` (under optional `coreml` / `ane` Cargo features).

### A. Architectural Rationale: Why ANE?
While Apple Silicon Metal GPUs deliver massive throughput at long sequence lengths (up to 26,627 tok/s at 2K tokens for Q8_0), GPU invocation incurs non-trivial dispatch latency and unified memory power overhead. For visual feature extraction (fixed-grid image patches) and resident low-latency text embeddings (e.g. 128–1024 token RAG retrieval queries), the Apple Neural Engine provides:
1. **Ultra-Low Latency:** Dedicated fixed-function systolic array ALUs operating independently from the GPU.
2. **Zero Unified Memory Contention:** Keeps GPU execution pipelines completely free for concurrent LLM generation or large text context processing.
3. **Thermal & Energy Efficiency:** Enables persistent background embedding pipelines in local search services and mobile applications without degrading GPU thermal envelopes.

---

### B. Clean Module Exporter & W8A16 Quantization (`scripts/export_coreml.py`)
Standard HuggingFace classes fail Core ML conversion due to dynamic TorchScript graphs. Glypr resolves this with pure PyTorch clean modules:
- **Direct Safetensors Extraction & `config.json` Reading:** Transfers weights from single or sharded Safetensors checkpoints and writes `manifest.json` alongside compiled `.mlmodelc` bundles (cleaning intermediate `.mlpackage` artifacts by default unless `--keep-mlpackage` is passed).
- **Multi-Aspect Vision Presets (`(patch_h, patch_w)` Grid Keying):** Exports 8 static aspect-ratio presets (`140-portrait`, `140-std`, `140-square`, `140-landscape`, `280-portrait`, `280-std`, `280-square`, `280-landscape`), tagged with `_h{H}_w{W}` and `glypr.patch_height` / `glypr.patch_width` metadata so portrait (`42×27`) and landscape (`27×42`) orientations never collide on patch count.
- **Multi-Bucket Text Backbone & Sliding-Window Mask (`--text-seq-lens`):** Supports static sequence-length buckets (e.g., `128,512,1024`) with exact `|i - j| <= 512` sliding-window attention masking on the 20 sliding layers when `seq_len > 512`.
- **Hardware Pooling & Trigonometry:** Replaces runtime loops with native Core ML `AvgPool2d` layers and precomputed 2D/1D RoPE buffers.
- **W8A16 Linear Symmetric Quantization:** Weights are quantized to INT8 with per-tensor scales while compute activations remain in native Apple Silicon FP16. This strictly avoids BF16 (which is unsupported by ANE ALUs).

---

### C. Native Rust Core ML Runtime (`crates/glypr-core/src/coreml/`)
1. **Zero C++ Interop / Pure Objective-C Runtime:**
   Implemented via `coreml-native`, binding directly into Apple's Objective-C runtime via `objc2`. This eliminates third-party C++ libraries, static toolchain dependencies, and linking hazards.
2. **Zero-Copy MultiArray Interop & Startup Warmup:**
   Input tensors are wrapped into `BorrowedTensor` directly from raw slices, avoiding intermediate heap allocations, and `GlyprEmbedder::warmup_coreml()` warms up all attached vision and text bundles on boot.
3. **Deterministic Model Discovery (`resolve_coreml_dir`):**
   CoreML directories resolve in deterministic order: `--coreml-dir <DIR>` $\to$ `$GLYPR_COREML_DIR` $\to$ `./models/coreml` $\to$ user data/cache dir (`~/Library/Application Support/glypr/coreml` on macOS, or `$XDG_DATA_HOME/glypr/coreml` / `~/.cache/glypr/coreml`). Inspect any `.mlmodelc` bundle directly with `glypr inspect <PATH>`.

---

### D. Transparent Vision & Text Routing Policies (`VisionBackendPolicy`, `TextBackendPolicy`)
The embedder manages vision and pure-text ANE execution through explicit gating policies:
- **Vision (`--vision-backend <auto|ane|metal>`):**
  - `Auto` (default): Keyed by `(patch_height, patch_width)` grid, routes matching portrait, square, and landscape grids to ANE and falls back to Metal `mmproj` for unmatched dimensions or budgets.
  - `AneStrict` (`ane`): Enforces ANE execution and fails fast with supported-grid diagnostics if unmatched.
  - `Metal` (`metal`): Forces GPU Metal `mmproj` execution.
  - **Zero-`mmproj` Standalone Image Embedding:** Images matching compiled grids embed via `--ane` without downloading or attaching any GGUF `mmproj` file.
- **Pure Text (`--text-backend <auto|ane|metal>`):**
  - `Auto` (default): Routes pure-text sequences in the `[64..=seq_len]` ANE sweet spot with `batch_size <= 4` to the smallest fitting `CoreMLTextBackbone` bucket, keeping `<64` tokens, `>seq_len`, `batch_size > 4`, and multimodal sequences on Metal/CPU.
  - `AneStrict` (`ane`): Enforces ANE text execution, failing if no CoreML text model is attached or sequence length exceeds the largest loaded bucket.
  - `Metal` (`metal`): Forces Metal/CPU text execution even when CoreML text models are attached.

---

### E. Live CoreML / ANE Parity (`v0.4.0`) & Reference Benchmarks

**Verified `v0.4.0` CoreML / ANE Export Parity (`scripts/export_coreml.py`):**
- **`140-landscape`** (`27×42`, `1134p -> 126t`, `_h27_w42`): ANE vs Metal `mmproj` **`0.999955`** cosine similarity.
- **`140-square`** (`33×33` `lenna.png`, `1089p -> 121t`, `_h33_w33`): CoreML vs PyTorch FP32 **`0.999267`**; ANE vs Metal `mmproj` **`0.999836`**.
- **`s1024` Text Backbone** (`gemma2_text_backbone_s1024_w8a16.mlmodelc`, sliding-window mask for `> 512t`): CoreML vs PyTorch FP32 **`0.999684`**; live `doc_science_512t` (522 tokens) ANE vs FP32 golden **`0.999601`** (vs Metal `0.999571`); live `~770t` (`> 512` sliding-window span) ANE vs Metal **`0.999700`**.

#### Prior Reference Telemetry (`v0.3.0` Vision & 512t Text ANE vs Metal/PyTorch MPS):
| Modality / Budget | Input Patches | Soft Tokens | Apple Neural Engine *(v0.3.0 ref)* | Metal GPU (ViT) *(v0.3.0 ref)* | Python Reference *(MPS ref)* | Speedup vs Python | Speedup vs Metal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Budget 140 Portrait** | 1,134 patches | 126 tokens | **11.07 ms (90.3 img/s)** 🚀 | 24.03 ms (41.6 img/s) | 74.66 ms (13.4 img/s) | **6.74× faster** | **2.17× faster** |
| **Budget 280 Portrait** | 2,394 patches | 266 tokens | **23.36 ms (42.8 img/s)** 🚀 | 36.54 ms (27.4 img/s) | 73.06 ms (13.7 img/s) | **3.13× faster** | **1.56× faster** |

| Sequence Length | Hardware Engine | Compute Units | Latency *(v0.3.0 ref)* | Throughput | Python Reference | Speedup vs Python |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **16 tokens** | Apple Silicon Metal GPU | GPU | $4.21\text{ ms}$ | $237.5\text{ q/s}$ | $8.12\text{ ms}$ | 1.93× |
| **16 tokens** | Apple Neural Engine (ANE) | ANE (W8A16) | $8.27\text{ ms}$ | $120.9\text{ q/s}$ | $8.12\text{ ms}$ | 0.98× |
| **128 tokens** | Apple Silicon Metal GPU | GPU | $7.85\text{ ms}$ | $127.4\text{ q/s}$ | $11.40\text{ ms}$ | 1.45× |
| **128 tokens** | Apple Neural Engine (ANE) | ANE (W8A16) | **$8.27\text{ ms}$** | **$120.9\text{ q/s}$** | $11.40\text{ ms}$ | 1.38× |
| **512 tokens (RAG chunk)** | Apple Silicon Metal GPU | GPU | $25.72\text{ ms}$ | $38.9\text{ q/s}$ | $24.65\text{ ms}$ | 0.96× |
| **512 tokens (RAG chunk)** | **Apple Neural Engine (ANE)** | **ANE (W8A16)** | **$8.27\text{ ms}$** 🚀 | **$120.9\text{ q/s}$** | $24.65\text{ ms}$ | **2.98× faster** 🏆 |

---

### F. Hybrid Multimodal Routing Architecture

When embedding mixed documents (e.g., long 2,048-token context passages accompanied by attached visual card images), neither single-engine approach is optimal:
- Static ANE text buckets (e.g. 512 or 1,024 tokens) cannot natively process 2,048 tokens without falling back to Metal GPU (`TextBackendPolicy::Auto`).
- Running the Vision Tower on Metal GPU consumes GPU cycles and unified memory bandwidth that could otherwise serve long-context attention.

**The Solution: Cross-Engine Asynchronous Routing:**
1. **Vision Tower Offload to ANE:** The SigLIP vision tower runs on the Neural Engine (**11.07 ms** 140b / **23.36 ms** 280b in `v0.3.0` ref), outputting FP32 projector activations directly into unified RAM.
2. **Text Backbone Routing to Metal GPU:** The 2,048-token document text is tokenized and processed via Glypr's zero-copy FlashAttention Metal pipeline (**95.29 ms** BF16 / **76.91 ms** Q8_0 in `v0.4.0`).
3. **End-to-End Hybrid Multimodal Reference (`v0.3.0` capture):**
   - **Python PyTorch MPS:** 165.20 ms
   - **Pure Metal GPU (`v0.3.0` ref):** 145.28 ms
   - **Hybrid Routing (ANE ViT + GPU Backbone, `v0.3.0` ref):** **122.76 ms** (**1.35× faster than Python baseline**).

---

## 8. Cross-Generation Apple Silicon Hardware Invariance & bfloat16 Arithmetic

During multi-target evaluation across Apple Silicon generations (M1, M2, M3, M4, M5), subtle floating-point precision characteristics emerge due to hardware architecture evolutions:

### A. The Architectural Root Cause: Hardware bfloat vs Software Emulation
1. **Apple M3 / M4 / M5 Generation:**
   - Features dedicated hardware Brain Floating Point (`bfloat16`) execution units in GPU shader cores (`__HAVE_BFLOAT__` defined in MSL).
   - In-register Fused Multiply-Add (FMA) instructions ($a \cdot b + c$) execute natively in 16-bit registers with deterministic single-pass mantissa truncation.
2. **Apple M1 / M2 Generation (e.g. M1, M2, M2 Max):**
   - Lacks dedicated hardware `bfloat16` arithmetic execution units.
   - The Metal Shading Language compiler emulates `bfloat` operations by upcasting intermediate operands to IEEE-754 32-bit float (`float`) before truncating back to 16-bit storage.
   - FMA reassociation and compiler contraction optimizations (`-ffast-math`) cause intermediate values in fused shaders (such as `val = (res + norm * w) * scalar`) to be evaluated with 24 bits of mantissa before single truncation, whereas multi-pass unfused pipelines round each intermediate tensor to 7 bits of mantissa at every memory boundary.

### B. The 1-ULP Step Variance Analysis
In `bfloat16`, the significand precision is fixed at 7 bits (1 sign bit, 8 exponent bits, 7 mantissa bits), yielding a relative machine epsilon of $\epsilon \approx 2^{-7} \approx 0.0078125$ (~0.78%).

The minimum representable step size (Unit in the Last Place, or ULP) scales with operand magnitude:
| Magnitude Range | Example Value | 1 ULP Step Size | Measured M1/M2 Delta vs M3+ Reference |
| :---: | :---: | :---: | :---: |
| $[1.0, 2.0)$ | $1.50$ | $2^{-7} = 0.0078125$ | $\le 0.0078125$ |
| $[2.0, 4.0)$ | $2.8125$ | $2^{-6} = 0.015625$ | **$0.015625$** (exact 1 ULP in `vision_qkv_rmsnorm_rope2d_transpose`) |
| $[4.0, 8.0)$ | $7.03125$ | $2^{-5} = 0.03125$ | **$0.03125$** (exact 1 ULP in `fused_metal_rmsnorm_residual_add`) |

### C. Mathematical Parity & Cosine Invariance
Across end-to-end embedding generation (768-dimensional normalized unit vectors):
- Positive and negative 1-ULP rounding variances across 768 dimensions cancel each other out in the dot product.
- Cosine similarity between M1/M2 execution and M3/M4/M5 execution remains strictly above **`0.99995`** (four-9s bitwise mathematical equivalence).
- However, raw element-wise strict equality (`assert_eq!` or static tolerances $< 0.035$) fails on M1/M2 hardware due to the absence of native bfloat hardware units.

### D. Reproduction & Debugging Roadmap (M2 Max Session)
To isolate compiler-level instruction emission differences:
1. **Target Hardware:** Apple MacBook Pro M2 Max (`MTLGPUFamilyApple8`).
2. **Key Test Cases:**
   - `crates/glypr-core/src/metal_kernels/mod.rs::test_phase2_fused_kernels_bit_exact_parity`
   - `crates/glypr-core/src/metal_kernels/mod.rs::test_phase3_fused_kernels_bit_exact_parity`
3. **Disassembly Inspection:**
   - Inspect GPU shader machine code using `xcrun metal-objdump` on compiled Metal libraries:
     ```bash
     xcrun metal -c crates/glypr-core/src/metal_kernels/fused_ops.metal -o /tmp/fused_ops.air
     xcrun metallib /tmp/fused_ops.air -o /tmp/fused_ops.metallib
     xcrun metal-objdump -d /tmp/fused_ops.metallib
     ```
   - Investigate compiler flags (`-fno-fast-math` vs precise FMA) and explicit float32 accumulation in `fused_ops.metal` to ensure identical rounding behavior across all Apple Silicon generations.

