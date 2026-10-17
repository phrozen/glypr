# Empirical Memory, VRAM & Transient Allocation Architecture

This document provides a rigorous analysis of the memory footprint of **EmbeddingGemma 2** running in Glypr (`glypr-core`). It covers Apple Silicon unified memory accounting mechanics, transient tensor churn, empirical VRAM scaling benchmarks, and the architectural Pareto trade-off between **Chunked Sliding Attention** and **Fused Metal FlashAttention**.

---

## 1. Unified Memory Mechanics: The CPU RSS Accounting Trap

On Apple Silicon (M-series processors), the CPU, GPU, and Neural Engine share physical unified LPDDR memory over a high-bandwidth system bus. However, macOS enforces strict architectural separation between user-space virtual memory page tables and hardware IOGPU Metal allocations:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Physical Unified Memory (RAM)                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│     CPU Virtual Memory Pages    │   │      IOGPU Metal Allocations    │
│    (Accounted by getrusage)     │   │ (Accounted by currentAllocated) │
├─────────────────────────────────┤   ├─────────────────────────────────┤
│ • Process heap & stack          │   │ • Model weight buffers          │
│ • Tokenizer data structures     │   │ • Command buffer command queues │
│ • Rust thread runtime & Rayon   │   │ • Metal FlashAttention scratch  │
│                                 │   │ • Intermediate activation tiles │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

### Why Standard Process RSS Fails:
Querying standard POSIX process telemetry (`ps`, `libc::getrusage(RUSAGE_SELF)`, or `mach_task_basic_info`) **only captures the CPU virtual page tables**. Metal device allocations are managed asynchronously in kernel space by the IOGPU driver:
- **The Failure Mode:** Under unoptimized 8,192-token attention, POSIX RSS reported a deceptively placid **$1,211\text{ MB}$**, while the physical system was thrashing under **$14,590\text{ MB}$** of GPU unified allocations.
- **The Ground-Truth Metric:** True hardware memory consumption must be polled via the Objective-C runtime method:
  ```rust
  // Direct hardware Metal allocation inquiry
  let allocated_bytes: usize = msg_send![device, currentAllocatedSize];
  ```

---

## 2. Anatomy of Transient Allocations: From 14.6 GB to 3.0 GB

Prior to hardware optimization, running full 8,192-token embeddings produced massive transient memory spikes that frequently triggered macOS kernel memory pressure warnings. Glypr identified and eradicated three distinct memory leaks:

```
14,590 MB ──┐ (Unoptimized Baseline)
            │
            ├─► [-2,013 MB] Sliced PLE Normalization
            │
            ├─► [-5,304 MB] Query-Chunked Softmax
            │
            ├─► [-134 MB] In-Register Tile-Skipping (0 MB Mask)
            ▼
 3,094 MB ──┘ (Optimized Chunked) / 3,951 MB (Fused Metal Flash)
```

### 1. Un-sliced 4D PLE RMSNorm (+2,013 MB)
- **Bottleneck:** Gemma-2 projects token embeddings into Per-Layer Embeddings (PLE) across all 24 layers simultaneously: $[1, N, 24, 512]$. Naive Candle evaluation performed `RMSNorm` over this entire 4D tensor in float32, creating 5 intermediate tensors ($402.6\text{ MB} \times 5 = \mathbf{2,013\text{ MB}}$) before Layer 0 even began.
- **Resolution:** Because RMSNorm operates along the last dimension ($D=512$), layers are mathematically independent. Glypr normalizes each layer's slice $[1, N, 512]$ on-demand inside the forward loop, slashing active norm allocations to **$16.7\text{ MB}$** with **$0.0$ numerical difference**.

### 2. Full-Context Dense Softmax (+5,304 MB)
- **Bottleneck:** In layers 5, 11, 17, and 23 (global full attention), materializing an $[8192 \times 8192]$ float32 attention matrix consumed $1.07\text{ GB}$. The intermediate operations inside softmax (`sub(max)`, `exp`, `sum`, `div`) allocated over **$5.3\text{ GB}$** of transient buffers per layer.
- **Resolution:** Query-chunked attention partitions the query dimension into blocks of $512$ tokens attending to all $8,192$ keys. The active scratchpad dropped from $8192 \times 8192$ ($1.07\text{ GB}$) to $512 \times 8192$ ($67\text{ MB}$), yielding **$100\%$ bit-for-bit mathematical identity**.

### 3. Mask Tensor Materialization (+134 MB)
- **Bottleneck:** Naive sliding-window implementations allocated an $[8192 \times 8192]$ additive mask tensor filled with $0.0$ and $-\infty$ ($134.2\text{ MB}$).
- **Resolution:** Handled analytically inside GPU thread registers: `if (abs(q_pos - k_pos) > 512) score = -INFINITY`, reducing mask allocation to **$0.0\text{ MB}$**.

---

## 3. Empirical Hardware Memory Scorecard (Apple Silicon)

The following metrics were captured on Apple Silicon hardware using an isolated subprocess harness with a high-resolution 1ms background polling thread querying `[MTLDevice currentAllocatedSize]` and `libc::getrusage`:

| Context Length | Framework / Precision | Attention Backend | Model VRAM | Peak Metal VRAM | Transient VRAM | Max Process RSS | Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1,024 tokens** | `glypr-core` BF16 | Standard Dense | 781.0 MB | 1,045.7 MB | 264.6 MB | 2,037.3 MB | 95.58 ms |
| **1,024 tokens** | `glypr-core` BF16 | Chunked Sliding | 785.0 MB | 1,180.3 MB | 395.3 MB | 2,038.5 MB | 95.58 ms |
| **1,024 tokens** | `glypr-core` BF16 | Metal Flash (Fused) | 785.0 MB | **986.2 MB** | **201.2 MB** | 2,038.5 MB | **52.88 ms** |
| **1,024 tokens** | `glypr-core` Q8_0 | Metal Flash (Fused) | **606.3 MB** | **754.1 MB** | **147.8 MB** | **1,678.5 MB** | **55.00 ms** |
| | | | | | | | |
| **2,048 tokens** | `glypr-core` BF16 | Standard Dense | 781.0 MB | 1,370.0 MB | 589.0 MB | 2,037.2 MB | 258.30 ms |
| **2,048 tokens** | `glypr-core` BF16 | Chunked Sliding | 785.0 MB | 1,430.3 MB | 645.3 MB | 2,038.5 MB | 204.18 ms |
| **2,048 tokens** | `glypr-core` BF16 | Metal Flash (Fused) | 785.0 MB | **1,120.4 MB** | **335.4 MB** | 2,038.5 MB | **114.73 ms** |
| **2,048 tokens** | `glypr-core` Q8_0 | Metal Flash (Fused) | **606.3 MB** | **935.2 MB** | **328.9 MB** | **1,678.5 MB** | **116.54 ms** |
| | | | | | | | |
| **4,096 tokens** | `glypr-core` BF16 | Standard Dense | 781.0 MB | 2,118.8 MB | 1,337.7 MB | 2,037.3 MB | 831.51 ms |
| **4,096 tokens** | `glypr-core` BF16 | Chunked Sliding | 785.0 MB | 2,006.5 MB | 1,221.5 MB | 2,038.6 MB | 462.23 ms |
| **4,096 tokens** | `glypr-core` BF16 | Metal Flash (Fused) | 785.0 MB | **1,177.4 MB** | **392.4 MB** | 2,038.6 MB | **277.24 ms** |
| **4,096 tokens** | `glypr-core` Q8_0 | Metal Flash (Fused) | **606.3 MB** | **1,086.7 MB** | **480.4 MB** | **1,678.6 MB** | **287.25 ms** |
| | | | | | | | |
| **8,192 tokens** | **PyTorch MPS (Golden)**| **MPSGraph SDPA** | **1,472.5 MB** | **6,592.8 MB** | 5,120.3 MB | **1,211.1 MB** | **875.00 ms** |
| **8,192 tokens** | `glypr-core` BF16 | Standard Dense | 781.0 MB | 4,608.2 MB | 3,827.2 MB | 2,037.2 MB | 2,937.26 ms |
| **8,192 tokens** | `glypr-core` BF16 | Chunked Sliding | 785.0 MB | **2,880.0 MB** | **2,095.0 MB** | 2,038.5 MB | 1,118.50 ms |
| **8,192 tokens** | `glypr-core` BF16 | Metal Flash (Fused) | 785.0 MB | **1,473.5 MB** | **688.4 MB** | 2,038.5 MB | **774.83 ms** 🏆 |
| **8,192 tokens** | `glypr-core` Q8_0 | Metal Flash (Fused) | **606.3 MB** | **1,582.7 MB** | **976.4 MB** | **1,678.5 MB** | **796.23 ms** |

---

## 4. The Pareto Frontier: Chunked Sliding vs. Metal Flash

The data reveals a clean **Pareto optimization frontier** between memory efficiency and execution latency at long sequence lengths ($N = 8,192$):

```
Peak Metal VRAM (MB)
  ▲
  │   [PyTorch MPS: 6,593 MB / 875 ms]
  │
4K│                    [Metal Flash BF16: 3,952 MB / 868 ms] ◄── Lowest Latency (Beats PyTorch)
  │                    [Metal Flash Q8_0: 3,727 MB / 895 ms] ◄── Balanced Edge Champion
  │
3K│   [Chunked Sliding: 3,095 MB / 1,288 ms] ◄── Lowest Memory (Strict Envelopes)
  │
  └──────────────────────────────────────────────────────────► Execution Latency (ms)
      800ms                   1,000ms                 1,400ms
```

### Architectural Trade-off Summary:

1. **`AttentionMode::ChunkedSliding` (The Memory Champion):**
   - **Peak VRAM:** **$2,880.0\text{ MB}$** at 8,192 tokens (**$56.3\%$ lower memory than PyTorch MPS**).
   - **Mechanism:** Queries are evaluated in strict chunks of 512 tokens. Intermediate matrix materializations never exceed a $3W$ window.
   - **Recommended For:** Background embedding daemons, memory-constrained machines (8 GB / 16 GB unified RAM), and concurrent multi-model serving.

2. **`AttentionMode::MetalFlash` (The Default & Speed Champion):**
   - **Execution Latency:** **$774.83\text{ ms}$** at 8,192 tokens (**faster than PyTorch MPS's $875.00\text{ ms}$ by over 100 ms**).
   - **Peak VRAM:** **$1,473.5\text{ MB}$** at 8,192 tokens (**$-77.7\%$ lower than PyTorch MPS**; down from $4,079.5\text{ MB}$ prior to Phase 1's native $d_{\text{head}}=512$ kernel).
   - **Transient VRAM:** Slashed by **$79.1\%$** from $3,298.4\text{ MB} \to \mathbf{688.4\text{ MB}}$.
   - **Mechanism:** GPU thread registers compute in-place tile reductions with zero-copy GQA across both $d_{\text{head}}=256$ and $d_{\text{head}}=512$, with zero mask allocation for unpadded sequences, fused backbone RMSNorm/RoPE/GeGLU/Residual kernels, and zero-overhead weight concatenation views.
   - **Recommended For:** Default engine across text, multimodal, and server serving.

3. **`GGUF Q8_0` + `MetalFlash` (The Production Edge Sweet Spot):**
   - Cuts base model disk storage down to **$295.5\text{ MB}$** ($-44.5\%$).
   - Base model VRAM is **$606.3\text{ MB}$** (with 0.0 MB weight-fusion overhead).
   - Peak VRAM at 8K context is **$1,582.7\text{ MB}$** (down from $3,855.4\text{ MB}$ prior to Phase 1) with transient VRAM reduced to **$976.4\text{ MB}$** ($-70.5\%$).
   - Native Metal `block_q8_0` dequant kernels eliminate all 432 `BF16 <-> F32` cast dispatches per pass.


---

## 5. CPU Memory Scaling & Footprint

When executing on CPU with multi-core Rayon parallelism and `pulp` SIMD kernels:

| Context Length | Model Precision | Peak Process RSS (CPU RAM) | Notes |
| :---: | :---: | :---: | :--- |
| **64 tokens** | GGUF Q8_0 | **1,040.9 MB** | Minimal cold allocation footprint |
| **64 tokens** | GGUF BF16 | 1,738.7 MB | Full BF16 weight mapping |
| **256 tokens** | GGUF Q8_0 | **1,134.7 MB** | Constant working memory |
| **512 tokens** | GGUF Q8_0 | **1,134.6 MB** | Zero heap churn from SIMD operators |
| **1,024 tokens** | GGUF Q8_0 | **1,251.0 MB** | Stable Rayon threadpool buffers |
| **2,048 tokens** | GGUF Q8_0 | **1,406.2 MB** | Predictable linear growth |
| **4,096 tokens** | GGUF Q8_0 | **2,054.6 MB** | Safe on 4 GB and 8 GB cloud instances |

Thanks to `simd_rmsnorm` and `simd_geglu` operating directly on continuous storage without intermediate tensor allocations, the CPU memory footprint remains flat and deterministic, eliminating memory fragmentation in long-running processes.

---

## 6. Multimodal Memory Footprint & Elimination of 30 GB Spikes

During multimodal batch processing (such as indexing card images or visual documents), naive vision attention implementations suffer from severe unified memory inflation that can easily drive system RAM consumption **north of 30 GB**:

### The Anatomy of the 30 GB Spike:
1. **The $4.0\text{ GB}$ Single-Image Scratchpad:**
   - In unoptimized Vision Transformer self-attention, each layer materializes a dense attention weight matrix: $[1, 16, 1134, 1134]$ ($41.1\text{ MB}$ in BF16).
   - In float32 upcast, this requires $82.3\text{ MB}$.
   - Softmax creates another $82.3\text{ MB}$ intermediate tensor.
   - Downcasting back to BF16 creates a $41.1\text{ MB}$ tensor.
   - Total transient allocations per layer: $\sim 247\text{ MB}$. Across 16 layers: **$\approx 3.95\text{ GB}$ per single image**.
2. **Unified Memory Autorelease Churn:**
   - On Apple Silicon, Metal GPU buffers share physical RAM with the operating system.
   - Under batch loops or concurrent requests, `MTLCommandBuffer` retained transient memory buffers until the command queue completed and drained.
   - At a batch of 8 to 16 cards, $8 \times 3.95\text{ GB} = \mathbf{31.6\text{ GB}}$ of transient allocations accumulated simultaneously, triggering aggressive swap thrashing and kernel memory pressure warnings.

### The Glypr Fused SDPA & Autorelease Scoping Solution:
1. **Fused Metal SDPA Attention:**
   - The $N \times N$ attention weight matrix is **never materialized in memory**.
   - Scaled dot-product attention and online softmax are computed on-the-fly inside hardware threadgroup registers.
   - All $3.95\text{ GB}$ of transient allocations per image are **reduced to $0\text{ MB}$**.
2. **Apple Metal Autorelease Pool Scoping (`AutoreleaseScope`):**
   - Metal command buffer allocations and transient tensor buffers are scoped via `objc_autoreleasePoolPush()` / `objc_autoreleasePoolPop()`.
   - Every batch iteration drains immediately, ensuring zero accumulation across consecutive batches or continuous server operation.
   - CPU image decompression (PNG/JPEG) is pre-executed concurrently in Tokio blocking worker threads prior to GPU submission, leaving the GPU worker thread free from CPU heap inflation.

### Empirical Multimodal Image Batch Memory Profile (Budget 140):

| Execution Configuration | Batch Size | Total Wallclock | Per-Image Latency | Throughput | Parity | Metal VRAM | Peak Process RSS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sequential Baseline** | 1 | 2,413.57 ms (32 items) | 75.40 ms / item | 13.26 images/sec | 1.000000 | 1,632.88 MB | **2,037.92 MB** |
| **Batched (`ChunkedSliding`)** | 4 | 259.80 ms | **64.95 ms / item** | **15.40 images/sec** | **1.000000** | 1,641.67 MB | **2,037.92 MB** |
| **Batched (`ChunkedSliding`)** | 8 | 523.38 ms | **65.42 ms / item** | **15.29 images/sec** | **1.000000** | 1,673.69 MB | **2,037.92 MB** |
| **Batched (`ChunkedSliding`)** | 16 | 1,068.99 ms | **66.81 ms / item** | **14.97 images/sec** | **1.000000** | 1,967.72 MB | **2,037.92 MB** |
| **Batched (`ChunkedSliding`)** | 32 | 2,215.20 ms | **69.23 ms / item** | **14.45 images/sec** | **1.000000** | 3,130.28 MB | **2,037.92 MB** |
| **Batched (`MetalFlash`)** | 8 | 509.55 ms | **63.69 ms / item** | **15.70 images/sec** | 0.999931 | 1,673.69 MB | **2,037.92 MB** |
| **Batched (`MetalFlash`)** | 32 | 2,147.53 ms | **67.11 ms / item** | **14.90 images/sec** | 0.999931 | 3,130.28 MB | **2,037.92 MB** |

> [!NOTE]
> Peak process RSS remains strictly locked at **$2,037.92\text{ MB}$** throughout the entire 32-image batch execution, with **zero memory leakage** and zero swap inflation. Parity with sequential single-item execution is **100% bit-exact (1.000000)** under `ChunkedSliding`.
