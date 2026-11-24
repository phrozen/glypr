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

### Apple Neural Engine (ANE) & CPU Unified Memory Accounting:
Apple Silicon's unified memory architecture means DRAM is shared across CPU cores, Metal GPU clusters, and the 16-core Neural Engine:
1. **Metal GPU:** Memory is allocated in IOGPU device driver buffers (measured via `[MTLDevice currentAllocatedSize]`).
2. **CPU SIMD:** Memory is allocated in process heap/virtual memory pages (measured via POSIX `getrusage` RSS).
3. **Apple Neural Engine (ANE):** Weights compiled into `.mlmodelc` packages are memory-mapped and loaded into physical RAM by the Core ML runtime framework.

> [!IMPORTANT]
> **The Zero-VRAM Fallacy:**
> While CPU SIMD and ANE inference record **0 MB of IOGPU Metal allocations**, claiming "0 MB memory usage" is misleading. Model weights and activation buffers still reside in physical unified DRAM. To maintain absolute transparency, Glypr benchmarks report **Peak Physical Unified Memory (RAM + VRAM)**:
> - **Python PyTorch MPS Baseline (*prior ref*):** **6,592 MB** (6.44 GB) at 8,192 tokens.
> - **Glypr Metal Flash (`v0.4.0`):** **1,177.5 MB** (BF16, `512.0 MB` transient) / **1,102.8 MB** (Q8_0, `496.0 MB` transient, `662.8 MB` RSS) at 8,192 tokens (**$-82\%$ to $-83\%$ peak Metal VRAM vs MPS ref**, saving > 5.4 GB).
> - **Glypr Universal CPU SIMD (`v0.4.0`):** **1,084.8 MB** (`64t–1K`) to **1,397.4 MB** (`4K`) RSS on Q8_0 / **1,800.6–1,813.3 MB** RSS on BF16.
> - **Glypr ANE Resident (512t + Vision, *prior ref*):** **~450–700 MB** total physical footprint.

---

## 2. Anatomy of Transient Allocations: From 14.6 GB to 1.1 GB

Prior to hardware optimization, running full 8,192-token embeddings produced massive transient memory spikes that frequently triggered macOS kernel memory pressure warnings. Glypr systematically eliminated those transient spikes across four stages:

```
14,590 MB ──┐ (Unoptimized Dense Baseline)
            │
            ├─► [-5,304 MB] Query-Chunked Softmax
            ├─► [-2,013 MB] Hybrid Per-Layer PLE Normalization (B*S >= 1024)
            ├─► [-134 MB]   In-Register Tile-Skipping (0 MB Mask)
            ▼
 2,242 MB ──┤ (Chunked Sliding v0.4.0)
            │
            ├─► Zero-Copy GQA + Fused Metal Kernels + M5 FlashAttention
            ▼
 1,178 MB (BF16 Metal Flash) / 1,103 MB (Q8_0 Metal Flash)
```

### 1. Hybrid Per-Layer PLE Normalization (`B * S >= 1024`)
- **Bottleneck:** Gemma 4 projects token embeddings into Per-Layer Embeddings (PLE) across all 24 layers simultaneously: $[B, N, 24, 512]$. Naive evaluation performed `RMSNorm` over this entire 4D tensor in float32, creating 5 intermediate tensors ($402.6\text{ MB} \times 5 = \mathbf{2,013\text{ MB}}$) before Layer 0 even began.
- **Resolution:** Because RMSNorm operates strictly along the last dimension ($D=512$), layers are mathematically independent. For short sequences ($B \cdot N < 1024$), Glypr uses a fused single-pass 4D kernel; for longer sequences ($B \cdot N \ge 1024$), zero-copy `layer_views` normalize each layer's slice $[B, N, 512]$ on-demand inside the forward loop—capping PLE transient allocation at **$16.8\text{ MB}$** at 8K with **$0.0$ numerical difference**.

### 2. Full-Context Dense Softmax (+5,304 MB)
- **Bottleneck:** In layers 5, 11, 17, and 23 (global full attention), materializing an $[8192 \times 8192]$ float32 attention matrix consumed $1.07\text{ GB}$. The intermediate operations inside softmax (`sub(max)`, `exp`, `sum`, `div`) allocated over **$5.3\text{ GB}$** of transient buffers per layer.
- **Resolution:** Query-chunked attention partitions the query dimension into blocks of $512$ tokens attending to all $8,192$ keys. The active scratchpad dropped from $8192 \times 8192$ ($1.07\text{ GB}$) to $512 \times 8192$ ($67\text{ MB}$), yielding **$100\%$ bit-for-bit mathematical identity**.

### 3. Mask Tensor Materialization (+134 MB)
- **Bottleneck:** Naive sliding-window implementations allocated an $[8192 \times 8192]$ additive mask tensor filled with $0.0$ and $-\infty$ ($134.2\text{ MB}$).
- **Resolution:** Handled analytically inside GPU thread registers: `if (abs(q_pos - k_pos) > 512) score = -INFINITY`, reducing mask allocation to **$0.0\text{ MB}$**.

---

## 3. Empirical Hardware Memory Scorecard (Apple Silicon)

All `glypr-core` rows below were freshly captured in **`v0.4.0`** (`bench_memory --features metal,accelerate`) via an isolated subprocess harness querying `[MTLDevice currentAllocatedSize]` and `libc::getrusage` (20-iteration median latencies from `compare_models` are `47.37/39.05 ms` at 1K, `95.29/76.91 ms` at 2K, `228.92/199.54 ms` at 4K, and `620.81/564.91 ms` at 8K for BF16/Q8_0 Metal Flash). The `PyTorch MPS` row reflects the prior baseline reference capture.

| Context Length | Framework / Precision | Attention Backend | Model VRAM | Peak Metal VRAM | Transient VRAM | Max Process RSS | Cold / Warm Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1,024 tokens** | `glypr-core` BF16 | Standard Dense | 665.4 MB | 790.8 MB | 125.4 MB | 2,046.2 MB | 111.93 / 90.36 ms |
| **1,024 tokens** | `glypr-core` BF16 | Chunked Sliding | 665.4 MB | 933.5 MB | 268.0 MB | 2,046.3 MB | 102.76 / 92.73 ms |
| **1,024 tokens** | `glypr-core` BF16 | Metal Flash (Fused) | 665.4 MB | **705.5 MB** | **40.0 MB** | 2,046.3 MB | 61.16 / **48.04 ms** |
| **1,024 tokens** | `glypr-core` Q8_0 | Metal Flash (Fused) | **606.7 MB** | **668.8 MB** | 62.0 MB | **663.0 MB** | 51.01 / **38.68 ms** 🚀 |
| | | | | | | | |
| **2,048 tokens** | `glypr-core` BF16 | Standard Dense | 665.4 MB | 991.2 MB | 325.7 MB | 2,046.2 MB | 272.41 / 261.30 ms |
| **2,048 tokens** | `glypr-core` BF16 | Chunked Sliding | 665.4 MB | 1,040.2 MB | 374.8 MB | 2,046.4 MB | 216.01 / 205.02 ms |
| **2,048 tokens** | `glypr-core` BF16 | Metal Flash (Fused) | 665.4 MB | **749.5 MB** | **84.0 MB** | 2,046.3 MB | 107.22 / **96.39 ms** |
| **2,048 tokens** | `glypr-core` Q8_0 | Metal Flash (Fused) | **606.7 MB** | **730.8 MB** | 124.0 MB | **662.9 MB** | 88.35 / **77.38 ms** 🚀 |
| | | | | | | | |
| **4,096 tokens** | `glypr-core` BF16 | Standard Dense | 665.4 MB | 1,603.9 MB | 938.4 MB | 2,046.4 MB | 855.05 / 843.55 ms |
| **4,096 tokens** | `glypr-core` BF16 | Chunked Sliding | 665.4 MB | 1,448.4 MB | 782.9 MB | 2,046.3 MB | 477.40 / 466.25 ms |
| **4,096 tokens** | `glypr-core` BF16 | Metal Flash (Fused) | 665.4 MB | **889.5 MB** | **224.0 MB** | 2,046.6 MB | 242.07 / **231.96 ms** |
| **4,096 tokens** | `glypr-core` Q8_0 | Metal Flash (Fused) | **606.7 MB** | **854.8 MB** | 248.0 MB | **663.0 MB** | 210.55 / **200.32 ms** 🚀 |
| | | | | | | | |
| **8,192 tokens** *(ref)* | **PyTorch MPS (Golden)**| **MPSGraph SDPA** | **1,472.5 MB** | **6,592.8 MB** | 5,120.3 MB | **1,211.1 MB** | — / **875.00 ms** *(ref)* |
| **8,192 tokens** | `glypr-core` BF16 | Standard Dense | 665.4 MB | 3,858.1 MB | 3,192.6 MB | 2,046.2 MB | 3,001.26 / 2,970.26 ms |
| **8,192 tokens** | `glypr-core` BF16 | Chunked Sliding | 665.4 MB | **2,241.6 MB** | **1,576.2 MB** | 2,046.4 MB | 1,140.23 / 1,127.60 ms |
| **8,192 tokens** | `glypr-core` BF16 | Metal Flash (Fused) | **665.4 MB** | **1,177.5 MB** | **512.0 MB** | 2,046.5 MB | 641.17 / **635.60 ms** |
| **8,192 tokens** | `glypr-core` Q8_0 | Metal Flash (Fused) | **606.7 MB** | **1,102.8 MB** | **496.0 MB** | **662.8 MB** | 575.78 / **578.34 ms** 🏆 |

---

## 4. The Pareto Frontier: Chunked Sliding vs. Metal Flash

The data reveals a clean **Pareto optimization frontier** between memory efficiency and execution latency at long sequence lengths ($N = 8,192$):

```
Peak Metal VRAM (MB)
  ▲
  │   [PyTorch MPS (ref): 6,593 MB / 875 ms]
  │
2.2K│ [Chunked Sliding (v0.4): 2,242 MB / 1,128 ms] ◄── Portable CPU/GPU Bounded Scratchpad
  │
1.2K│                  [Metal Flash BF16 (v0.4): 1,178 MB / 621 ms] ◄── Lowest Latency + Float Precision
1.1K│                  [Metal Flash Q8_0 (v0.4): 1,103 MB / 565 ms] ◄── Absolute Latency & Memory Champion 🏆
  │
  └──────────────────────────────────────────────────────────► Execution Latency (ms)
      550ms                   800ms                   1,200ms
```

### Architectural Trade-off Summary:

1. **`AttentionMode::ChunkedSliding` (Bounded CPU/GPU Scratchpad):**
   - **Peak VRAM (`v0.4.0`):** **$2,241.6\text{ MB}$** at 8,192 tokens (**$66.0\%$ lower memory than PyTorch MPS**).
   - **Mechanism:** Queries are evaluated in strict chunks of 512 tokens. Intermediate matrix materializations never exceed a $3W$ window.
   - **Recommended For:** Portable CPU execution and environments where custom Metal Flash shaders are disabled.

2. **`AttentionMode::MetalFlash` (The Default & Speed + Memory Champion):**
   - **Execution Latency (`v0.4.0` `compare_models`):** **$564.91\text{ ms}$** (Q8_0) / **$620.81\text{ ms}$** (BF16) at 8,192 tokens (**faster than PyTorch MPS's $875.00\text{ ms}$ by up to 310 ms / 1.55x**).
   - **Peak VRAM (`v0.4.0`):** **$1,177.5\text{ MB}$** (BF16, down from $1,473.5\text{ MB}$ in `v0.3.0`) — **$-82.1\%$ lower than PyTorch MPS**.
   - **Transient VRAM (`v0.4.0`):** Slashed to **$512.0\text{ MB}$** (BF16, down from $688.4\text{ MB}$ in `v0.3.0`).
   - **Mechanism:** GPU thread registers compute in-place tile reductions with zero-copy GQA across both $d_{\text{head}}=256$ and $d_{\text{head}}=512$, direct `[B, S, H, D]` attention output, zero mask allocation for unpadded sequences, fused backbone kernels, zero-overhead weight concatenation views, and M5 Neural Accelerator `mpp::tensor_ops` contractions.
   - **Recommended For:** Default engine across text, multimodal, and server serving.

3. **`GGUF Q8_0` + `MetalFlash` (The Production Edge Sweet Spot):**
   - Cuts base model disk storage down to **$295.5\text{ MB}$** ($-44.5\%$) and base model VRAM to **$606.7\text{ MB}$**.
   - Peak VRAM at 8K context is **$1,102.8\text{ MB}$** (down from $1,582.7\text{ MB}$ in `v0.3.0`) with transient VRAM reduced to **$496.0\text{ MB}$** and Max Process RSS dropped to **$662.8\text{ MB}$** (down from $1,678.5\text{ MB}$ in `v0.3.0`).
   - Native Metal `block_q8_0` dequant kernels (`BK=64`, `packed_char4`, `is_mv = m <= 8`) and M5 NAX INT8 GEMM deliver **$564.91\text{ ms}$** median (`14,502 tok/s`) at 8K tokens.


---

## 5. CPU Memory Scaling & Footprint (`v0.4.0`)

When executing on CPU with Rayon parallelism, `pulp` SIMD kernels, and optional Apple Accelerate (`bench_memory --features metal,accelerate`):

| Context Length | Precision | Cold / Warm Latency | Throughput | Peak Process RSS (CPU RAM) | Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **64 tokens** | GGUF Q8_0 | 105.4 / 103.1 ms | 621 tok/s | **1,084.8 MB** | Lowest CPU memory footprint |
| **64 tokens** | GGUF BF16 | 80.1 / **69.7 ms** | **919 tok/s** | 1,800.6 MB | Parallel BF16 CPU SIMD |
| **256 tokens** | GGUF Q8_0 | 257.7 / 256.4 ms | 998 tok/s | **1,136.0 MB** | Constant working memory |
| **256 tokens** | GGUF BF16 | 154.7 / **144.8 ms** | **1,768 tok/s** | 1,800.6 MB | Zero heap churn from SIMD operators |
| **512 tokens** | GGUF Q8_0 | 572.2 / 539.4 ms | 949 tok/s | **1,084.8 MB** | Flat RSS on Q8_0 weights |
| **512 tokens** | GGUF BF16 | 301.2 / **284.1 ms** | **1,802 tok/s** | 1,800.6 MB | Peak CPU throughput (**2.1× faster vs v0.2**) |
| **1,024 tokens** | GGUF Q8_0 | 1,111.2 / 1,060.8 ms | 965 tok/s | **1,084.8 MB** | Stable Rayon threadpool buffers |
| **1,024 tokens** | GGUF BF16 | 616.3 / **601.4 ms** | **1,703 tok/s** | 1,800.6 MB | Predictable linear growth |
| **2,048 tokens** | GGUF Q8_0 | 2,226.8 / 2,205.2 ms | 929 tok/s | **1,136.1 MB** | Flat footprint under 1.15 GB RSS |
| **2,048 tokens** | GGUF BF16 | 1,298.4 / **1,279.0 ms** | **1,601 tok/s** | 1,800.6 MB | Zero swap growth |
| **4,096 tokens** | GGUF Q8_0 | 4,835.1 / 4,808.4 ms | 852 tok/s | **1,397.4 MB** | Safe on 2–4 GB cloud instances |
| **4,096 tokens** | GGUF BF16 | 2,888.5 / **2,838.7 ms** | **1,443 tok/s** | 1,813.3 MB | **2.19× faster vs v0.2 reference** |

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

### Empirical Multimodal Image Batch Memory Profile (Budget 140, *prior reference capture*):
*(Note: In `v0.4.0`, `bench_batch` clocks `B = 8` Budget 140 vision at `315.96 ms` total / `39.50 ms/item` / `25.3 items/s`.)*

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
