#!/usr/bin/env python3
"""Benchmark Python baseline performance for EmbeddingGemma 2 on Apple Silicon MPS / CPU.

Measures:
1. Cold start / Model load time
2. Latency for short (32t), medium (128t), long (512t) inputs
3. Throughput across batch sizes (1, 8, 32)
4. Memory footprint (Process RSS)
"""

import os
import sys
import time
import json
from pathlib import Path
import resource
import torch
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from baseline.embedder import EmbeddingGemmaBaseline

def get_process_memory_mb() -> float:
    # On macOS, ru_maxrss is in bytes
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss_bytes / (1024 * 1024)

def run_benchmarks():
    print("=" * 60)
    print("  EmbeddingGemma 2 (270M Text Baseline) - Python Benchmark")
    print("=" * 60)

    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    print(f"Target Device: {device}")
    print(f"Initial Process RSS: {get_process_memory_mb():.2f} MB")

    # 1. Measure Load Time
    t0 = time.perf_counter()
    embedder = EmbeddingGemmaBaseline(mode="text_only", device=device, torch_dtype=torch.bfloat16)
    load_time_sec = time.perf_counter() - t0
    post_load_rss = get_process_memory_mb()
    print(f"Model Load Time: {load_time_sec:.3f} s")
    print(f"Post-load Process RSS: {post_load_rss:.2f} MB (Delta: +{post_load_rss - get_process_memory_mb():.2f} MB)")

    # 2. Warmup
    print("\nWarming up...")
    _ = embedder.encode_query("Warmup query to trigger kernel compilation.", task="SearchQuery")
    if device == "mps":
        torch.mps.synchronize()

    # 3. Sequence Length Latency Benchmarks
    test_corpora = {
        "short_query (~10 words)": "What is the capital and largest city of France?",
        "medium_passage (~50 words)": "Photosynthesis is the biological process used by plants, algae, and certain bacteria to harness energy from sunlight and turn it into chemical energy. Photosynthesis in green plants involves the chlorophyll pigment absorbing light energy.",
        "long_passage (~200 words)": " ".join([
            "The quick brown fox jumps over the lazy dog in an iterative search paradigm."
        ] * 15),
    }

    latency_results = {}
    print("\n--- Latency per Input Length (Batch Size = 1, N=20 runs) ---")
    for label, text in test_corpora.items():
        times = []
        # Run 20 iterations
        for _ in range(20):
            if device == "mps":
                torch.mps.synchronize()
            t_start = time.perf_counter()
            _ = embedder.encode_document(text, truncate_dim=768)
            if device == "mps":
                torch.mps.synchronize()
            t_end = time.perf_counter()
            times.append((t_end - t_start) * 1000.0) # ms

        median_ms = float(np.median(times))
        p95_ms = float(np.percentile(times, 95))
        min_ms = float(np.min(times))
        latency_results[label] = {
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "min_ms": min_ms,
        }
        print(f"  {label:<30}: Median = {median_ms:6.2f} ms | P95 = {p95_ms:6.2f} ms | Min = {min_ms:6.2f} ms")

    # 4. Batch Throughput Benchmarks
    sample_text = "Dense vector representation and embedding search optimization using Apple Silicon MPS."
    batch_sizes = [1, 8, 16, 32]
    throughput_results = {}

    print("\n--- Batch Throughput (N=10 runs) ---")
    for bs in batch_sizes:
        batch = [sample_text] * bs
        times = []
        for _ in range(10):
            if device == "mps":
                torch.mps.synchronize()
            t_start = time.perf_counter()
            _ = embedder.encode_document(batch, truncate_dim=768)
            if device == "mps":
                torch.mps.synchronize()
            t_end = time.perf_counter()
            times.append(t_end - t_start)

        median_sec = float(np.median(times))
        items_per_sec = bs / median_sec
        throughput_results[f"batch_{bs}"] = {
            "batch_size": bs,
            "median_time_ms": median_sec * 1000.0,
            "items_per_sec": items_per_sec,
        }
        print(f"  Batch Size {bs:2d}: {median_sec * 1000.0:6.2f} ms total | {items_per_sec:6.1f} items/sec")

    # 5. Export Metrics Summary
    output_path = Path(__file__).resolve().parent.parent / "golden_vectors" / "python_metrics.json"
    results = {
        "device": device,
        "dtype": "bfloat16",
        "load_time_sec": load_time_sec,
        "peak_rss_mb": get_process_memory_mb(),
        "latencies_ms": latency_results,
        "throughput": throughput_results,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved Python benchmark metrics to {output_path}")

if __name__ == "__main__":
    run_benchmarks()
