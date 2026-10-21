"""Automated Python PyTorch MPS benchmark harness for Glypr performance matrix."""

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
import torch
from PIL import Image

# Ensure tests module is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.baseline.embedder import EmbeddingGemmaBaseline


def get_peak_memory_mb() -> float:
    """Return total physical unified memory (Metal allocations + CPU RSS) in MB."""
    mps_mem = 0.0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_mem = torch.mps.current_allocated_memory() / (1024 * 1024)
    rss_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    return mps_mem + rss_mem


def benchmark_fn(fn, warmup: int = 3, iters: int = 10) -> float:
    """Run warmup and return median latency in milliseconds."""
    for _ in range(warmup):
        _ = fn()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]


def main():
    print("=" * 60)
    print("  PYTHON PYTORCH MPS BENCHMARK HARNESS")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    results = {"device": device, "measurements": {}}

    # Load corpus entries
    corpus_path = Path("tests/golden_vectors/corpus_8k.json")
    if not corpus_path.exists():
        corpus_path = Path("../../tests/golden_vectors/corpus_8k.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    corpus_map = {e["id"]: e["content"] for e in corpus["entries"]}
    text_512 = corpus_map.get("doc_physics_512t", "quantum " * 358)
    text_2048 = corpus_map.get("doc_physics_2048t", "physics " * 1421)
    text_8192 = corpus_map.get("doc_physics_8192t", "general relativity " * 2800)

    # 1. Text Only Benchmarks
    print("\n[1/3] Loading text-only model...")
    text_embedder = EmbeddingGemmaBaseline(mode="text_only", device=device)

    print("  -> Benchmarking text_512...")
    t_512 = benchmark_fn(lambda: text_embedder.encode_document(text_512))
    results["measurements"]["text_512"] = {"latency_ms": round(t_512, 2)}
    print(f"     Result: {t_512:.2f} ms")

    print("  -> Benchmarking text_8192...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    t_8192 = benchmark_fn(lambda: text_embedder.encode_document(text_8192), warmup=2, iters=5)
    mem_8k = get_peak_memory_mb()
    results["measurements"]["text_8192"] = {"latency_ms": round(t_8192, 2)}
    results["measurements"]["peak_memory_8k_mb"] = round(mem_8k, 1)
    print(f"     Result: {t_8192:.2f} ms (Peak Memory: {mem_8k:.1f} MB)")

    del text_embedder
    gc.collect()

    # 2. Multimodal Benchmarks
    print("\n[2/3] Loading multimodal model (text_vision)...")
    mm_embedder = EmbeddingGemmaBaseline(mode="text_vision", device=device)

    portrait_path = "tests/golden_vectors/portrait_test.png"
    baboon_path = "tests/golden_vectors/baboon.png"
    pepper_path = "tests/golden_vectors/pepper.png"

    img_portrait = Image.open(portrait_path).convert("RGB")
    img_baboon = Image.open(baboon_path).convert("RGB")
    img_pepper = Image.open(pepper_path).convert("RGB")

    print("  -> Benchmarking vision_140...")
    t_v140 = benchmark_fn(lambda: mm_embedder.encode_media(img_portrait))
    results["measurements"]["vision_140"] = {"latency_ms": round(t_v140, 2)}
    print(f"     Result: {t_v140:.2f} ms")

    print("  -> Benchmarking vision_280...")
    t_v280 = benchmark_fn(lambda: mm_embedder.encode_media(img_baboon))
    results["measurements"]["vision_280"] = {"latency_ms": round(t_v280, 2)}
    print(f"     Result: {t_v280:.2f} ms")

    print("  -> Benchmarking vision_560...")
    t_v560 = benchmark_fn(lambda: mm_embedder.encode_media(img_pepper))
    results["measurements"]["vision_560"] = {"latency_ms": round(t_v560, 2)}
    print(f"     Result: {t_v560:.2f} ms")

    # 3. Hybrid Benchmarks
    print("\n[3/3] Benchmarking hybrid interleaved sequences...")
    print("  -> Benchmarking hybrid_512...")
    t_h512 = benchmark_fn(lambda: mm_embedder.model.encode([{"text": "Card description: ", "image": img_portrait}]))
    results["measurements"]["hybrid_512"] = {"latency_ms": round(t_h512, 2)}
    print(f"     Result: {t_h512:.2f} ms")

    print("  -> Benchmarking hybrid_long (2K text + image)...")
    t_hlong = benchmark_fn(lambda: mm_embedder.model.encode([{"text": text_2048, "image": img_portrait}]), warmup=1, iters=3)
    results["measurements"]["hybrid_long"] = {"latency_ms": round(t_hlong, 2)}
    print(f"     Result: {t_hlong:.2f} ms")

    out_file = Path("target/bench_matrix_python.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Python benchmarks successfully saved to:", out_file)


if __name__ == "__main__":
    main()
