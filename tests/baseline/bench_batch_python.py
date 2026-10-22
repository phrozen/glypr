"""Automated Python PyTorch MPS batch benchmark harness for Glypr."""

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.baseline.embedder import EmbeddingGemmaBaseline


def get_peak_memory_mb() -> float:
    """Return total physical unified memory (Metal allocations + CPU RSS) in MB."""
    mps_mem = 0.0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_mem = torch.mps.current_allocated_memory() / (1024 * 1024)
    rss_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    return mps_mem + rss_mem


def benchmark_batch_fn(fn, warmup: int = 2, iters: int = 5) -> float:
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
    print("=" * 65)
    print("  PYTHON PYTORCH MPS BATCH BENCHMARK (Batch Size = 8)")
    print("=" * 65)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    results = {"device": device, "batch_size": 8, "measurements": {}}

    # Load corpus entries
    corpus_path = Path("tests/golden_vectors/corpus_8k.json")
    if not corpus_path.exists():
        corpus_path = Path("../../tests/golden_vectors/corpus_8k.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    corpus_map = {e["id"]: e["content"] for e in corpus["entries"]}
    text_512 = corpus_map.get("doc_physics_512t", "quantum " * 358)
    text_2048 = corpus_map.get("doc_physics_2048t", "physics " * 1421)
    text_4096 = corpus_map.get("doc_physics_4096t", "thermodynamics " * 2800)
    text_8192 = corpus_map.get("doc_physics_8192t", "general relativity " * 5600)

    # -------------------------------------------------------------------------
    # 1. Text Batch Benchmarks (B = 8)
    # -------------------------------------------------------------------------
    print("\n[1/3] Benchmarking Batched Text (B = 8)...")
    text_embedder = EmbeddingGemmaBaseline(mode="text_only", device=device)

    print("  -> text_512_b8...")
    t_512 = benchmark_batch_fn(lambda: text_embedder.model.encode([text_512] * 8, batch_size=8))
    results["measurements"]["text_512_b8"] = {"latency_ms": round(t_512, 2)}
    print(f"     Result: {t_512:.2f} ms")

    print("  -> text_2048_b8...")
    t_2k = benchmark_batch_fn(lambda: text_embedder.model.encode([text_2048] * 8, batch_size=8))
    results["measurements"]["text_2048_b8"] = {"latency_ms": round(t_2k, 2)}
    print(f"     Result: {t_2k:.2f} ms")

    print("  -> text_4096_b8...")
    t_4k = benchmark_batch_fn(lambda: text_embedder.model.encode([text_4096] * 8, batch_size=8), warmup=1, iters=3)
    results["measurements"]["text_4096_b8"] = {"latency_ms": round(t_4k, 2)}
    print(f"     Result: {t_4k:.2f} ms")

    print("  -> text_8192_b8...")
    t_8k = benchmark_batch_fn(lambda: text_embedder.model.encode([text_8192] * 8, batch_size=8), warmup=1, iters=3)
    results["measurements"]["text_8192_b8"] = {"latency_ms": round(t_8k, 2)}
    print(f"     Result: {t_8k:.2f} ms")

    del text_embedder
    gc.collect()

    # -------------------------------------------------------------------------
    # 2. Images Batch Across 4 Budgets (B = 8)
    # -------------------------------------------------------------------------
    print("\n[2/3] Benchmarking Batched Vision Across 4 Budgets (B = 8)...")
    mm_embedder = EmbeddingGemmaBaseline(mode="text_vision", device=device)

    img_portrait = Image.open("tests/golden_vectors/portrait_test.png").convert("RGB")
    img_baboon = Image.open("tests/golden_vectors/baboon.png").convert("RGB")
    img_pepper = Image.open("tests/golden_vectors/pepper.png").convert("RGB")

    print("  -> vision_70_b8 (clamped in HF)...")
    t_v70 = benchmark_batch_fn(lambda: mm_embedder.model.encode([{"image": img_portrait}] * 8, batch_size=8))
    results["measurements"]["vision_70_b8"] = {"latency_ms": round(t_v70, 2)}
    print(f"     Result: {t_v70:.2f} ms")

    print("  -> vision_140_b8...")
    t_v140 = benchmark_batch_fn(lambda: mm_embedder.model.encode([{"image": img_portrait}] * 8, batch_size=8))
    results["measurements"]["vision_140_b8"] = {"latency_ms": round(t_v140, 2)}
    print(f"     Result: {t_v140:.2f} ms")

    print("  -> vision_280_b8...")
    t_v280 = benchmark_batch_fn(lambda: mm_embedder.model.encode([{"image": img_baboon}] * 8, batch_size=8))
    results["measurements"]["vision_280_b8"] = {"latency_ms": round(t_v280, 2)}
    print(f"     Result: {t_v280:.2f} ms")

    print("  -> vision_560_b8...")
    t_v560 = benchmark_batch_fn(lambda: mm_embedder.model.encode([{"image": img_pepper}] * 8, batch_size=8))
    results["measurements"]["vision_560_b8"] = {"latency_ms": round(t_v560, 2)}
    print(f"     Result: {t_v560:.2f} ms")

    del mm_embedder
    gc.collect()

    # -------------------------------------------------------------------------
    # 3. Hybrid Multimodal Batch (B = 8)
    # -------------------------------------------------------------------------
    print("\n[3/3] Benchmarking Batched Hybrid Multimodal (B = 8)...")
    full_embedder = EmbeddingGemmaBaseline(mode="full", device=device)

    # 1.0s audio waveform (16000 samples)
    audio_wav = np.zeros(16000, dtype=np.float32)

    print("  -> hybrid_text_vision_b8...")
    hybrid_tv = [{"text": "A photograph of an artifact in a museum: ", "image": img_portrait}] * 8
    t_tv = benchmark_batch_fn(lambda: full_embedder.model.encode(hybrid_tv, batch_size=8))
    results["measurements"]["hybrid_text_vision_b8"] = {"latency_ms": round(t_tv, 2)}
    print(f"     Result: {t_tv:.2f} ms")

    print("  -> hybrid_text_vision_audio_b8...")
    hybrid_tva = [{"text": "A video clip with background music: ", "image": img_portrait, "audio": audio_wav}] * 8
    t_tva = benchmark_batch_fn(lambda: full_embedder.model.encode(hybrid_tva, batch_size=8))
    results["measurements"]["hybrid_text_vision_audio_b8"] = {"latency_ms": round(t_tva, 2)}
    print(f"     Result: {t_tva:.2f} ms")

    out_file = Path("target/bench_batch_python.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Python batch benchmarks saved to:", out_file)


if __name__ == "__main__":
    main()
