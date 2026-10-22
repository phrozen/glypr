#!/usr/bin/env python3
"""Aggregates Python and Rust Metal batch benchmark results into a consolidated matrix."""

import json
from pathlib import Path


def main():
    py_path = Path("target/bench_batch_python.json")
    rust_path = Path("target/bench_batch_rust.json")

    if not py_path.exists():
        print(f"Error: {py_path} not found.")
        return
    if not rust_path.exists():
        print(f"Error: {rust_path} not found.")
        return

    with open(py_path, "r", encoding="utf-8") as f:
        py_data = json.load(f)
    with open(rust_path, "r", encoding="utf-8") as f:
        rust_data = json.load(f)

    py_meas = py_data["measurements"]
    rust_meas = rust_data["measurements"]

    workloads = [
        # (Key, Category, Workload name, Target Input / Details)
        ("text_512_b8", "Text Batch (B=8)", "**Text @ 512 Tokens**", "8 × 512t passages (4,096 total tokens)"),
        ("text_2048_b8", "Text Batch (B=8)", "**Text @ 2,048 Tokens**", "8 × 2Kt documents (16,384 total tokens)"),
        ("text_4096_b8", "Text Batch (B=8)", "**Text @ 4,096 Tokens**", "8 × 4Kt documents (32,768 total tokens)"),
        ("text_8192_b8", "Text Batch (B=8)", "**Text @ 8,192 Tokens**", "8 × 8Kt full contexts (65,536 total tokens)"),
        ("vision_70_b8", "Vision Batch (B=8)", "**Vision @ 70 Budget**", "8 × compact images (4,536 total patches)"),
        ("vision_140_b8", "Vision Batch (B=8)", "**Vision @ 140 Budget**", "8 × portrait docs (9,072 total patches)"),
        ("vision_280_b8", "Vision Batch (B=8)", "**Vision @ 280 Budget**", "8 × standard images (19,152 total patches)"),
        ("vision_560_b8", "Vision Batch (B=8)", "**Vision @ 560 Budget**", "8 × high-res images (38,304 total patches)"),
        ("hybrid_text_vision_b8", "Hybrid Multimodal (B=8)", "**Hybrid: Text + Vision**", "8 × [Query text + 140b image]"),
        ("hybrid_text_vision_audio_b8", "Hybrid Multimodal (B=8)", "**Hybrid: Text + Vision + Audio**", "8 × [Text + 140b image + 1.0s audio]"),
    ]

    matrix = {
        "batch_size": 8,
        "results": []
    }

    print("\n" + "=" * 90)
    print("  GLYPR BATCH PERFORMANCE MATRIX (Batch Size = 8)")
    print("=" * 90)

    print("| Workload / Modality | Batch Size & Input | Python Reference (PyTorch MPS) | Glypr (Metal 4 GPU) | Throughput (Glypr) | Speedup vs. Python |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: |")

    for key, category, name, details in workloads:
        py_item = py_meas.get(key, {})
        rust_item = rust_meas.get(key, {})

        py_ms = py_item.get("latency_ms", 0.0)
        rust_ms = rust_item.get("latency_ms", 0.0)
        rust_per_item = rust_item.get("per_item_ms", 0.0)
        rust_tput = rust_item.get("throughput_items_per_sec", 0.0)

        speedup_str = "—"
        if py_ms > 0 and rust_ms > 0:
            speedup = py_ms / rust_ms
            speedup_str = f"**{speedup:.2f}× faster**"

        row = {
            "key": key,
            "category": category,
            "name": name,
            "details": details,
            "python_mps_ms": py_ms,
            "glypr_metal_ms": rust_ms,
            "glypr_per_item_ms": rust_per_item,
            "glypr_throughput": rust_tput,
            "speedup_vs_python": round(py_ms / rust_ms, 2) if (py_ms > 0 and rust_ms > 0) else None,
        }
        matrix["results"].append(row)

        print(f"| {name} | {details} | {py_ms:.2f} ms | **{rust_ms:.2f} ms** ({rust_per_item:.1f} ms/item) | **{rust_tput:.1f} items/s** | {speedup_str} |")

    out_file = Path("tests/golden_vectors/benchmark_batch.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

    print("\n✓ Consolidated batch benchmark matrix saved to:", out_file)


if __name__ == "__main__":
    main()
