"""Generate golden reference test vectors for EmbeddingGemma 2 to test the native Rust implementation."""

import json
import os
import wave
import struct
import math
from pathlib import Path
import sys

# Ensure tests directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file

from baseline.embedder import EmbeddingGemmaBaseline


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "golden_vectors"
OUTPUT_DIR.mkdir(exist_ok=True)

def create_synthetic_image(path: Path) -> Path:
    """Create a deterministic 64x64 RGB gradient image."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            arr[y, x, 0] = (x * 4) % 256
            arr[y, x, 1] = (y * 4) % 256
            arr[y, x, 2] = ((x + y) * 2) % 256
    img = Image.fromarray(arr, mode="RGB")
    img.save(path)
    print(f"Created synthetic test image: {path}")
    return path

def create_synthetic_audio(path: Path, duration_sec: float = 2.0, sample_rate: int = 16000) -> Path:
    """Create a deterministic 16 kHz mono sine wave WAV file."""
    num_samples = int(duration_sec * sample_rate)
    freq = 440.0  # A4 note
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)  # mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            value = int(math.sin(2.0 * math.pi * freq * (i / sample_rate)) * 16383.0)
            frames.extend(struct.pack("<h", value))
        wav.writeframes(frames)
    print(f"Created synthetic test audio: {path}")
    return path

def main():
    print("=" * 60)
    print("Generating Golden Reference Test Vectors for EmbeddingGemma 2")
    print("=" * 60)

    # 1. Initialize embedder
    embedder = EmbeddingGemmaBaseline(mode="text_only")

    # 2. Define deterministic test cases
    test_cases = [
        {
            "id": "search_query_1",
            "type": "query",
            "task": "SearchQuery",
            "text": "What causes the northern lights?",
        },
        {
            "id": "search_doc_1_titled",
            "type": "document",
            "title": "Aurora Borealis Explained",
            "text": "The northern lights are caused by charged solar wind particles colliding with Earth atmospheric gases.",
        },
        {
            "id": "search_doc_2_untitled",
            "type": "document",
            "title": None,
            "text": "Solar energetic particles interact with the magnetosphere near Earth magnetic poles.",
        },
        {
            "id": "code_query_1",
            "type": "query",
            "task": "CodeRetrieval",
            "text": "Fast inverse square root implementation in C",
        },
        {
            "id": "code_doc_1",
            "type": "document",
            "title": "q_rsqrt.c",
            "text": "float Q_rsqrt(float number) { long i; float x2, y; const float threehalfs = 1.5F; x2 = number * 0.5F; y = number; i = * ( long * ) &y; i = 0x5f3759df - ( i >> 1 ); y = * ( float * ) &i; y = y * ( threehalfs - ( x2 * y * y ) ); return y; }",
        },
        {
            "id": "sim_sentence_1",
            "type": "symmetric",
            "task": "SentenceSimilarity",
            "text": "The stock market dropped significantly on Friday.",
        },
        {
            "id": "sim_sentence_2",
            "type": "symmetric",
            "task": "SentenceSimilarity",
            "text": "Equities experienced steep declines at the end of the trading week.",
        },
    ]

    tensors_to_save = {}
    manifest = {
        "model_id": "google/embeddinggemma-2",
        "description": "Deterministic ground-truth reference vectors for Rust implementation validation",
        "supported_dims": [768, 512, 256, 128],
        "test_cases": {},
        "similarities": {},
    }

    # Generate text vectors across all MRL dimensions
    for case in test_cases:
        cid = case["id"]
        print(f"\nProcessing test case: {cid}")
        case_meta = dict(case)

        # 768d Full
        if case["type"] == "query":
            emb_768 = embedder.encode_query(case["text"], task=case["task"])
            emb_512 = embedder.encode_query(case["text"], task=case["task"], truncate_dim=512)
            emb_256 = embedder.encode_query(case["text"], task=case["task"], truncate_dim=256)
            emb_128 = embedder.encode_query(case["text"], task=case["task"], truncate_dim=128)
        elif case["type"] == "document":
            emb_768 = embedder.encode_document(case["text"], title=case.get("title"))
            emb_512 = embedder.encode_document(case["text"], title=case.get("title"), truncate_dim=512)
            emb_256 = embedder.encode_document(case["text"], title=case.get("title"), truncate_dim=256)
            emb_128 = embedder.encode_document(case["text"], title=case.get("title"), truncate_dim=128)
        else:
            emb_768 = embedder.encode_symmetric(case["text"], task=case["task"])
            emb_512 = embedder.encode_symmetric(case["text"], task=case["task"], truncate_dim=512)
            emb_256 = embedder.encode_symmetric(case["text"], task=case["task"], truncate_dim=256)
            emb_128 = embedder.encode_symmetric(case["text"], task=case["task"], truncate_dim=128)

        # Store tensors as float32 numpy arrays for cross-language compatibility
        tensors_to_save[f"{cid}_768d"] = np.ascontiguousarray(emb_768.astype(np.float32))
        tensors_to_save[f"{cid}_512d"] = np.ascontiguousarray(emb_512.astype(np.float32))
        tensors_to_save[f"{cid}_256d"] = np.ascontiguousarray(emb_256.astype(np.float32))
        tensors_to_save[f"{cid}_128d"] = np.ascontiguousarray(emb_128.astype(np.float32))

        # Verification of L2 norm
        norm_768 = float(np.linalg.norm(emb_768))
        norm_512 = float(np.linalg.norm(emb_512))
        norm_256 = float(np.linalg.norm(emb_256))
        norm_128 = float(np.linalg.norm(emb_128))

        case_meta["l2_norms"] = {
            "768d": norm_768,
            "512d": norm_512,
            "256d": norm_256,
            "128d": norm_128,
        }
        manifest["test_cases"][cid] = case_meta
        print(f"  Norms: 768d={norm_768:.6f}, 512d={norm_512:.6f}, 256d={norm_256:.6f}, 128d={norm_128:.6f}")

    # Compute reference similarities for regression assertions
    manifest["similarities"]["search_query_1_vs_doc_1_titled_768d"] = float(
        np.dot(tensors_to_save["search_query_1_768d"], tensors_to_save["search_doc_1_titled_768d"])
    )
    manifest["similarities"]["search_query_1_vs_doc_1_titled_512d"] = float(
        np.dot(tensors_to_save["search_query_1_512d"], tensors_to_save["search_doc_1_titled_512d"])
    )
    manifest["similarities"]["code_query_1_vs_code_doc_1_768d"] = float(
        np.dot(tensors_to_save["code_query_1_768d"], tensors_to_save["code_doc_1_768d"])
    )
    manifest["similarities"]["sim_sentence_1_vs_2_768d"] = float(
        np.dot(tensors_to_save["sim_sentence_1_768d"], tensors_to_save["sim_sentence_2_768d"])
    )

    # 3. Create synthetic multimodal inputs
    img_path = create_synthetic_image(OUTPUT_DIR / "test_gradient.png")
    audio_path = create_synthetic_audio(OUTPUT_DIR / "test_tone_440hz.wav")
    manifest["multimodal_files"] = {
        "image": str(img_path.name),
        "audio": str(audio_path.name),
    }

    # Save safetensors
    weights_path = OUTPUT_DIR / "baseline_vectors.safetensors"
    save_file(tensors_to_save, str(weights_path))
    print(f"\nSaved golden vectors safetensors: {weights_path}")

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved golden manifest: {manifest_path}")

    print("\n✓ Golden reference dataset generation complete!")

if __name__ == "__main__":
    main()
