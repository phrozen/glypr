#!/usr/bin/env python3
"""Generate Golden Reference vectors for the expanded multi-domain, multilingual corpus."""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import safetensors.numpy
import torch

import sys
# Ensure tests directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.embedder import EmbeddingGemmaBaseline

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden_vectors"

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def run():
    corpus_path = GOLDEN_DIR / "corpus_8k.json"
    if not corpus_path.exists():
        print(f"Error: {corpus_path} not found. Run generate_corpus.py first.")
        sys.exit(1)

    with open(corpus_path) as f:
        corpus_data = json.load(f)

    entries = corpus_data["entries"]
    print(f"Loaded {len(entries)} corpus entries from {corpus_path}")

    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    print(f"Initializing EmbeddingGemmaBaseline on {device} (bfloat16)...")
    t0 = time.perf_counter()
    embedder = EmbeddingGemmaBaseline(mode="text_only", device=device, torch_dtype=torch.bfloat16)
    print(f"Loaded baseline in {time.perf_counter() - t0:.2f} s")

    dimensions = [768, 512, 256, 128]
    tensors_to_save = {}
    manifest_entries = {}

    print(f"\nEncoding {len(entries)} corpus entries across MRL dimensions [768, 512, 256, 128]...")
    t_start_all = time.perf_counter()

    for idx, entry in enumerate(entries):
        eid = entry["id"]
        etype = entry["type"]
        cat = entry.get("category", "unknown")
        bucket = entry.get("bucket_tokens", 0)

        # Single forward pass at full 768d
        if etype == "document":
            vec_768 = embedder.encode_document(entry["content"], title=entry.get("title"), truncate_dim=768)
        else:
            task_name = entry.get("task", "SearchQuery")
            vec_768 = embedder.encode_query(entry["content"], task=task_name, truncate_dim=768)

        vec_768 = np.ascontiguousarray(vec_768.flatten(), dtype=np.float32)

        entry_mrl_norms = {}
        for dim in dimensions:
            if dim == 768:
                vec_dim = vec_768
            else:
                v_slice = vec_768[:dim]
                norm = np.linalg.norm(v_slice)
                vec_dim = v_slice / norm if norm > 0 else v_slice

            tensor_key = f"{eid}_{dim}d"
            tensors_to_save[tensor_key] = vec_dim
            entry_mrl_norms[f"{dim}d"] = float(np.linalg.norm(vec_dim))

        manifest_entries[eid] = {
            "id": eid,
            "category": cat,
            "type": etype,
            "language": entry.get("language"),
            "bucket_tokens": bucket,
            "actual_tokens": entry.get("actual_tokens"),
            "domain": entry.get("domain"),
            "l2_norms": entry_mrl_norms,
            "target_doc_id": entry.get("target_doc_id"),
            "relevant_query_id": entry.get("relevant_query_id"),
        }

        if (idx + 1) % 10 == 0 or idx + 1 == len(entries):
            print(f"  Processed [{idx+1:2d}/{len(entries)}] entries ({eid})...")

    print(f"All {len(entries)} entries encoded in {time.perf_counter() - t_start_all:.2f} s")

    # Pairwise verification 1: Length scaling buckets
    print("\nPairwise Evaluations - Length Scaling Buckets:")
    pairwise_similarities = {}
    for bucket in corpus_data["buckets"]:
        q_id = f"query_physics_{bucket}t"
        pos_doc_id = f"doc_physics_{bucket}t"
        neg_doc_id = f"doc_science_{bucket}t"

        for dim in [768, 512]:
            q_vec = tensors_to_save[f"{q_id}_{dim}d"]
            pos_vec = tensors_to_save[f"{pos_doc_id}_{dim}d"]
            neg_vec = tensors_to_save[f"{neg_doc_id}_{dim}d"]

            pos_sim = cosine_similarity(q_vec, pos_vec)
            neg_sim = cosine_similarity(q_vec, neg_vec)
            margin = pos_sim - neg_sim

            pair_key = f"physics_{bucket}t_{dim}d"
            pairwise_similarities[pair_key] = {
                "query": q_id,
                "positive_doc": pos_doc_id,
                "negative_doc": neg_doc_id,
                "positive_sim": pos_sim,
                "negative_sim": neg_sim,
                "margin": margin,
            }
            print(f"  Bucket {bucket:4d}t ({dim}d): Pos = {pos_sim:.4f} | Neg = {neg_sim:.4f} | Margin = {margin:+.4f}")

    # Pairwise verification 2: Multilingual 10 Languages
    print("\nPairwise Evaluations - Multilingual 10 Languages (768d):")
    languages = ["en", "es", "de", "fr", "ja", "zh", "ko", "ru", "el", "ar"]
    for lang in languages:
        q_id = f"query_lang_{lang}"
        pos_id = f"doc_lang_{lang}"
        neg_id = f"distractor_lang_{lang}"

        if f"{q_id}_768d" in tensors_to_save and f"{pos_id}_768d" in tensors_to_save:
            q_vec = tensors_to_save[f"{q_id}_768d"]
            pos_vec = tensors_to_save[f"{pos_id}_768d"]
            neg_vec = tensors_to_save[f"{neg_id}_768d"]

            pos_sim = cosine_similarity(q_vec, pos_vec)
            neg_sim = cosine_similarity(q_vec, neg_vec)
            margin = pos_sim - neg_sim

            pair_key = f"multilingual_{lang}_768d"
            pairwise_similarities[pair_key] = {
                "language": lang,
                "query": q_id,
                "positive_doc": pos_id,
                "negative_doc": neg_id,
                "positive_sim": pos_sim,
                "negative_sim": neg_sim,
                "margin": margin,
            }
            print(f"  Lang [{lang:2s}]: Pos = {pos_sim:.4f} | Neg = {neg_sim:.4f} | Margin = {margin:+.4f}")

    # Save safetensors
    weights_path = GOLDEN_DIR / "corpus_8k_vectors.safetensors"
    safetensors.numpy.save_file(tensors_to_save, str(weights_path))
    print(f"\n✓ Saved {len(tensors_to_save)} golden tensors to {weights_path}")

    # Save manifest
    manifest_path = GOLDEN_DIR / "corpus_8k_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "model_id": "google/embeddinggemma-2",
            "description": "Expanded golden reference oracle embeddings (8k scaling, 10 languages, task prompts)",
            "dimensions": dimensions,
            "entries": manifest_entries,
            "pairwise_evaluations": pairwise_similarities,
        }, f, indent=2)
    print(f"✓ Saved manifest metadata to {manifest_path}")

if __name__ == "__main__":
    run()
