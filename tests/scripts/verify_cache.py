#!/usr/bin/env python3
"""Verify local Hugging Face cache for google/embeddinggemma-2."""

import os
import sys
from pathlib import Path
import json

from huggingface_hub import constants

def verify_cache():
    cache_dir = os.environ.get("HF_HUB_CACHE", constants.HF_HUB_CACHE)
    model_dir = Path(cache_dir) / "models--google--embeddinggemma-2"
    
    print(f"Checking cache directory: {model_dir}")
    if not model_dir.exists():
        print(f"ERROR: Model directory does not exist: {model_dir}")
        sys.exit(1)
        
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        print(f"ERROR: Snapshots directory does not exist: {snapshots_dir}")
        sys.exit(1)
        
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        print("ERROR: No snapshots found.")
        sys.exit(1)
        
    snapshot = snapshots[0]
    print(f"Found snapshot: {snapshot.name}")
    
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "modules.json",
        "config_sentence_transformers.json",
    ]
    
    missing = []
    for req in required_files:
        p = snapshot / req
        if not p.exists():
            missing.append(req)
        else:
            size_mb = p.resolve().stat().st_size / (1024 * 1024)
            print(f"  ✓ {req} ({size_mb:.2f} MB)")
            
    if missing:
        print(f"ERROR: Missing required files: {missing}")
        sys.exit(1)
        
    # Read config
    with open(snapshot / "config.json") as f:
        config = json.load(f)
        print("\nModel Architecture Summary from config.json:")
        print(f"  Model Type: {config.get('model_type')}")
        print(f"  Architectures: {config.get('architectures')}")
        print(f"  Vocab Size: {config.get('vocab_size')}")
        print(f"  Hidden Size (d_model): {config.get('hidden_size')}")
        print(f"  Intermediate Size: {config.get('intermediate_size')}")
        print(f"  Num Hidden Layers: {config.get('num_hidden_layers')}")
        print(f"  Num Attention Heads: {config.get('num_attention_heads')}")
        print(f"  Num Key-Value Heads: {config.get('num_key_value_heads')}")
        print(f"  Sliding Window: {config.get('sliding_window')}")
        print(f"  Max Position Embeddings: {config.get('max_position_embeddings')}")
        print(f"  Torch Dtype: {config.get('torch_dtype')}")
        
    print("\n✓ Local cache is intact and ready for baseline execution.")

if __name__ == "__main__":
    verify_cache()
