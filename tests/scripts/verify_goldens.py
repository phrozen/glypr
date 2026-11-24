#!/usr/bin/env python3
"""Validate integrity, shapes, finiteness, and norms of all golden vector fixtures."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open


GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden_vectors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-fp32",
        action="store_true",
        help="Enforce strict FP32 unit norm tolerance (1e-3 instead of 1e-2) and provenance dtype=float32",
    )
    return parser.parse_args()


def check_vector(name: str, vec: np.ndarray, expected_dim: int | None, norm_tol: float, errors: list[str]) -> None:
    if vec.ndim != 1:
        errors.append(f"{name}: expected 1D vector, got shape {vec.shape}")
        return
    if expected_dim is not None and vec.shape[0] != expected_dim:
        errors.append(f"{name}: expected dim {expected_dim}, got {vec.shape[0]}")
    if not np.all(np.isfinite(vec)):
        errors.append(f"{name}: contains NaN or Inf")
        return
    norm = float(np.linalg.norm(vec.astype(np.float64)))
    if abs(norm - 1.0) > norm_tol:
        errors.append(f"{name}: L2 norm {norm:.6f} deviates from 1.0 by > {norm_tol}")


def check_provenance(label: str, doc: dict, strict_fp32: bool, errors: list[str]) -> None:
    if strict_fp32:
        prov = doc.get("provenance")
        if not isinstance(prov, dict):
            errors.append(f"{label}: --strict-fp32 requires 'provenance' metadata")
        elif prov.get("dtype") != "float32":
            errors.append(f"{label}: expected provenance.dtype == 'float32', got {prov.get('dtype')!r}")


def verify_baseline(norm_tol: float, strict_fp32: bool, errors: list[str]) -> None:
    st_path = GOLDEN_DIR / "baseline_vectors.safetensors"
    manifest_path = GOLDEN_DIR / "manifest.json"
    if not st_path.exists() or not manifest_path.exists():
        errors.append("Missing baseline_vectors.safetensors or manifest.json")
        return

    manifest = json.loads(manifest_path.read_text())
    check_provenance("manifest.json", manifest, strict_fp32, errors)

    with safe_open(str(st_path), framework="numpy") as f:
        keys = list(f.keys())
        if len(keys) != 28:
            errors.append(f"baseline_vectors.safetensors: expected 28 tensors, found {len(keys)}")

        dims = manifest.get("supported_dims", [])
        cases = manifest.get("test_cases", {})
        for cid in cases:
            for dim in dims:
                key = f"{cid}_{dim}d"
                if key not in keys:
                    errors.append(f"baseline_vectors.safetensors: missing tensor {key}")
                    continue
                check_vector(f"baseline/{key}", f.get_tensor(key), dim, norm_tol, errors)

    print(f"[OK] baseline_vectors.safetensors + manifest.json ({len(keys)} tensors)")


def verify_corpus_8k(norm_tol: float, strict_fp32: bool, errors: list[str]) -> None:
    st_path = GOLDEN_DIR / "corpus_8k_vectors.safetensors"
    manifest_path = GOLDEN_DIR / "corpus_8k_manifest.json"
    if not st_path.exists() or not manifest_path.exists():
        errors.append("Missing corpus_8k_vectors.safetensors or corpus_8k_manifest.json")
        return

    manifest = json.loads(manifest_path.read_text())
    check_provenance("corpus_8k_manifest.json", manifest, strict_fp32, errors)

    with safe_open(str(st_path), framework="numpy") as f:
        keys = list(f.keys())
        if len(keys) != 292:
            errors.append(f"corpus_8k_vectors.safetensors: expected 292 tensors, found {len(keys)}")

        dims = manifest.get("dimensions", [])
        entries = manifest.get("entries", {})
        for eid in entries:
            for dim in dims:
                key = f"{eid}_{dim}d"
                if key not in keys:
                    errors.append(f"corpus_8k_vectors.safetensors: missing tensor {key}")
                    continue
                check_vector(f"corpus_8k/{key}", f.get_tensor(key), dim, norm_tol, errors)

    print(f"[OK] corpus_8k_vectors.safetensors + corpus_8k_manifest.json ({len(keys)} tensors)")


def verify_multimodal(norm_tol: float, strict_fp32: bool, errors: list[str]) -> None:
    mm_path = GOLDEN_DIR / "multimodal_golden.json"
    if not mm_path.exists():
        errors.append("Missing multimodal_golden.json")
        return

    data = json.loads(mm_path.read_text())
    check_provenance("multimodal_golden.json", data, strict_fp32, errors)

    cases = data.get("cases", {})
    if len(cases) != 24:
        errors.append(f"multimodal_golden.json: expected 24 cases, found {len(cases)}")

    for cid, cdata in cases.items():
        emb = np.asarray(cdata.get("embedding", []), dtype=np.float32)
        check_vector(f"multimodal/{cid}", emb, 768, norm_tol, errors)

    print(f"[OK] multimodal_golden.json ({len(cases)} cases)")


def verify_preprocessor(errors: list[str]) -> None:
    prep_path = GOLDEN_DIR / "preprocessor_golden.json"
    if not prep_path.exists():
        print("[SKIP] preprocessor_golden.json not present")
        return

    data = json.loads(prep_path.read_text())
    audio = data.get("audio", {})
    features = np.asarray(audio.get("features", []), dtype=np.float32)
    num_frames = audio.get("num_frames")
    num_mels = audio.get("num_mels")
    if audio.get("sample_rate") != 16000:
        errors.append(f"preprocessor/audio: expected sample_rate=16000, got {audio.get('sample_rate')}")
    if num_mels != 128:
        errors.append(f"preprocessor/audio: expected num_mels=128, got {num_mels}")
    if not isinstance(num_frames, int) or features.size != num_frames * 128:
        errors.append(f"preprocessor/audio: feature length {features.size} != {num_frames} * 128")
    if not np.all(np.isfinite(features)):
        errors.append("preprocessor/audio: features contain NaN or Inf")

    image = data.get("image", {})
    pixels = np.asarray(image.get("pixel_values", []), dtype=np.float32)
    pw = image.get("patch_width")
    ph = image.get("patch_height")
    tw = image.get("target_width")
    th = image.get("target_height")
    if not isinstance(pw, int) or not isinstance(ph, int) or tw != pw * 16 or th != ph * 16:
        errors.append(f"preprocessor/image: inconsistent patch grid ({pw}x{ph}) vs target ({tw}x{th})")
    elif pixels.size != pw * ph * 16 * 16 * 3:
        errors.append(f"preprocessor/image: pixel_values len {pixels.size} != {pw * ph * 768}")
    if not np.all(np.isfinite(pixels)):
        errors.append("preprocessor/image: pixel_values contain NaN or Inf")

    print(f"[OK] preprocessor_golden.json (audio={num_frames}x{num_mels}, image={tw}x{th})")


def main() -> int:
    args = parse_args()
    norm_tol = 1e-3 if args.strict_fp32 else 1e-2
    errors: list[str] = []

    verify_baseline(norm_tol, args.strict_fp32, errors)
    verify_corpus_8k(norm_tol, args.strict_fp32, errors)
    verify_multimodal(norm_tol, args.strict_fp32, errors)
    verify_preprocessor(errors)

    if errors:
        print("\nGolden verification FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"\nAll golden checks passed (norm_tol={norm_tol:.0e}, strict_fp32={args.strict_fp32}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
