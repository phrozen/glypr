#!/usr/bin/env python3
"""Generate deterministic audio & image preprocessor golden vectors from HuggingFace AutoProcessor."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import transformers
from transformers import AutoProcessor


MODEL_ID = "google/embeddinggemma-2"
GOLDEN_PATH = Path(__file__).resolve().parent.parent / "golden_vectors" / "preprocessor_golden.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load processor using local_files_only=True",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GOLDEN_PATH,
        help=f"Output JSON path (default: {GOLDEN_PATH})",
    )
    return parser.parse_args()


def _to_numpy(val: Any) -> np.ndarray:
    if hasattr(val, "detach"):
        return val.detach().cpu().numpy()
    return np.asarray(val)


def extract_audio_golden(processor: Any) -> dict[str, Any]:
    sample_rate = 16000
    freq_hz = 440.0
    num_samples = 1600
    samples = np.array(
        [math.sin(2.0 * math.pi * freq_hz * (i / sample_rate)) for i in range(num_samples)],
        dtype=np.float32,
    )

    fe = getattr(processor, "feature_extractor", None) or processor
    if hasattr(fe, "_extract_spectrogram"):
        feats, mask = fe._extract_spectrogram(samples, np.ones(len(samples), dtype=np.int32))
        feats = np.asarray(feats, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    else:
        try:
            aud_out = fe(
                [samples],
                sampling_rate=sample_rate,
                pad_to_multiple_of=None,
                return_tensors="np",
            )
        except TypeError:
            aud_out = fe([samples], sampling_rate=sample_rate, return_tensors="np")

        feat_key = "input_features" if "input_features" in aud_out else next(
            k for k in aud_out.keys() if "feature" in k and "mask" not in k
        )
        feats = _to_numpy(aud_out[feat_key]).astype(np.float32)

        if feats.ndim == 3 and feats.shape[0] == 1:
            feats = feats[0]
        elif feats.ndim == 1:
            raise ValueError(f"Unexpected 1D audio feature shape: {feats.shape}")

        if "input_features_mask" in aud_out:
            mask = _to_numpy(aud_out["input_features_mask"])
            if mask.ndim == 2 and mask.shape[0] == 1:
                mask = mask[0]
            mask = mask.astype(bool)
            if feats.ndim == 2 and feats.shape[0] == mask.shape[0]:
                feats = feats[mask]
            elif feats.ndim == 2 and feats.shape[1] == mask.shape[0]:
                feats = feats[:, mask]

    if feats.ndim == 2 and feats.shape[1] != 128 and feats.shape[0] == 128:
        feats = feats.T

    if feats.ndim != 2 or feats.shape[1] != 128:
        raise ValueError(f"Expected audio features shape (num_frames, 128), got {feats.shape}")

    num_frames, num_mels = int(feats.shape[0]), int(feats.shape[1])
    return {
        "sample_rate": sample_rate,
        "freq_hz": freq_hz,
        "num_samples": num_samples,
        "num_frames": num_frames,
        "num_mels": num_mels,
        "features": np.ascontiguousarray(feats, dtype=np.float32).reshape(-1).tolist(),
    }


def _patchify_chw(chw: np.ndarray, patch_size: int) -> tuple[np.ndarray, int, int]:
    c, h, w = chw.shape
    ph, pw = h // patch_size, w // patch_size
    patches = chw.reshape(c, ph, patch_size, pw, patch_size).transpose(1, 3, 2, 4, 0)
    return patches.reshape(ph * pw, -1), pw, ph


def extract_image_golden(processor: Any) -> dict[str, Any]:
    orig_width = 32
    orig_height = 48
    budget = 70

    arr = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
    for y in range(orig_height):
        for x in range(orig_width):
            arr[y, x, 0] = (x * 7) % 256
            arr[y, x, 1] = (y * 5) % 256
            arr[y, x, 2] = ((x + y) * 3) % 256
    img = Image.fromarray(arr, mode="RGB")

    img_proc = getattr(processor, "image_processor", None) or processor
    patch_size = int(getattr(img_proc, "patch_size", 16))
    orig_budget = getattr(img_proc, "max_soft_tokens", None)
    try:
        try:
            img_out = img_proc(img, max_soft_tokens=budget, return_tensors="np")
        except TypeError:
            if hasattr(img_proc, "max_soft_tokens"):
                img_proc.max_soft_tokens = budget
            img_out = img_proc(img, return_tensors="np")
    finally:
        if orig_budget is not None and hasattr(img_proc, "max_soft_tokens"):
            img_proc.max_soft_tokens = orig_budget

    pv = _to_numpy(img_out["pixel_values"]).astype(np.float32)
    while pv.ndim >= 3 and pv.shape[0] == 1:
        pv = pv[0]

    pos_key = next((k for k in ("image_position_ids", "pixel_position_ids") if k in img_out), None)
    if pos_key is not None and pv.ndim == 2:
        pos = _to_numpy(img_out[pos_key])
        while pos.ndim >= 3 and pos.shape[0] == 1:
            pos = pos[0]
        valid_mask = (pos[:, 0] >= 0) & (pos[:, 1] >= 0)
        pv = pv[valid_mask]
        valid_pos = pos[valid_mask]
        pw = int(valid_pos[:, 0].max()) + 1
        ph = int(valid_pos[:, 1].max()) + 1
    elif pv.ndim == 3 and pv.shape[0] == 3:
        pv, pw, ph = _patchify_chw(pv, patch_size)
    elif pv.ndim == 3 and pv.shape[2] == 3:
        pv, pw, ph = _patchify_chw(pv.transpose(2, 0, 1), patch_size)
    else:
        raise ValueError(f"Unhandled pixel_values shape from HF image_processor: {pv.shape}")

    # Gemma4ImageProcessor outputs [0, 1] and scales 2 * (px - 0.5) inside the model;
    # Rust preprocess_image emits normalized [-1.0, 1.0] directly.
    if not getattr(img_proc, "do_normalize", False):
        pv = 2.0 * (pv - 0.5)

    tw = pw * patch_size
    th = ph * patch_size
    return {
        "orig_width": orig_width,
        "orig_height": orig_height,
        "budget": budget,
        "target_width": tw,
        "target_height": th,
        "patch_width": pw,
        "patch_height": ph,
        "pixel_values": [
            round(float(x), 6) for x in np.ascontiguousarray(pv, dtype=np.float32).reshape(-1)
        ],
    }


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=args.offline)

    audio_data = extract_audio_golden(processor)
    image_data = extract_image_golden(processor)

    payload = {
        "schema_version": 2,
        "provenance": {
            "model_id": MODEL_ID,
            "transformers_version": transformers.__version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "audio": audio_data,
        "image": image_data,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = args.output.stat().st_size / 1024.0
    print(
        f"Saved {args.output} ({size_kb:.1f} KB): "
        f"audio=[{audio_data['num_frames']}x{audio_data['num_mels']}], "
        f"image={image_data['target_width']}x{image_data['target_height']} "
        f"({image_data['patch_width']}x{image_data['patch_height']} patches)"
    )


if __name__ == "__main__":
    main()
