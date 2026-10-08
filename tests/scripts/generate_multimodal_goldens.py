#!/usr/bin/env python3
"""
Generate multimodal golden reference vectors for EmbeddingGemma 2.
Produces:
  1. golden_vectors/multimodal_test_image.png (Synthetic deterministic image)
  2. golden_vectors/multimodal_test_audio.wav (Synthetic deterministic 16 kHz audio)
  3. golden_vectors/lenna.png & golden_vectors/lena_16k.wav (Real-world assets)
  4. golden_vectors/multimodal_golden.json (Reference embeddings & metadata)
"""

import json
import wave
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import signal
from sentence_transformers import SentenceTransformer

def ensure_resampled_audio(src_wav: Path, dst_wav: Path, target_sr: int = 16000) -> np.ndarray:
    """Ensure audio is resampled to target_sr using polyphase filtering."""
    if not src_wav.exists() and dst_wav.exists():
        with wave.open(str(dst_wav), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            return pcm.astype(np.float32) / 32767.0

    with wave.open(str(src_wav), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        raw = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        if ch > 1:
            raw = raw.reshape(-1, ch).mean(axis=1)

    if sr != target_sr:
        print(f"Resampling {src_wav.name} ({sr} Hz) -> {dst_wav.name} ({target_sr} Hz)...")
        # Polyphase rational resample: gcd(target_sr, sr)
        from math import gcd
        g = gcd(target_sr, sr)
        up = target_sr // g
        down = sr // g
        resampled = signal.resample_poly(raw, up, down).astype(np.float32)
    else:
        resampled = raw

    with wave.open(str(dst_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        pcm = np.clip(resampled * 32767.0, -32768.0, 32767.0).astype(np.int16)
        wf.writeframes(pcm.tobytes())

    with wave.open(str(dst_wav), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return pcm.astype(np.float32) / 32767.0

def main():
    out_dir = Path(__file__).resolve().parent.parent / "golden_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SentenceTransformer('google/embeddinggemma-2') on CPU...")
    model = SentenceTransformer("google/embeddinggemma-2", device="cpu")

    # 1. Deterministic synthetic test image (224x224 RGB)
    np.random.seed(1337)
    img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        for x in range(224):
            img_arr[y, x, 0] = int((x / 224.0) * 255)
            img_arr[y, x, 1] = int((y / 224.0) * 255)
            img_arr[y, x, 2] = int(((x + y) / 448.0) * 255)

    synth_img = Image.fromarray(img_arr, mode="RGB")
    synth_img_path = out_dir / "multimodal_test_image.png"
    synth_img.save(synth_img_path)
    print(f"Saved synthetic test image to {synth_img_path}")

    # 2. Deterministic synthetic test audio (1 second, 16000 Hz)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    synth_audio = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.25 * np.sin(2 * np.pi * 880.0 * t)
    synth_audio *= np.sin(np.pi * t / duration)

    synth_audio_path = out_dir / "multimodal_test_audio.wav"
    with wave.open(str(synth_audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm_data = (synth_audio * 32767.0).astype(np.int16).tobytes()
        wf.writeframes(pcm_data)
    print(f"Saved synthetic test audio to {synth_audio_path}")

    # Read back exact PCM audio as float32 array, ensuring golden embeddings match the saved wav artifact
    with wave.open(str(synth_audio_path), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(int(sr * duration)), dtype=np.int16)
        synth_audio = pcm.astype(np.float32) / 32767.0

    # 3. Real-world assets (Lenna image & Lena audio)
    real_img_path = out_dir / "lenna.png"
    if not real_img_path.exists():
        raise FileNotFoundError(f"Missing {real_img_path}")
    real_img = Image.open(real_img_path).convert("RGB")
    print(f"Loaded real-world image: {real_img_path} ({real_img.size}, mode={real_img.mode})")

    real_audio_src = out_dir / "lena.wav"
    real_audio_dst = out_dir / "lena_16k.wav"
    real_audio = ensure_resampled_audio(real_audio_src, real_audio_dst, target_sr=16000)
    print(f"Loaded real-world audio: {real_audio_dst} ({len(real_audio)} samples, {len(real_audio)/16000:.2f}s)")

    # 4. Generate Reference Embeddings
    print("\nEncoding multimodal inputs via SentenceTransformer:")

    cases = {}

    # Synthetic Case 1: Image only
    emb_synth_img_only = model.encode([{"role": "user", "content": [{"type": "image", "image": synth_img}]}], device="cpu").flatten()
    print(f"  [1] Synthetic Image Only: shape={emb_synth_img_only.shape}, norm={np.linalg.norm(emb_synth_img_only):.6f}")
    cases["image_only"] = {
        "prompt": None,
        "embedding": [float(x) for x in emb_synth_img_only],
    }

    # Synthetic Case 2: Image + Prompt
    prompt_img = "Describe this picture"
    emb_synth_img_prompt = model.encode([{"role": "user", "content": [
        {"type": "image", "image": synth_img},
        {"type": "text", "text": prompt_img},
    ]}], device="cpu").flatten()
    print(f"  [2] Synthetic Image + Prompt: shape={emb_synth_img_prompt.shape}, norm={np.linalg.norm(emb_synth_img_prompt):.6f}")
    cases["image_prompt"] = {
        "prompt": prompt_img,
        "embedding": [float(x) for x in emb_synth_img_prompt],
    }

    # Synthetic Case 3: Audio only
    emb_synth_aud_only = model.encode([{"role": "user", "content": [{"type": "audio", "audio": synth_audio}]}], device="cpu").flatten()
    print(f"  [3] Synthetic Audio Only: shape={emb_synth_aud_only.shape}, norm={np.linalg.norm(emb_synth_aud_only):.6f}")
    cases["audio_only"] = {
        "prompt": None,
        "embedding": [float(x) for x in emb_synth_aud_only],
    }

    # Synthetic Case 4: Audio + Prompt
    prompt_aud = "A musical sound"
    emb_synth_aud_prompt = model.encode([{"role": "user", "content": [
        {"type": "audio", "audio": synth_audio},
        {"type": "text", "text": prompt_aud},
    ]}], device="cpu").flatten()
    print(f"  [4] Synthetic Audio + Prompt: shape={emb_synth_aud_prompt.shape}, norm={np.linalg.norm(emb_synth_aud_prompt):.6f}")
    cases["audio_prompt"] = {
        "prompt": prompt_aud,
        "embedding": [float(x) for x in emb_synth_aud_prompt],
    }

    # Real-World Case 5: Real Image (Lenna) Only
    emb_real_img_only = model.encode([{"role": "user", "content": [{"type": "image", "image": real_img}]}], device="cpu").flatten()
    print(f"  [5] Real Image (Lenna) Only: shape={emb_real_img_only.shape}, norm={np.linalg.norm(emb_real_img_only):.6f}")
    cases["real_image_only"] = {
        "prompt": None,
        "embedding": [float(x) for x in emb_real_img_only],
    }

    # Real-World Case 6: Real Image + Prompt
    prompt_real_img = "A classic portrait photograph of a woman wearing a hat"
    emb_real_img_prompt = model.encode([{"role": "user", "content": [
        {"type": "image", "image": real_img},
        {"type": "text", "text": prompt_real_img},
    ]}], device="cpu").flatten()
    print(f"  [6] Real Image + Prompt: shape={emb_real_img_prompt.shape}, norm={np.linalg.norm(emb_real_img_prompt):.6f}")
    cases["real_image_prompt"] = {
        "prompt": prompt_real_img,
        "embedding": [float(x) for x in emb_real_img_prompt],
    }

    # Real-World Case 7: Real Audio (Lena speech) Only
    emb_real_aud_only = model.encode([{"role": "user", "content": [{"type": "audio", "audio": real_audio}]}], device="cpu").flatten()
    print(f"  [7] Real Audio (Lena) Only: shape={emb_real_aud_only.shape}, norm={np.linalg.norm(emb_real_aud_only):.6f}")
    cases["real_audio_only"] = {
        "prompt": None,
        "embedding": [float(x) for x in emb_real_aud_only],
    }

    # Real-World Case 8: Real Audio + Prompt
    prompt_real_aud = "A woman speaking English into a microphone"
    emb_real_aud_prompt = model.encode([{"role": "user", "content": [
        {"type": "audio", "audio": real_audio},
        {"type": "text", "text": prompt_real_aud},
    ]}], device="cpu").flatten()
    print(f"  [8] Real Audio + Prompt: shape={emb_real_aud_prompt.shape}, norm={np.linalg.norm(emb_real_aud_prompt):.6f}")
    cases["real_audio_prompt"] = {
        "prompt": prompt_real_aud,
        "embedding": [float(x) for x in emb_real_aud_prompt],
    }

    # Save to JSON
    golden_data = {
        "synthetic_image_file": str(synth_img_path.name),
        "synthetic_audio_file": str(synth_audio_path.name),
        "real_image_file": str(real_img_path.name),
        "real_audio_file": str(real_audio_dst.name),
        "image_file": str(synth_img_path.name),
        "audio_file": str(synth_audio_path.name),
        "cases": cases,
    }

    golden_json_path = out_dir / "multimodal_golden.json"
    with open(golden_json_path, "w") as f:
        json.dump(golden_data, f, indent=2)

    print(f"\nSaved 8 golden embeddings to {golden_json_path}")
    print("DONE!")

if __name__ == "__main__":
    main()
