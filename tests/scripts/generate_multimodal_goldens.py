#!/usr/bin/env python3
"""
Generate comprehensive multimodal golden reference vectors for EmbeddingGemma 2.
Covers:
  1. Synthetic and real-world single images (square, landscape 16:9, portrait 2:3)
  2. Synthetic and real-world audio (vocal speech, acoustic instruments)
  3. Dynamic vision token budget sweeps (70, 140, 280, 560 tokens)
  4. Multi-image composite inputs (multiple <|image|> placeholders)
  5. Interleaved multimodal inputs (text + image + audio)
  6. Semantic cross-modal text retrieval queries
"""

import json
import wave
from pathlib import Path
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

def read_wav_pcm(path: Path) -> np.ndarray:
    """Read a 16 kHz mono WAV file as a float32 array in [-1.0, 1.0]."""
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
        pcm = np.frombuffer(frames, dtype=np.int16)
        return pcm.astype(np.float32) / 32767.0

def main():
    out_dir = Path(__file__).resolve().parent.parent / "golden_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SentenceTransformer('google/embeddinggemma-2') on CPU...")
    model = SentenceTransformer("google/embeddinggemma-2", device="cpu")

    # 1. Deterministic synthetic test image (224x224 RGB)
    img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        for x in range(224):
            img_arr[y, x, 0] = int((x / 224.0) * 255)
            img_arr[y, x, 1] = int((y / 224.0) * 255)
            img_arr[y, x, 2] = int(((x + y) / 448.0) * 255)

    synth_img = Image.fromarray(img_arr, mode="RGB")
    synth_img_path = out_dir / "multimodal_test_image.png"
    synth_img.save(synth_img_path)

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

    synth_audio = read_wav_pcm(synth_audio_path)

    # 3. Load all image assets
    images = {
        "synth": synth_img,
        "lenna": Image.open(out_dir / "lenna.png").convert("RGB"),
        "baboon": Image.open(out_dir / "baboon.png").convert("RGB"),
        "pepper": Image.open(out_dir / "pepper.png").convert("RGB"),
        "countryside": Image.open(out_dir / "countryside.png").convert("RGB"),
        "dog": Image.open(out_dir / "dog.png").convert("RGB"),
    }

    # 4. Load all audio assets
    audios = {
        "synth": synth_audio,
        "lena": read_wav_pcm(out_dir / "lena_16k.wav"),
        "female_voice": read_wav_pcm(out_dir / "female_voice_16k.wav"),
        "music": read_wav_pcm(out_dir / "music_16k.wav"),
    }

    cases = {}

    def add_case(cid: str, emb: np.ndarray, meta: dict = None):
        vec = [float(x) for x in emb.flatten()]
        meta_dict = meta or {}
        meta_dict["embedding"] = vec
        cases[cid] = meta_dict
        norm = float(np.linalg.norm(vec))
        print(f"  ✓ {cid:32s} (dim={len(vec)}, norm={norm:.4f})")

    print("\n[A] Encoding Baseline Synthetic & Lenna Assets...")
    # Baseline Case 1: Synthetic Image
    emb = model.encode([{"role": "user", "content": [{"type": "image", "image": images["synth"]}]}], device="cpu")
    add_case("image_only", emb)

    # Baseline Case 2: Synthetic Image + Prompt
    prompt_img = "Describe this picture"
    emb = model.encode([{"role": "user", "content": [{"type": "image", "image": images["synth"]}, {"type": "text", "text": prompt_img}]}], device="cpu")
    add_case("image_prompt", emb, {"prompt": prompt_img})

    # Baseline Case 3: Synthetic Audio
    emb = model.encode([{"role": "user", "content": [{"type": "audio", "audio": audios["synth"]}]}], device="cpu")
    add_case("audio_only", emb)

    # Baseline Case 4: Synthetic Audio + Prompt
    prompt_aud = "A musical sound"
    emb = model.encode([{"role": "user", "content": [{"type": "audio", "audio": audios["synth"]}, {"type": "text", "text": prompt_aud}]}], device="cpu")
    add_case("audio_prompt", emb, {"prompt": prompt_aud})

    # Baseline Case 5: Lenna Only
    emb = model.encode([{"role": "user", "content": [{"type": "image", "image": images["lenna"]}]}], device="cpu")
    add_case("real_image_only", emb)

    # Baseline Case 6: Lenna + Prompt
    prompt_lenna = "A classic portrait photograph of a woman wearing a hat"
    emb = model.encode([{"role": "user", "content": [{"type": "image", "image": images["lenna"]}, {"type": "text", "text": prompt_lenna}]}], device="cpu")
    add_case("real_image_prompt", emb, {"prompt": prompt_lenna})

    # Baseline Case 7: Lena Audio Only
    emb = model.encode([{"role": "user", "content": [{"type": "audio", "audio": audios["lena"]}]}], device="cpu")
    add_case("real_audio_only", emb)

    # Baseline Case 8: Lena Audio + Prompt
    prompt_lena_aud = "A woman speaking English into a microphone"
    emb = model.encode([{"role": "user", "content": [{"type": "audio", "audio": audios["lena"]}, {"type": "text", "text": prompt_lena_aud}]}], device="cpu")
    add_case("real_audio_prompt", emb, {"prompt": prompt_lena_aud})

    print("\n[B] Encoding Extended Image Assets (Aspect Ratios & Textures)...")
    add_case("baboon_only", model.encode([{"role": "user", "content": [{"type": "image", "image": images["baboon"]}]}], device="cpu"), {"file": "baboon.png"})
    add_case("pepper_only", model.encode([{"role": "user", "content": [{"type": "image", "image": images["pepper"]}]}], device="cpu"), {"file": "pepper.png"})
    add_case("countryside_only", model.encode([{"role": "user", "content": [{"type": "image", "image": images["countryside"]}]}], device="cpu"), {"file": "countryside.png", "aspect": "16:9"})
    add_case("dog_only", model.encode([{"role": "user", "content": [{"type": "image", "image": images["dog"]}]}], device="cpu"), {"file": "dog.png", "aspect": "2:3"})

    print("\n[C] Encoding Extended Audio Assets...")
    add_case("female_voice_only", model.encode([{"role": "user", "content": [{"type": "audio", "audio": audios["female_voice"]}]}], device="cpu"), {"file": "female_voice_16k.wav"})
    add_case("music_only", model.encode([{"role": "user", "content": [{"type": "audio", "audio": audios["music"]}]}], device="cpu"), {"file": "music_16k.wav"})

    print("\n[D] Encoding Dynamic Vision Token Budget Sweeps (pepper.png)...")
    for budget in [70, 140, 280, 560]:
        model[0].processor.image_processor.max_soft_tokens = budget
        emb_budget = model.encode({"image": images["pepper"]}, device="cpu")
        add_case(f"pepper_budget_{budget}", emb_budget, {"file": "pepper.png", "budget": budget})
    # Reset default budget
    model[0].processor.image_processor.max_soft_tokens = 280

    print("\n[E] Encoding Multi-Image Composite Inputs...")
    composite_text = "A comparison of two pictures. First: <|image|> Second: <|image|>"
    emb_comp1 = model.encode({
        "text": composite_text,
        "image": [images["baboon"], images["pepper"]],
    }, device="cpu")
    add_case("composite_multi_image", emb_comp1, {
        "text": composite_text,
        "images": ["baboon.png", "pepper.png"],
    })

    print("\n[F] Encoding Interleaved Multimodal (Text + Image + Audio)...")
    interleaved_text = "A picture and some audio: <|image|> <|audio|>"
    emb_comp2 = model.encode({
        "text": interleaved_text,
        "image": images["baboon"],
        "audio": audios["music"],
    }, device="cpu")
    add_case("composite_interleaved", emb_comp2, {
        "text": interleaved_text,
        "image": "baboon.png",
        "audio": "music_16k.wav",
    })

    print("\n[G] Encoding Cross-Modal Retrieval Text Queries...")
    queries = {
        "query_dog": "A photograph of a dog sitting in grass",
        "query_countryside": "A wide landscape view of rolling hills and trees in the countryside",
        "query_music": "Acoustic guitar instrumental music",
        "query_female_voice": "A woman speaking English clearly",
    }
    for qid, qtext in queries.items():
        emb_q = model.encode(qtext, prompt_name="SearchQuery", device="cpu")
        add_case(qid, emb_q, {"query": qtext, "task": "SearchQuery"})

    golden_data = {
        "synthetic_image_file": str(synth_img_path.name),
        "synthetic_audio_file": str(synth_audio_path.name),
        "real_image_file": "lenna.png",
        "real_audio_file": "lena_16k.wav",
        "image_file": str(synth_img_path.name),
        "audio_file": str(synth_audio_path.name),
        "cases": cases,
    }

    golden_json_path = out_dir / "multimodal_golden.json"
    with open(golden_json_path, "w") as f:
        json.dump(golden_data, f, indent=2)

    print(f"\n✓ Successfully saved {len(cases)} multimodal golden embeddings to {golden_json_path}!")

if __name__ == "__main__":
    main()
