#!/usr/bin/env python3
"""Warm live differential parity runner comparing Rust `glypr serve` against Python `EmbeddingGemmaBaseline`."""

import argparse
import base64
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
import wave

import numpy as np
from PIL import Image
import torch

# Ensure tests directory is on sys.path
TESTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TESTS_DIR.parent
GOLDEN_DIR = TESTS_DIR / "golden_vectors"
sys.path.insert(0, str(TESTS_DIR))

from baseline.embedder import EmbeddingGemmaBaseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="Python baseline torch dtype (default: float32)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps"],
        help="Execution device for Python/Rust (default: auto)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="TCP port for glypr serve (default: 0 for ephemeral free port)",
    )
    parser.add_argument(
        "--bin-path",
        type=str,
        default=None,
        help="Optional prebuilt glypr binary path (default: build release binary)",
    )
    parser.add_argument(
        "--mmproj",
        default="bf16",
        choices=["bf16", "q8", "none"],
        help="Multimodal projector mode (default: bf16)",
    )
    parser.add_argument(
        "--ane",
        action="store_true",
        help="Pass --ane to glypr serve",
    )
    return parser.parse_args()


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_or_build_binary(bin_path: str | None) -> Path:
    if bin_path is not None:
        resolved = Path(bin_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Provided --bin-path does not exist: {resolved}")
        return resolved

    print("Building release glypr binary (--features metal,ane)...")
    cmd = ["cargo", "build", "--release", "--features", "metal,ane", "--bin", "glypr"]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", REPO_ROOT / "target"))
    built = target_dir / "release" / "glypr"
    if not built.exists():
        raise FileNotFoundError(f"Expected built binary at {built}")
    return built


def wait_for_server_ready(host: str, port: int, proc: subprocess.Popen[str], timeout_secs: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_secs
    last_err = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out, _ = proc.communicate(timeout=5)
            raise RuntimeError(f"glypr serve exited prematurely with code {proc.returncode}:\n{out}")

        try:
            conn = http.client.HTTPConnection(host, port, timeout=2.0)
            conn.request("GET", "/health")
            health_resp = conn.getresponse()
            health_resp.read()
            if health_resp.status == 200:
                conn.request("GET", "/ready")
                ready_resp = conn.getresponse()
                ready_resp.read()
                conn.close()
                if ready_resp.status == 200:
                    return
                last_err = f"/ready returned HTTP {ready_resp.status}"
            else:
                conn.close()
                last_err = f"/health returned HTTP {health_resp.status}"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)

        time.sleep(0.25)

    raise TimeoutError(f"Timed out waiting for glypr serve on {host}:{port} ({last_err})")


class KeepAliveClient:
    """Persistent HTTP/1.1 keep-alive client for POST /v1/embeddings."""

    def __init__(self, host: str, port: int, timeout: float = 180.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.conn = http.client.HTTPConnection(host, port, timeout=timeout)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def embed(self, payload: dict[str, Any]) -> list[np.ndarray]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }
        for attempt in range(2):
            try:
                self.conn.request("POST", "/v1/embeddings", body=body, headers=headers)
                resp = self.conn.getresponse()
                raw = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"POST /v1/embeddings failed ({resp.status}): {raw.decode('utf-8', 'replace')}")
                parsed = json.loads(raw)
                items = sorted(parsed["data"], key=lambda item: item["index"])
                return [np.asarray(item["embedding"], dtype=np.float32) for item in items]
            except (http.client.HTTPException, ConnectionError, OSError):
                if attempt == 1:
                    raise
                self.close()
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        raise RuntimeError("Unreachable")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = np.asarray(a, dtype=np.float64).reshape(-1)
    b_f = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


def file_to_data_uri(path: Path, mime: str) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def read_wav_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16)
        return pcm.astype(np.float32) / 32767.0


def main() -> int:
    args = parse_args()

    py_device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    rust_device = "metal" if args.device == "mps" else args.device
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    baseline_mode = "text_only" if args.mmproj == "none" else "full"

    bin_path = resolve_or_build_binary(args.bin_path)
    port = args.port if args.port > 0 else pick_free_port()

    cmd = [str(bin_path), "--device", rust_device]
    if args.mmproj == "bf16":
        cmd.append("--bf16-mmproj")
    elif args.mmproj == "q8":
        cmd.append("--q8-mmproj")
    if args.ane:
        cmd.append("--ane")
    cmd.extend(["serve", "--host", "127.0.0.1", "--port", str(port)])

    print(f"Launching warm Rust server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    client: KeepAliveClient | None = None
    results: list[tuple[str, float, float, float, bool]] = []

    try:
        wait_for_server_ready("127.0.0.1", port, proc)
        print(f"Rust server ready on 127.0.0.1:{port}")

        baseline = EmbeddingGemmaBaseline(
            mode=baseline_mode,
            device=py_device,
            torch_dtype=torch_dtype,
        )
        client = KeepAliveClient("127.0.0.1", port)

        def record(name: str, py_vec: np.ndarray, rust_vec: np.ndarray, threshold: float) -> None:
            py_arr = np.asarray(py_vec, dtype=np.float32).reshape(-1)
            rust_arr = np.asarray(rust_vec, dtype=np.float32).reshape(-1)
            finite_ok = bool(np.all(np.isfinite(py_arr)) and np.all(np.isfinite(rust_arr)))
            rust_norm = float(np.linalg.norm(rust_arr))
            cos = cosine_sim(py_arr, rust_arr) if finite_ok else 0.0
            norm_ok = abs(rust_norm - 1.0) < 1e-3
            passed = finite_ok and norm_ok and (cos >= threshold)
            results.append((name, cos, rust_norm, threshold, passed))

        # 1. Text lengths 64t, 512t, 2048t, 8192t from corpus_8k.json
        corpus_path = GOLDEN_DIR / "corpus_8k.json"
        corpus = json.loads(corpus_path.read_text())
        entries_by_id = {entry["id"]: entry for entry in corpus["entries"]}

        for bucket, thresh in [(64, 0.999), (512, 0.999), (2048, 0.999), (8192, 0.998)]:
            entry = entries_by_id[f"doc_physics_{bucket}t"]
            title = entry.get("title")
            py_vec = baseline.encode_document(entry["content"], title=title)
            rust_payload: dict[str, Any] = {"input": entry["content"], "task": "Document"}
            if title is not None:
                rust_payload["title"] = title
            rust_vec = client.embed(rust_payload)[0]
            record(f"text_{bucket}t", py_vec, rust_vec, thresh)

        # 2. All 8 Tasks
        task_cases: list[tuple[str, str, str | None, str]] = [
            ("SearchQuery", "What causes the northern lights?", None, "query"),
            ("Document", "Solar energetic particles interact with Earth's magnetosphere.", "Magnetosphere", "document"),
            ("CodeRetrieval", "Fast inverse square root in Rust using f32::to_bits", None, "query"),
            ("QuestionAnswering", "How does Matryoshka representation learning truncate embeddings?", None, "query"),
            ("FactChecking", "Water boils at 100 degrees Celsius at standard atmospheric pressure.", None, "query"),
            ("Classification", "Quarterly revenue beat analyst consensus by 12 percent.", None, "symmetric"),
            ("Clustering", "Distributed consensus protocols for replicated state machines.", None, "symmetric"),
            ("SentenceSimilarity", "The quick brown fox jumps over the lazy dog.", None, "symmetric"),
        ]
        for task_name, text, title, kind in task_cases:
            if kind == "document":
                py_vec = baseline.encode_document(text, title=title)
                payload: dict[str, Any] = {"input": text, "task": task_name, "title": title}
            elif kind == "query":
                py_vec = baseline.encode_query(text, task=task_name)
                payload = {"input": text, "task": task_name}
            else:
                py_vec = baseline.encode_symmetric(text, task=task_name)
                payload = {"input": text, "task": task_name}
            rust_vec = client.embed(payload)[0]
            record(f"task_{task_name}", py_vec, rust_vec, 0.999)

        # 3. MRL dimensions (768, 512, 256, 128)
        mrl_text = "Matryoshka representation learning enables adaptive vector dimensionality."
        for dim in [768, 512, 256, 128]:
            py_vec = baseline.encode_query(mrl_text, task="SearchQuery", truncate_dim=dim)
            rust_vec = client.embed({"input": mrl_text, "task": "SearchQuery", "dimensions": dim})[0]
            record(f"mrl_{dim}d", py_vec, rust_vec, 0.999)

        # 4. Batched text [q1, q2, doc1]
        batch_texts = [
            "What is the speed of light in vacuum?",
            "How do transformers compute scaled dot-product attention?",
            "Light travels in vacuum at exactly 299,792,458 meters per second.",
        ]
        py_batch = baseline.encode_query(batch_texts, task="SearchQuery")
        rust_batch = client.embed({"input": batch_texts, "task": "SearchQuery"})
        for idx in range(len(batch_texts)):
            record(f"batch_item_{idx}", py_batch[idx], rust_batch[idx], 0.999)

        # 5. Multimodal cases when --mmproj != none
        if args.mmproj != "none":
            pepper_path = GOLDEN_DIR / "pepper.png"
            pepper_img = Image.open(pepper_path).convert("RGB")
            pepper_uri = file_to_data_uri(pepper_path, "image/png")

            img_proc = baseline.model[0].processor.image_processor
            orig_budget = img_proc.max_soft_tokens
            for budget in [70, 140, 280, 560]:
                try:
                    img_proc.max_soft_tokens = budget
                    py_vec = baseline.encode_media({"image": pepper_img})
                finally:
                    img_proc.max_soft_tokens = orig_budget
                rust_vec = client.embed({"input": {"image": pepper_uri, "budget": budget}})[0]
                record(f"image_pepper_b{budget}", py_vec, rust_vec, 0.998)

            lenna_path = GOLDEN_DIR / "lenna.png"
            lenna_img = Image.open(lenna_path).convert("RGB")
            lenna_uri = file_to_data_uri(lenna_path, "image/png")
            img_prompt = "A classic portrait photograph of a woman wearing a hat"
            py_lenna = baseline.encode_media(
                [{"role": "user", "content": [{"type": "image", "image": lenna_img}, {"type": "text", "text": img_prompt}]}]
            )
            rust_lenna = client.embed({"input": {"image": lenna_uri, "text": img_prompt}})[0]
            record("image_lenna_prompt", py_lenna, rust_lenna, 0.998)

            music_path = GOLDEN_DIR / "music_16k.wav"
            music_pcm = read_wav_pcm(music_path)
            music_uri = file_to_data_uri(music_path, "audio/wav")

            py_aud = baseline.encode_media([{"role": "user", "content": [{"type": "audio", "audio": music_pcm}]}])
            rust_aud = client.embed({"input": {"audio": music_uri}})[0]
            record("audio_music_only", py_aud, rust_aud, 0.998)

            aud_prompt = "Acoustic guitar instrumental music"
            py_aud_p = baseline.encode_media(
                [{"role": "user", "content": [{"type": "audio", "audio": music_pcm}, {"type": "text", "text": aud_prompt}]}]
            )
            rust_aud_p = client.embed({"input": {"audio": music_uri, "text": aud_prompt}})[0]
            record("audio_music_prompt", py_aud_p, rust_aud_p, 0.998)

    finally:
        if client is not None:
            client.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    # Print Scorecard
    print("\n" + "=" * 78)
    print(f"{'case':<26} {'cos_sim':>10} {'rust_norm':>11} {'threshold':>10} {'status':>8}")
    print("-" * 78)
    failures = 0
    for name, cos, rust_norm, thresh, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            failures += 1
        print(f"{name:<26} {cos:>10.6f} {rust_norm:>11.6f} {thresh:>10.3f} {status:>8}")
    print("=" * 78)

    if failures > 0 or not results:
        print(f"FAILED: {failures} / {len(results)} cases failed.")
        return 1

    print(f"PASSED: All {len(results)} differential parity cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
