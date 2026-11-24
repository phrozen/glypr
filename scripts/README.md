# Apple Neural Engine (ANE) Core ML Exporter

This directory contains the tooling for **Glypr's Apple Neural Engine (ANE) integration**, enabling offline export of Google's EmbeddingGemma 2 subsystems into compiled Core ML bundles (`.mlmodelc`) with **W8A16 linear symmetric INT8 quantization**. See also the [ANE Runtime Deployment Guide](../docs/deployment/quantization-and-runtimes.md#10-apple-neural-engine-ane--core-ml-runtime-deployment) and [Project README](../README.md).

---

## 1. Architectural Highlights

### Why Not HuggingFace `torch.jit.trace` Direct Export?
Standard HuggingFace model classes (`Gemma4VisionModel`, `EmbeddingGemma2Model`) contain dynamic tensor constructs that break TorchScript tracing and Core ML MIL translation:
1. Dynamic boolean bitwise OR (`__or__`) in masking utils causing TorchScript runtime assertions.
2. Dynamic KV-head repeating (`repeat_kv` via `torch.cat`) causing MIL linear broadcast shape mismatch (`ValueError: Incompatible dim 1 in shapes (1, 4) vs. (1, 2)`).
3. NumPy 2.0+ scalar casting incompatibility in `coremltools/converters/mil/frontend/torch/ops.py:_cast`.
4. Dynamic shape views creating dynamic TorchScript graphs that fail ANE compilation.

### The Clean Module Pattern
`scripts/export_coreml.py` solves this by introducing clean, pure PyTorch modules mirroring `crates/glypr-core/src/multimodal/vision.rs` and `model.rs`:
- **Direct Weight Transfer & `config.json` Reading:** Transfers weights from single or sharded (`model.safetensors.index.json`) Safetensors checkpoints and reads `config.json` hyperparameters when available.
- **Static Patch-Grid Vision Presets:** Supports 8 static aspect-ratio presets (`140-portrait`, `140-std`, `140-square`, `140-landscape`, `280-portrait`, `280-std`, `280-square`, `280-landscape`) tagged with `_h{H}_w{W}` and `glypr.patch_height` / `glypr.patch_width` metadata so portrait and landscape orientations never collide.
- **Static Text Buckets & Sliding-Window Masking:** Exports static sequence-length buckets (default `512`, or multi-bucket `--text-seq-lens 128,512,1024`) with exact `|i - j| <= 512` sliding-window attention masking on the 20 sliding layers when `seq_len > 512`.
- **Hardware-Accelerated Primitives:**
  - `nn.AvgPool2d(kernel_size=3, stride=3)` for 2D spatial patch averaging (compiles to native ANE hardware pooling).
  - Precomputed static 2D Axial RoPE buffers and static 1D RoPE buffers eliminating runtime trigonometric computations.
  - Broadcastable static additive attention masking (`(1.0 - mask) * -10000.0`).
  - GQA expansion via `torch.repeat_interleave` compiling cleanly to MIL tile/reshape operations.
- **W8A16 Linear Symmetric INT8 Quantization:**
  - Weights quantized to INT8 with per-tensor scales.
  - Activations and compute remain in native Apple Silicon FP16.
  - Strictly avoids BF16 (unsupported by ANE ALUs).

---

## 2. Exported Models & Specifications

| Model / Preset | Static Grid (`patch_h x patch_w`) | Static Input Shape | Output Shape |
| :--- | :---: | :--- | :--- |
| **Vision `140-portrait`** | `42 x 27` (432×672 px) | `pixel_values: [1, 1134, 768]` | `[1, 126, 512]` |
| **Vision `140-std`** | `42 x 30` (480×672 px) | `pixel_values: [1, 1260, 768]` | `[1, 140, 512]` |
| **Vision `140-square`** | `33 x 33` (528×528 px) | `pixel_values: [1, 1089, 768]` | `[1, 121, 512]` |
| **Vision `140-landscape`** | `27 x 42` (672×432 px) | `pixel_values: [1, 1134, 768]` | `[1, 126, 512]` |
| **Vision `280-portrait`** | `57 x 42` (672×912 px) | `pixel_values: [1, 2394, 768]` | `[1, 266, 512]` |
| **Vision `280-std`** | `60 x 42` (672×960 px) | `pixel_values: [1, 2520, 768]` | `[1, 280, 512]` |
| **Vision `280-square`** | `48 x 48` (768×768 px) | `pixel_values: [1, 2304, 768]` | `[1, 256, 512]` |
| **Vision `280-landscape`** | `42 x 57` (912×672 px) | `pixel_values: [1, 2394, 768]` | `[1, 266, 512]` |
| **Text Backbone (`s{N}`)** | Static `seq_len` (default `512`) | `input_ids: [1, N]`, `attention_mask: [1, N]` | `[1, 768]` |

---

## 3. Usage Guide

### Prerequisites
Sync the Python environment with `uv` (dependencies including `coremltools` are declared in `pyproject.toml`):
```bash
uv sync
```

### Full Export (Default Multi-Aspect Vision + Text Backbone)
```bash
# Export default multi-aspect vision presets + 512-token text backbone
uv run python scripts/export_coreml.py --target all --vision-preset default --output-dir models/coreml

# Or export all 8 vision presets + multiple text sequence-length buckets
uv run python scripts/export_coreml.py --target all --vision-preset all --text-seq-lens 128,512,1024 --output-dir models/coreml
```

### Vision Tower Presets
```bash
# Export specific comma-separated presets
uv run python scripts/export_coreml.py --target vision --vision-preset 140-portrait,280-square,280-landscape --output-dir models/coreml
```

### Text Backbone Buckets
```bash
# Single 512-token bucket
uv run python scripts/export_coreml.py --target text --text-seq-len 512 --output-dir models/coreml

# Multiple buckets (s128, s512, s1024 with sliding-window mask for >512)
uv run python scripts/export_coreml.py --target text --text-seq-lens 128,512,1024 --output-dir models/coreml
```

### Model Discovery & Placement (`GLYPR_COREML_DIR`)
By default, intermediate `.mlpackage` bundles are cleaned after compiling to `.mlmodelc` (pass `--keep-mlpackage` to retain them), and `manifest.json` is written alongside the `.mlmodelc` bundles.

`glypr` discovers CoreML models automatically in this order:
1. `--coreml-dir <DIR>`
2. `GLYPR_COREML_DIR` environment variable
3. `./models/coreml` (repo-local checkout)
4. User data directory: `~/Library/Application Support/glypr/coreml` on macOS (or `$XDG_DATA_HOME/glypr/coreml` / `~/.cache/glypr/coreml`)

For global `cargo install` usage outside the repository, export directly to your user CoreML directory or set `GLYPR_COREML_DIR`:
```bash
uv run python scripts/export_coreml.py --target all --vision-preset default --output-dir "$HOME/Library/Application Support/glypr/coreml"
```

### CLI Flags
- `--model-id`: Hugging Face repo ID, local model directory, or `.safetensors` path (default: `google/embeddinggemma-2`).
- `--output-dir`: Output directory for `.mlmodelc` bundles and `manifest.json` (default: `models/coreml`).
- `--target`: `all`, `vision`, or `text` (default: `all`).
- `--vision-preset`: Preset name, comma-separated list, `default` (`140-portrait,280-portrait,280-square,280-landscape`), or `all` (default: `140-portrait`).
- `--text-seq-len`: Single static sequence length for resident text model (default: `512`).
- `--text-seq-lens`: Comma-separated static sequence lengths (e.g. `128,512,1024`; overrides `--text-seq-len`).
- `--quantize`: `w8a16` (INT8 weights + FP16 compute) or `fp16` (default: `w8a16`).
- `--keep-mlpackage`: Retain intermediate `.mlpackage` bundles after compiling to `.mlmodelc` (deleted by default).
- `--no-compile`: Skip compilation to `.mlmodelc`.
- `--no-verify`: Skip parity verification against PyTorch.
- `--offline`: Fail immediately if weights/config are not present in local HF cache (`HF_HUB_OFFLINE=1` also honored).

---

## 4. Verification & Testing

Every export automatically runs verification inference targeting `ComputeUnit.CPU_AND_NE`:
```
Vision Tower Cosine Similarity vs PyTorch FP32: > 0.998 (typically ~0.9991–0.9993)
Text Backbone Cosine Similarity vs PyTorch FP32: > 0.995 (typically ~0.9997)
```
Both models meet strict cosine similarity parity requirements for production embedding retrieval.
