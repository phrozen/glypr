# Apple Neural Engine (ANE) Core ML Exporter

This directory contains the tooling for **Phase A of Glypr's Apple Neural Engine (ANE) integration**, enabling offline export of Google's EmbeddingGemma 2 subsystems into highly optimized Core ML packages (`.mlpackage` and `.mlmodelc`) with **W8A16 linear symmetric INT8 quantization**.

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
- **Direct Weight Transfer:** Transfers weights directly from `model.safetensors`.
- **Completely Static Shapes:** Static patch count (`1134` or `1260`) for Vision Tower; static sequence length (`512`) for Text Backbone.
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

| Model | Architecture | Static Input Shape | Output Shape | Weights Size (`.mlmodelc`) | Parity vs PyTorch FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision Tower** | 170M SigLIP / Gemma-4 ViT (16 layers, 12 heads) | `pixel_values: [1, 1134, 768]` (in `[-1.0, 1.0]`) | `[1, 126, 512]` | **147 MB** | **`0.999106` Cosine Sim** |
| **Text Backbone** | 270M Gemma 2 (24 layers, 4 Q heads, 1-2 KV heads) | `input_ids: [1, 512]` (int32), `attention_mask: [1, 512]` (fp32) | `[1, 768]` | **261 MB** | **`0.999684` Cosine Sim** |
| **Combined** | Resident Dual-Engine | — | — | **408 MB total** | **`> 0.999` Cosine Sim** |

> [!NOTE]
> The Vision Tower input preset `140-mtg` ($42 \times 27 = 1,134$ patches) exactly corresponds to Glypr's aspect-ratio-preserving preprocessor on $480 \times 680$ MTG card images ($672 \times 432$ px), outputting $14 \times 9 = 126$ soft tokens.

---

## 3. Usage Guide

### Prerequisites
Run with `uv` in the Glypr environment:
```bash
uv add coremltools
```

### Full Export (Vision Tower + Text Backbone)
```bash
uv run python scripts/export_coreml.py --target all --output-dir models/coreml
```

### Vision Tower Only (MTG Preset vs Standard Preset)
```bash
# MTG Card 480x680 optimized preset (1134 patches -> 126 soft tokens)
uv run python scripts/export_coreml.py --target vision --vision-preset 140-mtg --output-dir models/coreml

# Standard 140 budget preset (1260 patches -> 140 soft tokens)
uv run python scripts/export_coreml.py --target vision --vision-preset 140-std --output-dir models/coreml
```

### Text Backbone Only (Static 512 Tokens)
```bash
uv run python scripts/export_coreml.py --target text --text-seq-len 512 --output-dir models/coreml
```

### CLI Flags
- `--model-id`: Hugging Face repo ID or local path (default: `google/embeddinggemma-2`). Auto-detects local HF cache.
- `--output-dir`: Output directory for `.mlpackage` and `.mlmodelc` (default: `models/coreml`).
- `--target`: `all`, `vision`, or `text` (default: `all`).
- `--vision-preset`: `140-mtg` or `140-std` (default: `140-mtg`).
- `--text-seq-len`: Text sequence length (default: `512`).
- `--quantize`: `w8a16` (INT8 weights + FP16 compute) or `fp16` (default: `w8a16`).
- `--no-compile`: Skip compilation to `.mlmodelc`.
- `--no-verify`: Skip numerical cosine similarity assertion against PyTorch FP32.

---

## 4. Verification & Testing

Every export automatically runs verification inference targeting `ComputeUnit.CPU_AND_NE`:
```
Vision Tower Cosine Similarity vs PyTorch FP32: 0.999106
Text Backbone Cosine Similarity vs PyTorch FP32: 0.999684
```
Both models meet the strict $>0.998$ cosine similarity requirement for production embedding retrieval.
