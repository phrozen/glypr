# EmbeddingGemma 2 Multimodal GGUF Nuances & Conversion Reference

This document serves as the definitive engineering reference for the architectural quirks, tensor naming conventions, structural transformations, and subtle mathematical transformations present in Google DeepMind's EmbeddingGemma 2 weights when packaged into the **`ggml-org` GGUF container** (`embeddinggemma-2-*.gguf` and `mmproj-embeddinggemma-2-*.gguf`).

Anyone re-implementing, debugging, or converting EmbeddingGemma 2 models in native runtimes (Rust, C++, Metal, CUDA) should refer to this guide to avoid the pitfalls encountered during the development of `glypr`.

---

## 1. High-Level GGUF Container Topology

Unlike older multimodal models (e.g. LLaVA, CLIP) where projectors consist of 2–3 linear layers, EmbeddingGemma 2 contains **two complete external neural networks** inside `mmproj`:
1. **Vision Tower (`v.*`)**: 16-layer SigLIP-derived Transformer (210 tensors, ~170M params).
2. **Audio Tower (`a.*`)**: 12-layer Conformer with 2-stage subsampling convolution (751 tensors, ~300M params).
3. **Multimodal Projectors (`mm.*`)**: Linear projections mapping tower representations into the 512-dim language embedding space (2 tensors).

In total, the GGUF `mmproj` file packages **963 weight tensors**, all of which map to Google's official `model.safetensors` distribution.

---

## 2. Tensor Mapping & Naming Dictionary

### A. Non-Block Tensors

| Safetensors (Hugging Face) | GGUF (`mmproj`) | Shape / Dtype | Description |
| :--- | :--- | :---: | :--- |
| `embed_vision.embedding_projection.weight` | `mm.input_projection.weight` | `[512, 768]` BF16 | Vision Projector (maps pooled vision tokens to language embedding dim) |
| `embed_audio.embedding_projection.weight` | `mm.a.input_projection.weight` | `[512, 1536]` BF16 | Audio Projector (maps Conformer tokens to language embedding dim) |
| `vision_tower.patch_embedder.input_proj.weight` | `v.patch_embd.weight` | `[768, 3, 16, 16]` BF16 | Vision patch linear projection (transposed, see Section 4) |
| `vision_tower.patch_embedder.position_embedding_table` | `v.position_embd.weight` | `[2, 10240, 768]` BF16 | 2D coordinate grid positional embedding table |
| `audio_tower.subsample_conv_projection.layer0.conv.weight` | `a.conv1d.0.weight` | `[128, 1, 3, 3]` BF16 | Audio Subsampling Conv Stage 1 (2D Conv despite "conv1d" name) |
| `audio_tower.subsample_conv_projection.layer0.norm.weight` | `a.conv1d.0.norm.weight` | `[128]` BF16 | LayerNorm for Subsampling Stage 1 |
| `audio_tower.subsample_conv_projection.layer1.conv.weight` | `a.conv1d.1.weight` | `[32, 128, 3, 3]` BF16 | Audio Subsampling Conv Stage 2 (2D Conv) |
| `audio_tower.subsample_conv_projection.layer1.norm.weight` | `a.conv1d.1.norm.weight` | `[32]` BF16 | LayerNorm for Subsampling Stage 2 |
| `audio_tower.subsample_conv_projection.input_proj_linear.weight` | `a.input_projection.weight` | `[1024, 1024]` BF16 | Linear projection following flattened subsampling convolution |
| `audio_tower.output_proj.weight` | `a.pre_encode.out.weight` | `[1536, 1024]` BF16 | Audio Tower final linear projection |
| `audio_tower.output_proj.bias` | `a.pre_encode.out.bias` | `[1536]` BF16 | Audio Tower final linear bias |

---

### B. Vision Blocks (`v.blk.{L}.*`, $L \in [0, 15]$)

| Safetensors | GGUF | Note |
| :--- | :--- | :--- |
| `...layers.{L}.input_layernorm.weight` | `v.blk.{L}.ln1.weight` | Pre-attention RMSNorm |
| `...layers.{L}.self_attn.q_proj.linear.weight` | `v.blk.{L}.attn_q.weight` | Query projection `[768, 768]` |
| `...layers.{L}.self_attn.k_proj.linear.weight` | `v.blk.{L}.attn_k.weight` | Key projection `[768, 768]` |
| `...layers.{L}.self_attn.v_proj.linear.weight` | `v.blk.{L}.attn_v.weight` | Value projection `[768, 768]` |
| `...layers.{L}.self_attn.o_proj.linear.weight` | `v.blk.{L}.attn_out.weight` | Output projection `[768, 768]` |
| `...layers.{L}.self_attn.q_norm.weight` | `v.blk.{L}.attn_q_norm.weight` | Query per-head RMSNorm |
| `...layers.{L}.self_attn.k_norm.weight` | `v.blk.{L}.attn_k_norm.weight` | Key per-head RMSNorm |
| `...layers.{L}.post_attention_layernorm.weight`| `v.blk.{L}.attn_post_norm.weight`| Post-attention RMSNorm |
| `...layers.{L}.pre_feedforward_layernorm.weight`| `v.blk.{L}.ln2.weight` | Pre-FFN RMSNorm |
| `...layers.{L}.mlp.gate_proj.linear.weight` | `v.blk.{L}.ffn_gate.weight` | FFN Gate projection `[3072, 768]` |
| `...layers.{L}.mlp.up_proj.linear.weight` | `v.blk.{L}.ffn_up.weight` | FFN Up projection `[3072, 768]` |
| `...layers.{L}.mlp.down_proj.linear.weight` | `v.blk.{L}.ffn_down.weight` | FFN Down projection `[768, 3072]` |
| `...layers.{L}.post_feedforward_layernorm.weight`| `v.blk.{L}.ffn_post_norm.weight`| Post-FFN RMSNorm |

---

### C. Audio Blocks (`a.blk.{L}.*`, $L \in [0, 11]$)

Each of the 12 Conformer blocks contains **63 tensors**, categorized as follows:

1. **Self-Attention:**
   - Weights: `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_out.weight` `[1024, 1024]`.
   - Relative position projection: `attn_k_rel.weight` `[1024, 1024]`.
   - Attention norms: `attn_pre_norm.weight`, `attn_post_norm.weight` `[1024]`.
   - Head scale: `per_dim_scale.weight` `[128]` (Pre-computed `softplus`, see Section 5).
2. **Macaron FFN 1 & 2:**
   - Pre/post norms: `ffn_norm.weight`, `ffn_post_norm.weight`, `ffn_norm_1.weight`, `ffn_post_norm_1.weight` `[1024]`.
   - Projections: `ffn_up.weight`, `ffn_down.weight`, `ffn_up_1.weight`, `ffn_down_1.weight` (`[4096, 1024]` and `[1024, 4096]`).
3. **Lightweight 1D Convolution (`lconv1d`):**
   - Norms: `conv_norm.weight`, `norm_conv.weight` (**Swapped names**, see Section 6).
   - Linear projections: `conv_pw1.weight` `[2048, 1024]` (Pointwise 1 with GLU), `conv_pw2.weight` `[1024, 1024]` (Pointwise 2).
   - Depthwise 1D Conv: `conv_dw.weight` `[1024, 5]` (Kernel size 5).
4. **Block Output Norm:**
   - `ln2.weight` `[1024]`.
5. **Scalar Activation Bounds (40 tensors per block):**
   - For all 10 linear operators, 4 scalar bounds: `input_min`, `input_max`, `output_min`, `output_max` (see Section 3).

---

## 3. The Gemma 4 Clippable Linears Architecture

### The Problem
When running the 12 Conformer blocks using standard linear operations ($y = x W^T$), the activations begin deviating at layer 1 and suffer catastrophic explosion by layer 3 (divergence $> 29.21$).

### The Cause
Gemma 4 Audio uses Google's `Gemma4ClippableLinear` (`use_clipped_linears: true`). Every linear layer enforces scalar dynamic range clamping before and after matrix multiplication.

### The GGUF Representation
GGUF serializes these scalar bounds as single-element float32 tensors with matching base names:
- `<tensor_prefix>.input_min`
- `<tensor_prefix>.input_max`
- `<tensor_prefix>.output_min`
- `<tensor_prefix>.output_max`

Across the 10 linear operators per block $\times 12$ blocks, there are **480 scalar bound tensors** in the file.

### Required Execution Semantics
In your runtime, every linear forward pass must execute as:

```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    // 1. Pre-projection input clamping
    let x_clamped = if let (Some(min), Some(max)) = (self.input_min, self.input_max) {
        x.clamp(min, max)?
    } else {
        x.clone()
    };

    // 2. Standard linear projection
    let mut out = self.linear.forward(&x_clamped)?;

    // 3. Post-projection output clamping
    if let (Some(min), Some(max)) = (self.output_min, self.output_max) {
        out = out.clamp(min, max)?;
    }

    Ok(out)
}
```

Applying this drops the Conformer layer divergence from **29.21 down to 0.05** (the mathematical limit of BFloat16 precision).

---

## 4. Tensor Transpositions & Dimension Swaps

### Vision Patch Embedder (`v.patch_embd.weight`)
- **PyTorch Safetensors:** Shape `[768, 768]` representing `[out_channels, in_channels * patch_h * patch_w]` where the flattened order is `(16, 16, 3)`.
- **GGUF Export Converter:** Converted into a 4D tensor `[768, 3, 16, 16]` following standard PyTorch NCHW image layout conventions:
  $$\text{PyTorch } [768, 16, 16, 3] \xrightarrow{\text{GGUF export}} [768, 3, 16, 16]$$
- **Runtime Requirement:** If your engine treats patch embedding as a matrix multiplication on flattened RGB patches (`[batch, num_patches, 768]`), you **must** permute the GGUF tensor back:
  ```rust
  // Load GGUF [768, 3, 16, 16], permute to [768, 16, 16, 3], reshape to [768, 768]
  let patch_weight = gguf_tensor
      .permute((0, 2, 3, 1))?
      .contiguous()?
      .reshape((768, 768))?;
  ```

---

## 5. Pre-Computed Parameters in GGUF Export

### Conformer Attention `per_dim_scale.weight`
In PyTorch `Gemma4AudioAttention`, the per-head scaling vector is stored unconstrained and transformed dynamically via `softplus`:
$$\text{scale} = \text{softplus}(\text{per\_dim\_scale}) = \ln(1 + e^{\text{per\_dim\_scale}})$$

**The GGUF Quirks:**
The GGUF export script **already applies $\text{softplus}$** during the export process!
- Safetensors raw values: `[-0.535, -0.730, ...]`
- GGUF tensor values: `[softplus(-0.535), softplus(-0.730), ...] = [0.448, 0.385, ...]`

> [!WARNING]
> Do **NOT** apply `softplus` to `per_dim_scale.weight` loaded from GGUF! It is already pre-computed. Applying `softplus` a second time will distort the query-key scaling factor and produce completely corrupted attention logits.

---

## 6. Naming Inversions in Audio `lconv1d`

In the Gemma 4 Audio Conformer block, the lightweight 1D convolution block contains two LayerNorms/RMSNorms:
1. `pre_layer_norm`: Normalized before the first pointwise linear (`linear_start` / `conv_pw1`).
2. `conv_norm`: Normalized after depthwise convolution, before the activation function and `linear_end` (`conv_pw2`).

During GGUF export, their names were **inverted**:

| Architectural Role | PyTorch Safetensors Name | GGUF Name |
| :--- | :--- | :--- |
| Pre-LayerNorm (before `pw1`) | `lconv1d.pre_layer_norm.weight` | `a.blk.{L}.conv_norm.weight` ⚠️ |
| Post-Depthwise Norm (after `dw`) | `lconv1d.conv_norm.weight` | `a.blk.{L}.norm_conv.weight` ⚠️ |

If loaded naively by name matching, the two norms will be swapped, distorting the GLU and SiLU activations.

---

## 7. Audio Blocked Chunking Attention & `_rel_shift`

The Conformer self-attention operates over overlapping blocks rather than global tokens:

1. **Chunking Parameters:**
   - `chunk_size = 12`
   - `context_left = 13` $\implies \text{past\_horizon} = 12$
   - `context_right = 0` $\implies \text{future\_horizon} = 0$
   - Context window size: $12 + 12 + 0 = 24$.
2. **Extracting Block Context:**
   - Keys and Values are padded by 12 on the left and 11 on the right:
     $$\text{pad}(k, \text{left}=12, \text{right}=11)$$
   - Sliced into overlapping windows of size 24 strided by 12: block $b$ starts at index $b \times 12$.
3. **Relative Position Embeddings ($R$):**
   - Sinusoidal embeddings with context size 24.
   - Position IDs run in reverse from $12 \to 0$ (13 positions total):
     $$\text{position\_ids} = [12, 11, 10, \dots, 1, 0]$$
   - Projected via `relative_k_proj` (`attn_k_rel.weight`) into `[13, num_heads, head_dim]`.
4. **Relative Position Shift (`_rel_shift`):**
   - Multiplying queries by relative keys yields `matrix_bd` of shape `[batch, heads, blocks, 12, 13]`.
   - To align relative positions with key-query pairs:
     1. Zero-pad dimension 4 on the right by 12: `[..., 12, 25]`.
     2. Flatten dimensions 3 and 4: `[..., 12 * 25 = 300]`.
     3. Slice the first $12 \times 24 = 288$ elements: `[..., 288]`.
     4. Reshape back to `[..., 12, 24]`.
   $$\text{attn\_weights} = \text{matrix\_ac} + \text{matrix\_bd}$$
5. **5D Attention Mask:**
   - Attention logits are masked using a 5D boolean mask:
     $$\text{mask}[b, q, c] = \begin{cases} 0 & \text{if } t_k \ge 0 \land t_k < \text{seq\_len} \land t_k \le t_q \land (t_q - t_k) \le 11 \\ -\infty & \text{otherwise} \end{cases}$$
     where $t_q = b \times 12 + q$ and $t_k = b \times 12 - 12 + c$.

---

## 8. Artifact Quantization Nuance (WAV vs Continuous Floats)

When evaluating audio golden parity against PyTorch `SentenceTransformer`:
- **The Pitfall:** In Python, passing a synthesized `numpy.ndarray` (continuous 32-bit float values) directly into `model.encode()` bypasses audio container quantization.
- When saved to a standard `.wav` file, samples are quantized to 16-bit signed PCM integers ($[-32768, 32767]$) with precision step $\Delta = \frac{1}{32767} \approx 3.05 \times 10^{-5}$.
- This 16-bit quantization noise shifts the final 768d pooled embedding vector by **~0.00518** cosine similarity in PyTorch itself:
  $$\text{CosineSim}(\text{emb}_{\text{float32}}, \text{emb}_{\text{wav\_pcm16}}) = 0.99482$$
- **Rule for Golden Testing:** Reference embeddings **must** be generated by reading back the actual 16-bit PCM `.wav` file from disk, ensuring both Python and Rust test harnesses evaluate identical input bytes.

---

## 9. Multimodal Special Token Vocabulary IDs

When splicing soft tokens into the language model input sequence:

| Special Token | String Literal | Token ID | Role |
| :--- | :--- | :---: | :--- |
| $\langle\text{pad}\rangle$ | `<pad>` | `0` | Padding token |
| $\langle\text{eos}\rangle$ | `<eos>` | `1` | End-of-sequence delimiter |
| $\langle\text{bos}\rangle$ | `<bos>` | `2` | Beginning-of-sequence delimiter |
| $\langle\text{boi}\rangle$ | `<\|image\|>` (open) | `255999` | Beginning of Image block |
| $\langle\text{boa}\rangle$ | `<\|audio\|>` (open) | `256000` | Beginning of Audio block |
| $\langle\text{image}\rangle$ | `<\|image\|>` (placeholder) | `258880` | Soft token slot (repeated 280 times per image) |
| $\langle\text{audio}\rangle$ | `<\|audio\|>` (placeholder) | `258881` | Soft token slot (repeated 25 times per audio second) |
| $\langle\text{eoi}\rangle$ | `<image\|>` (close) | `258882` | End of Image block |
| $\langle\text{eoa}\rangle$ | `<audio\|>` (close) | `258883` | End of Audio block |

### Splicing Rules
1. Replace all $\langle\text{image}\rangle$ embeddings with the normalized soft tokens from the Vision Tower.
2. Replace all $\langle\text{audio}\rangle$ embeddings with the normalized soft tokens from the Audio Tower.
3. Scale all other vocabulary tokens (including $\langle\text{bos}\rangle$, $\langle\text{boi}\rangle$, $\langle\text{eoi}\rangle$, text prompt, and $\langle\text{eos}\rangle$) by:
   $$\text{embed\_scale} = \sqrt{d_{\text{hidden}}} = \sqrt{512} \approx 22.6274$$
4. Forward the spliced sequence through the language model using `forward_embeds()`.
