#!/usr/bin/env python3
"""
Apple Neural Engine (ANE) Core ML Exporter for Glypr / EmbeddingGemma 2.

Exports:
  1. SigLIP / Gemma-4 Vision Tower at static budget 140 (W8A16 linear symmetric INT8)
  2. Gemma 2 Text Backbone at static 512 tokens (W8A16 linear symmetric INT8)

Uses pure PyTorch modules mirroring `crates/glypr-core/src/multimodal/vision.rs`
and `model.rs` (the clean module pattern) to produce static MIL graphs that target
the Apple Neural Engine (ANE) via `ComputeUnit.CPU_AND_NE`.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
import coremltools.optimize.coreml as cto
import coremltools.converters.mil.frontend.torch.ops as torch_ops
from coremltools.converters.mil.mil import Builder as mb

# ---------------------------------------------------------------------------
# NumPy 2.0+ Compatibility Patch for coremltools
# ---------------------------------------------------------------------------
orig_cast = torch_ops._cast

def patched_cast(context, node, dtype, dtype_name):
    inputs = torch_ops._get_inputs(context, node, expected=1)
    x = inputs[0]
    if x.can_be_folded_to_const():
        v = x.val.item() if hasattr(x.val, "item") and x.val.size == 1 else x.val
        if not isinstance(v, dtype):
            res = mb.const(val=dtype(v), name=node.name)
        else:
            res = x
        context.add(res, node.name)
    else:
        orig_cast(context, node, dtype, dtype_name)

torch_ops._cast = patched_cast


# ===========================================================================
# 1. Common Neural Network Modules
# ===========================================================================

class CleanRMSNorm(nn.Module):
    """Numerically stable RMSNorm in float32 with optional learnable scale weight."""

    def __init__(self, dim: int, eps: float = 1e-6, has_weight: bool = True):
        super().__init__()
        self.eps = eps
        if has_weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            normed = normed * self.weight.float()
        return normed.to(orig_dtype)


# ===========================================================================
# 2. Vision Tower Subsystem (170M SigLIP / Gemma-4 Vision)
# ===========================================================================

class CleanVisionAttention(nn.Module):
    """Multi-Head Self-Attention with Q/K RMSNorm and 2D Axial RoPE."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 12, head_dim: int = 64, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = CleanRMSNorm(head_dim, eps=eps, has_weight=True)
        self.k_norm = CleanRMSNorm(head_dim, eps=eps, has_weight=True)
        self.v_norm = CleanRMSNorm(head_dim, eps=eps, has_weight=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)  # [b, heads, s, head_dim]
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_norm(v).transpose(1, 2)

        # 2D Axial RoPE: cos, sin are [1, 1, s, head_dim]
        def apply_rope_2d(t):
            h0, h1 = t[..., 0:16], t[..., 16:32]
            w0, w1 = t[..., 32:48], t[..., 48:64]
            rot_t = torch.cat([-h1, h0, -w1, w0], dim=-1)
            return t * cos + rot_t * sin

        q = apply_rope_2d(q)
        k = apply_rope_2d(k)

        # Non-causal unscaled attention (scale=1.0)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)

        context = context.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(context)


class CleanVisionEncoderLayer(nn.Module):
    """Vision Transformer Layer: Pre-Norm Attn + Pre-Norm MLP with Residual Adds."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 12, head_dim: int = 64, intermediate_size: int = 3072, eps: float = 1e-6):
        super().__init__()
        self.ln1 = CleanRMSNorm(hidden_size, eps=eps)
        self.self_attn = CleanVisionAttention(hidden_size, num_heads, head_dim, eps=eps)
        self.attn_post_norm = CleanRMSNorm(hidden_size, eps=eps)
        self.ln2 = CleanRMSNorm(hidden_size, eps=eps)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.ffn_post_norm = CleanRMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(self.ln1(x), cos, sin)
        x = x + self.attn_post_norm(attn_out)

        normed = self.ln2(x)
        gate = self.gate_proj(normed)
        up = self.up_proj(normed)
        act = F.gelu(gate, approximate="tanh") * up
        mlp_out = self.down_proj(act)
        x = x + self.ffn_post_norm(mlp_out)
        return x


class CleanGemma2VisionTower(nn.Module):
    """Full Vision Tower: Patch Embedder + 16 ViT Layers + 3x3 Pooler + Projector."""

    def __init__(
        self,
        patch_h: int = 42,
        patch_w: int = 27,
        hidden_size: int = 768,
        num_layers: int = 16,
        num_heads: int = 12,
        head_dim: int = 64,
        intermediate_size: int = 3072,
        out_features: int = 512,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_patches = patch_h * patch_w
        self.hidden_size = hidden_size
        self.out_h = patch_h // 3
        self.out_w = patch_w // 3
        self.num_soft_tokens = self.out_h * self.out_w

        self.input_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.register_buffer("pos_emb", torch.zeros(1, self.num_patches, hidden_size))
        self.register_buffer("cos", torch.zeros(1, 1, self.num_patches, head_dim))
        self.register_buffer("sin", torch.zeros(1, 1, self.num_patches, head_dim))

        self.layers = nn.ModuleList([
            CleanVisionEncoderLayer(hidden_size, num_heads, head_dim, intermediate_size, eps=eps)
            for _ in range(num_layers)
        ])

        self.proj_norm = CleanRMSNorm(hidden_size, eps=eps, has_weight=False)
        self.proj = nn.Linear(hidden_size, out_features, bias=False)

    def init_static_buffers(self, pos_table: torch.Tensor, rope_theta: float = 100.0):
        # 1. Position table lookup: pos_table is [2, 10240, 768]
        pos_emb = torch.zeros(1, self.num_patches, self.hidden_size)
        idx = 0
        for py in range(self.patch_h):
            for px in range(self.patch_w):
                pos_emb[0, idx] = pos_table[0, px] + pos_table[1, py]
                idx += 1
        self.pos_emb.copy_(pos_emb)

        # 2. 2D Axial RoPE: head_dim=64 -> spatial_dim=32 -> num_freqs=16
        spatial_dim = 32
        num_freqs = 16
        inv_freq = [1.0 / (rope_theta ** (2 * i / spatial_dim)) for i in range(num_freqs)]

        cos_t = torch.zeros(1, 1, self.num_patches, 64)
        sin_t = torch.zeros(1, 1, self.num_patches, 64)

        idx = 0
        for py in range(self.patch_h):
            for px in range(self.patch_w):
                for i in range(num_freqs):
                    th_x = px * inv_freq[i]
                    th_y = py * inv_freq[i]
                    cx, sx = math.cos(th_x), math.sin(th_x)
                    cy, sy = math.cos(th_y), math.sin(th_y)
                    # X coords: indices 0..16, 16..32
                    cos_t[0, 0, idx, i] = cx
                    cos_t[0, 0, idx, i + 16] = cx
                    sin_t[0, 0, idx, i] = sx
                    sin_t[0, 0, idx, i + 16] = sx
                    # Y coords: indices 32..48, 48..64
                    cos_t[0, 0, idx, i + 32] = cy
                    cos_t[0, 0, idx, i + 48] = cy
                    sin_t[0, 0, idx, i + 32] = sy
                    sin_t[0, 0, idx, i + 48] = sy
                idx += 1

        self.cos.copy_(cos_t)
        self.sin.copy_(sin_t)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [1, num_patches, 768] in [-1.0, 1.0]
        x = self.input_proj(pixel_values) + self.pos_emb

        for layer in self.layers:
            x = layer(x, self.cos, self.sin)

        # 2D Spatial pooling: AvgPool2d(3, stride=3) + sqrt(hidden_size) scaling
        x_spatial = x.view(1, self.patch_h, self.patch_w, self.hidden_size).permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(x_spatial, kernel_size=3, stride=3)
        pooled = pooled.permute(0, 2, 3, 1).reshape(1, self.num_soft_tokens, self.hidden_size)
        pooled = pooled * math.sqrt(self.hidden_size)

        normed = self.proj_norm(pooled)
        return self.proj(normed)


# ===========================================================================
# 3. Text Backbone Subsystem (270M Gemma 2 Text Model)
# ===========================================================================

class CleanTextAttention(nn.Module):
    """Grouped-Query Attention with Q/K RMSNorm, unweighted V RMSNorm, and 1D RoPE."""

    def __init__(self, seq_len: int, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = CleanRMSNorm(head_dim, eps=eps, has_weight=True)
        self.k_norm = CleanRMSNorm(head_dim, eps=eps, has_weight=True)
        self.v_norm = CleanRMSNorm(head_dim, eps=eps, has_weight=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x).reshape(1, self.seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(1, self.seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(1, self.seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)  # [1, num_heads, seq_len, head_dim]
        k = self.k_norm(k).transpose(1, 2)  # [1, num_kv_heads, seq_len, head_dim]
        v = self.v_norm(v).transpose(1, 2)  # [1, num_kv_heads, seq_len, head_dim]

        # 1D RoPE
        def rotate_half(t):
            d = t.shape[-1]
            return torch.cat([-t[..., d // 2:], t[..., :d // 2]], dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # GQA KV expansion
        if self.num_groups > 1:
            k = torch.repeat_interleave(k, self.num_groups, dim=1)
            v = torch.repeat_interleave(v, self.num_groups, dim=1)

        # Unscaled non-causal attention
        scores = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            scores = scores + mask
        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)

        ctx = ctx.transpose(1, 2).reshape(1, self.seq_len, self.num_heads * self.head_dim)
        return self.o_proj(ctx)


class CleanPLEBlock(nn.Module):
    """Per-Layer Embeddings (PLE) Mixing Block."""

    def __init__(self, hidden_size: int = 512, eps: float = 1e-6):
        super().__init__()
        self.per_layer_input_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.per_layer_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.post_per_layer_input_norm = CleanRMSNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor, per_layer_input: torch.Tensor) -> torch.Tensor:
        gate = self.per_layer_input_gate(hidden_states)
        act = F.gelu(gate, approximate="tanh") * per_layer_input
        proj = self.per_layer_projection(act)
        normed = self.post_per_layer_input_norm(proj)
        return hidden_states + normed


class CleanTextEncoderLayer(nn.Module):
    """Gemma 2 Decoder Layer: Attn + MLP + PLE Block + Layer Scalar."""

    def __init__(
        self,
        seq_len: int,
        hidden_size: int = 512,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 256,
        intermediate_size: int = 2048,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_layernorm = CleanRMSNorm(hidden_size, eps=eps)
        self.self_attn = CleanTextAttention(seq_len, hidden_size, num_heads, num_kv_heads, head_dim, eps=eps)
        self.post_attention_layernorm = CleanRMSNorm(hidden_size, eps=eps)

        self.pre_feedforward_layernorm = CleanRMSNorm(hidden_size, eps=eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.post_feedforward_layernorm = CleanRMSNorm(hidden_size, eps=eps)

        self.ple_block = CleanPLEBlock(hidden_size, eps=eps)
        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, per_layer_input: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(self.input_layernorm(x), mask, cos, sin)
        x = x + self.post_attention_layernorm(attn_out)

        normed = self.pre_feedforward_layernorm(x)
        gate = self.gate_proj(normed)
        up = self.up_proj(normed)
        act = F.gelu(gate, approximate="tanh") * up
        mlp_out = self.down_proj(act)
        x = x + self.post_feedforward_layernorm(mlp_out)

        x = self.ple_block(x, per_layer_input)
        return x * self.layer_scalar


class CleanGemma2TextBackbone(nn.Module):
    """Full Gemma 2 Text Model with Mean Pooling & L2 Normalization."""

    def __init__(
        self,
        seq_len: int = 512,
        vocab_size: int = 262144,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        out_features: int = 768,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_scale = math.sqrt(hidden_size)

        self.ple_proj = nn.Linear(hidden_size, 24 * hidden_size, bias=False)
        self.ple_norm = CleanRMSNorm(hidden_size, eps=eps)

        self.register_buffer("cos_sliding", torch.zeros(1, 1, seq_len, 256))
        self.register_buffer("sin_sliding", torch.zeros(1, 1, seq_len, 256))
        self.register_buffer("cos_full", torch.zeros(1, 1, seq_len, 512))
        self.register_buffer("sin_full", torch.zeros(1, 1, seq_len, 512))

        self.layers = nn.ModuleList()
        for i in range(24):
            if (i + 1) % 6 == 0:
                layer = CleanTextEncoderLayer(seq_len, hidden_size, num_heads=4, num_kv_heads=1, head_dim=512, intermediate_size=intermediate_size, eps=eps)
            else:
                layer = CleanTextEncoderLayer(seq_len, hidden_size, num_heads=4, num_kv_heads=2, head_dim=256, intermediate_size=intermediate_size, eps=eps)
            self.layers.append(layer)

        self.norm = CleanRMSNorm(hidden_size, eps=eps)
        self.embedding_projection = nn.Linear(hidden_size, out_features, bias=False)

    def init_static_rope(self, theta_sliding: float = 10000.0, theta_full: float = 1000000.0):
        pos = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)

        dim_s = 256
        inv_freq_s = 1.0 / (theta_sliding ** (torch.arange(0, dim_s, 2, dtype=torch.float) / dim_s))
        freqs_s = pos * inv_freq_s
        emb_s = torch.cat([freqs_s, freqs_s], dim=-1).unsqueeze(0).unsqueeze(0)
        self.cos_sliding.copy_(emb_s.cos())
        self.sin_sliding.copy_(emb_s.sin())

        dim_f = 512
        inv_freq_f = 1.0 / (theta_full ** (torch.arange(0, dim_f, 2, dtype=torch.float) / dim_f))
        freqs_f = pos * inv_freq_f
        emb_f = torch.cat([freqs_f, freqs_f], dim=-1).unsqueeze(0).unsqueeze(0)
        self.cos_full.copy_(emb_f.cos())
        self.sin_full.copy_(emb_f.sin())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: [1, seq_len] (long)
        # attention_mask: [1, seq_len] (1.0 for valid, 0.0 for padding)
        x = self.embed_tokens(input_ids) * self.embed_scale

        ple_raw = self.ple_proj(x) * (1.0 / self.embed_scale)
        ple_reshaped = ple_raw.reshape(1, self.seq_len, 24, self.hidden_size)
        ple_inputs = self.ple_norm(ple_reshaped)

        # Broadcastable additive mask: [1, 1, 1, seq_len]
        additive_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0

        for i, layer in enumerate(self.layers):
            per_layer_slice = ple_inputs[:, :, i, :]
            if (i + 1) % 6 == 0:
                cos, sin = self.cos_full, self.sin_full
            else:
                cos, sin = self.cos_sliding, self.sin_sliding
            x = layer(x, per_layer_slice, additive_mask, cos, sin)

        normed = self.norm(x)
        token_embs = self.embedding_projection(normed)

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1)
        sum_embs = (token_embs * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embs / sum_mask

        # L2 normalize to unit length
        return pooled / pooled.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)


# ===========================================================================
# 4. Model Loading & Weight Migration
# ===========================================================================

def resolve_safetensors_path(model_id: str) -> Path:
    """Locate local cached model.safetensors or download via HuggingFace Hub."""
    direct_path = Path(model_id)
    if direct_path.is_file() and direct_path.name.endswith(".safetensors"):
        return direct_path

    # Check common cache locations
    search_roots = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HF_HOME", "") + "/hub" if os.environ.get("HF_HOME") else None,
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    repo_folder = f"models--{model_id.replace('/', '--')}"

    for root in search_roots:
        if root and os.path.isdir(root):
            cand = Path(root) / repo_folder / "snapshots"
            if cand.is_dir():
                for snapshot in cand.iterdir():
                    st_file = snapshot / "model.safetensors"
                    if st_file.is_file():
                        print(f"Found cached safetensors at: {st_file}")
                        return st_file

    # Fallback to downloading
    print(f"Downloading {model_id}/model.safetensors via huggingface_hub...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    return Path(path)


def load_clean_vision_model(st_weights: Dict[str, torch.Tensor], patch_h: int, patch_w: int) -> CleanGemma2VisionTower:
    """Instantiate CleanGemma2VisionTower and transfer weights from state dict."""
    model = CleanGemma2VisionTower(patch_h=patch_h, patch_w=patch_w)

    model.input_proj.weight.data.copy_(st_weights["vision_tower.patch_embedder.input_proj.weight"].float())
    pos_table = st_weights["vision_tower.patch_embedder.position_embedding_table"].float()
    model.init_static_buffers(pos_table)

    for i in range(16):
        prefix = f"vision_tower.encoder.layers.{i}."
        dst = model.layers[i]

        dst.ln1.weight.data.copy_(st_weights[f"{prefix}input_layernorm.weight"].float())
        dst.self_attn.q_proj.weight.data.copy_(st_weights[f"{prefix}self_attn.q_proj.linear.weight"].float())
        dst.self_attn.k_proj.weight.data.copy_(st_weights[f"{prefix}self_attn.k_proj.linear.weight"].float())
        dst.self_attn.v_proj.weight.data.copy_(st_weights[f"{prefix}self_attn.v_proj.linear.weight"].float())
        dst.self_attn.o_proj.weight.data.copy_(st_weights[f"{prefix}self_attn.o_proj.linear.weight"].float())
        dst.self_attn.q_norm.weight.data.copy_(st_weights[f"{prefix}self_attn.q_norm.weight"].float())
        dst.self_attn.k_norm.weight.data.copy_(st_weights[f"{prefix}self_attn.k_norm.weight"].float())
        dst.attn_post_norm.weight.data.copy_(st_weights[f"{prefix}post_attention_layernorm.weight"].float())
        dst.ln2.weight.data.copy_(st_weights[f"{prefix}pre_feedforward_layernorm.weight"].float())

        dst.gate_proj.weight.data.copy_(st_weights[f"{prefix}mlp.gate_proj.linear.weight"].float())
        dst.up_proj.weight.data.copy_(st_weights[f"{prefix}mlp.up_proj.linear.weight"].float())
        dst.down_proj.weight.data.copy_(st_weights[f"{prefix}mlp.down_proj.linear.weight"].float())
        dst.ffn_post_norm.weight.data.copy_(st_weights[f"{prefix}post_feedforward_layernorm.weight"].float())

    model.proj.weight.data.copy_(st_weights["embed_vision.embedding_projection.weight"].float())
    model.float().eval()
    return model


def load_clean_text_model(st_weights: Dict[str, torch.Tensor], seq_len: int) -> CleanGemma2TextBackbone:
    """Instantiate CleanGemma2TextBackbone and transfer weights from state dict."""
    model = CleanGemma2TextBackbone(seq_len=seq_len)
    model.init_static_rope()

    model.embed_tokens.weight.data.copy_(st_weights["language_model.embed_tokens.weight"].float())
    model.ple_proj.weight.data.copy_(st_weights["language_model.ple.per_layer_model_projection.weight"].float())
    model.ple_norm.weight.data.copy_(st_weights["language_model.ple.per_layer_projection_norm.weight"].float())

    for i in range(24):
        prefix = f"language_model.layers.{i}."
        dst = model.layers[i]

        dst.input_layernorm.weight.data.copy_(st_weights[f"{prefix}input_layernorm.weight"].float())
        dst.self_attn.q_proj.weight.data.copy_(st_weights[f"{prefix}self_attn.q_proj.weight"].float())
        dst.self_attn.k_proj.weight.data.copy_(st_weights[f"{prefix}self_attn.k_proj.weight"].float())
        dst.self_attn.v_proj.weight.data.copy_(st_weights[f"{prefix}self_attn.v_proj.weight"].float())
        dst.self_attn.o_proj.weight.data.copy_(st_weights[f"{prefix}self_attn.o_proj.weight"].float())
        dst.self_attn.q_norm.weight.data.copy_(st_weights[f"{prefix}self_attn.q_norm.weight"].float())
        dst.self_attn.k_norm.weight.data.copy_(st_weights[f"{prefix}self_attn.k_norm.weight"].float())
        dst.post_attention_layernorm.weight.data.copy_(st_weights[f"{prefix}post_attention_layernorm.weight"].float())

        dst.pre_feedforward_layernorm.weight.data.copy_(st_weights[f"{prefix}pre_feedforward_layernorm.weight"].float())
        dst.gate_proj.weight.data.copy_(st_weights[f"{prefix}mlp.gate_proj.weight"].float())
        dst.up_proj.weight.data.copy_(st_weights[f"{prefix}mlp.up_proj.weight"].float())
        dst.down_proj.weight.data.copy_(st_weights[f"{prefix}mlp.down_proj.weight"].float())
        dst.post_feedforward_layernorm.weight.data.copy_(st_weights[f"{prefix}post_feedforward_layernorm.weight"].float())

        dst.ple_block.per_layer_input_gate.weight.data.copy_(st_weights[f"{prefix}ple_block.per_layer_input_gate.weight"].float())
        dst.ple_block.per_layer_projection.weight.data.copy_(st_weights[f"{prefix}ple_block.per_layer_projection.weight"].float())
        dst.ple_block.post_per_layer_input_norm.weight.data.copy_(st_weights[f"{prefix}ple_block.post_per_layer_input_norm.weight"].float())
        dst.layer_scalar.data.copy_(st_weights[f"{prefix}layer_scalar"].float())

    model.norm.weight.data.copy_(st_weights["language_model.norm.weight"].float())
    model.embedding_projection.weight.data.copy_(st_weights["language_model.embedding_projection.weight"].float())
    model.float().eval()
    return model


# ===========================================================================
# 5. Export Pipelines
# ===========================================================================

def compile_mlpackage(mlpackage_path: Path) -> Path:
    """Compile .mlpackage to .mlmodelc using CoreML framework via coremltools."""
    compiled_name = mlpackage_path.stem + ".mlmodelc"
    out_dir = mlpackage_path.parent
    target_path = out_dir / compiled_name
    print(f"Compiling Core ML model via ct.models.utils.compile_model...")
    t0 = time.time()
    compiled_temp = ct.models.utils.compile_model(str(mlpackage_path))
    if target_path.exists():
        shutil.rmtree(target_path)
    shutil.copytree(compiled_temp, target_path)
    print(f"Compilation finished in {time.time()-t0:.2f}s -> {target_path}")
    return target_path


def export_vision_tower(
    st_weights: Dict[str, torch.Tensor],
    output_dir: Path,
    patch_h: int = 42,
    patch_w: int = 27,
    quantize: str = "w8a16",
    do_compile: bool = True,
    do_verify: bool = True,
) -> Path:
    num_patches = patch_h * patch_w
    num_soft_tokens = (patch_h // 3) * (patch_w // 3)
    target_name = f"gemma2_vision_tower_p{num_patches}_t{num_soft_tokens}_{quantize}"
    package_path = output_dir / f"{target_name}.mlpackage"

    print("---------------------------------------------------------------------------")
    print(f"Exporting Vision Tower: {patch_h}x{patch_w} patches ({num_patches} patches -> {num_soft_tokens} soft tokens)")
    print(f"Destination: {package_path}")
    print("---------------------------------------------------------------------------")

    model = load_clean_vision_model(st_weights, patch_h=patch_h, patch_w=patch_w)
    dummy_pixels = torch.randn(1, num_patches, 768)

    with torch.no_grad():
        py_out = model(dummy_pixels)

    print("Tracing CleanVisionTower...")
    traced = torch.jit.trace(model, (dummy_pixels,))

    print("Converting to Core ML FP16...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=dummy_pixels.shape, dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    if quantize == "w8a16":
        print("Applying W8A16 linear symmetric weight quantization...")
        opt_config = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric")
        )
        mlmodel = cto.linear_quantize_weights(mlmodel, config=opt_config)

    if package_path.exists():
        shutil.rmtree(package_path)
    mlmodel.save(str(package_path))
    print(f"Saved .mlpackage to: {package_path}")

    if do_verify:
        print("Verifying Core ML inference against PyTorch ground truth (CPU_AND_NE)...")
        coreml_out = mlmodel.predict({"pixel_values": dummy_pixels.numpy()})
        out_key = list(coreml_out.keys())[0]
        c_out = torch.from_numpy(coreml_out[out_key])
        cos_sim = F.cosine_similarity(py_out.flatten(), c_out.flatten(), dim=0).item()
        print(f"Vision Tower Cosine Similarity vs PyTorch FP32: {cos_sim:.6f}")
        assert cos_sim > 0.998, f"Vision parity failed: {cos_sim} < 0.998"

    if do_compile:
        compile_mlpackage(package_path)

    return package_path


def export_text_backbone(
    st_weights: Dict[str, torch.Tensor],
    output_dir: Path,
    seq_len: int = 512,
    quantize: str = "w8a16",
    do_compile: bool = True,
    do_verify: bool = True,
) -> Path:
    target_name = f"gemma2_text_backbone_s{seq_len}_{quantize}"
    package_path = output_dir / f"{target_name}.mlpackage"

    print("---------------------------------------------------------------------------")
    print(f"Exporting Text Backbone: static seq_len={seq_len} tokens")
    print(f"Destination: {package_path}")
    print("---------------------------------------------------------------------------")

    model = load_clean_text_model(st_weights, seq_len=seq_len)

    # Realistic prompt tokens ("task: SearchQuery | query: What causes the northern lights?")
    sample_tokens = [2, 603, 235292, 1146, 235284, 1489, 235292, 235248, 1841, 6496, 573, 11956, 5028, 235336]
    input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    attention_mask = torch.zeros(1, seq_len, dtype=torch.float32)
    valid_len = min(len(sample_tokens), seq_len)
    input_ids[0, :valid_len] = torch.tensor(sample_tokens[:valid_len], dtype=torch.long)
    attention_mask[0, :valid_len] = 1.0

    with torch.no_grad():
        py_out = model(input_ids, attention_mask)

    print("Tracing CleanGemma2TextBackbone...")
    traced = torch.jit.trace(model, (input_ids, attention_mask))

    print("Converting to Core ML FP16...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.float32),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    if quantize == "w8a16":
        print("Applying W8A16 linear symmetric weight quantization...")
        opt_config = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric")
        )
        mlmodel = cto.linear_quantize_weights(mlmodel, config=opt_config)

    if package_path.exists():
        shutil.rmtree(package_path)
    mlmodel.save(str(package_path))
    print(f"Saved .mlpackage to: {package_path}")

    if do_verify:
        print("Verifying Core ML inference against PyTorch ground truth (CPU_AND_NE)...")
        coreml_out = mlmodel.predict({
            "input_ids": input_ids.numpy().astype(np.int32),
            "attention_mask": attention_mask.numpy().astype(np.float32),
        })
        out_key = list(coreml_out.keys())[0]
        c_out = torch.from_numpy(coreml_out[out_key])
        cos_sim = F.cosine_similarity(py_out.flatten(), c_out.flatten(), dim=0).item()
        print(f"Text Backbone Cosine Similarity vs PyTorch FP32: {cos_sim:.6f}")
        assert cos_sim > 0.995, f"Text parity failed: {cos_sim} < 0.995"

    if do_compile:
        compile_mlpackage(package_path)

    return package_path


# ===========================================================================
# 6. Main CLI Entrypoint
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Export EmbeddingGemma 2 Subsystems to Core ML for Apple Neural Engine (ANE)")
    parser.add_argument("--model-id", type=str, default="google/embeddinggemma-2", help="Hugging Face model ID or path to safetensors")
    parser.add_argument("--output-dir", type=Path, default=Path("models/coreml"), help="Directory for exported .mlpackage and .mlmodelc")
    parser.add_argument("--target", choices=["all", "vision", "text"], default="all", help="Subsystems to export")
    parser.add_argument("--vision-preset", choices=["140-mtg", "140-std"], default="140-mtg",
                        help="Vision patch grid preset: '140-mtg' (42x27=1134 patches -> 126 soft tokens for 480x680 images), or '140-std' (42x30=1260 patches -> 140 soft tokens)")
    parser.add_argument("--text-seq-len", type=int, default=512, help="Static sequence length for resident RAG text model (default: 512)")
    parser.add_argument("--quantize", choices=["w8a16", "fp16"], default="w8a16", help="Linear quantization mode (w8a16 recommended for ANE)")
    parser.add_argument("--no-compile", action="store_true", help="Skip compilation to .mlmodelc")
    parser.add_argument("--no-verify", action="store_true", help="Skip parity verification against PyTorch")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("===========================================================================")
    print("  Glypr Apple Neural Engine (ANE) Core ML Exporter — Phase A")
    print("===========================================================================")
    print(f"  Model ID       : {args.model_id}")
    print(f"  Target         : {args.target}")
    print(f"  Vision Preset  : {args.vision_preset}")
    print(f"  Text Seq Len   : {args.text_seq_len}")
    print(f"  Quantization   : {args.quantize}")
    print(f"  Output Dir     : {args.output_dir.resolve()}")
    print("===========================================================================")

    st_path = resolve_safetensors_path(args.model_id)
    t0 = time.time()
    print(f"Loading weights from: {st_path} ...")
    st_weights = safetensors.torch.load_file(str(st_path))
    print(f"Loaded {len(st_weights)} tensors in {time.time()-t0:.2f}s")

    if args.vision_preset == "140-mtg":
        patch_h, patch_w = 42, 27
    else:
        patch_h, patch_w = 42, 30

    if args.target in ["all", "vision"]:
        export_vision_tower(
            st_weights=st_weights,
            output_dir=args.output_dir,
            patch_h=patch_h,
            patch_w=patch_w,
            quantize=args.quantize,
            do_compile=not args.no_compile,
            do_verify=not args.no_verify,
        )

    if args.target in ["all", "text"]:
        export_text_backbone(
            st_weights=st_weights,
            output_dir=args.output_dir,
            seq_len=args.text_seq_len,
            quantize=args.quantize,
            do_compile=not args.no_compile,
            do_verify=not args.no_verify,
        )

    print("\n===========================================================================")
    print("  Phase A Core ML Export Complete!")
    print(f"  Artifacts saved in: {args.output_dir.resolve()}")
    print("===========================================================================")


if __name__ == "__main__":
    main()
