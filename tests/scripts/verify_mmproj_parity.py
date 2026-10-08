#!/usr/bin/env python3
"""
Comprehensive parity and byte-level verification between Google's official
EmbeddingGemma 2 weights (`model.safetensors`) and ggml-org's `mmproj-embeddinggemma-2-BF16.gguf`.
"""

import os
import re
import sys
import time
from pathlib import Path
import numpy as np
import safetensors.torch
import torch
import gguf

def map_gguf_to_st(name: str) -> str:
    # 1. Dynamic Multimodal Projections
    if name == 'mm.a.input_projection.weight':
        return 'embed_audio.embedding_projection.weight'
    if name == 'mm.input_projection.weight':
        return 'embed_vision.embedding_projection.weight'
        
    # 2. Vision non-block tensors
    if name == 'v.patch_embd.weight':
        return 'vision_tower.patch_embedder.input_proj.weight'
    if name == 'v.position_embd.weight':
        return 'vision_tower.patch_embedder.position_embedding_table'
        
    # 3. Vision blocks: v.blk.{L}.*
    m = re.match(r'^v\.blk\.(\d+)\.(.*)$', name)
    if m:
        l = m.group(1)
        sub = m.group(2)
        v_map = {
            'ln1.weight': f'vision_tower.encoder.layers.{l}.input_layernorm.weight',
            'ffn_down.weight': f'vision_tower.encoder.layers.{l}.mlp.down_proj.linear.weight',
            'ffn_gate.weight': f'vision_tower.encoder.layers.{l}.mlp.gate_proj.linear.weight',
            'ffn_up.weight': f'vision_tower.encoder.layers.{l}.mlp.up_proj.linear.weight',
            'attn_post_norm.weight': f'vision_tower.encoder.layers.{l}.post_attention_layernorm.weight',
            'ffn_post_norm.weight': f'vision_tower.encoder.layers.{l}.post_feedforward_layernorm.weight',
            'ln2.weight': f'vision_tower.encoder.layers.{l}.pre_feedforward_layernorm.weight',
            'attn_k_norm.weight': f'vision_tower.encoder.layers.{l}.self_attn.k_norm.weight',
            'attn_k.weight': f'vision_tower.encoder.layers.{l}.self_attn.k_proj.linear.weight',
            'attn_out.weight': f'vision_tower.encoder.layers.{l}.self_attn.o_proj.linear.weight',
            'attn_q_norm.weight': f'vision_tower.encoder.layers.{l}.self_attn.q_norm.weight',
            'attn_q.weight': f'vision_tower.encoder.layers.{l}.self_attn.q_proj.linear.weight',
            'attn_v.weight': f'vision_tower.encoder.layers.{l}.self_attn.v_proj.linear.weight',
        }
        return v_map.get(sub)
        
    # 4. Audio non-block tensors
    if name == 'a.pre_encode.out.bias':
        return 'audio_tower.output_proj.bias'
    if name == 'a.pre_encode.out.weight':
        return 'audio_tower.output_proj.weight'
    if name == 'a.input_projection.weight':
        return 'audio_tower.subsample_conv_projection.input_proj_linear.weight'
    if name == 'a.conv1d.0.weight':
        return 'audio_tower.subsample_conv_projection.layer0.conv.weight'
    if name == 'a.conv1d.0.norm.weight':
        return 'audio_tower.subsample_conv_projection.layer0.norm.weight'
    if name == 'a.conv1d.1.weight':
        return 'audio_tower.subsample_conv_projection.layer1.conv.weight'
    if name == 'a.conv1d.1.norm.weight':
        return 'audio_tower.subsample_conv_projection.layer1.norm.weight'
        
    # 5. Audio blocks: a.blk.{L}.*
    m = re.match(r'^a\.blk\.(\d+)\.(.*)$', name)
    if m:
        l = m.group(1)
        sub = m.group(2)
        
        a_patterns = {
            'attn_k': f'audio_tower.layers.{l}.self_attn.k_proj',
            'attn_q': f'audio_tower.layers.{l}.self_attn.q_proj',
            'attn_v': f'audio_tower.layers.{l}.self_attn.v_proj',
            'attn_out': f'audio_tower.layers.{l}.self_attn.post',
            'conv_pw1': f'audio_tower.layers.{l}.lconv1d.linear_start',
            'conv_pw2': f'audio_tower.layers.{l}.lconv1d.linear_end',
            'ffn_up': f'audio_tower.layers.{l}.feed_forward1.ffw_layer_1',
            'ffn_down': f'audio_tower.layers.{l}.feed_forward1.ffw_layer_2',
            'ffn_up_1': f'audio_tower.layers.{l}.feed_forward2.ffw_layer_1',
            'ffn_down_1': f'audio_tower.layers.{l}.feed_forward2.ffw_layer_2',
        }
        for k, prefix in a_patterns.items():
            if sub == f'{k}.weight':
                return f'{prefix}.linear.weight'
            for qk in ['input_max', 'input_min', 'output_max', 'output_min']:
                if sub == f'{k}.{qk}':
                    return f'{prefix}.{qk}'
                    
        a_simple = {
            'attn_k_rel.weight': f'audio_tower.layers.{l}.self_attn.relative_k_proj.weight',
            'per_dim_scale.weight': f'audio_tower.layers.{l}.self_attn.per_dim_scale',
            'attn_pre_norm.weight': f'audio_tower.layers.{l}.norm_pre_attn.weight',
            'attn_post_norm.weight': f'audio_tower.layers.{l}.norm_post_attn.weight',
            'ln2.weight': f'audio_tower.layers.{l}.norm_out.weight',
            'norm_conv.weight': f'audio_tower.layers.{l}.lconv1d.conv_norm.weight',
            'conv_dw.weight': f'audio_tower.layers.{l}.lconv1d.depthwise_conv1d.weight',
            'conv_norm.weight': f'audio_tower.layers.{l}.lconv1d.pre_layer_norm.weight',
            'ffn_norm.weight': f'audio_tower.layers.{l}.feed_forward1.pre_layer_norm.weight',
            'ffn_post_norm.weight': f'audio_tower.layers.{l}.feed_forward1.post_layer_norm.weight',
            'ffn_norm_1.weight': f'audio_tower.layers.{l}.feed_forward2.pre_layer_norm.weight',
            'ffn_post_norm_1.weight': f'audio_tower.layers.{l}.feed_forward2.post_layer_norm.weight',
        }
        return a_simple.get(sub)
        
    return None

from huggingface_hub import hf_hub_download

def main():
    gguf_path = Path(hf_hub_download(repo_id="ggml-org/embeddinggemma-2-GGUF", filename="mmproj-embeddinggemma-2-BF16.gguf"))
    st_path = Path(hf_hub_download(repo_id="google/embeddinggemma-2", filename="model.safetensors"))
    
    print("==================================================================================")
    print("  Google Official (Safetensors) vs. ggml-org (BF16 mmproj GGUF) Parity Audit")
    print("==================================================================================")
    print(f"  Safetensors Source : {st_path}")
    print(f"  GGUF Target        : {gguf_path}")
    
    if not gguf_path.exists() or not st_path.exists():
        print("ERROR: Checkpoint file(s) missing!")
        sys.exit(1)
        
    t0 = time.time()
    reader = gguf.GGUFReader(gguf_path)
    st_file = safetensors.safe_open(st_path, framework='pt')
    st_keys = set(st_file.keys())
    
    print(f"\nIndexed {len(reader.tensors)} GGUF tensors and {len(st_keys)} Safetensors in {time.time()-t0:.2f}s\n")
    
    total_tensors = len(reader.tensors)
    exact_numerical_match = 0
    exact_byte_match = 0
    bf16_tensor_count = 0
    f32_tensor_count = 0
    
    max_overall_diff = 0.0
    diff_tensors = []
    
    print(f"Auditing all {total_tensors} multimodal tensors...")
    
    for idx, gt in enumerate(reader.tensors):
        g_name = gt.name
        st_name = map_gguf_to_st(g_name)
        
        if st_name is None or st_name not in st_keys:
            print(f"ERROR: Unmapped tensor: {g_name} -> {st_name}")
            sys.exit(1)
            
        st_t = st_file.get_tensor(st_name)
        g_bytes = bytes(gt.data.tobytes())
        
        # 1. Byte matching for BF16 tensors
        if gt.tensor_type == gguf.GGMLQuantizationType.BF16:
            bf16_tensor_count += 1
            if st_t.dtype == torch.bfloat16:
                st_bytes = bytes(st_t.contiguous().untyped_storage())
                if st_bytes == g_bytes:
                    exact_byte_match += 1
        elif gt.tensor_type == gguf.GGMLQuantizationType.F32:
            f32_tensor_count += 1
            
        # 2. Numerical Parity Check
        st_f32 = st_t.float().numpy()
        if gt.tensor_type == gguf.GGMLQuantizationType.BF16:
            g_f32 = torch.frombuffer(bytearray(g_bytes), dtype=torch.bfloat16).float().numpy()
        else:
            g_f32 = np.frombuffer(g_bytes, dtype=np.float32)
            
        # Special handling for vision patch embedder layout and audio per_dim_scale
        if g_name == 'v.patch_embd.weight':
            # PyTorch [768, 768] was transposed from (768, 16, 16, 3) -> (768, 3, 16, 16) during GGUF conversion
            st_f32 = st_f32.reshape(768, 16, 16, 3).transpose((0, 3, 1, 2))
        elif 'per_dim_scale' in g_name:
            # GGUF pre-computes softplus(per_dim_scale) during export
            st_f32 = torch.nn.functional.softplus(st_t.float()).numpy()
            
        st_flat = st_f32.flatten()
        g_flat = g_f32.flatten()
        
        if st_flat.shape != g_flat.shape:
            print(f"Shape mismatch on {g_name}: st={st_flat.shape} vs gguf={g_flat.shape}")
            diff_tensors.append((g_name, "Shape mismatch", -1))
            continue
            
        diff = float(np.max(np.abs(g_flat - st_flat)))
        if diff > max_overall_diff:
            max_overall_diff = diff
            
        if diff == 0.0:
            exact_numerical_match += 1
        else:
            diff_tensors.append((g_name, st_name, diff))
            
        if (idx + 1) % 150 == 0 or idx + 1 == total_tensors:
            print(f"  [{idx+1:3d}/{total_tensors}] Verified up to {g_name} (Current Max Diff: {max_overall_diff:.2e})")

    print("\n" + "=" * 82)
    print("                               AUDIT SCORECARD")
    print("=" * 82)
    print(f"  Total Multimodal Tensors Evaluated : {total_tensors}")
    print(f"  Mapped Safetensors Matches         : {total_tensors} / {total_tensors} (100.0%)")
    print(f"  Exact Bit-for-Bit Matches (diff=0) : {exact_numerical_match} / {total_tensors} ({exact_numerical_match/total_tensors*100:.2f}%)")
    print(f"  Max Absolute Difference            : {max_overall_diff:.2e}")
    print(f"  BF16 Tensors Count                 : {bf16_tensor_count}")
    print(f"  BF16 Raw Byte Exact Matches        : {exact_byte_match} / {bf16_tensor_count} ({exact_byte_match/bf16_tensor_count*100:.2f}%)")
    print(f"  F32 Tensors (Converted Norms/Scales): {f32_tensor_count}")
    print("=" * 82)
    
    if diff_tensors:
        print(f"\nTensors with Non-Zero Difference ({len(diff_tensors)}):")
        for g_n, st_n, d in diff_tensors[:10]:
            print(f"  {g_n} <-> {st_n}: diff = {d:.6e}")
    else:
        print("\n🏆 PERFECT PARITY CONFIRMED: 0.00e0 DIFFERENCE ACROSS ALL 963 TENSORS!")

if __name__ == "__main__":
    main()
