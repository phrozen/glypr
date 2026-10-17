#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_BFLOAT__)
typedef bfloat MLX_BF16;
#else
struct MLX_BF16 {
    uint16_t bits_;
    MLX_BF16() thread = default;
    MLX_BF16() device = default;
    MLX_BF16() constant = default;
    
    MLX_BF16(float x) thread {
        union { float f; uint32_t i; } u;
        u.f = x;
        // Round to nearest even
        uint32_t lsb = (u.i >> 16) & 1;
        uint32_t rounding_bias = 0x7fff + lsb;
        bits_ = static_cast<uint16_t>((u.i + rounding_bias) >> 16);
    }
    
    operator float() const thread {
        union { uint32_t i; float f; } u;
        u.i = static_cast<uint32_t>(bits_) << 16;
        return u.f;
    }
    operator float() const device {
        union { uint32_t i; float f; } u;
        u.i = static_cast<uint32_t>(bits_) << 16;
        return u.f;
    }
    operator float() const constant {
        union { uint32_t i; float f; } u;
        u.i = static_cast<uint32_t>(bits_) << 16;
        return u.f;
    }
};
#endif

// =========================================================================
// 1. Fused RMSNorm Kernel (Direct Register Reduction via SIMD Shuffle)
// =========================================================================
kernel void rms_norm_bf16(
    device const MLX_BF16* x [[buffer(0)]],
    device const MLX_BF16* weight [[buffer(1)]],
    device MLX_BF16* out [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant int& D [[buffer(4)]],
    constant int& has_weight [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    device const MLX_BF16* x_row = x + size_t(row) * size_t(D);
    device MLX_BF16* out_row = out + size_t(row) * size_t(D);
    
    float thread_sum_sq = 0.0f;
    for (int i = int(tid); i < D; i += 32) {
        float v = float(x_row[i]);
        thread_sum_sq += v * v;
    }
    
    // In-register SIMD warp reduction across all 32 lanes
    float sum_sq = simd_sum(thread_sum_sq);
    float inv_rms = rsqrt(sum_sq / float(D) + eps);
    
    for (int i = int(tid); i < D; i += 32) {
        float v = float(x_row[i]);
        float normed = v * inv_rms;
        if (has_weight) {
            normed *= float(weight[i]);
        }
        out_row[i] = MLX_BF16(normed);
    }
}

// =========================================================================
// 2. Fused GeGLU Kernel (Single-Pass In-Register Tanh Activation & Mul)
// =========================================================================
kernel void geglu_bf16(
    device const MLX_BF16* gate [[buffer(0)]],
    device const MLX_BF16* up [[buffer(1)]],
    device MLX_BF16* out [[buffer(2)]],
    constant int& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (int(id) >= total_elements) return;
    
    float g = float(gate[id]);
    float u = float(up[id]);
    
    // Exact PyTorch gelu_pytorch_tanh formula:
    // 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Clamping g to [-10.0f, 10.0f] avoids tanh overflow in float (inner > 88.0f)
    // while being bit-for-bit identical (tanh saturates to +/- 1.0 beyond 9.0).
    float g_clamped = metal::clamp(g, -10.0f, 10.0f);
    float g3 = g_clamped * g_clamped * g_clamped;
    float inner = 0.7978845608028654f * (g_clamped + 0.044715f * g3);
    float tanh_val = metal::tanh(inner);
    float act = 0.5f * g * (1.0f + tanh_val);
    
    out[id] = MLX_BF16(act * u);
}

kernel void ple_geglu_bf16(
    device const MLX_BF16* gate [[buffer(0)]],
    device const MLX_BF16* up [[buffer(1)]],
    device MLX_BF16* out [[buffer(2)]],
    constant int& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    #pragma clang fp reassociate(off)
    if (int(id) >= total_elements) return;

    float g = float(gate[id]);
    float u = float(up[id]);

    float x2 = float(MLX_BF16(g * g));
    float x3 = float(MLX_BF16(x2 * g));
    float scaled_x3 = float(MLX_BF16(x3 * 0.044715f));
    float sum = float(MLX_BF16(g + scaled_x3));
    float inner = float(MLX_BF16(sum * 0.7978845608028654f));
    float t = float(MLX_BF16(metal::tanh(metal::clamp(inner, -10.0f, 10.0f))));
    float one_plus_t = float(MLX_BF16(t + 1.0f));
    float half_x = float(MLX_BF16(g * 0.5f));
    float act = float(MLX_BF16(half_x * one_plus_t));

    out[id] = MLX_BF16(act * u);
}

// =========================================================================
// 3. Fused Post-Norm + Residual Add (+ Optional Layer Scalar) Kernel
// =========================================================================
kernel void rms_norm_residual_add_bf16(
    device const MLX_BF16* x [[buffer(0)]],
    device const MLX_BF16* residual [[buffer(1)]],
    device const MLX_BF16* weight [[buffer(2)]],
    device const MLX_BF16* scale_buf [[buffer(3)]],
    device MLX_BF16* out [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant int& D [[buffer(6)]],
    constant int& has_weight [[buffer(7)]],
    constant int& has_scale [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    device const MLX_BF16* x_row = x + size_t(row) * size_t(D);
    device const MLX_BF16* res_row = residual + size_t(row) * size_t(D);
    device MLX_BF16* out_row = out + size_t(row) * size_t(D);

    float thread_sum_sq = 0.0f;
    for (int i = int(tid); i < D; i += 32) {
        float v = float(x_row[i]);
        thread_sum_sq += v * v;
    }

    float sum_sq = simd_sum(thread_sum_sq);
    float inv_rms = rsqrt(sum_sq / float(D) + eps);
    float layer_scale = has_scale ? float(scale_buf[0]) : 1.0f;

    for (int i = int(tid); i < D; i += 32) {
        float v = float(x_row[i]);
        float normed = v * inv_rms;
        if (has_weight) {
            normed *= float(weight[i]);
        }
        float normed_bf16 = float(MLX_BF16(normed));
        float sum_bf16 = float(MLX_BF16(float(res_row[i]) + normed_bf16));
        if (has_scale) {
            out_row[i] = MLX_BF16(sum_bf16 * layer_scale);
        } else {
            out_row[i] = MLX_BF16(sum_bf16);
        }
    }
}

// =========================================================================
// 4. Fused Q/K RMSNorm + RoPE + [B, S, H, D] -> [B, H, S, D] Transpose
// =========================================================================
kernel void rms_norm_rope_transpose_bf16(
    device const MLX_BF16* x [[buffer(0)]],
    device const MLX_BF16* weight [[buffer(1)]],
    device const MLX_BF16* cos [[buffer(2)]],
    device const MLX_BF16* sin [[buffer(3)]],
    device MLX_BF16* out [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant int& S [[buffer(6)]],
    constant int& H [[buffer(7)]],
    constant int& D [[buffer(8)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint h = tg_pos.x;
    uint s = tg_pos.y;
    uint b = tg_pos.z;

    size_t in_row = (size_t(b) * size_t(S) + size_t(s)) * size_t(H) + size_t(h);
    size_t out_row = (size_t(b) * size_t(H) + size_t(h)) * size_t(S) + size_t(s);

    device const MLX_BF16* x_row = x + in_row * size_t(D);
    device MLX_BF16* out_row_ptr = out + out_row * size_t(D);

    float thread_sum_sq = 0.0f;
    for (int i = int(tid); i < D; i += 32) {
        float v = float(x_row[i]);
        thread_sum_sq += v * v;
    }

    float sum_sq = simd_sum(thread_sum_sq);
    float inv_rms = rsqrt(sum_sq / float(D) + eps);

    int half_D = D / 2;
    device const MLX_BF16* cos_row = cos + size_t(s) * size_t(D);
    device const MLX_BF16* sin_row = sin + size_t(s) * size_t(D);

    for (int i = int(tid); i < half_D; i += 32) {
        #pragma clang fp reassociate(off)
        int j = i + half_D;
        float n1 = (float(x_row[i]) * inv_rms) * float(weight[i]);
        float n2 = (float(x_row[j]) * inv_rms) * float(weight[j]);
        float x1 = float(MLX_BF16(n1));
        float x2 = float(MLX_BF16(n2));
        float c1 = float(cos_row[i]);
        float s1 = float(sin_row[i]);
        float c2 = float(cos_row[j]);
        float s2 = float(sin_row[j]);
        float rot1 = float(MLX_BF16(x1 * c1)) + float(MLX_BF16((-x2) * s1));
        float rot2 = float(MLX_BF16(x2 * c2)) + float(MLX_BF16(x1 * s2));
        out_row_ptr[i] = MLX_BF16(rot1);
        out_row_ptr[j] = MLX_BF16(rot2);
    }
}

// =========================================================================
// 5. Fused V RMSNorm + [B, S, H, D] -> [B, H, S, D] Transpose
// =========================================================================
kernel void rms_norm_transpose_bf16(
    device const MLX_BF16* x [[buffer(0)]],
    device const MLX_BF16* weight [[buffer(1)]],
    device MLX_BF16* out [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant int& S [[buffer(4)]],
    constant int& H [[buffer(5)]],
    constant int& D [[buffer(6)]],
    constant int& has_weight [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint h = tg_pos.x;
    uint s = tg_pos.y;
    uint b = tg_pos.z;

    size_t in_row = (size_t(b) * size_t(S) + size_t(s)) * size_t(H) + size_t(h);
    size_t out_row = (size_t(b) * size_t(H) + size_t(h)) * size_t(S) + size_t(s);

    device const MLX_BF16* x_row = x + in_row * size_t(D);
    device MLX_BF16* out_row_ptr = out + out_row * size_t(D);

    float thread_sum_sq = 0.0f;
    for (int i = int(tid); i < D; i += 32) {
        float v = float(x_row[i]);
        thread_sum_sq += v * v;
    }

    float sum_sq = simd_sum(thread_sum_sq);
    float inv_rms = rsqrt(sum_sq / float(D) + eps);

    for (int i = int(tid); i < D; i += 32) {
        float v = float(x_row[i]);
        float normed = v * inv_rms;
        if (has_weight) {
            normed *= float(weight[i]);
        }
        out_row_ptr[i] = MLX_BF16(normed);
    }
}

// =========================================================================
// 6. Fused 4D PLE Scale + RMSNorm + [B, S, L, D] -> [L, B, S, D] Permute
// =========================================================================
kernel void ple_scale_rms_norm_transpose_bf16(
    device const MLX_BF16* x [[buffer(0)]],
    device const MLX_BF16* weight [[buffer(1)]],
    device MLX_BF16* out [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant int& B [[buffer(5)]],
    constant int& S [[buffer(6)]],
    constant int& L [[buffer(7)]],
    constant int& D [[buffer(8)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    #pragma clang fp reassociate(off)
    uint l = tg_pos.x;
    uint s = tg_pos.y;
    uint b = tg_pos.z;

    size_t in_row = (size_t(b) * size_t(S) + size_t(s)) * size_t(L) + size_t(l);
    size_t out_row = (size_t(l) * size_t(B) + size_t(b)) * size_t(S) + size_t(s);

    device const MLX_BF16* x_row = x + in_row * size_t(D);
    device MLX_BF16* out_row_ptr = out + out_row * size_t(D);

    float thread_sum_sq = 0.0f;
    for (int i = int(tid); i < D; i += 32) {
        float v = float(MLX_BF16(float(x_row[i]) * scale));
        thread_sum_sq += v * v;
    }

    float sum_sq = simd_sum(thread_sum_sq);
    float inv_rms = rsqrt(sum_sq / float(D) + eps);

    for (int i = int(tid); i < D; i += 32) {
        float v = float(MLX_BF16(float(x_row[i]) * scale));
        float normed = (v * inv_rms) * float(weight[i]);
        out_row_ptr[i] = MLX_BF16(normed);
    }
}

// =========================================================================
// 7. Fused Gate+Up GeGLU Kernel (Reads Concatenated [rows, 2*D] Directly)
// =========================================================================
kernel void geglu_gate_up_bf16(
    device const MLX_BF16* gate_up [[buffer(0)]],
    device MLX_BF16* out [[buffer(1)]],
    constant int& total_elements [[buffer(2)]],
    constant int& D [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (int(id) >= total_elements) return;

    int row = int(id) / D;
    int col = int(id) % D;
    size_t base = size_t(row) * size_t(2 * D) + size_t(col);

    float g = float(gate_up[base]);
    float u = float(gate_up[base + size_t(D)]);

    float g_clamped = metal::clamp(g, -10.0f, 10.0f);
    float g3 = g_clamped * g_clamped * g_clamped;
    float inner = 0.7978845608028654f * (g_clamped + 0.044715f * g3);
    float tanh_val = metal::tanh(inner);
    float act = 0.5f * g * (1.0f + tanh_val);

    out[id] = MLX_BF16(act * u);
}

// =========================================================================
// 8. Fused Single-Dispatch QKV RMSNorm + RoPE + Transpose (Text Attention)
// =========================================================================
kernel void qkv_rmsnorm_rope_transpose_bf16(
    device const MLX_BF16* qkv [[buffer(0)]],
    device const MLX_BF16* q_weight [[buffer(1)]],
    device const MLX_BF16* k_weight [[buffer(2)]],
    device const MLX_BF16* v_weight [[buffer(3)]],
    device const MLX_BF16* cos [[buffer(4)]],
    device const MLX_BF16* sin [[buffer(5)]],
    device MLX_BF16* q_out [[buffer(6)]],
    device MLX_BF16* k_out [[buffer(7)]],
    device MLX_BF16* v_out [[buffer(8)]],
    constant float& q_eps [[buffer(9)]],
    constant float& k_eps [[buffer(10)]],
    constant float& v_eps [[buffer(11)]],
    constant int& S [[buffer(12)]],
    constant int& H_q [[buffer(13)]],
    constant int& H_kv [[buffer(14)]],
    constant int& D [[buffer(15)]],
    constant int& has_v_weight [[buffer(16)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    int h_total = int(tg_pos.x);
    int s = int(tg_pos.y);
    int b = int(tg_pos.z);

    int total_heads = H_q + 2 * H_kv;
    size_t in_row = (size_t(b) * size_t(S) + size_t(s)) * size_t(total_heads) + size_t(h_total);
    device const MLX_BF16* x_row = qkv + in_row * size_t(D);

    float thread_sum_sq = 0.0f;
    for (int i = int(tid); i < D; i += 32) {
        float v = float(x_row[i]);
        thread_sum_sq += v * v;
    }
    float sum_sq = simd_sum(thread_sum_sq);

    if (h_total < H_q + H_kv) {
        bool is_q = (h_total < H_q);
        int h = is_q ? h_total : (h_total - H_q);
        int H_out = is_q ? H_q : H_kv;
        float eps = is_q ? q_eps : k_eps;
        device const MLX_BF16* weight = is_q ? q_weight : k_weight;
        device MLX_BF16* out_base = is_q ? q_out : k_out;

        float inv_rms = rsqrt(sum_sq / float(D) + eps);
        size_t out_row = (size_t(b) * size_t(H_out) + size_t(h)) * size_t(S) + size_t(s);
        device MLX_BF16* out_row_ptr = out_base + out_row * size_t(D);

        int half_D = D / 2;
        device const MLX_BF16* cos_row = cos + size_t(s) * size_t(D);
        device const MLX_BF16* sin_row = sin + size_t(s) * size_t(D);

        for (int i = int(tid); i < half_D; i += 32) {
            #pragma clang fp reassociate(off)
            int j = i + half_D;
            float n1 = (float(x_row[i]) * inv_rms) * float(weight[i]);
            float n2 = (float(x_row[j]) * inv_rms) * float(weight[j]);
            float x1 = float(MLX_BF16(n1));
            float x2 = float(MLX_BF16(n2));
            float c1 = float(cos_row[i]);
            float s1 = float(sin_row[i]);
            float c2 = float(cos_row[j]);
            float s2 = float(sin_row[j]);
            float rot1 = float(MLX_BF16(x1 * c1)) + float(MLX_BF16((-x2) * s1));
            float rot2 = float(MLX_BF16(x2 * c2)) + float(MLX_BF16(x1 * s2));
            out_row_ptr[i] = MLX_BF16(rot1);
            out_row_ptr[j] = MLX_BF16(rot2);
        }
    } else {
        int h = h_total - H_q - H_kv;
        float inv_rms = rsqrt(sum_sq / float(D) + v_eps);
        size_t out_row = (size_t(b) * size_t(H_kv) + size_t(h)) * size_t(S) + size_t(s);
        device MLX_BF16* out_row_ptr = v_out + out_row * size_t(D);

        for (int i = int(tid); i < D; i += 32) {
            float v = float(x_row[i]);
            float normed = v * inv_rms;
            if (has_v_weight) {
                normed *= float(v_weight[i]);
            }
            out_row_ptr[i] = MLX_BF16(normed);
        }
    }
}

// =========================================================================
// 9. Fused Single-Dispatch Vision QKV RMSNorm + 2D Axial RoPE + Transpose
// =========================================================================
kernel void vision_qkv_rmsnorm_rope2d_transpose_bf16(
    device const MLX_BF16* qkv [[buffer(0)]],
    device const MLX_BF16* q_weight [[buffer(1)]],
    device const MLX_BF16* k_weight [[buffer(2)]],
    device const MLX_BF16* v_weight [[buffer(3)]],
    device const MLX_BF16* cos [[buffer(4)]],
    device const MLX_BF16* sin [[buffer(5)]],
    device MLX_BF16* q_out [[buffer(6)]],
    device MLX_BF16* k_out [[buffer(7)]],
    device MLX_BF16* v_out [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant int& S [[buffer(10)]],
    constant int& H [[buffer(11)]],
    constant int& D [[buffer(12)]],
    constant int& has_v_weight [[buffer(13)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    int h_total = int(tg_pos.x);
    int s = int(tg_pos.y);
    int b = int(tg_pos.z);

    int total_heads = 3 * H;
    size_t in_row = (size_t(b) * size_t(S) + size_t(s)) * size_t(total_heads) + size_t(h_total);
    device const MLX_BF16* x_row = qkv + in_row * size_t(D);

    float thread_sum_sq = 0.0f;
    for (int i = int(tid); i < D; i += 32) {
        float v = float(x_row[i]);
        thread_sum_sq += v * v;
    }
    float sum_sq = simd_sum(thread_sum_sq);
    float inv_rms = rsqrt(sum_sq / float(D) + eps);

    if (h_total < 2 * H) {
        bool is_q = (h_total < H);
        int h = is_q ? h_total : (h_total - H);
        device const MLX_BF16* weight = is_q ? q_weight : k_weight;
        device MLX_BF16* out_base = is_q ? q_out : k_out;

        size_t out_row = (size_t(b) * size_t(H) + size_t(h)) * size_t(S) + size_t(s);
        device MLX_BF16* out_row_ptr = out_base + out_row * size_t(D);

        int quarter_D = D / 4; // 16 for head_dim = 64
        device const MLX_BF16* cos_row = cos + size_t(s) * size_t(D);
        device const MLX_BF16* sin_row = sin + size_t(s) * size_t(D);

        for (int idx = int(tid); idx < 2 * quarter_D; idx += 32) {
            #pragma clang fp reassociate(off)
            int i = (idx < quarter_D) ? idx : (idx + quarter_D);
            int j = i + quarter_D;
            float n1 = (float(x_row[i]) * inv_rms) * float(weight[i]);
            float n2 = (float(x_row[j]) * inv_rms) * float(weight[j]);
            float x1 = float(MLX_BF16(n1));
            float x2 = float(MLX_BF16(n2));
            float c1 = float(cos_row[i]);
            float s1 = float(sin_row[i]);
            float c2 = float(cos_row[j]);
            float s2 = float(sin_row[j]);
            float rot1 = float(MLX_BF16(x1 * c1)) + float(MLX_BF16((-x2) * s1));
            float rot2 = float(MLX_BF16(x2 * c2)) + float(MLX_BF16(x1 * s2));
            out_row_ptr[i] = MLX_BF16(rot1);
            out_row_ptr[j] = MLX_BF16(rot2);
        }
    } else {
        int h = h_total - 2 * H;
        size_t out_row = (size_t(b) * size_t(H) + size_t(h)) * size_t(S) + size_t(s);
        device MLX_BF16* out_row_ptr = v_out + out_row * size_t(D);

        for (int i = int(tid); i < D; i += 32) {
            float v = float(x_row[i]);
            float normed = v * inv_rms;
            if (has_v_weight) {
                normed *= float(v_weight[i]);
            }
            out_row_ptr[i] = MLX_BF16(normed);
        }
    }
}

// =========================================================================
// 10. Fused Audio GLU + 5-Tap Causal Depthwise Conv1D Kernel
// =========================================================================
kernel void audio_glu_causal_dwconv5_bf16(
    device const MLX_BF16* pw1 [[buffer(0)]],
    device const MLX_BF16* dw_weight [[buffer(1)]],
    device MLX_BF16* out [[buffer(2)]],
    constant int& B [[buffer(3)]],
    constant int& S [[buffer(4)]],
    constant int& C [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    #pragma clang fp reassociate(off)
    int total = B * S * C;
    if (int(id) >= total) return;

    int c = int(id) % C;
    int bs = int(id) / C;
    int s = bs % S;
    int b = bs / S;

    device const MLX_BF16* w_ptr = dw_weight + size_t(c) * 5;
    float acc = 0.0f;

    #pragma unroll(5)
    for (int k = 0; k < 5; k++) {
        int src_s = s + k - 4;
        float glu_val = 0.0f;
        if (src_s >= 0) {
            size_t base = (size_t(b) * size_t(S) + size_t(src_s)) * size_t(2 * C) + size_t(c);
            float h0 = float(pw1[base]);
            float h1 = float(pw1[base + size_t(C)]);
            float sig = float(MLX_BF16(1.0f / (1.0f + metal::exp(-h1))));
            glu_val = float(MLX_BF16(h0 * sig));
        }
        float term = float(MLX_BF16(glu_val * float(w_ptr[k])));
        if (k == 0) {
            acc = term;
        } else {
            acc = float(MLX_BF16(acc + term));
        }
    }

    out[id] = MLX_BF16(acc);
}

// =========================================================================
// 11. Native Metal Q8_0 Quantized MatMul & MatVec (Zero F32 Cast Overhead)
// =========================================================================
#define QK8_0 32
typedef struct {
    half d;            // delta (2 bytes)
    int8_t qs[QK8_0];  // quants (32 bytes)
} block_q8_0;

inline void dequantize_q8_0(device const block_q8_0* xb, short il, thread half4x4& reg) {
    device const int8_t* qs = (device const int8_t*)xb->qs;
    const half d = xb->d;
    #pragma unroll(16)
    for (int i = 0; i < 16; i++) {
        reg[i / 4][i % 4] = (qs[i + 16 * il] * d);
    }
}

template <typename ActT>
inline void q8_0_mv_impl(
    device const uchar* src0,
    device const ActT* src1,
    device ActT* dst,
    int ne00,
    int ne0,
    int ne1,
    uint3 tgpig,
    uint tiisg,
    uint sgitg
) {
    const int nr = 4;
    const int nsg = 2;
    const int nw = 32;

    const int nb = ne00 / QK8_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;

    const int first_row = (r0 * nsg + int(sgitg)) * nr;
    if (first_row >= ne0 || r1 >= ne1) return;

    const size_t nb01 = size_t(nb) * 34u;
    device const block_q8_0* x = (device const block_q8_0*)(src0 + size_t(first_row) * nb01);
    device const ActT* y = src1 + size_t(r1) * size_t(ne00);

    float yl[8];
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int ix = int(tiisg) / 4;
    const int il = int(tiisg) % 4;

    device const ActT* yb = y + ix * QK8_0 + 8 * il;

    for (int ib = ix; ib < nb; ib += nw / 4) {
        #pragma unroll(8)
        for (int i = 0; i < 8; ++i) {
            yl[i] = float(yb[i]);
        }

        #pragma unroll(4)
        for (int row = 0; row < 4; row++) {
            if (first_row + row < ne0) {
                device const int8_t* qs = x[ib + row * nb].qs + 8 * il;
                float sumq = 0.0f;
                #pragma unroll(8)
                for (int iq = 0; iq < 8; ++iq) {
                    sumq += float(qs[iq]) * yl[iq];
                }
                sumf[row] += sumq * float(x[ib + row * nb].d);
            }
        }

        yb += 8 * nw;
    }

    #pragma unroll(4)
    for (int row = 0; row < 4; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne0) {
            dst[size_t(r1) * size_t(ne0) + size_t(first_row + row)] = ActT(tot);
        }
    }
}

template <typename ActT>
inline void q8_0_mm_impl(
    device const uchar* src0,
    device const ActT* src1,
    device ActT* dst,
    int ne00,
    int ne0,
    int ne1,
    threadgroup uchar* shared_memory,
    uint3 tgpig,
    uint tiitg,
    uint sgitg
) {
    threadgroup half* sa = (threadgroup half*)(shared_memory);
    threadgroup float* sb = (threadgroup float*)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;

    if (int(r0 * 64) >= ne0 || int(r1 * 32) >= ne1) return;

    short n_rows = (ne0 - int(r0 * 64) < 64) ? short(ne0 - int(r0 * 64)) : 64;
    short n_cols = (ne1 - int(r1 * 32) < 32) ? short(ne1 - int(r1 * 32)) : 32;

    short thread_row = ((short)tiitg / 2) < n_rows ? ((short)tiitg / 2) : n_rows - 1;
    short thread_col = ((short)tiitg / 4) < n_cols ? ((short)tiitg / 4) : n_cols - 1;

    simdgroup_half8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    #pragma unroll(8)
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    short il = (tiitg % 2);
    size_t nb01 = size_t(ne00 / 32) * 34u;

    device const block_q8_0* x = (device const block_q8_0*)(src0 + size_t(r0 * 64 + uint(thread_row)) * nb01);
    device const ActT* y = src1 + size_t(r1 * 32 + uint(thread_col)) * size_t(ne00) + size_t(8 * (tiitg % 4));

    for (int loop_k = 0; loop_k < ne00; loop_k += 32) {
        half4x4 temp_a;
        dequantize_q8_0(x, il, temp_a);
        float temp_b[8];
        #pragma unroll(8)
        for (short i = 0; i < 8; i++) {
            temp_b[i] = float(y[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            *(sa + 64 * ((tiitg / 2 / 8) + (tiitg % 2) * 16 + (i / 8) * 8)
                 + (tiitg / 2) % 8 + (i & 7) * 8) = temp_a[i / 4][i % 4];
        }

        threadgroup float* sb_dst = sb + (tiitg % 4) * 256 + 8 * (tiitg / 4);
        #pragma unroll(8)
        for (short i = 0; i < 8; i++) {
            sb_dst[i] = temp_b[i];
        }

        x += 1;
        y += 32;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup float* lsmb = sb + 2 * 64 * (sgitg / 2);

        #pragma unroll(4)
        for (short ik = 0; ik < 4; ik++) {
            #pragma unroll(4)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;

            #pragma unroll(8)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* temp_base = (threadgroup float*)shared_memory;
    threadgroup float* temp_str = temp_base + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
    #pragma unroll(8)
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4), 64);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int total_out = int(n_cols) * int(n_rows);
    for (int idx = int(tiitg); idx < total_out; idx += 128) {
        int j = idx / int(n_rows);
        int i = idx % int(n_rows);
        dst[size_t(r1 * 32 + uint(j)) * size_t(ne0) + size_t(r0 * 64 + uint(i))] = ActT(temp_base[j * 64 + i]);
    }
}

kernel void q8_0_mv_bf16(
    device const uchar* src0 [[buffer(0)]],
    device const MLX_BF16* src1 [[buffer(1)]],
    device MLX_BF16* dst [[buffer(2)]],
    constant int& ne00 [[buffer(3)]],
    constant int& ne0 [[buffer(4)]],
    constant int& ne1 [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    q8_0_mv_impl<MLX_BF16>(src0, src1, dst, ne00, ne0, ne1, tgpig, tiisg, sgitg);
}

kernel void q8_0_mm_bf16(
    device const uchar* src0 [[buffer(0)]],
    device const MLX_BF16* src1 [[buffer(1)]],
    device MLX_BF16* dst [[buffer(2)]],
    constant int& ne00 [[buffer(3)]],
    constant int& ne0 [[buffer(4)]],
    constant int& ne1 [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uchar shared_memory[8192];
    q8_0_mm_impl<MLX_BF16>(src0, src1, dst, ne00, ne0, ne1, shared_memory, tgpig, tiitg, sgitg);
}

kernel void q8_0_mv_f32(
    device const uchar* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant int& ne00 [[buffer(3)]],
    constant int& ne0 [[buffer(4)]],
    constant int& ne1 [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    q8_0_mv_impl<float>(src0, src1, dst, ne00, ne0, ne1, tgpig, tiisg, sgitg);
}

kernel void q8_0_mm_f32(
    device const uchar* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant int& ne00 [[buffer(3)]],
    constant int& ne0 [[buffer(4)]],
    constant int& ne1 [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uchar shared_memory[8192];
    q8_0_mm_impl<float>(src0, src1, dst, ne00, ne0, ne1, shared_memory, tgpig, tiitg, sgitg);
}

