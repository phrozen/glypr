#include <metal_stdlib>
using namespace metal;

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
