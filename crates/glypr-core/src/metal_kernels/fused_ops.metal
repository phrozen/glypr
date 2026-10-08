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
    device const MLX_BF16* x_row = x + row * D;
    device MLX_BF16* out_row = out + row * D;
    
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
