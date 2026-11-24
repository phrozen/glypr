#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define QK8_0 32

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

// =========================================================================
// M5 Neural Accelerator (mpp::tensor_ops::matmul2d) Q8_0 Quantized GEMM (BK=64)
// =========================================================================
// Computes dst[M, N] = src1[M, K] @ src0[N, K]^T
// where:
//   src0: Q8_0 weights of shape [N = ne0, K = ne00]
//   src1: BF16 activations of shape [M = ne1, K = ne00]
//   dst:  BF16 output of shape [M = ne1, N = ne0]
//
// Each threadgroup (128 threads = 4 simdgroups) computes a [BM=32, BN=64] output tile
// over BK=64 K-slices per barrier pair:
//   - Activations X [32, 64] are read directly from device bfloat memory via tensor_inline.
//   - 128 threads cooperatively dequantize 64 rows x 2 Q8_0 blocks (64 cols = 4096 weights)
//     using vectorized packed_char4 loads into threadgroup half Ws[64 * 64] (8 KB shared memory).
//   - mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> executes C += X @ W^T on the
//     per-core M5 Neural Accelerators, accumulating in float32 cooperative_tensor registers.
[[kernel, max_total_threads_per_threadgroup(128)]]
void q8_0_mm_m5_bf16(
    device const uchar* src0 [[buffer(0)]],
    device bfloat* src1 [[buffer(1)]],
    device bfloat* dst [[buffer(2)]],
    constant int& ne00 [[buffer(3)]],
    constant int& ne0 [[buffer(4)]],
    constant int& ne1 [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]]
) {
    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 64;

    const int n_base = int(tgpig.y) * BN;
    const int m_base = int(tgpig.x) * BM;
    if (n_base >= ne0 || m_base >= ne1) return;

    threadgroup half Ws[BN * BK];

    // Wrap full activation matrix src1 [M = ne1, K = ne00] in row-major order:
    // extent(0) = contiguous dimension (ne00), extent(1) = row dimension (ne1).
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> X_t(
        src1,
        dextents<int32_t, 2>(ne00, ne1)
    );
    // Wrap threadgroup weight tile Ws [BN = 64, BK = 64] in row-major order:
    // extent(0) = contiguous dimension (BK = 64), extent(1) = row dimension (BN = 64).
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> W_t(
        Ws,
        dextents<int32_t, 2>(BK, BN)
    );

    constexpr auto desc = matmul2d_descriptor(
        BM,
        BN,
        BK,
        false, // transpose_left = false  (X is [BM, BK])
        true,  // transpose_right = true  (W is [BN, BK], so W^T is [BK, BN])
        false, // relaxed_precision = false (exact FP32 accumulation)
        matmul2d_descriptor::mode::multiply_accumulate
    );
    matmul2d<desc, execution_simdgroups<4>> op;

    auto mX0 = X_t.slice<BK, BM>(0, m_base);
    auto mW0 = W_t.slice<BK, BN>(0, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mX0), decltype(mW0), float>();

    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    const int nb = ne00 / 32;
    const size_t row_bytes = size_t(nb) * 34u;

    // 128 threads cooperatively dequantize 64 rows x 2 Q8_0 blocks (32 weights/block = 4096 weights)
    const int w_row_local = int(tiitg) / 2;   // [0..63]
    const int b_local     = int(tiitg) % 2;   // 0 or 1 (32-element block within BK=64)
    const int w_row_global = n_base + w_row_local;
    const bool valid_w_row = w_row_global < ne0;

    device const block_q8_0* w_row_ptr = valid_w_row
        ? (device const block_q8_0*)(src0 + size_t(w_row_global) * row_bytes)
        : (device const block_q8_0*)src0;

    threadgroup half4* ws_thread4 = (threadgroup half4*)(Ws + w_row_local * BK + b_local * 32);

    for (int k_block = 0; k_block < nb; k_block += 2) {
        const int blk_idx = k_block + b_local;
        if (valid_w_row && blk_idx < nb) {
            half d = w_row_ptr[blk_idx].d;
            device const packed_char4* qs4 = (device const packed_char4*)(w_row_ptr[blk_idx].qs);
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; ++i) {
                ws_thread4[i] = half4(char4(qs4[i])) * d;
            }
        } else {
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; ++i) {
                ws_thread4[i] = half4(0.0h);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mX = X_t.slice<BK, BM>(k_block * 32, m_base);
        op.run(mX, mW0, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulated FP32 cooperative_tensor directly to device bfloat dst [M = ne1, N = ne0]
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            int col_n = n_base + int(idx[0]);
            int row_m = m_base + int(idx[1]);
            if (row_m < ne1 && col_n < ne0) {
                dst[size_t(row_m) * size_t(ne0) + size_t(col_n)] = bfloat(cT[i]);
            }
        }
    }
}

// =========================================================================
// M5 Neural Accelerator (mpp::tensor_ops::matmul2d) Dense BF16 GEMM
// =========================================================================
// Computes dst[M, N] = src1[M, K] @ src0[N, K]^T
// Zero threadgroup memory and zero threadgroup_barrier() calls.
[[kernel, max_total_threads_per_threadgroup(128)]]
void bf16_mm_m5(
    device bfloat* src0 [[buffer(0)]], // W [N, K]
    device bfloat* src1 [[buffer(1)]], // X [M, K]
    device bfloat* dst  [[buffer(2)]], // Y [M, N]
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& M [[buffer(5)]],
    constant int& lda [[buffer(6)]], // stride of W along N (K for contiguous)
    uint3 tgpig [[threadgroup_position_in_grid]]
) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    const int m_base = int(tgpig.x) * BM;
    const int n_base = int(tgpig.y) * BN;
    if (m_base >= M || n_base >= N) return;

    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> X_t(
        src1, dextents<int32_t, 2>(K, M));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> W_t(
        src0, dextents<int32_t, 2>(K, N), array<int32_t, 2>{1, lda});

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mX0 = X_t.slice<BK, BM>(0, m_base);
    auto mW0 = W_t.slice<BK, BN>(0, n_base);
    auto cT = op.get_destination_cooperative_tensor<decltype(mX0), decltype(mW0), float>();

    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }
    for (int k = 0; k < K; k += BK) {
        auto mX = X_t.slice<BK, BM>(k, m_base);
        auto mW = W_t.slice<BK, BN>(k, n_base);
        op.run(mX, mW, cT);
    }
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            int col_n = n_base + int(idx[0]);
            int row_m = m_base + int(idx[1]);
            if (row_m < M && col_n < N) {
                dst[size_t(row_m) * size_t(N) + size_t(col_n)] = bfloat(cT[i]);
            }
        }
    }
}
