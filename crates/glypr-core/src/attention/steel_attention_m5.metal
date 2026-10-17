#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct AttnParams {
    int B;
    int H;
    int D;
    int qL;
    int kL;
    int gqa_factor;
    float scale;
    float softcapping;
    int NQ;
    int NK;
    int NQ_aligned;
    int NK_aligned;
    int qL_rem;
    int kL_rem;
    int qL_off;
    int window_size;
    int64_t Q_strides[3];
    int64_t K_strides[3];
    int64_t V_strides[3];
    int64_t O_strides[3];
};

struct AttnMaskParams {
    int64_t M_strides[3];
};

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];
constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];

// =========================================================================
// M5 Neural Accelerator FlashAttention (mpp::tensor_ops::matmul2d)
// =========================================================================
// Each threadgroup has WM=4 independent SIMD groups (128 threads total).
// Each SIMD group (execution_simdgroup) independently computes a [BQ_SG = BQ/4 = 16, BK = 32]
// query slice using hardware-native single-SIMD-group cooperative tensors:
//   - Zero threadgroup memory (0 KB shared memory) and zero threadgroup_barrier() calls!
//   - S = Q_sg @ K^T -> sT (float cooperative_tensor)
//   - Online softmax via hardware reduce_rows(sT, ..., reduction_operation::max / sum)
//     and exact map_iterator(it) coordinate mapping!
//   - pT = pv_op.get_left_input_cooperative_tensor<bfloat, bfloat, float>(sT_bf16) in registers!
//   - O += pT @ V directly from cooperative_tensor registers (split into 2x256 output tiles when BD=512
//     to stay within the 256-column matmul2d tile limit and keep register occupancy high)!
template <int BQ, int BK, int BD, int WM>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]]
void steel_attention_m5(
    device bfloat* Q [[buffer(0)]],
    device bfloat* K [[buffer(1)]],
    device bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device bfloat* mask [[buffer(6), function_constant(has_mask)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]
) {
    constexpr int BQ_SG = BQ / WM; // 16 rows per SIMD group when BQ=64, WM=4
    constexpr int PV_D  = (BD > 256) ? 256 : BD;

    const ulong3 tidl{tid.x, tid.y, tid.z};
    const int q_block_base = int(tid.x) * BQ;
    const int q_sg_base = q_block_base + int(simd_group_id) * BQ_SG;
    if (q_sg_base >= params->qL) return;

    device bfloat* Q_head = Q + tidl.z * params->Q_strides[0] + tidl.y * params->Q_strides[1];
    const ulong kv_head_idx = int(tid.y) / params->gqa_factor;
    device bfloat* K_head = K + tidl.z * params->K_strides[0] + kv_head_idx * params->K_strides[1];
    device bfloat* V_head = V + tidl.z * params->V_strides[0] + kv_head_idx * params->V_strides[1];
    device bfloat* O_head = O + tidl.z * params->O_strides[0] + tidl.y * params->O_strides[1];

    const device bfloat* mask_head = nullptr;
    if (has_mask) {
        mask_head = mask + tidl.z * mask_params->M_strides[0] + tidl.y * mask_params->M_strides[1];
    }

    int kb_start = 0;
    int kb_lim = params->NK;
    if (params->window_size > 0) {
        int q_min = q_sg_base + params->qL_off;
        int q_max = q_sg_base + BQ_SG + params->qL_off;
        kb_start = (q_min > params->window_size) ? (q_min - params->window_size) / BK : 0;
        kb_lim = metal::min(params->NK, (q_max + params->window_size + BK - 1) / BK);
    } else if (do_causal) {
        int q_max = q_sg_base + BQ_SG + params->qL_off;
        kb_lim = (q_max + BK - 1) / BK;
    }

    // Wrap head tensors in Metal 4 tensor_inline (hardware bounds-checked!):
    // extent(0) = contiguous dimension (BD), extent(1) = sequence dimension (qL / kL)
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Q_t(
        Q_head,
        dextents<int32_t, 2>(BD, params->qL),
        array<int32_t, 2>{1, int32_t(params->Q_strides[2])}
    );
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> K_t(
        K_head,
        dextents<int32_t, 2>(BD, params->kL),
        array<int32_t, 2>{1, int32_t(params->K_strides[2])}
    );
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> V_t(
        V_head,
        dextents<int32_t, 2>(BD, params->kL),
        array<int32_t, 2>{1, int32_t(params->V_strides[2])}
    );

    // 1. Q @ K^T descriptor per SIMD group: [BQ_SG, BD] @ [BK, BD]^T -> [BQ_SG, BK]
    constexpr auto qk_desc = matmul2d_descriptor(
        BQ_SG,
        BK,
        BD,
        false, // transpose_left = false
        true,  // transpose_right = true (K^T)
        false, // relaxed_precision = false (exact FP32 accumulation)
        matmul2d_descriptor::mode::multiply
    );
    matmul2d<qk_desc, execution_simdgroup> qk_op;

    // 2. P @ V descriptor per SIMD group: [BQ_SG, BK] (left cooperative_tensor bfloat) @ [BK, PV_D] (device bfloat) -> [BQ_SG, PV_D]
    constexpr auto pv_desc = matmul2d_descriptor(
        BQ_SG,
        PV_D,
        BK,
        false, // transpose_left = false
        false, // transpose_right = false
        false, // relaxed_precision = false
        matmul2d_descriptor::mode::multiply_accumulate
    );
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    auto mQ = Q_t.slice<BD, BQ_SG>(0, q_sg_base);
    auto mK0 = K_t.slice<BD, BK>(0, 0);
    auto mV0 = V_t.slice<PV_D, BK>(0, 0);

    // Score tile S [BQ_SG, BK] in FP32 cooperative_tensor registers
    auto sT = qk_op.template get_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();
    // Destination cooperative_tensor in bfloat with identical layout to sT so we can copy/repack into pv_op left input!
    auto sT_bf16 = qk_op.template get_destination_cooperative_tensor<decltype(mQ), decltype(mK0), bfloat>();
    // Output tile(s) O in FP32 cooperative_tensor registers
    auto pT_dummy = pv_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto oT0 = pv_op.template get_destination_cooperative_tensor<decltype(pT_dummy), decltype(mV0), float>();
    auto oT1 = pv_op.template get_destination_cooperative_tensor<decltype(pT_dummy), decltype(mV0), float>();

    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT0.get_capacity(); ++i) {
        if (oT0.is_valid_element(i)) {
            oT0[i] = 0.0f;
            if constexpr (BD > 256) {
                oT1[i] = 0.0f;
            }
        }
    }

    // Online softmax state in row-reduction cooperative tensors
    auto max_score = qk_op.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();
    auto new_max   = qk_op.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();
    auto factor    = qk_op.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();
    auto sum_score = qk_op.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();
    auto sum_tmp   = qk_op.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();

    constexpr float neg_inf = -metal::numeric_limits<float>::infinity();
    const float score_scale = params->scale * 1.44269504089f;

    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < max_score.get_capacity(); ++i) {
        max_score[i] = neg_inf;
        new_max[i]   = neg_inf;
        sum_score[i] = 0.0f;
    }

    for (int kb = kb_start; kb < kb_lim; ++kb) {
        const int k_base = kb * BK;
        auto mK = K_t.slice<BD, BK>(0, k_base);

        // Execute S = Q @ K^T on M5 Neural Accelerator
        qk_op.run(mQ, mK, sT);

        // Scale and mask S in registers
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sT.get_capacity(); ++i) {
            if (sT.is_valid_element(i)) {
                auto idx = sT.get_multidimensional_index(i);
                const int col_pos = k_base + int(idx[0]);
                const int row_pos = q_sg_base + params->qL_off + int(idx[1]);
                const int q_idx   = q_sg_base + int(idx[1]);
                float val = sT[i] * score_scale;

                if (!align_Q && q_idx >= params->qL) {
                    val = neg_inf;
                }
                if (!align_K && col_pos >= params->kL) {
                    val = neg_inf;
                }
                if (do_causal && row_pos < col_pos) {
                    val = neg_inf;
                }
                if (params->window_size > 0 && metal::abs(row_pos - col_pos) > params->window_size) {
                    val = neg_inf;
                }
                if (has_mask) {
                    if (q_idx < params->qL && col_pos < params->kL) {
                        val += 1.44269504089f * float(mask_head[size_t(q_idx) * size_t(mask_params->M_strides[2]) + size_t(col_pos)]);
                    } else {
                        val = neg_inf;
                    }
                }
                sT[i] = val;
            } else {
                sT[i] = neg_inf;
            }
        }

        // Row max reduction: reduce_rows overwrites new_max with row_max(sT), using -inf as identity!
        reduce_rows(sT, new_max, reduction_operation::max, neg_inf);

        // Update running max and rescaling factor per row
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < max_score.get_capacity(); ++i) {
            if (max_score.is_valid_element(i)) {
                const float m_old = max_score[i];
                const float m_cur = metal::max(m_old, new_max[i]);
                new_max[i] = m_cur;
                factor[i] = (m_old == neg_inf) ? 0.0f : fast::exp2(m_old - m_cur);
                max_score[i] = m_cur;
            }
        }

        // Compute exp2(sT - new_max) using map_iterator and populate sT_bf16
        #pragma clang loop unroll(full)
        for (auto it = sT.begin(); it != sT.end(); ++it) {
            if (it.is_valid_element()) {
                auto max_it = new_max.map_iterator(it);
                const float m_cur = *max_it;
                const float p_val = (m_cur == neg_inf) ? 0.0f : fast::exp2(*it - m_cur);
                *it = p_val;
            } else {
                *it = 0.0f;
            }
        }

        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sT.get_capacity(); ++i) {
            sT_bf16[i] = bfloat(sT[i]);
        }

        // Row sum reduction: reduce_rows overwrites sum_tmp with row_sum(sT), using 0.0f as identity!
        reduce_rows(sT, sum_tmp, reduction_operation::sum, 0.0f);

        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sum_score.get_capacity(); ++i) {
            if (sum_score.is_valid_element(i)) {
                sum_score[i] = sum_score[i] * factor[i] + sum_tmp[i];
            }
        }

        // Rescale O tile(s) by factor using map_iterator
        #pragma clang loop unroll(full)
        for (auto o_it = oT0.begin(); o_it != oT0.end(); ++o_it) {
            if (o_it.is_valid_element()) {
                auto f_it = factor.map_iterator(o_it);
                const float f = *f_it;
                *o_it *= f;
            }
        }
        if constexpr (BD > 256) {
            #pragma clang loop unroll(full)
            for (auto o_it = oT1.begin(); o_it != oT1.end(); ++o_it) {
                if (o_it.is_valid_element()) {
                    auto f_it = factor.map_iterator(o_it);
                    const float f = *f_it;
                    *o_it *= f;
                }
            }
        }

        // Repack destination cooperative_tensor sT_bf16 into pv_op left-input cooperative_tensor pT in registers!
        auto pT = pv_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>(sT_bf16);

        // Execute O += P @ V directly from cooperative_tensor pT in registers!
        auto mV_lo = V_t.slice<PV_D, BK>(0, k_base);
        pv_op.run(pT, mV_lo, oT0);
        if constexpr (BD > 256) {
            auto mV_hi = V_t.slice<PV_D, BK>(256, k_base);
            pv_op.run(pT, mV_hi, oT1);
        }
    }

    // Normalize O tile(s) by 1 / sum_score using map_iterator and store to device bfloat O_head
    const size_t o_row_stride = size_t(params->O_strides[2]);
    #pragma clang loop unroll(full)
    for (auto o_it = oT0.begin(); o_it != oT0.end(); ++o_it) {
        if (o_it.is_valid_element()) {
            auto s_it = sum_score.map_iterator(o_it);
            const float denom = *s_it;
            const float val = (denom == 0.0f) ? 0.0f : (*o_it / denom);
            auto idx = o_it.get_multidimensional_index();
            const int d_col = int(idx[0]);
            const int q_row = q_sg_base + int(idx[1]);
            if (q_row < params->qL && d_col < PV_D) {
                O_head[size_t(q_row) * o_row_stride + size_t(d_col)] = bfloat(val);
            }
        }
    }
    if constexpr (BD > 256) {
        #pragma clang loop unroll(full)
        for (auto o_it = oT1.begin(); o_it != oT1.end(); ++o_it) {
            if (o_it.is_valid_element()) {
                auto s_it = sum_score.map_iterator(o_it);
                const float denom = *s_it;
                const float val = (denom == 0.0f) ? 0.0f : (*o_it / denom);
                auto idx = o_it.get_multidimensional_index();
                const int d_col = 256 + int(idx[0]);
                const int q_row = q_sg_base + int(idx[1]);
                if (q_row < params->qL && d_col < BD) {
                    O_head[size_t(q_row) * o_row_stride + size_t(d_col)] = bfloat(val);
                }
            }
        }
    }
}

template [[host_name("steel_attention_m5_bfloat16_bq64_bk32_bd256_wm4")]]
[[kernel]] void steel_attention_m5<64, 32, 256, 4>(
    device bfloat* Q [[buffer(0)]],
    device bfloat* K [[buffer(1)]],
    device bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device bfloat* mask [[buffer(6), function_constant(has_mask)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]
);

template [[host_name("steel_attention_m5_bfloat16_bq64_bk32_bd512_wm4")]]
[[kernel]] void steel_attention_m5<64, 32, 512, 4>(
    device bfloat* Q [[buffer(0)]],
    device bfloat* K [[buffer(1)]],
    device bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device bfloat* mask [[buffer(6), function_constant(has_mask)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]
);
