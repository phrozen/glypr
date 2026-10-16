use candle_core::{DType, Device, Result, Tensor};
use candle_metal_kernels::metal::{ComputePipeline, ConstantValues, Device as MtlDevice, Library, Value};
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::traits::AttentionBackend;
use super::chunked::ChunkedSlidingAttention;

static SLIDING_SHADER_SRC: &str = include_str!("steel_attention_sliding.metal");

#[derive(Debug)]
#[repr(C)]
struct AttnParams {
    b: i32,
    h: i32,
    d: i32,
    ql: i32,
    kl: i32,
    gqa_factor: i32,
    scale: f32,
    softcapping: f32,
    nq: i32,
    nk: i32,
    nq_aligned: i32,
    nk_aligned: i32,
    ql_rem: i32,
    kl_rem: i32,
    ql_off: i32,
    window_size: i32,
    q_strides: [i64; 3],
    k_strides: [i64; 3],
    v_strides: [i64; 3],
    o_strides: [i64; 3],
}

#[derive(Debug)]
#[repr(C)]
struct AttnMaskParams {
    m_strides: [i64; 3],
}

struct PipelineCache {
    library: Library,
    pipelines: Mutex<HashMap<(usize, bool, bool, bool), ComputePipeline>>,
}

static PIPELINE_CACHE: OnceLock<PipelineCache> = OnceLock::new();

fn get_or_compile_pipeline(
    device: &MtlDevice,
    head_dim: usize,
    align_q: bool,
    align_k: bool,
    has_mask: bool,
) -> Result<ComputePipeline> {
    let cache = PIPELINE_CACHE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SLIDING_SHADER_SRC, None)
            .expect("Failed to compile sliding-window Steel Attention Metal library");
        PipelineCache {
            library: lib,
            pipelines: Mutex::new(HashMap::new()),
        }
    });

    let key = (head_dim, align_q, align_k, has_mask);
    let mut map = cache.pipelines.lock().unwrap();
    if let Some(p) = map.get(&key) {
        return Ok(p.clone());
    }

    let cv = ConstantValues::new(vec![
        (200, Value::Bool(align_q)),
        (201, Value::Bool(align_k)),
        (300, Value::Bool(has_mask)),
        (301, Value::Bool(false)), // do_causal = false (bidirectional sliding window / full attention)
    ]);

    let func_name = match head_dim {
        256 => "steel_attention_bfloat16_bq32_bk16_bd256_wm4_wn1_maskbfloat16",
        512 => "steel_attention_bfloat16_bq16_bk16_bd512_wm2_wn1_maskbfloat16",
        _ => {
            return Err(candle_core::Error::Msg(format!(
                "Unsupported head_dim for MetalFlashAttention: {head_dim}"
            )))
        }
    };
    let func = cache
        .library
        .get_function(func_name, Some(&cv))
        .map_err(|e| candle_core::Error::Metal(format!("Failed to get function {func_name}: {e}").into()))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| candle_core::Error::Metal(format!("Failed to create compute pipeline: {e}").into()))?;

    map.insert(key, pipeline.clone());
    Ok(pipeline)
}

/// Dedicated Apple Silicon Sliding-Window & Full-Context Metal FlashAttention Engine.
///
/// Features:
/// - Custom Metal Shading Language kernel with hardware tile-skipping for sliding window attention.
/// - Supports both `head_dim = 256` (sliding layers) and `head_dim = 512` (global layers 5, 11, 17, 23).
/// - Zero mask allocation (0 MB VRAM overhead) on unpadded sequences up to 8K, and $O(B \cdot S)$
///   stride-broadcasted key-padding mask on heterogeneous batches.
/// - Native GQA without `repeat_kv` tensor duplication.
/// - Automatic fallback to ChunkedSlidingAttention when running on CPU or unsupported dtypes.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalFlashAttention;

impl AttentionBackend for MetalFlashAttention {
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        let mdev = match q.device() {
            Device::Metal(m) => m,
            _ => {
                // Graceful CPU fallback
                let (_b, h, _s, _d) = q.dims4()?;
                let (_kb, kh, _ks, _kd) = k.dims4()?;
                let gqa = h / kh;
                let k_rep = if gqa > 1 { super::repeat_kv(k, gqa)? } else { k.clone() };
                let v_rep = if gqa > 1 { super::repeat_kv(v, gqa)? } else { v.clone() };
                return ChunkedSlidingAttention.compute_attention(q, &k_rep, &v_rep, attention_mask, sliding_window);
            }
        };

        let (b, h, s, d) = q.dims4()?;
        let (_kb, kh, _ks, _kd) = k.dims4()?;
        let gqa_factor = (h / kh) as i32;

        // Metal Flash Kernel supports head_dim = 256 and 512 in BF16
        if (d != 256 && d != 512) || q.dtype() != DType::BF16 {
            let k_rep = if gqa_factor > 1 { super::repeat_kv(k, gqa_factor as usize)? } else { k.clone() };
            let v_rep = if gqa_factor > 1 { super::repeat_kv(v, gqa_factor as usize)? } else { v.clone() };
            return ChunkedSlidingAttention.compute_attention(q, &k_rep, &v_rep, attention_mask, sliding_window);
        }

        // Validate optional additive mask ([B, 1, 1, S] or [B, H|1, S|1, S] in BF16 on Metal)
        let mask_contig = if let Some(m) = attention_mask {
            if m.dtype() != DType::BF16 || !m.device().is_metal() || m.dims().len() != 4 || m.dim(3)? != s {
                let k_rep = if gqa_factor > 1 { super::repeat_kv(k, gqa_factor as usize)? } else { k.clone() };
                let v_rep = if gqa_factor > 1 { super::repeat_kv(v, gqa_factor as usize)? } else { v.clone() };
                return super::metal_sdpa::MetalSdpaAttention.compute_attention(q, &k_rep, &v_rep, attention_mask, sliding_window);
            }
            Some(if m.is_contiguous() { m.clone() } else { m.contiguous()? })
        } else {
            None
        };

        let (bq, bk, wm, wn) = if d == 256 {
            (32, 16, 4, 1)
        } else {
            (16, 16, 2, 1)
        };
        let align_q = (s % bq) == 0;
        let align_k = (s % bk) == 0;
        let has_mask = mask_contig.is_some();

        let pipeline = get_or_compile_pipeline(mdev.metal_device(), d, align_q, align_k, has_mask)?;

        let out_tensor = Tensor::zeros((b, h, s, d), DType::BF16, q.device())?;

        {
            let (q_storage, q_layout) = q.storage_and_layout();
            let (k_storage, k_layout) = k.storage_and_layout();
            let (v_storage, v_layout) = v.storage_and_layout();
            let (out_storage, out_layout) = out_tensor.storage_and_layout();

            let q_buf = match &*q_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
            let k_buf = match &*k_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
            let v_buf = match &*v_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
            let out_buf = match &*out_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };

            let nq = (s + bq - 1) / bq;
            let nk = (s + bk - 1) / bk;
            let nq_aligned = s / bq;
            let nk_aligned = s / bk;

            let window_size = sliding_window.unwrap_or(0) as i32;

            let params = AttnParams {
                b: b as i32,
                h: h as i32,
                d: d as i32,
                ql: s as i32,
                kl: s as i32,
                gqa_factor,
                scale: 1.0,
                softcapping: 1.0,
                nq: nq as i32,
                nk: nk as i32,
                nq_aligned: nq_aligned as i32,
                nk_aligned: nk_aligned as i32,
                ql_rem: (s - nq_aligned * bq) as i32,
                kl_rem: (s - nk_aligned * bk) as i32,
                ql_off: 0,
                window_size,
                q_strides: [q_layout.stride()[0] as i64, q_layout.stride()[1] as i64, q_layout.stride()[2] as i64],
                k_strides: [k_layout.stride()[0] as i64, k_layout.stride()[1] as i64, k_layout.stride()[2] as i64],
                v_strides: [v_layout.stride()[0] as i64, v_layout.stride()[1] as i64, v_layout.stride()[2] as i64],
                o_strides: [out_layout.stride()[0] as i64, out_layout.stride()[1] as i64, out_layout.stride()[2] as i64],
            };

            let elem_bytes = q.dtype().size_in_bytes();
            let q_offset = q_layout.start_offset() * elem_bytes;
            let k_offset = k_layout.start_offset() * elem_bytes;
            let v_offset = v_layout.start_offset() * elem_bytes;
            let out_offset = out_layout.start_offset() * elem_bytes;

            let encoder = mdev.command_encoder()?;
            let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
            enc_ref.set_compute_pipeline_state(&pipeline);

            enc_ref.set_input_buffer(0, Some(q_buf), q_offset);
            enc_ref.set_input_buffer(1, Some(k_buf), k_offset);
            enc_ref.set_input_buffer(2, Some(v_buf), v_offset);
            enc_ref.set_output_buffer(3, Some(out_buf), out_offset);
            enc_ref.set_bytes(4, &params);

            if let Some(ref m_tensor) = mask_contig {
                let (m_storage, m_layout) = m_tensor.storage_and_layout();
                let m_buf = match &*m_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
                let m_dims = m_layout.dims();
                let m_str = m_layout.stride();
                // Zero-stride broadcast across batch/head/query-row dimensions when dim == 1
                let stride_b = if m_dims[0] == 1 { 0 } else { m_str[0] as i64 };
                let stride_h = if m_dims[1] == 1 { 0 } else { m_str[1] as i64 };
                let stride_q = if m_dims[2] == 1 { 0 } else { m_str[2] as i64 };
                let mask_params = AttnMaskParams {
                    m_strides: [stride_b, stride_h, stride_q],
                };
                let m_offset = m_layout.start_offset() * m_tensor.dtype().size_in_bytes();
                enc_ref.set_bytes(5, &mask_params);
                enc_ref.set_input_buffer(6, Some(m_buf), m_offset);
            }

            let grid_dims = MTLSize { width: nq, height: h, depth: b };
            let group_dims = MTLSize { width: 32, height: wm, depth: wn };
            enc_ref.dispatch_thread_groups(grid_dims, group_dims);
        }

        Ok(out_tensor)
    }
}
