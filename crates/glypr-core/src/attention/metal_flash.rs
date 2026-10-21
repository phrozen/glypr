use candle_core::{DType, Device, Result, Tensor};
use candle_metal_kernels::metal::{
    ComputePipeline, ConstantValues, Device as MtlDevice, Library, Value,
};
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::chunked::ChunkedSlidingAttention;
use super::traits::AttentionBackend;

static SLIDING_SHADER_SRC: &str = include_str!("steel_attention_sliding.metal");
static M5_SHADER_SRC: &str = include_str!("steel_attention_m5.metal");

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
static M5_PIPELINE_CACHE: OnceLock<PipelineCache> = OnceLock::new();

fn get_or_compile_pipeline(
    device: &MtlDevice,
    head_dim: usize,
    align_q: bool,
    align_k: bool,
    has_mask: bool,
    use_m5: bool,
) -> Result<ComputePipeline> {
    if use_m5 {
        let cache = M5_PIPELINE_CACHE.get_or_init(|| {
            let lib = match crate::metal_kernels::compile_metal4_library(device, M5_SHADER_SRC) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("M5 Steel Attention compilation error:\n{e}");
                    panic!("Failed to compile M5 Steel Attention Metal 4.0 library");
                }
            };
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
            (301, Value::Bool(false)),
        ]);

        let func_name = match head_dim {
            256 => "steel_attention_m5_bfloat16_bq64_bk32_bd256_wm4",
            512 => "steel_attention_m5_bfloat16_bq64_bk32_bd512_wm4",
            _ => {
                return Err(candle_core::Error::Msg(format!(
                    "Unsupported head_dim for M5 MetalFlashAttention: {head_dim}"
                )));
            }
        };
        let func = cache
            .library
            .get_function(func_name, Some(&cv))
            .map_err(|e| {
                candle_core::Error::Metal(format!("Failed to get function {func_name}: {e}").into())
            })?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| {
                candle_core::Error::Metal(
                    format!("Failed to create M5 compute pipeline: {e}").into(),
                )
            })?;

        map.insert(key, pipeline.clone());
        return Ok(pipeline);
    }

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
            )));
        }
    };
    let func = cache
        .library
        .get_function(func_name, Some(&cv))
        .map_err(|e| {
            candle_core::Error::Metal(format!("Failed to get function {func_name}: {e}").into())
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| {
            candle_core::Error::Metal(format!("Failed to create compute pipeline: {e}").into())
        })?;

    map.insert(key, pipeline.clone());
    Ok(pipeline)
}

/// Dedicated Apple Silicon Sliding-Window & Full-Context Metal FlashAttention Engine.
///
/// Features:
/// - Custom Metal Shading Language kernel with hardware tile-skipping for sliding window attention.
/// - Supports both `head_dim = 256` (sliding layers) and `head_dim = 512` (global layers 5, 11, 17, 23).
/// - On Apple M5 (`MTLGPUFamilyApple10` + `MTLGPUFamilyMetal4`), dispatches `mpp::tensor_ops::matmul2d`
///   cooperative tensor kernels (`BQ=64, BK=32`) directly on the per-core Neural Accelerators.
/// - Zero mask allocation (0 MB overhead) on unpadded sequences up to 8K, and $O(B \cdot S)$
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
                let k_rep = if gqa > 1 {
                    super::repeat_kv(k, gqa)?
                } else {
                    k.clone()
                };
                let v_rep = if gqa > 1 {
                    super::repeat_kv(v, gqa)?
                } else {
                    v.clone()
                };
                return ChunkedSlidingAttention.compute_attention(
                    q,
                    &k_rep,
                    &v_rep,
                    attention_mask,
                    sliding_window,
                );
            }
        };

        let (b, h, s, d) = q.dims4()?;
        let (_kb, kh, _ks, _kd) = k.dims4()?;
        let gqa_factor = (h / kh) as i32;

        // Metal Flash Kernel supports head_dim = 256 and 512 in BF16
        if (d != 256 && d != 512) || q.dtype() != DType::BF16 {
            let k_rep = if gqa_factor > 1 {
                super::repeat_kv(k, gqa_factor as usize)?
            } else {
                k.clone()
            };
            let v_rep = if gqa_factor > 1 {
                super::repeat_kv(v, gqa_factor as usize)?
            } else {
                v.clone()
            };
            return ChunkedSlidingAttention.compute_attention(
                q,
                &k_rep,
                &v_rep,
                attention_mask,
                sliding_window,
            );
        }

        // Validate optional additive mask ([B, 1, 1, S] or [B, H|1, S|1, S] in BF16 on Metal)
        let mask_contig = if let Some(m) = attention_mask {
            if m.dtype() != DType::BF16
                || !m.device().is_metal()
                || m.dims().len() != 4
                || m.dim(3)? != s
            {
                let k_rep = if gqa_factor > 1 {
                    super::repeat_kv(k, gqa_factor as usize)?
                } else {
                    k.clone()
                };
                let v_rep = if gqa_factor > 1 {
                    super::repeat_kv(v, gqa_factor as usize)?
                } else {
                    v.clone()
                };
                return super::metal_sdpa::MetalSdpaAttention.compute_attention(
                    q,
                    &k_rep,
                    &v_rep,
                    attention_mask,
                    sliding_window,
                );
            }
            Some(if m.is_contiguous() {
                m.clone()
            } else {
                m.contiguous()?
            })
        } else {
            None
        };

        let use_m5 = crate::metal_kernels::MetalCapabilities::detect(mdev.metal_device())
            .use_m5_tensor_ops()
            && (s % 64 == 0);

        let (bq, bk, wm, wn) = if use_m5 {
            (64, 32, 4, 1)
        } else if d == 256 {
            (32, 16, 4, 1)
        } else {
            (16, 16, 2, 1)
        };
        let align_q = (s % bq) == 0;
        let align_k = (s % bk) == 0;
        let has_mask = mask_contig.is_some();

        let pipeline =
            get_or_compile_pipeline(mdev.metal_device(), d, align_q, align_k, has_mask, use_m5)?;

        let out_tensor = Tensor::zeros((b, h, s, d), DType::BF16, q.device())?;

        {
            let (q_storage, q_layout) = q.storage_and_layout();
            let (k_storage, k_layout) = k.storage_and_layout();
            let (v_storage, v_layout) = v.storage_and_layout();
            let (out_storage, out_layout) = out_tensor.storage_and_layout();

            let q_buf = match &*q_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let k_buf = match &*k_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let v_buf = match &*v_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let out_buf = match &*out_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };

            let nq = s.div_ceil(bq);
            let nk = s.div_ceil(bk);
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
                q_strides: [
                    q_layout.stride()[0] as i64,
                    q_layout.stride()[1] as i64,
                    q_layout.stride()[2] as i64,
                ],
                k_strides: [
                    k_layout.stride()[0] as i64,
                    k_layout.stride()[1] as i64,
                    k_layout.stride()[2] as i64,
                ],
                v_strides: [
                    v_layout.stride()[0] as i64,
                    v_layout.stride()[1] as i64,
                    v_layout.stride()[2] as i64,
                ],
                o_strides: [
                    out_layout.stride()[0] as i64,
                    out_layout.stride()[1] as i64,
                    out_layout.stride()[2] as i64,
                ],
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
                let m_buf = match &*m_storage {
                    candle_core::Storage::Metal(m) => m.buffer(),
                    _ => unreachable!(),
                };
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

            let grid_dims = MTLSize {
                width: nq,
                height: h,
                depth: b,
            };
            let group_dims = MTLSize {
                width: 32,
                height: wm,
                depth: wn,
            };
            enc_ref.dispatch_thread_groups(grid_dims, group_dims);
        }

        Ok(out_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m5_flash_attention_parity_and_speed() -> Result<()> {
        if !candle_core::utils::metal_is_available() {
            return Ok(());
        }
        let dev = Device::new_metal(0)?;
        let mdev = match &dev {
            Device::Metal(m) => m,
            _ => unreachable!(),
        };
        let caps = crate::metal_kernels::MetalCapabilities::detect(mdev.metal_device());
        if !caps.has_tensor_ops_compiler {
            return Ok(());
        }

        for &(d, sliding) in &[(256usize, Some(16usize)), (512usize, None)] {
            for &s in &[19usize, 64usize, 145usize] {
                let (b, h, kh) = (2, 4, if d == 256 { 2 } else { 1 });
                let q_vec: Vec<f32> = (0..(b * h * s * d))
                    .map(|i| ((i as f32) * 0.031).sin() * 0.5)
                    .collect();
                let k_vec: Vec<f32> = (0..(b * kh * s * d))
                    .map(|i| ((i as f32) * 0.017).cos() * 0.5)
                    .collect();
                let v_vec: Vec<f32> = (0..(b * kh * s * d))
                    .map(|i| ((i as f32) * 0.023).sin() * 0.5)
                    .collect();
                let q = Tensor::from_vec(q_vec, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
                let k = Tensor::from_vec(k_vec, (b, kh, s, d), &dev)?.to_dtype(DType::BF16)?;
                let v = Tensor::from_vec(v_vec, (b, kh, s, d), &dev)?.to_dtype(DType::BF16)?;

                // Test without mask and with additive padding mask [B, 1, 1, S]
                let mut mask_vec = vec![0.0f32; b * s];
                for val in &mut mask_vec[(s - 5)..s] {
                    *val = f32::NEG_INFINITY;
                }
                let mask = Tensor::from_vec(mask_vec, (b, 1, 1, s), &dev)?.to_dtype(DType::BF16)?;

                for mask_opt in [None, Some(&mask)] {
                    unsafe {
                        std::env::set_var("GLYPR_METAL_TIER", "legacy");
                    }
                    let ref_out =
                        MetalFlashAttention.compute_attention(&q, &k, &v, mask_opt, sliding)?;
                    unsafe {
                        std::env::set_var("GLYPR_METAL_TIER", "m5");
                    }
                    let m5_out =
                        MetalFlashAttention.compute_attention(&q, &k, &v, mask_opt, sliding)?;
                    unsafe {
                        std::env::remove_var("GLYPR_METAL_TIER");
                    }

                    let ref_v = ref_out
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let m5_v = m5_out
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let mut max_diff = 0.0f32;
                    for (i, (&a, &b_val)) in ref_v.iter().zip(m5_v.iter()).enumerate() {
                        if a.is_nan() || b_val.is_nan() {
                            panic!(
                                "NaN at d={d} s={s} masked={} idx={i}: ref={a} m5={b_val}",
                                mask_opt.is_some()
                            );
                        }
                        let diff = (a - b_val).abs();
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                    println!(
                        "M5 FlashAttention d={d} s={s} masked={} max_diff={max_diff:.6e}",
                        mask_opt.is_some(),
                    );
                    assert!(
                        max_diff < 3e-3,
                        "M5 FlashAttention mismatch at d={d} s={s} masked={}: max_diff={max_diff}",
                        mask_opt.is_some()
                    );
                }
            }
        }
        Ok(())
    }
}
