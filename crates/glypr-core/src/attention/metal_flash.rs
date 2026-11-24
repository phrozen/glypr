use candle_core::{DType, Device, Result, Tensor};
use candle_metal_kernels::metal::{
    ComputePipeline, ConstantValues, Device as MtlDevice, Library, Value,
};
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

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

static SLIDING_LIB_CACHE: LazyLock<RwLock<HashMap<usize, Library>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static M5_LIB_CACHE: LazyLock<RwLock<HashMap<usize, Library>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
type PipelineCacheKey = (usize, bool, usize, bool, bool, bool);
static COMPUTE_PIPELINE_CACHE: LazyLock<RwLock<HashMap<PipelineCacheKey, ComputePipeline>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn get_or_compile_library(device: &MtlDevice, use_m5: bool) -> Result<Library> {
    let cache_lock = if use_m5 {
        &M5_LIB_CACHE
    } else {
        &SLIDING_LIB_CACHE
    };
    let dev_ptr = device.as_ref() as *const _ as usize;
    {
        let cache = cache_lock.read().unwrap_or_else(|e| e.into_inner());
        if let Some(lib) = cache.get(&dev_ptr) {
            return Ok(lib.clone());
        }
    }

    let library = if use_m5 {
        crate::metal_kernels::compile_metal4_library(device, M5_SHADER_SRC).map_err(|e| {
            candle_core::Error::Metal(
                format!("Failed to compile M5 Steel Attention Metal 4.0 library: {e}").into(),
            )
        })?
    } else {
        device
            .new_library_with_source(SLIDING_SHADER_SRC, None)
            .map_err(|e| {
                candle_core::Error::Metal(
                    format!("Failed to compile sliding-window Steel Attention Metal library: {e}")
                        .into(),
                )
            })?
    };

    let mut cache = cache_lock.write().unwrap_or_else(|e| e.into_inner());
    cache.insert(dev_ptr, library.clone());
    Ok(library)
}

fn get_or_compile_pipeline(
    device: &MtlDevice,
    head_dim: usize,
    align_q: bool,
    align_k: bool,
    has_mask: bool,
    use_m5: bool,
) -> Result<ComputePipeline> {
    let dev_ptr = device.as_ref() as *const _ as usize;
    let key = (dev_ptr, use_m5, head_dim, align_q, align_k, has_mask);
    {
        let cache = COMPUTE_PIPELINE_CACHE
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(p) = cache.get(&key) {
            return Ok(p.clone());
        }
    }

    let library = get_or_compile_library(device, use_m5)?;

    let cv = ConstantValues::new(vec![
        (200, Value::Bool(align_q)),
        (201, Value::Bool(align_k)),
        (300, Value::Bool(has_mask)),
        (301, Value::Bool(false)), // do_causal = false (bidirectional sliding window / full attention)
    ]);

    let func_name = match (use_m5, head_dim) {
        (true, 64) => "steel_attention_m5_bfloat16_bq64_bk32_bd64_wm4",
        (true, 256) => "steel_attention_m5_bfloat16_bq64_bk32_bd256_wm4",
        (true, 512) => "steel_attention_m5_bfloat16_bq64_bk32_bd512_wm4",
        (true, _) => {
            return Err(candle_core::Error::Msg(format!(
                "Unsupported head_dim for M5 MetalFlashAttention: {head_dim}"
            )));
        }
        (false, 64) => "steel_attention_bfloat16_bq32_bk32_bd64_wm4_wn1_maskbfloat16",
        (false, 256) => "steel_attention_bfloat16_bq32_bk16_bd256_wm4_wn1_maskbfloat16",
        (false, 512) => "steel_attention_bfloat16_bq16_bk16_bd512_wm2_wn1_maskbfloat16",
        (false, _) => {
            return Err(candle_core::Error::Msg(format!(
                "Unsupported head_dim for MetalFlashAttention: {head_dim}"
            )));
        }
    };

    let func = library.get_function(func_name, Some(&cv)).map_err(|e| {
        candle_core::Error::Metal(format!("Failed to get function {func_name}: {e}").into())
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| {
            candle_core::Error::Metal(format!("Failed to create compute pipeline: {e}").into())
        })?;

    let mut cache = COMPUTE_PIPELINE_CACHE
        .write()
        .unwrap_or_else(|e| e.into_inner());
    cache.insert(key, pipeline.clone());
    Ok(pipeline)
}

/// Dedicated Apple Silicon Sliding-Window & Full-Context Metal FlashAttention Engine.
///
/// Features:
/// - Custom Metal Shading Language kernel with hardware tile-skipping for sliding window attention.
/// - Supports `head_dim = 64` (Vision Tower), `head_dim = 256` (sliding layers), and `head_dim = 512` (global layers).
/// - On Apple M5 (`MTLGPUFamilyApple10` + `MTLGPUFamilyMetal4`), dispatches `mpp::tensor_ops::matmul2d`
///   cooperative tensor kernels (`BQ=64, BK=32`) directly on the per-core Neural Accelerators.
/// - Zero mask allocation (0 MB overhead) on unpadded sequences up to 8K, and $O(B \cdot S)$
///   stride-broadcasted key-padding mask on heterogeneous batches.
/// - Native GQA without `repeat_kv` tensor duplication.
/// - Automatic fallback to ChunkedSlidingAttention when running on CPU or unsupported dtypes.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalFlashAttention;

impl MetalFlashAttention {
    /// Pre-compile Steel FlashAttention pipelines for standard text head dimensions (256, 512).
    pub(crate) fn warmup_pipelines(device: &MtlDevice) -> Result<()> {
        let use_m5 = crate::metal_kernels::MetalCapabilities::detect(device).use_m5_tensor_ops();
        for head_dim in [256usize, 512] {
            for has_mask in [false, true] {
                let _ = get_or_compile_pipeline(device, head_dim, false, false, has_mask, use_m5)?;
            }
        }
        Ok(())
    }

    /// Compute FlashAttention returning output directly in `[B, S, H, D]` layout
    /// so callers can zero-copy `.reshape((B, S, H * D))` without a GPU transpose/copy.
    pub fn compute_attention_bshd(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        self.compute_attention_layout_with_tier(
            q,
            k,
            v,
            attention_mask,
            sliding_window,
            true,
            crate::metal_kernels::MetalTier::from_env(),
        )
    }

    pub(crate) fn compute_attention_with_tier(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
        tier: crate::metal_kernels::MetalTier,
    ) -> Result<Tensor> {
        self.compute_attention_layout_with_tier(
            q,
            k,
            v,
            attention_mask,
            sliding_window,
            false,
            tier,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_attention_layout_with_tier(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
        bshd_output: bool,
        tier: crate::metal_kernels::MetalTier,
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
                let out = ChunkedSlidingAttention.compute_attention(
                    q,
                    &k_rep,
                    &v_rep,
                    attention_mask,
                    sliding_window,
                )?;
                return if bshd_output {
                    out.transpose(1, 2)?.contiguous()
                } else {
                    Ok(out)
                };
            }
        };

        let (b, h, s, d) = q.dims4()?;
        let (kb, kh, ks, kd) = k.dims4()?;
        let (vb, vh, vs, vd) = v.dims4()?;
        if kh == 0 || kh != vh || !h.is_multiple_of(kh) {
            candle_core::bail!(
                "MetalFlashAttention: invalid GQA head counts q={h}, k={kh}, v={vh}"
            );
        }
        if (kb, ks, kd) != (b, s, d) || (vb, vs, vd) != (b, s, d) {
            candle_core::bail!(
                "MetalFlashAttention: shape mismatch q=({b},{h},{s},{d}) k=({kb},{kh},{ks},{kd}) v=({vb},{vh},{vs},{vd})"
            );
        }
        let gqa_factor = (h / kh) as i32;

        // Metal Flash Kernel supports head_dim = 64, 256, and 512 in BF16
        if !(d == 64 || d == 256 || d == 512)
            || q.dtype() != DType::BF16
            || k.dtype() != DType::BF16
            || v.dtype() != DType::BF16
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
            let out = ChunkedSlidingAttention.compute_attention(
                q,
                &k_rep,
                &v_rep,
                attention_mask,
                sliding_window,
            )?;
            return if bshd_output {
                out.transpose(1, 2)?.contiguous()
            } else {
                Ok(out)
            };
        }

        // Enforce unit last-dim stride for Metal Flash kernels
        let q = if q.layout().stride()[3] == 1 {
            q.clone()
        } else {
            q.contiguous()?
        };
        let k = if k.layout().stride()[3] == 1 {
            k.clone()
        } else {
            k.contiguous()?
        };
        let v = if v.layout().stride()[3] == 1 {
            v.clone()
        } else {
            v.contiguous()?
        };

        if b == 0 || h == 0 || s == 0 {
            let out_shape = if bshd_output {
                (b, s, h, d)
            } else {
                (b, h, s, d)
            };
            return Tensor::zeros(out_shape, DType::BF16, q.device());
        }

        // Validate optional additive mask ([B|1, H|1, S|1, S]) and convert to BF16 on Metal
        // rather than falling back to MetalSdpaAttention (which ignores sliding_window).
        let mask_contig = if let Some(m) = attention_mask {
            if !m.device().is_metal() {
                candle_core::bail!("MetalFlashAttention: attention_mask must be on Metal device");
            }
            let m_dims = m.dims();
            if m_dims.len() != 4
                || (m_dims[0] != 1 && m_dims[0] != b)
                || (m_dims[1] != 1 && m_dims[1] != h)
                || (m_dims[2] != 1 && m_dims[2] != s)
                || m_dims[3] != s
            {
                candle_core::bail!(
                    "MetalFlashAttention: invalid attention_mask shape {:?} for q shape [{b}, {h}, {s}, {d}]",
                    m_dims
                );
            }
            let m_bf16 = if m.dtype() != DType::BF16 {
                m.to_dtype(DType::BF16)?
            } else {
                m.clone()
            };
            Some(if m_bf16.is_contiguous() {
                m_bf16
            } else {
                m_bf16.contiguous()?
            })
        } else {
            None
        };

        let use_m5 = crate::metal_kernels::MetalCapabilities::detect(mdev.metal_device())
            .use_m5_tensor_ops_for_tier(tier);

        let (bq, bk, wm, wn) = if use_m5 {
            (64, 32, 4, 1)
        } else if d == 64 {
            (32, 32, 4, 1)
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

        let out_shape = if bshd_output {
            (b, s, h, d)
        } else {
            (b, h, s, d)
        };
        let out_tensor = Tensor::zeros(out_shape, DType::BF16, q.device())?;

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

            let o_strides = if bshd_output {
                [
                    out_layout.stride()[0] as i64, // batch stride: S * H * D
                    out_layout.stride()[2] as i64, // head stride: D
                    out_layout.stride()[1] as i64, // seq stride: H * D
                ]
            } else {
                [
                    out_layout.stride()[0] as i64, // batch stride: H * S * D
                    out_layout.stride()[1] as i64, // head stride: S * D
                    out_layout.stride()[2] as i64, // seq stride: D
                ]
            };

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
                o_strides,
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

impl AttentionBackend for MetalFlashAttention {
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        self.compute_attention_with_tier(
            q,
            k,
            v,
            attention_mask,
            sliding_window,
            crate::metal_kernels::MetalTier::from_env(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m5_flash_attention_parity_and_speed() -> Result<()> {
        if !crate::metal_kernels::is_metal_device_available() {
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

        for &(d, sliding) in &[(64usize, None), (256usize, Some(16usize)), (512usize, None)] {
            for &s in &[19usize, 64usize, 128usize, 145usize] {
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
                    // Verify M5 NAX pipeline compiles for this (d, s, has_mask) combination
                    assert!(
                        get_or_compile_pipeline(
                            mdev.metal_device(),
                            d,
                            (s % 64) == 0,
                            (s % 32) == 0,
                            mask_opt.is_some(),
                            true,
                        )
                        .is_ok(),
                        "Failed to compile M5 pipeline for d={d} s={s} masked={}",
                        mask_opt.is_some()
                    );

                    let ref_out = MetalFlashAttention.compute_attention_with_tier(
                        &q,
                        &k,
                        &v,
                        mask_opt,
                        sliding,
                        crate::metal_kernels::MetalTier::Legacy,
                    )?;
                    let m5_out = MetalFlashAttention.compute_attention_with_tier(
                        &q,
                        &k,
                        &v,
                        mask_opt,
                        sliding,
                        crate::metal_kernels::MetalTier::M5,
                    )?;
                    let auto_out = MetalFlashAttention.compute_attention_with_tier(
                        &q,
                        &k,
                        &v,
                        mask_opt,
                        sliding,
                        crate::metal_kernels::MetalTier::Auto,
                    )?;
                    let bshd_out = MetalFlashAttention.compute_attention_layout_with_tier(
                        &q,
                        &k,
                        &v,
                        mask_opt,
                        sliding,
                        true,
                        crate::metal_kernels::MetalTier::M5,
                    )?;

                    let ref_v = ref_out
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let m5_v = m5_out
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    if caps.use_m5_tensor_ops_for_tier(crate::metal_kernels::MetalTier::Auto) {
                        let auto_v = auto_out
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        assert_eq!(
                            auto_v,
                            m5_v,
                            "Auto tier must dispatch M5 NAX for d={d} s={s} masked={}",
                            mask_opt.is_some()
                        );
                    }
                    assert_eq!(bshd_out.dims(), &[b, s, h, d]);
                    let bshd_ref_v = m5_out
                        .transpose(1, 2)?
                        .contiguous()?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let bshd_v = bshd_out
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert_eq!(
                        bshd_v,
                        bshd_ref_v,
                        "BSHD layout output mismatch at d={d} s={s} masked={}",
                        mask_opt.is_some()
                    );
                    let mut max_diff = 0.0f32;
                    let mut dot = 0.0f64;
                    let mut na = 0.0f64;
                    let mut nb = 0.0f64;
                    for (i, (&a, &b_val)) in ref_v.iter().zip(m5_v.iter()).enumerate() {
                        assert!(
                            a.is_finite() && b_val.is_finite(),
                            "Non-finite at d={d} s={s} masked={} idx={i}: ref={a} m5={b_val}",
                            mask_opt.is_some()
                        );
                        let mag = a.abs().max(b_val.abs());
                        let tol = (0.01f32).max(mag * 0.016);
                        let diff = (a - b_val).abs();
                        if diff > max_diff {
                            max_diff = diff;
                        }
                        assert!(
                            diff <= tol,
                            "M5 FlashAttention mismatch at d={d} s={s} masked={} idx={i}: ref={a} m5={b_val} (diff={diff} > tol={tol})",
                            mask_opt.is_some()
                        );
                        dot += (a as f64) * (b_val as f64);
                        na += (a as f64) * (a as f64);
                        nb += (b_val as f64) * (b_val as f64);
                    }
                    let cos = if na > 0.0 && nb > 0.0 {
                        dot / (na.sqrt() * nb.sqrt())
                    } else {
                        1.0
                    };
                    println!(
                        "M5 FlashAttention d={d} s={s} masked={} max_diff={max_diff:.6e} cos={cos:.6} rms={:.6}",
                        mask_opt.is_some(),
                        (na / (ref_v.len() as f64)).sqrt(),
                    );
                    let min_cos = if d == 64 && s >= 128 { 0.9998 } else { 0.99995 };
                    assert!(
                        cos >= min_cos,
                        "M5 FlashAttention cosine similarity {cos:.6} < {min_cos} at d={d} s={s} masked={}",
                        mask_opt.is_some()
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_metal_flash_f32_mask_preserves_sliding_window_and_validates_dims() -> Result<()> {
        if !crate::metal_kernels::is_metal_device_available() {
            return Ok(());
        }
        let dev = Device::new_metal(0)?;
        let (b, h, kh, s, d) = (1, 2, 1, 32, 256);
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

        // F32 additive mask must convert to BF16 and STILL honor sliding_window=4
        let mut mask_vec = vec![0.0f32; b * s];
        mask_vec[s - 1] = f32::NEG_INFINITY;
        let mask_f32 = Tensor::from_vec(mask_vec, (b, 1, 1, s), &dev)?;
        let mask_bf16 = mask_f32.to_dtype(DType::BF16)?;

        let out_f32_mask =
            MetalFlashAttention.compute_attention(&q, &k, &v, Some(&mask_f32), Some(4))?;
        let out_bf16_mask =
            MetalFlashAttention.compute_attention(&q, &k, &v, Some(&mask_bf16), Some(4))?;
        let out_no_sw =
            MetalFlashAttention.compute_attention(&q, &k, &v, Some(&mask_bf16), None)?;

        let v_f32 = out_f32_mask
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let v_bf16 = out_bf16_mask
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let v_no_sw = out_no_sw
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            v_f32, v_bf16,
            "F32 mask must match BF16 mask on MetalFlashAttention"
        );
        assert_ne!(
            v_f32, v_no_sw,
            "F32 mask must preserve sliding_window instead of dropping it"
        );

        // Non-unit stride[3] tensor (transposed last two dims) must materialize and succeed
        let q_transposed = Tensor::zeros((b, h, d, s), DType::BF16, &dev)?.transpose(2, 3)?;
        assert_ne!(q_transposed.layout().stride()[3], 1);
        let out_t = MetalFlashAttention.compute_attention(&q_transposed, &k, &v, None, Some(4))?;
        assert_eq!(out_t.dims(), &[b, h, s, d]);

        // Invalid mask dimensions must return Err
        let bad_mask = Tensor::zeros((b, h + 1, 1, s), DType::BF16, &dev)?;
        assert!(
            MetalFlashAttention
                .compute_attention(&q, &k, &v, Some(&bad_mask), Some(4))
                .is_err()
        );
        Ok(())
    }
}
