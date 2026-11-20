pub mod capabilities;
pub use capabilities::{
    MetalCapabilities, MetalTier, compile_metal4_library, is_metal_device_available,
};

use candle_core::{DType, Device, Result, Tensor};
use candle_metal_kernels::metal::{ComputePipeline, Device as MtlDevice, Library};
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

const FUSED_OPS_SOURCE: &str = include_str!("fused_ops.metal");
const M5_OPS_SOURCE: &str = include_str!("m5_ops.metal");

static FUSED_LIB_CACHE: LazyLock<RwLock<HashMap<usize, Library>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static M5_LIB_CACHE: LazyLock<RwLock<HashMap<usize, Library>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static PIPELINE_CACHE: LazyLock<RwLock<HashMap<(usize, &'static str), ComputePipeline>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn get_or_compile_fused_library(dev: &MtlDevice) -> Result<Library> {
    let dev_ptr = dev.as_ref() as *const _ as usize;
    {
        let cache = FUSED_LIB_CACHE.read().unwrap_or_else(|e| e.into_inner());
        if let Some(lib) = cache.get(&dev_ptr) {
            return Ok(lib.clone());
        }
    }

    let library = dev
        .new_library_with_source(FUSED_OPS_SOURCE, None)
        .map_err(|e| {
            candle_core::Error::Metal(format!("Failed to compile fused_ops.metal: {:?}", e).into())
        })?;

    let mut cache = FUSED_LIB_CACHE.write().unwrap_or_else(|e| e.into_inner());
    cache.insert(dev_ptr, library.clone());
    Ok(library)
}

fn get_or_compile_m5_library(dev: &MtlDevice) -> Result<Library> {
    let dev_ptr = dev.as_ref() as *const _ as usize;
    {
        let cache = M5_LIB_CACHE.read().unwrap_or_else(|e| e.into_inner());
        if let Some(lib) = cache.get(&dev_ptr) {
            return Ok(lib.clone());
        }
    }

    let library = compile_metal4_library(dev, M5_OPS_SOURCE).map_err(|e| {
        candle_core::Error::Metal(format!("Failed to compile m5_ops.metal: {:?}", e).into())
    })?;

    let mut cache = M5_LIB_CACHE.write().unwrap_or_else(|e| e.into_inner());
    cache.insert(dev_ptr, library.clone());
    Ok(library)
}

fn get_or_compile_pipeline(dev: &MtlDevice, fn_name: &'static str) -> Result<ComputePipeline> {
    let dev_key = (dev.as_ref() as *const _ as usize, fn_name);
    {
        let cache = PIPELINE_CACHE.read().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = cache.get(&dev_key) {
            return Ok(p.clone());
        }
    }

    let library = get_or_compile_fused_library(dev)?;

    let function = library.get_function(fn_name, None).map_err(|e| {
        candle_core::Error::Metal(format!("Function {} not found: {:?}", fn_name, e).into())
    })?;

    let pipeline = dev
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            candle_core::Error::Metal(
                format!("Failed to create {} pipeline: {:?}", fn_name, e).into(),
            )
        })?;

    let mut cache = PIPELINE_CACHE.write().unwrap_or_else(|e| e.into_inner());
    cache.insert(dev_key, pipeline.clone());
    Ok(pipeline)
}

fn get_or_compile_m5_pipeline(dev: &MtlDevice, fn_name: &'static str) -> Result<ComputePipeline> {
    let dev_key = (dev.as_ref() as *const _ as usize, fn_name);
    {
        let cache = PIPELINE_CACHE.read().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = cache.get(&dev_key) {
            return Ok(p.clone());
        }
    }

    let library = get_or_compile_m5_library(dev)?;

    let function = library.get_function(fn_name, None).map_err(|e| {
        candle_core::Error::Metal(format!("Function {} not found: {:?}", fn_name, e).into())
    })?;

    let pipeline = dev
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            candle_core::Error::Metal(
                format!("Failed to create {} pipeline: {:?}", fn_name, e).into(),
            )
        })?;

    let mut cache = PIPELINE_CACHE.write().unwrap_or_else(|e| e.into_inner());
    cache.insert(dev_key, pipeline.clone());
    Ok(pipeline)
}

/// Pre-compile hot fused Metal pipelines (and M5 Neural Accelerator pipelines when supported)
/// so the first inference request incurs zero shader-compilation jitter.
pub(crate) fn warmup_pipelines(dev: &MtlDevice) -> Result<()> {
    for name in [
        "rms_norm_bf16",
        "rms_norm_residual_add_bf16",
        "qkv_rmsnorm_rope_transpose_bf16",
        "ple_scale_rms_norm_transpose_bf16",
        "ple_scale_rms_norm_bf16",
        "geglu_gate_up_bf16",
        "pool_mean_bf16_to_f32",
        "l2norm_f32_to_bf16",
        "q8_0_mv_bf16",
    ] {
        let _ = get_or_compile_pipeline(dev, name)?;
    }

    if MetalCapabilities::detect(dev).use_m5_tensor_ops() {
        for name in ["bf16_mm_m5", "q8_0_mm_m5_bf16"] {
            let _ = get_or_compile_m5_pipeline(dev, name)?;
        }
    }

    Ok(())
}

fn metal_device_of<'a>(
    t: &'a Tensor,
    fn_name: &str,
    arg_name: &str,
) -> Result<&'a candle_core::MetalDevice> {
    match t.device() {
        Device::Metal(m) => Ok(m),
        _ => candle_core::bail!("{fn_name}: {arg_name} requires a Metal device"),
    }
}

fn ensure_metal_bf16<'a>(
    t: &'a Tensor,
    fn_name: &str,
    arg_name: &str,
) -> Result<&'a candle_core::MetalDevice> {
    let mdev = metal_device_of(t, fn_name, arg_name)?;
    if t.dtype() != DType::BF16 {
        candle_core::bail!(
            "{fn_name}: {arg_name} must have dtype BF16, got {:?}",
            t.dtype()
        );
    }
    if t.dims().is_empty() {
        candle_core::bail!("{fn_name}: {arg_name} cannot be 0-D");
    }
    Ok(mdev)
}

fn prepare_metal_weight(
    w: &Tensor,
    expected_len: usize,
    base_device: &Device,
    fn_name: &str,
    arg_name: &str,
) -> Result<Tensor> {
    if !w.device().same_device(base_device) {
        candle_core::bail!("{fn_name}: {arg_name} must be on the same Metal device");
    }
    if w.dims() != [expected_len] {
        candle_core::bail!(
            "{fn_name}: {arg_name} shape {:?} does not match expected [{expected_len}]",
            w.dims()
        );
    }
    let w = if !w.is_contiguous() {
        w.contiguous()?
    } else {
        w.clone()
    };
    if w.dtype() != DType::BF16 {
        w.to_dtype(DType::BF16)
    } else {
        Ok(w)
    }
}

fn prepare_metal_rope_table(
    t: &Tensor,
    seq_len: usize,
    head_dim: usize,
    base_device: &Device,
    fn_name: &str,
    arg_name: &str,
) -> Result<Tensor> {
    if !t.device().same_device(base_device) {
        candle_core::bail!("{fn_name}: {arg_name} must be on the same Metal device");
    }
    if t.dtype() != DType::BF16 {
        candle_core::bail!(
            "{fn_name}: {arg_name} must have dtype BF16, got {:?}",
            t.dtype()
        );
    }
    let dims = t.dims();
    let expected_elems = seq_len
        .checked_mul(head_dim)
        .ok_or_else(|| candle_core::Error::Msg(format!("{fn_name}: {arg_name} size overflow")))?;
    if dims.is_empty() || dims[dims.len() - 1] != head_dim || t.elem_count() < expected_elems {
        candle_core::bail!(
            "{fn_name}: {arg_name} shape {:?} incompatible with seq_len={seq_len}, head_dim={head_dim}",
            dims
        );
    }
    if !t.is_contiguous() {
        t.contiguous()
    } else {
        Ok(t.clone())
    }
}

fn checked_4d_elems(
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    fn_name: &str,
) -> Result<usize> {
    batch
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(heads))
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| candle_core::Error::Msg(format!("{fn_name}: tensor dimension overflow")))
}

/// Fused Metal RMSNorm executing directly in threadgroup registers via SIMD shuffle reduction.
pub fn fused_metal_rmsnorm(x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let mdev = ensure_metal_bf16(x, "fused_metal_rmsnorm", "x")?;
    let dims = x.dims();
    let d = dims[dims.len() - 1];
    if d == 0 {
        candle_core::bail!("fused_metal_rmsnorm: last dimension must be > 0");
    }

    let weight_holder: Option<Tensor> = match weight {
        Some(w) => Some(prepare_metal_weight(
            w,
            d,
            x.device(),
            "fused_metal_rmsnorm",
            "weight",
        )?),
        None => None,
    };
    let weight_ref = weight_holder.as_ref();

    let x = if !x.is_contiguous() {
        x.contiguous()?
    } else {
        x.clone()
    };

    let orig_shape = x.shape();
    let num_rows = x.elem_count() / d;
    if num_rows == 0 {
        return Tensor::zeros(orig_shape.clone(), DType::BF16, x.device());
    }
    let out = Tensor::zeros(orig_shape.clone(), DType::BF16, x.device())?;

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "rms_norm_bf16")?;

    let has_weight: i32 = if weight_ref.is_some() { 1 } else { 0 };
    let d_i32 = d as i32;
    let eps_f32 = eps as f32;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let x_buf = match &*x_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = x.dtype().size_in_bytes();
        let x_offset = x_layout.start_offset() * elem_bytes;
        let out_offset = out_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(x_buf), x_offset);

        if let Some(w) = weight_ref {
            let (w_storage, w_layout) = w.storage_and_layout();
            let w_buf = match &*w_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let w_offset = w_layout.start_offset() * w.dtype().size_in_bytes();
            enc_ref.set_input_buffer(1, Some(w_buf), w_offset);
        } else {
            // Null or placeholder buffer
            enc_ref.set_input_buffer(1, Some(x_buf), x_offset);
        }

        enc_ref.set_output_buffer(2, Some(out_buf), out_offset);
        enc_ref.set_bytes(3, &eps_f32);
        enc_ref.set_bytes(4, &d_i32);
        enc_ref.set_bytes(5, &has_weight);

        let grid_dims = MTLSize {
            width: num_rows,
            height: 1,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Metal Post-Norm + Residual Add (+ optional Layer Scalar) in a single dispatch.
pub fn fused_metal_rmsnorm_residual_add(
    x: &Tensor,
    residual: &Tensor,
    weight: Option<&Tensor>,
    layer_scalar: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    let fn_name = "fused_metal_rmsnorm_residual_add";
    let mdev = ensure_metal_bf16(x, fn_name, "x")?;
    ensure_metal_bf16(residual, fn_name, "residual")?;
    if !residual.device().same_device(x.device()) {
        candle_core::bail!("{fn_name}: residual must be on the same Metal device");
    }
    if x.shape() != residual.shape() {
        candle_core::bail!(
            "{fn_name}: shape mismatch x={:?} vs residual={:?}",
            x.dims(),
            residual.dims()
        );
    }
    let dims = x.dims();
    let d = dims[dims.len() - 1];
    if d == 0 {
        candle_core::bail!("{fn_name}: last dimension must be > 0");
    }

    let weight_holder: Option<Tensor> = match weight {
        Some(w) => Some(prepare_metal_weight(w, d, x.device(), fn_name, "weight")?),
        None => None,
    };
    let weight_ref = weight_holder.as_ref();

    let scalar_holder: Option<Tensor> = match layer_scalar {
        Some(s) => {
            if !s.device().same_device(x.device()) {
                candle_core::bail!("{fn_name}: layer_scalar must be on the same Metal device");
            }
            if s.elem_count() != 1 {
                candle_core::bail!(
                    "{fn_name}: layer_scalar must have 1 element, got shape {:?}",
                    s.dims()
                );
            }
            let s = if !s.is_contiguous() {
                s.contiguous()?
            } else {
                s.clone()
            };
            Some(if s.dtype() != DType::BF16 {
                s.to_dtype(DType::BF16)?
            } else {
                s
            })
        }
        None => None,
    };
    let scalar_ref = scalar_holder.as_ref();

    let x = if !x.is_contiguous() {
        x.contiguous()?
    } else {
        x.clone()
    };
    let residual = if !residual.is_contiguous() {
        residual.contiguous()?
    } else {
        residual.clone()
    };

    let orig_shape = x.shape();
    let num_rows = x.elem_count() / d;
    if num_rows == 0 {
        return Tensor::zeros(orig_shape.clone(), DType::BF16, x.device());
    }
    let out = Tensor::zeros(orig_shape.clone(), DType::BF16, x.device())?;

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "rms_norm_residual_add_bf16")?;

    let has_weight: i32 = if weight_ref.is_some() { 1 } else { 0 };
    let has_scale: i32 = if scalar_ref.is_some() { 1 } else { 0 };
    let d_i32 = d as i32;
    let eps_f32 = eps as f32;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (res_storage, res_layout) = residual.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let x_buf = match &*x_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let res_buf = match &*res_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = x.dtype().size_in_bytes();
        let x_offset = x_layout.start_offset() * elem_bytes;
        let res_offset = res_layout.start_offset() * elem_bytes;
        let out_offset = out_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(x_buf), x_offset);
        enc_ref.set_input_buffer(1, Some(res_buf), res_offset);

        if let Some(w) = weight_ref {
            let (w_storage, w_layout) = w.storage_and_layout();
            let w_buf = match &*w_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let w_offset = w_layout.start_offset() * w.dtype().size_in_bytes();
            enc_ref.set_input_buffer(2, Some(w_buf), w_offset);
        } else {
            enc_ref.set_input_buffer(2, Some(x_buf), x_offset);
        }

        if let Some(s) = scalar_ref {
            let (s_storage, s_layout) = s.storage_and_layout();
            let s_buf = match &*s_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let s_offset = s_layout.start_offset() * s.dtype().size_in_bytes();
            enc_ref.set_input_buffer(3, Some(s_buf), s_offset);
        } else {
            enc_ref.set_input_buffer(3, Some(x_buf), x_offset);
        }

        enc_ref.set_output_buffer(4, Some(out_buf), out_offset);
        enc_ref.set_bytes(5, &eps_f32);
        enc_ref.set_bytes(6, &d_i32);
        enc_ref.set_bytes(7, &has_weight);
        enc_ref.set_bytes(8, &has_scale);

        let grid_dims = MTLSize {
            width: num_rows,
            height: 1,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Metal Q/K RMSNorm + RoPE + [B, S, H, D] -> [B, H, S, D] transpose in a single dispatch.
#[allow(clippy::too_many_arguments)]
pub fn fused_metal_qk_rmsnorm_rope_transpose(
    x: &Tensor,
    weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f64,
) -> Result<Tensor> {
    let fn_name = "fused_metal_qk_rmsnorm_rope_transpose";
    let mdev = ensure_metal_bf16(x, fn_name, "x")?;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        candle_core::bail!("{fn_name}: head_dim must be a positive even integer, got {head_dim}");
    }
    if num_heads == 0 {
        candle_core::bail!("{fn_name}: num_heads must be > 0");
    }
    let expected_elems = checked_4d_elems(batch, seq_len, num_heads, head_dim, fn_name)?;
    if x.elem_count() != expected_elems {
        candle_core::bail!(
            "{fn_name}: x elem_count {} does not match batch*seq*heads*head_dim = {}",
            x.elem_count(),
            expected_elems
        );
    }

    let weight = prepare_metal_weight(weight, head_dim, x.device(), fn_name, "weight")?;
    let cos = prepare_metal_rope_table(cos, seq_len, head_dim, x.device(), fn_name, "cos")?;
    let sin = prepare_metal_rope_table(sin, seq_len, head_dim, x.device(), fn_name, "sin")?;

    let x = if !x.is_contiguous() {
        x.contiguous()?
    } else {
        x.clone()
    };

    if batch == 0 || seq_len == 0 {
        return Tensor::zeros(
            (batch, num_heads, seq_len, head_dim),
            DType::BF16,
            x.device(),
        );
    }
    let out = Tensor::zeros(
        (batch, num_heads, seq_len, head_dim),
        DType::BF16,
        x.device(),
    )?;

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "rms_norm_rope_transpose_bf16")?;

    let eps_f32 = eps as f32;
    let s_i32 = seq_len as i32;
    let h_i32 = num_heads as i32;
    let d_i32 = head_dim as i32;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (cos_storage, cos_layout) = cos.storage_and_layout();
        let (sin_storage, sin_layout) = sin.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let x_buf = match &*x_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let w_buf = match &*w_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let cos_buf = match &*cos_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let sin_buf = match &*sin_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = DType::BF16.size_in_bytes();
        let x_offset = x_layout.start_offset() * elem_bytes;
        let w_offset = w_layout.start_offset() * elem_bytes;
        let cos_offset = cos_layout.start_offset() * elem_bytes;
        let sin_offset = sin_layout.start_offset() * elem_bytes;
        let out_offset = out_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(x_buf), x_offset);
        enc_ref.set_input_buffer(1, Some(w_buf), w_offset);
        enc_ref.set_input_buffer(2, Some(cos_buf), cos_offset);
        enc_ref.set_input_buffer(3, Some(sin_buf), sin_offset);
        enc_ref.set_output_buffer(4, Some(out_buf), out_offset);
        enc_ref.set_bytes(5, &eps_f32);
        enc_ref.set_bytes(6, &s_i32);
        enc_ref.set_bytes(7, &h_i32);
        enc_ref.set_bytes(8, &d_i32);

        let grid_dims = MTLSize {
            width: num_heads,
            height: seq_len,
            depth: batch,
        };
        let group_dims = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Metal V RMSNorm + [B, S, H, D] -> [B, H, S, D] transpose in a single dispatch.
pub fn fused_metal_v_rmsnorm_transpose(
    x: &Tensor,
    weight: Option<&Tensor>,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f64,
) -> Result<Tensor> {
    let fn_name = "fused_metal_v_rmsnorm_transpose";
    let mdev = ensure_metal_bf16(x, fn_name, "x")?;
    if head_dim == 0 || num_heads == 0 {
        candle_core::bail!("{fn_name}: num_heads and head_dim must be > 0");
    }
    let expected_elems = checked_4d_elems(batch, seq_len, num_heads, head_dim, fn_name)?;
    if x.elem_count() != expected_elems {
        candle_core::bail!(
            "{fn_name}: x elem_count {} does not match batch*seq*heads*head_dim = {}",
            x.elem_count(),
            expected_elems
        );
    }

    let weight_holder: Option<Tensor> = match weight {
        Some(w) => Some(prepare_metal_weight(
            w,
            head_dim,
            x.device(),
            fn_name,
            "weight",
        )?),
        None => None,
    };
    let weight_ref = weight_holder.as_ref();

    let x = if !x.is_contiguous() {
        x.contiguous()?
    } else {
        x.clone()
    };

    if batch == 0 || seq_len == 0 {
        return Tensor::zeros(
            (batch, num_heads, seq_len, head_dim),
            DType::BF16,
            x.device(),
        );
    }
    let out = Tensor::zeros(
        (batch, num_heads, seq_len, head_dim),
        DType::BF16,
        x.device(),
    )?;

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "rms_norm_transpose_bf16")?;

    let eps_f32 = eps as f32;
    let s_i32 = seq_len as i32;
    let h_i32 = num_heads as i32;
    let d_i32 = head_dim as i32;
    let has_weight: i32 = if weight_ref.is_some() { 1 } else { 0 };

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let x_buf = match &*x_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = DType::BF16.size_in_bytes();
        let x_offset = x_layout.start_offset() * elem_bytes;
        let out_offset = out_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(x_buf), x_offset);
        if let Some(w) = weight_ref {
            let (w_storage, w_layout) = w.storage_and_layout();
            let w_buf = match &*w_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let w_offset = w_layout.start_offset() * elem_bytes;
            enc_ref.set_input_buffer(1, Some(w_buf), w_offset);
        } else {
            enc_ref.set_input_buffer(1, Some(x_buf), x_offset);
        }
        enc_ref.set_output_buffer(2, Some(out_buf), out_offset);
        enc_ref.set_bytes(3, &eps_f32);
        enc_ref.set_bytes(4, &s_i32);
        enc_ref.set_bytes(5, &h_i32);
        enc_ref.set_bytes(6, &d_i32);
        enc_ref.set_bytes(7, &has_weight);

        let grid_dims = MTLSize {
            width: num_heads,
            height: seq_len,
            depth: batch,
        };
        let group_dims = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Metal 4D PLE Scale + RMSNorm + [B, S, L, D] -> [L, B, S, D] Permute in a single dispatch.
/// Eliminates the 256 MB `(proj * scale)` buffer at 8K tokens and all 24 per-layer `.contiguous()` copies.
#[allow(clippy::too_many_arguments)]
pub fn fused_metal_ple_scale_rmsnorm_transpose(
    x: &Tensor,
    weight: &Tensor,
    scale: f64,
    eps: f64,
    batch: usize,
    seq_len: usize,
    num_layers: usize,
    d: usize,
) -> Result<Tensor> {
    let fn_name = "fused_metal_ple_scale_rmsnorm_transpose";
    let mdev = ensure_metal_bf16(x, fn_name, "x")?;
    if d == 0 || num_layers == 0 {
        candle_core::bail!("{fn_name}: num_layers and d must be > 0");
    }
    let expected_elems = checked_4d_elems(batch, seq_len, num_layers, d, fn_name)?;
    if x.elem_count() != expected_elems {
        candle_core::bail!(
            "{fn_name}: x elem_count {} does not match batch*seq*layers*d = {}",
            x.elem_count(),
            expected_elems
        );
    }

    let weight = prepare_metal_weight(weight, d, x.device(), fn_name, "weight")?;
    let x = if !x.is_contiguous() {
        x.contiguous()?
    } else {
        x.clone()
    };

    if batch == 0 || seq_len == 0 {
        return Tensor::zeros((num_layers, batch, seq_len, d), DType::BF16, x.device());
    }
    let out = Tensor::zeros((num_layers, batch, seq_len, d), DType::BF16, x.device())?;

    let pipeline =
        get_or_compile_pipeline(mdev.metal_device(), "ple_scale_rms_norm_transpose_bf16")?;

    let scale_f32 = scale as f32;
    let eps_f32 = eps as f32;
    let b_i32 = batch as i32;
    let s_i32 = seq_len as i32;
    let l_i32 = num_layers as i32;
    let d_i32 = d as i32;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let x_buf = match &*x_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let w_buf = match &*w_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = DType::BF16.size_in_bytes();
        let x_offset = x_layout.start_offset() * elem_bytes;
        let w_offset = w_layout.start_offset() * elem_bytes;
        let out_offset = out_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(x_buf), x_offset);
        enc_ref.set_input_buffer(1, Some(w_buf), w_offset);
        enc_ref.set_output_buffer(2, Some(out_buf), out_offset);
        enc_ref.set_bytes(3, &scale_f32);
        enc_ref.set_bytes(4, &eps_f32);
        enc_ref.set_bytes(5, &b_i32);
        enc_ref.set_bytes(6, &s_i32);
        enc_ref.set_bytes(7, &l_i32);
        enc_ref.set_bytes(8, &d_i32);

        let grid_dims = MTLSize {
            width: num_layers,
            height: seq_len,
            depth: batch,
        };
        let group_dims = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Metal Single-Layer PLE Scale + RMSNorm (`[B, S, D] -> [B, S, D]`).
/// Used by the hybrid per-layer PLE path for large `B * S` to avoid holding `[L, B, S, D]` in VRAM.
pub fn fused_metal_ple_scale_rmsnorm(
    x: &Tensor,
    weight: &Tensor,
    scale: f64,
    eps: f64,
    d: usize,
) -> Result<Tensor> {
    let fn_name = "fused_metal_ple_scale_rmsnorm";
    let mdev = ensure_metal_bf16(x, fn_name, "x")?;
    if d == 0 {
        candle_core::bail!("{fn_name}: d must be > 0");
    }
    let dims = x.dims();
    if dims[dims.len() - 1] != d {
        candle_core::bail!("{fn_name}: x last dim {:?} does not match d={d}", dims);
    }

    let weight = prepare_metal_weight(weight, d, x.device(), fn_name, "weight")?;
    let x = if !x.is_contiguous() {
        x.contiguous()?
    } else {
        x.clone()
    };

    let orig_shape = x.shape();
    let num_rows = x.elem_count() / d;
    if num_rows == 0 {
        return Tensor::zeros(orig_shape.clone(), DType::BF16, x.device());
    }
    let out = Tensor::zeros(orig_shape.clone(), DType::BF16, x.device())?;

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "ple_scale_rms_norm_bf16")?;

    let scale_f32 = scale as f32;
    let eps_f32 = eps as f32;
    let d_i32 = d as i32;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let x_buf = match &*x_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let w_buf = match &*w_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = DType::BF16.size_in_bytes();
        let x_offset = x_layout.start_offset() * elem_bytes;
        let w_offset = w_layout.start_offset() * elem_bytes;
        let out_offset = out_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(x_buf), x_offset);
        enc_ref.set_input_buffer(1, Some(w_buf), w_offset);
        enc_ref.set_output_buffer(2, Some(out_buf), out_offset);
        enc_ref.set_bytes(3, &scale_f32);
        enc_ref.set_bytes(4, &eps_f32);
        enc_ref.set_bytes(5, &d_i32);

        let grid_dims = MTLSize {
            width: num_rows,
            height: 1,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Metal Mean Pooling (FP32 accumulation) + L2 Normalization (+ optional MRL truncation)
/// in two tiny dispatches sharing one `[B, out_dim]` FP32 scratch buffer instead of `[B, S, H]` FP32.
pub fn fused_metal_pool_and_l2norm(
    token_embeddings: &Tensor,
    attention_mask: Option<&Tensor>,
    truncate_dim: Option<usize>,
) -> Result<Tensor> {
    let fn_name = "fused_metal_pool_and_l2norm";
    let mdev = ensure_metal_bf16(token_embeddings, fn_name, "token_embeddings")?;
    let (batch, seq_len, hidden_dim) = token_embeddings.dims3()?;
    if hidden_dim == 0 {
        candle_core::bail!("{fn_name}: hidden_dim must be > 0");
    }
    if let Some(dim) = truncate_dim
        && (dim == 0 || dim > hidden_dim)
    {
        candle_core::bail!(
            "Invalid Matryoshka truncate_dim {}: must be in 1..={}",
            dim,
            hidden_dim
        );
    }
    let out_dim = truncate_dim.unwrap_or(hidden_dim);
    if batch == 0 {
        return Tensor::zeros((0, out_dim), DType::BF16, token_embeddings.device());
    }
    if seq_len == 0 {
        candle_core::bail!("{fn_name}: seq_len must be > 0");
    }

    let mask_holder: Option<Tensor> = match attention_mask {
        Some(m) => {
            if !m.device().same_device(token_embeddings.device()) {
                candle_core::bail!("{fn_name}: attention_mask must be on the same Metal device");
            }
            if m.dims() != [batch, seq_len] {
                candle_core::bail!(
                    "{fn_name}: attention_mask shape {:?} does not match [{batch}, {seq_len}]",
                    m.dims()
                );
            }
            let m_f32 = if m.dtype() != DType::F32 {
                m.to_dtype(DType::F32)?
            } else {
                m.clone()
            };
            Some(if !m_f32.is_contiguous() {
                m_f32.contiguous()?
            } else {
                m_f32
            })
        }
        None => None,
    };
    let mask_ref = mask_holder.as_ref();

    let x = if !token_embeddings.is_contiguous() {
        token_embeddings.contiguous()?
    } else {
        token_embeddings.clone()
    };

    let pooled_f32 = Tensor::zeros((batch, out_dim), DType::F32, x.device())?;
    let out = Tensor::zeros((batch, out_dim), DType::BF16, x.device())?;

    let pool_pipeline = get_or_compile_pipeline(mdev.metal_device(), "pool_mean_bf16_to_f32")?;
    let norm_pipeline = get_or_compile_pipeline(mdev.metal_device(), "l2norm_f32_to_bf16")?;

    let s_i32 = seq_len as i32;
    let h_i32 = hidden_dim as i32;
    let out_dim_i32 = out_dim as i32;
    let has_mask: i32 = if mask_ref.is_some() { 1 } else { 0 };

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (p_storage, p_layout) = pooled_f32.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let x_buf = match &*x_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let p_buf = match &*p_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let x_offset = x_layout.start_offset() * DType::BF16.size_in_bytes();
        let p_offset = p_layout.start_offset() * DType::F32.size_in_bytes();
        let out_offset = out_layout.start_offset() * DType::BF16.size_in_bytes();

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();

        // Dispatch 1: BF16 -> FP32 mean pool
        enc_ref.set_compute_pipeline_state(&pool_pipeline);
        enc_ref.set_input_buffer(0, Some(x_buf), x_offset);
        if let Some(m) = mask_ref {
            let (m_storage, m_layout) = m.storage_and_layout();
            let m_buf = match &*m_storage {
                candle_core::Storage::Metal(mb) => mb.buffer(),
                _ => unreachable!(),
            };
            let m_offset = m_layout.start_offset() * DType::F32.size_in_bytes();
            enc_ref.set_input_buffer(1, Some(m_buf), m_offset);
        } else {
            enc_ref.set_input_buffer(1, Some(x_buf), x_offset);
        }
        enc_ref.set_output_buffer(2, Some(p_buf), p_offset);
        enc_ref.set_bytes(3, &s_i32);
        enc_ref.set_bytes(4, &h_i32);
        enc_ref.set_bytes(5, &out_dim_i32);
        enc_ref.set_bytes(6, &has_mask);

        let pool_grid = MTLSize {
            width: out_dim.div_ceil(16),
            height: batch,
            depth: 1,
        };
        let pool_group = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(pool_grid, pool_group);

        // Dispatch 2: FP32 L2-normalize -> BF16
        enc_ref.set_compute_pipeline_state(&norm_pipeline);
        enc_ref.set_input_buffer(0, Some(p_buf), p_offset);
        enc_ref.set_output_buffer(1, Some(out_buf), out_offset);
        enc_ref.set_bytes(2, &out_dim_i32);

        let norm_grid = MTLSize {
            width: batch,
            height: 1,
            depth: 1,
        };
        let norm_group = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(norm_grid, norm_group);
    }

    Ok(out)
}

/// Fused Metal GeGLU evaluating exact PyTorch tanh GELU approximation in a single streaming pass.
pub fn fused_metal_geglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    fused_metal_geglu_impl(gate, up, "geglu_bf16")
}

/// Fused Metal PLE GeGLU matching BF16 `gelu_tanh(gate) * per_layer_input` bit-for-bit in a single pass.
pub fn fused_metal_ple_geglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    fused_metal_geglu_impl(gate, up, "ple_geglu_bf16")
}

fn fused_metal_geglu_impl(gate: &Tensor, up: &Tensor, kernel_name: &'static str) -> Result<Tensor> {
    let mdev = ensure_metal_bf16(gate, kernel_name, "gate")?;
    ensure_metal_bf16(up, kernel_name, "up")?;
    if !up.device().same_device(gate.device()) {
        candle_core::bail!("{kernel_name}: up must be on the same Metal device");
    }
    if gate.shape() != up.shape() {
        candle_core::bail!(
            "{kernel_name}: shape mismatch gate={:?} vs up={:?}",
            gate.dims(),
            up.dims()
        );
    }
    let dims = gate.dims();
    if dims[dims.len() - 1] == 0 {
        candle_core::bail!("{kernel_name}: last dimension must be > 0");
    }

    let gate = if !gate.is_contiguous() {
        gate.contiguous()?
    } else {
        gate.clone()
    };
    let up = if !up.is_contiguous() {
        up.contiguous()?
    } else {
        up.clone()
    };

    let total_elements = gate.elem_count();
    if total_elements == 0 {
        return Tensor::zeros(gate.shape().clone(), DType::BF16, gate.device());
    }
    let out = Tensor::zeros(gate.shape().clone(), DType::BF16, gate.device())?;
    let pipeline = get_or_compile_pipeline(mdev.metal_device(), kernel_name)?;
    let total_i32 = total_elements as i32;

    {
        let (gate_storage, gate_layout) = gate.storage_and_layout();
        let (up_storage, up_layout) = up.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let gate_buf = match &*gate_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let up_buf = match &*up_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = gate.dtype().size_in_bytes();
        let gate_offset = gate_layout.start_offset() * elem_bytes;
        let up_offset = up_layout.start_offset() * elem_bytes;
        let out_offset = out_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(gate_buf), gate_offset);
        enc_ref.set_input_buffer(1, Some(up_buf), up_offset);
        enc_ref.set_output_buffer(2, Some(out_buf), out_offset);
        enc_ref.set_bytes(3, &total_i32);

        let thread_group_size = 256;
        let num_groups = total_elements.div_ceil(thread_group_size);
        let grid_dims = MTLSize {
            width: num_groups,
            height: 1,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: thread_group_size,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Metal GeGLU reading directly from a concatenated `[..., 2 * D]` `gate_up` projection tensor.
pub fn fused_metal_geglu_gate_up(gate_up: &Tensor) -> Result<Tensor> {
    let fn_name = "fused_metal_geglu_gate_up";
    let mdev = ensure_metal_bf16(gate_up, fn_name, "gate_up")?;

    let dims = gate_up.dims();
    let last_dim = dims[dims.len() - 1];
    if last_dim == 0 || !last_dim.is_multiple_of(2) {
        candle_core::bail!(
            "{fn_name} expects positive last dim divisible by 2, got {:?}",
            dims
        );
    }

    let gate_up = if !gate_up.is_contiguous() {
        gate_up.contiguous()?
    } else {
        gate_up.clone()
    };
    let d = last_dim / 2;
    let mut out_dims = dims.to_vec();
    *out_dims.last_mut().unwrap() = d;

    let total_elements = gate_up.elem_count() / 2;
    if total_elements == 0 {
        return Tensor::zeros(out_dims, DType::BF16, gate_up.device());
    }
    let out = Tensor::zeros(out_dims, DType::BF16, gate_up.device())?;
    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "geglu_gate_up_bf16")?;
    let d_i32 = d as i32;

    {
        let (gu_storage, gu_layout) = gate_up.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let gu_buf = match &*gu_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = DType::BF16.size_in_bytes();
        let gu_offset = gu_layout.start_offset() * elem_bytes;
        let out_offset = out_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(gu_buf), gu_offset);
        enc_ref.set_output_buffer(1, Some(out_buf), out_offset);
        enc_ref.set_bytes(2, &d_i32);

        let thread_group_size = 256;
        let num_rows = total_elements / d;
        let num_groups_x = d.div_ceil(thread_group_size);
        let grid_dims = MTLSize {
            width: num_groups_x,
            height: num_rows,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: thread_group_size,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Single-Dispatch Metal QKV RMSNorm + RoPE + `[B, S, H, D] -> [B, H, S, D]` Transpose.
#[allow(clippy::too_many_arguments)]
pub fn fused_metal_qkv_rmsnorm_rope_transpose(
    qkv: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: Option<&Tensor>,
    cos: &Tensor,
    sin: &Tensor,
    batch: usize,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_eps: f64,
    k_eps: f64,
    v_eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let fn_name = "fused_metal_qkv_rmsnorm_rope_transpose";
    let mdev = ensure_metal_bf16(qkv, fn_name, "qkv")?;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        candle_core::bail!("{fn_name}: head_dim must be a positive even integer, got {head_dim}");
    }
    if num_q_heads == 0 || num_kv_heads == 0 {
        candle_core::bail!("{fn_name}: num_q_heads and num_kv_heads must be > 0");
    }
    let total_heads = num_q_heads
        .checked_add(2 * num_kv_heads)
        .ok_or_else(|| candle_core::Error::Msg(format!("{fn_name}: head count overflow")))?;
    let expected_elems = checked_4d_elems(batch, seq_len, total_heads, head_dim, fn_name)?;
    if qkv.elem_count() != expected_elems {
        candle_core::bail!(
            "{fn_name}: qkv elem_count {} does not match batch*seq*heads*head_dim = {}",
            qkv.elem_count(),
            expected_elems
        );
    }

    let q_weight = prepare_metal_weight(q_weight, head_dim, qkv.device(), fn_name, "q_weight")?;
    let k_weight = prepare_metal_weight(k_weight, head_dim, qkv.device(), fn_name, "k_weight")?;
    let v_weight_holder: Option<Tensor> = match v_weight {
        Some(w) => Some(prepare_metal_weight(
            w,
            head_dim,
            qkv.device(),
            fn_name,
            "v_weight",
        )?),
        None => None,
    };
    let v_weight_ref = v_weight_holder.as_ref();
    let cos = prepare_metal_rope_table(cos, seq_len, head_dim, qkv.device(), fn_name, "cos")?;
    let sin = prepare_metal_rope_table(sin, seq_len, head_dim, qkv.device(), fn_name, "sin")?;

    let qkv = if !qkv.is_contiguous() {
        qkv.contiguous()?
    } else {
        qkv.clone()
    };

    if batch == 0 || seq_len == 0 {
        return Ok((
            Tensor::zeros(
                (batch, num_q_heads, seq_len, head_dim),
                DType::BF16,
                qkv.device(),
            )?,
            Tensor::zeros(
                (batch, num_kv_heads, seq_len, head_dim),
                DType::BF16,
                qkv.device(),
            )?,
            Tensor::zeros(
                (batch, num_kv_heads, seq_len, head_dim),
                DType::BF16,
                qkv.device(),
            )?,
        ));
    }
    let q_out = Tensor::zeros(
        (batch, num_q_heads, seq_len, head_dim),
        DType::BF16,
        qkv.device(),
    )?;
    let k_out = Tensor::zeros(
        (batch, num_kv_heads, seq_len, head_dim),
        DType::BF16,
        qkv.device(),
    )?;
    let v_out = Tensor::zeros(
        (batch, num_kv_heads, seq_len, head_dim),
        DType::BF16,
        qkv.device(),
    )?;

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "qkv_rmsnorm_rope_transpose_bf16")?;

    let q_eps_f32 = q_eps as f32;
    let k_eps_f32 = k_eps as f32;
    let v_eps_f32 = v_eps as f32;
    let s_i32 = seq_len as i32;
    let h_q_i32 = num_q_heads as i32;
    let h_kv_i32 = num_kv_heads as i32;
    let d_i32 = head_dim as i32;
    let has_v_weight: i32 = if v_weight_ref.is_some() { 1 } else { 0 };

    {
        let (qkv_storage, qkv_layout) = qkv.storage_and_layout();
        let (qw_storage, qw_layout) = q_weight.storage_and_layout();
        let (kw_storage, kw_layout) = k_weight.storage_and_layout();
        let (cos_storage, cos_layout) = cos.storage_and_layout();
        let (sin_storage, sin_layout) = sin.storage_and_layout();
        let (qo_storage, qo_layout) = q_out.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();
        let (vo_storage, vo_layout) = v_out.storage_and_layout();

        let qkv_buf = match &*qkv_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let qw_buf = match &*qw_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let kw_buf = match &*kw_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let cos_buf = match &*cos_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let sin_buf = match &*sin_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let qo_buf = match &*qo_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let ko_buf = match &*ko_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let vo_buf = match &*vo_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = DType::BF16.size_in_bytes();
        let qkv_offset = qkv_layout.start_offset() * elem_bytes;
        let qw_offset = qw_layout.start_offset() * elem_bytes;
        let kw_offset = kw_layout.start_offset() * elem_bytes;
        let cos_offset = cos_layout.start_offset() * elem_bytes;
        let sin_offset = sin_layout.start_offset() * elem_bytes;
        let qo_offset = qo_layout.start_offset() * elem_bytes;
        let ko_offset = ko_layout.start_offset() * elem_bytes;
        let vo_offset = vo_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(qkv_buf), qkv_offset);
        enc_ref.set_input_buffer(1, Some(qw_buf), qw_offset);
        enc_ref.set_input_buffer(2, Some(kw_buf), kw_offset);
        if let Some(vw) = v_weight_ref {
            let (vw_storage, vw_layout) = vw.storage_and_layout();
            let vw_buf = match &*vw_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let vw_offset = vw_layout.start_offset() * elem_bytes;
            enc_ref.set_input_buffer(3, Some(vw_buf), vw_offset);
        } else {
            enc_ref.set_input_buffer(3, Some(qw_buf), qw_offset);
        }
        enc_ref.set_input_buffer(4, Some(cos_buf), cos_offset);
        enc_ref.set_input_buffer(5, Some(sin_buf), sin_offset);
        enc_ref.set_output_buffer(6, Some(qo_buf), qo_offset);
        enc_ref.set_output_buffer(7, Some(ko_buf), ko_offset);
        enc_ref.set_output_buffer(8, Some(vo_buf), vo_offset);
        enc_ref.set_bytes(9, &q_eps_f32);
        enc_ref.set_bytes(10, &k_eps_f32);
        enc_ref.set_bytes(11, &v_eps_f32);
        enc_ref.set_bytes(12, &s_i32);
        enc_ref.set_bytes(13, &h_q_i32);
        enc_ref.set_bytes(14, &h_kv_i32);
        enc_ref.set_bytes(15, &d_i32);
        enc_ref.set_bytes(16, &has_v_weight);

        let grid_dims = MTLSize {
            width: num_q_heads + 2 * num_kv_heads,
            height: seq_len,
            depth: batch,
        };
        let group_dims = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok((q_out, k_out, v_out))
}

/// Fused Single-Dispatch Metal Vision QKV RMSNorm + 2D Axial RoPE + `[B, S, H, D] -> [B, H, S, D]` Transpose.
#[allow(clippy::too_many_arguments)]
pub fn fused_metal_vision_qkv_rmsnorm_rope2d_transpose(
    qkv: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: Option<&Tensor>,
    cos: &Tensor,
    sin: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let fn_name = "fused_metal_vision_qkv_rmsnorm_rope2d_transpose";
    let mdev = ensure_metal_bf16(qkv, fn_name, "qkv")?;
    if head_dim == 0 || !head_dim.is_multiple_of(4) {
        candle_core::bail!(
            "{fn_name}: head_dim must be a positive multiple of 4 for 2D Vision RoPE, got {head_dim}"
        );
    }
    if num_heads == 0 {
        candle_core::bail!("{fn_name}: num_heads must be > 0");
    }
    let total_heads = 3usize
        .checked_mul(num_heads)
        .ok_or_else(|| candle_core::Error::Msg(format!("{fn_name}: head count overflow")))?;
    let expected_elems = checked_4d_elems(batch, seq_len, total_heads, head_dim, fn_name)?;
    if qkv.elem_count() != expected_elems {
        candle_core::bail!(
            "{fn_name}: qkv elem_count {} does not match batch*seq*3*heads*head_dim = {}",
            qkv.elem_count(),
            expected_elems
        );
    }

    let q_weight = prepare_metal_weight(q_weight, head_dim, qkv.device(), fn_name, "q_weight")?;
    let k_weight = prepare_metal_weight(k_weight, head_dim, qkv.device(), fn_name, "k_weight")?;
    let v_weight_holder: Option<Tensor> = match v_weight {
        Some(w) => Some(prepare_metal_weight(
            w,
            head_dim,
            qkv.device(),
            fn_name,
            "v_weight",
        )?),
        None => None,
    };
    let v_weight_ref = v_weight_holder.as_ref();
    let cos = prepare_metal_rope_table(cos, seq_len, head_dim, qkv.device(), fn_name, "cos")?;
    let sin = prepare_metal_rope_table(sin, seq_len, head_dim, qkv.device(), fn_name, "sin")?;

    let qkv = if !qkv.is_contiguous() {
        qkv.contiguous()?
    } else {
        qkv.clone()
    };

    if batch == 0 || seq_len == 0 {
        return Ok((
            Tensor::zeros(
                (batch, num_heads, seq_len, head_dim),
                DType::BF16,
                qkv.device(),
            )?,
            Tensor::zeros(
                (batch, num_heads, seq_len, head_dim),
                DType::BF16,
                qkv.device(),
            )?,
            Tensor::zeros(
                (batch, num_heads, seq_len, head_dim),
                DType::BF16,
                qkv.device(),
            )?,
        ));
    }
    let q_out = Tensor::zeros(
        (batch, num_heads, seq_len, head_dim),
        DType::BF16,
        qkv.device(),
    )?;
    let k_out = Tensor::zeros(
        (batch, num_heads, seq_len, head_dim),
        DType::BF16,
        qkv.device(),
    )?;
    let v_out = Tensor::zeros(
        (batch, num_heads, seq_len, head_dim),
        DType::BF16,
        qkv.device(),
    )?;

    let pipeline = get_or_compile_pipeline(
        mdev.metal_device(),
        "vision_qkv_rmsnorm_rope2d_transpose_bf16",
    )?;

    let eps_f32 = eps as f32;
    let s_i32 = seq_len as i32;
    let h_i32 = num_heads as i32;
    let d_i32 = head_dim as i32;
    let has_v_weight: i32 = if v_weight_ref.is_some() { 1 } else { 0 };

    {
        let (qkv_storage, qkv_layout) = qkv.storage_and_layout();
        let (qw_storage, qw_layout) = q_weight.storage_and_layout();
        let (kw_storage, kw_layout) = k_weight.storage_and_layout();
        let (cos_storage, cos_layout) = cos.storage_and_layout();
        let (sin_storage, sin_layout) = sin.storage_and_layout();
        let (qo_storage, qo_layout) = q_out.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();
        let (vo_storage, vo_layout) = v_out.storage_and_layout();

        let qkv_buf = match &*qkv_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let qw_buf = match &*qw_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let kw_buf = match &*kw_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let cos_buf = match &*cos_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let sin_buf = match &*sin_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let qo_buf = match &*qo_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let ko_buf = match &*ko_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let vo_buf = match &*vo_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = DType::BF16.size_in_bytes();
        let qkv_offset = qkv_layout.start_offset() * elem_bytes;
        let qw_offset = qw_layout.start_offset() * elem_bytes;
        let kw_offset = kw_layout.start_offset() * elem_bytes;
        let cos_offset = cos_layout.start_offset() * elem_bytes;
        let sin_offset = sin_layout.start_offset() * elem_bytes;
        let qo_offset = qo_layout.start_offset() * elem_bytes;
        let ko_offset = ko_layout.start_offset() * elem_bytes;
        let vo_offset = vo_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(qkv_buf), qkv_offset);
        enc_ref.set_input_buffer(1, Some(qw_buf), qw_offset);
        enc_ref.set_input_buffer(2, Some(kw_buf), kw_offset);
        if let Some(vw) = v_weight_ref {
            let (vw_storage, vw_layout) = vw.storage_and_layout();
            let vw_buf = match &*vw_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let vw_offset = vw_layout.start_offset() * elem_bytes;
            enc_ref.set_input_buffer(3, Some(vw_buf), vw_offset);
        } else {
            enc_ref.set_input_buffer(3, Some(qw_buf), qw_offset);
        }
        enc_ref.set_input_buffer(4, Some(cos_buf), cos_offset);
        enc_ref.set_input_buffer(5, Some(sin_buf), sin_offset);
        enc_ref.set_output_buffer(6, Some(qo_buf), qo_offset);
        enc_ref.set_output_buffer(7, Some(ko_buf), ko_offset);
        enc_ref.set_output_buffer(8, Some(vo_buf), vo_offset);
        enc_ref.set_bytes(9, &eps_f32);
        enc_ref.set_bytes(10, &s_i32);
        enc_ref.set_bytes(11, &h_i32);
        enc_ref.set_bytes(12, &d_i32);
        enc_ref.set_bytes(13, &has_v_weight);

        let grid_dims = MTLSize {
            width: 3 * num_heads,
            height: seq_len,
            depth: batch,
        };
        let group_dims = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok((q_out, k_out, v_out))
}

/// Fused Metal Audio GLU + 5-Tap Causal Depthwise Conv1D in a single dispatch.
pub fn fused_metal_audio_glu_causal_dwconv5(pw1: &Tensor, dw_weight: &Tensor) -> Result<Tensor> {
    let fn_name = "fused_metal_audio_glu_causal_dwconv5";
    let mdev = ensure_metal_bf16(pw1, fn_name, "pw1")?;
    if !dw_weight.device().same_device(pw1.device()) {
        candle_core::bail!("{fn_name}: dw_weight must be on the same Metal device");
    }

    let (batch, seq_len, two_c) = pw1.dims3()?;
    if two_c == 0 || !two_c.is_multiple_of(2) {
        candle_core::bail!("{fn_name}: pw1 last dim must be positive and even, got {two_c}");
    }
    let c = two_c / 2;
    if dw_weight.elem_count() != c * 5 {
        candle_core::bail!(
            "{fn_name}: dw_weight elem_count {} does not match c*5 = {}",
            dw_weight.elem_count(),
            c * 5
        );
    }

    let pw1 = if !pw1.is_contiguous() {
        pw1.contiguous()?
    } else {
        pw1.clone()
    };
    let dw_weight = if !dw_weight.is_contiguous() {
        dw_weight.contiguous()?
    } else {
        dw_weight.clone()
    };
    let dw_weight = if dw_weight.dtype() != DType::BF16 {
        dw_weight.to_dtype(DType::BF16)?
    } else {
        dw_weight
    };

    let total_elements = batch * seq_len * c;
    if total_elements == 0 {
        return Tensor::zeros((batch, seq_len, c), DType::BF16, pw1.device());
    }
    let out = Tensor::zeros((batch, seq_len, c), DType::BF16, pw1.device())?;

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "audio_glu_causal_dwconv5_bf16")?;

    let b_i32 = batch as i32;
    let s_i32 = seq_len as i32;
    let c_i32 = c as i32;

    {
        let (pw1_storage, pw1_layout) = pw1.storage_and_layout();
        let (dw_storage, dw_layout) = dw_weight.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let pw1_buf = match &*pw1_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let dw_buf = match &*dw_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = DType::BF16.size_in_bytes();
        let pw1_offset = pw1_layout.start_offset() * elem_bytes;
        let dw_offset = dw_layout.start_offset() * elem_bytes;
        let out_offset = out_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(pw1_buf), pw1_offset);
        enc_ref.set_input_buffer(1, Some(dw_buf), dw_offset);
        enc_ref.set_output_buffer(2, Some(out_buf), out_offset);
        enc_ref.set_bytes(3, &b_i32);
        enc_ref.set_bytes(4, &s_i32);
        enc_ref.set_bytes(5, &c_i32);

        let thread_group_size = 256;
        let num_groups = total_elements.div_ceil(thread_group_size);
        let grid_dims = MTLSize {
            width: num_groups,
            height: 1,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: thread_group_size,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Native Metal `Q8_0` Quantized MatMul/MatVec operating directly on `BF16` or `F32` activations without `F32` cast copies.
pub fn fused_metal_q8_0_matmul(
    weight_q8_bytes: &Tensor,
    x: &Tensor,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor> {
    let fn_name = "fused_metal_q8_0_matmul";
    let mdev = metal_device_of(x, fn_name, "x")?;
    if !weight_q8_bytes.device().same_device(x.device()) {
        candle_core::bail!("{fn_name}: weight_q8_bytes must be on the same Metal device");
    }
    if weight_q8_bytes.dtype() != DType::U8 {
        candle_core::bail!(
            "{fn_name}: weight_q8_bytes must have dtype U8, got {:?}",
            weight_q8_bytes.dtype()
        );
    }
    if in_dim == 0 || !in_dim.is_multiple_of(32) {
        candle_core::bail!("{fn_name}: in_dim must be a positive multiple of 32, got {in_dim}");
    }
    if out_dim == 0 {
        candle_core::bail!("{fn_name}: out_dim must be > 0");
    }

    let x_dims = x.dims();
    if x_dims.is_empty() || x_dims[x_dims.len() - 1] != in_dim {
        candle_core::bail!(
            "{fn_name} shape mismatch: x={:?}, expected in_dim={}",
            x_dims,
            in_dim
        );
    }
    let expected_q8_bytes = out_dim
        .checked_mul((in_dim / 32) * 34)
        .ok_or_else(|| candle_core::Error::Msg(format!("{fn_name}: Q8_0 byte size overflow")))?;
    if weight_q8_bytes.elem_count() != expected_q8_bytes {
        candle_core::bail!(
            "{fn_name}: weight_q8_bytes elem_count {} does not match expected {}",
            weight_q8_bytes.elem_count(),
            expected_q8_bytes
        );
    }

    let dtype = x.dtype();
    if !matches!(dtype, DType::BF16 | DType::F32 | DType::F16) {
        candle_core::bail!(
            "{fn_name}: unsupported activation dtype {:?} (expected BF16, F32, or F16)",
            dtype
        );
    }

    let m = x.elem_count() / in_dim;
    let mut out_dims = x_dims.to_vec();
    *out_dims.last_mut().unwrap() = out_dim;
    if m == 0 {
        return Tensor::zeros(out_dims, dtype, x.device());
    }

    let x = if !x.is_contiguous() {
        x.contiguous()?
    } else {
        x.clone()
    };
    let weight_q8_bytes = if !weight_q8_bytes.is_contiguous() {
        weight_q8_bytes.contiguous()?
    } else {
        weight_q8_bytes.clone()
    };

    let is_mv = m <= 8;
    let use_m5 = !is_mv
        && dtype == DType::BF16
        && MetalCapabilities::detect(mdev.metal_device()).use_m5_tensor_ops();

    let pipeline = if use_m5 {
        get_or_compile_m5_pipeline(mdev.metal_device(), "q8_0_mm_m5_bf16")?
    } else {
        let kernel_name = match (dtype, is_mv) {
            (DType::BF16, true) => "q8_0_mv_bf16",
            (DType::BF16, false) => "q8_0_mm_bf16",
            (DType::F32, true) => "q8_0_mv_f32",
            (DType::F32, false) => "q8_0_mm_f32",
            _ => {
                let x_f32 = x.to_dtype(DType::F32)?;
                let out_f32 = fused_metal_q8_0_matmul(&weight_q8_bytes, &x_f32, out_dim, in_dim)?;
                return out_f32.to_dtype(dtype);
            }
        };
        get_or_compile_pipeline(mdev.metal_device(), kernel_name)?
    };
    let out = Tensor::zeros(out_dims, dtype, x.device())?;

    let ne00 = in_dim as i32;
    let ne0 = out_dim as i32;
    let ne1 = m as i32;

    {
        let (w_storage, w_layout) = weight_q8_bytes.storage_and_layout();
        let (x_storage, x_layout) = x.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let w_buf = match &*w_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let x_buf = match &*x_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let w_offset = w_layout.start_offset() * weight_q8_bytes.dtype().size_in_bytes();
        let x_offset = x_layout.start_offset() * dtype.size_in_bytes();
        let out_offset = out_layout.start_offset() * dtype.size_in_bytes();

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(w_buf), w_offset);
        enc_ref.set_input_buffer(1, Some(x_buf), x_offset);
        enc_ref.set_output_buffer(2, Some(out_buf), out_offset);
        enc_ref.set_bytes(3, &ne00);
        enc_ref.set_bytes(4, &ne0);
        enc_ref.set_bytes(5, &ne1);

        if is_mv {
            let grid_dims = MTLSize {
                width: out_dim.div_ceil(8),
                height: m,
                depth: 1,
            };
            let group_dims = MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            };
            enc_ref.dispatch_thread_groups(grid_dims, group_dims);
        } else {
            let grid_dims = MTLSize {
                width: m.div_ceil(32),
                height: out_dim.div_ceil(64),
                depth: 1,
            };
            let group_dims = MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            };
            enc_ref.dispatch_thread_groups(grid_dims, group_dims);
        }
    }

    Ok(out)
}

/// M5 Neural Accelerator (`mpp::tensor_ops::matmul2d`) Dense BF16 MatMul:
/// Computes `x @ weight^T` for `weight` `[N, K]` (supporting zero-copy `narrow` views with `stride[1] == 1`)
/// and arbitrary ND `x` `[..., K] -> [..., N]`.
pub fn fused_metal_bf16_matmul(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    let fn_name = "fused_metal_bf16_matmul";
    let mdev = ensure_metal_bf16(x, fn_name, "x")?;
    if !weight.device().same_device(x.device()) {
        candle_core::bail!("{fn_name}: weight must be on the same Metal device");
    }
    if weight.dtype() != DType::BF16 {
        candle_core::bail!(
            "{fn_name}: weight must have dtype BF16, got {:?}",
            weight.dtype()
        );
    }

    let (n, k) = weight.dims2()?;
    if n == 0 || k == 0 || !k.is_multiple_of(32) {
        candle_core::bail!(
            "{fn_name}: weight shape ({n}, {k}) must have non-zero dims and K % 32 == 0"
        );
    }
    let x_dims = x.dims();
    if x_dims.is_empty() || x_dims[x_dims.len() - 1] != k {
        candle_core::bail!(
            "{fn_name}: shape mismatch x={:?} vs weight=({n}, {k})",
            x_dims
        );
    }

    let m = x.elem_count() / k;
    let mut out_dims = x_dims.to_vec();
    *out_dims.last_mut().unwrap() = n;
    if m == 0 {
        return Tensor::zeros(out_dims, DType::BF16, x.device());
    }

    let x = if !x.is_contiguous() {
        x.contiguous()?
    } else {
        x.clone()
    };
    let weight = if weight.layout().stride()[1] == 1 {
        weight.clone()
    } else {
        weight.contiguous()?
    };
    let lda = weight.layout().stride()[0] as i32;

    let pipeline = get_or_compile_m5_pipeline(mdev.metal_device(), "bf16_mm_m5")?;
    let out = Tensor::zeros(out_dims, DType::BF16, x.device())?;

    let k_i32 = k as i32;
    let n_i32 = n as i32;
    let m_i32 = m as i32;

    {
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (x_storage, x_layout) = x.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let w_buf = match &*w_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let x_buf = match &*x_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = DType::BF16.size_in_bytes();
        let w_offset = w_layout.start_offset() * elem_bytes;
        let x_offset = x_layout.start_offset() * elem_bytes;
        let out_offset = out_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(w_buf), w_offset);
        enc_ref.set_input_buffer(1, Some(x_buf), x_offset);
        enc_ref.set_output_buffer(2, Some(out_buf), out_offset);
        enc_ref.set_bytes(3, &k_i32);
        enc_ref.set_bytes(4, &n_i32);
        enc_ref.set_bytes(5, &m_i32);
        enc_ref.set_bytes(6, &lda);

        let grid_dims = MTLSize {
            width: m.div_ceil(64),
            height: n.div_ceil(64),
            depth: 1,
        };
        let group_dims = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Direct Metal `Q8_0 -> BF16` dequantization kernel (avoids intermediate `F32` allocation at load time).
pub fn dequantize_q8_0_to_bf16(
    weight_q8_bytes: &Tensor,
    rows: usize,
    cols: usize,
) -> Result<Tensor> {
    let fn_name = "dequantize_q8_0_to_bf16";
    let mdev = metal_device_of(weight_q8_bytes, fn_name, "weight_q8_bytes")?;
    if weight_q8_bytes.dtype() != DType::U8 {
        candle_core::bail!(
            "{fn_name}: weight_q8_bytes must have dtype U8, got {:?}",
            weight_q8_bytes.dtype()
        );
    }
    if cols == 0 || !cols.is_multiple_of(32) {
        candle_core::bail!("{fn_name}: cols must be a positive multiple of 32, got {cols}");
    }
    if rows == 0 {
        return Tensor::zeros((0, cols), DType::BF16, weight_q8_bytes.device());
    }
    let blocks_per_row = cols / 32;
    let expected_bytes = rows
        .checked_mul(blocks_per_row * 34)
        .ok_or_else(|| candle_core::Error::Msg(format!("{fn_name}: Q8_0 byte size overflow")))?;
    if weight_q8_bytes.elem_count() != expected_bytes {
        candle_core::bail!(
            "{fn_name}: elem_count {} does not match expected {}",
            weight_q8_bytes.elem_count(),
            expected_bytes
        );
    }

    let weight_q8_bytes = if !weight_q8_bytes.is_contiguous() {
        weight_q8_bytes.contiguous()?
    } else {
        weight_q8_bytes.clone()
    };

    let out = Tensor::zeros((rows, cols), DType::BF16, weight_q8_bytes.device())?;
    let total_blocks = rows * blocks_per_row;
    let total_blocks_u32 = total_blocks as u32;
    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "q8_0_dequant_bf16")?;

    {
        let (w_storage, w_layout) = weight_q8_bytes.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let w_buf = match &*w_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let w_offset = w_layout.start_offset() * DType::U8.size_in_bytes();
        let out_offset = out_layout.start_offset() * DType::BF16.size_in_bytes();

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(w_buf), w_offset);
        enc_ref.set_output_buffer(1, Some(out_buf), out_offset);
        enc_ref.set_bytes(2, &total_blocks_u32);

        let total_threads = total_blocks * 8;
        let group_size = 256;
        let grid_dims = MTLSize {
            width: total_threads.div_ceil(group_size),
            height: 1,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: group_size,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Metal Vision 2D Spatial Average Pooling (`[B, P, H] -> [B, out_h * out_w, H]` in BF16).
pub fn fused_metal_vision_spatial_pool(
    x: &Tensor,
    patch_width: usize,
    patch_height: usize,
    kernel_size: usize,
    scale: f64,
) -> Result<Tensor> {
    let fn_name = "fused_metal_vision_spatial_pool";
    let mdev = ensure_metal_bf16(x, fn_name, "x")?;
    let (batch, num_patches, hidden_size) = x.dims3()?;
    if kernel_size == 0
        || !patch_width.is_multiple_of(kernel_size)
        || !patch_height.is_multiple_of(kernel_size)
    {
        candle_core::bail!(
            "{fn_name}: patch grid {patch_width}x{patch_height} must be divisible by kernel_size={kernel_size}"
        );
    }
    if patch_width * patch_height != num_patches {
        candle_core::bail!("{fn_name}: num_patches {num_patches} != {patch_width}*{patch_height}");
    }
    let out_w = patch_width / kernel_size;
    let out_h = patch_height / kernel_size;
    let out_tokens = out_w * out_h;
    if batch == 0 || out_tokens == 0 || hidden_size == 0 {
        return Tensor::zeros((batch, out_tokens, hidden_size), DType::BF16, x.device());
    }

    let x = if !x.is_contiguous() {
        x.contiguous()?
    } else {
        x.clone()
    };
    let out = Tensor::zeros((batch, out_tokens, hidden_size), DType::BF16, x.device())?;
    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "vision_spatial_pool_3x3_bf16")?;

    let pw_i32 = patch_width as i32;
    let ph_i32 = patch_height as i32;
    let k_i32 = kernel_size as i32;
    let h_i32 = hidden_size as i32;
    let scale_f32 = scale as f32;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();
        let x_buf = match &*x_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let elem_bytes = DType::BF16.size_in_bytes();
        let x_offset = x_layout.start_offset() * elem_bytes;
        let out_offset = out_layout.start_offset() * elem_bytes;

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(x_buf), x_offset);
        enc_ref.set_output_buffer(1, Some(out_buf), out_offset);
        enc_ref.set_bytes(2, &pw_i32);
        enc_ref.set_bytes(3, &ph_i32);
        enc_ref.set_bytes(4, &k_i32);
        enc_ref.set_bytes(5, &h_i32);
        enc_ref.set_bytes(6, &scale_f32);

        let grid_dims = MTLSize {
            width: hidden_size.div_ceil(256),
            height: out_tokens,
            depth: batch,
        };
        let group_dims = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Metal Audio Conformer Chunked Relative Attention (`[B, S, H*D]` BF16 -> `[B, S, H*D]` BF16).
#[allow(clippy::too_many_arguments)]
pub fn fused_metal_audio_chunked_rel_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    per_dim_scale: &Tensor,
    rel_k: &Tensor,
    mask: &Tensor,
    num_heads: usize,
    head_dim: usize,
    chunk_size: usize,
    max_past_horizon: usize,
    context_size: usize,
    q_scale: f64,
    k_scale: f64,
    softcap: f64,
) -> Result<Tensor> {
    let fn_name = "fused_metal_audio_chunked_rel_attn";
    let mdev = ensure_metal_bf16(q, fn_name, "q")?;
    ensure_metal_bf16(k, fn_name, "k")?;
    ensure_metal_bf16(v, fn_name, "v")?;
    if head_dim == 0 || head_dim > 256 || num_heads == 0 || chunk_size == 0 {
        candle_core::bail!(
            "{fn_name}: invalid dims num_heads={num_heads}, head_dim={head_dim}, chunk_size={chunk_size}"
        );
    }

    let (batch, seq_len, hidden_q) = q.dims3()?;
    if hidden_q != num_heads * head_dim
        || k.dims() != [batch, seq_len, hidden_q]
        || v.dims() != [batch, seq_len, hidden_q]
    {
        candle_core::bail!("{fn_name}: q/k/v shape mismatch");
    }
    if batch == 0 || seq_len == 0 {
        return Tensor::zeros((batch, seq_len, hidden_q), DType::BF16, q.device());
    }

    if !per_dim_scale.device().same_device(q.device())
        || per_dim_scale.dtype() != DType::F32
        || per_dim_scale.dims() != [head_dim]
    {
        candle_core::bail!("{fn_name}: invalid per_dim_scale");
    }
    let (rk_b, rk_h, rk_d, num_rel_pos) = rel_k.dims4()?;
    if !rel_k.device().same_device(q.device())
        || rel_k.dtype() != DType::F32
        || rk_b != 1
        || rk_h != num_heads
        || rk_d != head_dim
    {
        candle_core::bail!("{fn_name}: invalid rel_k shape {:?}", rel_k.dims());
    }
    if num_rel_pos > context_size + 1 {
        candle_core::bail!(
            "{fn_name}: num_rel_pos={num_rel_pos} > context_size+1={}",
            context_size + 1
        );
    }

    let num_blocks = seq_len.div_ceil(chunk_size);
    if !mask.device().same_device(q.device())
        || mask.dtype() != DType::F32
        || mask.dims() != [1, 1, num_blocks, chunk_size, context_size]
    {
        candle_core::bail!("{fn_name}: invalid mask shape {:?}", mask.dims());
    }

    let q = if !q.is_contiguous() {
        q.contiguous()?
    } else {
        q.clone()
    };
    let k = if !k.is_contiguous() {
        k.contiguous()?
    } else {
        k.clone()
    };
    let v = if !v.is_contiguous() {
        v.contiguous()?
    } else {
        v.clone()
    };
    let per_dim_scale = if !per_dim_scale.is_contiguous() {
        per_dim_scale.contiguous()?
    } else {
        per_dim_scale.clone()
    };
    let rel_k = if !rel_k.is_contiguous() {
        rel_k.contiguous()?
    } else {
        rel_k.clone()
    };
    let mask = if !mask.is_contiguous() {
        mask.contiguous()?
    } else {
        mask.clone()
    };

    let out = Tensor::zeros((batch, seq_len, hidden_q), DType::BF16, q.device())?;
    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "audio_chunked_rel_attn_bf16")?;

    let s_i32 = seq_len as i32;
    let h_i32 = num_heads as i32;
    let d_i32 = head_dim as i32;
    let chunk_i32 = chunk_size as i32;
    let past_i32 = max_past_horizon as i32;
    let ctx_i32 = context_size as i32;
    let rel_i32 = num_rel_pos as i32;
    let qs_f32 = q_scale as f32;
    let ks_f32 = k_scale as f32;
    let sc_f32 = softcap as f32;

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let (pds_storage, pds_layout) = per_dim_scale.storage_and_layout();
        let (rk_storage, rk_layout) = rel_k.storage_and_layout();
        let (m_storage, m_layout) = mask.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

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
        let pds_buf = match &*pds_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let rk_buf = match &*rk_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let m_buf = match &*m_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let bf16_bytes = DType::BF16.size_in_bytes();
        let f32_bytes = DType::F32.size_in_bytes();
        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(q_buf), q_layout.start_offset() * bf16_bytes);
        enc_ref.set_input_buffer(1, Some(k_buf), k_layout.start_offset() * bf16_bytes);
        enc_ref.set_input_buffer(2, Some(v_buf), v_layout.start_offset() * bf16_bytes);
        enc_ref.set_input_buffer(3, Some(pds_buf), pds_layout.start_offset() * f32_bytes);
        enc_ref.set_input_buffer(4, Some(rk_buf), rk_layout.start_offset() * f32_bytes);
        enc_ref.set_input_buffer(5, Some(m_buf), m_layout.start_offset() * f32_bytes);
        enc_ref.set_output_buffer(6, Some(out_buf), out_layout.start_offset() * bf16_bytes);
        enc_ref.set_bytes(7, &s_i32);
        enc_ref.set_bytes(8, &h_i32);
        enc_ref.set_bytes(9, &d_i32);
        enc_ref.set_bytes(10, &chunk_i32);
        enc_ref.set_bytes(11, &past_i32);
        enc_ref.set_bytes(12, &ctx_i32);
        enc_ref.set_bytes(13, &rel_i32);
        enc_ref.set_bytes(14, &qs_f32);
        enc_ref.set_bytes(15, &ks_f32);
        enc_ref.set_bytes(16, &sc_f32);

        let grid_dims = MTLSize {
            width: num_blocks,
            height: num_heads,
            depth: batch,
        };
        let group_dims = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_bf16_ulp_eq(a: &[f32], b: &[f32], label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: len mismatch");
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                va.is_finite() && vb.is_finite(),
                "{label} non-finite at idx {i}: {va} vs {vb}"
            );
            let mag = va.abs().max(vb.abs());
            // BF16 has 7 mantissa bits (1 ULP = 2^-7 ~ 0.0078125). Allow 2 ULPs + 1e-2 floor.
            let tol = (0.01f32).max(mag * 0.016);
            let diff = (va - vb).abs();
            assert!(
                diff <= tol,
                "{label} mismatch at idx {i}: ref={va} vs got={vb} (diff={diff} > tol={tol})"
            );
            dot += (va as f64) * (vb as f64);
            na += (va as f64) * (va as f64);
            nb += (vb as f64) * (vb as f64);
        }
        if na > 0.0 && nb > 0.0 {
            let cos = dot / (na.sqrt() * nb.sqrt());
            assert!(
                cos >= 0.99995,
                "{label} cosine similarity {cos:.6} < 0.99995"
            );
        }
    }

    #[test]
    fn test_metal_kernels_respect_start_offset() -> Result<()> {
        if !is_metal_device_available() {
            return Ok(());
        }
        let dev = Device::new_metal(0)?;

        // Create a [2, 512] tensor where row 0 and row 1 have distinct values
        let mut data = Vec::with_capacity(2 * 512);
        for i in 0..512 {
            data.push((i as f32 + 1.0) * 0.01);
        }
        for i in 0..512 {
            data.push((512 - i) as f32 * 0.03 - 2.0);
        }
        let full = Tensor::from_vec(data, (2, 512), &dev)?.to_dtype(DType::BF16)?;
        let weight = Tensor::ones(512, DType::BF16, &dev)?;

        // Narrow to row 1 (contiguous == true, start_offset == 512)
        let row1_view = full.narrow(0, 1, 1)?;
        assert!(row1_view.is_contiguous());
        assert_eq!(row1_view.layout().start_offset(), 512);

        // Copy row 1 into a fresh offset-0 buffer for reference
        let row1_fresh = Tensor::from_vec(
            row1_view
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            (1, 512),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        assert_eq!(row1_fresh.layout().start_offset(), 0);

        let norm_view = fused_metal_rmsnorm(&row1_view, Some(&weight), 1e-6)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let norm_fresh = fused_metal_rmsnorm(&row1_fresh, Some(&weight), 1e-6)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            norm_view, norm_fresh,
            "fused_metal_rmsnorm ignored start_offset!"
        );

        let geglu_view = fused_metal_geglu(&row1_view, &row1_view)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let geglu_fresh = fused_metal_geglu(&row1_fresh, &row1_fresh)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            geglu_view, geglu_fresh,
            "fused_metal_geglu ignored start_offset!"
        );

        Ok(())
    }

    // NOTE(Hardware Invariance):
    // Apple Silicon M3/M4/M5 contains native hardware bfloat16 ALUs (__HAVE_BFLOAT__), whereas
    // M1/M2 silicon (e.g. M1, M2, M2 Max) emulates bfloat16 with float32 intermediates in MSL shaders.
    // Due to differing Fused Multiply-Add (FMA) instruction contraction order, element-wise differences
    // of exactly 1 bfloat16 ULP (0.015625 for magnitude ~2.8, 0.03125 for magnitude ~7.0) occur between
    // fused single-pass shaders and multi-pass unfused reference pipelines on M1/M2 hardware.
    // See docs/architecture/hardware-acceleration.md (Section 8) for detailed analysis and M2 Max debug plan.
    #[test]
    fn test_phase2_fused_kernels_bit_exact_parity() -> Result<()> {
        if !is_metal_device_available() {
            return Ok(());
        }
        let dev = Device::new_metal(0)?;
        let eps = 1e-6;

        // 1. Test fused_metal_rmsnorm_residual_add (with and without scalar) across wide dynamic range
        let x_vec: Vec<f32> = (0..(20 * 512))
            .map(|i| ((i as f32) * 0.137).sin() * 12.0)
            .collect();
        let res_vec: Vec<f32> = (0..(20 * 512))
            .map(|i| ((i as f32) * 0.029).cos() * 8.0)
            .collect();
        let w_vec: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.073).sin() * 3.0).collect();
        let x = Tensor::from_vec(x_vec, (2, 10, 512), &dev)?.to_dtype(DType::BF16)?;
        let res = Tensor::from_vec(res_vec, (2, 10, 512), &dev)?.to_dtype(DType::BF16)?;
        let w = Tensor::from_vec(w_vec, (512,), &dev)?.to_dtype(DType::BF16)?;
        let scalar = Tensor::from_vec(vec![0.87109375f32], (1,), &dev)?.to_dtype(DType::BF16)?;

        let ref_no_scale = res.add(&fused_metal_rmsnorm(&x, Some(&w), eps)?)?;
        let got_no_scale = fused_metal_rmsnorm_residual_add(&x, &res, Some(&w), None, eps)?;
        assert_bf16_ulp_eq(
            &ref_no_scale
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            &got_no_scale
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            "fused_metal_rmsnorm_residual_add (no scale)",
        );

        let ref_with_scale = ref_no_scale.broadcast_mul(&scalar)?;
        let got_with_scale =
            fused_metal_rmsnorm_residual_add(&x, &res, Some(&w), Some(&scalar), eps)?;
        assert_bf16_ulp_eq(
            &ref_with_scale
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            &got_with_scale
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            "fused_metal_rmsnorm_residual_add (with scale)",
        );

        // 2. Test fused_metal_qk_rmsnorm_rope_transpose & fused_metal_v_rmsnorm_transpose for d=256 and d=512
        for &d in &[256usize, 512usize] {
            let (b, s, h) = (2, 17, 4);
            let q_vec: Vec<f32> = (0..(b * s * h * d))
                .map(|i| ((i as f32) * 0.117).sin() * 15.0)
                .collect();
            let qw_vec: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.053).cos() * 2.5).collect();
            let q_in = Tensor::from_vec(q_vec, (b, s, h * d), &dev)?.to_dtype(DType::BF16)?;
            let qw = Tensor::from_vec(qw_vec, (d,), &dev)?.to_dtype(DType::BF16)?;

            let rope = crate::rope::RotaryEmbedding::new(d, 10_000.0, &dev)?;
            let (cos, sin) = rope.get_cos_sin(s, &dev, DType::BF16)?;

            let q_unfused = q_in.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?;
            let q_unfused = fused_metal_rmsnorm(&q_unfused, Some(&qw), eps)?;
            let q_unfused = crate::rope::apply_rotary_pos_emb(&q_unfused, &cos, &sin)?;

            let q_fused =
                fused_metal_qk_rmsnorm_rope_transpose(&q_in, &qw, &cos, &sin, b, s, h, d, eps)?;
            assert_bf16_ulp_eq(
                &q_unfused
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &q_fused
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &format!("fused_metal_qk_rmsnorm_rope_transpose d={d}"),
            );

            let v_unfused = q_in.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?;
            let v_unfused = fused_metal_rmsnorm(&v_unfused, None, eps)?;
            let v_fused = fused_metal_v_rmsnorm_transpose(&q_in, None, b, s, h, d, eps)?;
            assert_bf16_ulp_eq(
                &v_unfused
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &v_fused
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &format!("fused_metal_v_rmsnorm_transpose d={d}"),
            );
        }

        // 3. Test fused_metal_ple_scale_rmsnorm_transpose with wide dynamic range
        let (b, s, l, d_ple) = (2, 5, 24, 512);
        let ple_vec: Vec<f32> = (0..(b * s * l * d_ple))
            .map(|i| ((i as f32) * 0.137).sin() * 12.0)
            .collect();
        let ple_in = Tensor::from_vec(ple_vec, (b, s, l * d_ple), &dev)?.to_dtype(DType::BF16)?;
        let scale = (d_ple as f64).powf(-0.5);

        let ple_fused =
            fused_metal_ple_scale_rmsnorm_transpose(&ple_in, &w, scale, eps, b, s, l, d_ple)?;
        let scaled_unfused = (&ple_in * scale)?.reshape((b, s, l, d_ple))?;
        for layer_idx in 0..l {
            let slice_unfused = scaled_unfused.narrow(2, layer_idx, 1)?.squeeze(2)?;
            let norm_unfused = fused_metal_rmsnorm(&slice_unfused, Some(&w), eps)?;
            let slice_fused = ple_fused.get(layer_idx)?;
            assert!(slice_fused.is_contiguous());
            let v_unfused = norm_unfused
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let v_fused = slice_fused
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_bf16_ulp_eq(
                &v_unfused,
                &v_fused,
                &format!("fused_metal_ple_scale_rmsnorm_transpose layer {layer_idx}"),
            );

            let single_layer_in = ple_in
                .reshape((b, s, l, d_ple))?
                .narrow(2, layer_idx, 1)?
                .squeeze(2)?
                .contiguous()?;
            let single_layer_got =
                fused_metal_ple_scale_rmsnorm(&single_layer_in, &w, scale, eps, d_ple)?;
            let v_single = single_layer_got
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(
                v_single, v_fused,
                "ple_scale_rms_norm_bf16 must match ple_scale_rms_norm_transpose_bf16 bit-for-bit at layer {layer_idx}"
            );
        }

        // 3b. Test fused_metal_pool_and_l2norm (unmasked, masked, MRL-truncated, and 8K length)
        {
            let (pb, ps, ph) = (2usize, 129usize, 768usize);
            let p_vec: Vec<f32> = (0..(pb * ps * ph))
                .map(|i| ((i as f32) * 0.037).sin() * 2.5 + 0.1)
                .collect();
            let p_bf16 = Tensor::from_vec(p_vec, (pb, ps, ph), &dev)?.to_dtype(DType::BF16)?;
            let mut mask_vec = vec![1u32; pb * ps];
            for val in &mut mask_vec[(ps - 11)..ps] {
                *val = 0;
            }
            for val in &mut mask_vec[(pb * ps - 3)..(pb * ps)] {
                *val = 0;
            }
            let p_mask = Tensor::from_vec(mask_vec, (pb, ps), &dev)?;

            for mask_opt in [None, Some(&p_mask)] {
                for trunc_opt in [None, Some(256usize)] {
                    let embeds_f32 = p_bf16.to_dtype(DType::F32)?;
                    let pooled_f32 = if let Some(m) = mask_opt {
                        let m_f32 = m.to_dtype(DType::F32)?.unsqueeze(2)?;
                        let sum_e = embeds_f32.broadcast_mul(&m_f32)?.sum(1)?;
                        let sum_m = Tensor::clamp(&m_f32.sum(1)?, 1e-9f32, f32::MAX)?;
                        sum_e.broadcast_div(&sum_m)?
                    } else {
                        embeds_f32.mean(1)?
                    };
                    let n1 = Tensor::clamp(
                        &pooled_f32
                            .sqr()?
                            .sum_keepdim(candle_core::D::Minus1)?
                            .sqrt()?,
                        1e-12f32,
                        f32::MAX,
                    )?;
                    let mut norm_ref = pooled_f32.broadcast_div(&n1)?;
                    if let Some(dim) = trunc_opt {
                        let sl = norm_ref.narrow(candle_core::D::Minus1, 0, dim)?;
                        let n2 = Tensor::clamp(
                            &sl.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?,
                            1e-12f32,
                            f32::MAX,
                        )?;
                        norm_ref = sl.broadcast_div(&n2)?;
                    }
                    let ref_bf16 = norm_ref.to_dtype(DType::BF16)?;
                    let got_bf16 = fused_metal_pool_and_l2norm(&p_bf16, mask_opt, trunc_opt)?;
                    assert_bf16_ulp_eq(
                        &ref_bf16
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        &got_bf16
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        &format!(
                            "fused_metal_pool_and_l2norm masked={} trunc={:?}",
                            mask_opt.is_some(),
                            trunc_opt
                        ),
                    );
                }
            }
        }

        // 4. Test fused_metal_ple_geglu vs unfused gelu_tanh + broadcast_mul across [-20, 20]
        let x_wide_vec: Vec<f32> = (0..(20 * 512))
            .map(|i| ((i as f32) * 0.071).sin() * 20.0)
            .collect();
        let x_wide = Tensor::from_vec(x_wide_vec, (2, 10, 512), &dev)?.to_dtype(DType::BF16)?;
        let ref_ple_geglu = crate::ple::gelu_tanh(&x_wide)?.broadcast_mul(&res)?;
        let got_ple_geglu = fused_metal_ple_geglu(&x_wide, &res)?;
        let v_ref_geglu = ref_ple_geglu
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let v_got_geglu = got_ple_geglu
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_bf16_ulp_eq(&v_ref_geglu, &v_got_geglu, "ple_geglu");

        // 5. Test cached RoPE vs direct seq_len RoPE for both sliding and full attention
        let seq_test = 315usize;
        for &(d, theta) in &[(256usize, 10_000.0f64), (512usize, 1_000_000.0f64)] {
            let rope = crate::rope::RotaryEmbedding::new(d, theta, &dev)?;
            let (cos_cached, sin_cached) = rope.get_cos_sin(seq_test, &dev, DType::BF16)?;
            let pos_direct = Tensor::arange(0u32, seq_test as u32, &dev)?
                .to_dtype(DType::F32)?
                .reshape((1, seq_test, 1))?;
            let half_dim = d / 2;
            let mut inv_freq = Vec::with_capacity(half_dim);
            for i in 0..half_dim {
                inv_freq.push((1.0 / theta.powf((2 * i) as f64 / d as f64)) as f32);
            }
            let inv_freq = Tensor::from_vec(inv_freq, (1, 1, half_dim), &dev)?;
            let freqs = pos_direct.broadcast_mul(&inv_freq)?;
            let emb = Tensor::cat(&[&freqs, &freqs], candle_core::D::Minus1)?;
            let cos_direct = emb
                .cos()?
                .reshape((1, 1, seq_test, d))?
                .to_dtype(DType::BF16)?;
            let sin_direct = emb
                .sin()?
                .reshape((1, 1, seq_test, d))?
                .to_dtype(DType::BF16)?;
            assert_bf16_ulp_eq(
                &cos_cached
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &cos_direct
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &format!("cos_cached vs cos_direct d={d}"),
            );
            assert_bf16_ulp_eq(
                &sin_cached
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &sin_direct
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &format!("sin_cached vs sin_direct d={d}"),
            );
        }

        Ok(())
    }

    // NOTE(Hardware Invariance):
    // See test_phase2 note and docs/architecture/hardware-acceleration.md (Section 8).
    // On M1/M2 silicon, 2D RoPE + RMSNorm fused vision transpositions exhibit 1-ULP differences
    // (diff=0.015625 for values ~2.8125) due to software bfloat16 emulation vs hardware bfloat on M3+.
    #[test]
    fn test_phase3_fused_kernels_bit_exact_parity() -> Result<()> {
        if !is_metal_device_available() {
            return Ok(());
        }
        let dev = Device::new_metal(0)?;
        let eps = 1e-6;

        // 1. Test fused_metal_geglu_gate_up vs fused_metal_geglu
        let (b, s, d_mlp) = (2, 13, 2048);
        let gate_vec: Vec<f32> = (0..(b * s * d_mlp))
            .map(|i| ((i as f32) * 0.091).sin() * 14.0)
            .collect();
        let up_vec: Vec<f32> = (0..(b * s * d_mlp))
            .map(|i| ((i as f32) * 0.047).cos() * 9.0)
            .collect();
        let gate = Tensor::from_vec(gate_vec, (b, s, d_mlp), &dev)?.to_dtype(DType::BF16)?;
        let up = Tensor::from_vec(up_vec, (b, s, d_mlp), &dev)?.to_dtype(DType::BF16)?;
        let gate_up = Tensor::cat(&[&gate, &up], candle_core::D::Minus1)?;

        let ref_geglu = fused_metal_geglu(&gate, &up)?;
        let got_geglu = fused_metal_geglu_gate_up(&gate_up)?;
        assert_bf16_ulp_eq(
            &ref_geglu
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            &got_geglu
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            "fused_metal_geglu_gate_up",
        );

        // 2. Test fused_metal_qkv_rmsnorm_rope_transpose vs separate Q/K/V fused kernels
        for &(h_q, h_kv, d) in &[(4usize, 2usize, 256usize), (4usize, 1usize, 512usize)] {
            let (b, s) = (2, 19);
            let q_vec: Vec<f32> = (0..(b * s * h_q * d))
                .map(|i| ((i as f32) * 0.113).sin() * 11.0)
                .collect();
            let k_vec: Vec<f32> = (0..(b * s * h_kv * d))
                .map(|i| ((i as f32) * 0.079).cos() * 10.0)
                .collect();
            let v_vec: Vec<f32> = (0..(b * s * h_kv * d))
                .map(|i| ((i as f32) * 0.053).sin() * 8.0)
                .collect();
            let qw_vec: Vec<f32> = (0..d)
                .map(|i| ((i as f32) * 0.037).cos() * 2.0 + 1.0)
                .collect();
            let kw_vec: Vec<f32> = (0..d)
                .map(|i| ((i as f32) * 0.061).sin() * 2.0 + 1.0)
                .collect();

            let q_in = Tensor::from_vec(q_vec, (b, s, h_q * d), &dev)?.to_dtype(DType::BF16)?;
            let k_in = Tensor::from_vec(k_vec, (b, s, h_kv * d), &dev)?.to_dtype(DType::BF16)?;
            let v_in = Tensor::from_vec(v_vec, (b, s, h_kv * d), &dev)?.to_dtype(DType::BF16)?;
            let qkv_in = Tensor::cat(&[&q_in, &k_in, &v_in], candle_core::D::Minus1)?;

            let qw = Tensor::from_vec(qw_vec, (d,), &dev)?.to_dtype(DType::BF16)?;
            let kw = Tensor::from_vec(kw_vec, (d,), &dev)?.to_dtype(DType::BF16)?;
            let rope = crate::rope::RotaryEmbedding::new(d, 10_000.0, &dev)?;
            let (cos, sin) = rope.get_cos_sin(s, &dev, DType::BF16)?;

            let q_ref =
                fused_metal_qk_rmsnorm_rope_transpose(&q_in, &qw, &cos, &sin, b, s, h_q, d, eps)?;
            let k_ref =
                fused_metal_qk_rmsnorm_rope_transpose(&k_in, &kw, &cos, &sin, b, s, h_kv, d, eps)?;
            let v_ref = fused_metal_v_rmsnorm_transpose(&v_in, None, b, s, h_kv, d, eps)?;

            let (q_got, k_got, v_got) = fused_metal_qkv_rmsnorm_rope_transpose(
                &qkv_in, &qw, &kw, None, &cos, &sin, b, s, h_q, h_kv, d, eps, eps, eps,
            )?;

            assert_bf16_ulp_eq(
                &q_ref
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &q_got
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &format!("qkv_rmsnorm_rope_transpose Q d={d}"),
            );
            assert_bf16_ulp_eq(
                &k_ref
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &k_got
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &format!("qkv_rmsnorm_rope_transpose K d={d}"),
            );
            assert_bf16_ulp_eq(
                &v_ref
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &v_got
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &format!("qkv_rmsnorm_rope_transpose V d={d}"),
            );
        }

        // 3. Test fused_metal_vision_qkv_rmsnorm_rope2d_transpose vs unfused VisionAttention path
        {
            let (b, s, h, d) = (1, 36, 12, 64);
            let q_vec: Vec<f32> = (0..(b * s * h * d))
                .map(|i| ((i as f32) * 0.083).sin() * 9.0)
                .collect();
            let k_vec: Vec<f32> = (0..(b * s * h * d))
                .map(|i| ((i as f32) * 0.067).cos() * 7.0)
                .collect();
            let v_vec: Vec<f32> = (0..(b * s * h * d))
                .map(|i| ((i as f32) * 0.041).sin() * 6.0)
                .collect();
            let qw_vec: Vec<f32> = (0..d)
                .map(|i| ((i as f32) * 0.031).cos() * 1.5 + 1.0)
                .collect();
            let kw_vec: Vec<f32> = (0..d)
                .map(|i| ((i as f32) * 0.059).sin() * 1.5 + 1.0)
                .collect();

            let q_in = Tensor::from_vec(q_vec, (b, s, h * d), &dev)?.to_dtype(DType::BF16)?;
            let k_in = Tensor::from_vec(k_vec, (b, s, h * d), &dev)?.to_dtype(DType::BF16)?;
            let v_in = Tensor::from_vec(v_vec, (b, s, h * d), &dev)?.to_dtype(DType::BF16)?;
            let qkv_in = Tensor::cat(&[&q_in, &k_in, &v_in], candle_core::D::Minus1)?;

            let qw = Tensor::from_vec(qw_vec, (d,), &dev)?.to_dtype(DType::BF16)?;
            let kw = Tensor::from_vec(kw_vec, (d,), &dev)?.to_dtype(DType::BF16)?;

            let mut pos_ids = Vec::with_capacity(s * 2);
            for p in 0..s {
                pos_ids.push((p % 6) as u32);
                pos_ids.push((p / 6) as u32);
            }
            let pos_tensor = Tensor::from_vec(pos_ids, (s, 2), &dev)?;
            let vrope = crate::multimodal::vision::VisionRotaryEmbedding::new(d, 100.0);
            let (cos, sin) = vrope.precompute(&pos_tensor, s, DType::BF16, &dev)?;

            let q_ref = fused_metal_rmsnorm(&q_in.reshape((b, s, h, d))?, Some(&qw), eps)?
                .permute((0, 2, 1, 3))?
                .contiguous()?;
            let q_ref = vrope.apply_precomputed(&q_ref, &cos, &sin)?;
            let k_ref = fused_metal_rmsnorm(&k_in.reshape((b, s, h, d))?, Some(&kw), eps)?
                .permute((0, 2, 1, 3))?
                .contiguous()?;
            let k_ref = vrope.apply_precomputed(&k_ref, &cos, &sin)?;
            let v_ref = fused_metal_rmsnorm(&v_in.reshape((b, s, h, d))?, None, eps)?
                .permute((0, 2, 1, 3))?
                .contiguous()?;

            let (q_got, k_got, v_got) = fused_metal_vision_qkv_rmsnorm_rope2d_transpose(
                &qkv_in, &qw, &kw, None, &cos, &sin, b, s, h, d, eps,
            )?;

            assert_bf16_ulp_eq(
                &q_ref
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &q_got
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                "vision_qkv_rmsnorm_rope2d_transpose Q",
            );
            assert_bf16_ulp_eq(
                &k_ref
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &k_got
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                "vision_qkv_rmsnorm_rope2d_transpose K",
            );
            assert_bf16_ulp_eq(
                &v_ref
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &v_got
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                "vision_qkv_rmsnorm_rope2d_transpose V",
            );
        }

        // 4. Test fused_metal_audio_glu_causal_dwconv5 vs unfused AudioLightConv1d GLU + 5-tap conv
        {
            let (b, s, c) = (1, 25, 1024);
            let pw1_vec: Vec<f32> = (0..(b * s * 2 * c))
                .map(|i| ((i as f32) * 0.043).sin() * 5.0)
                .collect();
            let dw_vec: Vec<f32> = (0..(c * 5))
                .map(|i| ((i as f32) * 0.071).cos() * 0.5)
                .collect();
            let pw1 = Tensor::from_vec(pw1_vec, (b, s, 2 * c), &dev)?.to_dtype(DType::BF16)?;
            let dw_w = Tensor::from_vec(dw_vec, (1, c, 5), &dev)?.to_dtype(DType::BF16)?;

            let half0 = pw1.narrow(candle_core::D::Minus1, 0, c)?;
            let half1 = pw1.narrow(candle_core::D::Minus1, c, c)?;
            let sig = candle_nn::ops::sigmoid(&half1)?;
            let glu_out = (half0 * sig)?;
            let dw_in = glu_out.transpose(1, 2)?.contiguous()?.pad_with_zeros(
                candle_core::D::Minus1,
                4,
                0,
            )?;
            let mut dw_ref = dw_in
                .narrow(2, 0, s)?
                .broadcast_mul(&dw_w.narrow(2, 0, 1)?)?;
            for k in 1..5 {
                let term = dw_in
                    .narrow(2, k, s)?
                    .broadcast_mul(&dw_w.narrow(2, k, 1)?)?;
                dw_ref = (dw_ref + term)?;
            }
            let dw_ref = dw_ref.transpose(1, 2)?.contiguous()?;

            let dw_got = fused_metal_audio_glu_causal_dwconv5(&pw1, &dw_w)?;
            assert_bf16_ulp_eq(
                &dw_ref
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &dw_got
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                "fused_metal_audio_glu_causal_dwconv5",
            );
        }

        // 5. Test fused_metal_q8_0_matmul (MV m=1..=8, MM m>=9, and odd K-blocks) + dequantize_q8_0_to_bf16
        for &(out_dim, in_dim) in &[(128usize, 256usize), (70usize, 96usize)] {
            let w_f32_vec: Vec<f32> = (0..(out_dim * in_dim))
                .map(|i| ((i as f32) * 0.031).sin() * 1.5)
                .collect();
            let w_cpu = Tensor::from_vec(w_f32_vec, (out_dim, in_dim), &Device::Cpu)?;
            let qt_cpu = candle_core::quantized::QTensor::quantize(
                &w_cpu,
                candle_core::quantized::GgmlDType::Q8_0,
            )?;
            let raw_bytes = qt_cpu.data()?.into_owned();
            let row_bytes = (in_dim / 32) * 34;
            let q8_bytes_metal = Tensor::from_slice(&raw_bytes, (out_dim, row_bytes), &dev)?;

            let qt_metal = candle_core::quantized::QTensor::quantize_onto(
                &w_cpu,
                candle_core::quantized::GgmlDType::Q8_0,
                &dev,
            )?;
            let dequant_ref = qt_metal.dequantize(&dev)?.to_dtype(DType::BF16)?;
            let dequant_got = dequantize_q8_0_to_bf16(&q8_bytes_metal, out_dim, in_dim)?;
            assert_bf16_ulp_eq(
                &dequant_ref
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &dequant_got
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &format!("dequantize_q8_0_to_bf16 {out_dim}x{in_dim}"),
            );

            let qm_ref = candle_core::quantized::QMatMul::from_qtensor(qt_metal)?;

            for &m in &[1usize, 4usize, 8usize, 9usize, 19usize, 64usize] {
                let x_vec: Vec<f32> = (0..(m * in_dim))
                    .map(|i| ((i as f32) * 0.059).cos() * 3.0)
                    .collect();
                let x_bf16 =
                    Tensor::from_vec(x_vec, (1, m, in_dim), &dev)?.to_dtype(DType::BF16)?;

                let ref_out = candle_core::Module::forward(&qm_ref, &x_bf16.to_dtype(DType::F32)?)?
                    .to_dtype(DType::BF16)?;
                let got_out = fused_metal_q8_0_matmul(&q8_bytes_metal, &x_bf16, out_dim, in_dim)?;

                let ref_vec = ref_out
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let got_vec = got_out
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_bf16_ulp_eq(
                    &ref_vec,
                    &got_vec,
                    &format!("fused_metal_q8_0_matmul {out_dim}x{in_dim} m={m}"),
                );
            }
        }

        // 6. Test M5 bf16_mm_m5 (including narrow(0, ...) view and non-multiple-of-64 dimensions)
        if MetalCapabilities::detect(dev.as_metal_device()?.metal_device()).use_m5_tensor_ops() {
            let (full_n, k) = (256usize, 128usize);
            let w_vec: Vec<f32> = (0..(full_n * k))
                .map(|i| ((i as f32) * 0.029).sin() * 1.2)
                .collect();
            let w_full = Tensor::from_vec(w_vec, (full_n, k), &dev)?.to_dtype(DType::BF16)?;
            let w_view = w_full.narrow(0, 70, 130)?; // non-zero start_offset and N=130

            let (b, s) = (2usize, 35usize); // M = 70 (non-multiple of 64)
            let x_vec: Vec<f32> = (0..(b * s * k))
                .map(|i| ((i as f32) * 0.043).cos() * 2.0)
                .collect();
            let x_bf16 = Tensor::from_vec(x_vec, (b, s, k), &dev)?.to_dtype(DType::BF16)?;

            let ref_linear = candle_nn::Linear::new(w_view.clone(), None);
            let ref_out = candle_core::Module::forward(&ref_linear, &x_bf16)?;
            let got_out = fused_metal_bf16_matmul(&w_view, &x_bf16)?;
            assert_eq!(got_out.dims(), &[b, s, 130]);

            assert_bf16_ulp_eq(
                &ref_out
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &got_out
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                "fused_metal_bf16_matmul narrow view",
            );

            // Check large-K / non-64 M=25,125 Audio-like shapes against F32 ground truth
            for &(am, ak, an) in &[
                (25usize, 1024usize, 1024usize),
                (25usize, 4096usize, 1024usize),
                (125usize, 1024usize, 4096usize),
            ] {
                let w_vec: Vec<f32> = (0..(an * ak))
                    .map(|i| ((i as f32) * 0.017).sin() * 0.5)
                    .collect();
                let x_vec: Vec<f32> = (0..(am * ak))
                    .map(|i| ((i as f32) * 0.023).cos() * 10.0)
                    .collect();
                let w_bf16 = Tensor::from_vec(w_vec, (an, ak), &dev)?.to_dtype(DType::BF16)?;
                let x_bf16 = Tensor::from_vec(x_vec, (1, am, ak), &dev)?.to_dtype(DType::BF16)?;
                let ref_f32 = x_bf16
                    .to_dtype(DType::F32)?
                    .reshape((am, ak))?
                    .matmul(&w_bf16.to_dtype(DType::F32)?.t()?)?
                    .to_dtype(DType::BF16)?;
                let lin = candle_nn::Linear::new(w_bf16.clone(), None);
                let ref_lin = candle_core::Module::forward(&lin, &x_bf16)?;
                let got_m5 = fused_metal_bf16_matmul(&w_bf16, &x_bf16)?;
                assert_bf16_ulp_eq(
                    &ref_f32
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    &got_m5
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    &format!("bf16_mm_m5 vs F32 ({am},{ak},{an})"),
                );
                assert_bf16_ulp_eq(
                    &ref_lin
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    &got_m5
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    &format!("bf16_mm_m5 vs Linear ({am},{ak},{an})"),
                );
            }
        }

        // 7. Test fused_metal_vision_spatial_pool vs unfused VisionPooler F32 reference
        {
            let (vb, pw, ph, k_sz, hs) = (2usize, 18usize, 27usize, 3usize, 768usize);
            let p_vec: Vec<f32> = (0..(vb * pw * ph * hs))
                .map(|i| ((i as f32) * 0.041).sin() * 2.5)
                .collect();
            let p_bf16 = Tensor::from_vec(p_vec, (vb, pw * ph, hs), &dev)?.to_dtype(DType::BF16)?;
            let scale = (hs as f64).sqrt() / ((k_sz * k_sz) as f64);

            let out_w = pw / k_sz;
            let out_h = ph / k_sz;
            let h_sum = p_bf16
                .to_dtype(DType::F32)?
                .reshape((vb * out_h, k_sz, pw * hs))?
                .sum(1)?;
            let hw_sum = h_sum.reshape((vb * out_h, out_w, k_sz, hs))?.sum(2)?;
            let ref_pool = (&hw_sum * scale)?
                .reshape((vb, out_h * out_w, hs))?
                .to_dtype(DType::BF16)?;
            let got_pool = fused_metal_vision_spatial_pool(&p_bf16, pw, ph, k_sz, scale)?;
            assert_bf16_ulp_eq(
                &ref_pool
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &got_pool
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                "fused_metal_vision_spatial_pool",
            );
        }

        // 8. Test fused_metal_audio_chunked_rel_attn vs unfused F32 block-loop AudioAttention
        {
            let (ab, as_len, ah, ad) = (1usize, 31usize, 8usize, 128usize);
            let chunk_size = 12usize;
            let max_past_horizon = 12usize;
            let context_right = 0usize;
            let context_size = chunk_size + max_past_horizon + context_right;
            let num_rel_pos = 13usize;
            let num_blocks = as_len.div_ceil(chunk_size);
            let q_scale = (ad as f64).powf(-0.5) / 2.0f64.ln();
            let k_scale = (1.0 + std::f64::consts::E).ln() / 2.0f64.ln();
            let softcap = 50.0f64;

            let q_vec: Vec<f32> = (0..(ab * as_len * ah * ad))
                .map(|i| ((i as f32) * 0.037).sin() * 0.8)
                .collect();
            let k_vec: Vec<f32> = (0..(ab * as_len * ah * ad))
                .map(|i| ((i as f32) * 0.029).cos() * 0.8)
                .collect();
            let v_vec: Vec<f32> = (0..(ab * as_len * ah * ad))
                .map(|i| ((i as f32) * 0.019).sin() * 0.8)
                .collect();
            let pds_vec: Vec<f32> = (0..ad)
                .map(|i| 0.5 + ((i as f32) * 0.05).cos() * 0.3)
                .collect();
            let rk_vec: Vec<f32> = (0..(ah * ad * num_rel_pos))
                .map(|i| ((i as f32) * 0.043).sin() * 0.2)
                .collect();

            let q_bf16 =
                Tensor::from_vec(q_vec, (ab, as_len, ah * ad), &dev)?.to_dtype(DType::BF16)?;
            let k_bf16 =
                Tensor::from_vec(k_vec, (ab, as_len, ah * ad), &dev)?.to_dtype(DType::BF16)?;
            let v_bf16 =
                Tensor::from_vec(v_vec, (ab, as_len, ah * ad), &dev)?.to_dtype(DType::BF16)?;
            let per_dim_scale = Tensor::from_vec(pds_vec, (ad,), &dev)?;
            let rel_k = Tensor::from_vec(rk_vec, (1, ah, ad, num_rel_pos), &dev)?;

            let mut mask_vec = vec![-1e9f32; num_blocks * chunk_size * context_size];
            for blk in 0..num_blocks {
                for qi in 0..chunk_size {
                    let t_q = blk * chunk_size + qi;
                    if t_q < as_len {
                        for c in 0..context_size {
                            let t_k = (blk as i32) * (chunk_size as i32)
                                - (max_past_horizon as i32)
                                + (c as i32);
                            if t_k >= 0
                                && (t_k as usize) < as_len
                                && (t_k as usize) <= t_q + context_right
                                && t_q.saturating_sub(t_k as usize)
                                    <= max_past_horizon.saturating_sub(1)
                            {
                                mask_vec[blk * chunk_size * context_size + qi * context_size + c] =
                                    0.0f32;
                            }
                        }
                    }
                }
            }
            let mask_tensor =
                Tensor::from_vec(mask_vec, (1, 1, num_blocks, chunk_size, context_size), &dev)?;

            // Unfused F32 reference matching AudioAttention::forward
            let q_f = (q_bf16
                .to_dtype(DType::F32)?
                .reshape((ab, as_len, ah, ad))?
                .broadcast_mul(&per_dim_scale)?
                * q_scale)?;
            let k_f = (k_bf16.to_dtype(DType::F32)?.reshape((ab, as_len, ah, ad))? * k_scale)?;
            let v_f = v_bf16.to_dtype(DType::F32)?.reshape((ab, as_len, ah, ad))?;
            let pad_amount = num_blocks * chunk_size - as_len;
            let q_pad = if pad_amount > 0 {
                q_f.pad_with_zeros(1, 0, pad_amount)?
            } else {
                q_f
            };
            let q_blocks = q_pad
                .reshape((ab, num_blocks, chunk_size, ah, ad))?
                .permute((0, 3, 1, 2, 4))?
                .contiguous()?;
            let k_pad = k_f.pad_with_zeros(1, max_past_horizon, context_right + chunk_size - 1)?;
            let v_pad = v_f.pad_with_zeros(1, max_past_horizon, context_right + chunk_size - 1)?;
            let mut kbv = Vec::with_capacity(num_blocks);
            let mut vbv = Vec::with_capacity(num_blocks);
            for blk in 0..num_blocks {
                kbv.push(k_pad.narrow(1, blk * chunk_size, context_size)?);
                vbv.push(v_pad.narrow(1, blk * chunk_size, context_size)?);
            }
            let k_ctx = Tensor::stack(&kbv, 1)?;
            let v_ctx = Tensor::stack(&vbv, 1)?;
            let b_h_n = ab * ah * num_blocks;
            let k_perm = k_ctx.permute((0, 3, 1, 4, 2))?.contiguous()?;
            let ac = q_blocks
                .reshape((b_h_n, chunk_size, ad))?
                .matmul(&k_perm.reshape((b_h_n, ad, context_size))?)?
                .reshape((ab, ah, num_blocks, chunk_size, context_size))?;
            let bd = q_blocks
                .reshape((ab, ah, num_blocks * chunk_size, ad))?
                .contiguous()?
                .matmul(&rel_k)?
                .reshape((ab, ah, num_blocks, chunk_size, num_rel_pos))?;
            let rel_pad = (context_size + 1) - num_rel_pos;
            let matrix_bd = bd
                .pad_with_zeros(4, 0, rel_pad)?
                .reshape((ab, ah, num_blocks, chunk_size * (context_size + 1)))?
                .narrow(3, 0, chunk_size * context_size)?
                .reshape((ab, ah, num_blocks, chunk_size, context_size))?;
            let mut w = (ac + matrix_bd)?;
            w = ((w / softcap)?.tanh()? * softcap)?;
            w = w.broadcast_add(&mask_tensor)?;
            let probs = candle_nn::ops::softmax(&w, candle_core::D::Minus1)?.contiguous()?;
            let v_perm = v_ctx.permute((0, 3, 1, 2, 4))?.contiguous()?;
            let ref_audio = probs
                .reshape((b_h_n, chunk_size, context_size))?
                .matmul(&v_perm.reshape((b_h_n, context_size, ad))?)?
                .reshape((ab, ah, num_blocks, chunk_size, ad))?
                .permute((0, 2, 3, 1, 4))?
                .contiguous()?
                .reshape((ab, num_blocks * chunk_size, ah * ad))?
                .narrow(1, 0, as_len)?
                .to_dtype(DType::BF16)?;

            let got_audio = fused_metal_audio_chunked_rel_attn(
                &q_bf16,
                &k_bf16,
                &v_bf16,
                &per_dim_scale,
                &rel_k,
                &mask_tensor,
                ah,
                ad,
                chunk_size,
                max_past_horizon,
                context_size,
                q_scale,
                k_scale,
                softcap,
            )?;
            assert_bf16_ulp_eq(
                &ref_audio
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                &got_audio
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                "fused_metal_audio_chunked_rel_attn",
            );
        }

        Ok(())
    }

    #[test]
    fn test_fused_metal_kernels_reject_invalid_inputs() -> Result<()> {
        if !is_metal_device_available() {
            return Ok(());
        }
        let dev = Device::new_metal(0)?;

        // 1. Wrong dtype (F32) rejected cleanly
        let f32_x = Tensor::ones((1, 4, 8), DType::F32, &dev)?;
        assert!(fused_metal_rmsnorm(&f32_x, None, 1e-6).is_err());
        assert!(fused_metal_geglu(&f32_x, &f32_x).is_err());
        assert!(fused_metal_geglu_gate_up(&f32_x).is_err());

        // 2. 0-D and 0-len last dim rejected cleanly
        let scalar = Tensor::new(1.0f32, &dev)?.to_dtype(DType::BF16)?;
        assert!(fused_metal_rmsnorm(&scalar, None, 1e-6).is_err());
        let zero_d = Tensor::zeros((2, 0), DType::BF16, &dev)?;
        assert!(fused_metal_rmsnorm(&zero_d, None, 1e-6).is_err());

        // 3. Shape mismatches rejected cleanly
        let a = Tensor::ones((2, 8), DType::BF16, &dev)?;
        let b = Tensor::ones((2, 16), DType::BF16, &dev)?;
        assert!(fused_metal_rmsnorm_residual_add(&a, &b, None, None, 1e-6).is_err());
        assert!(fused_metal_geglu(&a, &b).is_err());
        let bad_w = Tensor::ones((4,), DType::BF16, &dev)?;
        assert!(fused_metal_rmsnorm(&a, Some(&bad_w), 1e-6).is_err());

        // 4. Odd head_dim rejected for 1D RoPE; non-multiple-of-4 rejected for 2D Vision RoPE
        let odd_d = 3usize;
        let qk_in = Tensor::ones((1, 2, 1, odd_d), DType::BF16, &dev)?;
        let qk_w = Tensor::ones((odd_d,), DType::BF16, &dev)?;
        let cos_odd = Tensor::ones((1, 1, 2, odd_d), DType::BF16, &dev)?;
        let sin_odd = Tensor::ones((1, 1, 2, odd_d), DType::BF16, &dev)?;
        assert!(
            fused_metal_qk_rmsnorm_rope_transpose(
                &qk_in, &qk_w, &cos_odd, &sin_odd, 1, 2, 1, odd_d, 1e-6
            )
            .is_err()
        );
        let odd_qkv_in = Tensor::ones((1, 2, 3 * odd_d), DType::BF16, &dev)?;
        assert!(
            fused_metal_qkv_rmsnorm_rope_transpose(
                &odd_qkv_in,
                &qk_w,
                &qk_w,
                None,
                &cos_odd,
                &sin_odd,
                1,
                2,
                1,
                1,
                odd_d,
                1e-6,
                1e-6,
                1e-6
            )
            .is_err()
        );

        let vis_d = 6usize; // even, but not divisible by 4!
        let vis_qkv = Tensor::ones((1, 2, 3 * vis_d), DType::BF16, &dev)?;
        let vis_w = Tensor::ones((vis_d,), DType::BF16, &dev)?;
        let cos_vis = Tensor::ones((1, 1, 2, vis_d), DType::BF16, &dev)?;
        let sin_vis = Tensor::ones((1, 1, 2, vis_d), DType::BF16, &dev)?;
        assert!(
            fused_metal_vision_qkv_rmsnorm_rope2d_transpose(
                &vis_qkv, &vis_w, &vis_w, None, &cos_vis, &sin_vis, 1, 2, 1, vis_d, 1e-6
            )
            .is_err()
        );

        // 5. F32 cos/sin rejected cleanly
        let cos_f32 = Tensor::ones((1, 1, 2, 8), DType::F32, &dev)?;
        let qk_in_8 = Tensor::ones((1, 2, 8), DType::BF16, &dev)?;
        let w8 = Tensor::ones((8,), DType::BF16, &dev)?;
        assert!(
            fused_metal_qk_rmsnorm_rope_transpose(
                &qk_in_8, &w8, &cos_f32, &cos_f32, 1, 2, 1, 8, 1e-6
            )
            .is_err()
        );

        // 6. Q8_0 matmul rejects in_dim not divisible by 32
        let bad_q8 = Tensor::zeros((4, 34), DType::U8, &dev)?;
        let bad_x = Tensor::ones((1, 16), DType::BF16, &dev)?;
        assert!(fused_metal_q8_0_matmul(&bad_q8, &bad_x, 4, 16).is_err());

        Ok(())
    }
}
