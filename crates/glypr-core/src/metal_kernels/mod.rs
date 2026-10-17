pub mod capabilities;
pub use capabilities::{compile_metal4_library, MetalCapabilities, MetalTier};

use candle_core::{DType, Device, Result, Tensor};
use candle_metal_kernels::metal::{ComputePipeline, Device as MtlDevice};
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

const FUSED_OPS_SOURCE: &str = include_str!("fused_ops.metal");
const M5_OPS_SOURCE: &str = include_str!("m5_ops.metal");

static PIPELINE_CACHE: LazyLock<RwLock<HashMap<(usize, &'static str), ComputePipeline>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn get_or_compile_pipeline(dev: &MtlDevice, fn_name: &'static str) -> Result<ComputePipeline> {
    let dev_key = (dev.as_ref() as *const _ as usize, fn_name);
    {
        let cache = PIPELINE_CACHE.read().unwrap();
        if let Some(p) = cache.get(&dev_key) {
            return Ok(p.clone());
        }
    }

    let library = dev
        .new_library_with_source(FUSED_OPS_SOURCE, None)
        .map_err(|e| candle_core::Error::Metal(format!("Failed to compile fused_ops.metal: {:?}", e).into()))?;

    let function = library
        .get_function(fn_name, None)
        .map_err(|e| candle_core::Error::Metal(format!("Function {} not found: {:?}", fn_name, e).into()))?;

    let pipeline = dev
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| candle_core::Error::Metal(format!("Failed to create {} pipeline: {:?}", fn_name, e).into()))?;

    let mut cache = PIPELINE_CACHE.write().unwrap();
    cache.insert(dev_key, pipeline.clone());
    Ok(pipeline)
}

fn get_or_compile_m5_pipeline(dev: &MtlDevice, fn_name: &'static str) -> Result<ComputePipeline> {
    let dev_key = (dev.as_ref() as *const _ as usize, fn_name);
    {
        let cache = PIPELINE_CACHE.read().unwrap();
        if let Some(p) = cache.get(&dev_key) {
            return Ok(p.clone());
        }
    }

    let library = compile_metal4_library(dev, M5_OPS_SOURCE)
        .map_err(|e| candle_core::Error::Metal(format!("Failed to compile m5_ops.metal: {:?}", e).into()))?;

    let function = library
        .get_function(fn_name, None)
        .map_err(|e| candle_core::Error::Metal(format!("Function {} not found: {:?}", fn_name, e).into()))?;

    let pipeline = dev
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| candle_core::Error::Metal(format!("Failed to create {} pipeline: {:?}", fn_name, e).into()))?;

    let mut cache = PIPELINE_CACHE.write().unwrap();
    cache.insert(dev_key, pipeline.clone());
    Ok(pipeline)
}

/// Fused Metal RMSNorm executing directly in threadgroup registers via SIMD shuffle reduction.
pub fn fused_metal_rmsnorm(x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let mdev = match x.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("fused_metal_rmsnorm requires a Metal device"),
    };

    let x = if !x.is_contiguous() { x.contiguous()? } else { x.clone() };
    let weight_holder: Option<Tensor> = match weight {
        Some(w) if !w.is_contiguous() => Some(w.contiguous()?),
        Some(w) if w.dtype() != DType::BF16 => Some(w.to_dtype(DType::BF16)?),
        Some(w) => Some(w.clone()),
        None => None,
    };
    let weight_ref = weight_holder.as_ref();

    let orig_shape = x.shape();
    let d = orig_shape.dims()[orig_shape.dims().len() - 1];
    let num_rows = x.elem_count() / d;

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "rms_norm_bf16")?;
    let out = Tensor::zeros(orig_shape.clone(), DType::BF16, x.device())?;

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

        let grid_dims = MTLSize { width: num_rows, height: 1, depth: 1 };
        let group_dims = MTLSize { width: 32, height: 1, depth: 1 };
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
    let mdev = match x.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("fused_metal_rmsnorm_residual_add requires a Metal device"),
    };

    let x = if !x.is_contiguous() { x.contiguous()? } else { x.clone() };
    let residual = if !residual.is_contiguous() { residual.contiguous()? } else { residual.clone() };

    let weight_holder: Option<Tensor> = match weight {
        Some(w) if !w.is_contiguous() => Some(w.contiguous()?),
        Some(w) if w.dtype() != DType::BF16 => Some(w.to_dtype(DType::BF16)?),
        Some(w) => Some(w.clone()),
        None => None,
    };
    let weight_ref = weight_holder.as_ref();

    let scalar_holder: Option<Tensor> = match layer_scalar {
        Some(s) if !s.is_contiguous() => Some(s.contiguous()?),
        Some(s) if s.dtype() != DType::BF16 => Some(s.to_dtype(DType::BF16)?),
        Some(s) => Some(s.clone()),
        None => None,
    };
    let scalar_ref = scalar_holder.as_ref();

    let orig_shape = x.shape();
    let d = orig_shape.dims()[orig_shape.dims().len() - 1];
    let num_rows = x.elem_count() / d;

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "rms_norm_residual_add_bf16")?;
    let out = Tensor::zeros(orig_shape.clone(), DType::BF16, x.device())?;

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

        let grid_dims = MTLSize { width: num_rows, height: 1, depth: 1 };
        let group_dims = MTLSize { width: 32, height: 1, depth: 1 };
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
    let mdev = match x.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("fused_metal_qk_rmsnorm_rope_transpose requires a Metal device"),
    };

    let x = if !x.is_contiguous() { x.contiguous()? } else { x.clone() };
    let weight = if !weight.is_contiguous() { weight.contiguous()? } else { weight.clone() };
    let weight = if weight.dtype() != DType::BF16 { weight.to_dtype(DType::BF16)? } else { weight };
    let cos = if !cos.is_contiguous() { cos.contiguous()? } else { cos.clone() };
    let sin = if !sin.is_contiguous() { sin.contiguous()? } else { sin.clone() };

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "rms_norm_rope_transpose_bf16")?;
    let out = Tensor::zeros((batch, num_heads, seq_len, head_dim), DType::BF16, x.device())?;

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

        let grid_dims = MTLSize { width: num_heads, height: seq_len, depth: batch };
        let group_dims = MTLSize { width: 32, height: 1, depth: 1 };
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
    let mdev = match x.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("fused_metal_v_rmsnorm_transpose requires a Metal device"),
    };

    let x = if !x.is_contiguous() { x.contiguous()? } else { x.clone() };
    let weight_holder: Option<Tensor> = match weight {
        Some(w) if !w.is_contiguous() => Some(w.contiguous()?),
        Some(w) if w.dtype() != DType::BF16 => Some(w.to_dtype(DType::BF16)?),
        Some(w) => Some(w.clone()),
        None => None,
    };
    let weight_ref = weight_holder.as_ref();

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "rms_norm_transpose_bf16")?;
    let out = Tensor::zeros((batch, num_heads, seq_len, head_dim), DType::BF16, x.device())?;

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

        let grid_dims = MTLSize { width: num_heads, height: seq_len, depth: batch };
        let group_dims = MTLSize { width: 32, height: 1, depth: 1 };
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
    let mdev = match x.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("fused_metal_ple_scale_rmsnorm_transpose requires a Metal device"),
    };

    let x = if !x.is_contiguous() { x.contiguous()? } else { x.clone() };
    let weight = if !weight.is_contiguous() { weight.contiguous()? } else { weight.clone() };
    let weight = if weight.dtype() != DType::BF16 { weight.to_dtype(DType::BF16)? } else { weight };

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "ple_scale_rms_norm_transpose_bf16")?;
    let out = Tensor::zeros((num_layers, batch, seq_len, d), DType::BF16, x.device())?;

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

        let grid_dims = MTLSize { width: num_layers, height: seq_len, depth: batch };
        let group_dims = MTLSize { width: 32, height: 1, depth: 1 };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
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
    let mdev = match gate.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("{} requires a Metal device", kernel_name),
    };

    let gate = if !gate.is_contiguous() { gate.contiguous()? } else { gate.clone() };
    let up = if !up.is_contiguous() { up.contiguous()? } else { up.clone() };

    let total_elements = gate.elem_count();
    let pipeline = get_or_compile_pipeline(mdev.metal_device(), kernel_name)?;
    let out = Tensor::zeros(gate.shape().clone(), DType::BF16, gate.device())?;
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
        let num_groups = (total_elements + thread_group_size - 1) / thread_group_size;
        let grid_dims = MTLSize { width: num_groups, height: 1, depth: 1 };
        let group_dims = MTLSize { width: thread_group_size, height: 1, depth: 1 };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Metal GeGLU reading directly from a concatenated `[..., 2 * D]` `gate_up` projection tensor.
pub fn fused_metal_geglu_gate_up(gate_up: &Tensor) -> Result<Tensor> {
    let mdev = match gate_up.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("fused_metal_geglu_gate_up requires a Metal device"),
    };

    let gate_up = if !gate_up.is_contiguous() { gate_up.contiguous()? } else { gate_up.clone() };
    let dims = gate_up.dims();
    if dims.is_empty() || dims[dims.len() - 1] % 2 != 0 {
        candle_core::bail!("fused_metal_geglu_gate_up expects last dim divisible by 2, got {:?}", dims);
    }
    let d = dims[dims.len() - 1] / 2;
    let mut out_dims = dims.to_vec();
    *out_dims.last_mut().unwrap() = d;

    let total_elements = gate_up.elem_count() / 2;
    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "geglu_gate_up_bf16")?;
    let out = Tensor::zeros(out_dims, DType::BF16, gate_up.device())?;
    let total_i32 = total_elements as i32;
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
        enc_ref.set_bytes(2, &total_i32);
        enc_ref.set_bytes(3, &d_i32);

        let thread_group_size = 256;
        let num_groups = (total_elements + thread_group_size - 1) / thread_group_size;
        let grid_dims = MTLSize { width: num_groups, height: 1, depth: 1 };
        let group_dims = MTLSize { width: thread_group_size, height: 1, depth: 1 };
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
    let mdev = match qkv.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("fused_metal_qkv_rmsnorm_rope_transpose requires a Metal device"),
    };

    let qkv = if !qkv.is_contiguous() { qkv.contiguous()? } else { qkv.clone() };
    let q_weight = if !q_weight.is_contiguous() { q_weight.contiguous()? } else { q_weight.clone() };
    let q_weight = if q_weight.dtype() != DType::BF16 { q_weight.to_dtype(DType::BF16)? } else { q_weight };
    let k_weight = if !k_weight.is_contiguous() { k_weight.contiguous()? } else { k_weight.clone() };
    let k_weight = if k_weight.dtype() != DType::BF16 { k_weight.to_dtype(DType::BF16)? } else { k_weight };
    let v_weight_holder: Option<Tensor> = match v_weight {
        Some(w) if !w.is_contiguous() => Some(w.contiguous()?),
        Some(w) if w.dtype() != DType::BF16 => Some(w.to_dtype(DType::BF16)?),
        Some(w) => Some(w.clone()),
        None => None,
    };
    let v_weight_ref = v_weight_holder.as_ref();
    let cos = if !cos.is_contiguous() { cos.contiguous()? } else { cos.clone() };
    let sin = if !sin.is_contiguous() { sin.contiguous()? } else { sin.clone() };

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "qkv_rmsnorm_rope_transpose_bf16")?;
    let q_out = Tensor::zeros((batch, num_q_heads, seq_len, head_dim), DType::BF16, qkv.device())?;
    let k_out = Tensor::zeros((batch, num_kv_heads, seq_len, head_dim), DType::BF16, qkv.device())?;
    let v_out = Tensor::zeros((batch, num_kv_heads, seq_len, head_dim), DType::BF16, qkv.device())?;

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

        let qkv_buf = match &*qkv_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let qw_buf = match &*qw_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let kw_buf = match &*kw_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let cos_buf = match &*cos_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let sin_buf = match &*sin_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let qo_buf = match &*qo_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let ko_buf = match &*ko_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let vo_buf = match &*vo_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };

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
            let vw_buf = match &*vw_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
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
        let group_dims = MTLSize { width: 32, height: 1, depth: 1 };
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
    let mdev = match qkv.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("fused_metal_vision_qkv_rmsnorm_rope2d_transpose requires a Metal device"),
    };

    let qkv = if !qkv.is_contiguous() { qkv.contiguous()? } else { qkv.clone() };
    let q_weight = if !q_weight.is_contiguous() { q_weight.contiguous()? } else { q_weight.clone() };
    let q_weight = if q_weight.dtype() != DType::BF16 { q_weight.to_dtype(DType::BF16)? } else { q_weight };
    let k_weight = if !k_weight.is_contiguous() { k_weight.contiguous()? } else { k_weight.clone() };
    let k_weight = if k_weight.dtype() != DType::BF16 { k_weight.to_dtype(DType::BF16)? } else { k_weight };
    let v_weight_holder: Option<Tensor> = match v_weight {
        Some(w) if !w.is_contiguous() => Some(w.contiguous()?),
        Some(w) if w.dtype() != DType::BF16 => Some(w.to_dtype(DType::BF16)?),
        Some(w) => Some(w.clone()),
        None => None,
    };
    let v_weight_ref = v_weight_holder.as_ref();
    let cos = if !cos.is_contiguous() { cos.contiguous()? } else { cos.clone() };
    let sin = if !sin.is_contiguous() { sin.contiguous()? } else { sin.clone() };

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "vision_qkv_rmsnorm_rope2d_transpose_bf16")?;
    let q_out = Tensor::zeros((batch, num_heads, seq_len, head_dim), DType::BF16, qkv.device())?;
    let k_out = Tensor::zeros((batch, num_heads, seq_len, head_dim), DType::BF16, qkv.device())?;
    let v_out = Tensor::zeros((batch, num_heads, seq_len, head_dim), DType::BF16, qkv.device())?;

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

        let qkv_buf = match &*qkv_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let qw_buf = match &*qw_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let kw_buf = match &*kw_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let cos_buf = match &*cos_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let sin_buf = match &*sin_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let qo_buf = match &*qo_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let ko_buf = match &*ko_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let vo_buf = match &*vo_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };

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
            let vw_buf = match &*vw_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
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
        let group_dims = MTLSize { width: 32, height: 1, depth: 1 };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok((q_out, k_out, v_out))
}

/// Fused Metal Audio GLU + 5-Tap Causal Depthwise Conv1D in a single dispatch.
pub fn fused_metal_audio_glu_causal_dwconv5(pw1: &Tensor, dw_weight: &Tensor) -> Result<Tensor> {
    let mdev = match pw1.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("fused_metal_audio_glu_causal_dwconv5 requires a Metal device"),
    };

    let pw1 = if !pw1.is_contiguous() { pw1.contiguous()? } else { pw1.clone() };
    let dw_weight = if !dw_weight.is_contiguous() { dw_weight.contiguous()? } else { dw_weight.clone() };
    let dw_weight = if dw_weight.dtype() != DType::BF16 {
        dw_weight.to_dtype(DType::BF16)?
    } else {
        dw_weight
    };

    let (batch, seq_len, two_c) = pw1.dims3()?;
    let c = two_c / 2;
    let total_elements = batch * seq_len * c;

    let pipeline = get_or_compile_pipeline(mdev.metal_device(), "audio_glu_causal_dwconv5_bf16")?;
    let out = Tensor::zeros((batch, seq_len, c), DType::BF16, pw1.device())?;

    let b_i32 = batch as i32;
    let s_i32 = seq_len as i32;
    let c_i32 = c as i32;

    {
        let (pw1_storage, pw1_layout) = pw1.storage_and_layout();
        let (dw_storage, dw_layout) = dw_weight.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let pw1_buf = match &*pw1_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let dw_buf = match &*dw_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let out_buf = match &*out_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };

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
        let num_groups = (total_elements + thread_group_size - 1) / thread_group_size;
        let grid_dims = MTLSize { width: num_groups, height: 1, depth: 1 };
        let group_dims = MTLSize { width: thread_group_size, height: 1, depth: 1 };
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
    let mdev = match x.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("fused_metal_q8_0_matmul requires a Metal device"),
    };

    let x = if !x.is_contiguous() { x.contiguous()? } else { x.clone() };
    let weight_q8_bytes = if !weight_q8_bytes.is_contiguous() {
        weight_q8_bytes.contiguous()?
    } else {
        weight_q8_bytes.clone()
    };

    let x_dims = x.dims();
    if x_dims.is_empty() || x_dims[x_dims.len() - 1] != in_dim {
        candle_core::bail!(
            "fused_metal_q8_0_matmul shape mismatch: x={:?}, expected in_dim={}",
            x_dims,
            in_dim
        );
    }
    let m = x.elem_count() / in_dim;
    let mut out_dims = x_dims.to_vec();
    *out_dims.last_mut().unwrap() = out_dim;

    let dtype = x.dtype();
    let is_mv = m == 1;
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

        let w_buf = match &*w_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let x_buf = match &*x_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };
        let out_buf = match &*out_storage { candle_core::Storage::Metal(m) => m.buffer(), _ => unreachable!() };

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
            let group_dims = MTLSize { width: 8, height: 8, depth: 1 };
            enc_ref.dispatch_thread_groups(grid_dims, group_dims);
        } else {
            let grid_dims = MTLSize {
                width: m.div_ceil(32),
                height: out_dim.div_ceil(64),
                depth: 1,
            };
            let group_dims = MTLSize { width: 128, height: 1, depth: 1 };
            enc_ref.dispatch_thread_groups(grid_dims, group_dims);
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_kernels_respect_start_offset() -> Result<()> {
        if !candle_core::utils::metal_is_available() {
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
            row1_view.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
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
        assert_eq!(norm_view, norm_fresh, "fused_metal_rmsnorm ignored start_offset!");

        let geglu_view = fused_metal_geglu(&row1_view, &row1_view)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let geglu_fresh = fused_metal_geglu(&row1_fresh, &row1_fresh)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(geglu_view, geglu_fresh, "fused_metal_geglu ignored start_offset!");

        Ok(())
    }

    #[test]
    fn test_phase2_fused_kernels_bit_exact_parity() -> Result<()> {
        if !candle_core::utils::metal_is_available() {
            return Ok(());
        }
        let dev = Device::new_metal(0)?;
        let eps = 1e-6;

        // 1. Test fused_metal_rmsnorm_residual_add (with and without scalar) across wide dynamic range
        let x_vec: Vec<f32> = (0..(20 * 512)).map(|i| ((i as f32) * 0.137).sin() * 12.0).collect();
        let res_vec: Vec<f32> = (0..(20 * 512)).map(|i| ((i as f32) * 0.029).cos() * 8.0).collect();
        let w_vec: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.073).sin() * 3.0).collect();
        let x = Tensor::from_vec(x_vec, (2, 10, 512), &dev)?.to_dtype(DType::BF16)?;
        let res = Tensor::from_vec(res_vec, (2, 10, 512), &dev)?.to_dtype(DType::BF16)?;
        let w = Tensor::from_vec(w_vec, (512,), &dev)?.to_dtype(DType::BF16)?;
        let scalar = Tensor::from_vec(vec![0.87109375f32], (1,), &dev)?.to_dtype(DType::BF16)?;

        let ref_no_scale = res.add(&fused_metal_rmsnorm(&x, Some(&w), eps)?)?;
        let got_no_scale = fused_metal_rmsnorm_residual_add(&x, &res, Some(&w), None, eps)?;
        assert_eq!(
            ref_no_scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            got_no_scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            "fused_metal_rmsnorm_residual_add (no scale) mismatch!"
        );

        let ref_with_scale = ref_no_scale.broadcast_mul(&scalar)?;
        let got_with_scale = fused_metal_rmsnorm_residual_add(&x, &res, Some(&w), Some(&scalar), eps)?;
        assert_eq!(
            ref_with_scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            got_with_scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            "fused_metal_rmsnorm_residual_add (with scale) mismatch!"
        );

        // 2. Test fused_metal_qk_rmsnorm_rope_transpose & fused_metal_v_rmsnorm_transpose for d=256 and d=512
        for &d in &[256usize, 512usize] {
            let (b, s, h) = (2, 17, 4);
            let q_vec: Vec<f32> = (0..(b * s * h * d)).map(|i| ((i as f32) * 0.117).sin() * 15.0).collect();
            let qw_vec: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.053).cos() * 2.5).collect();
            let q_in = Tensor::from_vec(q_vec, (b, s, h * d), &dev)?.to_dtype(DType::BF16)?;
            let qw = Tensor::from_vec(qw_vec, (d,), &dev)?.to_dtype(DType::BF16)?;

            let rope = crate::rope::RotaryEmbedding::new(d, 10_000.0, &dev)?;
            let (cos, sin) = rope.get_cos_sin(s, &dev, DType::BF16)?;

            let q_unfused = q_in.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?;
            let q_unfused = fused_metal_rmsnorm(&q_unfused, Some(&qw), eps)?;
            let q_unfused = crate::rope::apply_rotary_pos_emb(&q_unfused, &cos, &sin)?;

            let q_fused = fused_metal_qk_rmsnorm_rope_transpose(&q_in, &qw, &cos, &sin, b, s, h, d, eps)?;
            assert_eq!(
                q_unfused.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                q_fused.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                "fused_metal_qk_rmsnorm_rope_transpose mismatch for d={d}!"
            );

            let v_unfused = q_in.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?;
            let v_unfused = fused_metal_rmsnorm(&v_unfused, None, eps)?;
            let v_fused = fused_metal_v_rmsnorm_transpose(&q_in, None, b, s, h, d, eps)?;
            assert_eq!(
                v_unfused.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                v_fused.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                "fused_metal_v_rmsnorm_transpose mismatch for d={d}!"
            );
        }

        // 3. Test fused_metal_ple_scale_rmsnorm_transpose with wide dynamic range
        let (b, s, l, d_ple) = (2, 5, 24, 512);
        let ple_vec: Vec<f32> = (0..(b * s * l * d_ple))
            .map(|i| ((i as f32) * 0.137).sin() * 12.0)
            .collect();
        let ple_in = Tensor::from_vec(ple_vec, (b, s, l * d_ple), &dev)?.to_dtype(DType::BF16)?;
        let scale = (d_ple as f64).powf(-0.5);

        let ple_fused = fused_metal_ple_scale_rmsnorm_transpose(&ple_in, &w, scale, eps, b, s, l, d_ple)?;
        let scaled_unfused = (&ple_in * scale)?.reshape((b, s, l, d_ple))?;
        for layer_idx in 0..l {
            let slice_unfused = scaled_unfused.narrow(2, layer_idx, 1)?.squeeze(2)?;
            let norm_unfused = fused_metal_rmsnorm(&slice_unfused, Some(&w), eps)?;
            let slice_fused = ple_fused.get(layer_idx)?;
            assert!(slice_fused.is_contiguous());
            let v_unfused = norm_unfused.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let v_fused = slice_fused.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(
                v_unfused, v_fused,
                "fused_metal_ple_scale_rmsnorm_transpose mismatch at layer {layer_idx}!"
            );
        }

        // 4. Test fused_metal_ple_geglu vs unfused gelu_tanh + broadcast_mul across [-20, 20]
        let x_wide_vec: Vec<f32> = (0..(20 * 512)).map(|i| ((i as f32) * 0.071).sin() * 20.0).collect();
        let x_wide = Tensor::from_vec(x_wide_vec, (2, 10, 512), &dev)?.to_dtype(DType::BF16)?;
        let ref_ple_geglu = crate::ple::gelu_tanh(&x_wide)?.broadcast_mul(&res)?;
        let got_ple_geglu = fused_metal_ple_geglu(&x_wide, &res)?;
        let v_ref_geglu = ref_ple_geglu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let v_got_geglu = got_ple_geglu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(v_ref_geglu.len(), v_got_geglu.len());
        for (idx, (&a, &b)) in v_ref_geglu.iter().zip(v_got_geglu.iter()).enumerate() {
            assert_eq!(a, b, "ple_geglu mismatch at idx {idx}: {a} vs {b}");
        }

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
            let cos_direct = emb.cos()?.reshape((1, 1, seq_test, d))?.to_dtype(DType::BF16)?;
            let sin_direct = emb.sin()?.reshape((1, 1, seq_test, d))?.to_dtype(DType::BF16)?;
            assert_eq!(
                cos_cached.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                cos_direct.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                "cos_cached vs cos_direct mismatch for d={d}!"
            );
            assert_eq!(
                sin_cached.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                sin_direct.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                "sin_cached vs sin_direct mismatch for d={d}!"
            );
        }

        Ok(())
    }

    fn assert_vec_eq(a: &[f32], b: &[f32], label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: len mismatch");
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            if va != vb {
                panic!("{label} mismatch at idx {i}: ref={va} vs got={vb} (diff={})", (va - vb).abs());
            }
        }
    }

    #[test]
    fn test_phase3_fused_kernels_bit_exact_parity() -> Result<()> {
        if !candle_core::utils::metal_is_available() {
            return Ok(());
        }
        let dev = Device::new_metal(0)?;
        let eps = 1e-6;

        // 1. Test fused_metal_geglu_gate_up vs fused_metal_geglu
        let (b, s, d_mlp) = (2, 13, 2048);
        let gate_vec: Vec<f32> = (0..(b * s * d_mlp)).map(|i| ((i as f32) * 0.091).sin() * 14.0).collect();
        let up_vec: Vec<f32> = (0..(b * s * d_mlp)).map(|i| ((i as f32) * 0.047).cos() * 9.0).collect();
        let gate = Tensor::from_vec(gate_vec, (b, s, d_mlp), &dev)?.to_dtype(DType::BF16)?;
        let up = Tensor::from_vec(up_vec, (b, s, d_mlp), &dev)?.to_dtype(DType::BF16)?;
        let gate_up = Tensor::cat(&[&gate, &up], candle_core::D::Minus1)?;

        let ref_geglu = fused_metal_geglu(&gate, &up)?;
        let got_geglu = fused_metal_geglu_gate_up(&gate_up)?;
        assert_vec_eq(
            &ref_geglu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            &got_geglu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            "fused_metal_geglu_gate_up",
        );

        // 2. Test fused_metal_qkv_rmsnorm_rope_transpose vs separate Q/K/V fused kernels
        for &(h_q, h_kv, d) in &[(4usize, 2usize, 256usize), (4usize, 1usize, 512usize)] {
            let (b, s) = (2, 19);
            let q_vec: Vec<f32> = (0..(b * s * h_q * d)).map(|i| ((i as f32) * 0.113).sin() * 11.0).collect();
            let k_vec: Vec<f32> = (0..(b * s * h_kv * d)).map(|i| ((i as f32) * 0.079).cos() * 10.0).collect();
            let v_vec: Vec<f32> = (0..(b * s * h_kv * d)).map(|i| ((i as f32) * 0.053).sin() * 8.0).collect();
            let qw_vec: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.037).cos() * 2.0 + 1.0).collect();
            let kw_vec: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.061).sin() * 2.0 + 1.0).collect();

            let q_in = Tensor::from_vec(q_vec, (b, s, h_q * d), &dev)?.to_dtype(DType::BF16)?;
            let k_in = Tensor::from_vec(k_vec, (b, s, h_kv * d), &dev)?.to_dtype(DType::BF16)?;
            let v_in = Tensor::from_vec(v_vec, (b, s, h_kv * d), &dev)?.to_dtype(DType::BF16)?;
            let qkv_in = Tensor::cat(&[&q_in, &k_in, &v_in], candle_core::D::Minus1)?;

            let qw = Tensor::from_vec(qw_vec, (d,), &dev)?.to_dtype(DType::BF16)?;
            let kw = Tensor::from_vec(kw_vec, (d,), &dev)?.to_dtype(DType::BF16)?;
            let rope = crate::rope::RotaryEmbedding::new(d, 10_000.0, &dev)?;
            let (cos, sin) = rope.get_cos_sin(s, &dev, DType::BF16)?;

            let q_ref = fused_metal_qk_rmsnorm_rope_transpose(&q_in, &qw, &cos, &sin, b, s, h_q, d, eps)?;
            let k_ref = fused_metal_qk_rmsnorm_rope_transpose(&k_in, &kw, &cos, &sin, b, s, h_kv, d, eps)?;
            let v_ref = fused_metal_v_rmsnorm_transpose(&v_in, None, b, s, h_kv, d, eps)?;

            let (q_got, k_got, v_got) = fused_metal_qkv_rmsnorm_rope_transpose(
                &qkv_in, &qw, &kw, None, &cos, &sin, b, s, h_q, h_kv, d, eps, eps, eps,
            )?;

            assert_vec_eq(
                &q_ref.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &q_got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &format!("qkv_rmsnorm_rope_transpose Q d={d}"),
            );
            assert_vec_eq(
                &k_ref.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &k_got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &format!("qkv_rmsnorm_rope_transpose K d={d}"),
            );
            assert_vec_eq(
                &v_ref.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &v_got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &format!("qkv_rmsnorm_rope_transpose V d={d}"),
            );
        }

        // 3. Test fused_metal_vision_qkv_rmsnorm_rope2d_transpose vs unfused VisionAttention path
        {
            let (b, s, h, d) = (1, 36, 12, 64);
            let q_vec: Vec<f32> = (0..(b * s * h * d)).map(|i| ((i as f32) * 0.083).sin() * 9.0).collect();
            let k_vec: Vec<f32> = (0..(b * s * h * d)).map(|i| ((i as f32) * 0.067).cos() * 7.0).collect();
            let v_vec: Vec<f32> = (0..(b * s * h * d)).map(|i| ((i as f32) * 0.041).sin() * 6.0).collect();
            let qw_vec: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.031).cos() * 1.5 + 1.0).collect();
            let kw_vec: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.059).sin() * 1.5 + 1.0).collect();

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

            assert_vec_eq(
                &q_ref.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &q_got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                "vision_qkv_rmsnorm_rope2d_transpose Q",
            );
            assert_vec_eq(
                &k_ref.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &k_got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                "vision_qkv_rmsnorm_rope2d_transpose K",
            );
            assert_vec_eq(
                &v_ref.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &v_got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                "vision_qkv_rmsnorm_rope2d_transpose V",
            );
        }

        // 4. Test fused_metal_audio_glu_causal_dwconv5 vs unfused AudioLightConv1d GLU + 5-tap conv
        {
            let (b, s, c) = (1, 25, 1024);
            let pw1_vec: Vec<f32> = (0..(b * s * 2 * c)).map(|i| ((i as f32) * 0.043).sin() * 5.0).collect();
            let dw_vec: Vec<f32> = (0..(c * 5)).map(|i| ((i as f32) * 0.071).cos() * 0.5).collect();
            let pw1 = Tensor::from_vec(pw1_vec, (b, s, 2 * c), &dev)?.to_dtype(DType::BF16)?;
            let dw_w = Tensor::from_vec(dw_vec, (1, c, 5), &dev)?.to_dtype(DType::BF16)?;

            let half0 = pw1.narrow(candle_core::D::Minus1, 0, c)?;
            let half1 = pw1.narrow(candle_core::D::Minus1, c, c)?;
            let sig = candle_nn::ops::sigmoid(&half1)?;
            let glu_out = (half0 * sig)?;
            let dw_in = glu_out.transpose(1, 2)?.contiguous()?.pad_with_zeros(candle_core::D::Minus1, 4, 0)?;
            let mut dw_ref = dw_in.narrow(2, 0, s)?.broadcast_mul(&dw_w.narrow(2, 0, 1)?)?;
            for k in 1..5 {
                let term = dw_in.narrow(2, k, s)?.broadcast_mul(&dw_w.narrow(2, k, 1)?)?;
                dw_ref = (dw_ref + term)?;
            }
            let dw_ref = dw_ref.transpose(1, 2)?.contiguous()?;

            let dw_got = fused_metal_audio_glu_causal_dwconv5(&pw1, &dw_w)?;
            assert_vec_eq(
                &dw_ref.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &dw_got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                "fused_metal_audio_glu_causal_dwconv5",
            );
        }

        // 5. Test fused_metal_q8_0_matmul (both MV m=1 and MM m=19) vs Candle QMatMul F32->BF16
        {
            let (out_dim, in_dim) = (128, 256);
            let w_f32_vec: Vec<f32> = (0..(out_dim * in_dim)).map(|i| ((i as f32) * 0.031).sin() * 1.5).collect();
            let w_cpu = Tensor::from_vec(w_f32_vec, (out_dim, in_dim), &Device::Cpu)?;
            let qt_cpu = candle_core::quantized::QTensor::quantize(&w_cpu, candle_core::quantized::GgmlDType::Q8_0)?;
            let raw_bytes = qt_cpu.data()?.into_owned();
            let row_bytes = (in_dim / 32) * 34;
            let q8_bytes_metal = Tensor::from_slice(&raw_bytes, (out_dim, row_bytes), &dev)?;

            let qt_metal = candle_core::quantized::QTensor::quantize_onto(
                &w_cpu,
                candle_core::quantized::GgmlDType::Q8_0,
                &dev,
            )?;
            let qm_ref = candle_core::quantized::QMatMul::from_qtensor(qt_metal)?;

            for &m in &[1usize, 19usize, 64usize] {
                let x_vec: Vec<f32> = (0..(m * in_dim)).map(|i| ((i as f32) * 0.059).cos() * 3.0).collect();
                let x_bf16 = Tensor::from_vec(x_vec, (1, m, in_dim), &dev)?.to_dtype(DType::BF16)?;

                let ref_out = candle_core::Module::forward(&qm_ref, &x_bf16.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)?;
                let got_out = fused_metal_q8_0_matmul(&q8_bytes_metal, &x_bf16, out_dim, in_dim)?;

                let ref_vec = ref_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let got_vec = got_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                if m == 1 {
                    assert_vec_eq(&ref_vec, &got_vec, "fused_metal_q8_0_matmul m=1");
                } else {
                    // On M5 Neural Accelerators (mpp::tensor_ops::matmul2d), bfloat x half -> float
                    // vs legacy float x half -> float can differ by at most 1 BF16 ULP (~0.78% rel or 5e-4 abs).
                    for (i, (&va, &vb)) in ref_vec.iter().zip(got_vec.iter()).enumerate() {
                        let diff = (va - vb).abs();
                        assert!(
                            diff <= 1e-3,
                            "fused_metal_q8_0_matmul m={m} mismatch at idx {i}: ref={va} vs got={vb} (diff={diff})"
                        );
                    }
                }
            }
        }

        Ok(())
    }
}


