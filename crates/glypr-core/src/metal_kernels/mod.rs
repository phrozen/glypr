use candle_core::{DType, Device, Result, Tensor};
use candle_metal_kernels::metal::{ComputePipeline, Device as MtlDevice};
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

const FUSED_OPS_SOURCE: &str = include_str!("fused_ops.metal");

static RMSNORM_PIPELINES: LazyLock<RwLock<HashMap<usize, ComputePipeline>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

static GEGLU_PIPELINES: LazyLock<RwLock<HashMap<usize, ComputePipeline>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn get_or_compile_rmsnorm_pipeline(dev: &MtlDevice) -> Result<ComputePipeline> {
    let dev_key = dev.as_ref() as *const _ as usize;
    {
        let cache = RMSNORM_PIPELINES.read().unwrap();
        if let Some(p) = cache.get(&dev_key) {
            return Ok(p.clone());
        }
    }

    let library = dev
        .new_library_with_source(FUSED_OPS_SOURCE, None)
        .map_err(|e| candle_core::Error::Metal(format!("Failed to compile fused_ops.metal: {:?}", e).into()))?;

    let function = library
        .get_function("rms_norm_bf16", None)
        .map_err(|e| candle_core::Error::Metal(format!("Function rms_norm_bf16 not found: {:?}", e).into()))?;

    let pipeline = dev
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| candle_core::Error::Metal(format!("Failed to create rmsnorm pipeline: {:?}", e).into()))?;

    let mut cache = RMSNORM_PIPELINES.write().unwrap();
    cache.insert(dev_key, pipeline.clone());
    Ok(pipeline)
}

fn get_or_compile_geglu_pipeline(dev: &MtlDevice) -> Result<ComputePipeline> {
    let dev_key = dev.as_ref() as *const _ as usize;
    {
        let cache = GEGLU_PIPELINES.read().unwrap();
        if let Some(p) = cache.get(&dev_key) {
            return Ok(p.clone());
        }
    }

    let library = dev
        .new_library_with_source(FUSED_OPS_SOURCE, None)
        .map_err(|e| candle_core::Error::Metal(format!("Failed to compile fused_ops.metal: {:?}", e).into()))?;

    let function = library
        .get_function("geglu_bf16", None)
        .map_err(|e| candle_core::Error::Metal(format!("Function geglu_bf16 not found: {:?}", e).into()))?;

    let pipeline = dev
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| candle_core::Error::Metal(format!("Failed to create geglu pipeline: {:?}", e).into()))?;

    let mut cache = GEGLU_PIPELINES.write().unwrap();
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

    let pipeline = get_or_compile_rmsnorm_pipeline(mdev.metal_device())?;
    let out = Tensor::zeros(orig_shape.clone(), DType::BF16, x.device())?;

    let has_weight: i32 = if weight_ref.is_some() { 1 } else { 0 };
    let d_i32 = d as i32;
    let eps_f32 = eps as f32;

    {
        let (x_storage, _) = x.storage_and_layout();
        let (out_storage, _) = out.storage_and_layout();

        let x_buf = match &*x_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };
        let out_buf = match &*out_storage {
            candle_core::Storage::Metal(m) => m.buffer(),
            _ => unreachable!(),
        };

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(x_buf), 0);

        if let Some(w) = weight_ref {
            let (w_storage, _) = w.storage_and_layout();
            let w_buf = match &*w_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            enc_ref.set_input_buffer(1, Some(w_buf), 0);
        } else {
            // Null or placeholder buffer
            enc_ref.set_input_buffer(1, Some(x_buf), 0);
        }

        enc_ref.set_output_buffer(2, Some(out_buf), 0);
        enc_ref.set_bytes(3, &eps_f32);
        enc_ref.set_bytes(4, &d_i32);
        enc_ref.set_bytes(5, &has_weight);

        let grid_dims = MTLSize { width: num_rows, height: 1, depth: 1 };
        let group_dims = MTLSize { width: 32, height: 1, depth: 1 };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}

/// Fused Metal GeGLU evaluating exact PyTorch tanh GELU approximation in a single streaming pass.
pub fn fused_metal_geglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let mdev = match gate.device() {
        Device::Metal(m) => m,
        _ => candle_core::bail!("fused_metal_geglu requires a Metal device"),
    };

    let gate = if !gate.is_contiguous() { gate.contiguous()? } else { gate.clone() };
    let up = if !up.is_contiguous() { up.contiguous()? } else { up.clone() };

    let total_elements = gate.elem_count();
    let pipeline = get_or_compile_geglu_pipeline(mdev.metal_device())?;
    let out = Tensor::zeros(gate.shape().clone(), DType::BF16, gate.device())?;
    let total_i32 = total_elements as i32;

    {
        let (gate_storage, _) = gate.storage_and_layout();
        let (up_storage, _) = up.storage_and_layout();
        let (out_storage, _) = out.storage_and_layout();

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

        let encoder = mdev.command_encoder()?;
        let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc_ref.set_compute_pipeline_state(&pipeline);

        enc_ref.set_input_buffer(0, Some(gate_buf), 0);
        enc_ref.set_input_buffer(1, Some(up_buf), 0);
        enc_ref.set_output_buffer(2, Some(out_buf), 0);
        enc_ref.set_bytes(3, &total_i32);

        let thread_group_size = 256;
        let num_groups = (total_elements + thread_group_size - 1) / thread_group_size;
        let grid_dims = MTLSize { width: num_groups, height: 1, depth: 1 };
        let group_dims = MTLSize { width: thread_group_size, height: 1, depth: 1 };
        enc_ref.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(out)
}
