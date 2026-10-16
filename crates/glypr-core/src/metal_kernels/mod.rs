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
}

