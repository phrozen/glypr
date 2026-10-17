use candle_core::{DType, Device, Module, Result, Tensor};
use candle_core::quantized::QMatMul;
use candle_nn::Linear;

#[derive(Debug, Clone)]
pub enum LinearOp {
    Standard(Linear),
    Quantized(QMatMul),
    #[cfg(feature = "metal")]
    MetalQ8_0 {
        weight_q8_bytes: Tensor,
        out_dim: usize,
        in_dim: usize,
    },
}

impl LinearOp {
    pub fn new_standard(linear: Linear) -> Self {
        Self::Standard(linear)
    }

    pub fn new_quantized(qmatmul: QMatMul) -> Self {
        Self::Quantized(qmatmul)
    }

    #[cfg(feature = "metal")]
    pub fn new_metal_q8_0(weight_q8_bytes: Tensor, out_dim: usize, in_dim: usize) -> Self {
        Self::MetalQ8_0 {
            weight_q8_bytes,
            out_dim,
            in_dim,
        }
    }

    /// Load a linear operator from GGUF, using native zero-cast `MetalQ8_0` when on Metal with `Q8_0` weights.
    pub fn from_gguf_tensor<R: std::io::Read + std::io::Seek>(
        content: &candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        name: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let info = content.tensor_infos.get(name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Missing tensor info for {}", name))
        })?;
        match info.ggml_dtype {
            candle_core::quantized::GgmlDType::F32
            | candle_core::quantized::GgmlDType::F16
            | candle_core::quantized::GgmlDType::BF16 => {
                let qt = content.tensor(reader, name, device)?;
                let t = qt.dequantize(device)?.to_dtype(dtype)?;
                Ok(Self::new_standard(Linear::new(t, None)))
            }
            #[cfg(feature = "metal")]
            candle_core::quantized::GgmlDType::Q8_0
                if device.is_metal() && info.shape.rank() == 2 =>
            {
                let (out_dim, in_dim) = info.shape.dims2()?;
                if in_dim % 32 == 0 {
                    let qt_cpu = content.tensor(reader, name, &Device::Cpu)?;
                    let raw_bytes = qt_cpu.data()?;
                    let row_bytes = (in_dim / 32) * 34;
                    let weight_q8_bytes =
                        Tensor::from_slice(&raw_bytes, (out_dim, row_bytes), device)?;
                    return Ok(Self::new_metal_q8_0(weight_q8_bytes, out_dim, in_dim));
                }
                let qt = content.tensor(reader, name, device)?;
                let qm = QMatMul::from_qtensor(qt)?;
                Ok(Self::new_quantized(qm))
            }
            _ => {
                let qt = content.tensor(reader, name, device)?;
                let qm = QMatMul::from_qtensor(qt)?;
                Ok(Self::new_quantized(qm))
            }
        }
    }

    /// Concatenate multiple compatible bias-free `LinearOp`s along `out_dim` (dim 0),
    /// returning `(fused_op, zero_copy_narrow_views)` so all views share the single fused buffer.
    pub fn try_concat(ops: &[&LinearOp]) -> Option<(LinearOp, Vec<LinearOp>)> {
        if ops.len() < 2 {
            return None;
        }
        if ops.iter().all(|op| matches!(op, Self::Standard(l) if l.bias().is_none())) {
            let mut weights = Vec::with_capacity(ops.len());
            let mut row_counts = Vec::with_capacity(ops.len());
            for op in ops {
                if let Self::Standard(l) = op {
                    let w = l.weight();
                    row_counts.push(w.dim(0).ok()?);
                    weights.push(w);
                }
            }
            let fused_w = Tensor::cat(&weights, 0).ok()?;
            let mut views = Vec::with_capacity(ops.len());
            let mut offset = 0usize;
            for &rows in &row_counts {
                let slice = fused_w.narrow(0, offset, rows).ok()?;
                views.push(Self::new_standard(Linear::new(slice, None)));
                offset += rows;
            }
            let fused_op = Self::new_standard(Linear::new(fused_w, None));
            return Some((fused_op, views));
        }

        #[cfg(feature = "metal")]
        if ops.iter().all(|op| matches!(op, Self::MetalQ8_0 { .. })) {
            let mut byte_tensors = Vec::with_capacity(ops.len());
            let mut row_counts = Vec::with_capacity(ops.len());
            let mut common_in_dim = None;
            for op in ops {
                if let Self::MetalQ8_0 {
                    weight_q8_bytes,
                    out_dim,
                    in_dim,
                } = op
                {
                    if let Some(expected_in) = common_in_dim {
                        if *in_dim != expected_in {
                            return None;
                        }
                    } else {
                        common_in_dim = Some(*in_dim);
                    }
                    row_counts.push(*out_dim);
                    byte_tensors.push(weight_q8_bytes);
                }
            }
            let in_dim = common_in_dim?;
            let fused_bytes = Tensor::cat(&byte_tensors, 0).ok()?;
            let mut views = Vec::with_capacity(ops.len());
            let mut offset = 0usize;
            for &rows in &row_counts {
                let slice = fused_bytes.narrow(0, offset, rows).ok()?;
                views.push(Self::new_metal_q8_0(slice, rows, in_dim));
                offset += rows;
            }
            let fused_op = Self::new_metal_q8_0(fused_bytes, offset, in_dim);
            return Some((fused_op, views));
        }

        None
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(l) => l.forward(x),
            Self::Quantized(q) => {
                let orig_dtype = x.dtype();
                if orig_dtype != DType::F32 {
                    let x_f32 = x.to_dtype(DType::F32)?;
                    let out_f32 = q.forward(&x_f32)?;
                    out_f32.to_dtype(orig_dtype)
                } else {
                    q.forward(x)
                }
            }
            #[cfg(feature = "metal")]
            Self::MetalQ8_0 {
                weight_q8_bytes,
                out_dim,
                in_dim,
            } => crate::metal_kernels::fused_metal_q8_0_matmul(
                weight_q8_bytes,
                x,
                *out_dim,
                *in_dim,
            ),
        }
    }
}

