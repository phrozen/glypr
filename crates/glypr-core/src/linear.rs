use candle_core::{DType, Module, Result, Tensor};
use candle_core::quantized::QMatMul;
use candle_nn::Linear;

#[derive(Debug, Clone)]
pub enum LinearOp {
    Standard(Linear),
    Quantized(QMatMul),
}

impl LinearOp {
    pub fn new_standard(linear: Linear) -> Self {
        Self::Standard(linear)
    }

    pub fn new_quantized(qmatmul: QMatMul) -> Self {
        Self::Quantized(qmatmul)
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
        }
    }
}
