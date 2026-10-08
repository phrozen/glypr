use candle_core::{DType, Result, Tensor};

#[derive(Debug, Clone)]
pub struct RMSNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl RMSNorm {
    pub fn new(weight: Option<Tensor>, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if x.device().is_metal() && x.dtype() == DType::BF16 {
            return crate::metal_kernels::fused_metal_rmsnorm(x, self.weight.as_ref(), self.eps);
        }

        if x.device().is_cpu() {
            return crate::cpu_simd::cpu_rmsnorm(x, self.weight.as_ref(), self.eps);
        }

        let orig_dtype = x.dtype();
        // Compute in float32 for stability matching Python reference
        let x_f32 = if orig_dtype != DType::F32 {
            x.to_dtype(DType::F32)?
        } else {
            x.clone()
        };

        // mean_squared = mean(x^2, dim=-1, keepdim=True) + eps
        let sq = x_f32.sqr()?;
        let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
        let mean_sq_eps = (mean_sq + self.eps)?;
        let rsqrt = mean_sq_eps.powf(-0.5)?;
        let normed = x_f32.broadcast_mul(&rsqrt)?;

        let out_f32 = if let Some(ref w) = self.weight {
            let w_f32 = if w.dtype() != DType::F32 {
                w.to_dtype(DType::F32)?
            } else {
                w.clone()
            };
            normed.broadcast_mul(&w_f32)?
        } else {
            normed
        };

        if orig_dtype != DType::F32 {
            out_f32.to_dtype(orig_dtype)
        } else {
            Ok(out_f32)
        }
    }
}
