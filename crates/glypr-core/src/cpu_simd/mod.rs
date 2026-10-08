use candle_core::{Result, Tensor};

pub mod rmsnorm;
pub mod geglu;

pub use rmsnorm::*;
pub use geglu::*;

/// Fast CPU RMSNorm using portable SIMD (NEON / AVX2 / AVX-512) and zero intermediate allocations.
pub fn cpu_rmsnorm(x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let weight_f32 = match weight {
        Some(w) => {
            let w_f32 = if w.dtype() != candle_core::DType::F32 {
                w.to_dtype(candle_core::DType::F32)?
            } else {
                w.clone()
            };
            Some(w_f32.flatten_all()?.to_vec1::<f32>()?)
        }
        None => None,
    };
    let op = CpuRMSNormOp {
        weight_f32,
        eps: eps as f32,
    };
    let x_cont = if !x.is_contiguous() { x.contiguous()? } else { x.clone() };
    x_cont.apply_op1_no_bwd(&op)
}

/// Fast CPU GeGLU using in-register tanh activation and single-pass streaming with zero intermediate allocations.
pub fn cpu_geglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let gate_cont = if !gate.is_contiguous() { gate.contiguous()? } else { gate.clone() };
    let up_cont = if !up.is_contiguous() { up.contiguous()? } else { up.clone() };
    gate_cont.apply_op2_no_bwd(&up_cont, &CpuGeGLUOp)
}
