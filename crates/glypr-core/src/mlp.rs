use candle_core::{Result, Tensor, D};
use crate::linear::LinearOp;
use crate::ple::gelu_tanh;

#[derive(Debug, Clone)]
pub struct MLP {
    pub gate_proj: LinearOp,
    pub up_proj: LinearOp,
    pub down_proj: LinearOp,
    pub fused_gate_up: Option<LinearOp>,
}

impl MLP {
    pub fn new(mut gate_proj: LinearOp, mut up_proj: LinearOp, down_proj: LinearOp) -> Self {
        let fused_gate_up = if let Some((fused, mut views)) = LinearOp::try_concat(&[&gate_proj, &up_proj]) {
            up_proj = views.pop().unwrap();
            gate_proj = views.pop().unwrap();
            Some(fused)
        } else {
            None
        };
        Self {
            gate_proj,
            up_proj,
            down_proj,
            fused_gate_up,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(ref fused) = self.fused_gate_up {
            let gate_up = fused.forward(x)?;

            #[cfg(feature = "metal")]
            if gate_up.device().is_metal() && gate_up.dtype() == candle_core::DType::BF16 {
                let gated = crate::metal_kernels::fused_metal_geglu_gate_up(&gate_up)?;
                return self.down_proj.forward(&gated);
            }

            let d = gate_up.dim(D::Minus1)? / 2;
            let gate = gate_up.narrow(D::Minus1, 0, d)?.contiguous()?;
            let up = gate_up.narrow(D::Minus1, d, d)?.contiguous()?;
            if gate.device().is_cpu() {
                let gated = crate::cpu_simd::cpu_geglu(&gate, &up)?;
                return self.down_proj.forward(&gated);
            }
            let gate_act = gelu_tanh(&gate)?;
            let gated = gate_act.broadcast_mul(&up)?;
            return self.down_proj.forward(&gated);
        }

        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        #[cfg(feature = "metal")]
        if gate.device().is_metal() && gate.dtype() == candle_core::DType::BF16 {
            let gated = crate::metal_kernels::fused_metal_geglu(&gate, &up)?;
            return self.down_proj.forward(&gated);
        }

        if gate.device().is_cpu() {
            let gated = crate::cpu_simd::cpu_geglu(&gate, &up)?;
            return self.down_proj.forward(&gated);
        }

        let gate_act = gelu_tanh(&gate)?;
        let gated = gate_act.broadcast_mul(&up)?;
        self.down_proj.forward(&gated)
    }
}

