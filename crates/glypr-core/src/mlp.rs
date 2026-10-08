use candle_core::{Result, Tensor};
use crate::linear::LinearOp;
use crate::ple::gelu_tanh;

#[derive(Debug, Clone)]
pub struct MLP {
    pub gate_proj: LinearOp,
    pub up_proj: LinearOp,
    pub down_proj: LinearOp,
}

impl MLP {
    pub fn new(gate_proj: LinearOp, up_proj: LinearOp, down_proj: LinearOp) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
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
