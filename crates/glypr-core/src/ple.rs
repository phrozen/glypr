use candle_core::{Result, Tensor};
use crate::linear::LinearOp;
use crate::norm::RMSNorm;

/// Exact PyTorch gelu_pytorch_tanh approximation:
/// 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    let sqrt_2_over_pi = 0.7978845608028654f64;
    let coeff = 0.044715f64;

    let x3 = x.sqr()?.broadcast_mul(x)?;
    let scaled_x3 = (x3 * coeff)?;
    let sum = x.add(&scaled_x3)?;
    let inner = (sum * sqrt_2_over_pi)?;
    let tanh = inner.tanh()?;
    let one_plus_tanh = (tanh + 1.0)?;
    let half_x = (x * 0.5)?;
    half_x.broadcast_mul(&one_plus_tanh)
}

#[derive(Debug, Clone)]
pub struct TextPLE {
    pub projection: LinearOp,
    pub scale: f64,
    pub norm: RMSNorm,
    pub num_layers: usize,
    pub hidden_size_per_layer_input: usize,
}

impl TextPLE {
    pub fn new(
        projection: LinearOp,
        scale: f64,
        norm: RMSNorm,
        num_layers: usize,
        hidden_size_per_layer_input: usize,
    ) -> Self {
        Self {
            projection,
            scale,
            norm,
            num_layers,
            hidden_size_per_layer_input,
        }
    }

    /// Project and reshape without computing float32 RMSNorm on the full 4D tensor.
    /// Returns unnormalized projections [batch, seq_len, num_layers, hidden_size_per_layer_input].
    pub fn project(&self, inputs_embeds: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = inputs_embeds.dims3()?;
        let proj = self.projection.forward(inputs_embeds)?;
        let scaled_proj = (proj * self.scale)?;
        scaled_proj.reshape((
            batch,
            seq_len,
            self.num_layers,
            self.hidden_size_per_layer_input,
        ))
    }

    /// Forward pass producing [batch, seq_len, num_layers, hidden_size_per_layer_input]
    pub fn forward(&self, inputs_embeds: &Tensor) -> Result<Tensor> {
        let reshaped = self.project(inputs_embeds)?;
        self.norm.forward(&reshaped)
    }
}

#[derive(Debug, Clone)]
pub struct TextPLEBlock {
    pub gate: LinearOp,
    pub projection: LinearOp,
    pub norm: RMSNorm,
}

impl TextPLEBlock {
    pub fn new(gate: LinearOp, projection: LinearOp, norm: RMSNorm) -> Self {
        Self { gate, projection, norm }
    }

    /// Forward pass mixing the layer's slice of per-layer inputs
    /// hidden_states: [batch, seq_len, hidden_size]
    /// per_layer_input: [batch, seq_len, hidden_size]
    pub fn forward(&self, hidden_states: &Tensor, per_layer_input: &Tensor) -> Result<Tensor> {
        let h = self.gate.forward(hidden_states)?;
        let h = gelu_tanh(&h)?;
        let h = h.broadcast_mul(per_layer_input)?;
        let h = self.projection.forward(&h)?;
        let normed = self.norm.forward(&h)?;
        hidden_states.add(&normed)
    }
}
