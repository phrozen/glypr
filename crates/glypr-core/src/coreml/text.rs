use crate::coreml::CoreMLComputeUnits;
use crate::task::Task;
use crate::tokenizer::GlyprTokenizer;
use anyhow::{Context, Result, anyhow, bail};
use candle_core::{Device, Tensor};
use coreml_native::{AsMultiArray, BorrowedTensor, Model};
use std::path::Path;

/// CoreML-accelerated Text Backbone targeting the Apple Neural Engine (ANE) and GPU.
///
/// Designed for low-power, low-latency resident document and search query embedding
/// with a static sequence length (typically 512 tokens).
pub struct CoreMLTextBackbone {
    model: Model,
    seq_len: usize,
    output_name: String,
    compute_units: CoreMLComputeUnits,
}

impl CoreMLTextBackbone {
    /// Load a compiled CoreML text backbone model (`.mlmodelc`).
    pub fn from_path<P: AsRef<Path>>(path: P, compute_units: CoreMLComputeUnits) -> Result<Self> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            bail!("CoreML text model not found at {}", path_ref.display());
        }

        let model = Model::load(path_ref, compute_units.into()).with_context(|| {
            format!("Failed to load CoreML text model at {}", path_ref.display())
        })?;

        let inputs = model.inputs();
        let id_input = inputs
            .iter()
            .find(|inp| inp.name() == "input_ids")
            .ok_or_else(|| anyhow!("Model does not contain 'input_ids' input feature"))?;

        let seq_len = id_input
            .shape()
            .and_then(|s| if s.len() >= 2 { Some(s[1]) } else { None })
            .unwrap_or(512);

        let outputs = model.outputs();
        let out_desc = outputs
            .first()
            .ok_or_else(|| anyhow!("Model contains no output features"))?;

        let output_name = out_desc.name().to_string();

        Ok(Self {
            model,
            seq_len,
            output_name,
            compute_units,
        })
    }

    /// Embed a sequence of token IDs into a 768-dimensional L2-normalized embedding vector.
    ///
    /// If `token_ids.len() > self.seq_len`, fails immediately with a descriptive error.
    /// Sequences shorter than `self.seq_len` are automatically right-padded with zero mask.
    pub fn embed_token_ids(&self, token_ids: &[i32]) -> Result<Vec<f32>> {
        let num_tokens = token_ids.len();
        if num_tokens > self.seq_len {
            bail!(
                "Token sequence length ({}) exceeds CoreML text backbone static limit ({})",
                num_tokens,
                self.seq_len
            );
        }

        let mut padded_ids = vec![0i32; self.seq_len];
        let mut padded_mask = vec![0.0f32; self.seq_len];

        padded_ids[..num_tokens].copy_from_slice(token_ids);
        padded_mask[..num_tokens].fill(1.0);

        let t_ids = BorrowedTensor::from_i32(&padded_ids, &[1, self.seq_len])
            .context("Failed to create BorrowedTensor for input_ids")?;
        let t_mask = BorrowedTensor::from_f32(&padded_mask, &[1, self.seq_len])
            .context("Failed to create BorrowedTensor for attention_mask")?;

        let inputs: &[(&str, &dyn AsMultiArray)] =
            &[("input_ids", &t_ids), ("attention_mask", &t_mask)];

        let prediction = self
            .model
            .predict(inputs)
            .context("CoreML prediction failed during text backbone forward pass")?;

        let (mut emb_data, shape) = prediction
            .get_f32(&self.output_name)
            .with_context(|| format!("Failed to read output feature '{}'", self.output_name))?;

        if emb_data.len() != 768 {
            bail!(
                "Unexpected output length {} (expected 768, shape: {:?})",
                emb_data.len(),
                shape
            );
        }

        // Defensive normalization check
        let norm_sq: f32 = emb_data.iter().map(|&x| x * x).sum();
        if norm_sq.is_nan() || norm_sq <= 0.0 {
            bail!("CoreML text embedding returned invalid or NaN vector");
        }

        // Ensure bit-exact unit length
        let inv_norm = 1.0 / norm_sq.sqrt();
        for val in &mut emb_data {
            *val *= inv_norm;
        }

        Ok(emb_data)
    }

    /// Embed a text string, optionally applying the prompt format for the given task.
    pub fn embed_text(
        &self,
        text: &str,
        tokenizer: &GlyprTokenizer,
        task: Option<Task>,
    ) -> Result<Vec<f32>> {
        let formatted = if let Some(t) = task {
            t.format_text(text)
        } else {
            text.to_string()
        };

        let encoding = tokenizer.encode(&formatted)?;
        let token_ids: Vec<i32> = encoding.iter().map(|&id| id as i32).collect();
        self.embed_token_ids(&token_ids)
    }

    /// Embed a text string directly into a Candle `Tensor` `[1, 768]` on the specified device.
    pub fn embed_to_tensor(
        &self,
        text: &str,
        tokenizer: &GlyprTokenizer,
        task: Option<Task>,
        device: &Device,
    ) -> Result<Tensor> {
        let vec = self.embed_text(text, tokenizer, task)?;
        Ok(Tensor::from_vec(vec, (1, 768), device)?)
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn compute_units(&self) -> CoreMLComputeUnits {
        self.compute_units
    }
}
