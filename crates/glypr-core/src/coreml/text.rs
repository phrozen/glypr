use crate::coreml::CoreMLComputeUnits;
use crate::error::{GlyprError, GlyprResult};
use crate::task::Task;
use crate::tokenizer::GlyprTokenizer;
use crate::util::truncate_mrl;
use anyhow::{Context, Result, anyhow, bail};
use candle_core::{Device, Tensor};
use coreml_native::{AsMultiArray, BorrowedTensor, Model};
use std::path::{Path, PathBuf};

const TEXT_EMBEDDING_DIM: usize = 768;

struct TextModelEntry {
    #[cfg(not(test))]
    model: Model,
    #[cfg(test)]
    model: Option<Model>,
    model_name: String,
    seq_len: usize,
    output_name: String,
}

/// CoreML-accelerated Text Backbone targeting the Apple Neural Engine (ANE) and GPU.
///
/// Supports one or more compiled static-sequence-length text buckets (for example `s128`,
/// `s512`, `s1024`), automatically selecting the smallest loaded bucket that fits each
/// tokenized input.
pub struct CoreMLTextBackbone {
    buckets: Vec<TextModelEntry>,
    compute_units: CoreMLComputeUnits,
}

fn validate_text_mrl_dim(truncate_dim: Option<usize>) -> GlyprResult<Option<usize>> {
    match truncate_dim {
        None => Ok(None),
        Some(0) => Err(GlyprError::InvalidMrlDimension {
            dim: 0,
            max: TEXT_EMBEDDING_DIM,
        }),
        Some(d) if d > TEXT_EMBEDDING_DIM => Err(GlyprError::InvalidMrlDimension {
            dim: d,
            max: TEXT_EMBEDDING_DIM,
        }),
        Some(TEXT_EMBEDDING_DIM) => Ok(None),
        Some(d) => Ok(Some(d)),
    }
}

impl CoreMLTextBackbone {
    fn inspect_and_build_entry(model: Model, path: &Path) -> Result<TextModelEntry> {
        let inputs = model.inputs();
        let id_input = inputs
            .iter()
            .find(|inp| inp.name() == "input_ids")
            .ok_or_else(|| {
                anyhow!(
                    "Model at {} does not contain 'input_ids' input feature",
                    path.display()
                )
            })?;

        if !inputs.iter().any(|inp| inp.name() == "attention_mask") {
            bail!(
                "Model at {} does not contain 'attention_mask' input feature",
                path.display()
            );
        }

        let seq_len = id_input
            .shape()
            .and_then(|s| if s.len() >= 2 { Some(s[1]) } else { None })
            .ok_or_else(|| {
                anyhow!(
                    "Model at {} has missing or non-2D 'input_ids' shape",
                    path.display()
                )
            })?;

        if seq_len == 0 {
            bail!("Model at {} has invalid static seq_len=0", path.display());
        }

        let outputs = model.outputs();
        let out_desc = outputs
            .first()
            .ok_or_else(|| anyhow!("Model at {} contains no output features", path.display()))?;

        let output_name = out_desc.name().to_string();
        let model_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("text_backbone.mlmodelc")
            .to_string();

        Ok(TextModelEntry {
            #[cfg(not(test))]
            model,
            #[cfg(test)]
            model: Some(model),
            model_name,
            seq_len,
            output_name,
        })
    }

    /// Load a single compiled CoreML text backbone model (`.mlmodelc`).
    pub fn from_path<P: AsRef<Path>>(
        path: P,
        compute_units: CoreMLComputeUnits,
    ) -> GlyprResult<Self> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Err(GlyprError::CoreMLNotFound {
                path: path_ref.to_path_buf(),
            });
        }

        let model = Model::load(path_ref, compute_units.into()).map_err(|e| {
            GlyprError::ModelLoad(format!(
                "Failed to load CoreML text model at {}: {}",
                path_ref.display(),
                e
            ))
        })?;

        let entry = Self::inspect_and_build_entry(model, path_ref)?;
        Ok(Self {
            buckets: vec![entry],
            compute_units,
        })
    }

    /// Load all compiled CoreML text backbone models found inside a directory (e.g. `models/coreml`).
    ///
    /// Introspects every `*.mlmodelc` bundle by checking for `"input_ids"` and `"attention_mask"`
    /// inputs (cleanly skipping vision models), sorts buckets by `seq_len`, and surfaces
    /// `.mlpackage` / load diagnostics if no usable text models are found.
    pub fn from_dir<P: AsRef<Path>>(
        dir: P,
        compute_units: CoreMLComputeUnits,
    ) -> GlyprResult<Self> {
        let dir_ref = dir.as_ref();
        if !dir_ref.is_dir() {
            return Err(GlyprError::CoreMLNotFound {
                path: dir_ref.to_path_buf(),
            });
        }

        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir_ref)?
            .filter_map(|e| e.ok().map(|entry| entry.path()))
            .collect();
        entries.sort();

        let mut buckets: Vec<TextModelEntry> = Vec::new();
        let mut load_errors: Vec<(PathBuf, String)> = Vec::new();
        let mut mlpackage_paths: Vec<PathBuf> = Vec::new();

        for path in entries {
            if !path.is_dir() {
                continue;
            }
            match path.extension().and_then(|s| s.to_str()) {
                Some("mlpackage") => {
                    mlpackage_paths.push(path);
                }
                Some("mlmodelc") => {
                    let model = match Model::load(&path, compute_units.into()) {
                        Ok(m) => m,
                        Err(e) => {
                            load_errors.push((path, format!("Model::load failed: {e}")));
                            continue;
                        }
                    };

                    let inputs = model.inputs();
                    let has_text_inputs = inputs.iter().any(|i| i.name() == "input_ids")
                        && inputs.iter().any(|i| i.name() == "attention_mask");
                    if !has_text_inputs {
                        let is_vision_model = inputs.iter().any(|i| i.name() == "pixel_values");
                        if !is_vision_model
                            && path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .is_some_and(|s| s.contains("text"))
                        {
                            load_errors.push((
                                path,
                                "Missing expected 'input_ids' / 'attention_mask' input features"
                                    .to_string(),
                            ));
                        }
                        continue;
                    }

                    match Self::inspect_and_build_entry(model, &path) {
                        Ok(entry) => {
                            if let Some(existing) =
                                buckets.iter().find(|b| b.seq_len == entry.seq_len)
                            {
                                eprintln!(
                                    "Warning: duplicate CoreML text seq_len={} in '{}' (keeping '{}')",
                                    entry.seq_len,
                                    path.display(),
                                    existing.model_name
                                );
                            } else {
                                buckets.push(entry);
                            }
                        }
                        Err(e) => {
                            load_errors.push((path, e.to_string()));
                        }
                    }
                }
                _ => {}
            }
        }

        if buckets.is_empty() {
            let mut msg = format!(
                "No compiled CoreML text backbone models found in {}",
                dir_ref.display()
            );
            if !mlpackage_paths.is_empty() {
                msg.push_str(&format!(
                    ". Found {} uncompiled .mlpackage bundle(s), but CoreML runtime requires compiled .mlmodelc bundles. \
                     Run 'xcrun coremlcompiler compile <pkg>.mlpackage {}' or re-run 'uv run python scripts/export_coreml.py' without --no-compile.",
                    mlpackage_paths.len(),
                    dir_ref.display()
                ));
            }
            if !load_errors.is_empty() {
                let details: Vec<String> = load_errors
                    .iter()
                    .map(|(p, e)| format!("{}: {}", p.display(), e))
                    .collect();
                msg.push_str(&format!(". Load diagnostics: [{}]", details.join("; ")));
            }
            return Err(GlyprError::ModelLoad(msg));
        }

        buckets.sort_by_key(|b| b.seq_len);
        Ok(Self {
            buckets,
            compute_units,
        })
    }

    fn select_bucket(&self, num_tokens: usize) -> GlyprResult<&TextModelEntry> {
        if let Some(entry) = self.buckets.iter().find(|b| b.seq_len >= num_tokens) {
            return Ok(entry);
        }
        Err(GlyprError::ContextLimitExceeded {
            actual: num_tokens,
            max: self.max_seq_len(),
        })
    }

    /// Run a dummy 1-token forward pass across all loaded text buckets to pre-warm ANE compilation.
    pub fn warmup(&self) -> GlyprResult<()> {
        for entry in &self.buckets {
            let mut padded_ids = vec![0i32; entry.seq_len];
            let mut padded_mask = vec![0.0f32; entry.seq_len];
            padded_ids[0] = 2; // <bos>
            padded_mask[0] = 1.0;

            let t_ids = BorrowedTensor::from_i32(&padded_ids, &[1, entry.seq_len])
                .context("Failed to create warmup BorrowedTensor for input_ids")?;
            let t_mask = BorrowedTensor::from_f32(&padded_mask, &[1, entry.seq_len])
                .context("Failed to create warmup BorrowedTensor for attention_mask")?;

            let inputs: &[(&str, &dyn AsMultiArray)] =
                &[("input_ids", &t_ids), ("attention_mask", &t_mask)];

            #[cfg(not(test))]
            let model = &entry.model;
            #[cfg(test)]
            let model = entry
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("Mock CoreML text model cannot execute warmup"))?;

            let _ = model.predict(inputs).with_context(|| {
                format!(
                    "CoreML text warmup failed for {} (seq_len={})",
                    entry.model_name, entry.seq_len
                )
            })?;
        }
        Ok(())
    }

    /// Embed a sequence of token IDs into a 768-dimensional L2-normalized embedding vector.
    ///
    /// If `token_ids.len() > self.max_seq_len()`, fails immediately with a descriptive error.
    /// Sequences shorter than the chosen bucket's `seq_len` are automatically right-padded with zero mask.
    pub fn embed_token_ids(&self, token_ids: &[i32]) -> GlyprResult<Vec<f32>> {
        self.embed_token_ids_with_dim(token_ids, None)
    }

    /// Embed a sequence of token IDs with optional Matryoshka (MRL) dimension truncation (`1..=768`).
    pub fn embed_token_ids_with_dim(
        &self,
        token_ids: &[i32],
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(GlyprError::InvalidInput(
                "Cannot embed an empty token sequence".to_string(),
            ));
        }
        let truncate_dim = validate_text_mrl_dim(truncate_dim)?;
        let num_tokens = token_ids.len();
        let entry = self.select_bucket(num_tokens)?;

        let mut padded_ids = vec![0i32; entry.seq_len];
        let mut padded_mask = vec![0.0f32; entry.seq_len];

        padded_ids[..num_tokens].copy_from_slice(token_ids);
        padded_mask[..num_tokens].fill(1.0);

        let t_ids = BorrowedTensor::from_i32(&padded_ids, &[1, entry.seq_len])
            .context("Failed to create BorrowedTensor for input_ids")?;
        let t_mask = BorrowedTensor::from_f32(&padded_mask, &[1, entry.seq_len])
            .context("Failed to create BorrowedTensor for attention_mask")?;

        let inputs: &[(&str, &dyn AsMultiArray)] =
            &[("input_ids", &t_ids), ("attention_mask", &t_mask)];

        #[cfg(not(test))]
        let model = &entry.model;
        #[cfg(test)]
        let model = entry
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("Mock CoreML text model cannot execute predict"))?;

        let prediction = model
            .predict(inputs)
            .context("CoreML prediction failed during text backbone forward pass")?;

        let (mut emb_data, shape) = prediction
            .get_f32(&entry.output_name)
            .with_context(|| format!("Failed to read output feature '{}'", entry.output_name))?;

        if emb_data.len() != TEXT_EMBEDDING_DIM {
            return Err(GlyprError::Other(anyhow!(
                "Unexpected output length {} (expected {}, shape: {:?})",
                emb_data.len(),
                TEXT_EMBEDDING_DIM,
                shape
            )));
        }

        // Defensive normalization check
        let norm_sq: f32 = emb_data.iter().map(|&x| x * x).sum();
        if norm_sq.is_nan() || norm_sq <= 0.0 {
            return Err(GlyprError::Other(anyhow!(
                "CoreML text embedding returned invalid or NaN vector"
            )));
        }

        // Ensure bit-exact unit length
        let inv_norm = 1.0 / norm_sq.sqrt();
        for val in &mut emb_data {
            *val *= inv_norm;
        }

        match truncate_dim {
            Some(dim) => truncate_mrl(&emb_data, dim),
            None => Ok(emb_data),
        }
    }

    /// Embed a text string, optionally applying the prompt format for the given task.
    pub fn embed_text(
        &self,
        text: &str,
        tokenizer: &GlyprTokenizer,
        task: Option<Task>,
    ) -> GlyprResult<Vec<f32>> {
        self.embed_text_with_dim(text, tokenizer, task, None)
    }

    /// Embed a text string with optional task formatting and MRL dimension truncation (`1..=768`).
    pub fn embed_text_with_dim(
        &self,
        text: &str,
        tokenizer: &GlyprTokenizer,
        task: Option<Task>,
        truncate_dim: Option<usize>,
    ) -> GlyprResult<Vec<f32>> {
        let formatted = if let Some(t) = task {
            t.format_text(text)
        } else {
            text.to_string()
        };

        let encoding = tokenizer.encode(&formatted)?;
        let token_ids: Vec<i32> = encoding.iter().map(|&id| id as i32).collect();
        self.embed_token_ids_with_dim(&token_ids, truncate_dim)
    }

    /// Embed a text string directly into a Candle `Tensor` `[1, 768]` on the specified device.
    pub fn embed_to_tensor(
        &self,
        text: &str,
        tokenizer: &GlyprTokenizer,
        task: Option<Task>,
        device: &Device,
    ) -> GlyprResult<Tensor> {
        let vec = self.embed_text(text, tokenizer, task)?;
        Ok(Tensor::from_vec(vec, (1, TEXT_EMBEDDING_DIM), device)?)
    }

    /// Maximum static sequence length supported across loaded buckets.
    pub fn seq_len(&self) -> usize {
        self.max_seq_len()
    }

    /// Maximum static sequence length supported across loaded buckets.
    pub fn max_seq_len(&self) -> usize {
        self.buckets.last().map(|b| b.seq_len).unwrap_or(0)
    }

    /// Sorted list of all loaded static sequence-length buckets.
    pub fn supported_seq_lens(&self) -> Vec<usize> {
        self.buckets.iter().map(|b| b.seq_len).collect()
    }

    /// Compiled bundle names for loaded text buckets.
    pub fn model_names(&self) -> Vec<(usize, &str)> {
        self.buckets
            .iter()
            .map(|b| (b.seq_len, b.model_name.as_str()))
            .collect()
    }

    /// Configured CoreML compute units.
    pub fn compute_units(&self) -> CoreMLComputeUnits {
        self.compute_units
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_text_mrl_dim() {
        assert_eq!(validate_text_mrl_dim(None).unwrap(), None);
        assert_eq!(validate_text_mrl_dim(Some(768)).unwrap(), None);
        assert_eq!(validate_text_mrl_dim(Some(256)).unwrap(), Some(256));
        assert!(matches!(
            validate_text_mrl_dim(Some(0)).unwrap_err(),
            GlyprError::InvalidMrlDimension { dim: 0, max: 768 }
        ));
        let err = validate_text_mrl_dim(Some(1024)).unwrap_err();
        assert!(matches!(
            err,
            GlyprError::InvalidMrlDimension {
                dim: 1024,
                max: 768
            }
        ));
    }

    #[test]
    fn test_coreml_text_bucket_selection_and_limits() {
        let backbone = CoreMLTextBackbone {
            buckets: vec![
                TextModelEntry {
                    model: None,
                    model_name: "gemma2_text_backbone_s128_w8a16.mlmodelc".to_string(),
                    seq_len: 128,
                    output_name: "output".to_string(),
                },
                TextModelEntry {
                    model: None,
                    model_name: "gemma2_text_backbone_s512_w8a16.mlmodelc".to_string(),
                    seq_len: 512,
                    output_name: "output".to_string(),
                },
            ],
            compute_units: CoreMLComputeUnits::CpuAndNeuralEngine,
        };

        assert_eq!(backbone.max_seq_len(), 512);
        assert_eq!(backbone.supported_seq_lens(), vec![128, 512]);
        assert_eq!(backbone.select_bucket(64).unwrap().seq_len, 128);
        assert_eq!(backbone.select_bucket(128).unwrap().seq_len, 128);
        assert_eq!(backbone.select_bucket(129).unwrap().seq_len, 512);
        assert_eq!(backbone.select_bucket(512).unwrap().seq_len, 512);

        let err = backbone.select_bucket(513).err().unwrap();
        assert!(
            matches!(
                err,
                GlyprError::ContextLimitExceeded {
                    actual: 513,
                    max: 512
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_coreml_text_from_dir_reports_uncompiled_mlpackage() {
        let tmp =
            std::env::temp_dir().join(format!("glypr_coreml_text_pkg_test_{}", std::process::id()));
        let pkg = tmp.join("gemma2_text_backbone_s512_w8a16.mlpackage");
        std::fs::create_dir_all(&pkg).unwrap();

        let err = CoreMLTextBackbone::from_dir(&tmp, CoreMLComputeUnits::CpuAndNeuralEngine)
            .err()
            .expect("from_dir should fail when only .mlpackage exists");
        assert!(matches!(err, GlyprError::ModelLoad(_)), "got: {err:?}");
        let msg = err.to_string();
        assert!(msg.contains(".mlpackage"), "got: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_coreml_text_missing_path_and_empty_dir_errors() {
        let missing_path = std::env::temp_dir().join(format!(
            "glypr_coreml_text_missing_{}.mlmodelc",
            std::process::id()
        ));
        let err_path =
            CoreMLTextBackbone::from_path(&missing_path, CoreMLComputeUnits::CpuAndNeuralEngine)
                .err()
                .expect("from_path on missing file must fail");
        assert!(matches!(err_path, GlyprError::CoreMLNotFound { .. }));

        let missing_dir = std::env::temp_dir().join(format!(
            "glypr_coreml_text_missing_dir_{}",
            std::process::id()
        ));
        let err_missing_dir =
            CoreMLTextBackbone::from_dir(&missing_dir, CoreMLComputeUnits::CpuAndNeuralEngine)
                .err()
                .expect("from_dir on missing dir must fail");
        assert!(matches!(err_missing_dir, GlyprError::CoreMLNotFound { .. }));

        let empty_dir = std::env::temp_dir().join(format!(
            "glypr_coreml_text_empty_dir_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&empty_dir).unwrap();
        let err_empty =
            CoreMLTextBackbone::from_dir(&empty_dir, CoreMLComputeUnits::CpuAndNeuralEngine)
                .err()
                .expect("from_dir on empty dir must fail");
        assert!(matches!(err_empty, GlyprError::ModelLoad(_)));
        let _ = std::fs::remove_dir_all(&empty_dir);
    }
}
