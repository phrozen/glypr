use crate::coreml::CoreMLComputeUnits;
use crate::multimodal::ProcessedImage;
use anyhow::{Context, Result, anyhow, bail};
use candle_core::{Device, Tensor};
use coreml_native::{AsMultiArray, BorrowedTensor, Model};
use std::collections::HashMap;
use std::path::Path;

/// A compiled CoreML vision model entry with its static patch count and output feature name.
struct VisionModelEntry {
    model: Model,
    num_patches: usize,
    num_soft_tokens: usize,
    output_name: String,
}

/// CoreML-accelerated Vision Tower targeting the Apple Neural Engine (ANE) and GPU.
pub struct CoreMLVisionTower {
    models: HashMap<usize, VisionModelEntry>,
    compute_units: CoreMLComputeUnits,
}

impl CoreMLVisionTower {
    /// Load a single compiled CoreML vision model (`.mlmodelc`).
    ///
    /// Automatically introspects the model to discover its static patch input count
    /// and output feature name.
    pub fn from_path<P: AsRef<Path>>(path: P, compute_units: CoreMLComputeUnits) -> Result<Self> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            bail!("CoreML model not found at {}", path_ref.display());
        }

        let model = Model::load(path_ref, compute_units.into()).with_context(|| {
            format!(
                "Failed to load CoreML vision model at {}",
                path_ref.display()
            )
        })?;

        let entry = Self::inspect_and_build_entry(model, path_ref)?;
        let mut models = HashMap::new();
        models.insert(entry.num_patches, entry);

        Ok(Self {
            models,
            compute_units,
        })
    }

    /// Load all compiled vision models found inside a directory (e.g. `models/coreml`).
    ///
    /// Matches models supporting different budgets (e.g. 1134 patches for Budget 140,
    /// 2394 patches for Budget 280).
    pub fn from_dir<P: AsRef<Path>>(dir: P, compute_units: CoreMLComputeUnits) -> Result<Self> {
        let dir_ref = dir.as_ref();
        if !dir_ref.is_dir() {
            bail!("Directory not found at {}", dir_ref.display());
        }

        let mut models = HashMap::new();
        for entry in std::fs::read_dir(dir_ref)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() && path.extension().and_then(|s| s.to_str()) == Some("mlmodelc") {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if stem.contains("vision_tower")
                    && let Ok(model) = Model::load(&path, compute_units.into())
                    && let Ok(v_entry) = Self::inspect_and_build_entry(model, &path)
                {
                    models.insert(v_entry.num_patches, v_entry);
                }
            }
        }

        if models.is_empty() {
            bail!(
                "No compiled CoreML vision models found in {}",
                dir_ref.display()
            );
        }

        Ok(Self {
            models,
            compute_units,
        })
    }

    fn inspect_and_build_entry(model: Model, path: &Path) -> Result<VisionModelEntry> {
        let inputs = model.inputs();
        let pixel_input = inputs
            .iter()
            .find(|inp| inp.name() == "pixel_values")
            .ok_or_else(|| {
                anyhow!(
                    "Model at {} does not have 'pixel_values' input",
                    path.display()
                )
            })?;

        let in_shape = pixel_input
            .shape()
            .ok_or_else(|| anyhow!("Model at {} does not specify input shape", path.display()))?;

        if in_shape.len() != 3 || in_shape[2] != 768 {
            bail!(
                "Invalid input shape {:?} for vision model at {} (expected [1, num_patches, 768])",
                in_shape,
                path.display()
            );
        }

        let num_patches = in_shape[1];

        let outputs = model.outputs();
        let out_desc = outputs
            .first()
            .ok_or_else(|| anyhow!("Model at {} has no output features", path.display()))?;

        let output_name = out_desc.name().to_string();
        let num_soft_tokens = if let Some(out_shape) = out_desc.shape() {
            if out_shape.len() >= 2 {
                out_shape[1]
            } else {
                num_patches / 9
            }
        } else {
            num_patches / 9
        };

        Ok(VisionModelEntry {
            model,
            num_patches,
            num_soft_tokens,
            output_name,
        })
    }

    /// Returns the patch counts supported by currently loaded models.
    pub fn supported_patch_counts(&self) -> Vec<usize> {
        let mut keys: Vec<usize> = self.models.keys().copied().collect();
        keys.sort_unstable();
        keys
    }

    /// Check if a specific patch count is supported.
    pub fn supports_patches(&self, patches: usize) -> bool {
        self.models.contains_key(&patches)
    }

    /// Number of output soft tokens produced for a given patch count.
    pub fn num_soft_tokens_for_patches(&self, patches: usize) -> Option<usize> {
        self.models.get(&patches).map(|m| m.num_soft_tokens)
    }

    /// Raw forward pass encoding a `ProcessedImage` into flat f32 data.
    ///
    /// Returns `(output_buffer, shape)` where shape is `[1, num_soft_tokens, 512]`.
    pub fn encode_raw(&self, image: &ProcessedImage) -> Result<(Vec<f32>, Vec<usize>)> {
        let entry = self.models.get(&image.num_patches).ok_or_else(|| {
            anyhow!(
                "No CoreML vision model loaded for {} patches (available: {:?})",
                image.num_patches,
                self.supported_patch_counts()
            )
        })?;

        let expected_elements = entry.num_patches * 768;
        if image.pixel_values.len() != expected_elements {
            bail!(
                "Image pixel values buffer size {} does not match expected size {}",
                image.pixel_values.len(),
                expected_elements
            );
        }

        let tensor = BorrowedTensor::from_f32(&image.pixel_values, &[1, entry.num_patches, 768])
            .context("Failed to create BorrowedTensor for pixel_values")?;

        let inputs: &[(&str, &dyn AsMultiArray)] = &[("pixel_values", &tensor)];
        let prediction = entry
            .model
            .predict(inputs)
            .context("CoreML prediction failed during vision tower execution")?;

        let (buf, shape) = prediction
            .get_f32(&entry.output_name)
            .with_context(|| format!("Failed to read output feature '{}'", entry.output_name))?;

        Ok((buf, shape))
    }

    /// Encode a `ProcessedImage` directly into a Candle `Tensor` of shape `[num_soft_tokens, 512]`.
    ///
    /// Copies the output buffer ($126 \times 512 \times 4\text{ bytes} \approx 258\text{ KB}$) into
    /// device memory ready for splicing into the multimodal prompt sequence.
    pub fn encode_to_tensor(&self, image: &ProcessedImage, device: &Device) -> Result<Tensor> {
        self.encode_to_tensor_dtype(image, device, candle_core::DType::F32)
    }

    /// Encode a `ProcessedImage` directly into a Candle `Tensor` with the requested data type.
    pub fn encode_to_tensor_dtype(
        &self,
        image: &ProcessedImage,
        device: &Device,
        dtype: candle_core::DType,
    ) -> Result<Tensor> {
        let (data, shape) = self.encode_raw(image)?;
        let num_tokens = if shape.len() >= 2 {
            shape[1]
        } else {
            image.num_soft_tokens
        };

        let tensor = Tensor::from_vec(data, (1, num_tokens, 512), device)
            .context("Failed to create Candle Tensor from CoreML vision output")?;

        let squeezed = tensor.squeeze(0)?;
        if squeezed.dtype() != dtype {
            Ok(squeezed.to_dtype(dtype)?)
        } else {
            Ok(squeezed)
        }
    }

    pub fn compute_units(&self) -> CoreMLComputeUnits {
        self.compute_units
    }
}
