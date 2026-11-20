use crate::coreml::CoreMLComputeUnits;
use crate::error::{GlyprError, GlyprResult};
use crate::multimodal::ProcessedImage;
use anyhow::{Context, Result, anyhow, bail};
use candle_core::{Device, Tensor};
use coreml_native::{AsMultiArray, BorrowedTensor, Model};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Summary metadata for a loaded CoreML vision model bucket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionGridInfo {
    /// Compiled bundle directory name (e.g. `gemma2_vision_tower_p1134_t126_h42_w27_w8a16.mlmodelc`).
    pub model_name: String,
    /// Height in 16x16 patches.
    pub patch_height: usize,
    /// Width in 16x16 patches.
    pub patch_width: usize,
    /// Total patches (`patch_height * patch_width`).
    pub num_patches: usize,
    /// Number of pooled soft tokens emitted by this vision bucket.
    pub num_soft_tokens: usize,
}

/// A compiled CoreML vision model entry with its static grid dimensions and output feature name.
struct VisionModelEntry {
    #[cfg(not(test))]
    model: Model,
    #[cfg(test)]
    model: Option<Model>,
    model_name: String,
    num_patches: usize,
    patch_height: usize,
    patch_width: usize,
    num_soft_tokens: usize,
    output_name: String,
}

/// CoreML-accelerated Vision Tower targeting the Apple Neural Engine (ANE) and GPU.
pub struct CoreMLVisionTower {
    models: HashMap<(usize, usize), VisionModelEntry>,
    compute_units: CoreMLComputeUnits,
}

/// Parse `(patch_height, patch_width)` from `_h{H}_w{W}` filename tags or known legacy portrait stems.
fn parse_vision_grid_from_path(path: &Path, num_patches: usize) -> Result<(usize, usize)> {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let parts: Vec<&str> = stem.split('_').collect();
    for pair in parts.windows(2) {
        if let (Some(h_str), Some(w_str)) = (pair[0].strip_prefix('h'), pair[1].strip_prefix('w'))
            && !h_str.is_empty()
            && !w_str.is_empty()
            && h_str.bytes().all(|b| b.is_ascii_digit())
            && w_str.bytes().all(|b| b.is_ascii_digit())
            && let (Ok(h), Ok(w)) = (h_str.parse::<usize>(), w_str.parse::<usize>())
        {
            if h == 0 || w == 0 {
                bail!(
                    "Invalid zero grid dimension ({}x{}) in CoreML vision model {}",
                    h,
                    w,
                    path.display()
                );
            }
            if h * w != num_patches {
                bail!(
                    "CoreML vision model {} grid {}x{} ({} patches) does not match input shape num_patches={}",
                    path.display(),
                    h,
                    w,
                    h * w,
                    num_patches
                );
            }
            return Ok((h, w));
        }
    }

    let legacy_grid = if stem.contains("p1134_t126") {
        Some((42, 27))
    } else if stem.contains("p1260_t140") {
        Some((42, 30))
    } else if stem.contains("p2394_t266") {
        Some((57, 42))
    } else if stem.contains("p2520_t280") {
        Some((60, 42))
    } else {
        None
    };

    match legacy_grid {
        Some((h, w)) if h * w == num_patches => Ok((h, w)),
        Some((h, w)) => bail!(
            "Legacy CoreML vision model {} grid {}x{} does not match input shape num_patches={}",
            path.display(),
            h,
            w,
            num_patches
        ),
        None => bail!(
            "Cannot determine CoreML vision grid (patch_height, patch_width) for {} (num_patches={}); expected _h{{H}}_w{{W}} in filename",
            path.display(),
            num_patches
        ),
    }
}

impl CoreMLVisionTower {
    /// Load a single compiled CoreML vision model (`.mlmodelc`).
    ///
    /// Automatically introspects the model to discover its static patch input count
    /// and output feature name.
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
                "Failed to load CoreML vision model at {}: {}",
                path_ref.display(),
                e
            ))
        })?;

        let entry = Self::inspect_and_build_entry(model, path_ref)?;
        let mut models = HashMap::new();
        models.insert((entry.patch_height, entry.patch_width), entry);

        Ok(Self {
            models,
            compute_units,
        })
    }

    /// Load all compiled vision models found inside a directory (e.g. `models/coreml`).
    ///
    /// Discovers every `*.mlmodelc` bundle containing a `"pixel_values"` input feature
    /// (skipping `"input_ids"` text backbones without relying on filename prefixes),
    /// surfaces per-file load diagnostics, and detects uncompiled `.mlpackage` bundles.
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

        let mut models: HashMap<(usize, usize), VisionModelEntry> = HashMap::new();
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
                    let has_pixel_values = inputs.iter().any(|inp| inp.name() == "pixel_values");
                    if !has_pixel_values {
                        let is_text_model = inputs.iter().any(|inp| inp.name() == "input_ids");
                        if !is_text_model
                            && path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .is_some_and(|s| s.contains("vision"))
                        {
                            load_errors.push((
                                path,
                                "Missing expected 'pixel_values' input feature".to_string(),
                            ));
                        }
                        continue;
                    }

                    match Self::inspect_and_build_entry(model, &path) {
                        Ok(v_entry) => {
                            let grid = (v_entry.patch_height, v_entry.patch_width);
                            if let Some(existing) = models.get(&grid) {
                                eprintln!(
                                    "Warning: duplicate CoreML vision grid {}x{} in '{}' (keeping '{}')",
                                    grid.0,
                                    grid.1,
                                    path.display(),
                                    existing.model_name
                                );
                            } else {
                                models.insert(grid, v_entry);
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

        if models.is_empty() {
            let mut msg = format!(
                "No compiled CoreML vision models found in {}",
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
        let (patch_height, patch_width) = parse_vision_grid_from_path(path, num_patches)?;

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
        let model_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("vision_tower.mlmodelc")
            .to_string();

        Ok(VisionModelEntry {
            #[cfg(not(test))]
            model,
            #[cfg(test)]
            model: Some(model),
            model_name,
            num_patches,
            patch_height,
            patch_width,
            num_soft_tokens,
            output_name,
        })
    }

    /// Returns the `(patch_height, patch_width)` grids supported by currently loaded models.
    pub fn supported_grids(&self) -> Vec<(usize, usize)> {
        let mut keys: Vec<(usize, usize)> = self.models.keys().copied().collect();
        keys.sort_unstable();
        keys
    }

    /// Check if a specific `(patch_height, patch_width)` grid is supported.
    pub fn supports_grid(&self, patch_height: usize, patch_width: usize) -> bool {
        self.models.contains_key(&(patch_height, patch_width))
    }

    /// Returns the patch counts supported by currently loaded models.
    pub fn supported_patch_counts(&self) -> Vec<usize> {
        let mut counts: Vec<usize> = self.models.values().map(|m| m.num_patches).collect();
        counts.sort_unstable();
        counts.dedup();
        counts
    }

    /// Check if a specific patch count is supported.
    pub fn supports_patches(&self, patches: usize) -> bool {
        self.models.values().any(|m| m.num_patches == patches)
    }

    /// Number of output soft tokens produced for a given patch count.
    pub fn num_soft_tokens_for_patches(&self, patches: usize) -> Option<usize> {
        self.models
            .values()
            .find(|m| m.num_patches == patches)
            .map(|m| m.num_soft_tokens)
    }

    /// Number of output soft tokens produced for a given `(patch_height, patch_width)` grid.
    pub fn num_soft_tokens_for_grid(
        &self,
        patch_height: usize,
        patch_width: usize,
    ) -> Option<usize> {
        self.models
            .get(&(patch_height, patch_width))
            .map(|m| m.num_soft_tokens)
    }

    /// Returns summary metadata for all loaded vision grids, sorted by `(patch_height, patch_width)`.
    pub fn grid_infos(&self) -> Vec<VisionGridInfo> {
        let mut infos: Vec<VisionGridInfo> = self
            .models
            .values()
            .map(|entry| VisionGridInfo {
                model_name: entry.model_name.clone(),
                patch_height: entry.patch_height,
                patch_width: entry.patch_width,
                num_patches: entry.num_patches,
                num_soft_tokens: entry.num_soft_tokens,
            })
            .collect();
        infos.sort_by_key(|info| (info.patch_height, info.patch_width));
        infos
    }

    /// Execute a single dummy forward pass on every loaded CoreML vision model to pre-warm ANE compilation.
    pub fn warmup(&self) -> GlyprResult<()> {
        for entry in self.models.values() {
            let dummy = vec![0.0f32; entry.num_patches * 768];
            let tensor = BorrowedTensor::from_f32(&dummy, &[1, entry.num_patches, 768])
                .context("Failed to create warmup BorrowedTensor for pixel_values")?;
            let inputs: &[(&str, &dyn AsMultiArray)] = &[("pixel_values", &tensor)];
            #[cfg(not(test))]
            let model = &entry.model;
            #[cfg(test)]
            let model = entry
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("Mock CoreML vision model cannot execute warmup"))?;
            let _ = model.predict(inputs).with_context(|| {
                format!(
                    "CoreML vision warmup failed for {} ({}x{})",
                    entry.model_name, entry.patch_height, entry.patch_width
                )
            })?;
        }
        Ok(())
    }

    /// Raw forward pass encoding a `ProcessedImage` into flat f32 data.
    ///
    /// Returns `(output_buffer, shape)` where shape is `[1, num_soft_tokens, 512]`.
    pub fn encode_raw(&self, image: &ProcessedImage) -> GlyprResult<(Vec<f32>, Vec<usize>)> {
        let entry = match self.models.get(&(image.patch_height, image.patch_width)) {
            Some(e) => e,
            None => {
                return Err(GlyprError::UnsupportedCoreMLPatchGrid {
                    patch_height: image.patch_height,
                    patch_width: image.patch_width,
                    num_patches: image.num_patches,
                    supported_grids: self.supported_grids(),
                });
            }
        };

        if image.num_patches != entry.num_patches {
            return Err(GlyprError::InvalidInput(format!(
                "CoreML vision patch count mismatch: image has {} patches ({}x{}), expected {}",
                image.num_patches, image.patch_height, image.patch_width, entry.num_patches
            )));
        }

        let expected_elements = entry.num_patches * 768;
        if image.pixel_values.len() != expected_elements {
            return Err(GlyprError::InvalidInput(format!(
                "Image pixel values buffer size {} does not match expected size {}",
                image.pixel_values.len(),
                expected_elements
            )));
        }

        let tensor = BorrowedTensor::from_f32(&image.pixel_values, &[1, entry.num_patches, 768])
            .context("Failed to create BorrowedTensor for pixel_values")?;

        let inputs: &[(&str, &dyn AsMultiArray)] = &[("pixel_values", &tensor)];
        #[cfg(not(test))]
        let model = &entry.model;
        #[cfg(test)]
        let model = entry
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("Mock CoreML vision model cannot execute predict"))?;
        let prediction = model
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
    pub fn encode_to_tensor(&self, image: &ProcessedImage, device: &Device) -> GlyprResult<Tensor> {
        self.encode_to_tensor_dtype(image, device, candle_core::DType::F32)
    }

    /// Encode a `ProcessedImage` directly into a Candle `Tensor` with the requested data type.
    pub fn encode_to_tensor_dtype(
        &self,
        image: &ProcessedImage,
        device: &Device,
        dtype: candle_core::DType,
    ) -> GlyprResult<Tensor> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vision_grid_tags_and_legacy_stems() {
        assert_eq!(
            parse_vision_grid_from_path(
                Path::new("gemma2_vision_tower_p1134_t126_h42_w27_w8a16.mlmodelc"),
                1134
            )
            .unwrap(),
            (42, 27)
        );
        assert_eq!(
            parse_vision_grid_from_path(
                Path::new("gemma2_vision_tower_p1134_t126_h27_w42_w8a16.mlmodelc"),
                1134
            )
            .unwrap(),
            (27, 42)
        );
        assert_eq!(
            parse_vision_grid_from_path(
                Path::new("gemma2_vision_tower_p1134_t126_w8a16.mlmodelc"),
                1134
            )
            .unwrap(),
            (42, 27)
        );
        assert_eq!(
            parse_vision_grid_from_path(
                Path::new("gemma2_vision_tower_p1260_t140_w8a16.mlmodelc"),
                1260
            )
            .unwrap(),
            (42, 30)
        );
        assert_eq!(
            parse_vision_grid_from_path(
                Path::new("gemma2_vision_tower_p2394_t266_w8a16.mlmodelc"),
                2394
            )
            .unwrap(),
            (57, 42)
        );
        assert_eq!(
            parse_vision_grid_from_path(
                Path::new("gemma2_vision_tower_p2520_t280_w8a16.mlmodelc"),
                2520
            )
            .unwrap(),
            (60, 42)
        );

        // Mismatched h*w vs num_patches or ambiguous stem must fail
        assert!(
            parse_vision_grid_from_path(
                Path::new("gemma2_vision_tower_p1134_t126_h42_w30_w8a16.mlmodelc"),
                1134
            )
            .is_err()
        );
        assert!(
            parse_vision_grid_from_path(Path::new("gemma2_vision_tower_p1134.mlmodelc"), 1134)
                .is_err()
        );
    }

    #[test]
    fn test_coreml_vision_tower_rejects_landscape_for_portrait_grid() {
        let mut models = HashMap::new();
        models.insert(
            (42, 27),
            VisionModelEntry {
                model: None,
                model_name: "gemma2_vision_tower_p1134_t126_h42_w27_w8a16.mlmodelc".to_string(),
                num_patches: 1134,
                patch_height: 42,
                patch_width: 27,
                num_soft_tokens: 126,
                output_name: "output".to_string(),
            },
        );
        let tower = CoreMLVisionTower {
            models,
            compute_units: CoreMLComputeUnits::CpuAndNeuralEngine,
        };

        assert!(tower.supports_grid(42, 27));
        assert!(!tower.supports_grid(27, 42));
        assert!(tower.supports_patches(1134));
        assert_eq!(tower.supported_grids(), vec![(42, 27)]);
        assert_eq!(tower.grid_infos().len(), 1);

        let landscape = ProcessedImage {
            pixel_values: vec![0.0; 1134 * 768],
            position_ids: vec![0; 1134 * 2],
            num_patches: 1134,
            num_soft_tokens: 126,
            patch_width: 42,
            patch_height: 27,
            target_width: 42 * 16,
            target_height: 27 * 16,
        };
        let err = tower.encode_raw(&landscape).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("27x42"), "unexpected error message: {msg}");
        assert!(
            msg.contains("42x27"),
            "expected loaded grid in error message: {msg}"
        );
        assert!(
            matches!(
                err,
                GlyprError::UnsupportedCoreMLPatchGrid {
                    patch_height: 27,
                    patch_width: 42,
                    num_patches: 1134,
                    ..
                }
            ),
            "expected GlyprError::UnsupportedCoreMLPatchGrid, got: {err:?}"
        );
    }

    #[test]
    fn test_coreml_vision_from_dir_reports_uncompiled_mlpackage() {
        let tmp =
            std::env::temp_dir().join(format!("glypr_coreml_pkg_test_{}", std::process::id()));
        let pkg = tmp.join("gemma2_vision_tower_p1134_t126_h42_w27_w8a16.mlpackage");
        std::fs::create_dir_all(&pkg).unwrap();

        let err = CoreMLVisionTower::from_dir(&tmp, CoreMLComputeUnits::CpuAndNeuralEngine)
            .err()
            .expect("from_dir should fail when only .mlpackage exists");
        assert!(matches!(err, GlyprError::ModelLoad(_)), "got: {err:?}");
        let msg = err.to_string();
        assert!(msg.contains(".mlpackage"), "got: {msg}");
        assert!(msg.contains("coremlcompiler"), "got: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
