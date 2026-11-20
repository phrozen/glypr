pub mod text;
pub mod vision;

pub use crate::{CoreMLComputeUnits, TextBackendPolicy};
pub use text::CoreMLTextBackbone;
pub use vision::{CoreMLVisionTower, VisionGridInfo};

impl From<CoreMLComputeUnits> for coreml_native::ComputeUnits {
    fn from(cu: CoreMLComputeUnits) -> Self {
        match cu {
            CoreMLComputeUnits::CpuAndNeuralEngine => {
                coreml_native::ComputeUnits::CpuAndNeuralEngine
            }
            CoreMLComputeUnits::All => coreml_native::ComputeUnits::All,
            CoreMLComputeUnits::CpuAndGpu => coreml_native::ComputeUnits::CpuAndGpu,
            CoreMLComputeUnits::CpuOnly => coreml_native::ComputeUnits::CpuOnly,
        }
    }
}

/// Input or output feature metadata for a compiled CoreML `.mlmodelc` bundle.
#[derive(Debug, Clone)]
pub struct CoreMLFeatureInfo {
    /// Feature name (e.g. `"pixel_values"`, `"input_ids"`, `"attention_mask"`).
    pub name: String,
    /// Optional static tensor shape.
    pub shape: Option<Vec<usize>>,
}

/// Introspection summary for a compiled `.mlmodelc` bundle.
#[derive(Debug, Clone)]
pub struct CoreMLModelInfo {
    /// Bundle path on disk.
    pub path: std::path::PathBuf,
    /// Input feature descriptions.
    pub inputs: Vec<CoreMLFeatureInfo>,
    /// Output feature descriptions.
    pub outputs: Vec<CoreMLFeatureInfo>,
}

/// Load and inspect a compiled `.mlmodelc` bundle's input/output feature names and static shapes.
pub fn inspect_mlmodelc<P: AsRef<std::path::Path>>(
    path: P,
    compute_units: CoreMLComputeUnits,
) -> anyhow::Result<CoreMLModelInfo> {
    let path_ref = path.as_ref();
    if !path_ref.exists() {
        anyhow::bail!("CoreML bundle not found at {}", path_ref.display());
    }
    let model = coreml_native::Model::load(path_ref, compute_units.into()).map_err(|e| {
        anyhow::anyhow!(
            "Failed to load CoreML model at {}: {}",
            path_ref.display(),
            e
        )
    })?;
    let inputs = model
        .inputs()
        .iter()
        .map(|f| CoreMLFeatureInfo {
            name: f.name().to_string(),
            shape: f.shape().map(|s| s.to_vec()),
        })
        .collect();
    let outputs = model
        .outputs()
        .iter()
        .map(|f| CoreMLFeatureInfo {
            name: f.name().to_string(),
            shape: f.shape().map(|s| s.to_vec()),
        })
        .collect();
    Ok(CoreMLModelInfo {
        path: path_ref.to_path_buf(),
        inputs,
        outputs,
    })
}
