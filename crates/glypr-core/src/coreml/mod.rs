pub mod text;
pub mod vision;

pub use text::CoreMLTextBackbone;
pub use vision::CoreMLVisionTower;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoreMLComputeUnits {
    #[default]
    CpuAndNeuralEngine,
    All,
    CpuAndGpu,
    CpuOnly,
}

impl From<CoreMLComputeUnits> for coreml_native::ComputeUnits {
    fn from(cu: CoreMLComputeUnits) -> Self {
        match cu {
            CoreMLComputeUnits::CpuAndNeuralEngine => coreml_native::ComputeUnits::CpuAndNeuralEngine,
            CoreMLComputeUnits::All => coreml_native::ComputeUnits::All,
            CoreMLComputeUnits::CpuAndGpu => coreml_native::ComputeUnits::CpuAndGpu,
            CoreMLComputeUnits::CpuOnly => coreml_native::ComputeUnits::CpuOnly,
        }
    }
}
