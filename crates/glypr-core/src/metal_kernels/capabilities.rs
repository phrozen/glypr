use candle_metal_kernels::metal::Device as MtlDevice;
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLGPUFamily, MTLLanguageVersion};
use std::sync::OnceLock;

/// Metal execution tier selection (configurable via `GLYPR_METAL_TIER=auto|m5|legacy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalTier {
    /// Automatically enable M5 Neural Accelerator (`mpp::tensor_ops`) kernels when hardware + compiler support it.
    Auto,
    /// Force M5 Neural Accelerator (`mpp::tensor_ops`) kernels if compiler probe succeeds.
    M5,
    /// Force M1–M4 `simdgroup_matrix` path even on M5 hardware (useful for A/B benchmarking).
    Legacy,
}

impl MetalTier {
    /// Resolve tier from `GLYPR_METAL_TIER` environment variable (`auto` default).
    pub fn from_env() -> Self {
        match std::env::var("GLYPR_METAL_TIER")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "m5" | "metal4" | "nax" => Self::M5,
            "legacy" | "m1" | "m2" | "m3" | "m4" | "simdgroup" => Self::Legacy,
            _ => Self::Auto,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::M5 => "m5",
            Self::Legacy => "legacy",
        }
    }
}

/// Runtime hardware and compiler capability profile for the active Apple Silicon Metal GPU.
#[derive(Debug, Clone)]
pub struct MetalCapabilities {
    /// Human-readable GPU name (e.g. `"Apple M5 Max"`).
    pub device_name: String,
    /// Metal architecture identifier (e.g. `"applegpu_g18s"`).
    pub architecture: String,
    /// `MTLGPUFamilyApple9` (`1009`) or newer: native hardware `bfloat` ALU & `simdgroup_matrix<bfloat, 8, 8>`.
    pub has_native_bfloat16: bool,
    /// `MTLGPUFamilyApple10` (`1010`) or newer: Apple M5 generation GPU with per-core Neural Accelerators (NAX).
    pub has_apple10_m5: bool,
    /// `MTLGPUFamilyMetal4` (`5002`): Metal 4 API and MSL 4.0 support.
    pub has_metal4_family: bool,
    /// Runtime MSL 4.0 compiler probe for `<metal_tensor>` and `mpp::tensor_ops::matmul2d`.
    pub has_tensor_ops_compiler: bool,
    /// Maximum threadgroup memory length in bytes (typically 32,768 bytes on Apple Silicon).
    pub max_threadgroup_memory: usize,
    /// Recommended maximum working set size in bytes (unified VRAM budget).
    pub recommended_max_working_set_size: usize,
}

const M5_TENSOR_OPS_PROBE_SRC: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void glypr_m5_tensor_ops_probe(
    device bfloat* A_ptr [[buffer(0)]],
    device bfloat* B_ptr [[buffer(1)]],
    device float*  C_ptr [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A(A_ptr, dextents<int32_t, 2>(32, 32));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> B(B_ptr, dextents<int32_t, 2>(32, 32));
    tensor<device float,  dextents<int32_t, 2>, tensor_inline> C(C_ptr, dextents<int32_t, 2>(32, 32));

    constexpr auto desc = matmul2d_descriptor(32, 32, 32, false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    op.run(A, B, C);
}
"#;

static CAPABILITIES_CACHE: OnceLock<MetalCapabilities> = OnceLock::new();

/// Compile a Metal 4.0 (`MTLLanguageVersion::Version4_0`) shader library on `dev`.
pub fn compile_metal4_library(
    dev: &MtlDevice,
    source: &str,
) -> std::result::Result<
    candle_metal_kernels::metal::Library,
    candle_metal_kernels::MetalKernelError,
> {
    let opts = MTLCompileOptions::new();
    opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
    dev.new_library_with_source(source, Some(&opts))
}

impl MetalCapabilities {
    /// Detect and cache the capabilities of the provided Metal device.
    pub fn detect(dev: &MtlDevice) -> &'static Self {
        CAPABILITIES_CACHE.get_or_init(|| Self::probe_device(dev))
    }

    /// Detect capabilities on the system default Metal device (if available).
    pub fn detect_default() -> Option<&'static Self> {
        let dev = MtlDevice::system_default()?;
        Some(Self::detect(&dev))
    }

    fn probe_device(dev: &MtlDevice) -> Self {
        crate::util::with_autorelease_pool(|| {
            let raw = dev.as_ref();
            let device_name = raw.name().to_string();
            let architecture = dev.architecture_name();

            let has_native_bfloat16 = raw.supportsFamily(MTLGPUFamily::Apple9);
            let has_apple10_m5 = raw.supportsFamily(MTLGPUFamily::Apple10);
            let has_metal4_family = raw.supportsFamily(MTLGPUFamily::Metal4);
            let max_threadgroup_memory = raw.maxThreadgroupMemoryLength();
            let recommended_max_working_set_size = dev.recommended_max_working_set_size();

            let has_tensor_ops_compiler = if has_apple10_m5 || has_metal4_family {
                compile_metal4_library(dev, M5_TENSOR_OPS_PROBE_SRC)
                    .and_then(|lib| {
                        let f = lib.get_function("glypr_m5_tensor_ops_probe", None)?;
                        dev.new_compute_pipeline_state_with_function(&f)
                    })
                    .is_ok()
            } else {
                false
            };

            Self {
                device_name,
                architecture,
                has_native_bfloat16,
                has_apple10_m5,
                has_metal4_family,
                has_tensor_ops_compiler,
                max_threadgroup_memory,
                recommended_max_working_set_size,
            }
        })
    }

    /// Returns `true` if M5 Neural Accelerator (`mpp::tensor_ops`) kernels should be used
    /// after accounting for hardware support, compiler probe, and `GLYPR_METAL_TIER`.
    #[inline]
    pub fn use_m5_tensor_ops(&self) -> bool {
        if !self.has_tensor_ops_compiler {
            return false;
        }
        match MetalTier::from_env() {
            MetalTier::Legacy => false,
            MetalTier::M5 => true,
            MetalTier::Auto => self.has_apple10_m5 && self.has_metal4_family,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use objc2_metal::MTLSize;

    #[test]
    fn test_metal_capabilities_detection_and_probe() -> anyhow::Result<()> {
        if !candle_core::utils::metal_is_available() {
            return Ok(());
        }
        let dev = Device::new_metal(0)?;
        let mdev = match &dev {
            Device::Metal(m) => m,
            _ => unreachable!(),
        };
        let caps = MetalCapabilities::detect(mdev.metal_device());
        println!("Detected MetalCapabilities: {:#?}", caps);
        assert!(!caps.device_name.is_empty());
        assert!(caps.max_threadgroup_memory >= 32768);

        if caps.has_tensor_ops_compiler {
            // Test executing a 32x32 bfloat x bfloat -> float matmul via mpp::tensor_ops::matmul2d!
            let lib = compile_metal4_library(mdev.metal_device(), M5_TENSOR_OPS_PROBE_SRC)?;
            let f = lib.get_function("glypr_m5_tensor_ops_probe", None)?;
            let pipeline = mdev
                .metal_device()
                .new_compute_pipeline_state_with_function(&f)?;

            let a_vec: Vec<f32> = (0..(32 * 32)).map(|i| ((i as f32) * 0.031).sin()).collect();
            let b_vec: Vec<f32> = (0..(32 * 32)).map(|i| ((i as f32) * 0.017).cos()).collect();
            let a = Tensor::from_vec(a_vec, (32, 32), &dev)?.to_dtype(DType::BF16)?;
            let b = Tensor::from_vec(b_vec, (32, 32), &dev)?.to_dtype(DType::BF16)?;
            let c = Tensor::zeros((32, 32), DType::F32, &dev)?;

            {
                let (a_s, _) = a.storage_and_layout();
                let (b_s, _) = b.storage_and_layout();
                let (c_s, _) = c.storage_and_layout();
                let a_buf = match &*a_s {
                    candle_core::Storage::Metal(m) => m.buffer(),
                    _ => unreachable!(),
                };
                let b_buf = match &*b_s {
                    candle_core::Storage::Metal(m) => m.buffer(),
                    _ => unreachable!(),
                };
                let c_buf = match &*c_s {
                    candle_core::Storage::Metal(m) => m.buffer(),
                    _ => unreachable!(),
                };

                let encoder = mdev.command_encoder()?;
                let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
                enc_ref.set_compute_pipeline_state(&pipeline);
                enc_ref.set_input_buffer(0, Some(a_buf), 0);
                enc_ref.set_input_buffer(1, Some(b_buf), 0);
                enc_ref.set_output_buffer(2, Some(c_buf), 0);
                enc_ref.dispatch_thread_groups(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            }

            let ref_c = a.to_dtype(DType::F32)?.matmul(&b.to_dtype(DType::F32)?)?;
            let diff: f32 = (&c - &ref_c)?.abs()?.max_all()?.to_scalar()?;
            println!("M5 tensor_ops 32x32 matmul max_diff vs F32 reference: {diff:.6e}");
            assert!(diff < 1e-4, "M5 tensor_ops matmul diff too large: {diff}");
        }
        Ok(())
    }
}

