#![allow(dead_code)]

use anyhow::Result;
use candle_core::{DType, Device};
use glypr_core::GlyprEmbedder;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

pub type SharedEmbedder = Arc<Mutex<GlyprEmbedder>>;
type CachedEmbedder = std::result::Result<Option<SharedEmbedder>, String>;

/// Returns a Metal device if both `metal` is enabled and `MTLResidencySetDescriptor` is
/// safely available on the host OS, otherwise falls back to `Device::Cpu`.
pub fn safe_test_device() -> Device {
    #[cfg(feature = "metal")]
    if glypr_core::metal_kernels::is_metal_device_available()
        && let Ok(dev) = Device::new_metal(0)
    {
        return dev;
    }
    Device::Cpu
}

/// Returns `DType::BF16` for Metal devices and `DType::F32` for CPU.
pub fn default_dtype_for_device(device: &Device) -> DType {
    if device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Returns `true` when `GLYPR_ALLOW_MISSING_WEIGHTS` is set to `"1"` or `"true"`.
pub fn allow_missing_weights() -> bool {
    matches!(
        std::env::var("GLYPR_ALLOW_MISSING_WEIGHTS")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true"
    )
}

/// Returns `true` when `err` represents a missing model file or Hugging Face Hub resolution
/// failure rather than a corrupt weight file or runtime inference error.
pub fn is_missing_weight_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        if let Some(io_err) = cause.downcast_ref::<std::io::Error>()
            && io_err.kind() == std::io::ErrorKind::NotFound
        {
            return true;
        }
        let msg = cause.to_string();
        msg.contains("Failed to resolve '")
            || msg.contains("Failed to initialize Hugging Face Hub API client")
            || msg.contains("No such file or directory")
    })
}

/// Converts missing-weight / Hub resolution errors into `Ok(None)` when
/// [`allow_missing_weights()`] is enabled, while preserving all other errors.
pub fn try_or_skip_missing<T>(res: Result<T>, context_label: &str) -> Result<Option<T>> {
    match res {
        Ok(val) => Ok(Some(val)),
        Err(err) if allow_missing_weights() && is_missing_weight_error(&err) => {
            eprintln!("Skipping {context_label} (GLYPR_ALLOW_MISSING_WEIGHTS=1): {err:#}");
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

/// Resolves a weight-dependent `Result<T>` expression, returning `Ok(())` cleanly from the
/// enclosing test function when `GLYPR_ALLOW_MISSING_WEIGHTS` permits missing weight errors.
#[allow(unused_macros)]
macro_rules! require_weights {
    ($expr:expr $(,)?) => {
        match $crate::common::try_or_skip_missing($expr, stringify!($expr))? {
            Some(val) => val,
            None => return Ok(()),
        }
    };
    ($expr:expr, $label:expr $(,)?) => {
        match $crate::common::try_or_skip_missing($expr, $label)? {
            Some(val) => val,
            None => return Ok(()),
        }
    };
}
#[allow(unused_imports)]
pub(crate) use require_weights;

/// Returns the workspace `tests/golden_vectors` directory.
pub fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("glypr-core crate should have crates/ parent")
        .parent()
        .expect("crates/ should have workspace root parent")
        .join("tests/golden_vectors")
}

fn load_gguf_fixture(device: Device, use_q8: bool, label: &str) -> Result<Option<SharedEmbedder>> {
    let base_gguf = match try_or_skip_missing(glypr_core::hub::resolve_canonical_model(use_q8), label)?
    {
        Some(path) => path,
        None => return Ok(None),
    };

    let mut embedder = if use_q8 {
        GlyprEmbedder::load_gguf_q8(&base_gguf, &device)?
    } else {
        GlyprEmbedder::load_gguf_bf16(&base_gguf, &device)?
    };

    match glypr_core::hub::resolve_canonical_mmproj(use_q8) {
        Ok(mmproj) if mmproj.exists() => {
            embedder.attach_multimodal_projector(&mmproj)?;
        }
        Ok(_) => {}
        Err(err) if allow_missing_weights() && is_missing_weight_error(&err) => {
            eprintln!("Note: optional mmproj not cached for {label}: {err:#}");
        }
        Err(err) => return Err(err),
    }

    Ok(Some(Arc::new(Mutex::new(embedder))))
}

fn load_safetensors_fixture(device: Device, label: &str) -> Result<Option<SharedEmbedder>> {
    let weights_path =
        match try_or_skip_missing(glypr_core::hub::resolve_google_weights(), label)? {
            Some(path) => path,
            None => return Ok(None),
        };
    let snapshot_dir = weights_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Invalid Safetensors path: {}", weights_path.display()))?;
    let dtype = default_dtype_for_device(&device);
    let embedder = GlyprEmbedder::load(snapshot_dir, &device, dtype)?;
    Ok(Some(Arc::new(Mutex::new(embedder))))
}

fn get_or_init_shared(
    cell: &OnceLock<CachedEmbedder>,
    init: impl FnOnce() -> Result<Option<SharedEmbedder>>,
) -> Result<Option<SharedEmbedder>> {
    let cached = cell.get_or_init(|| init().map_err(|e| format!("{e:#}")));
    match cached {
        Ok(Some(embedder)) => Ok(Some(Arc::clone(embedder))),
        Ok(None) => Ok(None),
        Err(msg) => Err(anyhow::anyhow!("{msg}")),
    }
}

/// Process-wide `OnceLock`-cached BF16 GGUF embedder on [`safe_test_device()`] (with optional
/// BF16 multimodal projector attached).
pub fn shared_gguf_bf16_metal() -> Result<Option<SharedEmbedder>> {
    static CELL: OnceLock<CachedEmbedder> = OnceLock::new();
    get_or_init_shared(&CELL, || {
        load_gguf_fixture(safe_test_device(), false, "shared_gguf_bf16_metal")
    })
}

/// Process-wide `OnceLock`-cached Q8_0 GGUF embedder on [`safe_test_device()`] (with optional
/// Q8_0 multimodal projector attached).
pub fn shared_gguf_q8_metal() -> Result<Option<SharedEmbedder>> {
    static CELL: OnceLock<CachedEmbedder> = OnceLock::new();
    get_or_init_shared(&CELL, || {
        load_gguf_fixture(safe_test_device(), true, "shared_gguf_q8_metal")
    })
}

/// Process-wide `OnceLock`-cached Q8_0 GGUF embedder on CPU (with optional Q8_0 multimodal
/// projector attached).
pub fn shared_gguf_q8_cpu() -> Result<Option<SharedEmbedder>> {
    static CELL: OnceLock<CachedEmbedder> = OnceLock::new();
    get_or_init_shared(&CELL, || {
        load_gguf_fixture(Device::Cpu, true, "shared_gguf_q8_cpu")
    })
}

/// Process-wide `OnceLock`-cached Safetensors F32 embedder on CPU.
pub fn shared_safetensors_cpu() -> Result<Option<SharedEmbedder>> {
    static CELL: OnceLock<CachedEmbedder> = OnceLock::new();
    get_or_init_shared(&CELL, || {
        load_safetensors_fixture(Device::Cpu, "shared_safetensors_cpu")
    })
}

/// Process-wide `OnceLock`-cached Safetensors embedder on [`safe_test_device()`].
pub fn shared_safetensors_metal() -> Result<Option<SharedEmbedder>> {
    static CELL: OnceLock<CachedEmbedder> = OnceLock::new();
    get_or_init_shared(&CELL, || {
        load_safetensors_fixture(safe_test_device(), "shared_safetensors_metal")
    })
}
