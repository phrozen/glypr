use crate::error::GlyprError;
use anyhow::{Context, Result};
use hf_hub::{Cache, api::sync::Api, api::sync::ApiBuilder};
use std::path::{Path, PathBuf};

pub const DEFAULT_GGUF_REPO: &str = "ggml-org/embeddinggemma-2-GGUF";

pub const GGUF_BF16_FILE: &str = "embeddinggemma-2-BF16.gguf";
pub const MMPROJ_BF16_FILE: &str = "mmproj-embeddinggemma-2-BF16.gguf";

pub const GGUF_Q8_0_FILE: &str = "embeddinggemma-2-Q8_0.gguf";
pub const MMPROJ_Q8_0_FILE: &str = "mmproj-embeddinggemma-2-Q8_0.gguf";

pub const DEFAULT_GGUF_FILE: &str = GGUF_BF16_FILE;
pub const DEFAULT_MMPROJ_FILE: &str = MMPROJ_BF16_FILE;

pub const GOOGLE_OFFICIAL_REPO: &str = "google/embeddinggemma-2";
pub const GOOGLE_SAFETENSORS_FILE: &str = "model.safetensors";
pub const GOOGLE_TOKENIZER_FILE: &str = "tokenizer.json";

/// Returns the base directory of the active Hugging Face cache.
pub fn get_hf_cache_dir() -> PathBuf {
    if let Ok(hub_cache) = std::env::var("HF_HUB_CACHE") {
        PathBuf::from(hub_cache)
    } else if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home).join("hub")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache/huggingface/hub")
    } else {
        PathBuf::from(".cache/huggingface/hub")
    }
}

/// Returns a configured Hugging Face cache instance respecting:
/// 1. `HF_HUB_CACHE` environment variable
/// 2. `HF_HOME` environment variable (via `Cache::from_env()`)
/// 3. Default Hugging Face cache location (`~/.cache/huggingface/hub`)
pub fn get_hf_cache() -> Cache {
    Cache::new(get_hf_cache_dir())
}

/// Check if a canonical file is already present in the local Hugging Face cache (zero network).
pub fn check_cached_file(repo_id: &str, filename: &str) -> Option<PathBuf> {
    let cache = get_hf_cache();
    let repo = cache.model(repo_id.to_string());
    repo.get(filename)
}

/// Returns `true` when `HF_HUB_OFFLINE` is set to `"1"`, `"true"`, or `"yes"` (case-insensitive).
pub fn is_offline_mode() -> bool {
    std::env::var("HF_HUB_OFFLINE")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Builds a synchronous Hugging Face Hub API client configured with the active cache.
pub fn get_hf_api() -> Result<Api> {
    let cache = get_hf_cache();
    ApiBuilder::from_cache(cache)
        .build()
        .context("Failed to initialize Hugging Face Hub API client")
}

/// Resolves a file from a Hugging Face model repository with explicit offline control.
///
/// Checks the local Hugging Face cache first (zero network I/O). If the file is missing and
/// `offline || is_offline_mode()`, fails immediately with an actionable [`GlyprError::OfflineCacheMiss`]
/// without attempting any network requests.
pub fn resolve_file_with_options(repo_id: &str, filename: &str, offline: bool) -> Result<PathBuf> {
    if let Some(cached) = check_cached_file(repo_id, filename) {
        return Ok(cached);
    }

    if offline || is_offline_mode() {
        let cache_dir = get_hf_cache_dir();
        let err = GlyprError::OfflineCacheMiss {
            repo_id: repo_id.to_string(),
            filename: filename.to_string(),
            cache_dir: cache_dir.clone(),
        };
        return Err(anyhow::Error::new(err)).with_context(|| {
            format!(
                "File '{filename}' from '{repo_id}' is not in local HF cache ({}) and offline mode is enabled. \
                 Run 'glypr pull' online or pass --model-file <PATH>.",
                cache_dir.display()
            )
        });
    }

    let api = get_hf_api()?;
    let repo = api.model(repo_id.to_string());
    repo.get(filename)
        .with_context(|| format!("Failed to resolve '{filename}' from repository '{repo_id}'"))
}

/// Resolves a file from a Hugging Face model repository.
///
/// If the file is already cached locally, it returns the local file path immediately (zero network overhead).
/// If the file is not yet cached locally, it downloads it from the hub (unless `HF_HUB_OFFLINE=1`).
pub fn resolve_file(repo_id: &str, filename: &str) -> Result<PathBuf> {
    resolve_file_with_options(repo_id, filename, false)
}

/// Resolves the canonical base model GGUF (BF16 default or Q8_0) with explicit offline control.
pub fn resolve_canonical_model_with_options(use_q8: bool, offline: bool) -> Result<PathBuf> {
    let filename = if use_q8 {
        GGUF_Q8_0_FILE
    } else {
        GGUF_BF16_FILE
    };
    resolve_file_with_options(DEFAULT_GGUF_REPO, filename, offline)
}

/// Resolves the canonical base model GGUF (BF16 default or Q8_0).
pub fn resolve_canonical_model(use_q8: bool) -> Result<PathBuf> {
    resolve_canonical_model_with_options(use_q8, false)
}

/// Resolves the canonical multimodal projector GGUF (BF16 default or Q8_0) with explicit offline control.
pub fn resolve_canonical_mmproj_with_options(use_q8: bool, offline: bool) -> Result<PathBuf> {
    let filename = if use_q8 {
        MMPROJ_Q8_0_FILE
    } else {
        MMPROJ_BF16_FILE
    };
    resolve_file_with_options(DEFAULT_GGUF_REPO, filename, offline)
}

/// Resolves the canonical multimodal projector GGUF (BF16 default or Q8_0).
pub fn resolve_canonical_mmproj(use_q8: bool) -> Result<PathBuf> {
    resolve_canonical_mmproj_with_options(use_q8, false)
}

/// Resolves the default BF16 base model GGUF (`ggml-org/embeddinggemma-2-GGUF/embeddinggemma-2-BF16.gguf`).
pub fn resolve_default_gguf() -> Result<PathBuf> {
    resolve_canonical_model(false)
}

/// Resolves the default BF16 multimodal projector GGUF (`ggml-org/embeddinggemma-2-GGUF/mmproj-embeddinggemma-2-BF16.gguf`).
pub fn resolve_default_mmproj() -> Result<PathBuf> {
    resolve_canonical_mmproj(false)
}

/// Resolves Google's official `model.safetensors` (`google/embeddinggemma-2/model.safetensors`) with offline control.
pub fn resolve_google_weights_with_options(offline: bool) -> Result<PathBuf> {
    resolve_file_with_options(GOOGLE_OFFICIAL_REPO, GOOGLE_SAFETENSORS_FILE, offline)
}

/// Resolves Google's official `model.safetensors` (`google/embeddinggemma-2/model.safetensors`).
pub fn resolve_google_weights() -> Result<PathBuf> {
    resolve_google_weights_with_options(false)
}

/// Resolves Google's official `tokenizer.json` (`google/embeddinggemma-2/tokenizer.json`) with offline control.
pub fn resolve_google_tokenizer_with_options(offline: bool) -> Result<PathBuf> {
    resolve_file_with_options(GOOGLE_OFFICIAL_REPO, GOOGLE_TOKENIZER_FILE, offline)
}

/// Resolves Google's official `tokenizer.json` (`google/embeddinggemma-2/tokenizer.json`).
pub fn resolve_google_tokenizer() -> Result<PathBuf> {
    resolve_google_tokenizer_with_options(false)
}

/// Returns the default user-level CoreML models directory.
///
/// Precedence:
/// 1. `$XDG_DATA_HOME/glypr/coreml`
/// 2. macOS: `~/Library/Application Support/glypr/coreml`
/// 3. Other platforms / fallback: `~/.cache/glypr/coreml`
pub fn get_user_coreml_dir() -> PathBuf {
    if let Ok(xdg_data) = std::env::var("XDG_DATA_HOME")
        && !xdg_data.trim().is_empty()
    {
        return PathBuf::from(xdg_data.trim()).join("glypr/coreml");
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        #[cfg(target_os = "macos")]
        {
            return PathBuf::from(home.trim()).join("Library/Application Support/glypr/coreml");
        }
        #[cfg(not(target_os = "macos"))]
        {
            return PathBuf::from(home.trim()).join(".cache/glypr/coreml");
        }
    }
    PathBuf::from(".cache/glypr/coreml")
}

/// Resolve the CoreML models directory following the deterministic lookup chain:
///
/// 1. Explicit `--coreml-dir` path (returned unconditionally if provided so callers can validate it)
/// 2. `GLYPR_COREML_DIR` environment variable (if set and non-empty)
/// 3. `./models/coreml` (repo-local checkout, when directory exists)
/// 4. User data/cache directory (`get_user_coreml_dir()`, or `~/.cache/glypr/coreml` if it exists)
pub fn resolve_coreml_dir(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf());
    }

    if let Ok(env_dir) = std::env::var("GLYPR_COREML_DIR") {
        let trimmed = env_dir.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }

    let local = PathBuf::from("models/coreml");
    if local.is_dir() {
        return Some(local);
    }

    let user_dir = get_user_coreml_dir();
    if user_dir.is_dir() {
        return Some(user_dir);
    }

    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        let fallback_cache = PathBuf::from(home.trim()).join(".cache/glypr/coreml");
        if fallback_cache.is_dir() {
            return Some(fallback_cache);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_file_with_options_offline_cache_miss() {
        let err = resolve_file_with_options(
            "glypr-nonexistent-org/nonexistent-repo-xyz",
            "missing-model.gguf",
            true,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("offline mode is enabled"), "got: {msg}");
        assert!(msg.contains("glypr pull"), "got: {msg}");
        assert!(
            err.downcast_ref::<GlyprError>().is_some(),
            "expected GlyprError downcast"
        );
    }

    #[test]
    fn test_resolve_coreml_dir_explicit_takes_precedence() {
        let explicit = Path::new("/tmp/custom-glypr-coreml-dir");
        assert_eq!(
            resolve_coreml_dir(Some(explicit)),
            Some(explicit.to_path_buf())
        );
    }
}
