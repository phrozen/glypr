use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, api::sync::ApiBuilder, Cache};
use std::path::PathBuf;

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

/// Builds a synchronous Hugging Face Hub API client configured with the active cache.
pub fn get_hf_api() -> Result<Api> {
    let cache = get_hf_cache();
    ApiBuilder::from_cache(cache)
        .build()
        .context("Failed to initialize Hugging Face Hub API client")
}

/// Resolves a file from a Hugging Face model repository.
///
/// If the file is already cached locally, it returns the local file path immediately (zero network overhead).
/// If the file is not yet cached locally, it downloads it from the hub.
pub fn resolve_file(repo_id: &str, filename: &str) -> Result<PathBuf> {
    let api = get_hf_api()?;
    let repo = api.model(repo_id.to_string());
    repo.get(filename)
        .with_context(|| format!("Failed to resolve '{filename}' from repository '{repo_id}'"))
}

/// Resolves the canonical base model GGUF (BF16 default or Q8_0).
pub fn resolve_canonical_model(use_q8: bool) -> Result<PathBuf> {
    let filename = if use_q8 {
        GGUF_Q8_0_FILE
    } else {
        GGUF_BF16_FILE
    };
    resolve_file(DEFAULT_GGUF_REPO, filename)
}

/// Resolves the canonical multimodal projector GGUF (BF16 default or Q8_0).
pub fn resolve_canonical_mmproj(use_q8: bool) -> Result<PathBuf> {
    let filename = if use_q8 {
        MMPROJ_Q8_0_FILE
    } else {
        MMPROJ_BF16_FILE
    };
    resolve_file(DEFAULT_GGUF_REPO, filename)
}

/// Resolves the default BF16 base model GGUF (`ggml-org/embeddinggemma-2-GGUF/embeddinggemma-2-BF16.gguf`).
pub fn resolve_default_gguf() -> Result<PathBuf> {
    resolve_canonical_model(false)
}

/// Resolves the default BF16 multimodal projector GGUF (`ggml-org/embeddinggemma-2-GGUF/mmproj-embeddinggemma-2-BF16.gguf`).
pub fn resolve_default_mmproj() -> Result<PathBuf> {
    resolve_canonical_mmproj(false)
}

/// Resolves Google's official `model.safetensors` (`google/embeddinggemma-2/model.safetensors`).
pub fn resolve_google_weights() -> Result<PathBuf> {
    resolve_file(GOOGLE_OFFICIAL_REPO, GOOGLE_SAFETENSORS_FILE)
}

/// Resolves Google's official `tokenizer.json` (`google/embeddinggemma-2/tokenizer.json`).
pub fn resolve_google_tokenizer() -> Result<PathBuf> {
    resolve_file(GOOGLE_OFFICIAL_REPO, GOOGLE_TOKENIZER_FILE)
}
