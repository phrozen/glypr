use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, api::sync::ApiBuilder, Cache};
use std::path::PathBuf;

pub const DEFAULT_GGUF_REPO: &str = "ggml-org/embeddinggemma-2-GGUF";
pub const DEFAULT_GGUF_FILE: &str = "embeddinggemma-2-BF16.gguf";
pub const DEFAULT_MMPROJ_FILE: &str = "mmproj-embeddinggemma-2-BF16.gguf";

pub const GOOGLE_OFFICIAL_REPO: &str = "google/embeddinggemma-2";
pub const GOOGLE_SAFETENSORS_FILE: &str = "model.safetensors";
pub const GOOGLE_TOKENIZER_FILE: &str = "tokenizer.json";

/// Returns a configured Hugging Face cache instance respecting:
/// 1. `HF_HUB_CACHE` environment variable
/// 2. `HF_HOME` environment variable (via `Cache::from_env()`)
/// 3. Default Hugging Face cache location (`~/.cache/huggingface/hub`)
pub fn get_hf_cache() -> Cache {
    if let Ok(hub_cache) = std::env::var("HF_HUB_CACHE") {
        Cache::new(PathBuf::from(hub_cache))
    } else {
        Cache::from_env()
    }
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

/// Resolves the default BF16 base model GGUF (`ggml-org/embeddinggemma-2-GGUF/embeddinggemma-2-BF16.gguf`).
pub fn resolve_default_gguf() -> Result<PathBuf> {
    resolve_file(DEFAULT_GGUF_REPO, DEFAULT_GGUF_FILE)
}

/// Resolves the default BF16 multimodal projector GGUF (`ggml-org/embeddinggemma-2-GGUF/mmproj-embeddinggemma-2-BF16.gguf`).
pub fn resolve_default_mmproj() -> Result<PathBuf> {
    resolve_file(DEFAULT_GGUF_REPO, DEFAULT_MMPROJ_FILE)
}

/// Resolves Google's official `model.safetensors` (`google/embeddinggemma-2/model.safetensors`).
pub fn resolve_google_weights() -> Result<PathBuf> {
    resolve_file(GOOGLE_OFFICIAL_REPO, GOOGLE_SAFETENSORS_FILE)
}

/// Resolves Google's official `tokenizer.json` (`google/embeddinggemma-2/tokenizer.json`).
pub fn resolve_google_tokenizer() -> Result<PathBuf> {
    resolve_file(GOOGLE_OFFICIAL_REPO, GOOGLE_TOKENIZER_FILE)
}
