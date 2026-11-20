//! `glypr` - Native Embedding & Retrieval Engine for EmbeddingGemma 2
//!
//! Re-exports the complete engine from `glypr_core` along with the testable CLI argument
//! definitions and pure validation helpers in [`cli`].

pub mod cli;

pub use cli::*;
pub use glypr_core::*;
