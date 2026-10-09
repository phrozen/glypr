use anyhow::Result;
use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use glypr_core::{cosine_similarity, truncate_mrl, GlyprEmbedder, Task};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "glypr", version, about = "Native Rust Embedding & Retrieval Engine for EmbeddingGemma 2")]
struct Cli {
    /// Path to model file (.gguf) or directory containing model weights
    #[arg(short, long, global = true)]
    model: Option<PathBuf>,

    /// Path to model directory (alias for --model)
    #[arg(long, global = true)]
    model_dir: Option<PathBuf>,

    /// Optional path to multimodal projector GGUF (mmproj-*.gguf)
    #[arg(long, global = true)]
    mmproj: Option<PathBuf>,

    /// Use Apple Silicon Metal GPU acceleration
    #[arg(long, global = true)]
    metal: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate an embedding vector for text, image, or audio
    Embed {
        /// Text content to encode (or optional query prompt for image/audio)
        #[arg(default_value = "")]
        text: String,

        /// Path to an image file (.png, .jpg, .webp)
        #[arg(long)]
        image: Option<PathBuf>,

        /// Path to an audio file (.wav)
        #[arg(long)]
        audio: Option<PathBuf>,

        /// Task prompt name: SearchQuery, CodeRetrieval, QuestionAnswering, Document
        #[arg(short, long, default_value = "SearchQuery")]
        task: String,

        /// Document title (if encoding a titled document)
        #[arg(long)]
        title: Option<String>,

        /// Matryoshka dimension truncation (768, 512, 256, 128)
        #[arg(short, long, default_value_t = 768)]
        dim: usize,
    },
    /// Compute semantic cosine similarity between two texts
    Sim {
        text1: String,
        text2: String,

        /// Matryoshka dimension truncation (768, 512, 256, 128)
        #[arg(short, long, default_value_t = 768)]
        dim: usize,
    },
    /// Launch an OpenAI-compatible HTTP embedding server
    Serve {
        /// Host address to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Maximum dynamic batch size
        #[arg(long, default_value_t = 32)]
        max_batch_size: usize,
    },
}

enum ModelTarget {
    Gguf(PathBuf),
    Directory(PathBuf),
}

fn resolve_model(explicit_model: Option<PathBuf>, explicit_dir: Option<PathBuf>) -> Result<ModelTarget> {
    let explicit = explicit_model.or(explicit_dir);
    if let Some(p) = explicit {
        if p.is_file() {
            return Ok(ModelTarget::Gguf(p));
        } else if p.is_dir() {
            let q8_candidate = p.join("embeddinggemma-2-Q8_0.gguf");
            if q8_candidate.exists() {
                return Ok(ModelTarget::Gguf(q8_candidate));
            }
            let bf16_candidate = p.join("embeddinggemma-2-BF16.gguf");
            if bf16_candidate.exists() {
                return Ok(ModelTarget::Gguf(bf16_candidate));
            }
            return Ok(ModelTarget::Directory(p));
        }
    }

    // Try resolving GGUF via Hub API (checks cache, downloads if needed)
    if let Ok(p) = glypr_core::hub::resolve_file("ggml-org/embeddinggemma-2-GGUF", "embeddinggemma-2-Q8_0.gguf") {
        if p.exists() {
            return Ok(ModelTarget::Gguf(p));
        }
    }
    if let Ok(p) = glypr_core::hub::resolve_default_gguf() {
        if p.exists() {
            return Ok(ModelTarget::Gguf(p));
        }
    }
    if let Ok(p) = glypr_core::hub::resolve_google_weights() {
        if let Some(parent) = p.parent() {
            return Ok(ModelTarget::Directory(parent.to_path_buf()));
        }
    }

    anyhow::bail!("Could not resolve EmbeddingGemma 2 model. Specify --model <PATH.gguf> or configure Hugging Face Hub.")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let target = resolve_model(cli.model, cli.model_dir)?;

    let (device, dtype) = if cli.metal {
        #[cfg(feature = "metal")]
        {
            (Device::new_metal(0)?, DType::BF16)
        }
        #[cfg(not(feature = "metal"))]
        {
            anyhow::bail!("Binary was compiled without --features metal");
        }
    } else {
        (Device::Cpu, DType::F32)
    };

    let mut embedder = match target {
        ModelTarget::Gguf(path) => {
            GlyprEmbedder::load_gguf(&path, &device, dtype)?
        }
        ModelTarget::Directory(dir) => {
            GlyprEmbedder::load(&dir, &device, dtype)?
        }
    };
    embedder.set_attention_mode(glypr_core::AttentionMode::ChunkedSliding);

    if let Some(mmproj_path) = cli.mmproj {
        embedder.attach_multimodal_projector(&mmproj_path)?;
        println!("✓ Attached multimodal projector: {:?}", mmproj_path);
    }

    match cli.command {
        Commands::Embed { text, image, audio, task, title, dim } => {
            let prompt = if text.trim().is_empty() {
                None
            } else {
                Some(text.trim())
            };

            let mut vec = if let Some(img_path) = image {
                let img = image::open(&img_path)
                    .map_err(|e| anyhow::anyhow!("Failed to open image {:?}: {}", img_path, e))?;
                embedder.embed_image(&img, prompt)?
            } else if let Some(aud_path) = audio {
                let mut reader = hound::WavReader::open(&aud_path)
                    .map_err(|e| anyhow::anyhow!("Failed to open WAV audio {:?}: {}", aud_path, e))?;
                let samples: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32767.0).collect();
                embedder.embed_audio(&samples, prompt)?
            } else {
                let task_enum = Task::from_name_and_title(&task, title.as_deref());
                embedder.embed_text(&text, task_enum)?
            };

            if dim < 768 {
                vec = truncate_mrl(&vec, dim)?;
            }

            println!("Embedding dimension: {}", vec.len());
            let preview = if vec.len() > 6 {
                format!("[{:.4}, {:.4}, {:.4}, ..., {:.4}, {:.4}, {:.4}]",
                    vec[0], vec[1], vec[2], vec[vec.len()-3], vec[vec.len()-2], vec[vec.len()-1])
            } else {
                format!("{:?}", vec)
            };
            println!("Vector preview: {}", preview);
        }
        Commands::Sim { text1, text2, dim } => {
            let mut vec1 = embedder.embed_text(&text1, Task::SearchQuery)?;
            let mut vec2 = embedder.embed_text(&text2, Task::Document { title: None })?;
            if dim < 768 {
                vec1 = truncate_mrl(&vec1, dim)?;
                vec2 = truncate_mrl(&vec2, dim)?;
            }
            let sim = cosine_similarity(&vec1, &vec2);
            println!("Cosine Similarity ({}d): {:.6}", dim, sim);
        }
        Commands::Serve { host, port, max_batch_size } => {
            let config = glypr_server::ServerConfig {
                host,
                port,
                max_batch_size,
                model_id: "embeddinggemma-2".to_string(),
            };
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                glypr_server::run_server(config, embedder).await
            })?;
        }
    }

    Ok(())
}
