use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::{Parser, Subcommand, ValueEnum};
use glypr_core::{
    hub::{
        check_cached_file, resolve_canonical_mmproj, resolve_canonical_model, resolve_file,
        DEFAULT_GGUF_REPO, GGUF_BF16_FILE, GGUF_Q8_0_FILE, MMPROJ_BF16_FILE, MMPROJ_Q8_0_FILE,
    },
    truncate_mrl, GlyprEmbedder, ModalityPart, Task,
};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
enum DeviceArg {
    #[default]
    Auto,
    Metal,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
enum EncodingArg {
    #[default]
    Float,
    Int8,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
enum OutputArg {
    #[default]
    Preview,
    Json,
    Raw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
enum AttentionArg {
    #[default]
    Auto,
    Flash,
    Chunked,
    Standard,
}

#[derive(Parser)]
#[command(
    name = "glypr",
    version,
    about = "Native Rust Embedding & Retrieval Engine for EmbeddingGemma 2"
)]
struct Cli {
    /// Local model file (.gguf) or directory override (disables automatic downloads)
    #[arg(short, long, global = true)]
    model: Option<PathBuf>,

    /// Path to model directory (alias for --model)
    #[arg(long, global = true)]
    model_dir: Option<PathBuf>,

    /// Local multimodal projector (.gguf) override
    #[arg(long, global = true)]
    mmproj: Option<PathBuf>,

    /// Use 8-bit quantized weights (Q8_0) instead of default BF16
    #[arg(long, global = true)]
    q8: bool,

    /// Hardware acceleration device: auto, metal, cpu
    #[arg(long, global = true, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Attention kernel implementation: auto (Flash on Metal, Chunked on CPU), flash, chunked, standard
    #[arg(long, global = true, value_enum, default_value_t = AttentionArg::Auto)]
    attention: AttentionArg,

    /// Legacy flag for Metal acceleration (alias for --device metal)
    #[arg(long, global = true, hide = true)]
    metal: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate an embedding vector for text, image, audio, or interleaved inputs
    Embed {
        /// Positional text and media paths (interleaved). If omitted or '-', reads from STDIN.
        #[arg(value_name = "INPUTS")]
        inputs: Vec<String>,

        /// Task prompt name: Raw, SearchQuery, CodeRetrieval, QuestionAnswering, Document
        #[arg(short, long, default_value = "Raw")]
        task: String,

        /// Document title (if encoding a titled document)
        #[arg(long)]
        title: Option<String>,

        /// Matryoshka dimension truncation (768, 512, 256, 128)
        #[arg(short, long, default_value_t = 768)]
        dim: usize,

        /// Output encoding format: float, int8, binary
        #[arg(short, long, value_enum, default_value_t = EncodingArg::Float)]
        encoding: EncodingArg,

        /// Output display mode: preview, json, raw (binary bytes)
        #[arg(short, long, value_enum, default_value_t = OutputArg::Preview)]
        output: OutputArg,

        /// Image token budget: 70, 140, 280, 560, 1120
        #[arg(long, default_value_t = 280)]
        image_token_budget: usize,
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

    /// Pre-fetch canonical GGUF models into Hugging Face cache
    Pull {
        /// Pull BF16 base weights (embeddinggemma-2-BF16.gguf)
        #[arg(long)]
        bf16: bool,

        /// Pull BF16 multimodal projector (mmproj-embeddinggemma-2-BF16.gguf)
        #[arg(long)]
        bf16_mmproj: bool,

        /// Pull Q8_0 base weights (embeddinggemma-2-Q8_0.gguf)
        #[arg(long)]
        q8: bool,

        /// Pull Q8_0 multimodal projector (mmproj-embeddinggemma-2-Q8_0.gguf)
        #[arg(long)]
        q8_mmproj: bool,

        /// Pull all 4 canonical models (default if no specific flags provided)
        #[arg(long)]
        all: bool,
    },

    /// Display cache status, hardware detection, and engine capabilities
    Info,
}

enum ModelTarget {
    Gguf(PathBuf),
    Directory(PathBuf),
}

fn resolve_model(
    explicit_model: Option<PathBuf>,
    explicit_dir: Option<PathBuf>,
    use_q8: bool,
) -> Result<ModelTarget> {
    let explicit = explicit_model.or(explicit_dir);
    if let Some(p) = explicit {
        if p.is_file() {
            return Ok(ModelTarget::Gguf(p));
        } else if p.is_dir() {
            let q8_candidate = p.join(GGUF_Q8_0_FILE);
            if q8_candidate.exists() {
                return Ok(ModelTarget::Gguf(q8_candidate));
            }
            let bf16_candidate = p.join(GGUF_BF16_FILE);
            if bf16_candidate.exists() {
                return Ok(ModelTarget::Gguf(bf16_candidate));
            }
            return Ok(ModelTarget::Directory(p));
        }
    }

    // Default: resolve canonical GGUF from official repo
    let path = resolve_canonical_model(use_q8)?;
    Ok(ModelTarget::Gguf(path))
}

fn resolve_device_and_dtype(device_arg: DeviceArg, legacy_metal: bool) -> Result<(Device, DType)> {
    let use_metal = match device_arg {
        DeviceArg::Metal => true,
        DeviceArg::Cpu => false,
        DeviceArg::Auto => {
            if legacy_metal {
                true
            } else {
                #[cfg(feature = "metal")]
                {
                    candle_core::utils::metal_is_available()
                }
                #[cfg(not(feature = "metal"))]
                {
                    false
                }
            }
        }
    };

    if use_metal {
        #[cfg(feature = "metal")]
        {
            Ok((Device::new_metal(0)?, DType::BF16))
        }
        #[cfg(not(feature = "metal"))]
        {
            anyhow::bail!("Metal acceleration requested but binary was compiled without --features metal");
        }
    } else {
        Ok((Device::Cpu, DType::F32))
    }
}

fn resolve_attention_mode(arg: AttentionArg, device: &Device) -> Result<glypr_core::AttentionMode> {
    match arg {
        AttentionArg::Auto => {
            #[cfg(feature = "metal")]
            if device.is_metal() {
                return Ok(glypr_core::AttentionMode::MetalFlash);
            }
            Ok(glypr_core::AttentionMode::ChunkedSliding)
        }
        AttentionArg::Flash => {
            #[cfg(feature = "metal")]
            if device.is_metal() {
                return Ok(glypr_core::AttentionMode::MetalFlash);
            }
            anyhow::bail!(
                "Flash attention requested but not supported on device {:?}. Flash attention requires Metal GPU (compiled with --features metal).",
                device
            );
        }
        AttentionArg::Chunked => Ok(glypr_core::AttentionMode::ChunkedSliding),
        AttentionArg::Standard => Ok(glypr_core::AttentionMode::Standard),
    }
}

enum OwnedInputItem {
    Text(String),
    Image(image::DynamicImage),
    Audio(Vec<f32>),
}

fn parse_interleaved_inputs(inputs: Vec<String>) -> Result<(Vec<OwnedInputItem>, bool)> {
    // Check if input should be read from STDIN
    if inputs.is_empty() || (inputs.len() == 1 && inputs[0] == "-") {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .context("Failed to read text from STDIN")?;
        let trimmed = buffer.trim();
        if trimmed.is_empty() {
            anyhow::bail!("No input provided via arguments or STDIN");
        }
        return Ok((vec![OwnedInputItem::Text(trimmed.to_string())], false));
    }

    let mut parts = Vec::new();
    let mut has_media = false;

    for arg in inputs {
        let p = Path::new(&arg);
        if p.is_file() {
            let ext = p
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext == "png" {
                let img =
                    image::open(p).with_context(|| format!("Failed to open PNG image {:?}", p))?;
                parts.push(OwnedInputItem::Image(img));
                has_media = true;
            } else if ext == "wav" {
                let mut reader = hound::WavReader::open(p)
                    .with_context(|| format!("Failed to open WAV audio {:?}", p))?;
                let samples: Vec<f32> = reader
                    .samples::<i16>()
                    .map(|s| s.unwrap_or(0) as f32 / 32767.0)
                    .collect();
                parts.push(OwnedInputItem::Audio(samples));
                has_media = true;
            } else {
                anyhow::bail!(
                    "Unsupported media file format '.{}' for {:?}. V1 supports .png images and .wav audio files.",
                    ext,
                    p
                );
            }
        } else {
            parts.push(OwnedInputItem::Text(arg));
        }
    }

    Ok((parts, has_media))
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle Pull command (runs without loading weights into GPU/RAM)
    if let Commands::Pull {
        bf16,
        bf16_mmproj,
        q8,
        q8_mmproj,
        all,
    } = cli.command
    {
        let pull_all = all || (!bf16 && !bf16_mmproj && !q8 && !q8_mmproj);
        let mut targets = Vec::new();

        if pull_all || bf16 {
            targets.push(GGUF_BF16_FILE);
        }
        if pull_all || bf16_mmproj {
            targets.push(MMPROJ_BF16_FILE);
        }
        if pull_all || q8 {
            targets.push(GGUF_Q8_0_FILE);
        }
        if pull_all || q8_mmproj {
            targets.push(MMPROJ_Q8_0_FILE);
        }

        println!("Pulling canonical GGUF models from '{}'...", DEFAULT_GGUF_REPO);
        for file in targets {
            print!("  - Resolving {} ... ", file);
            std::io::stdout().flush()?;
            let path = resolve_file(DEFAULT_GGUF_REPO, file)?;
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            println!("✓ ({})", format_bytes(size));
        }
        println!("All requested models verified in Hugging Face cache.");
        return Ok(());
    }

    // Handle Info command
    if let Commands::Info = cli.command {
        let cache_dir = glypr_core::hub::get_hf_cache_dir();
        println!("================================================================================");
        println!("  GLYPR ENGINE INFORMATION");
        println!("================================================================================");
        println!("Hugging Face Cache:   {}", cache_dir.display());
        println!("Default Repository:   {}", DEFAULT_GGUF_REPO);
        println!();
        println!("Canonical GGUF Models:");

        for (name, file) in [
            ("BF16 Base", GGUF_BF16_FILE),
            ("BF16 Projector", MMPROJ_BF16_FILE),
            ("Q8_0 Base", GGUF_Q8_0_FILE),
            ("Q8_0 Projector", MMPROJ_Q8_0_FILE),
        ] {
            if let Some(path) = check_cached_file(DEFAULT_GGUF_REPO, file) {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                println!("  [✓] {:<16}: {} ({})", name, file, format_bytes(size));
            } else {
                println!("  [ ] {:<16}: {} (Not cached)", name, file);
            }
        }

        println!();
        #[cfg(feature = "metal")]
        let metal_avail = candle_core::utils::metal_is_available();
        #[cfg(not(feature = "metal"))]
        let metal_avail = false;

        println!("Hardware & Backend:");
        if metal_avail {
            println!("  Device:             Apple Silicon Metal GPU (BF16)");
            println!("  Attention Kernel:   Metal Flash Attention (Fused Hardware Tile-Skipping, Auto)");
            println!("  Supported Kernels:  Flash (MetalFlash), Chunked (ChunkedSliding), Standard");
        } else {
            println!("  Device:             CPU SIMD (FP32)");
            println!("  Attention Kernel:   Chunked Sliding Attention (Window: 1024, Auto)");
            println!("  Supported Kernels:  Chunked (ChunkedSliding), Standard");
        }
        println!("  Max Context Window: 8,320 tokens");
        println!();
        println!("Supported Capabilities:");
        println!("  Native Dimensions:  768");
        println!("  MRL Truncations:    512, 256, 128");
        println!("  Quantizations:      Float32, Static INT8 ([-127, 127]), Binary 1-bit (POPCNT)");
        println!("  Vision Budgets:     70, 140 (Native 480x680), 280, 560, 1120");
        println!("================================================================================");
        return Ok(());
    }

    // Load Model & Engine
    let target = resolve_model(cli.model, cli.model_dir, cli.q8)?;
    let (device, dtype) = resolve_device_and_dtype(cli.device, cli.metal)?;

    let mut embedder = match target {
        ModelTarget::Gguf(path) => GlyprEmbedder::load_gguf(&path, &device, dtype)?,
        ModelTarget::Directory(dir) => GlyprEmbedder::load(&dir, &device, dtype)?,
    };
    let attn_mode = resolve_attention_mode(cli.attention, &device)?;
    embedder.set_attention_mode(attn_mode);

    // Resolve Multimodal Projector
    let mmproj_override = cli.mmproj;

    match cli.command {
        Commands::Embed {
            inputs,
            task,
            title,
            dim,
            encoding,
            output,
            image_token_budget,
        } => {
            let (owned_parts, has_media) = parse_interleaved_inputs(inputs)?;

            // Attach multimodal projector if media is present
            if has_media && embedder.multimodal.is_none() {
                let proj_path = match mmproj_override {
                    Some(p) => p,
                    None => resolve_canonical_mmproj(cli.q8)?,
                };
                embedder.attach_multimodal_projector(&proj_path)?;
            }

            let parts: Vec<ModalityPart<'_>> = owned_parts
                .iter()
                .map(|part| match part {
                    OwnedInputItem::Text(s) => ModalityPart::Text(s.as_str()),
                    OwnedInputItem::Image(img) => ModalityPart::Image {
                        image: img,
                        budget: Some(image_token_budget),
                    },
                    OwnedInputItem::Audio(samples) => ModalityPart::Audio {
                        samples: samples.as_slice(),
                    },
                })
                .collect();

            let task_enum = Task::from_name_and_title(&task, title.as_deref());
            let mut vec = embedder.embed_interleaved(&parts, task_enum)?;

            if dim < 768 {
                vec = truncate_mrl(&vec, dim)?;
            }

            match encoding {
                EncodingArg::Float => match output {
                    OutputArg::Preview => {
                        println!("Embedding dimension: {}", vec.len());
                        let preview = if vec.len() > 6 {
                            format!(
                                "[{:.4}, {:.4}, {:.4}, ..., {:.4}, {:.4}, {:.4}]",
                                vec[0],
                                vec[1],
                                vec[2],
                                vec[vec.len() - 3],
                                vec[vec.len() - 2],
                                vec[vec.len() - 1]
                            )
                        } else {
                            format!("{:?}", vec)
                        };
                        println!("Vector preview: {}", preview);
                    }
                    OutputArg::Json => {
                        println!("{}", serde_json::to_string(&vec)?);
                    }
                    OutputArg::Raw => {
                        let bytes: &[u8] = bytemuck::cast_slice(&vec);
                        std::io::stdout().write_all(bytes)?;
                        std::io::stdout().flush()?;
                    }
                },
                EncodingArg::Int8 => {
                    let quantized = glypr_core::quantize_int8(&vec);
                    match output {
                        OutputArg::Preview => {
                            println!("Embedding dimension: {} (INT8)", quantized.len());
                            let preview = if quantized.len() > 6 {
                                format!(
                                    "[{}, {}, {}, ..., {}, {}, {}]",
                                    quantized[0],
                                    quantized[1],
                                    quantized[2],
                                    quantized[quantized.len() - 3],
                                    quantized[quantized.len() - 2],
                                    quantized[quantized.len() - 1]
                                )
                            } else {
                                format!("{:?}", quantized)
                            };
                            println!("Vector preview: {}", preview);
                        }
                        OutputArg::Json => {
                            println!("{}", serde_json::to_string(&quantized)?);
                        }
                        OutputArg::Raw => {
                            let bytes: &[u8] = bytemuck::cast_slice(&quantized);
                            std::io::stdout().write_all(bytes)?;
                            std::io::stdout().flush()?;
                        }
                    }
                }
                EncodingArg::Binary => {
                    let quantized = glypr_core::quantize_binary(&vec)?;
                    match output {
                        OutputArg::Preview => {
                            let total_bits = quantized.len() * 64;
                            println!(
                                "Embedding dimension: {} (Binary, {} u64 words / {} bytes)",
                                total_bits,
                                quantized.len(),
                                quantized.len() * 8
                            );
                            let preview = if quantized.len() > 4 {
                                format!(
                                    "[{:#018x}, {:#018x}, ..., {:#018x}, {:#018x}]",
                                    quantized[0],
                                    quantized[1],
                                    quantized[quantized.len() - 2],
                                    quantized[quantized.len() - 1]
                                )
                            } else {
                                format!("{:x?}", quantized)
                            };
                            println!("Words preview: {}", preview);
                        }
                        OutputArg::Json => {
                            println!("{}", serde_json::to_string(&quantized)?);
                        }
                        OutputArg::Raw => {
                            let bytes: &[u8] = bytemuck::cast_slice(&quantized);
                            std::io::stdout().write_all(bytes)?;
                            std::io::stdout().flush()?;
                        }
                    }
                }
            }
        }

        Commands::Serve {
            host,
            port,
            max_batch_size,
        } => {
            // In server mode, automatically attach canonical multimodal projector if available
            let proj_path = match mmproj_override {
                Some(p) => Some(p),
                None => resolve_canonical_mmproj(cli.q8).ok(),
            };
            if let Some(ref p) = proj_path {
                embedder.attach_multimodal_projector(p)?;
            }

            let config = glypr_server::ServerConfig {
                host,
                port,
                max_batch_size,
                model_id: "embeddinggemma-2".to_string(),
            };
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async { glypr_server::run_server(config, embedder).await })?;
        }

        Commands::Pull { .. } | Commands::Info => unreachable!(),
    }

    Ok(())
}
