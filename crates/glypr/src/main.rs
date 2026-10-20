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
    #[value(alias = "int8")]
    Int,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
enum VisionBackendArg {
    #[default]
    Auto,
    Ane,
    Metal,
}

#[derive(Parser)]
#[command(
    name = "glypr",
    version,
    about = "Native Rust Embedding & Retrieval Engine for EmbeddingGemma 2"
)]
struct Cli {
    /// Use canonical BF16 base weights (embeddinggemma-2-BF16.gguf) [default]
    #[arg(long, global = true)]
    bf16: bool,

    /// Use canonical Q8_0 base weights (embeddinggemma-2-Q8_0.gguf)
    #[arg(long, global = true)]
    q8: bool,

    /// Attach canonical BF16 multimodal projector (mmproj-embeddinggemma-2-BF16.gguf)
    #[arg(long, global = true)]
    bf16_mmproj: bool,

    /// Attach canonical Q8_0 multimodal projector (mmproj-embeddinggemma-2-Q8_0.gguf)
    #[arg(long, global = true)]
    q8_mmproj: bool,

    /// Override base model with local GGUF file (disables automatic downloads)
    #[arg(long, global = true, value_name = "FILE")]
    model_file: Option<PathBuf>,

    /// Override multimodal projector with local GGUF file
    #[arg(long, global = true, value_name = "FILE")]
    mmproj_file: Option<PathBuf>,

    /// Hardware acceleration device: auto (default), metal, cpu
    #[arg(long, global = true, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Attention kernel: auto (default), flash, chunked, standard
    #[arg(long, global = true, value_enum, default_value_t = AttentionArg::Auto)]
    attention: AttentionArg,

    /// Enable Apple Neural Engine (ANE) CoreML acceleration for vision models
    #[arg(long, global = true)]
    ane: bool,

    /// Vision execution backend: auto (default), ane, metal
    #[arg(long, global = true, value_enum, default_value_t = VisionBackendArg::Auto)]
    vision_backend: VisionBackendArg,

    /// Path to compiled CoreML models directory (.mlmodelc) [default: models/coreml]
    #[arg(long, global = true, value_name = "DIR")]
    coreml_dir: Option<PathBuf>,

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

        /// Task prompt name: Raw (default), SearchQuery, CodeRetrieval, Document, QuestionAnswering
        #[arg(short, long, default_value = "Raw")]
        task: String,

        /// Document title (when --task Document is used)
        #[arg(long)]
        title: Option<String>,

        /// Output vector dimension: 768 (default), 512, 256, 128
        #[arg(short, long, default_value_t = 768)]
        dim: usize,

        /// Output encoding format: float (default), int, binary
        #[arg(short, long, value_enum, default_value_t = EncodingArg::Float)]
        encoding: EncodingArg,

        /// Output display mode: preview (default), json, raw
        #[arg(short, long, value_enum, default_value_t = OutputArg::Preview)]
        output: OutputArg,

        /// Image soft token budget [default: 280] [range: 16-2048]
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
        /// Pull all 4 canonical models (default if no specific flags provided)
        #[arg(long)]
        all: bool,
    },

    /// Display cache status, hardware detection, and engine capabilities
    Info,
}

fn resolve_base_model_path(model_file: Option<PathBuf>, use_q8: bool) -> Result<PathBuf> {
    if let Some(file) = model_file {
        if !file.is_file() {
            anyhow::bail!("Specified --model-file '{:?}' is not a valid file", file);
        }
        return Ok(file);
    }
    resolve_canonical_model(use_q8)
}

fn resolve_projector_path(
    mmproj_file: Option<PathBuf>,
    use_bf16_mmproj: bool,
    use_q8_mmproj: bool,
) -> Result<Option<PathBuf>> {
    if let Some(file) = mmproj_file {
        if !file.is_file() {
            anyhow::bail!("Specified --mmproj-file '{:?}' is not a valid file", file);
        }
        return Ok(Some(file));
    }
    if use_bf16_mmproj && use_q8_mmproj {
        anyhow::bail!("Cannot specify both --bf16-mmproj and --q8-mmproj simultaneously");
    }
    if use_bf16_mmproj {
        return Ok(Some(resolve_canonical_mmproj(false)?));
    }
    if use_q8_mmproj {
        return Ok(Some(resolve_canonical_mmproj(true)?));
    }
    Ok(None)
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

fn resolve_attention_mode(arg: AttentionArg, _device: &Device) -> Result<glypr_core::AttentionMode> {
    match arg {
        AttentionArg::Auto => {
            #[cfg(feature = "metal")]
            if _device.is_metal() {
                return Ok(glypr_core::AttentionMode::MetalFlash);
            }
            Ok(glypr_core::AttentionMode::ChunkedSliding)
        }
        AttentionArg::Flash => {
            #[cfg(feature = "metal")]
            if _device.is_metal() {
                return Ok(glypr_core::AttentionMode::MetalFlash);
            }
            eprintln!("Warning: Flash attention requested but running on CPU. Falling back to Chunked Sliding Attention.");
            Ok(glypr_core::AttentionMode::ChunkedSliding)
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
            match ext.as_str() {
                "png" | "jpg" | "jpeg" | "webp" => {
                    let img = image::open(p)
                        .with_context(|| format!("Failed to open image file {:?}", p))?;
                    parts.push(OwnedInputItem::Image(img));
                    has_media = true;
                }
                "wav" => {
                    let samples = glypr_core::multimodal::decode_wav_file(p)
                        .with_context(|| format!("Failed to decode WAV audio {:?}", p))?;
                    parts.push(OwnedInputItem::Audio(samples));
                    has_media = true;
                }
                _ => {
                    anyhow::bail!(
                        "Unsupported media file format '.{}' for {:?}. Supported image formats: .png, .jpg, .jpeg, .webp; supported audio formats: .wav.",
                        ext,
                        p
                    );
                }
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
    if let Commands::Pull { all } = cli.command {
        let pull_all = all || (!cli.bf16 && !cli.bf16_mmproj && !cli.q8 && !cli.q8_mmproj);
        let mut targets = Vec::new();

        if pull_all || cli.bf16 {
            targets.push(GGUF_BF16_FILE);
        }
        if pull_all || cli.bf16_mmproj {
            targets.push(MMPROJ_BF16_FILE);
        }
        if pull_all || cli.q8 {
            targets.push(GGUF_Q8_0_FILE);
        }
        if pull_all || cli.q8_mmproj {
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
        println!("Canonical Models:");

        for (flag, file, desc) in [
            ("--bf16", GGUF_BF16_FILE, "base model, default"),
            ("--q8", GGUF_Q8_0_FILE, "base model"),
            ("--bf16-mmproj", MMPROJ_BF16_FILE, "projector"),
            ("--q8-mmproj", MMPROJ_Q8_0_FILE, "projector"),
        ] {
            if let Some(path) = check_cached_file(DEFAULT_GGUF_REPO, file) {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                println!(
                    "  [✓] {:<16}: {} ({}) [{}]",
                    flag,
                    file,
                    format_bytes(size),
                    desc
                );
            } else {
                println!(
                    "  [ ] {:<16}: {} (Not cached) [{}]",
                    flag, file, desc
                );
            }
        }

        println!();
        #[cfg(feature = "metal")]
        let metal_avail = candle_core::utils::metal_is_available();
        #[cfg(not(feature = "metal"))]
        let metal_avail = false;

        println!("Hardware & Runtime:");
        if metal_avail {
            #[cfg(feature = "metal")]
            if let Some(caps) = glypr_core::metal_kernels::MetalCapabilities::detect_default() {
                let tier = glypr_core::metal_kernels::MetalTier::from_env();
                let nax_status = if caps.use_m5_tensor_ops() {
                    "Active (Metal 4 mpp::tensor_ops)"
                } else if caps.has_tensor_ops_compiler {
                    "Supported (Disabled via GLYPR_METAL_TIER=legacy)"
                } else {
                    "Not available (Using M1–M4 simdgroup_matrix)"
                };
                println!(
                    "  Device:             {} ({}, BF16)",
                    caps.device_name, caps.architecture
                );
                println!(
                    "  GPU Families:       Apple9={} | Apple10(M5)={} | Metal4={}",
                    caps.has_native_bfloat16, caps.has_apple10_m5, caps.has_metal4_family
                );
                println!(
                    "  Neural Accel (NAX): {} [tier={}]",
                    nax_status,
                    tier.as_str()
                );
                println!(
                    "  Memory Limits:      {} threadgroup | {} recommended VRAM",
                    format_bytes(caps.max_threadgroup_memory as u64),
                    format_bytes(caps.recommended_max_working_set_size as u64)
                );
            } else {
                println!("  Device:             Apple Silicon Metal GPU (BF16)");
            }
            #[cfg(not(feature = "metal"))]
            println!("  Device:             Apple Silicon Metal GPU (BF16)");
        } else {
            println!("  Device:             CPU SIMD (FP32)");
        }
        println!("  Context Window:     8,192 tokens");
        println!();
        println!("Attention Kernels (--attention):");
        if metal_avail {
            println!("  flash               : Fused hardware FlashAttention (default)");
            println!("  chunked             : Sliding-window banded attention");
            println!("  standard            : Full dense NxN attention");
        } else {
            println!("  chunked             : Sliding-window banded attention (default)");
            println!("  standard            : Full dense NxN attention");
            println!("  flash               : Fused hardware FlashAttention (Requires Metal GPU)");
        }
        println!();
        println!("Output Dimensions (--dim):");
        println!("  768 (default), 512, 256, 128");
        println!();
        println!("Encoding Format (--encoding):");
        println!("  float               : 32-bit floating point (default)");
        println!("  int                 : 8-bit signed integer (4x compression)");
        println!("  binary              : 1-bit packed binary (32x compression)");
        println!();
        println!("Vision Budget (--image-token-budget):");
        println!("  280 (default) [range: 16–2048]");
        println!("================================================================================");
        return Ok(());
    }

    // Load Model & Engine
    let model_path = resolve_base_model_path(cli.model_file, cli.q8)?;
    let (device, dtype) = resolve_device_and_dtype(cli.device, cli.metal)?;

    let mut embedder = GlyprEmbedder::load_gguf(&model_path, &device, dtype)?;
    let attn_mode = resolve_attention_mode(cli.attention, &device)?;
    embedder.set_attention_mode(attn_mode);

    // Resolve Multimodal Projector (opt-in; default is None)
    let proj_path = resolve_projector_path(cli.mmproj_file, cli.bf16_mmproj, cli.q8_mmproj)?;
    if let Some(ref p) = proj_path {
        embedder.attach_multimodal_projector(p)?;
    }

    #[cfg(feature = "coreml")]
    {
        let policy = match cli.vision_backend {
            VisionBackendArg::Auto => glypr_core::VisionBackendPolicy::Auto,
            VisionBackendArg::Ane => glypr_core::VisionBackendPolicy::AneStrict,
            VisionBackendArg::Metal => glypr_core::VisionBackendPolicy::Metal,
        };
        embedder.set_vision_backend(policy);

        if cli.ane || cli.vision_backend != VisionBackendArg::Metal {
            let coreml_dir = cli.coreml_dir.unwrap_or_else(|| PathBuf::from("models/coreml"));
            if coreml_dir.is_dir() {
                if let Err(e) = embedder.attach_coreml_vision_dir(&coreml_dir, glypr_core::coreml::CoreMLComputeUnits::CpuAndNeuralEngine) {
                    if cli.ane || cli.vision_backend == VisionBackendArg::Ane {
                        anyhow::bail!("Failed to load CoreML vision models from {}: {}", coreml_dir.display(), e);
                    }
                }
            } else if cli.ane || cli.vision_backend == VisionBackendArg::Ane {
                anyhow::bail!("CoreML models directory not found at: {}", coreml_dir.display());
            }
        }
    }

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

            // Reject media if no multimodal engine is attached
            let has_coreml = {
                #[cfg(feature = "coreml")]
                { embedder.coreml_vision.is_some() }
                #[cfg(not(feature = "coreml"))]
                { false }
            };

            if has_media && embedder.multimodal.is_none() && !has_coreml {
                anyhow::bail!(
                    "Media input detected in arguments, but neither multimodal projector nor CoreML vision model is attached.\n\
                    Tip: Add --ane to enable Apple Neural Engine vision, or add --bf16-mmproj / --q8-mmproj / --mmproj-file <FILE>."
                );
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
                EncodingArg::Int => {
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
