use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
#[cfg(any(feature = "coreml", feature = "ane"))]
use glypr_core::hub::resolve_coreml_dir;
use glypr_core::{
    CoreMLComputeUnits, GlyprEmbedder, ModalityPart, MultimodalInput, Task, TextBackendPolicy,
    VisionBackendPolicy,
    hub::{
        DEFAULT_GGUF_REPO, GGUF_BF16_FILE, GGUF_Q8_0_FILE, GOOGLE_OFFICIAL_REPO,
        GOOGLE_SAFETENSORS_FILE, GOOGLE_TOKENIZER_FILE, MMPROJ_BF16_FILE, MMPROJ_Q8_0_FILE,
        check_cached_file, get_user_coreml_dir, is_offline_mode,
        resolve_canonical_mmproj_with_options, resolve_canonical_model_with_options,
        resolve_file_with_options,
    },
    validate_vision_budget,
};
use std::collections::BTreeMap;
use std::io::{BufRead, Read, Write};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
enum TextBackendArg {
    #[default]
    Auto,
    Ane,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
enum CoreMLComputeUnitsArg {
    #[default]
    Ane,
    All,
    Gpu,
    Cpu,
}

impl From<CoreMLComputeUnitsArg> for CoreMLComputeUnits {
    fn from(val: CoreMLComputeUnitsArg) -> Self {
        match val {
            CoreMLComputeUnitsArg::Ane => CoreMLComputeUnits::CpuAndNeuralEngine,
            CoreMLComputeUnitsArg::All => CoreMLComputeUnits::All,
            CoreMLComputeUnitsArg::Gpu => CoreMLComputeUnits::CpuAndGpu,
            CoreMLComputeUnitsArg::Cpu => CoreMLComputeUnits::CpuOnly,
        }
    }
}

fn task_help_footer() -> String {
    let mut s = String::from("Supported --task names:\n");
    for (name, desc) in Task::all_names() {
        s.push_str(&format!("  {:<20} {}\n", name, desc));
    }
    s.push_str("  <custom>             Custom prefix or 'task: <name> | query: '");
    s
}

#[derive(Parser)]
#[command(
    name = "glypr",
    version,
    about = "Native Rust Embedding & Retrieval Engine for EmbeddingGemma 2"
)]
struct Cli {
    /// Use canonical BF16 base weights (embeddinggemma-2-BF16.gguf) [default]
    #[arg(long, global = true, conflicts_with = "q8")]
    bf16: bool,

    /// Use canonical Q8_0 base weights (embeddinggemma-2-Q8_0.gguf)
    #[arg(long, global = true, conflicts_with = "bf16")]
    q8: bool,

    /// Attach canonical BF16 multimodal projector (mmproj-embeddinggemma-2-BF16.gguf)
    #[arg(long, global = true, conflicts_with = "q8_mmproj")]
    bf16_mmproj: bool,

    /// Attach canonical Q8_0 multimodal projector (mmproj-embeddinggemma-2-Q8_0.gguf)
    #[arg(long, global = true, conflicts_with = "bf16_mmproj")]
    q8_mmproj: bool,

    /// Override base model with local GGUF file (disables automatic downloads)
    #[arg(long, global = true, value_name = "FILE")]
    model_file: Option<PathBuf>,

    /// Override multimodal projector with local GGUF file
    #[arg(long, global = true, value_name = "FILE")]
    mmproj_file: Option<PathBuf>,

    /// Hardware acceleration device: auto (default), metal, cpu
    #[arg(long, global = true, value_enum, default_value_t = DeviceArg::Auto, conflicts_with = "metal")]
    device: DeviceArg,

    /// Attention kernel: auto (default), flash, chunked, standard
    #[arg(long, global = true, value_enum, default_value_t = AttentionArg::Auto)]
    attention: AttentionArg,

    /// Enable Apple Neural Engine (ANE) CoreML acceleration (vision + pure-text routing)
    #[arg(long, global = true)]
    ane: bool,

    /// Vision execution backend: auto (default), ane, metal
    #[arg(long, global = true, value_enum, default_value_t = VisionBackendArg::Auto)]
    vision_backend: VisionBackendArg,

    /// Pure-text ANE execution policy: auto (default, 64..=seq_len & batch<=4), ane, metal
    #[arg(long, global = true, value_enum, default_value_t = TextBackendArg::Auto)]
    text_backend: TextBackendArg,

    /// Path to compiled CoreML models directory (.mlmodelc)
    #[arg(long, global = true, value_name = "DIR")]
    coreml_dir: Option<PathBuf>,

    /// CoreML compute units: ane (default), all, gpu, cpu
    #[arg(long, global = true, value_enum, default_value_t = CoreMLComputeUnitsArg::Ane)]
    coreml_compute_units: CoreMLComputeUnitsArg,

    /// Offline mode: fail immediately if weights are not in local Hugging Face cache
    #[arg(long, global = true)]
    offline: bool,

    /// Enable verbose diagnostic logging on stderr
    #[arg(short, long, global = true, conflicts_with = "quiet")]
    verbose: bool,

    /// Suppress non-essential diagnostic output on stderr
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    quiet: bool,

    /// Legacy flag for Metal acceleration (alias for --device metal)
    #[arg(long, global = true, hide = true, conflicts_with = "device")]
    metal: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate embedding vectors for text, image, audio, interleaved, or batched inputs
    #[command(after_help = task_help_footer())]
    Embed {
        /// Positional text and media paths (interleaved, or independent items with --batch). If omitted or '-', reads from STDIN.
        #[arg(value_name = "INPUTS")]
        inputs: Vec<String>,

        /// Explicit image file path(s) to include (repeatable; avoids path-vs-text ambiguity)
        #[arg(long = "image", value_name = "PATH", action = ArgAction::Append)]
        images: Vec<PathBuf>,

        /// Explicit WAV audio file path(s) to include (repeatable; 16 kHz mono recommended)
        #[arg(long = "audio", value_name = "PATH", action = ArgAction::Append)]
        audios: Vec<PathBuf>,

        /// Explicit UTF-8 text file path(s) to embed (repeatable; reads file contents as text)
        #[arg(long = "text-file", value_name = "PATH", action = ArgAction::Append)]
        text_files: Vec<PathBuf>,

        /// Embed each positional input or STDIN line as an independent vector
        #[arg(long, conflicts_with = "jsonl")]
        batch: bool,

        /// Stream JSONL objects (or plain lines) from STDIN and emit one JSON embedding object per line
        #[arg(long, conflicts_with = "batch")]
        jsonl: bool,

        /// Task prompt name: Raw (default), SearchQuery, Document, CodeRetrieval, QuestionAnswering, FactChecking, Classification, Clustering, SentenceSimilarity
        #[arg(short, long, default_value = "Raw")]
        task: String,

        /// Document title (when --task Document is used)
        #[arg(long)]
        title: Option<String>,

        /// Output vector dimension: 768 (default), 512, 256, 128 (or 1..=768)
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
        /// Host address to bind to [env: GLYPR_HOST]
        #[arg(long)]
        host: Option<String>,

        /// Port to listen on [env: GLYPR_PORT]
        #[arg(short, long)]
        port: Option<u16>,

        /// Maximum dynamic coalescing batch size [env: GLYPR_MAX_BATCH_SIZE]
        #[arg(long)]
        max_batch_size: Option<usize>,

        /// Served model identifier in /v1/models and responses [env: GLYPR_MODEL_ID]
        #[arg(long)]
        model_id: Option<String>,

        /// Maximum HTTP request body size in MiB [env: GLYPR_MAX_BODY_MB, default: 32]
        #[arg(long)]
        max_body_mb: Option<usize>,

        /// Maximum items per /v1/embeddings request array [env: GLYPR_MAX_BATCH_ITEMS, default: 256]
        #[arg(long)]
        max_batch_items: Option<usize>,

        /// Per-request inference timeout in seconds [env: GLYPR_REQUEST_TIMEOUT_SECS, default: 120]
        #[arg(long)]
        request_timeout_secs: Option<u64>,

        /// Micro-batch coalescing wait window in microseconds (0 = no wait) [env: GLYPR_BATCH_WAIT_US, default: 0]
        #[arg(long)]
        batch_wait_us: Option<u64>,

        /// Opt-in to allow reading local filesystem paths from HTTP media inputs [env: GLYPR_ALLOW_LOCAL_FILES]
        #[arg(long)]
        allow_local_files: bool,

        /// Pre-warm CoreML models at startup (default: true when CoreML is attached; pass --warmup-coreml=false to skip)
        #[arg(long)]
        warmup_coreml: Option<bool>,
    },

    /// Pre-fetch canonical GGUF (or Safetensors) models into Hugging Face cache
    Pull {
        /// Pull all 4 canonical GGUF models (default if no specific flags provided)
        #[arg(long)]
        all: bool,

        /// Also pull Google Safetensors weights + config + tokenizer for CoreML export
        #[arg(long)]
        safetensors: bool,
    },

    /// Display cache status, hardware detection, CoreML models, and supported tasks
    Info,

    /// Inspect metadata, shapes, and tensors of a .gguf file or .mlmodelc CoreML bundle
    Inspect {
        /// Path to a .gguf file or .mlmodelc directory
        path: PathBuf,
    },
}

fn is_recognized_media_ext(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "webp" => Some("image"),
        "wav" => Some("audio"),
        _ => None,
    }
}

fn validate_coreml_feature_and_flags(cli: &Cli) -> Result<()> {
    #[cfg(not(any(feature = "coreml", feature = "ane")))]
    {
        if cli.ane
            || cli.vision_backend == VisionBackendArg::Ane
            || cli.text_backend == TextBackendArg::Ane
            || cli.coreml_dir.is_some()
        {
            anyhow::bail!(
                "Apple Neural Engine (CoreML) requested, but this binary was built without --features ane. \
                 Rebuild/reinstall with: cargo install --path crates/glypr --features metal,ane"
            );
        }
    }

    #[cfg(any(feature = "coreml", feature = "ane"))]
    {
        if cli.ane
            && cli.vision_backend == VisionBackendArg::Metal
            && cli.text_backend != TextBackendArg::Ane
        {
            anyhow::bail!(
                "Contradictory flags: --ane enables CoreML vision, while --vision-backend metal disables it. \
                 Use --text-backend ane without --ane if you only want ANE text routing."
            );
        }
        if let Some(ref dir) = cli.coreml_dir
            && !dir.is_dir()
        {
            anyhow::bail!(
                "Specified --coreml-dir '{}' does not exist or is not a directory",
                dir.display()
            );
        }
    }

    Ok(())
}

fn resolve_base_model_path(
    model_file: Option<PathBuf>,
    use_q8: bool,
    offline: bool,
) -> Result<PathBuf> {
    if let Some(file) = model_file {
        if !file.is_file() {
            anyhow::bail!(
                "Specified --model-file '{}' is not a valid file",
                file.display()
            );
        }
        return Ok(file);
    }
    Ok(resolve_canonical_model_with_options(use_q8, offline)?)
}

fn resolve_projector_path(
    mmproj_file: Option<PathBuf>,
    use_bf16_mmproj: bool,
    use_q8_mmproj: bool,
    offline: bool,
) -> Result<Option<PathBuf>> {
    if let Some(file) = mmproj_file {
        if !file.is_file() {
            anyhow::bail!(
                "Specified --mmproj-file '{}' is not a valid file",
                file.display()
            );
        }
        return Ok(Some(file));
    }
    if use_bf16_mmproj && use_q8_mmproj {
        anyhow::bail!("Cannot specify both --bf16-mmproj and --q8-mmproj simultaneously");
    }
    if use_bf16_mmproj {
        return Ok(Some(resolve_canonical_mmproj_with_options(false, offline)?));
    }
    if use_q8_mmproj {
        return Ok(Some(resolve_canonical_mmproj_with_options(true, offline)?));
    }
    Ok(None)
}

fn resolve_device_and_dtype(device_arg: DeviceArg, legacy_metal: bool) -> Result<(Device, DType)> {
    let use_metal = match device_arg {
        DeviceArg::Metal => true,
        DeviceArg::Cpu => false,
        DeviceArg::Auto => {
            #[cfg(feature = "metal")]
            {
                legacy_metal || glypr_core::metal_kernels::is_metal_device_available()
            }
            #[cfg(not(feature = "metal"))]
            {
                legacy_metal
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
            anyhow::bail!(
                "Metal acceleration requested, but this binary was compiled without --features metal. \
                 Rebuild/reinstall with: cargo install --path crates/glypr --features metal"
            );
        }
    } else {
        Ok((Device::Cpu, DType::F32))
    }
}

fn resolve_attention_mode(
    arg: AttentionArg,
    device: &Device,
    quiet: bool,
) -> Result<glypr_core::AttentionMode> {
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
            let _ = device;
            if !quiet {
                eprintln!(
                    "Warning: Flash attention requested but running on CPU. Falling back to Chunked Sliding Attention."
                );
            }
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

fn load_media_path(p: &Path, kind: &str) -> Result<OwnedInputItem> {
    if !p.is_file() {
        anyhow::bail!("Media file not found: {}", p.display());
    }
    match kind {
        "image" => {
            let img = image::open(p)
                .with_context(|| format!("Failed to open image file {}", p.display()))?;
            Ok(OwnedInputItem::Image(img))
        }
        "audio" => {
            let samples = glypr_core::multimodal::decode_wav_file(p)
                .with_context(|| format!("Failed to decode WAV audio {}", p.display()))?;
            Ok(OwnedInputItem::Audio(samples))
        }
        _ => unreachable!(),
    }
}

fn parse_interleaved_inputs(
    inputs: Vec<String>,
    explicit_images: Vec<PathBuf>,
    explicit_audios: Vec<PathBuf>,
    explicit_text_files: Vec<PathBuf>,
) -> Result<(Vec<OwnedInputItem>, bool, bool)> {
    let mut parts = Vec::new();
    let mut has_image = false;
    let mut has_audio = false;

    for img_path in explicit_images {
        parts.push(load_media_path(&img_path, "image")?);
        has_image = true;
    }
    for aud_path in explicit_audios {
        parts.push(load_media_path(&aud_path, "audio")?);
        has_audio = true;
    }
    for txt_path in explicit_text_files {
        if !txt_path.is_file() {
            anyhow::bail!("Text file not found: {}", txt_path.display());
        }
        let content = std::fs::read_to_string(&txt_path)
            .with_context(|| format!("Failed to read --text-file {}", txt_path.display()))?;
        if content.trim().is_empty() {
            anyhow::bail!("Text file '{}' is empty", txt_path.display());
        }
        parts.push(OwnedInputItem::Text(content));
    }

    let read_stdin =
        (inputs.is_empty() && parts.is_empty()) || (inputs.len() == 1 && inputs[0] == "-");
    if read_stdin {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .context("Failed to read text from STDIN")?;
        let trimmed = buffer.trim();
        if trimmed.is_empty() && parts.is_empty() {
            anyhow::bail!("No input provided via arguments or STDIN");
        }
        if !trimmed.is_empty() {
            parts.push(OwnedInputItem::Text(trimmed.to_string()));
        }
        return Ok((parts, has_image, has_audio));
    }

    for arg in inputs {
        let p = Path::new(&arg);
        if let Some(kind) = is_recognized_media_ext(p) {
            parts.push(load_media_path(p, kind)?);
            if kind == "image" {
                has_image = true;
            } else {
                has_audio = true;
            }
        } else {
            parts.push(OwnedInputItem::Text(arg));
        }
    }

    Ok((parts, has_image, has_audio))
}

struct PreparedBatchItem {
    part: OwnedInputItem,
    task_name: Option<String>,
    title: Option<String>,
    budget: Option<usize>,
    dim: Option<usize>,
}

struct JsonlInputLine {
    text: Option<String>,
    task: Option<String>,
    title: Option<String>,
    image: Option<String>,
    audio: Option<String>,
    budget: Option<usize>,
    dim: Option<usize>,
}

impl JsonlInputLine {
    fn from_json_str(raw: &str) -> Result<Self> {
        let val: serde_json::Value = serde_json::from_str(raw)?;
        let obj = val
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Expected JSON object"))?;
        let opt_str = |k: &str| -> Result<Option<String>> {
            match obj.get(k) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
                Some(_) => anyhow::bail!("Field '{}' must be a string", k),
            }
        };
        let opt_usize = |k: &str| -> Result<Option<usize>> {
            match obj.get(k) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(serde_json::Value::Number(n)) => n
                    .as_u64()
                    .map(|u| Some(u as usize))
                    .ok_or_else(|| anyhow::anyhow!("Field '{}' must be a non-negative integer", k)),
                Some(_) => anyhow::bail!("Field '{}' must be an integer", k),
            }
        };

        Ok(Self {
            text: opt_str("text")?,
            task: opt_str("task")?,
            title: opt_str("title")?,
            image: opt_str("image")?,
            audio: opt_str("audio")?,
            budget: opt_usize("budget")?,
            dim: opt_usize("dim")?,
        })
    }
}

fn parse_batch_inputs(
    inputs: Vec<String>,
    explicit_images: Vec<PathBuf>,
    explicit_audios: Vec<PathBuf>,
    explicit_text_files: Vec<PathBuf>,
) -> Result<(Vec<PreparedBatchItem>, bool, bool)> {
    let mut items = Vec::new();
    let mut has_image = false;
    let mut has_audio = false;

    for img_path in explicit_images {
        items.push(PreparedBatchItem {
            part: load_media_path(&img_path, "image")?,
            task_name: None,
            title: None,
            budget: None,
            dim: None,
        });
        has_image = true;
    }
    for aud_path in explicit_audios {
        items.push(PreparedBatchItem {
            part: load_media_path(&aud_path, "audio")?,
            task_name: None,
            title: None,
            budget: None,
            dim: None,
        });
        has_audio = true;
    }
    for txt_path in explicit_text_files {
        if !txt_path.is_file() {
            anyhow::bail!("Text file not found: {}", txt_path.display());
        }
        let content = std::fs::read_to_string(&txt_path)
            .with_context(|| format!("Failed to read --text-file {}", txt_path.display()))?;
        items.push(PreparedBatchItem {
            part: OwnedInputItem::Text(content),
            task_name: None,
            title: None,
            budget: None,
            dim: None,
        });
    }

    let read_stdin =
        (inputs.is_empty() && items.is_empty()) || (inputs.len() == 1 && inputs[0] == "-");
    if read_stdin {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = line.context("Failed to read line from STDIN")?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let p = Path::new(trimmed);
            if let Some(kind) = is_recognized_media_ext(p) {
                items.push(PreparedBatchItem {
                    part: load_media_path(p, kind)?,
                    task_name: None,
                    title: None,
                    budget: None,
                    dim: None,
                });
                if kind == "image" {
                    has_image = true;
                } else {
                    has_audio = true;
                }
            } else {
                items.push(PreparedBatchItem {
                    part: OwnedInputItem::Text(trimmed.to_string()),
                    task_name: None,
                    title: None,
                    budget: None,
                    dim: None,
                });
            }
        }
    } else {
        for arg in inputs {
            let p = Path::new(&arg);
            if let Some(kind) = is_recognized_media_ext(p) {
                items.push(PreparedBatchItem {
                    part: load_media_path(p, kind)?,
                    task_name: None,
                    title: None,
                    budget: None,
                    dim: None,
                });
                if kind == "image" {
                    has_image = true;
                } else {
                    has_audio = true;
                }
            } else {
                items.push(PreparedBatchItem {
                    part: OwnedInputItem::Text(arg),
                    task_name: None,
                    title: None,
                    budget: None,
                    dim: None,
                });
            }
        }
    }

    if items.is_empty() {
        anyhow::bail!("No batch items provided via arguments or STDIN");
    }

    Ok((items, has_image, has_audio))
}

fn parse_jsonl_inputs(
    inputs: Vec<String>,
    default_dim: usize,
    encoding: EncodingArg,
) -> Result<(Vec<PreparedBatchItem>, bool, bool)> {
    let raw_lines: Vec<String> = if inputs.is_empty() || (inputs.len() == 1 && inputs[0] == "-") {
        std::io::stdin()
            .lock()
            .lines()
            .collect::<std::io::Result<Vec<_>>>()
            .context("Failed to read JSONL lines from STDIN")?
    } else if inputs.len() == 1 && Path::new(&inputs[0]).is_file() {
        let content = std::fs::read_to_string(&inputs[0])
            .with_context(|| format!("Failed to read JSONL file {}", inputs[0]))?;
        content.lines().map(|l| l.to_string()).collect()
    } else {
        anyhow::bail!(
            "--jsonl expects JSONL lines on STDIN (or a single .jsonl file path argument)"
        );
    };

    let mut items = Vec::new();
    let mut has_image = false;
    let mut has_audio = false;

    for (line_idx, line) in raw_lines.into_iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('{') {
            let obj = JsonlInputLine::from_json_str(trimmed)
                .with_context(|| format!("Invalid JSON object on JSONL line {}", line_idx + 1))?;
            if let Some(d) = obj.dim {
                validate_dim_arg(d, encoding)?;
            } else {
                validate_dim_arg(default_dim, encoding)?;
            }
            if let Some(b) = obj.budget {
                validate_vision_budget(b)?;
            }
            if obj.image.is_some() && obj.audio.is_some() {
                anyhow::bail!(
                    "JSONL line {} specifies both 'image' and 'audio' (choose one per item)",
                    line_idx + 1
                );
            }
            let part = if let Some(ref img_path) = obj.image {
                has_image = true;
                load_media_path(Path::new(img_path), "image")?
            } else if let Some(ref aud_path) = obj.audio {
                has_audio = true;
                load_media_path(Path::new(aud_path), "audio")?
            } else if let Some(txt) = obj.text {
                OwnedInputItem::Text(txt)
            } else {
                anyhow::bail!(
                    "JSONL line {} must specify 'text', 'image', or 'audio'",
                    line_idx + 1
                );
            };
            items.push(PreparedBatchItem {
                part,
                task_name: obj.task,
                title: obj.title,
                budget: obj.budget,
                dim: obj.dim,
            });
        } else {
            items.push(PreparedBatchItem {
                part: OwnedInputItem::Text(trimmed.to_string()),
                task_name: None,
                title: None,
                budget: None,
                dim: None,
            });
        }
    }

    if items.is_empty() {
        anyhow::bail!("No non-empty lines found in JSONL input");
    }

    Ok((items, has_image, has_audio))
}

fn validate_dim_arg(dim: usize, encoding: EncodingArg) -> Result<()> {
    if dim == 0 || dim > 768 {
        anyhow::bail!("Invalid --dim {}: must be in 1..=768", dim);
    }
    if encoding == EncodingArg::Binary && !dim.is_multiple_of(64) {
        anyhow::bail!(
            "Binary encoding requires --dim to be a multiple of 64, got {}",
            dim
        );
    }
    Ok(())
}

fn validate_media_engine_flags(cli: &Cli, has_image: bool, has_audio: bool) -> Result<()> {
    let has_mmproj = cli.bf16_mmproj || cli.q8_mmproj || cli.mmproj_file.is_some();
    let has_ane_vision =
        cli.ane || cli.vision_backend == VisionBackendArg::Ane || cli.coreml_dir.is_some();

    if has_audio && !has_mmproj {
        anyhow::bail!(
            "Audio input detected, but audio embeddings require an attached GGUF multimodal projector. \
             Add --bf16-mmproj, --q8-mmproj, or --mmproj-file <FILE> (CoreML/ANE supports vision and text only)."
        );
    }

    if has_image && !has_mmproj && !has_ane_vision {
        anyhow::bail!(
            "Media input detected in arguments, but no multimodal projector was attached.\n\
             Tip: Add --ane to enable Apple Neural Engine vision, or add --bf16-mmproj / --q8-mmproj / --mmproj-file <FILE> to enable image/audio embeddings."
        );
    }

    Ok(())
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

fn env_parse<T: std::str::FromStr>(key: &str) -> Result<Option<T>> {
    match std::env::var(key) {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            let parsed = trimmed
                .parse::<T>()
                .map_err(|_| anyhow::anyhow!("Invalid value '{}' for env var {}", trimmed, key))?;
            Ok(Some(parsed))
        }
        Err(_) => Ok(None),
    }
}

fn emit_vector_output(
    vec: &[f32],
    encoding: EncodingArg,
    output: OutputArg,
    prefix: Option<&str>,
) -> Result<()> {
    match encoding {
        EncodingArg::Float => match output {
            OutputArg::Preview => {
                if let Some(p) = prefix {
                    println!("{p} Embedding dimension: {}", vec.len());
                } else {
                    println!("Embedding dimension: {}", vec.len());
                }
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
                println!("{}", serde_json::to_string(vec)?);
            }
            OutputArg::Raw => {
                let bytes: &[u8] = bytemuck::cast_slice(vec);
                std::io::stdout().write_all(bytes)?;
                std::io::stdout().flush()?;
            }
        },
        EncodingArg::Int => {
            let quantized = glypr_core::quantize_int8(vec);
            match output {
                OutputArg::Preview => {
                    if let Some(p) = prefix {
                        println!("{p} Embedding dimension: {} (INT8)", quantized.len());
                    } else {
                        println!("Embedding dimension: {} (INT8)", quantized.len());
                    }
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
            let quantized = glypr_core::quantize_binary(vec)?;
            match output {
                OutputArg::Preview => {
                    let total_bits = quantized.len() * 64;
                    if let Some(p) = prefix {
                        println!(
                            "{p} Embedding dimension: {} (Binary, {} u64 words / {} bytes)",
                            total_bits,
                            quantized.len(),
                            quantized.len() * 8
                        );
                    } else {
                        println!(
                            "Embedding dimension: {} (Binary, {} u64 words / {} bytes)",
                            total_bits,
                            quantized.len(),
                            quantized.len() * 8
                        );
                    }
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
    Ok(())
}

fn handle_inspect(path: &Path, compute_units: CoreMLComputeUnits) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Inspect path does not exist: {}", path.display());
    }

    if path.is_dir() && path.extension().and_then(|s| s.to_str()) == Some("mlmodelc") {
        #[cfg(any(feature = "coreml", feature = "ane"))]
        {
            let info = glypr_core::coreml::inspect_mlmodelc(path, compute_units)?;
            println!("CoreML Bundle: {}", info.path.display());
            println!("Inputs ({}):", info.inputs.len());
            for inp in &info.inputs {
                println!("  - {:<20} shape={:?}", inp.name, inp.shape);
            }
            println!("Outputs ({}):", info.outputs.len());
            for out in &info.outputs {
                println!("  - {:<20} shape={:?}", out.name, out.shape);
            }
            return Ok(());
        }
        #[cfg(not(any(feature = "coreml", feature = "ane")))]
        {
            let _ = compute_units;
            anyhow::bail!(
                "Inspecting .mlmodelc bundles requires CoreML support. Rebuild with: cargo install --path crates/glypr --features metal,ane"
            );
        }
    }

    if path.is_file() {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open file {}", path.display()))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to parse GGUF container from {}", path.display()))?;

        println!("GGUF File: {}", path.display());
        println!("Metadata Keys: {}", content.metadata.len());
        let mut keys: Vec<&String> = content.metadata.keys().collect();
        keys.sort();
        for k in keys {
            let val = &content.metadata[k];
            let rendered = match val {
                candle_core::quantized::gguf_file::Value::Array(arr) => {
                    format!("Array(len={})", arr.len())
                }
                other => format!("{:?}", other),
            };
            println!("  {:<44} : {}", k, rendered);
        }

        let mut dtype_counts: BTreeMap<String, usize> = BTreeMap::new();
        for info in content.tensor_infos.values() {
            *dtype_counts
                .entry(format!("{:?}", info.ggml_dtype))
                .or_insert(0) += 1;
        }
        println!("Tensors: {}", content.tensor_infos.len());
        println!("Quantization / DType Summary:");
        for (dt, count) in dtype_counts {
            println!("  {:<16} : {} tensors", dt, count);
        }
        return Ok(());
    }

    anyhow::bail!(
        "Unsupported path '{}' for inspect (expected a .gguf file or .mlmodelc directory)",
        path.display()
    );
}

fn handle_info(cli: &Cli) -> Result<()> {
    let cache_dir = glypr_core::hub::get_hf_cache_dir();
    println!("================================================================================");
    println!("  GLYPR ENGINE INFORMATION");
    println!("================================================================================");
    println!("Hugging Face Cache:   {}", cache_dir.display());
    println!("Offline Mode:         {}", cli.offline || is_offline_mode());
    println!("Default GGUF Repo:    {}", DEFAULT_GGUF_REPO);
    println!("Default CoreML Dir:   {}", get_user_coreml_dir().display());
    if let Ok(env_coreml) = std::env::var("GLYPR_COREML_DIR") {
        println!("GLYPR_COREML_DIR:     {}", env_coreml);
    }
    if let Ok(env_tier) = std::env::var("GLYPR_METAL_TIER") {
        println!("GLYPR_METAL_TIER:     {}", env_tier);
    }
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
            println!("  [ ] {:<16}: {} (Not cached) [{}]", flag, file, desc);
        }
    }

    println!();
    #[cfg(feature = "metal")]
    let metal_avail = glypr_core::metal_kernels::is_metal_device_available();
    #[cfg(not(feature = "metal"))]
    let metal_avail = false;

    println!("Hardware & Runtime:");
    #[cfg(not(feature = "metal"))]
    println!("  Metal GPU:          Disabled at compile time (rebuild with --features metal)");

    if metal_avail {
        #[cfg(feature = "metal")]
        if let Some(caps) = glypr_core::metal_kernels::MetalCapabilities::detect_default() {
            let tier = glypr_core::metal_kernels::MetalTier::from_env();
            let nax_status = if caps.use_m5_tensor_ops() {
                "Active (Metal 4 mpp::tensor_ops)"
            } else if caps.has_tensor_ops_compiler
                && tier == glypr_core::metal_kernels::MetalTier::Legacy
            {
                "Supported (Disabled via GLYPR_METAL_TIER=legacy)"
            } else if caps.has_tensor_ops_compiler {
                "Supported (Unused on non-M5 GPU / Auto)"
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

    #[cfg(any(feature = "coreml", feature = "ane"))]
    {
        println!("  Neural Engine (ANE): Supported (CoreML native runtime)");
        let compute_units: CoreMLComputeUnits = cli.coreml_compute_units.into();
        if let Some(coreml_dir) = resolve_coreml_dir(cli.coreml_dir.as_deref()) {
            if coreml_dir.is_dir() {
                match glypr_core::coreml::CoreMLVisionTower::from_dir(&coreml_dir, compute_units) {
                    Ok(tower) => {
                        let grids: Vec<String> = tower
                            .grid_infos()
                            .into_iter()
                            .map(|g| {
                                format!("{}x{} ({}p)", g.patch_height, g.patch_width, g.num_patches)
                            })
                            .collect();
                        println!(
                            "  ANE Vision Models:  Loaded from {} [{}]",
                            coreml_dir.display(),
                            grids.join(", ")
                        );
                    }
                    Err(e) => {
                        println!(
                            "  ANE Vision Models:  None in {} ({})",
                            coreml_dir.display(),
                            e
                        );
                    }
                }
                match glypr_core::coreml::CoreMLTextBackbone::from_dir(&coreml_dir, compute_units) {
                    Ok(text_bb) => {
                        println!(
                            "  ANE Text Backbone:  Loaded from {} (seq_lens: {:?})",
                            coreml_dir.display(),
                            text_bb.supported_seq_lens()
                        );
                    }
                    Err(e) => {
                        println!(
                            "  ANE Text Backbone:  None in {} ({})",
                            coreml_dir.display(),
                            e
                        );
                    }
                }
            } else {
                println!(
                    "  ANE CoreML Models:  Directory {} not found",
                    coreml_dir.display()
                );
            }
        } else {
            println!(
                "  ANE CoreML Models:  None found (run 'uv run python scripts/export_coreml.py' or set GLYPR_COREML_DIR)"
            );
        }
    }
    #[cfg(not(any(feature = "coreml", feature = "ane")))]
    {
        println!(
            "  Neural Engine (ANE): Disabled at compile time (rebuild with --features metal,ane)"
        );
    }

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

    #[cfg(any(feature = "coreml", feature = "ane"))]
    {
        println!();
        println!("Vision & Text Backends (--vision-backend / --text-backend):");
        println!("  auto (vision)       : ANE when grid matches, Metal mmproj fallback (default)");
        println!("  ane  (vision)       : Strict Apple Neural Engine execution");
        println!("  metal (vision)      : GPU Metal multimodal projector");
        println!(
            "  auto (text)         : ANE for pure text in [64..=seq_len] (batch <= 4), else Metal"
        );
        println!("  ane  (text)         : Strict Apple Neural Engine text execution");
        println!("  metal (text)        : Force Metal/CPU text backbone");
    }

    println!();
    println!("Supported Tasks (--task):");
    for (name, desc) in Task::all_names() {
        println!("  {:<20}: {}", name, desc);
    }
    println!();
    println!("Output Dimensions (--dim):");
    println!("  768 (default), 512, 256, 128 [range: 1..=768]");
    println!();
    println!("Encoding Format (--encoding):");
    println!("  float               : 32-bit floating point (default)");
    println!("  int                 : 8-bit signed integer (4x compression)");
    println!("  binary              : 1-bit packed binary (32x compression, dim % 64 == 0)");
    println!();
    println!("Vision Budget (--image-token-budget):");
    println!("  280 (default) [range: 16–2048]");
    println!("================================================================================");
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let offline = cli.offline || is_offline_mode();

    // 1. Handle Pull command (runs without loading weights into GPU/RAM)
    if let Commands::Pull { all, safetensors } = &cli.command {
        let pull_all =
            *all || (!cli.bf16 && !cli.bf16_mmproj && !cli.q8 && !cli.q8_mmproj && !*safetensors);
        let mut targets = Vec::new();

        if pull_all || cli.bf16 {
            targets.push((DEFAULT_GGUF_REPO, GGUF_BF16_FILE));
        }
        if pull_all || cli.bf16_mmproj {
            targets.push((DEFAULT_GGUF_REPO, MMPROJ_BF16_FILE));
        }
        if pull_all || cli.q8 {
            targets.push((DEFAULT_GGUF_REPO, GGUF_Q8_0_FILE));
        }
        if pull_all || cli.q8_mmproj {
            targets.push((DEFAULT_GGUF_REPO, MMPROJ_Q8_0_FILE));
        }
        if *safetensors {
            targets.push((GOOGLE_OFFICIAL_REPO, GOOGLE_SAFETENSORS_FILE));
            targets.push((GOOGLE_OFFICIAL_REPO, GOOGLE_TOKENIZER_FILE));
            targets.push((GOOGLE_OFFICIAL_REPO, "config.json"));
        }

        let cache_dir = glypr_core::hub::get_hf_cache_dir();
        println!("Hugging Face Cache: {}", cache_dir.display());
        for (repo, file) in targets {
            print!("  - Resolving {}/{} ... ", repo, file);
            std::io::stdout().flush()?;
            let path = resolve_file_with_options(repo, file, offline)?;
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            println!("✓ ({})", format_bytes(size));
        }
        println!(
            "All requested models verified in cache ({}).",
            cache_dir.display()
        );
        println!("Next step: run 'glypr info' to inspect cache status and hardware capabilities.");
        return Ok(());
    }

    // 2. Handle Info & Inspect commands (weightless)
    if let Commands::Info = cli.command {
        return handle_info(&cli);
    }
    if let Commands::Inspect { ref path } = cli.command {
        return handle_inspect(path, cli.coreml_compute_units.into());
    }

    // 3. Validate CoreML/ANE feature flags & contradictions before loading weights
    validate_coreml_feature_and_flags(&cli)?;

    // 4. Pre-validate Embed arguments and read/decode inputs BEFORE loading 532 MB+ model weights
    enum PreValidatedCommand {
        EmbedInterleaved {
            parts: Vec<OwnedInputItem>,
            task: String,
            title: Option<String>,
            dim: usize,
            encoding: EncodingArg,
            output: OutputArg,
            budget: usize,
        },
        EmbedBatch {
            items: Vec<PreparedBatchItem>,
            is_jsonl: bool,
            task: String,
            title: Option<String>,
            dim: usize,
            encoding: EncodingArg,
            output: OutputArg,
            budget: usize,
        },
        Serve {
            config: glypr_server::ServerConfig,
            warmup_coreml: bool,
        },
    }

    let prepared_cmd = match &cli.command {
        Commands::Embed {
            inputs,
            images,
            audios,
            text_files,
            batch,
            jsonl,
            task,
            title,
            dim,
            encoding,
            output,
            image_token_budget,
        } => {
            validate_dim_arg(*dim, *encoding)?;
            validate_vision_budget(*image_token_budget)?;

            if *jsonl {
                let (items, has_image, has_audio) =
                    parse_jsonl_inputs(inputs.clone(), *dim, *encoding)?;
                validate_media_engine_flags(&cli, has_image, has_audio)?;
                PreValidatedCommand::EmbedBatch {
                    items,
                    is_jsonl: true,
                    task: task.clone(),
                    title: title.clone(),
                    dim: *dim,
                    encoding: *encoding,
                    output: *output,
                    budget: *image_token_budget,
                }
            } else if *batch {
                let (items, has_image, has_audio) = parse_batch_inputs(
                    inputs.clone(),
                    images.clone(),
                    audios.clone(),
                    text_files.clone(),
                )?;
                validate_media_engine_flags(&cli, has_image, has_audio)?;
                PreValidatedCommand::EmbedBatch {
                    items,
                    is_jsonl: false,
                    task: task.clone(),
                    title: title.clone(),
                    dim: *dim,
                    encoding: *encoding,
                    output: *output,
                    budget: *image_token_budget,
                }
            } else {
                let (parts, has_image, has_audio) = parse_interleaved_inputs(
                    inputs.clone(),
                    images.clone(),
                    audios.clone(),
                    text_files.clone(),
                )?;
                validate_media_engine_flags(&cli, has_image, has_audio)?;
                PreValidatedCommand::EmbedInterleaved {
                    parts,
                    task: task.clone(),
                    title: title.clone(),
                    dim: *dim,
                    encoding: *encoding,
                    output: *output,
                    budget: *image_token_budget,
                }
            }
        }
        Commands::Serve {
            host,
            port,
            max_batch_size,
            model_id,
            max_body_mb,
            max_batch_items,
            request_timeout_secs,
            batch_wait_us,
            allow_local_files,
            warmup_coreml,
        } => {
            let defaults = glypr_server::ServerConfig::default();
            let resolved_host = host
                .clone()
                .or_else(|| {
                    std::env::var("GLYPR_HOST")
                        .ok()
                        .filter(|s| !s.trim().is_empty())
                })
                .unwrap_or(defaults.host);
            let resolved_port = port
                .or(env_parse::<u16>("GLYPR_PORT")?)
                .unwrap_or(defaults.port);
            let resolved_batch_size = max_batch_size
                .or(env_parse::<usize>("GLYPR_MAX_BATCH_SIZE")?)
                .unwrap_or(defaults.max_batch_size);
            let resolved_model_id = model_id
                .clone()
                .or_else(|| {
                    std::env::var("GLYPR_MODEL_ID")
                        .ok()
                        .filter(|s| !s.trim().is_empty())
                })
                .unwrap_or(defaults.model_id);
            let resolved_body_mb = max_body_mb
                .or(env_parse::<usize>("GLYPR_MAX_BODY_MB")?)
                .unwrap_or(defaults.max_body_bytes / (1024 * 1024));
            let resolved_batch_items = max_batch_items
                .or(env_parse::<usize>("GLYPR_MAX_BATCH_ITEMS")?)
                .unwrap_or(defaults.max_batch_items);
            let resolved_timeout = request_timeout_secs
                .or(env_parse::<u64>("GLYPR_REQUEST_TIMEOUT_SECS")?)
                .unwrap_or(defaults.request_timeout_secs);
            let resolved_batch_wait_us = batch_wait_us
                .or(env_parse::<u64>("GLYPR_BATCH_WAIT_US")?)
                .unwrap_or(defaults.batch_wait_us);
            let env_allow_local = std::env::var("GLYPR_ALLOW_LOCAL_FILES")
                .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
                .unwrap_or(false);

            let config = glypr_server::ServerConfig {
                host: resolved_host,
                port: resolved_port,
                max_batch_size: resolved_batch_size,
                model_id: resolved_model_id,
                max_body_bytes: resolved_body_mb.max(1) * 1024 * 1024,
                max_batch_items: resolved_batch_items.max(1),
                allow_local_files: *allow_local_files || env_allow_local,
                request_timeout_secs: resolved_timeout.max(1),
                batch_wait_us: resolved_batch_wait_us,
            };
            PreValidatedCommand::Serve {
                config,
                warmup_coreml: warmup_coreml.unwrap_or(true),
            }
        }
        Commands::Pull { .. } | Commands::Info | Commands::Inspect { .. } => unreachable!(),
    };

    // 5. Resolve device/model paths & load engine
    let (device, dtype) = resolve_device_and_dtype(cli.device, cli.metal)?;
    let model_path = resolve_base_model_path(cli.model_file.clone(), cli.q8, offline)?;
    let proj_path = resolve_projector_path(
        cli.mmproj_file.clone(),
        cli.bf16_mmproj,
        cli.q8_mmproj,
        offline,
    )?;

    let mut embedder = GlyprEmbedder::load_gguf(&model_path, &device, dtype)?;
    let attn_mode = resolve_attention_mode(cli.attention, &device, cli.quiet)?;
    embedder.set_attention_mode(attn_mode);

    if let Some(ref p) = proj_path {
        embedder.attach_multimodal_projector(p)?;
    }

    let vision_policy = match cli.vision_backend {
        VisionBackendArg::Auto => VisionBackendPolicy::Auto,
        VisionBackendArg::Ane => VisionBackendPolicy::AneStrict,
        VisionBackendArg::Metal => VisionBackendPolicy::Metal,
    };
    let text_policy = match cli.text_backend {
        TextBackendArg::Auto => TextBackendPolicy::Auto,
        TextBackendArg::Ane => TextBackendPolicy::AneStrict,
        TextBackendArg::Metal => TextBackendPolicy::Metal,
    };
    embedder.set_vision_backend(vision_policy);
    embedder.set_text_backend(text_policy);

    #[cfg(any(feature = "coreml", feature = "ane"))]
    {
        let want_coreml = cli.ane
            || cli.coreml_dir.is_some()
            || cli.vision_backend == VisionBackendArg::Ane
            || cli.text_backend == TextBackendArg::Ane;

        if want_coreml {
            let coreml_dir = resolve_coreml_dir(cli.coreml_dir.as_deref()).ok_or_else(|| {
                anyhow::anyhow!(
                    "CoreML/ANE acceleration requested, but no CoreML models directory was found.\n\
                     Searched: --coreml-dir, GLYPR_COREML_DIR, ./models/coreml, and {}.\n\
                     Export models with: uv run python scripts/export_coreml.py --target all --vision-preset default",
                    get_user_coreml_dir().display()
                )
            })?;
            if !coreml_dir.is_dir() {
                anyhow::bail!(
                    "CoreML models directory not found at: {}",
                    coreml_dir.display()
                );
            }

            let compute_units: CoreMLComputeUnits = cli.coreml_compute_units.into();
            let report = embedder
                .attach_coreml_dir(&coreml_dir, compute_units)
                .with_context(|| {
                    format!("Failed to load CoreML models from {}", coreml_dir.display())
                })?;

            if cli.vision_backend == VisionBackendArg::Ane && !embedder.has_coreml_vision() {
                anyhow::bail!(
                    "--vision-backend ane requested, but no CoreML vision models were found in {}",
                    coreml_dir.display()
                );
            }
            if cli.text_backend == TextBackendArg::Ane && !embedder.has_coreml_text() {
                anyhow::bail!(
                    "--text-backend ane requested, but no CoreML text backbone was found in {}",
                    coreml_dir.display()
                );
            }
            if cli.verbose && !cli.quiet {
                eprintln!(
                    "CoreML attached from {} (vision_grids={:?}, text_seq_lens={:?})",
                    coreml_dir.display(),
                    report.vision_grids,
                    report.text_seq_lens
                );
            }
        }
    }

    // 6. Execute command
    match prepared_cmd {
        PreValidatedCommand::EmbedInterleaved {
            parts: owned_parts,
            task,
            title,
            dim,
            encoding,
            output,
            budget,
        } => {
            let parts: Vec<ModalityPart<'_>> = owned_parts
                .iter()
                .map(|part| match part {
                    OwnedInputItem::Text(s) => ModalityPart::Text(s.as_str()),
                    OwnedInputItem::Image(img) => ModalityPart::Image {
                        image: img,
                        budget: Some(budget),
                    },
                    OwnedInputItem::Audio(samples) => ModalityPart::Audio {
                        samples: samples.as_slice(),
                    },
                })
                .collect();

            let task_enum = Task::from_name_and_title(&task, title.as_deref());
            let truncate_dim = if dim < embedder.embedding_dim() {
                Some(dim)
            } else {
                None
            };
            let vec = embedder.embed_interleaved_with_dim(&parts, task_enum, truncate_dim)?;
            emit_vector_output(&vec, encoding, output, None)?;
        }

        PreValidatedCommand::EmbedBatch {
            items,
            is_jsonl,
            task,
            title,
            dim,
            encoding,
            output,
            budget,
        } => {
            if is_jsonl {
                for (idx, item) in items.iter().enumerate() {
                    let item_task_name = item.task_name.as_deref().unwrap_or(task.as_str());
                    let item_title = item.title.as_deref().or(title.as_deref());
                    let item_task = Task::from_name_and_title(item_task_name, item_title);
                    let item_dim = item.dim.unwrap_or(dim);
                    let item_budget = item.budget.unwrap_or(budget);
                    let truncate_dim = if item_dim < embedder.embedding_dim() {
                        Some(item_dim)
                    } else {
                        None
                    };

                    let single_part = match &item.part {
                        OwnedInputItem::Text(s) => ModalityPart::Text(s.as_str()),
                        OwnedInputItem::Image(img) => ModalityPart::Image {
                            image: img,
                            budget: Some(item_budget),
                        },
                        OwnedInputItem::Audio(samples) => ModalityPart::Audio {
                            samples: samples.as_slice(),
                        },
                    };
                    let parts_slice = [single_part];
                    let (tok_ids, _) = embedder
                        .tokenizer()
                        .encode_modality_sequence(&parts_slice, item_task)?;
                    let vec = embedder.embed_interleaved_with_dim(
                        &parts_slice,
                        item_task,
                        truncate_dim,
                    )?;

                    let emb_json = match encoding {
                        EncodingArg::Float => serde_json::to_value(&vec)?,
                        EncodingArg::Int => {
                            let q = glypr_core::quantize_int8(&vec);
                            serde_json::to_value(q)?
                        }
                        EncodingArg::Binary => {
                            let q = glypr_core::quantize_binary(&vec)?;
                            serde_json::to_value(q)?
                        }
                    };

                    let record = serde_json::json!({
                        "index": idx,
                        "dim": vec.len(),
                        "tokens": tok_ids.len(),
                        "embedding": emb_json,
                    });
                    println!("{}", serde_json::to_string(&record)?);
                }
            } else {
                let default_task = Task::from_name_and_title(&task, title.as_deref());
                let truncate_dim = if dim < embedder.embedding_dim() {
                    Some(dim)
                } else {
                    None
                };

                let mm_inputs: Vec<MultimodalInput<'_>> = items
                    .iter()
                    .map(|item| {
                        let part = match &item.part {
                            OwnedInputItem::Text(s) => ModalityPart::Text(s.as_str()),
                            OwnedInputItem::Image(img) => ModalityPart::Image {
                                image: img,
                                budget: Some(item.budget.unwrap_or(budget)),
                            },
                            OwnedInputItem::Audio(samples) => ModalityPart::Audio {
                                samples: samples.as_slice(),
                            },
                        };
                        MultimodalInput {
                            parts: std::borrow::Cow::Owned(vec![part]),
                            task: default_task,
                        }
                    })
                    .collect();

                let vectors = embedder.embed_batch(&mm_inputs, truncate_dim)?;
                for (idx, vec) in vectors.iter().enumerate() {
                    let pfx = format!("[{idx}]");
                    emit_vector_output(vec, encoding, output, Some(&pfx))?;
                }
            }
        }

        PreValidatedCommand::Serve {
            config,
            warmup_coreml,
        } => {
            let log_level = if cli.verbose {
                "debug"
            } else if cli.quiet {
                "warn"
            } else {
                "info"
            };
            glypr_server::init_tracing(log_level);

            #[cfg(any(feature = "coreml", feature = "ane"))]
            if warmup_coreml && (embedder.has_coreml_vision() || embedder.has_coreml_text()) {
                if !cli.quiet {
                    eprintln!("Warming up CoreML ANE models...");
                }
                embedder.warmup_coreml()?;
            }
            #[cfg(not(any(feature = "coreml", feature = "ane")))]
            let _ = warmup_coreml;

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async { glypr_server::run_server(config, embedder).await })?;
        }
    }

    Ok(())
}
