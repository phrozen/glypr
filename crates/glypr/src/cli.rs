use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use glypr_core::{
    CoreMLComputeUnits, Task,
    hub::{resolve_canonical_mmproj_with_options, resolve_canonical_model_with_options},
    validate_vision_budget,
};
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum DeviceArg {
    #[default]
    Auto,
    Metal,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum EncodingArg {
    #[default]
    Float,
    #[value(alias = "int8")]
    Int,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum OutputArg {
    #[default]
    Preview,
    Json,
    Raw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum AttentionArg {
    #[default]
    Auto,
    Flash,
    Chunked,
    Standard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum VisionBackendArg {
    #[default]
    Auto,
    Ane,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum TextBackendArg {
    #[default]
    Auto,
    Ane,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum CoreMLComputeUnitsArg {
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

pub fn task_help_footer() -> String {
    let mut s = String::from("Supported --task names:\n");
    for (name, desc) in Task::all_names() {
        s.push_str(&format!("  {:<20} {}\n", name, desc));
    }
    s.push_str("  <custom>             Custom prefix or 'task: <name> | query: '");
    s
}

#[derive(Debug, Parser)]
#[command(
    name = "glypr",
    version,
    about = "Native Rust Embedding & Retrieval Engine for EmbeddingGemma 2"
)]
pub struct Cli {
    /// Use canonical BF16 base weights (`embeddinggemma-2-BF16.gguf`) (default)
    #[arg(long, global = true, conflicts_with = "q8")]
    pub bf16: bool,

    /// Use canonical Q8_0 base weights (`embeddinggemma-2-Q8_0.gguf`)
    #[arg(long, global = true, conflicts_with = "bf16")]
    pub q8: bool,

    /// Attach canonical BF16 multimodal projector (`mmproj-embeddinggemma-2-BF16.gguf`)
    #[arg(long, global = true, conflicts_with = "q8_mmproj")]
    pub bf16_mmproj: bool,

    /// Attach canonical Q8_0 multimodal projector (`mmproj-embeddinggemma-2-Q8_0.gguf`)
    #[arg(long, global = true, conflicts_with = "bf16_mmproj")]
    pub q8_mmproj: bool,

    /// Override base model with local GGUF file (disables automatic downloads)
    #[arg(long, global = true, value_name = "FILE")]
    pub model_file: Option<PathBuf>,

    /// Override multimodal projector with local GGUF file
    #[arg(long, global = true, value_name = "FILE")]
    pub mmproj_file: Option<PathBuf>,

    /// Hardware acceleration device: auto (default), metal, cpu
    #[arg(long, global = true, value_enum, default_value_t = DeviceArg::Auto, conflicts_with = "metal")]
    pub device: DeviceArg,

    /// Attention kernel: auto (default), flash, chunked, standard
    #[arg(long, global = true, value_enum, default_value_t = AttentionArg::Auto)]
    pub attention: AttentionArg,

    /// Enable Apple Neural Engine (ANE) CoreML acceleration (vision + pure-text routing)
    #[arg(long, global = true)]
    pub ane: bool,

    /// Vision execution backend: auto (default), ane, metal
    #[arg(long, global = true, value_enum, default_value_t = VisionBackendArg::Auto)]
    pub vision_backend: VisionBackendArg,

    /// Pure-text ANE execution policy: auto (default, 64..=seq_len & batch<=4), ane, metal
    #[arg(long, global = true, value_enum, default_value_t = TextBackendArg::Auto)]
    pub text_backend: TextBackendArg,

    /// Path to compiled CoreML models directory (.mlmodelc)
    #[arg(long, global = true, value_name = "DIR")]
    pub coreml_dir: Option<PathBuf>,

    /// CoreML compute units: ane (default), all, gpu, cpu
    #[arg(long, global = true, value_enum, default_value_t = CoreMLComputeUnitsArg::Ane)]
    pub coreml_compute_units: CoreMLComputeUnitsArg,

    /// Offline mode: fail immediately if weights are not in local Hugging Face cache
    #[arg(long, global = true)]
    pub offline: bool,

    /// Enable verbose diagnostic logging on stderr
    #[arg(short, long, global = true, conflicts_with = "quiet")]
    pub verbose: bool,

    /// Suppress non-essential diagnostic output on stderr
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,

    /// Legacy flag for Metal acceleration (alias for --device metal)
    #[arg(long, global = true, hide = true, conflicts_with = "device")]
    pub metal: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
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

        /// Image soft token budget (default: 280, range: 16-2048)
        #[arg(long, default_value_t = 280)]
        image_token_budget: usize,
    },

    /// Launch an OpenAI-compatible HTTP embedding server
    Serve {
        /// Host address to bind to (env: `GLYPR_HOST`)
        #[arg(long)]
        host: Option<String>,

        /// Port to listen on (env: `GLYPR_PORT`)
        #[arg(short, long)]
        port: Option<u16>,

        /// Maximum dynamic coalescing batch size (env: `GLYPR_MAX_BATCH_SIZE`)
        #[arg(long)]
        max_batch_size: Option<usize>,

        /// Served model identifier in `/v1/models` and responses (env: `GLYPR_MODEL_ID`)
        #[arg(long)]
        model_id: Option<String>,

        /// Maximum HTTP request body size in MiB (env: `GLYPR_MAX_BODY_MB`, default: 32)
        #[arg(long)]
        max_body_mb: Option<usize>,

        /// Maximum items per `/v1/embeddings` request array (env: `GLYPR_MAX_BATCH_ITEMS`, default: 256)
        #[arg(long)]
        max_batch_items: Option<usize>,

        /// Per-request inference timeout in seconds (env: `GLYPR_REQUEST_TIMEOUT_SECS`, default: 120)
        #[arg(long)]
        request_timeout_secs: Option<u64>,

        /// Micro-batch coalescing wait window in microseconds (0 = no wait, env: `GLYPR_BATCH_WAIT_US`)
        #[arg(long)]
        batch_wait_us: Option<u64>,

        /// Opt-in to allow reading local filesystem paths from HTTP media inputs (env: `GLYPR_ALLOW_LOCAL_FILES`)
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

    /// Inspect metadata, shapes, and tensors of a `.gguf` file or `.mlmodelc` CoreML bundle
    Inspect {
        /// Path to a `.gguf` file or `.mlmodelc` directory
        path: PathBuf,
    },
}

pub fn is_recognized_media_ext(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "webp" => Some("image"),
        "wav" => Some("audio"),
        _ => None,
    }
}

pub fn validate_coreml_feature_and_flags(cli: &Cli) -> Result<()> {
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

pub fn resolve_base_model_path(
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

pub fn resolve_projector_path(
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

pub fn resolve_device_and_dtype(
    device_arg: DeviceArg,
    legacy_metal: bool,
) -> Result<(Device, DType)> {
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

pub fn resolve_attention_mode(
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

#[derive(Debug, Clone)]
pub enum OwnedInputItem {
    Text(String),
    Image(image::DynamicImage),
    Audio(Vec<f32>),
}

pub fn load_media_path(p: &Path, kind: &str) -> Result<OwnedInputItem> {
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

pub fn parse_interleaved_inputs(
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

#[derive(Debug, Clone)]
pub struct PreparedBatchItem {
    pub part: OwnedInputItem,
    pub task_name: Option<String>,
    pub title: Option<String>,
    pub budget: Option<usize>,
    pub dim: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct JsonlInputLine {
    pub text: Option<String>,
    pub task: Option<String>,
    pub title: Option<String>,
    pub image: Option<String>,
    pub audio: Option<String>,
    pub budget: Option<usize>,
    pub dim: Option<usize>,
}

impl JsonlInputLine {
    pub fn from_json_str(raw: &str) -> Result<Self> {
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

pub fn parse_batch_inputs(
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

pub fn parse_jsonl_inputs(
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

pub fn validate_dim_arg(dim: usize, encoding: EncodingArg) -> Result<()> {
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

pub fn validate_media_engine_flags(cli: &Cli, has_image: bool, has_audio: bool) -> Result<()> {
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

pub fn format_bytes(bytes: u64) -> String {
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

pub fn env_parse<T: std::str::FromStr>(key: &str) -> Result<Option<T>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_cli_try_parse_from_subcommands_aliases_and_conflicts() {
        // 1. embed with aliases (--encoding int8, -d, -t, -o, --batch)
        let cli = Cli::try_parse_from([
            "glypr",
            "--q8",
            "embed",
            "hello",
            "--encoding",
            "int8",
            "-d",
            "256",
            "-t",
            "SearchQuery",
            "-o",
            "json",
            "--batch",
        ])
        .unwrap();
        assert!(cli.q8);
        match cli.command {
            Commands::Embed {
                inputs,
                dim,
                encoding,
                task,
                output,
                batch,
                jsonl,
                image_token_budget,
                ..
            } => {
                assert_eq!(inputs, vec!["hello"]);
                assert_eq!(dim, 256);
                assert_eq!(encoding, EncodingArg::Int);
                assert_eq!(task, "SearchQuery");
                assert_eq!(output, OutputArg::Json);
                assert!(batch);
                assert!(!jsonl);
                assert_eq!(image_token_budget, 280);
            }
            other => panic!("expected Embed, got {other:?}"),
        }

        // 2. serve defaults and explicit options
        let cli_serve = Cli::try_parse_from([
            "glypr",
            "--ane",
            "serve",
            "--port",
            "9090",
            "--allow-local-files",
            "--warmup-coreml=false",
        ])
        .unwrap();
        assert!(cli_serve.ane);
        match cli_serve.command {
            Commands::Serve {
                port,
                allow_local_files,
                warmup_coreml,
                ..
            } => {
                assert_eq!(port, Some(9090));
                assert!(allow_local_files);
                assert_eq!(warmup_coreml, Some(false));
            }
            other => panic!("expected Serve, got {other:?}"),
        }

        // 3. pull & info
        assert!(Cli::try_parse_from(["glypr", "pull", "--all", "--safetensors"]).is_ok());
        assert!(Cli::try_parse_from(["glypr", "info"]).is_ok());

        // 4. Conflicts & invalid flags
        assert!(Cli::try_parse_from(["glypr", "--bf16", "--q8", "info"]).is_err());
        assert!(Cli::try_parse_from(["glypr", "--bf16-mmproj", "--q8-mmproj", "info"]).is_err());
        assert!(Cli::try_parse_from(["glypr", "--device", "cpu", "--metal", "info"]).is_err());
        assert!(Cli::try_parse_from(["glypr", "--verbose", "--quiet", "info"]).is_err());
        assert!(Cli::try_parse_from(["glypr", "embed", "--batch", "--jsonl", "x"]).is_err());
        assert!(Cli::try_parse_from(["glypr", "--not-a-flag", "info"]).is_err());
    }

    #[test]
    fn test_resolve_projector_and_base_model_paths() {
        // Conflicting mmproj flags rejected
        let err = resolve_projector_path(None, true, true, true).unwrap_err();
        assert!(err.to_string().contains("--bf16-mmproj"));

        // Nonexistent --mmproj-file / --model-file rejected
        let missing = PathBuf::from("/tmp/nonexistent_glypr_unit_test_file.gguf");
        assert!(resolve_projector_path(Some(missing.clone()), false, false, true).is_err());
        assert!(resolve_base_model_path(Some(missing), false, true).is_err());

        // None when nothing requested
        assert_eq!(
            resolve_projector_path(None, false, false, true).unwrap(),
            None
        );

        // Existing temp file resolves cleanly
        let tmp_file =
            std::env::temp_dir().join(format!("glypr_cli_test_dummy_{}.gguf", std::process::id()));
        std::fs::write(&tmp_file, b"GGUF").unwrap();
        assert_eq!(
            resolve_base_model_path(Some(tmp_file.clone()), false, true).unwrap(),
            tmp_file
        );
        assert_eq!(
            resolve_projector_path(Some(tmp_file.clone()), false, false, true).unwrap(),
            Some(tmp_file.clone())
        );
        let _ = std::fs::remove_file(&tmp_file);
    }

    #[test]
    fn test_resolve_device_and_attention_mode() {
        let (cpu_dev, cpu_dtype) = resolve_device_and_dtype(DeviceArg::Cpu, false).unwrap();
        assert!(cpu_dev.is_cpu());
        assert_eq!(cpu_dtype, DType::F32);

        #[cfg(not(feature = "metal"))]
        {
            let err = resolve_device_and_dtype(DeviceArg::Metal, false).unwrap_err();
            assert!(err.to_string().contains("--features metal"));
        }

        assert_eq!(
            resolve_attention_mode(AttentionArg::Auto, &Device::Cpu, true).unwrap(),
            glypr_core::AttentionMode::ChunkedSliding
        );
        assert_eq!(
            resolve_attention_mode(AttentionArg::Flash, &Device::Cpu, true).unwrap(),
            glypr_core::AttentionMode::ChunkedSliding
        );
        assert_eq!(
            resolve_attention_mode(AttentionArg::Standard, &Device::Cpu, true).unwrap(),
            glypr_core::AttentionMode::Standard
        );
    }

    #[test]
    fn test_parse_interleaved_inputs_and_media_validation() {
        let tmp_dir =
            std::env::temp_dir().join(format!("glypr_cli_parse_test_{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        // Create valid PNG and WAV files
        let png_path = tmp_dir.join("sample.png");
        let img = image::RgbImage::from_pixel(16, 16, image::Rgb([10, 20, 30]));
        image::DynamicImage::ImageRgb8(img).save(&png_path).unwrap();

        let wav_path = tmp_dir.join("sample.wav");
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut buf = Vec::new();
            let mut writer = hound::WavWriter::new(Cursor::new(&mut buf), spec).unwrap();
            for i in 0..320 {
                writer.write_sample((i * 10) as i16).unwrap();
            }
            writer.finalize().unwrap();
            std::fs::write(&wav_path, buf).unwrap();
        }

        // .txt / .mp4 are not recognized media extensions (treated as literal text)
        assert_eq!(is_recognized_media_ext(Path::new("notes.txt")), None);
        assert_eq!(is_recognized_media_ext(Path::new("clip.mp4")), None);
        assert_eq!(is_recognized_media_ext(&png_path), Some("image"));
        assert_eq!(is_recognized_media_ext(&wav_path), Some("audio"));

        let (parts, has_image, has_audio) = parse_interleaved_inputs(
            vec![
                "prefix text".to_string(),
                png_path.to_string_lossy().to_string(),
                wav_path.to_string_lossy().to_string(),
                "clip.mp4".to_string(),
            ],
            vec![],
            vec![],
            vec![],
        )
        .unwrap();
        assert!(has_image);
        assert!(has_audio);
        assert_eq!(parts.len(), 4);
        assert!(matches!(parts[0], OwnedInputItem::Text(ref s) if s == "prefix text"));
        assert!(matches!(parts[1], OwnedInputItem::Image(_)));
        assert!(matches!(parts[2], OwnedInputItem::Audio(_)));
        assert!(matches!(parts[3], OwnedInputItem::Text(ref s) if s == "clip.mp4"));

        // Explicit --image with non-image / missing file fails
        let txt_path = tmp_dir.join("not_an_image.txt");
        std::fs::write(&txt_path, "just text").unwrap();
        assert!(parse_interleaved_inputs(vec![], vec![txt_path.clone()], vec![], vec![]).is_err());
        assert!(
            parse_interleaved_inputs(vec![], vec![tmp_dir.join("missing.png")], vec![], vec![])
                .is_err()
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_validate_dim_and_format_bytes() {
        assert!(validate_dim_arg(0, EncodingArg::Float).is_err());
        assert!(validate_dim_arg(769, EncodingArg::Float).is_err());
        assert!(validate_dim_arg(100, EncodingArg::Binary).is_err());
        assert!(validate_dim_arg(128, EncodingArg::Binary).is_ok());
        assert!(validate_dim_arg(768, EncodingArg::Float).is_ok());

        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.0 MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.0 GB");
    }
}
