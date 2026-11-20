use anyhow::{Context, Result};
use clap::Parser;
use glypr::cli::{
    Cli, Commands, EncodingArg, OutputArg, OwnedInputItem, PreparedBatchItem, TextBackendArg,
    VisionBackendArg, env_parse, format_bytes, parse_batch_inputs, parse_interleaved_inputs,
    parse_jsonl_inputs, resolve_attention_mode, resolve_base_model_path, resolve_device_and_dtype,
    resolve_projector_path, validate_coreml_feature_and_flags, validate_dim_arg,
    validate_media_engine_flags,
};
#[cfg(any(feature = "coreml", feature = "ane"))]
use glypr_core::hub::resolve_coreml_dir;
use glypr_core::{
    CoreMLComputeUnits, GlyprEmbedder, ModalityPart, MultimodalInput, Task, TextBackendPolicy,
    VisionBackendPolicy,
    hub::{
        DEFAULT_GGUF_REPO, GGUF_BF16_FILE, GGUF_Q8_0_FILE, GOOGLE_OFFICIAL_REPO,
        GOOGLE_SAFETENSORS_FILE, GOOGLE_TOKENIZER_FILE, MMPROJ_BF16_FILE, MMPROJ_Q8_0_FILE,
        check_cached_file, get_user_coreml_dir, is_offline_mode, resolve_file_with_options,
    },
    validate_vision_budget,
};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

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
