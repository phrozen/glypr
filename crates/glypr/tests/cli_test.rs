use std::io::Write;
use std::process::{Command, Stdio};

fn allow_missing_weights() -> bool {
    matches!(
        std::env::var("GLYPR_ALLOW_MISSING_WEIGHTS")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true"
    )
}

fn should_skip_weight_test() -> bool {
    if !allow_missing_weights() {
        return false;
    }
    let cache_dir = std::env::var("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .or_else(|_| std::env::var("HF_HOME").map(|h| std::path::PathBuf::from(h).join("hub")))
        .or_else(|_| {
            std::env::var("HOME")
                .map(|h| std::path::PathBuf::from(h).join(".cache/huggingface/hub"))
        })
        .unwrap_or_else(|_| std::path::PathBuf::from(".cache/huggingface/hub"));
    let repo_dir = cache_dir.join("models--ggml-org--embeddinggemma-2-GGUF/snapshots");
    if !repo_dir.is_dir() {
        return true;
    }
    let has_bf16 = std::fs::read_dir(&repo_dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| e.path().join("embeddinggemma-2-BF16.gguf").exists());
    !has_bf16
}

#[test]
fn test_cli_help() {
    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .arg("--help")
        .output()
        .expect("Failed to execute glypr --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("embed"));
    assert!(stdout.contains("serve"));
    assert!(stdout.contains("pull"));
    assert!(stdout.contains("info"));
    assert!(stdout.contains("inspect"));
    assert!(stdout.contains("--offline"));
    assert!(stdout.contains("--text-backend"));
    assert!(!stdout.contains("sim")); // sim was removed for V1
}

#[test]
fn test_cli_serve_help() {
    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["serve", "--help"])
        .output()
        .expect("Failed to execute glypr serve --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in [
        "--host",
        "--port",
        "--max-batch-size",
        "--model-id",
        "--max-body-mb",
        "--max-batch-items",
        "--request-timeout-secs",
        "--batch-wait-us",
        "--allow-local-files",
        "--warmup-coreml",
    ] {
        assert!(
            stdout.contains(expected),
            "serve --help missing '{expected}':\n{stdout}"
        );
    }
}

#[test]
fn test_cli_embed_help_lists_all_tasks_and_new_flags() {
    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "--help"])
        .output()
        .expect("Failed to execute glypr embed --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in [
        "--batch",
        "--jsonl",
        "--image",
        "--audio",
        "--text-file",
        "search_query",
        "document",
        "code_retrieval",
        "question_answering",
        "fact_checking",
        "classification",
        "clustering",
        "sentence_similarity",
        "raw",
    ] {
        assert!(
            stdout.contains(expected),
            "embed --help missing '{expected}':\n{stdout}"
        );
    }
}

#[test]
fn test_cli_info() {
    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .arg("info")
        .output()
        .expect("Failed to execute glypr info");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GLYPR ENGINE INFORMATION"));
    assert!(stdout.contains("Hugging Face Cache:"));
    assert!(stdout.contains("Canonical Models:"));
    assert!(stdout.contains("--bf16"));
    assert!(stdout.contains("--bf16-mmproj"));
    assert!(stdout.contains("Hardware & Runtime:"));
    assert!(stdout.contains("8,192 tokens"));
    assert!(stdout.contains("Attention Kernels (--attention):"));
    assert!(stdout.contains("Supported Tasks (--task):"));
    assert!(stdout.contains("Output Dimensions (--dim):"));
    assert!(stdout.contains("Encoding Format (--encoding):"));
    assert!(stdout.contains("Vision Budget (--image-token-budget):"));
    assert!(stdout.contains("Neural Engine (ANE):"));
}

#[test]
fn test_cli_flag_contradictions_and_preload_validation() {
    // 1. --bf16 --q8 contradiction
    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["--bf16", "--q8", "info"])
        .output()
        .unwrap();
    assert!(!out.status.success());

    // 2. --device cpu --metal contradiction
    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["--device", "cpu", "--metal", "info"])
        .output()
        .unwrap();
    assert!(!out.status.success());

    // 3. --dim 0 and --dim 1024 rejected before loading model
    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "--dim", "0", "hi"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Invalid --dim 0"));

    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "--dim", "1024", "hi"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Invalid --dim 1024"));

    // 4. --encoding binary --dim 100 rejected before loading model
    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "--encoding", "binary", "--dim", "100", "hi"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("multiple of 64"));

    // 5. --image-token-budget out of range rejected before loading model
    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "--image-token-budget", "5", "hi"])
        .output()
        .unwrap();
    assert!(!out.status.success());

    // 6. Missing media extension positional arg errors immediately
    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["--bf16-mmproj", "embed", "./does_not_exist_glypr_test.png"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Media file not found"));

    // 7. Explicit missing --coreml-dir fails fast
    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "--coreml-dir",
            "/tmp/nonexistent_glypr_coreml_dir_999",
            "embed",
            "hi",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());

    // 8. Offline pull cache miss fails cleanly without network
    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["--offline", "--q8", "pull"])
        .env("HF_HUB_CACHE", "/tmp/empty_glypr_hf_cache_test")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("offline mode is enabled"));

    // 8b. --batch conflicts with --jsonl; missing explicit --image / --text-file fail fast;
    // README.md is treated as text (reaches --dim validation rather than failing as media)
    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "--batch", "--jsonl", "hi"])
        .output()
        .unwrap();
    assert!(!out.status.success());

    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "--bf16-mmproj",
            "embed",
            "--image",
            "/tmp/nonexistent_glypr_img.png",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Media file not found"));

    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "--text-file", "/tmp/nonexistent_glypr_txt.txt"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Text file not found"));

    let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "README.md", "--dim", "1024"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Invalid --dim 1024"));

    // 9. Compile-time ANE feature error or contradiction
    #[cfg(not(any(feature = "coreml", feature = "ane")))]
    {
        let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
            .args(["--ane", "embed", "hi"])
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("--features ane"));
    }
    #[cfg(any(feature = "coreml", feature = "ane"))]
    {
        let out = Command::new(env!("CARGO_BIN_EXE_glypr"))
            .args(["--ane", "--vision-backend", "metal", "embed", "hi"])
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("Contradictory flags"));
    }
}

#[test]
fn test_cli_inspect_synthetic_gguf() {
    let tmp = std::env::temp_dir().join(format!("glypr_inspect_test_{}.gguf", std::process::id()));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes()); // version 3
    bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count = 1
    let key = b"general.architecture";
    bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
    bytes.extend_from_slice(key);
    bytes.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING = 8
    let val = b"embeddinggemma2";
    bytes.extend_from_slice(&(val.len() as u64).to_le_bytes());
    bytes.extend_from_slice(val);
    std::fs::write(&tmp, &bytes).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["inspect", tmp.to_str().unwrap()])
        .output()
        .expect("Failed to run glypr inspect");

    let _ = std::fs::remove_file(&tmp);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GGUF File:"));
    assert!(stdout.contains("embeddinggemma2"));
    assert!(stdout.contains("Tensors: 0"));
}

#[test]
fn test_cli_pull_dry_or_cached() {
    if should_skip_weight_test() {
        eprintln!(
            "Skipping test_cli_pull_dry_or_cached: GLYPR_ALLOW_MISSING_WEIGHTS=1 and cache empty"
        );
        return;
    }
    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["--q8", "pull"])
        .output()
        .expect("Failed to execute glypr --q8 pull");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("embeddinggemma-2-Q8_0.gguf"));
}

#[test]
fn test_cli_embed_text_json_truncated_dim() {
    if should_skip_weight_test() {
        eprintln!("Skipping test_cli_embed_text_json_truncated_dim: GLYPR_ALLOW_MISSING_WEIGHTS=1");
        return;
    }
    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "embed",
            "Deep learning and quantum computing",
            "--dim",
            "128",
            "-o",
            "json",
        ])
        .output()
        .expect("Failed to execute glypr embed");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let values: Vec<f32> =
        serde_json::from_str(stdout.trim()).expect("Output should be valid JSON array of floats");
    assert_eq!(values.len(), 128);

    // Verify norm is approximately 1.0 (L2 normalized)
    let norm_sq: f32 = values.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    assert!((norm - 1.0).abs() < 1e-2, "Expected unit norm, got {norm}");
}

#[test]
fn test_cli_embed_stdin_piped() {
    if should_skip_weight_test() {
        eprintln!("Skipping test_cli_embed_stdin_piped: GLYPR_ALLOW_MISSING_WEIGHTS=1");
        return;
    }
    let mut child = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "--dim", "64", "-o", "json"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to spawn glypr embed for stdin test");

    {
        let stdin = child.stdin.as_mut().expect("Failed to open stdin");
        stdin
            .write_all(b"Piped input directly through standard input stream")
            .expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to wait on child");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let values: Vec<f32> = serde_json::from_str(stdout.trim())
        .expect("Piped output should be valid JSON array of floats");
    assert_eq!(values.len(), 64);
}

#[test]
fn test_cli_embed_batch_and_jsonl_and_readme_literal() {
    if should_skip_weight_test() {
        eprintln!("Skipping batch/jsonl/readme test: GLYPR_ALLOW_MISSING_WEIGHTS=1");
        return;
    }

    // 1. Existing non-media file like README.md embeds as literal text
    let readme_out = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "README.md", "--dim", "64", "-o", "json"])
        .output()
        .unwrap();
    assert!(readme_out.status.success());
    let vec: Vec<f32> = serde_json::from_slice(&readme_out.stdout).unwrap();
    assert_eq!(vec.len(), 64);

    // 2. --batch via STDIN lines
    let mut child = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "--batch", "--dim", "64", "-o", "json"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"first line\nsecond line\n")
        .unwrap();
    let batch_out = child.wait_with_output().unwrap();
    assert!(batch_out.status.success());
    let lines: Vec<&str> = std::str::from_utf8(&batch_out.stdout)
        .unwrap()
        .lines()
        .collect();
    assert_eq!(lines.len(), 2);

    // 3. --jsonl streaming
    let mut child = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args(["embed", "--jsonl", "--dim", "64"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"{\"text\":\"hello\",\"task\":\"SearchQuery\"}\nplain line\n")
        .unwrap();
    let jsonl_out = child.wait_with_output().unwrap();
    assert!(jsonl_out.status.success());
    let jsonl_lines: Vec<&str> = std::str::from_utf8(&jsonl_out.stdout)
        .unwrap()
        .lines()
        .collect();
    assert_eq!(jsonl_lines.len(), 2);
    let obj0: serde_json::Value = serde_json::from_str(jsonl_lines[0]).unwrap();
    assert_eq!(obj0["index"], 0);
    assert_eq!(obj0["dim"], 64);
}

#[test]
fn test_cli_embed_q8_int8_json() {
    if should_skip_weight_test() {
        eprintln!("Skipping test_cli_embed_q8_int8_json: GLYPR_ALLOW_MISSING_WEIGHTS=1");
        return;
    }
    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "--q8",
            "embed",
            "Magic: The Gathering Black Lotus card text",
            "--encoding",
            "int8",
            "--dim",
            "64",
            "-o",
            "json",
        ])
        .output()
        .expect("Failed to execute glypr --q8 embed int8");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let values: Vec<i8> = serde_json::from_str(stdout.trim())
        .expect("Output should be valid JSON array of i8 integers");
    assert_eq!(values.len(), 64);

    // Ensure non-trivial values
    assert!(values.iter().any(|&v| v != 0));
}

#[test]
fn test_cli_embed_q8_int_json() {
    if should_skip_weight_test() {
        eprintln!("Skipping test_cli_embed_q8_int_json: GLYPR_ALLOW_MISSING_WEIGHTS=1");
        return;
    }
    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "--q8",
            "embed",
            "Magic: The Gathering Black Lotus card text",
            "--encoding",
            "int",
            "--dim",
            "64",
            "-o",
            "json",
        ])
        .output()
        .expect("Failed to execute glypr --q8 embed int");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let values: Vec<i8> = serde_json::from_str(stdout.trim())
        .expect("Output should be valid JSON array of i8 integers");
    assert_eq!(values.len(), 64);

    // Ensure non-trivial values
    assert!(values.iter().any(|&v| v != 0));
}

#[test]
fn test_cli_embed_binary_raw_bytes() {
    if should_skip_weight_test() {
        eprintln!("Skipping test_cli_embed_binary_raw_bytes: GLYPR_ALLOW_MISSING_WEIGHTS=1");
        return;
    }
    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "embed",
            "Binary quantization raw stream test",
            "--encoding",
            "binary",
            "--dim",
            "64",
            "-o",
            "raw",
        ])
        .output()
        .expect("Failed to execute glypr embed binary raw");

    assert!(output.status.success());
    // 64 dimensions / 8 bits per byte = exactly 8 bytes of raw binary payload
    assert_eq!(output.stdout.len(), 8);
}

#[test]
fn test_cli_embed_multimodal_interleaved() {
    if should_skip_weight_test() {
        eprintln!("Skipping test_cli_embed_multimodal_interleaved: GLYPR_ALLOW_MISSING_WEIGHTS=1");
        return;
    }
    let image_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden_vectors/multimodal_test_image.png"
    );

    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "--bf16-mmproj",
            "embed",
            "Card illustration:",
            image_path,
            "and title card.",
            "--dim",
            "64",
            "-o",
            "json",
        ])
        .output()
        .expect("Failed to execute glypr embed multimodal");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let values: Vec<f32> = serde_json::from_str(stdout.trim())
        .expect("Multimodal output should be valid JSON array of floats");
    assert_eq!(values.len(), 64);
}

#[test]
fn test_cli_embed_multimodal_fails_without_mmproj() {
    let image_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden_vectors/multimodal_test_image.png"
    );

    let output = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "embed",
            "Card illustration:",
            image_path,
            "and title card.",
            "--dim",
            "64",
        ])
        .output()
        .expect("Failed to execute glypr embed");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(
            "Media input detected in arguments, but no multimodal projector was attached"
        )
    );
    assert!(stderr.contains("--bf16-mmproj"));
}

#[test]
fn test_cli_attention_modes() {
    if should_skip_weight_test() {
        eprintln!("Skipping test_cli_attention_modes: GLYPR_ALLOW_MISSING_WEIGHTS=1");
        return;
    }
    // 1. Explicit --attention chunked
    let output_chunked = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "--attention",
            "chunked",
            "embed",
            "Attention mode chunked verification",
            "--dim",
            "32",
            "-o",
            "json",
        ])
        .output()
        .expect("Failed to execute glypr with --attention chunked");
    assert!(output_chunked.status.success());

    // 2. Explicit --attention flash
    let output_flash = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "--attention",
            "flash",
            "embed",
            "Attention mode flash verification",
            "--dim",
            "32",
            "-o",
            "json",
        ])
        .output()
        .expect("Failed to execute glypr with --attention flash");
    assert!(output_flash.status.success());
}

#[test]
fn test_cli_embed_jpeg_and_webp() {
    if should_skip_weight_test() {
        eprintln!("Skipping test_cli_embed_jpeg_and_webp: GLYPR_ALLOW_MISSING_WEIGHTS=1");
        return;
    }
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden_vectors/multimodal_test_image.png"
    );
    let img = image::open(png_path).expect("Failed to open test PNG");

    let tmp_dir = std::env::temp_dir();
    let jpg_path = tmp_dir.join("glypr_test_image.jpg");
    let webp_path = tmp_dir.join("glypr_test_image.webp");

    img.save(&jpg_path).expect("Failed to save test JPEG");
    img.save(&webp_path).expect("Failed to save test WebP");

    // 1. Embed JPEG
    let output_jpg = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "--bf16-mmproj",
            "embed",
            "JPEG card image:",
            jpg_path.to_str().unwrap(),
            "--dim",
            "64",
            "-o",
            "json",
        ])
        .output()
        .expect("Failed to execute glypr embed with JPEG");
    assert!(output_jpg.status.success());
    let stdout_jpg = String::from_utf8_lossy(&output_jpg.stdout);
    let values_jpg: Vec<f32> = serde_json::from_str(stdout_jpg.trim())
        .expect("JPEG output should be valid JSON array of floats");
    assert_eq!(values_jpg.len(), 64);

    // 2. Embed WebP
    let output_webp = Command::new(env!("CARGO_BIN_EXE_glypr"))
        .args([
            "--bf16-mmproj",
            "embed",
            "WebP card image:",
            webp_path.to_str().unwrap(),
            "--dim",
            "64",
            "-o",
            "json",
        ])
        .output()
        .expect("Failed to execute glypr embed with WebP");
    assert!(output_webp.status.success());
    let stdout_webp = String::from_utf8_lossy(&output_webp.stdout);
    let values_webp: Vec<f32> = serde_json::from_str(stdout_webp.trim())
        .expect("WebP output should be valid JSON array of floats");
    assert_eq!(values_webp.len(), 64);

    let _ = std::fs::remove_file(jpg_path);
    let _ = std::fs::remove_file(webp_path);
}
