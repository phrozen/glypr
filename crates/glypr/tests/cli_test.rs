use std::io::Write;
use std::process::{Command, Stdio};

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
    assert!(!stdout.contains("sim")); // sim was removed for V1
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
    assert!(stdout.contains("Output Dimensions (--dim):"));
    assert!(stdout.contains("Encoding Format (--encoding):"));
    assert!(stdout.contains("Vision Budget (--image-token-budget):"));
}

#[test]
fn test_cli_pull_dry_or_cached() {
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
    let values: Vec<f32> = serde_json::from_str(stdout.trim())
        .expect("Output should be valid JSON array of floats");
    assert_eq!(values.len(), 128);

    // Verify norm is approximately 1.0 (L2 normalized)
    let norm_sq: f32 = values.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    assert!((norm - 1.0).abs() < 1e-2, "Expected unit norm, got {norm}");
}

#[test]
fn test_cli_embed_stdin_piped() {
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
fn test_cli_embed_q8_int8_json() {
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
    assert!(stderr.contains("Media input detected in arguments, but no multimodal projector was attached"));
    assert!(stderr.contains("--bf16-mmproj"));
}

#[test]
fn test_cli_attention_modes() {
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

