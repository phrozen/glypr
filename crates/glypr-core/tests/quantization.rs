use anyhow::Result;
use candle_core::Device;
use glypr_core::util::{
    binary_cosine_similarity, cosine_similarity, hamming_distance, int8_cosine_similarity,
    quantize_binary, quantize_int8, truncate_mrl,
};
use glypr_core::{GlyprEmbedder, Task};

#[test]
fn test_quantization_unit_invariants() {
    // 1. Binary requires multiple of 64
    assert!(quantize_binary(&[1.0; 63]).is_err());
    assert!(quantize_binary(&[1.0; 65]).is_err());
    assert!(quantize_binary(&[1.0; 128]).is_ok());

    let a64 = vec![1.0; 64];
    let b64 = vec![-1.0; 64];
    let bin_a = quantize_binary(&a64).unwrap();
    let bin_b = quantize_binary(&b64).unwrap();
    assert_eq!(bin_a.len(), 1);
    assert_eq!(bin_b.len(), 1);
    assert_eq!(hamming_distance(&bin_a, &bin_b), 64);
    assert_eq!(hamming_distance(&bin_a, &bin_a), 0);
    assert!((binary_cosine_similarity(&bin_a, &bin_b) - (-1.0)).abs() < 1e-6);
    assert!((binary_cosine_similarity(&bin_a, &bin_a) - 1.0).abs() < 1e-6);

    // 2. Symmetric INT8 maps [-1.0, 1.0] to [-127, 127]
    let static_in = vec![1.0, -1.0, 0.0, 0.5];
    let static_q = quantize_int8(&static_in);
    assert_eq!(static_q, vec![127, -127, 0, 64]);
    assert!((int8_cosine_similarity(&static_q, &static_q) - 1.0).abs() < 1e-5);
}

#[test]
fn test_quantization_mrl_parity_across_dimensions() -> Result<()> {
    let device = if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    let model_path = glypr_core::hub::resolve_default_gguf()?;
    if !model_path.exists() {
        eprintln!("Model not found at {:?}, skipping parity test", model_path);
        return Ok(());
    }

    let embedder = GlyprEmbedder::load_gguf_bf16(&model_path, &device)?;

    // Define 3 real text items for semantic similarity and ranking tests
    let query = "what is gravitational lensing in astronomy?";
    let doc_relevant = "Gravitational lensing occurs when a massive celestial body warps spacetime, deflecting light from distant sources.";
    let doc_unrelated =
        "To bake sourdough bread, mix active starter with flour, water, and salt before proofing.";

    let q_emb_768 = embedder.embed_text(query, Task::SearchQuery)?;
    let rel_emb_768 = embedder.embed_text(
        doc_relevant,
        Task::Document {
            title: Some("Astrophysics"),
        },
    )?;
    let unrel_emb_768 = embedder.embed_text(
        doc_unrelated,
        Task::Document {
            title: Some("Cooking"),
        },
    )?;

    let dimensions = [768, 512, 256, 128];

    println!("\n=== MRL & QUANTIZATION FIDELITY REPORT ===");
    println!(
        "{:<6} | {:<12} | {:<14} | {:<14} | {:<10}",
        "Dim", "Metric Pair", "Float32 Cosim", "INT8 Cosim", "Binary Cos"
    );
    println!("{}", "-".repeat(66));

    for &dim in &dimensions {
        // Truncate embeddings using MRL
        let q_dim = truncate_mrl(&q_emb_768, dim)?;
        let rel_dim = truncate_mrl(&rel_emb_768, dim)?;
        let unrel_dim = truncate_mrl(&unrel_emb_768, dim)?;

        // 1. Float32 baseline
        let f32_rel = cosine_similarity(&q_dim, &rel_dim);
        let f32_unrel = cosine_similarity(&q_dim, &unrel_dim);

        // 2. INT8 (symmetric [-127, 127], zero metadata)
        let q_int8 = quantize_int8(&q_dim);
        let rel_int8 = quantize_int8(&rel_dim);
        let unrel_int8 = quantize_int8(&unrel_dim);

        let int8_rel = int8_cosine_similarity(&q_int8, &rel_int8);
        let int8_unrel = int8_cosine_similarity(&q_int8, &unrel_int8);

        // 3. Binary (1-bit / POPCNT angular similarity)
        let q_bin = quantize_binary(&q_dim)?;
        let rel_bin = quantize_binary(&rel_dim)?;
        let unrel_bin = quantize_binary(&unrel_dim)?;

        let bin_rel = binary_cosine_similarity(&q_bin, &rel_bin);
        let bin_unrel = binary_cosine_similarity(&q_bin, &unrel_bin);

        println!(
            "{:<6} | {:<12} | {:<14.4} | {:<14.4} | {:<10.4}",
            dim, "Relevant", f32_rel, int8_rel, bin_rel
        );
        println!(
            "{:<6} | {:<12} | {:<14.4} | {:<14.4} | {:<10.4}",
            dim, "Unrelated", f32_unrel, int8_unrel, bin_unrel
        );

        // Assertions:
        // A. INT8 vs Float32 delta is very small (< 0.015)
        let delta_int8_rel = (f32_rel - int8_rel).abs();
        let delta_int8_unrel = (f32_unrel - int8_unrel).abs();
        assert!(
            delta_int8_rel < 0.015,
            "INT8 rel delta {} too large at dim {}",
            delta_int8_rel,
            dim
        );
        assert!(
            delta_int8_unrel < 0.015,
            "INT8 unrel delta {} too large at dim {}",
            delta_int8_unrel,
            dim
        );

        // B. Ranking preservation: relevant must always score significantly higher than unrelated
        assert!(
            f32_rel > f32_unrel + 0.20,
            "Float32 failed ranking separation at dim {}",
            dim
        );
        assert!(
            int8_rel > int8_unrel + 0.20,
            "INT8 failed ranking separation at dim {}",
            dim
        );
        assert!(
            bin_rel > bin_unrel + 0.15,
            "Binary failed ranking separation at dim {}",
            dim
        );
    }
    println!("{}", "-".repeat(66));

    Ok(())
}
