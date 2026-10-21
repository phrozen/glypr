use glypr_core::util::{
    binary_cosine_similarity, cosine_similarity, hamming_distance, int8_cosine_similarity,
    int8_dot_product, l2_normalize, quantize_binary, quantize_int8,
};
use std::time::Instant;

#[test]
fn bench_quantization_and_similarity_across_dimensions() {
    let dimensions = [768, 512, 256, 128];

    println!(
        "\n=========================================================================================="
    );
    println!(
        "                           QUANTIZATION & DISTANCE BENCHMARKS                             "
    );
    println!(
        "=========================================================================================="
    );

    // =========================================================================
    // PART 1: QUANTIZATION ENCODING BENCHMARK (Float32 -> Quantized format)
    // =========================================================================
    println!("\n--- Part 1: Quantization Encoding Latency & Throughput ---");
    println!(
        "{:<6} | {:<16} | {:<14} | {:<14} | {:<12}",
        "Dim", "Quant Scheme", "Latency (ns)", "Throughput (Mv/s)", "Relative Speed"
    );
    println!("{}", "-".repeat(74));

    let enc_iters = 50_000;

    for &dim in &dimensions {
        let raw_v: Vec<f32> = (0..dim)
            .map(|i| (i as f32 * 0.1).sin() + (i as f32 * 0.2).cos())
            .collect();
        let norm_v = l2_normalize(&raw_v);

        // 1. INT8 (symmetric [-127, 127], zero metadata)
        let _ = quantize_int8(&norm_v);
        let t0 = Instant::now();
        for _ in 0..enc_iters {
            std::hint::black_box(quantize_int8(std::hint::black_box(&norm_v)));
        }
        let int8_ns = t0.elapsed().as_nanos() as f64 / enc_iters as f64;
        let int8_mvecs = 1_000.0 / int8_ns;

        // 2. Binary (1-bit / 64-bit word packing)
        let _ = quantize_binary(&norm_v).unwrap();
        let t0 = Instant::now();
        for _ in 0..enc_iters {
            std::hint::black_box(quantize_binary(std::hint::black_box(&norm_v)).unwrap());
        }
        let bin_ns = t0.elapsed().as_nanos() as f64 / enc_iters as f64;
        let bin_mvecs = 1_000.0 / bin_ns;

        println!(
            "{:<6} | {:<16} | {:<14.1} | {:<14.2} | {:<12}",
            dim, "INT8 (1 byte/d)", int8_ns, int8_mvecs, "1.00x (baseline)"
        );
        println!(
            "{:<6} | {:<16} | {:<14.1} | {:<14.2} | {:<10.2}x",
            dim,
            "Binary (1 bit/d)",
            bin_ns,
            bin_mvecs,
            bin_ns / int8_ns
        );
        println!("{}", "-".repeat(74));
    }

    // =========================================================================
    // PART 2: DISTANCE & SIMILARITY COMPUTE BENCHMARK
    // =========================================================================
    println!("\n--- Part 2: Similarity & Distance Compute Latency & Throughput ---");
    println!(
        "{:<6} | {:<22} | {:<14} | {:<14} | {:<12}",
        "Dim", "Distance / Similarity", "Latency (ns)", "Throughput (Mop/s)", "Speedup vs F32"
    );
    println!("{}", "-".repeat(80));

    let sim_iters = 500_000;

    for &dim in &dimensions {
        let raw_a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).sin()).collect();
        let raw_b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05 + 0.1).sin()).collect();
        let a_f32 = l2_normalize(&raw_a);
        let b_f32 = l2_normalize(&raw_b);

        // Pre-quantize vectors for distance benchmarks
        let a_int8 = quantize_int8(&a_f32);
        let b_int8 = quantize_int8(&b_f32);

        let a_bin = quantize_binary(&a_f32).unwrap();
        let b_bin = quantize_binary(&b_f32).unwrap();

        // 1. Float32 Cosine Similarity
        let _ = cosine_similarity(&a_f32, &b_f32);
        let t0 = Instant::now();
        for _ in 0..sim_iters {
            std::hint::black_box(cosine_similarity(&a_f32, &b_f32));
        }
        let f32_ns = t0.elapsed().as_nanos() as f64 / sim_iters as f64;
        let f32_mops = 1_000.0 / f32_ns;

        // 2a. INT8 Dot Product (SIMD single dot product, unit-norm cosine equivalent)
        let _ = int8_dot_product(&a_int8, &b_int8);
        let t0 = Instant::now();
        for _ in 0..sim_iters {
            std::hint::black_box(int8_dot_product(&a_int8, &b_int8));
        }
        let int8_dot_ns = t0.elapsed().as_nanos() as f64 / sim_iters as f64;
        let int8_dot_mops = 1_000.0 / int8_dot_ns;

        // 2b. INT8 Cosine Similarity (3x SIMD dot products for on-the-fly norm)
        let _ = int8_cosine_similarity(&a_int8, &b_int8);
        let t0 = Instant::now();
        for _ in 0..sim_iters {
            std::hint::black_box(int8_cosine_similarity(&a_int8, &b_int8));
        }
        let int8_cos_ns = t0.elapsed().as_nanos() as f64 / sim_iters as f64;
        let int8_cos_mops = 1_000.0 / int8_cos_ns;

        // 3. Binary POPCNT Hamming Distance
        let _ = hamming_distance(&a_bin, &b_bin);
        let t0 = Instant::now();
        for _ in 0..sim_iters {
            std::hint::black_box(hamming_distance(&a_bin, &b_bin));
        }
        let bin_hamm_ns = t0.elapsed().as_nanos() as f64 / sim_iters as f64;
        let bin_hamm_mops = 1_000.0 / bin_hamm_ns;

        // 4. Binary Angular Cosine Similarity
        let _ = binary_cosine_similarity(&a_bin, &b_bin);
        let t0 = Instant::now();
        for _ in 0..sim_iters {
            std::hint::black_box(binary_cosine_similarity(&a_bin, &b_bin));
        }
        let bin_cos_ns = t0.elapsed().as_nanos() as f64 / sim_iters as f64;
        let bin_cos_mops = 1_000.0 / bin_cos_ns;

        println!(
            "{:<6} | {:<22} | {:<14.2} | {:<14.1} | {:<12}",
            dim, "Float32 Cosine", f32_ns, f32_mops, "1.00x"
        );
        println!(
            "{:<6} | {:<22} | {:<14.2} | {:<14.1} | {:<12.2}x",
            dim,
            "INT8 Dot (SIMD)",
            int8_dot_ns,
            int8_dot_mops,
            f32_ns / int8_dot_ns
        );
        println!(
            "{:<6} | {:<22} | {:<14.2} | {:<14.1} | {:<12.2}x",
            dim,
            "INT8 Cosine (3x dots)",
            int8_cos_ns,
            int8_cos_mops,
            f32_ns / int8_cos_ns
        );
        println!(
            "{:<6} | {:<22} | {:<14.2} | {:<14.1} | {:<12.2}x",
            dim,
            "Binary POPCNT Hamming",
            bin_hamm_ns,
            bin_hamm_mops,
            f32_ns / bin_hamm_ns
        );
        println!(
            "{:<6} | {:<22} | {:<14.2} | {:<14.1} | {:<12.2}x",
            dim,
            "Binary Angular Cosine",
            bin_cos_ns,
            bin_cos_mops,
            f32_ns / bin_cos_ns
        );
        println!("{}", "-".repeat(80));
    }
}
