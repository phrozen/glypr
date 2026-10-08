use candle_core::{Device, Tensor};
use glypr_core::cpu_simd::{cpu_geglu, cpu_rmsnorm};
use std::time::Instant;

#[test]
fn bench_cpu_simd_vs_baseline() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let seq_len = 512;
    let d = 2304; // Gemma-2 hidden dim
    let x = Tensor::randn(0.0f32, 1.0f32, (1, seq_len, d), &dev)?;
    let weight = Tensor::randn(1.0f32, 0.2f32, (d,), &dev)?;
    let eps = 1e-6;

    let iters = 100;

    // Warmup
    let _ = cpu_rmsnorm(&x, Some(&weight), eps)?;

    // 1. Baseline Candle RMSNorm
    let t0 = Instant::now();
    for _ in 0..iters {
        let sq = x.sqr()?;
        let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
        let rsqrt = (mean_sq + eps)?.powf(-0.5)?;
        let _ = x.broadcast_mul(&rsqrt)?.broadcast_mul(&weight)?;
    }
    let baseline_rmsnorm_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // 2. Pulp SIMD RMSNorm
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = cpu_rmsnorm(&x, Some(&weight), eps)?;
    }
    let simd_rmsnorm_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    println!("\n=== CPU RMSNorm Benchmark (512x2304) ===");
    println!("  Candle Baseline : {:.3} ms (5 intermediate tensor allocs per call)", baseline_rmsnorm_ms);
    println!("  Pulp SIMD       : {:.3} ms (0 intermediate tensor allocs)", simd_rmsnorm_ms);
    println!("  Speedup         : {:.2}x", baseline_rmsnorm_ms / simd_rmsnorm_ms);

    // 3. GeGLU Benchmark (512x9216)
    let ffn_dim = 9216;
    let gate = Tensor::randn(0.0f32, 1.0f32, (1, seq_len, ffn_dim), &dev)?;
    let up = Tensor::randn(0.0f32, 1.0f32, (1, seq_len, ffn_dim), &dev)?;

    let t0 = Instant::now();
    for _ in 0..iters {
        let sqrt_2_over_pi = 0.7978845608028654f64;
        let coeff = 0.044715f64;
        let x3 = gate.sqr()?.broadcast_mul(&gate)?;
        let scaled_x3 = (x3 * coeff)?;
        let sum = gate.add(&scaled_x3)?;
        let inner = (sum * sqrt_2_over_pi)?;
        let tanh = inner.tanh()?;
        let one_plus_tanh = (tanh + 1.0)?;
        let half_x = (&gate * 0.5)?;
        let gate_act = half_x.broadcast_mul(&one_plus_tanh)?;
        let _ = gate_act.broadcast_mul(&up)?;
    }
    let baseline_geglu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = cpu_geglu(&gate, &up)?;
    }
    let simd_geglu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    println!("\n=== CPU GeGLU Benchmark (512x9216) ===");
    println!("  Candle Baseline : {:.3} ms (10 intermediate tensor allocs per call)", baseline_geglu_ms);
    println!("  Pulp Streaming  : {:.3} ms (0 intermediate tensor allocs)", simd_geglu_ms);
    println!("  Speedup         : {:.2}x\n", baseline_geglu_ms / simd_geglu_ms);

    Ok(())
}
