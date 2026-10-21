use candle_core::{Device, Tensor};
use glypr_core::cpu_simd::{cpu_geglu, cpu_rmsnorm};

#[test]
fn test_cpu_rmsnorm_parity_with_candle() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let shape = (2, 8, 256);
    let x = Tensor::randn(0.0f32, 1.0f32, shape, &dev)?;
    let weight = Tensor::randn(1.0f32, 0.2f32, (256,), &dev)?;
    let eps = 1e-6;

    // 1. Candle baseline formula
    let sq = x.sqr()?;
    let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
    let rsqrt = (mean_sq + eps)?.powf(-0.5)?;
    let expected = x.broadcast_mul(&rsqrt)?.broadcast_mul(&weight)?;

    // 2. CPU SIMD
    let got = cpu_rmsnorm(&x, Some(&weight), eps)?;

    let exp_vec = expected.flatten_all()?.to_vec1::<f32>()?;
    let got_vec = got.flatten_all()?.to_vec1::<f32>()?;

    let mut max_diff = 0.0f32;
    for (e, g) in exp_vec.iter().zip(got_vec.iter()) {
        let diff = (e - g).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("RMSNorm max diff: {:e}", max_diff);
    assert!(
        max_diff < 1e-5,
        "Max diff {:e} exceeded tolerance 1e-5",
        max_diff
    );
    Ok(())
}

#[test]
fn test_cpu_geglu_parity_with_candle() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let shape = (1, 16, 512);
    let gate = Tensor::randn(0.0f32, 2.0f32, shape, &dev)?;
    let up = Tensor::randn(0.0f32, 1.0f32, shape, &dev)?;

    // 1. Candle baseline formula
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
    let expected = gate_act.broadcast_mul(&up)?;

    // 2. CPU SIMD GeGLU
    let got = cpu_geglu(&gate, &up)?;

    let exp_vec = expected.flatten_all()?.to_vec1::<f32>()?;
    let got_vec = got.flatten_all()?.to_vec1::<f32>()?;

    let mut max_diff = 0.0f32;
    for (e, g) in exp_vec.iter().zip(got_vec.iter()) {
        let diff = (e - g).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("GeGLU max diff: {:e}", max_diff);
    assert!(
        max_diff < 1e-5,
        "Max diff {:e} exceeded tolerance 1e-5",
        max_diff
    );
    Ok(())
}
