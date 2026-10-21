#[cfg(feature = "metal")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "metal")]
use candle_metal_kernels::metal::{ConstantValues, Value};
#[cfg(feature = "metal")]
use glypr_core::attention::AttentionBackend;
#[cfg(feature = "metal")]
use objc2_metal::MTLSize;

#[derive(Debug)]
#[repr(C)]
#[allow(dead_code)]
struct AttnParams {
    b: i32,
    h: i32,
    d: i32,
    ql: i32,
    kl: i32,
    gqa_factor: i32,
    scale: f32,
    softcapping: f32,
    nq: i32,
    nk: i32,
    nq_aligned: i32,
    nk_aligned: i32,
    ql_rem: i32,
    kl_rem: i32,
    ql_off: i32,
    window_size: i32,
    q_strides: [i64; 3],
    k_strides: [i64; 3],
    v_strides: [i64; 3],
    o_strides: [i64; 3],
}

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "metal")]
    {
        let device = Device::new_metal(0)?;
        let mdev = match &device {
            Device::Metal(m) => m,
            _ => unreachable!(),
        };

        let path = "crates/glypr-core/src/attention/steel_attention_sliding.metal";
        let src = std::fs::read_to_string(path)?;
        let lib = mdev.metal_device().new_library_with_source(&src, None)?;

        let seq_lens = [64, 512, 1024, 4096, 8192];
        for &seq_len in &seq_lens {
            let q =
                Tensor::randn(0f32, 1f32, (1, 4, seq_len, 256), &device)?.to_dtype(DType::BF16)?;
            let k =
                Tensor::randn(0f32, 1f32, (1, 2, seq_len, 256), &device)?.to_dtype(DType::BF16)?;
            let v =
                Tensor::randn(0f32, 1f32, (1, 2, seq_len, 256), &device)?.to_dtype(DType::BF16)?;

            let window_size = 512; // Test sliding window attention with GQA!

            // 1. StandardAttention Ground Truth: explicitly repeat KV heads for reference
            let k_rep = glypr_core::attention::repeat_kv(&k, 2)?;
            let v_rep = glypr_core::attention::repeat_kv(&v, 2)?;
            let ref_mask = {
                let mut mask_vec = vec![0.0f32; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let diff = i.abs_diff(j);
                        if diff > window_size {
                            mask_vec[i * seq_len + j] = -1e9f32;
                        }
                    }
                }
                Tensor::from_vec(mask_vec, (1, 1, seq_len, seq_len), &device)?
                    .to_dtype(DType::BF16)?
            };
            let ref_start = std::time::Instant::now();
            let ref_out = glypr_core::attention::StandardAttention.compute_attention(
                &q,
                &k_rep,
                &v_rep,
                Some(&ref_mask),
                None,
            )?;
            let _ = ref_out.to_device(&Device::Cpu)?;
            let ref_elapsed = ref_start.elapsed();

            // 2. Custom Metal Sliding FlashAttention
            let (bq, bk, wm, wn) = (32, 16, 4, 1);
            let align_q = (seq_len % bq) == 0;
            let align_k = (seq_len % bk) == 0;

            let cv = ConstantValues::new(vec![
                (200, Value::Bool(align_q)),
                (201, Value::Bool(align_k)),
                (300, Value::Bool(false)), // has_mask = false
                (301, Value::Bool(false)), // do_causal = false
            ]);
            let func_name = "steel_attention_bfloat16_bq32_bk16_bd256_wm4_wn1_maskbfloat16";
            let func = lib.get_function(func_name, Some(&cv))?;
            let pipeline = mdev
                .metal_device()
                .new_compute_pipeline_state_with_function(&func)?;

            let out_tensor = Tensor::zeros((1, 4, seq_len, 256), DType::BF16, &device)?;

            let (q_storage, q_layout) = q.storage_and_layout();
            let (k_storage, k_layout) = k.storage_and_layout();
            let (v_storage, v_layout) = v.storage_and_layout();
            let (out_storage, out_layout) = out_tensor.storage_and_layout();

            let q_buf = match &*q_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let k_buf = match &*k_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let v_buf = match &*v_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };
            let out_buf = match &*out_storage {
                candle_core::Storage::Metal(m) => m.buffer(),
                _ => unreachable!(),
            };

            let nq = seq_len.div_ceil(bq);
            let nk = seq_len.div_ceil(bk);
            let nq_aligned = seq_len / bq;
            let nk_aligned = seq_len / bk;

            let params = AttnParams {
                b: 1,
                h: 4,
                d: 256,
                ql: seq_len as i32,
                kl: seq_len as i32,
                gqa_factor: 2,
                scale: 1.0,
                softcapping: 1.0,
                nq: nq as i32,
                nk: nk as i32,
                nq_aligned: nq_aligned as i32,
                nk_aligned: nk_aligned as i32,
                ql_rem: (seq_len - nq_aligned * bq) as i32,
                kl_rem: (seq_len - nk_aligned * bk) as i32,
                ql_off: 0,
                window_size: window_size as i32,
                q_strides: [
                    q_layout.stride()[0] as i64,
                    q_layout.stride()[1] as i64,
                    q_layout.stride()[2] as i64,
                ],
                k_strides: [
                    k_layout.stride()[0] as i64,
                    k_layout.stride()[1] as i64,
                    k_layout.stride()[2] as i64,
                ],
                v_strides: [
                    v_layout.stride()[0] as i64,
                    v_layout.stride()[1] as i64,
                    v_layout.stride()[2] as i64,
                ],
                o_strides: [
                    out_layout.stride()[0] as i64,
                    out_layout.stride()[1] as i64,
                    out_layout.stride()[2] as i64,
                ],
            };

            let kernel_start = std::time::Instant::now();
            {
                let encoder = mdev.command_encoder()?;
                let enc_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
                enc_ref.set_compute_pipeline_state(&pipeline);

                enc_ref.set_input_buffer(0, Some(q_buf), 0);
                enc_ref.set_input_buffer(1, Some(k_buf), 0);
                enc_ref.set_input_buffer(2, Some(v_buf), 0);
                enc_ref.set_output_buffer(3, Some(out_buf), 0);
                enc_ref.set_bytes(4, &params);

                let grid_dims = MTLSize {
                    width: nq,
                    height: 4,
                    depth: 1,
                };
                let group_dims = MTLSize {
                    width: 32,
                    height: wm,
                    depth: wn,
                };
                enc_ref.dispatch_thread_groups(grid_dims, group_dims);
            }
            let _ = out_tensor.to_device(&Device::Cpu)?;
            let kernel_elapsed = kernel_start.elapsed();

            // Parity check
            let diff =
                (&ref_out.to_dtype(DType::F32)? - &out_tensor.to_dtype(DType::F32)?)?.abs()?;
            let max_diff: f32 = diff.max_all()?.to_scalar()?;
            let ref_vec: Vec<f32> = ref_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let out_vec: Vec<f32> = out_tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let dot: f32 = ref_vec.iter().zip(out_vec.iter()).map(|(a, b)| a * b).sum();
            let norm_ref: f32 = ref_vec.iter().map(|a| a * a).sum::<f32>().sqrt();
            let norm_out: f32 = out_vec.iter().map(|a| a * a).sum::<f32>().sqrt();
            let sim = dot / (norm_ref * norm_out);

            println!(
                "seq_len={:4}: sim={:.7}, max_diff={:.5e} | Ref took {:>8.2?} | Custom Flash took {:>8.2?} ({:.1}x)",
                seq_len,
                sim,
                max_diff,
                ref_elapsed,
                kernel_elapsed,
                ref_elapsed.as_secs_f64() / kernel_elapsed.as_secs_f64()
            );

            if seq_len == 8192 {
                println!("Row 0 ref[:5]: {:?}", &ref_vec[0..5]);
                println!("Row 0 out[:5]: {:?}", &out_vec[0..5]);
                // Let's also check row 4095
                let mid = 4095 * 256;
                println!("Row 4095 ref[:5]: {:?}", &ref_vec[mid..mid + 5]);
                println!("Row 4095 out[:5]: {:?}", &out_vec[mid..mid + 5]);
                // Let's also check row 8191
                let last = 8191 * 256;
                println!("Row 8191 ref[:5]: {:?}", &ref_vec[last..last + 5]);
                println!("Row 8191 out[:5]: {:?}", &out_vec[last..last + 5]);
            }
        }

        println!("\n--- Testing Fused Metal RMSNorm ---");
        let x_norm = Tensor::randn(0f32, 1f32, (1, 8192, 512), &device)?.to_dtype(DType::BF16)?;
        let weight = Tensor::randn(0f32, 1f32, (512,), &device)?.to_dtype(DType::BF16)?;

        let ref_norm = {
            let x_f32 = x_norm.to_dtype(DType::F32)?;
            let sq = x_f32.sqr()?;
            let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
            let rsqrt = (mean_sq + 1e-6)?.powf(-0.5)?;
            let normed = x_f32.broadcast_mul(&rsqrt)?;
            normed
                .broadcast_mul(&weight.to_dtype(DType::F32)?)?
                .to_dtype(DType::BF16)?
        };

        let t0 = std::time::Instant::now();
        let fused_norm =
            glypr_core::metal_kernels::fused_metal_rmsnorm(&x_norm, Some(&weight), 1e-6)?;
        let _ = fused_norm.to_device(&Device::Cpu)?;
        let fused_norm_dur = t0.elapsed();

        let diff = (&ref_norm.to_dtype(DType::F32)? - &fused_norm.to_dtype(DType::F32)?)?.abs()?;
        let max_diff: f32 = diff.max_all()?.to_scalar()?;
        let ref_vec: Vec<f32> = ref_norm.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let out_vec: Vec<f32> = fused_norm.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let dot: f32 = ref_vec.iter().zip(out_vec.iter()).map(|(a, b)| a * b).sum();
        let sim = dot
            / (ref_vec.iter().map(|a| a * a).sum::<f32>().sqrt()
                * out_vec.iter().map(|a| a * a).sum::<f32>().sqrt());
        println!(
            "RMSNorm 8Kx512: sim={:.7}, max_diff={:.5e}, latency={:?}",
            sim, max_diff, fused_norm_dur
        );

        println!("\n--- Testing Fused Metal RMSNorm with Non-Contiguous Slice ---");
        let x_4d = Tensor::randn(0f32, 1f32, (1, 64, 24, 512), &device)?.to_dtype(DType::BF16)?;
        let x_slice = x_4d.narrow(2, 5, 1)?.squeeze(2)?;
        println!("x_slice is_contiguous: {}", x_slice.is_contiguous());
        let ref_slice_norm = {
            let x_f32 = x_slice.to_dtype(DType::F32)?;
            let sq = x_f32.sqr()?;
            let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
            let rsqrt = (mean_sq + 1e-6)?.powf(-0.5)?;
            x_f32.broadcast_mul(&rsqrt)?.to_dtype(DType::BF16)?
        };
        let fused_slice_norm =
            glypr_core::metal_kernels::fused_metal_rmsnorm(&x_slice, None, 1e-6)?;
        let _diff = (&ref_slice_norm.to_dtype(DType::F32)?
            - &fused_slice_norm.to_dtype(DType::F32)?)?
            .abs()?;
        let ref_vec: Vec<f32> = ref_slice_norm
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let out_vec: Vec<f32> = fused_slice_norm
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let dot: f32 = ref_vec.iter().zip(out_vec.iter()).map(|(a, b)| a * b).sum();
        let sim = dot
            / (ref_vec.iter().map(|a| a * a).sum::<f32>().sqrt()
                * out_vec.iter().map(|a| a * a).sum::<f32>().sqrt());
        println!("Non-contiguous slice RMSNorm: sim={:.7}", sim);

        println!("\n--- Testing Fused Metal GeGLU ---");
        let gate = Tensor::randn(0f32, 1f32, (1, 8192, 2048), &device)?.to_dtype(DType::BF16)?;
        let up = Tensor::randn(0f32, 1f32, (1, 8192, 2048), &device)?.to_dtype(DType::BF16)?;

        let ref_geglu = {
            let gate_act = glypr_core::ple::gelu_tanh(&gate)?;
            gate_act.broadcast_mul(&up)?
        };

        let t0 = std::time::Instant::now();
        let fused_geglu = glypr_core::metal_kernels::fused_metal_geglu(&gate, &up)?;
        let _ = fused_geglu.to_device(&Device::Cpu)?;
        let fused_geglu_dur = t0.elapsed();

        let diff =
            (&ref_geglu.to_dtype(DType::F32)? - &fused_geglu.to_dtype(DType::F32)?)?.abs()?;
        let max_diff: f32 = diff.max_all()?.to_scalar()?;
        let ref_vec: Vec<f32> = ref_geglu.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let out_vec: Vec<f32> = fused_geglu.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let dot: f32 = ref_vec.iter().zip(out_vec.iter()).map(|(a, b)| a * b).sum();
        let sim = dot
            / (ref_vec.iter().map(|a| a * a).sum::<f32>().sqrt()
                * out_vec.iter().map(|a| a * a).sum::<f32>().sqrt());
        println!(
            "GeGLU 8Kx2048: sim={:.7}, max_diff={:.5e}, latency={:?}",
            sim, max_diff, fused_geglu_dur
        );
    }
    #[cfg(not(feature = "metal"))]
    {
        println!("This probe requires --features metal");
    }
    Ok(())
}
