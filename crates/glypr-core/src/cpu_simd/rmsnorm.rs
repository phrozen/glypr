use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape};
use half::bf16;
use pulp::Simd;

#[pulp::with_simd(simd_rmsnorm_row_f32_dispatch = ::pulp::Arch::new())]
pub fn simd_rmsnorm_row_f32<S: Simd>(
    simd: S,
    x: &[f32],
    weight: Option<&[f32]>,
    out: &mut [f32],
    eps: f32,
) {
    let d = x.len();
    let (x_head, x_tail) = S::as_simd_f32s(x);

    let mut sum_sq_vec = simd.splat_f32s(0.0);
    for &xv in x_head {
        sum_sq_vec = simd.mul_add_f32s(xv, xv, sum_sq_vec);
    }
    let mut sum_sq = simd.reduce_sum_f32s(sum_sq_vec);
    for &xv in x_tail {
        sum_sq += xv * xv;
    }

    let inv_rms = 1.0f32 / (sum_sq / (d as f32) + eps).sqrt();
    let inv_rms_vec = simd.splat_f32s(inv_rms);

    let (out_head, out_tail) = S::as_mut_simd_f32s(out);

    if let Some(w) = weight {
        let (w_head, w_tail) = S::as_simd_f32s(w);
        for ((ov, &xv), &wv) in out_head.iter_mut().zip(x_head.iter()).zip(w_head.iter()) {
            *ov = simd.mul_f32s(simd.mul_f32s(xv, inv_rms_vec), wv);
        }
        for ((ov, &xv), &wv) in out_tail.iter_mut().zip(x_tail.iter()).zip(w_tail.iter()) {
            *ov = xv * inv_rms * wv;
        }
    } else {
        for (ov, &xv) in out_head.iter_mut().zip(x_head.iter()) {
            *ov = simd.mul_f32s(xv, inv_rms_vec);
        }
        for (ov, &xv) in out_tail.iter_mut().zip(x_tail.iter()) {
            *ov = xv * inv_rms;
        }
    }
}

pub struct CpuRMSNormOp {
    pub weight_f32: Option<Vec<f32>>,
    pub eps: f32,
}

impl CustomOp1 for CpuRMSNormOp {
    fn name(&self) -> &'static str {
        "cpu_rmsnorm_simd"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let shape = layout.shape();
        let dims = shape.dims();
        if dims.is_empty() {
            candle_core::bail!("cpu_rmsnorm requires at least 1 dimension");
        }
        let d = dims[dims.len() - 1];
        if d == 0 {
            candle_core::bail!("cpu_rmsnorm requires non-zero last dimension");
        }
        if let Some(ref w) = self.weight_f32
            && w.len() != d
        {
            candle_core::bail!(
                "cpu_rmsnorm weight length {} does not match last dimension {}",
                w.len(),
                d
            );
        }
        let total_elem = shape.elem_count();

        match storage {
            CpuStorage::F32(data) => {
                let slice = if layout.is_contiguous() {
                    let offset = layout.start_offset();
                    &data[offset..offset + total_elem]
                } else {
                    candle_core::bail!("cpu_rmsnorm requires contiguous input layout");
                };

                let mut out = vec![0.0f32; total_elem];
                let weight_ref = self.weight_f32.as_deref();

                use rayon::prelude::*;
                out.par_chunks_exact_mut(d)
                    .zip(slice.par_chunks_exact(d))
                    .for_each(|(out_row, x_row)| {
                        simd_rmsnorm_row_f32_dispatch(x_row, weight_ref, out_row, self.eps);
                    });

                Ok((CpuStorage::F32(out), shape.clone()))
            }
            CpuStorage::BF16(data) => {
                let slice = if layout.is_contiguous() {
                    let offset = layout.start_offset();
                    &data[offset..offset + total_elem]
                } else {
                    candle_core::bail!("cpu_rmsnorm requires contiguous input layout");
                };

                // Parallelize BF16 rows across Rayon threads, reusing per-thread scratch F32 buffers
                let mut out = vec![bf16::from_f32(0.0); total_elem];
                let weight_ref = self.weight_f32.as_deref();

                use rayon::prelude::*;
                out.par_chunks_exact_mut(d)
                    .zip(slice.par_chunks_exact(d))
                    .for_each_init(
                        || (vec![0.0f32; d], vec![0.0f32; d]),
                        |(row_buf, out_buf), (out_row_bf16, x_row_bf16)| {
                            for (dst, src) in row_buf.iter_mut().zip(x_row_bf16.iter()) {
                                *dst = src.to_f32();
                            }
                            simd_rmsnorm_row_f32_dispatch(row_buf, weight_ref, out_buf, self.eps);
                            for (dst, src) in out_row_bf16.iter_mut().zip(out_buf.iter()) {
                                *dst = bf16::from_f32(*src);
                            }
                        },
                    );

                Ok((CpuStorage::BF16(out), shape.clone()))
            }
            _ => candle_core::bail!("cpu_rmsnorm unsupported dtype: {:?}", storage.dtype()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::cpu_rmsnorm;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_cpu_rmsnorm_rejects_invalid_shapes_and_weights() {
        // 0-D scalar tensor
        let scalar = Tensor::new(1.0f32, &Device::Cpu).unwrap();
        assert!(cpu_rmsnorm(&scalar, None, 1e-6).is_err());

        // 0-length last dimension
        let zero_d = Tensor::zeros((2, 0), DType::F32, &Device::Cpu).unwrap();
        assert!(cpu_rmsnorm(&zero_d, None, 1e-6).is_err());

        // Mismatched weight length
        let x = Tensor::ones((2, 4), DType::F32, &Device::Cpu).unwrap();
        let bad_w = Tensor::ones((3,), DType::F32, &Device::Cpu).unwrap();
        assert!(cpu_rmsnorm(&x, Some(&bad_w), 1e-6).is_err());

        // Valid case works
        let good_w = Tensor::ones((4,), DType::F32, &Device::Cpu).unwrap();
        let out = cpu_rmsnorm(&x, Some(&good_w), 1e-6).unwrap();
        assert_eq!(out.dims(), &[2, 4]);
    }

    #[test]
    fn test_cpu_rmsnorm_bf16_parity() {
        let rows = 5usize;
        let d = 73usize; // non-SIMD-aligned dimension to exercise head + tail
        let x_data: Vec<f32> = (0..(rows * d))
            .map(|i| ((i as f32) * 0.13 - 3.0).sin() * 2.5)
            .collect();
        let w_data: Vec<f32> = (0..d)
            .map(|i| 0.5 + ((i as f32) * 0.07).cos() * 0.4)
            .collect();

        let x_f32 = Tensor::from_vec(x_data, (rows, d), &Device::Cpu).unwrap();
        let w_f32 = Tensor::from_vec(w_data, (d,), &Device::Cpu).unwrap();
        let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap();
        let w_bf16 = w_f32.to_dtype(DType::BF16).unwrap();

        let ref_with_w = cpu_rmsnorm(
            &x_bf16.to_dtype(DType::F32).unwrap(),
            Some(&w_bf16.to_dtype(DType::F32).unwrap()),
            1e-6,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let got_with_w = cpu_rmsnorm(&x_bf16, Some(&w_bf16), 1e-6)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let max_diff_w = ref_with_w
            .iter()
            .zip(got_with_w.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff_w < 1e-3,
            "BF16 RMSNorm (weighted) parity mismatch: {max_diff_w}"
        );

        let ref_no_w = cpu_rmsnorm(&x_bf16.to_dtype(DType::F32).unwrap(), None, 1e-6)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got_no_w = cpu_rmsnorm(&x_bf16, None, 1e-6)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let max_diff_no_w = ref_no_w
            .iter()
            .zip(got_no_w.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff_no_w < 1e-3,
            "BF16 RMSNorm (unweighted) parity mismatch: {max_diff_no_w}"
        );
    }
}
