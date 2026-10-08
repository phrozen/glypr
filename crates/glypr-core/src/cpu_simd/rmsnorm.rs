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
        let d = dims[dims.len() - 1];
        let total_elem = shape.elem_count();
        let num_rows = total_elem / d;

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

                // Convert BF16 input row to F32 in stack buffer, compute SIMD, convert back to BF16
                let mut out = vec![bf16::from_f32(0.0); total_elem];
                let weight_ref = self.weight_f32.as_deref();
                let mut row_buf = vec![0.0f32; d];
                let mut out_buf = vec![0.0f32; d];

                for r in 0..num_rows {
                    let start = r * d;
                    let end = start + d;
                    let x_row_bf16 = &slice[start..end];
                    for i in 0..d {
                        row_buf[i] = x_row_bf16[i].to_f32();
                    }
                    simd_rmsnorm_row_f32_dispatch(&row_buf, weight_ref, &mut out_buf, self.eps);
                    let out_row_bf16 = &mut out[start..end];
                    for i in 0..d {
                        out_row_bf16[i] = bf16::from_f32(out_buf[i]);
                    }
                }

                Ok((CpuStorage::BF16(out), shape.clone()))
            }
            _ => candle_core::bail!("cpu_rmsnorm unsupported dtype: {:?}", storage.dtype()),
        }
    }
}
