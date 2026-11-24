use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, Layout, Result, Shape};
use half::bf16;

#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vvtanhf(y: *mut f32, x: *const f32, n: *const libc::c_int);
}

const GEGLU_CHUNK_SIZE: usize = 1024;

#[cfg(not(all(feature = "accelerate", target_os = "macos")))]
#[inline(always)]
fn geglu_scalar_f32(g: f32, u: f32) -> f32 {
    if g <= -10.0f32 {
        0.0f32
    } else if g >= 10.0f32 {
        g * u
    } else {
        let g3 = g * g * g;
        let inner = 0.797_884_6_f32 * (g + 0.044715f32 * g3);
        let tanh_val = inner.tanh();
        let act = 0.5f32 * g * (1.0f32 + tanh_val);
        act * u
    }
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn geglu_chunk_f32(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    let mut inner_buf = [0.0f32; GEGLU_CHUNK_SIZE];
    let mut tanh_buf = [0.0f32; GEGLU_CHUNK_SIZE];
    for ((g_chunk, u_chunk), o_chunk) in gate
        .chunks(GEGLU_CHUNK_SIZE)
        .zip(up.chunks(GEGLU_CHUNK_SIZE))
        .zip(out.chunks_mut(GEGLU_CHUNK_SIZE))
    {
        let n = g_chunk.len();
        for i in 0..n {
            let g = g_chunk[i].clamp(-10.0, 10.0);
            let g3 = g * g * g;
            inner_buf[i] = 0.797_884_6_f32 * (g + 0.044715f32 * g3);
        }
        let count = n as libc::c_int;
        // SAFETY: inner_buf and tanh_buf have >= n elements (n <= GEGLU_CHUNK_SIZE).
        unsafe {
            vvtanhf(tanh_buf.as_mut_ptr(), inner_buf.as_ptr(), &count);
        }
        for i in 0..n {
            let g = g_chunk[i];
            let u = u_chunk[i];
            o_chunk[i] = if g <= -10.0 {
                0.0
            } else if g >= 10.0 {
                g * u
            } else {
                0.5f32 * g * (1.0f32 + tanh_buf[i]) * u
            };
        }
    }
}

#[cfg(not(all(feature = "accelerate", target_os = "macos")))]
fn geglu_chunk_f32(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for i in 0..out.len() {
        out[i] = geglu_scalar_f32(gate[i], up[i]);
    }
}

fn geglu_chunk_bf16(
    gate: &[bf16],
    up: &[bf16],
    out: &mut [bf16],
    gate_f32: &mut Vec<f32>,
    up_f32: &mut Vec<f32>,
    out_f32: &mut Vec<f32>,
) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for ((g_chunk, u_chunk), o_chunk) in gate
        .chunks(GEGLU_CHUNK_SIZE)
        .zip(up.chunks(GEGLU_CHUNK_SIZE))
        .zip(out.chunks_mut(GEGLU_CHUNK_SIZE))
    {
        let n = g_chunk.len();
        gate_f32.resize(n, 0.0);
        up_f32.resize(n, 0.0);
        out_f32.resize(n, 0.0);
        for i in 0..n {
            gate_f32[i] = g_chunk[i].to_f32();
            up_f32[i] = u_chunk[i].to_f32();
        }
        geglu_chunk_f32(&gate_f32[..n], &up_f32[..n], &mut out_f32[..n]);
        for i in 0..n {
            o_chunk[i] = bf16::from_f32(out_f32[i]);
        }
    }
}

pub struct CpuGeGLUOp;

impl CustomOp2 for CpuGeGLUOp {
    fn name(&self) -> &'static str {
        "cpu_geglu_simd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if l1.shape() != l2.shape() {
            candle_core::bail!(
                "cpu_geglu shape mismatch: {:?} vs {:?}",
                l1.shape(),
                l2.shape()
            );
        }
        let total_elem = l1.shape().elem_count();

        match (s1, s2) {
            (CpuStorage::F32(gate_data), CpuStorage::F32(up_data)) => {
                let gate_slice = if l1.is_contiguous() {
                    let offset = l1.start_offset();
                    &gate_data[offset..offset + total_elem]
                } else {
                    candle_core::bail!("cpu_geglu requires contiguous gate tensor");
                };

                let up_slice = if l2.is_contiguous() {
                    let offset = l2.start_offset();
                    &up_data[offset..offset + total_elem]
                } else {
                    candle_core::bail!("cpu_geglu requires contiguous up tensor");
                };

                let mut out = vec![0.0f32; total_elem];

                use rayon::prelude::*;
                out.par_chunks_mut(GEGLU_CHUNK_SIZE)
                    .zip(gate_slice.par_chunks(GEGLU_CHUNK_SIZE))
                    .zip(up_slice.par_chunks(GEGLU_CHUNK_SIZE))
                    .for_each(|((out_chunk, gate_chunk), up_chunk)| {
                        geglu_chunk_f32(gate_chunk, up_chunk, out_chunk);
                    });

                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::BF16(gate_data), CpuStorage::BF16(up_data)) => {
                let gate_slice = if l1.is_contiguous() {
                    let offset = l1.start_offset();
                    &gate_data[offset..offset + total_elem]
                } else {
                    candle_core::bail!("cpu_geglu requires contiguous gate tensor");
                };

                let up_slice = if l2.is_contiguous() {
                    let offset = l2.start_offset();
                    &up_data[offset..offset + total_elem]
                } else {
                    candle_core::bail!("cpu_geglu requires contiguous up tensor");
                };

                let mut out = vec![bf16::from_f32(0.0); total_elem];

                use rayon::prelude::*;
                out.par_chunks_mut(GEGLU_CHUNK_SIZE)
                    .zip(gate_slice.par_chunks(GEGLU_CHUNK_SIZE))
                    .zip(up_slice.par_chunks(GEGLU_CHUNK_SIZE))
                    .for_each_init(
                        || {
                            (
                                Vec::with_capacity(GEGLU_CHUNK_SIZE),
                                Vec::with_capacity(GEGLU_CHUNK_SIZE),
                                Vec::with_capacity(GEGLU_CHUNK_SIZE),
                            )
                        },
                        |(gate_f32, up_f32, out_f32), ((out_chunk, gate_chunk), up_chunk)| {
                            geglu_chunk_bf16(
                                gate_chunk, up_chunk, out_chunk, gate_f32, up_f32, out_f32,
                            );
                        },
                    );

                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            _ => candle_core::bail!(
                "cpu_geglu unsupported dtypes: {:?}, {:?}",
                s1.dtype(),
                s2.dtype()
            ),
        }
    }
}

/// Fused CPU GeGLU reading concatenated `[..., 2 * D]` rows directly in parallel,
/// avoiding intermediate `.narrow().contiguous()` tensor copies.
pub struct CpuGeGLUGateUpOp;

impl CustomOp1 for CpuGeGLUGateUpOp {
    fn name(&self) -> &'static str {
        "cpu_geglu_gate_up_simd"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let dims = layout.shape().dims();
        if dims.is_empty() {
            candle_core::bail!("cpu_geglu_gate_up requires at least 1 dimension");
        }
        let two_d = dims[dims.len() - 1];
        if two_d == 0 || !two_d.is_multiple_of(2) {
            candle_core::bail!(
                "cpu_geglu_gate_up requires non-zero even last dimension, got {}",
                two_d
            );
        }
        if !layout.is_contiguous() {
            candle_core::bail!("cpu_geglu_gate_up requires contiguous input layout");
        }

        let d = two_d / 2;
        let mut out_dims = dims.to_vec();
        out_dims[dims.len() - 1] = d;
        let out_shape = Shape::from_dims(&out_dims);

        let total_in = layout.shape().elem_count();
        let total_out = out_shape.elem_count();
        let offset = layout.start_offset();

        match storage {
            CpuStorage::F32(data) => {
                let slice = &data[offset..offset + total_in];
                let mut out = vec![0.0f32; total_out];

                use rayon::prelude::*;
                out.par_chunks_exact_mut(d)
                    .zip(slice.par_chunks_exact(two_d))
                    .for_each(|(out_row, in_row)| {
                        let (gate_row, up_row) = in_row.split_at(d);
                        geglu_chunk_f32(gate_row, up_row, out_row);
                    });

                Ok((CpuStorage::F32(out), out_shape))
            }
            CpuStorage::BF16(data) => {
                let slice = &data[offset..offset + total_in];
                let mut out = vec![bf16::from_f32(0.0); total_out];

                use rayon::prelude::*;
                out.par_chunks_exact_mut(d)
                    .zip(slice.par_chunks_exact(two_d))
                    .for_each_init(
                        || {
                            (
                                Vec::with_capacity(GEGLU_CHUNK_SIZE),
                                Vec::with_capacity(GEGLU_CHUNK_SIZE),
                                Vec::with_capacity(GEGLU_CHUNK_SIZE),
                            )
                        },
                        |(gate_f32, up_f32, out_f32), (out_row, in_row)| {
                            let (gate_row, up_row) = in_row.split_at(d);
                            geglu_chunk_bf16(gate_row, up_row, out_row, gate_f32, up_f32, out_f32);
                        },
                    );

                Ok((CpuStorage::BF16(out), out_shape))
            }
            _ => candle_core::bail!("cpu_geglu_gate_up unsupported dtype: {:?}", storage.dtype()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{cpu_geglu, cpu_geglu_gate_up};
    use candle_core::{D, DType, Device, Tensor};

    #[test]
    fn test_cpu_geglu_gate_up_f32_and_bf16_match_cpu_geglu() {
        let rows = 3usize;
        let cols = 4usize;
        let d = 19usize;
        let data: Vec<f32> = (0..(rows * cols * 2 * d))
            .map(|i| (((i as f32) * 0.17).sin()) * 4.0)
            .collect();

        let gate_up_f32 = Tensor::from_vec(data, (rows, cols, 2 * d), &Device::Cpu).unwrap();

        for (dtype, tol) in [(DType::F32, 1e-5f32), (DType::BF16, 1e-3f32)] {
            let gate_up = gate_up_f32.to_dtype(dtype).unwrap();
            let gate = gate_up
                .narrow(D::Minus1, 0, d)
                .unwrap()
                .contiguous()
                .unwrap();
            let up = gate_up
                .narrow(D::Minus1, d, d)
                .unwrap()
                .contiguous()
                .unwrap();

            let expected = cpu_geglu(&gate, &up)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            let got = cpu_geglu_gate_up(&gate_up)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            assert_eq!(expected.len(), got.len());
            let max_diff = expected
                .iter()
                .zip(got.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= tol,
                "dtype={dtype:?} max_diff={max_diff} exceeded tol={tol}"
            );
        }

        // Also check non-contiguous fallback works cleanly
        let non_contig = gate_up_f32.transpose(0, 1).unwrap();
        assert!(!non_contig.is_contiguous());
        let out = cpu_geglu_gate_up(&non_contig).unwrap();
        assert_eq!(out.dims(), &[cols, rows, d]);
    }
}
