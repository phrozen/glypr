use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape};
use half::bf16;

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
            candle_core::bail!("cpu_geglu shape mismatch: {:?} vs {:?}", l1.shape(), l2.shape());
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
                const CHUNK_SIZE: usize = 1024;
                out.par_chunks_mut(CHUNK_SIZE)
                    .zip(gate_slice.par_chunks(CHUNK_SIZE))
                    .zip(up_slice.par_chunks(CHUNK_SIZE))
                    .for_each(|((out_chunk, gate_chunk), up_chunk)| {
                        for i in 0..out_chunk.len() {
                            out_chunk[i] = geglu_scalar_f32(gate_chunk[i], up_chunk[i]);
                        }
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
                const CHUNK_SIZE: usize = 1024;
                out.par_chunks_mut(CHUNK_SIZE)
                    .zip(gate_slice.par_chunks(CHUNK_SIZE))
                    .zip(up_slice.par_chunks(CHUNK_SIZE))
                    .for_each(|((out_chunk, gate_chunk), up_chunk)| {
                        for i in 0..out_chunk.len() {
                            let g = gate_chunk[i].to_f32();
                            let u = up_chunk[i].to_f32();
                            let val = geglu_scalar_f32(g, u);
                            out_chunk[i] = bf16::from_f32(val);
                        }
                    });

                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            _ => candle_core::bail!("cpu_geglu unsupported dtypes: {:?}, {:?}", s1.dtype(), s2.dtype()),
        }
    }
}
