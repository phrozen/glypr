use anyhow::{bail, Result};
use pulp::Simd;

/// Truncate a normalized embedding to a Matryoshka (MRL) dimension (e.g. 512, 256, 128)
/// and re-normalize with L2 norm.
pub fn truncate_mrl(embedding: &[f32], target_dim: usize) -> Result<Vec<f32>> {
    if target_dim == 0 {
        bail!("Target Matryoshka dimension cannot be 0");
    }
    if target_dim > embedding.len() {
        bail!(
            "Target dimension {} exceeds embedding dimension {}",
            target_dim,
            embedding.len()
        );
    }
    let slice = &embedding[..target_dim];
    Ok(l2_normalize(slice))
}

/// L2 normalize a slice of floats.
pub fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        vec.to_vec()
    } else {
        vec.iter().map(|x| x / norm).collect()
    }
}

/// Compute cosine similarity between two float vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

// =========================================================================
// Binary Quantization (1-bit / POPCNT Hamming)
// =========================================================================

/// Quantize a float vector to a 1-bit binary vector packed into `u64` words.
/// Bit `i` is set to 1 if `embedding[i] > 0.0`, else 0.
///
/// Strictly requires `embedding.len()` to be a multiple of 64 (e.g. 768, 512, 256, 128, 64),
/// eliminating partial padding-bit edge cases.
pub fn quantize_binary(embedding: &[f32]) -> Result<Vec<u64>> {
    let dim = embedding.len();
    if !dim.is_multiple_of(64) {
        bail!(
            "Binary quantization requires dimension to be a multiple of 64, got {}",
            dim
        );
    }
    let num_words = dim / 64;
    let mut words = vec![0u64; num_words];

    for (i, &val) in embedding.iter().enumerate() {
        if val > 0.0 {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }

    Ok(words)
}

/// Compute exact Hamming distance between two packed `u64` binary vectors using hardware POPCNT.
#[inline]
pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Dimension mismatch in hamming_distance: {} words vs {} words",
        a.len(),
        b.len()
    );

    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    let mut s0 = 0u32;
    let mut s1 = 0u32;
    let mut s2 = 0u32;
    let mut s3 = 0u32;

    for (ca, cb) in chunks_a.zip(chunks_b) {
        s0 += (ca[0] ^ cb[0]).count_ones();
        s1 += (ca[1] ^ cb[1]).count_ones();
        s2 += (ca[2] ^ cb[2]).count_ones();
        s3 += (ca[3] ^ cb[3]).count_ones();
    }

    let mut rem_sum = 0u32;
    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        rem_sum += (x ^ y).count_ones();
    }

    s0 + s1 + s2 + s3 + rem_sum
}

/// Compute angular cosine similarity approximation between two packed `u64` binary vectors
/// using Charikar's formula: `cos(pi * hamming_distance / dim)`.
#[inline]
pub fn binary_cosine_similarity(a: &[u64], b: &[u64]) -> f32 {
    let dim = a.len() * 64;
    if dim == 0 {
        return 1.0;
    }
    let h = hamming_distance(a, b) as f32;
    (std::f32::consts::PI * (h / dim as f32)).cos()
}

// =========================================================================
// Symmetric INT8 Quantization (Fixed [-1.0, 1.0] -> [-127, 127], Zero-Metadata)
// =========================================================================

/// Symmetric INT8 quantization mapping `[-1.0, 1.0] -> [-127, 127]`.
/// Payload is purely `Vec<i8>` with 0 metadata bytes (dim == size in bytes).
#[inline]
pub fn quantize_int8(embedding: &[f32]) -> Vec<i8> {
    embedding
        .iter()
        .map(|&v| (v * 127.0).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

#[pulp::with_simd(int8_dot_product_dispatch = ::pulp::Arch::new())]
pub fn int8_dot_product_simd<S: Simd>(_simd: S, a: &[i8], b: &[i8]) -> i32 {
    let mut sum0 = 0i32;
    let mut sum1 = 0i32;
    let mut sum2 = 0i32;
    let mut sum3 = 0i32;

    let chunks_a = a.chunks_exact(16);
    let chunks_b = b.chunks_exact(16);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        for i in 0..4 {
            sum0 += (ca[i] as i32) * (cb[i] as i32);
            sum1 += (ca[i + 4] as i32) * (cb[i + 4] as i32);
            sum2 += (ca[i + 8] as i32) * (cb[i + 8] as i32);
            sum3 += (ca[i + 12] as i32) * (cb[i + 12] as i32);
        }
    }

    let mut rem_sum = 0i32;
    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        rem_sum += (x as i32) * (y as i32);
    }

    sum0 + sum1 + sum2 + sum3 + rem_sum
}

/// Compute integer dot product between two `i8` slices using SIMD dynamic dispatch.
#[inline]
pub fn int8_dot_product(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Dimension mismatch in int8_dot_product: {} vs {}",
        a.len(),
        b.len()
    );
    int8_dot_product_dispatch(a, b)
}

/// Compute cosine similarity between two `i8` slices:
/// `sum(a_i * b_i) / (sqrt(sum(a_i^2)) * sqrt(sum(b_i^2)))`.
#[inline]
pub fn int8_cosine_similarity(a: &[i8], b: &[i8]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Dimension mismatch in int8_cosine_similarity: {} vs {}",
        a.len(),
        b.len()
    );
    let norm_sq_a = int8_dot_product(a, a);
    let norm_sq_b = int8_dot_product(b, b);
    if norm_sq_a == 0 || norm_sq_b == 0 {
        return 0.0;
    }
    let dot = int8_dot_product(a, b);
    let denom = (norm_sq_a as f32).sqrt() * (norm_sq_b as f32).sqrt();
    ((dot as f32) / denom).clamp(-1.0, 1.0)
}

// =========================================================================
// Memory Lifecycle & Objective-C Autorelease Pool Scoping
// =========================================================================

#[cfg(target_os = "macos")]
#[allow(clippy::duplicated_attributes)]
unsafe extern "C" {
    fn objc_autoreleasePoolPush() -> *mut std::ffi::c_void;
    fn objc_autoreleasePoolPop(pool: *mut std::ffi::c_void);
}

/// RAII scope guard to ensure Apple Metal command buffers and transient objects
/// are drained immediately after batch execution.
#[derive(Debug, Default)]
pub struct AutoreleaseScope {
    #[cfg(target_os = "macos")]
    pool: *mut std::ffi::c_void,
}

impl AutoreleaseScope {
    #[inline(always)]
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        unsafe {
            Self {
                pool: objc_autoreleasePoolPush(),
            }
        }
        #[cfg(not(target_os = "macos"))]
        Self {}
    }
}

impl Drop for AutoreleaseScope {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        unsafe {
            if !self.pool.is_null() {
                objc_autoreleasePoolPop(self.pool);
            }
        }
    }
}

/// Execute a closure within a scoped Objective-C autorelease pool on macOS (no-op on Linux/Windows).
#[inline(always)]
pub fn with_autorelease_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _scope = AutoreleaseScope::new();
    f()
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_mrl_and_norm() {
        let v = vec![3.0, 4.0];
        let norm = l2_normalize(&v);
        assert!((norm[0] - 0.6).abs() < 1e-6);
        assert!((norm[1] - 0.8).abs() < 1e-6);

        let v4 = vec![1.0, 1.0, 1.0, 1.0];
        let norm_v4 = l2_normalize(&v4);
        let trunc = truncate_mrl(&norm_v4, 2).unwrap();
        assert_eq!(trunc.len(), 2);
        assert!((cosine_similarity(&trunc, &trunc) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_binary_divisibility() {
        assert!(quantize_binary(&vec![0.1; 63]).is_err());
        assert!(quantize_binary(&vec![0.1; 65]).is_err());
        assert!(quantize_binary(&vec![0.1; 128]).is_ok());

        let v = vec![1.0; 64];
        let bin = quantize_binary(&v).unwrap();
        assert_eq!(bin.len(), 1);
        assert_eq!(bin[0], u64::MAX);

        let neg = vec![-1.0; 64];
        let bin_neg = quantize_binary(&neg).unwrap();
        assert_eq!(bin_neg[0], 0);

        assert_eq!(hamming_distance(&bin, &bin_neg), 64);
        assert!((binary_cosine_similarity(&bin, &bin_neg) - (-1.0)).abs() < 1e-6);
        assert!((binary_cosine_similarity(&bin, &bin) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_int8_and_simd_dot() {
        let a = vec![1.0, -1.0, 0.0, 0.5];
        let q = quantize_int8(&a);
        assert_eq!(q, vec![127, -127, 0, 64]);

        let b = vec![1.0, 1.0, 0.0, 0.5];
        let qb = quantize_int8(&b);

        let dot = int8_dot_product(&q, &qb);
        let expected = 127 * 127 + (-127) * 127 + 0 + 64 * 64;
        assert_eq!(dot, expected);

        assert!((int8_cosine_similarity(&q, &q) - 1.0).abs() < 1e-5);
    }
}
