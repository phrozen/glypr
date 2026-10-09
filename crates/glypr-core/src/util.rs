use anyhow::{bail, Result};

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

/// Compute cosine similarity between two float vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let norm = l2_normalize(&v);
        assert!((norm[0] - 0.6).abs() < 1e-6);
        assert!((norm[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_truncate_mrl() {
        let v = vec![1.0, 1.0, 1.0, 1.0];
        let norm_v = l2_normalize(&v); // [0.5, 0.5, 0.5, 0.5]
        let trunc = truncate_mrl(&norm_v, 2).unwrap();
        assert_eq!(trunc.len(), 2);
        let len: f32 = trunc.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-6);

        // Error cases
        assert!(truncate_mrl(&norm_v, 0).is_err());
        assert!(truncate_mrl(&norm_v, 10).is_err());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }
}
