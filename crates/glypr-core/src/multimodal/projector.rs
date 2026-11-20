use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use image::DynamicImage;
use std::path::Path;

use crate::config::MultimodalConfig;
use crate::multimodal::audio::AudioTower;
use crate::multimodal::audio_preprocessor::{ProcessedAudio, preprocess_audio};
use crate::multimodal::image_preprocessor::{ProcessedImage, preprocess_image};
use crate::multimodal::vision::VisionTower;

/// Unified Multimodal Projector managing Vision and Audio towers loaded from GGUF
pub struct MultimodalProjector {
    pub vision_tower: Option<VisionTower>,
    pub audio_tower: Option<AudioTower>,
    pub config: MultimodalConfig,
    pub device: Device,
    pub dtype: DType,
}

impl MultimodalProjector {
    /// Load multimodal projector directly from a GGUF file (`mmproj-*.gguf`)
    pub fn from_gguf_file<P: AsRef<Path>>(path: P, device: &Device, dtype: DType) -> Result<Self> {
        let p = path.as_ref();
        if !p.exists() {
            bail!("Multimodal projector file not found at {:?}", p);
        }

        let mut file = std::fs::File::open(p)
            .with_context(|| format!("Failed to open mmproj GGUF file {:?}", p))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to read mmproj GGUF container from {:?}", p))?;

        let config = MultimodalConfig::from_gguf_metadata(&content.metadata);

        let vision_tower = if let Some(v_cfg) = config.vision_config.clone() {
            let tower = VisionTower::from_gguf(&content, &mut file, v_cfg, device, dtype)
                .context("Failed to load Vision Tower from mmproj GGUF")?;
            Some(tower)
        } else {
            None
        };

        let audio_tower = if let Some(a_cfg) = config.audio_config.clone() {
            let tower = AudioTower::from_gguf(&content, &mut file, a_cfg, device, dtype)
                .context("Failed to load Audio Tower from mmproj GGUF")?;
            Some(tower)
        } else {
            None
        };

        if vision_tower.is_none() && audio_tower.is_none() {
            bail!(
                "GGUF file at {:?} contains neither vision nor audio projector weights",
                p
            );
        }

        Ok(Self {
            vision_tower,
            audio_tower,
            config,
            device: device.clone(),
            dtype,
        })
    }

    /// Encode an image into soft tokens [num_soft_tokens, 512] using default token budget (280)
    pub fn encode_image(&self, img: &DynamicImage) -> Result<(Tensor, usize)> {
        let budget = self
            .vision_tower
            .as_ref()
            .map(|t| t.config.default_output_length)
            .unwrap_or(280);
        self.encode_image_with_budget(img, budget)
    }

    /// Encode an image with a specific token budget (e.g. 70, 140, 280, 560, 1120) into soft tokens [num_soft_tokens, 512]
    pub fn encode_image_with_budget(
        &self,
        img: &DynamicImage,
        budget: usize,
    ) -> Result<(Tensor, usize)> {
        let tower = self.vision_tower.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Vision Tower is not available in this multimodal projector")
        })?;

        let processed: ProcessedImage = preprocess_image(
            img,
            budget,
            tower.config.patch_size,
            tower.config.pooling_kernel_size,
        )?;

        let num_soft_tokens = processed.num_soft_tokens;
        let soft_tokens = tower.encode(&processed)?;
        Ok((soft_tokens, num_soft_tokens))
    }

    /// Encode a batch of images with specified token budgets.
    /// If all images share identical patch layouts (e.g. batch of cards), they execute in a single
    /// batched ViT forward pass. Preprocessing is automatically parallelized with Rayon.
    pub fn encode_images_batch(
        &self,
        images_and_budgets: &[(&DynamicImage, usize)],
    ) -> Result<Vec<(Tensor, usize)>> {
        if images_and_budgets.is_empty() {
            return Ok(Vec::new());
        }
        let tower = self.vision_tower.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Vision Tower is not available in this multimodal projector")
        })?;

        // Parallel image preprocessing across CPU cores with Rayon
        use rayon::prelude::*;
        let patch_size = tower.config.patch_size;
        let pooling_kernel_size = tower.config.pooling_kernel_size;

        let processed: Vec<Result<ProcessedImage>> = images_and_budgets
            .into_par_iter()
            .map(|(img, b)| preprocess_image(img, *b, patch_size, pooling_kernel_size))
            .collect();

        let mut processed_images = Vec::with_capacity(processed.len());
        for res in processed {
            processed_images.push(res?);
        }

        // Shape-bucketed batching by (patch_width, patch_height), preserving original output order.
        use std::collections::HashMap;
        let mut buckets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (idx, p) in processed_images.iter().enumerate() {
            buckets
                .entry((p.patch_width, p.patch_height))
                .or_default()
                .push(idx);
        }

        let mut result: Vec<Option<(Tensor, usize)>> = vec![None; processed_images.len()];
        for (_dims, indices) in buckets {
            if indices.len() == 1 {
                let idx = indices[0];
                let p = &processed_images[idx];
                let soft = tower.encode(p)?;
                result[idx] = Some((soft, p.num_soft_tokens));
            } else {
                let group_refs: Vec<&ProcessedImage> =
                    indices.iter().map(|&idx| &processed_images[idx]).collect();
                let batched_tokens = tower.encode_batch_refs(&group_refs)?; // [g, num_soft_tokens, 512]
                for (pos, &orig_idx) in indices.iter().enumerate() {
                    let single = batched_tokens.get(pos)?; // [num_soft_tokens, 512]
                    result[orig_idx] = Some((single, processed_images[orig_idx].num_soft_tokens));
                }
            }
        }

        Ok(result.into_iter().map(|opt| opt.unwrap()).collect())
    }

    /// Encode raw 16 kHz audio samples into soft tokens [num_soft_tokens, 512]
    pub fn encode_audio(&self, samples: &[f32]) -> Result<(Tensor, usize)> {
        let tower = self.audio_tower.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Audio Tower is not available in this multimodal projector")
        })?;

        let processed: ProcessedAudio = preprocess_audio(samples)?;
        let soft_tokens = tower.encode(&processed)?;
        let num_soft_tokens = soft_tokens.dim(0)?;
        Ok((soft_tokens, num_soft_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{MultimodalConfig, VisionConfig};
    use crate::linear::LinearOp;
    use crate::multimodal::vision::{
        VisionPatchEmbedder, VisionPooler, VisionProjector, VisionRotaryEmbedding,
    };
    use crate::norm::RMSNorm;
    use candle_nn::Linear;
    use image::{DynamicImage, RgbImage};

    #[test]
    fn test_encode_images_batch_shape_bucketing_preserves_order() -> Result<()> {
        // Build a minimal CPU VisionTower with 0 encoder layers and distinct input_proj weights
        let proj_w = Tensor::ones((768, 768), DType::F32, &Device::Cpu)?;
        let input_proj = LinearOp::new_standard(Linear::new(proj_w, None));
        let pos_table = Tensor::zeros((2, 1024, 768), DType::F32, &Device::Cpu)?;
        let patch_embedder = VisionPatchEmbedder::new(input_proj, &pos_table, 768)?;

        let out_w = Tensor::ones((512, 768), DType::F32, &Device::Cpu)?;
        let tower = VisionTower {
            patch_embedder,
            rotary_emb: VisionRotaryEmbedding::new(64, 100.0),
            layers: Vec::new(),
            pooler: VisionPooler::new(3, 768),
            projector: VisionProjector {
                norm: RMSNorm::new(None, 1e-6),
                projection: LinearOp::new_standard(Linear::new(out_w, None)),
            },
            config: VisionConfig::default(),
            device: Device::Cpu,
            dtype: DType::F32,
        };

        let projector = MultimodalProjector {
            vision_tower: Some(tower),
            audio_tower: None,
            config: MultimodalConfig::from_gguf_metadata(&std::collections::HashMap::new()),
            device: Device::Cpu,
            dtype: DType::F32,
        };

        // Mix square and landscape images in interleaved order: [square, landscape, square]
        let img_sq1 =
            DynamicImage::ImageRgb8(RgbImage::from_pixel(128, 128, image::Rgb([10, 20, 30])));
        let img_wide =
            DynamicImage::ImageRgb8(RgbImage::from_pixel(256, 128, image::Rgb([40, 50, 60])));
        let img_sq2 =
            DynamicImage::ImageRgb8(RgbImage::from_pixel(128, 128, image::Rgb([70, 80, 90])));

        let batch_out =
            projector.encode_images_batch(&[(&img_sq1, 140), (&img_wide, 140), (&img_sq2, 140)])?;
        assert_eq!(batch_out.len(), 3);

        let solo_0 = projector.encode_image_with_budget(&img_sq1, 140)?;
        let solo_1 = projector.encode_image_with_budget(&img_wide, 140)?;
        let solo_2 = projector.encode_image_with_budget(&img_sq2, 140)?;

        for (got, solo) in batch_out.iter().zip([solo_0, solo_1, solo_2].iter()) {
            assert_eq!(got.1, solo.1);
            assert_eq!(got.0.dims(), solo.0.dims());
            let gv = got.0.flatten_all()?.to_vec1::<f32>()?;
            let sv = solo.0.flatten_all()?.to_vec1::<f32>()?;
            for (a, b) in gv.iter().zip(sv.iter()) {
                assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
            }
        }
        Ok(())
    }
}
