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

        // Check if all images have the same patch grid dimensions
        let is_homogeneous = processed_images.len() > 1
            && processed_images.windows(2).all(|w| {
                w[0].num_patches == w[1].num_patches
                    && w[0].patch_width == w[1].patch_width
                    && w[0].patch_height == w[1].patch_height
            });

        if is_homogeneous {
            let batched_tokens = tower.encode_batch(&processed_images)?; // [b, num_soft_tokens, 512]
            let b = processed_images.len();
            let mut result = Vec::with_capacity(b);
            for (i, p) in processed_images.iter().enumerate() {
                let single = batched_tokens.get(i)?; // [num_soft_tokens, 512]
                result.push((single, p.num_soft_tokens));
            }
            Ok(result)
        } else {
            // Heterogeneous images or single image: encode individually
            let mut result = Vec::with_capacity(processed_images.len());
            for p in &processed_images {
                let soft = tower.encode(p)?;
                result.push((soft, p.num_soft_tokens));
            }
            Ok(result)
        }
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
