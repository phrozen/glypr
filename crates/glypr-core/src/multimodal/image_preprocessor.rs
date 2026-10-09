use anyhow::{bail, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView};

/// Standard discrete vision token budget operating points:
pub const VISION_BUDGET_LOW: usize = 70;
pub const VISION_BUDGET_MEDIUM: usize = 140;
pub const VISION_BUDGET_DEFAULT: usize = 280;
pub const VISION_BUDGET_HIGH: usize = 560;
pub const VISION_BUDGET_ULTRA: usize = 1120;

pub const VISION_BUDGET_MIN: usize = 16;
pub const VISION_BUDGET_MAX: usize = 2048;

/// Validate that a vision token budget is within supported bounds [16..2048]
pub fn validate_vision_budget(budget: usize) -> Result<()> {
    if budget < VISION_BUDGET_MIN || budget > VISION_BUDGET_MAX {
        bail!(
            "Vision token budget {} is out of supported range [{}..{}]",
            budget,
            VISION_BUDGET_MIN,
            VISION_BUDGET_MAX
        );
    }
    Ok(())
}

/// Processed image ready for Vision Tower encoding
#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// Normalized patch pixel values: [num_patches, 768] (16 x 16 x 3, in [-1.0, 1.0])
    pub pixel_values: Vec<f32>,
    /// 2D patch coordinate positions: [num_patches, 2] (x, y)
    pub position_ids: Vec<u32>,
    /// Number of extracted patches
    pub num_patches: usize,
    /// Number of output soft tokens after 3x3 spatial pooling (num_patches / 9)
    pub num_soft_tokens: usize,
    /// Width in patches
    pub patch_width: usize,
    /// Height in patches
    pub patch_height: usize,
    /// Resized pixel width
    pub target_width: usize,
    /// Resized pixel height
    pub target_height: usize,
}

/// Calculate aspect-ratio-preserving dimensions that produce at most `max_patches` patches,
/// with both dimensions divisible by `side_mult = pooling_kernel_size * patch_size` (typically 48).
pub fn get_aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        bail!("Image dimensions must be non-zero");
    }

    let side_mult = pooling_kernel_size * patch_size;
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let total_px = (height * width) as f64;
    let factor = (target_px / total_px).sqrt();

    let ideal_height = factor * (height as f64);
    let ideal_width = factor * (width as f64);

    let mut target_height = ((ideal_height / (side_mult as f64)).floor() as usize) * side_mult;
    let mut target_width = ((ideal_width / (side_mult as f64)).floor() as usize) * side_mult;

    let max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;

    if target_height == 0 && target_width == 0 {
        bail!("Attempting to resize to 0x0 image");
    }

    if target_height == 0 {
        target_height = side_mult;
        let w_ratio = ((width as f64) / (height as f64)).floor() as usize;
        target_width = (w_ratio * side_mult).min(max_side_length);
    } else if target_width == 0 {
        target_width = side_mult;
        let h_ratio = ((height as f64) / (width as f64)).floor() as usize;
        target_height = (h_ratio * side_mult).min(max_side_length);
    }

    if (target_height * target_width) as f64 > target_px {
        bail!(
            "Resizing [{}x{}] to [{}x{}] exceeds {} patches with patch_size {}",
            height,
            width,
            target_height,
            target_width,
            max_patches,
            patch_size
        );
    }

    Ok((target_height, target_width))
}

/// Calculate the exact number of soft tokens an image will produce for a given token budget
pub fn compute_soft_token_count(
    orig_width: usize,
    orig_height: usize,
    max_soft_tokens: usize,
    patch_size: usize,
    pooling_kernel_size: usize,
) -> Result<usize> {
    validate_vision_budget(max_soft_tokens)?;
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;
    let (target_height, target_width) = get_aspect_ratio_preserving_size(
        orig_height,
        orig_width,
        patch_size,
        max_patches,
        pooling_kernel_size,
    )?;
    let patch_width = target_width / patch_size;
    let patch_height = target_height / patch_size;
    let num_patches = patch_width * patch_height;
    let k_squared = pooling_kernel_size * pooling_kernel_size;
    Ok(num_patches / k_squared)
}

/// Calculate the total number of tokens for an image including framing tokens (<|image> and <image|>)
pub fn compute_image_total_tokens(
    orig_width: usize,
    orig_height: usize,
    budget: usize,
) -> Result<usize> {
    let soft_tokens = compute_soft_token_count(orig_width, orig_height, budget, 16, 3)?;
    Ok(soft_tokens + 2)
}

/// Preprocess image matching Google Gemma 4 Vision Image Processor
pub fn preprocess_image(
    img: &DynamicImage,
    max_soft_tokens: usize,
    patch_size: usize,
    pooling_kernel_size: usize,
) -> Result<ProcessedImage> {
    validate_vision_budget(max_soft_tokens)?;
    let (orig_width, orig_height) = img.dimensions();
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;

    // 1. Calculate aspect-ratio-preserving dimensions
    let (target_height, target_width) = get_aspect_ratio_preserving_size(
        orig_height as usize,
        orig_width as usize,
        patch_size,
        max_patches,
        pooling_kernel_size,
    )?;

    // 2. Resize image using Catmull-Rom bicubic interpolation (matching PIL BICUBIC)
    let resized = img.resize_exact(
        target_width as u32,
        target_height as u32,
        FilterType::CatmullRom,
    );
    let rgb = resized.to_rgb8();

    let patch_width = target_width / patch_size;
    let patch_height = target_height / patch_size;
    let num_patches = patch_width * patch_height;
    let k_squared = pooling_kernel_size * pooling_kernel_size;
    let num_soft_tokens = num_patches / k_squared;

    // 3. Extract patches and normalize pixels: 2 * (px / 255.0 - 0.5)
    // Layout within each patch: (y * 16 + x) * 3 + c (row-major RGB)
    let mut pixel_values = Vec::with_capacity(num_patches * patch_size * patch_size * 3);
    let mut position_ids = Vec::with_capacity(num_patches * 2);

    for py in 0..patch_height {
        for px in 0..patch_width {
            position_ids.push(px as u32);
            position_ids.push(py as u32);

            let start_x = px * patch_size;
            let start_y = py * patch_size;

            for y in 0..patch_size {
                for x in 0..patch_size {
                    let pixel = rgb.get_pixel((start_x + x) as u32, (start_y + y) as u32);
                    // Rescale to [-1.0, 1.0]: Gemma4 scales pixels in patch embedder with 2 * (x - 0.5)
                    let r = 2.0 * ((pixel[0] as f32) / 255.0 - 0.5);
                    let g = 2.0 * ((pixel[1] as f32) / 255.0 - 0.5);
                    let b = 2.0 * ((pixel[2] as f32) / 255.0 - 0.5);
                    pixel_values.push(r);
                    pixel_values.push(g);
                    pixel_values.push(b);
                }
            }
        }
    }

    Ok(ProcessedImage {
        pixel_values,
        position_ids,
        num_patches,
        num_soft_tokens,
        patch_width,
        patch_height,
        target_width,
        target_height,
    })
}
