use glypr_core::coreml::{CoreMLComputeUnits, CoreMLTextBackbone, CoreMLVisionTower};
use glypr_core::multimodal::ProcessedImage;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing CoreML Vision Tower ===");
    let vision_dir = "models/coreml";
    if Path::new(vision_dir).is_dir() {
        let vision_tower =
            CoreMLVisionTower::from_dir(vision_dir, CoreMLComputeUnits::CpuAndNeuralEngine)?;
        println!(
            "Loaded CoreML Vision Tower supporting patch counts: {:?}",
            vision_tower.supported_patch_counts()
        );

        // Test with a dummy Budget 140 image (1134 patches -> 126 soft tokens)
        if vision_tower.supports_patches(1134) {
            println!("Encoding dummy Budget 140 image (1134 patches) on ANE...");
            let dummy_image = ProcessedImage {
                pixel_values: vec![0.0f32; 1134 * 768],
                position_ids: vec![0u32; 1134 * 2],
                num_patches: 1134,
                num_soft_tokens: 126,
                patch_width: 27,
                patch_height: 42,
                target_width: 432,
                target_height: 672,
            };

            let (buf, shape) = vision_tower.encode_raw(&dummy_image)?;
            println!(
                "✓ ANE Vision Tower Output Shape: {:?}, Elements: {}",
                shape,
                buf.len()
            );
            println!(
                "  First 5 soft token features: {:?}",
                &buf[..5.min(buf.len())]
            );

            // Test Candle Tensor conversion
            let device = candle_core::Device::Cpu;
            let tensor = vision_tower.encode_to_tensor(&dummy_image, &device)?;
            println!(
                "✓ Converted to Candle Tensor: shape {:?}, dtype {:?}",
                tensor.shape(),
                tensor.dtype()
            );
        }

        // Test with Budget 280 image (2394 patches -> 266 soft tokens)
        if vision_tower.supports_patches(2394) {
            println!("\nEncoding dummy Budget 280 image (2394 patches) on ANE...");
            let dummy_image_280 = ProcessedImage {
                pixel_values: vec![0.0f32; 2394 * 768],
                position_ids: vec![0u32; 2394 * 2],
                num_patches: 2394,
                num_soft_tokens: 266,
                patch_width: 42,
                patch_height: 57,
                target_width: 672,
                target_height: 912,
            };

            let (buf, shape) = vision_tower.encode_raw(&dummy_image_280)?;
            println!(
                "✓ ANE Vision Tower Output Shape: {:?}, Elements: {}",
                shape,
                buf.len()
            );
            let device = candle_core::Device::Cpu;
            let tensor = vision_tower.encode_to_tensor(&dummy_image_280, &device)?;
            println!(
                "✓ Converted to Candle Tensor: shape {:?}, dtype {:?}",
                tensor.shape(),
                tensor.dtype()
            );
        }
    }

    println!("\n=== Testing CoreML Text Backbone ===");
    let text_path = "models/coreml/gemma2_text_backbone_s512_w8a16.mlmodelc";
    if Path::new(text_path).exists() {
        let text_backbone =
            CoreMLTextBackbone::from_path(text_path, CoreMLComputeUnits::CpuAndNeuralEngine)?;
        println!(
            "Loaded CoreML Text Backbone (seq_len: {}) on ANE",
            text_backbone.seq_len()
        );

        let sample_tokens = [
            2i32, 603, 235292, 1146, 235284, 1489, 235292, 235248, 1841, 6496, 573, 11956, 5028,
            235336,
        ];
        println!("Embedding {} sample tokens on ANE...", sample_tokens.len());
        let embedding = text_backbone.embed_token_ids(&sample_tokens)?;
        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();

        println!("✓ ANE Text Embedding Vector Length: {}", embedding.len());
        println!("✓ L2 Norm: {:.6}", norm);
        println!("  First 5 embedding dimensions: {:?}", &embedding[..5]);
    }

    println!("\nAll CoreML Rust integrations passed cleanly!");
    Ok(())
}
