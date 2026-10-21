use anyhow::Result;
use candle_core::quantized::gguf_file;
use std::fs::File;

fn main() -> Result<()> {
    let mmproj_path = glypr_core::hub::resolve_default_mmproj()?;
    let mut mmproj_file = File::open(&mmproj_path)?;
    let mmproj_content = gguf_file::Content::read(&mut mmproj_file)?;
    println!("\nMMProj Metadata Keys: {}", mmproj_content.metadata.len());
    for (k, v) in &mmproj_content.metadata {
        println!("  [MMPROJ] {:<35} : {:?}", k, v);
    }
    println!("\nMMProj Tensors: {}", mmproj_content.tensor_infos.len());
    for (k, info) in &mmproj_content.tensor_infos {
        println!(
            "  [MMPROJ TENSOR] {:<35} : shape={:?}, dtype={:?}",
            k, info.shape, info.ggml_dtype
        );
    }
    Ok(())
}
