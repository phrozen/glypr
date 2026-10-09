use anyhow::Result;
use candle_core::Device;
use glypr_core::{
    cosine_similarity, l2_normalize, truncate_mrl, validate_vision_budget, GlyprEmbedder,
    ModalityPart, Task, VISION_BUDGET_DEFAULT, VISION_BUDGET_HIGH, VISION_BUDGET_LOW,
    VISION_BUDGET_MAX, VISION_BUDGET_MEDIUM, VISION_BUDGET_MIN, VISION_BUDGET_ULTRA,
};
use std::path::PathBuf;

#[test]
fn test_task_formatting_and_parsing() {
    assert_eq!(
        Task::SearchQuery.format_text("what is rust"),
        "task: search result | query: what is rust"
    );
    assert_eq!(
        Task::Document { title: Some("Rust Guide") }.format_text("Rust is memory safe."),
        "title: Rust Guide | text: Rust is memory safe."
    );
    assert_eq!(
        Task::Document { title: None }.format_text("Rust is memory safe."),
        "title: none | text: Rust is memory safe."
    );
    assert_eq!(
        Task::Classification.format_text("happy mood"),
        "task: classification | query: happy mood"
    );
    assert_eq!(
        Task::CodeRetrieval.format_text("fn main() {}"),
        "task: code retrieval | query: fn main() {}"
    );
    assert_eq!(
        Task::Clustering.format_text("cluster topic"),
        "task: clustering | query: cluster topic"
    );
    assert_eq!(
        Task::QuestionAnswering.format_text("where is Paris?"),
        "task: question answering | query: where is Paris?"
    );
    assert_eq!(
        Task::FactChecking.format_text("water boils at 100C"),
        "task: fact checking | query: water boils at 100C"
    );
    assert_eq!(
        Task::SentenceSimilarity.format_text("two sentences"),
        "task: sentence similarity | query: two sentences"
    );
    assert_eq!(
        Task::Raw.format_text("no prefix text"),
        "no prefix text"
    );
    assert_eq!(
        Task::Custom("my custom task").format_text("custom input"),
        "task: my custom task | query: custom input"
    );

    // Parsing from name and title strings
    assert_eq!(Task::from_name_and_title("SearchQuery", None), Task::SearchQuery);
    assert_eq!(Task::from_name_and_title("query", None), Task::SearchQuery);
    assert_eq!(Task::from_name_and_title("retrieval", None), Task::SearchQuery);
    assert_eq!(
        Task::from_name_and_title("Document", Some("Doc1")),
        Task::Document { title: Some("Doc1") }
    );
    assert_eq!(
        Task::from_name_and_title("document", None),
        Task::Document { title: None }
    );
    assert_eq!(Task::from_name_and_title("code", None), Task::CodeRetrieval);
    assert_eq!(Task::from_name_and_title("classification", None), Task::Classification);
    assert_eq!(Task::from_name_and_title("raw", None), Task::Raw);
    assert_eq!(Task::from_name_and_title("none", None), Task::Raw);
    assert_eq!(
        Task::from_name_and_title("SpecialTask", None),
        Task::Custom("SpecialTask")
    );
}

#[test]
fn test_util_mrl_and_normalization() -> Result<()> {
    let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let norm = l2_normalize(&raw);
    let mag: f32 = norm.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((mag - 1.0).abs() < 1e-6);

    let zero = vec![0.0, 0.0, 0.0];
    let zero_norm = l2_normalize(&zero);
    assert_eq!(zero_norm, vec![0.0, 0.0, 0.0]);

    // Truncate MRL
    let mrl_3 = truncate_mrl(&norm, 3)?;
    assert_eq!(mrl_3.len(), 3);
    let mrl_mag: f32 = mrl_3.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((mrl_mag - 1.0).abs() < 1e-6);

    // MRL out of bounds error
    assert!(truncate_mrl(&norm, 10).is_err());
    assert!(truncate_mrl(&norm, 0).is_err());

    // Cosine similarity
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![1.0, 0.0, 0.0];
    let v3 = vec![0.0, 1.0, 0.0];
    let v4 = vec![-1.0, 0.0, 0.0];

    assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-6);
    assert!(cosine_similarity(&v1, &v3).abs() < 1e-6);
    assert!((cosine_similarity(&v1, &v4) - (-1.0)).abs() < 1e-6);

    Ok(())
}

#[test]
fn test_vision_budget_constants_and_validation() {
    assert_eq!(VISION_BUDGET_LOW, 70);
    assert_eq!(VISION_BUDGET_MEDIUM, 140);
    assert_eq!(VISION_BUDGET_DEFAULT, 280);
    assert_eq!(VISION_BUDGET_HIGH, 560);
    assert_eq!(VISION_BUDGET_ULTRA, 1120);
    assert_eq!(VISION_BUDGET_MIN, 16);
    assert_eq!(VISION_BUDGET_MAX, 2048);

    assert!(validate_vision_budget(70).is_ok());
    assert!(validate_vision_budget(140).is_ok());
    assert!(validate_vision_budget(280).is_ok());
    assert!(validate_vision_budget(560).is_ok());
    assert!(validate_vision_budget(1120).is_ok());
    assert!(validate_vision_budget(16).is_ok());
    assert!(validate_vision_budget(2048).is_ok());

    assert!(validate_vision_budget(15).is_err());
    assert!(validate_vision_budget(0).is_err());
    assert!(validate_vision_budget(2049).is_err());
    assert!(validate_vision_budget(5000).is_err());
}

#[test]
fn test_unified_api_model_inference() -> Result<()> {
    let gguf_path = glypr_core::hub::resolve_default_gguf()?;
    if !gguf_path.exists() {
        eprintln!("Skipping inference test: GGUF model not found at {:?}", gguf_path);
        return Ok(());
    }

    let device = if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    let mut embedder = GlyprEmbedder::load_gguf_bf16(&gguf_path, &device)?;

    let mmproj_path = glypr_core::hub::resolve_default_mmproj()?;
    let has_mmproj = mmproj_path.exists();
    if has_mmproj {
        embedder.attach_multimodal_projector(&mmproj_path)?;
    }

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
    let golden_dir = repo_root.join("tests/golden_vectors");

    // 1. Text embedding with Task enum
    let q_emb = embedder.embed_text("what is gravitational lensing?", Task::SearchQuery)?;
    assert_eq!(q_emb.len(), 768);
    let q_norm: f32 = q_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((q_norm - 1.0).abs() < 1e-3, "q_norm was {}", q_norm);

    let doc_emb = embedder.embed_text(
        "Gravitational lensing occurs when a massive celestial body warps spacetime.",
        Task::Document { title: Some("Astrophysics 101") },
    )?;
    assert_eq!(doc_emb.len(), 768);

    // High semantic similarity between query and relevant document
    let sim_text = cosine_similarity(&q_emb, &doc_emb);
    println!("Text SearchQuery vs Document similarity: {:.4}", sim_text);
    assert!(sim_text > 0.5, "Expected high semantic similarity, got {}", sim_text);

    // 2. MRL Truncation from native 768d vector
    let q_emb_256 = truncate_mrl(&q_emb, 256)?;
    assert_eq!(q_emb_256.len(), 256);
    let mrl_norm: f32 = q_emb_256.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((mrl_norm - 1.0).abs() < 1e-3, "mrl_norm was {}", mrl_norm);

    // 3. Batch text embedding
    let batch_texts = ["query alpha", "query beta", "query gamma"];
    let batch_embs = embedder.embed_texts(&batch_texts, Task::SearchQuery)?;
    assert_eq!(batch_embs.len(), 3);
    for emb in &batch_embs {
        assert_eq!(emb.len(), 768);
    }

    // 4. Interleaved text-only sequence
    let interleaved_text = embedder.embed_interleaved(
        &[ModalityPart::Text("what is gravitational lensing?")],
        Task::SearchQuery,
    )?;
    assert_eq!(interleaved_text.len(), 768);
    let sim_interleaved_text = cosine_similarity(&q_emb, &interleaved_text);
    assert!(
        sim_interleaved_text > 0.999,
        "Interleaved text-only should match embed_text (sim={})",
        sim_interleaved_text
    );

    // 5. Multimodal tests if mmproj available
    if has_mmproj {
        let pepper_path = golden_dir.join("pepper.png");
        if pepper_path.exists() {
            let img = image::open(&pepper_path)?;

            // embed_image with default budget
            let img_emb = embedder.embed_image(&img, Some("A bell pepper"))?;
            assert_eq!(img_emb.len(), 768);

            // embed_image_with_budget with low budget
            let img_low_emb = embedder.embed_image_with_budget(&img, None, VISION_BUDGET_LOW)?;
            assert_eq!(img_low_emb.len(), 768);

            // embed_image_with_budget out-of-range budget fails fast
            assert!(embedder.embed_image_with_budget(&img, None, 10).is_err());
            assert!(embedder.embed_image_with_budget(&img, None, 5000).is_err());

            // Interleaved text + image
            let interleaved_multimodal = embedder.embed_interleaved(
                &[
                    ModalityPart::Text("A photograph of "),
                    ModalityPart::Image { image: &img, budget: Some(VISION_BUDGET_DEFAULT) },
                ],
                Task::SearchQuery,
            )?;
            assert_eq!(interleaved_multimodal.len(), 768);

            let sim_img_multimodal = cosine_similarity(&img_emb, &interleaved_multimodal);
            println!("Image prompt vs interleaved similarity: {:.4}", sim_img_multimodal);
            assert!(sim_img_multimodal > 0.7);
        }
    }

    Ok(())
}
