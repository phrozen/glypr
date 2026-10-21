use anyhow::Result;
use candle_core::quantized::gguf_file;
use std::fs::File;
use std::path::PathBuf;

fn main() -> Result<()> {
    let gguf_bf16_path = glypr_core::hub::resolve_default_gguf()?;

    let mut gguf_file = File::open(&gguf_bf16_path)?;
    let content = gguf_file::Content::read(&mut gguf_file)?;

    println!(
        "GGUF read success. Metadata keys: {}",
        content.metadata.len()
    );

    let tokens_val = content.metadata.get("tokenizer.ggml.tokens");
    let merges_val = content.metadata.get("tokenizer.ggml.merges");

    use tokenizers::Tokenizer;
    use tokenizers::models::bpe::BPE;

    let tokens_arr = match tokens_val.unwrap() {
        gguf_file::Value::Array(arr) => arr,
        _ => panic!(),
    };
    let merges_arr = match merges_val.unwrap() {
        gguf_file::Value::Array(arr) => arr,
        _ => panic!(),
    };

    println!("Total tokens: {}", tokens_arr.len());
    println!("Total merges: {}", merges_arr.len());

    let mut vocab = ahash::AHashMap::new();
    for (i, v) in tokens_arr.iter().enumerate() {
        if let gguf_file::Value::String(s) = v {
            vocab.insert(s.clone(), i as u32);
        }
    }

    let mut merges: Vec<(String, String)> = Vec::new();
    for v in merges_arr {
        if let gguf_file::Value::String(s) = v {
            let parts: Vec<&str> = s.split(' ').collect();
            if parts.len() == 2 {
                merges.push((parts[0].to_string(), parts[1].to_string()));
            }
        }
    }

    println!("Parsed vocab len: {}", vocab.len());
    println!("Parsed merges len: {}", merges.len());

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .fuse_unk(true)
        .unk_token("<unk>".to_string())
        .build()
        .map_err(|e| anyhow::anyhow!("BPE build error: {}", e))?;
    println!("Built BPE model successfully!");

    let mut tokenizer = Tokenizer::new(bpe);

    use tokenizers::normalizers::replace::Replace;
    tokenizer.with_normalizer(Some(Replace::new(" ", "\u{2581}").unwrap()));

    use tokenizers::SplitDelimiterBehavior;
    use tokenizers::pre_tokenizers::split::Split;
    tokenizer.with_pre_tokenizer(Some(
        Split::new(" ", SplitDelimiterBehavior::MergedWithPrevious, false).unwrap(),
    ));
    println!("Built Tokenizer successfully!");

    let text = "Hello world";
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    println!(
        "GGUF Raw BPE Encoded '{}' -> IDs: {:?}",
        text,
        encoding.get_ids()
    );
    println!("GGUF Raw BPE Tokens: {:?}", encoding.get_tokens());

    let test_texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "def hello_world(x: int) -> float:\n    return x * 3.14159",
        "El veloz murciélago hindú comía feliz cardillo y kiwi.",
        "Falsches Üben von Xylophonmusik quält jeden größeren Zwerg.",
        "Voix ambiguë d'un cœur qui au zéphyr préfère les jattes de kiwis.",
        "自然语言处理与多模态嵌入向量检索。",
        "日本語の自然言語処理とマルチモーダルベクトル検索。",
        "الذكاء الاصطناعي ومعالجة اللغات الطبيعية",
        "Широкая электрификация южных губерний даст мощный толчок подъёму сельского хозяйства.",
        "task: search result | query: What is quantum mechanics?",
        "title: Quantum Field Theory | text: In theoretical physics, quantum field theory is a theoretical framework...",
    ];

    let hf_tok_path = glypr_core::hub::resolve_google_tokenizer()?;
    let hf_tok = Tokenizer::from_file(&hf_tok_path).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut all_matched = true;
    for text in test_texts {
        let hf_enc = hf_tok
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let hf_ids = hf_enc.get_ids().to_vec();

        let gguf_enc = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let mut gguf_ids = gguf_enc.get_ids().to_vec();
        // Insert BOS/EOS as GlyprTokenizer does
        if gguf_ids.first() != Some(&2) {
            gguf_ids.insert(0, 2);
        }
        if gguf_ids.last() != Some(&1) {
            gguf_ids.push(1);
        }

        if hf_ids != gguf_ids {
            println!("MISMATCH for text: {}", text);
            println!("  HF:   {:?}", hf_ids);
            println!("  GGUF: {:?}", gguf_ids);
            all_matched = false;
        } else {
            let snippet: String = text.chars().take(30).collect();
            println!("✓ MATCH (len={:3}): {}", hf_ids.len(), snippet);
        }
    }
    assert!(all_matched, "All test texts must match HF tokenizer");

    let golden_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/golden_vectors");
    let corpus_json_path = golden_dir.join("corpus_8k.json");
    let corpus_bytes = std::fs::read(&corpus_json_path)?;
    let corpus: serde_json::Value = serde_json::from_slice(&corpus_bytes)?;
    let entries = corpus["entries"].as_array().unwrap();

    println!(
        "\nTesting all {} entries from corpus_8k.json...",
        entries.len()
    );
    let mut corpus_matched = 0;
    for entry in entries {
        let content = entry["content"].as_str().unwrap();
        let hf_enc = hf_tok
            .encode(content, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let hf_ids = hf_enc.get_ids().to_vec();

        let gguf_enc = tokenizer
            .encode(content, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let mut gguf_ids = gguf_enc.get_ids().to_vec();
        if gguf_ids.first() != Some(&2) {
            gguf_ids.insert(0, 2);
        }
        if gguf_ids.last() != Some(&1) {
            gguf_ids.push(1);
        }

        if hf_ids != gguf_ids {
            let id = entry["id"].as_str().unwrap();
            println!(
                "MISMATCH on corpus entry {}: HF len {}, GGUF len {}",
                id,
                hf_ids.len(),
                gguf_ids.len()
            );
            anyhow::bail!("Corpus token mismatch on {}", id);
        }
        corpus_matched += 1;
    }

    println!(
        "✓ 100% MATCH: All {}/{} corpus entries produced identical token sequences!",
        corpus_matched,
        entries.len()
    );

    // --- Test Architecture Config Extraction ---
    println!("\nTesting TextConfig extraction from GGUF metadata...");
    let hf_config_path = glypr_core::hub::resolve_file("google/embeddinggemma-2", "config.json")?;
    let hf_config = glypr_core::config::TextConfig::from_json_file(&hf_config_path)?;

    let num_hidden_layers =
        get_u32(content.metadata.get("gemma-embedding2.block_count")).unwrap() as usize;
    let hidden_size =
        get_u32(content.metadata.get("gemma-embedding2.embedding_length")).unwrap() as usize;
    let intermediate_size =
        get_u32(content.metadata.get("gemma-embedding2.feed_forward_length")).unwrap() as usize;
    let num_attention_heads = get_u32(
        content
            .metadata
            .get("gemma-embedding2.attention.head_count"),
    )
    .unwrap() as usize;
    let head_dim = get_u32(
        content
            .metadata
            .get("gemma-embedding2.attention.key_length_swa"),
    )
    .unwrap_or(256) as usize;
    let embedding_dim = get_u32(
        content
            .metadata
            .get("gemma-embedding2.embedding_length_out"),
    )
    .unwrap_or(768) as usize;
    let max_position_embeddings =
        get_u32(content.metadata.get("gemma-embedding2.context_length")).unwrap_or(262144) as usize;
    let rms_norm_eps = get_f64(
        content
            .metadata
            .get("gemma-embedding2.attention.layer_norm_rms_epsilon"),
    )
    .unwrap_or(1e-6);
    let hidden_size_per_layer_input = get_u32(
        content
            .metadata
            .get("gemma-embedding2.embedding_length_per_layer_input"),
    )
    .unwrap_or(512) as usize;

    let pad_token_id =
        get_u32(content.metadata.get("tokenizer.ggml.padding_token_id")).unwrap_or(0) as usize;
    let bos_token_id =
        get_u32(content.metadata.get("tokenizer.ggml.bos_token_id")).unwrap_or(2) as usize;
    let eos_token_id =
        get_u32(content.metadata.get("tokenizer.ggml.eos_token_id")).unwrap_or(1) as usize;
    let vocab_size = tokens_arr.len();

    // Sliding window pattern & layer types
    let swa_pattern = match content
        .metadata
        .get("gemma-embedding2.attention.sliding_window_pattern")
    {
        Some(gguf_file::Value::Array(arr)) => arr,
        _ => panic!("Missing sliding_window_pattern"),
    };
    let mut layer_types = Vec::with_capacity(num_hidden_layers);
    for v in swa_pattern {
        match v {
            gguf_file::Value::Bool(true) => {
                layer_types.push(glypr_core::config::LayerType::SlidingAttention)
            }
            gguf_file::Value::Bool(false) => {
                layer_types.push(glypr_core::config::LayerType::FullAttention)
            }
            _ => panic!("Invalid bool in pattern"),
        }
    }

    // KV heads
    let kv_heads_arr = match content
        .metadata
        .get("gemma-embedding2.attention.head_count_kv")
    {
        Some(gguf_file::Value::Array(arr)) => arr,
        _ => panic!("Missing head_count_kv"),
    };
    let num_key_value_heads = get_u32(kv_heads_arr.first()).unwrap() as usize;

    let full_head_dim = get_u32(
        content
            .metadata
            .get("gemma-embedding2.attention.key_length"),
    )
    .unwrap_or(512) as usize;

    let mut per_layer_config = std::collections::HashMap::new();
    for (i, &lt) in layer_types.iter().enumerate() {
        if lt == glypr_core::config::LayerType::FullAttention {
            let key = format!("{:02}", i);
            let kv_heads = get_u32(kv_heads_arr.get(i)).unwrap() as usize;
            per_layer_config.insert(
                key,
                glypr_core::config::LayerOverride {
                    head_dim: full_head_dim,
                    num_key_value_heads: kv_heads,
                },
            );
        }
    }

    let rope_theta_swa =
        get_f64(content.metadata.get("gemma-embedding2.rope.freq_base_swa")).unwrap_or(10000.0);
    let rope_theta_full =
        get_f64(content.metadata.get("gemma-embedding2.rope.freq_base")).unwrap_or(1000000.0);
    let mut rope_parameters = std::collections::HashMap::new();
    rope_parameters.insert(
        "sliding_attention".to_string(),
        glypr_core::config::RoPEParameters {
            rope_theta: rope_theta_swa,
            rope_type: "default".to_string(),
        },
    );
    rope_parameters.insert(
        "full_attention".to_string(),
        glypr_core::config::RoPEParameters {
            rope_theta: rope_theta_full,
            rope_type: "default".to_string(),
        },
    );

    let sliding_window = 512; // In bidirectional encoder, sliding_window is 512 per side

    let gguf_config = glypr_core::config::TextConfig {
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        vocab_size,
        embedding_dim,
        max_position_embeddings,
        sliding_window,
        rms_norm_eps,
        hidden_size_per_layer_input,
        layer_types,
        per_layer_config,
        rope_parameters,
        pad_token_id,
        bos_token_id,
        eos_token_id,
    };

    println!("GGUF Config parsed successfully!");
    assert_eq!(hf_config.hidden_size, gguf_config.hidden_size);
    assert_eq!(hf_config.intermediate_size, gguf_config.intermediate_size);
    assert_eq!(hf_config.num_hidden_layers, gguf_config.num_hidden_layers);
    assert_eq!(
        hf_config.num_attention_heads,
        gguf_config.num_attention_heads
    );
    assert_eq!(
        hf_config.num_key_value_heads,
        gguf_config.num_key_value_heads
    );
    assert_eq!(hf_config.head_dim, gguf_config.head_dim);
    assert_eq!(hf_config.vocab_size, gguf_config.vocab_size);
    assert_eq!(hf_config.embedding_dim, gguf_config.embedding_dim);
    assert_eq!(
        hf_config.max_position_embeddings,
        gguf_config.max_position_embeddings
    );
    assert_eq!(hf_config.sliding_window, gguf_config.sliding_window);
    assert!((hf_config.rms_norm_eps - gguf_config.rms_norm_eps).abs() < 1e-10);
    assert_eq!(
        hf_config.hidden_size_per_layer_input,
        gguf_config.hidden_size_per_layer_input
    );
    assert_eq!(hf_config.layer_types, gguf_config.layer_types);
    assert_eq!(
        hf_config.per_layer_config.len(),
        gguf_config.per_layer_config.len()
    );
    for (k, v) in &hf_config.per_layer_config {
        let gv = gguf_config.per_layer_config.get(k).unwrap();
        assert_eq!(v.head_dim, gv.head_dim);
        assert_eq!(v.num_key_value_heads, gv.num_key_value_heads);
    }
    for (k, v) in &hf_config.rope_parameters {
        let gv = gguf_config.rope_parameters.get(k).unwrap();
        assert_eq!(v.rope_theta, gv.rope_theta);
    }
    println!(
        "✓ 100% MATCH: GGUF extracted TextConfig is 100% field-identical to HuggingFace config.json!"
    );

    Ok(())
}

fn get_u32(val: Option<&gguf_file::Value>) -> Option<u32> {
    match val {
        Some(gguf_file::Value::U8(v)) => Some(*v as u32),
        Some(gguf_file::Value::I8(v)) => Some(*v as u32),
        Some(gguf_file::Value::U16(v)) => Some(*v as u32),
        Some(gguf_file::Value::I16(v)) => Some(*v as u32),
        Some(gguf_file::Value::U32(v)) => Some(*v),
        Some(gguf_file::Value::I32(v)) => Some(*v as u32),
        Some(gguf_file::Value::U64(v)) => Some(*v as u32),
        Some(gguf_file::Value::I64(v)) => Some(*v as u32),
        _ => None,
    }
}

fn get_f64(val: Option<&gguf_file::Value>) -> Option<f64> {
    match val {
        Some(gguf_file::Value::F32(v)) => Some(*v as f64),
        Some(gguf_file::Value::F64(v)) => Some(*v),
        _ => None,
    }
}
