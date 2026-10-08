use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct GlyprTokenizer {
    tokenizer: Tokenizer,
    prompts: HashMap<String, String>,
}

impl GlyprTokenizer {
    fn default_prompts() -> HashMap<String, String> {
        let mut prompts = HashMap::new();
        prompts.insert("searchquery".to_string(), "task: search result | query: ".to_string());
        prompts.insert("retrieval".to_string(), "task: search result | query: ".to_string());
        prompts.insert("coderetrieval".to_string(), "task: code retrieval | query: ".to_string());
        prompts.insert("questionanswering".to_string(), "task: question answering | query: ".to_string());
        prompts.insert("factchecking".to_string(), "task: fact checking | query: ".to_string());
        prompts.insert("sentencesimilarity".to_string(), "task: sentence similarity | query: ".to_string());
        prompts.insert("sts".to_string(), "task: sentence similarity | query: ".to_string());
        prompts.insert("summarization".to_string(), "task: sentence similarity | query: ".to_string());
        prompts.insert("classification".to_string(), "task: classification | query: ".to_string());
        prompts.insert("clustering".to_string(), "task: clustering | query: ".to_string());
        prompts.insert("reranking".to_string(), "task: search result | query: ".to_string());
        prompts.insert("bitextmining".to_string(), "task: search result | query: ".to_string());
        prompts.insert("document".to_string(), "title: none | text: ".to_string());
        prompts
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("Tokenizer load error: {}", e))?;
        Ok(Self {
            tokenizer,
            prompts: Self::default_prompts(),
        })
    }

    /// Construct GlyprTokenizer directly from GGUF metadata without external files
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
    ) -> Result<Self> {
        let tokens_val = metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow::anyhow!("Missing tokenizer.ggml.tokens in GGUF metadata"))?;
        let merges_val = metadata
            .get("tokenizer.ggml.merges")
            .ok_or_else(|| anyhow::anyhow!("Missing tokenizer.ggml.merges in GGUF metadata"))?;

        let tokens_arr = match tokens_val {
            candle_core::quantized::gguf_file::Value::Array(arr) => arr,
            _ => anyhow::bail!("tokenizer.ggml.tokens must be an Array"),
        };
        let merges_arr = match merges_val {
            candle_core::quantized::gguf_file::Value::Array(arr) => arr,
            _ => anyhow::bail!("tokenizer.ggml.merges must be an Array"),
        };

        let mut vocab = ahash::AHashMap::with_capacity(tokens_arr.len());
        for (i, v) in tokens_arr.iter().enumerate() {
            if let candle_core::quantized::gguf_file::Value::String(s) = v {
                vocab.insert(s.clone(), i as u32);
            }
        }

        let mut merges = Vec::with_capacity(merges_arr.len());
        for v in merges_arr {
            if let candle_core::quantized::gguf_file::Value::String(s) = v {
                let parts: Vec<&str> = s.split(' ').collect();
                if parts.len() == 2 {
                    merges.push((parts[0].to_string(), parts[1].to_string()));
                }
            }
        }

        let bpe = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, merges)
            .byte_fallback(true)
            .fuse_unk(true)
            .unk_token("<unk>".to_string())
            .build()
            .map_err(|e| anyhow::anyhow!("BPE build error: {}", e))?;

        let mut tokenizer = Tokenizer::new(bpe);

        // Gemma normalization: Replace " " with " " (\u{2581})
        use tokenizers::normalizers::replace::Replace;
        tokenizer.with_normalizer(Some(
            Replace::new(" ", "\u{2581}").map_err(|e| anyhow::anyhow!("Normalizer error: {}", e))?,
        ));

        // Gemma pre-tokenization: Split on " " (\u{2581}) merged with previous token
        use tokenizers::pre_tokenizers::split::Split;
        use tokenizers::SplitDelimiterBehavior;
        tokenizer.with_pre_tokenizer(Some(
            Split::new(" ", SplitDelimiterBehavior::MergedWithPrevious, false)
                .map_err(|e| anyhow::anyhow!("Pre-tokenizer error: {}", e))?,
        ));

        Ok(Self {
            tokenizer,
            prompts: Self::default_prompts(),
        })
    }

    pub fn format_query(&self, query: &str, task: &str) -> String {
        let key = task.to_lowercase().replace(['-', '_', ' '], "");
        let prefix = self.prompts.get(&key).cloned().unwrap_or_else(|| {
            format!("task: {} | query: ", task.to_lowercase())
        });
        format!("{}{}", prefix, query)
    }

    pub fn format_document(&self, content: &str, title: Option<&str>) -> String {
        match title {
            Some(t) => format!("title: {} | text: {}", t, content),
            None => format!("title: none | text: {}", content),
        }
    }

    /// Encode text with BOS (2) and EOS (1) tokens
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
        let mut ids = encoding.get_ids().to_vec();

        // Ensure BOS at start and EOS at end if not already present
        if ids.first() != Some(&2) {
            ids.insert(0, 2);
        }
        if ids.last() != Some(&1) {
            ids.push(1);
        }

        Ok(ids)
    }

    /// Encode a batch of texts into padded [batch_size, max_seq_len] tensors with attention mask
    pub fn encode_batch_padded(
        &self,
        texts: &[&str],
        device: &candle_core::Device,
    ) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
        let batch_size = texts.len();
        if batch_size == 0 {
            anyhow::bail!("Cannot encode an empty batch");
        }

        let mut all_ids = Vec::with_capacity(batch_size);
        let mut max_len = 0;
        for &text in texts {
            let ids = self.encode(text)?;
            if ids.len() > max_len {
                max_len = ids.len();
            }
            all_ids.push(ids);
        }

        let mut input_ids_vec = vec![0u32; batch_size * max_len];
        let mut attention_mask_vec = vec![0u32; batch_size * max_len];

        for (b, ids) in all_ids.iter().enumerate() {
            for (pos, &id) in ids.iter().enumerate() {
                let idx = b * max_len + pos;
                input_ids_vec[idx] = id;
                attention_mask_vec[idx] = 1;
            }
        }

        let input_ids = candle_core::Tensor::from_vec(input_ids_vec, (batch_size, max_len), device)?;
        let attention_mask = candle_core::Tensor::from_vec(attention_mask_vec, (batch_size, max_len), device)?;

        Ok((input_ids, attention_mask))
    }

    /// Construct token sequence for an image query:
    /// <bos> <boi> <image>*N <eoi> [prompt] <eos>
    pub fn encode_image_query(&self, prompt: Option<&str>, num_soft_tokens: usize) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(num_soft_tokens + 16);
        tokens.push(TOKEN_BOS);
        tokens.push(TOKEN_BOI);
        for _ in 0..num_soft_tokens {
            tokens.push(TOKEN_IMAGE);
        }
        tokens.push(TOKEN_EOI);

        if let Some(text) = prompt {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                let encoding = self
                    .tokenizer
                    .encode(trimmed, false)
                    .map_err(|e| anyhow::anyhow!("Encode prompt error: {}", e))?;
                for &id in encoding.get_ids() {
                    tokens.push(id);
                }
            }
        }

        tokens.push(TOKEN_EOS);
        Ok(tokens)
    }

    /// Construct token sequence for an audio query:
    /// <bos> <boa> <audio>*N <eoa> [prompt] <eos>
    pub fn encode_audio_query(&self, prompt: Option<&str>, num_soft_tokens: usize) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(num_soft_tokens + 16);
        tokens.push(TOKEN_BOS);
        tokens.push(TOKEN_BOA);
        for _ in 0..num_soft_tokens {
            tokens.push(TOKEN_AUDIO);
        }
        tokens.push(TOKEN_EOA);

        if let Some(text) = prompt {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                let encoding = self
                    .tokenizer
                    .encode(trimmed, false)
                    .map_err(|e| anyhow::anyhow!("Encode prompt error: {}", e))?;
                for &id in encoding.get_ids() {
                    tokens.push(id);
                }
            }
        }

        tokens.push(TOKEN_EOS);
        Ok(tokens)
    }
}

pub const TOKEN_PAD: u32 = 0;
pub const TOKEN_EOS: u32 = 1;
pub const TOKEN_BOS: u32 = 2;
pub const TOKEN_BOI: u32 = 255999;
pub const TOKEN_BOA: u32 = 256000;
pub const TOKEN_IMAGE: u32 = 258880;
pub const TOKEN_AUDIO: u32 = 258881;
pub const TOKEN_EOI: u32 = 258882;
pub const TOKEN_EOA: u32 = 258883;

