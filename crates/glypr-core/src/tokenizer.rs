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

    /// Encode a batch of texts into padded [batch_size, max_seq_len] tensors with attention mask,
    /// enforcing an optional upper limit on maximum sequence length.
    pub fn encode_batch_padded_with_limit(
        &self,
        texts: &[&str],
        max_limit: usize,
        device: &candle_core::Device,
    ) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
        let batch_size = texts.len();
        if batch_size == 0 {
            anyhow::bail!("Cannot encode an empty batch");
        }

        let mut all_ids = Vec::with_capacity(batch_size);
        let mut max_len = 0;
        for (i, &text) in texts.iter().enumerate() {
            let ids = self.encode(text)?;
            if ids.len() > max_limit {
                anyhow::bail!(
                    "Batch item {} sequence length ({} tokens) exceeds maximum context limit ({} tokens)",
                    i,
                    ids.len(),
                    max_limit
                );
            }
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

    /// Encode a batch of texts into padded [batch_size, max_seq_len] tensors with attention mask
    pub fn encode_batch_padded(
        &self,
        texts: &[&str],
        device: &candle_core::Device,
    ) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
        self.encode_batch_padded_with_limit(texts, usize::MAX, device)
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

    /// Construct token sequence and slot map for an interleaved multimodal sequence:
    /// Supports manual placeholder tags `<|image|>`, `<|audio|>`, `<|video|>` in prompt,
    /// or prepends media placeholders if no manual tags are present.
    /// Returns (token_ids, Vec<MediaSlot>).
    pub fn encode_interleaved(
        &self,
        prompt: Option<&str>,
        media_items: &[MediaPlaceholder],
    ) -> Result<(Vec<u32>, Vec<MediaSlot>)> {
        let mut tokens = Vec::new();
        let mut slots = Vec::with_capacity(media_items.len());

        tokens.push(TOKEN_BOS);

        let prompt_text = prompt.unwrap_or("").trim();
        let has_placeholders = prompt_text.contains("<|image|>")
            || prompt_text.contains("<|audio|>")
            || prompt_text.contains("<|video|>");

        if has_placeholders {
            let mut media_idx = 0;
            let tags = ["<|image|>", "<|audio|>", "<|video|>"];

            let mut remaining = prompt_text;
            while !remaining.is_empty() {
                let mut earliest_pos = None;
                let mut found_tag = "";

                for tag in &tags {
                    if let Some(pos) = remaining.find(tag) {
                        if earliest_pos.map_or(true, |p| pos < p) {
                            earliest_pos = Some(pos);
                            found_tag = tag;
                        }
                    }
                }

                if let Some(pos) = earliest_pos {
                    let prefix = &remaining[..pos];
                    if !prefix.is_empty() {
                        let enc = self
                            .tokenizer
                            .encode(prefix, false)
                            .map_err(|e| anyhow::anyhow!("Encode prefix error: {}", e))?;
                        tokens.extend(enc.get_ids());
                    }

                    if media_idx < media_items.len() {
                        match &media_items[media_idx] {
                            MediaPlaceholder::Image { num_soft_tokens } => {
                                tokens.push(TOKEN_BOI);
                                let slot_start = tokens.len();
                                for _ in 0..*num_soft_tokens {
                                    tokens.push(TOKEN_IMAGE);
                                }
                                tokens.push(TOKEN_EOI);
                                slots.push(MediaSlot {
                                    start_idx: slot_start,
                                    length: *num_soft_tokens,
                                });
                            }
                            MediaPlaceholder::Audio { num_soft_tokens } => {
                                tokens.push(TOKEN_BOA);
                                let slot_start = tokens.len();
                                for _ in 0..*num_soft_tokens {
                                    tokens.push(TOKEN_AUDIO);
                                }
                                tokens.push(TOKEN_EOA);
                                slots.push(MediaSlot {
                                    start_idx: slot_start,
                                    length: *num_soft_tokens,
                                });
                            }
                            MediaPlaceholder::Video { num_frames, soft_tokens_per_frame } => {
                                for _ in 0..*num_frames {
                                    tokens.push(TOKEN_BOI);
                                    let slot_start = tokens.len();
                                    for _ in 0..*soft_tokens_per_frame {
                                        tokens.push(TOKEN_VIDEO);
                                    }
                                    tokens.push(TOKEN_EOI);
                                    slots.push(MediaSlot {
                                        start_idx: slot_start,
                                        length: *soft_tokens_per_frame,
                                    });
                                }
                            }
                        }
                        media_idx += 1;
                    }

                    remaining = &remaining[pos + found_tag.len()..];
                } else {
                    let enc = self
                        .tokenizer
                        .encode(remaining, false)
                        .map_err(|e| anyhow::anyhow!("Encode text error: {}", e))?;
                    tokens.extend(enc.get_ids());
                    break;
                }
            }
        } else {
            for item in media_items {
                match item {
                    MediaPlaceholder::Image { num_soft_tokens } => {
                        tokens.push(TOKEN_BOI);
                        let slot_start = tokens.len();
                        for _ in 0..*num_soft_tokens {
                            tokens.push(TOKEN_IMAGE);
                        }
                        tokens.push(TOKEN_EOI);
                        slots.push(MediaSlot {
                            start_idx: slot_start,
                            length: *num_soft_tokens,
                        });
                    }
                    MediaPlaceholder::Audio { num_soft_tokens } => {
                        tokens.push(TOKEN_BOA);
                        let slot_start = tokens.len();
                        for _ in 0..*num_soft_tokens {
                            tokens.push(TOKEN_AUDIO);
                        }
                        tokens.push(TOKEN_EOA);
                        slots.push(MediaSlot {
                            start_idx: slot_start,
                            length: *num_soft_tokens,
                        });
                    }
                    MediaPlaceholder::Video { num_frames, soft_tokens_per_frame } => {
                        for _ in 0..*num_frames {
                            tokens.push(TOKEN_BOI);
                            let slot_start = tokens.len();
                            for _ in 0..*soft_tokens_per_frame {
                                tokens.push(TOKEN_VIDEO);
                            }
                            tokens.push(TOKEN_EOI);
                            slots.push(MediaSlot {
                                start_idx: slot_start,
                                length: *soft_tokens_per_frame,
                            });
                        }
                    }
                }
            }

            if !prompt_text.is_empty() {
                let enc = self
                    .tokenizer
                    .encode(prompt_text, false)
                    .map_err(|e| anyhow::anyhow!("Encode prompt error: {}", e))?;
                tokens.extend(enc.get_ids());
            }
        }

        tokens.push(TOKEN_EOS);
        Ok((tokens, slots))
    }

    /// Encode an ordered sequence of ModalityPart items with an optional Task prompt.
    /// Returns (token_ids, Vec<MediaSlot>).
    pub fn encode_modality_sequence(
        &self,
        parts: &[crate::multimodal::ModalityPart],
        task: crate::task::Task,
    ) -> Result<(Vec<u32>, Vec<MediaSlot>)> {
        let mut tokens = Vec::new();
        let mut slots = Vec::new();

        tokens.push(TOKEN_BOS);

        // Prepend task prompt prefix if present
        if let Some(prefix) = task.prompt_prefix() {
            let enc = self
                .tokenizer
                .encode(prefix.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Encode task prefix error: {}", e))?;
            tokens.extend(enc.get_ids());
        }

        for part in parts {
            match part {
                crate::multimodal::ModalityPart::Text(s) => {
                    if !s.is_empty() {
                        let enc = self
                            .tokenizer
                            .encode(*s, false)
                            .map_err(|e| anyhow::anyhow!("Encode text part error: {}", e))?;
                        tokens.extend(enc.get_ids());
                    }
                }
                crate::multimodal::ModalityPart::Image { image, budget } => {
                    let b = budget.unwrap_or(crate::multimodal::VISION_BUDGET_DEFAULT);
                    let n_tokens = crate::multimodal::compute_soft_token_count(
                        image.width() as usize,
                        image.height() as usize,
                        b,
                        16,
                        3,
                    )?;
                    tokens.push(TOKEN_BOI);
                    let slot_start = tokens.len();
                    for _ in 0..n_tokens {
                        tokens.push(TOKEN_IMAGE);
                    }
                    tokens.push(TOKEN_EOI);
                    slots.push(MediaSlot {
                        start_idx: slot_start,
                        length: n_tokens,
                    });
                }
                crate::multimodal::ModalityPart::Audio { samples } => {
                    let n_tokens = crate::multimodal::compute_audio_soft_token_count(samples.len());
                    tokens.push(TOKEN_BOA);
                    let slot_start = tokens.len();
                    for _ in 0..n_tokens {
                        tokens.push(TOKEN_AUDIO);
                    }
                    tokens.push(TOKEN_EOA);
                    slots.push(MediaSlot {
                        start_idx: slot_start,
                        length: n_tokens,
                    });
                }
                crate::multimodal::ModalityPart::VideoFrame { frame, budget } => {
                    let b = budget.unwrap_or(crate::multimodal::VISION_BUDGET_MEDIUM);
                    let n_tokens = crate::multimodal::compute_soft_token_count(
                        frame.width() as usize,
                        frame.height() as usize,
                        b,
                        16,
                        3,
                    )?;
                    tokens.push(TOKEN_BOI);
                    let slot_start = tokens.len();
                    for _ in 0..n_tokens {
                        tokens.push(TOKEN_VIDEO);
                    }
                    tokens.push(TOKEN_EOI);
                    slots.push(MediaSlot {
                        start_idx: slot_start,
                        length: n_tokens,
                    });
                }
            }
        }

        tokens.push(TOKEN_EOS);
        Ok((tokens, slots))
    }
}

#[derive(Debug, Clone)]
pub enum MediaPlaceholder {
    Image { num_soft_tokens: usize },
    Audio { num_soft_tokens: usize },
    Video { num_frames: usize, soft_tokens_per_frame: usize },
}

#[derive(Debug, Clone)]
pub struct MediaSlot {
    pub start_idx: usize,
    pub length: usize,
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
pub const TOKEN_VIDEO: u32 = 258884;

