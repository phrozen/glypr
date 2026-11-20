use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct GlyprTokenizer {
    tokenizer: Tokenizer,
}

impl GlyprTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Tokenizer load error: {}", e))?;
        Ok(Self { tokenizer })
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
            Replace::new(" ", "\u{2581}")
                .map_err(|e| anyhow::anyhow!("Normalizer error: {}", e))?,
        ));

        // Gemma pre-tokenization: Split on ASCII space " " (NOT "\u{2581}") with MergedWithPrevious.
        // Because normalization already replaced " " with "\u{2581}" before pre-tokenization,
        // splitting on " " is a no-op that preserves "\u{2581}word" prefix tokens and matches
        // Google's tokenizer.json 100% across corpus_8k.json (changing to "\u{2581}" would turn
        // "\u{2581}" into a word suffix and break BPE).
        use tokenizers::SplitDelimiterBehavior;
        use tokenizers::pre_tokenizers::split::Split;
        tokenizer.with_pre_tokenizer(Some(
            Split::new(" ", SplitDelimiterBehavior::MergedWithPrevious, false)
                .map_err(|e| anyhow::anyhow!("Pre-tokenizer error: {}", e))?,
        ));

        Ok(Self { tokenizer })
    }

    pub fn format_query(&self, query: &str, task: &str) -> String {
        crate::task::Task::from_name_and_title(task, None).format_text(query)
    }

    pub fn format_document(&self, content: &str, title: Option<&str>) -> String {
        crate::task::Task::Document { title }.format_text(content)
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

        let input_ids =
            candle_core::Tensor::from_vec(input_ids_vec, (batch_size, max_len), device)?;
        let attention_mask =
            candle_core::Tensor::from_vec(attention_mask_vec, (batch_size, max_len), device)?;

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
    /// `<bos> <boi> <image>*N <eoi> [prompt] <eos>`
    pub fn encode_image_query(
        &self,
        prompt: Option<&str>,
        num_soft_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(num_soft_tokens + 16);
        tokens.push(TOKEN_BOS);
        tokens.push(TOKEN_BOI);
        tokens.resize(tokens.len() + num_soft_tokens, TOKEN_IMAGE);
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
    /// `<bos> <boa> <audio>*N <eoa> [prompt] <eos>`
    pub fn encode_audio_query(
        &self,
        prompt: Option<&str>,
        num_soft_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(num_soft_tokens + 16);
        tokens.push(TOKEN_BOS);
        tokens.push(TOKEN_BOA);
        tokens.resize(tokens.len() + num_soft_tokens, TOKEN_AUDIO);
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
    /// Returns `(token_ids, Vec<MediaSlot>)`.
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
                    if let Some(pos) = remaining.find(tag)
                        && earliest_pos.is_none_or(|p| pos < p)
                    {
                        earliest_pos = Some(pos);
                        found_tag = tag;
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

                    if media_idx >= media_items.len() {
                        anyhow::bail!(
                            "Found excess placeholder tag '{}' in prompt (only {} media item(s) provided)",
                            found_tag,
                            media_items.len()
                        );
                    }
                    let item = &media_items[media_idx];
                    let matches_tag = matches!(
                        (found_tag, item),
                        ("<|image|>", MediaPlaceholder::Image { .. })
                            | ("<|audio|>", MediaPlaceholder::Audio { .. })
                            | ("<|video|>", MediaPlaceholder::Video { .. })
                    );
                    if !matches_tag {
                        anyhow::bail!(
                            "Placeholder tag '{}' at media index {} does not match provided media modality ({})",
                            found_tag,
                            media_idx,
                            item.kind_name()
                        );
                    }
                    push_media_placeholder(&mut tokens, &mut slots, item);
                    media_idx += 1;

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

            if media_idx != media_items.len() {
                anyhow::bail!(
                    "Prompt contains {} placeholder tag(s), but {} media item(s) were provided",
                    media_idx,
                    media_items.len()
                );
            }
        } else {
            for item in media_items {
                push_media_placeholder(&mut tokens, &mut slots, item);
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
    /// Returns `(token_ids, Vec<MediaSlot>)`.
    pub fn encode_modality_sequence(
        &self,
        parts: &[crate::multimodal::ModalityPart],
        task: crate::task::Task,
    ) -> Result<(Vec<u32>, Vec<MediaSlot>)> {
        let mut tokens = Vec::new();
        let mut slots = Vec::new();

        tokens.push(TOKEN_BOS);

        let mut pending_prefix = task.prompt_prefix();

        for part in parts {
            match part {
                crate::multimodal::ModalityPart::Text(s) => {
                    let full_str = if let Some(pfx) = pending_prefix.take() {
                        format!("{}{}", pfx, s)
                    } else {
                        s.to_string()
                    };
                    if !full_str.is_empty() {
                        let enc = self
                            .tokenizer
                            .encode(full_str.as_str(), false)
                            .map_err(|e| anyhow::anyhow!("Encode text part error: {}", e))?;
                        tokens.extend(enc.get_ids());
                    }
                }
                crate::multimodal::ModalityPart::Image { image, budget } => {
                    if let Some(pfx) = pending_prefix.take() {
                        let enc = self
                            .tokenizer
                            .encode(pfx.as_str(), false)
                            .map_err(|e| anyhow::anyhow!("Encode task prefix error: {}", e))?;
                        tokens.extend(enc.get_ids());
                    }
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
                    tokens.resize(tokens.len() + n_tokens, TOKEN_IMAGE);
                    tokens.push(TOKEN_EOI);
                    slots.push(MediaSlot {
                        start_idx: slot_start,
                        length: n_tokens,
                    });
                }
                crate::multimodal::ModalityPart::Audio { samples } => {
                    if samples.is_empty() {
                        anyhow::bail!("Input audio cannot be empty");
                    }
                    if let Some(pfx) = pending_prefix.take() {
                        let enc = self
                            .tokenizer
                            .encode(pfx.as_str(), false)
                            .map_err(|e| anyhow::anyhow!("Encode task prefix error: {}", e))?;
                        tokens.extend(enc.get_ids());
                    }
                    let n_tokens = crate::multimodal::compute_audio_soft_token_count(samples.len());

                    tokens.push(TOKEN_BOA);
                    let slot_start = tokens.len();
                    tokens.resize(tokens.len() + n_tokens, TOKEN_AUDIO);
                    tokens.push(TOKEN_EOA);
                    slots.push(MediaSlot {
                        start_idx: slot_start,
                        length: n_tokens,
                    });
                }
                crate::multimodal::ModalityPart::VideoFrame { frame, budget } => {
                    if let Some(pfx) = pending_prefix.take() {
                        let enc = self
                            .tokenizer
                            .encode(pfx.as_str(), false)
                            .map_err(|e| anyhow::anyhow!("Encode task prefix error: {}", e))?;
                        tokens.extend(enc.get_ids());
                    }
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
                    tokens.resize(tokens.len() + n_tokens, TOKEN_VIDEO);
                    tokens.push(TOKEN_EOI);
                    slots.push(MediaSlot {
                        start_idx: slot_start,
                        length: n_tokens,
                    });
                }
            }
        }

        if let Some(pfx) = pending_prefix.take() {
            let enc = self
                .tokenizer
                .encode(pfx.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Encode task prefix error: {}", e))?;
            tokens.extend(enc.get_ids());
        }

        tokens.push(TOKEN_EOS);
        Ok((tokens, slots))
    }
}

#[derive(Debug, Clone)]
pub enum MediaPlaceholder {
    Image { num_soft_tokens: usize },
    Audio { num_soft_tokens: usize },
    Video { frame_soft_tokens: Vec<usize> },
}

impl MediaPlaceholder {
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Image { .. } => "image",
            Self::Audio { .. } => "audio",
            Self::Video { .. } => "video",
        }
    }
}

fn push_media_placeholder(
    tokens: &mut Vec<u32>,
    slots: &mut Vec<MediaSlot>,
    item: &MediaPlaceholder,
) {
    match item {
        MediaPlaceholder::Image { num_soft_tokens } => {
            tokens.push(TOKEN_BOI);
            let slot_start = tokens.len();
            tokens.resize(tokens.len() + *num_soft_tokens, TOKEN_IMAGE);
            tokens.push(TOKEN_EOI);
            slots.push(MediaSlot {
                start_idx: slot_start,
                length: *num_soft_tokens,
            });
        }
        MediaPlaceholder::Audio { num_soft_tokens } => {
            tokens.push(TOKEN_BOA);
            let slot_start = tokens.len();
            tokens.resize(tokens.len() + *num_soft_tokens, TOKEN_AUDIO);
            tokens.push(TOKEN_EOA);
            slots.push(MediaSlot {
                start_idx: slot_start,
                length: *num_soft_tokens,
            });
        }
        MediaPlaceholder::Video { frame_soft_tokens } => {
            for &n_tokens in frame_soft_tokens {
                tokens.push(TOKEN_BOI);
                let slot_start = tokens.len();
                tokens.resize(tokens.len() + n_tokens, TOKEN_VIDEO);
                tokens.push(TOKEN_EOI);
                slots.push(MediaSlot {
                    start_idx: slot_start,
                    length: n_tokens,
                });
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file::Value;

    fn build_synthetic_gguf_tokenizer() -> GlyprTokenizer {
        let vocab = vec![
            "<unk>",
            "</s>",
            "<s>",
            "\u{2581}",
            "H",
            "e",
            "l",
            "o",
            "w",
            "r",
            "d",
            "f",
            "b",
            "a",
            "He",
            "Hel",
            "Hell",
            "Hello",
            "wo",
            "wor",
            "worl",
            "world",
            "\u{2581}world",
            "fo",
            "foo",
            "ba",
            "bar",
            "\u{2581}bar",
            "Hello\u{2581}",
        ];
        let merges = vec![
            "H e",
            "He l",
            "Hel l",
            "Hell o",
            "w o",
            "wo r",
            "wor l",
            "worl d",
            "\u{2581} world",
            "f o",
            "fo o",
            "b a",
            "ba r",
            "\u{2581} bar",
        ];
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(
                vocab
                    .into_iter()
                    .map(|s| Value::String(s.to_string()))
                    .collect(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_string(),
            Value::Array(
                merges
                    .into_iter()
                    .map(|s| Value::String(s.to_string()))
                    .collect(),
            ),
        );
        GlyprTokenizer::from_gguf_metadata(&metadata).unwrap()
    }

    #[test]
    fn test_gguf_tokenizer_preserves_word_prefix_underscore() {
        let tok = build_synthetic_gguf_tokenizer();
        // BOS=2, "Hello"=17, "\u{2581}world"=22, EOS=1
        let ids = tok.encode("Hello world").unwrap();
        assert_eq!(ids, vec![2, 17, 22, 1]);

        // Leading space also yields prefix token "\u{2581}world"
        let leading = tok.encode(" world").unwrap();
        assert_eq!(leading, vec![2, 22, 1]);

        // Multiple spaces preserve individual "\u{2581}" (3) + "\u{2581}bar" (27)
        let multi = tok.encode("foo  bar").unwrap();
        assert_eq!(multi, vec![2, 24, 3, 27, 1]);
    }

    #[test]
    fn test_format_query_matches_task_aliases() {
        let tok = build_synthetic_gguf_tokenizer();
        let query = "neural retrieval";
        for alias in [
            "search_query",
            "retrieval",
            "retrieval_query",
            "query",
            "reranking",
            "bitextmining",
            "document",
            "retrieval_document",
            "code_retrieval",
            "code",
            "classification",
            "multilabel_classification",
            "clustering",
            "question_answering",
            "qa",
            "fact_checking",
            "sentence_similarity",
            "sts",
            "pair_classification",
            "summarization",
            "raw",
            "none",
        ] {
            let expected = crate::task::Task::from_name_and_title(alias, None).format_text(query);
            assert_eq!(
                tok.format_query(query, alias),
                expected,
                "mismatch for task alias '{}'",
                alias
            );
        }
        assert_eq!(tok.format_query(query, "raw"), query);
        assert_eq!(tok.format_query(query, "none"), query);
        assert_eq!(
            tok.format_query(query, "reranking"),
            "task: search result | query: neural retrieval"
        );
        assert_eq!(
            tok.format_query(query, "bitextmining"),
            "task: search result | query: neural retrieval"
        );
    }

    #[test]
    fn test_encode_interleaved_strict_placeholder_validation() {
        let tok = build_synthetic_gguf_tokenizer();

        // Valid multi-modal with variable per-frame video tokens
        let (tokens, slots) = tok
            .encode_interleaved(
                Some("Hello <|image|> world <|video|>"),
                &[
                    MediaPlaceholder::Image { num_soft_tokens: 3 },
                    MediaPlaceholder::Video {
                        frame_soft_tokens: vec![2, 4],
                    },
                ],
            )
            .unwrap();
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].length, 3);
        assert_eq!(slots[1].length, 2);
        assert_eq!(slots[2].length, 4);
        assert_eq!(
            &tokens[slots[0].start_idx..slots[0].start_idx + 3],
            &[TOKEN_IMAGE; 3]
        );
        assert_eq!(
            &tokens[slots[1].start_idx..slots[1].start_idx + 2],
            &[TOKEN_VIDEO; 2]
        );
        assert_eq!(
            &tokens[slots[2].start_idx..slots[2].start_idx + 4],
            &[TOKEN_VIDEO; 4]
        );

        // Excess placeholder tag must fail
        assert!(
            tok.encode_interleaved(
                Some("<|image|> <|image|>"),
                &[MediaPlaceholder::Image { num_soft_tokens: 2 }],
            )
            .is_err()
        );

        // Missing placeholder tag must fail
        assert!(
            tok.encode_interleaved(
                Some("<|image|>"),
                &[
                    MediaPlaceholder::Image { num_soft_tokens: 2 },
                    MediaPlaceholder::Audio { num_soft_tokens: 2 },
                ],
            )
            .is_err()
        );

        // Modality mismatch must fail
        assert!(
            tok.encode_interleaved(
                Some("<|audio|>"),
                &[MediaPlaceholder::Image { num_soft_tokens: 2 }],
            )
            .is_err()
        );
        assert!(
            tok.encode_interleaved(
                Some("<|image|>"),
                &[MediaPlaceholder::Video {
                    frame_soft_tokens: vec![2]
                }],
            )
            .is_err()
        );
    }

    #[test]
    fn test_encode_modality_sequence_rejects_empty_audio() {
        let tok = build_synthetic_gguf_tokenizer();
        let err = tok
            .encode_modality_sequence(
                &[crate::multimodal::ModalityPart::Audio { samples: &[] }],
                crate::task::Task::Raw,
            )
            .unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_from_gguf_metadata_missing_keys_error() {
        let mut meta = HashMap::new();
        assert!(GlyprTokenizer::from_gguf_metadata(&meta).is_err());

        meta.insert(
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(vec![Value::String("<unk>".to_string())]),
        );
        // Missing merges
        assert!(GlyprTokenizer::from_gguf_metadata(&meta).is_err());

        // Non-array merges
        meta.insert("tokenizer.ggml.merges".to_string(), Value::U32(42));
        assert!(GlyprTokenizer::from_gguf_metadata(&meta).is_err());

        // Non-array tokens
        meta.insert("tokenizer.ggml.tokens".to_string(), Value::Bool(true));
        meta.insert("tokenizer.ggml.merges".to_string(), Value::Array(vec![]));
        assert!(GlyprTokenizer::from_gguf_metadata(&meta).is_err());
    }

    #[test]
    fn test_encode_adds_bos_eos_idempotently() {
        let tok = build_synthetic_gguf_tokenizer();
        let plain = tok.encode("Hello").unwrap();
        assert_eq!(plain, vec![TOKEN_BOS, 17, TOKEN_EOS]);

        // Vocab contains "<s>" at index 2 and "</s>" at index 1; encoding them directly
        // must not duplicate BOS/EOS.
        let with_specials = tok.encode("<s>Hello</s>").unwrap();
        assert_eq!(with_specials.first(), Some(&TOKEN_BOS));
        assert_eq!(with_specials.last(), Some(&TOKEN_EOS));
        assert_eq!(
            with_specials.iter().filter(|&&id| id == TOKEN_BOS).count(),
            1
        );
        assert_eq!(
            with_specials.iter().filter(|&&id| id == TOKEN_EOS).count(),
            1
        );
    }

    #[test]
    fn test_encode_batch_padded_with_limit_masks_and_rejects_oversized() {
        let tok = build_synthetic_gguf_tokenizer();
        let dev = candle_core::Device::Cpu;

        // Empty batch must fail
        assert!(tok.encode_batch_padded_with_limit(&[], 16, &dev).is_err());

        // "Hello" -> [2, 17, 1] (len 3); "Hello world" -> [2, 17, 22, 1] (len 4)
        let (ids, mask) = tok
            .encode_batch_padded_with_limit(&["Hello", "Hello world"], 4, &dev)
            .unwrap();
        assert_eq!(ids.dims(), &[2, 4]);
        assert_eq!(mask.dims(), &[2, 4]);
        assert_eq!(
            ids.to_vec2::<u32>().unwrap(),
            vec![vec![2, 17, 1, 0], vec![2, 17, 22, 1]]
        );
        assert_eq!(
            mask.to_vec2::<u32>().unwrap(),
            vec![vec![1, 1, 1, 0], vec![1, 1, 1, 1]]
        );

        // Oversized item rejected cleanly
        let err = tok
            .encode_batch_padded_with_limit(&["Hello", "Hello world"], 3, &dev)
            .unwrap_err();
        assert!(err.to_string().contains("exceeds maximum context limit"));
    }

    #[test]
    fn test_encode_interleaved_manual_tags_vs_auto_prepend() {
        let tok = build_synthetic_gguf_tokenizer();
        let media = [
            MediaPlaceholder::Image { num_soft_tokens: 3 },
            MediaPlaceholder::Audio { num_soft_tokens: 2 },
        ];

        // Auto-prepend when no manual tags exist: <bos> <boi> <image>*3 <eoi> <boa> <audio>*2 <eoa> Hello <eos>
        let (auto_tokens, auto_slots) = tok.encode_interleaved(Some("Hello"), &media).unwrap();
        assert_eq!(auto_slots.len(), 2);
        assert_eq!(auto_slots[0].start_idx, 2);
        assert_eq!(auto_slots[0].length, 3);
        assert_eq!(auto_slots[1].start_idx, 7);
        assert_eq!(auto_slots[1].length, 2);
        assert_eq!(auto_tokens[0], TOKEN_BOS);
        assert_eq!(auto_tokens[1], TOKEN_BOI);
        assert_eq!(&auto_tokens[2..5], &[TOKEN_IMAGE; 3]);
        assert_eq!(auto_tokens[5], TOKEN_EOI);
        assert_eq!(auto_tokens[6], TOKEN_BOA);
        assert_eq!(&auto_tokens[7..9], &[TOKEN_AUDIO; 2]);
        assert_eq!(auto_tokens[9], TOKEN_EOA);
        assert_eq!(auto_tokens[10], 17); // "Hello"
        assert_eq!(auto_tokens[11], TOKEN_EOS);

        // Manual tags place slots at the exact tag positions
        let (manual_tokens, manual_slots) = tok
            .encode_interleaved(Some("Hello <|image|> foo <|audio|>"), &media)
            .unwrap();
        assert_eq!(manual_slots.len(), 2);
        assert_eq!(manual_slots[0].length, 3);
        assert_eq!(manual_slots[1].length, 2);
        assert!(manual_slots[0].start_idx > 2);
        assert_eq!(
            &manual_tokens[manual_slots[0].start_idx..manual_slots[0].start_idx + 3],
            &[TOKEN_IMAGE; 3]
        );
        assert_eq!(
            &manual_tokens[manual_slots[1].start_idx..manual_slots[1].start_idx + 2],
            &[TOKEN_AUDIO; 2]
        );
        assert_eq!(manual_tokens.first(), Some(&TOKEN_BOS));
        assert_eq!(manual_tokens.last(), Some(&TOKEN_EOS));
    }

    #[test]
    fn test_encode_modality_sequence_task_prefix_placement() {
        let tok = build_synthetic_gguf_tokenizer();
        let img = image::DynamicImage::new_rgb8(48, 48);
        let task = crate::task::Task::SearchQuery;
        let expected_prefix_ids = tok
            .tokenizer
            .encode(task.prompt_prefix().unwrap().as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();

        // Case 1: Leading Text part gets prefix prepended before first text, before media slot
        let (tokens_text_first, slots_text_first) = tok
            .encode_modality_sequence(
                &[
                    crate::multimodal::ModalityPart::Text("Hello"),
                    crate::multimodal::ModalityPart::Image {
                        image: &img,
                        budget: Some(16),
                    },
                ],
                task,
            )
            .unwrap();
        assert_eq!(slots_text_first.len(), 1);
        let boi_pos_1 = tokens_text_first
            .iter()
            .position(|&t| t == TOKEN_BOI)
            .unwrap();
        assert!(boi_pos_1 > 1 + expected_prefix_ids.len());
        assert_eq!(
            &tokens_text_first[1..1 + expected_prefix_ids.len()],
            expected_prefix_ids.as_slice()
        );

        // Case 2: Leading Media part gets prefix emitted before <boi>
        let (tokens_media_first, slots_media_first) = tok
            .encode_modality_sequence(
                &[
                    crate::multimodal::ModalityPart::Image {
                        image: &img,
                        budget: Some(16),
                    },
                    crate::multimodal::ModalityPart::Text("Hello"),
                ],
                task,
            )
            .unwrap();
        assert_eq!(slots_media_first.len(), 1);
        let boi_pos_2 = tokens_media_first
            .iter()
            .position(|&t| t == TOKEN_BOI)
            .unwrap();
        assert_eq!(boi_pos_2, 1 + expected_prefix_ids.len());
        assert_eq!(
            &tokens_media_first[1..boi_pos_2],
            expected_prefix_ids.as_slice()
        );
        assert_eq!(slots_media_first[0].start_idx, boi_pos_2 + 1);
    }
}
