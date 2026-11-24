use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerOverride {
    pub head_dim: usize,
    pub num_key_value_heads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoPEParameters {
    pub rope_theta: f64,
    pub rope_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub rms_norm_eps: f64,
    pub hidden_size_per_layer_input: usize,
    pub layer_types: Vec<LayerType>,
    pub per_layer_config: HashMap<String, LayerOverride>,
    pub rope_parameters: HashMap<String, RoPEParameters>,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
}

impl TextConfig {
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let json_val: serde_json::Value = serde_json::from_str(&content)?;

        // If nested in text_config
        let config_obj = if let Some(tc) = json_val.get("text_config") {
            tc
        } else {
            &json_val
        };

        let config: TextConfig = serde_json::from_value(config_obj.clone())?;
        config.validate()?;
        Ok(config)
    }

    /// Validate structural invariants of `TextConfig`.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.num_hidden_layers == 0 {
            anyhow::bail!("TextConfig.num_hidden_layers must be > 0");
        }
        if self.layer_types.len() != self.num_hidden_layers {
            anyhow::bail!(
                "TextConfig.layer_types length ({}) does not match num_hidden_layers ({})",
                self.layer_types.len(),
                self.num_hidden_layers
            );
        }
        if self.num_attention_heads == 0 {
            anyhow::bail!("TextConfig.num_attention_heads must be > 0");
        }
        if self.num_key_value_heads == 0 {
            anyhow::bail!("TextConfig.num_key_value_heads must be > 0");
        }
        if self.head_dim == 0 {
            anyhow::bail!("TextConfig.head_dim must be > 0");
        }
        if self.embedding_dim == 0 {
            anyhow::bail!("TextConfig.embedding_dim must be > 0");
        }
        for (layer_key, override_cfg) in &self.per_layer_config {
            if override_cfg.head_dim == 0 {
                anyhow::bail!(
                    "TextConfig.per_layer_config[{}].head_dim must be > 0",
                    layer_key
                );
            }
            if override_cfg.num_key_value_heads == 0 {
                anyhow::bail!(
                    "TextConfig.per_layer_config[{}].num_key_value_heads must be > 0",
                    layer_key
                );
            }
        }
        Ok(())
    }

    /// Construct TextConfig directly from GGUF metadata
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
    ) -> anyhow::Result<Self> {
        let num_hidden_layers = get_u32(metadata.get("gemma-embedding2.block_count"))
            .ok_or_else(|| anyhow::anyhow!("Missing gemma-embedding2.block_count"))?
            as usize;
        let hidden_size = get_u32(metadata.get("gemma-embedding2.embedding_length"))
            .ok_or_else(|| anyhow::anyhow!("Missing gemma-embedding2.embedding_length"))?
            as usize;
        let intermediate_size = get_u32(metadata.get("gemma-embedding2.feed_forward_length"))
            .ok_or_else(|| anyhow::anyhow!("Missing gemma-embedding2.feed_forward_length"))?
            as usize;
        let num_attention_heads = get_u32(metadata.get("gemma-embedding2.attention.head_count"))
            .ok_or_else(|| anyhow::anyhow!("Missing gemma-embedding2.attention.head_count"))?
            as usize;
        let head_dim = get_u32(metadata.get("gemma-embedding2.attention.key_length_swa"))
            .unwrap_or(256) as usize;
        let embedding_dim =
            get_u32(metadata.get("gemma-embedding2.embedding_length_out")).unwrap_or(768) as usize;
        let max_position_embeddings =
            get_u32(metadata.get("gemma-embedding2.context_length")).unwrap_or(262144) as usize;
        let rms_norm_eps =
            get_f64(metadata.get("gemma-embedding2.attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-6);
        let hidden_size_per_layer_input =
            get_u32(metadata.get("gemma-embedding2.embedding_length_per_layer_input"))
                .unwrap_or(512) as usize;

        let pad_token_id =
            get_u32(metadata.get("tokenizer.ggml.padding_token_id")).unwrap_or(0) as usize;
        let bos_token_id =
            get_u32(metadata.get("tokenizer.ggml.bos_token_id")).unwrap_or(2) as usize;
        let eos_token_id =
            get_u32(metadata.get("tokenizer.ggml.eos_token_id")).unwrap_or(1) as usize;

        let vocab_size = match metadata.get("tokenizer.ggml.tokens") {
            Some(candle_core::quantized::gguf_file::Value::Array(arr)) => arr.len(),
            _ => 262144,
        };

        let mut layer_types = Vec::with_capacity(num_hidden_layers);
        if let Some(candle_core::quantized::gguf_file::Value::Array(swa_pattern)) =
            metadata.get("gemma-embedding2.attention.sliding_window_pattern")
        {
            for v in swa_pattern {
                match v {
                    candle_core::quantized::gguf_file::Value::Bool(true) => {
                        layer_types.push(LayerType::SlidingAttention)
                    }
                    candle_core::quantized::gguf_file::Value::Bool(false) => {
                        layer_types.push(LayerType::FullAttention)
                    }
                    _ => anyhow::bail!("Invalid boolean in sliding_window_pattern"),
                }
            }
        } else {
            // Default 5 sliding + 1 full repeating pattern
            for i in 0..num_hidden_layers {
                if (i + 1) % 6 == 0 {
                    layer_types.push(LayerType::FullAttention);
                } else {
                    layer_types.push(LayerType::SlidingAttention);
                }
            }
        }

        let kv_heads_arr = match metadata.get("gemma-embedding2.attention.head_count_kv") {
            Some(candle_core::quantized::gguf_file::Value::Array(arr)) => Some(arr),
            _ => None,
        };
        let num_key_value_heads = kv_heads_arr
            .and_then(|arr| get_u32(arr.first()))
            .unwrap_or(2) as usize;

        let full_head_dim =
            get_u32(metadata.get("gemma-embedding2.attention.key_length")).unwrap_or(512) as usize;

        let mut per_layer_config = HashMap::new();
        for (i, &lt) in layer_types.iter().enumerate() {
            if lt == LayerType::FullAttention {
                let key = format!("{:02}", i);
                let kv_heads = kv_heads_arr
                    .and_then(|arr| arr.get(i))
                    .and_then(|v| get_u32(Some(v)))
                    .unwrap_or(1) as usize;
                per_layer_config.insert(
                    key,
                    LayerOverride {
                        head_dim: full_head_dim,
                        num_key_value_heads: kv_heads,
                    },
                );
            }
        }

        let rope_theta_swa =
            get_f64(metadata.get("gemma-embedding2.rope.freq_base_swa")).unwrap_or(10000.0);
        let rope_theta_full =
            get_f64(metadata.get("gemma-embedding2.rope.freq_base")).unwrap_or(1000000.0);
        let mut rope_parameters = HashMap::new();
        rope_parameters.insert(
            "sliding_attention".to_string(),
            RoPEParameters {
                rope_theta: rope_theta_swa,
                rope_type: "default".to_string(),
            },
        );
        rope_parameters.insert(
            "full_attention".to_string(),
            RoPEParameters {
                rope_theta: rope_theta_full,
                rope_type: "default".to_string(),
            },
        );

        // llama.cpp exports gemma-embedding2.attention.sliding_window as the full
        // bidirectional span (2 * radius = 1024), whereas HF config.json and glypr's
        // attention masks / FlashAttention kernels use the one-sided half-window radius (512).
        let sliding_window = get_u32(metadata.get("gemma-embedding2.attention.sliding_window"))
            .filter(|&w| w > 0)
            .map(|w| (w as usize / 2).max(1))
            .unwrap_or(512);

        let config = Self {
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
        config.validate()?;
        Ok(config)
    }

    /// Get head dimension for a specific layer index
    pub fn get_head_dim(&self, layer_idx: usize) -> usize {
        let key = format!("{:02}", layer_idx);
        if let Some(override_cfg) = self.per_layer_config.get(&key) {
            override_cfg.head_dim
        } else {
            self.head_dim
        }
    }

    /// Get number of key/value heads for a specific layer index
    pub fn get_num_kv_heads(&self, layer_idx: usize) -> usize {
        let key = format!("{:02}", layer_idx);
        if let Some(override_cfg) = self.per_layer_config.get(&key) {
            override_cfg.num_key_value_heads
        } else {
            self.num_key_value_heads
        }
    }

    /// Get RoPE base theta for a specific layer index
    pub fn get_rope_theta(&self, layer_idx: usize) -> f64 {
        let layer_type = self.layer_types[layer_idx];
        let key = match layer_type {
            LayerType::SlidingAttention => "sliding_attention",
            LayerType::FullAttention => "full_attention",
        };
        if let Some(params) = self.rope_parameters.get(key) {
            params.rope_theta
        } else {
            match layer_type {
                LayerType::SlidingAttention => 10_000.0,
                LayerType::FullAttention => 1_000_000.0,
            }
        }
    }
}

pub(crate) fn get_u32(val: Option<&candle_core::quantized::gguf_file::Value>) -> Option<u32> {
    use candle_core::quantized::gguf_file::Value;
    match val? {
        Value::U8(v) => Some(*v as u32),
        Value::I8(v) => u32::try_from(*v).ok(),
        Value::U16(v) => Some(*v as u32),
        Value::I16(v) => u32::try_from(*v).ok(),
        Value::U32(v) => Some(*v),
        Value::I32(v) => u32::try_from(*v).ok(),
        Value::U64(v) => u32::try_from(*v).ok(),
        Value::I64(v) => u32::try_from(*v).ok(),
        _ => None,
    }
}

pub(crate) fn get_bool(val: Option<&candle_core::quantized::gguf_file::Value>) -> Option<bool> {
    match val {
        Some(candle_core::quantized::gguf_file::Value::Bool(v)) => Some(*v),
        _ => None,
    }
}

pub(crate) fn get_f64(val: Option<&candle_core::quantized::gguf_file::Value>) -> Option<f64> {
    match val {
        Some(candle_core::quantized::gguf_file::Value::F32(v)) => Some(*v as f64),
        Some(candle_core::quantized::gguf_file::Value::F64(v)) => Some(*v),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub projection_dim: usize,
    pub default_output_length: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 16,
            num_attention_heads: 12,
            num_key_value_heads: 12,
            head_dim: 64,
            patch_size: 16,
            pooling_kernel_size: 3,
            position_embedding_size: 10240,
            rope_theta: 100.0,
            rms_norm_eps: 1e-6,
            projection_dim: 512,
            default_output_length: 280,
        }
    }
}

impl VisionConfig {
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
    ) -> Option<Self> {
        let has_vision = get_bool(metadata.get("clip.has_vision_encoder")).unwrap_or(false);
        if !has_vision && !metadata.keys().any(|k| k.starts_with("clip.vision.")) {
            return None;
        }

        let mut cfg = Self::default();
        if let Some(v) = get_u32(metadata.get("clip.vision.embedding_length")) {
            cfg.hidden_size = v as usize;
        }
        if let Some(v) = get_u32(metadata.get("clip.vision.feed_forward_length")) {
            cfg.intermediate_size = v as usize;
        }
        if let Some(v) = get_u32(metadata.get("clip.vision.block_count")) {
            cfg.num_hidden_layers = v as usize;
        }
        if let Some(v) = get_u32(metadata.get("clip.vision.attention.head_count")) {
            if v == 0 {
                return None;
            }
            cfg.num_attention_heads = v as usize;
            cfg.num_key_value_heads = v as usize;
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
        }
        if let Some(v) = get_u32(metadata.get("clip.vision.patch_size")) {
            cfg.patch_size = v as usize;
        }
        if let Some(v) = get_u32(metadata.get("clip.vision.projection_dim")) {
            cfg.projection_dim = v as usize;
        }
        if let Some(v) = get_f64(metadata.get("clip.vision.attention.layer_norm_epsilon")) {
            cfg.rms_norm_eps = v;
        }

        Some(cfg)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_mel_bins: usize,
    pub output_proj_dims: usize,
    pub projection_dim: usize,
    pub rms_norm_eps: f64,
    pub conv_kernel_size: usize,
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    pub attention_context_right: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 12,
            num_attention_heads: 8,
            num_mel_bins: 128,
            output_proj_dims: 1536,
            projection_dim: 512,
            rms_norm_eps: 1e-6,
            conv_kernel_size: 5,
            attention_chunk_size: 12,
            attention_context_left: 13,
            attention_context_right: 0,
        }
    }
}

impl AudioConfig {
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
    ) -> Option<Self> {
        let has_audio = get_bool(metadata.get("clip.has_audio_encoder")).unwrap_or(false);
        if !has_audio && !metadata.keys().any(|k| k.starts_with("clip.audio.")) {
            return None;
        }

        let mut cfg = Self::default();
        if let Some(v) = get_u32(metadata.get("clip.audio.embedding_length")) {
            cfg.hidden_size = v as usize;
        }
        if let Some(v) = get_u32(metadata.get("clip.audio.feed_forward_length")) {
            cfg.intermediate_size = v as usize;
        }
        if let Some(v) = get_u32(metadata.get("clip.audio.block_count")) {
            cfg.num_hidden_layers = v as usize;
        }
        if let Some(v) = get_u32(metadata.get("clip.audio.attention.head_count")) {
            if v == 0 {
                return None;
            }
            cfg.num_attention_heads = v as usize;
        }
        if let Some(v) = get_u32(metadata.get("clip.audio.num_mel_bins")) {
            cfg.num_mel_bins = v as usize;
        }
        if let Some(v) = get_u32(metadata.get("clip.audio.projection_dim")) {
            cfg.projection_dim = v as usize;
        }
        if let Some(v) = get_f64(metadata.get("clip.audio.attention.layer_norm_epsilon")) {
            cfg.rms_norm_eps = v;
        }

        Some(cfg)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalConfig {
    pub vision_config: Option<VisionConfig>,
    pub audio_config: Option<AudioConfig>,
    pub boi_token_id: usize,
    pub eoi_token_id: usize,
    pub image_token_id: usize,
    pub boa_token_id: usize,
    pub eoa_token_id: usize,
    pub audio_token_id: usize,
}

impl MultimodalConfig {
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
    ) -> Self {
        let vision_config = VisionConfig::from_gguf_metadata(metadata);
        let audio_config = AudioConfig::from_gguf_metadata(metadata);

        Self {
            vision_config,
            audio_config,
            boi_token_id: 255999,
            eoi_token_id: 258882,
            image_token_id: 258880,
            boa_token_id: 256000,
            eoa_token_id: 258883,
            audio_token_id: 258881,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file::Value;

    #[test]
    fn test_get_u32_rejects_negative_and_overflow() {
        assert_eq!(get_u32(Some(&Value::I8(-1))), None);
        assert_eq!(get_u32(Some(&Value::I16(-42))), None);
        assert_eq!(get_u32(Some(&Value::I32(-100))), None);
        assert_eq!(get_u32(Some(&Value::I64(-1))), None);
        assert_eq!(get_u32(Some(&Value::I64(u32::MAX as i64 + 1))), None);
        assert_eq!(get_u32(Some(&Value::U64(u32::MAX as u64 + 1))), None);

        assert_eq!(get_u32(Some(&Value::U8(7))), Some(7));
        assert_eq!(get_u32(Some(&Value::I8(7))), Some(7));
        assert_eq!(get_u32(Some(&Value::I32(512))), Some(512));
        assert_eq!(get_u32(Some(&Value::U64(u32::MAX as u64))), Some(u32::MAX));
    }

    #[test]
    fn test_text_and_vision_config_validation() {
        let mut meta = HashMap::new();
        meta.insert("gemma-embedding2.block_count".to_string(), Value::U32(6));
        meta.insert(
            "gemma-embedding2.embedding_length".to_string(),
            Value::U32(512),
        );
        meta.insert(
            "gemma-embedding2.feed_forward_length".to_string(),
            Value::U32(2048),
        );
        meta.insert(
            "gemma-embedding2.attention.head_count".to_string(),
            Value::U32(4),
        );
        meta.insert(
            "gemma-embedding2.attention.sliding_window".to_string(),
            Value::U32(1024),
        );

        let cfg = TextConfig::from_gguf_metadata(&meta).unwrap();
        assert_eq!(cfg.sliding_window, 512);
        assert_eq!(cfg.layer_types.len(), 6);

        // Mismatched sliding_window_pattern length must fail
        meta.insert(
            "gemma-embedding2.attention.sliding_window_pattern".to_string(),
            Value::Array(vec![Value::Bool(true), Value::Bool(false)]),
        );
        assert!(TextConfig::from_gguf_metadata(&meta).is_err());
        meta.remove("gemma-embedding2.attention.sliding_window_pattern");

        // Zero attention heads must fail
        meta.insert(
            "gemma-embedding2.attention.head_count".to_string(),
            Value::U32(0),
        );
        assert!(TextConfig::from_gguf_metadata(&meta).is_err());

        // VisionConfig / AudioConfig must not divide by zero when head_count == 0
        let mut vis_meta = HashMap::new();
        vis_meta.insert("clip.has_vision_encoder".to_string(), Value::Bool(true));
        vis_meta.insert(
            "clip.vision.attention.head_count".to_string(),
            Value::U32(0),
        );
        assert!(VisionConfig::from_gguf_metadata(&vis_meta).is_none());

        let mut aud_meta = HashMap::new();
        aud_meta.insert("clip.has_audio_encoder".to_string(), Value::Bool(true));
        aud_meta.insert("clip.audio.attention.head_count".to_string(), Value::U32(0));
        assert!(AudioConfig::from_gguf_metadata(&aud_meta).is_none());
    }

    #[test]
    fn test_text_config_from_gguf_metadata() {
        let mut meta = HashMap::new();
        meta.insert("gemma-embedding2.block_count".to_string(), Value::U32(6));
        meta.insert(
            "gemma-embedding2.embedding_length".to_string(),
            Value::U32(768),
        );
        meta.insert(
            "gemma-embedding2.feed_forward_length".to_string(),
            Value::U32(3072),
        );
        meta.insert(
            "gemma-embedding2.attention.head_count".to_string(),
            Value::U32(8),
        );
        meta.insert(
            "gemma-embedding2.attention.key_length_swa".to_string(),
            Value::U32(256),
        );
        meta.insert(
            "gemma-embedding2.attention.key_length".to_string(),
            Value::U32(512),
        );
        meta.insert(
            "gemma-embedding2.attention.head_count_kv".to_string(),
            Value::Array(vec![
                Value::U32(2),
                Value::U32(2),
                Value::U32(2),
                Value::U32(2),
                Value::U32(2),
                Value::U32(1),
            ]),
        );
        meta.insert(
            "gemma-embedding2.attention.sliding_window".to_string(),
            Value::U32(1024),
        );
        meta.insert(
            "gemma-embedding2.rope.freq_base_swa".to_string(),
            Value::F32(10_000.0),
        );
        meta.insert(
            "gemma-embedding2.rope.freq_base".to_string(),
            Value::F32(1_000_000.0),
        );
        meta.insert(
            "gemma-embedding2.embedding_length_out".to_string(),
            Value::U32(768),
        );
        meta.insert(
            "gemma-embedding2.embedding_length_per_layer_input".to_string(),
            Value::U32(512),
        );
        meta.insert(
            "gemma-embedding2.context_length".to_string(),
            Value::U32(32768),
        );
        meta.insert("tokenizer.ggml.padding_token_id".to_string(), Value::U32(0));
        meta.insert("tokenizer.ggml.bos_token_id".to_string(), Value::U32(2));
        meta.insert("tokenizer.ggml.eos_token_id".to_string(), Value::U32(1));
        meta.insert(
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(vec![Value::String("a".to_string()); 128]),
        );

        let cfg = TextConfig::from_gguf_metadata(&meta).unwrap();
        assert_eq!(cfg.num_hidden_layers, 6);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.vocab_size, 128);
        assert_eq!(cfg.sliding_window, 512);
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert_eq!(
            cfg.layer_types,
            vec![
                LayerType::SlidingAttention,
                LayerType::SlidingAttention,
                LayerType::SlidingAttention,
                LayerType::SlidingAttention,
                LayerType::SlidingAttention,
                LayerType::FullAttention,
            ]
        );

        for i in 0..5 {
            assert_eq!(cfg.get_head_dim(i), 256);
            assert_eq!(cfg.get_num_kv_heads(i), 2);
            assert!((cfg.get_rope_theta(i) - 10_000.0).abs() < 1e-6);
        }
        assert_eq!(cfg.get_head_dim(5), 512);
        assert_eq!(cfg.get_num_kv_heads(5), 1);
        assert!((cfg.get_rope_theta(5) - 1_000_000.0).abs() < 1e-6);
        assert!(cfg.per_layer_config.contains_key("05"));
        assert!(!cfg.per_layer_config.contains_key("00"));
    }

    #[test]
    fn test_vision_and_audio_config_from_gguf_metadata() {
        let empty = HashMap::new();
        assert!(VisionConfig::from_gguf_metadata(&empty).is_none());
        assert!(AudioConfig::from_gguf_metadata(&empty).is_none());
        let mm_empty = MultimodalConfig::from_gguf_metadata(&empty);
        assert!(mm_empty.vision_config.is_none());
        assert!(mm_empty.audio_config.is_none());

        let mut meta = HashMap::new();
        meta.insert("clip.has_vision_encoder".to_string(), Value::Bool(true));
        meta.insert("clip.vision.embedding_length".to_string(), Value::U32(512));
        meta.insert(
            "clip.vision.feed_forward_length".to_string(),
            Value::U32(2048),
        );
        meta.insert("clip.vision.block_count".to_string(), Value::U32(12));
        meta.insert(
            "clip.vision.attention.head_count".to_string(),
            Value::U32(8),
        );
        meta.insert("clip.vision.patch_size".to_string(), Value::U32(14));
        meta.insert("clip.vision.projection_dim".to_string(), Value::U32(256));
        meta.insert(
            "clip.vision.attention.layer_norm_epsilon".to_string(),
            Value::F64(1e-5),
        );

        meta.insert("clip.has_audio_encoder".to_string(), Value::Bool(true));
        meta.insert("clip.audio.embedding_length".to_string(), Value::U32(640));
        meta.insert(
            "clip.audio.feed_forward_length".to_string(),
            Value::U32(1920),
        );
        meta.insert("clip.audio.block_count".to_string(), Value::U32(6));
        meta.insert(
            "clip.audio.attention.head_count".to_string(),
            Value::U32(10),
        );
        meta.insert("clip.audio.num_mel_bins".to_string(), Value::U32(80));
        meta.insert("clip.audio.projection_dim".to_string(), Value::U32(384));
        meta.insert(
            "clip.audio.attention.layer_norm_epsilon".to_string(),
            Value::F32(1e-5),
        );

        let mm = MultimodalConfig::from_gguf_metadata(&meta);
        let vis = mm.vision_config.expect("vision config present");
        assert_eq!(vis.hidden_size, 512);
        assert_eq!(vis.intermediate_size, 2048);
        assert_eq!(vis.num_hidden_layers, 12);
        assert_eq!(vis.num_attention_heads, 8);
        assert_eq!(vis.num_key_value_heads, 8);
        assert_eq!(vis.head_dim, 64);
        assert_eq!(vis.patch_size, 14);
        assert_eq!(vis.projection_dim, 256);
        assert!((vis.rms_norm_eps - 1e-5).abs() < 1e-12);

        let aud = mm.audio_config.expect("audio config present");
        assert_eq!(aud.hidden_size, 640);
        assert_eq!(aud.intermediate_size, 1920);
        assert_eq!(aud.num_hidden_layers, 6);
        assert_eq!(aud.num_attention_heads, 10);
        assert_eq!(aud.num_mel_bins, 80);
        assert_eq!(aud.projection_dim, 384);
        assert!((aud.rms_norm_eps - 1e-5).abs() < 1e-9);
    }

    #[test]
    fn test_text_config_from_json_nested_and_flat() {
        let mut rope_parameters = HashMap::new();
        rope_parameters.insert(
            "sliding_attention".to_string(),
            RoPEParameters {
                rope_theta: 10_000.0,
                rope_type: "default".to_string(),
            },
        );
        rope_parameters.insert(
            "full_attention".to_string(),
            RoPEParameters {
                rope_theta: 1_000_000.0,
                rope_type: "default".to_string(),
            },
        );
        let mut per_layer_config = HashMap::new();
        per_layer_config.insert(
            "01".to_string(),
            LayerOverride {
                head_dim: 16,
                num_key_value_heads: 1,
            },
        );

        let sample = TextConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            vocab_size: 128,
            embedding_dim: 32,
            max_position_embeddings: 256,
            sliding_window: 16,
            rms_norm_eps: 1e-6,
            hidden_size_per_layer_input: 16,
            layer_types: vec![LayerType::SlidingAttention, LayerType::FullAttention],
            per_layer_config,
            rope_parameters,
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 1,
        };

        let dir = std::env::temp_dir().join(format!("glypr_cfg_json_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let flat_path = dir.join("flat.json");
        let nested_path = dir.join("nested.json");

        std::fs::write(&flat_path, serde_json::to_string_pretty(&sample).unwrap()).unwrap();
        let nested_val = serde_json::json!({ "text_config": &sample });
        std::fs::write(
            &nested_path,
            serde_json::to_string_pretty(&nested_val).unwrap(),
        )
        .unwrap();

        let flat_cfg = TextConfig::from_json_file(&flat_path).unwrap();
        let nested_cfg = TextConfig::from_json_file(&nested_path).unwrap();

        assert_eq!(flat_cfg.hidden_size, 32);
        assert_eq!(nested_cfg.hidden_size, 32);
        assert_eq!(flat_cfg.get_head_dim(0), 8);
        assert_eq!(flat_cfg.get_head_dim(1), 16);
        assert_eq!(nested_cfg.get_num_kv_heads(0), 2);
        assert_eq!(nested_cfg.get_num_kv_heads(1), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
