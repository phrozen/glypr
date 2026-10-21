use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigFile {
    pub text_config: Option<TextConfig>,
    // Root config fallback fields if text_config is absent
    pub hidden_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub embedding_dim: Option<usize>,
}

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
        Ok(config)
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

        let sliding_window = 512;

        Ok(Self {
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
        })
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
    match val {
        Some(candle_core::quantized::gguf_file::Value::U8(v)) => Some(*v as u32),
        Some(candle_core::quantized::gguf_file::Value::I8(v)) => Some(*v as u32),
        Some(candle_core::quantized::gguf_file::Value::U16(v)) => Some(*v as u32),
        Some(candle_core::quantized::gguf_file::Value::I16(v)) => Some(*v as u32),
        Some(candle_core::quantized::gguf_file::Value::U32(v)) => Some(*v),
        Some(candle_core::quantized::gguf_file::Value::I32(v)) => Some(*v as u32),
        Some(candle_core::quantized::gguf_file::Value::U64(v)) => Some(*v as u32),
        Some(candle_core::quantized::gguf_file::Value::I64(v)) => Some(*v as u32),
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
