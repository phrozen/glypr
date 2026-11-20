mod common;

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear};
use glypr_core::attention::{AttentionMode, SelfAttention};
use glypr_core::config::{LayerType, RoPEParameters, TextConfig};
use glypr_core::layer::EncoderLayer;
use glypr_core::linear::LinearOp;
use glypr_core::mlp::MLP;
use glypr_core::model::EmbeddingGemma2Model;
use glypr_core::norm::RMSNorm;
use glypr_core::ple::{TextPLE, TextPLEBlock};
use glypr_core::rope::RotaryEmbedding;
use glypr_core::util::cosine_similarity;
use std::collections::HashMap;

fn deterministic_tensor(
    shape: &[usize],
    seed: usize,
    scale: f32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let phase = ((i + seed * 31) as f32) * 0.37 + (seed as f32) * 0.13;
            phase.sin() * scale
        })
        .collect();
    Ok(Tensor::from_vec(data, shape, device)?.to_dtype(dtype)?)
}

fn deterministic_norm_weight(
    len: usize,
    seed: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let data: Vec<f32> = (0..len)
        .map(|i| 1.0 + 0.05 * (((i + seed * 17) as f32) * 0.5).cos())
        .collect();
    Ok(Tensor::from_vec(data, (len,), device)?.to_dtype(dtype)?)
}

fn tiny_config(max_position_embeddings: usize) -> TextConfig {
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

    TextConfig {
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 8,
        vocab_size: 64,
        embedding_dim: 16,
        max_position_embeddings,
        sliding_window: 4,
        rms_norm_eps: 1e-6,
        hidden_size_per_layer_input: 8,
        layer_types: vec![LayerType::SlidingAttention, LayerType::FullAttention],
        per_layer_config: HashMap::new(),
        rope_parameters,
        pad_token_id: 0,
        bos_token_id: 2,
        eos_token_id: 1,
    }
}

fn build_tiny_model(
    device: &Device,
    dtype: DType,
    max_position_embeddings: usize,
) -> Result<EmbeddingGemma2Model> {
    let config = tiny_config(max_position_embeddings);
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let ple_dim = config.hidden_size_per_layer_input;

    let embed_weight = deterministic_tensor(&[config.vocab_size, h], 1, 0.25, device, dtype)?;
    let embed_tokens = Embedding::new(embed_weight, h);
    let embed_scale = (h as f64).sqrt();

    let ple_proj_w = deterministic_tensor(
        &[config.num_hidden_layers * ple_dim, h],
        2,
        0.2,
        device,
        dtype,
    )?;
    let ple_norm_w = deterministic_norm_weight(ple_dim, 3, device, dtype)?;
    let ple = TextPLE::new(
        LinearOp::new_standard(Linear::new(ple_proj_w, None)),
        (h as f64).powf(-0.5),
        RMSNorm::new(Some(ple_norm_w), config.rms_norm_eps),
        config.num_hidden_layers,
        ple_dim,
    );

    let rope_sliding = RotaryEmbedding::new(config.head_dim, 10_000.0, device)?;
    let rope_full = RotaryEmbedding::new(config.head_dim, 1_000_000.0, device)?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let s = 10 + i * 20;
        let q_w = deterministic_tensor(
            &[config.num_attention_heads * config.head_dim, h],
            s + 1,
            0.2,
            device,
            dtype,
        )?;
        let k_w = deterministic_tensor(
            &[config.num_key_value_heads * config.head_dim, h],
            s + 2,
            0.2,
            device,
            dtype,
        )?;
        let v_w = deterministic_tensor(
            &[config.num_key_value_heads * config.head_dim, h],
            s + 3,
            0.2,
            device,
            dtype,
        )?;
        let o_w = deterministic_tensor(
            &[h, config.num_attention_heads * config.head_dim],
            s + 4,
            0.2,
            device,
            dtype,
        )?;

        let q_norm = RMSNorm::new(
            Some(deterministic_norm_weight(
                config.head_dim,
                s + 5,
                device,
                dtype,
            )?),
            config.rms_norm_eps,
        );
        let k_norm = RMSNorm::new(
            Some(deterministic_norm_weight(
                config.head_dim,
                s + 6,
                device,
                dtype,
            )?),
            config.rms_norm_eps,
        );
        let v_norm = RMSNorm::new(None, config.rms_norm_eps);

        let sliding_window = match config.layer_types[i] {
            LayerType::SlidingAttention => Some(config.sliding_window),
            LayerType::FullAttention => None,
        };

        let self_attn = SelfAttention::new(
            LinearOp::new_standard(Linear::new(q_w, None)),
            LinearOp::new_standard(Linear::new(k_w, None)),
            LinearOp::new_standard(Linear::new(v_w, None)),
            LinearOp::new_standard(Linear::new(o_w, None)),
            q_norm,
            k_norm,
            v_norm,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            sliding_window,
        );

        let gate_w = deterministic_tensor(&[inter, h], s + 7, 0.2, device, dtype)?;
        let up_w = deterministic_tensor(&[inter, h], s + 8, 0.2, device, dtype)?;
        let down_w = deterministic_tensor(&[h, inter], s + 9, 0.2, device, dtype)?;
        let mlp = MLP::new(
            LinearOp::new_standard(Linear::new(gate_w, None)),
            LinearOp::new_standard(Linear::new(up_w, None)),
            LinearOp::new_standard(Linear::new(down_w, None)),
        );

        let ple_gate_w = deterministic_tensor(&[ple_dim, h], s + 10, 0.2, device, dtype)?;
        let ple_block_proj_w = deterministic_tensor(&[h, ple_dim], s + 11, 0.2, device, dtype)?;
        let ple_block_norm = RMSNorm::new(
            Some(deterministic_norm_weight(h, s + 12, device, dtype)?),
            config.rms_norm_eps,
        );
        let ple_block = TextPLEBlock::new(
            LinearOp::new_standard(Linear::new(ple_gate_w, None)),
            LinearOp::new_standard(Linear::new(ple_block_proj_w, None)),
            ple_block_norm,
        );

        let layer_scalar = Tensor::from_vec(vec![0.95f32], (1,), device)?.to_dtype(dtype)?;

        layers.push(EncoderLayer {
            self_attn,
            mlp,
            input_layernorm: RMSNorm::new(
                Some(deterministic_norm_weight(h, s + 13, device, dtype)?),
                config.rms_norm_eps,
            ),
            post_attention_layernorm: RMSNorm::new(
                Some(deterministic_norm_weight(h, s + 14, device, dtype)?),
                config.rms_norm_eps,
            ),
            pre_feedforward_layernorm: RMSNorm::new(
                Some(deterministic_norm_weight(h, s + 15, device, dtype)?),
                config.rms_norm_eps,
            ),
            post_feedforward_layernorm: RMSNorm::new(
                Some(deterministic_norm_weight(h, s + 16, device, dtype)?),
                config.rms_norm_eps,
            ),
            ple_block,
            layer_scalar,
        });
    }

    let final_norm = RMSNorm::new(
        Some(deterministic_norm_weight(h, 90, device, dtype)?),
        config.rms_norm_eps,
    );
    let proj_w = deterministic_tensor(&[config.embedding_dim, h], 91, 0.25, device, dtype)?;

    Ok(EmbeddingGemma2Model {
        config,
        embed_tokens,
        embed_scale,
        ple,
        layers,
        norm: final_norm,
        embedding_projection: LinearOp::new_standard(Linear::new(proj_w, None)),
        rope_sliding,
        rope_full,
        device: device.clone(),
        dtype,
    })
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_standard_vs_chunked_check(device: &Device, dtype: DType) -> Result<()> {
    let mut model = build_tiny_model(device, dtype, 64)?;

    // seq_len = 11 > sliding_window = 4, exercising multiple chunks and tail remainder
    let seq_len = 11usize;
    let ids_unmasked = Tensor::from_vec(
        vec![2u32, 5, 8, 12, 17, 23, 29, 31, 40, 55, 1],
        (1, seq_len),
        device,
    )?;

    model.set_attention_mode(AttentionMode::Standard);
    let out_std = model
        .forward(&ids_unmasked, None)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    model.set_attention_mode(AttentionMode::ChunkedSliding);
    let out_chk = model
        .forward(&ids_unmasked, None)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let diff_unmasked = max_abs_diff(&out_std, &out_chk);
    let cos_unmasked = cosine_similarity(&out_std, &out_chk);
    if dtype == DType::F32 {
        assert!(
            diff_unmasked < 1e-5,
            "Standard vs ChunkedSliding unmasked max_diff={diff_unmasked} >= 1e-5"
        );
    } else {
        assert!(
            cos_unmasked >= 0.9999 && diff_unmasked < 2e-2,
            "BF16 Standard vs ChunkedSliding unmasked cos={cos_unmasked}, max_diff={diff_unmasked}"
        );
    }

    // Masked batch of 2 sequences with seq_len = 11 (item 1 has 4 trailing pads)
    let ids_batch = Tensor::from_vec(
        vec![
            2u32, 5, 8, 12, 17, 23, 29, 31, 40, 55, 1, // item 0: 11 valid
            2, 9, 14, 21, 33, 44, 1, 0, 0, 0, 0, // item 1: 7 valid + 4 pads
        ],
        (2, seq_len),
        device,
    )?;
    let mask_batch = Tensor::from_vec(
        vec![
            1u32, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        ],
        (2, seq_len),
        device,
    )?;

    model.set_attention_mode(AttentionMode::Standard);
    let masked_std = model
        .forward(&ids_batch, Some(&mask_batch))?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    model.set_attention_mode(AttentionMode::ChunkedSliding);
    let masked_chk = model
        .forward(&ids_batch, Some(&mask_batch))?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let diff_masked = max_abs_diff(&masked_std, &masked_chk);
    let cos_masked = cosine_similarity(&masked_std, &masked_chk);
    if dtype == DType::F32 {
        assert!(
            diff_masked < 1e-5,
            "Standard vs ChunkedSliding masked max_diff={diff_masked} >= 1e-5"
        );
    } else {
        assert!(
            cos_masked >= 0.9999 && diff_masked < 2e-2,
            "BF16 Standard vs ChunkedSliding masked cos={cos_masked}, max_diff={diff_masked}"
        );
    }

    Ok(())
}

#[test]
fn test_synthetic_standard_vs_chunked_attention_cpu_and_metal() -> Result<()> {
    run_standard_vs_chunked_check(&Device::Cpu, DType::F32)?;

    let metal_dev = common::safe_test_device();
    if metal_dev.is_metal() {
        run_standard_vs_chunked_check(&metal_dev, DType::BF16)?;
    }
    Ok(())
}

#[test]
fn test_build_attention_mask_fast_vs_dense_equivalence() -> Result<()> {
    let dev = Device::Cpu;
    let model = build_tiny_model(&dev, DType::F32, 64)?;

    let batch = 2usize;
    let seq_len = 6usize;
    let pad_vec = vec![
        1u32, 1, 1, 1, 1, 1, //
        1, 1, 1, 0, 0, 0,
    ];
    let pad_mask = Tensor::from_vec(pad_vec.clone(), (batch, seq_len), &dev)?;

    // Fast O(B*S) path: sliding_window=None -> [B, 1, 1, S]
    let fast_mask = model
        .build_attention_mask(batch, seq_len, None, Some(&pad_mask))?
        .expect("padding mask must yield Some(tensor)");
    assert_eq!(fast_mask.dims(), &[batch, 1, 1, seq_len]);

    // Reference dense O(B*S^2) padding mask [B, 1, S, S]
    let mut dense_ref_vec = vec![0.0f32; batch * seq_len * seq_len];
    for b in 0..batch {
        for i in 0..seq_len {
            for j in 0..seq_len {
                if pad_vec[b * seq_len + j] == 0 {
                    dense_ref_vec[b * seq_len * seq_len + i * seq_len + j] = -1e9f32;
                }
            }
        }
    }
    let fast_broadcast = fast_mask
        .broadcast_as((batch, 1, seq_len, seq_len))?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(max_abs_diff(&fast_broadcast, &dense_ref_vec), 0.0);

    // Dense O(B*S^2) sliding + padding path: seq_len = 6 > window = 4 -> [B, 1, S, S]
    let dense_sliding_pad = model
        .build_attention_mask(batch, seq_len, Some(4), Some(&pad_mask))?
        .expect("dense sliding+pad mask must yield Some(tensor)");
    assert_eq!(dense_sliding_pad.dims(), &[batch, 1, seq_len, seq_len]);

    let dense_sliding_only = model
        .build_attention_mask(batch, seq_len, Some(4), None)?
        .expect("dense sliding-only mask must yield Some(tensor)");
    let combined_fast_and_sliding = fast_mask
        .broadcast_as((batch, 1, seq_len, seq_len))?
        .minimum(&dense_sliding_only)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let dense_flat = dense_sliding_pad.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(max_abs_diff(&combined_fast_and_sliding, &dense_flat), 0.0);

    // Unmasked short sequence <= sliding_window returns None
    assert!(model.build_attention_mask(1, 4, Some(4), None)?.is_none());
    Ok(())
}

#[test]
fn test_pool_and_normalize_padding_mrl_and_4096_bf16_precision() -> Result<()> {
    let dev = Device::Cpu;
    let model = build_tiny_model(&dev, DType::F32, 64)?;

    // 1. Padding mask zeros out arbitrary garbage in padded tail tokens
    let valid_tokens = vec![2u32, 7, 15, 22, 1];
    let single_ids = Tensor::from_vec(valid_tokens.clone(), (1, 5), &dev)?;
    let single_out = model.forward(&single_ids, None)?;
    let single_emb = model
        .pool_and_normalize(&single_out, None, None)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let padded_ids = Tensor::from_vec(
        vec![
            2u32, 7, 15, 22, 1, 50, 51, 52, // item 0: 5 valid + 3 garbage pads
            2, 9, 10, 11, 12, 13, 14, 1, // item 1: 8 valid
        ],
        (2, 8),
        &dev,
    )?;
    let padded_mask = Tensor::from_vec(
        vec![
            1u32, 1, 1, 1, 1, 0, 0, 0, //
            1, 1, 1, 1, 1, 1, 1, 1,
        ],
        (2, 8),
        &dev,
    )?;
    let batch_out = model.forward(&padded_ids, Some(&padded_mask))?;
    let batch_emb = model
        .pool_and_normalize(&batch_out, Some(&padded_mask), None)?
        .to_vec2::<f32>()?;
    let diff = max_abs_diff(&single_emb, &batch_emb[0]);
    assert!(
        diff < 1e-6,
        "Masked padded batch item diverged from unpadded single item: max_diff={diff}"
    );

    // 2. MRL truncation Some(8) matches manual slice + L2 normalize
    let mrl8 = model
        .pool_and_normalize(&single_out, None, Some(8))?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(mrl8.len(), 8);
    let mrl8_norm: f32 = mrl8.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((mrl8_norm - 1.0).abs() < 1e-6, "MRL norm={mrl8_norm}");
    let expected_mrl8 = glypr_core::truncate_mrl(&single_emb, 8)?;
    assert!(max_abs_diff(&mrl8, &expected_mrl8) < 1e-6);

    // 3. 4096-token BF16 FP32-accumulation precision vs F64 ground truth
    let seq_len = 4096usize;
    let dim = 16usize;
    let mut raw_f32 = Vec::with_capacity(seq_len * dim);
    for t in 0..seq_len {
        for d in 0..dim {
            let v = ((t as f32) * 0.017 + (d as f32) * 0.31).sin() * 2.5 + 0.1;
            raw_f32.push(v);
        }
    }
    let bf16_vals: Vec<half::bf16> = raw_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();

    // Exact F64 reference computed from the exact BF16-quantized input values
    let mut sum_f64 = vec![0.0f64; dim];
    for t in 0..seq_len {
        for d in 0..dim {
            sum_f64[d] += bf16_vals[t * dim + d].to_f64();
        }
    }
    let mean_f64: Vec<f64> = sum_f64.iter().map(|&s| s / (seq_len as f64)).collect();
    let norm_f64 = mean_f64.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let ref_f32: Vec<f32> = mean_f64.iter().map(|&x| (x / norm_f64) as f32).collect();

    let bf16_tensor = Tensor::from_vec(bf16_vals, (1, seq_len, dim), &dev)?;
    let pooled_bf16 = model
        .pool_and_normalize(&bf16_tensor, None, None)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let cos_cpu = cosine_similarity(&pooled_bf16, &ref_f32);
    assert!(
        cos_cpu >= 0.99999,
        "4096-token BF16 pooling CPU cosine {cos_cpu:.7} < 0.99999"
    );

    let metal_dev = common::safe_test_device();
    if metal_dev.is_metal() {
        let bf16_metal = bf16_tensor.to_device(&metal_dev)?;
        let pooled_metal = model
            .pool_and_normalize(&bf16_metal, None, None)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let cos_metal = cosine_similarity(&pooled_metal, &ref_f32);
        assert!(
            cos_metal >= 0.99995,
            "4096-token BF16 pooling Metal cosine {cos_metal:.7} < 0.99995"
        );
    }

    Ok(())
}

#[test]
fn test_max_position_embeddings_guard_and_forward_embeds_equivalence() -> Result<()> {
    let dev = Device::Cpu;
    let max_pos = 16usize;
    let model = build_tiny_model(&dev, DType::F32, max_pos)?;

    // max_pos + 1 must fail in both forward and forward_embeds
    let too_long_ids = Tensor::zeros((1, max_pos + 1), DType::U32, &dev)?;
    let err_ids = model.forward(&too_long_ids, None).unwrap_err();
    assert!(
        err_ids.to_string().contains("max_position_embeddings"),
        "unexpected error: {err_ids}"
    );

    let too_long_embeds =
        Tensor::zeros((1, max_pos + 1, model.config.hidden_size), DType::F32, &dev)?;
    let err_embeds = model.forward_embeds(&too_long_embeds, None).unwrap_err();
    assert!(
        err_embeds.to_string().contains("max_position_embeddings"),
        "unexpected error: {err_embeds}"
    );

    // Exact equivalence between forward and forward_embeds for valid seq_len <= max_pos
    let ids = Tensor::from_vec(vec![2u32, 4, 11, 19, 27, 35, 1], (1, 7), &dev)?;
    let out_forward = model.forward(&ids, None)?.flatten_all()?.to_vec1::<f32>()?;

    let inputs_embeds = (model.embed_tokens.forward(&ids)? * model.embed_scale)?;
    let out_embeds = model
        .forward_embeds(&inputs_embeds, None)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let diff = max_abs_diff(&out_forward, &out_embeds);
    assert!(
        diff < 1e-6,
        "forward vs forward_embeds diverged: max_diff={diff}"
    );

    Ok(())
}
