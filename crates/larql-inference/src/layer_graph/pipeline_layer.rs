//! Shared FullPipelineLayer construction from ModelWeights + VectorIndex.
//!
//! Single source of truth for extracting per-layer architecture parameters
//! from larql-models and wiring them into larql-compute's FullPipelineLayer.
//! Both GPU and CPU paths use this — no duplicated param extraction.

use larql_compute::{QuantWeight, QuantFormat, FullPipelineLayer};
use crate::model::ModelWeights;

/// Extract per-layer architecture parameters into a FullPipelineLayer.
///
/// This is the single construction site for all per-layer params:
/// head_dim, num_q/kv_heads, rope_base, attn_scale, rotary_dim,
/// sliding_window, norm offsets, activation, FFN type, V-norm, layer scalar.
///
/// The attention weights (wq/wk/wv/wo) and FFN weights (gate/up/down)
/// must be provided separately since they come from different sources
/// (Q4_K from vindex, Q8 from vindex, f32 from model weights).
#[allow(clippy::too_many_arguments)]
pub fn build_arch_params<'a>(
    weights: &'a ModelWeights,
    layer: usize,
    wq: QuantWeight<'a>,
    wk: QuantWeight<'a>,
    wv: QuantWeight<'a>,
    wo: QuantWeight<'a>,
    gate: QuantWeight<'a>,
    up: QuantWeight<'a>,
    down: QuantWeight<'a>,
) -> FullPipelineLayer<'a> {
    let arch = &*weights.arch;
    let layer_hd = arch.head_dim_for_layer(layer);
    let layer_nq = arch.num_q_heads_for_layer(layer);
    let layer_nkv = arch.num_kv_heads_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let rotary_dim = if rotary_frac >= 1.0 { 0 } else { (layer_hd as f64 * rotary_frac) as usize };
    let sw = if arch.is_sliding_window_layer(layer) {
        arch.sliding_window_size().unwrap_or(0)
    } else {
        0
    };
    let layer_scalar = arch.layer_scalar_key(layer)
        .and_then(|k| weights.vectors.get(&k)
            // Fallback: GGUF normalizes `layer_output_scale.weight` → `layer_scalar.weight`
            // while safetensors uses `layer_scalar` (no .weight suffix for scalars).
            .or_else(|| weights.vectors.get(&format!("{k}.weight"))))
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0);

    FullPipelineLayer {
        wq, wk, wv, wo,
        gate, up, down,
        input_norm: weights.vectors.get(&arch.input_layernorm_key(layer))
            .map(|v| v.as_slice()).unwrap_or(&[]),
        post_attn_norm: weights.vectors.get(&arch.post_attention_layernorm_key(layer))
            // Fallback: GGUF normalizes `ffn_norm` to `pre_feedforward_layernorm`.
            // For 2-norm models (Llama) where HF calls this norm `post_attention_layernorm`,
            // look up `pre_feedforward_layernorm` as a fallback.
            .or_else(|| arch.pre_feedforward_layernorm_key(layer)
                .and_then(|k| weights.vectors.get(&k)))
            .map(|v| v.as_slice()).unwrap_or(&[]),
        pre_ffn_norm: arch.pre_feedforward_layernorm_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
        post_ffn_norm: arch.post_feedforward_layernorm_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
        norm_offset: arch.norm_weight_offset(),
        has_post_norms: arch.has_post_norms(),
        activation: match arch.activation() {
            larql_models::Activation::GeluTanh => larql_compute::Activation::GeluTanh,
            _ => larql_compute::Activation::Silu,
        },
        qk_norm_offset: arch.qk_norm_weight_offset(),
        eps: arch.norm_eps(),
        norm_type: match arch.norm_type() {
            larql_models::NormType::LayerNorm => larql_compute::NormType::LayerNorm,
            _ => larql_compute::NormType::RmsNorm,
        },
        ffn_type: match arch.ffn_type() {
            larql_models::FfnType::Standard => larql_compute::FfnType::Standard,
            _ => larql_compute::FfnType::Gated,
        },
        attn_scale: arch.attention_scale_for_layer(layer) as f32,
        head_dim: layer_hd,
        num_q_heads: layer_nq,
        num_kv_heads: layer_nkv,
        rope_base: arch.rope_base_for_layer(layer) as f32,
        rotary_dim,
        sliding_window: sw,
        has_v_norm: arch.has_v_norm(),
        layer_scalar,
        input_norm_bias: None,
        q_norm_weight: arch.attn_q_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
        k_norm_weight: arch.attn_k_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
        post_attn_norm_bias: None,
        softcap: arch.attn_logit_softcapping().unwrap_or(0.0),
        ffn_up_bias: arch.ffn_up_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
        ffn_down_bias: arch.ffn_down_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
        router_weight: None,
        expert_gate_up: None,
        expert_down: None,
        expert_down_scale: None,
        is_moe_layer: false,
        num_experts: 0,
        num_active_experts: 0,
        expert_intermediate: 0,
    }
}

/// Helper: resolve attention weights from vindex (Q4_K preferred, Q8 fallback).
pub fn resolve_attn_weights<'a>(
    index: &'a larql_vindex::VectorIndex,
    layer: usize,
) -> Option<(QuantWeight<'a>, QuantWeight<'a>, QuantWeight<'a>, QuantWeight<'a>)> {
    fn to_format(s: &str) -> QuantFormat {
        match s {
            "Q6_K" => QuantFormat::Q6_K,
            "Q4_KF" | "Q4_K_GGUF" => QuantFormat::Q4_KF,
            "Q8_0" => QuantFormat::Q8_0Gguf,
            _ => QuantFormat::Q4_K,
        }
    }

    if let Some([q, k, v, o]) = index.attn_q4k_layer_data(layer) {
        Some((
            QuantWeight { data: q.0, scales: None, format: to_format(q.1) },
            QuantWeight { data: k.0, scales: None, format: to_format(k.1) },
            QuantWeight { data: v.0, scales: None, format: to_format(v.1) },
            QuantWeight { data: o.0, scales: None, format: to_format(o.1) },
        ))
    } else if let Some([q, k, v, o]) = index.attn_q8_layer_data(layer) {
        Some((
            QuantWeight { data: q.0, scales: Some(q.1), format: QuantFormat::Q8_0 },
            QuantWeight { data: k.0, scales: Some(k.1), format: QuantFormat::Q8_0 },
            QuantWeight { data: v.0, scales: Some(v.1), format: QuantFormat::Q8_0 },
            QuantWeight { data: o.0, scales: Some(o.1), format: QuantFormat::Q8_0 },
        ))
    } else {
        None
    }
}

/// Helper: resolve FFN weights from vindex interleaved mmap.
///
/// Supports the Ollama / build_q4k_weights mixed layout where gate and up
/// are Q4_K but down is Q6_K. Pass `Some(down_bytes_per_matrix)` and
/// `Some(down_format)` to override; `None` falls back to uniform layout.
pub fn resolve_ffn_weights_mixed<'a>(
    q4_ffn_mmap: &'a [u8],
    layer: usize,
    gate_up_bytes: usize,
    gate_up_format: QuantFormat,
    down_bytes: usize,
    down_format: QuantFormat,
) -> (QuantWeight<'a>, QuantWeight<'a>, QuantWeight<'a>) {
    let per_layer = gate_up_bytes * 2 + down_bytes;
    let fs = layer * per_layer;
    (
        QuantWeight { data: &q4_ffn_mmap[fs..fs + gate_up_bytes], scales: None, format: gate_up_format },
        QuantWeight { data: &q4_ffn_mmap[fs + gate_up_bytes..fs + 2 * gate_up_bytes], scales: None, format: gate_up_format },
        QuantWeight { data: &q4_ffn_mmap[fs + 2 * gate_up_bytes..fs + 2 * gate_up_bytes + down_bytes], scales: None, format: down_format },
    )
}

/// Uniform-Q4 layout (gate/up/down all same size + format).
pub fn resolve_ffn_weights<'a>(
    q4_ffn_mmap: &'a [u8],
    layer: usize,
    q4_ffn_per_matrix: usize,
    ffn_format: QuantFormat,
) -> (QuantWeight<'a>, QuantWeight<'a>, QuantWeight<'a>) {
    resolve_ffn_weights_mixed(q4_ffn_mmap, layer, q4_ffn_per_matrix, ffn_format, q4_ffn_per_matrix, ffn_format)
}

/// Build a complete Vec<FullPipelineLayer> for a range of layers.
/// Single source of truth — used by both GPU decode and GPU prefill paths.
#[allow(clippy::too_many_arguments)]
pub fn build_pipeline_layers<'a>(
    weights: &'a ModelWeights,
    index: &'a larql_vindex::VectorIndex,
    layer_range: std::ops::Range<usize>,
    q4_ffn_mmap: &'a [u8],
    q4_ffn_per_matrix: usize,
    ffn_format: QuantFormat,
) -> Vec<FullPipelineLayer<'a>> {
    let num_layers_in_file = weights.num_layers;
    // Supported FFN layouts (per-layer byte counts per 256 values × superblocks):
    //   uniform Q4K Ollama (148 B): per_layer = 3 × q4_per_matrix
    //   Q4K + Q4K + Q6K (Ollama):   per_layer = 2 × q4_per_matrix + q6_per_matrix
    //   all Q6K:                    per_layer = 3 × q6_per_matrix
    //   Q4KF + Q4KF + Q6K:          per_layer = 2 × q4kf_per_matrix + q6_per_matrix (llama.cpp)
    //   uniform Q4KF:               per_layer = 3 × q4kf_per_matrix
    // q4_ffn_per_matrix is the Ollama Q4K size (148). Q4KF (GGUF) is 144, ratio 144/148.
    // q6_per_matrix is 210/148 × q4_per_matrix.
    let q4kf_per_matrix = q4_ffn_per_matrix * 144 / 148;
    let q6_per_matrix = q4_ffn_per_matrix * 210 / 148;
    let _uniform_q4_per_layer = q4_ffn_per_matrix * 3;
    let mixed_q4q6_per_layer = 2 * q4_ffn_per_matrix + q6_per_matrix;
    let uniform_q6_per_layer = q6_per_matrix * 3;
    let mixed_q4kf_q6_per_layer = 2 * q4kf_per_matrix + q6_per_matrix;
    let uniform_q4kf_per_layer = q4kf_per_matrix * 3;
    let actual_per_layer = q4_ffn_mmap.len() / num_layers_in_file.max(1);
    // Choose the layout whose per-layer size matches actual.
    let (gate_bytes, gate_format, up_bytes, up_format, down_bytes, down_format) =
        if actual_per_layer == mixed_q4kf_q6_per_layer {
            (q4kf_per_matrix, QuantFormat::Q4_KF, q4kf_per_matrix, QuantFormat::Q4_KF, q6_per_matrix, QuantFormat::Q6_K)
        } else if actual_per_layer == uniform_q4kf_per_layer {
            (q4kf_per_matrix, QuantFormat::Q4_KF, q4kf_per_matrix, QuantFormat::Q4_KF, q4kf_per_matrix, QuantFormat::Q4_KF)
        } else if actual_per_layer == uniform_q6_per_layer {
            (q6_per_matrix, QuantFormat::Q6_K, q6_per_matrix, QuantFormat::Q6_K, q6_per_matrix, QuantFormat::Q6_K)
        } else if actual_per_layer == mixed_q4q6_per_layer {
            (q4_ffn_per_matrix, ffn_format, q4_ffn_per_matrix, ffn_format, q6_per_matrix, QuantFormat::Q6_K)
        } else {
            (q4_ffn_per_matrix, ffn_format, q4_ffn_per_matrix, ffn_format, q4_ffn_per_matrix, ffn_format)
        };
    layer_range.map(|layer| {
        let (wq, wk, wv, wo) = resolve_attn_weights(index, layer)
            .expect("No attention weights available for layer");
        let per_layer = gate_bytes + up_bytes + down_bytes;
        let base = layer * per_layer;
        let gate = QuantWeight {
            data: &q4_ffn_mmap[base..base + gate_bytes],
            scales: None, format: gate_format,
        };
        let up = QuantWeight {
            data: &q4_ffn_mmap[base + gate_bytes..base + gate_bytes + up_bytes],
            scales: None, format: up_format,
        };
        let down = QuantWeight {
            data: &q4_ffn_mmap[base + gate_bytes + up_bytes..base + per_layer],
            scales: None, format: down_format,
        };
        build_arch_params(weights, layer, wq, wk, wv, wo, gate, up, down)
    }).collect()
}
