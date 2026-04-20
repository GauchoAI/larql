//! GGUF inference pipeline — loads a GGUF file directly (no vindex),
//! exposes per-token embedding lookup, layer construction for `decode_token`,
//! and GPU lm_head matvec. Shared by both `examples/gguf_infer.rs` and
//! `larql-server` (when the vindex dir contains a `weights.gguf`).
//!
//! See the example for the seven debugging fixes that made this path work
//! (commits 0485c84, 6ebeb8a, bff503f, 6b84c3f, 3e2d081). When extending
//! this module, audit those commits before changing format routing,
//! norm-offset handling, dim interpretation, or RoPE scaling.

use std::collections::HashMap;
use std::path::Path;

use larql_compute::{
    Activation as ComputeActivation, ComputeBackend, FfnType as ComputeFfnType,
    FullPipelineLayer, NormType as ComputeNormType, QuantFormat, QuantWeight,
};
use larql_models::{
    GgufFile, GgufQuantizedData, ModelArchitecture, ModelError,
    detect_from_json,
    quant::ggml,
};

/// Owns everything needed to run inference from a GGUF file.
/// Layers (FullPipelineLayer) are built on demand and borrow from this struct.
pub struct GgufPipeline {
    pub gguf: GgufFile,
    pub qdata: GgufQuantizedData,
    pub arch: Box<dyn ModelArchitecture>,
    /// f32 norm weights and small tensors loaded eagerly.
    pub vectors: HashMap<String, Vec<f32>>,
    /// Embedding raw bytes (mmap'd quantized).
    pub embed_bytes: &'static [u8],
    pub embed_type: u32,
    pub embed_vocab: usize,
    pub embed_bytes_per_row: usize,
    pub embed_scale: f32,
    /// LM head: usually tied to embed; falls back to `output.weight`.
    pub lm_head_bytes: &'static [u8],
    pub lm_head_type: u32,
    pub lm_head_vocab: usize,
    pub lm_head_bytes_per_row: usize,
    /// Final RMSNorm weight key (e.g. "norm.weight").
    pub final_norm_key: String,
    pub norm_offset: f32,
    pub final_logit_softcap: f32,
}

// SAFETY: All fields are owned-and-immutable after construction. Architectures
// are stateless (just hold a ModelConfig); GgufFile/GgufQuantizedData wrap
// memory-mapped read-only data; the &'static slices into the mmap are valid
// for the life of the struct (qdata holds the mmap alive). Read-only sharing
// across threads is sound.
unsafe impl Send for GgufPipeline {}
unsafe impl Sync for GgufPipeline {}

impl GgufPipeline {
    /// Open a GGUF file and prepare for inference.
    pub fn open(path: &Path) -> Result<Self, ModelError> {
        let gguf = GgufFile::open(path)?;
        let config_json = gguf.to_config_json();
        let arch: Box<dyn ModelArchitecture> = detect_from_json(&config_json);
        let cfg = arch.config();
        let hidden = cfg.hidden_size;

        let (_tensors, vectors) = gguf.load_tensors_filtered(2_000_000)
            .map_err(|e| ModelError::Parse(format!("load_tensors_filtered: {e}")))?;

        let qdata = GgufQuantizedData::open(path, gguf.data_offset)?;

        // GGUF dims are inner-axis-first: dims=[hidden, vocab] for the
        // [vocab, hidden] embedding matrix. (See 6b84c3f / 6ebeb8a.)
        let embed_info = gguf.find_tensor("token_embd.weight")
            .ok_or_else(|| ModelError::Parse("missing token_embd.weight".into()))?;
        let embed_bytes_local: &[u8] = qdata.tensor_bytes(embed_info);
        assert_eq!(embed_info.dims[0] as usize, hidden,
            "GGUF embed dims[0] should be hidden, got {:?}", embed_info.dims);
        let embed_vocab = embed_info.dims[1] as usize;
        let embed_bytes_per_row = ggml::tensor_data_size(embed_info.tensor_type, hidden)?;
        let embed_type = embed_info.tensor_type;

        let (lm_head_bytes_local, lm_head_type, lm_head_vocab): (&[u8], u32, usize) =
            if let Some(info) = gguf.find_tensor("output.weight") {
                let bytes = qdata.tensor_bytes(info);
                assert_eq!(info.dims[0] as usize, hidden, "lm_head dims[0] should be hidden");
                (bytes, info.tensor_type, info.dims[1] as usize)
            } else {
                (embed_bytes_local, embed_type, embed_vocab)
            };
        let lm_head_bytes_per_row = ggml::tensor_data_size(lm_head_type, hidden)?;

        // SAFETY: `qdata` is moved into Self below; its mmap stays alive for
        // the life of the GgufPipeline. The byte slices point into that
        // mmap and never move. Read-only access only.
        let embed_bytes_static: &'static [u8] = unsafe {
            std::slice::from_raw_parts(embed_bytes_local.as_ptr(), embed_bytes_local.len())
        };
        let lm_head_bytes_static: &'static [u8] = unsafe {
            std::slice::from_raw_parts(lm_head_bytes_local.as_ptr(), lm_head_bytes_local.len())
        };

        let final_norm_key = arch.final_norm_key().to_string();
        // Gemma3 GGUF norms have +1 baked in (convert_hf_to_gguf adds 1.0).
        let norm_offset = if arch.family() == "gemma3" { 0.0 } else { arch.norm_weight_offset() };
        let final_logit_softcap = cfg.final_logit_softcapping.unwrap_or(0.0) as f32;
        let embed_scale = arch.embed_scale();

        Ok(Self {
            gguf,
            qdata,
            arch,
            vectors,
            embed_bytes: embed_bytes_static,
            embed_type,
            embed_vocab,
            embed_bytes_per_row,
            embed_scale,
            lm_head_bytes: lm_head_bytes_static,
            lm_head_type,
            lm_head_vocab,
            lm_head_bytes_per_row,
            final_norm_key,
            norm_offset,
            final_logit_softcap,
        })
    }

    pub fn hidden(&self) -> usize { self.arch.config().hidden_size }
    pub fn intermediate(&self) -> usize { self.arch.config().intermediate_size }
    pub fn num_layers(&self) -> usize { self.arch.config().num_layers }
    pub fn num_q_heads(&self) -> usize { self.arch.config().num_q_heads }
    pub fn num_kv_heads(&self) -> usize { self.arch.config().num_kv_heads }
    pub fn head_dim(&self) -> usize { self.arch.config().head_dim }

    /// Dequantize one token's embedding row (contiguous `hidden` elements).
    pub fn embed_row(&self, token_id: u32) -> Vec<f32> {
        let offset = token_id as usize * self.embed_bytes_per_row;
        let row_bytes = &self.embed_bytes[offset..offset + self.embed_bytes_per_row];
        let mut row = ggml::dequantize(row_bytes, self.embed_type, self.hidden()).unwrap();
        for v in &mut row { *v *= self.embed_scale; }
        row
    }

    /// Final RMSNorm weight (None if missing).
    pub fn final_norm(&self) -> Option<&Vec<f32>> {
        self.vectors.get(&self.final_norm_key)
    }

    /// Apply final RMSNorm to the last hidden state.
    pub fn apply_final_norm(&self, x: &[f32]) -> Vec<f32> {
        let eps = self.arch.norm_eps();
        let n = x.len();
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        let offset = self.norm_offset;
        match self.final_norm() {
            Some(w) if w.len() == n => x.iter().zip(w.iter())
                .map(|(&xi, &wi)| xi * inv_rms * (wi + offset)).collect(),
            _ => x.iter().map(|&xi| xi * inv_rms).collect(),
        }
    }

    /// Compute full vocab logits via Q8_0 GPU matvec when supported,
    /// otherwise CPU dequant + dot.
    pub fn compute_logits(&self, h: &[f32], backend: &dyn ComputeBackend) -> Vec<f32> {
        let logits = if self.lm_head_type == ggml::TYPE_Q8_0 {
            backend.matvec_q8_0_gguf(self.lm_head_bytes, h, self.lm_head_vocab, self.hidden())
        } else { None };
        let mut logits = logits.unwrap_or_else(|| {
            let mut out = vec![0.0f32; self.lm_head_vocab];
            let chunk = 256;
            for start in (0..self.lm_head_vocab).step_by(chunk) {
                let end = (start + chunk).min(self.lm_head_vocab);
                let chunk_bytes = &self.lm_head_bytes[start * self.lm_head_bytes_per_row .. end * self.lm_head_bytes_per_row];
                let chunk_f32 = ggml::dequantize(chunk_bytes, self.lm_head_type, (end - start) * self.hidden()).unwrap();
                for i in 0..(end - start) {
                    let row = &chunk_f32[i * self.hidden()..(i + 1) * self.hidden()];
                    out[start + i] = row.iter().zip(h.iter()).map(|(a, b)| a * b).sum();
                }
            }
            out
        });
        if self.final_logit_softcap > 0.0 {
            let cap = self.final_logit_softcap;
            for v in &mut logits { *v = cap * (*v / cap).tanh(); }
        }
        logits
    }

    /// Build the per-layer pipeline. Returned vector borrows from `self`.
    /// Callers should hold the borrow for the duration of a decode pass.
    pub fn build_layers(&self) -> Vec<FullPipelineLayer<'_>> {
        let arch = &*self.arch;
        let num_layers = arch.config().num_layers;
        (0..num_layers).map(|l| build_one_layer(&self.gguf, &self.qdata, &self.vectors, arch, l)).collect()
    }

    /// Run prefill + decode for one token's worth of generation: feed all
    /// `token_ids` through `decode_token`, return top-k predictions for the
    /// next token plus the final hidden state at the probe layer (if any).
    ///
    /// Caller holds the inference lock and resets the KV cache before calling.
    pub fn predict_top_k(
        &self,
        token_ids: &[u32],
        top_k: usize,
        backend: &dyn ComputeBackend,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Vec<(String, f32)> {
        let layers = self.build_layers();
        let cfg = self.arch.config();
        let max_inter = cfg.intermediate_size;
        let max_q_dim: usize = (0..cfg.num_layers)
            .map(|l| self.arch.num_q_heads_for_layer(l) * self.arch.head_dim_for_layer(l))
            .max().unwrap_or(0);
        let max_kv_dim: usize = (0..cfg.num_layers)
            .map(|l| self.arch.num_kv_heads_for_layer(l) * self.arch.head_dim_for_layer(l))
            .max().unwrap_or(0);

        let mut last_h: Option<Vec<f32>> = None;
        for &tid in token_ids {
            let x = self.embed_row(tid);
            let h = backend.decode_token(
                &layers, &x, self.hidden(), max_inter, max_q_dim, max_kv_dim,
                cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
                self.arch.rope_base_for_layer(0) as f32,
            ).expect("decode_token returned None");
            last_h = Some(h);
        }
        let h_out = last_h.expect("token_ids must be non-empty");
        let h_normed = self.apply_final_norm(&h_out);
        let logits = self.compute_logits(&h_normed, backend);

        // Softmax → probabilities, take top_k. Numeric-stable max-subtract.
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = logits.iter().map(|v| (v - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 { for e in &mut exps { *e /= sum; } }

        let mut indexed: Vec<(usize, f32)> = exps.iter().enumerate()
            .map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);
        indexed.into_iter().map(|(tid, p)| {
            let tok = tokenizer.decode(&[tid as u32], true).unwrap_or_default();
            (tok, p)
        }).collect()
    }
}

fn lookup_norm_vec<'a>(vectors: &'a HashMap<String, Vec<f32>>, key: &str) -> &'a [f32] {
    vectors.get(key).map(|v| v.as_slice()).unwrap_or(&[])
}

fn gguf_tensor_type_to_quant_format(tensor_type: u32) -> QuantFormat {
    match tensor_type {
        ggml::TYPE_Q4_0 => QuantFormat::Q4_0,
        ggml::TYPE_Q8_0 => QuantFormat::Q8_0Gguf,
        ggml::TYPE_Q4_K => QuantFormat::Q4_KF,
        ggml::TYPE_Q6_K => QuantFormat::Q6_K,
        ggml::TYPE_Q5_0 => QuantFormat::Q8_0Gguf, // fallback (no Metal Q5_0 kernel)
        ggml::TYPE_Q5_K => QuantFormat::Q6_K,
        _ => QuantFormat::Q4_KF,
    }
}

fn build_one_layer<'a>(
    gguf: &'a GgufFile,
    qdata: &'a GgufQuantizedData,
    vectors: &'a HashMap<String, Vec<f32>>,
    arch: &dyn ModelArchitecture,
    l: usize,
) -> FullPipelineLayer<'a> {
    let wq_info = gguf.find_tensor(&format!("blk.{l}.attn_q.weight"))
        .unwrap_or_else(|| panic!("missing blk.{l}.attn_q.weight"));
    let wq_data = qdata.tensor_bytes(wq_info);
    let wq_format = gguf_tensor_type_to_quant_format(wq_info.tensor_type);

    let wk_info = gguf.find_tensor(&format!("blk.{l}.attn_k.weight"))
        .unwrap_or_else(|| panic!("missing blk.{l}.attn_k.weight"));
    let wk_data = qdata.tensor_bytes(wk_info);
    let wk_format = gguf_tensor_type_to_quant_format(wk_info.tensor_type);

    let (wv_data, wv_format) = match gguf.find_tensor(&format!("blk.{l}.attn_v.weight")) {
        Some(info) => (qdata.tensor_bytes(info), gguf_tensor_type_to_quant_format(info.tensor_type)),
        None => (wk_data, wk_format),
    };

    let wo_info = gguf.find_tensor(&format!("blk.{l}.attn_output.weight"))
        .unwrap_or_else(|| panic!("missing blk.{l}.attn_output.weight"));
    let wo_data = qdata.tensor_bytes(wo_info);
    let wo_format = gguf_tensor_type_to_quant_format(wo_info.tensor_type);

    let empty_qw = QuantWeight { data: &[], scales: None, format: QuantFormat::Q4_K };
    let gate = gguf.find_tensor(&format!("blk.{l}.ffn_gate.weight"))
        .map(|info| QuantWeight { data: qdata.tensor_bytes(info), scales: None, format: gguf_tensor_type_to_quant_format(info.tensor_type) })
        .unwrap_or(empty_qw);
    let up = gguf.find_tensor(&format!("blk.{l}.ffn_up.weight"))
        .map(|info| QuantWeight { data: qdata.tensor_bytes(info), scales: None, format: gguf_tensor_type_to_quant_format(info.tensor_type) })
        .unwrap_or(empty_qw);
    let down = gguf.find_tensor(&format!("blk.{l}.ffn_down.weight"))
        .map(|info| QuantWeight { data: qdata.tensor_bytes(info), scales: None, format: gguf_tensor_type_to_quant_format(info.tensor_type) })
        .unwrap_or(empty_qw);

    let input_norm = lookup_norm_vec(vectors, &arch.input_layernorm_key(l));
    let post_attn_norm = lookup_norm_vec(vectors, &arch.post_attention_layernorm_key(l));
    let pre_ffn_norm = arch.pre_feedforward_layernorm_key(l)
        .and_then(|k| vectors.get(&k)).map(|v| v.as_slice());
    let post_ffn_norm = arch.post_feedforward_layernorm_key(l)
        .and_then(|k| vectors.get(&k)).map(|v| v.as_slice());
    let q_norm_weight = arch.attn_q_norm_key(l)
        .and_then(|k| vectors.get(&k)).map(|v| v.as_slice());
    let k_norm_weight = arch.attn_k_norm_key(l)
        .and_then(|k| vectors.get(&k)).map(|v| v.as_slice());

    let head_dim = arch.head_dim_for_layer(l);
    let num_q_heads = arch.num_q_heads_for_layer(l);
    let num_kv_heads = arch.num_kv_heads_for_layer(l);
    let rotary_frac = arch.rotary_fraction_for_layer(l);
    let rotary_dim = if rotary_frac >= 1.0 { 0 } else { (head_dim as f64 * rotary_frac) as usize };
    let sliding_window = if arch.is_sliding_window_layer(l) {
        arch.sliding_window_size().unwrap_or(0)
    } else { 0 };

    let layer_scalar = arch.layer_scalar_key(l)
        .and_then(|k| vectors.get(&k).or_else(|| vectors.get(&format!("{k}.weight"))))
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0);

    // MoE bits (None for dense Gemma 3)
    let router_weight = gguf.find_tensor(&format!("blk.{l}.ffn_gate_inp.weight"))
        .and_then(|info| qdata.tensor_f32(info));
    let expert_gu_info = gguf.find_tensor(&format!("blk.{l}.ffn_gate_up_exps.weight"));
    let expert_gate_up = expert_gu_info.map(|info| QuantWeight {
        data: qdata.tensor_bytes(info), scales: None,
        format: gguf_tensor_type_to_quant_format(info.tensor_type),
    });
    let expert_down_info = gguf.find_tensor(&format!("blk.{l}.ffn_down_exps.weight"));
    let expert_down = expert_down_info.map(|info| QuantWeight {
        data: qdata.tensor_bytes(info), scales: None,
        format: gguf_tensor_type_to_quant_format(info.tensor_type),
    });
    let has_moe = router_weight.is_some() && expert_gate_up.is_some();
    let num_experts = if has_moe { arch.num_experts() } else { 0 };
    let num_active_experts = if has_moe { arch.num_experts_per_token() } else { 0 };
    let expert_inter = expert_gu_info
        .map(|info| if info.dims.len() >= 2 { info.dims[1] as usize / 2 } else { 0 })
        .unwrap_or(0);

    let norm_offset = if arch.family() == "gemma3" { 0.0 } else { arch.norm_weight_offset() };
    let qk_norm_offset = if arch.family() == "gemma3" { 0.0 } else { arch.qk_norm_weight_offset() };

    FullPipelineLayer {
        wq: QuantWeight { data: wq_data, scales: None, format: wq_format },
        wk: QuantWeight { data: wk_data, scales: None, format: wk_format },
        wv: QuantWeight { data: wv_data, scales: None, format: wv_format },
        wo: QuantWeight { data: wo_data, scales: None, format: wo_format },
        gate, up, down,
        input_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm,
        q_norm_weight, k_norm_weight,
        input_norm_bias: None, post_attn_norm_bias: None,
        norm_offset, qk_norm_offset,
        eps: arch.norm_eps(),
        has_post_norms: arch.has_post_norms(),
        norm_type: match arch.norm_type() {
            larql_models::NormType::LayerNorm => ComputeNormType::LayerNorm,
            _ => ComputeNormType::RmsNorm,
        },
        ffn_type: match arch.ffn_type() {
            larql_models::FfnType::Standard => ComputeFfnType::Standard,
            _ => ComputeFfnType::Gated,
        },
        activation: match arch.activation() {
            larql_models::Activation::GeluTanh => ComputeActivation::GeluTanh,
            _ => ComputeActivation::Silu,
        },
        attn_scale: arch.attention_scale_for_layer(l) as f32,
        head_dim, num_q_heads, num_kv_heads,
        rope_base: arch.rope_base_for_layer(l) as f32,
        rope_freq_scale: arch.rope_freq_scale_for_layer(l) as f32,
        rotary_dim, sliding_window,
        has_v_norm: arch.has_v_norm(),
        layer_scalar,
        softcap: arch.attn_logit_softcapping().unwrap_or(0.0),
        ffn_up_bias: None, ffn_down_bias: None,
        router_weight, expert_gate_up, expert_down,
        expert_down_scale: None,
        is_moe_layer: has_moe,
        num_experts, num_active_experts,
        expert_intermediate: expert_inter,
    }
}
