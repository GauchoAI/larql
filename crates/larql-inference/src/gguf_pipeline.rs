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

    /// Apply the per-layer pre-FFN norm to a residual probe vector (for KNN).
    /// Mirrors what the residual goes through between attention and FFN.
    pub fn apply_pre_ffn_norm_for_layer(&self, residual: &[f32], layer: usize) -> Vec<f32> {
        let key = if self.arch.has_post_norms() {
            self.arch.pre_feedforward_layernorm_key(layer)
        } else {
            Some(self.arch.post_attention_layernorm_key(layer))
        };
        let weight = key.and_then(|k| self.vectors.get(&k));
        let n = residual.len();
        let eps = self.arch.norm_eps();
        let sum_sq: f32 = residual.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        let offset = self.norm_offset;
        match weight {
            Some(w) if w.len() == n => residual.iter().zip(w.iter())
                .map(|(&xi, &wi)| xi * inv_rms * (wi + offset)).collect(),
            _ => residual.iter().map(|&xi| xi * inv_rms).collect(),
        }
    }

    /// Result of `predict_top_k_with_knn`: top-k predictions plus an optional
    /// KNN override hit (cosine + entry data) when one fires.
    pub fn predict_top_k_with_knn(
        &self,
        token_ids: &[u32],
        top_k: usize,
        backend: &dyn ComputeBackend,
        tokenizer: &tokenizers::Tokenizer,
        knn_store: Option<&larql_vindex::KnnStore>,
    ) -> PredictResult {
        const KNN_COSINE_THRESHOLD: f32 = 0.75;
        let layers = self.build_layers();
        let cfg = self.arch.config();
        let max_inter = cfg.intermediate_size;
        let max_q_dim: usize = (0..cfg.num_layers)
            .map(|l| self.arch.num_q_heads_for_layer(l) * self.arch.head_dim_for_layer(l))
            .max().unwrap_or(0);
        let max_kv_dim: usize = (0..cfg.num_layers)
            .map(|l| self.arch.num_kv_heads_for_layer(l) * self.arch.head_dim_for_layer(l))
            .max().unwrap_or(0);

        // Probe at the first layer that has KNN entries (matches the existing
        // vindex-path behavior in predict_honest_with_knn_ffn).
        let knn_probe_layer: Option<usize> = knn_store
            .and_then(|s| s.layers().into_iter().next());

        let mut last_h: Option<Vec<f32>> = None;
        let mut last_probe: Option<Vec<f32>> = None;
        let last_idx = token_ids.len().saturating_sub(1);
        for (i, &tid) in token_ids.iter().enumerate() {
            let x = self.embed_row(tid);
            // Only probe on the LAST token (we want the residual at the
            // generation-position layer, not earlier prompt positions).
            let probe_for_this_token = if i == last_idx { knn_probe_layer } else { None };
            let (h, probe) = backend.decode_token_with_probe(
                &layers, &x, self.hidden(), max_inter, max_q_dim, max_kv_dim,
                cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
                self.arch.rope_base_for_layer(0) as f32,
                probe_for_this_token,
            ).expect("decode_token_with_probe returned None");
            last_h = Some(h);
            if i == last_idx { last_probe = probe; }
        }

        // KNN overlay check
        if let (Some(probe), Some(store), Some(pl)) = (last_probe.as_ref(), knn_store, knn_probe_layer) {
            let normed_probe = self.apply_pre_ffn_norm_for_layer(probe, pl);
            if let Some((entry, cosine)) = store.query_top1(pl, &normed_probe) {
                if std::env::var("LARQL_TRACE_KNN").ok().as_deref() == Some("1") {
                    eprintln!("[knn-gguf L{pl}] top1={} cos={cosine:.4} (threshold={KNN_COSINE_THRESHOLD})",
                        entry.target_token);
                }
                if cosine > KNN_COSINE_THRESHOLD {
                    let label = format!(
                        "{} (KNN override, cos={:.2}, L{})",
                        entry.target_token, cosine, pl
                    );
                    return PredictResult {
                        predictions: vec![(label, 1.0)],
                        knn_override: true,
                        knn_cosine: Some(cosine),
                    };
                }
            }
        }

        // Standard prediction: lm_head + softmax + top-k
        let h_out = last_h.expect("token_ids must be non-empty");
        let h_normed = self.apply_final_norm(&h_out);
        let logits = self.compute_logits(&h_normed, backend);
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = logits.iter().map(|v| (v - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 { for e in &mut exps { *e /= sum; } }
        let mut indexed: Vec<(usize, f32)> = exps.iter().enumerate()
            .map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);
        let predictions = indexed.into_iter().map(|(tid, p)| {
            let tok = tokenizer.decode(&[tid as u32], true).unwrap_or_default();
            (tok, p)
        }).collect();
        PredictResult { predictions, knn_override: false, knn_cosine: None }
    }

    /// Run the prompt through decode and capture the residual at `probe_layer`
    /// AT THE LAST TOKEN POSITION, with the same pre-FFN normalization that
    /// KNN queries use. Returns the f32 vector ready for `KnnStore::add`.
    ///
    /// Used by `/v1/insert mode=knn` to record the "fact key" for a prompt.
    pub fn capture_residual_at_layer(
        &self,
        token_ids: &[u32],
        probe_layer: usize,
        backend: &dyn ComputeBackend,
    ) -> Option<Vec<f32>> {
        let layers = self.build_layers();
        let cfg = self.arch.config();
        let max_inter = cfg.intermediate_size;
        let max_q_dim: usize = (0..cfg.num_layers)
            .map(|l| self.arch.num_q_heads_for_layer(l) * self.arch.head_dim_for_layer(l))
            .max().unwrap_or(0);
        let max_kv_dim: usize = (0..cfg.num_layers)
            .map(|l| self.arch.num_kv_heads_for_layer(l) * self.arch.head_dim_for_layer(l))
            .max().unwrap_or(0);

        let last_idx = token_ids.len().saturating_sub(1);
        let mut last_probe: Option<Vec<f32>> = None;
        for (i, &tid) in token_ids.iter().enumerate() {
            let x = self.embed_row(tid);
            let probe_for = if i == last_idx { Some(probe_layer) } else { None };
            let (_h, probe) = backend.decode_token_with_probe(
                &layers, &x, self.hidden(), max_inter, max_q_dim, max_kv_dim,
                cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
                self.arch.rope_base_for_layer(0) as f32,
                probe_for,
            )?;
            if i == last_idx { last_probe = probe; }
        }
        last_probe.map(|raw| self.apply_pre_ffn_norm_for_layer(&raw, probe_layer))
    }

    /// Convenience wrapper: same as `predict_top_k_with_knn` with no KNN store.
    pub fn predict_top_k(
        &self,
        token_ids: &[u32],
        top_k: usize,
        backend: &dyn ComputeBackend,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Vec<(String, f32)> {
        self.predict_top_k_with_knn(token_ids, top_k, backend, tokenizer, None).predictions
    }
}

/// Output of `predict_top_k_with_knn`: top predictions plus KNN-override
/// signal so the caller can surface `knn_override:true` in the response.
#[derive(Debug, Clone)]
pub struct PredictResult {
    pub predictions: Vec<(String, f32)>,
    pub knn_override: bool,
    pub knn_cosine: Option<f32>,
}

/// Holds per-call decode state across token steps in a streaming generate
/// loop. Owns the layer-build and dimension caches so we don't rebuild
/// `FullPipelineLayer` between every token.
pub struct DecodeSession<'a> {
    pipeline: &'a GgufPipeline,
    layers: Vec<FullPipelineLayer<'a>>,
    max_inter: usize,
    max_q_dim: usize,
    max_kv_dim: usize,
}

impl<'a> DecodeSession<'a> {
    pub fn new(pipeline: &'a GgufPipeline) -> Self {
        let cfg = pipeline.arch.config();
        let max_inter = cfg.intermediate_size;
        let max_q_dim: usize = (0..cfg.num_layers)
            .map(|l| pipeline.arch.num_q_heads_for_layer(l) * pipeline.arch.head_dim_for_layer(l))
            .max().unwrap_or(0);
        let max_kv_dim: usize = (0..cfg.num_layers)
            .map(|l| pipeline.arch.num_kv_heads_for_layer(l) * pipeline.arch.head_dim_for_layer(l))
            .max().unwrap_or(0);
        Self {
            layers: pipeline.build_layers(),
            pipeline,
            max_inter, max_q_dim, max_kv_dim,
        }
    }

    /// Feed one token through all layers, return the final hidden state
    /// (pre-final-norm) and an optional probe at `probe_layer`.
    pub fn step(
        &self,
        token_id: u32,
        probe_layer: Option<usize>,
        backend: &dyn ComputeBackend,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        let cfg = self.pipeline.arch.config();
        let x = self.pipeline.embed_row(token_id);
        backend.decode_token_with_probe(
            &self.layers, &x, self.pipeline.hidden(), self.max_inter,
            self.max_q_dim, self.max_kv_dim,
            cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
            self.pipeline.arch.rope_base_for_layer(0) as f32,
            probe_layer,
        ).expect("decode_token_with_probe returned None")
    }

    /// Compute logits (final-norm + lm_head) for the latest hidden state,
    /// then build a top-N sampler-compatible (token_id, logit, prob) list.
    /// `n_keep` typically 64 — large enough for top-p, small enough to be
    /// near-free vs O(vocab log vocab) full sort.
    pub fn finalize_for_sampler(
        &self,
        h_out: &[f32],
        n_keep: usize,
        backend: &dyn ComputeBackend,
    ) -> Vec<(u32, f32, f64)> {
        let h_normed = self.pipeline.apply_final_norm(h_out);
        let logits = self.pipeline.compute_logits(&h_normed, backend);
        // Partial selection: pick top n_keep by logit (O(n)), then sort those.
        let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
            .map(|(i, &l)| (i as u32, l)).collect();
        let n_keep = n_keep.min(indexed.len());
        if n_keep < indexed.len() {
            indexed.select_nth_unstable_by(n_keep, |a, b|
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(n_keep);
        }
        indexed.sort_by(|a, b|
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Compute probs over the kept candidates (numeric-stable softmax).
        let max_l = indexed.first().map(|(_, l)| *l).unwrap_or(0.0);
        let exps: Vec<f64> = indexed.iter()
            .map(|(_, l)| ((*l as f64) - max_l as f64).exp()).collect();
        let sum: f64 = exps.iter().sum();
        indexed.into_iter().zip(exps).map(|((tid, l), e)| {
            (tid, l, if sum > 0.0 { e / sum } else { 0.0 })
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
