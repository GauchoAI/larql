//! Model weight tensors — the loaded representation of a model's parameters.

use std::collections::HashMap;
use ndarray::ArcArray2;
use crate::ModelArchitecture;

/// Type alias for weight tensors — ArcArray2 supports both owned and shared storage.
/// Owned: from safetensors loading (heap). Shared: from mmap (zero-copy).
pub type WeightArray = ArcArray2<f32>;

/// A loaded model's weight tensors, configuration, and architecture.
pub struct ModelWeights {
    pub tensors: HashMap<String, WeightArray>,
    pub vectors: HashMap<String, Vec<f32>>,
    pub embed: WeightArray,
    /// Output projection matrix. Same as embed if tie_word_embeddings=true,
    /// separate lm_head.weight otherwise.
    pub lm_head: WeightArray,
    pub arch: Box<dyn ModelArchitecture>,
    // Cached from arch.config() for convenience — these are hot-path values.
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
}

impl ModelWeights {
    /// Drop FFN weight tensors (gate, up, down projections) from memory.
    /// After this, only attention, embedding, norm, and logits weights remain.
    /// Returns the number of bytes freed.
    ///
    /// Use when running walk-only mode — FFN is served from vindex mmap.
    /// Typical savings: ~13GB for a 4B model.
    pub fn drop_ffn_weights(&mut self) -> usize {
        let mut freed = 0usize;
        let ffn_patterns = ["gate_proj", "up_proj", "down_proj",
                           "ffn_gate", "ffn_up", "ffn_down",
                           "mlp.experts", "block_sparse_moe.experts",
                           "packed_gate_up_blocks", "packed_down_blocks"];
        let keys_to_remove: Vec<String> = self.tensors.keys()
            .filter(|k| ffn_patterns.iter().any(|p| k.contains(p)))
            .cloned()
            .collect();
        for key in &keys_to_remove {
            if let Some(arr) = self.tensors.remove(key) {
                freed += arr.len() * std::mem::size_of::<f32>();
            }
        }
        // Also drop FFN bias vectors
        let vec_keys: Vec<String> = self.vectors.keys()
            .filter(|k| ffn_patterns.iter().any(|p| k.contains(p)))
            .cloned()
            .collect();
        for key in &vec_keys {
            if let Some(v) = self.vectors.remove(key) {
                freed += v.len() * std::mem::size_of::<f32>();
            }
        }
        freed
    }

    /// Drop the eagerly-loaded f32 `lm_head` tensor. Inference through the
    /// vindex's `lm_head_knn_backend` (Q4 or f32 mmap) doesn't need this
    /// matrix — it's a leftover from safetensors load. Frees ~2.6 GB on
    /// Gemma 3 4B (vocab 262144 × hidden 2560 × 4 bytes).
    ///
    /// Safe to call when the caller guarantees that no code path will
    /// reach `weights.lm_head.row(...)` (i.e. no trace/capture flows, only
    /// `predict_honest_with_knn_ffn` / walk-only `ask`). Returns bytes freed.
    pub fn drop_lm_head_weight(&mut self) -> usize {
        let freed = self.lm_head.len() * std::mem::size_of::<f32>();
        // Replace with an empty ArcArray2 so the field type stays valid and
        // the original Arc refcount drops to zero, releasing the backing f32
        // buffer back to the allocator. A 0×hidden shape keeps metadata
        // consistent (hidden_size reads still work if anything accidentally
        // accesses this structure).
        self.lm_head = ndarray::Array2::<f32>::zeros((0, self.hidden_size)).into_shared();
        freed
    }

    /// Also drop the embed tensor when the caller knows the token embedding
    /// lookups won't fire (e.g. mid-stream test harnesses that start from
    /// a pre-embedded residual). For normal inference you DO need embed —
    /// don't call this from `ask`.
    pub fn drop_embed_weight(&mut self) -> usize {
        let freed = self.embed.len() * std::mem::size_of::<f32>();
        self.embed = ndarray::Array2::<f32>::zeros((0, self.hidden_size)).into_shared();
        freed
    }

    /// Drop attention Q/K/V/O projection tensors. Safe to call ONLY when the
    /// caller has a Q6_K or Q4_K mmap-backed attention path active (e.g.
    /// `run_attention_kv_cached_f32_opt` with LARQL_ATTN_Q6K=1). Frees
    /// ~1 GB on Gemma 3 4B. Returns bytes freed.
    pub fn drop_attn_weights(&mut self) -> usize {
        let mut freed = 0usize;
        let attn_patterns = ["q_proj", "k_proj", "v_proj", "o_proj",
                             "attn_q", "attn_k", "attn_v", "attn_output"];
        let keys_to_remove: Vec<String> = self.tensors.keys()
            .filter(|k| attn_patterns.iter().any(|p| k.contains(p)))
            .cloned()
            .collect();
        for key in &keys_to_remove {
            if let Some(arr) = self.tensors.remove(key) {
                freed += arr.len() * std::mem::size_of::<f32>();
            }
        }
        let vec_keys: Vec<String> = self.vectors.keys()
            .filter(|k| attn_patterns.iter().any(|p| k.contains(p)))
            .cloned()
            .collect();
        for key in &vec_keys {
            if let Some(v) = self.vectors.remove(key) {
                freed += v.len() * std::mem::size_of::<f32>();
            }
        }
        freed
    }
}
