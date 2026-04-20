//! `ComputeBackend` trait — the surface used by the GGUF hot path.
//!
//! Three things only:
//!  - `decode_token_with_probe` — full forward pass through all layers,
//!     optional KNN probe at one layer.
//!  - `matvec_q8_0_gguf` — final lm_head dispatch (single Metal kernel).
//!  - `reset_kv_cache` — between requests.
//!
//! Everything else (matmul, q4_matvec, prefill, walk-FFN) was deleted with
//! the GGUF-only refactor; the inference and server crates never call them.

/// Hardware compute backend.
pub trait ComputeBackend: Send + Sync {
    /// Reset the KV cache (call between requests).
    fn reset_kv_cache(&self) {}

    /// Decode one token through all layers using the cached KV state.
    /// Optional `probe_layer` returns the post-attention residual at that
    /// layer (used for KNN overlay). Returns (final_h, optional_probe).
    #[allow(clippy::too_many_arguments)]
    fn decode_token_with_probe(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize, _inter: usize,
        _q_dim: usize, _kv_dim: usize,
        _num_q_heads: usize, _num_kv_heads: usize, _head_dim: usize,
        _rope_base: f32,
        _probe_layer: Option<usize>,
    ) -> Option<(Vec<f32>, Option<Vec<f32>>)> { None }

    /// Single-shot Q8_0Gguf matvec: y[n] = sum_k W[n,k] * x[k].
    /// Used by the GGUF lm_head matmul.
    fn matvec_q8_0_gguf(&self, _weight: &[u8], _x: &[f32], _n: usize, _k: usize)
        -> Option<Vec<f32>> { None }

    /// Human-readable backend name.
    fn name(&self) -> &str;

    /// Device info string for logging.
    fn device_info(&self) -> String { self.name().to_string() }
}
