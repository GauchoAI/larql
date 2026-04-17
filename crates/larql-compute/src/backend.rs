//! `ComputeBackend` trait — the single interface for all hardware backends.
//!
//! Callers use this trait exclusively. The implementation behind it can be
//! CPU BLAS, Metal GPU, CUDA, or anything else. The trait covers:
//!
//! - f32 matrix operations (matmul, matmul_transb, batch)
//! - Q4 quantized operations (matvec, vecmat, batched pairs)
//! - Metadata (name, capabilities)

use ndarray::{Array2, ArrayView2};

/// A single matmul operation for batch dispatch.
pub struct MatMulOp {
    pub a: Array2<f32>,
    pub b: Array2<f32>,
    pub transpose_b: bool,
}

/// Hardware compute backend.
///
/// Implementations provide f32 matmul and optionally Q4 quantized operations.
/// All methods accept `ArrayView2` (zero-copy borrowed views) to avoid
/// unnecessary data copies for mmap'd weight matrices.
pub trait ComputeBackend: Send + Sync {
    // ── f32 matrix operations ──

    /// C = A × B where A is [m, k] and B is [k, n].
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32>;

    /// C = A × B^T where A is [m, k] and B is [n, k].
    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32>;

    /// Multiple matmuls in one submission. Default: serial dispatch.
    /// GPU backends can override with parallel command buffer encoding.
    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter().map(|op| {
            if op.transpose_b {
                self.matmul_transb(op.a.view(), op.b.view())
            } else {
                self.matmul(op.a.view(), op.b.view())
            }
        }).collect()
    }

    // ── Q4 quantized operations (optional) ──

    /// Q4 matrix-vector: scores[N] = Q4[N,K] @ Q8_x[K].
    /// Returns None if backend doesn't support Q4.
    fn q4_matvec(
        &self,
        _q4_data: &[u8], _q8_x: &[i8], _q8_scales: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Q4 vector-matrix: out[K] = activation[N] @ Q4[N,K].
    fn q4_vecmat(
        &self,
        _activation: &[f32], _q4_data: &[u8],
        _intermediate: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// f32 sparse matvec (Option C): out[k] = Σ_h W[indices[k], h] * x[h].
    /// Reads only K rows of the [N, hidden] f32 matrix — bandwidth ∝ K.
    /// Used for walk-FFN up projection when the gate-KNN has narrowed to
    /// top-K features.
    fn f32_sparse_matvec(
        &self,
        _weights: &[f32], _x: &[f32], _indices: &[u32], _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// f32 sparse vecmat (Option C): out[h] = Σ_k activation[k] * W[indices[k], h].
    /// Used for walk-FFN down projection.
    fn f32_sparse_vecmat(
        &self,
        _weights: &[f32], _activation: &[f32], _indices: &[u32], _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Batched Q4 gate+up for all seq positions in one submission.
    #[allow(clippy::type_complexity)]
    fn q4_matvec_pair_batch(
        &self,
        _gate_q4: &[u8], _up_q4: &[u8],
        _x_matrix: &[f32], _seq_len: usize,
        _num_rows: usize, _hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> { None }

    /// Full pipeline: ALL Q4 (attention + FFN) in one command buffer for all layers.
    /// Each layer: Q4 Q/K/V proj → fused attention (RoPE+GQA+softcap) → Q4 O proj → Q4 FFN.
    /// No CPU-GPU round-trips between layers.
    #[allow(clippy::too_many_arguments)]
    fn full_pipeline_q4(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize, _inter: usize,
        _q_dim: usize, _kv_dim: usize,
        _seq_len: usize,
        _num_q_heads: usize, _num_kv_heads: usize, _head_dim: usize,
        _rope_base: f32, _use_qk_norm: bool, _softcap: f32,
    ) -> Option<Vec<f32>> { None }

    /// Multi-layer Q4 FFN in one submission: gate → up → GEGLU → down, chained.
    /// All layers processed in one command buffer — no CPU-GPU round-trips.
    /// Input: per-layer (gate_q4, up_q4, down_t_q4), initial residual x.
    /// Returns: final residual after all FFN layers.
    fn multi_layer_q4_ffn(
        &self,
        _layers_q4: &[(&[u8], &[u8], &[u8])],
        _x: &[f32],
        _inter: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Whether this backend supports KV cache decode operations.
    fn has_kv_cache(&self) -> bool { false }

    /// Populate KV cache with prefill K/V data for one layer.
    /// k_data/v_data: [seq_len, kv_dim] as flat f32.
    fn populate_kv_layer(
        &self, _layer: usize,
        _k_data: &[f32], _v_data: &[f32],
        _seq_len: usize, _num_kv_heads: usize, _head_dim: usize,
    ) { /* no-op for non-KV backends */ }

    /// Reset KV cache (for new prompt).
    fn reset_kv_cache(&self) {}

    /// Rollback KV cache: discard the last `n` tokens from all layers.
    /// Used by speculative decoding to undo rejected draft tokens.
    fn rollback_kv_cache(&self, _n: usize) {}

    /// Diagnostic: read back K and V cache contents for one layer.
    /// Returns (k_flat, v_flat, current_len) or None if unsupported.
    #[allow(clippy::type_complexity)]
    fn debug_read_kv_layer(&self, _layer: usize) -> Option<(Vec<f32>, Vec<f32>, usize)> { None }

    /// Decode one token through all layers with KV cache.
    /// Q8 attention + KV cache + Q4 FFN, one command buffer.
    #[allow(clippy::too_many_arguments)]
    fn decode_token(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize, _inter: usize,
        _q_dim: usize, _kv_dim: usize,
        _num_q_heads: usize, _num_kv_heads: usize, _head_dim: usize,
        _rope_base: f32,
    ) -> Option<Vec<f32>> { None }

    /// Like decode_token but also probes h_post_attn at a specific layer.
    /// Returns (final_h, Some(probe_h)) when probe_layer is set.
    /// The probe is a single residual_copy inside the same cmd buffer — zero
    /// pipeline break, ~0.01 ms cost. Used for KNN overlay checks: caller
    /// runs the full GPU pipeline, then checks KNN on the probed residual.
    /// If KNN matches, override the prediction. If not, use the GPU result.
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

    /// Q4_K matvec: scores[N] = Q4_K[N,K] @ f32_x[K]. Returns None if not supported.
    fn q4k_matvec(
        &self,
        _q4k_data: &[u8], _x: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Q6_K matvec: scores[N] = Q6_K[N,K] @ f32_x[K]. Returns None if not supported.
    fn q6k_matvec(
        &self,
        _q6k_data: &[u8], _x: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Two Q4_K matvecs sharing the same input x, dispatched in ONE Metal
    /// command buffer (one commit + one wait + two reads vs three of each).
    /// Used by walk-FFN to fuse gate + up projections per layer.
    /// Returns (a_scores[num_rows], b_scores[num_rows]).
    #[allow(clippy::too_many_arguments)]
    fn q4k_matvec_pair(
        &self,
        _q4k_a: &[u8], _q4k_b: &[u8],
        _x: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> { None }

    /// Three f32 matmul_transb operations sharing the same `a` (input),
    /// dispatched in ONE Metal command buffer. Used by attention to fuse
    /// Q + K + V projections at decode time. One upload of `a`, one commit,
    /// one wait, three reads. Falls through to None if not implemented.
    /// Each result is `a × b_i^T` where a:[m,k], b_i:[n_i,k] → out_i:[m,n_i].
    fn matmul_transb_triple_share_a(
        &self,
        _a: ArrayView2<f32>,
        _b_q: ArrayView2<f32>,
        _b_k: ArrayView2<f32>,
        _b_v: ArrayView2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>, Array2<f32>)> { None }

    /// Fused Q4_K GEGLU + down projection: computes
    ///   out[h] = Σ_i activation_kind(gate[i]) * up[i] * W_down[h, i]
    /// in a single Metal kernel. Eliminates the CPU activation loop and the
    /// gate/up readback that the separate q4k_matvec path requires.
    /// `activation` selects the activation function: `"silu"` (Llama family)
    /// or `"gelu_tanh"` (Gemma family). Returns None if backend doesn't have
    /// the fused kernel.
    #[allow(clippy::too_many_arguments)]
    fn q4k_geglu_down(
        &self,
        _down_q4k: &[u8],
        _gate: &[f32],
        _up: &[f32],
        _hidden: usize,
        _intermediate: usize,
        _activation: &str,
    ) -> Option<Vec<f32>> { None }

    /// **S1/P11**: Fully fused Q4_K FFN — gate + up + GEGLU + down in ONE
    /// Metal command buffer with three encoders sharing GPU buffers. Only
    /// `down_out` is read back to CPU; `gate_out` and `up_out` stay GPU-resident
    /// between encoders. Eliminates two readbacks + two re-uploads + one
    /// `wait_until_completed` per layer vs the q4k_matvec_pair + q4k_geglu_down
    /// chain.
    /// Returns the down projection output `out[hidden]` or None if unsupported.
    #[allow(clippy::too_many_arguments)]
    fn q4k_ffn_full(
        &self,
        _gate_q4k: &[u8],
        _up_q4k: &[u8],
        _down_q4k: &[u8],
        _x: &[f32],
        _hidden: usize,
        _intermediate: usize,
        _activation: &str,
    ) -> Option<Vec<f32>> { None }

    /// Metal fused multi-token attention: causal Q·K^T → softmax → @V with GQA.
    /// Caller has already applied RoPE and QK-norm on Q/K. `softcap = 0.0` disables
    /// softcap (Gemma 3). Input layouts (row-major):
    ///   q: [seq_len, num_q * head_dim]
    ///   k: [seq_len, num_kv * head_dim]
    ///   v: [seq_len, num_kv * head_dim]
    /// Returns out: [seq_len * num_q * head_dim] or None if unsupported.
    #[allow(clippy::too_many_arguments)]
    fn fused_attention_prefill(
        &self,
        _q: &[f32], _k: &[f32], _v: &[f32],
        _seq_len: usize,
        _num_q: usize, _num_kv: usize, _head_dim: usize,
        _scale: f32, _softcap: f32,
    ) -> Option<Vec<f32>> { None }

    /// Prefill: full pipeline for seq>1 with KV cache population.
    /// Runs Q4 attention + FFN for all layers, stores post-RoPE K/V in KV cache.
    /// Returns the final hidden state [seq_len * hidden] for all positions.
    #[allow(clippy::too_many_arguments)]
    fn prefill_q4(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize, _inter: usize,
        _q_dim: usize, _kv_dim: usize,
        _seq_len: usize,
        _num_q_heads: usize, _num_kv_heads: usize, _head_dim: usize,
        _rope_base: f32, _use_qk_norm: bool, _softcap: f32,
    ) -> Option<Vec<f32>> { None }

    /// Whether this backend supports Q4 fused operations.
    fn has_q4(&self) -> bool { false }

    // ── Metadata ──

    /// Human-readable backend name.
    fn name(&self) -> &str;

    /// Device info string (for logging/diagnostics).
    fn device_info(&self) -> String { self.name().to_string() }
}

// ── Helper functions for callers ──

/// dot_proj through a backend: a @ b^T.
/// If backend is None, falls back to ndarray BLAS (CPU).
pub fn dot_proj_gpu(
    a: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    b: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    match backend {
        Some(be) => be.matmul_transb(a.view(), b.view()),
        None => a.dot(&b.t()),
    }
}

/// matmul through a backend: a @ b (no transpose).
pub fn matmul_gpu(
    a: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    b: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    match backend {
        Some(be) => be.matmul(a.view(), b.view()),
        None => a.dot(b),
    }
}
