//! WalkFfn — FFN backend that replaces dense matmul with vindex lookups.
//!
//! Sparse walk path (preferred):
//!   gate_knn (HNSW or brute) → K up dot products → GEGLU → K down accumulations
//!   No dense matmuls. Reads only K feature vectors from mmap.
//!
//! Fallback paths:
//!   exact: gate/up from model weights + down from mmap (3 dense matmuls)
//!   full_mmap: all three from mmap (3 dense matmuls)
//!   sparse_model: gate KNN + sparse gather from model weights

use ndarray::Array2;

use larql_compute::ComputeBackend;
use crate::ffn::FfnBackend;
use crate::ffn::sparse_compute::sparse_ffn_forward;
use crate::model::ModelWeights;

use larql_vindex::{GateIndex, WalkHit, WalkTrace};

pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn GateIndex,
    pub top_k: usize,
    pub backend: Option<&'a dyn ComputeBackend>,
    trace_residuals: std::cell::RefCell<Vec<(usize, Vec<f32>)>>,
    record_trace: bool,
}

impl<'a> WalkFfn<'a> {
    /// Create a WalkFfn with unlimited K (uses all features above activation threshold).
    /// The gate KNN returns all features; sparsity comes from the activation threshold.
    pub fn new(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        Self {
            weights, index, top_k, backend: None,
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: false,
        }
    }

    /// Create with unlimited K — no artificial cap on feature count.
    pub fn new_unlimited(weights: &'a ModelWeights, index: &'a dyn GateIndex) -> Self {
        Self::new(weights, index, usize::MAX)
    }

    pub fn new_with_backend(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        top_k: usize,
        backend: &'a dyn ComputeBackend,
    ) -> Self {
        Self {
            weights, index, top_k, backend: Some(backend),
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: false,
        }
    }

    /// Create with backend and unlimited K.
    pub fn new_unlimited_with_backend(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        backend: &'a dyn ComputeBackend,
    ) -> Self {
        Self::new_with_backend(weights, index, usize::MAX, backend)
    }

    pub fn new_with_trace(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        Self {
            weights, index, top_k, backend: None,
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: true,
        }
    }

    /// Unlimited top_k plus residual tracing. Used by `exec_infer`
    /// whenever a patched session has installed slots — bounded
    /// top_k drops features from the activation sum, which is
    /// harmless on a clean model (dropped features have tiny
    /// activations) but catastrophic once a strong (×30 gate scale)
    /// INSERT slot is in the mix: the slot's activation then
    /// dominates a half-weakened baseline and hijacks every prompt
    /// to whichever installed target has the largest lm_head
    /// alignment. Matching the dense FFN by processing every
    /// feature keeps the baseline intact and the installed slot
    /// proportional.
    pub fn new_unlimited_with_trace(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
    ) -> Self {
        Self::new_with_trace(weights, index, usize::MAX)
    }

    /// Take raw per-layer residuals (the exact vectors gate_knn sees during inference).
    /// These are the normalized post-attention hidden states at the last token position.
    pub fn take_residuals(&self) -> Vec<(usize, Vec<f32>)> {
        self.trace_residuals.borrow_mut().drain(..).collect()
    }

    pub fn take_trace(&self) -> WalkTrace {
        let residuals = self.trace_residuals.borrow_mut().drain(..).collect::<Vec<_>>();
        let mut layers = Vec::with_capacity(residuals.len());
        for (layer, residual) in residuals {
            let r = ndarray::Array1::from_vec(residual);
            let hits = self.index.gate_knn(layer, &r, self.top_k);
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.index.feature_meta(layer, feature)?.clone();
                    Some(WalkHit { layer, feature, gate_score, meta })
                })
                .collect();
            layers.push((layer, walk_hits));
        }
        WalkTrace { layers }
    }

    /// Sparse walk FFN: zero matrix multiplications.
    ///
    /// Per position:
    ///   1. gate_knn → top-K features with gate scores (HNSW graph search, no matmul)
    ///   2. For each feature: up_score = up_mmap[feat] · x  (dot product)
    ///   3. activation = silu(gate_score) * up_score          (GEGLU)
    ///   4. out += activation * down_mmap[feat]               (scaled vector add)
    ///
    /// Operations: K dot products + K scaled adds per position. No matmuls.
    fn walk_ffn_sparse(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let up_view = self.index.up_layer_matrix(layer)?;
        let down_view = self.index.down_layer_matrix(layer)?;

        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];
        let intermediate = self.index.num_features(layer);

        let arch = &*self.weights.arch;
        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        for s in 0..seq_len {
            let x_row = x.row(s);
            let x_owned = x_row.to_owned();

            // Gate: try fastest path available
            //   0. HNSW graph search (O(log N), ~0.1ms) if enabled
            //   1. gate_walk (brute-force BLAS gemv, ~5ms) if available
            //   2. Q4 gate KNN via compute backend (0.5ms Metal, 1ms CPU Q4)
            //   3. f32 brute-force BLAS (1.1ms) as fallback
            let hits = if self.index.is_hnsw_enabled() {
                self.index.gate_knn(layer, &x_owned, self.top_k)
            } else {
                self.index.gate_walk(layer, &x_owned, self.top_k)
                    .or_else(|| {
                        self.backend.and_then(|be|
                            self.index.gate_knn_q4(layer, &x_owned, self.top_k, be)
                        )
                    })
                    .unwrap_or_else(|| self.index.gate_knn(layer, &x_owned, self.top_k))
            };

            let mut out_row = out.row_mut(s);

            for (feat, gate_score) in hits {
                let act = if is_gated {
                    // Up: prefer the override slot (set by INSERT) before
                    // falling back to the mmap'd `up_features.bin` row.
                    // This is the parallel of the down_override path
                    // below — installing a fact rewrites all three
                    // FFN slot components (gate via overlay, up here,
                    // down via base.down_overrides) so the slot's
                    // activation reflects the constellation install
                    // instead of the original weak free-slot up vector.
                    let up_score = if let Some(up_ov) = self.index.up_override(layer, feat) {
                        if up_ov.len() == hidden {
                            let ov = ndarray::ArrayView1::from(up_ov);
                            ov.dot(&x_row)
                        } else {
                            up_view.row(feat).dot(&x_row)
                        }
                    } else {
                        up_view.row(feat).dot(&x_row)
                    };
                    let activated_gate = if use_gelu {
                        crate::ffn::gelu_tanh(gate_score)
                    } else {
                        gate_score * crate::ffn::sigmoid(gate_score)
                    };
                    activated_gate * up_score
                } else {
                    let mut v = gate_score;
                    if let Some(bias) = arch.ffn_up_bias_key(layer)
                        .and_then(|bk| self.weights.vectors.get(&bk))
                    {
                        if feat < bias.len() { v += bias[feat]; }
                    }
                    if use_gelu { crate::ffn::gelu_tanh(v) } else { v * crate::ffn::sigmoid(v) }
                };

                full_activation[[s, feat]] = act;

                if act.abs() > 1e-10 {
                    // Down: scaled vector add from mmap (not a matmul)
                    if let Some(override_down) = self.index.down_override(layer, feat) {
                        if override_down.len() == hidden {
                            let ov = ndarray::ArrayView1::from(override_down);
                            out_row.scaled_add(act, &ov);
                            continue;
                        }
                    }
                    let down_row = down_view.row(feat);
                    out_row.scaled_add(act, &down_row);
                }
            }
        }

        // Down bias
        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, full_activation))
    }

    /// Q4 interleaved walk: C kernel with vdotq_s32 for gate/up, scalar for down.
    /// Reads 44MB per layer instead of 315MB. Matches BLAS f32 speed on warm,
    /// faster on cold cache (7x less data to page in).
    fn walk_ffn_q4_interleaved(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        use larql_compute::cpu::ops::{q4_matvec, q4_vecmat};

        let q4_mmap = self.index.interleaved_q4_mmap_ref()?;
        let intermediate = self.index.num_features(layer);
        if intermediate == 0 { return None; }
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];

        let q4_bytes_per_matrix = intermediate * hidden / 32 * 18;
        let q4_bytes_per_layer = q4_bytes_per_matrix * 3;
        let layer_start = layer * q4_bytes_per_layer;

        let gate_q4 = &q4_mmap[layer_start..layer_start + q4_bytes_per_matrix];
        let up_q4 = &q4_mmap[layer_start + q4_bytes_per_matrix..layer_start + 2 * q4_bytes_per_matrix];
        let down_q4 = &q4_mmap[layer_start + 2 * q4_bytes_per_matrix..layer_start + 3 * q4_bytes_per_matrix];

        // Prefetch next layer
        self.index.prefetch_interleaved_q4_layer(layer + 1);

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        // Check for Metal Q4 backend
        let metal_q4 = self.backend.and_then(|be| if be.has_q4() { Some(be) } else { None });

        if let Some(be) = metal_q4 {
            // Metal: ONE GPU submission for all gate+up across ALL seq positions
            let x_flat = x.as_slice().unwrap();
            let (all_gate, all_up) = be.q4_matvec_pair_batch(
                gate_q4, up_q4, x_flat, seq_len, intermediate, hidden,
            ).unwrap();

            // GEGLU on CPU (element-wise, all positions)
            let mut all_activation: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            for s in 0..seq_len {
                let mut activation = vec![0.0f32; intermediate];
                for i in 0..intermediate {
                    let g = all_gate[s][i];
                    let u = all_up[s][i];
                    activation[i] = if use_gelu {
                        crate::ffn::gelu_tanh(g) * u
                    } else {
                        g * crate::ffn::sigmoid(g) * u
                    };
                    full_activation[[s, i]] = activation[i];
                }
                all_activation.push(activation);
            }

            // Down: one submission per position (GPU vecmat)
            for (s, activation_row) in all_activation.iter().enumerate().take(seq_len) {
                let down_result = be.q4_vecmat(activation_row, down_q4, intermediate, hidden).unwrap();
                let mut out_row = out.row_mut(s);
                for j in 0..hidden { out_row[j] = down_result[j]; }
            }
        } else {
            // C kernel path: vdotq for gate/up, scalar for down
            for s in 0..seq_len {
                let x_row = x.row(s);
                let x_slice = x_row.as_slice().unwrap();

                let gate_scores = q4_matvec::dispatch(gate_q4, x_slice, intermediate, hidden);
                let up_scores = q4_matvec::dispatch(up_q4, x_slice, intermediate, hidden);

                let mut activation = vec![0.0f32; intermediate];
                for i in 0..intermediate {
                    let g = gate_scores[i];
                    let u = up_scores[i];
                    activation[i] = if use_gelu {
                        crate::ffn::gelu_tanh(g) * u
                    } else {
                        g * crate::ffn::sigmoid(g) * u
                    };
                    full_activation[[s, i]] = activation[i];
                }

                let down_result = q4_vecmat::dispatch(&activation, down_q4, intermediate, hidden);
                let mut out_row = out.row_mut(s);
                for j in 0..hidden { out_row[j] = down_result[j]; }
            }
        }

        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, full_activation))
    }

    /// Walk-FFN on the f32 interleaved mmap (`interleaved.bin`), sparse
    /// gather via Metal. Option C: bit-exact with dense f32, reads only top-K
    /// rows per layer via the new `f32_sparse_matvec` / `f32_sparse_vecmat`
    /// Metal kernels. Bandwidth per decode ~680 MB vs dense's 10.6 GB.
    fn walk_ffn_f32_sparse(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let gate_view = self.index.interleaved_gate(layer)?;
        let up_view = self.index.interleaved_up(layer)?;
        let down_view = self.index.interleaved_down(layer)?;
        let up_flat = up_view.as_slice()?; // row-major [intermediate, hidden]
        let down_flat = down_view.as_slice()?;

        let intermediate = gate_view.shape()[0];
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];
        let backend = self.backend?;
        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // We dense-compute gate and up so activation[] has the same numeric
        // content as Q4_K walk / dense reference. The SPARSITY win comes from
        // the down projection only: instead of activation[intermediate] @
        // down[intermediate, hidden] reading 104 MB per layer, we sort
        // activation by magnitude, keep the top_k, and sparse-gather only
        // those rows of down. Bandwidth per layer drops from 3 × 104 MB
        // (dense f32 walk) to 2 × 104 MB + (top_k/intermediate) × 104 MB.
        // Output is bit-close to dense because the dropped contributions are
        // the smallest-magnitude activations.
        let top_k_cap = self.top_k.min(intermediate);
        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        for s in 0..seq_len {
            let x_row: Vec<f32> = x.row(s).to_vec();

            // 1. Dense gate matvec via Metal (matches reference exactly).
            let gate_scores_2d = larql_compute::dot_proj_gpu(
                &x.slice(ndarray::s![s..s+1, ..]), &gate_view, Some(backend),
            );
            let gate_scores: Vec<f32> = gate_scores_2d.row(0).to_vec();

            // 2. Dense up matvec via Metal.
            let up_scores_2d = larql_compute::dot_proj_gpu(
                &x.slice(ndarray::s![s..s+1, ..]), &up_view, Some(backend),
            );
            let up_scores: Vec<f32> = up_scores_2d.row(0).to_vec();

            // 3. GEGLU for all intermediate features.
            let mut activation = vec![0.0f32; intermediate];
            for i in 0..intermediate {
                let g = gate_scores[i];
                let u = up_scores[i];
                activation[i] = if use_gelu {
                    crate::ffn::gelu_tanh(g) * u
                } else {
                    g * crate::ffn::sigmoid(g) * u
                };
                full_activation[[s, i]] = activation[i];
            }

            // 4. Pick top_k features by activation magnitude. Linear scan
            //    with partial sort — fine for intermediate ~10 K.
            let mut idx_by_mag: Vec<(u32, f32)> = activation.iter().enumerate()
                .map(|(i, &a)| (i as u32, a.abs()))
                .collect();
            idx_by_mag.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx_by_mag.truncate(top_k_cap);
            let indices: Vec<u32> = idx_by_mag.iter().map(|&(i, _)| i).collect();
            let top_activations: Vec<f32> = indices.iter().map(|&i| activation[i as usize]).collect();

            // 5. Sparse down vecmat: only top_k rows contribute. Bandwidth ∝ K.
            let down_out = backend.f32_sparse_vecmat(down_flat, &top_activations, &indices, hidden)?;

            let mut out_row = out.row_mut(s);
            for j in 0..hidden { out_row[j] = down_out[j]; }
        }

        // Suppress unused-import warning for up_flat (kept in scope for future
        // sparse-up optimization when gate_knn is good enough to drive it).
        let _ = up_flat;

        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }
        Some((out, full_activation))
    }

    /// Walk-FFN on Q4_K interleaved data (148-byte Ollama layout from
    /// `interleaved_q4k_real.bin`). Dispatches the existing `q4k_matvec`
    /// Metal shader for gate+up per layer, then a CPU vec×mat for down
    /// (no Metal q4k_vecmat kernel exists yet — CPU fallback is acceptable
    /// since down is ~5% of per-layer cost at top_k=1024).
    ///
    /// Advantage over Q4_0 walk: per-sub-block scales + 6-bit mins give
    /// ~3× better dequant precision, reducing token drift on long generation.
    fn walk_ffn_q4k_interleaved(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let q4k_mmap = self.index.interleaved_q4k_real_mmap_ref()?;
        let intermediate = self.index.num_features(layer);
        if intermediate == 0 { return None; }
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];

        let q4k_bytes_per_matrix = intermediate * hidden / 256 * 148;
        let q4k_bytes_per_layer = q4k_bytes_per_matrix * 3;
        let layer_start = layer * q4k_bytes_per_layer;
        if layer_start + q4k_bytes_per_layer > q4k_mmap.len() { return None; }

        let gate = &q4k_mmap[layer_start..layer_start + q4k_bytes_per_matrix];
        let up = &q4k_mmap[layer_start + q4k_bytes_per_matrix
            ..layer_start + 2 * q4k_bytes_per_matrix];
        let down = &q4k_mmap[layer_start + 2 * q4k_bytes_per_matrix
            ..layer_start + 3 * q4k_bytes_per_matrix];

        let backend = self.backend?;
        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        // Idea 5: gate-KNN-predicted layer skip. If the layer's max|activation|
        // falls below LARQL_LAYER_SKIP_THRESHOLD, skip the down projection
        // entirely and return zeros. Tracing can be enabled separately to see
        // the distribution.
        let skip_threshold: f32 = std::env::var("LARQL_LAYER_SKIP_THRESHOLD")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let trace_skip = std::env::var("LARQL_LAYER_SKIP_TRACE")
            .ok().as_deref() == Some("1");
        // Experimental: truncate activation to top-K by magnitude, zeroing
        // smaller features. Forces the chained path so we can see/modify the
        // activation array before it goes into down. Set to 0 / unset to
        // disable. Use to find the correctness cliff: at what K does the
        // model break? Below the cliff, sparse-walk wins are unsafe.
        let topk_truncate: usize = std::env::var("LARQL_WALK_TOPK")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        // Sub-block-level truncation: pick top-S of total_subs (intermediate/32)
        // sub-blocks ranked by sum-of-|activation|, zero the rest. Element-level
        // top-K leaves activations scattered across all subs, so the down matvec
        // still reads every block. Sub-level top-S is what a real sparse Metal
        // kernel could actually skip. This cliff test tells us whether the
        // shader is worth building.
        let topk_subs: usize = std::env::var("LARQL_WALK_TOPK_SUBS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);

        let measure = crate::perf::is_enabled();

        for s in 0..seq_len {
            let x_row: Vec<f32> = x.row(s).to_vec();

            // S1/P11: try the fully-fused FFN first — one Metal command buffer
            // with three encoders (gate matvec, up matvec, GEGLU+down) all
            // sharing GPU buffers. Eliminates the gate/up readback + re-upload
            // and 1 wait_until_completed per layer. Validated bit-close to the
            // chained path by `q4k_ffn_full_matches_chained` cargo test.
            // Falls back to the chained gate+up + GEGLU+down path when
            // LARQL_LAYER_SKIP_THRESHOLD is set (needs intermediate gate/up
            // values for max_act computation) or backend lacks the full kernel.
            let activation_kind = if use_gelu { "gelu_tanh" } else { "silu" };
            let need_activation_array = skip_threshold > 0.0 || trace_skip
                || topk_truncate > 0 || topk_subs > 0;
            let mut down_result_opt: Option<Vec<f32>> = None;
            if !need_activation_array {
                let t_full = std::time::Instant::now();
                if let Some(d) = backend.q4k_ffn_full(
                    gate, up, down, &x_row,
                    hidden, intermediate, activation_kind,
                ) {
                    if measure { crate::perf::record("walk.ffn_full_fused", t_full.elapsed().as_micros()); }
                    down_result_opt = Some(d);
                }
            }

            // Chained fallback paths (P5 + P8b): used when full-fused isn't
            // available or layer-skip needs intermediate activations.
            let mut gate_scores_opt: Option<Vec<f32>> = None;
            let mut up_scores_opt: Option<Vec<f32>> = None;
            if down_result_opt.is_none() {
                let t_gu = std::time::Instant::now();
                let (gate_scores, up_scores) = match backend.q4k_matvec_pair(
                    gate, up, &x_row, intermediate, hidden,
                ) {
                    Some(pair) => pair,
                    None => (
                        backend.q4k_matvec(gate, &x_row, intermediate, hidden)?,
                        backend.q4k_matvec(up, &x_row, intermediate, hidden)?,
                    ),
                };
                if measure { crate::perf::record("walk.gate_up_dispatch", t_gu.elapsed().as_micros()); }
                gate_scores_opt = Some(gate_scores);
                up_scores_opt = Some(up_scores);

                if !need_activation_array {
                    let t_fused = std::time::Instant::now();
                    if let Some(d) = backend.q4k_geglu_down(
                        down, gate_scores_opt.as_ref().unwrap(), up_scores_opt.as_ref().unwrap(),
                        hidden, intermediate, activation_kind,
                    ) {
                        if measure { crate::perf::record("walk.geglu_down_fused", t_fused.elapsed().as_micros()); }
                        down_result_opt = Some(d);
                    }
                }
            }

            let down_result = match down_result_opt {
                Some(d) => d,
                None => {
                    // Both fused paths failed (or layer-skip was requested).
                    // gate_scores/up_scores are guaranteed populated by the
                    // chained-fallback block above when down_result_opt is None.
                    let gate_scores = gate_scores_opt.expect("gate_scores must be populated when fused path failed");
                    let up_scores = up_scores_opt.expect("up_scores must be populated when fused path failed");

                    let t_act = std::time::Instant::now();
                    let mut activation = vec![0.0f32; intermediate];
                    let mut max_act = 0.0f32;
                    for i in 0..intermediate {
                        let g = gate_scores[i];
                        let u = up_scores[i];
                        activation[i] = if use_gelu {
                            crate::ffn::gelu_tanh(g) * u
                        } else {
                            g * crate::ffn::sigmoid(g) * u
                        };
                        let a = activation[i].abs();
                        if a > max_act { max_act = a; }
                        full_activation[[s, i]] = activation[i];
                    }
                    if measure { crate::perf::record("walk.activation", t_act.elapsed().as_micros()); }

                    if trace_skip {
                        let max_gate = gate_scores.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                        eprintln!("[layer-skip L{layer:02} s={s}] max|gate|={max_gate:.3} max|act|={max_act:.3} skip={}",
                            if skip_threshold > 0.0 && max_act < skip_threshold { "yes" } else { "no" });
                    }
                    if skip_threshold > 0.0 && max_act < skip_threshold { continue; }

                    // P15 experiment: truncate activation to top-K by magnitude.
                    // Simulates "what if a graph index pre-selected only the top-K
                    // most relevant features?" Sweep LARQL_WALK_TOPK to find the
                    // correctness cliff. K=intermediate is a no-op identity.
                    if topk_truncate > 0 && topk_truncate < intermediate {
                        let mut idx: Vec<u32> = (0..intermediate as u32).collect();
                        idx.sort_unstable_by(|&a, &b| {
                            activation[b as usize].abs()
                                .partial_cmp(&activation[a as usize].abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        // Zero everything below top_k.
                        for &i in idx.iter().skip(topk_truncate) {
                            activation[i as usize] = 0.0;
                        }
                    }

                    // P16: sub-block-level top-S. Group activation into 32-element
                    // sub-blocks (matches the q4k_geglu_*_down inner loop), rank
                    // by sum-of-|act|, zero the bottom (total_subs - S) subs.
                    // This is the granularity a sparse-down Metal kernel could
                    // skip — element-level top-K can't, since avg occupancy
                    // per sub stays well above zero.
                    let total_subs = intermediate / 32;
                    if topk_subs > 0 && topk_subs < total_subs {
                        let mut sub_score: Vec<(usize, f32)> = (0..total_subs)
                            .map(|s| {
                                let base = s * 32;
                                let sum: f32 = (0..32).map(|i| activation[base + i].abs()).sum();
                                (s, sum)
                            })
                            .collect();
                        sub_score.sort_unstable_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        for &(s, _) in sub_score.iter().skip(topk_subs) {
                            let base = s * 32;
                            for i in 0..32 { activation[base + i] = 0.0; }
                        }
                    }

                    let t_down = std::time::Instant::now();
                    let r = backend.q4k_matvec(down, &activation, hidden, intermediate)?;
                    if measure { crate::perf::record("walk.down_dispatch", t_down.elapsed().as_micros()); }
                    r
                }
            };
            let mut out_row = out.row_mut(s);
            for j in 0..hidden { out_row[j] = down_result[j]; }
        }

        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, full_activation))
    }

    /// Interleaved walk: gate + up + down from one contiguous mmap per layer.
    /// Eliminates TLB thrash from 3 separate files. Prefetches next layer.
    fn walk_ffn_interleaved(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        // All three matrices from one contiguous region
        let gate_view = self.index.interleaved_gate(layer)?;
        let up_view = self.index.interleaved_up(layer)?;
        let down_view = self.index.interleaved_down(layer)?;

        // Prefetch next layer while we compute this one
        self.index.prefetch_interleaved_layer(layer + 1);

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // gate_scores = gate_vectors @ x^T (one BLAS gemv from contiguous region)
        let gate_scores = larql_compute::dot_proj_gpu(x, &gate_view, self.backend);

        // up_scores = x @ up_vectors^T (contiguous, right after gate in memory)
        let up_scores = larql_compute::dot_proj_gpu(x, &up_view, self.backend);

        // GEGLU
        let activation = if use_gelu {
            crate::ffn::gelu_tanh_gate_up(&gate_scores, &up_scores)
        } else {
            crate::ffn::silu_gate_up(&gate_scores, &up_scores)
        };

        // down: activation @ down_matrix (contiguous, right after up in memory)
        let mut out = larql_compute::matmul_gpu(&activation, &down_view, self.backend);

        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, activation))
    }

    /// Full mmap walk: gate + up + down all from mmap. Zero safetensor reads.
    /// Currently slower than exact path due to 3 separate mmap file reads.
    #[allow(dead_code)]
    ///
    /// gate_scores = gate_vectors @ x^T     (mmap, one BLAS gemm)
    /// up_scores   = up_vectors @ x^T       (mmap, one BLAS gemm)
    /// activation  = silu(gate) * up         (exact GEGLU)
    /// output      = activation @ down       (mmap, one BLAS gemm)
    ///
    /// Three mmap gemms. Same computation as dense. Zero model weight reads.
    fn walk_ffn_full_mmap(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let gate_scores = self.index.gate_scores_batch(layer, x)?;
        let up_view = self.index.up_layer_matrix(layer)?;
        let down_view = self.index.down_layer_matrix(layer)?;

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // up_scores = x @ up_vectors^T = [seq, intermediate]
        let up_scores = larql_compute::dot_proj_gpu(x, &up_view, self.backend);

        // GEGLU: silu(gate) * up  (exact, same as dense)
        let activation = if use_gelu {
            crate::ffn::gelu_tanh_gate_up(&gate_scores, &up_scores)
        } else {
            crate::ffn::silu_gate_up(&gate_scores, &up_scores)
        };

        // Down: activation @ down_matrix (mmap)
        let mut out = larql_compute::matmul_gpu(&activation, &down_view, self.backend);

        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, activation))
    }

    /// KNN-direct walk: gate scores as activations + down from mmap.
    /// NOTE: Produces wrong answer without up projection (tested: Jack instead of Paris).
    /// Kept for future research when combined gate+up vectors are available.
    #[allow(dead_code)]
    ///
    /// Gate KNN scores = x @ gate_vectors^T = the gate projection.
    /// Apply SiLU activation. Multiply by down matrix. Done.
    /// No gate matmul from model weights. No up matmul. No GEGLU.
    /// Two BLAS gemms: gate_knn + down. Reads 205MB instead of 315MB.
    fn walk_ffn_knn_direct(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let down_view = self.index.down_layer_matrix(layer)?;
        let gate_scores = self.index.gate_scores_batch(layer, x)?;

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // Gate scores → SiLU/GELU activation (no up projection)
        let activation = if use_gelu {
            gate_scores.mapv(crate::ffn::gelu_tanh)
        } else {
            gate_scores.mapv(|v| v * crate::ffn::sigmoid(v))
        };

        // activation[seq, intermediate] @ down[intermediate, hidden] → [seq, hidden]
        let mut out = larql_compute::matmul_gpu(&activation, &down_view, self.backend);

        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, activation))
    }

    /// Walk FFN: gate/up from model weights + down from mmap.
    ///
    /// Uses dense gate/up matmul (exact, sequential reads) and reads the down
    /// matrix directly from the feature-major mmap (zero-copy BLAS gemm).
    /// Total: gate(105MB) + up(105MB) + down_mmap(105MB) = 315MB.
    /// Same bandwidth as dense but down read is from mmap (potentially cached).
    fn walk_ffn_exact(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;

        // If FFN weights were dropped (walk-only mode), fall through to full mmap
        let w_up = match self.weights.tensors.get(&arch.ffn_up_key(layer)) {
            Some(w) => w,
            None => {
                // No model FFN weights — use full mmap path
                if let Some(result) = self.walk_ffn_full_mmap(layer, x) {
                    return result;
                }
                panic!("walk_ffn_exact: no FFN weights and no mmap data for layer {layer}");
            }
        };

        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // Gate + up + GEGLU: exact computation from model weights
        let activation = if is_gated {
            let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
            let gate = crate::forward::dot_proj(x, w_gate);
            let up = crate::forward::dot_proj(x, w_up);
            if use_gelu {
                crate::ffn::gelu_tanh_gate_up(&gate, &up)
            } else {
                crate::ffn::silu_gate_up(&gate, &up)
            }
        } else {
            let mut proj = crate::forward::dot_proj(x, w_up);
            if let Some(bias) = arch.ffn_up_bias_key(layer)
                .and_then(|bk| self.weights.vectors.get(&bk))
            {
                crate::forward::add_bias(&mut proj, bias);
            }
            if use_gelu {
                proj.mapv(crate::ffn::gelu_tanh)
            } else {
                proj.mapv(|v| v * crate::ffn::sigmoid(v))
            }
        };

        // Down: zero-copy BLAS gemm against mmap'd feature-major matrix
        let out = if let Some(down_view) = self.index.down_layer_matrix(layer) {
            // Zero-copy: mmap reinterpreted as ArrayView2, routed through compute backend
            larql_compute::matmul_gpu(&activation, &down_view, self.backend)
        } else {
            // Fallback: read W_down from model weights via compute backend
            let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
            larql_compute::dot_proj_gpu(&activation, w_down, self.backend)
        };

        let mut out = out;
        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        (out, activation)
    }
}

impl<'a> FfnBackend for WalkFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let num_features = self.index.num_features(layer);
        if num_features == 0 {
            let dense_ffn = crate::ffn::WeightFfn { weights: self.weights };
            return dense_ffn.forward_with_activation(layer, x);
        }

        // Record for deferred trace
        if self.record_trace {
            let seq_len = x.shape()[0];
            let last_row = x.row(seq_len - 1).to_vec();
            self.trace_residuals.borrow_mut().push((layer, last_row));
        }

        // Override-aware routing: when this layer has any patched
        // gate / up / down vectors (i.e. INSERT has touched it), force
        // the per-feature `walk_ffn_sparse` path. That path checks all
        // three override slots before falling back to the mmap'd row;
        // the BLAS / interleaved paths below operate on whole-layer
        // matrices and only have a partial post-hoc down-override
        // correction, which silently produces wrong activations for
        // overridden features. The sparse path is correct by
        // construction and the only path that respects up_override,
        // so anything with overrides goes here.
        if self.index.has_overrides_at(layer) {
            if let Some(result) = self.walk_ffn_sparse(layer, x) {
                return result;
            }
        }

        // LARQL_WALK_FORMAT lets a caller pin one branch for benchmarking:
        //   "q4_k" → Metal Q4_K walk (Option A) — 10.5 tok/s, 9.3 GB RSS
        //   "q4_0" → Metal Q4_0 walk (Option B) — 4.7 tok/s, 11.1 GB RSS
        //   "f32_sparse" → Metal f32 sparse gather (Option C proper) — bit-exact
        //                 with dense f32, reads only top-K rows per layer
        //   "f32"  → f32 interleaved walk (dense matmul over 10.2 GB mmap —
        //            correct but bandwidth-bound at 0.09 tok/s)
        //   unset  → priority chain (Q4_K first if available, then Q4_0, then f32)
        let forced = std::env::var("LARQL_WALK_FORMAT").ok();
        let want = forced.as_deref();

        // Option C: f32 sparse-gather Metal walk. Only fires when explicitly
        // requested, because the dense interleaved mmap (10.2 GB) needs to be
        // loaded but only top-K rows are *touched* per decode.
        if want == Some("f32_sparse")
            && self.index.has_interleaved()
            && self.backend.is_some()
        {
            if let Some(result) = self.walk_ffn_f32_sparse(layer, x) {
                return result;
            }
        }

        // Q4_K walk (Option A): higher precision than Q4_0, matches llama.cpp/Ollama
        // production quant strategy. Fires when the real Q4_K file is loaded AND
        // a Metal q4k_matvec kernel is available.
        if (want.is_none() || want == Some("q4_k"))
            && self.index.has_interleaved_q4k_real()
            && self.backend.is_some()
        {
            if let Some(result) = self.walk_ffn_q4k_interleaved(layer, x) {
                return result;
            }
        }

        // Q4 interleaved: preferred when GPU Q4 is available (Metal shader faster than BLAS).
        // CPU Q4 C kernel is slower than CPU BLAS at these dimensions — only use with GPU.
        if (want.is_none() || want == Some("q4_0"))
            && self.index.has_interleaved_q4()
            && self.backend.is_some_and(|be| be.has_q4())
        {
            if let Some(result) = self.walk_ffn_q4_interleaved(layer, x) {
                return result;
            }
        }

        // f32 interleaved: gate+up+down contiguous per layer.
        if (want.is_none() || want == Some("f32")) && self.index.has_interleaved() {
            if let Some(result) = self.walk_ffn_interleaved(layer, x) {
                return result;
            }
        }

        // Full mmap walk: gate + up + down from 3 separate mmap files.
        // At high K (>50% intermediate), uses full mmap matmuls.
        // At low K (<50%), uses per-feature sparse walk.
        //
        if self.index.has_full_mmap_ffn() {
            let intermediate = self.index.num_features(layer);
            if intermediate > 0 && self.top_k * 2 < intermediate {
                // Low K: per-feature sparse (no matmul, graph walk)
                if let Some(result) = self.walk_ffn_sparse(layer, x) {
                    return result;
                }
            } else {
                // High K: full mmap matmuls (production path)
                if let Some(mut result) = self.walk_ffn_full_mmap(layer, x) {
                    // Apply down overrides from INSERT as post-hoc corrections.
                    // For each overridden feature, subtract the model's down contribution
                    // and add the override's down contribution using the same activation.
                    if self.index.has_overrides_at(layer) {
                        let hidden = x.shape()[1];
                        let seq_len = x.shape()[0];
                        let (ref mut out, ref activation) = result;
                        if let Some(down_view) = self.index.down_layer_matrix(layer) {
                            for s in 0..seq_len {
                                let mut out_row = out.row_mut(s);
                                // Check each overridden feature
                                for feat in 0..intermediate {
                                    if let Some(override_down) = self.index.down_override(layer, feat) {
                                        if override_down.len() != hidden { continue; }
                                        let act = activation[[s, feat]];
                                        if act.abs() <= 1e-10 { continue; }
                                        // Subtract original down contribution
                                        let orig_down = down_view.row(feat);
                                        out_row.scaled_add(-act, &orig_down);
                                        // Add override down contribution
                                        let ov = ndarray::ArrayView1::from(override_down);
                                        out_row.scaled_add(act, &ov);
                                    }
                                }
                            }
                        }
                    }
                    return result;
                }
            }
        }

        // Fallback: partial mmap (gate/up from model weights + down from mmap)
        if self.index.has_down_features() {
            return self.walk_ffn_exact(layer, x);
        }

        // Gate KNN needed only for sparse fallback (no mmap down).
        // PatchedVindex::gate_knn_batch applies the gate overlay so any
        // installed slot lands in the candidate set even when its
        // original disk-side gate is weak.
        let features = self.index.gate_knn_batch(layer, x, self.top_k);

        // Fallback: sparse matmul against model weights.
        //
        // We always need gate-aware overrides on the patched session
        // because INSERT writes the strong gate / up / down trio into
        // the overlay. The dense gather above reads the original (weak)
        // free-slot gate / up at the installed feature, so the activation
        // would be tiny without the override-aware computation.
        // sparse_ffn_forward_with_full_overrides re-computes
        // `silu(gate_override · x) * (up_override · x)` for any slot
        // with an overlay entry, then applies the down override.
        let has_any_override = features.iter().any(|&f| {
            self.index.down_override(layer, f).is_some()
                || self.index.up_override(layer, f).is_some()
        }) || self.index.has_overrides_at(layer);

        if has_any_override {
            let slot_overrides: Vec<crate::ffn::FeatureSlotOverride<'_>> = features
                .iter()
                .map(|&f| crate::ffn::FeatureSlotOverride {
                    feature: f,
                    // gate override lives on the patched overlay, accessed
                    // via the new accessor on the GateIndex trait.
                    gate: self.index.gate_override(layer, f),
                    up: self.index.up_override(layer, f),
                    down: self.index.down_override(layer, f),
                })
                .filter(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some())
                .collect();
            crate::ffn::sparse_ffn_forward_with_full_overrides(
                self.weights, layer, x, &features, &slot_overrides,
            )
        } else {
            sparse_ffn_forward(self.weights, layer, x, &features)
        }
    }

    fn name(&self) -> &str {
        "walk"
    }
}

/// Q4_K vector × matrix for the walk-FFN down projection.
///
/// `down_q4k` is one layer's down matrix of shape `[intermediate, hidden]`
/// in 148-byte Ollama Q4_K blocks. `activation` is the GEGLU output of
/// length `intermediate`. Returns `out[hidden] = sum_i activation[i] * dequant(down_row_i)`.
///
/// Per-row dequant, scaled-accumulate. Skips rows with near-zero activation
/// (`|act| < 1e-10`) — cheap sparsity win when the GEGLU output is sparse.
/// No Metal Q4_K vec×mat shader exists yet; CPU fallback is acceptable
/// because down is ~5-10% of per-layer wall time once gate+up are on Metal.
fn q4k_vecmat_cpu(
    down_q4k: &[u8],
    activation: &[f32],
    intermediate: usize,
    hidden: usize,
) -> Vec<f32> {
    // Dequantize the entire matrix once. Costs `intermediate * hidden * 4` bytes
    // of f32 memory but sidesteps the subtle bugs we hit trying to dequant per-row
    // and per-block. Roundtrip correctness is established (Metal q4k_matvec
    // uses the same decoder semantics and produces right outputs for gate/up),
    // so calling dequant on the whole blob + `activation @ matrix` is safe.
    let full = larql_models::quant::ggml::dequantize_q4_k(down_q4k, intermediate * hidden)
        .expect("Q4_K matrix dequant");
    // full is row-major [intermediate, hidden]. out[j] = Σ_i activation[i] * full[i*hidden + j].
    let mut out = vec![0.0f32; hidden];
    for i in 0..intermediate {
        let act = activation[i];
        if act.abs() < 1e-10 { continue; }
        let row = &full[i * hidden..(i + 1) * hidden];
        for j in 0..hidden {
            out[j] += act * row[j];
        }
    }
    out
}
