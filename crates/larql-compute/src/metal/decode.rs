use super::*;

impl MetalBackend {
    /// Create a KV cache for decode mode.
    pub fn create_kv_cache(&self, num_layers: usize, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new(&self.bufs, num_layers, max_seq, num_kv_heads, head_dim)
    }

    /// Decode one token through all layers with KV cache.
    ///
    /// **Single command buffer**, one encoder per layer, no explicit barriers
    /// (Apple Silicon serialises compute dispatches within an encoder).
    ///
    /// Per-layer pipeline (~10 dispatches):
    ///   1. Input norm
    ///   2. Fused QKV projection (Q4_K or Q4_KF)
    ///   3. Batched RoPE (all Q heads), batched RoPE (all K heads)
    ///   4. Batched V-norm (optional, Gemma 4)
    ///   5. KV cache append + KV attend
    ///   6. O projection
    ///   7. Residual + norm (f32 for Q4_K/Q4_KF, +Q8 for Q4_0)
    ///   8. FFN: fused gate+up (Q4_K) or separate gate/up (Q4_KF) + GEGLU + down
    ///   9. Post-FFN residual + optional layer scalar
    ///
    /// Format-aware FFN routing:
    ///   - Q4_KF: llama.cpp-exact kernel (q4kf_proj) with register-cached input
    ///   - Q4_K:  fused gate+up kernel + q4k_matvec (uint4, 8 rows/TG, nr0=2)
    ///   - Q4_0:  legacy Q8-input path
    #[allow(clippy::too_many_arguments)]
    /// Decode one token, optionally probing h_post_attn at a specific layer.
    ///
    /// When `probe_layer` is Some(L), copies `h_post_attn` at layer L into a
    /// side buffer and returns it alongside the final output. This lets the
    /// caller run KNN overlay checks on the probed residual WITHOUT breaking
    /// the GPU pipeline — the entire 34-layer decode stays in one cmd buffer.
    ///
    /// Returns (final_h, Option<probe_h>).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token_with_probe(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        probe_layer: Option<usize>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        let result = self.decode_token_inner(kv_cache, layers, x, hidden, inter,
            q_dim, kv_dim, _num_q_heads, _num_kv_heads, _head_dim, _rope_base,
            probe_layer);
        result
    }

    pub fn decode_token(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
    ) -> Vec<f32> {
        self.decode_token_inner(kv_cache, layers, x, hidden, inter,
            q_dim, kv_dim, _num_q_heads, _num_kv_heads, _head_dim, _rope_base,
            None).0
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_token_inner(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        probe_layer: Option<usize>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        // If any layer is MoE, use the per-layer command buffer path
        // (router readback requires cmd buffer split mid-layer).
        let has_moe = layers.iter().any(|l| l.is_moe_layer);
        if has_moe {
            return self.decode_token_inner_moe(
                kv_cache, layers, x, hidden, inter,
                q_dim, kv_dim, _num_q_heads, _num_kv_heads, _head_dim, _rope_base,
                probe_layer,
            );
        }

        let num_layers = layers.len();
        let hidden_val = hidden as u32;
        let inter_val = inter as u32;

        // Pre-cache weight buffers
        let wq_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wq.data)).collect();
        let wk_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wk.data)).collect();
        let wv_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wv.data)).collect();
        let wo_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wo.data)).collect();
        let wq_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wq.scales.unwrap_or(&[]))).collect();
        let wk_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wk.scales.unwrap_or(&[]))).collect();
        let wv_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wv.scales.unwrap_or(&[]))).collect();
        let wo_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wo.scales.unwrap_or(&[]))).collect();
        let gate_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.gate.data)).collect();
        let up_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.up.data)).collect();
        let down_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.down.data)).collect();
        let input_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.input_norm)).collect();
        let post_attn_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.post_attn_norm)).collect();
        // Optional per-head QK-norm weights (Gemma 3 / Gemma 4). Zero-length slice if absent.
        let q_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.q_norm_weight.unwrap_or(&[]))).collect();
        let k_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.k_norm_weight.unwrap_or(&[]))).collect();

        // Two h buffers for ping-pong: even layers write to h_a, odd to h_b.
        let h_init = self.bufs.transient_from_f32(x);
        let h_a = self.bufs.output((hidden * 4) as u64);
        let h_b = self.bufs.output((hidden * 4) as u64);
        let mut h_buf = &h_init;

        // Pre-allocate scratch buffers reused across layers.
        // GPU processes layers sequentially within one cmd buffer, so
        // these buffers are never read and written simultaneously.
        let q_out = self.bufs.output((q_dim * 4) as u64);
        let k_out = self.bufs.output((kv_dim * 4) as u64);
        let v_out = self.bufs.output((kv_dim * 4) as u64);
        let norm_f32_buf = self.bufs.output((hidden * 4) as u64);
        let attn_out_buf = self.bufs.output((q_dim * 4) as u64);
        let o_out_buf = self.bufs.output((hidden * 4) as u64);
        let h_post_attn = self.bufs.output((hidden * 4) as u64);
        // Probe buffer: if probe_layer is set, we copy h_post_attn here at that layer.
        let probe_buf = if probe_layer.is_some() {
            Some(self.bufs.output((hidden * 4) as u64))
        } else {
            None
        };
        let ffn_norm_out = self.bufs.output((hidden * 4) as u64);
        let ffn_q8 = self.bufs.output(hidden as u64);
        let ffn_q8s = self.bufs.output((hidden / 32 * 4) as u64);
        let up_out = self.bufs.output((inter * 4) as u64);
        let act_buf = self.bufs.output((inter * 4) as u64);
        let down_out = self.bufs.output((hidden * 4) as u64);
        let gate_out_scratch = self.bufs.output((inter * 4) as u64);
        // new_h is ping-ponged via h_a/h_b above
        let normed_scratch = self.bufs.output((hidden * 4) as u64);
        let o_q8_scratch = self.bufs.output(q_dim as u64);
        let o_q8s_scratch = self.bufs.output((q_dim / 32 * 4) as u64);
        let scaled_scratch = self.bufs.output((hidden * 4) as u64);

        // Single command buffer + single encoder for ALL layers.
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        for l in 0..num_layers {
            let layer = &layers[l];
            let norm_offset = layer.norm_offset;
            let eps = layer.eps;
            let scale = layer.attn_scale;
            let layer_head_dim = layer.head_dim;
            let layer_num_q_heads = layer.num_q_heads;
            let layer_num_kv_heads = layer.num_kv_heads;
            let layer_rope_base = layer.rope_base;
            let layer_rotary_dim = if layer.rotary_dim > 0 { layer.rotary_dim } else { layer_head_dim };
            let uses_q4k = layer.wq.format == crate::QuantFormat::Q4_K
                || layer.wq.format == crate::QuantFormat::Q6_K
                || layer.wq.format == crate::QuantFormat::Q4_KF
                || layer.wq.format == crate::QuantFormat::Q8_0Gguf;
            let layer_q_dim = layer_num_q_heads * layer_head_dim;
            let _layer_kv_dim = layer_num_kv_heads * layer_head_dim;
            let window_size = layer.sliding_window as u32;

            // ── Step 1: Input norm + Q/K/V projection ──
            if uses_q4k {
                let all_same_format = layer.wq.format == layer.wk.format && layer.wk.format == layer.wv.format;

                // Fused norm+QKV: skip separate norm dispatch, each simdgroup
                // computes RMS inline. Saves 1 dispatch per layer.
                if all_same_format && layer.wq.format == crate::QuantFormat::Q4_K
                    && layer.norm_type != crate::NormType::LayerNorm
                {
                    let total_rows = (q_dim + kv_dim + kv_dim) as u32;
                    let q_rows_val = q_dim as u32;
                    let k_rows_val = kv_dim as u32;
                    let v_rows_val = kv_dim as u32;
                    let k_val = hidden as u32;
                    let rows_per_tg = crate::metal::shaders::q4k_qkv_proj::ROWS_PER_TG;
                    let num_tgs = (total_rows as u64).div_ceil(rows_per_tg);
                    enc.set_compute_pipeline_state(&self.q4k_norm_qkv_pipeline);
                    enc.set_buffer(0, Some(&wq_bufs[l]), 0);
                    enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                    enc.set_buffer(2, Some(&wv_bufs[l]), 0);
                    enc.set_buffer(3, Some(&h_buf), 0);  // raw h, not normed
                    enc.set_buffer(4, Some(&input_norm_bufs[l]), 0);  // norm weights
                    enc.set_buffer(5, Some(&q_out), 0);
                    enc.set_buffer(6, Some(&k_out), 0);
                    enc.set_buffer(7, Some(&v_out), 0);
                    enc.set_bytes(8, 4, &q_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(9, 4, &k_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(10, 4, &v_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(11, 4, &k_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(12, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(13, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    let threads_per_tg = crate::metal::shaders::q4k_qkv_proj::THREADS_PER_TG;
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs, 1, 1),
                        MTLSize::new(threads_per_tg, 1, 1),
                    );
                } else {
                // Fallback: separate norm + QKV (LayerNorm, Q4_KF, mixed formats)
                use crate::metal::ops::full_pipeline::encode_rms_norm;
                if layer.norm_type == crate::NormType::LayerNorm {
                    let len_val = hidden as u32;
                    if let Some(bias) = layer.input_norm_bias {
                        let bias_buf = self.bufs.transient_from_f32(bias);
                        enc.set_compute_pipeline_state(&self.layer_norm_pipeline);
                        enc.set_buffer(0, Some(&h_buf), 0);
                        enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                        enc.set_buffer(2, Some(&bias_buf), 0);
                        enc.set_buffer(3, Some(&norm_f32_buf), 0);
                        enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                        enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    } else {
                        enc.set_compute_pipeline_state(&self.layer_norm_no_bias_pipeline);
                        enc.set_buffer(0, Some(&h_buf), 0);
                        enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                        enc.set_buffer(2, Some(&norm_f32_buf), 0);
                        enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    }
                    enc.dispatch_threads(
                        MTLSize::new(hidden as u64, 1, 1),
                        MTLSize::new(256.min(hidden as u64), 1, 1),
                    );
                } else {
                    encode_rms_norm(enc, &self.rms_norm_pipeline,
                        &h_buf, &input_norm_bufs[l], &norm_f32_buf,
                        hidden, eps, norm_offset);
                }

                let all_same_format = layer.wq.format == layer.wk.format && layer.wk.format == layer.wv.format;
                if all_same_format && layer.wq.format != crate::QuantFormat::Q6_K && layer.wq.format != crate::QuantFormat::Q8_0Gguf {
                    let total_rows = (q_dim + kv_dim + kv_dim) as u32;
                    let q_rows_val = q_dim as u32;
                    let k_rows_val = kv_dim as u32;
                    let v_rows_val = kv_dim as u32;
                    let k_val = hidden as u32;
                    let (qkv_pipeline, rows_per_tg) = if layer.wq.format == crate::QuantFormat::Q4_KF {
                        (&self.q4kf_qkv_proj_pipeline, crate::metal::shaders::q4kf_qkv_proj::ROWS_PER_TG)
                    } else {
                        (&self.q4k_qkv_proj_pipeline, crate::metal::shaders::q4k_qkv_proj::ROWS_PER_TG)
                    };
                    let num_tgs = (total_rows as u64).div_ceil(rows_per_tg);
                    enc.set_compute_pipeline_state(qkv_pipeline);
                    enc.set_buffer(0, Some(&wq_bufs[l]), 0);
                    enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                    enc.set_buffer(2, Some(&wv_bufs[l]), 0);
                    enc.set_buffer(3, Some(&norm_f32_buf), 0);
                    enc.set_buffer(4, Some(&q_out), 0);
                    enc.set_buffer(5, Some(&k_out), 0);
                    enc.set_buffer(6, Some(&v_out), 0);
                    enc.set_bytes(7, 4, &q_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &k_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(9, 4, &v_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(10, 4, &k_val as *const u32 as *const std::ffi::c_void);
                    let threads_per_tg = if layer.wq.format == crate::QuantFormat::Q4_KF {
                        crate::metal::shaders::q4kf_qkv_proj::THREADS_PER_TG
                    } else {
                        crate::metal::shaders::q4k_qkv_proj::THREADS_PER_TG
                    };
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs, 1, 1),
                        MTLSize::new(threads_per_tg, 1, 1),
                    );
                } else {
                    // Mixed formats: dispatch each projection separately.
                    // This handles Q4_K Q/K + Q6_K V (Ollama strategy).
                    let k_val = hidden as u32;

                    // Helper: dispatch one projection with format-appropriate shader
                    fn encode_single_proj(
                        enc: &metal::ComputeCommandEncoderRef,
                        w_buf: &metal::Buffer, x_buf: &metal::Buffer, out_buf: &metal::Buffer,
                        rows: usize, k: u32, format: crate::QuantFormat,
                        q4k_pipeline: &metal::ComputePipelineState,
                        q4kf_pipeline: &metal::ComputePipelineState,
                        q6k_pipeline: &metal::ComputePipelineState,
                        q8_gguf_pipeline: &metal::ComputePipelineState,
                    ) {
                        match format {
                            crate::QuantFormat::Q8_0Gguf => {
                                use crate::metal::shaders::q8_0_gguf_matvec as q8gm;
                                let n = rows as u32;
                                let num_tgs = (rows as u64).div_ceil(q8gm::ROWS_PER_TG);
                                enc.set_compute_pipeline_state(q8_gguf_pipeline);
                                enc.set_buffer(0, Some(w_buf), 0);
                                enc.set_buffer(1, Some(x_buf), 0);
                                enc.set_buffer(2, Some(out_buf), 0);
                                enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
                                enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
                                enc.dispatch_thread_groups(
                                    MTLSize::new(num_tgs, 1, 1),
                                    MTLSize::new(q8gm::THREADS_PER_TG, 1, 1),
                                );
                            }
                            crate::QuantFormat::Q6_K => {
                                use crate::metal::shaders::q6k_matvec as q6k;
                                let n = rows as u32;
                                let num_tgs = (rows as u64).div_ceil(q6k::ROWS_PER_TG);
                                enc.set_compute_pipeline_state(q6k_pipeline);
                                enc.set_buffer(0, Some(w_buf), 0);
                                enc.set_buffer(1, Some(x_buf), 0);
                                enc.set_buffer(2, Some(out_buf), 0);
                                enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
                                enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
                                enc.dispatch_thread_groups(
                                    MTLSize::new(num_tgs, 1, 1),
                                    MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
                                );
                            }
                            crate::QuantFormat::Q4_KF => {
                                use crate::metal::shaders::q4kf_qkv_proj as proj_sh;
                                let n = rows as u32;
                                let num_tgs = (rows as u64).div_ceil(proj_sh::ROWS_PER_TG);
                                enc.set_compute_pipeline_state(q4kf_pipeline);
                                enc.set_buffer(0, Some(w_buf), 0);
                                enc.set_buffer(1, Some(x_buf), 0);
                                enc.set_buffer(2, Some(out_buf), 0);
                                enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
                                enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
                                enc.dispatch_thread_groups(
                                    MTLSize::new(num_tgs, 1, 1),
                                    MTLSize::new(proj_sh::THREADS_PER_TG, 1, 1),
                                );
                            }
                            _ => {
                                // Q4_K standard
                                use crate::metal::shaders::q4k_matvec as q4k;
                                let n = rows as u32;
                                let num_tgs = (rows as u64).div_ceil(q4k::ROWS_PER_TG);
                                enc.set_compute_pipeline_state(q4k_pipeline);
                                enc.set_buffer(0, Some(w_buf), 0);
                                enc.set_buffer(1, Some(x_buf), 0);
                                enc.set_buffer(2, Some(out_buf), 0);
                                enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
                                enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
                                enc.dispatch_thread_groups(
                                    MTLSize::new(num_tgs, 1, 1),
                                    MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
                                );
                            }
                        }
                    }

                    encode_single_proj(enc, &wq_bufs[l], &norm_f32_buf, &q_out,
                        q_dim, k_val, layer.wq.format,
                        &self.q4k_matvec_pipeline, &self.q4kf_proj_pipeline, &self.q6k_matvec_pipeline, &self.q8_0_gguf_matvec_pipeline);
                    encode_single_proj(enc, &wk_bufs[l], &norm_f32_buf, &k_out,
                        kv_dim, k_val, layer.wk.format,
                        &self.q4k_matvec_pipeline, &self.q4kf_proj_pipeline, &self.q6k_matvec_pipeline, &self.q8_0_gguf_matvec_pipeline);
                    encode_single_proj(enc, &wv_bufs[l], &norm_f32_buf, &v_out,
                        kv_dim, k_val, layer.wv.format,
                        &self.q4k_matvec_pipeline, &self.q4kf_proj_pipeline, &self.q6k_matvec_pipeline, &self.q8_0_gguf_matvec_pipeline);
                }
                } // end fallback (non-fused norm+QKV)
            } else {
                // Q8 path: norm+Q8 → Q8 QKV (reuse ffn_q8/q8s scratch)
                let q8_buf = &ffn_q8;
                let q8s_buf = &ffn_q8s;

                enc.set_compute_pipeline_state(&self.rms_norm_q8_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                enc.set_buffer(2, Some(&q8_buf), 0);
                enc.set_buffer(3, Some(&q8s_buf), 0);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));

                let total_rows = (q_dim + kv_dim + kv_dim) as u32;
                let q_rows = q_dim as u32;
                let k_rows = kv_dim as u32;
                let v_rows = kv_dim as u32;
                let k_val = hidden as u32;
                enc.set_compute_pipeline_state(&self.q8_qkv_proj_pipeline);
                enc.set_buffer(0, Some(&wq_bufs[l]), 0);
                enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                enc.set_buffer(2, Some(&wv_bufs[l]), 0);
                enc.set_buffer(3, Some(&q8_buf), 0);
                enc.set_buffer(4, Some(&wq_scale_bufs[l]), 0);
                enc.set_buffer(5, Some(&wk_scale_bufs[l]), 0);
                enc.set_buffer(6, Some(&wv_scale_bufs[l]), 0);
                enc.set_buffer(7, Some(&q8s_buf), 0);
                enc.set_buffer(8, Some(&q_out), 0);
                enc.set_buffer(9, Some(&k_out), 0);
                enc.set_buffer(10, Some(&v_out), 0);
                enc.set_bytes(11, 4, &q_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(12, 4, &k_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(13, 4, &v_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(14, 4, &k_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new((total_rows as u64).div_ceil(8), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }

            // ── Step 1.5: QK-norm (Gemma 3 / Gemma 4) ──
            // Per-head RMSNorm applied to Q and K between QKV projection and RoPE.
            // Gemma 3 stores learned per-head norm weights in q_norm.weight / k_norm.weight
            // with a "+1.0" offset convention (qk_norm_offset = 1.0). Without this step the
            // residual's large norm-weight multipliers leave Q and K un-scaled, and softmax
            // overflows to inf for any reasonable KV-cache length.
            if layer.q_norm_weight.is_some() {
                // Fused multi-head QK-norm: 2 dispatches instead of 12.
                // Each dispatch normalizes all heads in parallel (one TG per head).
                let hd_val = layer_head_dim as u32;
                let eps_val = layer.eps;
                let off_val = layer.qk_norm_offset;
                let tg_threads = 256u64.min(layer_head_dim as u64);

                enc.set_compute_pipeline_state(&self.rms_norm_multihead_pipeline);
                enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &eps_val as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &off_val as *const f32 as *const std::ffi::c_void);
                // Q heads — all in one dispatch
                enc.set_buffer(0, Some(&q_out), 0);
                enc.set_buffer(1, Some(&q_norm_bufs[l]), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(layer_num_q_heads as u64, 1, 1),
                    MTLSize::new(tg_threads, 1, 1),
                );
                // K heads — all in one dispatch
                enc.set_buffer(0, Some(&k_out), 0);
                enc.set_buffer(1, Some(&k_norm_bufs[l]), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(layer_num_kv_heads as u64, 1, 1),
                    MTLSize::new(tg_threads, 1, 1),
                );
            }

            // ── Step 2: RoPE on Q and K heads (batched — one dispatch each) ──
            {
                let pos = kv_cache.layers[l].current_len as u32;
                let hd = layer_head_dim as u32;
                let rdim = layer_rotary_dim as u32;
                let rope_pairs = (layer_rotary_dim / 2) as u64;
                let num_q = layer_num_q_heads as u32;
                let num_kv = layer_num_kv_heads as u32;
                let freq_scale = layer.rope_freq_scale;

                // Q heads — all in one dispatch
                enc.set_compute_pipeline_state(&self.rope_at_pos_batched_pipeline);
                enc.set_buffer(0, Some(&q_out), 0);
                enc.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(2, 4, &layer_rope_base as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &pos as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &rdim as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &num_q as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &freq_scale as *const f32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(rope_pairs, layer_num_q_heads as u64, 1),
                    MTLSize::new(rope_pairs.min(256), 1, 1),
                );

                // K heads — all in one dispatch
                enc.set_buffer(0, Some(&k_out), 0);
                enc.set_bytes(5, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(rope_pairs, layer_num_kv_heads as u64, 1),
                    MTLSize::new(rope_pairs.min(256), 1, 1),
                );
            }

            // ── Step 3: V-norm batched (optional, Gemma 4) ──
            if layer.has_v_norm {
                let hd_val = layer_head_dim as u32;
                let num_kv = layer_num_kv_heads as u32;
                enc.set_compute_pipeline_state(&self.v_norm_batched_pipeline);
                enc.set_buffer(0, Some(&v_out), 0);
                enc.set_buffer(1, Some(&v_out), 0);
                enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(layer_head_dim as u64, layer_num_kv_heads as u64, 1),
                    MTLSize::new((layer_head_dim as u64).min(256), 1, 1),
                );
            }

            // No explicit barriers — Apple Silicon executes compute dispatches
            // within a single encoder in submission order. Verified by tests.

            let attn_out = &attn_out_buf;
            ops::kv_cache::encode_kv_append(
                enc, &kv_cache.layers[l],
                &self.kv_append_pipeline, &k_out, &v_out,
            );
            // Pick fast (4KB tg, high occupancy) or long (32KB tg) based on T
            let t_after = kv_cache.layers[l].current_len + 1;
            let kv_attend = if t_after <= 1024 {
                &self.kv_attend_fast_pipeline
            } else {
                &self.kv_attend_long_pipeline
            };
            ops::kv_cache::encode_kv_attend_softcap(
                enc, &kv_cache.layers[l],
                kv_attend, &q_out, &attn_out,
                layer_num_q_heads, scale, window_size, layer.softcap,
            );
            kv_cache.layers[l].current_len += 1;


            // Scratch buffers pre-allocated above — reused each layer.
            let new_h = if l % 2 == 0 { &h_a } else { &h_b };
            {
                if uses_q4k {
                    // Pipeline choice decides ROWS/THREADS per TG: q4k_proj uses 8/256,
                    // q4kf_proj uses 4/64, q6k_matvec uses 4/128. Previously both
                    // Q4_K/Q4_KF paths used q4kf dims, causing q4k_proj to dispatch
                    // only 2 simdgroups per TG (expected 8) — 75% of output rows
                    // never written. Q6_K path now explicitly supported for O proj
                    // too (makes attention-all-Q6K layouts work for Gemma 3).
                    let o_rows = hidden as u32;
                    let o_k = layer_q_dim as u32;
                    let (o_pipeline, rows_per_tg, threads_per_tg) = match layer.wo.format {
                        crate::QuantFormat::Q4_KF => {
                            use crate::metal::shaders::q4kf_qkv_proj as kf;
                            (&self.q4kf_proj_pipeline, kf::ROWS_PER_TG, kf::THREADS_PER_TG)
                        }
                        crate::QuantFormat::Q6_K => {
                            use crate::metal::shaders::q6k_matvec as q6k;
                            (&self.q6k_matvec_pipeline, q6k::ROWS_PER_TG, q6k::THREADS_PER_TG)
                        }
                        crate::QuantFormat::Q8_0Gguf => {
                            use crate::metal::shaders::q8_0_gguf_matvec as q8gm;
                            (&self.q8_0_gguf_matvec_pipeline, q8gm::ROWS_PER_TG, q8gm::THREADS_PER_TG)
                        }
                        _ => {
                            use crate::metal::shaders::q4k_qkv_proj as k;
                            (&self.q4k_proj_pipeline, k::ROWS_PER_TG, k::THREADS_PER_TG)
                        }
                    };
                    let num_tgs = (hidden as u64).div_ceil(rows_per_tg);
                    enc.set_compute_pipeline_state(o_pipeline);
                    enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                    enc.set_buffer(1, Some(&attn_out), 0);
                    enc.set_buffer(2, Some(&o_out_buf), 0);
                    enc.set_bytes(3, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &o_k as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs, 1, 1),
                        MTLSize::new(threads_per_tg, 1, 1),
                    );
                } else {
                    let o_q8 = &o_q8_scratch;
                    let o_q8s = &o_q8s_scratch;
                    let dim_val = layer_q_dim as u32;
                    let blocks = (layer_q_dim / 32) as u32;
                    enc.set_compute_pipeline_state(&self.q8_quant_pipeline);
                    enc.set_buffer(0, Some(&attn_out), 0);
                    enc.set_buffer(1, Some(&o_q8), 0);
                    enc.set_buffer(2, Some(&o_q8s), 0);
                    enc.set_bytes(3, 4, &dim_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(blocks as u64, 1, 1), MTLSize::new(256.min(blocks as u64), 1, 1));

                    let o_rows = hidden as u32;
                    let o_k = layer_q_dim as u32;
                    enc.set_compute_pipeline_state(&self.q8_matvec_pipeline);
                    enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                    enc.set_buffer(1, Some(&o_q8), 0);
                    enc.set_buffer(2, Some(&wo_scale_bufs[l]), 0);
                    enc.set_buffer(3, Some(&o_q8s), 0);
                    enc.set_buffer(4, Some(&o_out_buf), 0);
                    enc.set_bytes(5, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &o_k as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new((hidden as u64).div_ceil(8), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                }
            }

            // ── Step 5: Residual + norm (format-aware: Q4_K skips Q8 quantize) ──
            let ffn_uses_q4k = layer.gate.format == crate::QuantFormat::Q4_K
                || layer.gate.format == crate::QuantFormat::Q4_KF
                || layer.gate.format == crate::QuantFormat::Q6_K
                || layer.gate.format == crate::QuantFormat::Q8_0Gguf;
            // ffn_norm_out pre-allocated above

            let has_post_norms = layer.has_post_norms;
            if has_post_norms {
                let normed_o = &normed_scratch;
                {
                    use crate::metal::ops::full_pipeline::encode_rms_norm;
                    encode_rms_norm(enc, &self.rms_norm_pipeline,
                        &o_out_buf, &post_attn_norm_bufs[l], &normed_o, hidden, eps, norm_offset);
                }
                let pre_ffn_buf = if let Some(pfn) = layer.pre_ffn_norm {
                    self.bufs.transient_from_f32(pfn)
                } else {
                    post_attn_norm_bufs[l].clone()
                };
                if ffn_uses_q4k {
                    // Q4_K path: residual+norm → f32 output (no Q8)
                    enc.set_compute_pipeline_state(&self.residual_norm_pipeline);
                    enc.set_buffer(0, Some(&h_buf), 0);
                    enc.set_buffer(1, Some(&normed_o), 0);
                    enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(3, Some(&ffn_norm_out), 0);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                    // h_post_attn = h + normed_o (residual_norm also writes this to buffer 3? No — residual_norm only outputs normed.
                    // We need the pre-norm residual for the post-FFN add. Use residual_add separately.
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    encode_residual_add(enc, &self.residual_add_pipeline,
                        &h_buf, &normed_o, &h_post_attn, hidden);
                } else {
                    enc.set_compute_pipeline_state(&self.residual_norm_q8_pipeline);
                    enc.set_buffer(0, Some(&h_buf), 0);
                    enc.set_buffer(1, Some(&normed_o), 0);
                    enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(3, Some(&ffn_q8), 0);
                    enc.set_buffer(4, Some(&ffn_q8s), 0);
                    enc.set_buffer(5, Some(&h_post_attn), 0);
                    enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                }
            } else if ffn_uses_q4k {
                // Q4_K path: residual+norm → f32 output (no Q8)
                enc.set_compute_pipeline_state(&self.residual_norm_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&o_out_buf), 0);
                enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0);
                enc.set_buffer(3, Some(&ffn_norm_out), 0);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                // h_post_attn = h + o (pre-norm residual for post-FFN add)
                use crate::metal::ops::full_pipeline::encode_residual_add;
                encode_residual_add(enc, &self.residual_add_pipeline,
                    &h_buf, &o_out_buf, &h_post_attn, hidden);
            } else {
                enc.set_compute_pipeline_state(&self.residual_norm_q8_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&o_out_buf), 0);
                enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0);
                enc.set_buffer(3, Some(&ffn_q8), 0);
                enc.set_buffer(4, Some(&ffn_q8s), 0);
                enc.set_buffer(5, Some(&h_post_attn), 0);
                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
            }

            // ── KNN probe: snapshot h_post_attn at the probe layer ──
            // Uses residual_add(h_post_attn, zero, probe_buf) as a copy.
            // One dispatch (~0.01 ms), no pipeline break.
            if let Some(pl) = probe_layer {
                if l == pl {
                    if let Some(ref pb) = probe_buf {
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        let zero_vec = vec![0.0f32; hidden];
                        let zero_buf = self.bufs.transient_from_f32(&zero_vec);
                        encode_residual_add(enc, &self.residual_add_pipeline,
                            &h_post_attn, &zero_buf, pb, hidden);
                    }
                }
            }

            // DIAGNOSTIC: LARQL_SKIP_FFN=1 bypasses FFN entirely, returning h_post_attn
            // as the layer output. Used to isolate whether the Gemma 3 decode NaN
            // comes from attention+O-proj+post-attn-norm vs. FFN+post-FFN-norm.
            let skip_ffn = std::env::var("LARQL_SKIP_FFN").ok().as_deref() == Some("1");
            if skip_ffn {
                // Copy h_post_attn to new_h so downstream logic still works.
                let len_val = hidden as u32;
                enc.set_compute_pipeline_state(&self.residual_add_pipeline);
                // residual_add(a, b, out) = a + b; pass zero-buffer as b.
                let zero_vec = vec![0.0f32; hidden];
                let zero_buf = self.bufs.transient_from_f32(&zero_vec);
                enc.set_buffer(0, Some(&h_post_attn), 0);
                enc.set_buffer(1, Some(&zero_buf), 0);
                enc.set_buffer(2, Some(&new_h), 0);
                enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                // residual_add is per-element (no cooperative reduction), so dispatch ONE
                // thread per output element across many threadgroups.
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                h_buf = new_h;
                continue; // skip steps 6-8
            }

            // ── Step 6: FFN (format-aware: Q4_KF uses llama.cpp kernel, Q4_K uses our kernel, Q4_0 uses Q8) ──
            {
                let ffn_is_q4kf = layer.gate.format == crate::QuantFormat::Q4_KF;
                let ffn_is_q8_gguf = layer.gate.format == crate::QuantFormat::Q8_0Gguf;

                if ffn_is_q8_gguf {
                    // Q8_0Gguf FFN: f32 input → q8_0_gguf matvec for gate/up/down.
                    use crate::metal::shaders::q8_0_gguf_matvec as q8gm;
                    let n_tgs_inter = (inter as u64).div_ceil(q8gm::ROWS_PER_TG);
                    let n_tgs_hidden = (hidden as u64).div_ceil(q8gm::ROWS_PER_TG);

                    if layer.is_gated() {
                        let gate_out = &gate_out_scratch;
                        // Gate
                        enc.set_compute_pipeline_state(&self.q8_0_gguf_matvec_pipeline);
                        enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_norm_out), 0);
                        enc.set_buffer(2, Some(&gate_out), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_inter, 1, 1), MTLSize::new(q8gm::THREADS_PER_TG, 1, 1));
                        // Up
                        enc.set_compute_pipeline_state(&self.q8_0_gguf_matvec_pipeline);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_norm_out), 0);
                        enc.set_buffer(2, Some(&up_out), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_inter, 1, 1), MTLSize::new(q8gm::THREADS_PER_TG, 1, 1));
                        // GEGLU
                        let geglu = match layer.activation {
                            crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                            _ => &self.geglu_pipeline,
                        };
                        enc.set_compute_pipeline_state(geglu);
                        enc.set_buffer(0, Some(&gate_out), 0);
                        enc.set_buffer(1, Some(&up_out), 0);
                        enc.set_buffer(2, Some(&act_buf), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        // Down
                        enc.set_compute_pipeline_state(&self.q8_0_gguf_matvec_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_hidden, 1, 1), MTLSize::new(q8gm::THREADS_PER_TG, 1, 1));
                    } else {
                        // Up
                        enc.set_compute_pipeline_state(&self.q8_0_gguf_matvec_pipeline);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_norm_out), 0);
                        enc.set_buffer(2, Some(&up_out), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_inter, 1, 1), MTLSize::new(q8gm::THREADS_PER_TG, 1, 1));
                        // Activation
                        let activation_pipeline = match layer.activation {
                            crate::Activation::GeluTanh => &self.gelu_tanh_pipeline,
                            _ => &self.silu_pipeline,
                        };
                        enc.set_compute_pipeline_state(activation_pipeline);
                        enc.set_buffer(0, Some(&up_out), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        // Down
                        enc.set_compute_pipeline_state(&self.q8_0_gguf_matvec_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_hidden, 1, 1), MTLSize::new(q8gm::THREADS_PER_TG, 1, 1));
                    }
                } else if ffn_is_q4kf {
                    // Q4_KF (GGUF) FFN path: llama.cpp-exact kernel
                    use crate::metal::shaders::q4kf_qkv_proj as q4kf;
                    use crate::metal::shaders::q4kf_ffn_gate_up as q4kf_gu;
                    use crate::metal::shaders::q6k_matvec as q6k_mv;
                    // Down format may differ from gate/up (Q4_KF gate+up + Q6_K down
                    // is the standard Gemma 3 llama.cpp layout). Pick the down
                    // pipeline + dims by down.format; the old code unconditionally
                    // used q4kf_proj which produced garbage / NaN on Q6_K bytes.
                    let down_is_q6k = layer.down.format == crate::QuantFormat::Q6_K;
                    let n_tgs_down = if down_is_q6k {
                        (hidden as u64).div_ceil(q6k_mv::ROWS_PER_TG)
                    } else {
                        (hidden as u64).div_ceil(q4kf::ROWS_PER_TG)
                    };
                    let down_threads = if down_is_q6k { q6k_mv::THREADS_PER_TG } else { q4kf::THREADS_PER_TG };
                    let down_pipeline = if down_is_q6k {
                        &self.q6k_matvec_pipeline
                    } else {
                        &self.q4kf_proj_pipeline
                    };

                    if layer.is_gated() {
                        let gate_out = &gate_out_scratch;
                        // Fused gate+up: one dispatch, shared input (llama.cpp inner loop)
                        let n_tgs_per_mat = (inter as u64).div_ceil(q4kf_gu::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4kf_ffn_gate_up_pipeline);
                        enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                        enc.set_buffer(1, Some(&up_bufs[l]), 0);
                        enc.set_buffer(2, Some(&ffn_norm_out), 0);
                        enc.set_buffer(3, Some(&gate_out), 0);
                        enc.set_buffer(4, Some(&up_out), 0);
                        enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(n_tgs_per_mat * 2, 1, 1),
                            MTLSize::new(q4kf_gu::THREADS_PER_TG, 1, 1),
                        );
                        // GEGLU
                        let geglu = match layer.activation {
                            crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                            _ => &self.geglu_pipeline,
                        };
                        enc.set_compute_pipeline_state(geglu);
                        enc.set_buffer(0, Some(&gate_out), 0);
                        enc.set_buffer(1, Some(&up_out), 0);
                        enc.set_buffer(2, Some(&act_buf), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        // Down — format-aware dispatch.
                        enc.set_compute_pipeline_state(down_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(down_threads, 1, 1));
                    } else {
                        let n_tgs_up = (inter as u64).div_ceil(q4kf::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_norm_out), 0);
                        enc.set_buffer(2, Some(&up_out), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_up, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
                        let activation_pipeline = match layer.activation {
                            crate::Activation::GeluTanh => &self.gelu_tanh_pipeline,
                            _ => &self.silu_pipeline,
                        };
                        enc.set_compute_pipeline_state(activation_pipeline);
                        enc.set_buffer(0, Some(&up_out), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
                    }
                } else if ffn_uses_q4k {
                    // Q4_K FFN path: f32 input → Q4_K (or Q6_K) matvec
                    use crate::metal::shaders::q4k_matvec as q4k;
                    use crate::metal::shaders::q6k_matvec as q6k;
                    use crate::metal::shaders::q4k_ffn_gate_up as q4k_gu;
                    // Down projection dispatch depends on down.format (Ollama uses Q6_K).
                    let down_is_q6k = layer.down.format == crate::QuantFormat::Q6_K;
                    let n_tgs_down = if down_is_q6k {
                        (hidden as u64).div_ceil(q6k::ROWS_PER_TG)
                    } else {
                        (hidden as u64).div_ceil(q4k::ROWS_PER_TG)
                    };

                    if layer.is_gated() {
                        let gate_out = &gate_out_scratch;
                        let gate_is_q6k = layer.gate.format == crate::QuantFormat::Q6_K;
                        let up_is_q6k = layer.up.format == crate::QuantFormat::Q6_K;
                        let n_tgs_per_mat = (inter as u64).div_ceil(q4k_gu::ROWS_PER_TG);
                        // DIAGNOSTIC: LARQL_SEPARATE_GATE_UP=1 bypasses the fused gate_up kernel
                        // and uses two independent matvec dispatches. Also required when gate
                        // or up is Q6_K (fused kernel only accepts Q4_K).
                        let separate_gate_up = std::env::var("LARQL_SEPARATE_GATE_UP").ok().as_deref() == Some("1")
                            || gate_is_q6k || up_is_q6k;
                        if separate_gate_up {
                            let dispatch_single = |enc_ref: &metal::ComputeCommandEncoderRef,
                                                    w_buf: &metal::Buffer,
                                                    out_buf: &metal::Buffer,
                                                    is_q6k: bool| {
                                if is_q6k {
                                    let n_tgs = (inter as u64).div_ceil(q6k::ROWS_PER_TG);
                                    enc_ref.set_compute_pipeline_state(&self.q6k_matvec_pipeline);
                                    enc_ref.set_buffer(0, Some(w_buf), 0);
                                    enc_ref.set_buffer(1, Some(&ffn_norm_out), 0);
                                    enc_ref.set_buffer(2, Some(out_buf), 0);
                                    enc_ref.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                                    enc_ref.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                                    enc_ref.dispatch_thread_groups(MTLSize::new(n_tgs, 1, 1), MTLSize::new(q6k::THREADS_PER_TG, 1, 1));
                                } else {
                                    let n_tgs = (inter as u64).div_ceil(q4k::ROWS_PER_TG);
                                    enc_ref.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                                    enc_ref.set_buffer(0, Some(w_buf), 0);
                                    enc_ref.set_buffer(1, Some(&ffn_norm_out), 0);
                                    enc_ref.set_buffer(2, Some(out_buf), 0);
                                    enc_ref.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                                    enc_ref.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                                    enc_ref.dispatch_thread_groups(MTLSize::new(n_tgs, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
                                }
                            };
                            dispatch_single(enc, &gate_bufs[l], &gate_out, gate_is_q6k);
                            dispatch_single(enc, &up_bufs[l], &up_out, up_is_q6k);
                        } else {
                            enc.set_compute_pipeline_state(&self.q4k_ffn_gate_up_pipeline);
                            enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                            enc.set_buffer(1, Some(&up_bufs[l]), 0);
                            enc.set_buffer(2, Some(&ffn_norm_out), 0);
                            enc.set_buffer(3, Some(&gate_out), 0);
                            enc.set_buffer(4, Some(&up_out), 0);
                            enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(
                                MTLSize::new(n_tgs_per_mat * 2, 1, 1),
                                MTLSize::new(q4k_gu::THREADS_PER_TG, 1, 1),
                            );
                        }
                        // Fused GEGLU + down projection (S1 technique: activation
                        // computed on-the-fly per row, no intermediate activation buffer).
                        // Saves one full dispatch + one read/write of the inter-sized buffer.
                        // Falls back to separate dispatches for Q6_K down weights.
                        let skip_down = std::env::var("LARQL_SKIP_DOWN").ok().as_deref() == Some("1");
                        if skip_down {
                            // DIAGNOSTIC: zero-fill down_out
                            let zero_vec = vec![0.0f32; hidden];
                            let zero_buf = self.bufs.transient_from_f32(&zero_vec);
                            use crate::metal::ops::full_pipeline::encode_residual_add;
                            encode_residual_add(enc, &self.residual_add_pipeline,
                                &zero_buf, &zero_buf, &down_out, hidden);
                        } else if !down_is_q6k {
                            // Q4_K down: fused GEGLU+down kernel (same as walk-FFN S1)
                            use crate::metal::shaders::q4k_geglu_down as q4k_gd;
                            let geglu_down_pipeline = match layer.activation {
                                crate::Activation::GeluTanh => &self.q4k_geglu_gelu_tanh_down_pipeline,
                                _ => &self.q4k_geglu_silu_down_pipeline,
                            };
                            let n_tgs_geglu = (hidden as u64).div_ceil(q4k_gd::ROWS_PER_TG);
                            enc.set_compute_pipeline_state(geglu_down_pipeline);
                            enc.set_buffer(0, Some(&down_bufs[l]), 0);
                            enc.set_buffer(1, Some(&gate_out), 0);
                            enc.set_buffer(2, Some(&up_out), 0);
                            enc.set_buffer(3, Some(&down_out), 0);
                            enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(
                                MTLSize::new(n_tgs_geglu, 1, 1),
                                MTLSize::new(q4k_gd::THREADS_PER_TG, 1, 1),
                            );
                        } else {
                            // Q6_K down: separate GEGLU + down (no fused kernel for Q6_K)
                            let geglu = match layer.activation {
                                crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                                _ => &self.geglu_pipeline,
                            };
                            enc.set_compute_pipeline_state(geglu);
                            enc.set_buffer(0, Some(&gate_out), 0);
                            enc.set_buffer(1, Some(&up_out), 0);
                            enc.set_buffer(2, Some(&act_buf), 0);
                            enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                            enc.set_compute_pipeline_state(&self.q6k_matvec_pipeline);
                            enc.set_buffer(0, Some(&down_bufs[l]), 0);
                            enc.set_buffer(1, Some(&act_buf), 0);
                            enc.set_buffer(2, Some(&down_out), 0);
                            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q6k::THREADS_PER_TG, 1, 1));
                        }
                    } else {
                        let n_tgs_up = (inter as u64).div_ceil(q4k::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_norm_out), 0);
                        enc.set_buffer(2, Some(&up_out), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_up, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
                        let activation_pipeline = match layer.activation {
                            crate::Activation::GeluTanh => &self.gelu_tanh_pipeline,
                            _ => &self.silu_pipeline,
                        };
                        enc.set_compute_pipeline_state(activation_pipeline);
                        enc.set_buffer(0, Some(&up_out), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
                    }
                } else {
                    // Q4_0 FFN path: Q8 input → Q4_0 matvec (legacy)
                    use crate::metal::shaders::q4_matvec as q4mv;
                    let n_tgs_ffn = (inter as u64).div_ceil(q4mv::ROWS_PER_TG);

                    if layer.is_gated() {
                        let gate_out = &gate_out_scratch;
                        enc.set_compute_pipeline_state(&self.q4.matvec);
                        enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_q8), 0);
                        enc.set_buffer(2, Some(&ffn_q8s), 0);
                        enc.set_buffer(3, Some(&gate_out), 0);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(3, Some(&up_out), 0);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                        let geglu = match layer.activation {
                            crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                            _ => &self.geglu_pipeline,
                        };
                        enc.set_compute_pipeline_state(geglu);
                        enc.set_buffer(0, Some(&gate_out), 0);
                        enc.set_buffer(1, Some(&up_out), 0);
                        enc.set_buffer(2, Some(&act_buf), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                    } else {
                        enc.set_compute_pipeline_state(&self.q4.matvec);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_q8), 0);
                        enc.set_buffer(2, Some(&ffn_q8s), 0);
                        enc.set_buffer(3, Some(&up_out), 0);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                        let activation_pipeline = match layer.activation {
                            crate::Activation::GeluTanh => &self.gelu_tanh_pipeline,
                            _ => &self.silu_pipeline,
                        };
                        enc.set_compute_pipeline_state(activation_pipeline);
                        enc.set_buffer(0, Some(&up_out), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                    }

                    enc.set_compute_pipeline_state(&self.q4.f32_matvec);
                    enc.set_buffer(0, Some(&down_bufs[l]), 0);
                    enc.set_buffer(1, Some(&act_buf), 0);
                    enc.set_buffer(2, Some(&down_out), 0);
                    enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
                }
            }

            // ── Step 7: Post-FFN residual ──
            let skip_post_ffn_norm = std::env::var("LARQL_SKIP_POST_FFN_NORM").ok().as_deref() == Some("1");
            if has_post_norms && !skip_post_ffn_norm {
                if let Some(post_ffn) = layer.post_ffn_norm {
                    let post_ffn_buf = self.bufs.transient_from_f32(post_ffn);
                    let normed_ffn = &normed_scratch;
                    use crate::metal::ops::full_pipeline::encode_rms_norm;
                    encode_rms_norm(enc, &self.rms_norm_pipeline,
                        &down_out, &post_ffn_buf, &normed_ffn, hidden, eps, norm_offset);
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    encode_residual_add(enc, &self.residual_add_pipeline,
                        &h_post_attn, &normed_ffn, &new_h, hidden);
                } else {
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    encode_residual_add(enc, &self.residual_add_pipeline,
                        &h_post_attn, &down_out, &new_h, hidden);
                }
            } else {
                let len_val = hidden as u32;
                enc.set_compute_pipeline_state(&self.residual_add_pipeline);
                enc.set_buffer(0, Some(&h_post_attn), 0);
                enc.set_buffer(1, Some(&down_out), 0);
                enc.set_buffer(2, Some(&new_h), 0);
                enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
            }

            // ── Step 8: Optional layer scalar ──
            if layer.layer_scalar != 0.0 {
                let scaled = &scaled_scratch;
                let scalar_val = layer.layer_scalar;
                enc.set_compute_pipeline_state(&self.scale_vector_pipeline);
                enc.set_buffer(0, Some(&new_h), 0);
                enc.set_buffer(1, Some(&scaled), 0);
                enc.set_bytes(2, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &scalar_val as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                h_buf = scaled;
            } else {
                h_buf = new_h;
            }

        }

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // DIAGNOSTIC: LARQL_READBACK=1 dumps intermediate buffer stats to stderr.
        // Limited to buffers defined OUTSIDE the per-layer loop (the others would
        // need explicit copy-out during the loop).
        if std::env::var("LARQL_READBACK").ok().as_deref() == Some("1") {
            // Dump specific channel values at Gemma 3 L0 divergent channels.
            let chs = [443usize, 368, 1762, 1365, 1638];
            for (name, buf, n_expected) in [
                ("o_out", &o_out_buf, hidden),
                ("h_post_attn", &h_post_attn, hidden),
                ("ffn_norm_out", &ffn_norm_out, hidden),
                ("down_out", &down_out, hidden),
                ("h_a", &h_a, hidden),
                ("h_b", &h_b, hidden),
            ] {
                let data = super::buffers::read_buffer_f32(buf, n_expected);
                let vals: Vec<String> = chs.iter().filter(|&&i| i < n_expected)
                    .map(|&i| format!("[{i}]={:.3}", data[i])).collect();
                eprintln!("[readback-ch] {:<14} {}", name, vals.join(" "));
            }
            for (name, buf) in [
                ("norm_f32_buf", &norm_f32_buf),
                ("q_out", &q_out),
                ("k_out", &k_out),
                ("v_out", &v_out),
                ("attn_out", &attn_out_buf),
                ("o_out", &o_out_buf),
                ("h_post_attn", &h_post_attn),
                ("ffn_norm_out", &ffn_norm_out),
                ("gate_out", &gate_out_scratch),
                ("up_out", &up_out),
                ("act_buf", &act_buf),
                ("down_out", &down_out),
                ("h_a", &h_a),
                ("h_b", &h_b),
            ] {
                let n = (buf.length() / 4) as usize;
                let data = super::buffers::read_buffer_f32(buf, n);
                let ninf = data.iter().filter(|v| v.is_infinite()).count();
                let nnan = data.iter().filter(|v| v.is_nan()).count();
                let fmx = data.iter().filter(|v| v.is_finite()).map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("[readback] {:<14} len={:<6} {} inf, {} nan, max|finite|={:.2}",
                    name, n, ninf, nnan, fmx);
            }
        }

        let final_h = super::buffers::read_buffer_f32(&h_buf, hidden);
        let probe_h = probe_buf.as_ref().map(|pb| super::buffers::read_buffer_f32(pb, hidden));
        (final_h, probe_h)
    }

    /// MoE decode: per-layer command buffers (needed for router readback).
    ///
    /// Mirrors `decode_token_inner` for attention steps, but uses MoE FFN dispatch
    /// for layers with `is_moe_layer == true` and falls back to dense FFN for others.
    /// Each layer gets its own command buffer because MoE router readback requires
    /// a GPU→CPU sync mid-layer.
    #[allow(clippy::too_many_arguments)]
    fn decode_token_inner_moe(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        probe_layer: Option<usize>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        let num_layers = layers.len();
        let hidden_val = hidden as u32;
        let inter_val = inter as u32;

        // ── Pre-cache weight buffers ──
        let _t_bufs = std::time::Instant::now();
        let wq_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wq.data)).collect();
        let wk_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wk.data)).collect();
        let wv_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wv.data)).collect();
        let wo_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wo.data)).collect();
        let wq_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wq.scales.unwrap_or(&[]))).collect();
        let wk_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wk.scales.unwrap_or(&[]))).collect();
        let wv_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wv.scales.unwrap_or(&[]))).collect();
        let wo_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wo.scales.unwrap_or(&[]))).collect();
        let gate_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.gate.data)).collect();
        let up_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.up.data)).collect();
        let down_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.down.data)).collect();
        let input_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.input_norm)).collect();
        let post_attn_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.post_attn_norm)).collect();
        let q_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.q_norm_weight.unwrap_or(&[]))).collect();
        let k_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.k_norm_weight.unwrap_or(&[]))).collect();

        // ── MoE weight buffers ──
        // NOTE: MoE expert buffers are created per-layer inside the loop
        // to avoid registering 16+ GB of Metal buffers upfront (swap thrash).
        // Router, expert_gate_up, and expert_down buffers are created on demand.

        // ── MoE dimensions (from first MoE layer) ──
        let moe_layer = layers.iter().find(|l| l.is_moe_layer);
        let moe_experts = moe_layer.map(|l| l.num_experts).unwrap_or(0);
        let moe_inter = moe_layer.map(|l| l.expert_intermediate).unwrap_or(0);

        // ── Two h buffers for ping-pong ──
        let h_init = self.bufs.transient_from_f32(x);
        let h_a = self.bufs.output((hidden * 4) as u64);
        let h_b = self.bufs.output((hidden * 4) as u64);
        let mut h_buf = &h_init;

        // ── Scratch buffers (reused across layers) ──
        let q_out = self.bufs.output((q_dim * 4) as u64);
        let k_out = self.bufs.output((kv_dim * 4) as u64);
        let v_out = self.bufs.output((kv_dim * 4) as u64);
        let norm_f32_buf = self.bufs.output((hidden * 4) as u64);
        let attn_out_buf = self.bufs.output((q_dim * 4) as u64);
        let o_out_buf = self.bufs.output((hidden * 4) as u64);
        let h_post_attn = self.bufs.output((hidden * 4) as u64);
        let probe_buf = if probe_layer.is_some() {
            Some(self.bufs.output((hidden * 4) as u64))
        } else {
            None
        };
        let ffn_norm_out = self.bufs.output((hidden * 4) as u64);
        let ffn_q8 = self.bufs.output(hidden as u64);
        let ffn_q8s = self.bufs.output((hidden / 32 * 4) as u64);
        let up_out = self.bufs.output((inter * 4) as u64);
        let act_buf = self.bufs.output((inter * 4) as u64);
        let down_out = self.bufs.output((hidden * 4) as u64);
        let gate_out_scratch = self.bufs.output((inter * 4) as u64);
        let normed_scratch = self.bufs.output((hidden * 4) as u64);
        let o_q8_scratch = self.bufs.output(q_dim as u64);
        let o_q8s_scratch = self.bufs.output((q_dim / 32 * 4) as u64);
        let scaled_scratch = self.bufs.output((hidden * 4) as u64);

        // ── MoE scratch buffers ──
        let router_out = if moe_experts > 0 {
            self.bufs.output((moe_experts * 4) as u64)
        } else {
            self.bufs.output(4)
        };
        let moe_accum = if moe_experts > 0 {
            self.bufs.output((hidden * 4) as u64)
        } else {
            self.bufs.output(4)
        };
        let expert_gu_out = if moe_inter > 0 {
            self.bufs.output((moe_inter * 2 * 4) as u64)
        } else {
            self.bufs.output(4)
        };
        let expert_act_buf = if moe_inter > 0 {
            self.bufs.output((moe_inter * 4) as u64)
        } else {
            self.bufs.output(4)
        };
        let expert_down_out_buf = if moe_experts > 0 {
            self.bufs.output((hidden * 4) as u64)
        } else {
            self.bufs.output(4)
        };

        let _trace_moe = std::env::var("LARQL_TRACE_MOE").ok().as_deref() == Some("1");
        if _trace_moe {
            eprintln!("[moe-decode] buffers created in {:.1}ms", _t_bufs.elapsed().as_secs_f64() * 1000.0);
        }

        // ── Per-layer loop with per-layer command buffers ──
        for l in 0..num_layers {
            let _t_layer = std::time::Instant::now();
            let layer = &layers[l];
            let norm_offset = layer.norm_offset;
            let eps = layer.eps;
            let scale = layer.attn_scale;
            let layer_head_dim = layer.head_dim;
            let layer_num_q_heads = layer.num_q_heads;
            let layer_num_kv_heads = layer.num_kv_heads;
            let layer_rope_base = layer.rope_base;
            let layer_rotary_dim = if layer.rotary_dim > 0 { layer.rotary_dim } else { layer_head_dim };
            let uses_q4k = layer.wq.format == crate::QuantFormat::Q4_K
                || layer.wq.format == crate::QuantFormat::Q6_K
                || layer.wq.format == crate::QuantFormat::Q4_KF
                || layer.wq.format == crate::QuantFormat::Q8_0Gguf;
            let layer_q_dim = layer_num_q_heads * layer_head_dim;
            let _layer_kv_dim = layer_num_kv_heads * layer_head_dim;
            let window_size = layer.sliding_window as u32;

            let new_h = if l % 2 == 0 { &h_a } else { &h_b };

            // ── Command buffer for attention half of this layer ──
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            // ── Step 1: Input norm + Q/K/V projection ──
            // (Identical to dense path)
            if uses_q4k {
                let all_same_format = layer.wq.format == layer.wk.format && layer.wk.format == layer.wv.format;
                if all_same_format && layer.wq.format == crate::QuantFormat::Q4_K
                    && layer.norm_type != crate::NormType::LayerNorm
                {
                    let total_rows = (q_dim + kv_dim + kv_dim) as u32;
                    let q_rows_val = q_dim as u32;
                    let k_rows_val = kv_dim as u32;
                    let v_rows_val = kv_dim as u32;
                    let k_val = hidden as u32;
                    let rows_per_tg = crate::metal::shaders::q4k_qkv_proj::ROWS_PER_TG;
                    let num_tgs = (total_rows as u64).div_ceil(rows_per_tg);
                    enc.set_compute_pipeline_state(&self.q4k_norm_qkv_pipeline);
                    enc.set_buffer(0, Some(&wq_bufs[l]), 0);
                    enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                    enc.set_buffer(2, Some(&wv_bufs[l]), 0);
                    enc.set_buffer(3, Some(&h_buf), 0);
                    enc.set_buffer(4, Some(&input_norm_bufs[l]), 0);
                    enc.set_buffer(5, Some(&q_out), 0);
                    enc.set_buffer(6, Some(&k_out), 0);
                    enc.set_buffer(7, Some(&v_out), 0);
                    enc.set_bytes(8, 4, &q_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(9, 4, &k_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(10, 4, &v_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(11, 4, &k_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(12, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(13, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    let threads_per_tg = crate::metal::shaders::q4k_qkv_proj::THREADS_PER_TG;
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs, 1, 1),
                        MTLSize::new(threads_per_tg, 1, 1),
                    );
                } else {
                    use crate::metal::ops::full_pipeline::encode_rms_norm;
                    if layer.norm_type == crate::NormType::LayerNorm {
                        let len_val = hidden as u32;
                        if let Some(bias) = layer.input_norm_bias {
                            let bias_buf = self.bufs.transient_from_f32(bias);
                            enc.set_compute_pipeline_state(&self.layer_norm_pipeline);
                            enc.set_buffer(0, Some(&h_buf), 0);
                            enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                            enc.set_buffer(2, Some(&bias_buf), 0);
                            enc.set_buffer(3, Some(&norm_f32_buf), 0);
                            enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                            enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                        } else {
                            enc.set_compute_pipeline_state(&self.layer_norm_no_bias_pipeline);
                            enc.set_buffer(0, Some(&h_buf), 0);
                            enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                            enc.set_buffer(2, Some(&norm_f32_buf), 0);
                            enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
                            enc.set_bytes(5, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                        }
                        enc.dispatch_threads(
                            MTLSize::new(hidden as u64, 1, 1),
                            MTLSize::new(256.min(hidden as u64), 1, 1),
                        );
                    } else {
                        encode_rms_norm(enc, &self.rms_norm_pipeline,
                            &h_buf, &input_norm_bufs[l], &norm_f32_buf,
                            hidden, eps, norm_offset);
                    }

                    let all_same_format_inner = layer.wq.format == layer.wk.format && layer.wk.format == layer.wv.format;
                    if all_same_format_inner && layer.wq.format != crate::QuantFormat::Q6_K && layer.wq.format != crate::QuantFormat::Q8_0Gguf {
                        let total_rows = (q_dim + kv_dim + kv_dim) as u32;
                        let q_rows_val = q_dim as u32;
                        let k_rows_val = kv_dim as u32;
                        let v_rows_val = kv_dim as u32;
                        let k_val = hidden as u32;
                        let (qkv_pipeline, rows_per_tg) = if layer.wq.format == crate::QuantFormat::Q4_KF {
                            (&self.q4kf_qkv_proj_pipeline, crate::metal::shaders::q4kf_qkv_proj::ROWS_PER_TG)
                        } else {
                            (&self.q4k_qkv_proj_pipeline, crate::metal::shaders::q4k_qkv_proj::ROWS_PER_TG)
                        };
                        let num_tgs = (total_rows as u64).div_ceil(rows_per_tg);
                        enc.set_compute_pipeline_state(qkv_pipeline);
                        enc.set_buffer(0, Some(&wq_bufs[l]), 0);
                        enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                        enc.set_buffer(2, Some(&wv_bufs[l]), 0);
                        enc.set_buffer(3, Some(&norm_f32_buf), 0);
                        enc.set_buffer(4, Some(&q_out), 0);
                        enc.set_buffer(5, Some(&k_out), 0);
                        enc.set_buffer(6, Some(&v_out), 0);
                        enc.set_bytes(7, 4, &q_rows_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(8, 4, &k_rows_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(9, 4, &v_rows_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(10, 4, &k_val as *const u32 as *const std::ffi::c_void);
                        let threads_per_tg = if layer.wq.format == crate::QuantFormat::Q4_KF {
                            crate::metal::shaders::q4kf_qkv_proj::THREADS_PER_TG
                        } else {
                            crate::metal::shaders::q4k_qkv_proj::THREADS_PER_TG
                        };
                        enc.dispatch_thread_groups(
                            MTLSize::new(num_tgs, 1, 1),
                            MTLSize::new(threads_per_tg, 1, 1),
                        );
                    } else {
                        // Mixed formats: dispatch each projection separately.
                        let k_val = hidden as u32;
                        fn encode_single_proj_moe(
                            enc: &metal::ComputeCommandEncoderRef,
                            w_buf: &metal::Buffer, x_buf: &metal::Buffer, out_buf: &metal::Buffer,
                            rows: usize, k: u32, format: crate::QuantFormat,
                            q4k_pipeline: &metal::ComputePipelineState,
                            q4kf_pipeline: &metal::ComputePipelineState,
                            q6k_pipeline: &metal::ComputePipelineState,
                        ) {
                            match format {
                                crate::QuantFormat::Q6_K => {
                                    use crate::metal::shaders::q6k_matvec as q6k;
                                    let n = rows as u32;
                                    let num_tgs = (rows as u64).div_ceil(q6k::ROWS_PER_TG);
                                    enc.set_compute_pipeline_state(q6k_pipeline);
                                    enc.set_buffer(0, Some(w_buf), 0);
                                    enc.set_buffer(1, Some(x_buf), 0);
                                    enc.set_buffer(2, Some(out_buf), 0);
                                    enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
                                    enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
                                    enc.dispatch_thread_groups(
                                        MTLSize::new(num_tgs, 1, 1),
                                        MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
                                    );
                                }
                                crate::QuantFormat::Q4_KF => {
                                    use crate::metal::shaders::q4kf_qkv_proj as proj_sh;
                                    let n = rows as u32;
                                    let num_tgs = (rows as u64).div_ceil(proj_sh::ROWS_PER_TG);
                                    enc.set_compute_pipeline_state(q4kf_pipeline);
                                    enc.set_buffer(0, Some(w_buf), 0);
                                    enc.set_buffer(1, Some(x_buf), 0);
                                    enc.set_buffer(2, Some(out_buf), 0);
                                    enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
                                    enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
                                    enc.dispatch_thread_groups(
                                        MTLSize::new(num_tgs, 1, 1),
                                        MTLSize::new(proj_sh::THREADS_PER_TG, 1, 1),
                                    );
                                }
                                _ => {
                                    use crate::metal::shaders::q4k_matvec as q4k;
                                    let n = rows as u32;
                                    let num_tgs = (rows as u64).div_ceil(q4k::ROWS_PER_TG);
                                    enc.set_compute_pipeline_state(q4k_pipeline);
                                    enc.set_buffer(0, Some(w_buf), 0);
                                    enc.set_buffer(1, Some(x_buf), 0);
                                    enc.set_buffer(2, Some(out_buf), 0);
                                    enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
                                    enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
                                    enc.dispatch_thread_groups(
                                        MTLSize::new(num_tgs, 1, 1),
                                        MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
                                    );
                                }
                            }
                        }
                        encode_single_proj_moe(enc, &wq_bufs[l], &norm_f32_buf, &q_out,
                            q_dim, k_val, layer.wq.format,
                            &self.q4k_matvec_pipeline, &self.q4kf_proj_pipeline, &self.q6k_matvec_pipeline);
                        encode_single_proj_moe(enc, &wk_bufs[l], &norm_f32_buf, &k_out,
                            kv_dim, k_val, layer.wk.format,
                            &self.q4k_matvec_pipeline, &self.q4kf_proj_pipeline, &self.q6k_matvec_pipeline);
                        encode_single_proj_moe(enc, &wv_bufs[l], &norm_f32_buf, &v_out,
                            kv_dim, k_val, layer.wv.format,
                            &self.q4k_matvec_pipeline, &self.q4kf_proj_pipeline, &self.q6k_matvec_pipeline);
                    }
                }
            } else {
                // Q8 path
                let q8_buf = &ffn_q8;
                let q8s_buf = &ffn_q8s;
                enc.set_compute_pipeline_state(&self.rms_norm_q8_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                enc.set_buffer(2, Some(&q8_buf), 0);
                enc.set_buffer(3, Some(&q8s_buf), 0);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                let total_rows = (q_dim + kv_dim + kv_dim) as u32;
                let q_rows = q_dim as u32;
                let k_rows = kv_dim as u32;
                let v_rows = kv_dim as u32;
                let k_val = hidden as u32;
                enc.set_compute_pipeline_state(&self.q8_qkv_proj_pipeline);
                enc.set_buffer(0, Some(&wq_bufs[l]), 0);
                enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                enc.set_buffer(2, Some(&wv_bufs[l]), 0);
                enc.set_buffer(3, Some(&q8_buf), 0);
                enc.set_buffer(4, Some(&wq_scale_bufs[l]), 0);
                enc.set_buffer(5, Some(&wk_scale_bufs[l]), 0);
                enc.set_buffer(6, Some(&wv_scale_bufs[l]), 0);
                enc.set_buffer(7, Some(&q8s_buf), 0);
                enc.set_buffer(8, Some(&q_out), 0);
                enc.set_buffer(9, Some(&k_out), 0);
                enc.set_buffer(10, Some(&v_out), 0);
                enc.set_bytes(11, 4, &q_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(12, 4, &k_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(13, 4, &v_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(14, 4, &k_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new((total_rows as u64).div_ceil(8), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }

            // ── Step 1.5: QK-norm (Gemma 3 / Gemma 4) ──
            if layer.q_norm_weight.is_some() {
                let hd_val = layer_head_dim as u32;
                let eps_val = layer.eps;
                let off_val = layer.qk_norm_offset;
                let tg_threads = 256u64.min(layer_head_dim as u64);
                enc.set_compute_pipeline_state(&self.rms_norm_multihead_pipeline);
                enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &eps_val as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &off_val as *const f32 as *const std::ffi::c_void);
                enc.set_buffer(0, Some(&q_out), 0);
                enc.set_buffer(1, Some(&q_norm_bufs[l]), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(layer_num_q_heads as u64, 1, 1),
                    MTLSize::new(tg_threads, 1, 1),
                );
                enc.set_buffer(0, Some(&k_out), 0);
                enc.set_buffer(1, Some(&k_norm_bufs[l]), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(layer_num_kv_heads as u64, 1, 1),
                    MTLSize::new(tg_threads, 1, 1),
                );
            }

            // ── Step 2: RoPE on Q and K heads (batched) ──
            {
                let pos = kv_cache.layers[l].current_len as u32;
                let hd = layer_head_dim as u32;
                let rdim = layer_rotary_dim as u32;
                let rope_pairs = (layer_rotary_dim / 2) as u64;
                let num_q = layer_num_q_heads as u32;
                let num_kv = layer_num_kv_heads as u32;
                let freq_scale = layer.rope_freq_scale;
                enc.set_compute_pipeline_state(&self.rope_at_pos_batched_pipeline);
                enc.set_buffer(0, Some(&q_out), 0);
                enc.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(2, 4, &layer_rope_base as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &pos as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &rdim as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &num_q as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &freq_scale as *const f32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(rope_pairs, layer_num_q_heads as u64, 1),
                    MTLSize::new(rope_pairs.min(256), 1, 1),
                );
                enc.set_buffer(0, Some(&k_out), 0);
                enc.set_bytes(5, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(rope_pairs, layer_num_kv_heads as u64, 1),
                    MTLSize::new(rope_pairs.min(256), 1, 1),
                );
            }

            // ── Step 3: V-norm batched (optional, Gemma 4) ──
            if layer.has_v_norm {
                let hd_val = layer_head_dim as u32;
                let num_kv = layer_num_kv_heads as u32;
                enc.set_compute_pipeline_state(&self.v_norm_batched_pipeline);
                enc.set_buffer(0, Some(&v_out), 0);
                enc.set_buffer(1, Some(&v_out), 0);
                enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(layer_head_dim as u64, layer_num_kv_heads as u64, 1),
                    MTLSize::new((layer_head_dim as u64).min(256), 1, 1),
                );
            }

            // ── Step 4: KV cache append + attend ──
            let attn_out = &attn_out_buf;
            ops::kv_cache::encode_kv_append(
                enc, &kv_cache.layers[l],
                &self.kv_append_pipeline, &k_out, &v_out,
            );
            let t_after = kv_cache.layers[l].current_len + 1;
            let kv_attend = if t_after <= 1024 {
                &self.kv_attend_fast_pipeline
            } else {
                &self.kv_attend_long_pipeline
            };
            ops::kv_cache::encode_kv_attend_softcap(
                enc, &kv_cache.layers[l],
                kv_attend, &q_out, &attn_out,
                layer_num_q_heads, scale, window_size, layer.softcap,
            );
            kv_cache.layers[l].current_len += 1;

            // ── Step 5: O projection ──
            {
                if uses_q4k {
                    let o_rows = hidden as u32;
                    let o_k = layer_q_dim as u32;
                    let (o_pipeline, rows_per_tg, threads_per_tg) = match layer.wo.format {
                        crate::QuantFormat::Q4_KF => {
                            use crate::metal::shaders::q4kf_qkv_proj as kf;
                            (&self.q4kf_proj_pipeline, kf::ROWS_PER_TG, kf::THREADS_PER_TG)
                        }
                        crate::QuantFormat::Q6_K => {
                            use crate::metal::shaders::q6k_matvec as q6k;
                            (&self.q6k_matvec_pipeline, q6k::ROWS_PER_TG, q6k::THREADS_PER_TG)
                        }
                        crate::QuantFormat::Q8_0Gguf => {
                            use crate::metal::shaders::q8_0_gguf_matvec as q8gm;
                            (&self.q8_0_gguf_matvec_pipeline, q8gm::ROWS_PER_TG, q8gm::THREADS_PER_TG)
                        }
                        _ => {
                            use crate::metal::shaders::q4k_qkv_proj as k;
                            (&self.q4k_proj_pipeline, k::ROWS_PER_TG, k::THREADS_PER_TG)
                        }
                    };
                    let num_tgs = (hidden as u64).div_ceil(rows_per_tg);
                    enc.set_compute_pipeline_state(o_pipeline);
                    enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                    enc.set_buffer(1, Some(&attn_out), 0);
                    enc.set_buffer(2, Some(&o_out_buf), 0);
                    enc.set_bytes(3, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &o_k as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs, 1, 1),
                        MTLSize::new(threads_per_tg, 1, 1),
                    );
                } else {
                    let o_q8 = &o_q8_scratch;
                    let o_q8s = &o_q8s_scratch;
                    let dim_val = layer_q_dim as u32;
                    let blocks = (layer_q_dim / 32) as u32;
                    enc.set_compute_pipeline_state(&self.q8_quant_pipeline);
                    enc.set_buffer(0, Some(&attn_out), 0);
                    enc.set_buffer(1, Some(&o_q8), 0);
                    enc.set_buffer(2, Some(&o_q8s), 0);
                    enc.set_bytes(3, 4, &dim_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(blocks as u64, 1, 1), MTLSize::new(256.min(blocks as u64), 1, 1));
                    let o_rows = hidden as u32;
                    let o_k = layer_q_dim as u32;
                    enc.set_compute_pipeline_state(&self.q8_matvec_pipeline);
                    enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                    enc.set_buffer(1, Some(&o_q8), 0);
                    enc.set_buffer(2, Some(&wo_scale_bufs[l]), 0);
                    enc.set_buffer(3, Some(&o_q8s), 0);
                    enc.set_buffer(4, Some(&o_out_buf), 0);
                    enc.set_bytes(5, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &o_k as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new((hidden as u64).div_ceil(8), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                }
            }

            // ── Step 5b: Residual + norm (format-aware) ──
            let ffn_uses_q4k = layer.gate.format == crate::QuantFormat::Q4_K
                || layer.gate.format == crate::QuantFormat::Q4_KF
                || layer.gate.format == crate::QuantFormat::Q6_K;

            let has_post_norms = layer.has_post_norms;
            if has_post_norms {
                let normed_o = &normed_scratch;
                {
                    use crate::metal::ops::full_pipeline::encode_rms_norm;
                    encode_rms_norm(enc, &self.rms_norm_pipeline,
                        &o_out_buf, &post_attn_norm_bufs[l], &normed_o, hidden, eps, norm_offset);
                }
                let pre_ffn_buf = if let Some(pfn) = layer.pre_ffn_norm {
                    self.bufs.transient_from_f32(pfn)
                } else {
                    post_attn_norm_bufs[l].clone()
                };
                if ffn_uses_q4k || layer.is_moe_layer {
                    enc.set_compute_pipeline_state(&self.residual_norm_pipeline);
                    enc.set_buffer(0, Some(&h_buf), 0);
                    enc.set_buffer(1, Some(&normed_o), 0);
                    enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(3, Some(&ffn_norm_out), 0);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    encode_residual_add(enc, &self.residual_add_pipeline,
                        &h_buf, &normed_o, &h_post_attn, hidden);
                } else {
                    enc.set_compute_pipeline_state(&self.residual_norm_q8_pipeline);
                    enc.set_buffer(0, Some(&h_buf), 0);
                    enc.set_buffer(1, Some(&normed_o), 0);
                    enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(3, Some(&ffn_q8), 0);
                    enc.set_buffer(4, Some(&ffn_q8s), 0);
                    enc.set_buffer(5, Some(&h_post_attn), 0);
                    enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                }
            } else if ffn_uses_q4k || layer.is_moe_layer {
                enc.set_compute_pipeline_state(&self.residual_norm_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&o_out_buf), 0);
                enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0);
                enc.set_buffer(3, Some(&ffn_norm_out), 0);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                use crate::metal::ops::full_pipeline::encode_residual_add;
                encode_residual_add(enc, &self.residual_add_pipeline,
                    &h_buf, &o_out_buf, &h_post_attn, hidden);
            } else {
                enc.set_compute_pipeline_state(&self.residual_norm_q8_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&o_out_buf), 0);
                enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0);
                enc.set_buffer(3, Some(&ffn_q8), 0);
                enc.set_buffer(4, Some(&ffn_q8s), 0);
                enc.set_buffer(5, Some(&h_post_attn), 0);
                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
            }

            // ── KNN probe: snapshot h_post_attn at the probe layer ──
            if let Some(pl) = probe_layer {
                if l == pl {
                    if let Some(ref pb) = probe_buf {
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        let zero_vec = vec![0.0f32; hidden];
                        let zero_buf = self.bufs.transient_from_f32(&zero_vec);
                        encode_residual_add(enc, &self.residual_add_pipeline,
                            &h_post_attn, &zero_buf, pb, hidden);
                    }
                }
            }

            // ── Step 6: FFN (MoE or dense) ──
            if layer.is_moe_layer && layer.router_weight.is_some() {
                // ── MoE FFN path ──
                let num_active = layer.num_active_experts;
                let expert_inter = layer.expert_intermediate;
                let expert_inter_2 = expert_inter * 2; // fused gate+up

                // Create per-layer MoE buffers on demand (avoid 16+ GB upfront registration)
                let router_buf = self.bufs.transient_from_f32(layer.router_weight.unwrap());
                let expert_gu_buf = layer.expert_gate_up.as_ref()
                    .map(|w| self.bufs.get_bytes(w.data))
                    .unwrap_or_else(|| self.bufs.output(4));
                let expert_down_buf = layer.expert_down.as_ref()
                    .map(|w| self.bufs.get_bytes(w.data))
                    .unwrap_or_else(|| self.bufs.output(4));

                // A) Router: sgemm_transb(ffn_norm_out[1, hidden], router[num_experts, hidden]) → scores[num_experts]
                {
                    let m_val = 1u32;
                    let n_val = layer.num_experts as u32;
                    let k_val = hidden_val;
                    crate::metal::f32_ops::F32Ops::encode_static(
                        &self.f32_ops.transb_pipeline, enc,
                        &ffn_norm_out, &router_buf, &router_out,
                        m_val as usize, n_val as usize, k_val as usize,
                    );
                }

                // B) Read back router scores — requires command buffer boundary
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();

                let scores = super::buffers::read_buffer_f32(&router_out, layer.num_experts);

                // C) Top-K selection + softmax on CPU
                let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top_k_experts: Vec<(usize, f32)> = indexed.into_iter().take(num_active).collect();

                let max_score = top_k_experts.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = top_k_experts.iter().map(|(_, s)| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let softmax_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

                // D) New command buffer for expert dispatches
                let cmd2 = self.queue.new_command_buffer();
                let enc2 = cmd2.new_compute_command_encoder();

                // Zero the MoE accumulator
                {
                    let zero_vec = vec![0.0f32; hidden];
                    let zero_buf = self.bufs.transient_from_f32(&zero_vec);
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    encode_residual_add(enc2, &self.residual_add_pipeline,
                        &zero_buf, &zero_buf, &moe_accum, hidden);
                }

                // E) Dispatch each selected expert
                // Gate+up bytes per expert — format-aware block size
                let gu_format = layer.expert_gate_up.as_ref().map(|w| w.format).unwrap_or(crate::QuantFormat::Q4_K);
                let q4k_blocks_per_row = (hidden + 255) / 256;
                let q4k_block_bytes: usize = match gu_format {
                    crate::QuantFormat::Q4_KF => 144,  // GGUF Q4_K layout
                    _ => 148,                           // larql/Ollama Q4_K layout
                };
                let q4k_bytes_per_row = q4k_blocks_per_row * q4k_block_bytes;
                let q4k_bytes_per_expert = expert_inter_2 * q4k_bytes_per_row;

                // Q8_0 GGUF down bytes per expert
                let q8_blocks_per_row = (expert_inter + 31) / 32;
                let q8_bytes_per_row = q8_blocks_per_row * 34;
                let q8_bytes_per_expert = hidden * q8_bytes_per_row;

                for (idx, &(expert_id, _)) in top_k_experts.iter().enumerate() {
                    let weight = softmax_weights[idx];

                    // Gate+up: Q4_K matvec at offset into expert_gate_up
                    let gu_offset = (expert_id * q4k_bytes_per_expert) as u64;
                    {
                        let n_val = expert_inter_2 as u32;
                        let k_val = hidden_val;
                        if gu_format == crate::QuantFormat::Q4_KF {
                            // GGUF 144-byte Q4_K layout — use q4kf_proj shader
                            use crate::metal::shaders::q4kf_qkv_proj as q4kf;
                            let num_tgs = (expert_inter_2 as u64).div_ceil(q4kf::ROWS_PER_TG);
                            enc2.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                            enc2.set_buffer(0, Some(&expert_gu_buf), gu_offset);
                            enc2.set_buffer(1, Some(&ffn_norm_out), 0);
                            enc2.set_buffer(2, Some(&expert_gu_out), 0);
                            enc2.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
                            enc2.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
                            enc2.dispatch_thread_groups(
                                MTLSize::new(num_tgs, 1, 1),
                                MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
                            );
                        } else {
                            // larql/Ollama 148-byte Q4_K layout
                            use crate::metal::shaders::q4k_matvec as q4k;
                            let num_tgs = (expert_inter_2 as u64).div_ceil(q4k::ROWS_PER_TG);
                            enc2.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                            enc2.set_buffer(0, Some(&expert_gu_buf), gu_offset);
                            enc2.set_buffer(1, Some(&ffn_norm_out), 0);
                            enc2.set_buffer(2, Some(&expert_gu_out), 0);
                            enc2.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
                            enc2.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
                            enc2.dispatch_thread_groups(
                                MTLSize::new(num_tgs, 1, 1),
                                MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
                            );
                        }
                    }

                    // GEGLU: gelu_tanh(gate[0..expert_inter]) * up[expert_inter..expert_inter_2]
                    {
                        let inter_u32 = expert_inter as u32;
                        let geglu = match layer.activation {
                            crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                            _ => &self.geglu_pipeline,
                        };
                        enc2.set_compute_pipeline_state(geglu);
                        enc2.set_buffer(0, Some(&expert_gu_out), 0); // gate (first half)
                        enc2.set_buffer(1, Some(&expert_gu_out), (expert_inter * 4) as u64); // up (second half)
                        enc2.set_buffer(2, Some(&expert_act_buf), 0);
                        enc2.set_bytes(3, 4, &inter_u32 as *const u32 as *const std::ffi::c_void);
                        enc2.dispatch_threads(
                            MTLSize::new(expert_inter as u64, 1, 1),
                            MTLSize::new(256.min(expert_inter as u64), 1, 1),
                        );
                    }

                    // Down: Q8_0 GGUF matvec at offset into expert_down
                    let down_offset = (expert_id * q8_bytes_per_expert) as u64;
                    {
                        use crate::metal::shaders::q8_0_gguf_matvec as q8gm;
                        let n_val = hidden_val;
                        let k_val = expert_inter as u32;
                        let num_tgs = (hidden as u64).div_ceil(q8gm::ROWS_PER_TG);
                        enc2.set_compute_pipeline_state(&self.q8_0_gguf_matvec_pipeline);
                        enc2.set_buffer(0, Some(&expert_down_buf), down_offset);
                        enc2.set_buffer(1, Some(&expert_act_buf), 0);
                        enc2.set_buffer(2, Some(&expert_down_out_buf), 0);
                        enc2.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
                        enc2.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
                        enc2.dispatch_thread_groups(
                            MTLSize::new(num_tgs, 1, 1),
                            MTLSize::new(q8gm::THREADS_PER_TG, 1, 1),
                        );
                    }

                    // Weighted accumulate: moe_accum += weight * expert_down_out
                    {
                        // Scale expert_down_out in-place
                        let weight_val = weight;
                        enc2.set_compute_pipeline_state(&self.scale_vector_pipeline);
                        enc2.set_buffer(0, Some(&expert_down_out_buf), 0);
                        enc2.set_buffer(1, Some(&expert_down_out_buf), 0);
                        enc2.set_bytes(2, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc2.set_bytes(3, 4, &weight_val as *const f32 as *const std::ffi::c_void);
                        enc2.dispatch_threads(
                            MTLSize::new(hidden as u64, 1, 1),
                            MTLSize::new(256.min(hidden as u64), 1, 1),
                        );
                        // Add to accumulator
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        encode_residual_add(enc2, &self.residual_add_pipeline,
                            &moe_accum, &expert_down_out_buf, &moe_accum, hidden);
                    }
                }

                // F) Shared expert (standard dense FFN using gate/up/down from the layer)
                {
                    use crate::metal::shaders::q4k_matvec as q4k;
                    // Gate
                    let n_tgs = (inter as u64).div_ceil(q4k::ROWS_PER_TG);
                    enc2.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                    enc2.set_buffer(0, Some(&gate_bufs[l]), 0);
                    enc2.set_buffer(1, Some(&ffn_norm_out), 0);
                    enc2.set_buffer(2, Some(&gate_out_scratch), 0);
                    enc2.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc2.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc2.dispatch_thread_groups(
                        MTLSize::new(n_tgs, 1, 1),
                        MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
                    );
                    // Up
                    enc2.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                    enc2.set_buffer(0, Some(&up_bufs[l]), 0);
                    enc2.set_buffer(1, Some(&ffn_norm_out), 0);
                    enc2.set_buffer(2, Some(&up_out), 0);
                    enc2.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc2.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc2.dispatch_thread_groups(
                        MTLSize::new(n_tgs, 1, 1),
                        MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
                    );
                }
                // GEGLU for shared expert
                {
                    let geglu = match layer.activation {
                        crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                        _ => &self.geglu_pipeline,
                    };
                    enc2.set_compute_pipeline_state(geglu);
                    enc2.set_buffer(0, Some(&gate_out_scratch), 0);
                    enc2.set_buffer(1, Some(&up_out), 0);
                    enc2.set_buffer(2, Some(&act_buf), 0);
                    enc2.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc2.dispatch_threads(
                        MTLSize::new(inter as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                }
                // Down for shared expert — check format
                {
                    let down_is_q6k = layer.down.format == crate::QuantFormat::Q6_K;
                    let down_is_q8_gguf = layer.down.format == crate::QuantFormat::Q8_0Gguf;
                    if down_is_q8_gguf {
                        use crate::metal::shaders::q8_0_gguf_matvec as q8gm;
                        let n_val = hidden_val;
                        let k_val = inter_val;
                        let num_tgs = (hidden as u64).div_ceil(q8gm::ROWS_PER_TG);
                        enc2.set_compute_pipeline_state(&self.q8_0_gguf_matvec_pipeline);
                        enc2.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc2.set_buffer(1, Some(&act_buf), 0);
                        enc2.set_buffer(2, Some(&down_out), 0);
                        enc2.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
                        enc2.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
                        enc2.dispatch_thread_groups(
                            MTLSize::new(num_tgs, 1, 1),
                            MTLSize::new(q8gm::THREADS_PER_TG, 1, 1),
                        );
                    } else if down_is_q6k {
                        use crate::metal::shaders::q6k_matvec as q6k;
                        let n_tgs = (hidden as u64).div_ceil(q6k::ROWS_PER_TG);
                        enc2.set_compute_pipeline_state(&self.q6k_matvec_pipeline);
                        enc2.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc2.set_buffer(1, Some(&act_buf), 0);
                        enc2.set_buffer(2, Some(&down_out), 0);
                        enc2.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc2.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc2.dispatch_thread_groups(
                            MTLSize::new(n_tgs, 1, 1),
                            MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
                        );
                    } else {
                        use crate::metal::shaders::q4k_matvec as q4k;
                        let n_tgs = (hidden as u64).div_ceil(q4k::ROWS_PER_TG);
                        enc2.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                        enc2.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc2.set_buffer(1, Some(&act_buf), 0);
                        enc2.set_buffer(2, Some(&down_out), 0);
                        enc2.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc2.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc2.dispatch_thread_groups(
                            MTLSize::new(n_tgs, 1, 1),
                            MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
                        );
                    }
                }
                // G) Add shared expert output to MoE accumulator → down_out
                {
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    encode_residual_add(enc2, &self.residual_add_pipeline,
                        &moe_accum, &down_out, &down_out, hidden);
                }

                // ── Step 7: Post-FFN residual ──
                if has_post_norms {
                    if let Some(post_ffn) = layer.post_ffn_norm {
                        let post_ffn_buf = self.bufs.transient_from_f32(post_ffn);
                        let normed_ffn = &normed_scratch;
                        use crate::metal::ops::full_pipeline::encode_rms_norm;
                        encode_rms_norm(enc2, &self.rms_norm_pipeline,
                            &down_out, &post_ffn_buf, &normed_ffn, hidden, eps, norm_offset);
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        encode_residual_add(enc2, &self.residual_add_pipeline,
                            &h_post_attn, &normed_ffn, &new_h, hidden);
                    } else {
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        encode_residual_add(enc2, &self.residual_add_pipeline,
                            &h_post_attn, &down_out, &new_h, hidden);
                    }
                } else {
                    let len_val = hidden as u32;
                    enc2.set_compute_pipeline_state(&self.residual_add_pipeline);
                    enc2.set_buffer(0, Some(&h_post_attn), 0);
                    enc2.set_buffer(1, Some(&down_out), 0);
                    enc2.set_buffer(2, Some(&new_h), 0);
                    enc2.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                    enc2.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                }

                // ── Step 8: Optional layer scalar ──
                if layer.layer_scalar != 0.0 {
                    let scaled = &scaled_scratch;
                    let scalar_val = layer.layer_scalar;
                    enc2.set_compute_pipeline_state(&self.scale_vector_pipeline);
                    enc2.set_buffer(0, Some(&new_h), 0);
                    enc2.set_buffer(1, Some(&scaled), 0);
                    enc2.set_bytes(2, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc2.set_bytes(3, 4, &scalar_val as *const f32 as *const std::ffi::c_void);
                    enc2.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                    h_buf = scaled;
                } else {
                    h_buf = new_h;
                }

                enc2.end_encoding();
                cmd2.commit();
                cmd2.wait_until_completed();
            } else {
                // ── Dense FFN path (same as decode_token_inner) ──
                {
                    let ffn_is_q4kf = layer.gate.format == crate::QuantFormat::Q4_KF;
                    if ffn_is_q4kf {
                        use crate::metal::shaders::q4kf_qkv_proj as q4kf;
                        use crate::metal::shaders::q4kf_ffn_gate_up as q4kf_gu;
                        use crate::metal::shaders::q6k_matvec as q6k_mv;
                        let down_is_q6k = layer.down.format == crate::QuantFormat::Q6_K;
                        let n_tgs_down = if down_is_q6k {
                            (hidden as u64).div_ceil(q6k_mv::ROWS_PER_TG)
                        } else {
                            (hidden as u64).div_ceil(q4kf::ROWS_PER_TG)
                        };
                        let down_threads = if down_is_q6k { q6k_mv::THREADS_PER_TG } else { q4kf::THREADS_PER_TG };
                        let down_pipeline = if down_is_q6k {
                            &self.q6k_matvec_pipeline
                        } else {
                            &self.q4kf_proj_pipeline
                        };
                        if layer.is_gated() {
                            let gate_out = &gate_out_scratch;
                            let n_tgs_per_mat = (inter as u64).div_ceil(q4kf_gu::ROWS_PER_TG);
                            enc.set_compute_pipeline_state(&self.q4kf_ffn_gate_up_pipeline);
                            enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                            enc.set_buffer(1, Some(&up_bufs[l]), 0);
                            enc.set_buffer(2, Some(&ffn_norm_out), 0);
                            enc.set_buffer(3, Some(&gate_out), 0);
                            enc.set_buffer(4, Some(&up_out), 0);
                            enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(
                                MTLSize::new(n_tgs_per_mat * 2, 1, 1),
                                MTLSize::new(q4kf_gu::THREADS_PER_TG, 1, 1),
                            );
                            let geglu = match layer.activation {
                                crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                                _ => &self.geglu_pipeline,
                            };
                            enc.set_compute_pipeline_state(geglu);
                            enc.set_buffer(0, Some(&gate_out), 0);
                            enc.set_buffer(1, Some(&up_out), 0);
                            enc.set_buffer(2, Some(&act_buf), 0);
                            enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                            enc.set_compute_pipeline_state(down_pipeline);
                            enc.set_buffer(0, Some(&down_bufs[l]), 0);
                            enc.set_buffer(1, Some(&act_buf), 0);
                            enc.set_buffer(2, Some(&down_out), 0);
                            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(down_threads, 1, 1));
                        } else {
                            let n_tgs_up = (inter as u64).div_ceil(q4kf::ROWS_PER_TG);
                            enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                            enc.set_buffer(0, Some(&up_bufs[l]), 0);
                            enc.set_buffer(1, Some(&ffn_norm_out), 0);
                            enc.set_buffer(2, Some(&up_out), 0);
                            enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(MTLSize::new(n_tgs_up, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
                            let activation_pipeline = match layer.activation {
                                crate::Activation::GeluTanh => &self.gelu_tanh_pipeline,
                                _ => &self.silu_pipeline,
                            };
                            enc.set_compute_pipeline_state(activation_pipeline);
                            enc.set_buffer(0, Some(&up_out), 0);
                            enc.set_buffer(1, Some(&act_buf), 0);
                            enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                            enc.set_compute_pipeline_state(down_pipeline);
                            enc.set_buffer(0, Some(&down_bufs[l]), 0);
                            enc.set_buffer(1, Some(&act_buf), 0);
                            enc.set_buffer(2, Some(&down_out), 0);
                            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
                        }
                    } else if ffn_uses_q4k {
                        use crate::metal::shaders::q4k_matvec as q4k;
                        use crate::metal::shaders::q6k_matvec as q6k;
                        use crate::metal::shaders::q4k_ffn_gate_up as q4k_gu;
                        let down_is_q6k = layer.down.format == crate::QuantFormat::Q6_K;
                        let n_tgs_down = if down_is_q6k {
                            (hidden as u64).div_ceil(q6k::ROWS_PER_TG)
                        } else {
                            (hidden as u64).div_ceil(q4k::ROWS_PER_TG)
                        };
                        if layer.is_gated() {
                            let gate_out = &gate_out_scratch;
                            let gate_is_q6k = layer.gate.format == crate::QuantFormat::Q6_K;
                            let up_is_q6k = layer.up.format == crate::QuantFormat::Q6_K;
                            let n_tgs_per_mat = (inter as u64).div_ceil(q4k_gu::ROWS_PER_TG);
                            let separate_gate_up = std::env::var("LARQL_SEPARATE_GATE_UP").ok().as_deref() == Some("1")
                                || gate_is_q6k || up_is_q6k;
                            if separate_gate_up {
                                let dispatch_single = |enc_ref: &metal::ComputeCommandEncoderRef,
                                                        w_buf: &metal::Buffer,
                                                        out_buf: &metal::Buffer,
                                                        is_q6k: bool| {
                                    if is_q6k {
                                        let n_tgs = (inter as u64).div_ceil(q6k::ROWS_PER_TG);
                                        enc_ref.set_compute_pipeline_state(&self.q6k_matvec_pipeline);
                                        enc_ref.set_buffer(0, Some(w_buf), 0);
                                        enc_ref.set_buffer(1, Some(&ffn_norm_out), 0);
                                        enc_ref.set_buffer(2, Some(out_buf), 0);
                                        enc_ref.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                                        enc_ref.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                                        enc_ref.dispatch_thread_groups(MTLSize::new(n_tgs, 1, 1), MTLSize::new(q6k::THREADS_PER_TG, 1, 1));
                                    } else {
                                        let n_tgs = (inter as u64).div_ceil(q4k::ROWS_PER_TG);
                                        enc_ref.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                                        enc_ref.set_buffer(0, Some(w_buf), 0);
                                        enc_ref.set_buffer(1, Some(&ffn_norm_out), 0);
                                        enc_ref.set_buffer(2, Some(out_buf), 0);
                                        enc_ref.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                                        enc_ref.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                                        enc_ref.dispatch_thread_groups(MTLSize::new(n_tgs, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
                                    }
                                };
                                dispatch_single(enc, &gate_bufs[l], &gate_out, gate_is_q6k);
                                dispatch_single(enc, &up_bufs[l], &up_out, up_is_q6k);
                            } else {
                                enc.set_compute_pipeline_state(&self.q4k_ffn_gate_up_pipeline);
                                enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                                enc.set_buffer(1, Some(&up_bufs[l]), 0);
                                enc.set_buffer(2, Some(&ffn_norm_out), 0);
                                enc.set_buffer(3, Some(&gate_out), 0);
                                enc.set_buffer(4, Some(&up_out), 0);
                                enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                                enc.dispatch_thread_groups(
                                    MTLSize::new(n_tgs_per_mat * 2, 1, 1),
                                    MTLSize::new(q4k_gu::THREADS_PER_TG, 1, 1),
                                );
                            }
                            let skip_down = std::env::var("LARQL_SKIP_DOWN").ok().as_deref() == Some("1");
                            if skip_down {
                                let zero_vec = vec![0.0f32; hidden];
                                let zero_buf = self.bufs.transient_from_f32(&zero_vec);
                                use crate::metal::ops::full_pipeline::encode_residual_add;
                                encode_residual_add(enc, &self.residual_add_pipeline,
                                    &zero_buf, &zero_buf, &down_out, hidden);
                            } else if !down_is_q6k {
                                use crate::metal::shaders::q4k_geglu_down as q4k_gd;
                                let geglu_down_pipeline = match layer.activation {
                                    crate::Activation::GeluTanh => &self.q4k_geglu_gelu_tanh_down_pipeline,
                                    _ => &self.q4k_geglu_silu_down_pipeline,
                                };
                                let n_tgs_geglu = (hidden as u64).div_ceil(q4k_gd::ROWS_PER_TG);
                                enc.set_compute_pipeline_state(geglu_down_pipeline);
                                enc.set_buffer(0, Some(&down_bufs[l]), 0);
                                enc.set_buffer(1, Some(&gate_out), 0);
                                enc.set_buffer(2, Some(&up_out), 0);
                                enc.set_buffer(3, Some(&down_out), 0);
                                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                                enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                                enc.dispatch_thread_groups(
                                    MTLSize::new(n_tgs_geglu, 1, 1),
                                    MTLSize::new(q4k_gd::THREADS_PER_TG, 1, 1),
                                );
                            } else {
                                let geglu = match layer.activation {
                                    crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                                    _ => &self.geglu_pipeline,
                                };
                                enc.set_compute_pipeline_state(geglu);
                                enc.set_buffer(0, Some(&gate_out), 0);
                                enc.set_buffer(1, Some(&up_out), 0);
                                enc.set_buffer(2, Some(&act_buf), 0);
                                enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                                enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                                enc.set_compute_pipeline_state(&self.q6k_matvec_pipeline);
                                enc.set_buffer(0, Some(&down_bufs[l]), 0);
                                enc.set_buffer(1, Some(&act_buf), 0);
                                enc.set_buffer(2, Some(&down_out), 0);
                                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                                enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                                enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q6k::THREADS_PER_TG, 1, 1));
                            }
                        } else {
                            let n_tgs_up = (inter as u64).div_ceil(q4k::ROWS_PER_TG);
                            enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                            enc.set_buffer(0, Some(&up_bufs[l]), 0);
                            enc.set_buffer(1, Some(&ffn_norm_out), 0);
                            enc.set_buffer(2, Some(&up_out), 0);
                            enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(MTLSize::new(n_tgs_up, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
                            let activation_pipeline = match layer.activation {
                                crate::Activation::GeluTanh => &self.gelu_tanh_pipeline,
                                _ => &self.silu_pipeline,
                            };
                            enc.set_compute_pipeline_state(activation_pipeline);
                            enc.set_buffer(0, Some(&up_out), 0);
                            enc.set_buffer(1, Some(&act_buf), 0);
                            enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                            enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                            enc.set_buffer(0, Some(&down_bufs[l]), 0);
                            enc.set_buffer(1, Some(&act_buf), 0);
                            enc.set_buffer(2, Some(&down_out), 0);
                            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
                        }
                    } else {
                        // Q4_0 FFN path
                        use crate::metal::shaders::q4_matvec as q4mv;
                        let n_tgs_ffn = (inter as u64).div_ceil(q4mv::ROWS_PER_TG);
                        if layer.is_gated() {
                            let gate_out = &gate_out_scratch;
                            enc.set_compute_pipeline_state(&self.q4.matvec);
                            enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                            enc.set_buffer(1, Some(&ffn_q8), 0);
                            enc.set_buffer(2, Some(&ffn_q8s), 0);
                            enc.set_buffer(3, Some(&gate_out), 0);
                            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                            enc.set_buffer(0, Some(&up_bufs[l]), 0);
                            enc.set_buffer(3, Some(&up_out), 0);
                            enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                            let geglu = match layer.activation {
                                crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                                _ => &self.geglu_pipeline,
                            };
                            enc.set_compute_pipeline_state(geglu);
                            enc.set_buffer(0, Some(&gate_out), 0);
                            enc.set_buffer(1, Some(&up_out), 0);
                            enc.set_buffer(2, Some(&act_buf), 0);
                            enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        } else {
                            enc.set_compute_pipeline_state(&self.q4.matvec);
                            enc.set_buffer(0, Some(&up_bufs[l]), 0);
                            enc.set_buffer(1, Some(&ffn_q8), 0);
                            enc.set_buffer(2, Some(&ffn_q8s), 0);
                            enc.set_buffer(3, Some(&up_out), 0);
                            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                            let activation_pipeline = match layer.activation {
                                crate::Activation::GeluTanh => &self.gelu_tanh_pipeline,
                                _ => &self.silu_pipeline,
                            };
                            enc.set_compute_pipeline_state(activation_pipeline);
                            enc.set_buffer(0, Some(&up_out), 0);
                            enc.set_buffer(1, Some(&act_buf), 0);
                            enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        }
                        enc.set_compute_pipeline_state(&self.q4.f32_matvec);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
                    }
                }

                // ── Step 7: Post-FFN residual (dense path) ──
                if has_post_norms {
                    if let Some(post_ffn) = layer.post_ffn_norm {
                        let post_ffn_buf = self.bufs.transient_from_f32(post_ffn);
                        let normed_ffn = &normed_scratch;
                        use crate::metal::ops::full_pipeline::encode_rms_norm;
                        encode_rms_norm(enc, &self.rms_norm_pipeline,
                            &down_out, &post_ffn_buf, &normed_ffn, hidden, eps, norm_offset);
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        encode_residual_add(enc, &self.residual_add_pipeline,
                            &h_post_attn, &normed_ffn, &new_h, hidden);
                    } else {
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        encode_residual_add(enc, &self.residual_add_pipeline,
                            &h_post_attn, &down_out, &new_h, hidden);
                    }
                } else {
                    let len_val = hidden as u32;
                    enc.set_compute_pipeline_state(&self.residual_add_pipeline);
                    enc.set_buffer(0, Some(&h_post_attn), 0);
                    enc.set_buffer(1, Some(&down_out), 0);
                    enc.set_buffer(2, Some(&new_h), 0);
                    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                }

                // ── Step 8: Optional layer scalar ──
                if layer.layer_scalar != 0.0 {
                    let scaled = &scaled_scratch;
                    let scalar_val = layer.layer_scalar;
                    enc.set_compute_pipeline_state(&self.scale_vector_pipeline);
                    enc.set_buffer(0, Some(&new_h), 0);
                    enc.set_buffer(1, Some(&scaled), 0);
                    enc.set_bytes(2, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(3, 4, &scalar_val as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                    h_buf = scaled;
                } else {
                    h_buf = new_h;
                }

                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
            if _trace_moe {
                eprintln!("[moe-decode] layer {l}/{num_layers} done in {:.1}ms (moe={})",
                    _t_layer.elapsed().as_secs_f64() * 1000.0, layer.is_moe_layer);
            }
        } // end per-layer loop

        let final_h = super::buffers::read_buffer_f32(&h_buf, hidden);
        let probe_h = probe_buf.as_ref().map(|pb| super::buffers::read_buffer_f32(pb, hidden));
        (final_h, probe_h)
    }

    /// Batch-decode K tokens through all layers in ONE Metal command buffer.
    ///
    /// For each layer: dispatches K copies of each operation (norm, QKV, RoPE,
    /// FFN, etc.) using buffer offsets, then ONE batched KV append + attend.
    /// All K × 34-layers of dispatches share one encoder + one commit + one wait.
    ///
    /// This is the core enabler for speculative decoding parallel verification:
    /// at 100% acceptance with K=4, gives 5 tokens per 24ms = 200+ tok/s.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token_batch(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x_batch: &[f32],     // [K * hidden] — K token embeddings concatenated
        batch_size: usize,    // K
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
    ) -> Vec<f32> {
        use crate::metal::shaders::q4k_matvec as q4k;
        use crate::metal::shaders::q6k_matvec as q6k;
        use crate::metal::shaders::q4k_geglu_down as q4k_gd;

        let k = batch_size;
        let num_layers = layers.len();
        if num_layers == 0 || k == 0 {
            return x_batch.to_vec();
        }

        // ── Pre-cache weight buffers (shared across batch positions) ──
        let wq_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wq.data)).collect();
        let wk_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wk.data)).collect();
        let wv_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wv.data)).collect();
        let wo_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wo.data)).collect();
        let gate_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.gate.data)).collect();
        let up_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.up.data)).collect();
        let down_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.down.data)).collect();
        let input_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.input_norm)).collect();
        let post_attn_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.post_attn_norm)).collect();
        let q_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.q_norm_weight.unwrap_or(&[]))).collect();
        let k_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.k_norm_weight.unwrap_or(&[]))).collect();

        // ── K-sized batch buffers ──
        let b = |n: usize| self.bufs.output((k as u64) * (n as u64) * 4);
        let h_init = self.bufs.transient_from_f32(x_batch);
        let h_a = b(hidden);
        let h_b = b(hidden);
        let mut h_buf = &h_init;
        let norm_buf = b(hidden);
        let q_out = b(q_dim);
        let k_out = b(kv_dim);
        let v_out = b(kv_dim);
        let attn_out = b(q_dim);
        let o_out = b(hidden);
        let h_post_attn = b(hidden);
        let ffn_norm_out = b(hidden);
        let normed_scratch = b(hidden);
        let gate_out = b(inter);
        let up_out = b(inter);
        let down_out = b(hidden);

        let hidden_val = hidden as u32;
        let inter_val = inter as u32;

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        for l in 0..num_layers {
            let layer = &layers[l];
            let norm_offset = layer.norm_offset;
            let eps = layer.eps;
            let scale = layer.attn_scale;
            let layer_hd = layer.head_dim;
            let layer_nq = layer.num_q_heads;
            let layer_nkv = layer.num_kv_heads;
            let layer_rope = layer.rope_base;
            let layer_rotary_dim = if layer.rotary_dim > 0 { layer.rotary_dim } else { layer_hd };
            let layer_q_dim = layer_nq * layer_hd;
            let window_size = layer.sliding_window as u32;

            let new_h = if l % 2 == 0 { &h_a } else { &h_b };

            // ── Phase 1: Input norms for ALL K positions ──
            // Must complete before batch QKV projection reads norm_buf.
            {
                let len_val = hidden as u32;
                enc.set_compute_pipeline_state(&self.rms_norm_pipeline);
                enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                for bi in 0..k {
                    let h_off = (bi * hidden * 4) as u64;
                    let norm_off_bi = (bi * hidden * 4) as u64;
                    enc.set_buffer(0, Some(h_buf), h_off);
                    enc.set_buffer(2, Some(&norm_buf), norm_off_bi);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                }
            }

            // ── Phase 2: Batched QKV projection (shared weight reads) ──
            {
                let k_val = hidden as u32;
                let m_val = k as u32;
                let dispatch_batch_proj = |enc: &metal::ComputeCommandEncoderRef,
                                           w_buf: &metal::Buffer,
                                           in_buf: &metal::Buffer,
                                           out_buf: &metal::Buffer,
                                           rows: usize, fmt: crate::QuantFormat| {
                    let n_val = rows as u32;
                    match fmt {
                        crate::QuantFormat::Q6_K => {
                            let n_tgs = (rows as u64).div_ceil(q6k::ROWS_PER_TG);
                            enc.set_compute_pipeline_state(&self.q6k_matvec_batch_pipeline);
                            enc.set_buffer(0, Some(w_buf), 0);
                            enc.set_buffer(1, Some(in_buf), 0);
                            enc.set_buffer(2, Some(out_buf), 0);
                            enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(5, 4, &m_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(
                                MTLSize::new(n_tgs, m_val as u64, 1),
                                MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
                            );
                        }
                        _ => {
                            let n_tgs = (rows as u64).div_ceil(q4k::ROWS_PER_TG);
                            enc.set_compute_pipeline_state(&self.q4k_matvec_batch_pipeline);
                            enc.set_buffer(0, Some(w_buf), 0);
                            enc.set_buffer(1, Some(in_buf), 0);
                            enc.set_buffer(2, Some(out_buf), 0);
                            enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(5, 4, &m_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(
                                MTLSize::new(n_tgs, m_val as u64, 1),
                                MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
                            );
                        }
                    }
                };
                dispatch_batch_proj(enc, &wq_bufs[l], &norm_buf, &q_out, layer_q_dim, layer.wq.format);
                dispatch_batch_proj(enc, &wk_bufs[l], &norm_buf, &k_out, kv_dim, layer.wk.format);
                dispatch_batch_proj(enc, &wv_bufs[l], &norm_buf, &v_out, kv_dim, layer.wv.format);
            }

            // ── Phase 3: Per-position QK-norm, RoPE, V-norm ──
            for bi in 0..k {
                let q_off = (bi * q_dim * 4) as u64;
                let k_off = (bi * kv_dim * 4) as u64;
                let v_off = (bi * kv_dim * 4) as u64;

                // QK-norm (Gemma 3)
                if layer.q_norm_weight.is_some() {
                    let hd_val = layer_hd as u32;
                    let qk_off_val = layer.qk_norm_offset;
                    let tg_threads = 256u64.min(layer_hd as u64);
                    enc.set_compute_pipeline_state(&self.rms_norm_pipeline);
                    enc.set_bytes(3, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &qk_off_val as *const f32 as *const std::ffi::c_void);
                    enc.set_buffer(1, Some(&q_norm_bufs[l]), 0);
                    for head in 0..layer_nq {
                        let off = q_off + (head * layer_hd * 4) as u64;
                        enc.set_buffer(0, Some(&q_out), off);
                        enc.set_buffer(2, Some(&q_out), off);
                        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg_threads, 1, 1));
                    }
                    enc.set_buffer(1, Some(&k_norm_bufs[l]), 0);
                    for head in 0..layer_nkv {
                        let off = k_off + (head * layer_hd * 4) as u64;
                        enc.set_buffer(0, Some(&k_out), off);
                        enc.set_buffer(2, Some(&k_out), off);
                        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg_threads, 1, 1));
                    }
                }

                // RoPE
                {
                    let pos = (kv_cache.layers[l].current_len + bi) as u32;
                    let hd = layer_hd as u32;
                    let rdim = layer_rotary_dim as u32;
                    let rope_pairs = (layer_rotary_dim / 2) as u64;
                    let num_q = layer_nq as u32;
                    let num_kv = layer_nkv as u32;
                    let freq_scale = layer.rope_freq_scale;
                    enc.set_compute_pipeline_state(&self.rope_at_pos_batched_pipeline);
                    enc.set_buffer(0, Some(&q_out), q_off);
                    enc.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(2, 4, &layer_rope as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(3, 4, &pos as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &rdim as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &num_q as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &freq_scale as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_threads(
                        MTLSize::new(rope_pairs, layer_nq as u64, 1),
                        MTLSize::new(rope_pairs.min(256), 1, 1),
                    );
                    enc.set_buffer(0, Some(&k_out), k_off);
                    enc.set_bytes(5, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(
                        MTLSize::new(rope_pairs, layer_nkv as u64, 1),
                        MTLSize::new(rope_pairs.min(256), 1, 1),
                    );
                }

                // V-norm (if needed)
                if layer.has_v_norm {
                    let hd_val = layer_hd as u32;
                    let num_kv = layer_nkv as u32;
                    enc.set_compute_pipeline_state(&self.v_norm_batched_pipeline);
                    enc.set_buffer(0, Some(&v_out), v_off);
                    enc.set_buffer(1, Some(&v_out), v_off);
                    enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(
                        MTLSize::new(layer_hd as u64, layer_nkv as u64, 1),
                        MTLSize::new((layer_hd as u64).min(256), 1, 1),
                    );
                }
            }

            // ── Step 6: Batched KV append + attend (ONE dispatch for all K) ──
            let cache_len_before = kv_cache.layers[l].current_len;
            ops::kv_cache::encode_kv_append_batch(
                enc, &kv_cache.layers[l],
                &self.kv_append_batch_pipeline,
                &k_out, &v_out, k,
            );
            ops::kv_cache::encode_kv_attend_batch(
                enc, &kv_cache.layers[l],
                &self.kv_attend_batched_pipeline,
                &q_out, &attn_out, k, cache_len_before,
                layer_nq, scale, window_size, layer.softcap,
            );
            kv_cache.layers[l].current_len += k;

            // ── Per-position dispatches (O projection through post-FFN) ──
            for bi in 0..k {
                let _attn_off = (bi * q_dim * 4) as u64;
                let o_off = (bi * hidden * 4) as u64;
                let h_off = (bi * hidden * 4) as u64;
                let ffn_off = (bi * hidden * 4) as u64;
                let gate_off_bi = (bi * inter * 4) as u64;
                let down_off = (bi * hidden * 4) as u64;

                // Step 7: O projection — 2D batched matvec at bi==0
                if bi == 0 {
                    let o_k_val = layer_q_dim as u32;
                    let m_val = k as u32;
                    match layer.wo.format {
                        crate::QuantFormat::Q6_K => {
                            let n_tgs = (hidden as u64).div_ceil(q6k::ROWS_PER_TG);
                            enc.set_compute_pipeline_state(&self.q6k_matvec_batch_pipeline);
                            enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                            enc.set_buffer(1, Some(&attn_out), 0);
                            enc.set_buffer(2, Some(&o_out), 0);
                            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(4, 4, &o_k_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(5, 4, &m_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(MTLSize::new(n_tgs, m_val as u64, 1), MTLSize::new(q6k::THREADS_PER_TG, 1, 1));
                        }
                        _ => {
                            let n_tgs = (hidden as u64).div_ceil(q4k::ROWS_PER_TG);
                            enc.set_compute_pipeline_state(&self.q4k_matvec_batch_pipeline);
                            enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                            enc.set_buffer(1, Some(&attn_out), 0);
                            enc.set_buffer(2, Some(&o_out), 0);
                            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(4, 4, &o_k_val as *const u32 as *const std::ffi::c_void);
                            enc.set_bytes(5, 4, &m_val as *const u32 as *const std::ffi::c_void);
                            enc.dispatch_thread_groups(MTLSize::new(n_tgs, m_val as u64, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
                        }
                    }
                }

                // Step 8: Post-attn norm + residual (Gemma 3 post-norm)
                if layer.has_post_norms {
                    let normed_o_off = (bi * hidden * 4) as u64;
                    let len_val = hidden as u32;

                    // rms_norm(o_out) → normed_scratch
                    enc.set_compute_pipeline_state(&self.rms_norm_pipeline);
                    enc.set_buffer(0, Some(&o_out), o_off);
                    enc.set_buffer(1, Some(&post_attn_norm_bufs[l]), 0);
                    enc.set_buffer(2, Some(&normed_scratch), normed_o_off);
                    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));

                    // pre_ffn_norm(h + normed_o) → ffn_norm_out (fused residual_norm)
                    let pre_ffn_buf = if let Some(pfn) = layer.pre_ffn_norm {
                        self.bufs.transient_from_f32(pfn)
                    } else {
                        post_attn_norm_bufs[l].clone()
                    };
                    enc.set_compute_pipeline_state(&self.residual_norm_pipeline);
                    enc.set_buffer(0, Some(h_buf), h_off);
                    enc.set_buffer(1, Some(&normed_scratch), normed_o_off);
                    enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(3, Some(&ffn_norm_out), ffn_off);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));

                    // h_post_attn = h + normed_o (direct dispatch with offsets)
                    enc.set_compute_pipeline_state(&self.residual_add_pipeline);
                    enc.set_buffer(0, Some(h_buf), h_off);
                    enc.set_buffer(1, Some(&normed_scratch), normed_o_off);
                    enc.set_buffer(2, Some(&h_post_attn), h_off);
                    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                }

                // Step 9: FFN — fused GEGLU+down (Q4_K path)
                let down_is_q6k = layer.down.format == crate::QuantFormat::Q6_K;
                if !down_is_q6k && layer.is_gated() {
                    // Gate + Up (per-position)
                    let n_tgs_inter = (inter as u64).div_ceil(q4k::ROWS_PER_TG);
                    enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                    enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                    enc.set_buffer(1, Some(&ffn_norm_out), ffn_off);
                    enc.set_buffer(2, Some(&gate_out), gate_off_bi);
                    enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(n_tgs_inter, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));

                    enc.set_buffer(0, Some(&up_bufs[l]), 0);
                    enc.set_buffer(2, Some(&up_out), gate_off_bi);
                    enc.dispatch_thread_groups(MTLSize::new(n_tgs_inter, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));

                    // Fused GEGLU + down
                    let geglu_down = match layer.activation {
                        crate::Activation::GeluTanh => &self.q4k_geglu_gelu_tanh_down_pipeline,
                        _ => &self.q4k_geglu_silu_down_pipeline,
                    };
                    let n_tgs_geglu = (hidden as u64).div_ceil(q4k_gd::ROWS_PER_TG);
                    enc.set_compute_pipeline_state(geglu_down);
                    enc.set_buffer(0, Some(&down_bufs[l]), 0);
                    enc.set_buffer(1, Some(&gate_out), gate_off_bi);
                    enc.set_buffer(2, Some(&up_out), gate_off_bi);
                    enc.set_buffer(3, Some(&down_out), down_off);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(n_tgs_geglu, 1, 1),
                        MTLSize::new(q4k_gd::THREADS_PER_TG, 1, 1),
                    );
                } else {
                    // Fallback: separate GEGLU + down (Q6_K or non-gated)
                    let n_tgs_inter = (inter as u64).div_ceil(if down_is_q6k { q6k::ROWS_PER_TG } else { q4k::ROWS_PER_TG });
                    let pipeline = if down_is_q6k { &self.q6k_matvec_pipeline } else { &self.q4k_matvec_pipeline };
                    let threads = if down_is_q6k { q6k::THREADS_PER_TG } else { q4k::THREADS_PER_TG };

                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                    enc.set_buffer(1, Some(&ffn_norm_out), ffn_off);
                    enc.set_buffer(2, Some(&gate_out), gate_off_bi);
                    enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(n_tgs_inter, 1, 1), MTLSize::new(threads, 1, 1));

                    enc.set_buffer(0, Some(&up_bufs[l]), 0);
                    enc.set_buffer(2, Some(&up_out), gate_off_bi);
                    enc.dispatch_thread_groups(MTLSize::new(n_tgs_inter, 1, 1), MTLSize::new(threads, 1, 1));

                    let geglu = match layer.activation {
                        crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                        _ => &self.geglu_pipeline,
                    };
                    enc.set_compute_pipeline_state(geglu);
                    enc.set_buffer(0, Some(&gate_out), gate_off_bi);
                    enc.set_buffer(1, Some(&up_out), gate_off_bi);
                    enc.set_buffer(2, Some(&up_out), gate_off_bi);
                    enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));

                    let n_tgs_down = (hidden as u64).div_ceil(if down_is_q6k { q6k::ROWS_PER_TG } else { q4k::ROWS_PER_TG });
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(&down_bufs[l]), 0);
                    enc.set_buffer(1, Some(&up_out), gate_off_bi);
                    enc.set_buffer(2, Some(&down_out), down_off);
                    enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(threads, 1, 1));
                }

                // Step 10: Post-FFN norm + residual → new_h
                let len_val_h = hidden as u32;
                if layer.has_post_norms {
                    if let Some(post_ffn) = layer.post_ffn_norm {
                        let post_ffn_buf = self.bufs.transient_from_f32(post_ffn);
                        let normed_ffn_off = (bi * hidden * 4) as u64;

                        // rms_norm(down_out) → normed_scratch
                        enc.set_compute_pipeline_state(&self.rms_norm_pipeline);
                        enc.set_buffer(0, Some(&down_out), down_off);
                        enc.set_buffer(1, Some(&post_ffn_buf), 0);
                        enc.set_buffer(2, Some(&normed_scratch), normed_ffn_off);
                        enc.set_bytes(3, 4, &len_val_h as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));

                        // new_h = h_post_attn + normed_ffn
                        enc.set_compute_pipeline_state(&self.residual_add_pipeline);
                        enc.set_buffer(0, Some(&h_post_attn), h_off);
                        enc.set_buffer(1, Some(&normed_scratch), normed_ffn_off);
                        enc.set_buffer(2, Some(new_h), h_off);
                        enc.set_bytes(3, 4, &len_val_h as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                    }
                } else {
                    // Pre-norm: new_h = h_post_attn + down_out
                    enc.set_compute_pipeline_state(&self.residual_add_pipeline);
                    enc.set_buffer(0, Some(&h_post_attn), h_off);
                    enc.set_buffer(1, Some(&down_out), down_off);
                    enc.set_buffer(2, Some(new_h), h_off);
                    enc.set_bytes(3, 4, &len_val_h as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                }
            }

            h_buf = new_h;
        }

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        super::buffers::read_buffer_f32(h_buf, k * hidden)
    }
}
