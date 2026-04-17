use super::*;

// ── ComputeBackend trait implementation ──

impl ComputeBackend for MetalBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul(&self.queue, &self.bufs, a, b, self.flop_threshold.load(Ordering::Relaxed))
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul_transb(&self.queue, &self.bufs, a, b, self.flop_threshold.load(Ordering::Relaxed))
    }

    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter().map(|op| {
            if op.transpose_b { self.matmul_transb(op.a.view(), op.b.view()) }
            else { self.matmul(op.a.view(), op.b.view()) }
        }).collect()
    }

    fn matmul_transb_triple_share_a(
        &self,
        a: ArrayView2<f32>,
        b_q: ArrayView2<f32>,
        b_k: ArrayView2<f32>,
        b_v: ArrayView2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n_q = b_q.shape()[0];
        let n_k = b_k.shape()[0];
        let n_v = b_v.shape()[0];
        if b_q.shape()[1] != k || b_k.shape()[1] != k || b_v.shape()[1] != k { return None; }

        // Below threshold: fall back to three CPU matmuls (faster than Metal for tiny shapes)
        let total_flops = 2 * m * (n_q + n_k + n_v) * k;
        if total_flops < self.flop_threshold.load(Ordering::Relaxed) {
            return None;
        }

        let a_owned;
        let a_data: &[f32] = match a.as_slice() {
            Some(s) => s,
            None => { a_owned = a.as_standard_layout().into_owned(); a_owned.as_slice().unwrap() }
        };
        let bq_owned;
        let bq_data: &[f32] = match b_q.as_slice() {
            Some(s) => s,
            None => { bq_owned = b_q.as_standard_layout().into_owned(); bq_owned.as_slice().unwrap() }
        };
        let bk_owned;
        let bk_data: &[f32] = match b_k.as_slice() {
            Some(s) => s,
            None => { bk_owned = b_k.as_standard_layout().into_owned(); bk_owned.as_slice().unwrap() }
        };
        let bv_owned;
        let bv_data: &[f32] = match b_v.as_slice() {
            Some(s) => s,
            None => { bv_owned = b_v.as_standard_layout().into_owned(); bv_owned.as_slice().unwrap() }
        };

        let buf_a = self.bufs.get_f32(a_data);
        let buf_q = self.bufs.get_f32(bq_data);
        let buf_k = self.bufs.get_f32(bk_data);
        let buf_v = self.bufs.get_f32(bv_data);
        let buf_out_q = self.bufs.output((m * n_q * 4) as u64);
        let buf_out_k = self.bufs.output((m * n_k * 4) as u64);
        let buf_out_v = self.bufs.output((m * n_v * 4) as u64);

        let pipeline = &self.f32_ops.transb_pipeline;
        let cmd = self.queue.new_command_buffer();
        for (buf_b, buf_c, n_i) in &[(&buf_q, &buf_out_q, n_q), (&buf_k, &buf_out_k, n_k), (&buf_v, &buf_out_v, n_v)] {
            let enc = cmd.new_compute_command_encoder();
            super::f32_ops::F32Ops::encode_static(pipeline, enc, &buf_a, buf_b, buf_c, m, *n_i, k);
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();

        let out_q = Array2::from_shape_vec((m, n_q), super::buffers::read_buffer_f32(&buf_out_q, m * n_q)).ok()?;
        let out_k = Array2::from_shape_vec((m, n_k), super::buffers::read_buffer_f32(&buf_out_k, m * n_k)).ok()?;
        let out_v = Array2::from_shape_vec((m, n_v), super::buffers::read_buffer_f32(&buf_out_v, m * n_v)).ok()?;
        Some((out_q, out_k, out_v))
    }

    fn q4_matvec(
        &self, q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32],
        num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_matvec_direct(q4_data, q8_x, q8_scales, num_rows, hidden))
    }

    fn q4_vecmat(
        &self, activation: &[f32], q4_data: &[u8],
        intermediate: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_vecmat_direct(activation, q4_data, intermediate, hidden))
    }

    fn q4_matvec_pair_batch(
        &self, gate_q4: &[u8], up_q4: &[u8],
        x_matrix: &[f32], seq_len: usize,
        num_rows: usize, hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        Some(self.q4_matvec_pair_batch_direct(gate_q4, up_q4, x_matrix, seq_len, num_rows, hidden))
    }

    fn full_pipeline_q4(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        seq_len: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32, use_qk_norm: bool, softcap: f32,
    ) -> Option<Vec<f32>> {
        let geglu = if layers.first().is_some_and(|l| l.activation == crate::Activation::GeluTanh) {
            &self.geglu_gelu_tanh_pipeline
        } else {
            &self.geglu_pipeline
        };
        Some(ops::full_pipeline::dispatch_full_pipeline(
            &self.queue, &self.bufs, &self.q4,
            geglu,
            &self.geglu_gelu_tanh_pipeline,
            &self.silu_pipeline,
            &self.gelu_tanh_pipeline,
            &self.q8_quant_pipeline,
            Some(&self.fused_attn_pipeline),
            &self.q8_matvec_pipeline,
            &self.q8_qkv_proj_pipeline,
            &self.q4k_matvec_pipeline, &self.q6k_matvec_pipeline,
            &self.rms_norm_pipeline, &self.residual_add_pipeline,
            &self.rms_norm_q8_pipeline, &self.residual_norm_q8_pipeline,
            Some(&self.q4k_qkv_proj_pipeline), Some(&self.q4k_proj_pipeline),
            None, None, // no rope_at_pos or KV cache for standard full_pipeline_q4
            layers, x, hidden, inter, q_dim, kv_dim,
            seq_len, num_q_heads, num_kv_heads, head_dim,
            rope_base, use_qk_norm, softcap,
        ))
    }

    fn multi_layer_q4_ffn(
        &self,
        layers_q4: &[(&[u8], &[u8], &[u8])],
        x: &[f32],
        inter: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(MetalBackend::multi_layer_q4_ffn(self, layers_q4, x, inter, hidden))
    }

    fn q4k_matvec(
        &self, q4k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        use crate::metal::shaders::q4k_matvec as q4k;
        let buf_w = self.bufs.get_bytes(q4k_data);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        let num_tgs = (num_rows as u64).div_ceil(q4k::ROWS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(super::buffers::read_buffer_f32(&buf_out, num_rows))
    }

    fn q4k_matvec_pair(
        &self,
        q4k_a: &[u8], q4k_b: &[u8],
        x: &[f32],
        num_rows: usize, hidden: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        use crate::metal::shaders::q4k_matvec as q4k;
        let buf_a = self.bufs.get_bytes(q4k_a);
        let buf_b = self.bufs.get_bytes(q4k_b);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out_a = self.bufs.output((num_rows * 4) as u64);
        let buf_out_b = self.bufs.output((num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        let num_tgs = (num_rows as u64).div_ceil(q4k::ROWS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        for (buf_w, buf_out) in &[(&buf_a, &buf_out_a), (&buf_b, &buf_out_b)] {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
            enc.set_buffer(0, Some(*buf_w), 0);
            enc.set_buffer(1, Some(&buf_x), 0);
            enc.set_buffer(2, Some(*buf_out), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(num_tgs, 1, 1),
                metal::MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
            );
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();

        Some((
            super::buffers::read_buffer_f32(&buf_out_a, num_rows),
            super::buffers::read_buffer_f32(&buf_out_b, num_rows),
        ))
    }

    fn q4k_ffn_full(
        &self,
        gate_q4k: &[u8],
        up_q4k: &[u8],
        down_q4k: &[u8],
        x: &[f32],
        hidden: usize,
        intermediate: usize,
        activation: &str,
    ) -> Option<Vec<f32>> {
        use crate::metal::shaders::q4k_matvec as q4k;
        use crate::metal::shaders::q4k_geglu_down as q4k_gd;

        let geglu_pipeline = match activation {
            "silu"      => &self.q4k_geglu_silu_down_pipeline,
            "gelu_tanh" => &self.q4k_geglu_gelu_tanh_down_pipeline,
            _ => return None,
        };

        let buf_gate_w = self.bufs.get_bytes(gate_q4k);
        let buf_up_w   = self.bufs.get_bytes(up_q4k);
        let buf_down_w = self.bufs.get_bytes(down_q4k);
        let buf_x      = self.bufs.transient_from_f32(x);
        // gate_out / up_out are GPU-resident scratch — the GEGLU+down encoder
        // reads them directly without a CPU round-trip.
        let buf_gate_out = self.bufs.output((intermediate * 4) as u64);
        let buf_up_out   = self.bufs.output((intermediate * 4) as u64);
        let buf_down_out = self.bufs.output((hidden * 4) as u64);

        let n_inter = intermediate as u32;
        let k_hidden = hidden as u32;
        let n_hidden = hidden as u32;
        let k_inter = intermediate as u32;
        let n_tgs_matvec = (intermediate as u64).div_ceil(q4k::ROWS_PER_TG);
        let n_tgs_geglu = (hidden as u64).div_ceil(q4k_gd::ROWS_PER_TG);

        let cmd = self.queue.new_command_buffer();

        // Encoder 1: gate matvec → buf_gate_out
        for (buf_w, buf_out) in &[(&buf_gate_w, &buf_gate_out), (&buf_up_w, &buf_up_out)] {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
            enc.set_buffer(0, Some(*buf_w), 0);
            enc.set_buffer(1, Some(&buf_x), 0);
            enc.set_buffer(2, Some(*buf_out), 0);
            enc.set_bytes(3, 4, &n_inter as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k_hidden as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(n_tgs_matvec, 1, 1),
                metal::MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
            );
            enc.end_encoding();
        }

        // Encoder 3: GEGLU + down — reads buf_gate_out and buf_up_out directly,
        // no intermediate CPU transfer. Metal's command-buffer scheduler
        // implicitly orders this after encoders 1-2 because they write the
        // buffers this encoder reads.
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(geglu_pipeline);
        enc.set_buffer(0, Some(&buf_down_w), 0);
        enc.set_buffer(1, Some(&buf_gate_out), 0);
        enc.set_buffer(2, Some(&buf_up_out), 0);
        enc.set_buffer(3, Some(&buf_down_out), 0);
        enc.set_bytes(4, 4, &n_hidden as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &k_inter as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(n_tgs_geglu, 1, 1),
            metal::MTLSize::new(q4k_gd::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        Some(super::buffers::read_buffer_f32(&buf_down_out, hidden))
    }

    fn q4k_geglu_down(
        &self,
        down_q4k: &[u8],
        gate: &[f32],
        up: &[f32],
        hidden: usize,
        intermediate: usize,
        activation: &str,
    ) -> Option<Vec<f32>> {
        use crate::metal::shaders::q4k_geglu_down as q4k_gd;
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARMED: AtomicBool = AtomicBool::new(false);

        let pipeline = match activation {
            "silu"      => &self.q4k_geglu_silu_down_pipeline,
            "gelu_tanh" => &self.q4k_geglu_gelu_tanh_down_pipeline,
            _ => return None,
        };
        let buf_w = self.bufs.get_bytes(down_q4k);
        let buf_g = self.bufs.transient_from_f32(gate);
        let buf_u = self.bufs.transient_from_f32(up);
        let buf_out = self.bufs.output((hidden * 4) as u64);
        let n = hidden as u32;
        let k = intermediate as u32;
        let n_tgs = (hidden as u64).div_ceil(q4k_gd::ROWS_PER_TG);

        let dispatch_once = || {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(&buf_w), 0);
            enc.set_buffer(1, Some(&buf_g), 0);
            enc.set_buffer(2, Some(&buf_u), 0);
            enc.set_buffer(3, Some(&buf_out), 0);
            enc.set_bytes(4, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(n_tgs, 1, 1),
                metal::MTLSize::new(q4k_gd::THREADS_PER_TG, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        };

        // First-ever invocation of this pipeline returns garbage on M-series
        // — Metal pipeline specialisation appears not to be fully ready until
        // a second dispatch. One throw-away dispatch on first call only;
        // subsequent calls take the normal single-dispatch path.
        if !WARMED.swap(true, Ordering::Relaxed) {
            dispatch_once();
        }
        dispatch_once();

        Some(super::buffers::read_buffer_f32(&buf_out, hidden))
    }

    fn q6k_matvec(
        &self, q6k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        use crate::metal::shaders::q6k_matvec as q6k;
        let buf_w = self.bufs.get_bytes(q6k_data);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        let num_tgs = (num_rows as u64).div_ceil(q6k::ROWS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.q6k_matvec_pipeline);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(super::buffers::read_buffer_f32(&buf_out, num_rows))
    }

    fn fused_attention_prefill(
        &self,
        q: &[f32], k: &[f32], v: &[f32],
        seq_len: usize,
        num_q: usize, num_kv: usize, head_dim: usize,
        scale: f32, softcap: f32,
    ) -> Option<Vec<f32>> {
        if seq_len == 0 || num_q == 0 || num_kv == 0 || head_dim == 0 { return None; }
        if head_dim > 512 { return None; } // shader threadgroup `tg_q[512]` cap
        if seq_len > 4096 { return None; } // shader `tg_scores[4096]` cap

        let q_elems = seq_len * num_q * head_dim;
        let buf_q = self.bufs.transient_from_f32(q);
        let buf_k = self.bufs.transient_from_f32(k);
        let buf_v = self.bufs.transient_from_f32(v);
        let buf_out = self.bufs.output((q_elems * 4) as u64);

        let seq_val = seq_len as u32;
        let hd_val = head_dim as u32;
        let nq_val = num_q as u32;
        let nkv_val = num_kv as u32;
        let rope_base_dummy: f32 = 10000.0; // unused when skip_rope=1
        let qknorm_val: u32 = 0; // CPU already applied QK-norm
        let skip_rope_val: u32 = 1; // CPU already applied partial RoPE
        let rotary_dim_val: u32 = 0;

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.fused_attn_pipeline);
        enc.set_buffer(0, Some(&buf_q), 0);
        enc.set_buffer(1, Some(&buf_k), 0);
        enc.set_buffer(2, Some(&buf_v), 0);
        enc.set_buffer(3, Some(&buf_out), 0);
        enc.set_bytes(4, 4, &seq_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &hd_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &nq_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &nkv_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(9, 4, &rope_base_dummy as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(10, 4, &qknorm_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(11, 4, &softcap as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(12, 4, &skip_rope_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(13, 4, &rotary_dim_val as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_q as u64, seq_len as u64, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(super::buffers::read_buffer_f32(&buf_out, q_elems))
    }

    fn f32_sparse_matvec(
        &self,
        weights: &[f32],
        x: &[f32],
        indices: &[u32],
        hidden: usize,
    ) -> Option<Vec<f32>> {
        use crate::metal::shaders::f32_sparse_walk as fs;
        let k = indices.len();
        if k == 0 { return Some(Vec::new()); }
        // SAFETY: weights is a mmap'd f32 slice; we only read from it.
        let buf_w = self.bufs.get_bytes(unsafe {
            std::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len() * 4)
        });
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_idx = self.bufs.get_bytes(unsafe {
            std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 4)
        });
        let buf_out = self.bufs.output((k * 4) as u64);
        let k_u = k as u32;
        let h_u = hidden as u32;
        let num_tgs = (k as u64).div_ceil(fs::THREADS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.f32_sparse_matvec_pipeline);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_idx), 0);
        enc.set_buffer(3, Some(&buf_out), 0);
        enc.set_bytes(4, 4, &k_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &h_u as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(fs::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(super::buffers::read_buffer_f32(&buf_out, k))
    }

    fn f32_sparse_vecmat(
        &self,
        weights: &[f32],
        activation: &[f32],
        indices: &[u32],
        hidden: usize,
    ) -> Option<Vec<f32>> {
        use crate::metal::shaders::f32_sparse_walk as fs;
        let k = indices.len();
        if k == 0 { return Some(vec![0.0; hidden]); }
        let buf_w = self.bufs.get_bytes(unsafe {
            std::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len() * 4)
        });
        let buf_act = self.bufs.transient_from_f32(activation);
        let buf_idx = self.bufs.get_bytes(unsafe {
            std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 4)
        });
        let buf_out = self.bufs.output((hidden * 4) as u64);
        let k_u = k as u32;
        let h_u = hidden as u32;
        let num_tgs = (hidden as u64).div_ceil(fs::THREADS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.f32_sparse_vecmat_pipeline);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_act), 0);
        enc.set_buffer(2, Some(&buf_idx), 0);
        enc.set_buffer(3, Some(&buf_out), 0);
        enc.set_bytes(4, 4, &k_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &h_u as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(fs::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(super::buffers::read_buffer_f32(&buf_out, hidden))
    }

    fn prefill_q4(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        seq_len: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32, use_qk_norm: bool, softcap: f32,
    ) -> Option<Vec<f32>> {
        // Use full_pipeline with KV cache population via separate RoPE + skip_rope=1
        let num_layers = layers.len();
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(num_layers, 4096, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        while kv.layers.len() < num_layers {
            kv.layers.push(ops::kv_cache::LayerKVCache::new(&self.bufs, 4096, num_kv_heads, head_dim));
        }
        let geglu = if layers.first().is_some_and(|l| l.activation == crate::Activation::GeluTanh) {
            &self.geglu_gelu_tanh_pipeline
        } else {
            &self.geglu_pipeline
        };
        Some(ops::full_pipeline::dispatch_full_pipeline(
            &self.queue, &self.bufs, &self.q4,
            geglu,
            &self.geglu_gelu_tanh_pipeline,
            &self.silu_pipeline,
            &self.gelu_tanh_pipeline,
            &self.q8_quant_pipeline,
            Some(&self.fused_attn_pipeline),
            &self.q8_matvec_pipeline,
            &self.q8_qkv_proj_pipeline,
            &self.q4k_matvec_pipeline, &self.q6k_matvec_pipeline,
            &self.rms_norm_pipeline, &self.residual_add_pipeline,
            &self.rms_norm_q8_pipeline, &self.residual_norm_q8_pipeline,
            Some(&self.q4k_qkv_proj_pipeline), Some(&self.q4k_proj_pipeline),
            Some(&self.rope_at_pos_pipeline), Some(kv),
            layers, x, hidden, inter, q_dim, kv_dim,
            seq_len, num_q_heads, num_kv_heads, head_dim,
            rope_base, use_qk_norm, softcap,
        ))
    }

    fn has_kv_cache(&self) -> bool { true }

    fn populate_kv_layer(
        &self, layer: usize,
        k_data: &[f32], v_data: &[f32],
        seq_len: usize, num_kv_heads: usize, head_dim: usize,
    ) {
        if layer == 0 && std::env::var("LARQL_DUMP_KV").ok().as_deref() == Some("1") {
            let expected = seq_len * num_kv_heads * head_dim;
            let k_max = k_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let v_max = v_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let k_nf = k_data.iter().filter(|v| !v.is_finite()).count();
            let v_nf = v_data.iter().filter(|v| !v.is_finite()).count();
            eprintln!("[populate_kv_layer] L{layer} seq_len={seq_len} num_kv={num_kv_heads} head_dim={head_dim} expected={expected}");
            eprintln!("  k: len={} max|val|={k_max:.3} non-finite={k_nf}  (first 8: {:?})",
                k_data.len(), &k_data[..k_data.len().min(8)]);
            eprintln!("  v: len={} max|val|={v_max:.3} non-finite={v_nf}  (first 8: {:?})",
                v_data.len(), &v_data[..v_data.len().min(8)]);
        }
        let mut cache_guard = self.kv_cache.lock().unwrap();
        // Ensure KV cache exists with enough layers
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(layer + 1, 4096, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        // Extend if needed
        while kv.layers.len() <= layer {
            kv.layers.push(ops::kv_cache::LayerKVCache::new(&self.bufs, 4096, num_kv_heads, head_dim));
        }

        let lc = &mut kv.layers[layer];
        // Write K/V data directly to Metal buffers
        let total = seq_len * num_kv_heads * head_dim;
        let k_ptr = lc.k_cache.contents() as *mut f32;
        let v_ptr = lc.v_cache.contents() as *mut f32;
        // SAFETY: k_ptr/v_ptr point to pre-allocated Metal buffers sized for max_seq * kv_dim.
        // k_data/v_data are borrow-checked &[f32] params. Copy size is bounded by min(total, src.len()).
        unsafe {
            std::ptr::copy_nonoverlapping(k_data.as_ptr(), k_ptr, total.min(k_data.len()));
            std::ptr::copy_nonoverlapping(v_data.as_ptr(), v_ptr, total.min(v_data.len()));
        }
        lc.current_len = seq_len;
    }

    fn reset_kv_cache(&self) {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        *cache_guard = None; // drop entirely so next decode_token re-creates with correct layer count
    }

    fn rollback_kv_cache(&self, n: usize) {
        if let Some(kv) = self.kv_cache.lock().unwrap().as_mut() {
            kv.rollback(n);
        }
    }

    fn debug_read_kv_layer(&self, layer: usize) -> Option<(Vec<f32>, Vec<f32>, usize)> {
        let guard = self.kv_cache.lock().ok()?;
        let kv = guard.as_ref()?;
        let lc = kv.layers.get(layer)?;
        let n = lc.current_len;
        let total = n * lc.num_kv_heads * lc.head_dim;
        let k = super::buffers::read_buffer_f32(&lc.k_cache, total);
        let v = super::buffers::read_buffer_f32(&lc.v_cache, total);
        Some((k, v, n))
    }

    fn decode_token(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<f32>> {
        let num_layers = layers.len();
        // Lazily initialize KV cache
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(num_layers, 4096, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        Some(MetalBackend::decode_token(self, kv, layers, x, hidden, inter, q_dim, kv_dim,
            num_q_heads, num_kv_heads, head_dim, rope_base))
    }

    fn decode_token_with_probe(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32,
        probe_layer: Option<usize>,
    ) -> Option<(Vec<f32>, Option<Vec<f32>>)> {
        let num_layers = layers.len();
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(num_layers, 4096, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        Some(MetalBackend::decode_token_with_probe(self, kv, layers, x, hidden, inter,
            q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base, probe_layer))
    }

    fn has_q4(&self) -> bool { true }

    fn name(&self) -> &str { "metal (GPU)" }

    fn device_info(&self) -> String {
        format!("Metal GPU, FLOP threshold: {}", self.flop_threshold())
    }
}
