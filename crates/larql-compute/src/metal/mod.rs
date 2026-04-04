//! Metal GPU compute backend — Apple Silicon.
//!
//! All operations go through the [`ComputeBackend`] trait. Metal-specific
//! optimisations: simdgroup Q4 dot products, threadgroup shared memory,
//! zero-copy mmap buffers, multi-layer command buffer pipeline.
//!
//! ## Modules
//!
//! - `shaders/`:  Metal Shading Language — one file per kernel (9 shaders)
//! - `ops/`:      GPU dispatch — one file per operation (6 dispatchers)
//! - `buffers`:   GPU buffer cache (zero-copy mmap, transient allocation)
//! - `f32_ops`:   f32 tiled matmul dispatch with GPU/CPU routing
//! - `calibrate`: CPU vs GPU auto-calibration
//!
//! ## Performance (M3 Max)
//!
//! - Q4 matvec: 0.57ms (simdgroup, 14.7MB matrix)
//! - Multi-layer FFN: 8.5ms (21 layers, one command buffer)
//! - Full layer: 1.7ms (attention + FFN, seq=1)

pub mod shaders;   // modular: shaders/mod.rs → one file per shader
pub mod buffers;
pub mod f32_ops;
pub mod ops;        // modular: ops/mod.rs → one file per operation
pub mod calibrate;

use std::sync::atomic::{AtomicUsize, Ordering};
use ndarray::{Array2, ArrayView2};
use metal::*;

use crate::backend::{ComputeBackend, MatMulOp};
use buffers::BufferCache;
use f32_ops::F32Ops;
use ops::q4_common::Q4Pipelines;

/// Metal GPU compute backend.
pub struct MetalBackend {
    queue: CommandQueue,
    bufs: BufferCache,
    f32_ops: F32Ops,
    q4: Q4Pipelines,
    causal_attn_pipeline: ComputePipelineState,
    geglu_pipeline: ComputePipelineState,
    q8_quant_pipeline: ComputePipelineState,
    flop_threshold: AtomicUsize,
}

impl MetalBackend {
    /// Create a Metal backend. Returns None if no Metal device is available.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let all_src = shaders::all_shaders();
        let library = device
            .new_library_with_source(&all_src, &opts)
            .map_err(|e| eprintln!("[metal] shader compile error: {e}"))
            .ok()?;

        let sgemm_fn = library.get_function("sgemm", None).ok()?;
        let transb_fn = library.get_function("sgemm_transb", None).ok()?;
        let q4_matvec_fn = library.get_function("q4_matvec", None).ok()?;
        let q4_vecmat_fn = library.get_function("q4_vecmat", None).ok()?;

        let f32_ops = F32Ops {
            sgemm_pipeline: device.new_compute_pipeline_state_with_function(&sgemm_fn).ok()?,
            transb_pipeline: device.new_compute_pipeline_state_with_function(&transb_fn).ok()?,
        };

        let q4_f32_matvec_fn = library.get_function("q4_f32_matvec", None).ok()?;
        let geglu_fn = library.get_function("geglu_silu", None).ok()?;
        let q8_quant_fn = library.get_function("quantize_q8", None).ok()?;
        let causal_attn_fn = library.get_function("causal_attention", None).ok()?;
        let causal_attn_pipeline = device.new_compute_pipeline_state_with_function(&causal_attn_fn).ok()?;

        let q4 = Q4Pipelines {
            matvec: device.new_compute_pipeline_state_with_function(&q4_matvec_fn).ok()?,
            vecmat: device.new_compute_pipeline_state_with_function(&q4_vecmat_fn).ok()?,
            f32_matvec: device.new_compute_pipeline_state_with_function(&q4_f32_matvec_fn).ok()?,
        };

        let bufs = BufferCache::new(&device);

        let geglu_pipeline = device.new_compute_pipeline_state_with_function(&geglu_fn).ok()?;
        let q8_quant_pipeline = device.new_compute_pipeline_state_with_function(&q8_quant_fn).ok()?;

        Some(Self {
            queue, bufs, f32_ops, q4, causal_attn_pipeline,
            geglu_pipeline, q8_quant_pipeline,
            flop_threshold: AtomicUsize::new(calibrate::DEFAULT_FLOP_THRESHOLD),
        })
    }

    /// Auto-calibrate CPU vs GPU threshold.
    pub fn calibrate(&self) {
        let threshold = calibrate::calibrate(&self.f32_ops, &self.queue, &self.bufs);
        self.flop_threshold.store(threshold, Ordering::Relaxed);
    }

    pub fn flop_threshold(&self) -> usize { self.flop_threshold.load(Ordering::Relaxed) }
    pub fn set_flop_threshold(&self, t: usize) { self.flop_threshold.store(t.max(calibrate::MIN_FLOP_FLOOR), Ordering::Relaxed); }
    pub fn cache_size(&self) -> usize { self.bufs.len() }

    // ── Direct Q4 ops (for benchmarking outside the trait) ──

    pub fn q4_matvec_direct(
        &self, q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32],
        num_rows: usize, hidden: usize,
    ) -> Vec<f32> {
        ops::q4_matvec::dispatch(&self.queue, &self.bufs, &self.q4.matvec, q4_data, q8_x, q8_scales, num_rows, hidden)
    }

    pub fn q4_vecmat_direct(
        &self, activation: &[f32], q4_data: &[u8],
        intermediate: usize, hidden: usize,
    ) -> Vec<f32> {
        ops::q4_vecmat::dispatch(&self.queue, &self.bufs, &self.q4.vecmat, activation, q4_data, intermediate, hidden)
    }

    /// Q4 × f32 matvec (for transposed down projection).
    pub fn q4_f32_matvec_direct(
        &self, q4_data: &[u8], x: &[f32], num_rows: usize, hidden: usize,
    ) -> Vec<f32> {
        ops::q4_f32_matvec::dispatch(&self.queue, &self.bufs, &self.q4.f32_matvec, q4_data, x, num_rows, hidden)
    }

    /// Full layer pipeline: attention + FFN in one Metal command buffer.
    pub fn full_layer_direct(
        &self,
        w_q: &[f32], w_k: &[f32], w_v: &[f32], w_o: &[f32],
        gate_q4: &[u8], up_q4: &[u8], down_t_q4: &[u8],
        x: &[f32], seq_len: usize, hidden: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        inter: usize, attn_scale: f32,
    ) -> Vec<f32> {
        ops::full_layer::dispatch(
            &self.queue, &self.bufs,
            &self.f32_ops.transb_pipeline,
            &self.causal_attn_pipeline,
            &self.q4,
            w_q, w_k, w_v, w_o,
            gate_q4, up_q4, down_t_q4,
            x, seq_len, hidden,
            num_q_heads, num_kv_heads, head_dim, inter, attn_scale,
        )
    }

    /// Multi-layer Q4 FFN in ONE command buffer.
    /// gate → up → GEGLU → down → Q8 quantize → next layer.
    /// All on GPU, no CPU return between layers.
    pub fn multi_layer_q4_ffn(
        &self,
        layers_q4: &[(Vec<u8>, Vec<u8>, Vec<u8>)], // [(gate, up, down_t)]
        x: &[f32],
        inter: usize,
        hidden: usize,
    ) -> Vec<f32> {
        ops::q4_batched::multi_layer_ffn(
            &self.queue, &self.bufs, &self.q4,
            &self.geglu_pipeline, &self.q8_quant_pipeline,
            layers_q4, x, inter, hidden,
        )
    }

    pub fn q4_matvec_pair_batch_direct(
        &self, gate_q4: &[u8], up_q4: &[u8],
        x_matrix: &[f32], seq_len: usize,
        num_rows: usize, hidden: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        ops::q4_batched::pair_batch(
            &self.queue, &self.bufs, &self.q4,
            gate_q4, up_q4, x_matrix, seq_len, num_rows, hidden,
        )
    }
}

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

    fn has_q4(&self) -> bool { true }

    fn name(&self) -> &str { "metal (GPU)" }

    fn device_info(&self) -> String {
        format!("Metal GPU, FLOP threshold: {}", self.flop_threshold())
    }
}
