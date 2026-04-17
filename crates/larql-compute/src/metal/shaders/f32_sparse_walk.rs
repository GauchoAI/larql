//! f32 sparse walk kernels — true Option C.
//!
//! Reads only the top-K rows of an f32 matrix instead of the full N rows,
//! delivering bit-exact f32 arithmetic at a fraction of the memory bandwidth
//! of dense walk over the same mmap.
//!
//! Two kernels:
//!
//! - `f32_sparse_matvec`: gate / up projection direction.
//!   out[k] = Σ_h W[indices[k], h] * x[h]
//!   One thread per output row k; reads 1 row × hidden f32 per thread.
//!
//! - `f32_sparse_vecmat`: down projection direction.
//!   out[h] = Σ_k activation[k] * W[indices[k], h]
//!   One thread per output column h; reads K random rows' column h per
//!   thread. Poorer cache locality than matvec but avoids atomic_fadd.
//!
//! Bandwidth per decode token at K=1024, hidden=2560: (up 10 MB + down 10
//! MB) × 34 layers ≈ 680 MB vs dense f32 walk's 10.6 GB. That's the real
//! Option-C memory-vs-bandwidth win.

pub const SHADER: &str = r#"
kernel void f32_sparse_matvec(
    device const float*  W        [[buffer(0)]],   // [N, hidden] f32, mmap
    device const float*  x        [[buffer(1)]],   // [hidden]
    device const uint*   indices  [[buffer(2)]],   // [K]
    device float*        out      [[buffer(3)]],   // [K]
    constant uint&       K        [[buffer(4)]],
    constant uint&       hidden   [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= K) return;
    uint row = indices[tid];
    device const float* r = W + row * hidden;
    float acc = 0.0f;
    for (uint h = 0; h < hidden; h++) {
        acc += r[h] * x[h];
    }
    out[tid] = acc;
}

kernel void f32_sparse_vecmat(
    device const float*  W          [[buffer(0)]],   // [N, hidden] f32, mmap
    device const float*  activation [[buffer(1)]],   // [K]
    device const uint*   indices    [[buffer(2)]],   // [K]
    device float*        out        [[buffer(3)]],   // [hidden]
    constant uint&       K          [[buffer(4)]],
    constant uint&       hidden     [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= hidden) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        uint row = indices[k];
        acc += activation[k] * W[row * hidden + tid];
    }
    out[tid] = acc;
}
"#;

pub const THREADS_PER_TG: u64 = 256;
