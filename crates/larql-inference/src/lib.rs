//! larql-inference — the GGUF hot path.
//!
//! Two responsibilities:
//! 1. `gguf_pipeline::GgufPipeline` — load a GGUF, build per-layer
//!    `FullPipelineLayer` views, run `decode_token` through Metal,
//!    capture residuals for KNN, sample logits.
//! 2. `sampling::Sampler` — temperature/top-p/top-k token sampling
//!    used by the streaming generate path.
//!
//! Everything else (vindex weight loading, walk-FFN, dense matmul,
//! per-layer trace) was deleted with the GGUF-only refactor.

extern crate blas_src;

pub mod gguf_pipeline;
pub mod sampling;

// Re-export dependencies used by the server.
pub use larql_models;
pub use larql_vindex;
pub use larql_compute::{ComputeBackend, default_backend};
#[cfg(feature = "metal")]
pub use larql_compute::MetalBackend;
