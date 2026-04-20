//! # larql-compute
//!
//! Hardware compute backend for the LARQL GGUF hot path. The single
//! supported backend is Metal — `default_backend()` returns it on macOS,
//! and panics elsewhere because the only consumer (the GGUF inference
//! pipeline) doesn't have a CPU fallback.

extern crate blas_src;

pub mod backend;
pub mod pipeline;

#[cfg(feature = "metal")]
pub mod metal;

pub use pipeline::{
    QuantFormat, QuantWeight,
    NormType, FfnType, Activation,
    FullPipelineLayer,
};

pub use backend::ComputeBackend;

#[cfg(feature = "metal")]
pub use metal::MetalBackend;

/// Create the best available backend. Metal-only; panics if unavailable.
pub fn default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(feature = "metal")]
    {
        if let Some(m) = metal::MetalBackend::new() {
            return Box::new(m);
        }
        panic!("Metal backend not available — the GGUF inference path requires Metal.");
    }
    #[cfg(not(feature = "metal"))]
    panic!("larql-compute was built without --features metal; no inference backend available.");
}
