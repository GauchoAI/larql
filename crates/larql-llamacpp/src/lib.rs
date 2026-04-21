//! Thin, safe wrapper around llama.cpp that exposes a greedy `generate`
//! path plus a hook for residual-stream inspection/override via
//! `cb_eval`.  The long-term goal is to host larql's KNN overlay on top
//! of this crate and retire our custom Metal pipeline.

mod backend;
mod pipeline;
pub mod probe;
pub mod sampling;

pub use pipeline::{GenerateConfig, LlamaPipeline, LlamaPipelineError};
pub use probe::{NullProbe, OneShot, ProbeHandler, ProbeNode};
pub use sampling::{Sampler, SamplingConfig};
