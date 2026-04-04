//! Metal shader registry — one file per shader, compiled as one library.
//!
//! Each shader module exports a `SHADER` constant with the MSL source.
//! `all_shaders()` concatenates them with the common header for compilation.

pub mod common;
pub mod sgemm;
pub mod sgemm_transb;
pub mod q4_matvec;
pub mod q4_vecmat;
pub mod q4_f32_matvec;
pub mod geglu;
pub mod quantize_q8;
pub mod causal_attention;

/// Concatenate all shaders into one MSL source string for compilation.
pub fn all_shaders() -> String {
    let mut src = String::with_capacity(8192);
    src.push_str(common::HEADER);
    src.push_str(sgemm::SHADER);
    src.push_str(sgemm_transb::SHADER);
    src.push_str(q4_matvec::SHADER);
    src.push_str(q4_vecmat::SHADER);
    src.push_str(q4_f32_matvec::SHADER);
    src.push_str(geglu::SHADER);
    src.push_str(quantize_q8::SHADER);
    src.push_str(causal_attention::SHADER);
    src
}
