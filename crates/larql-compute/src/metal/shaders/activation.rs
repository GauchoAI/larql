//! Standalone activation functions for non-gated FFN (StarCoder2, GPT-2).
//!
//! Unlike GEGLU which multiplies gate*up, these apply activation in-place
//! to a single buffer: out[i] = activation(input[i]).
//!
//! Used when ffn_type == Standard: up → activation → down (no gate).

pub const SHADER: &str = r#"
// SiLU / Swish: out = x / (1 + exp(-x))
kernel void silu(
    device const float* input [[buffer(0)]],
    device float*       out   [[buffer(1)]],
    constant uint&      N     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float x = input[tid];
    out[tid] = x / (1.0f + exp(-x));
}

// GELU with tanh approximation: out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_tanh(
    device const float* input [[buffer(0)]],
    device float*       out   [[buffer(1)]],
    constant uint&      N     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float x = input[tid];
    float c = 0.7978845608f; // sqrt(2/pi)
    // Clamp tanh argument: Metal tanh() returns NaN for |arg| > ~44 because
    // (e^2*arg) overflows f32 and (inf-1)/(inf+1) = NaN. tanh saturates to
    // ±1 within 1e-9 by |arg|=10, so clamping here is safe.
    float arg = clamp(c * (x + 0.044715f * x * x * x), -10.0f, 10.0f);
    out[tid] = 0.5f * x * (1.0f + tanh(arg));
}
"#;
