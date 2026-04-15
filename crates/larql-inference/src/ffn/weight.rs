//! Dense FFN backend — full matrix multiply, architecture-correct.
//! This is the ground truth: identical to model inference.

use ndarray::Array2;

use crate::forward::{add_bias, dot_proj};
use crate::model::ModelWeights;
use super::{sigmoid, gelu_tanh, silu_gate_up, gelu_tanh_gate_up, FfnBackend};

/// Dense FFN: follows the model architecture exactly.
/// Gated: activation(x @ gate.T) * (x @ up.T) @ down.T + bias
/// Non-gated: activation(x @ up.T + bias) @ down.T + bias
///
/// Supports all model families via the ModelArchitecture trait:
/// SiLU (Gemma/Llama), GELU (Qwen/StarCoder), gated/non-gated, bias/no-bias.
pub struct WeightFfn<'a> {
    pub weights: &'a ModelWeights,
}

impl<'a> FfnBackend for WeightFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        dense_ffn_forward(self.weights, layer, x)
    }

    fn name(&self) -> &str {
        "weights"
    }
}

/// WeightFfn with a ComputeBackend for matmuls — use this to run the f32
/// architecture-correct FFN through Metal instead of CPU ndarray. Fast path
/// for models where quantisation precision isn't sufficient (Gemma 3 4B).
pub struct WeightFfnGpu<'a> {
    pub weights: &'a ModelWeights,
    pub backend: &'a dyn larql_compute::ComputeBackend,
}

impl<'a> FfnBackend for WeightFfnGpu<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        dense_ffn_forward_gpu(self.weights, layer, x, Some(self.backend))
    }

    fn name(&self) -> &str {
        "weights-gpu"
    }
}

/// Architecture-correct dense FFN computation.
/// Used by WeightFfn and as fallback by sparse backends when K is high.
pub fn dense_ffn_forward(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    dense_ffn_forward_gpu(weights, layer, x, None)
}

/// Dense FFN with optional backend for matmuls. `backend=None` → CPU ndarray;
/// `backend=Some(metal)` → Metal `matmul_transb` on f32 weights.
pub fn dense_ffn_forward_gpu(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> (Array2<f32>, Array2<f32>) {
    let arch = &*weights.arch;
    let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
    let w_down = weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();

    // Backend-aware projection — transposed dot: out = x @ W.T.
    // Uses ArrayView2 so the closure accepts any ArrayBase representation
    // (`weights.tensors` returns OwnedArcRepr, locals are OwnedRepr).
    let project = |x: ndarray::ArrayView2<f32>, w: ndarray::ArrayView2<f32>| -> Array2<f32> {
        match backend {
            Some(be) => be.matmul_transb(x, w),
            None => x.dot(&w.t()),
        }
    };

    let activation = if arch.ffn_type() == larql_models::FfnType::Gated {
        let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        let gate = project(x.view(), w_gate.view());
        let up = project(x.view(), w_up.view());
        match arch.activation() {
            larql_models::Activation::GeluTanh => gelu_tanh_gate_up(&gate, &up),
            _ => silu_gate_up(&gate, &up),
        }
    } else {
        let mut projected = project(x.view(), w_up.view());
        if let Some(bias) = arch.ffn_up_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
            add_bias(&mut projected, bias);
        }
        match arch.activation() {
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu => projected.mapv(gelu_tanh),
            _ => projected.mapv(|v| v * sigmoid(v)),
        }
    };

    let mut out = project(activation.view(), w_down.view());
    if let Some(bias) = arch.ffn_down_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut out, bias);
    }
    (out, activation)
}
