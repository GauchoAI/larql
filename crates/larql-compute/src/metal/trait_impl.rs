use super::*;

impl ComputeBackend for MetalBackend {
    fn reset_kv_cache(&self) {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        // Drop entirely so the next decode_token re-creates with the right
        // layer count for the model in use.
        *cache_guard = None;
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
            *cache_guard = Some(self.create_kv_cache(num_layers, super::KV_MAX_SEQ, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        Some(MetalBackend::decode_token_with_probe(self, kv, layers, x, hidden, inter,
            q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base, probe_layer))
    }

    fn matvec_q8_0_gguf(&self, weight: &[u8], x: &[f32], n: usize, k: usize)
        -> Option<Vec<f32>> {
        Some(MetalBackend::matvec_q8_0_gguf(self, weight, x, n, k))
    }

    fn name(&self) -> &str { "metal (GPU)" }

    fn device_info(&self) -> String {
        format!("Metal GPU, FLOP threshold: {}", self.flop_threshold())
    }
}
