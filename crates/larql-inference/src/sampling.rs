//! Token sampling for generation.
//!
//! The raw `predict_*` fns return top-k (token, logit, prob) tuples. Callers
//! that want greedy decoding pick `[0]`; callers that want varied output use
//! `sample_token()` here. Default config is greedy (temperature = 0) so
//! existing behavior is preserved.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Sampling parameters. All fields default to the greedy-argmax behavior.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Softmax temperature. `0.0` = greedy argmax. `1.0` = use raw logits.
    /// Higher values (1.2-2.0) spread probability mass, producing more variety.
    pub temperature: f32,
    /// Nucleus sampling cutoff. `1.0` = use full distribution. `0.9` = sample
    /// only from the smallest set of tokens whose cumulative probability ≥ 0.9.
    /// Ignored when temperature == 0.
    pub top_p: f32,
    /// Top-k truncation. `0` = no limit (use whatever predict returned).
    /// When set, only the top-k tokens by logit are considered.
    pub top_k: usize,
    /// Deterministic seed for reproducibility. `None` = thread_rng.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
        }
    }
}

/// A sampler that owns its RNG. Construct once, call `sample` per token.
pub struct Sampler {
    cfg: SamplingConfig,
    rng: StdRng,
}

impl Sampler {
    pub fn new(cfg: SamplingConfig) -> Self {
        let rng = match cfg.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        Self { cfg, rng }
    }

    /// Pick a token from the prediction list. `predictions` is the
    /// `raw_predictions` field from `PredictResult`: `(token_id, logit, prob)`
    /// tuples, sorted by logit descending.
    pub fn sample(&mut self, predictions: &[(u32, f32, f64)]) -> Option<u32> {
        if predictions.is_empty() { return None; }

        // Greedy fast-path.
        if self.cfg.temperature <= 0.0 {
            return predictions.first().map(|&(tid, _, _)| tid);
        }

        // Top-k truncation before temperature-rescaling.
        let cutoff = if self.cfg.top_k == 0 {
            predictions.len()
        } else {
            self.cfg.top_k.min(predictions.len())
        };
        let candidates = &predictions[..cutoff];

        // Apply temperature to logits then renormalize.
        let inv_t = 1.0 / self.cfg.temperature as f64;
        let max_logit = candidates.iter()
            .map(|&(_, l, _)| l as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<f64> = candidates.iter()
            .map(|&(_, l, _)| ((l as f64 - max_logit) * inv_t).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return predictions.first().map(|&(tid, _, _)| tid);
        }
        for w in &mut weights { *w /= total; }

        // Top-p nucleus filter — sort indexed by weight desc, cut at cumulative p.
        if self.cfg.top_p < 1.0 && self.cfg.top_p > 0.0 {
            let mut idx: Vec<usize> = (0..weights.len()).collect();
            idx.sort_by(|&a, &b| weights[b].partial_cmp(&weights[a]).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0;
            let mut keep = 0usize;
            for (k, &i) in idx.iter().enumerate() {
                cum += weights[i];
                keep = k + 1;
                if cum >= self.cfg.top_p as f64 { break; }
            }
            // Zero out the tail, renormalize.
            let mut mask = vec![false; weights.len()];
            for &i in &idx[..keep] { mask[i] = true; }
            let mut new_total = 0.0;
            for i in 0..weights.len() {
                if !mask[i] { weights[i] = 0.0; }
                new_total += weights[i];
            }
            if new_total > 0.0 {
                for w in &mut weights { *w /= new_total; }
            }
        }

        // Categorical sample.
        let u: f64 = self.rng.random::<f64>();
        let mut cum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cum += w;
            if u <= cum {
                return Some(candidates[i].0);
            }
        }
        // Fallback (shouldn't happen with proper normalization).
        candidates.last().map(|&(tid, _, _)| tid)
    }
}
