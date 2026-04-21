//! Token sampling for generation.  Self-contained: takes
//! `(token_id, logit, prob)` tuples (as produced by
//! `LlamaPipeline::top_k_at`) and returns a sampled token.  Default
//! config is greedy (temperature = 0).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Sampling parameters.  All fields default to greedy argmax.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Softmax temperature.  `0.0` = greedy argmax.  `1.0` = raw logits.
    pub temperature: f32,
    /// Nucleus cutoff.  `1.0` = full distribution, `0.9` = sample from
    /// the smallest set whose cumulative probability ≥ 0.9.  Ignored
    /// when temperature == 0.
    pub top_p: f32,
    /// Top-k truncation.  `0` = no limit.
    pub top_k: usize,
    /// Deterministic seed.  `None` = OS RNG.
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

/// A sampler that owns its RNG.  Construct once, call `sample` per token.
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

    /// Pick a token from a `(token_id, logit, prob)` list sorted
    /// logit-descending.
    pub fn sample(&mut self, predictions: &[(u32, f32, f64)]) -> Option<u32> {
        if predictions.is_empty() {
            return None;
        }
        if self.cfg.temperature <= 0.0 {
            return predictions.first().map(|&(tid, _, _)| tid);
        }
        let cutoff = if self.cfg.top_k == 0 {
            predictions.len()
        } else {
            self.cfg.top_k.min(predictions.len())
        };
        let candidates = &predictions[..cutoff];

        let inv_t = 1.0 / self.cfg.temperature as f64;
        let max_logit = candidates
            .iter()
            .map(|&(_, l, _)| l as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<f64> = candidates
            .iter()
            .map(|&(_, l, _)| ((l as f64 - max_logit) * inv_t).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return predictions.first().map(|&(tid, _, _)| tid);
        }
        for w in &mut weights {
            *w /= total;
        }

        if self.cfg.top_p < 1.0 && self.cfg.top_p > 0.0 {
            let mut idx: Vec<usize> = (0..weights.len()).collect();
            idx.sort_by(|&a, &b| {
                weights[b]
                    .partial_cmp(&weights[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut cum = 0.0;
            let mut keep = 0usize;
            for (k, &i) in idx.iter().enumerate() {
                cum += weights[i];
                keep = k + 1;
                if cum >= self.cfg.top_p as f64 {
                    break;
                }
            }
            let mut mask = vec![false; weights.len()];
            for &i in &idx[..keep] {
                mask[i] = true;
            }
            let mut new_total = 0.0;
            for i in 0..weights.len() {
                if !mask[i] {
                    weights[i] = 0.0;
                }
                new_total += weights[i];
            }
            if new_total > 0.0 {
                for w in &mut weights {
                    *w /= new_total;
                }
            }
        }

        let u: f64 = self.rng.random::<f64>();
        let mut cum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cum += w;
            if u <= cum {
                return Some(candidates[i].0);
            }
        }
        candidates.last().map(|&(tid, _, _)| tid)
    }
}
