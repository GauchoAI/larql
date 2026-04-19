//! Neural draft head for speculative decoding.
//!
//! A small MLP: h[2560] → Linear(2560, 1024) → GELU → Linear(1024, vocab)
//! Trained on (h_final, next_token) pairs captured during normal inference.
//! Runs in ~0.1ms on CPU (two matvecs of modest size).

use std::path::Path;

/// Trained draft head weights.
pub struct DraftHead {
    w1: Vec<f32>,   // [inner, hidden] row-major
    b1: Vec<f32>,   // [inner]
    w2: Vec<f32>,   // [vocab, inner] row-major
    b2: Vec<f32>,   // [vocab]
    pub hidden: usize,
    pub inner: usize,
    pub vocab: usize,
}

impl DraftHead {
    /// Load from binary file (DRFT header + f32 weights).
    pub fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        if data.len() < 20 || &data[0..4] != b"DRFT" { return None; }

        let hidden = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
        let inner = u32::from_le_bytes(data[8..12].try_into().ok()?) as usize;
        let vocab = u32::from_le_bytes(data[12..16].try_into().ok()?) as usize;

        let expected = 20 + (inner * hidden + inner + vocab * inner + vocab) * 4;
        if data.len() < expected { return None; }

        let mut offset = 20;
        let read_f32 = |off: &mut usize, n: usize| -> Vec<f32> {
            let bytes = &data[*off..*off + n * 4];
            *off += n * 4;
            bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };

        let w1 = read_f32(&mut offset, inner * hidden);
        let b1 = read_f32(&mut offset, inner);
        let w2 = read_f32(&mut offset, vocab * inner);
        let b2 = read_f32(&mut offset, vocab);

        eprintln!("[draft-head] loaded: hidden={hidden} inner={inner} vocab={vocab} ({:.0} MB)",
            data.len() as f64 / 1e6);

        Some(Self { w1, b1, w2, b2, hidden, inner, vocab })
    }

    /// Predict top-1 token from h_final. Returns (token_id, logit).
    /// Cost: ~0.1ms (two small matvecs on CPU).
    pub fn predict(&self, h: &[f32]) -> Option<u32> {
        if h.len() != self.hidden { return None; }

        // Layer 1: inner = GELU(W1 @ h + b1)
        let mut mid = vec![0.0f32; self.inner];
        for i in 0..self.inner {
            let mut sum = self.b1[i];
            let row = &self.w1[i * self.hidden..(i + 1) * self.hidden];
            for j in 0..self.hidden {
                sum += row[j] * h[j];
            }
            // GELU approximation
            let v3 = sum * sum * sum;
            let arg = (0.7978845608f32 * (sum + 0.044715 * v3)).clamp(-10.0, 10.0);
            mid[i] = 0.5 * sum * (1.0 + arg.tanh());
        }

        // Layer 2: logits = W2 @ mid + b2, return argmax
        let mut best_id = 0u32;
        let mut best_logit = f32::NEG_INFINITY;
        for i in 0..self.vocab {
            let mut sum = self.b2[i];
            let row = &self.w2[i * self.inner..(i + 1) * self.inner];
            for j in 0..self.inner {
                sum += row[j] * mid[j];
            }
            if sum > best_logit {
                best_logit = sum;
                best_id = i as u32;
            }
        }

        Some(best_id)
    }

    /// Draft K tokens by chaining predictions.
    /// Each step: predict next token, embed it (lookup h from decode), predict again.
    /// For now, just returns the single top-1 prediction repeated
    /// (proper chaining requires running embed → draft_head for each step).
    pub fn draft(&self, h_final: &[f32], max_k: usize) -> Vec<u32> {
        // Single-step draft: predict one token from h_final
        match self.predict(h_final) {
            Some(tid) => vec![tid],  // only 1 token — can't chain without embed
            None => vec![],
        }
    }
}
