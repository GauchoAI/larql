//! N-gram draft cache for speculative decoding.
//!
//! Maps bigrams (token_{t-2}, token_{t-1}) → continuation tokens with counts.
//! Populated from prompt tokens and accepted generation. Multi-step drafts
//! chain lookups: after drafting token_0, use (token_{t-1}, draft_0) to get
//! draft_1, and so on.
//!
//! Cost: one hash lookup per draft step (~0.001ms). Zero model inference.
//! Expected acceptance: ~90%+ on repetitive/structured text, ~0% on novel.

use std::collections::HashMap;

/// Bigram + trigram draft cache. Trigrams are more specific (higher
/// acceptance), bigrams are the fallback when trigram misses.
pub struct NgramCache {
    /// (tok_a, tok_b) → sorted list of (next_tok, count), descending by count.
    bigrams: HashMap<(u32, u32), Vec<(u32, u32)>>,
    /// (tok_a, tok_b, tok_c) → sorted list of (next_tok, count).
    trigrams: HashMap<(u32, u32, u32), Vec<(u32, u32)>>,
}

impl NgramCache {
    pub fn new() -> Self {
        Self {
            bigrams: HashMap::new(),
            trigrams: HashMap::new(),
        }
    }

    fn insert_into(map: &mut Vec<(u32, u32)>, tok: u32) {
        if let Some(pos) = map.iter().position(|&(t, _)| t == tok) {
            map[pos].1 += 1;
        } else {
            map.push((tok, 1));
        }
        map.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    }

    /// Record an observed trigram: after seeing (a, b), token c followed.
    pub fn insert(&mut self, a: u32, b: u32, c: u32) {
        Self::insert_into(self.bigrams.entry((a, b)).or_default(), c);
    }

    /// Populate from a token sequence (e.g., the prompt).
    pub fn populate(&mut self, tokens: &[u32]) {
        for w in tokens.windows(3) {
            self.insert(w[0], w[1], w[2]);
        }
        for w in tokens.windows(4) {
            Self::insert_into(
                self.trigrams.entry((w[0], w[1], w[2])).or_default(),
                w[3],
            );
        }
    }

    /// Draft up to `max_k` continuation tokens.
    /// Prefers trigram match (more specific), falls back to bigram.
    pub fn draft(&self, context: &[u32], max_k: usize) -> Vec<u32> {
        if context.len() < 2 || max_k == 0 { return vec![]; }

        let mut drafts = Vec::with_capacity(max_k);
        let n = context.len();
        let mut a = if n >= 3 { context[n - 3] } else { 0 };
        let mut b = context[n - 2];
        let mut c = context[n - 1];

        for _ in 0..max_k {
            // Try trigram first (higher specificity)
            let next = if n >= 3 || !drafts.is_empty() {
                self.trigrams.get(&(a, b, c))
                    .and_then(|v| v.first().map(|&(tok, _)| tok))
            } else {
                None
            };
            // Fall back to bigram
            let next = next.or_else(|| {
                self.bigrams.get(&(b, c))
                    .and_then(|v| v.first().map(|&(tok, _)| tok))
            });

            match next {
                Some(tok) => {
                    drafts.push(tok);
                    a = b;
                    b = c;
                    c = tok;
                }
                None => break,
            }
        }
        drafts
    }

    /// Number of distinct bigram entries.
    pub fn len(&self) -> usize { self.bigrams.len() }
    pub fn is_empty(&self) -> bool { self.bigrams.is_empty() }
}

impl Default for NgramCache {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_draft() {
        let mut cache = NgramCache::new();
        // "the capital of France is Paris"
        let tokens = vec![1, 2, 3, 4, 5, 6]; // the=1 capital=2 of=3 France=4 is=5 Paris=6
        cache.populate(&tokens);

        // After seeing (capital, of), should draft France
        assert_eq!(cache.draft(&[2, 3], 1), vec![4u32]);
        // Multi-step: (of, France) → is → Paris
        assert_eq!(cache.draft(&[3, 4], 2), vec![5u32, 6]);
    }

    #[test]
    fn frequency_ranking() {
        let mut cache = NgramCache::new();
        cache.insert(1, 2, 10); // (1,2) → 10 once
        cache.insert(1, 2, 20); // (1,2) → 20 once
        cache.insert(1, 2, 10); // (1,2) → 10 twice now
        // 10 has count=2, should be ranked first
        assert_eq!(cache.draft(&[1, 2], 1), vec![10u32]);
    }

    #[test]
    fn no_hit() {
        let cache = NgramCache::new();
        assert_eq!(cache.draft(&[1, 2], 4), Vec::<u32>::new());
    }

    #[test]
    fn chain_breaks() {
        let mut cache = NgramCache::new();
        cache.insert(1, 2, 3);
        // (2, 3) has no continuation → chain stops after first draft
        assert_eq!(cache.draft(&[1, 2], 5), vec![3u32]);
    }
}
