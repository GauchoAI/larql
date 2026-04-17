//! Speculative decoding loop — draft K tokens, verify, accept prefix.
//!
//! Phase 0: sequential verification (proves the loop, zero speedup).
//! Phase 1: parallel verification via decode_token_batch (the real unlock).

use larql_compute::ComputeBackend;
use crate::model::ModelWeights;
use super::ngram_cache::NgramCache;

/// Result of speculative generation.
pub struct SpecGenerateResult {
    pub tokens: Vec<(String, f64)>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_accepted: usize,
    pub total_drafted: usize,
    pub total_cycles: usize,
}

impl SpecGenerateResult {
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_drafted == 0 { 0.0 }
        else { self.total_accepted as f64 / self.total_drafted as f64 }
    }

    pub fn effective_tok_s(&self) -> f64 {
        if self.decode_ms <= 0.0 { 0.0 }
        else { self.tokens.len() as f64 / self.decode_ms * 1000.0 }
    }
}

/// Speculative generation with n-gram draft and sequential verification.
///
/// Uses `decode_token` for each verify step (same cost as normal generation).
/// Phase 0: validates infrastructure. Phase 1 will use `decode_token_batch`
/// for parallel verification.
#[allow(clippy::too_many_arguments)]
pub fn generate_speculative(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    max_draft_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    layers: &[larql_compute::FullPipelineLayer],
) -> SpecGenerateResult {
    let norm_offset = weights.arch.norm_weight_offset();
    let hidden = weights.hidden_size;
    let intermediate = weights.intermediate_size;
    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = weights.arch.rope_base_for_layer(0) as f32;
    let trace = std::env::var("LARQL_TRACE_SPEC").ok().as_deref() == Some("1");

    // ── Prefill via sequential GPU decode ──
    let prefill_start = std::time::Instant::now();
    backend.reset_kv_cache();
    let embeds = crate::forward::embed_tokens_pub(weights, token_ids);
    let seq_len = token_ids.len();
    let mut last_h = vec![0.0f32; hidden];
    for p in 0..seq_len {
        let x: Vec<f32> = embeds.row(p).to_vec();
        if let Some(result) = backend.decode_token(
            layers, &x, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
        ) {
            last_h = result;
        }
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // First token from prefill output
    let h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_h).unwrap();
    let h_final = crate::forward::apply_norm(weights, &h_arr, weights.arch.final_norm_key(), norm_offset);
    let h_1d = h_final.row(0).to_owned();
    let first_hits = index.lm_head_knn_backend(&h_1d, 5, backend);
    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();

    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut context: Vec<u32> = token_ids.to_vec();
    let mut current_token_id: u32;

    if let Some(&(tid, score)) = first_hits.first() {
        let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default();
        let prob = super::logits::softmax_prob(score, &first_hits, logits_scale, final_softcap);
        tokens.push((tok_str, prob));
        current_token_id = tid;
        context.push(tid);
    } else {
        return SpecGenerateResult {
            tokens, prefill_ms, decode_ms: 0.0,
            total_accepted: 0, total_drafted: 0, total_cycles: 0,
        };
    }

    // ── N-gram cache: seed from prompt ──
    let mut ngram = NgramCache::new();
    ngram.populate(&context);

    let mut total_accepted = 0usize;
    let mut total_drafted = 0usize;
    let mut total_cycles = 0usize;
    let decode_start = std::time::Instant::now();

    // Helper: run one decode_token and get top-1 prediction
    let decode_and_predict = |tok: u32| -> Option<(Vec<f32>, u32, f64)> {
        let h_tok = crate::forward::embed_tokens_pub(weights, &[tok]);
        let x: Vec<f32> = h_tok.row(0).to_vec();
        let h_out = backend.decode_token(
            layers, &x, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
        )?;
        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out.clone()).unwrap();
        let h_final = crate::forward::apply_norm(weights, &h_arr, weights.arch.final_norm_key(), norm_offset);
        let h_1d = h_final.row(0).to_owned();
        let hits = index.lm_head_knn_backend(&h_1d, 5, backend);
        let (tid, score) = hits.first().copied()?;
        let prob = super::logits::softmax_prob(score, &hits, logits_scale, final_softcap);
        Some((h_out, tid, prob))
    };

    // ── Speculative decode loop ──
    while tokens.len() < max_tokens {
        // Draft K tokens from n-gram cache
        let drafts = ngram.draft(&context, max_draft_k);
        let k = drafts.len();

        if k == 0 {
            // No draft available — normal single-token decode
            if let Some((_h, tid, prob)) = decode_and_predict(current_token_id) {
                let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default();
                let is_eos = tok_str.trim() == "<eos>" || tok_str.trim() == "</s>";
                tokens.push((tok_str, prob));
                context.push(tid);
                ngram.populate(&context[context.len().saturating_sub(3)..]);
                current_token_id = tid;
                if is_eos { break; }
            } else { break; }
            continue;
        }

        total_cycles += 1;
        total_drafted += k;

        // Verify each draft token sequentially
        let mut accepted = 0usize;
        let mut bonus_token: Option<(u32, f64)> = None;

        for i in 0..k {
            let verify_input = if i == 0 { current_token_id } else { drafts[i - 1] };
            if let Some((_h, model_tid, prob)) = decode_and_predict(verify_input) {
                if model_tid == drafts[i] {
                    // Draft matches model — accept
                    accepted += 1;
                    let tok_str = tokenizer.decode(&[model_tid], true).unwrap_or_default();
                    let is_eos = tok_str.trim() == "<eos>" || tok_str.trim() == "</s>";
                    tokens.push((tok_str, prob));
                    context.push(model_tid);
                    if is_eos || tokens.len() >= max_tokens { break; }
                } else {
                    // Reject — use model's prediction as the bonus token
                    bonus_token = Some((model_tid, prob));
                    // Rollback KV cache: we already appended this token's KV,
                    // but the DRAFT token for subsequent positions is wrong.
                    // Rollback the remaining (k - i - 1) un-verified positions.
                    // (The current position's KV is correct — model processed it.)
                    break;
                }
            } else { break; }
        }

        // Rollback un-verified draft positions from KV cache
        let unverified = k - accepted - if bonus_token.is_some() { 0 } else { 0 };
        // We only ran decode_token for (accepted + 1) tokens if we rejected,
        // or (accepted) if we ran out. No rollback needed for un-run tokens
        // since they were never appended to KV cache.

        // Add bonus token (model's prediction at rejection point)
        if let Some((tid, prob)) = bonus_token {
            let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default();
            let is_eos = tok_str.trim() == "<eos>" || tok_str.trim() == "</s>";
            tokens.push((tok_str, prob));
            context.push(tid);
            current_token_id = tid;
            if is_eos { break; }
        } else if accepted > 0 {
            current_token_id = drafts[accepted - 1];
        }

        // If all K drafts accepted, also get the model's prediction at position K
        if accepted == k && bonus_token.is_none() {
            if let Some((_h, tid, prob)) = decode_and_predict(drafts[k - 1]) {
                let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default();
                let is_eos = tok_str.trim() == "<eos>" || tok_str.trim() == "</s>";
                tokens.push((tok_str, prob));
                context.push(tid);
                current_token_id = tid;
                if is_eos { break; }
            }
        }

        total_accepted += accepted;
        ngram.populate(&context[context.len().saturating_sub(k + 3)..]);

        if trace {
            eprintln!("[spec] cycle {total_cycles}: drafted {k}, accepted {accepted}, total tokens {}",
                tokens.len());
        }
    }

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    SpecGenerateResult {
        tokens, prefill_ms, decode_ms,
        total_accepted, total_drafted, total_cycles,
    }
}
