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
    generate_speculative_inner(weights, tokenizer, token_ids, max_tokens, max_draft_k,
        index, backend, layers, None)
}

/// Speculative generation with optional pre-sampled first token.
/// When `first_token` is Some, skips prefill (assumes KV cache already populated).
pub fn generate_speculative_resume(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    max_draft_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    layers: &[larql_compute::FullPipelineLayer],
    first_token: u32,
) -> SpecGenerateResult {
    generate_speculative_inner(weights, tokenizer, token_ids, max_tokens, max_draft_k,
        index, backend, layers, Some(first_token))
}

fn generate_speculative_inner(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    max_draft_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    layers: &[larql_compute::FullPipelineLayer],
    first_token: Option<u32>,
) -> SpecGenerateResult {
    let norm_offset = weights.arch.norm_weight_offset();
    let hidden = weights.hidden_size;
    let intermediate = weights.intermediate_size;
    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = weights.arch.rope_base_for_layer(0) as f32;
    let trace = std::env::var("LARQL_TRACE_SPEC").ok().as_deref() == Some("1");

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut context: Vec<u32> = token_ids.to_vec();
    let mut current_token_id: u32;
    let prefill_ms;

    if let Some(ft) = first_token {
        // Resume mode: KV cache already populated, first token already sampled
        prefill_ms = 0.0;
        current_token_id = ft;
        context.push(ft);
        if trace { eprintln!("[spec] resume mode: first_token={ft}, skipping prefill"); }
    } else {
        // Full prefill
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
        prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        if trace {
            let amax = last_h.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            eprintln!("[spec] prefill done: seq_len={seq_len} h_amax={amax:.2} layers={}", layers.len());
        }

        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_h).unwrap();
        let h_final = crate::forward::apply_norm(weights, &h_arr, weights.arch.final_norm_key(), norm_offset);
        let h_1d = h_final.row(0).to_owned();
        let first_hits = index.lm_head_knn_backend(&h_1d, 5, backend);

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

        // Build batch: [current_token, draft[0], draft[1], ..., draft[K-1]]
        // We verify K+1 positions: current_token produces the first model prediction
        // (which should match draft[0]); draft[0] produces the second (match draft[1]); etc.
        let mut verify_ids = Vec::with_capacity(k + 1);
        verify_ids.push(current_token_id);
        verify_ids.extend_from_slice(&drafts);
        let k_plus_1 = verify_ids.len();

        // Embed all verify tokens
        let embeds = crate::forward::embed_tokens_pub(weights, &verify_ids);
        let x_batch: Vec<f32> = embeds.as_slice().unwrap_or(&[]).to_vec();

        // Batch decode: all K+1 tokens through the model
        let h_batch = backend.decode_token_batch(
            layers, &x_batch, k_plus_1,
            hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
        );

        let h_batch = match h_batch {
            Some(h) => h,
            None => break,
        };

        // Get model predictions for each position
        let mut model_tokens = Vec::with_capacity(k_plus_1);
        let mut model_probs = Vec::with_capacity(k_plus_1);
        for bi in 0..k_plus_1 {
            let h_slice = &h_batch[bi * hidden..(bi + 1) * hidden];
            let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_slice.to_vec()).unwrap();
            let h_final = crate::forward::apply_norm(weights, &h_arr, weights.arch.final_norm_key(), norm_offset);
            let h_1d = h_final.row(0).to_owned();
            let hits = index.lm_head_knn_backend(&h_1d, 5, backend);
            if let Some(&(tid, score)) = hits.first() {
                let prob = super::logits::softmax_prob(score, &hits, logits_scale, final_softcap);
                model_tokens.push(tid);
                model_probs.push(prob);
            } else {
                model_tokens.push(0);
                model_probs.push(0.0);
            }
        }

        // Accept longest matching prefix.
        // Position 0: model_tokens[0] should match drafts[0]
        // Position i: model_tokens[i] should match drafts[i]
        let mut accepted = 0usize;
        for i in 0..k {
            if model_tokens[i] == drafts[i] {
                accepted += 1;
                let tok_str = tokenizer.decode(&[drafts[i]], true).unwrap_or_default();
                tokens.push((tok_str, model_probs[i]));
                context.push(drafts[i]);
                if tokens.len() >= max_tokens { break; }
            } else {
                break;
            }
        }

        // Rollback un-accepted draft tokens from KV cache.
        // We appended k+1 tokens total (current + k drafts).
        // We want to keep: 1 (current) + accepted (matched drafts) + 1 (bonus).
        // Rollback: (k+1) - (accepted+1) - 1 = k - accepted - 1... but we also
        // want to keep the bonus token's KV.
        // Simpler: rollback (k - accepted) positions.
        let rollback_n = k - accepted;
        if rollback_n > 0 {
            backend.rollback_kv_cache(rollback_n);
        }

        // Bonus token: model's prediction at the rejection point (or after all accepted)
        let bonus_idx = accepted; // model_tokens[accepted] = what model predicted at draft[accepted]
        if bonus_idx < k_plus_1 {
            let tid = model_tokens[bonus_idx];
            let prob = model_probs[bonus_idx];
            let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default();
            let is_eos = tok_str.trim() == "<eos>" || tok_str.trim() == "</s>";
            tokens.push((tok_str, prob));
            context.push(tid);
            current_token_id = tid;
            if is_eos { break; }
        } else if accepted > 0 {
            current_token_id = drafts[accepted - 1];
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
