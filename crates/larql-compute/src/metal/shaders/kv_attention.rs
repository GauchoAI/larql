//! KV-cached attention for token generation (seq=1 decode).
//!
//! `kv_attention` handles up to 8192 past tokens (32KB threadgroup scores).
//! For Gemma 3 4B with 8K context window, this covers the full sequence.
//! Uses simd_max/simd_sum for reductions and float4 Q·K dot products.

pub const SHADER: &str = r#"
// Decode attention — 32KB scores = max 8192 tokens (Gemma 3 8K context).
kernel void kv_attention(
    device const float* Q       [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      T       [[buffer(4)]],
    constant uint&      head_dim[[buffer(5)]],
    constant uint&      num_q   [[buffer(6)]],
    constant uint&      num_kv  [[buffer(7)]],
    constant float&     scale   [[buffer(8)]],
    constant uint&      window_size [[buffer(9)]],
    constant float&     softcap [[buffer(10)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    uint head = tg_id;
    if (head >= num_q) return;
    uint kv_head = head / (num_q / num_kv);

    device const float* q = Q + head * head_dim;

    uint t_start = (window_size > 0 && T > window_size) ? T - window_size : 0;

    // Threadgroup scores — max 8160 tokens (32640 bytes, within 32KB limit
    // with tg_sg_vals[8] = 32 bytes → total 32672 < 32768)
    threadgroup float tg_scores[8160];

    // Phase 1: Q·K dot products + max
    float local_max = -1e30f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        device const float* k = K_cache + t * num_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d + 3 < head_dim; d += 4) {
            dot += q[d]*k[d] + q[d+1]*k[d+1] + q[d+2]*k[d+2] + q[d+3]*k[d+3];
        }
        for (uint d = (head_dim & ~3u); d < head_dim; d++) dot += q[d] * k[d];
        dot *= scale;
        // Optional softcap (Gemma 3: softcap=50). Clamp tanh argument to
        // [-30, 30] to avoid Metal tanh() NaN for |arg| > ~44.
        if (softcap > 0.0f) {
            float arg = clamp(dot / softcap, -30.0f, 30.0f);
            dot = tanh(arg) * softcap;
        }
        tg_scores[t - t_start] = dot;
        local_max = max(local_max, dot);
    }

    float sg_max = simd_max(local_max);
    threadgroup float tg_sg_vals[8];
    if (lane == 0) tg_sg_vals[sg_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = tg_sg_vals[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) global_max = max(global_max, tg_sg_vals[i]);

    // Phase 2: softmax
    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - global_max);
        tg_scores[t - t_start] = w;
        local_sum += w;
    }

    float sg_sum = simd_sum(local_sum);
    if (lane == 0) tg_sg_vals[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = tg_sg_vals[0];
    for (uint i = 1; i < n_sg; i++) global_sum += tg_sg_vals[i];
    float inv_sum = 1.0f / global_sum;

    // Normalize
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        tg_scores[t - t_start] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: weighted V sum
    device float* out_head = out + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_sz) {
        float acc = 0.0f;
        for (uint t = t_start; t < T; t++) {
            acc += tg_scores[t - t_start] * V_cache[t * num_kv * head_dim + kv_head * head_dim + d];
        }
        out_head[d] = acc;
    }
}

// ── Batched variants for speculative decoding ──

// Append K tokens to KV cache at consecutive positions [pos..pos+K).
// Grid: (num_kv * head_dim, batch_size, 1).
kernel void kv_cache_append_batch(
    device const float* new_k    [[buffer(0)]],   // [K, num_kv * head_dim]
    device const float* new_v    [[buffer(1)]],   // [K, num_kv * head_dim]
    device float*       K_cache  [[buffer(2)]],
    device float*       V_cache  [[buffer(3)]],
    constant uint&      pos      [[buffer(4)]],   // starting position
    constant uint&      num_kv   [[buffer(5)]],
    constant uint&      head_dim [[buffer(6)]],
    constant uint&      batch_size [[buffer(7)]],  // K
    uint tid_linear [[thread_position_in_grid]])
{
    // Grid is (kv_dim * batch_size, 1, 1). Decompose.
    uint total = num_kv * head_dim;
    uint elem = tid_linear % total;   // element within kv_dim
    uint bi = tid_linear / total;     // batch index [0..K)
    if (elem >= total || bi >= batch_size) return;
    K_cache[(pos + bi) * total + elem] = new_k[bi * total + elem];
    V_cache[(pos + bi) * total + elem] = new_v[bi * total + elem];
}

// Batched KV attention: K query positions, each with causal mask.
// Query position qi attends to [0..cache_len + qi + 1] (past + preceding drafts).
// All K draft tokens must already be in the cache (via kv_cache_append_batch).
// Grid: (num_q, batch_size, 1) — one threadgroup per (head, qi).
kernel void kv_attention_batched(
    device const float* Q_batch  [[buffer(0)]],   // [K, num_q * head_dim]
    device const float* K_cache  [[buffer(1)]],
    device const float* V_cache  [[buffer(2)]],
    device float*       out      [[buffer(3)]],   // [K, num_q * head_dim]
    constant uint&      cache_len[[buffer(4)]],   // positions already in cache BEFORE batch
    constant uint&      batch_size[[buffer(5)]],  // K
    constant uint&      head_dim [[buffer(6)]],
    constant uint&      num_q    [[buffer(7)]],
    constant uint&      num_kv   [[buffer(8)]],
    constant float&     scale    [[buffer(9)]],
    constant uint&      window_size [[buffer(10)]],
    constant float&     softcap  [[buffer(11)]],
    uint tg_linear [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    // Grid is (num_q * batch_size, 1, 1). Decompose linear index.
    uint head = tg_linear % num_q;
    uint qi = tg_linear / num_q;
    if (head >= num_q || qi >= batch_size) return;
    uint kv_head = head / (num_q / num_kv);

    // This query position sees [0..T) where T = cache_len + qi + 1
    uint T = cache_len + qi + 1;
    uint t_start = (window_size > 0 && T > window_size) ? T - window_size : 0;

    device const float* q = Q_batch + qi * num_q * head_dim + head * head_dim;

    threadgroup float tg_scores[1024];

    // Phase 1: Q·K dot products + max
    float local_max = -1e30f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        device const float* k = K_cache + t * num_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d + 3 < head_dim; d += 4) {
            dot += q[d]*k[d] + q[d+1]*k[d+1] + q[d+2]*k[d+2] + q[d+3]*k[d+3];
        }
        for (uint d = (head_dim & ~3u); d < head_dim; d++) dot += q[d] * k[d];
        dot *= scale;
        if (softcap > 0.0f) {
            float arg = clamp(dot / softcap, -30.0f, 30.0f);
            dot = tanh(arg) * softcap;
        }
        tg_scores[t - t_start] = dot;
        local_max = max(local_max, dot);
    }

    float sg_max = simd_max(local_max);
    threadgroup float tg_sg_vals[8];
    if (lane == 0) tg_sg_vals[sg_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = tg_sg_vals[0];
    uint n_sg = (tg_sz + 31) / 32;
    for (uint i = 1; i < n_sg; i++) global_max = max(global_max, tg_sg_vals[i]);

    // Phase 2: softmax
    float local_sum = 0.0f;
    for (uint t = t_start + tid; t < T; t += tg_sz) {
        float w = exp(tg_scores[t - t_start] - global_max);
        tg_scores[t - t_start] = w;
        local_sum += w;
    }
    float sg_sum = simd_sum(local_sum);
    if (lane == 0) tg_sg_vals[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = tg_sg_vals[0];
    for (uint i = 1; i < n_sg; i++) global_sum += tg_sg_vals[i];
    float inv_sum = 1.0f / global_sum;

    for (uint t = t_start + tid; t < T; t += tg_sz) {
        tg_scores[t - t_start] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: weighted V sum
    device float* out_head = out + qi * num_q * head_dim + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_sz) {
        float acc = 0.0f;
        for (uint t = t_start; t < T; t++) {
            acc += tg_scores[t - t_start] * V_cache[t * num_kv * head_dim + kv_head * head_dim + d];
        }
        out_head[d] = acc;
    }
}

// ── Single-token variants (existing) ──

kernel void kv_cache_append(
    device const float* new_k    [[buffer(0)]],
    device const float* new_v    [[buffer(1)]],
    device float*       K_cache  [[buffer(2)]],
    device float*       V_cache  [[buffer(3)]],
    constant uint&      pos      [[buffer(4)]],
    constant uint&      num_kv   [[buffer(5)]],
    constant uint&      head_dim [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = num_kv * head_dim;
    if (tid >= total) return;
    K_cache[pos * total + tid] = new_k[tid];
    V_cache[pos * total + tid] = new_v[tid];
}
"#;
