//! Performance accumulator — record per-component timings during inference,
//! then dump a structured markdown report at the end.
//!
//! Enable with `LARQL_PERF_RECORD=1` (or `enable()` programmatically). Trace
//! points (in attention/gpu.rs, layer_graph/predict.rs, vindex/walk_ffn.rs,
//! forward/layer.rs) push micros into a global keyed accumulator. The bench
//! (or any caller) calls `report_markdown(meta)` to render a markdown report
//! shaped like PERFORMANCE.md (headline + per-token decomposition + per-layer
//! sub-components + raw accumulator).
//!
//! Hot path is lock-free: a `RwLock<HashMap>` is read once to look up the
//! atomic accumulator for a key (registers it on first sight), then 3-5 atomic
//! ops on that accumulator. Zero allocation per record after first registration.
//! Reservoir-sampled percentile storage (1024 samples per key, ring-buffer)
//! lets us report p50/p95 without per-call alloc.

use std::collections::HashMap;
use std::sync::{RwLock, OnceLock, Arc};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Reservoir size per key for percentile estimation. Power of two for fast modulo.
const RESERVOIR: usize = 1024;
const RESERVOIR_MASK: usize = RESERVOIR - 1;

/// Lock-free per-key accumulator. All updates go through atomic ops; the
/// reservoir uses a ring-buffer (last-N samples) which is good enough for
/// p50/p95 estimation on stationary distributions.
pub struct AtomicAccum {
    pub total_us: AtomicU64,
    pub count: AtomicU64,
    pub min_us: AtomicU64,
    pub max_us: AtomicU64,
    /// Ring-buffer of recent samples. `next_idx` is the write position.
    pub reservoir: [AtomicU64; RESERVOIR],
    pub next_idx: AtomicUsize,
}

impl AtomicAccum {
    fn new() -> Self {
        Self {
            total_us: AtomicU64::new(0),
            count: AtomicU64::new(0),
            min_us: AtomicU64::new(u64::MAX),
            max_us: AtomicU64::new(0),
            reservoir: std::array::from_fn(|_| AtomicU64::new(0)),
            next_idx: AtomicUsize::new(0),
        }
    }

    fn add(&self, us: u64) {
        self.total_us.fetch_add(us, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        // Compare-exchange not needed — fetch_min/max are atomic.
        self.min_us.fetch_min(us, Ordering::Relaxed);
        self.max_us.fetch_max(us, Ordering::Relaxed);
        let i = self.next_idx.fetch_add(1, Ordering::Relaxed) & RESERVOIR_MASK;
        self.reservoir[i].store(us, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.total_us.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
        self.min_us.store(u64::MAX, Ordering::Relaxed);
        self.max_us.store(0, Ordering::Relaxed);
        self.next_idx.store(0, Ordering::Relaxed);
        // Don't bother zeroing reservoir; next_idx wrap-around will overwrite.
    }
}

/// Snapshot read by report formatter — owns its data so the live accumulator
/// stays untouched.
#[derive(Default, Clone)]
pub struct AccumStats {
    pub total_us: u128,
    pub count: u64,
    pub min_us: u128,
    pub max_us: u128,
    pub samples_us: Vec<u32>,
}

impl AccumStats {
    fn from_atomic(a: &AtomicAccum) -> Self {
        let count = a.count.load(Ordering::Relaxed);
        let next = a.next_idx.load(Ordering::Relaxed);
        let n_samples = (count as usize).min(RESERVOIR);
        // Read out reservoir contents — newest `n_samples` are valid.
        let mut samples_us = Vec::with_capacity(n_samples);
        let start = next.saturating_sub(n_samples) & !0; // logical start; modulo applied below
        for i in 0..n_samples {
            let idx = (start + i) & RESERVOIR_MASK;
            samples_us.push(a.reservoir[idx].load(Ordering::Relaxed) as u32);
        }
        Self {
            total_us: a.total_us.load(Ordering::Relaxed) as u128,
            count,
            min_us: a.min_us.load(Ordering::Relaxed) as u128,
            max_us: a.max_us.load(Ordering::Relaxed) as u128,
            samples_us,
        }
    }

    pub fn mean_ms(&self) -> f64 { (self.total_us as f64) / (self.count.max(1) as f64) / 1000.0 }
    pub fn min_ms(&self) -> f64 { if self.count == 0 { 0.0 } else { self.min_us as f64 / 1000.0 } }
    pub fn max_ms(&self) -> f64 { self.max_us as f64 / 1000.0 }
    pub fn total_ms(&self) -> f64 { self.total_us as f64 / 1000.0 }

    pub fn pct_ms(&self, p: f64) -> f64 {
        if self.samples_us.is_empty() { return 0.0; }
        let mut s = self.samples_us.clone();
        s.sort_unstable();
        let n = s.len();
        let idx = ((p / 100.0) * (n.saturating_sub(1) as f64)).round() as usize;
        s[idx.min(n - 1)] as f64 / 1000.0
    }
}

fn store() -> &'static RwLock<HashMap<&'static str, Arc<AtomicAccum>>> {
    static S: OnceLock<RwLock<HashMap<&'static str, Arc<AtomicAccum>>>> = OnceLock::new();
    S.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Fast path: one read-lock acquisition (cheap) + atomic ops. Falls back to
/// write-lock to register on first sight of a new key.
#[inline]
pub fn record(component: &'static str, micros: u128) {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let us = micros as u64;
    {
        let m = store().read().unwrap();
        if let Some(a) = m.get(component) {
            a.add(us);
            return;
        }
    }
    // First-sight: register, then add.
    let mut m = store().write().unwrap();
    let entry = m.entry(component).or_insert_with(|| Arc::new(AtomicAccum::new()));
    entry.add(us);
}

pub fn enable() { ENABLED.store(true, Ordering::Relaxed); }
pub fn disable() { ENABLED.store(false, Ordering::Relaxed); }
#[inline]
pub fn is_enabled() -> bool { ENABLED.load(Ordering::Relaxed) }

/// Reset all accumulators (zero counters, drop reservoirs).
pub fn reset() {
    let m = store().read().unwrap();
    for a in m.values() { a.reset(); }
}

/// Snapshot all accumulators for reporting.
pub fn snapshot() -> HashMap<&'static str, AccumStats> {
    let m = store().read().unwrap();
    m.iter().map(|(k, a)| (*k, AccumStats::from_atomic(a))).collect()
}

/// Run metadata captured by the caller and embedded in the markdown report.
pub struct ReportMeta<'a> {
    pub model: &'a str,
    pub vindex: &'a str,
    pub prompt: &'a str,
    pub prompt_tokens: usize,
    pub decode_tokens: usize,
    pub n_runs: usize,
    pub backend: &'a str,
    pub generated: &'a str,
    pub decode_tok_s_mean: f64,
    pub decode_ms_mean: f64,
    pub decode_ms_std: f64,
    pub decode_ms_min: f64,
    pub decode_ms_max: f64,
    pub prefill_ms_mean: f64,
    pub prefill_ms_std: f64,
    pub prefill_ms_min: f64,
    pub prefill_ms_max: f64,
    pub per_step_decode_ms: &'a [f64],
}

pub fn report_markdown(meta: &ReportMeta) -> String {
    use std::fmt::Write;
    let snap = snapshot();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let total_decode_steps = meta.n_runs * meta.decode_tokens.saturating_sub(1).max(1);

    let get = |name: &str| snap.get(name).cloned().unwrap_or_default();

    // For a per-decode-step component, sum_us across all decode-step records
    // divided by the number of decode steps gives per-step ms.
    let per_decode = |name: &str| -> f64 {
        let s = get(name);
        if s.count == 0 { return 0.0; }
        s.total_ms() / total_decode_steps as f64
    };

    let attn = per_decode("decode.attn");
    let ffn  = per_decode("decode.ffn");
    let knn  = per_decode("decode.knn_overlay");
    let lmh  = per_decode("decode.lm_head");
    let sum_components = attn + ffn + knn + lmh;
    let untracked = (meta.decode_ms_mean - sum_components).max(0.0);

    let pct = |v: f64| if meta.decode_ms_mean > 0.0 { v * 100.0 / meta.decode_ms_mean } else { 0.0 };

    let mut out = String::new();
    let _ = writeln!(out, "# larql perf report");
    let _ = writeln!(out);
    let _ = writeln!(out, "Generated: unix={now} | backend={} | model=`{}` | vindex=`{}`", meta.backend, meta.model, meta.vindex);
    let _ = writeln!(out, "Run shape: {} runs × (prefill {}-token + decode {}-token)", meta.n_runs, meta.prompt_tokens, meta.decode_tokens);
    let _ = writeln!(out, "Prompt: `{}`", meta.prompt);
    let _ = writeln!(out, "Recording: only enabled during decode steps (prefill walks excluded so per-component costs reflect single-token decode shape).");
    let _ = writeln!(out);

    let _ = writeln!(out, "## 1. Headline numbers");
    let _ = writeln!(out);
    let _ = writeln!(out, "| Metric | Value | Range |");
    let _ = writeln!(out, "|---|---:|---:|");
    let _ = writeln!(out, "| Decode tok/s (mean) | **{:.2}** | {:.2} – {:.2} |", meta.decode_tok_s_mean, 1000.0/meta.decode_ms_max.max(1e-6), 1000.0/meta.decode_ms_min.max(1e-6));
    let _ = writeln!(out, "| Decode ms/tok | {:.1} ± {:.1} | min {:.1}, max {:.1} |", meta.decode_ms_mean, meta.decode_ms_std, meta.decode_ms_min, meta.decode_ms_max);
    let _ = writeln!(out, "| Prefill ms | {:.0} ± {:.0} | min {:.0}, max {:.0} |", meta.prefill_ms_mean, meta.prefill_ms_std, meta.prefill_ms_min, meta.prefill_ms_max);
    let _ = writeln!(out);
    let _ = writeln!(out, "Generated text: `{}`", meta.generated.replace('\n', "\\n"));
    let _ = writeln!(out);

    let _ = writeln!(out, "## 2. Per-token decomposition (decode)");
    let _ = writeln!(out);
    let _ = writeln!(out, "| Component | ms/tok | % of decode | sample count |");
    let _ = writeln!(out, "|---|---:|---:|---:|");
    let _ = writeln!(out, "| Attention (Metal KV decode)   | {:>5.1} | {:>4.1}% | {} |", attn, pct(attn), get("decode.attn").count);
    let _ = writeln!(out, "| Walk FFN                       | {:>5.1} | {:>4.1}% | {} |", ffn,  pct(ffn),  get("decode.ffn").count);
    let _ = writeln!(out, "| KNN overlay                    | {:>5.1} | {:>4.1}% | {} |", knn,  pct(knn),  get("decode.knn_overlay").count);
    let _ = writeln!(out, "| LM head KNN                    | {:>5.1} | {:>4.1}% | {} |", lmh,  pct(lmh),  get("decode.lm_head").count);
    let _ = writeln!(out, "| _Untracked (instrumentation+scaffolding)_ | {:>5.1} | {:>4.1}% | — |", untracked, pct(untracked));
    let _ = writeln!(out, "| **Total decode**               | **{:>5.1}** | **100%** | {} |", meta.decode_ms_mean, total_decode_steps);
    let _ = writeln!(out);

    // ── Section 3: Attention sub-components (per-layer) ──
    let kv_read = get("attn.kv_read");
    let qkv = get("attn.qkv_proj");
    let softmax = get("attn.softmax");
    let o_proj = get("attn.o_proj");
    let kv_write = get("attn.kv_write");
    let attn_calls = kv_read.count + qkv.count + softmax.count + o_proj.count + kv_write.count;
    if attn_calls > 0 {
        let layers_per_step = (kv_read.count.max(1) / get("decode.attn").count.max(1)) as usize;
        let _ = writeln!(out, "## 3. Attention sub-components (per-layer)");
        let _ = writeln!(out);
        let _ = writeln!(out, "Calls per decode step ≈ {} layers. Per-token = mean × layers (CPU/Metal mix).", layers_per_step);
        let _ = writeln!(out);
        let _ = writeln!(out, "| Sub-component   | mean ms | p50 | p95 | max | per-token (×layers) |");
        let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|");
        let mut attn_per_tok = 0.0;
        for (name, s) in [
            ("KV cache read",   &kv_read),
            ("QKV projections", &qkv),
            ("Softmax + V sum", &softmax),
            ("O projection",    &o_proj),
            ("KV cache write",  &kv_write),
        ] {
            let per_tok = s.mean_ms() * layers_per_step as f64;
            attn_per_tok += per_tok;
            let _ = writeln!(out, "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2} |",
                name, s.mean_ms(), s.pct_ms(50.0), s.pct_ms(95.0), s.max_ms(), per_tok);
        }
        let attn_unaccounted = (attn - attn_per_tok).max(0.0);
        let _ = writeln!(out, "| _Untracked (RoPE, biases, norms, residual)_ | — | — | — | — | {:.2} |", attn_unaccounted);
        let _ = writeln!(out, "| **Per-token total** | | | | | **{:.2}** |", attn_per_tok + attn_unaccounted);
        let _ = writeln!(out);
    }

    // ── Section 4: Walk FFN sub-components (per-layer) ──
    let gu = get("walk.gate_up_dispatch");
    let act = get("walk.activation");
    let down = get("walk.down_dispatch");
    let pre_norm = get("walk.pre_ffn_norm");
    let post_norm = get("walk.post_ffn_norm_residual");
    let walk_calls = gu.count + act.count + down.count;
    if walk_calls > 0 {
        let layers_per_step = (gu.count.max(1) / get("decode.ffn").count.max(1)) as usize;
        let _ = writeln!(out, "## 4. Walk FFN sub-components (per-layer)");
        let _ = writeln!(out);
        let _ = writeln!(out, "Calls per decode step ≈ {} layers (run_ffn fires once per transformer layer).", layers_per_step);
        let _ = writeln!(out);
        let _ = writeln!(out, "| Sub-component                  | mean ms | p50 | p95 | max | per-token (×layers) |");
        let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|");
        let mut walk_per_tok = 0.0;
        for (name, s) in [
            ("pre_ffn_norm (CPU)",                 &pre_norm),
            ("gate+up dispatch (Metal q4k_pair)",  &gu),
            ("GEGLU activation loop (CPU)",        &act),
            ("down dispatch (Metal q4k_matvec)",   &down),
            ("post_ffn_norm + residual (CPU)",     &post_norm),
        ] {
            let per_tok = s.mean_ms() * layers_per_step as f64;
            walk_per_tok += per_tok;
            let _ = writeln!(out, "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2} |",
                name, s.mean_ms(), s.pct_ms(50.0), s.pct_ms(95.0), s.max_ms(), per_tok);
        }
        let walk_unaccounted = (ffn - walk_per_tok).max(0.0);
        let _ = writeln!(out, "| _Untracked_                    | — | — | — | — | {:.2} |", walk_unaccounted);
        let _ = writeln!(out, "| **Per-token total**            | | | | | **{:.2}** |", walk_per_tok + walk_unaccounted);
        let _ = writeln!(out);
    }

    // ── Section 5: Per-step decode times — surfaces warmup pattern ──
    if !meta.per_step_decode_ms.is_empty() {
        let _ = writeln!(out, "## 5. Per-step decode times (warmup visibility)");
        let _ = writeln!(out);
        let mut sorted = meta.per_step_decode_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50 = sorted[sorted.len() / 2];
        let p95_idx = (sorted.len() as f64 * 0.95) as usize;
        let p95 = sorted[p95_idx.min(sorted.len() - 1)];
        let _ = writeln!(out, "Distribution across **{}** steps: p50 = {:.1} ms, p95 = {:.1} ms, max = {:.1} ms.", sorted.len(), p50, p95, sorted[sorted.len() - 1]);
        let _ = writeln!(out);
        let _ = writeln!(out, "First 12 steps (warmup window — Metal pipeline specialisation, mmap page faults):");
        let _ = writeln!(out);
        let _ = writeln!(out, "| step | ms | vs p50 |");
        let _ = writeln!(out, "|---:|---:|---:|");
        for (i, &ms) in meta.per_step_decode_ms.iter().take(12).enumerate() {
            let mark = if ms > p50 * 1.5 { " ⚠ slow" } else if ms < p50 * 0.8 { " (fast)" } else { "" };
            let _ = writeln!(out, "| {} | {:.1} | {:+.0}%{} |", i, ms, (ms - p50) / p50 * 100.0, mark);
        }
        if meta.per_step_decode_ms.len() > 12 {
            let _ = writeln!(out, "| _… + {} more_ |  |  |", meta.per_step_decode_ms.len() - 12);
        }
        let _ = writeln!(out);
    }

    // ── Section 6: Raw accumulator ──
    let _ = writeln!(out, "## 6. Raw accumulator");
    let _ = writeln!(out);
    let _ = writeln!(out, "| key | calls | mean ms | p50 | p95 | min | max | total ms |");
    let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|");
    let mut keys: Vec<&str> = snap.keys().copied().collect();
    keys.sort();
    for k in keys {
        let s = &snap[&k];
        let _ = writeln!(out, "| `{}` | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1} |",
            k, s.count, s.mean_ms(), s.pct_ms(50.0), s.pct_ms(95.0), s.min_ms(), s.max_ms(), s.total_ms());
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "_Generated by `larql_inference::perf::report_markdown`. Recording uses lock-free atomic accumulators with reservoir-sampled percentiles ({} samples per key)._", RESERVOIR);

    out
}
