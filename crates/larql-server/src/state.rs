//! AppState: loaded vindex + config, shared across all handlers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use larql_inference::ComputeBackend;
use larql_models::ModelWeights;
use larql_vindex::{PatchedVindex, VindexConfig, tokenizers};
use tokio::sync::RwLock;

use crate::cache::DescribeCache;
use crate::session::SessionManager;

/// A single loaded model.
pub struct LoadedModel {
    /// Model ID derived from config (e.g., "gemma-3-4b-it").
    pub id: String,
    /// Vindex directory on disk.
    pub path: PathBuf,
    /// Vindex config (index.json).
    pub config: VindexConfig,
    /// Base index with patch overlay (starts with no patches).
    pub patched: RwLock<PatchedVindex>,
    /// Embeddings matrix + scale factor, loaded once.
    pub embeddings: larql_models::WeightArray,
    pub embed_scale: f32,
    /// Tokenizer for embedding lookups.
    pub tokenizer: tokenizers::Tokenizer,
    /// Whether inference is disabled (--no-infer).
    pub infer_disabled: bool,
    /// Model weights, lazy-loaded on first INFER request.
    pub weights: std::sync::OnceLock<ModelWeights>,
    /// ComputeBackend for the fast Metal path (mode=fast, mode=knn).
    /// Lazy-initialized on first use; serializes inference requests via
    /// `inference_lock` because the KV cache inside is shared mutable state.
    pub backend: std::sync::OnceLock<std::sync::Arc<dyn ComputeBackend>>,
    /// Serializes access to the Metal backend's KV cache. Held around any
    /// `predict_honest_with_knn` / `capture_residual_post_attn_norm` call.
    pub inference_lock: std::sync::Mutex<()>,
    /// Probe-confirmed feature labels: (layer, feature) → relation name.
    /// Loaded from feature_labels.json if present.
    pub probe_labels: HashMap<(usize, usize), String>,
    /// Optional GGUF weight source. When `weights.gguf` is present in the
    /// vindex dir, `mode=fast` reads attention/FFN weights from this pipeline
    /// instead of the (possibly stale) Q4_K binaries. The vindex still
    /// supplies KNN store, embeddings (for KNN), tokenizer, and patches.
    pub gguf: Option<Arc<larql_inference::gguf_pipeline::GgufPipeline>>,
}

impl LoadedModel {
    /// Get or lazy-load model weights for inference. Drops f32 FFN tensors
    /// when Q4_K FFN weights exist on disk (saves ~10.7 GB — the Q4_K pipeline
    /// reads from the mmap'd interleaved_q4k_real.bin, not from f32 tensors).
    /// Also drops in walk-only mode as before.
    pub fn get_or_load_weights(&self) -> Result<&ModelWeights, String> {
        if let Some(w) = self.weights.get() {
            return Ok(w);
        }
        // Skip loading f32 tensors when Q4_K equivalents exist on disk.
        // Never allocates them — no load-then-drop, no malloc arena ghost.
        let has_q4k_ffn = self.path.join("interleaved_q4k_real.bin").exists();
        let has_q4k_attn = self.path.join("attn_weights_q4k.bin").exists();
        let has_q4_lm = self.path.join("lm_head_q4.bin").exists();
        let mut skip: Vec<&str> = Vec::new();
        if has_q4k_ffn {
            skip.extend_from_slice(&["gate_proj", "up_proj", "down_proj",
                "ffn_gate", "ffn_up", "ffn_down", "mlp.experts",
                "packed_gate_up", "packed_down"]);
        }
        if has_q4k_attn {
            skip.extend_from_slice(&["q_proj", "k_proj", "v_proj", "o_proj",
                "attn_q", "attn_k", "attn_v", "attn_output"]);
        }
        if has_q4_lm { skip.push("lm_head"); }

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let mut weights = larql_vindex::load_model_weights_filtered(&self.path, &mut cb, &skip)
            .map_err(|e| format!("failed to load model weights: {e}"))?;
        if !skip.is_empty() {
            tracing::info!("[skip-load] skipped {} tensor patterns — never allocated", skip.len());
        }
        // Share embedding matrix: replace weights.embed with a clone of
        // LoadedModel.embeddings (same ArcArray2, one allocation).
        // Saves ~2.7 GB by deduplicating the embedding matrix.
        let old_embed_size = weights.embed.len() * 4;
        weights.embed = self.embeddings.clone(); // ArcArray2 clone = refcount bump, not copy
        tracing::info!("[shared embed] deduplicated embedding matrix: {:.1} GB freed",
            old_embed_size as f64 / 1e9);
        let _ = self.weights.set(weights);
        Ok(self.weights.get().unwrap())
    }

    /// Get or lazy-init the shared ComputeBackend (Metal when available).
    pub fn get_or_init_backend(&self) -> &std::sync::Arc<dyn ComputeBackend> {
        self.backend.get_or_init(|| {
            std::sync::Arc::from(larql_inference::default_backend())
        })
    }
}

/// Shared application state.
pub struct AppState {
    /// Loaded models, keyed by model ID.
    pub models: Vec<Arc<LoadedModel>>,
    /// Server start time for uptime reporting.
    pub started_at: std::time::Instant,
    /// Request counter.
    pub requests_served: std::sync::atomic::AtomicU64,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
    /// Per-session PatchedVindex manager.
    pub sessions: SessionManager,
    /// DESCRIBE result cache.
    pub describe_cache: DescribeCache,
}

impl AppState {
    /// Get model by ID, or the only model if single-model serving.
    pub fn model(&self, id: Option<&str>) -> Option<&Arc<LoadedModel>> {
        match id {
            Some(id) => self.models.iter().find(|m| m.id == id),
            None if self.models.len() == 1 => self.models.first(),
            None => None,
        }
    }

    /// Whether this is multi-model serving.
    pub fn is_multi_model(&self) -> bool {
        self.models.len() > 1
    }

    pub fn bump_requests(&self) {
        self.requests_served
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Load probe-confirmed feature labels from feature_labels.json.
/// Format: {"L{layer}_F{feature}": "relation_name", ...}
pub fn load_probe_labels(vindex_path: &std::path::Path) -> HashMap<(usize, usize), String> {
    let path = vindex_path.join("feature_labels.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return HashMap::new(),
    };
    let obj: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    let map = match obj.as_object() {
        Some(m) => m,
        None => return HashMap::new(),
    };

    let mut labels = HashMap::new();
    for (key, value) in map {
        if let Some(rel) = value.as_str() {
            let parts: Vec<&str> = key.split('_').collect();
            if parts.len() == 2 {
                if let (Some(layer), Some(feat)) = (
                    parts[0].strip_prefix('L').and_then(|s| s.parse::<usize>().ok()),
                    parts[1].strip_prefix('F').and_then(|s| s.parse::<usize>().ok()),
                ) {
                    labels.insert((layer, feat), rel.to_string());
                }
            }
        }
    }
    labels
}

/// Derive a short model ID from the full model name.
/// "google/gemma-3-4b-it" → "gemma-3-4b-it"
pub fn model_id_from_name(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_string()
}
