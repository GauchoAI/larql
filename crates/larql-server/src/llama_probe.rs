//! A single `ProbeHandler` that multiplexes the three operating modes
//! the server needs across requests: idle (no-op), capture (grab the
//! last-token residual at a layer for /v1/insert), or knn-query (fire
//! one-shot token override for /v1/infer).
//!
//! llama.cpp doesn't let us swap the `cb_eval` user_data after context
//! creation, so we keep a single handler alive for the pipeline's life
//! and mutate its mode per request via an `Arc<Mutex<ServerProbeState>>`
//! shared between route handlers and the probe.

use std::sync::{Arc, Mutex};

use larql_llamacpp::{ProbeHandler, ProbeNode};
use larql_vindex::patch::knn_store::KnnStore;

/// Per-request output from a KNN query pass.
#[derive(Debug, Clone)]
pub struct KnnMatch {
    pub entity: String,
    pub relation: String,
    pub target_token: String,
    pub target_id: i32,
    pub cosine: f32,
}

/// What the probe should do during the next decode pass.
#[derive(Default)]
pub enum Mode {
    #[default]
    Idle,
    /// Capture the last-token residual at `layer` on tensor `tensor_name`.
    /// When the pass completes, `captured` holds the f32 vector.
    Capture {
        layer: u32,
        tensor_name: String,
        captured: Option<Vec<f32>>,
    },
    /// Query the KnnStore at `layer` / `tensor_name`.  If top-1 cosine
    /// exceeds `threshold`, force the matching token (one-shot) and
    /// record the match in `result`.
    KnnQuery {
        layer: u32,
        tensor_name: String,
        threshold: f32,
        entries: Vec<KnnEntrySnapshot>,
        fired: bool,
        forced: Option<i32>,
        result: Option<KnnMatch>,
        /// Best cosine seen during observation, even when below
        /// threshold.  Diagnostic: tells us if the probe sees anything
        /// vaguely similar to stored keys.
        best_cosine: Option<f32>,
    },
}

/// A trimmed, server-owned snapshot of a `KnnEntry` — no `Mutex` on the
/// source store needs to be held while the probe is running.
#[derive(Clone)]
pub struct KnnEntrySnapshot {
    pub key_normalized: Vec<f32>,
    pub target_id: i32,
    pub target_token: String,
    pub entity: String,
    pub relation: String,
}

#[derive(Default)]
pub struct ServerProbeState {
    pub mode: Mode,
}

/// Handler that reads `mode` from shared state to decide what to do on
/// each cb_eval callback.  Cheap to clone (just clones the Arc).
#[derive(Clone)]
pub struct ServerProbe {
    state: Arc<Mutex<ServerProbeState>>,
}

impl ServerProbe {
    pub fn new(state: Arc<Mutex<ServerProbeState>>) -> Self {
        Self { state }
    }
}

/// Snapshot the entries of `store` at `layer`, L2-norm enforced by the
/// store itself on insert (see `KnnStore::add`).  Used to feed the
/// probe's `KnnQuery` mode without holding a lock during decode.
pub fn snapshot_layer(store: &KnnStore, layer: usize) -> Vec<KnnEntrySnapshot> {
    let entries = match store.entries().get(&layer) {
        Some(e) => e,
        None => return Vec::new(),
    };
    entries
        .iter()
        .map(|e| KnnEntrySnapshot {
            key_normalized: e.key.clone(),
            target_id: e.target_id as i32,
            target_token: e.target_token.clone(),
            entity: e.entity.clone(),
            relation: e.relation.clone(),
        })
        .collect()
}

impl ProbeHandler for ServerProbe {
    fn wants(&self, node: &ProbeNode<'_>) -> bool {
        let s = self.state.lock().unwrap();
        match &s.mode {
            Mode::Idle => false,
            Mode::Capture { layer, tensor_name, .. } => {
                node.layer == Some(*layer) && node.name == *tensor_name
            }
            Mode::KnnQuery {
                layer,
                tensor_name,
                fired,
                entries,
                ..
            } => {
                !*fired
                    && !entries.is_empty()
                    && node.layer == Some(*layer)
                    && node.name == *tensor_name
            }
        }
    }

    fn observe(&mut self, node: &ProbeNode<'_>, data: &[f32]) -> Option<Vec<f32>> {
        let n_embd = node.shape[0] as usize;
        let n_tokens = node.shape[1] as usize;
        if n_embd == 0 || n_tokens == 0 {
            return None;
        }
        let last_off = (n_tokens - 1) * n_embd;
        let last = &data[last_off..last_off + n_embd];

        let mut s = self.state.lock().unwrap();
        match &mut s.mode {
            Mode::Idle => {}
            Mode::Capture { captured, .. } => {
                *captured = Some(last.to_vec());
            }
            Mode::KnnQuery {
                threshold,
                entries,
                fired,
                forced,
                result,
                best_cosine,
                ..
            } => {
                if *fired {
                    return None;
                }
                let norm: f32 = last.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-12 {
                    return None;
                }
                let query: Vec<f32> = last.iter().map(|x| x / norm).collect();

                let mut best: Option<(usize, f32)> = None;
                for (i, e) in entries.iter().enumerate() {
                    if e.key_normalized.len() != n_embd {
                        continue;
                    }
                    let dot: f32 = e
                        .key_normalized
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    if best.map_or(true, |(_, c)| dot > c) {
                        best = Some((i, dot));
                    }
                }
                if let Some((idx, cos)) = best {
                    *best_cosine = Some(cos);
                    if cos > *threshold {
                        let e = &entries[idx];
                        *forced = Some(e.target_id);
                        *result = Some(KnnMatch {
                            entity: e.entity.clone(),
                            relation: e.relation.clone(),
                            target_token: e.target_token.clone(),
                            target_id: e.target_id,
                            cosine: cos,
                        });
                    }
                }
            }
        }
        None
    }

    fn forced_token(&self) -> Option<i32> {
        let s = self.state.lock().unwrap();
        match &s.mode {
            Mode::KnnQuery {
                fired,
                forced,
                ..
            } if !*fired => *forced,
            _ => None,
        }
    }

    fn reset_step(&mut self) {
        let mut s = self.state.lock().unwrap();
        if let Mode::KnnQuery { fired, forced, .. } = &mut s.mode {
            if forced.is_some() {
                *fired = true;
                *forced = None;
            }
        }
    }
}
