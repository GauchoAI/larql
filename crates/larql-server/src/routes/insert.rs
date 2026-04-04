//! POST /v1/insert — constellation knowledge insertion.
//!
//! Full trace-guided multi-layer insert: forward pass to capture residuals,
//! use as gate vectors, write down vector overrides with target embedding.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

#[derive(Deserialize)]
pub struct InsertRequest {
    pub entity: String,
    pub relation: String,
    pub target: String,
    #[serde(default)]
    pub layer: Option<usize>,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_alpha() -> f32 { 2.0 }
fn default_confidence() -> f32 { 0.9 }

fn run_insert(
    model: &LoadedModel,
    req: &InsertRequest,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();
    let hidden = model.embeddings.shape()[1];

    // Determine insert layers
    let last = model.config.num_layers.saturating_sub(1);
    let bands = model.config.layer_bands.clone()
        .or_else(|| larql_vindex::LayerBands::for_family(&model.config.family, model.config.num_layers))
        .unwrap_or(larql_vindex::LayerBands {
            syntax: (0, last),
            knowledge: (0, last),
            output: (0, last),
        });

    let insert_layers: Vec<usize> = if let Some(l) = req.layer {
        vec![l]
    } else {
        let mid = (bands.knowledge.0 + bands.knowledge.1) / 2;
        (mid..=bands.knowledge.1).collect()
    };

    // Target embedding for down vector
    let target_encoding = model.tokenizer.encode(req.target.as_str(), false)
        .map_err(|e| ServerError::Internal(format!("tokenize target: {e}")))?;
    let target_ids: Vec<u32> = target_encoding.get_ids().to_vec();
    let target_id = target_ids.first().copied().unwrap_or(0);

    let mut target_embed = vec![0.0f32; hidden];
    for &tok in &target_ids {
        let row = model.embeddings.row(tok as usize);
        for j in 0..hidden { target_embed[j] += row[j] * model.embed_scale; }
    }
    let n = target_ids.len().max(1) as f32;
    for v in &mut target_embed { *v /= n; }

    // Constellation: forward pass to capture residuals as gate vectors
    let residuals: Vec<(usize, Vec<f32>)> = if model.config.has_model_weights {
        let weights = model.get_or_load_weights()
            .map_err(ServerError::InferenceUnavailable)?;

        let prompt = format!("The {} of {} is",
            req.relation.replace('-', " ").replace('_', " "), req.entity);
        let encoding = model.tokenizer.encode(prompt.as_str(), true)
            .map_err(|e| ServerError::Internal(format!("tokenize prompt: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let patched = model.patched.blocking_read();
        let walk_ffn = larql_inference::vindex::WalkFfn::new_with_trace(weights, &*patched, 8092);
        let _result = larql_inference::predict_with_ffn(
            weights, &model.tokenizer, &token_ids, 1, &walk_ffn,
        );

        // Take the exact residuals gate_knn sees (normalized post-attention states)
        walk_ffn.take_residuals().into_iter()
            .filter(|(layer, _)| insert_layers.contains(layer))
            .collect()
    } else {
        Vec::new()
    };

    let use_constellation = !residuals.is_empty();

    // Insert features across layers
    let mut patched = model.patched.blocking_write();
    let mut inserted = 0usize;
    let mut features_inserted = Vec::new();

    for &layer in &insert_layers {
        let feature = match patched.find_free_feature(layer) {
            Some(f) => f,
            None => continue,
        };

        // Gate vector: residual (constellation) or entity embedding (fallback)
        let gate_vec: Vec<f32> = if let Some((_, ref residual)) = residuals.iter().find(|(l, _)| *l == layer) {
            let mut gv = residual.clone();
            if let Some(gate_matrix) = patched.base().gate_vectors_at(layer) {
                let sample = gate_matrix.nrows().min(100);
                if sample > 0 {
                    let avg_norm: f32 = (0..sample)
                        .map(|i| gate_matrix.row(i).dot(&gate_matrix.row(i)).sqrt())
                        .sum::<f32>() / sample as f32;
                    let res_norm: f32 = gv.iter().map(|v| v * v).sum::<f32>().sqrt();
                    if res_norm > 1e-8 && avg_norm > 0.0 {
                        let scale = avg_norm / res_norm;
                        for v in &mut gv { *v *= scale; }
                    }
                }
            }
            gv
        } else {
            // Fallback: entity embedding
            let enc = model.tokenizer.encode(req.entity.as_str(), false)
                .map_err(|e| ServerError::Internal(format!("tokenize entity: {e}")))?;
            let ids = enc.get_ids();
            let mut ev = vec![0.0f32; hidden];
            for &tok in ids {
                let row = model.embeddings.row(tok as usize);
                for j in 0..hidden { ev[j] += row[j] * model.embed_scale; }
            }
            let n = ids.len().max(1) as f32;
            for v in &mut ev { *v /= n; }
            // Normalise
            let norm: f32 = ev.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-8 { for v in &mut ev { *v /= norm; } }
            ev
        };

        let down_vec: Vec<f32> = target_embed.iter().map(|v| v * req.alpha).collect();

        let meta = larql_vindex::FeatureMeta {
            top_token: req.target.clone(),
            top_token_id: target_id,
            c_score: req.confidence,
            top_k: vec![larql_models::TopKEntry {
                token: req.target.clone(),
                token_id: target_id,
                logit: req.confidence,
            }],
        };

        patched.insert_feature(layer, feature, gate_vec, meta);
        patched.set_down_vector(layer, feature, down_vec);

        features_inserted.push(serde_json::json!({"layer": layer, "feature": feature}));
        inserted += 1;
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(serde_json::json!({
        "entity": req.entity,
        "relation": req.relation,
        "target": req.target,
        "inserted": inserted,
        "mode": if use_constellation { "constellation" } else { "embedding" },
        "alpha": req.alpha,
        "features": features_inserted,
        "latency_ms": (latency_ms * 10.0).round() / 10.0,
    }))
}

pub async fn handle_insert(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let result = tokio::task::spawn_blocking(move || run_insert(&model, &req))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

pub async fn handle_insert_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Json(req): Json<InsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(Some(&model_id))
        .ok_or_else(|| ServerError::NotFound(format!("model '{}' not found", model_id)))?;
    let model = Arc::clone(model);
    let result = tokio::task::spawn_blocking(move || run_insert(&model, &req))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
