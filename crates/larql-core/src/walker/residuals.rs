//! Capture residual stream vectors for entities via forward passes.
//!
//! Runs a tokenized prompt through transformer layers and captures the
//! last-token hidden state at specified layers. Output is NDJSON compatible
//! with vector-load for SurrealDB ingestion.

use std::path::Path;

use super::forward::capture_residuals;
use super::safetensors_loader::{load_model_dir, WalkerError};
use super::utils::current_date;
use super::vector_extractor::{TopKEntry, VectorFileHeader, VectorRecord, VectorWriter};
use super::weight_walker::resolve_model_path;

/// A single residual capture request.
pub struct ResidualConfig {
    pub layers: Vec<usize>,
    pub prompt_template: Option<String>,
}

impl Default for ResidualConfig {
    fn default() -> Self {
        Self {
            layers: vec![25],
            prompt_template: None,
        }
    }
}

/// Callbacks for residual capture progress.
pub trait ResidualCallbacks {
    fn on_entity_start(&mut self, _entity: &str, _index: usize, _total: usize) {}
    fn on_entity_done(&mut self, _entity: &str, _layers_captured: usize, _elapsed_ms: f64) {}
}

pub struct SilentResidualCallbacks;
impl ResidualCallbacks for SilentResidualCallbacks {}

/// Loaded model ready for residual capture.
pub struct ResidualCapturer {
    weights: super::safetensors_loader::ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
}

impl ResidualCapturer {
    pub fn load(model: &str) -> Result<Self, WalkerError> {
        let model_path = resolve_model_path(model)?;
        let weights = load_model_dir(&model_path)?;

        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(WalkerError::MissingTensor(
                "tokenizer.json not found".into(),
            ));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| WalkerError::Parse(e.to_string()))?;

        Ok(Self {
            weights,
            tokenizer,
            model_name: model.to_string(),
        })
    }

    pub fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.weights.hidden_size
    }

    /// Capture residuals for a list of entities and write to NDJSON.
    pub fn capture(
        &self,
        entities: &[String],
        config: &ResidualConfig,
        output_path: &Path,
        callbacks: &mut dyn ResidualCallbacks,
    ) -> Result<usize, WalkerError> {
        let mut writer = VectorWriter::create(output_path)?;
        writer.write_header(&VectorFileHeader {
            _header: true,
            component: "residuals".to_string(),
            model: self.model_name.clone(),
            dimension: self.weights.hidden_size,
            extraction_date: current_date(),
        })?;

        let total = entities.len();
        let mut count = 0;

        for (i, entity) in entities.iter().enumerate() {
            let start = std::time::Instant::now();
            callbacks.on_entity_start(entity, i, total);

            // Build prompt — bare entity or template
            let prompt = match &config.prompt_template {
                Some(tmpl) => tmpl.replace("{entity}", entity),
                None => entity.clone(),
            };

            // Tokenize
            let encoding = self
                .tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| WalkerError::Parse(format!("tokenize error: {e}")))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            if token_ids.is_empty() {
                continue;
            }

            // Forward pass — capture residuals at requested layers
            let residuals = capture_residuals(&self.weights, &token_ids, &config.layers);

            // Write each layer's residual
            for (layer, vector) in &residuals {
                // Project residual onto embedding to get top-k tokens
                let top_k = project_to_vocab(&self.weights.embed, vector, 10, &self.tokenizer);

                let (top_token, top_token_id, c_score) = if let Some(first) = top_k.first() {
                    (first.token.clone(), first.token_id, first.logit)
                } else {
                    (String::new(), 0, 0.0)
                };

                writer.write_record(&VectorRecord {
                    id: format!("{entity}_L{layer}"),
                    layer: *layer,
                    feature: 0, // not a feature — it's a residual
                    dim: vector.len(),
                    vector: vector.clone(),
                    top_token,
                    top_token_id,
                    c_score,
                    top_k,
                })?;
                count += 1;
            }

            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            callbacks.on_entity_done(entity, residuals.len(), elapsed_ms);
        }

        writer.flush()?;
        Ok(count)
    }
}

/// Project a residual vector onto the embedding matrix to find top-k tokens.
fn project_to_vocab(
    embed: &ndarray::Array2<f32>,
    residual: &[f32],
    k: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<TopKEntry> {
    let vocab_size = embed.shape()[0];
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(vocab_size);

    for i in 0..vocab_size {
        let row = embed.row(i);
        let dot: f32 = row.iter().zip(residual.iter()).map(|(a, b)| a * b).sum();
        scores.push((i, dot));
    }

    let k = k.min(scores.len());
    if k > 0 {
        scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    }
    scores.truncate(k);
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    scores
        .into_iter()
        .filter_map(|(idx, logit)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(|token| TopKEntry {
                    token,
                    token_id: idx as u32,
                    logit,
                })
        })
        .collect()
}

