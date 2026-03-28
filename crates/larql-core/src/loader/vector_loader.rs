//! Read intermediate vector NDJSON files and generate SurrealDB SQL for loading.
//!
//! Pure functions — no database I/O. The CLI module handles HTTP communication
//! to SurrealDB. This module provides:
//! - Streaming NDJSON reader for VectorRecord files
//! - Schema DDL generation per component type
//! - Batch INSERT SQL generation

use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::walker::vector_extractor::{VectorFileHeader, VectorRecord, ALL_COMPONENTS};

#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unknown component: {0}")]
    UnknownComponent(String),
    #[error("no vector files found in {0}")]
    NoVectorFiles(PathBuf),
    #[error("surreal error: {0}")]
    Surreal(String),
}

/// Configuration for the vector loader.
pub struct LoadConfig {
    pub tables: Option<Vec<String>>,
    pub layers: Option<Vec<usize>>,
    pub batch_size: usize,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            tables: None,
            layers: None,
            batch_size: 500,
        }
    }
}

/// Callbacks for load progress.
pub trait LoadCallbacks {
    fn on_table_start(&mut self, _table: &str, _total_records: usize) {}
    fn on_batch_done(&mut self, _table: &str, _batch_num: usize, _records_loaded: usize) {}
    fn on_table_done(&mut self, _table: &str, _total_loaded: usize, _elapsed_ms: f64) {}
}

pub struct SilentLoadCallbacks;
impl LoadCallbacks for SilentLoadCallbacks {}

/// Summary of a load run.
pub struct LoadSummary {
    pub tables: Vec<TableSummary>,
    pub total_records: usize,
    pub elapsed_secs: f64,
}

pub struct TableSummary {
    pub table: String,
    pub records_loaded: usize,
    pub elapsed_secs: f64,
}

// ═══════════════════════════════════════════════════
// NDJSON Reader
// ═══════════════════════════════════════════════════

/// Streaming reader for .vectors.jsonl files.
pub struct VectorReader {
    reader: BufReader<std::fs::File>,
    header: VectorFileHeader,
    line_buf: String,
}

impl VectorReader {
    /// Open a .vectors.jsonl file and read its header.
    pub fn open(path: &Path) -> Result<Self, LoaderError> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header line
        let mut header_line = String::new();
        reader
            .read_line(&mut header_line)
            .map_err(|e| LoaderError::Parse(format!("failed to read header: {e}")))?;

        let header: VectorFileHeader = serde_json::from_str(&header_line)
            .map_err(|e| LoaderError::Parse(format!("invalid header: {e}")))?;

        Ok(Self {
            reader,
            header,
            line_buf: String::new(),
        })
    }

    pub fn header(&self) -> &VectorFileHeader {
        &self.header
    }

    /// Read the next vector record. Returns None at EOF.
    pub fn next_record(&mut self) -> Result<Option<VectorRecord>, LoaderError> {
        self.line_buf.clear();
        let bytes = self.reader.read_line(&mut self.line_buf)?;
        if bytes == 0 {
            return Ok(None);
        }
        let trimmed = self.line_buf.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        let record: VectorRecord = serde_json::from_str(trimmed)
            .map_err(|e| LoaderError::Parse(format!("invalid record: {e}")))?;
        Ok(Some(record))
    }

    /// Collect all records, optionally filtered by layer.
    pub fn read_all(
        &mut self,
        layer_filter: Option<&HashSet<usize>>,
    ) -> Result<Vec<VectorRecord>, LoaderError> {
        let mut records = Vec::new();
        while let Some(record) = self.next_record()? {
            if let Some(layers) = layer_filter {
                if !layers.contains(&record.layer) {
                    continue;
                }
            }
            records.push(record);
        }
        Ok(records)
    }
}

/// Discover .vectors.jsonl files in a directory.
pub fn discover_vector_files(dir: &Path) -> Result<Vec<(String, PathBuf)>, LoaderError> {
    if !dir.is_dir() {
        return Err(LoaderError::NoVectorFiles(dir.to_path_buf()));
    }

    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.ends_with(".vectors.jsonl") {
                let component = name.trim_end_matches(".vectors.jsonl").to_string();
                if ALL_COMPONENTS.contains(&component.as_str()) {
                    files.push((component, path));
                }
            }
        }
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

// ═══════════════════════════════════════════════════
// Schema DDL Generation
// ═══════════════════════════════════════════════════

// SurQL templates embedded from surql/ directory at compile time.
const SETUP_TEMPLATE: &str = include_str!("../../../../surql/setup.surql");
const SCHEMA_TEMPLATE: &str = include_str!("../../../../surql/component_schema.surql");
const PROGRESS_TEMPLATE: &str = include_str!("../../../../surql/load_progress.surql");

/// Generate the full schema DDL for a namespace + database setup.
pub fn setup_sql(ns: &str, db: &str) -> String {
    SETUP_TEMPLATE
        .replace("{ns}", ns)
        .replace("{db}", db)
}

/// Generate schema DDL for a single component table.
pub fn schema_sql(component: &str, dimension: usize) -> Result<String, LoaderError> {
    if !ALL_COMPONENTS.contains(&component) {
        return Err(LoaderError::UnknownComponent(component.to_string()));
    }

    Ok(SCHEMA_TEMPLATE
        .replace("{table}", component)
        .replace("{dimension}", &dimension.to_string()))
}

/// Generate load_progress table schema.
pub fn progress_table_sql() -> String {
    PROGRESS_TEMPLATE.to_string()
}

// ═══════════════════════════════════════════════════
// Batch INSERT SQL Generation
// ═══════════════════════════════════════════════════

/// Generate a batch INSERT transaction for a slice of vector records.
///
/// Each record becomes a `CREATE table:id CONTENT {...}` statement
/// wrapped in a transaction.
pub fn batch_insert_sql(table: &str, records: &[VectorRecord]) -> String {
    let mut sql = String::from("BEGIN TRANSACTION;\n");

    for record in records {
        // Build CONTENT JSON with only the fields SurrealDB needs
        // (skip the 'id' and 'dim' fields — id is in the record key)
        let content = serde_json::json!({
            "layer": record.layer,
            "feature": record.feature,
            "vector": record.vector,
            "top_token": record.top_token,
            "top_token_id": record.top_token_id,
            "c_score": record.c_score,
            "top_k": record.top_k,
        });

        sql.push_str(&format!(
            "CREATE {table}:{id} CONTENT {json};\n",
            id = record.id,
            json = content,
        ));
    }

    sql.push_str("COMMIT TRANSACTION;");
    sql
}

/// Generate a single-record INSERT SQL. One record per request avoids
/// HTTP body size limits for large vectors (2560-dim = ~30KB per record).
pub fn single_insert_sql(table: &str, record: &VectorRecord) -> String {
    let content = serde_json::json!({
        "layer": record.layer,
        "feature": record.feature,
        "vector": record.vector,
        "top_token": record.top_token,
        "top_token_id": record.top_token_id,
        "c_score": record.c_score,
        "top_k": record.top_k,
    });

    format!(
        "CREATE {table}:{id} CONTENT {json};",
        id = record.id,
        json = content,
    )
}

/// Generate SQL to mark a layer as completed in load_progress.
pub fn mark_layer_done_sql(table: &str, layer: usize, count: usize) -> String {
    format!(
        "CREATE load_progress:{table}_L{layer} CONTENT {{\
            \"table_name\": \"{table}\",\
            \"layer\": {layer},\
            \"vectors_loaded\": {count},\
            \"completed\": true,\
            \"timestamp\": time::now()\
        }};"
    )
}

/// Generate SQL to query completed layers for a table.
pub fn completed_layers_sql(table: &str) -> String {
    format!(
        "SELECT layer FROM load_progress WHERE table_name = '{table}' AND completed = true;"
    )
}

/// Count records in a table.
pub fn count_sql(table: &str) -> String {
    format!("SELECT count() FROM {table} GROUP ALL;")
}
