use larql_core::loader::vector_loader::*;
use larql_core::walker::vector_extractor::{TopKEntry, VectorRecord, ALL_COMPONENTS};

#[test]
fn test_setup_sql() {
    let sql = setup_sql("test_ns", "test_db");
    assert!(sql.contains("DEFINE NAMESPACE IF NOT EXISTS test_ns"));
    assert!(sql.contains("USE NS test_ns"));
    assert!(sql.contains("DEFINE DATABASE IF NOT EXISTS test_db"));
    assert!(sql.contains("USE DB test_db"));
}

#[test]
fn test_schema_sql_valid_component() {
    let sql = schema_sql("ffn_down", 2560).unwrap();
    assert!(sql.contains("DEFINE TABLE IF NOT EXISTS ffn_down SCHEMAFULL"));
    assert!(sql.contains("DEFINE TABLE IF NOT EXISTS ffn_down"));
    assert!(sql.contains("layer"));
    assert!(sql.contains("vector"));
    assert!(sql.contains("top_token"));
    assert!(sql.contains("c_score"));
    assert!(sql.contains("HNSW DIMENSION 2560 DIST COSINE"));
    assert!(sql.contains("top_k[*].token"));
    assert!(sql.contains("top_k[*].logit"));
}

#[test]
fn test_schema_sql_all_components() {
    for component in ALL_COMPONENTS {
        let result = schema_sql(component, 2560);
        assert!(result.is_ok(), "schema_sql failed for {component}");
        let sql = result.unwrap();
        assert!(sql.contains(component));
        assert!(sql.contains("HNSW DIMENSION 2560"));
    }
}

#[test]
fn test_schema_sql_unknown_component() {
    let result = schema_sql("nonexistent", 2560);
    assert!(result.is_err());
}

#[test]
fn test_progress_table_sql() {
    let sql = progress_table_sql();
    assert!(sql.contains("DEFINE TABLE IF NOT EXISTS load_progress"));
    assert!(sql.contains("table_name"));
    assert!(sql.contains("layer"));
    assert!(sql.contains("completed"));
    assert!(sql.contains("TYPE datetime"));
}

#[test]
fn test_single_insert_sql() {
    let record = VectorRecord {
        id: "L26_F42".to_string(),
        layer: 26,
        feature: 42,
        vector: vec![1.0, 2.0, 3.0],
        dim: 3,
        top_token: "Paris".to_string(),
        top_token_id: 4196,
        c_score: 12.4,
        top_k: vec![TopKEntry {
            token: "Paris".to_string(),
            token_id: 4196,
            logit: 12.4,
        }],
    };

    let sql = single_insert_sql("ffn_down", &record);
    assert!(sql.contains("CREATE ffn_down:L26_F42 CONTENT"));
    assert!(sql.contains("\"layer\":26"));
    assert!(sql.contains("\"feature\":42"));
    assert!(sql.contains("\"top_token\":\"Paris\""));
    assert!(sql.contains("[1.0,2.0,3.0]"));
}

#[test]
fn test_batch_insert_sql() {
    let records = vec![
        VectorRecord {
            id: "L0_F0".to_string(),
            layer: 0,
            feature: 0,
            vector: vec![1.0],
            dim: 1,
            top_token: "the".to_string(),
            top_token_id: 1,
            c_score: 0.5,
            top_k: vec![],
        },
        VectorRecord {
            id: "L0_F1".to_string(),
            layer: 0,
            feature: 1,
            vector: vec![2.0],
            dim: 1,
            top_token: "a".to_string(),
            top_token_id: 2,
            c_score: 0.3,
            top_k: vec![],
        },
    ];

    let sql = batch_insert_sql("ffn_gate", &records);
    assert!(sql.contains("BEGIN TRANSACTION"));
    assert!(sql.contains("CREATE ffn_gate:L0_F0"));
    assert!(sql.contains("CREATE ffn_gate:L0_F1"));
    assert!(sql.contains("COMMIT TRANSACTION"));
}

#[test]
fn test_mark_layer_done_sql() {
    let sql = mark_layer_done_sql("ffn_down", 26, 10240);
    assert!(sql.contains("CREATE load_progress:ffn_down_L26"));
    assert!(sql.contains("\"table_name\": \"ffn_down\""));
    assert!(sql.contains("\"layer\": 26"));
    assert!(sql.contains("\"vectors_loaded\": 10240"));
    assert!(sql.contains("\"completed\": true"));
    assert!(sql.contains("time::now()"));
}

#[test]
fn test_completed_layers_sql() {
    let sql = completed_layers_sql("ffn_gate");
    assert!(sql.contains("SELECT layer FROM load_progress"));
    assert!(sql.contains("table_name = 'ffn_gate'"));
    assert!(sql.contains("completed = true"));
}

#[test]
fn test_count_sql() {
    let sql = count_sql("embeddings");
    assert!(sql.contains("SELECT count() FROM embeddings"));
}
