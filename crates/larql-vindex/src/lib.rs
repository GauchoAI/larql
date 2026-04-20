//! Vindex — load-time KNN store, graph index, and patches for the GGUF
//! hot path. Extract / clustering / vindexfile build paths were removed;
//! this crate is now read-only for the server's purposes (the vindex is
//! built externally and served as a directory).

pub mod config;
pub mod describe;
pub mod error;
pub mod format;
pub mod index;
pub mod patch;
pub mod mmap_util;

pub use ndarray;
pub use tokenizers;

// Config
pub use config::dtype::StorageDtype;
pub use config::types::{
    DownMetaRecord, DownMetaTopK, ExtractLevel, LayerBands, MoeConfig,
    VindexConfig, VindexLayerInfo, VindexModelConfig, VindexSource,
};

// Error
pub use error::VindexError;

// Index
pub use index::core::{
    FeatureMeta, GateIndex, IndexLoadCallbacks, SilentLoadCallbacks, VectorIndex, WalkHit, WalkTrace,
};

// Describe
pub use describe::{DescribeEdge, LabelSource};

// Format (read-only vindex loading)
pub use format::down_meta;
pub use format::load::{
    load_feature_labels, load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
};
pub use format::huggingface::{
    resolve_hf_vindex, is_hf_path,
};

// Patch
pub use patch::core::{PatchOp, PatchedVindex, VindexPatch};
pub use patch::knn_store::{KnnStore, KnnEntry};
