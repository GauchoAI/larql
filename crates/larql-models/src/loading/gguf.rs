//! GGUF format reader — parse GGUF files and load tensors as f32.
//!
//! GGUF is the GGML Universal Format used by llama.cpp.
//! We support reading unquantized (F32, F16, BF16) and quantized (Q4_0, Q4_1, Q8_0) tensors.
//! All tensors are dequantized to f32 for use with ModelWeights.

use std::collections::HashMap;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use ndarray::Array2;

use crate::weights::ModelWeights;
use crate::detect::ModelError;

// ═══════════════════════════════════════════════════════════════
// GGUF constants
// ═══════════════════════════════════════════════════════════════

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian

// Metadata value types
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

// Tensor type constants moved to format::quant::ggml

// ═══════════════════════════════════════════════════════════════
// GGUF metadata value
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::F32(v) => Some(*v as f64),
            GgufValue::F64(v) => Some(*v),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// GGUF tensor info
// ═══════════════════════════════════════════════════════════════

pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,
}

// ═══════════════════════════════════════════════════════════════
// GGUF reader
// ═══════════════════════════════════════════════════════════════

pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensor_infos: Vec<GgufTensorInfo>,
    pub data_offset: u64,
    pub path: std::path::PathBuf,
}

impl GgufFile {
    /// Parse a GGUF file header and tensor info (does not read tensor data yet).
    pub fn open(path: &Path) -> Result<Self, ModelError> {
        let file = std::fs::File::open(path)?;
        let mut r = BufReader::new(file);

        // Magic
        let magic = read_u32(&mut r)?;
        if magic != GGUF_MAGIC {
            return Err(ModelError::Parse(format!(
                "not a GGUF file (magic: 0x{:08X}, expected 0x{:08X})", magic, GGUF_MAGIC
            )));
        }

        // Version
        let version = read_u32(&mut r)?;
        if !(2..=3).contains(&version) {
            return Err(ModelError::Parse(format!("unsupported GGUF version: {version}")));
        }

        let n_tensors = read_u64(&mut r)? as usize;
        let n_metadata = read_u64(&mut r)? as usize;

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_metadata {
            let key = read_string(&mut r)?;
            let value = read_value(&mut r)?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensor_infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(&mut r)?;
            let n_dims = read_u32(&mut r)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut r)?);
            }
            let tensor_type = read_u32(&mut r)?;
            let offset = read_u64(&mut r)?;
            tensor_infos.push(GgufTensorInfo { name, n_dims, dims, tensor_type, offset });
        }

        // Data starts at next alignment boundary (32 bytes)
        let pos = r.stream_position()
            .map_err(ModelError::Io)?;
        let alignment = 32u64;
        let data_offset = pos.div_ceil(alignment) * alignment;

        Ok(GgufFile {
            metadata,
            tensor_infos,
            data_offset,
            path: path.to_path_buf(),
        })
    }

    /// Find a tensor by its GGUF name (e.g., "blk.0.attn_q.weight").
    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_infos.iter().find(|t| t.name == name)
    }

    /// Load all tensors, dequantizing to f32.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors(&self) -> Result<(HashMap<String, crate::WeightArray>, HashMap<String, Vec<f32>>), ModelError> {
        let file = std::fs::File::open(&self.path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();

        for info in &self.tensor_infos {
            // Skip vision encoder tensors (v.blk.*, mm.*)
            if info.name.starts_with("v.") || info.name.starts_with("mm.") {
                continue;
            }

            let abs_offset = self.data_offset + info.offset;
            let n_elements: u64 = info.dims.iter().product();

            let data_size = tensor_data_size(info.tensor_type, n_elements as usize)?;
            if abs_offset as usize + data_size > mmap.len() {
                return Err(ModelError::Parse(format!(
                    "tensor {} data out of bounds (offset {} + size {} > file {})",
                    info.name, abs_offset, data_size, mmap.len()
                )));
            }

            let raw = &mmap[abs_offset as usize..abs_offset as usize + data_size];
            let floats = dequantize(raw, info.tensor_type, n_elements as usize)?;

            // Normalize key name (strip GGUF prefixes)
            let key = normalize_gguf_key(&info.name);

            match info.n_dims {
                2 => {
                    // GGUF stores in row-major, dims[0] = rows, dims[1] = cols
                    let rows = info.dims[0] as usize;
                    let cols = info.dims[1] as usize;
                    let arr = Array2::from_shape_vec((rows, cols), floats)
                        .map_err(|e| ModelError::Parse(format!("tensor {}: {}", info.name, e)))?;
                    tensors.insert(key, arr.into_shared());
                }
                3 => {
                    // 3D tensor: expert stacks [K, N, num_experts] → flatten to [num_experts * N, K]
                    let k = info.dims[0] as usize;
                    let n = info.dims[1] as usize;
                    let num_experts = info.dims[2] as usize;
                    let arr = Array2::from_shape_vec((num_experts * n, k), floats)
                        .map_err(|e| ModelError::Parse(format!("3D tensor {}: {}", info.name, e)))?;
                    tensors.insert(key, arr.into_shared());
                }
                1 => {
                    vectors.insert(key, floats);
                }
                _ => {
                    // Skip 4D+ tensors (vision patch embeddings, etc.)
                }
            }
        }

        Ok((tensors, vectors))
    }

    /// Build a config.json-equivalent from GGUF metadata for architecture detection.
    pub fn to_config_json(&self) -> serde_json::Value {
        let get_str = |k: &str| self.metadata.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let _get_u32 = |k: &str| self.metadata.get(k).and_then(|v| v.as_u32()).unwrap_or(0);

        // GGUF uses "general.architecture" and "{arch}.*" keys
        let arch = get_str("general.architecture");
        let prefix = format!("{arch}.");

        let get_arch_u32 = |suffix: &str| -> u32 {
            self.metadata.get(&format!("{prefix}{suffix}"))
                .and_then(|v| v.as_u32())
                .unwrap_or(0)
        };
        let _get_arch_f64 = |suffix: &str| -> f64 {
            self.metadata.get(&format!("{prefix}{suffix}"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        };
        let get_arch_opt_u32 = |suffix: &str| -> Option<u32> {
            self.metadata.get(&format!("{prefix}{suffix}"))
                .and_then(|v| v.as_u32())
        };
        let get_arch_opt_f64 = |suffix: &str| -> Option<f64> {
            self.metadata.get(&format!("{prefix}{suffix}"))
                .and_then(|v| v.as_f64())
        };
        let get_arch_array_u32 = |suffix: &str| -> Option<Vec<u32>> {
            self.metadata.get(&format!("{prefix}{suffix}"))
                .and_then(|v| match v {
                    GgufValue::Array(arr) => {
                        Some(arr.iter().filter_map(|x| x.as_u32()).collect::<Vec<_>>())
                    }
                    _ => None,
                })
        };

        // Map GGUF architecture names to HF model_type
        let model_type = match arch.as_str() {
            "llama" => "llama",
            "gemma" | "gemma2" | "gemma3" | "gemma4" => &arch,
            "qwen" | "qwen2" => "qwen2",
            "mistral" => "mistral",
            "mixtral" => "mixtral",
            "phi" | "phi2" | "phi3" => "phi",
            "gpt2" => "gpt2",
            "deepseek" | "deepseek2" => "deepseek_v2",
            other => other,
        };

        // Handle per-layer arrays for head_count_kv and sliding_window_pattern
        let kv_heads_array = get_arch_array_u32("attention.head_count_kv");
        let num_kv_heads = kv_heads_array.as_ref()
            .and_then(|arr| arr.first().copied())
            .unwrap_or_else(|| get_arch_u32("attention.head_count_kv"));

        let sliding_pattern = get_arch_array_u32("attention.sliding_window_pattern");

        // For Gemma 4: use sliding head_dim as default, keep global head_dim separate
        let head_dim = if arch == "gemma4" {
            get_arch_opt_u32("attention.key_length_swa")
                .unwrap_or_else(|| get_arch_u32("attention.key_length"))
        } else {
            get_arch_u32("attention.key_length")
        };

        let mut config = serde_json::json!({
            "model_type": model_type,
            "hidden_size": get_arch_u32("embedding_length"),
            "num_hidden_layers": get_arch_u32("block_count"),
            "intermediate_size": get_arch_u32("feed_forward_length"),
        });
        // Only include optional fields if present in metadata.
        // Missing values let parse_model_config use arch-specific defaults
        // (e.g. Gemma3 defaults to rope_theta=1,000,000, head_dim=256).
        if let Some(v) = get_arch_opt_u32("attention.head_count") {
            config["num_attention_heads"] = serde_json::json!(v);
        }
        if num_kv_heads > 0 {
            config["num_key_value_heads"] = serde_json::json!(num_kv_heads);
        }
        if head_dim > 0 {
            config["head_dim"] = serde_json::json!(head_dim);
        }
        if let Some(rope) = get_arch_opt_f64("rope.freq_base") {
            config["rope_theta"] = serde_json::json!(rope);
        }
        if let Some(vs) = get_arch_opt_u32("vocab_size") {
            config["vocab_size"] = serde_json::json!(vs);
        }

        // Add MoE fields if present
        if let Some(expert_count) = get_arch_opt_u32("expert_count") {
            config["num_local_experts"] = serde_json::json!(expert_count);
        }
        if let Some(experts_per_tok) = get_arch_opt_u32("expert_used_count") {
            config["num_experts_per_token"] = serde_json::json!(experts_per_tok);
        }
        if let Some(softcap) = get_arch_opt_f64("final_logit_softcapping") {
            config["final_logit_softcapping"] = serde_json::json!(softcap);
        }

        // Gemma 4: global head_dim, sliding window, RoPE bases, attention_k_eq_v
        if arch == "gemma4" {
            if let Some(global_hd) = get_arch_opt_u32("attention.key_length") {
                config["global_head_dim"] = serde_json::json!(global_hd);
            }
            if let Some(sw) = get_arch_opt_u32("attention.sliding_window") {
                config["sliding_window"] = serde_json::json!(sw);
            }
            if let Some(swa_base) = get_arch_opt_f64("rope.freq_base_swa") {
                config["rope_local_base_freq"] = serde_json::json!(swa_base);
            }
            config["attention_k_eq_v"] = serde_json::json!(true);
        }

        // Convert sliding_window_pattern array to layer_types
        if let Some(ref pattern) = sliding_pattern {
            let types: Vec<String> = pattern.iter().map(|&v|
                if v == 1 { "full_attention".to_string() } else { "sliding_attention".to_string() }
            ).collect();
            config["layer_types"] = serde_json::json!(types);
        }

        // Find global KV head count from per-layer arrays
        if let (Some(ref kv_arr), Some(ref pat)) = (&kv_heads_array, &sliding_pattern) {
            if let Some(global_idx) = pat.iter().position(|&v| v == 1) {
                if let Some(&global_kv) = kv_arr.get(global_idx) {
                    config["num_global_key_value_heads"] = serde_json::json!(global_kv);
                }
            }
        }

        config
    }

    /// Load only small tensors (F32/F16 vectors and small matrices).
    /// Skips large quantized 2D/3D weight matrices to avoid dequantizing 17+ GB.
    /// Returns norms as vectors, small f32 matrices as tensors.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors_filtered(&self, max_elements: usize) -> Result<(HashMap<String, crate::WeightArray>, HashMap<String, Vec<f32>>), ModelError> {
        let file = std::fs::File::open(&self.path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();

        for info in &self.tensor_infos {
            // Skip vision tensors
            if info.name.starts_with("v.") || info.name.starts_with("mm.") {
                continue;
            }

            let n_elements: u64 = info.dims.iter().product();

            // Skip tensors larger than max_elements
            if n_elements as usize > max_elements {
                continue;
            }

            let abs_offset = self.data_offset + info.offset;
            let data_size = tensor_data_size(info.tensor_type, n_elements as usize)?;
            if abs_offset as usize + data_size > mmap.len() {
                continue;
            }

            let raw = &mmap[abs_offset as usize..abs_offset as usize + data_size];
            let floats = dequantize(raw, info.tensor_type, n_elements as usize)?;

            let key = normalize_gguf_key(&info.name);

            match info.n_dims {
                2 => {
                    let rows = info.dims[0] as usize;
                    let cols = info.dims[1] as usize;
                    let arr = ndarray::Array2::from_shape_vec((rows, cols), floats)
                        .map_err(|e| ModelError::Parse(format!("tensor {}: {}", info.name, e)))?;
                    tensors.insert(key, arr.into_shared());
                }
                1 => {
                    vectors.insert(key, floats);
                }
                _ => {}
            }
        }

        Ok((tensors, vectors))
    }
}

/// Holds an mmap'd GGUF file for zero-copy access to quantized tensor data.
/// The mmap stays alive as long as this struct exists.
pub struct GgufQuantizedData {
    mmap: memmap2::Mmap,
    data_offset: u64,
}

impl GgufQuantizedData {
    /// Open and mmap a GGUF file for raw quantized access.
    pub fn open(path: &std::path::Path, data_offset: u64) -> Result<Self, ModelError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { mmap, data_offset })
    }

    /// Get raw bytes for a tensor (no dequantization).
    pub fn tensor_bytes(&self, info: &GgufTensorInfo) -> &[u8] {
        let abs_offset = self.data_offset + info.offset;
        let n_elements: u64 = info.dims.iter().product();
        let data_size = tensor_data_size(info.tensor_type, n_elements as usize)
            .expect("unsupported tensor type");
        let end = abs_offset as usize + data_size;
        if end > self.mmap.len() {
            panic!(
                "tensor {} out of bounds: offset={} + size={} = {} > file_len={} (type={}, dims={:?})",
                info.name, abs_offset, data_size, end, self.mmap.len(),
                info.tensor_type, info.dims,
            );
        }
        &self.mmap[abs_offset as usize..end]
    }

    /// Get raw bytes for a tensor, reinterpreted as f32 slice.
    /// Only valid for F32 tensors (tensor_type == 0).
    pub fn tensor_f32(&self, info: &GgufTensorInfo) -> Option<&[f32]> {
        if info.tensor_type != 0 { return None; }  // GGML_TYPE_F32 = 0
        let bytes = self.tensor_bytes(info);
        let n = bytes.len() / 4;
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n) })
    }
}

/// Load a GGUF file into ModelWeights (dequantized to f32).
pub fn load_gguf(path: &Path) -> Result<ModelWeights, ModelError> {
    let gguf = GgufFile::open(path)?;

    // Detect architecture from GGUF metadata
    let config_json = gguf.to_config_json();
    let arch = crate::detect_from_json(&config_json);
    let prefixes = arch.key_prefixes_to_strip();

    // Load and dequantize all tensors
    let (mut tensors, mut vectors) = gguf.load_tensors()?;

    // Re-normalize keys through the architecture's prefix stripping
    let mut normalized_tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    for (k, v) in tensors.drain() {
        let key = super::safetensors::normalize_key_pub(&k, prefixes);
        normalized_tensors.insert(key, v);
    }

    // GGUF norm compatibility: `ffn_norm` normalizes to `pre_feedforward_layernorm`.
    // For 2-norm models (Llama, etc.) the f32 inference path looks up
    // `post_attention_layernorm`. Duplicate the vector under both keys so
    // both lookup patterns find it.
    if !arch.has_post_norms() {
        let num_layers = arch.config().num_layers;
        for layer in 0..num_layers {
            let pre_ffn_key = format!("layers.{layer}.pre_feedforward_layernorm.weight");
            let post_attn_key = format!("layers.{layer}.post_attention_layernorm.weight");
            if !vectors.contains_key(&post_attn_key) {
                if let Some(v) = vectors.get(&pre_ffn_key) {
                    let cloned = v.clone();
                    vectors.insert(post_attn_key, cloned);
                }
            }
        }
    }

    let embed_key = arch.embed_key();
    let embed = normalized_tensors
        .get(embed_key)
        .ok_or_else(|| ModelError::MissingTensor(embed_key.into()))?
        .clone();

    let lm_head = normalized_tensors
        .get("lm_head.weight")
        .or_else(|| normalized_tensors.get("output.weight"))
        .cloned()
        .unwrap_or_else(|| embed.clone());

    let vocab_size = lm_head.shape()[0];
    let cfg = arch.config();

    Ok(ModelWeights {
        tensors: normalized_tensors,
        vectors,
        embed,
        lm_head,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

// ═══════════════════════════════════════════════════════════════
// GGUF binary reading helpers
// ═══════════════════════════════════════════════════════════════

fn read_u8(r: &mut impl Read) -> Result<u8, ModelError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(r: &mut impl Read) -> Result<i8, ModelError> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut impl Read) -> Result<u16, ModelError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(r: &mut impl Read) -> Result<i16, ModelError> {
    Ok(read_u16(r)? as i16)
}

fn read_u32(r: &mut impl Read) -> Result<u32, ModelError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> Result<i32, ModelError> {
    Ok(read_u32(r)? as i32)
}

fn read_u64(r: &mut impl Read) -> Result<u64, ModelError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> Result<i64, ModelError> {
    Ok(read_u64(r)? as i64)
}

fn read_f32(r: &mut impl Read) -> Result<f32, ModelError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> Result<f64, ModelError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string(r: &mut impl Read) -> Result<String, ModelError> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| ModelError::Parse(e.to_string()))
}

fn read_value(r: &mut impl Read) -> Result<GgufValue, ModelError> {
    let vtype = read_u32(r)?;
    match vtype {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(r)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(read_i64(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(r)?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            let len = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_array_element(r, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        _ => Err(ModelError::Parse(format!("unknown GGUF metadata type: {vtype}"))),
    }
}

fn read_array_element(r: &mut impl Read, elem_type: u32) -> Result<GgufValue, ModelError> {
    match elem_type {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(r)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(read_i64(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(r)?)),
        _ => Err(ModelError::Parse(format!("unknown GGUF array element type: {elem_type}"))),
    }
}

// ═══════════════════════════════════════════════════════════════
// Dequantization — delegates to format::quant module
// ═══════════════════════════════════════════════════════════════

fn tensor_data_size(tensor_type: u32, n_elements: usize) -> Result<usize, ModelError> {
    crate::quant::ggml::tensor_data_size(tensor_type, n_elements)
}

fn dequantize(data: &[u8], tensor_type: u32, n_elements: usize) -> Result<Vec<f32>, ModelError> {
    crate::quant::ggml::dequantize(data, tensor_type, n_elements)
}

/// Normalize GGUF tensor key names to match HuggingFace conventions.
pub fn normalize_gguf_key(name: &str) -> String {
    // GGUF uses "blk.N.attn_q.weight" format
    // HF uses "model.layers.N.self_attn.q_proj.weight" format
    // We normalize to the HF style since that's what ModelArchitecture expects

    

    name
        .replace("blk.", "layers.")
        // 4-norm model keys (Gemma 3/4): MUST come before shorter patterns.
        // GGUF uses both conventions: "attn_post_norm"/"ffn_post_norm" (llama.cpp)
        // and "post_attention_norm"/"post_ffw_norm" (Gemma 4 GGUF).
        .replace("attn_post_norm.", "post_attention_layernorm.")
        .replace("post_attention_norm.", "post_attention_layernorm.")
        .replace("ffn_post_norm.", "post_feedforward_layernorm.")
        .replace("post_ffw_norm.", "post_feedforward_layernorm.")
        // Gemma 4 has extra MoE norms: post_ffw_norm_1, post_ffw_norm_2, pre_ffw_norm_2
        .replace("post_ffw_norm_1.", "post_feedforward_layernorm_1.")
        .replace("post_ffw_norm_2.", "post_feedforward_layernorm_2.")
        .replace("pre_ffw_norm_2.", "pre_feedforward_layernorm_2.")
        // QK norm keys (Gemma 3/4): MUST come before generic "attn_q."/"attn_k."
        .replace("attn_q_norm.", "self_attn.q_norm.")
        .replace("attn_k_norm.", "self_attn.k_norm.")
        // Per-layer scalar (Gemma 4)
        .replace("layer_output_scale.", "layer_scalar.")
        .replace("layer_scale", "layer_scalar")
        .replace("attn_q.", "self_attn.q_proj.")
        .replace("attn_k.", "self_attn.k_proj.")
        .replace("attn_v.", "self_attn.v_proj.")
        .replace("attn_output.", "self_attn.o_proj.")
        // Expert patterns MUST come before dense FFN patterns
        // to prevent "ffn_gate_up_exps" from matching "ffn_gate."
        .replace("ffn_gate_up_exps.", "mlp.experts.gate_up_proj.")
        .replace("ffn_gate_inp.", "mlp.router.")
        .replace("ffn_down_exps.", "mlp.experts.down_proj.")
        .replace("ffn_gate.", "mlp.gate_proj.")
        .replace("ffn_up.", "mlp.up_proj.")
        .replace("ffn_down.", "mlp.down_proj.")
        .replace("attn_norm.", "input_layernorm.")
        // ffn_norm = pre-FFN norm in ALL models. For 2-norm models,
        // HF calls this "post_attention_layernorm" (legacy naming).
        // For 4-norm models (Gemma 3/4), HF calls it "pre_feedforward_layernorm".
        // We normalize to "pre_feedforward_layernorm" — build_arch_params falls
        // back to this key when "post_attention_layernorm" is missing.
        .replace("ffn_norm.", "pre_feedforward_layernorm.")
        .replace("token_embd.", "embed_tokens.")
        .replace("output_norm.", "norm.")
        .replace("output.", "lm_head.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_gguf_key() {
        assert_eq!(
            normalize_gguf_key("blk.0.attn_q.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.15.ffn_gate.weight"),
            "layers.15.mlp.gate_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("token_embd.weight"),
            "embed_tokens.weight"
        );
        assert_eq!(
            normalize_gguf_key("output.weight"),
            "lm_head.weight"
        );
    }

    #[test]
    fn test_normalize_gguf_key_experts() {
        assert_eq!(
            normalize_gguf_key("blk.0.ffn_gate_up_exps.weight"),
            "layers.0.mlp.experts.gate_up_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.5.ffn_gate_inp.weight"),
            "layers.5.mlp.router.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.3.ffn_down_exps.weight"),
            "layers.3.mlp.experts.down_proj.weight"
        );
        // Ensure dense FFN patterns still work alongside expert patterns
        assert_eq!(
            normalize_gguf_key("blk.1.ffn_gate.weight"),
            "layers.1.mlp.gate_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.2.ffn_down.weight"),
            "layers.2.mlp.down_proj.weight"
        );
    }

    #[test]
    fn test_normalize_gguf_key_gemma_norms() {
        // 4-norm model keys (Gemma 3/4) — llama.cpp convention
        assert_eq!(
            normalize_gguf_key("blk.0.attn_post_norm.weight"),
            "layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.5.ffn_post_norm.weight"),
            "layers.5.post_feedforward_layernorm.weight"
        );
        // Gemma 4 GGUF convention
        assert_eq!(
            normalize_gguf_key("blk.0.post_attention_norm.weight"),
            "layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.0.post_ffw_norm.weight"),
            "layers.0.post_feedforward_layernorm.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.0.ffn_norm.weight"),
            "layers.0.pre_feedforward_layernorm.weight"
        );
        // QK norms
        assert_eq!(
            normalize_gguf_key("blk.3.attn_q_norm.weight"),
            "layers.3.self_attn.q_norm.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.3.attn_k_norm.weight"),
            "layers.3.self_attn.k_norm.weight"
        );
        // Layer scalar (Gemma 4)
        assert_eq!(
            normalize_gguf_key("blk.7.layer_output_scale.weight"),
            "layers.7.layer_scalar.weight"
        );
    }

    // Dequant tests are in format::quant::ggml::tests
}
