//! Interactive inference harness — loads model/vindex/backend ONCE,
//! reads commands from stdin. Zero reload per experiment.
//!
//! Commands:
//!   prompt "text"           — run dense predict_honest on prompt, show top-5 + time
//!   gen "text" N            — dense multi-token gen via predict_honest, growing prompt
//!   kvreset                 — clear KV cache
//!   kvprefill "text"        — prefill the prompt, populate KV, report time
//!   kvdecode N              — decode N single-token steps using current KV
//!   help / quit
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference --example bench_interactive -- \
//!     --model /path/to/gemma-3-4b-it --vindex /path/to/gemma3-4b.vindex

use std::io::{BufRead, Write};
use std::time::Instant;

use larql_inference::{InferenceModel, CachedLayerGraph, default_backend, predict_with_ffn};
use larql_inference::ffn::WeightFfn;
use larql_inference::vindex::WalkFfn;
use larql_inference::sampling::{Sampler, SamplingConfig};
use larql_vindex::{SilentLoadCallbacks, VectorIndex, KnnStore, VindexPatch, PatchOp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = std::path::PathBuf::new();
    let mut model_ref = String::from("google/gemma-3-4b-it");
    let mut patch_paths: Vec<std::path::PathBuf> = Vec::new();
    let mut walk_only = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => { i += 1; vindex_path = std::path::PathBuf::from(&args[i]); }
            "--model" => { i += 1; model_ref = args[i].clone(); }
            "--patch" => { i += 1; patch_paths.push(std::path::PathBuf::from(&args[i])); }
            "--walk-only" => { walk_only = true; }
            _ => {}
        }
        i += 1;
    }
    if !vindex_path.is_dir() {
        eprintln!("Usage: bench_interactive --model MODEL --vindex PATH [--patch FILE.vlp ...] [--walk-only]");
        std::process::exit(1);
    }

    eprintln!("[load] model {model_ref}{}", if walk_only { " (walk-only, FFN dropped)" } else { "" });
    let t0 = Instant::now();
    let model = if walk_only {
        InferenceModel::load_walk_only(&model_ref)?
    } else {
        InferenceModel::load(&model_ref)?
    };
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;
    eprintln!("[load] model: {:.1}s", t0.elapsed().as_secs_f64());

    eprintln!("[load] vindex {}", vindex_path.display());
    let t0 = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    let _ = index.load_lm_head(&vindex_path);
    let _ = index.load_lm_head_q4(&vindex_path);
    let _ = index.load_attn_q4k(&vindex_path);
    let _ = index.load_attn_q8(&vindex_path);
    // LARQL_WALK_FORMAT pins the walk path AND limits which mmap is loaded,
    // so RSS measurements are honest per-mode (otherwise we'd pay for all
    // three mmap'd files even when only one fires).
    let want_fmt = std::env::var("LARQL_WALK_FORMAT").ok();
    match want_fmt.as_deref() {
        Some("q4_0") => {
            match index.load_interleaved_q4(&vindex_path) {
                Ok(()) => eprintln!("[load] interleaved_q4.bin ok"),
                Err(e) => eprintln!("[load] interleaved_q4.bin failed: {e}"),
            }
        }
        Some("q4_k") => {
            match index.load_interleaved_q4k_real(&vindex_path) {
                Ok(()) => eprintln!("[load] interleaved_q4k_real.bin ok"),
                Err(e) => eprintln!("[load] interleaved_q4k_real.bin failed: {e}"),
            }
        }
        Some("f32") | Some("f32_sparse") => {
            match index.load_interleaved(&vindex_path) {
                Ok(()) => eprintln!("[load] interleaved.bin ok"),
                Err(e) => eprintln!("[load] interleaved.bin failed: {e}"),
            }
        }
        _ => {
            // Prefer Q4_K real (GPU decode compatible, validated cos=0.9994).
            // Only load other formats as fallback.
            if index.load_interleaved_q4k_real(&vindex_path).is_ok() {
                eprintln!("[load] interleaved_q4k_real.bin ok (GPU decode default)");
            } else {
                let _ = index.load_interleaved(&vindex_path);
                let _ = index.load_interleaved_q4(&vindex_path);
                let _ = index.load_interleaved_q4k(&vindex_path);
            }
        }
    }
    eprintln!("[load] vindex: {:.1}s", t0.elapsed().as_secs_f64());

    let backend = default_backend();
    let dense_ffn = WeightFfn { weights };
    let walk_ffn = WalkFfn::new_unlimited(weights, &index);
    let empty_cache_layers: Vec<usize> = Vec::new();

    // Load KNN overlay store from patch files (if any). Each .vlp may
    // contain `InsertKnn` ops with layer + base64-encoded residual keys.
    // These get consulted in the Gemma 3 Metal fast path after each
    // attention step; cosine > 0.75 overrides the model's prediction.
    let mut knn_store = KnnStore::default();
    for p in &patch_paths {
        match VindexPatch::load(p) {
            Ok(patch) => {
                let mut loaded = 0usize;
                for op in &patch.operations {
                    if let PatchOp::InsertKnn { layer, entity, relation, target, target_id, confidence, key_vector_b64 } = op {
                        if let Ok(key_vec) = larql_vindex::patch::core::decode_gate_vector(key_vector_b64) {
                            knn_store.add(
                                *layer, key_vec, *target_id,
                                target.clone(), entity.clone(), relation.clone(),
                                confidence.unwrap_or(1.0),
                            );
                            loaded += 1;
                        }
                    }
                }
                eprintln!("[patch] {}: {} KNN entries loaded (layers: {:?})",
                    p.display(), loaded, knn_store.layers());
            }
            Err(e) => eprintln!("[patch] failed to load {}: {e}", p.display()),
        }
    }
    // Warmup: Metal specializes pipeline state per seq_len (attention shapes
    // and FFN intermediate dims), and the first use of each specialization
    // triggers a shader compile. Without warmup, the first user question at a
    // given prompt length pays 10-20 s for compilation. Doing one dummy
    // forward per common seq_len up front moves that cost out of the
    // interactive loop. Skip with --no-warmup.
    let warmup_enabled = !args.iter().any(|a| a == "--no-warmup");
    if warmup_enabled {
        eprintln!("[warmup] compiling Metal shaders for seq_len ∈ {{1, 4, 8, 16, 32}} ...");
        let t0 = Instant::now();
        // Pick a token id that decodes cleanly — BOS (1) or similar. Use the
        // tokenizer's BOS so the forward is not garbage.
        let bos = tokenizer.encode("", true).ok()
            .and_then(|e| e.get_ids().first().copied())
            .unwrap_or(1);
        for &n in &[1usize, 4, 8, 16, 32] {
            let ids: Vec<u32> = std::iter::repeat(bos).take(n).collect();
            let cache = CachedLayerGraph::build(weights, &ids, &empty_cache_layers, &dense_ffn);
            backend.reset_kv_cache();
            let t = Instant::now();
            let _ = larql_inference::layer_graph::predict::predict_honest_with_knn(
                weights, tokenizer, &ids, 1, &index, &*backend, &cache, 0..num_layers, None,
            );
            eprintln!("[warmup]   seq_len={n} {:.1}s", t.elapsed().as_secs_f64());
        }
        backend.reset_kv_cache();
        eprintln!("[warmup] done in {:.1}s — subsequent queries will be fast", t0.elapsed().as_secs_f64());
    } else {
        eprintln!("[warmup] skipped (--no-warmup)");
    }

    // Build a sampler config from env vars — greedy by default, overrideable
    // per-session. LARQL_TEMP=0 gives greedy (current behaviour); >0 enables
    // probabilistic sampling with optional LARQL_TOP_P nucleus cutoff.
    let build_sampler = || -> Sampler {
        let temperature = std::env::var("LARQL_TEMP").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(0.0f32);
        let top_p = std::env::var("LARQL_TOP_P").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(1.0f32);
        let top_k = std::env::var("LARQL_TOP_K").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(0usize);
        let seed = std::env::var("LARQL_SEED").ok().and_then(|s| s.parse().ok());
        Sampler::new(SamplingConfig { temperature, top_p, top_k, seed })
    };
    // Gemma 3 stop tokens: 1 = <eos>, 106 = <end_of_turn>. Other models may
    // differ; override with LARQL_STOP_IDS="1,106,108".
    let stop_ids: Vec<u32> = std::env::var("LARQL_STOP_IDS").ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1u32, 106]);

    eprintln!("[ready] backend={} layers={num_layers}", backend.name());
    eprintln!("[ready] commands: ask, insert, save, prompt, gen, kvreset, kvprefill, kvdecode, gpuprefill, cpupredict, help, quit");
    eprintln!("[ready] sampling: set LARQL_TEMP / LARQL_TOP_P / LARQL_TOP_K to enable; default greedy. LARQL_STOP_IDS for EOS.");
    if !knn_store.is_empty() {
        eprintln!("[ready] KNN overlay ACTIVE ({} entries) — Metal fast path will consult it", knn_store.len());
    }
    // Helper: build an `Option<&KnnStore>` per call (so the store remains
    // mutably accessible to `insert` / `save` between commands).
    // Used below at each `predict_honest_with_knn` call site.

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    print!("> ");
    stdout.flush().ok();

    let mut last_prompt_tokens: Vec<u32> = Vec::new();
    let mut last_prediction: Option<u32> = None;

    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let line = line.trim().to_string();
        if line.is_empty() { print!("> "); stdout.flush().ok(); continue; }

        let (cmd, rest) = match line.split_once(' ') {
            Some((c, r)) => (c, r.trim()),
            None => (line.as_str(), ""),
        };

        match cmd {
            "quit" | "exit" => break,
            "help" => {
                println!("  ask \"text\" [N]    — fast prefill + decode N tokens (default 20), Metal + KNN overlay");
                println!("  spec \"text\" [N] [K] — speculative decode: n-gram draft K, verify, accept prefix");
                println!("  insert \"entity\" \"relation\" \"target\" [layer]");
                println!("                    — capture residual via Metal f32 forward, add to KNN overlay");
                println!("  save \"file.vlp\"   — persist the runtime KNN overlay as a .vlp patch");
                println!("  prompt \"text\"     — dense predict, top-5 + time");
                println!("  gen \"text\" N      — dense gen N tokens, growing prompt");
                println!("  kvreset           — clear backend KV cache");
                println!("  gpuprefill \"text\" — prefill prompt, show top-5 + time, populate KV");
                println!("  kvprefill \"text\"  — alias of gpuprefill");
                println!("  kvdecode N        — decode N single-token steps from current KV");
                println!("  cpupredict \"text\" — CPU-only ground-truth reference (slow)");
                println!("  quit");
            }
            "insert" => {
                // Parse: insert "Entity" "relation" "target" [layer]
                // Match LQL INSERT INTO EDGES (entity, relation, target) VALUES (...)
                // but run the capture through the Metal f32 path instead of the
                // 75-second CPU walk — ~1 s per fact.
                let parts = parse_quoted_list(rest);
                if parts.len() < 3 || parts.len() > 4 {
                    println!("  usage: insert \"entity\" \"relation\" \"target\" [layer]");
                    print!("> "); stdout.flush().ok(); continue;
                }
                let entity = parts[0].clone();
                let relation = parts[1].clone();
                let target = parts[2].clone();
                let install_layer: usize = parts.get(3)
                    .and_then(|s| s.parse().ok())
                    // Default matches LQL: knowledge.1 - 1 ≈ 26 for Gemma 3 4B.
                    .unwrap_or_else(|| num_layers.saturating_sub(8));

                // Build canonical prompt (same logic as LQL exec_insert).
                let rel_words = relation.replace(['-', '_'], " ");
                let prompt = format!("The {rel_words} of {entity} is");
                let prompt_enc = tokenizer.encode(prompt.as_str(), true).map_err(|e| e.to_string())?;
                let prompt_ids: Vec<u32> = prompt_enc.get_ids().to_vec();

                // Target token id — tokenize " {target}" and take first id.
                let spaced_target = format!(" {target}");
                let tgt_enc = tokenizer.encode(spaced_target.as_str(), false).map_err(|e| e.to_string())?;
                let target_id: u32 = tgt_enc.get_ids().first().copied().unwrap_or(0);

                eprintln!("[insert] capturing residual for \"{prompt}\" at L{install_layer} ...");
                let t = Instant::now();
                backend.reset_kv_cache();

                // Capture via GPU decode path (works with SKIP_FFN_LOAD).
                // Run sequential decode_token_with_probe for each prompt token,
                // probing h_post_attn at the install layer. The last token's
                // probe output IS the residual we need for KNN INSERT.
                let key = if walk_only {
                    // Build pipeline layers for decode_token
                    let gi: &dyn larql_vindex::GateIndex = &index;
                    let ins_mmap = gi.interleaved_q4k_real_mmap_ref()
                        .or_else(|| gi.interleaved_q4k_mmap_ref())
                        .unwrap_or(&[][..]);
                    let ins_inter = gi.num_features(0);
                    let ins_q4_per = (ins_inter * weights.hidden_size).div_ceil(256) * 148;
                    if ins_mmap.is_empty() || ins_inter == 0 {
                        None
                    } else {
                        let ins_layers = larql_inference::layer_graph::pipeline_layer::build_pipeline_layers(
                            weights, &index, 0..num_layers,
                            ins_mmap, ins_q4_per, larql_compute::QuantFormat::Q4_K,
                        );
                        let embeds = larql_inference::forward::embed_tokens_pub(weights, &prompt_ids);
                        let mut probe_h = None;
                        let rope = weights.arch.rope_base_for_layer(0) as f32;
                        for p in 0..prompt_ids.len() {
                            let x: Vec<f32> = embeds.row(p).to_vec();
                            let (_h, ph) = backend.decode_token_with_probe(
                                &ins_layers, &x, weights.hidden_size, ins_inter,
                                weights.num_q_heads * weights.head_dim,
                                weights.num_kv_heads * weights.head_dim,
                                weights.num_q_heads, weights.num_kv_heads,
                                weights.head_dim, rope,
                                Some(install_layer),
                            ).unwrap_or((vec![], None));
                            if ph.is_some() { probe_h = ph; }
                        }
                        // Apply pre_ffn_norm to the probed h_post_attn (same
                        // normalization the KNN check uses at query time).
                        probe_h.map(|ph| {
                            let norm_offset = weights.arch.norm_weight_offset();
                            let ph_arr = ndarray::Array2::from_shape_vec(
                                (1, weights.hidden_size), ph).unwrap();
                            let pre_ffn_key = weights.arch.pre_feedforward_layernorm_key(install_layer);
                            let normed = match pre_ffn_key {
                                Some(key) => larql_inference::forward::apply_norm(
                                    weights, &ph_arr, &key, norm_offset),
                                None => ph_arr,
                            };
                            normed.row(0).to_vec()
                        })
                    }
                } else {
                    larql_inference::capture_residual_post_attn_norm(
                        weights, &prompt_ids, install_layer, &*backend,
                    )
                };
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                match key {
                    Some(k) => {
                        knn_store.add(install_layer, k, target_id, target.clone(),
                            entity.clone(), relation.clone(), 1.0);
                        println!("  inserted: {entity} —[{relation}]→ {target}  L{install_layer}  ({:.0}ms)", ms);
                        println!("  KNN overlay now: {} entries across layers {:?}",
                            knn_store.len(), knn_store.layers());
                    }
                    None => println!("  insert failed — capture returned None (non-post-norm model or layer out of range)"),
                }
                // Clear the KV cache we polluted during capture; subsequent ask
                // calls reset anyway, but be polite.
                backend.reset_kv_cache();
            }
            "save" => {
                // Persist the runtime KNN overlay as a .vlp file — symmetric with LQL BEGIN/SAVE PATCH.
                let path_str = parse_quoted(rest);
                if path_str.is_empty() {
                    println!("  usage: save \"path.vlp\"");
                    print!("> "); stdout.flush().ok(); continue;
                }
                let mut ops: Vec<PatchOp> = Vec::new();
                for (layer, entries) in knn_store.entries().iter() {
                    for e in entries {
                        let b64 = larql_vindex::patch::core::encode_gate_vector(&e.key);
                        ops.push(PatchOp::InsertKnn {
                            layer: *layer,
                            entity: e.entity.clone(),
                            relation: e.relation.clone(),
                            target: e.target_token.clone(),
                            target_id: e.target_id,
                            confidence: Some(e.confidence),
                            key_vector_b64: b64,
                        });
                    }
                }
                let patch = VindexPatch {
                    version: 1,
                    base_model: model_ref.clone(),
                    base_checksum: None,
                    created_at: format!("{:?}", std::time::SystemTime::now()),
                    description: Some("runtime INSERTs captured via bench_interactive Metal f32 path".into()),
                    author: None,
                    tags: Vec::new(),
                    operations: ops,
                };
                match patch.save(std::path::Path::new(&path_str)) {
                    Ok(()) => println!("  saved {} entries to {}", knn_store.len(), path_str),
                    Err(e) => println!("  save failed: {e}"),
                }
            }
            "chat" => {
                // Chat mode: wrap in Gemma 3 chat template for instruction following.
                // Usage: chat Hello there!
                // Or: chat "Write a Python chess game" 200
                let (text_raw, n_raw) = split_trailing_int(rest);
                let user_text = parse_quoted(text_raw);
                // No hardcoded limit — model stops on <end_of_turn> naturally.
                // Cap at 4096 to prevent infinite loops on degenerate output.
                let n: usize = 4096;
                // System prompt — encourage complete responses
                let system = "You are a helpful coding assistant. Always give complete, detailed answers. When writing code, provide the full working program in a markdown code block. Never cut your response short.";
                let chat_prompt = format!(
                    "<start_of_turn>system\n{system}<end_of_turn>\n\
                     <start_of_turn>user\n{user_text}<end_of_turn>\n\
                     <start_of_turn>model\n"
                );
                let enc = tokenizer.encode(chat_prompt.as_str(), true).map_err(|e| e.to_string())?;
                let mut ids: Vec<u32> = enc.get_ids().to_vec();
                // Delegate to the same ask logic below
                // (fall through by rewriting cmd — Rust doesn't allow, so inline)
                if walk_only {
                    backend.reset_kv_cache();
                    let walk = WalkFfn::new_with_backend(weights, &index, 1024, &*backend);
                    let cache = CachedLayerGraph::from_residuals(Vec::new());
                    let mut sampler = build_sampler();

                    let t0 = Instant::now();
                    let r = larql_inference::predict_honest_with_knn_ffn(
                        weights, tokenizer, &ids, 20, &index, &*backend, &cache,
                        0..num_layers, knn_ref(&knn_store), Some(&walk),
                    );
                    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

                    // Don't echo the prompt (TUI handles display)
                    let first_label = r.predictions.first().map(|(s,_)| s.as_str()).unwrap_or("");
                    let knn_hit = first_label.contains("KNN override");
                    if knn_hit {
                        if let Some((s, _)) = r.predictions.first() { println!("{s}"); }
                        println!("  walk-only prefill: {:.0}ms (KNN override, no decode)", prefill_ms);
                        print!("> "); stdout.flush().ok();
                        continue;
                    }

                    let mut next = match sampler.sample(&r.raw_predictions) {
                        Some(tid) => {
                            let tok = tokenizer.decode(&[tid], true).unwrap_or_default();
                            print!("{tok}"); stdout.flush().ok();
                            tid
                        }
                        None => { println!(); print!("> "); stdout.flush().ok(); continue; }
                    };

                    let mut per: Vec<f64> = Vec::with_capacity(n);
                    let mut stopped = false;
                    for _ in 0..n {
                        let input = vec![next];
                        let t = Instant::now();
                        let r = larql_inference::predict_honest_with_knn_ffn(
                            weights, tokenizer, &input, 20, &index, &*backend, &cache,
                            0..num_layers, knn_ref(&knn_store), Some(&walk),
                        );
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        per.push(ms);
                        let tid = match sampler.sample(&r.raw_predictions) {
                            Some(t) => t,
                            None => break,
                        };
                        let tok = tokenizer.decode(&[tid], true).unwrap_or_default();
                        let is_eos = tok.trim() == "<eos>" || tok.trim() == "</s>"
                            || tok.trim() == "<end_of_turn>" || tok.trim() == "<|endoftext|>";
                        if is_eos { stopped = true; break; }
                        print!("{tok}"); stdout.flush().ok();
                        next = tid;
                    }
                    println!();
                    let avg = if per.is_empty() { 0.0 } else { per.iter().sum::<f64>() / per.len() as f64 };
                    println!("  prefill: {:.0}ms  decode: {:.0}ms/tok ({:.2} tok/s) over {} tokens{}",
                        prefill_ms, avg, if avg>0.0 {1000.0/avg} else {0.0}, per.len(),
                        if stopped { "  [stopped on EOS]" } else { "" });
                }
                print!("> "); stdout.flush().ok();
                continue;
            }
            "ask" => {
                // Raw text completion: prefill + decode + print. KNN overlay consulted.
                let (text_raw, n_raw) = split_trailing_int(rest);
                let text = parse_quoted(text_raw);
                let n: usize = n_raw.parse().unwrap_or(20);
                let enc = tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
                let mut ids: Vec<u32> = enc.get_ids().to_vec();

                // Walk-only mode: FFN weights dropped; route through WalkFfn
                // reading Q4_0 vindex instead. Uses predict_honest_with_knn_ffn
                // so we get Metal KV-cached attention AND sparse Q4_0 walk FFN
                // AND KNN overlay consultation — the model-as-graph hot path.
                if walk_only {
                    backend.reset_kv_cache();
                    let walk = WalkFfn::new_with_backend(weights, &index, 1024, &*backend);
                    // Empty cache — avoids triggering a spurious layer-0
                    // forward pass via DenseLayerGraph at build time (which
                    // was causing a second walk dispatch with a stale
                    // backend reference in the captured closure).
                    let cache = CachedLayerGraph::from_residuals(Vec::new());
                    let mut sampler = build_sampler();

                    let t0 = Instant::now();
                    let r = larql_inference::predict_honest_with_knn_ffn(
                        weights, tokenizer, &ids, 20, &index, &*backend, &cache,
                        0..num_layers, knn_ref(&knn_store), Some(&walk),
                    );
                    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

                    print!("{text}");
                    let first_label = r.predictions.first().map(|(s,_)| s.as_str()).unwrap_or("");
                    let knn_hit = first_label.contains("KNN override");
                    if knn_hit {
                        if let Some((s, _)) = r.predictions.first() { print!(" {s}"); }
                        println!();
                        println!("  walk-only prefill: {:.0}ms (KNN override, no decode)", prefill_ms);
                        last_prompt_tokens = ids;
                        last_prediction = r.raw_predictions.first().map(|&(t,_,_)|t);
                        print!("> "); stdout.flush().ok();
                        continue;
                    }

                    let mut next = match sampler.sample(&r.raw_predictions) {
                        Some(tid) => {
                            let tok = tokenizer.decode(&[tid], true).unwrap_or_default();
                            print!("{tok}"); stdout.flush().ok();
                            tid
                        }
                        None => { println!(); continue; }
                    };

                    let mut per: Vec<f64> = Vec::with_capacity(n);
                    let mut stopped = false;
                    for _ in 0..n {
                        let input = vec![next];
                        let t = Instant::now();
                        let r = larql_inference::predict_honest_with_knn_ffn(
                            weights, tokenizer, &input, 20, &index, &*backend, &cache,
                            0..num_layers, knn_ref(&knn_store), Some(&walk),
                        );
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        per.push(ms);
                        match sampler.sample(&r.raw_predictions) {
                            Some(tid) if stop_ids.contains(&tid) => { stopped = true; break; }
                            Some(tid) => {
                                let tok = tokenizer.decode(&[tid], true).unwrap_or_default();
                                print!("{tok}"); stdout.flush().ok();
                                next = tid;
                            }
                            None => break,
                        }
                    }
                    println!();
                    let avg = if !per.is_empty() { per.iter().sum::<f64>() / per.len() as f64 } else { 0.0 };
                    println!("  walk-only prefill: {:.0}ms  decode: {:.0}ms/tok ({:.2} tok/s) over {} tokens{}",
                        prefill_ms, avg, if avg>0.0 {1000.0/avg} else {0.0}, per.len(),
                        if stopped { "  [stopped on EOS]" } else { "" });
                    last_prompt_tokens = ids;
                    last_prediction = Some(next);
                    print!("> "); stdout.flush().ok();
                    continue;
                }

                backend.reset_kv_cache();
                let cache = CachedLayerGraph::build(weights, &ids, &empty_cache_layers, &dense_ffn);

                let t0 = Instant::now();
                let r = larql_inference::layer_graph::predict::predict_honest_with_knn(
                    weights, tokenizer, &ids, 5, &index, &*backend, &cache, 0..num_layers, knn_ref(&knn_store),
                );
                let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

                print!("{text}");
                let first_label = r.predictions.first().map(|(s,_)| s.as_str()).unwrap_or("");
                let knn_hit = first_label.contains("KNN override");
                if let Some((s, _)) = r.predictions.first() {
                    // KNN override labels lead with the bare token ("Rome ...")
                    // which butts against the end of the prompt. Add a space.
                    let prefix = if knn_hit { " " } else { "" };
                    print!("{prefix}{s}");
                }
                stdout.flush().ok();

                let mut next = match r.raw_predictions.first() { Some(&(tid,_,_)) => tid, None => { println!(); continue; } };

                if knn_hit {
                    println!();
                    println!("  prefill: {:.0}ms (KNN override, no decode needed)", prefill_ms);
                    last_prompt_tokens = ids;
                    last_prediction = Some(next);
                    print!("> "); stdout.flush().ok();
                    continue;
                }

                let mut sampler = build_sampler();
                let mut decode_per: Vec<f64> = Vec::with_capacity(n);
                let mut stopped = false;
                for _ in 0..n {
                    let input = vec![next];
                    let t = Instant::now();
                    // top_k=20 so the sampler has a proper candidate list to
                    // sample from; greedy still just picks index 0.
                    let r = larql_inference::layer_graph::predict::predict_honest_with_knn(
                        weights, tokenizer, &input, 20, &index, &*backend, &cache, 0..num_layers, knn_ref(&knn_store),
                    );
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    decode_per.push(ms);
                    match sampler.sample(&r.raw_predictions) {
                        Some(tid) if stop_ids.contains(&tid) => { stopped = true; break; }
                        Some(tid) => {
                            let tok = tokenizer.decode(&[tid], true).unwrap_or_default();
                            print!("{tok}"); stdout.flush().ok();
                            next = tid;
                        }
                        None => break,
                    }
                }
                println!();
                let avg = if !decode_per.is_empty() { decode_per.iter().sum::<f64>() / decode_per.len() as f64 } else { 0.0 };
                println!("  prefill: {:.0}ms  decode: {:.0}ms/tok avg ({:.2} tok/s) over {} tokens{}",
                    prefill_ms, avg, if avg>0.0 {1000.0/avg} else {0.0}, decode_per.len(),
                    if stopped { "  [stopped on EOS]" } else { "" });
                last_prompt_tokens = ids;
                last_prediction = Some(next);
            }
            "spec" => {
                // Speculative decode: n-gram draft K tokens, verify, accept prefix.
                // Usage: spec "prompt text" [N] [K]
                // N = max tokens (default 20), K = max draft length (default 4)
                let (text_raw, n_raw) = split_trailing_int(rest);
                let text = parse_quoted(text_raw);
                let parts: Vec<&str> = n_raw.split_whitespace().collect();
                let n: usize = parts.first().and_then(|s| s.parse().ok()).unwrap_or(20);
                let k: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
                let enc = tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
                let ids: Vec<u32> = enc.get_ids().to_vec();

                // Build pipeline layers for decode_token
                let gi: &dyn larql_vindex::GateIndex = &index;
                let spec_ffn_mmap = gi.interleaved_q4k_real_mmap_ref()
                    .or_else(|| gi.interleaved_q4k_mmap_ref())
                    .unwrap_or(&[][..]);
                let spec_inter = gi.num_features(0);
                let spec_q4_per = (spec_inter * weights.hidden_size).div_ceil(256) * 148;
                eprintln!("[spec] building layers: inter={spec_inter} mmap_len={} q4_per={spec_q4_per}",
                    spec_ffn_mmap.len());
                let spec_layers = if spec_ffn_mmap.is_empty() || spec_inter == 0 {
                    eprintln!("[spec] WARNING: no FFN mmap or zero features, using empty layers");
                    Vec::new()
                } else {
                    larql_inference::layer_graph::pipeline_layer::build_pipeline_layers(
                        weights, &index, 0..num_layers,
                        spec_ffn_mmap, spec_q4_per, larql_compute::QuantFormat::Q4_K,
                    )
                };
                let r = larql_inference::layer_graph::speculative::generate_speculative(
                    weights, tokenizer, &ids, n, k,
                    &index, &*backend, &spec_layers,
                );

                print!("{text}");
                for (tok, _) in &r.tokens { print!("{tok}"); }
                println!();
                println!("  prefill: {:.0}ms  decode: {:.0}ms  tokens: {}  effective: {:.2} tok/s",
                    r.prefill_ms, r.decode_ms, r.tokens.len(), r.effective_tok_s());
                println!("  spec stats: drafted {} accepted {} ({:.1}%) cycles {}",
                    r.total_drafted, r.total_accepted,
                    r.acceptance_rate() * 100.0, r.total_cycles);
            }
            "prompt" => {
                let text = parse_quoted(rest);
                let enc = tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
                let ids: Vec<u32> = enc.get_ids().to_vec();
                let cache = CachedLayerGraph::build(weights, &ids, &empty_cache_layers, &dense_ffn);
                let t = Instant::now();
                let r = larql_inference::layer_graph::predict::predict_honest_with_knn(
                    weights, tokenizer, &ids, 5, &index, &*backend, &cache, 0..num_layers, knn_ref(&knn_store),
                );
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                last_prompt_tokens = ids.clone();
                for (i, (s, p)) in r.predictions.iter().take(5).enumerate() {
                    println!("  {:>2}. {:?}  {:.2}%", i+1, s, p * 100.0);
                }
                println!("  time: {:.1}ms  ({} input tokens)", ms, ids.len());
            }
            "gen" | "walkgen" => {
                // `gen "text" N`     — uses dense WeightFfn
                // `walkgen "text" N` — uses WalkFfn (KNN walk over gate index)
                let use_walk = cmd == "walkgen";
                let (text_raw, n_raw) = split_trailing_int(rest);
                let text = parse_quoted(text_raw);
                let n: usize = n_raw.parse().unwrap_or(20);
                let enc = tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
                let mut ids: Vec<u32> = enc.get_ids().to_vec();

                print!("{text}"); stdout.flush().ok();
                let total_start = Instant::now();
                let mut per: Vec<f64> = Vec::with_capacity(n);
                for step in 0..n {
                    let t = Instant::now();
                    let r = if use_walk {
                        predict_with_ffn(weights, tokenizer, &ids, 1, &walk_ffn)
                    } else {
                        predict_with_ffn(weights, tokenizer, &ids, 1, &dense_ffn)
                    };
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    per.push(ms);
                    if let Some(&(tid, _, _)) = r.raw_predictions.first() {
                        let s = r.predictions.first().map(|(s,_)| s.clone()).unwrap_or_default();
                        print!("{s}"); stdout.flush().ok();
                        ids.push(tid);
                        eprintln!("[{} step {:>3} tid={:>6} {:>6.0}ms {:.2} tok/s]",
                            if use_walk {"walk"} else {"dense"}, step+1, tid, ms, 1000.0/ms);
                    } else {
                        eprintln!("[{} step {:>3} empty, stopping]",
                            if use_walk {"walk"} else {"dense"}, step+1);
                        break;
                    }
                }
                println!();
                let total = total_start.elapsed().as_secs_f64();
                let avg = if !per.is_empty() { per.iter().sum::<f64>() / per.len() as f64 } else { 0.0 };
                println!("  total: {:.1}s  avg: {:.0}ms/tok ({:.2} tok/s) [{}]",
                    total, avg, if avg>0.0 {1000.0/avg} else {0.0}, if use_walk {"WalkFfn"} else {"WeightFfn dense"});
            }
            "walkbench" => {
                // `walkbench "text" top_k N` — walk-FFN with a specific top_k cap.
                // Runs N decode steps (growing context), reports per-step timing
                // and prints the generated text. For top_k research: does the
                // sparse walk match dense at top_k=256? top_k=1024? top_k=4096?
                let parts = parse_quoted_list(rest);
                if parts.len() < 2 {
                    println!("  usage: walkbench \"text\" top_k [N]");
                    print!("> "); stdout.flush().ok(); continue;
                }
                let text = parts[0].clone();
                let top_k: usize = parts[1].parse().unwrap_or(8092);
                let n: usize = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

                // Build a fresh WalkFfn with the requested top_k cap, passing
                // the Metal backend so the Q4_K interleaved path can fire when
                // `interleaved_q4k.bin` is loaded.
                let walk = WalkFfn::new_with_backend(weights, &index, top_k, &*backend);

                let enc = tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
                let mut ids: Vec<u32> = enc.get_ids().to_vec();
                print!("{text}"); stdout.flush().ok();
                let total_start = Instant::now();
                let mut per: Vec<f64> = Vec::with_capacity(n);
                for step in 0..n {
                    let t = Instant::now();
                    let r = predict_with_ffn(weights, tokenizer, &ids, 1, &walk);
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    per.push(ms);
                    if let Some(&(tid, _, _)) = r.raw_predictions.first() {
                        let s = r.predictions.first().map(|(s,_)| s.clone()).unwrap_or_default();
                        print!("{s}"); stdout.flush().ok();
                        ids.push(tid);
                        eprintln!("[walk k={top_k} step {:>3} tid={:>6} {:>6.0}ms]", step+1, tid, ms);
                    } else { break; }
                }
                println!();
                let total = total_start.elapsed().as_secs_f64();
                let avg = if !per.is_empty() { per.iter().sum::<f64>() / per.len() as f64 } else { 0.0 };
                println!("  walk top_k={top_k}  total: {:.1}s  avg: {:.0}ms/tok ({:.2} tok/s)",
                    total, avg, if avg>0.0 {1000.0/avg} else {0.0});
            }
            "kvreset" => {
                backend.reset_kv_cache();
                println!("  kv cache cleared");
            }
            "cpupredict" => {
                // Pure CPU forward pass via forward::predict (matches bench_inference).
                // Use this as numerical ground truth to diff against Metal decode_token.
                let text = parse_quoted(rest);
                let enc = tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
                let ids: Vec<u32> = enc.get_ids().to_vec();
                eprintln!("[cpupredict] input token IDs: {:?}", ids);
                let t = Instant::now();
                let r = larql_inference::predict(weights, tokenizer, &ids, 5);
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                for (i, (s, p)) in r.predictions.iter().take(5).enumerate() {
                    println!("  {:>2}. {:?}  {:.2}%", i+1, s, p * 100.0);
                }
                println!("  cpupredict: {:.1}ms  ({} tokens, CPU forward::predict)", ms, ids.len());
            }
            "gpuprefill" => {
                // Full-prompt prefill in a SINGLE predict_honest call. For Gemma 3
                // (post-norm) this routes through the CPU+backend-matmul multi-token
                // path which handles cross-token attention correctly; for other
                // models it goes through prefill_q4 (Q4_K GPU prefill).
                // Populates the Metal KV cache for subsequent kvdecode calls.
                backend.reset_kv_cache();
                let text = parse_quoted(rest);
                let enc = tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
                let ids: Vec<u32> = enc.get_ids().to_vec();
                let cache = CachedLayerGraph::build(weights, &ids, &empty_cache_layers, &dense_ffn);
                let t = Instant::now();
                let r = larql_inference::layer_graph::predict::predict_honest_with_knn(
                    weights, tokenizer, &ids, 5, &index, &*backend, &cache, 0..num_layers, knn_ref(&knn_store),
                );
                let total_ms = t.elapsed().as_secs_f64() * 1000.0;
                for (i, (s, p)) in r.predictions.iter().take(5).enumerate() {
                    println!("  {:>2}. {:?}  {:.2}%", i+1, s, p * 100.0);
                }
                println!("  gpuprefill total: {:.1}ms  ({} input tokens)", total_ms, ids.len());
                last_prompt_tokens = ids;
                last_prediction = r.raw_predictions.first().map(|&(tid, _, _)| tid);
            }
            "kvprefill" => {
                let text = parse_quoted(rest);
                let enc = tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
                let ids: Vec<u32> = enc.get_ids().to_vec();
                let cache = CachedLayerGraph::build(weights, &ids, &empty_cache_layers, &dense_ffn);
                let t = Instant::now();
                let r = larql_inference::layer_graph::predict::predict_honest_with_knn(
                    weights, tokenizer, &ids, 5, &index, &*backend, &cache, 0..num_layers, knn_ref(&knn_store),
                );
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                last_prompt_tokens = ids.clone();
                for (i, (s, p)) in r.predictions.iter().take(5).enumerate() {
                    println!("  {:>2}. {:?}  {:.2}%", i+1, s, p * 100.0);
                }
                println!("  prefill: {:.1}ms  ({} input tokens, KV populated)", ms, ids.len());
            }
            "kvdecode" => {
                let n: usize = rest.parse().unwrap_or(1);
                if last_prompt_tokens.is_empty() {
                    println!("  no prompt yet — run kvprefill first");
                } else {
                    let mut per: Vec<f64> = Vec::with_capacity(n);
                    let cache = CachedLayerGraph::build(weights, &last_prompt_tokens, &empty_cache_layers, &dense_ffn);
                    // Start from the PREDICTED next token (from prompt/gpuprefill), not
                    // the last prompt input. That way kvdecode continues the sentence
                    // instead of re-processing the last input token.
                    let mut next: u32 = last_prediction.unwrap_or_else(||
                        last_prompt_tokens.last().copied().unwrap_or(1));
                    eprintln!("[kvdecode] starting with next={next}, last_prompt_len={}",
                        last_prompt_tokens.len());
                    for step in 0..n {
                        let input = vec![next];
                        let t = Instant::now();
                        let r = larql_inference::layer_graph::predict::predict_honest_with_knn(
                            weights, tokenizer, &input, 1, &index, &*backend, &cache, 0..num_layers, knn_ref(&knn_store),
                        );
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        per.push(ms);
                        if let Some(&(tid, _, _)) = r.raw_predictions.first() {
                            let s = r.predictions.first().map(|(s,_)| s.clone()).unwrap_or_default();
                            let top5: Vec<String> = r.predictions.iter().take(5)
                                .map(|(t, p)| format!("{t:?} {:.1}%", p * 100.0)).collect();
                            println!("  step {:>3}: tid={:>6} {:?} {:.0}ms  top5=[{}]",
                                step+1, tid, s, ms, top5.join(", "));
                            next = tid;
                        } else {
                            println!("  step {:>3}: empty, stopping", step+1);
                            break;
                        }
                    }
                    let avg = if !per.is_empty() { per.iter().sum::<f64>() / per.len() as f64 } else { 0.0 };
                    println!("  avg: {:.0}ms/tok ({:.2} tok/s)", avg, if avg>0.0 {1000.0/avg} else {0.0});
                }
            }
            other => println!("  unknown command: {other}"),
        }
        print!("> "); stdout.flush().ok();
    }
    eprintln!("[exit]");
    Ok(())
}

fn knn_ref(s: &KnnStore) -> Option<&KnnStore> {
    if s.is_empty() { None } else { Some(s) }
}

/// Split a whitespace-separated sequence honoring double-quoted groups.
/// `"a b" "c" d` → `["a b", "c", "d"]`.
fn parse_quoted_list(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut in_quote = false;
    for ch in s.chars() {
        match ch {
            '"' => {
                if in_quote {
                    out.push(std::mem::take(&mut buf));
                }
                in_quote = !in_quote;
            }
            c if c.is_whitespace() && !in_quote => {
                if !buf.is_empty() { out.push(std::mem::take(&mut buf)); }
            }
            c => buf.push(c),
        }
    }
    if !buf.is_empty() { out.push(buf); }
    out
}

fn parse_quoted(s: &str) -> String {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        s[1..s.len()-1].to_string()
    } else {
        s.to_string()
    }
}

fn split_trailing_int(s: &str) -> (&str, &str) {
    // Find trailing integer separated by whitespace
    let s = s.trim_end();
    if let Some(pos) = s.rfind(char::is_whitespace) {
        let (left, right) = s.split_at(pos);
        let right = right.trim();
        if right.chars().all(|c| c.is_ascii_digit()) {
            return (left.trim(), right);
        }
    }
    (s, "20")
}
