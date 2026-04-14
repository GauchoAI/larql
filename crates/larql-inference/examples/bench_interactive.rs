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
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = std::path::PathBuf::new();
    let mut model_ref = String::from("google/gemma-3-4b-it");
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => { i += 1; vindex_path = std::path::PathBuf::from(&args[i]); }
            "--model" => { i += 1; model_ref = args[i].clone(); }
            _ => {}
        }
        i += 1;
    }
    if !vindex_path.is_dir() {
        eprintln!("Usage: bench_interactive --model MODEL --vindex PATH");
        std::process::exit(1);
    }

    eprintln!("[load] model {model_ref}");
    let t0 = Instant::now();
    let model = InferenceModel::load(&model_ref)?;
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
    let _ = index.load_interleaved_q4(&vindex_path);
    let _ = index.load_interleaved_q4k(&vindex_path);
    eprintln!("[load] vindex: {:.1}s", t0.elapsed().as_secs_f64());

    let backend = default_backend();
    let dense_ffn = WeightFfn { weights };
    let walk_ffn = WalkFfn::new_unlimited(weights, &index);
    let empty_cache_layers: Vec<usize> = Vec::new();

    eprintln!("[ready] backend={} layers={num_layers}", backend.name());
    eprintln!("[ready] commands: prompt, gen, kvreset, kvprefill, kvdecode, help, quit");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    print!("> ");
    stdout.flush().ok();

    let mut last_prompt_tokens: Vec<u32> = Vec::new();

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
                println!("  prompt \"text\"     — dense predict, top-5 + time");
                println!("  gen \"text\" N      — dense gen N tokens, growing prompt");
                println!("  kvreset           — clear backend KV cache");
                println!("  kvprefill \"text\"  — prefill prompt, populate KV, time");
                println!("  kvdecode N        — decode N single-token steps from current KV");
                println!("  quit");
            }
            "prompt" => {
                let text = parse_quoted(rest);
                let enc = tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
                let ids: Vec<u32> = enc.get_ids().to_vec();
                let cache = CachedLayerGraph::build(weights, &ids, &empty_cache_layers, &dense_ffn);
                let t = Instant::now();
                let r = larql_inference::layer_graph::predict::predict_honest(
                    weights, tokenizer, &ids, 5, &index, &*backend, &cache, 0..num_layers,
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
            "kvreset" => {
                backend.reset_kv_cache();
                println!("  kv cache cleared");
            }
            "gpuprefill" => {
                // Prefill-via-decode-loop: reset cache, then call decode_token once per
                // prompt token in sequence. This keeps the entire forward pass in
                // Q4K-quantized space (matching subsequent kvdecode) instead of the
                // CPU-f16 mismatch that kvprefill creates.
                backend.reset_kv_cache();
                let text = parse_quoted(rest);
                let enc = tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
                let ids: Vec<u32> = enc.get_ids().to_vec();
                let cache = CachedLayerGraph::build(weights, &ids, &empty_cache_layers, &dense_ffn);
                let t = Instant::now();
                let mut last_preds: Vec<(String, f64)> = Vec::new();
                let mut last_raw: Vec<(u32, f32, f64)> = Vec::new();
                for (i, &tid) in ids.iter().enumerate() {
                    let t_step = Instant::now();
                    let r = larql_inference::layer_graph::predict::predict_honest(
                        weights, tokenizer, &[tid], 5, &index, &*backend, &cache, 0..num_layers,
                    );
                    let ms = t_step.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[gpuprefill tok {i}/{}  id={}  {:.1}ms]", ids.len(), tid, ms);
                    last_preds = r.predictions.clone();
                    last_raw = r.raw_predictions.clone();
                }
                let total_ms = t.elapsed().as_secs_f64() * 1000.0;
                for (i, (s, p)) in last_preds.iter().take(5).enumerate() {
                    println!("  {:>2}. {:?}  {:.2}%", i+1, s, p * 100.0);
                }
                println!("  gpuprefill total: {:.1}ms  ({} tokens, KV populated via decode loop)", total_ms, ids.len());
                last_prompt_tokens = ids;
                let _ = last_raw;
            }
            "kvprefill" => {
                let text = parse_quoted(rest);
                let enc = tokenizer.encode(text.as_str(), true).map_err(|e| e.to_string())?;
                let ids: Vec<u32> = enc.get_ids().to_vec();
                let cache = CachedLayerGraph::build(weights, &ids, &empty_cache_layers, &dense_ffn);
                let t = Instant::now();
                let r = larql_inference::layer_graph::predict::predict_honest(
                    weights, tokenizer, &ids, 5, &index, &*backend, &cache, 0..num_layers,
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
                    let mut next: u32 = last_prompt_tokens.last().copied().unwrap_or(1);
                    for step in 0..n {
                        let input = vec![next];
                        let t = Instant::now();
                        let r = larql_inference::layer_graph::predict::predict_honest(
                            weights, tokenizer, &input, 1, &index, &*backend, &cache, 0..num_layers,
                        );
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        per.push(ms);
                        if let Some(&(tid, _, _)) = r.raw_predictions.first() {
                            let s = r.predictions.first().map(|(s,_)| s.clone()).unwrap_or_default();
                            println!("  step {:>3}: tid={:>6} {:?} {:.0}ms", step+1, tid, s, ms);
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
