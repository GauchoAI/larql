//! Spike: cb_eval read + write + raw throughput measurement.
//!
//! Measures baseline llama.cpp decode speed on Gemma 3 4B Q8_0 (no KNN
//! overlay, no cb_eval overhead unless SPIKE_PROBE=1) so we can compare
//! against our custom Metal pipeline (44.6 tok/s, 9 GB) and Ollama
//! (~51 tok/s, similar RAM).
//!
//! SPIKE_PROBE=1  → register cb_eval observing attn_post_norm-26 + l_out-26
//! SPIKE_OVERRIDE=1 → also zero l_out-26 mid-graph (sanity check)
//! SPIKE_N=128    → number of tokens to generate (default 128)

use llama_cpp_sys_2 as sys;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::time::Instant;

struct ProbeState {
    observe: bool,
    override_l26: bool,
    saw_probe: u64,
    wrote: u64,
}

const PROBE_TARGETS: &[&str] = &["attn_post_norm-26", "l_out-26"];

unsafe extern "C" fn cb_eval(
    t: *mut sys::ggml_tensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    let state = &mut *(user_data as *mut ProbeState);

    let name_ptr = (&(*t).name) as *const c_char;
    let name = CStr::from_ptr(name_ptr).to_string_lossy();

    if ask {
        return state.observe && PROBE_TARGETS.iter().any(|k| name.contains(k));
    }

    if (*t).type_ != sys::GGML_TYPE_F32 {
        return true;
    }

    let nbytes = sys::ggml_nbytes(t);
    let nelements = sys::ggml_nelements(t) as usize;

    if name.contains("attn_post_norm-26") {
        state.saw_probe += 1;
    }

    if state.override_l26 && name.contains("l_out-26") {
        let zeros = vec![0f32; nelements];
        sys::ggml_backend_tensor_set(t, zeros.as_ptr() as *const c_void, 0, nbytes);
        state.wrote += 1;
    }

    true
}

fn rss_mb() -> u64 {
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .expect("ps failed");
    let s = String::from_utf8_lossy(&out.stdout);
    s.trim().parse::<u64>().unwrap_or(0) / 1024
}

fn main() {
    let path = std::env::var("LLAMA_GGUF")
        .unwrap_or_else(|_| "/tmp/gemma3-4b-stock-q8_0.gguf".to_string());
    let prompt = std::env::var("LLAMA_PROMPT")
        .unwrap_or_else(|_| "The capital of Australia is ".to_string());
    let n_gen: usize = std::env::var("SPIKE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let probe = std::env::var("SPIKE_PROBE").map(|v| v == "1").unwrap_or(false);
    let override_l26 = std::env::var("SPIKE_OVERRIDE").map(|v| v == "1").unwrap_or(false);

    println!("llama.cpp throughput spike");
    println!("  model    : {}", path);
    println!("  prompt   : {:?}", prompt);
    println!("  n_gen    : {}", n_gen);
    println!("  probe    : {}", probe);
    println!("  override : {}", override_l26);
    println!("  RSS before load: {} MB", rss_mb());

    unsafe {
        sys::llama_backend_init();

        let mut mparams = sys::llama_model_default_params();
        mparams.n_gpu_layers = 999;
        let path_c = CString::new(path).unwrap();
        let t0 = Instant::now();
        let model = sys::llama_model_load_from_file(path_c.as_ptr(), mparams);
        assert!(!model.is_null(), "model load failed");
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let vocab = sys::llama_model_get_vocab(model);

        let mut state = Box::new(ProbeState {
            observe: probe,
            override_l26,
            saw_probe: 0,
            wrote: 0,
        });
        let state_ptr: *mut ProbeState = &mut *state;

        let mut cparams = sys::llama_context_default_params();
        cparams.n_ctx = 2048;
        cparams.n_batch = 2048;
        if probe || override_l26 {
            cparams.cb_eval = Some(cb_eval);
            cparams.cb_eval_user_data = state_ptr as *mut c_void;
        }

        let ctx = sys::llama_init_from_model(model, cparams);
        assert!(!ctx.is_null(), "ctx create failed");

        // Tokenize.
        let prompt_c = CString::new(prompt.clone()).unwrap();
        let mut tokens = vec![0i32; 128];
        let n_tok = sys::llama_tokenize(
            vocab,
            prompt_c.as_ptr(),
            prompt_c.as_bytes().len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            true,
            true,
        );
        assert!(n_tok > 0, "tokenize failed");
        tokens.truncate(n_tok as usize);
        let n_prefill = n_tok as usize;

        // Prefill.
        let t_prefill = Instant::now();
        let prefill_batch = sys::llama_batch_get_one(tokens.as_mut_ptr(), n_tok);
        let rc = sys::llama_decode(ctx, prefill_batch);
        assert_eq!(rc, 0, "prefill decode failed");
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

        let n_vocab = sys::llama_vocab_n_tokens(vocab) as usize;

        // Greedy generation loop.
        let eos = sys::llama_vocab_eos(vocab);
        let mut generated = Vec::<i32>::with_capacity(n_gen);
        let mut next_pos = n_tok;
        let mut cur_tok = {
            let logits = sys::llama_get_logits_ith(ctx, n_tok - 1);
            let slice = std::slice::from_raw_parts(logits, n_vocab);
            let (argmax, _) = slice
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                    if v > acc.1 {
                        (i, v)
                    } else {
                        acc
                    }
                });
            argmax as i32
        };

        let t_gen = Instant::now();
        for _ in 0..n_gen {
            generated.push(cur_tok);
            if cur_tok == eos {
                break;
            }
            let mut single = [cur_tok];
            let batch = sys::llama_batch_get_one(single.as_mut_ptr(), 1);
            let rc = sys::llama_decode(ctx, batch);
            assert_eq!(rc, 0, "gen decode failed");
            next_pos += 1;

            let logits = sys::llama_get_logits_ith(ctx, 0);
            let slice = std::slice::from_raw_parts(logits, n_vocab);
            let (argmax, _) = slice
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                    if v > acc.1 {
                        (i, v)
                    } else {
                        acc
                    }
                });
            cur_tok = argmax as i32;
        }
        let gen_secs = t_gen.elapsed().as_secs_f64();
        let n_actual = generated.len();
        let tok_s = n_actual as f64 / gen_secs;

        // Detokenize.
        let mut piece = [0i8; 128];
        let mut text = String::new();
        for &t in &generated {
            let n = sys::llama_token_to_piece(
                vocab,
                t,
                piece.as_ptr() as *mut c_char,
                piece.len() as i32,
                0,
                true,
            );
            if n > 0 {
                let s = std::slice::from_raw_parts(piece.as_ptr() as *const u8, n as usize);
                text.push_str(&String::from_utf8_lossy(s));
            }
        }

        let rss_peak = rss_mb();

        println!();
        println!("=== Results ===");
        println!("  load      : {:>8.1} ms", load_ms);
        println!("  prefill   : {:>8.1} ms  ({} tokens, {:.1} tok/s)",
            prefill_ms, n_prefill, n_prefill as f64 / (prefill_ms / 1000.0));
        println!("  generate  : {:>8.3} s   ({} tokens, {:.2} tok/s)",
            gen_secs, n_actual, tok_s);
        println!("  RSS peak  : {:>8} MB", rss_peak);
        if probe {
            println!("  cb_eval observed attn_post_norm-26 : {} times", state.saw_probe);
        }
        if override_l26 {
            println!("  cb_eval wrote l_out-26 : {} times", state.wrote);
        }
        println!();
        println!("OUTPUT: {}{}", prompt, text);

        let _ = next_pos;
        sys::llama_free(ctx);
        sys::llama_model_free(model);
        sys::llama_backend_free();
    }
}
