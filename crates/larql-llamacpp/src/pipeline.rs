//! Safe wrapper over llama.cpp model + context with a minimal greedy
//! `generate` path.  Mirrors the throughput spike.

use llama_cpp_sys_2 as sys;
use std::ffi::CString;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use crate::backend;
use crate::probe::{cb_eval_trampoline, ProbeDispatch, ProbeHandler};

#[derive(Debug, thiserror::Error)]
pub enum LlamaPipelineError {
    #[error("model path contains interior NUL: {0:?}")]
    PathNul(PathBuf),
    #[error("failed to load GGUF model from {0:?}")]
    ModelLoadFailed(PathBuf),
    #[error("failed to create llama context")]
    ContextFailed,
    #[error("tokenize failed (rc={0}, text={1:?})")]
    TokenizeFailed(i32, String),
    #[error("prompt text contains interior NUL")]
    PromptNul,
    #[error("decode failed (rc={0})")]
    DecodeFailed(i32),
    #[error("prompt too long: {len} tokens exceeds n_ctx={n_ctx}")]
    PromptTooLong { len: usize, n_ctx: usize },
}

/// Greedy generation configuration.  Sampler support can be added later.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_tokens: usize,
    /// Stop when the model emits the vocabulary EOS token.
    pub stop_at_eos: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            stop_at_eos: true,
        }
    }
}

/// Loaded llama.cpp model + generation context.  Single-threaded: call
/// `generate` from one thread at a time.
pub struct LlamaPipeline {
    model: NonNull<sys::llama_model>,
    ctx: NonNull<sys::llama_context>,
    vocab: *const sys::llama_vocab,
    n_ctx: u32,
    n_vocab: usize,
    eos: sys::llama_token,
    /// Heap-pinned so the `cb_eval_user_data` pointer stays valid.  Must
    /// outlive `ctx`.
    _probe: Option<Box<ProbeDispatch>>,
}

// Safety: the underlying C pointers are owned exclusively by this struct
// and all access is through &mut self on the Rust side.
unsafe impl Send for LlamaPipeline {}

impl LlamaPipeline {
    /// Load a GGUF model onto GPU (Metal) and prepare a decode context.
    pub fn load(gguf: &Path, n_ctx: u32) -> Result<Self, LlamaPipelineError> {
        Self::load_inner(gguf, n_ctx, None)
    }

    /// Load with a residual-stream probe wired into `cb_eval`.  The probe
    /// observes every decode call for the lifetime of this pipeline.
    pub fn load_with_probe(
        gguf: &Path,
        n_ctx: u32,
        probe: Box<dyn ProbeHandler>,
    ) -> Result<Self, LlamaPipelineError> {
        Self::load_inner(gguf, n_ctx, Some(probe))
    }

    fn load_inner(
        gguf: &Path,
        n_ctx: u32,
        probe: Option<Box<dyn ProbeHandler>>,
    ) -> Result<Self, LlamaPipelineError> {
        backend::ensure_initialized();

        let path_c = CString::new(gguf.as_os_str().to_string_lossy().into_owned())
            .map_err(|_| LlamaPipelineError::PathNul(gguf.to_path_buf()))?;

        let model = unsafe {
            let mut mparams = sys::llama_model_default_params();
            mparams.n_gpu_layers = 999;
            sys::llama_model_load_from_file(path_c.as_ptr(), mparams)
        };
        let model = NonNull::new(model)
            .ok_or_else(|| LlamaPipelineError::ModelLoadFailed(gguf.to_path_buf()))?;

        let probe_dispatch = probe.map(ProbeDispatch::new);

        let ctx = unsafe {
            let mut cparams = sys::llama_context_default_params();
            cparams.n_ctx = n_ctx;
            cparams.n_batch = n_ctx.max(512);
            if let Some(d) = probe_dispatch.as_ref() {
                cparams.cb_eval = Some(cb_eval_trampoline);
                cparams.cb_eval_user_data =
                    (&**d as *const ProbeDispatch as *mut ProbeDispatch) as *mut _;
            }
            sys::llama_init_from_model(model.as_ptr(), cparams)
        };
        let ctx = match NonNull::new(ctx) {
            Some(c) => c,
            None => {
                unsafe { sys::llama_model_free(model.as_ptr()) };
                return Err(LlamaPipelineError::ContextFailed);
            }
        };

        let vocab = unsafe { sys::llama_model_get_vocab(model.as_ptr()) };
        let n_vocab = unsafe { sys::llama_vocab_n_tokens(vocab) } as usize;
        let eos = unsafe { sys::llama_vocab_eos(vocab) };

        Ok(Self {
            model,
            ctx,
            vocab,
            n_ctx,
            n_vocab,
            eos,
            _probe: probe_dispatch,
        })
    }

    /// Tokenize `text` with the model's vocab.  `add_special` controls BOS.
    fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<sys::llama_token>, LlamaPipelineError> {
        let c = CString::new(text).map_err(|_| LlamaPipelineError::PromptNul)?;
        // First call to discover required length.
        let mut buf: Vec<sys::llama_token> = vec![0; self.n_ctx as usize];
        let n = unsafe {
            sys::llama_tokenize(
                self.vocab,
                c.as_ptr(),
                c.as_bytes().len() as i32,
                buf.as_mut_ptr(),
                buf.len() as i32,
                add_special,
                true,
            )
        };
        if n < 0 {
            return Err(LlamaPipelineError::TokenizeFailed(n, text.to_string()));
        }
        buf.truncate(n as usize);
        Ok(buf)
    }

    fn detokenize(&self, tokens: &[sys::llama_token]) -> String {
        let piece = [0i8; 128];
        let mut out = String::new();
        for &t in tokens {
            let n = unsafe {
                sys::llama_token_to_piece(
                    self.vocab,
                    t,
                    piece.as_ptr() as *mut c_char,
                    piece.len() as i32,
                    0,
                    true,
                )
            };
            if n > 0 {
                let bytes = unsafe {
                    std::slice::from_raw_parts(piece.as_ptr() as *const u8, n as usize)
                };
                out.push_str(&String::from_utf8_lossy(bytes));
            }
        }
        out
    }

    fn argmax(&self, logits_last: i32) -> sys::llama_token {
        let ptr = unsafe { sys::llama_get_logits_ith(self.ctx.as_ptr(), logits_last) };
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.n_vocab) };
        let mut best_i = 0i32;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in slice.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i as i32;
            }
        }
        best_i
    }

    /// Greedy-decode from `prompt`.  Returns the generated text only
    /// (prompt is not echoed).
    pub fn generate(
        &mut self,
        prompt: &str,
        cfg: &GenerateConfig,
    ) -> Result<String, LlamaPipelineError> {
        let mut prompt_toks = self.tokenize(prompt, true)?;
        if prompt_toks.is_empty() {
            return Ok(String::new());
        }
        if prompt_toks.len() > self.n_ctx as usize {
            return Err(LlamaPipelineError::PromptTooLong {
                len: prompt_toks.len(),
                n_ctx: self.n_ctx as usize,
            });
        }

        // Prefill.
        let n_prefill = prompt_toks.len() as i32;
        let batch = unsafe { sys::llama_batch_get_one(prompt_toks.as_mut_ptr(), n_prefill) };
        let rc = unsafe { sys::llama_decode(self.ctx.as_ptr(), batch) };
        if rc != 0 {
            return Err(LlamaPipelineError::DecodeFailed(rc));
        }

        let mut next = self.pick_token(n_prefill - 1);
        let mut generated = Vec::<sys::llama_token>::with_capacity(cfg.max_tokens);
        for _ in 0..cfg.max_tokens {
            if cfg.stop_at_eos && next == self.eos {
                break;
            }
            generated.push(next);
            let mut single = [next];
            let batch = unsafe { sys::llama_batch_get_one(single.as_mut_ptr(), 1) };
            let rc = unsafe { sys::llama_decode(self.ctx.as_ptr(), batch) };
            if rc != 0 {
                return Err(LlamaPipelineError::DecodeFailed(rc));
            }
            next = self.pick_token(0);
        }
        Ok(self.detokenize(&generated))
    }

    /// Token selection for one decoded position.  Gives the probe handler
    /// a chance to force a token (KNN override); falls back to argmax.
    fn pick_token(&mut self, logits_last: i32) -> sys::llama_token {
        let forced = self
            ._probe
            .as_mut()
            .and_then(|p| p.handler_mut().forced_token());
        let tok = forced.unwrap_or_else(|| self.argmax(logits_last));
        if let Some(p) = self._probe.as_mut() {
            p.handler_mut().reset_step();
        }
        tok
    }

    /// Number of layers in the loaded model.
    pub fn n_layer(&self) -> i32 {
        unsafe { sys::llama_model_n_layer(self.model.as_ptr()) }
    }

    /// Embedding dimension.
    pub fn n_embd(&self) -> i32 {
        unsafe { sys::llama_model_n_embd(self.model.as_ptr()) }
    }

    /// Tokenize `text` (without BOS) and return the first token id.
    /// Handy for looking up vocabulary tokens for KNN targets.
    pub fn token_id_of(&self, text: &str) -> Option<sys::llama_token> {
        let toks = self.tokenize(text, false).ok()?;
        toks.into_iter().next()
    }

    /// Convert a single token id to its decoded piece string.
    pub fn decode_token(&self, tok: sys::llama_token) -> String {
        self.detokenize(&[tok])
    }

    /// Prefill `prompt`, then return the top-`k` (token_id, probability)
    /// pairs from the logits at the last prompt position — softmaxed.
    /// No new tokens are generated.
    pub fn prefill_and_top_k(
        &mut self,
        prompt: &str,
        k: usize,
    ) -> Result<Vec<(sys::llama_token, f32)>, LlamaPipelineError> {
        let mut toks = self.tokenize(prompt, true)?;
        if toks.is_empty() {
            return Ok(Vec::new());
        }
        if toks.len() > self.n_ctx as usize {
            return Err(LlamaPipelineError::PromptTooLong {
                len: toks.len(),
                n_ctx: self.n_ctx as usize,
            });
        }
        let n = toks.len() as i32;
        let batch = unsafe { sys::llama_batch_get_one(toks.as_mut_ptr(), n) };
        let rc = unsafe { sys::llama_decode(self.ctx.as_ptr(), batch) };
        if rc != 0 {
            return Err(LlamaPipelineError::DecodeFailed(rc));
        }
        let ptr = unsafe { sys::llama_get_logits_ith(self.ctx.as_ptr(), n - 1) };
        let logits = unsafe { std::slice::from_raw_parts(ptr, self.n_vocab) };

        // Softmax with max-subtraction for numerical stability.
        let mut max_v = f32::NEG_INFINITY;
        for &v in logits {
            if v > max_v {
                max_v = v;
            }
        }
        let exps: Vec<f32> = logits.iter().map(|&v| (v - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();

        // Partial top-k.  Small-k so linear scan is fine.
        let mut best: Vec<(i32, f32)> = Vec::with_capacity(k);
        for (i, &e) in exps.iter().enumerate() {
            let p = e / sum;
            if best.len() < k {
                best.push((i as i32, p));
                if best.len() == k {
                    best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                }
            } else if p > best.last().unwrap().1 {
                *best.last_mut().unwrap() = (i as i32, p);
                best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
        }
        if best.len() < k {
            best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }
        Ok(best)
    }

    /// Clear the KV cache so the next prefill starts fresh.  Keeps the
    /// context alive (much cheaper than reloading).
    pub fn reset_kv(&mut self) {
        unsafe {
            let mem = sys::llama_get_memory(self.ctx.as_ptr());
            sys::llama_memory_clear(mem, true);
        }
    }

    /// Low-level prefill: tokenize + decode, leaving the KV populated
    /// for subsequent `feed()` calls.  Returns the number of tokens
    /// pushed (the last of which is the position to sample from).
    pub fn prefill(&mut self, prompt: &str) -> Result<usize, LlamaPipelineError> {
        let mut toks = self.tokenize(prompt, true)?;
        if toks.is_empty() {
            return Ok(0);
        }
        if toks.len() > self.n_ctx as usize {
            return Err(LlamaPipelineError::PromptTooLong {
                len: toks.len(),
                n_ctx: self.n_ctx as usize,
            });
        }
        let n = toks.len() as i32;
        let batch = unsafe { sys::llama_batch_get_one(toks.as_mut_ptr(), n) };
        let rc = unsafe { sys::llama_decode(self.ctx.as_ptr(), batch) };
        if rc != 0 {
            return Err(LlamaPipelineError::DecodeFailed(rc));
        }
        Ok(n as usize)
    }

    /// Feed a single token through the model and update the KV cache.
    /// Intended as the step function of a generation loop.
    pub fn feed(&mut self, tok: sys::llama_token) -> Result<(), LlamaPipelineError> {
        let mut single = [tok];
        let batch = unsafe { sys::llama_batch_get_one(single.as_mut_ptr(), 1) };
        let rc = unsafe { sys::llama_decode(self.ctx.as_ptr(), batch) };
        if rc != 0 {
            return Err(LlamaPipelineError::DecodeFailed(rc));
        }
        Ok(())
    }

    /// Extract `(token_id, logit, prob)` tuples for the top-`k` logits
    /// at position `idx` of the most recent decode.  The format matches
    /// `larql_inference::sampling::Sampler::sample` and is sorted
    /// logit-descending.  Use `idx = n_prefill - 1` after prefill,
    /// `idx = 0` after a single `feed()` call.
    pub fn top_k_at(&self, idx: i32, k: usize) -> Vec<(u32, f32, f64)> {
        let ptr = unsafe { sys::llama_get_logits_ith(self.ctx.as_ptr(), idx) };
        if ptr.is_null() {
            return Vec::new();
        }
        let logits = unsafe { std::slice::from_raw_parts(ptr, self.n_vocab) };

        let mut max_v = f32::NEG_INFINITY;
        for &v in logits {
            if v > max_v {
                max_v = v;
            }
        }
        let exps: Vec<f64> = logits
            .iter()
            .map(|&v| ((v - max_v) as f64).exp())
            .collect();
        let sum: f64 = exps.iter().sum();

        let k_eff = if k == 0 { self.n_vocab } else { k.min(self.n_vocab) };
        let mut idxs: Vec<usize> = (0..self.n_vocab).collect();
        idxs.select_nth_unstable_by(k_eff.saturating_sub(1).max(0), |&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idxs.truncate(k_eff);
        idxs.sort_by(|&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idxs.into_iter()
            .map(|i| (i as u32, logits[i], exps[i] / sum))
            .collect()
    }

    /// Tokenize accessor (without BOS).
    pub fn tokenize_plain(&self, text: &str) -> Result<Vec<sys::llama_token>, LlamaPipelineError> {
        self.tokenize(text, false)
    }

    /// BOS-prefixed tokenize (mirrors `generate()`'s internal call).
    pub fn tokenize_with_bos(&self, text: &str) -> Result<Vec<sys::llama_token>, LlamaPipelineError> {
        self.tokenize(text, true)
    }

    /// Apply the model's built-in chat template to a list of
    /// `(role, content)` messages.  Returns the rendered prompt string
    /// with the assistant turn primed (`add_assistant = true`).  Falls
    /// back to a Gemma-style template if the model has none.
    pub fn apply_chat_template(
        &self,
        messages: &[(String, String)],
    ) -> Result<String, LlamaPipelineError> {
        // Hold the C strings alive for the duration of the FFI call.
        let owned: Vec<(CString, CString)> = messages
            .iter()
            .map(|(r, c)| {
                (
                    CString::new(r.as_str()).unwrap_or_default(),
                    CString::new(c.as_str()).unwrap_or_default(),
                )
            })
            .collect();
        let raw: Vec<sys::llama_chat_message> = owned
            .iter()
            .map(|(r, c)| sys::llama_chat_message {
                role: r.as_ptr(),
                content: c.as_ptr(),
            })
            .collect();

        // Use the template stored in the GGUF metadata when available.
        let tmpl_ptr = unsafe { sys::llama_model_chat_template(self.model.as_ptr(), std::ptr::null()) };

        // Estimate buffer length conservatively then grow if needed.
        let mut estimated: usize = messages
            .iter()
            .map(|(r, c)| r.len() + c.len() + 64)
            .sum::<usize>()
            .max(256);

        for _ in 0..3 {
            let mut buf = vec![0i8; estimated];
            let n = unsafe {
                sys::llama_chat_apply_template(
                    tmpl_ptr,
                    raw.as_ptr(),
                    raw.len(),
                    /*add_ass=*/ true,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                )
            };
            if n < 0 {
                // Either no template baked in or the FFI rejected it —
                // fall back to a Gemma chat scaffold so the demo path
                // still works on base / non-templated GGUFs.
                return Ok(render_gemma_chat(messages));
            }
            let n = n as usize;
            if n <= buf.len() {
                let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, n) };
                return Ok(String::from_utf8_lossy(bytes).into_owned());
            }
            estimated = n + 16;
        }
        Ok(render_gemma_chat(messages))
    }
}

/// Minimal Gemma-style chat scaffold used when the GGUF doesn't carry
/// its own template.  Matches gemma-3 instruct format.
fn render_gemma_chat(messages: &[(String, String)]) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        let r = match role.as_str() {
            "assistant" | "model" => "model",
            other => other,
        };
        out.push_str("<start_of_turn>");
        out.push_str(r);
        out.push('\n');
        out.push_str(content);
        out.push_str("<end_of_turn>\n");
    }
    out.push_str("<start_of_turn>model\n");
    out
}

impl Drop for LlamaPipeline {
    fn drop(&mut self) {
        unsafe {
            sys::llama_free(self.ctx.as_ptr());
            sys::llama_model_free(self.model.as_ptr());
        }
    }
}

