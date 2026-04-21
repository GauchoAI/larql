//! Token-override KNN probe end-to-end.
//!
//! Steps:
//!   1. Baseline generate from prompt, capture the residual at the probe
//!      layer at the moment the model would emit its first new token.
//!   2. Store that residual in a `KnnStore` keyed to a *different* target
//!      token (here: "Rome" — a factually wrong capital for Australia).
//!   3. Re-run with a `KnnProbe` wired to that store.  The probe observes
//!      the same residual, the cosine hits 1.0, and the first emitted
//!      token is forced to "Rome" instead of "Canberra".
//!
//! This mirrors the `capture → infer` loop of the HTTP `/v1/insert` +
//! `/v1/infer` path, but talks only to the new `larql-llamacpp` API.

use larql_llamacpp::{
    GenerateConfig, LlamaPipeline, OneShot, ProbeHandler, ProbeNode,
};
use larql_vindex::patch::knn_store::KnnStore;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const PROBE_LAYER: u32 = 26;
const PROBE_TENSOR: &str = "l_out-26";
const COS_THRESHOLD: f32 = 0.75;

/// Probe that:
///  - on every decode step, reads the last-token residual at `l_out-26`,
///  - queries `KnnStore::query_top1`,
///  - if cosine > threshold, forces the next emitted token to the
///    store's `target_id`.
struct KnnProbe {
    store: Arc<Mutex<KnnStore>>,
    last_residual: Option<Vec<f32>>,
    forced: Option<i32>,
}

impl KnnProbe {
    fn new(store: Arc<Mutex<KnnStore>>) -> Self {
        Self {
            store,
            last_residual: None,
            forced: None,
        }
    }

    fn take_residual(&mut self) -> Option<Vec<f32>> {
        self.last_residual.take()
    }
}

impl ProbeHandler for KnnProbe {
    fn wants(&self, node: &ProbeNode<'_>) -> bool {
        node.name == PROBE_TENSOR
    }

    fn observe(&mut self, node: &ProbeNode<'_>, data: &[f32]) -> Option<Vec<f32>> {
        let n_embd = node.shape[0] as usize;
        let n_tokens = node.shape[1] as usize;
        if n_embd == 0 || n_tokens == 0 {
            return None;
        }
        // Last token residual: rows are tokens, cols are embedding.
        let last_off = (n_tokens - 1) * n_embd;
        let last = data[last_off..last_off + n_embd].to_vec();

        // Consult the KNN store.
        let store = self.store.lock().unwrap();
        if let Some((entry, score)) = store.query_top1(PROBE_LAYER as usize, &last) {
            if score > COS_THRESHOLD {
                self.forced = Some(entry.target_id as i32);
                eprintln!(
                    "  [knn] matched entity={:?} target={:?} cos={:.3}",
                    entry.entity, entry.target_token, score
                );
            }
        }
        drop(store);

        self.last_residual = Some(last);
        None
    }

    fn forced_token(&self) -> Option<i32> {
        self.forced
    }

    fn reset_step(&mut self) {
        self.forced = None;
    }
}

/// Captures the last-token residual at the probe layer (no override).
struct CaptureProbe {
    last_residual: Arc<Mutex<Option<Vec<f32>>>>,
}

impl ProbeHandler for CaptureProbe {
    fn wants(&self, node: &ProbeNode<'_>) -> bool {
        node.name == PROBE_TENSOR
    }
    fn observe(&mut self, node: &ProbeNode<'_>, data: &[f32]) -> Option<Vec<f32>> {
        let n_embd = node.shape[0] as usize;
        let n_tokens = node.shape[1] as usize;
        if n_embd == 0 || n_tokens == 0 {
            return None;
        }
        let last_off = (n_tokens - 1) * n_embd;
        let last = data[last_off..last_off + n_embd].to_vec();
        *self.last_residual.lock().unwrap() = Some(last);
        None
    }
}

fn main() {
    let gguf: PathBuf = std::env::var("LLAMA_GGUF")
        .unwrap_or_else(|_| "/tmp/gemma3-4b-stock-q8_0.gguf".into())
        .into();
    let prompt = "The capital of Australia is ";

    // Cheap trick to find the "Rome" token id via llama.cpp's own vocab:
    //   generate with a different prompt that ends in "Rome" and read the
    //   last prefill argmax.  For the demo, hardcode by reading it from
    //   the baseline second-token position of "... is Rome" — but simpler
    //   still: tokenize "Rome" directly.  We'll do that via a throwaway
    //   Pipeline call.
    let rome_token = {
        let mut p = LlamaPipeline::load(&gguf, 512).expect("load");
        p.token_id_of(" Rome").expect("tokenize Rome")
    };
    let canberra_token = {
        let mut p = LlamaPipeline::load(&gguf, 512).expect("load");
        p.token_id_of(" Canberra").expect("tokenize Canberra")
    };
    println!("Target token ids: Rome={} Canberra={}", rome_token, canberra_token);

    // ---- Step 1: capture residual at l_out-26 for last prompt token ----
    let residual_slot: Arc<Mutex<Option<Vec<f32>>>> = Arc::new(Mutex::new(None));
    let capture = CaptureProbe {
        last_residual: Arc::clone(&residual_slot),
    };
    let mut cap_pipe = LlamaPipeline::load_with_probe(
        &gguf,
        1024,
        Box::new(capture),
    )
    .expect("load with probe");
    let cfg1 = GenerateConfig { max_tokens: 1, stop_at_eos: false };
    let baseline = cap_pipe.generate(prompt, &cfg1).expect("baseline");
    println!();
    println!("--- BASELINE ---");
    println!("{prompt}{baseline}");
    drop(cap_pipe);

    let residual = residual_slot
        .lock()
        .unwrap()
        .take()
        .expect("residual not captured");
    println!("  captured residual: len={}", residual.len());

    // ---- Step 2: store a KNN entry mapping that residual → Rome ----
    let store = Arc::new(Mutex::new(KnnStore::default()));
    store.lock().unwrap().add(
        PROBE_LAYER as usize,
        residual,
        rome_token as u32,
        "Rome".into(),
        "Australia".into(),
        "capital".into(),
        1.0,
    );
    println!("  KnnStore: {} entries", store.lock().unwrap().len());

    // ---- Step 3: re-run with OneShot<KnnProbe> (fires exactly once) ----
    let knn_probe = OneShot::new(KnnProbe::new(Arc::clone(&store)));
    let mut override_pipe = LlamaPipeline::load_with_probe(
        &gguf,
        1024,
        Box::new(knn_probe),
    )
    .expect("load with knn");
    let cfg2 = GenerateConfig { max_tokens: 12, stop_at_eos: false };
    println!();
    println!("--- OVERRIDE (KNN → Rome, one-shot) ---");
    let overridden = override_pipe.generate(prompt, &cfg2).expect("override");
    println!("{prompt}{overridden}");

    println!();
    if overridden.trim_start().starts_with("Rome") {
        println!("VERDICT: KNN token override works end-to-end (one-shot). ✓");
    } else {
        println!(
            "VERDICT: override did NOT take effect (got {:?}).",
            overridden
        );
    }
}
