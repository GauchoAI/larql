//! larql-server — HTTP server for vindex knowledge queries.

mod auth;
mod cache;
mod error;
mod etag;
mod grpc;
mod ratelimit;
mod routes;
mod session;
mod state;

use std::path::PathBuf;
use std::sync::Arc;

use axum::middleware;
use clap::Parser;
use tokio::sync::RwLock;
use tracing::{info, warn};

use larql_vindex::{
    PatchedVindex, SilentLoadCallbacks, VectorIndex,
    load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
};

use cache::DescribeCache;
use session::SessionManager;
use state::{AppState, LoadedModel, model_id_from_name, load_probe_labels};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Parser)]
#[command(
    name = "larql-server",
    version,
    about = "HTTP server for vindex knowledge queries and inference"
)]
struct Cli {
    /// Path to a .vindex directory (or hf:// path).
    #[arg(value_name = "VINDEX_PATH")]
    vindex_path: Option<String>,

    /// Serve all .vindex directories in this folder.
    #[arg(long)]
    dir: Option<PathBuf>,

    /// Listen port.
    #[arg(long, default_value = "3000")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Disable INFER endpoint (browse-only, reduces memory).
    #[arg(long)]
    no_infer: bool,

    /// Enable CORS for browser access.
    #[arg(long)]
    cors: bool,

    /// API key for authentication (clients send Authorization: Bearer <key>).
    #[arg(long)]
    api_key: Option<String>,

    /// Rate limit per IP (e.g., "100/min", "10/sec").
    #[arg(long)]
    rate_limit: Option<String>,

    /// Max concurrent requests.
    #[arg(long, default_value = "100")]
    max_concurrent: usize,

    /// Cache TTL for DESCRIBE results in seconds (0 = disabled).
    #[arg(long, default_value = "0")]
    cache_ttl: u64,

    /// Logging level.
    #[arg(long, default_value = "info")]
    log_level: String,

    /// gRPC port (enables gRPC server alongside HTTP).
    #[arg(long)]
    grpc_port: Option<u16>,

    /// TLS certificate path for HTTPS.
    #[arg(long)]
    tls_cert: Option<PathBuf>,

    /// TLS private key path for HTTPS.
    #[arg(long)]
    tls_key: Option<PathBuf>,

    /// Skip Metal shader warmup at startup. Without warmup the first request
    /// of each prompt length pays 10-20s of shader compilation; warmup moves
    /// that cost out of the interactive loop. Default: warmup enabled.
    #[arg(long)]
    no_warmup: bool,

    /// DEPRECATED: prefer dropping a `weights.gguf` into the vindex dir.
    /// Walk-only drops FFN weights and routes `/v1/infer mode=fast` through
    /// a sparse Q4_0 walk: ~11 GB RSS, ~2× slower decode. The GGUF path
    /// achieves ~5 GB RSS at full speed (see findings.md 2026-04-19/20).
    /// Kept for vindexes that lack a GGUF and still have `interleaved_q4.bin`.
    #[arg(long)]
    walk_only: bool,
}

fn load_single_vindex(path_str: &str, no_infer: bool, walk_only: bool) -> Result<LoadedModel, BoxError> {
    let path = if larql_vindex::is_hf_path(path_str) {
        info!("Resolving HuggingFace path: {}", path_str);
        larql_vindex::resolve_hf_vindex(path_str)?
    } else {
        PathBuf::from(path_str)
    };

    info!("Loading: {}", path.display());

    let config = load_vindex_config(&path)?;
    let model_name = config.model.clone();
    let id = model_id_from_name(&model_name);

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&path, &mut cb)?;
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();

    let has_weights = config.has_model_weights
        || config.extract_level == larql_vindex::ExtractLevel::Inference
        || config.extract_level == larql_vindex::ExtractLevel::All;

    info!(
        "  Model: {} ({} layers, {} features)",
        model_name, config.num_layers, total_features
    );

    // Load mmap'd feature-major vectors for walk FFN optimization
    match index.load_down_features(&path) {
        Ok(()) => info!("  Down features: loaded (mmap walk enabled)"),
        Err(_) => info!("  Down features: not available"),
    }
    if let Ok(()) = index.load_up_features(&path) { info!("  Up features: loaded (full mmap FFN)") }
    // Tiers the Metal fast path needs: lm_head (for finalize_logits), q4k/q8
    // attn weights, interleaved Q4_K FFN mmaps. Matches bench_interactive.
    let _ = index.load_lm_head(&path);
    let _ = index.load_lm_head_q4(&path);
    let _ = index.load_attn_q4k(&path);
    let _ = index.load_attn_q8(&path);
    // Prefer interleaved_q4k_real.bin (true Q4_K, validated cos=0.9994 on
    // Gemma 3 4B). Fall back to Q4_0 if the preferred file isn't present.
    if index.load_interleaved_q4k_real(&path).is_ok() {
        info!("  FFN: interleaved_q4k_real.bin (GPU decode default)");
    } else {
        let _ = index.load_interleaved_q4(&path);
    }
    index.warmup();
    info!("  Warmup: done");

    // Release pages for mmaps not needed during Q4_K inference.
    // The mmaps stay open (capability detection still works) but the OS
    // can reclaim the physical memory. Saves ~9.4 GB of resident pages.
    if path.join("interleaved_q4k_real.bin").exists() {
        index.advise_dontneed_unused();
    }

    let (embeddings_owned, embed_scale) = load_vindex_embeddings(&path)?;
    info!("  Embeddings: {}x{}", embeddings_owned.shape()[0], embeddings_owned.shape()[1]);
    let embeddings: larql_models::WeightArray = embeddings_owned.into_shared();

    let tokenizer = load_vindex_tokenizer(&path)?;
    let patched = PatchedVindex::new(index);

    let probe_labels = load_probe_labels(&path);
    if !probe_labels.is_empty() {
        info!("  Labels: {} probe-confirmed", probe_labels.len());
    }

    if no_infer {
        info!("  Infer: disabled (--no-infer)");
    } else if has_weights {
        info!("  Infer: available (weights detected, will lazy-load on first request)");
    } else {
        info!("  Infer: not available (no model weights in vindex)");
    }

    // Detect optional GGUF weight source. If present, fast-path inference
    // reads weights from it instead of the (possibly stale) Q4_K binaries.
    let gguf = {
        let p = path.join("weights.gguf");
        if p.exists() {
            match larql_inference::gguf_pipeline::GgufPipeline::open(&p) {
                Ok(g) => {
                    info!("  GGUF: loaded weights.gguf ({} layers, vocab={})",
                        g.num_layers(), g.lm_head_vocab);
                    if walk_only {
                        info!("  NOTE: --walk-only is redundant when weights.gguf is present \
                              (GGUF achieves smaller RSS at full decode speed)");
                    }
                    Some(Arc::new(g))
                }
                Err(e) => {
                    info!("  GGUF: failed to load weights.gguf: {e}");
                    None
                }
            }
        } else {
            None
        }
    };

    Ok(LoadedModel {
        id,
        path,
        config,
        patched: RwLock::new(patched),
        embeddings,
        embed_scale,
        tokenizer,
        infer_disabled: no_infer,
        weights: std::sync::OnceLock::new(),
        backend: std::sync::OnceLock::new(),
        inference_lock: std::sync::Mutex::new(()),
        probe_labels,
        walk_only,
        gguf,
    })
}

fn discover_vindexes(dir: &PathBuf) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join("index.json").exists() {
                paths.push(p);
            }
        }
    }
    paths.sort();
    paths
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // Accept both `larql-server <path>` and `larql-server serve <path>`.
    let args: Vec<String> = std::env::args().collect();
    let filtered: Vec<String> = if args.len() > 1 && args[1] == "serve" {
        std::iter::once(args[0].clone()).chain(args[2..].iter().cloned()).collect()
    } else {
        args
    };
    let cli = Cli::parse_from(filtered);

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .init();

    info!("larql-server v{}", env!("CARGO_PKG_VERSION"));

    let mut models: Vec<Arc<LoadedModel>> = Vec::new();

    if let Some(ref dir) = cli.dir {
        let paths = discover_vindexes(dir);
        if paths.is_empty() {
            return Err(format!("no .vindex directories found in {}", dir.display()).into());
        }
        info!("Found {} vindexes in {}", paths.len(), dir.display());
        for p in &paths {
            match load_single_vindex(&p.to_string_lossy(), cli.no_infer, cli.walk_only) {
                Ok(m) => models.push(Arc::new(m)),
                Err(e) => warn!("  Skipping {}: {}", p.display(), e),
            }
        }
    } else if let Some(ref vindex_path) = cli.vindex_path {
        let m = load_single_vindex(vindex_path, cli.no_infer, cli.walk_only)?;
        models.push(Arc::new(m));
    } else {
        return Err("must provide a vindex path or --dir".into());
    }

    if models.is_empty() {
        return Err("no vindexes loaded".into());
    }

    // Metal shader warmup: compile pipelines for common seq_lens up front
    // so clients don't pay 10-20s for the first request of each length.
    // Runs on spawn_blocking before the HTTP listener binds — we cannot
    // call `RwLock::blocking_read` directly from the async runtime.
    if !cli.no_warmup {
        for m in &models {
            if m.infer_disabled || !m.config.has_model_weights {
                continue;
            }
            let m_cl = Arc::clone(m);
            let id = m.id.clone();
            info!("Warming up Metal shaders for model '{}' ...", id);
            let t0 = std::time::Instant::now();
            let report = tokio::task::spawn_blocking(move || -> Result<Vec<(usize, f64)>, String> {
                let tw = std::time::Instant::now();
                let weights = m_cl.get_or_load_weights()?;
                tracing::info!("  weights loaded in {:.1}s", tw.elapsed().as_secs_f64());
                let backend = m_cl.get_or_init_backend();
                let bos = m_cl.tokenizer.encode("", true).ok()
                    .and_then(|e| e.get_ids().first().copied())
                    .unwrap_or(1);
                let patched = m_cl.patched.blocking_read();
                // Match the hot-path FFN at warmup time. When walk_only is set
                // we must warm the Q4_0 walk kernel, not the dense matmul (the
                // dense path would panic on dropped FFN weights anyway).
                let walk_ffn_warmup = if m_cl.walk_only {
                    Some(larql_inference::WalkFfn::new_with_backend(
                        weights, patched.base(), 1024, &**backend,
                    ))
                } else { None };
                let ffn_override: Option<&dyn larql_inference::ffn::FfnBackend> =
                    walk_ffn_warmup.as_ref().map(|w| w as &dyn larql_inference::ffn::FfnBackend);
                let mut timings = Vec::new();
                for &n in &[1usize, 4, 8, 16, 32] {
                    let ids: Vec<u32> = std::iter::repeat(bos).take(n).collect();
                    // Empty CachedLayerGraph — avoids invoking the FFN (which
                    // would panic in walk_only mode since FFN tensors were
                    // dropped after load).
                    let cache = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
                    backend.reset_kv_cache();
                    let t = std::time::Instant::now();
                    let _ = larql_inference::predict_honest_with_knn_ffn(
                        weights, &m_cl.tokenizer, &ids, 1, patched.base(),
                        &**backend, &cache, 0..weights.num_layers, None, ffn_override,
                    );
                    let s = t.elapsed().as_secs_f64();
                    tracing::info!("  seq_len={} {:.1}s", n, s);
                    timings.push((n, s));
                }
                backend.reset_kv_cache();
                Ok(timings)
            }).await.map_err(|e| e.to_string())?;
            match report {
                Ok(timings) => {
                    for (n, s) in timings { info!("  seq_len={} {:.1}s", n, s); }
                    info!("Warmup done for '{}' in {:.1}s", id, t0.elapsed().as_secs_f64());
                }
                Err(e) => warn!("  warmup failed for '{}': {}", id, e),
            }
        }
    }

    // Parse rate limiter if specified.
    let rate_limiter = cli.rate_limit.as_ref().and_then(|spec| {
        match ratelimit::RateLimiter::parse(spec) {
            Some(rl) => {
                info!("Rate limit: {}", spec);
                Some(Arc::new(rl))
            }
            None => {
                warn!("Invalid rate limit format: {} (expected e.g. '100/min')", spec);
                None
            }
        }
    });

    let state = Arc::new(AppState {
        models: models.clone(),
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
        api_key: cli.api_key.clone(),
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(cli.cache_ttl),
        rag_store: routes::rag::RagStore::new(),
        kv_rag_store: routes::kv_rag::KvRagStore::new(),
        vec_store: routes::vec_inject::VecStore::new(),
        kv_cache_store: routes::kv_cache::KvCacheStore::new(),
    });

    if cli.cache_ttl > 0 {
        info!("DESCRIBE cache: {}s TTL", cli.cache_ttl);
    }

    let is_multi = state.is_multi_model();
    let mut app = if is_multi {
        info!("Multi-model mode ({} models)", state.models.len());
        for m in &state.models {
            info!("  /v1/{}/...", m.id);
        }
        routes::multi_model_router(Arc::clone(&state))
    } else {
        let m = &models[0];
        info!("Single-model mode: {}", m.config.model);
        routes::single_model_router(Arc::clone(&state))
    };

    // Rate limiting middleware.
    if let Some(ref rl) = rate_limiter {
        app = app.layer(middleware::from_fn_with_state(
            Arc::clone(rl),
            ratelimit::rate_limit_middleware,
        ));
    }

    // Auth middleware (if --api-key set).
    if cli.api_key.is_some() {
        app = app.layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            auth::auth_middleware,
        ));
        info!("Auth: API key required");
    }

    // CORS middleware.
    if cli.cors {
        use tower_http::cors::CorsLayer;
        app = app.layer(CorsLayer::permissive());
        info!("CORS: enabled");
    }

    // Concurrency limit.
    app = app.layer(tower::limit::ConcurrencyLimitLayer::new(cli.max_concurrent));
    info!("Max concurrent: {}", cli.max_concurrent);

    // Trace middleware.
    app = app.layer(tower_http::trace::TraceLayer::new_for_http());

    // gRPC server (if --grpc-port set).
    if let Some(grpc_port) = cli.grpc_port {
        let grpc_addr = format!("{}:{}", cli.host, grpc_port).parse()?;
        let grpc_state = Arc::clone(&state);
        info!("gRPC: listening on {}", grpc_addr);
        tokio::spawn(async move {
            let svc = grpc::VindexGrpcService { state: grpc_state };
            if let Err(e) = tonic::transport::Server::builder()
                .add_service(grpc::proto::vindex_service_server::VindexServiceServer::new(svc))
                .serve(grpc_addr)
                .await
            {
                tracing::error!("gRPC server error: {}", e);
            }
        });
    }

    let addr = format!("{}:{}", cli.host, cli.port);

    // TLS or plain HTTP.
    if let (Some(cert_path), Some(key_path)) = (&cli.tls_cert, &cli.tls_key) {
        info!("TLS: enabled ({}, {})", cert_path.display(), key_path.display());
        info!("Listening: https://{}", addr);

        let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(
            cert_path, key_path,
        )
        .await?;

        axum_server::bind_rustls(addr.parse()?, tls_config)
            .serve(app.into_make_service())
            .await?;
    } else {
        info!("Listening: http://{}", addr);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
    }

    Ok(())
}
