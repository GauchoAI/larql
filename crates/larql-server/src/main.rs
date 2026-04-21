//! larql-server — HTTP server for vindex knowledge queries.

mod auth;
mod cache;
mod chat_log;
mod error;
mod etag;
mod llama_probe;
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

    /// TLS certificate path for HTTPS.
    #[arg(long)]
    tls_cert: Option<PathBuf>,

    /// TLS private key path for HTTPS.
    #[arg(long)]
    tls_key: Option<PathBuf>,

}

fn load_single_vindex(path_str: &str) -> Result<LoadedModel, BoxError> {
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
    let index = VectorIndex::load_vindex(&path, &mut cb)?;
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();

    info!(
        "  Model: {} ({} layers, {} features)",
        model_name, config.num_layers, total_features
    );

    let tokenizer = load_vindex_tokenizer(&path)?;
    let (embeddings_owned, embed_scale) = load_vindex_embeddings(&path)?;
    info!("  Embeddings: {}x{}", embeddings_owned.shape()[0], embeddings_owned.shape()[1]);
    let embeddings: larql_models::WeightArray = embeddings_owned.into_shared();
    let patched = PatchedVindex::new(index);

    let probe_labels = load_probe_labels(&path);
    if !probe_labels.is_empty() {
        info!("  Labels: {} probe-confirmed", probe_labels.len());
    }

    // llama.cpp is the only inference path.  Without weights.gguf the
    // vindex still serves graph browsing endpoints (DESCRIBE, SELECT,
    // RELATIONS, EXPLAIN).
    let gguf_path = path.join("weights.gguf");
    let probe_state: Arc<std::sync::Mutex<llama_probe::ServerProbeState>> =
        Arc::new(std::sync::Mutex::new(llama_probe::ServerProbeState::default()));
    let llama = if gguf_path.exists() {
        let probe = Box::new(llama_probe::ServerProbe::new(Arc::clone(&probe_state)));
        match larql_llamacpp::LlamaPipeline::load_with_probe(&gguf_path, 8192, probe) {
            Ok(p) => {
                info!(
                    "  llama.cpp: loaded ({} layers, n_embd={}, probe armed)",
                    p.n_layer(),
                    p.n_embd()
                );
                Some(std::sync::Mutex::new(p))
            }
            Err(e) => {
                warn!("  llama.cpp: failed to load: {e}");
                None
            }
        }
    } else {
        warn!("  no weights.gguf — inference endpoints will return 503");
        None
    };

    Ok(LoadedModel {
        id,
        config,
        patched: RwLock::new(patched),
        embeddings,
        embed_scale,
        tokenizer,
        probe_labels,
        llama,
        probe_state,
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
            match load_single_vindex(&p.to_string_lossy()) {
                Ok(m) => models.push(Arc::new(m)),
                Err(e) => warn!("  Skipping {}: {}", p.display(), e),
            }
        }
    } else if let Some(ref vindex_path) = cli.vindex_path {
        let m = load_single_vindex(vindex_path)?;
        models.push(Arc::new(m));
    } else {
        return Err("must provide a vindex path or --dir".into());
    }

    if models.is_empty() {
        return Err("no vindexes loaded".into());
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
