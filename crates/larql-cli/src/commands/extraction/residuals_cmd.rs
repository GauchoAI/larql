use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, Subcommand};
use larql_core::walker::residuals::{ResidualCallbacks, ResidualCapturer, ResidualConfig};

#[derive(Args)]
pub struct ResidualsArgs {
    #[command(subcommand)]
    command: ResidualsCommand,
}

#[derive(Subcommand)]
enum ResidualsCommand {
    /// Capture residual stream vectors for entities via forward passes.
    Capture(CaptureArgs),
}

#[derive(Args)]
struct CaptureArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Comma-separated entities, or path to a text file (one per line).
    #[arg(short, long)]
    entities: String,

    /// Layer to capture residuals at. Can specify multiple times.
    #[arg(short, long)]
    layer: Option<Vec<usize>>,

    /// Capture residuals at all layers.
    #[arg(long)]
    all_layers: bool,

    /// Output NDJSON file.
    #[arg(short, long)]
    output: PathBuf,

    /// Prompt template. {entity} is replaced with the entity name.
    /// Default: bare entity name.
    #[arg(long)]
    template: Option<String>,
}

struct ProgressCallbacks;

impl ResidualCallbacks for ProgressCallbacks {
    fn on_entity_start(&mut self, entity: &str, index: usize, total: usize) {
        eprint!("  [{}/{}] {entity}...", index + 1, total);
    }

    fn on_entity_done(&mut self, _entity: &str, layers_captured: usize, elapsed_ms: f64) {
        eprintln!(" {layers_captured} layers ({:.1}s)", elapsed_ms / 1000.0);
    }
}

pub fn run(args: ResidualsArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        ResidualsCommand::Capture(capture) => run_capture(capture),
    }
}

fn run_capture(args: CaptureArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let capturer = ResidualCapturer::load(&args.model)?;
    eprintln!(
        "  {} layers, hidden_size={}",
        capturer.num_layers(),
        capturer.hidden_size()
    );

    // Parse entities: comma-separated string or file path
    let entities: Vec<String> = if std::path::Path::new(&args.entities).exists() {
        std::fs::read_to_string(&args.entities)?
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect()
    } else {
        args.entities
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };

    // Determine layers
    let layers: Vec<usize> = if args.all_layers {
        (0..capturer.num_layers()).collect()
    } else {
        args.layer.unwrap_or_else(|| vec![25])
    };

    eprintln!("  entities: {} ({} total)", entities.iter().take(5).cloned().collect::<Vec<_>>().join(", "), entities.len());
    eprintln!("  layers: {:?}", layers);
    if let Some(ref tmpl) = args.template {
        eprintln!("  template: {tmpl}");
    }
    eprintln!();

    let config = ResidualConfig {
        layers,
        prompt_template: args.template,
    };

    let mut callbacks = ProgressCallbacks;
    let start = Instant::now();

    let count = capturer.capture(&entities, &config, &args.output, &mut callbacks)?;

    let elapsed = start.elapsed();
    let size = std::fs::metadata(&args.output)?.len();

    eprintln!("\nCompleted in {:.1}s", elapsed.as_secs_f64());
    eprintln!("  Residuals captured: {count}");
    eprintln!("  Entities: {}", entities.len());
    eprintln!(
        "  Output: {} ({:.1} MB)",
        args.output.display(),
        size as f64 / 1024.0 / 1024.0
    );

    Ok(())
}
