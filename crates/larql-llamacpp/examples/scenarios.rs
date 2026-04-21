//! Golden-file scenario runner.  Each scenario is a directory:
//!
//!   scenarios/<name>/
//!     input.txt    (required)  — stdin lines fed to the TUI; final
//!                                line should be `/quit`.
//!     setup.sh     (optional)  — bash; runs before the TUI launches.
//!                                $LARQL_SERVER is exported.
//!     post.sh      (optional)  — bash; runs after the TUI exits.
//!                                Output is appended to actual.txt.
//!     teardown.sh  (optional)  — bash; cleanup, output discarded.
//!     golden.txt   (optional)  — last accepted transcript.  If
//!                                missing, the first run records it.
//!     actual.txt   (generated) — current run's transcript.  Read this
//!                                to verify the agent did what you
//!                                expected (annotated, used a tool,
//!                                wrote a file, etc.).
//!
//! The runner is intentionally non-asserting: it produces the
//! transcript and reports whether it matches `golden.txt`.  You read
//! `actual.txt` and decide if the run is good — `cp actual.txt
//! golden.txt` to accept the new baseline.
//!
//! Usage:
//!   LARQL_SERVER=http://localhost:3000 \
//!     cargo run --release -p larql-llamacpp --example scenarios -- \
//!       /path/to/scenarios

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const TUI_DEFAULT: &str = "/Users/miguel_lemos/Desktop/llm-as-a-database/larql-stable/target/release/larql";
const SERVER_DEFAULT: &str = "http://localhost:3000";

struct Outcome {
    name: String,
    duration_ms: u128,
    /// "RECORDED" (no golden existed; we just wrote one),
    /// "MATCH"    (transcript identical to golden),
    /// "DIFFER"   (transcript differs — view actual.txt).
    status: &'static str,
    note: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("scenarios"));
    let server = std::env::var("LARQL_SERVER").unwrap_or_else(|_| SERVER_DEFAULT.into());
    let tui = std::env::var("LARQL_TUI").unwrap_or_else(|_| TUI_DEFAULT.into());

    eprintln!("[scenarios] dir    = {}", dir.display());
    eprintln!("[scenarios] server = {server}");
    eprintln!("[scenarios] tui    = {tui}");
    eprintln!();

    if !Path::new(&tui).exists() {
        eprintln!("[scenarios] TUI binary missing: {tui}");
        eprintln!("[scenarios] build:  cargo build --release \\");
        eprintln!("                    --manifest-path /Users/miguel_lemos/Desktop/llm-as-a-database/larql-stable/Cargo.toml \\");
        eprintln!("                    -p larql-tui");
        std::process::exit(2);
    }
    if !server_alive(&server) {
        eprintln!("[scenarios] cannot reach {server}/v1/health — is the server running?");
        std::process::exit(2);
    }

    let mut scenario_dirs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| {
            eprintln!("[scenarios] {} unreadable: {e}", dir.display());
            std::process::exit(2);
        })
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("input.txt").exists())
        .collect();
    scenario_dirs.sort();
    if scenario_dirs.is_empty() {
        eprintln!("[scenarios] no scenarios with input.txt under {}", dir.display());
        std::process::exit(2);
    }

    let mut outcomes: Vec<Outcome> = Vec::new();
    for sd in &scenario_dirs {
        let name = sd.file_name().unwrap().to_string_lossy().into_owned();
        eprintln!("─── {} ───", name);
        // Hermetic per-scenario: wipe the live KNN store so state from
        // an earlier scenario (or an old manual test) can't leak in.
        let _ = Command::new("curl")
            .args([
                "-s", "-o", "/dev/null",
                "-X", "POST",
                "--max-time", "5",
                &format!("{server}/v1/reset"),
            ])
            .status();
        let outcome = run_one(sd, &server, &tui);
        eprintln!("  {}  {}  ({} ms)", outcome.status, outcome.note, outcome.duration_ms);
        eprintln!();
        outcomes.push(outcome);
    }

    eprintln!("════════════════════════════════════");
    eprintln!("  Summary ({} scenarios)", outcomes.len());
    eprintln!("════════════════════════════════════");
    for o in &outcomes {
        eprintln!("  {:8} {}  ({} ms)", o.status, o.name, o.duration_ms);
    }
    eprintln!();
    eprintln!("  Read actual.txt in each scenario dir to verify the model");
    eprintln!("  did the right thing.  To bless a transcript as the new");
    eprintln!("  baseline:");
    eprintln!("    cp scenarios/<name>/actual.txt scenarios/<name>/golden.txt");
}

fn server_alive(server: &str) -> bool {
    Command::new("curl")
        .args([
            "-s", "-o", "/dev/null", "-w", "%{http_code}",
            "--max-time", "3",
            &format!("{server}/v1/health"),
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "200")
        .unwrap_or(false)
}

fn run_one(scenario_dir: &Path, server: &str, tui: &str) -> Outcome {
    let started = Instant::now();
    let name = scenario_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    // ── setup ──
    if let Some(out) = run_script(scenario_dir, "setup.sh", server) {
        if !out.trim().is_empty() {
            eprintln!("{}", indent(&out, "  setup: "));
        }
    }

    // ── drive the TUI ──
    let input = std::fs::read_to_string(scenario_dir.join("input.txt"))
        .unwrap_or_else(|_| "/quit\n".into());
    let transcript = run_tui_headless(tui, server, &input);

    // ── post ── (output appended to transcript so it's part of the golden)
    let mut full = transcript;
    if let Some(out) = run_script(scenario_dir, "post.sh", server) {
        full.push_str("\n========= post =========\n");
        full.push_str(&out);
    }

    let actual_path = scenario_dir.join("actual.txt");
    let golden_path = scenario_dir.join("golden.txt");
    if let Err(e) = std::fs::write(&actual_path, &full) {
        return Outcome {
            name,
            duration_ms: started.elapsed().as_millis(),
            status: "ERR",
            note: format!("write actual: {e}"),
        };
    }

    // ── teardown ── (always; output suppressed)
    let _ = run_script(scenario_dir, "teardown.sh", server);

    // ── golden compare ──
    let outcome = if !golden_path.exists() {
        if let Err(e) = std::fs::copy(&actual_path, &golden_path) {
            Outcome {
                name: name.clone(),
                duration_ms: started.elapsed().as_millis(),
                status: "ERR",
                note: format!("seed golden: {e}"),
            }
        } else {
            Outcome {
                name: name.clone(),
                duration_ms: started.elapsed().as_millis(),
                status: "RECORDED",
                note: "first run — golden.txt seeded; review it manually".into(),
            }
        }
    } else {
        let golden = std::fs::read_to_string(&golden_path).unwrap_or_default();
        if golden == full {
            Outcome {
                name: name.clone(),
                duration_ms: started.elapsed().as_millis(),
                status: "MATCH",
                note: "transcript identical to golden".into(),
            }
        } else {
            let (gl, al) = (golden.lines().count(), full.lines().count());
            Outcome {
                name: name.clone(),
                duration_ms: started.elapsed().as_millis(),
                status: "DIFFER",
                note: format!(
                    "transcript drifted ({} → {} lines); diff golden actual; cp actual golden to bless",
                    gl, al
                ),
            }
        }
    };
    outcome
}

fn run_script(dir: &Path, name: &str, server: &str) -> Option<String> {
    let path = dir.join(name);
    if !path.exists() {
        return None;
    }
    let out = Command::new("bash")
        .arg(&path)
        .env("LARQL_SERVER", server)
        .output();
    match out {
        Ok(o) => {
            let mut buf = String::from_utf8_lossy(&o.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&o.stderr);
            if !stderr.trim().is_empty() {
                buf.push_str(&format!("\n[stderr]\n{stderr}"));
            }
            Some(buf)
        }
        Err(e) => Some(format!("[script {name} failed to launch: {e}]")),
    }
}

fn run_tui_headless(tui: &str, server: &str, stdin_text: &str) -> String {
    let mut child = match Command::new(tui)
        .arg("--headless")
        .env("LARQL_SERVER", server)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return format!("[scenarios] spawn TUI failed: {e}"),
    };

    if let Some(stdin) = child.stdin.as_mut() {
        let _ = stdin.write_all(stdin_text.as_bytes());
    }

    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                break;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(_) => break,
        }
    }

    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut s) = child.stdout.take() {
        let _ = std::io::Read::read_to_string(&mut s, &mut stdout);
    }
    if let Some(mut s) = child.stderr.take() {
        let _ = std::io::Read::read_to_string(&mut s, &mut stderr);
    }
    // Combined transcript: model stream + TUI tool markers.
    format!("========= transcript =========\n{stdout}========= tui markers =========\n{stderr}")
}

fn indent(s: &str, prefix: &str) -> String {
    s.lines()
        .map(|l| format!("{prefix}{l}"))
        .collect::<Vec<_>>()
        .join("\n")
}
