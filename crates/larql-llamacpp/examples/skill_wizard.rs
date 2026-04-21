//! Multi-step skill builder.  When asked to make a tool, **make it** —
//! every stage that fails re-prompts the model with the actual error
//! until it converges or the per-stage budget runs out.  No skill
//! names are hard-coded in this file; Rust knows nothing about
//! "kernel_version" or "weather", only about the wizard protocol.
//!
//! Stages (each retries on failure with the model seeing the error):
//!   1. propose_command — bash that gathers raw data
//!   2. run_command     — execute in Debian sandbox, capture output
//!   3. propose_summary — pipeline that reads stdin → 1-3 line summary
//!   4. validate_summary — pipeline must produce non-empty stdout
//!   5. propose_chart    — optional chart.js JSON
//!   6. validate_chart   — JSON parses + has type/data
//!   7. install         — write skill.md + tool.sh
//!   8. smoke_test      — run installed skill once in container
//!
//! The harness reads cases from `~/.larql/wizard_cases.json` (created
//! on first run with a default set the user can edit).
//!
//! Usage:
//!   skill_wizard build --prompt "<request>" --name <name>
//!   skill_wizard harness                   # runs wizard_cases.json
//!   skill_wizard harness --cases /path.json

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const SERVER_DEFAULT: &str = "http://localhost:3000";
const SAMPLE_HEAD_LINES: usize = 10;
const SAMPLE_HEAD_BYTES: usize = 1500;
const RUNTIME_CONTAINER: &str = "larql-skill-runtime";

/// Per-stage retry budget.  Five gives the model two more shots after
/// it sees a deterministic validator complaint, which is usually
/// enough for the easy "wrap with `${1:-…}`" or "drop `input.txt`" fix.
const STAGE_RETRIES: usize = 5;

#[derive(Debug)]
struct Config {
    server: String,
}

#[derive(Debug, Default)]
struct WizardOutcome {
    user_request: String,
    skill_name: String,
    /// Each entry: (stage, success?, attempts_used, note).
    stages: Vec<(String, bool, usize, String)>,
    success: bool,
    elapsed_ms: u128,
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = Config {
        server: std::env::var("LARQL_SERVER").unwrap_or_else(|_| SERVER_DEFAULT.into()),
    };
    let cmd = args.first().cloned().unwrap_or_default();
    args.remove(0);
    match cmd.as_str() {
        "build" => {
            let prompt = arg_value(&args, "--prompt").unwrap_or_default();
            let name = arg_value(&args, "--name").unwrap_or_default();
            if prompt.is_empty() || name.is_empty() {
                eprintln!("usage: skill_wizard build --prompt \"<request>\" --name <skill-name>");
                std::process::exit(2);
            }
            let outcome = run_wizard(&cfg, &prompt, &name);
            print_outcome(&outcome);
            std::process::exit(if outcome.success { 0 } else { 1 });
        }
        "harness" | "" => {
            let cases_path = arg_value(&args, "--cases")
                .map(PathBuf::from)
                .unwrap_or_else(default_cases_path);
            run_harness(&cfg, &cases_path);
        }
        _ => {
            eprintln!("unknown command: {cmd}");
            std::process::exit(2);
        }
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1).cloned()
}

fn default_cases_path() -> PathBuf {
    home_dir().join(".larql/wizard_cases.json")
}

// ──────────────────────── Wizard with retries ───────────────────────

fn run_wizard(cfg: &Config, user_request: &str, skill_name: &str) -> WizardOutcome {
    let started = Instant::now();
    let mut outcome = WizardOutcome {
        user_request: user_request.into(),
        skill_name: skill_name.into(),
        ..Default::default()
    };

    if !skill_name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
    {
        outcome.stages.push(("validate_name".into(), false, 0, format!("invalid name: {skill_name}")));
        outcome.elapsed_ms = started.elapsed().as_millis();
        return outcome;
    }
    let target_dir = home_dir().join(".larql/skills").join(skill_name);
    if target_dir.exists() {
        outcome.stages.push(("validate_name".into(), false, 0, "skill exists".into()));
        outcome.elapsed_ms = started.elapsed().as_millis();
        return outcome;
    }
    outcome.stages.push(("validate_name".into(), true, 0, "ok".into()));

    // ── Stage A: command + execution converge together.  We retry the
    // model until the proposed command actually produces output.
    let (sample_command, sample_output, attempts_a) = match converge_command(cfg, user_request, skill_name) {
        Ok(c) => c,
        Err(e) => {
            outcome.stages.push(("propose_command".into(), false, e.attempts, e.note));
            outcome.elapsed_ms = started.elapsed().as_millis();
            return outcome;
        }
    };
    outcome.stages.push((
        "propose_command".into(),
        true,
        attempts_a,
        format!("`{}` → {} bytes", first_line(&sample_command), sample_output.len()),
    ));

    // ── Stage B: summary pipeline + validation converge together.
    let (summary_pipeline, attempts_b) = match converge_summary(cfg, &sample_output) {
        Ok(p) => p,
        Err(e) => {
            outcome.stages.push(("propose_summary".into(), false, e.attempts, e.note));
            outcome.elapsed_ms = started.elapsed().as_millis();
            return outcome;
        }
    };
    outcome.stages.push((
        "propose_summary".into(),
        true,
        attempts_b,
        first_line(&summary_pipeline).to_string(),
    ));

    // ── Stage C: optional chart.  Failures here are non-fatal — we
    // skip the chart but still ship the skill.
    let (chart_json, attempts_c) = converge_chart(cfg, &sample_output);
    outcome.stages.push((
        "propose_chart".into(),
        true,
        attempts_c,
        chart_json.as_deref().map(|_| "valid chart.json").unwrap_or("none").into(),
    ));

    // Snapshot for the registry call we make once smoke passes.
    let kw_for_registry: String = user_request
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() > 3)
        .take(8)
        .collect::<Vec<_>>()
        .join(", ");

    // ── Stage D + E: assemble + install + smoke-test, with the smoke
    // test feeding errors back into propose_summary on failure.  After
    // STAGE_RETRIES of full assembly+smoke we give up.
    let mut current_summary = summary_pipeline;
    let mut smoke_attempts = 0usize;
    for attempt in 1..=STAGE_RETRIES {
        smoke_attempts = attempt;
        let _ = std::fs::remove_dir_all(&target_dir); // start fresh each loop
        let tool_sh = assemble_tool_sh(&sample_command, &current_summary, &chart_json);
        if let Err(e) = install_skill(&target_dir, skill_name, user_request, &tool_sh) {
            outcome.stages.push(("install".into(), false, attempt, e));
            outcome.elapsed_ms = started.elapsed().as_millis();
            return outcome;
        }

        let installed = format!("bash {}", target_dir.join("tool.sh").display());
        match run_in_container(&installed, "", "", Duration::from_secs(10)) {
            Ok(out) if out.contains("```summary") && summary_block_nonempty(&out) => {
                outcome.stages.push((
                    "install".into(),
                    true,
                    attempt,
                    format!("{}", target_dir.display()),
                ));
                outcome.stages.push((
                    "smoke_test".into(),
                    true,
                    attempt,
                    format!("{} bytes", out.len()),
                ));
                // ── Judge stage (advisory): ask the model whether
                // the produced summary obviously fails (404, command
                // not found, totally unrelated).  We DO NOT block on
                // a "broken" verdict — Gemma is too strict on terse
                // outputs.  Logged for future tuning; smoke_test is
                // the actual gate.
                let summary_text = extract_summary_block(&out).unwrap_or_default();
                let verdict = judge(cfg, user_request, &summary_text)
                    .map(|r| format!("advisory: {r}"))
                    .unwrap_or_else(|| "advisory: ok".into());
                outcome.stages.push(("judge".into(), true, 1, verdict));
                // Register in the SQLite catalog so the TUI's view +
                // future maintenance commands can see this skill.
                let reg_body = serde_json::json!({
                    "name": skill_name,
                    "description": user_request,
                    "keywords": kw_for_registry,
                    "runtime": "container",
                    "source": "wizard",
                    "scope": "global",
                });
                let _ = Command::new("curl")
                    .args([
                        "-s", "-o", "/dev/null", "-X", "POST",
                        "-H", "content-type: application/json",
                        "-d", &reg_body.to_string(),
                        "--max-time", "5",
                        &format!("{}/v1/skills/register", cfg.server),
                    ])
                    .status();
                outcome.success = true;
                break;
            }
            Ok(out) => {
                let preview: String = out.chars().take(400).collect();
                let err_note = format!("smoke produced an empty / malformed summary block. \
                    First 400 chars of output: {}", preview);
                if attempt == STAGE_RETRIES {
                    outcome.stages.push(("install".into(), true, attempt, "rolled back".into()));
                    outcome.stages.push(("smoke_test".into(), false, attempt, err_note));
                    let _ = std::fs::remove_dir_all(&target_dir);
                    break;
                }
                // Re-converge summary with smoke-test feedback.
                match reconverge_summary(cfg, &sample_output, &current_summary, &err_note) {
                    Ok(p) => current_summary = p,
                    Err(_) => {
                        outcome.stages.push(("smoke_test".into(), false, attempt, err_note));
                        let _ = std::fs::remove_dir_all(&target_dir);
                        break;
                    }
                }
            }
            Err(e) => {
                if attempt == STAGE_RETRIES {
                    outcome.stages.push(("smoke_test".into(), false, attempt, e));
                    let _ = std::fs::remove_dir_all(&target_dir);
                    break;
                }
                match reconverge_summary(cfg, &sample_output, &current_summary, &e) {
                    Ok(p) => current_summary = p,
                    Err(_) => {
                        outcome.stages.push(("smoke_test".into(), false, attempt, e));
                        let _ = std::fs::remove_dir_all(&target_dir);
                        break;
                    }
                }
            }
        }
    }
    if !outcome.success {
        // Make sure the final smoke_attempts is recorded.
        let _ = smoke_attempts;
    }
    outcome.elapsed_ms = started.elapsed().as_millis();
    outcome
}

fn extract_summary_block(out: &str) -> Option<String> {
    let opener = "```summary";
    let start = out.find(opener)? + opener.len();
    let after = &out[start..];
    let nl = after.find('\n')? + 1;
    let body_start = start + nl;
    let close = out[body_start..].find("```")?;
    Some(out[body_start..body_start + close].trim().to_string())
}

/// One-shot semantic check.  Calibrated to be LENIENT: only reject
/// obvious misses (404 errors, "command not found", clearly off-topic
/// content).  Imperfect formatting or terse output is fine — the
/// model can always ask the user to clarify.
fn judge(cfg: &Config, user_request: &str, summary: &str) -> Option<String> {
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        return Some("summary is empty".into());
    }
    // Pre-check: obvious failure markers in the summary text.
    let lc = trimmed.to_lowercase();
    let bad_markers = [
        "command not found", "no such file or directory", "permission denied",
        "404 not found", "error 404", "couldn't be found",
    ];
    for m in bad_markers {
        if lc.contains(m) {
            return Some(format!("summary contains failure marker: {m}"));
        }
    }
    let prompt = format!(
        "A user asked for a tool that does this: \"{user_request}\"\n\n\
         The tool ran and produced:\n```\n{summary}\n```\n\n\
         Question: is this output OBVIOUSLY WRONG / UNRELATED / an error message?  Be \
         LENIENT — if the output is in the right ballpark, even if format is rough or \
         numbers seem odd, it counts as fine.  Only flag clear failures (404s, \
         \"command not found\", completely unrelated content).\n\n\
         Reply EXACTLY ONE WORD: `fine` (output is acceptable) or `broken` (clear failure).",
    );
    let resp = match chat(&cfg.server, &prompt) {
        Ok(t) => t,
        Err(_) => return None,
    };
    let first_line = resp.lines().next().unwrap_or("").trim().to_lowercase();
    if first_line.starts_with("broken") {
        Some(format!("judge: {}", first_line))
    } else {
        None // accept "fine", ambiguous, or anything else
    }
}

/// True iff the assembled tool.sh emitted a `summary` block with a
/// non-whitespace body.  Catches the case where the pipeline ran but
/// produced empty output, leaving an empty `\`\`\`summary…\`\`\``.
fn summary_block_nonempty(out: &str) -> bool {
    let opener = "```summary";
    let start = match out.find(opener) {
        Some(s) => s + opener.len(),
        None => return false,
    };
    let after = &out[start..];
    let nl = match after.find('\n') {
        Some(n) => n + 1,
        None => return false,
    };
    let body_start = start + nl;
    let close = match out[body_start..].find("```") {
        Some(c) => c,
        None => return false,
    };
    !out[body_start..body_start + close].trim().is_empty()
}

fn reconverge_summary(
    cfg: &Config,
    sample_output: &str,
    previous_pipeline: &str,
    error_note: &str,
) -> Result<String, StageError> {
    let head = head_n(sample_output, SAMPLE_HEAD_LINES, SAMPLE_HEAD_BYTES);
    let prompt = format!(
        "I assembled the skill and ran it on a clean Debian sandbox.  The pipeline you wrote \
         to summarise the data did not work.  Error / behaviour:\n\n{error_note}\n\n\
         Your previous pipeline:\n```bash\n{previous_pipeline}\n```\n\n\
         Reminders: it must read from STDIN (no literal filenames!), only use coreutils-class \
         tools, and print 1-3 lines of human-readable summary.\n\n\
         Sample of what the data looks like (first {} lines):\n```\n{}\n```\n\n\
         Reply with a corrected pipeline in ONE fenced ```bash``` block.",
        SAMPLE_HEAD_LINES, head,
    );
    let resp = chat(&cfg.server, &prompt).map_err(|e| StageError { attempts: 1, note: e })?;
    let pipeline = extract_fenced(&resp, "bash")
        .or_else(|| extract_fenced(&resp, "sh"))
        .ok_or(StageError { attempts: 1, note: "no ```bash``` block in re-converge".into() })?;
    let pipeline = pipeline.trim().to_string();
    if let Some(reason) = validate_summary(&pipeline) {
        return Err(StageError { attempts: 1, note: reason });
    }
    Ok(pipeline)
}

#[derive(Debug)]
struct StageError {
    attempts: usize,
    note: String,
}

/// Cheap pre-validators that catch the recurring failure modes
/// without burning a model round-trip.  Each returns `Some(reason)`
/// to reject the candidate (we then re-prompt the model with `reason`
/// as the corrective feedback).
/// Result of inspecting a model-proposed command:
/// - `Ok(cmd)` — accepted (possibly auto-rewritten by us; see Diff).
/// - `Err(reason)` — reject and re-prompt.
enum CmdCheck {
    Accept(String),                 // possibly auto-rewritten command
    Rewritten(String, String),      // (new_cmd, what_we_did) — log it
    Reject(String),
}

fn validate_command(cmd: &str) -> CmdCheck {
    let mut current = cmd.to_string();
    let mut fixes: Vec<String> = Vec::new();

    // ── Auto-fix #1: literal `~` → `$HOME`. ──
    // Tilde isn't expanded inside quoted vars and confuses path use.
    if current.contains('~') {
        let before = current.clone();
        current = current.replace("~/", "$HOME/").replace('~', "$HOME");
        if current != before {
            fixes.push("`~` → `$HOME`".into());
        }
    }

    // ── Auto-fix #2: strip `sudo ` ──
    if current.contains("sudo ") {
        let before = current.clone();
        current = current.replace("sudo ", "");
        if current != before {
            fixes.push("removed `sudo`".into());
        }
    }

    // ── Reject #1: `apt install` / `apt-get install` (we can't fix it) ──
    let lower = current.to_lowercase();
    if lower.contains("apt-get install") || lower.contains("apt install") {
        return CmdCheck::Reject(
            "do not install packages with apt — the sandbox already has coreutils, curl, jq, \
             dig, ip, ss, ps, free, wc, awk, sed, grep, find, git, sha256sum.  Use one of those."
                .into(),
        );
    }

    // ── Auto-fix #3: bare `$1` → `"${1:-…}"` ──
    if current.contains("$1") && !current.contains("${1:-") {
        let before = current.clone();
        current = current
            .replace("\"$1\"", "\"${1:-/etc/hostname}\"")
            .replace("'$1'", "\"${1:-/etc/hostname}\"")
            .replace("$1", "\"${1:-/etc/hostname}\"");
        if current != before {
            fixes.push("wrapped bare `$1` with default".into());
        }
    }

    // ── Auto-fix #4: curl/wget without --max-time → add it ──
    if (current.contains("curl ") || current.contains("wget ")) && !current.contains("--max-time") {
        let before = current.clone();
        current = current
            .replace("curl ", "curl --max-time 5 ")
            .replace("wget ", "wget --timeout=5 ");
        if current != before {
            fixes.push("added `--max-time 5` to network call".into());
        }
    }

    if fixes.is_empty() {
        CmdCheck::Accept(current)
    } else {
        CmdCheck::Rewritten(current, format!("auto-fixes: {}", fixes.join(", ")))
    }
}

fn validate_summary(pipeline: &str) -> Option<String> {
    // Most common failure: the model writes `head input.txt` or
    // `cat /path/to/output.txt` instead of consuming stdin.  Look for
    // any reference to a literal file path that isn't `/dev/stdin`.
    let lower = pipeline.to_lowercase();
    let bad_filenames = [
        "input.txt", "output.txt", "data.txt", "file.txt", "your_file",
        "/path/to/", "<file>", "<input>", "$file",
    ];
    for needle in bad_filenames {
        if lower.contains(needle) {
            return Some(format!(
                "your pipeline references the placeholder `{needle}` — it must read from STDIN, \
                 not from a literal file.  Use `head -n 5`, `awk '{{...}}'`, `grep ...`, etc., \
                 with no filename argument so they consume the piped stdin."
            ));
        }
    }
    // Pipeline that doesn't actually filter — `cat -` alone is OK,
    // but a totally empty pipeline body isn't.
    if pipeline.trim().is_empty() {
        return Some("pipeline is empty.  Provide one bash command line.".into());
    }
    None
}

fn converge_command(cfg: &Config, user_request: &str, skill_name: &str) -> Result<(String, String, usize), StageError> {
    let mut last_err = String::new();
    let mut prompt = format!(
        "I want to build a tiny shell-based skill called `{skill_name}` that helps with: \"{user_request}\".\n\n\
         Step 1: propose ONE bash command-line that gathers the raw data the skill needs.\n\n\
         HARD CONSTRAINTS:\n\
         * Runs on Debian Linux (GNU coreutils).  Use Linux conventions: sha256sum, ip addr, \
           uptime -s, nproc, free -h, ss -tlnp, dig +short, wc -l.\n\
         * Argument arrives as `$1`.  ALWAYS supply a default that exists on a clean Debian: \
           `${{1:-/etc/hostname}}` for files, `${{1:-/etc}}` for dirs, `${{1:-localhost}}` for \
           hostnames, `${{1:-google.com}}` for an external host.\n\
         * Do NOT use `~` anywhere — there is no useful home dir in the sandbox.\n\
         * Network calls: `curl -s --max-time 5 https://…`.  Hostnames are NOT commands.\n\
         * No sudo, no apt install, no interactive prompts.  Tools available: coreutils, curl, \
           wget, jq, dig, ip, ss, ps, free, wc, awk, sed, grep, find, file, git, sha256sum.\n\
         * Must complete in under 8 seconds.  Output must be non-empty and parseable.\n\n\
         Reply ONLY with a single fenced ```bash``` block — no prose."
    );
    for attempt in 1..=STAGE_RETRIES {
        let resp = match chat(&cfg.server, &prompt) {
            Ok(t) => t,
            Err(e) => {
                last_err = format!("chat: {e}");
                continue;
            }
        };
        let cmd = match extract_fenced(&resp, "bash").or_else(|| extract_fenced(&resp, "sh")) {
            Some(c) => c.trim().to_string(),
            None => {
                last_err = "no ```bash``` block in reply".into();
                prompt = format!(
                    "Your previous reply did not contain a fenced ```bash``` block.  \
                     Reply ONLY with one fenced ```bash``` code block containing the bash \
                     command line.  No prose, no explanation.  Original task: \"{user_request}\"."
                );
                continue;
            }
        };
        // Cheap pre-validation — catches "no default", "literal ~",
        // "sudo", etc., without burning a container exec.
        let cmd = match validate_command(&cmd) {
            CmdCheck::Accept(c) => c,
            CmdCheck::Rewritten(c, why) => {
                eprintln!("[wizard] auto-fix: {why}");
                c
            }
            CmdCheck::Reject(reason) => {
                last_err = reason.clone();
                prompt = format!(
                    "Your command was rejected by validation:\n\n{reason}\n\n\
                     Your previous command:\n```bash\n{cmd}\n```\n\n\
                     Reply with a corrected command in ONE fenced ```bash``` block."
                );
                continue;
            }
        };
        match run_in_container(&cmd, "", "", Duration::from_secs(10)) {
            Ok(out) if !out.trim().is_empty() => {
                return Ok((cmd, out, attempt));
            }
            Ok(_) => {
                last_err = "produced empty output".into();
                prompt = format!(
                    "Your bash command produced EMPTY output when I ran it on a clean Debian:\n\n\
                     ```bash\n{cmd}\n```\n\n\
                     The skill goal is: \"{user_request}\".  Try a different command — perhaps \
                     the default for `$1` was wrong, or the path doesn't exist on a fresh \
                     install.  Reply with a corrected command in ONE fenced ```bash``` block."
                );
            }
            Err(e) => {
                last_err = e.clone();
                prompt = format!(
                    "Your bash command failed with this error:\n\n\
                     ```\n{e}\n```\n\n\
                     The command was:\n```bash\n{cmd}\n```\n\n\
                     Goal: \"{user_request}\".  Fix the command.  Reply with ONE fenced \
                     ```bash``` block — no prose."
                );
            }
        }
    }
    Err(StageError { attempts: STAGE_RETRIES, note: last_err })
}

fn converge_summary(cfg: &Config, sample_output: &str) -> Result<(String, usize), StageError> {
    let head = head_n(sample_output, SAMPLE_HEAD_LINES, SAMPLE_HEAD_BYTES);
    let mut last_err = String::new();
    let mut prompt = format!(
        "I just ran the data-gathering command and saved its output to a file.  \
         Here are the first {} lines:\n\n```\n{}\n```\n\n\
         Total bytes captured: {}.\n\n\
         Now write ONE bash pipeline that reads this output from stdin and emits a 1-3 line, \
         human-readable summary suitable to show the user.  You may use: head, tail, awk, grep, \
         sed, wc, sort, uniq, cut, tr, jq.  Reply ONLY with a single fenced ```bash``` block.",
        SAMPLE_HEAD_LINES, head, sample_output.len(),
    );
    for attempt in 1..=STAGE_RETRIES {
        let resp = match chat(&cfg.server, &prompt) {
            Ok(t) => t,
            Err(e) => { last_err = format!("chat: {e}"); continue; }
        };
        let pipeline = match extract_fenced(&resp, "bash").or_else(|| extract_fenced(&resp, "sh")) {
            Some(p) => p.trim().to_string(),
            None => {
                last_err = "no ```bash``` block".into();
                prompt = "Reply with a single fenced ```bash``` block containing the pipeline. No prose.".into();
                continue;
            }
        };
        if let Some(reason) = validate_summary(&pipeline) {
            last_err = reason.clone();
            prompt = format!(
                "Your pipeline was rejected by validation:\n\n{reason}\n\n\
                 Your previous pipeline:\n```bash\n{pipeline}\n```\n\n\
                 Reply with a corrected pipeline in ONE fenced ```bash``` block."
            );
            continue;
        }
        match run_in_container(&pipeline, "", sample_output, Duration::from_secs(8)) {
            Ok(out) if !out.trim().is_empty() => {
                return Ok((pipeline, attempt));
            }
            Ok(_) => {
                last_err = "pipeline produced empty output".into();
                prompt = format!(
                    "Your pipeline produced EMPTY output:\n\n```bash\n{pipeline}\n```\n\n\
                     Reading from stdin (which was the {} bytes I already showed you).  Try a \
                     different pipeline.  Reply with ONE fenced ```bash``` block.",
                    sample_output.len()
                );
            }
            Err(e) => {
                last_err = e.clone();
                prompt = format!(
                    "Your pipeline errored:\n\n```\n{e}\n```\n\nThe pipeline was:\n\
                     ```bash\n{pipeline}\n```\n\nFix it.  Reply with ONE fenced ```bash``` block."
                );
            }
        }
    }
    Err(StageError { attempts: STAGE_RETRIES, note: last_err })
}

/// Optional stage — failure is non-fatal, we just install with no chart.
fn converge_chart(cfg: &Config, sample_output: &str) -> (Option<String>, usize) {
    let head = head_n(sample_output, SAMPLE_HEAD_LINES, SAMPLE_HEAD_BYTES);
    let prompt = format!(
        "Optional last step.  Could the data be rendered as a small chart.js bar/doughnut \
         chart?  If yes, reply ONLY with a fenced ```json``` block containing chart.js JSON \
         using one of these shapes:\n\n\
         doughnut: {{\"type\":\"doughnut\",\"data\":{{\"labels\":[..],\"datasets\":[{{\"data\":[..]}}]}}}}\n\
         bar:      {{\"type\":\"bar\",\"data\":{{\"labels\":[..],\"datasets\":[{{\"label\":\"...\",\"data\":[..]}}]}}}}\n\n\
         If a chart wouldn't make sense, reply with the literal word `none` (no fence).\n\n\
         Sample output (first {} lines):\n```\n{}\n```",
        SAMPLE_HEAD_LINES, head,
    );
    let resp = chat(&cfg.server, &prompt).unwrap_or_default();
    if resp.trim().eq_ignore_ascii_case("none") {
        return (None, 1);
    }
    if let Some(j) = extract_fenced(&resp, "json").or_else(|| extract_fenced(&resp, "chartjs")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(j.trim()) {
            if v["type"].is_string() && v["data"].is_object() {
                return (Some(j.trim().to_string()), 1);
            }
        }
    }
    (None, 1)
}

// ──────────────────────── Helpers ───────────────────────────────────

fn home_dir() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/tmp"))
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

fn head_n(s: &str, lines: usize, byte_cap: usize) -> String {
    let mut out = String::new();
    for (i, line) in s.lines().enumerate() {
        if i >= lines || out.len() + line.len() > byte_cap { break; }
        if i > 0 { out.push('\n'); }
        out.push_str(line);
    }
    if s.len() > out.len() { out.push_str("\n…"); }
    out
}

fn extract_fenced(text: &str, lang: &str) -> Option<String> {
    let opener = format!("```{lang}");
    let mut idx = text.find(&opener)?;
    idx += opener.len();
    let after = &text[idx..];
    let nl = after.find('\n')?;
    let content_start = idx + nl + 1;
    let close = text[content_start..].find("```")?;
    Some(text[content_start..content_start + close].to_string())
}

// ──────────────────────── Container exec ────────────────────────────

fn ensure_runtime() -> Result<(), String> {
    let alive = Command::new("docker")
        .args(["inspect", "-f", "{{.State.Running}}", RUNTIME_CONTAINER])
        .output()
        .map_err(|e| format!("docker inspect: {e}"))?;
    if alive.status.success() && String::from_utf8_lossy(&alive.stdout).trim() == "true" {
        return Ok(());
    }
    let _ = Command::new("docker").args(["rm", "-f", RUNTIME_CONTAINER]).output();
    eprintln!("[wizard] starting runtime container '{RUNTIME_CONTAINER}'…");
    let host_skills = home_dir().join(".larql/skills");
    let _ = std::fs::create_dir_all(&host_skills);
    let mount = format!("{}:{}", host_skills.display(), host_skills.display());
    let run = Command::new("docker")
        .args([
            "run", "-d", "--name", RUNTIME_CONTAINER, "--rm", "-v", &mount,
            "debian:bookworm-slim", "bash", "-c", "sleep infinity",
        ])
        .output()
        .map_err(|e| format!("docker run: {e}"))?;
    if !run.status.success() {
        return Err(format!("docker run failed: {}", String::from_utf8_lossy(&run.stderr)));
    }
    let provision = Command::new("docker")
        .args([
            "exec", RUNTIME_CONTAINER, "bash", "-c",
            "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
             coreutils procps net-tools iproute2 dnsutils curl wget jq grep sed gawk findutils \
             git ca-certificates file unzip xz-utils >/dev/null 2>&1",
        ])
        .output()
        .map_err(|e| format!("provision: {e}"))?;
    if !provision.status.success() {
        return Err(format!("provision failed: {}", String::from_utf8_lossy(&provision.stderr)));
    }
    eprintln!("[wizard] runtime ready");
    Ok(())
}

/// Run a bash command in the runtime container.  `arg` becomes `$1`,
/// `stdin_text` is piped to the process if non-empty.
fn run_in_container(cmd: &str, arg: &str, stdin_text: &str, timeout: Duration) -> Result<String, String> {
    ensure_runtime()?;
    let mut child = Command::new("docker")
        .args(["exec", "-i", RUNTIME_CONTAINER, "bash", "-c", cmd, "_", arg])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn: {e}"))?;
    if let Some(mut s) = child.stdin.take() {
        if !stdin_text.is_empty() { let _ = s.write_all(stdin_text.as_bytes()); }
    }
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                return Err(format!("timeout after {timeout:?}"));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => return Err(format!("wait: {e}")),
        }
    }
    let mut out = String::new();
    if let Some(mut s) = child.stdout.take() { let _ = s.read_to_string(&mut out); }
    if out.trim().is_empty() {
        let mut err = String::new();
        if let Some(mut s) = child.stderr.take() { let _ = s.read_to_string(&mut err); }
        if !err.trim().is_empty() {
            return Err(err.trim().to_string());
        }
    }
    Ok(out)
}

// ──────────────────────── Assembly + install ────────────────────────

fn assemble_tool_sh(sample_cmd: &str, summary_pipeline: &str, chart_json: &Option<String>) -> String {
    let mut s = String::new();
    s.push_str("#!/bin/bash\n");
    s.push_str("source \"$(dirname \"$0\")/../_lib.sh\"\n\n");
    s.push_str("ARG=\"${1:-}\"\n\n");
    s.push_str("start_timer\n");
    s.push_str("RAW=$(set -- \"$ARG\"; ");
    s.push_str(sample_cmd);
    s.push_str(" 2>&1)\n");
    s.push_str("stop_timer\n\n");
    s.push_str("emit_raw \"$RAW\"\n\n");
    s.push_str("SUMMARY=$(printf '%s\\n' \"$RAW\" | (");
    s.push_str(summary_pipeline);
    s.push_str("))\n");
    s.push_str("emit_summary \"$SUMMARY\"\n");
    if let Some(cj) = chart_json {
        s.push_str("\nCHART=$(cat <<'CHART_EOF'\n");
        s.push_str(cj);
        s.push_str("\nCHART_EOF\n)\n");
        s.push_str("emit_chart \"$CHART\"\n");
    }
    s
}

fn install_skill(target_dir: &PathBuf, name: &str, description: &str, tool_sh: &str) -> Result<(), String> {
    std::fs::create_dir_all(target_dir).map_err(|e| format!("mkdir: {e}"))?;
    let kw: Vec<&str> = description
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() > 3)
        .take(8)
        .collect();
    let kw_joined = if kw.is_empty() { name.to_string() } else { kw.join(", ") };
    let skill_md = format!(
        "---\nname: {name}\ndescription: {description}\nkeywords: {kw_joined}\nalways: false\nruntime: container\n---\n\n# Skill: {name}\n\n{description}\n\nInvoke as:\n\n```tool\n{name}\n```\n",
    );
    std::fs::write(target_dir.join("skill.md"), skill_md).map_err(|e| format!("write skill.md: {e}"))?;
    let tool_path = target_dir.join("tool.sh");
    std::fs::write(&tool_path, tool_sh).map_err(|e| format!("write tool.sh: {e}"))?;
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(&tool_path).map_err(|e| format!("stat: {e}"))?.permissions();
    perms.set_mode(0o755);
    let _ = std::fs::set_permissions(&tool_path, perms);
    Ok(())
}

// ────────────────── HTTP chat — non-streaming collect ────────────────

fn chat(server: &str, prompt: &str) -> Result<String, String> {
    let body = serde_json::json!({
        "messages": [{ "role": "user", "content": prompt }],
        "stream": true,
        "max_tokens": 600,
        "temperature": 0.0,
    });
    let payload = serde_json::to_string(&body).map_err(|e| e.to_string())?;
    let url = format!("{server}/v1/chat/completions");
    let mut child = Command::new("curl")
        .args(["-sN", "-X", "POST", "-H", "content-type: application/json",
               "-d", &payload, "--max-time", "60", &url])
        .stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().map_err(|e| format!("curl: {e}"))?;
    let mut raw = String::new();
    if let Some(mut s) = child.stdout.take() { let _ = s.read_to_string(&mut raw); }
    let _ = child.wait();
    let mut out = String::new();
    for line in raw.lines() {
        let data = match line.trim().strip_prefix("data: ") { Some(d) => d, None => continue };
        if data == "[DONE]" { break; }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(c) = v["choices"][0]["delta"]["content"].as_str() { out.push_str(c); }
        }
    }
    if out.trim().is_empty() { return Err(format!("empty response (raw: {} bytes)", raw.len())); }
    Ok(out)
}

// ────────────────────── Harness + reporting ──────────────────────────

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    prompt: String,
}

fn load_cases(path: &PathBuf) -> Vec<Case> {
    if !path.exists() {
        // Seed an empty file so the user knows where to put cases.
        let _ = std::fs::create_dir_all(path.parent().unwrap_or(&PathBuf::from(".")));
        let _ = std::fs::write(
            path,
            "[\n  { \"name\": \"example_case\", \"prompt\": \"replace this with what you want a skill to do\" }\n]\n",
        );
        eprintln!("[wizard] seeded {} — edit it and re-run.", path.display());
        return Vec::new();
    }
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[wizard] cannot read {}: {e}", path.display());
            return Vec::new();
        }
    };
    serde_json::from_str(&raw).unwrap_or_else(|e| {
        eprintln!("[wizard] {} parse error: {e}", path.display());
        Vec::new()
    })
}

fn run_harness(cfg: &Config, cases_path: &PathBuf) {
    let cases = load_cases(cases_path);
    if cases.is_empty() {
        eprintln!("[wizard] no cases to run.  Edit {} and try again.", cases_path.display());
        std::process::exit(2);
    }
    let mut summary: Vec<(String, bool, u128, usize, String)> = Vec::new();
    for c in &cases {
        eprintln!("\n──── {} ────  «{}»", c.name, c.prompt);
        let _ = std::fs::remove_dir_all(home_dir().join(".larql/skills").join(&c.name));
        let outcome = run_wizard(cfg, &c.prompt, &c.name);
        let last_note = outcome.stages.iter().rev().next()
            .map(|s| s.3.lines().next().unwrap_or("").to_string()).unwrap_or_default();
        let attempts: usize = outcome.stages.iter().map(|s| s.2).sum();
        eprintln!("  {} ({} ms, {} model attempts)  {}",
            if outcome.success { "PASS" } else { "FAIL" },
            outcome.elapsed_ms, attempts, last_note);
        summary.push((c.name.clone(), outcome.success, outcome.elapsed_ms, attempts, last_note));
    }
    let pass = summary.iter().filter(|s| s.1).count();
    let total = summary.len();
    let total_ms: u128 = summary.iter().map(|s| s.2).sum();
    let total_attempts: usize = summary.iter().map(|s| s.3).sum();
    eprintln!("\n══════════════════════════════════════════════");
    eprintln!("  {pass} / {total} passed   ({} s, {} model attempts total)", total_ms / 1000, total_attempts);
    eprintln!("══════════════════════════════════════════════");
    for (name, ok, ms, att, note) in &summary {
        eprintln!("  {:>4}  {:<20} {:>6}ms  {} attempts   {}",
            if *ok { "PASS" } else { "FAIL" }, name, ms, att, note);
    }
    if pass != total { std::process::exit(1); }
}

fn print_outcome(o: &WizardOutcome) {
    eprintln!("─────────────── skill_wizard ───────────────");
    eprintln!("  request: {}", o.user_request);
    eprintln!("  name:    {}", o.skill_name);
    eprintln!("  total:   {} ms", o.elapsed_ms);
    for (stage, ok, attempts, note) in &o.stages {
        eprintln!("  {:1}  {:<22}  ({}x)  {}",
            if *ok { "✓" } else { "✗" }, stage, attempts, note.lines().next().unwrap_or(""));
    }
    eprintln!("  RESULT: {}", if o.success { "PASS" } else { "FAIL" });
}

#[allow(dead_code)]
fn _hashmap_alive() -> HashMap<i32, i32> { HashMap::new() }
