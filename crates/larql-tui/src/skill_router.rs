//! Skill loader + TF-IDF router, ported from gaucho-code's
//! `frontmatter-tfidf-v1.md` (29/39 cases, ~22µs/route on bench).
//!
//! Each `~/.larql/skills/<name>/skill.md` may carry a tiny YAML
//! frontmatter:
//!
//!     ---
//!     name: list
//!     description: directory listing with chart
//!     keywords: list, ls, files, directory, folder, contents, browse
//!     always: false
//!     ---
//!
//! `keywords` is a comma-separated list (each gets confidence 1.0).
//! Set `always: true` on skills that should be in *every* prompt
//! regardless of routing — used for `annotate`, the meta-skill that
//! teaches the model to emit ```fact``` / ```status``` blocks.
//!
//! Skills without frontmatter are still loaded; they just route only
//! by name + description-words extracted from the body.

use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    /// Description from frontmatter or first heading.
    pub description: String,
    /// Phrases the router matches on, lowercased.
    pub keywords: Vec<String>,
    /// Full skill.md body (the part the model gets when routed).
    pub body: String,
    /// If true, always include in every primer (e.g. `annotate`).
    pub always: bool,
    /// Where the tool.sh runs.  `"host"` (default for legacy skills
    /// written for macOS) or `"container"` (Linux sandbox; what the
    /// wizard produces).  Routed by execute_skill_tool.
    pub runtime: String,
}

/// Load every `<dir>/<name>/skill.md` into a `Skill`.  `dirs` are
/// scanned in order; later dirs override earlier ones if they declare
/// the same skill name.
pub fn load_skills(dirs: &[PathBuf]) -> Vec<Skill> {
    let mut by_name: HashMap<String, Skill> = HashMap::new();
    for dir in dirs {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) if !n.starts_with('_') && !n.starts_with('.') => n.to_string(),
                _ => continue,
            };
            let md_path = path.join("skill.md");
            if !md_path.exists() {
                continue;
            }
            let raw = match std::fs::read_to_string(&md_path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let skill = parse_skill(name, raw);
            by_name.insert(skill.name.clone(), skill);
        }
    }
    let mut out: Vec<Skill> = by_name.into_values().collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Parse one skill.md.  Frontmatter is optional; without it we still
/// extract a description and treat the skill name itself as a keyword.
fn parse_skill(default_name: String, raw: String) -> Skill {
    let mut name = default_name.clone();
    let mut description = String::new();
    let mut keywords: Vec<String> = Vec::new();
    let mut always = false;
    let mut runtime = "host".to_string();
    let mut body_start = 0usize;

    if let Some(rest) = raw.strip_prefix("---\n") {
        if let Some(end) = rest.find("\n---") {
            let block = &rest[..end];
            for line in block.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    let key = k.trim();
                    let val = v.trim();
                    match key {
                        "name" if !val.is_empty() => name = val.to_string(),
                        "description" => description = val.to_string(),
                        "keywords" => {
                            keywords = val
                                .split(',')
                                .map(|s| s.trim().to_lowercase())
                                .filter(|s| !s.is_empty())
                                .collect();
                        }
                        "always" => {
                            always = matches!(val.to_lowercase().as_str(), "true" | "yes" | "1");
                        }
                        "runtime" => {
                            let v = val.to_lowercase();
                            if v == "container" || v == "host" {
                                runtime = v;
                            }
                        }
                        _ => {}
                    }
                }
            }
            body_start = 4 + end + 4; // "---\n" + block + "\n---"
            // Skip a possible blank line after the closing ---
            if raw.as_bytes().get(body_start) == Some(&b'\n') {
                body_start += 1;
            }
        }
    }

    let body = raw[body_start..].to_string();
    if description.is_empty() {
        description = body
            .lines()
            .find(|l| !l.trim().is_empty())
            .map(|l| l.trim_start_matches(['#', ' ']).trim().to_string())
            .unwrap_or_else(|| name.clone());
    }
    if keywords.is_empty() {
        keywords.push(name.to_lowercase());
    }

    Skill {
        name,
        description,
        keywords,
        body,
        always,
        runtime,
    }
}

// ── Routing ──────────────────────────────────────────────────────────

/// Pre-built TF-IDF index for one set of skills.  Build once at
/// startup, then call `route` per user prompt.
pub struct Index<'a> {
    pub skills: &'a [Skill],
    /// vectors[i] = term → tf-idf weight for skills[i] (only routable).
    vectors: Vec<HashMap<String, f32>>,
    /// Skills marked `always: true` — never need routing.
    pub always_idx: Vec<usize>,
}

const STOP: &[&str] = &[
    "the", "and", "for", "from", "with", "show", "that", "this", "are", "can", "has", "have",
    "its", "not", "but", "bar", "pie", "chart", "charts", "available", "you", "have", "tool",
    "called", "when", "user", "asks",
];

pub fn build_index(skills: &[Skill]) -> Index<'_> {
    // Routable skills (everything not flagged `always`).
    let routable: Vec<usize> = skills
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.always)
        .map(|(i, _)| i)
        .collect();
    let always_idx: Vec<usize> = skills
        .iter()
        .enumerate()
        .filter(|(_, s)| s.always)
        .map(|(i, _)| i)
        .collect();

    // Term frequencies per routable skill.
    let mut term_tables: Vec<HashMap<String, f32>> = vec![HashMap::new(); skills.len()];
    let mut doc_freq: HashMap<String, usize> = HashMap::new();

    for &i in &routable {
        let s = &skills[i];
        let terms = &mut term_tables[i];

        // Keywords (strongest signal, weight 3.0; sub-words 1.0).
        for kw in &s.keywords {
            *terms.entry(kw.clone()).or_insert(0.0) += 3.0;
            for word in kw.split_whitespace() {
                if word.len() > 2 {
                    *terms.entry(word.to_string()).or_insert(0.0) += 1.0;
                }
            }
        }
        // Skill name as keyword too (weight 2.0).
        let lc_name = s.name.to_lowercase();
        for word in lc_name.split('-') {
            if word.len() > 2 {
                *terms.entry(word.to_string()).or_insert(0.0) += 2.0;
            }
        }
        // Description words (weak, 0.3).
        for w in extract_words(&s.description.to_lowercase(), 4) {
            if !STOP.contains(&w.as_str()) {
                *terms.entry(w).or_insert(0.0) += 0.3;
            }
        }

        for term in terms.keys() {
            *doc_freq.entry(term.clone()).or_insert(0) += 1;
        }
    }

    let n_docs = routable.len().max(1) as f32;
    let mut vectors: Vec<HashMap<String, f32>> = vec![HashMap::new(); skills.len()];
    for &i in &routable {
        let terms = &term_tables[i];
        let max_tf = terms.values().cloned().fold(1.0f32, f32::max).max(1.0);
        let mut v: HashMap<String, f32> = HashMap::with_capacity(terms.len());
        for (term, raw_tf) in terms.iter() {
            let tf = raw_tf / max_tf;
            let df = *doc_freq.get(term).unwrap_or(&1) as f32;
            let idf = (n_docs / df).ln().max(0.0);
            v.insert(term.clone(), tf * idf);
        }
        vectors[i] = v;
    }

    Index {
        skills,
        vectors,
        always_idx,
    }
}

/// Score-and-pick.  Returns Some((skill_index, confidence)) if a
/// match clears the threshold; None if the prompt is too generic, an
/// obvious raw shell command, or no skill matches.
pub fn route(prompt: &str, idx: &Index) -> Option<(usize, f32)> {
    let p = prompt.to_lowercase();

    // Raw-command negatives: things that look like the user is just
    // typing a command at us, not asking for a skill.
    if looks_like_raw_command(&p) {
        return None;
    }

    let words: std::collections::HashSet<String> = extract_words(&p, 3).into_iter().collect();

    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(idx.vectors.len());
    for (i, vector) in idx.vectors.iter().enumerate() {
        if vector.is_empty() {
            continue;
        }
        let mut score = 0.0f32;
        for (term, tfidf) in vector.iter() {
            if term.contains(' ') {
                if p.contains(term) {
                    score += tfidf * 1.5;
                }
            } else if words.contains(term) {
                score += tfidf;
            }
        }
        if score > 0.0 {
            scores.push((i, score));
        }
    }
    if scores.is_empty() {
        return None;
    }
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (best_i, best_score) = scores[0];
    let runner_up = scores.get(1).map(|x| x.1).unwrap_or(0.0);
    let separation = if runner_up > 0.0 {
        best_score / runner_up
    } else {
        2.0
    };
    let confidence =
        ((best_score / 3.0) * (separation / 1.5).min(1.0)).min(1.0);
    // 0.15 was too trigger-happy: "lets MAKE a fibonacci.py" scored
    // 0.17 against `make_skill` purely on the word "make", pulling
    // the wizard primer into context when the user really wanted
    // `run`.  0.22 keeps intentional matches (usually 0.3+) while
    // rejecting one-keyword coincidences.
    if confidence >= 0.22 {
        Some((best_i, confidence))
    } else {
        None
    }
}

fn extract_words(text: &str, min_len: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            if cur.len() >= min_len {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if cur.len() >= min_len {
        out.push(cur);
    }
    out
}

fn looks_like_raw_command(p: &str) -> bool {
    let trimmed = p.trim_start();
    // Leading word matching common commands followed by an arg/flag.
    const CMDS: &[&str] = &[
        "git ", "ls ", "cat ", "cd ", "rm ", "cp ", "mv ", "find ", "grep ", "curl ", "wget ",
        "npm ", "cargo ", "pip ", "docker ", "python ", "python3 ", "ssh ", "du ", "df ",
    ];
    if CMDS.iter().any(|c| trimmed.starts_with(c)) {
        return true;
    }
    if trimmed.contains(" -") && !trimmed.contains(" - ") {
        return true; // looks like `cmd --flag`
    }
    if trimmed.contains(" | ") || trimmed.contains(" && ") || trimmed.contains(" > ") {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(name: &str, kw: &[&str]) -> Skill {
        Skill {
            name: name.into(),
            description: format!("test skill {name}"),
            keywords: kw.iter().map(|k| k.to_lowercase()).collect(),
            body: format!("# {name}\n\nbody"),
            always: false,
            runtime: "host".into(),
        }
    }

    #[test]
    fn routes_clock_for_time_questions() {
        let skills = vec![
            s("clock", &["clock", "time", "uptime", "load"]),
            s("list", &["list", "files", "directory"]),
            s("disk", &["disk", "free space", "mounts"]),
        ];
        let idx = build_index(&skills);
        let (i, conf) = route("what time is it on this machine", &idx).expect("matched");
        assert_eq!(skills[i].name, "clock");
        assert!(conf >= 0.15);
    }

    #[test]
    fn routes_list_for_file_questions() {
        // Use a representative-size skill set so IDF has discrimination
        // (with only 2 skills every term has ln(2/1)=0.693, too weak).
        let skills = vec![
            s("clock", &["clock", "time", "uptime"]),
            s("list", &["list", "files", "directory", "folder", "browse"]),
            s("disk", &["disk", "free space", "mounts"]),
            s("proc", &["processes", "running", "cpu"]),
            s("ports", &["ports", "listening", "network"]),
            s("git", &["git", "status", "commits", "branch"]),
            s("head", &["head", "first lines", "peek file"]),
            s("stats", &["stats", "language", "lines of code"]),
        ];
        let idx = build_index(&skills);
        let (i, _) = route("show me the files in /tmp", &idx).expect("matched");
        assert_eq!(skills[i].name, "list");
    }

    #[test]
    fn raw_command_routes_to_nothing() {
        let skills = vec![s("git", &["git", "status", "log"])];
        let idx = build_index(&skills);
        assert!(route("git status", &idx).is_none());
        assert!(route("ls -la /tmp", &idx).is_none());
        assert!(route("echo hello | grep h", &idx).is_none());
    }

    #[test]
    fn no_match_returns_none() {
        let skills = vec![s("clock", &["clock", "time"])];
        let idx = build_index(&skills);
        assert!(route("the quick brown fox jumps", &idx).is_none());
    }
}
