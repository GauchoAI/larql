//! Workflow store + plan/status block parser.
//!
//! Workflows are the planner-side counterpart to `fact:` annotations:
//! when the model emits a ```plan``` block we record a new workflow with
//! its steps; when it later emits a ```status``` block targeting that
//! workflow we update the matching step's state.  The whole store is
//! persisted to `~/.larql/workflows.json` so the next session can resume
//! and the sidebar can render live state.
//!
//! Block formats expected from the model (taught by the annotate skill):
//!
//! ```plan
//! workflow: add caching layer
//! step: research existing cache sites
//! step: implement Redis client
//! step: add tests
//! ```
//!
//! ```status
//! workflow: add caching layer
//! step: implement Redis client
//! state: done
//! output: shipped at v0.4.2
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum WorkflowState {
    #[default]
    Active,
    Done,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StepState {
    #[default]
    Pending,
    Active,
    Done,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub description: String,
    #[serde(default)]
    pub state: StepState,
    #[serde(default)]
    pub output: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub name: String,
    #[serde(default)]
    pub state: WorkflowState,
    #[serde(default)]
    pub steps: Vec<Step>,
    #[serde(default)]
    pub ts: u64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct WorkflowStore {
    #[serde(default)]
    pub workflows: Vec<Workflow>,
}

impl WorkflowStore {
    pub fn load(path: &PathBuf) -> Self {
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(_) => return Self::default(),
        };
        // Try the canonical format first.
        if let Ok(s) = serde_json::from_str::<WorkflowStore>(&raw) {
            return s;
        }
        // Fall back to a bare `[Workflow, ...]` (older flat format).
        if let Ok(list) = serde_json::from_str::<Vec<Workflow>>(&raw) {
            return Self { workflows: list };
        }
        Self::default()
    }

    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".into());
        std::fs::write(path, json)
    }

    /// Insert a new workflow or replace an existing one of the same
    /// name.  The "replace" path is used when the model re-plans —
    /// we treat the latest ```plan``` block as authoritative.
    pub fn upsert(&mut self, wf: Workflow) {
        if let Some(existing) = self.workflows.iter_mut().find(|w| w.name == wf.name) {
            *existing = wf;
        } else {
            self.workflows.push(wf);
        }
    }

    /// Apply a status update.  Matches the workflow by name (case
    /// insensitive, trimmed) and the step by description (substring,
    /// case insensitive).  Returns true if anything changed.
    pub fn apply_status(&mut self, upd: &StatusUpdate) -> bool {
        let wf = match self
            .workflows
            .iter_mut()
            .find(|w| w.name.eq_ignore_ascii_case(upd.workflow.trim()))
        {
            Some(w) => w,
            None => return false,
        };
        let needle = upd.step.trim().to_lowercase();
        let step = match wf
            .steps
            .iter_mut()
            .find(|s| s.description.to_lowercase().contains(&needle))
        {
            Some(s) => s,
            None => return false,
        };
        let mut changed = false;
        if let Some(state) = upd.state {
            if step.state != state {
                step.state = state;
                changed = true;
            }
        }
        if let Some(out) = &upd.output {
            if step.output.as_deref() != Some(out.as_str()) {
                step.output = Some(out.clone());
                changed = true;
            }
        }
        // If every step is done, mark the workflow done too.
        if wf.steps.iter().all(|s| s.state == StepState::Done) {
            if wf.state != WorkflowState::Done {
                wf.state = WorkflowState::Done;
                changed = true;
            }
        }
        changed
    }
}

#[derive(Debug, Clone)]
pub struct StatusUpdate {
    pub workflow: String,
    pub step: String,
    pub state: Option<StepState>,
    pub output: Option<String>,
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Pull every ```plan ... ``` block out of `text` and parse it into a
/// Workflow.  Blocks without a `workflow:` header are skipped.
pub fn extract_plan_blocks(text: &str) -> Vec<Workflow> {
    extract_blocks(text, "plan")
        .into_iter()
        .filter_map(|body| parse_plan_body(&body))
        .collect()
}

/// Pull every ```status ... ``` block out of `text` and parse it.
pub fn extract_status_updates(text: &str) -> Vec<StatusUpdate> {
    extract_blocks(text, "status")
        .into_iter()
        .filter_map(|body| parse_status_body(&body))
        .collect()
}

fn extract_blocks(text: &str, lang: &str) -> Vec<String> {
    let needle = format!("```{lang}");
    let mut out = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = text[cursor..].find(&needle) {
        let block_start = cursor + rel + needle.len();
        let after = &text[block_start..];
        let nl = match after.find('\n') {
            Some(n) => n + 1,
            None => break,
        };
        let body_start = block_start + nl;
        let close = match text[body_start..].find("```") {
            Some(c) => c,
            None => break,
        };
        out.push(text[body_start..body_start + close].to_string());
        cursor = body_start + close + 3;
    }
    out
}

fn parse_plan_body(body: &str) -> Option<Workflow> {
    let mut name = String::new();
    let mut steps: Vec<Step> = Vec::new();
    for line in body.lines() {
        let line = line.trim_end();
        let trimmed = line.trim_start();
        if let Some(v) = strip_field(line, "workflow").or_else(|| strip_field(line, "task")) {
            name = v.to_string();
        } else if let Some(v) = strip_field(line, "step") {
            // `step: <text>`
            steps.push(make_step(v));
        } else if let Some(stripped) = strip_list_marker(trimmed) {
            // Bullet-list style:  `1. ...`, `- ...`, `* ...`
            // (often appears under a `steps:` header).
            if !stripped.is_empty() {
                steps.push(make_step(stripped));
            }
        }
    }
    if name.trim().is_empty() || steps.is_empty() {
        return None;
    }
    Some(Workflow {
        name: name.trim().to_string(),
        state: WorkflowState::Active,
        steps,
        ts: now_unix(),
    })
}

fn make_step(raw: &str) -> Step {
    // The model sometimes appends a `[pending]` / `[done]` tag to the
    // step text — pull it out into the state if present.
    let mut desc = raw.trim().to_string();
    let mut state = StepState::Pending;
    if let Some(open) = desc.rfind('[') {
        if desc.ends_with(']') {
            let tag = &desc[open + 1..desc.len() - 1];
            if let Some(s) = parse_step_state(tag) {
                state = s;
                desc = desc[..open].trim_end().to_string();
            }
        }
    }
    Step { description: desc, state, output: None }
}

/// Strip a leading `1.`, `1)`, `-`, or `*` list marker and return the
/// remaining text.  Returns None if no marker matches (so non-list
/// lines like `steps:` get skipped instead of treated as a step).
fn strip_list_marker(line: &str) -> Option<&str> {
    if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
        return Some(rest.trim());
    }
    // Numbered: peel off leading digits + `.` or `)` + space.
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
    if i > 0 && i < bytes.len() && (bytes[i] == b'.' || bytes[i] == b')') {
        let after = &line[i + 1..];
        if let Some(rest) = after.strip_prefix(' ') {
            return Some(rest.trim());
        }
    }
    None
}

fn parse_status_body(body: &str) -> Option<StatusUpdate> {
    let mut wf = String::new();
    let mut step = String::new();
    let mut state: Option<StepState> = None;
    let mut output: Option<String> = None;
    let mut task_name: Option<String> = None;
    for line in body.lines() {
        let line = line.trim_end();
        if let Some(v) = strip_field(line, "workflow") {
            wf = v.to_string();
        } else if let Some(v) = strip_field(line, "task") {
            // Older skill format used `task:` for the workflow name when
            // it wasn't part of a multi-step plan.  Treat it as a
            // workflow-of-one — useful for the sidebar.
            task_name = Some(v.to_string());
        } else if let Some(v) = strip_field(line, "step") {
            // Strip a leading `N/M --` prefix if present so the parser
            // matches the step description rather than the index.
            step = strip_step_index(v).to_string();
        } else if let Some(v) = strip_field(line, "state") {
            state = parse_step_state(v.trim());
        } else if let Some(v) = strip_field(line, "detail").or_else(|| strip_field(line, "output")) {
            output = Some(v.to_string());
        }
    }
    if wf.trim().is_empty() {
        // Fall back to `task:` alone — synthesise a workflow-of-one.
        if let Some(t) = task_name {
            return Some(StatusUpdate {
                workflow: t.clone(),
                step: t,
                state,
                output,
            });
        }
        return None;
    }
    if step.trim().is_empty() {
        return None;
    }
    Some(StatusUpdate {
        workflow: wf,
        step,
        state,
        output,
    })
}

fn strip_step_index(s: &str) -> &str {
    // Matches `1/3 -- something` or `2 -- something`.
    let s = s.trim();
    if let Some(idx) = s.find("--") {
        let (head, tail) = s.split_at(idx);
        let head_ok = head.trim().chars().all(|c| c.is_ascii_digit() || c == '/');
        if head_ok && !head.trim().is_empty() {
            return tail.trim_start_matches('-').trim();
        }
    }
    s
}

fn strip_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}:");
    line.strip_prefix(&prefix).map(|s| s.trim())
}

fn parse_step_state(s: &str) -> Option<StepState> {
    match s.to_lowercase().as_str() {
        "pending" | "todo" | "queued" => Some(StepState::Pending),
        "active" | "in_progress" | "in-progress" | "wip" | "doing" => Some(StepState::Active),
        "done" | "complete" | "completed" | "finished" => Some(StepState::Done),
        "failed" | "error" | "blocked" => Some(StepState::Failed),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plan_block() {
        let txt = "blah\n```plan\nworkflow: build cache\nstep: research\nstep: implement\nstep: test\n```\nmore";
        let wfs = extract_plan_blocks(txt);
        assert_eq!(wfs.len(), 1);
        assert_eq!(wfs[0].name, "build cache");
        assert_eq!(wfs[0].steps.len(), 3);
        assert_eq!(wfs[0].steps[0].state, StepState::Pending);
    }

    #[test]
    fn parses_status_block_and_applies() {
        let mut store = WorkflowStore::default();
        store.upsert(Workflow {
            name: "build cache".into(),
            state: WorkflowState::Active,
            steps: vec![
                Step { description: "research".into(), state: StepState::Pending, output: None },
                Step { description: "implement Redis".into(), state: StepState::Pending, output: None },
            ],
            ts: 0,
        });
        let upds = extract_status_updates(
            "```status\nworkflow: build cache\nstep: implement\nstate: done\noutput: shipped\n```",
        );
        assert_eq!(upds.len(), 1);
        assert!(store.apply_status(&upds[0]));
        assert_eq!(store.workflows[0].steps[1].state, StepState::Done);
        assert_eq!(store.workflows[0].steps[1].output.as_deref(), Some("shipped"));
    }

    #[test]
    fn marks_workflow_done_when_all_steps_done() {
        let mut store = WorkflowStore::default();
        store.upsert(Workflow {
            name: "x".into(),
            state: WorkflowState::Active,
            steps: vec![Step { description: "only".into(), state: StepState::Pending, output: None }],
            ts: 0,
        });
        let upds = extract_status_updates("```status\nworkflow: x\nstep: only\nstate: done\n```");
        store.apply_status(&upds[0]);
        assert_eq!(store.workflows[0].state, WorkflowState::Done);
    }

    #[test]
    fn parses_skill_doc_format() {
        // Exactly the format documented in the `annotate` skill doc.
        let txt = "```plan\n\
            workflow: refactor cache\n\
            steps:\n  \
              1. survey call sites [pending]\n  \
              2. extract trait [pending]\n  \
              3. migrate one site as POC [pending]\n\
            ```";
        let wfs = extract_plan_blocks(txt);
        assert_eq!(wfs.len(), 1);
        assert_eq!(wfs[0].name, "refactor cache");
        assert_eq!(wfs[0].steps.len(), 3);
        assert_eq!(wfs[0].steps[0].description, "survey call sites");
        assert_eq!(wfs[0].steps[0].state, StepState::Pending);
    }

    #[test]
    fn parses_skill_doc_status_with_step_index() {
        let mut store = WorkflowStore::default();
        store.upsert(Workflow {
            name: "refactor cache".into(),
            state: WorkflowState::Active,
            steps: vec![
                Step { description: "survey call sites".into(), state: StepState::Pending, output: None },
                Step { description: "extract trait".into(), state: StepState::Pending, output: None },
            ],
            ts: 0,
        });
        let txt = "```status\n\
            workflow: refactor cache\n\
            step: 2/3 -- extract trait\n\
            state: active\n\
            detail: writing CacheBackend trait\n\
            ```";
        let upds = extract_status_updates(txt);
        assert_eq!(upds.len(), 1);
        assert!(store.apply_status(&upds[0]));
        assert_eq!(store.workflows[0].steps[1].state, StepState::Active);
        assert_eq!(store.workflows[0].steps[1].output.as_deref(), Some("writing CacheBackend trait"));
    }

    #[test]
    fn task_only_status_synthesises_workflow_of_one() {
        let upds = extract_status_updates("```status\ntask: ship sidebar\nstate: done\ndetail: yes\n```");
        assert_eq!(upds.len(), 1);
        assert_eq!(upds[0].workflow, "ship sidebar");
        assert_eq!(upds[0].step, "ship sidebar");
        assert_eq!(upds[0].state, Some(StepState::Done));
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = std::env::temp_dir().join(format!("larql-wf-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("wf.json");
        let mut a = WorkflowStore::default();
        a.upsert(Workflow {
            name: "ship sidebar".into(),
            state: WorkflowState::Active,
            steps: vec![
                Step { description: "data model".into(), state: StepState::Done, output: None },
                Step { description: "render".into(), state: StepState::Active, output: None },
            ],
            ts: 17,
        });
        a.save(&path).unwrap();
        let b = WorkflowStore::load(&path);
        assert_eq!(b.workflows.len(), 1);
        assert_eq!(b.workflows[0].name, "ship sidebar");
        assert_eq!(b.workflows[0].steps[0].state, StepState::Done);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn replan_replaces_existing() {
        let mut store = WorkflowStore::default();
        store.upsert(Workflow {
            name: "x".into(),
            state: WorkflowState::Active,
            steps: vec![Step { description: "old".into(), state: StepState::Done, output: None }],
            ts: 0,
        });
        let new_plans =
            extract_plan_blocks("```plan\nworkflow: x\nstep: new1\nstep: new2\n```");
        for w in new_plans {
            store.upsert(w);
        }
        assert_eq!(store.workflows.len(), 1);
        assert_eq!(store.workflows[0].steps.len(), 2);
        assert_eq!(store.workflows[0].steps[0].description, "new1");
    }
}
