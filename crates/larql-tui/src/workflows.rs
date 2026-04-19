use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StepState {
    Pending,
    Active,
    Done,
    Blocked,
}

impl std::fmt::Display for StepState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            StepState::Pending => write!(f, "pending"),
            StepState::Active => write!(f, "active"),
            StepState::Done => write!(f, "done"),
            StepState::Blocked => write!(f, "blocked"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub description: String,
    pub state: StepState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Workflow {
    pub name: String,
    pub state: StepState,
    pub steps: Vec<WorkflowStep>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    pub ts: String,
}

#[derive(Serialize, Deserialize)]
struct WorkflowFile {
    workflows: Vec<Workflow>,
}

pub struct WorkflowStore {
    path: PathBuf,
}

impl WorkflowStore {
    pub fn new() -> Self {
        Self {
            path: crate::app::home_dir().join(".larql/workflows.json"),
        }
    }

    pub fn load(&self) -> Vec<Workflow> {
        let content = match std::fs::read_to_string(&self.path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        // Try new format first
        if let Ok(wf) = serde_json::from_str::<WorkflowFile>(&content) {
            return wf.workflows;
        }
        // Fall back to old flat format: { "task": { "state": "...", "detail": "..." } }
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(map) = obj.as_object() {
                return map
                    .iter()
                    .map(|(name, v)| {
                        let state_str = v["state"].as_str().unwrap_or("pending");
                        let state = match state_str {
                            "active" => StepState::Active,
                            "done" => StepState::Done,
                            "blocked" => StepState::Blocked,
                            _ => StepState::Pending,
                        };
                        let detail = v["detail"].as_str().unwrap_or("").to_string();
                        Workflow {
                            name: name.clone(),
                            state: state.clone(),
                            steps: if detail.is_empty() {
                                vec![]
                            } else {
                                vec![WorkflowStep {
                                    description: detail,
                                    state: state.clone(),
                                    output: None,
                                }]
                            },
                            parent: None,
                            ts: v["ts"].as_str().unwrap_or("0").to_string(),
                        }
                    })
                    .collect();
            }
        }
        Vec::new()
    }

    pub fn save(&self, workflows: &[Workflow]) {
        let wf = WorkflowFile {
            workflows: workflows.to_vec(),
        };
        if let Ok(json) = serde_json::to_string_pretty(&wf) {
            let _ = std::fs::write(&self.path, json);
        }
    }

    /// Create a new workflow from a plan block.
    pub fn create_from_plan(&self, name: &str, step_descs: Vec<String>) -> Workflow {
        Workflow {
            name: name.to_string(),
            state: StepState::Pending,
            steps: step_descs
                .into_iter()
                .map(|desc| WorkflowStep {
                    description: desc,
                    state: StepState::Pending,
                    output: None,
                })
                .collect(),
            parent: None,
            ts: crate::app::chrono_now(),
        }
    }

    /// Update a specific step in a workflow.
    pub fn update_step(
        workflows: &mut [Workflow],
        workflow_name: &str,
        step_num: usize,
        state: StepState,
        detail: Option<String>,
    ) {
        if let Some(wf) = workflows.iter_mut().find(|w| w.name == workflow_name) {
            if step_num > 0 && step_num <= wf.steps.len() {
                wf.steps[step_num - 1].state = state;
                if let Some(d) = detail {
                    wf.steps[step_num - 1].output = Some(d);
                }
            }
            // Update workflow overall state
            let all_done = wf.steps.iter().all(|s| s.state == StepState::Done);
            let any_active = wf.steps.iter().any(|s| s.state == StepState::Active);
            let any_blocked = wf.steps.iter().any(|s| s.state == StepState::Blocked);
            wf.state = if all_done {
                StepState::Done
            } else if any_blocked {
                StepState::Blocked
            } else if any_active {
                StepState::Active
            } else {
                StepState::Pending
            };
        }
    }

    /// Update or create a flat workflow (legacy status block without steps).
    pub fn upsert_flat(
        workflows: &mut Vec<Workflow>,
        name: &str,
        state: StepState,
        detail: &str,
    ) {
        if let Some(wf) = workflows.iter_mut().find(|w| w.name == name) {
            wf.state = state;
            if !wf.steps.is_empty() {
                // Update the last active or first pending step
                if let Some(step) = wf.steps.iter_mut().find(|s| s.state == StepState::Active) {
                    step.output = Some(detail.to_string());
                }
            } else {
                wf.steps = vec![WorkflowStep {
                    description: detail.to_string(),
                    state: wf.state.clone(),
                    output: None,
                }];
            }
            wf.ts = crate::app::chrono_now();
        } else {
            workflows.push(Workflow {
                name: name.to_string(),
                state,
                steps: vec![WorkflowStep {
                    description: detail.to_string(),
                    state: StepState::Active,
                    output: None,
                }],
                parent: None,
                ts: crate::app::chrono_now(),
            });
        }
    }
}
