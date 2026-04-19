use serde::Serialize;

// ── Types ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub enum Message {
    User(String),
    Assistant(String),
    System(String),
    ToolUse { tool: String, detail: String },
    ToolResult { summary: String },
}

pub enum StreamEvent {
    Token(String),
    Done,
    Error(String),
}

pub struct AppState {
    pub input: String,
    pub cursor: usize,
    pub messages: Vec<Message>,
    pub status: String,
    pub is_generating: bool,
    pub server_url: String,
    pub workflows: Vec<crate::workflows::Workflow>,
    pub workflow_store: crate::workflows::WorkflowStore,
    pub logger: crate::log::Logger,
    pub sidebar_visible: bool,
}

impl AppState {
    pub fn new(server_url: &str) -> Self {
        let workflow_store = crate::workflows::WorkflowStore::new();
        let workflows = workflow_store.load();
        Self {
            input: String::new(),
            cursor: 0,
            messages: vec![Message::System(
                "larql — LLM as a Database. Type questions, use tools.".into(),
            )],
            status: format!("connecting to {server_url}..."),
            is_generating: false,
            server_url: server_url.to_string(),
            workflows,
            workflow_store,
            logger: crate::log::Logger::new(),
            sidebar_visible: true,
        }
    }

    /// Build chat messages — "system prompt" trigger + user message.
    /// The phrase "system prompt" is a KNN trigger that the model's
    /// residual at L26 matches against the stored expansion.
    /// When matched, the KNN overlay force-injects the full annotation
    /// instructions (~124 tokens) into the decode stream.
    /// No system prompt stuffing — the model gets the instructions
    /// through its own weights/activations.
    pub fn build_chat_messages(&self) -> Vec<ChatMsg> {
        let mut msgs = Vec::new();
        // System message with just the trigger phrase
        msgs.push(ChatMsg {
            role: "system".into(),
            content: "system prompt".into(),
        });
        if let Some(msg) = self
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m, Message::User(_)))
        {
            if let Message::User(text) = msg {
                msgs.push(ChatMsg {
                    role: "user".into(),
                    content: text.clone(),
                });
            }
        }
        msgs
    }

    /// Insert stored facts into the model via /v1/insert (KNN overlay).
    /// Called once at startup — facts live in the model, not the prompt.
    pub async fn insert_facts_into_model(&self) {
        let client = reqwest::Client::new();
        let facts_path = home_dir().join(".larql/facts.jsonl");

        if let Ok(content) = std::fs::read_to_string(&facts_path) {
            for line in content.lines() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    let key = v["key"].as_str().unwrap_or("");
                    let value = v["value"].as_str().unwrap_or("");
                    if key.is_empty() || value.is_empty() { continue; }

                    let _ = client
                        .post(format!("{}/v1/insert", self.server_url))
                        .json(&serde_json::json!({
                            "entity": key,
                            "relation": "is",
                            "target": value,
                        }))
                        .timeout(std::time::Duration::from_secs(5))
                        .send()
                        .await;
                }
            }
        }
    }
}

// ── HTTP types ──────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMsg>,
    pub stream: bool,
}

#[derive(Serialize, Clone)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

// ── Helpers ─────────────────────────────────────────────────────────────

pub fn home_dir() -> std::path::PathBuf {
    std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
}

pub fn chrono_now() -> String {
    // Simple ISO timestamp without chrono dependency
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", d.as_secs())
}
