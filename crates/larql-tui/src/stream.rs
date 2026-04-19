use crate::app::{ChatMsg, ChatRequest, StreamEvent};

/// Send chat messages to the server, streaming tokens via `tx`.
pub async fn chat_stream(
    url: &str,
    messages: Vec<ChatMsg>,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let req = ChatRequest {
        model: std::env::var("LARQL_MODEL").unwrap_or_else(|_| "gemma-3-4b".into()),
        messages,
        stream: true,
    };

    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&req)
        .send()
        .await
        .map_err(|e| format!("connection failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("server error: {}", resp.status()));
    }

    use futures::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("stream error: {e}"))?;
        let text = String::from_utf8_lossy(&chunk);
        buf.push_str(&text);

        while let Some(newline_pos) = buf.find('\n') {
            let line = buf[..newline_pos].to_string();
            buf = buf[newline_pos + 1..].to_string();

            let line = line.trim();
            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    return Ok(());
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(content) = v["choices"][0]["delta"]["content"].as_str() {
                        let _ = tx.send(StreamEvent::Token(content.to_string())).await;
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn spawn_chat(
    url: String,
    messages: Vec<ChatMsg>,
    tx: tokio::sync::mpsc::Sender<StreamEvent>,
) {
    tokio::spawn(async move {
        match chat_stream(&url, messages, &tx).await {
            Ok(()) => {
                let _ = tx.send(StreamEvent::Done).await;
            }
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(e)).await;
            }
        }
    });
}
