use crate::annotations::extract_block;
use crate::app::{home_dir, Message};

pub fn execute_skill_tool(text: &str, messages: &mut Vec<Message>) -> Option<String> {
    let open = "```tool";
    let start = text.find(open)?;
    let after = &text[start + open.len()..];
    let close = after.find("```")?;
    let tool_call = after[..close].trim();

    let parts: Vec<&str> = tool_call.splitn(2, char::is_whitespace).collect();
    let skill_name = parts.first()?;
    let skill_args = parts.get(1).unwrap_or(&"");

    let skills_dirs = vec![
        std::env::current_dir().unwrap_or_default().join(".skills"),
        home_dir().join(".larql/skills"),
    ];
    let mut tool_path = None;
    for dir in &skills_dirs {
        let candidate = dir.join(skill_name).join("tool.sh");
        if candidate.exists() {
            tool_path = Some(candidate);
            break;
        }
    }

    let path = tool_path?;
    messages.push(Message::ToolUse {
        tool: format!("{skill_name}"),
        detail: skill_args.chars().take(70).collect(),
    });

    match std::process::Command::new("bash")
        .arg(&path)
        .args(skill_args.split_whitespace())
        .output()
    {
        Ok(output) => {
            let tool_output = String::from_utf8_lossy(&output.stdout).to_string();

            if let Some(summary) = extract_block(&tool_output, "summary") {
                messages.push(Message::ToolResult {
                    summary: summary.clone(),
                });
                if let Some(chart) = extract_block(&tool_output, "chartjs") {
                    let chart_md = format!("```chartjs\n{chart}\n```");
                    messages.push(Message::ToolResult { summary: chart_md });
                }
                return Some(summary);
            }
            None
        }
        Err(e) => {
            messages.push(Message::System(format!("tool error: {e}")));
            None
        }
    }
}
