use std::io;

use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;

use crate::app::{AppState, Message};
use crate::workflows::StepState;

pub fn draw(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, state: &AppState) {
    terminal
        .draw(|f| {
            // Optional horizontal split for sidebar
            let (sidebar_area, main_area) =
                if state.sidebar_visible && !state.workflows.is_empty() && f.area().width >= 60 {
                    let h = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([
                            Constraint::Length(30), // sidebar width
                            Constraint::Min(40),    // main area
                        ])
                        .split(f.area());
                    (Some(h[0]), h[1])
                } else {
                    (None, f.area())
                };

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(5),
                    Constraint::Length(3),
                    Constraint::Length(1),
                ])
                .split(main_area);

            if let Some(sb) = sidebar_area {
                draw_sidebar(f, state, sb);
            }

            draw_messages(f, state, chunks[0]);
            draw_input(f, state, chunks[1]);
            draw_status(f, state, chunks[2]);
        })
        .ok();
}

fn draw_messages(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();

    for msg in &state.messages {
        match msg {
            Message::User(text) => {
                lines.push(Line::from(vec![
                    Span::styled(
                        "❯ ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        text.as_str(),
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));
                lines.push(Line::from(""));
            }
            Message::Assistant(text) => {
                lines.extend(gc_markdown::render_markdown(text, gc_markdown::Theme::Dark));
                lines.push(Line::from(""));
            }
            Message::System(text) => {
                lines.push(Line::from(Span::styled(
                    format!("  {text}"),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC),
                )));
                lines.push(Line::from(""));
            }
            Message::ToolUse { tool, detail } => {
                lines.push(Line::from(vec![
                    Span::styled("  ⚡ ", Style::default().fg(Color::Magenta)),
                    Span::styled(
                        tool.as_str(),
                        Style::default()
                            .fg(Color::Magenta)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" {detail}"),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
            Message::ToolResult { summary } => {
                lines.extend(gc_markdown::render_markdown(
                    summary,
                    gc_markdown::Theme::Dark,
                ));
                lines.push(Line::from(""));
            }
        }
    }

    // Scroll: estimate wrapped line count for proper auto-scroll.
    // Each Line can wrap across multiple screen rows. Use the inner
    // width (area minus borders) to approximate the wrapped height.
    let inner_width = area.width.saturating_sub(2) as usize;
    let wrapped_height: usize = lines
        .iter()
        .map(|line| {
            let content_len: usize = line.spans.iter().map(|s| s.content.len()).sum();
            if content_len == 0 {
                1
            } else {
                content_len.div_ceil(inner_width.max(1))
            }
        })
        .sum();
    let visible = area.height.saturating_sub(2) as usize;
    let scroll = if wrapped_height > visible {
        (wrapped_height - visible) as u16
    } else {
        0
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " larql ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));

    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    f.render_widget(para, area);
}

fn draw_input(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let (style, text) = if state.is_generating {
        (
            Style::default().fg(Color::DarkGray),
            "  generating...".to_string(),
        )
    } else if state.input.is_empty() {
        (
            Style::default().fg(Color::DarkGray),
            "  Type a question...".to_string(),
        )
    } else {
        (
            Style::default().fg(Color::White),
            format!("  {}", state.input),
        )
    };

    let block = Block::default().borders(Borders::ALL).border_style(
        Style::default().fg(if state.is_generating {
            Color::DarkGray
        } else {
            Color::Cyan
        }),
    );
    let para = Paragraph::new(Line::from(Span::styled(text, style))).block(block);
    f.render_widget(para, area);

    if !state.is_generating {
        f.set_cursor_position((area.x + 3 + state.cursor as u16, area.y + 1));
    }
}

fn draw_status(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let status = format!(" {} ", state.status);
    let para = Paragraph::new(Line::from(Span::styled(
        status,
        Style::default().fg(Color::White).bg(Color::DarkGray),
    )));
    f.render_widget(para, area);
}

fn draw_sidebar(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();

    for wf in &state.workflows {
        // Workflow header: icon + name + progress
        let icon = match wf.state {
            StepState::Active => "\u{1f504}",
            StepState::Done => "\u{2705}",
            StepState::Blocked => "\u{1f6ab}",
            StepState::Pending => "\u{23f3}",
        };
        let done_count = wf.steps.iter().filter(|s| s.state == StepState::Done).count();
        let total = wf.steps.len();
        let progress = if total > 0 {
            format!(" [{done_count}/{total}]")
        } else {
            String::new()
        };

        let header_style = match wf.state {
            StepState::Active => Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            StepState::Done => Style::default().fg(Color::Green),
            StepState::Blocked => Style::default().fg(Color::Red),
            _ => Style::default().fg(Color::White),
        };

        lines.push(Line::from(vec![
            Span::raw(format!("{icon} ")),
            Span::styled(&wf.name, header_style),
            Span::styled(progress, Style::default().fg(Color::DarkGray)),
        ]));

        // Show steps for active/pending workflows (collapse done ones)
        if wf.state != StepState::Done {
            for (i, step) in wf.steps.iter().enumerate() {
                let step_icon = match step.state {
                    StepState::Done => "  \u{2705}",
                    StepState::Active => "  \u{1f504}",
                    StepState::Blocked => "  \u{1f6ab}",
                    StepState::Pending => "  \u{23f3}",
                };
                let step_style = match step.state {
                    StepState::Active => Style::default().fg(Color::Yellow),
                    StepState::Done => Style::default().fg(Color::DarkGray),
                    _ => Style::default().fg(Color::White),
                };
                // Truncate step description to fit sidebar
                let max_desc_len = 20; // sidebar is 30 wide, minus icon + number
                let desc: String = step.description.chars().take(max_desc_len).collect();
                lines.push(Line::from(vec![
                    Span::raw(format!("{step_icon} ")),
                    Span::styled(format!("{}. {desc}", i + 1), step_style),
                ]));
            }
        }

        lines.push(Line::raw("")); // spacer between workflows
    }

    if lines.is_empty() {
        lines.push(Line::styled(
            "  No active workflows",
            Style::default().fg(Color::DarkGray),
        ));
    }

    let block = Block::default()
        .title(" Workflows ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });

    f.render_widget(paragraph, area);
}
