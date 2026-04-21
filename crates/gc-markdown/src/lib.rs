// Markdown renderer for ratatui
//
// Parses markdown text into ratatui Lines/Spans with proper styling.
// Handles standard markdown (via tui-markdown) plus custom fenced blocks:
//   - ```chartjs  → ASCII bar chart via ratatui's built-in BarChart
//   - ```diff     → red/green colored lines
//   - ```terminal → dimmed command output
//   - ```mermaid  → rendered as-is with box border (future: actual rendering)
//   - ```csv      → parsed into a styled table

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Render markdown string into styled ratatui Lines.
/// Splits text into blocks, renders standard markdown via tui-markdown,
/// and handles custom fenced blocks with specialized renderers.
fn trace(msg: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/gaucho-render.log") {
        let _ = writeln!(f, "{msg}");
    }
}

pub fn render_markdown(text: &str, theme: Theme) -> Vec<Line<'static>> {
    // Take first ~100 CHARACTERS (not bytes!) for the log — slicing
    // by byte index panics when the cut lands inside a multi-byte
    // UTF-8 sequence (e.g. the `·` separator in our tool summaries).
    let preview: String = text.chars().take(100).collect();
    trace(&format!("[render_markdown] input len={}, first 100 chars: {:?}", text.len(), preview));
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut i = 0;
    let raw_lines: Vec<&str> = text.lines().collect();

    while i < raw_lines.len() {
        // Check for markdown pipe table (collect consecutive | rows)
        if raw_lines[i].trim_start().starts_with('|') && raw_lines[i].trim_end().ends_with('|') {
            let mut table_rows: Vec<&str> = Vec::new();
            while i < raw_lines.len() {
                let trimmed = raw_lines[i].trim();
                if trimmed.starts_with('|') && trimmed.ends_with('|') {
                    table_rows.push(trimmed);
                    i += 1;
                } else {
                    break;
                }
            }
            lines.extend(render_pipe_table(&table_rows, theme));
            lines.push(Line::from(""));
            continue;
        }

        // Check for fenced code block
        if raw_lines[i].trim_start().starts_with("```") {
            let lang = raw_lines[i].trim_start().trim_start_matches('`').trim().to_lowercase();
            trace(&format!("[render_markdown] fenced block detected: lang={:?}", lang));
            i += 1;

            // Collect block content
            let mut block_content = String::new();
            while i < raw_lines.len() && !raw_lines[i].trim_start().starts_with("```") {
                block_content.push_str(raw_lines[i]);
                block_content.push('\n');
                i += 1;
            }
            if i < raw_lines.len() { i += 1; } // skip closing ```

            trace(&format!("[render_markdown] block content len={}, rendering as {:?}", block_content.len(), lang));
            let block_lines = render_fenced_block(&lang, &block_content, theme);
            trace(&format!("[render_markdown] block produced {} lines", block_lines.len()));
            lines.extend(block_lines);
            lines.push(Line::from(""));
            continue;
        }

        // Standard markdown line
        let line = raw_lines[i];
        lines.push(render_markdown_line(line, theme));
        i += 1;
    }

    lines
}

#[derive(Clone, Copy)]
pub enum Theme {
    Dark,
    Light,
}

// ── Fenced block renderers ─────────────────────────────────────────────────

fn render_fenced_block(lang: &str, content: &str, theme: Theme) -> Vec<Line<'static>> {
    match lang {
        "chartjs" => render_chartjs(content, theme),
        "diff" => render_diff(content, theme),
        "terminal" => render_terminal(content, theme),
        "csv" => render_csv(content, theme),
        "json" => render_code(content, "json", theme),
        "mermaid" => render_boxed(content, "mermaid", theme),
        "math" => render_boxed(content, "math", theme),
        _ => render_code(content, lang, theme),
    }
}

// ── chartjs: ASCII bar chart ───────────────────────────────────────────────

// Palette of distinct colors for pie/chart slices
const SLICE_COLORS: &[Color] = &[
    Color::Rgb(100, 180, 255),  // blue
    Color::Rgb(255, 160, 80),   // orange
    Color::Rgb(100, 200, 130),  // green
    Color::Rgb(220, 100, 100),  // red
    Color::Rgb(180, 130, 255),  // purple
    Color::Rgb(255, 220, 80),   // yellow
    Color::Rgb(100, 220, 220),  // cyan
    Color::Rgb(255, 130, 180),  // pink
    Color::Rgb(160, 200, 80),   // lime
    Color::Rgb(200, 160, 120),  // tan
    Color::Rgb(140, 140, 160),  // gray
];

/// Merge entries to fit a target count by combining adjacent pairs.
fn merge_entries(entries: &[(String, f64)], target: usize) -> Vec<(String, f64)> {
    if entries.len() <= target { return entries.to_vec(); }

    let step = (entries.len() + target - 1) / target; // ceil division
    let mut merged = Vec::new();
    for chunk in entries.chunks(step) {
        let sum: f64 = chunk.iter().map(|(_, v)| *v).sum();
        let label = if chunk.len() == 1 {
            chunk[0].0.clone()
        } else {
            // Use first and last label
            format!("{}..{}", chunk.first().unwrap().0, chunk.last().unwrap().0)
        };
        merged.push((label, sum));
    }
    merged
}

fn render_chartjs(content: &str, theme: Theme) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let (fg, label_color) = match theme {
        Theme::Dark => (Color::White, Color::Rgb(180, 180, 180)),
        Theme::Light => (Color::Black, Color::Rgb(80, 80, 80)),
    };

    let parsed: Result<serde_json::Value, _> = serde_json::from_str(content);
    let Ok(chart) = parsed else {
        lines.push(Line::from(Span::styled("  (invalid chart data)", Style::default().fg(Color::Red))));
        return lines;
    };

    let labels = chart["data"]["labels"].as_array();
    let datasets = chart["data"]["datasets"].as_array();
    let title = chart["options"]["title"].as_str();

    if let Some(title) = title {
        lines.push(Line::from(Span::styled(
            format!("  📊 {title}"),
            Style::default().fg(fg).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
    }

    let Some(labels) = labels else { return lines };
    let Some(datasets) = datasets else { return lines };
    let Some(data) = datasets.first().and_then(|d| d["data"].as_array()) else { return lines };

    let entries: Vec<(String, f64)> = labels.iter().zip(data.iter())
        .filter_map(|(l, v)| {
            let label = l.as_str().map(|s| s.to_string())
                .or_else(|| Some(l.to_string()))?;
            let val = v.as_f64()?;
            Some((label, val))
        })
        .collect();

    if entries.is_empty() { return lines; }

    // Adaptive: if too many entries, merge adjacent pairs to fit terminal
    let entries = if entries.len() > 24 {
        merge_entries(&entries, 20)
    } else {
        entries
    };

    let total: f64 = entries.iter().map(|(_, v)| *v).sum();
    let chart_type = chart["type"].as_str().unwrap_or("bar");

    // Dispatch by chart type
    if chart_type == "bar" {
        // ── Bar chart only ─────────────────────────────────────────
        let max_val = entries.iter().map(|(_, v)| *v).fold(0.0_f64, f64::max);
        let max_label_len = entries.iter().map(|(l, _)| l.len()).max().unwrap_or(10).min(25);
        let bar_width = 35u16;

        for (i, (label, val)) in entries.iter().enumerate() {
            let frac = if max_val > 0.0 { val / max_val } else { 0.0 };
            let filled = ((frac * bar_width as f64).round() as u16).max(if *val > 0.0 { 1 } else { 0 });
            let bar_str = "█".repeat(filled as usize);
            let padded_label = format!("{:>width$}", truncate_label(label, max_label_len), width = max_label_len);
            let color = SLICE_COLORS[i % SLICE_COLORS.len()];

            lines.push(Line::from(vec![
                Span::styled(format!("  {padded_label} "), Style::default().fg(label_color)),
                Span::styled(bar_str, Style::default().fg(color)),
                Span::styled(format!(" {}", format_value(*val)), Style::default().fg(label_color)),
            ]));
        }

        return lines;
    }

    // ── Pie chart (circular using half-blocks) ─────────────────────
    let radius = 6i32;
    let pie_lines = render_pie_circle(&entries, total, radius);
    // Legend beside the pie
    let legend_start = 2; // vertical offset to start legend
    let pie_width = (radius * 4 + 4) as usize; // approximate char width of pie

    for (row, pie_line_spans) in pie_lines.iter().enumerate() {
        let mut spans = vec![Span::styled("  ", Style::default())];
        spans.extend(pie_line_spans.clone());

        // Add legend item beside the pie
        let legend_idx = row as i32 - legend_start as i32;
        if legend_idx >= 0 && (legend_idx as usize) < entries.len() {
            let idx = legend_idx as usize;
            let (label, val) = &entries[idx];
            let color = SLICE_COLORS[idx % SLICE_COLORS.len()];
            let pct = if total > 0.0 { val / total * 100.0 } else { 0.0 };
            let val_str = format_value(*val);
            // Pad to align after pie
            let pad = pie_width.saturating_sub(pie_line_spans.iter().map(|s| s.content.len()).sum::<usize>());
            spans.push(Span::styled(" ".repeat(pad + 2), Style::default()));
            spans.push(Span::styled("■ ", Style::default().fg(color)));
            spans.push(Span::styled(
                format!("{} ({val_str}, {pct:.0}%)", truncate_label(label, 25)),
                Style::default().fg(label_color),
            ));
        }

        lines.push(Line::from(spans));
    }

    lines
}

/// Render a circular pie chart using Unicode half-block characters
fn render_pie_circle(entries: &[(String, f64)], total: f64, radius: i32) -> Vec<Vec<Span<'static>>> {
    let mut rows: Vec<Vec<Span<'static>>> = Vec::new();
    let height = radius * 2 + 1;
    let width = radius * 2 + 1;

    // Build angle→color map
    let mut angle_colors: Vec<(f64, Color)> = Vec::new();
    let mut cumulative = 0.0;
    for (i, (_, val)) in entries.iter().enumerate() {
        let fraction = if total > 0.0 { val / total } else { 0.0 };
        let end_angle = cumulative + fraction * 360.0;
        angle_colors.push((end_angle, SLICE_COLORS[i % SLICE_COLORS.len()]));
        cumulative = end_angle;
    }

    fn angle_to_color(angle: f64, map: &[(f64, Color)]) -> Color {
        let a = ((angle % 360.0) + 360.0) % 360.0;
        for (end, color) in map {
            if a <= *end { return *color; }
        }
        map.last().map(|(_, c)| *c).unwrap_or(Color::DarkGray)
    }

    for row in 0..height {
        let mut spans: Vec<Span<'static>> = Vec::new();
        for col in 0..width {
            // Map to circle coordinates (-1..1)
            let cx = (col - radius) as f64;
            let cy = (row - radius) as f64 * 2.0; // *2 because chars are taller than wide
            let dist = (cx * cx + cy * cy).sqrt();

            if dist <= radius as f64 {
                let angle = cy.atan2(cx).to_degrees() + 180.0; // 0-360
                let color = angle_to_color(angle, &angle_colors);
                spans.push(Span::styled("██", Style::default().fg(color)));
            } else {
                spans.push(Span::styled("  ", Style::default()));
            }
        }
        rows.push(spans);
    }

    rows
}

fn format_value(val: f64) -> String {
    if val >= 1024.0 * 1024.0 {
        format!("{:.1}G", val / (1024.0 * 1024.0))
    } else if val >= 1024.0 {
        format!("{:.0}M", val / 1024.0)
    } else {
        format!("{:.0}K", val)
    }
}

fn truncate_label(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() }
    else { format!("{}…", &s[..max - 1]) }
}

// ── diff: red/green colored ────────────────────────────────────────────────

fn render_diff(content: &str, _theme: Theme) -> Vec<Line<'static>> {
    content.lines().map(|line| {
        let style = if line.starts_with('+') && !line.starts_with("+++") {
            Style::default().fg(Color::Green)
        } else if line.starts_with('-') && !line.starts_with("---") {
            Style::default().fg(Color::Red)
        } else if line.starts_with("@@") {
            Style::default().fg(Color::Cyan)
        } else if line.starts_with("diff") || line.starts_with("index") {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        Line::from(Span::styled(format!("  {line}"), style))
    }).collect()
}

// ── terminal: dimmed command output ────────────────────────────────────────

fn render_terminal(content: &str, theme: Theme) -> Vec<Line<'static>> {
    let (cmd_color, output_color) = match theme {
        Theme::Dark => (Color::Rgb(100, 200, 100), Color::Rgb(160, 160, 160)),
        Theme::Light => (Color::Rgb(40, 120, 40), Color::Rgb(100, 100, 100)),
    };

    content.lines().map(|line| {
        if line.starts_with('$') || line.starts_with('>') {
            Line::from(Span::styled(format!("  {line}"), Style::default().fg(cmd_color).add_modifier(Modifier::BOLD)))
        } else {
            Line::from(Span::styled(format!("  {line}"), Style::default().fg(output_color)))
        }
    }).collect()
}

// ── csv: parse into table ──────────────────────────────────────────────────

fn render_csv(content: &str, theme: Theme) -> Vec<Line<'static>> {
    let (header_color, cell_color, border_color) = match theme {
        Theme::Dark => (Color::Rgb(100, 200, 255), Color::Rgb(200, 200, 200), Color::Rgb(80, 80, 90)),
        Theme::Light => (Color::Rgb(40, 100, 160), Color::Rgb(60, 60, 60), Color::Rgb(180, 180, 190)),
    };

    let rows: Vec<Vec<&str>> = content.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|c| c.trim()).collect())
        .collect();

    if rows.is_empty() { return vec![]; }

    let cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut col_widths = vec![0usize; cols];
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            if i < cols { col_widths[i] = col_widths[i].max(cell.len()); }
        }
    }

    let mut lines = Vec::new();
    for (row_idx, row) in rows.iter().enumerate() {
        let color = if row_idx == 0 { header_color } else { cell_color };
        let mut spans = vec![Span::styled("  ", Style::default())];
        for (i, cell) in row.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(" │ ", Style::default().fg(border_color)));
            }
            let w = col_widths.get(i).copied().unwrap_or(cell.len());
            spans.push(Span::styled(
                format!("{:<width$}", cell, width = w),
                Style::default().fg(color).add_modifier(if row_idx == 0 { Modifier::BOLD } else { Modifier::empty() }),
            ));
        }
        lines.push(Line::from(spans));

        // Separator after header
        if row_idx == 0 {
            let sep: String = col_widths.iter()
                .map(|w| "─".repeat(*w))
                .collect::<Vec<_>>()
                .join("─┼─");
            lines.push(Line::from(Span::styled(
                format!("  {sep}"),
                Style::default().fg(border_color),
            )));
        }
    }

    lines
}

// ── code: syntax-colored block ─────────────────────────────────────────────

fn render_code(content: &str, lang: &str, theme: Theme) -> Vec<Line<'static>> {
    let (border_color, code_color, lang_color) = match theme {
        Theme::Dark => (Color::Rgb(80, 80, 90), Color::Rgb(200, 200, 180), Color::Rgb(120, 120, 140)),
        Theme::Light => (Color::Rgb(180, 180, 190), Color::Rgb(50, 50, 40), Color::Rgb(140, 140, 150)),
    };

    let mut lines = Vec::new();
    if !lang.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("  ╭─ {lang} ─"),
            Style::default().fg(lang_color),
        )));
    }
    for line in content.lines() {
        lines.push(Line::from(vec![
            Span::styled("  │ ", Style::default().fg(border_color)),
            Span::styled(line.to_string(), Style::default().fg(code_color)),
        ]));
    }
    if !lang.is_empty() {
        lines.push(Line::from(Span::styled("  ╰─", Style::default().fg(lang_color))));
    }
    lines
}

// ── boxed: content in a border ─────────────────────────────────────────────

fn render_boxed(content: &str, label: &str, theme: Theme) -> Vec<Line<'static>> {
    let (border_color, text_color) = match theme {
        Theme::Dark => (Color::Rgb(100, 100, 120), Color::Rgb(200, 200, 200)),
        Theme::Light => (Color::Rgb(160, 160, 180), Color::Rgb(50, 50, 50)),
    };

    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        format!("  ┌─ {label} ─┐"),
        Style::default().fg(border_color),
    )));
    for line in content.lines() {
        lines.push(Line::from(vec![
            Span::styled("  │ ", Style::default().fg(border_color)),
            Span::styled(line.to_string(), Style::default().fg(text_color)),
        ]));
    }
    lines.push(Line::from(Span::styled("  └─────┘", Style::default().fg(border_color))));
    lines
}

// ── Standard markdown line ─────────────────────────────────────────────────

// ── Pipe table: collect all rows, compute column widths, render aligned ────

fn render_pipe_table(rows: &[&str], theme: Theme) -> Vec<Line<'static>> {
    let (fg, bold_color, dim_color, link_color) = match theme {
        Theme::Dark => (
            Color::Rgb(220, 220, 220), Color::Rgb(255, 255, 255),
            Color::Rgb(140, 140, 140), Color::Rgb(100, 180, 255),
        ),
        Theme::Light => (
            Color::Rgb(40, 40, 50), Color::Rgb(0, 0, 0),
            Color::Rgb(120, 120, 130), Color::Rgb(40, 100, 180),
        ),
    };

    // Parse all rows into cells
    let parsed: Vec<Vec<String>> = rows.iter().map(|row| {
        row.split('|')
            .filter(|c| !c.is_empty())
            .map(|c| c.trim().to_string())
            .collect()
    }).collect();

    // Find separator row (contains ---)
    let is_separator = |cells: &[String]| cells.iter().all(|c| c.contains("---") || c.contains(":-") || c.trim().is_empty());

    // Compute max width per column (ignoring separator rows, stripping markdown markers for width calc)
    let num_cols = parsed.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut col_widths = vec![0usize; num_cols];
    for row in &parsed {
        if is_separator(row) { continue; }
        for (i, cell) in row.iter().enumerate() {
            if i < num_cols {
                // Strip ** and ` for width calculation
                let clean_len = cell.replace("**", "").replace('`', "").len();
                col_widths[i] = col_widths[i].max(clean_len);
            }
        }
    }

    let mut lines = Vec::new();
    for (row_idx, row) in parsed.iter().enumerate() {
        if is_separator(row) {
            // Render separator
            let sep: String = col_widths.iter()
                .map(|w| "─".repeat(*w + 2))
                .collect::<Vec<_>>()
                .join("┼");
            lines.push(Line::from(Span::styled(format!("  {sep}"), Style::default().fg(dim_color))));
            continue;
        }

        let mut spans = vec![Span::styled("  ", Style::default())];
        for (i, cell) in row.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(" │ ", Style::default().fg(dim_color)));
            }
            let w = col_widths.get(i).copied().unwrap_or(cell.len());
            // Pad the cell content (strip markers for length calc, but render with formatting)
            let clean_len = cell.replace("**", "").replace('`', "").len();
            let padding = w.saturating_sub(clean_len);

            // Render inline markdown within the cell
            let cell_style = if row_idx == 0 { bold_color } else { fg };
            let cell_mod = if row_idx == 0 { Modifier::BOLD } else { Modifier::empty() };

            // For header row, render bold directly; for data rows, parse inline
            if row_idx == 0 {
                let cleaned = cell.replace("**", "");
                spans.push(Span::styled(cleaned, Style::default().fg(cell_style).add_modifier(cell_mod)));
            } else {
                spans.extend(render_inline_spans(cell, fg, bold_color, link_color));
            }
            if padding > 0 {
                spans.push(Span::styled(" ".repeat(padding), Style::default()));
            }
        }
        lines.push(Line::from(spans));
    }

    lines
}

fn render_markdown_line(line: &str, theme: Theme) -> Line<'static> {
    let (fg, bold_color, dim_color, heading_color, link_color) = match theme {
        Theme::Dark => (
            Color::Rgb(220, 220, 220),
            Color::Rgb(255, 255, 255),
            Color::Rgb(140, 140, 140),
            Color::Rgb(100, 200, 255),
            Color::Rgb(100, 180, 255),
        ),
        Theme::Light => (
            Color::Rgb(40, 40, 50),
            Color::Rgb(0, 0, 0),
            Color::Rgb(120, 120, 130),
            Color::Rgb(30, 90, 160),
            Color::Rgb(40, 100, 180),
        ),
    };

    let trimmed = line.trim_start();

    // Headers
    if trimmed.starts_with("### ") {
        return Line::from(Span::styled(
            format!("  ▸ {}", &trimmed[4..]),
            Style::default().fg(heading_color).add_modifier(Modifier::BOLD),
        ));
    }
    if trimmed.starts_with("## ") {
        return Line::from(Span::styled(
            format!("  ◆ {}", &trimmed[3..]),
            Style::default().fg(heading_color).add_modifier(Modifier::BOLD),
        ));
    }
    if trimmed.starts_with("# ") {
        return Line::from(Span::styled(
            format!("  ★ {}", &trimmed[2..]),
            Style::default().fg(heading_color).add_modifier(Modifier::BOLD),
        ));
    }

    // Horizontal rule
    if trimmed == "---" || trimmed == "***" || trimmed == "___" {
        return Line::from(Span::styled("  ────────────────────────────", Style::default().fg(dim_color)));
    }

    // List items
    if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
        let content = &trimmed[2..];
        let mut spans = vec![Span::styled("  • ", Style::default().fg(dim_color))];
        spans.extend(render_inline_spans(content, fg, bold_color, link_color));
        return Line::from(spans);
    }

    // Numbered list
    if trimmed.len() > 2 && trimmed.chars().next().unwrap_or(' ').is_ascii_digit() && trimmed.contains(". ") {
        let dot_pos = trimmed.find(". ").unwrap_or(0);
        let num = &trimmed[..dot_pos + 1];
        let content = &trimmed[dot_pos + 2..];
        let mut spans = vec![Span::styled(format!("  {num} "), Style::default().fg(dim_color))];
        spans.extend(render_inline_spans(content, fg, bold_color, link_color));
        return Line::from(spans);
    }

    // Table rows are now handled as blocks in render_markdown (collected above)

    // Italic/emphasis line: *text*
    if trimmed.starts_with('*') && trimmed.ends_with('*') && !trimmed.starts_with("**") {
        let inner = trimmed.trim_matches('*');
        return Line::from(Span::styled(
            format!("  {inner}"),
            Style::default().fg(dim_color).add_modifier(Modifier::ITALIC),
        ));
    }

    // Empty line
    if trimmed.is_empty() {
        return Line::from("");
    }

    // Regular paragraph with inline formatting
    let mut spans = vec![Span::styled("  ", Style::default())];
    spans.extend(render_inline_spans(trimmed, fg, bold_color, link_color));
    Line::from(spans)
}

/// Parse inline markdown into multiple styled spans: **bold**, `code`, plain text
fn render_inline_spans(text: &str, fg: Color, bold_color: Color, _link_color: Color) -> Vec<Span<'static>> {
    let code_color = Color::Rgb(200, 180, 100);
    let mut spans: Vec<Span<'static>> = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut buf = String::new();

    while i < len {
        // **bold**
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            // Flush buffer
            if !buf.is_empty() {
                spans.push(Span::styled(buf.clone(), Style::default().fg(fg)));
                buf.clear();
            }
            i += 2;
            let mut bold_buf = String::new();
            while i + 1 < len && !(chars[i] == '*' && chars[i + 1] == '*') {
                bold_buf.push(chars[i]);
                i += 1;
            }
            if i + 1 < len { i += 2; } // skip closing **
            spans.push(Span::styled(bold_buf, Style::default().fg(bold_color).add_modifier(Modifier::BOLD)));
            continue;
        }

        // `code`
        if chars[i] == '`' {
            // Flush buffer
            if !buf.is_empty() {
                spans.push(Span::styled(buf.clone(), Style::default().fg(fg)));
                buf.clear();
            }
            i += 1;
            let mut code_buf = String::new();
            while i < len && chars[i] != '`' {
                code_buf.push(chars[i]);
                i += 1;
            }
            if i < len { i += 1; } // skip closing `
            spans.push(Span::styled(code_buf, Style::default().fg(code_color)));
            continue;
        }

        buf.push(chars[i]);
        i += 1;
    }

    if !buf.is_empty() {
        spans.push(Span::styled(buf, Style::default().fg(fg)));
    }

    if spans.is_empty() {
        spans.push(Span::styled(text.to_string(), Style::default().fg(fg)));
    }

    spans
}
