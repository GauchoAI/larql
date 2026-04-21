//! Colour palette + theme selector.  Mirrors the structure used in
//! gaucho-code's gc-tui so both projects share a visual vocabulary.
//!
//! Two themes ship: `Dark` (default) and `Light` (for users who said
//! "the font is near-white, make it darker").  Switch at runtime
//! with Ctrl-L.

use ratatui::style::Color;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Theme {
    Dark,
    Light,
}

impl Theme {
    pub fn toggle(self) -> Self {
        match self {
            Theme::Dark => Theme::Light,
            Theme::Light => Theme::Dark,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Theme::Dark => "dark",
            Theme::Light => "light",
        }
    }
    /// Map onto gc-markdown's own Theme enum so the markdown renderer
    /// picks matching colours for body text / headings / links.
    pub fn gcm(self) -> gc_markdown::Theme {
        match self {
            Theme::Dark => gc_markdown::Theme::Dark,
            Theme::Light => gc_markdown::Theme::Light,
        }
    }
}

pub struct Palette {
    /// Background for every panel.  Dark mode lets the terminal's
    /// native bg show through (`Color::Reset`); Light mode forces a
    /// near-white bg so the dark text is readable even when the
    /// terminal itself is dark.
    pub bg: Color,
    /// Primary text colour (user-visible prose).  Gaucho-code's dark
    /// mode uses 220/220/220 — a touch dimmer than pure white, which
    /// is what our user asked for.
    pub fg: Color,
    /// Bolder emphasis — headings, names.
    pub fg_bold: Color,
    /// Dim secondary text (pipes, markers, ghost hints).
    pub fg_dim: Color,
    /// Muted mid-brightness (tool detail, inline metadata).
    pub fg_muted: Color,
    /// User-prompt glyph (❯) + user turns.
    pub user: Color,
    /// Accent — titles, headings, highlights.
    pub accent: Color,
    /// Borders (chat + sidebar).
    pub border: Color,
    /// ⚡ tool markers.
    pub tool: Color,
    /// Error / failed runs.
    pub error: Color,
    /// Success (done steps, exit-0).
    pub ok: Color,
    /// In-progress (active step).
    pub active: Color,
    /// Status bar background + foreground.
    pub status_bg: Color,
    pub status_fg: Color,
    /// Selected tab (background + foreground).
    pub tab_selected_bg: Color,
    pub tab_selected_fg: Color,
    pub tab_idle_bg: Color,
    pub tab_idle_fg: Color,
}

pub fn palette(theme: Theme) -> Palette {
    match theme {
        Theme::Dark => Palette {
            bg:             Color::Reset,   // use terminal's native dark bg
            fg:             Color::Rgb(220, 220, 220),
            fg_bold:        Color::Rgb(255, 255, 255),
            fg_dim:         Color::Rgb(110, 115, 125),
            fg_muted:       Color::Rgb(160, 160, 170),
            user:           Color::Rgb(100, 200, 255),
            accent:         Color::Rgb(100, 180, 255),
            border:         Color::Rgb(80, 140, 200),
            tool:           Color::Rgb(255, 200, 80),
            error:          Color::Rgb(255, 100, 100),
            ok:             Color::Rgb(120, 220, 140),
            active:         Color::Rgb(255, 200, 80),
            status_bg:      Color::Rgb(40, 80, 120),
            status_fg:      Color::Rgb(200, 220, 240),
            tab_selected_bg: Color::Rgb(60, 130, 200),
            tab_selected_fg: Color::Rgb(255, 255, 255),
            tab_idle_bg:    Color::Rgb(40, 45, 55),
            tab_idle_fg:    Color::Rgb(180, 185, 195),
        },
        Theme::Light => Palette {
            bg:             Color::Rgb(248, 248, 250), // force light bg even on dark terminal
            fg:             Color::Rgb(30, 30, 40),
            fg_bold:        Color::Rgb(0, 0, 0),
            fg_dim:         Color::Rgb(140, 140, 155),
            fg_muted:       Color::Rgb(85, 85, 95),
            user:           Color::Rgb(30, 90, 180),
            accent:         Color::Rgb(40, 100, 180),
            border:         Color::Rgb(140, 170, 200),
            tool:           Color::Rgb(170, 110, 20),
            error:          Color::Rgb(200, 50, 50),
            ok:             Color::Rgb(30, 130, 70),
            active:         Color::Rgb(170, 110, 20),
            status_bg:      Color::Rgb(210, 220, 235),
            status_fg:      Color::Rgb(30, 50, 80),
            tab_selected_bg: Color::Rgb(90, 140, 210),
            tab_selected_fg: Color::Rgb(255, 255, 255),
            tab_idle_bg:    Color::Rgb(225, 228, 232),
            tab_idle_fg:    Color::Rgb(60, 65, 75),
        },
    }
}

/// Disk path for the persisted theme preference (a single line with
/// "dark" or "light").  Lives alongside session data.
pub fn theme_pref_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    std::path::PathBuf::from(home).join(".larql/theme")
}

pub fn load_theme() -> Theme {
    match std::fs::read_to_string(theme_pref_path()).ok().as_deref().map(str::trim) {
        Some("light") => Theme::Light,
        _ => Theme::Dark,
    }
}

pub fn save_theme(theme: Theme) {
    let path = theme_pref_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, theme.name());
}
