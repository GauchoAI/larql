use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

pub struct Logger {
    path: PathBuf,
}

impl Logger {
    pub fn new() -> Self {
        let dir = crate::app::home_dir().join(".larql/logs");
        let _ = create_dir_all(&dir);
        let date = Self::date_today();
        Logger {
            path: dir.join(format!("{date}.jsonl")),
        }
    }

    pub fn log(&self, entry_type: &str, content: &str, metadata: Option<serde_json::Value>) {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&self.path) {
            let entry = serde_json::json!({
                "ts": crate::app::chrono_now(),
                "type": entry_type,
                "content": content,
                "metadata": metadata,
            });
            let _ = writeln!(f, "{}", entry);
        }
    }

    fn date_today() -> String {
        // Simple date from epoch (no chrono needed for this)
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Rough date calc (good enough for log filenames)
        let days = secs / 86400;
        let mut y = 1970i64;
        let mut remaining = days as i64;
        loop {
            let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                366
            } else {
                365
            };
            if remaining < days_in_year {
                break;
            }
            remaining -= days_in_year;
            y += 1;
        }
        let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
        let month_days = [
            31,
            if leap { 29 } else { 28 },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ];
        let mut m = 0usize;
        for (i, &md) in month_days.iter().enumerate() {
            if remaining < md as i64 {
                m = i;
                break;
            }
            remaining -= md as i64;
        }
        format!("{y:04}-{:02}-{:02}", m + 1, remaining + 1)
    }
}
