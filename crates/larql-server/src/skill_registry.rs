//! SQLite registry for skills.  Lives next to the session logs at
//! `~/.larql/skills.db` and tracks metadata that the filesystem
//! alone can't express: where a skill came from, when it was last
//! used, how often it has failed, what session it belongs to.
//!
//! The filesystem (`~/.larql/skills/<name>/skill.md` + `tool.sh`) is
//! still the source of truth for executable content.  This DB is the
//! *catalog* on top.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub name: String,
    pub description: String,
    pub keywords: String,
    pub runtime: String,
    pub source: String,
    pub scope: String,
    pub created_at: u64,
    pub last_used_ts: Option<u64>,
    pub last_success_ts: Option<u64>,
    pub use_count: u64,
    pub fail_count: u64,
}

pub struct Registry {
    conn: Mutex<Connection>,
}

impl Registry {
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS skills (
                name             TEXT PRIMARY KEY,
                description      TEXT NOT NULL DEFAULT '',
                keywords         TEXT NOT NULL DEFAULT '',
                runtime          TEXT NOT NULL DEFAULT 'host',
                source           TEXT NOT NULL DEFAULT 'manual',
                scope            TEXT NOT NULL DEFAULT 'global',
                created_at       INTEGER NOT NULL,
                last_used_ts     INTEGER,
                last_success_ts  INTEGER,
                use_count        INTEGER NOT NULL DEFAULT 0,
                fail_count       INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_skills_scope ON skills(scope);
            CREATE INDEX IF NOT EXISTS idx_skills_last_used ON skills(last_used_ts DESC);",
        )
        .map_err(|e| format!("schema: {e}"))?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    pub fn default_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join(".larql/skills.db")
    }

    pub fn upsert(
        &self,
        name: &str,
        description: &str,
        keywords: &str,
        runtime: &str,
        source: &str,
        scope: &str,
    ) -> Result<(), String> {
        let now = unix_now();
        let conn = self.conn.lock().map_err(|e| format!("lock: {e}"))?;
        conn.execute(
            "INSERT INTO skills (name, description, keywords, runtime, source, scope, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(name) DO UPDATE SET
                description = excluded.description,
                keywords    = excluded.keywords,
                runtime     = excluded.runtime,
                source      = COALESCE(excluded.source, source),
                scope       = COALESCE(excluded.scope, scope)",
            params![name, description, keywords, runtime, source, scope, now as i64],
        )
        .map_err(|e| format!("upsert {name}: {e}"))?;
        Ok(())
    }

    pub fn mark_used(&self, name: &str, success: bool) -> Result<(), String> {
        let now = unix_now() as i64;
        let conn = self.conn.lock().map_err(|e| format!("lock: {e}"))?;
        if success {
            conn.execute(
                "UPDATE skills SET use_count = use_count + 1,
                                   last_used_ts = ?1,
                                   last_success_ts = ?1
                 WHERE name = ?2",
                params![now, name],
            )
            .map_err(|e| format!("mark_used: {e}"))?;
        } else {
            conn.execute(
                "UPDATE skills SET use_count = use_count + 1,
                                   last_used_ts = ?1,
                                   fail_count = fail_count + 1
                 WHERE name = ?2",
                params![now, name],
            )
            .map_err(|e| format!("mark_used: {e}"))?;
        }
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<RegistryEntry>, String> {
        let conn = self.conn.lock().map_err(|e| format!("lock: {e}"))?;
        let mut stmt = conn
            .prepare(
                "SELECT name, description, keywords, runtime, source, scope, created_at,
                        last_used_ts, last_success_ts, use_count, fail_count
                 FROM skills
                 ORDER BY COALESCE(last_used_ts, 0) DESC, name ASC",
            )
            .map_err(|e| format!("prepare: {e}"))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(RegistryEntry {
                    name: row.get(0)?,
                    description: row.get(1)?,
                    keywords: row.get(2)?,
                    runtime: row.get(3)?,
                    source: row.get(4)?,
                    scope: row.get(5)?,
                    created_at: row.get::<_, i64>(6)? as u64,
                    last_used_ts: row.get::<_, Option<i64>>(7)?.map(|v| v as u64),
                    last_success_ts: row.get::<_, Option<i64>>(8)?.map(|v| v as u64),
                    use_count: row.get::<_, i64>(9)? as u64,
                    fail_count: row.get::<_, i64>(10)? as u64,
                })
            })
            .map_err(|e| format!("query: {e}"))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| format!("row: {e}"))?);
        }
        Ok(out)
    }

    pub fn delete(&self, name: &str) -> Result<bool, String> {
        let conn = self.conn.lock().map_err(|e| format!("lock: {e}"))?;
        let n = conn
            .execute("DELETE FROM skills WHERE name = ?1", params![name])
            .map_err(|e| format!("delete: {e}"))?;
        Ok(n > 0)
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
