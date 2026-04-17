# Database Troubleshooting

## Connection Pool Exhausted

**Symptom**: API returns 503 with "connection pool exhausted"

**Cause**: More than 20 concurrent connections per service instance.

**Fix**:
1. Check active connections: `SELECT count(*) FROM pg_stat_activity;`
2. If over 80: identify long-running queries with `SELECT * FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '5 minutes';`
3. Kill stuck queries: `SELECT pg_terminate_backend(pid);`
4. If recurring: increase PgBouncer pool size in `k8s/pgbouncer-config.yaml`

## Slow Migrations

**Symptom**: `sqlx migrate` hangs for more than 5 minutes

**Fix**:
1. Check for locks: `SELECT * FROM pg_locks WHERE NOT granted;`
2. Never run migrations during peak hours (9am-5pm UTC)
3. For large table alterations, use `pg_repack` instead of `ALTER TABLE`

## Backup & Restore

Daily backups run at 2am UTC via `pg_dump`.
Stored in S3 bucket `gaucho-db-backups` with 30-day retention.
To restore: `./scripts/restore-db.sh <backup-date>`
