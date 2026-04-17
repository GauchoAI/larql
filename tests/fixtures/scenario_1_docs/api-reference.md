# API Reference

## POST /auth/token

Request: `{"email": "user@example.com", "password": "..."}`
Response: `{"access_token": "...", "refresh_token": "...", "expires_in": 86400}`

## GET /api/projects

Lists all projects for the authenticated user.
Requires: Bearer token.
Response: `[{"id": 1, "name": "my-project", "created_at": "..."}]`

## POST /api/projects/:id/deploy

Triggers a deployment for the specified project.
Requires: Bearer token with `deploy` scope.
Request: `{"environment": "staging", "branch": "main"}`
Response: `{"deployment_id": "dep-abc123", "status": "queued"}`

## GET /api/health

No authentication required.
Response: `{"status": "ok", "version": "2.4.1", "uptime_seconds": 123456}`
