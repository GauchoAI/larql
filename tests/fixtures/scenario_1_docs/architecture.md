# Architecture

## Overview

The Gaucho platform uses a microservices architecture with three main components:
- **API Gateway** (port 8080): handles authentication and rate limiting
- **Worker Service** (port 9090): processes background jobs using Redis queues
- **Storage Layer**: PostgreSQL for structured data, S3 for binary assets

## Authentication

All API requests require a Bearer token in the Authorization header.
Tokens are JWT with RS256 signing, issued by the `/auth/token` endpoint.
Token expiry is 24 hours. Refresh tokens last 30 days.

## Database

Primary database is PostgreSQL 15. Connection pooling via PgBouncer.
Maximum pool size: 20 connections per service instance.
Migrations managed by `sqlx migrate`.
