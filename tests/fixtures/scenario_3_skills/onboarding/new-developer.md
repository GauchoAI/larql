# New Developer Onboarding

## Day 1

1. Get access to GitHub org `gaucho-platform`
2. Clone the monorepo: `git clone git@github.com:gaucho-platform/gaucho.git`
3. Install dependencies: `make setup` (requires Docker, Python 3.11, Rust 1.75+)
4. Run the test suite: `make test` — all tests must pass before first PR

## Day 2

1. Read `docs/architecture.md` for system overview
2. Set up local development: `docker compose up -d`
3. The local API runs at `http://localhost:8080`
4. Create a test account: `./scripts/create-dev-user.sh`

## Key Contacts

- **Tech Lead**: Sarah Chen (sarah@gaucho.io)
- **DevOps**: Marcus Rodriguez (marcus@gaucho.io)
- **On-call rotation**: see #oncall in Slack
