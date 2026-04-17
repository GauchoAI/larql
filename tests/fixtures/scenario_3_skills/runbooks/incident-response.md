# Incident Response Runbook

## Severity Levels

- **SEV1**: Production down, all users affected. Response: 5 minutes. Page on-call.
- **SEV2**: Major feature broken, >10% users affected. Response: 30 minutes.
- **SEV3**: Minor issue, workaround available. Response: next business day.

## SEV1 Procedure

1. Acknowledge in #incidents Slack channel
2. Start a Zoom bridge: `https://gaucho.zoom.us/j/incident`
3. Check dashboards: `https://grafana.gaucho.io/d/api-overview`
4. If API Gateway down: restart pods `kubectl rollout restart deployment/api-gateway`
5. If database down: failover to replica `./scripts/db-failover.sh`
6. Post-incident: write postmortem within 48 hours in `docs/postmortems/`

## Escalation

If on-call cannot resolve within 30 minutes:
- Escalate to Tech Lead (Sarah Chen): +1-555-0142
- Escalate to VP Engineering (James Park): +1-555-0199
