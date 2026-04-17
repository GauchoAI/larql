# Deployment

## Environments

- **staging**: `staging.gaucho.dev` — auto-deploys from `main` branch
- **production**: `app.gaucho.io` — manual deploy via `./scripts/deploy.sh prod`

## Requirements

- Docker 24+
- kubectl configured for the target cluster
- AWS credentials in `~/.aws/credentials`

## Deploy Steps

1. Build images: `make docker-build`
2. Push to ECR: `make docker-push`
3. Apply k8s manifests: `kubectl apply -f k8s/`
4. Verify: `curl https://app.gaucho.io/health`

## Rollback

To rollback to the previous version:
```bash
kubectl rollout undo deployment/api-gateway
kubectl rollout undo deployment/worker-service
```
