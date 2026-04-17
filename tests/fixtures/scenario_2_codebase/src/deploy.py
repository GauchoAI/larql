"""Deployment module for the Gaucho platform.

Manages Docker builds, ECR pushes, and Kubernetes deployments.
The staging URL is staging.gaucho.dev and production is app.gaucho.io.
"""

import subprocess
import sys


ENVIRONMENTS = {
    "staging": {
        "url": "staging.gaucho.dev",
        "cluster": "gaucho-staging",
        "auto_deploy": True,
        "branch": "main",
    },
    "production": {
        "url": "app.gaucho.io",
        "cluster": "gaucho-prod",
        "auto_deploy": False,
        "branch": None,
    },
}


def build_docker(tag: str = "latest") -> bool:
    """Build Docker images for all services."""
    result = subprocess.run(["make", "docker-build", f"TAG={tag}"])
    return result.returncode == 0


def deploy(environment: str, branch: str = "main") -> str:
    """Deploy to the specified environment.

    Args:
        environment: "staging" or "production"
        branch: Git branch to deploy

    Returns:
        Deployment ID string
    """
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {environment}")

    env = ENVIRONMENTS[environment]
    if environment == "production" and not _confirm_production():
        return "cancelled"

    # Build, push, apply
    build_docker(tag=branch)
    subprocess.run(["make", "docker-push"])
    subprocess.run(["kubectl", "--context", env["cluster"], "apply", "-f", "k8s/"])

    return f"dep-{branch[:8]}"


def rollback(environment: str) -> None:
    """Rollback to previous deployment version."""
    env = ENVIRONMENTS[environment]
    subprocess.run([
        "kubectl", "--context", env["cluster"],
        "rollout", "undo", "deployment/api-gateway"
    ])
    subprocess.run([
        "kubectl", "--context", env["cluster"],
        "rollout", "undo", "deployment/worker-service"
    ])


def _confirm_production() -> bool:
    return input("Deploy to PRODUCTION? (yes/no): ").strip().lower() == "yes"
