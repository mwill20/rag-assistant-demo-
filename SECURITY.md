# Security Policy

## Supported Versions
We support the latest `main` branch and the most recent tagged release.

## Reporting a Vulnerability
Email **security@your-domain.example** with:
- Description + impact
- Steps to reproduce
- Environment details (OS, Python)

We aim to acknowledge within 72 hours.

## Scope & Expectations
- Do not test with real secrets or sensitive data.
- Do not perform denial-of-service testing.
- Please avoid automated scanning of hosted demos.

## Defensive Measures in This Repo
- `.env` excluded from VCS; `.env_example` provided
- Citations required by design; “no-answer” fallback when context is weak
- Optional MMR/rerank to improve retrieval quality

## Roadmap
- Dependency audit in CI
- Secret scanning in CI
- SBOM generation in CI
