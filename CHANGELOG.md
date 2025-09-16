# Changelog
All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
- Add adversarial tests and lint/types in CI
- Add CI scans (deps, secrets) and SBOM
- Add basic request logging and rate-limit

## [0.1.0] - 2025-09-16
### Added
- Minimal RAG pipeline (ingest → embed → Chroma → retrieve → cite)
- CLI (`make ingest`, `make qa`) and FastAPI (`/healthz`, `/ask`)
- Golden test, MMR toggle, and “no-answer” fallback
- CONTRIBUTING.md
