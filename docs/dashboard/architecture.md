# Architecture

Truthound Dashboard 3.0 is a thin operational layer over Truthound 3.0.

## Principles

- Root API first: `truthound.check`, `profile`, `scan`, `mask`, `learn`
- Namespace features only when they exist upstream:
  `truthound.checkpoint`, `datadocs`, `drift`, `lineage`, `ml`,
  `observability`, `plugins`, `profiler`, `realtime`, `stores`
- No dashboard-only validation semantics
- Preserve the established dashboard UI and Intlayer application i18n

## Runtime shape

- Backend: FastAPI and service layer for persisted dashboard state
- Frontend: existing React layout and patterns, trimmed to the Truthound 3.0 surface
- Contracts: canonical Truthound 3.0 payloads, especially `ValidationRunResult`
- Docs: MkDocs in this repo, mirrored into the main Truthound docs portal
