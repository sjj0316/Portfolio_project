# AGENTS (Codex instructions)

You are working in a small portfolio project (**YOLO MCP Server**).
Follow these rules strictly:

## Tooling
- Use **uv** for Python dependency management and running commands.
- Do not introduce other package managers (pip/poetry/conda) unless explicitly requested.

## Commands (contract)
- Run smoke check: `uv run smoke --profile local`
- Run tests (if dev extra installed): `uv run test`
- Format/lint (if dev extra installed): `uv run format` and `uv run lint`
- Serve MCP: `uv run serve --transport streamable-http`

## Code style
- Keep changes small and focused.
- Prefer small, readable functions with clear names.
- Lazy-import heavy optional dependencies inside subcommands.

## Repo hygiene
- Do NOT commit large files (datasets, model weights, outputs).
- Keep configs in `configs/` and paths configurable via `--profile` / `PROFILE`.

## What / Why
- Expose the existing YOLO predict workflow as an MCP tool with optional heavy deps.

## How to test (local)
- [ ] `uv sync --extra dev`
- [ ] `uv run smoke --profile local`
- [ ] (If YOLO extra installed) `uv run test`

## Codex review
- [ ] Commented on this PR: `@codex review for correctness, Windows portability, uv usage, and CI readiness`
