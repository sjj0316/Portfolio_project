# AGENTS (Codex instructions)

You are working in a small portfolio project (**Portfolio Launcher UI**).
Follow these rules strictly:

## Tooling
- Use **uv** for Python dependency management and running commands.
- Do not introduce other package managers (pip/poetry/conda) unless explicitly requested.

## Commands (contract)
- Run smoke check: `uv run smoke --profile local`
- Run tests (if dev extra installed): `uv run test`
- Format/lint (if dev extra installed): `uv run format` and `uv run lint`

## Code style
- Keep changes small and focused.
- Prefer small, readable functions with clear names.
- Lazy-import UI dependencies inside subcommands that need them.

## Repo hygiene
- Do NOT commit large files (datasets, model weights, outputs).
- Keep configs in `configs/` and paths configurable via `--profile` / `PROFILE`.


## What / Why
- 

## How to test (local)
- [ ] `uv sync --extra dev`
- [ ] `uv run smoke --profile local`

## Codex review
- [ ] Commented on this PR: `@codex review for correctness, Windows portability, uv usage, and CI readiness`
