# AGENTS (Codex instructions)

You are working in a small portfolio project (**Image Generation**).
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
- Lazy-import heavy optional dependencies inside subcommands.

## Repo hygiene
- Do NOT commit large files (datasets, model weights, outputs).
- Keep configs in `configs/` and paths configurable via `--profile` / `PROFILE`.
