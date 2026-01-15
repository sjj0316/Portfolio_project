# AI Portfolio Monorepo - Agent Instructions (Codex)

## Operating rules
- Use `uv` only. Do not introduce pip/conda/poetry.
- Keep PRs small: one feature or one fix.
- Never commit large artifacts: .venv/, data/, models/, outputs/, runs/, *.pt, *.bin, *.ckpt, *.safetensors
- Keep commands English and stable: smoke/train/eval/predict/lint/test/format.

## Review guidelines
- Commands must run on Windows PowerShell (no bash-only assumptions).
- Keep `smoke` lightweight: it must succeed without heavy deps (torch/diffusers/yolo).
- Prefer lazy imports in CLI to avoid importing heavy libs in `smoke`.
- No hard-coded absolute paths; use configs and relative paths.
- If you touch logic, add/adjust tests.

## Git workflow (Codex standard)
- Do not commit directly to `main` (exception: `hotfix/<topic>`).
- Create a new branch before starting work for:
  - New features
  - Behavior changes or compatibility impact
  - Refactors or structure changes
  - Dependency/build/CI/config changes
- Branch naming:
  - `feat/<topic>`, `fix/<topic>`, `refactor/<topic>`, `chore/<topic>`, `docs/<topic>`
  - Keep `<topic>` short and specific (e.g., `feat/ui-launcher`, `fix/yolo-predict`).
- Commits:
  - Small, single-purpose commits.
  - Prefer Conventional Commits: `feat(ui): add launcher`, `fix(cv): handle missing model`.
- Merge checklist (minimum):
  - `uv run smoke --profile local` (project scope)
  - Lint/test if `--extra dev` is used
  - README or usage docs updated when behavior changes
  - No secrets committed; use `.env.sample` if needed
- After merge, delete the branch (local/remote).
