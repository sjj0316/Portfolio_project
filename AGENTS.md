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
