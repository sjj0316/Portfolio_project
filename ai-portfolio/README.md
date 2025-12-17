# AI Portfolio (Codex + Local) Starter

This is a multi-project starter designed for:
- Doing changes in Codex (PR-based)
- Pulling to your local machine and running with **uv**
- Keeping each project as an independent root (easy to split into separate repos later)

## Projects
- `01_nlp/nlp_sentiment_starter`
- `02_cv/realtime_yolo_starter`
- `03_image_gen/image_gen_starter`

Each project contains:
- `AGENTS.md` (Codex rules)
- English command contract via `pyproject.toml` console scripts
- `configs/local.yaml` + `configs/colab.yaml`

## Quickstart
Pick a project folder and run:

```bash
uv sync
uv run smoke --profile local
```

For heavier features (YOLO / Diffusers), install extras per-project:
- CV: `uv sync --extra yolo`
- Image gen: `uv sync --extra diffusers`


## Codex workflow (recommended)
1) In Codex, assign a small task (one feature or fix).
2) Ensure Codex follows `AGENTS.md` and uses these commands:
   - `uv run smoke --profile local`
   - `uv run test` (if `--extra dev` installed)
3) Let Codex open a PR.
4) Locally: pull the PR branch and run the same commands.

Keep PRs small. Avoid committing large artifacts (data/models/outputs).
