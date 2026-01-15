# AI Portfolio Monorepo (uv + Codex)

Detailed notes to understand and run the whole portfolio. Four AI demos plus a UI launcher and two MCP servers live side by side, each with its own isolated `uv` environment and a stable, English console-script contract that Codex can follow.

---

## Why this repo exists
- Build a portfolio of AI starters without dependency collisions; each project is an independent root that can be split out later.
- Keep commands stable and human-friendly (`smoke`, `train`, `eval`, `predict`, `lint`, `test`, `clean`, `setup`).
- Run locally or on Colab with the same commands by switching `--profile local|colab`.
- Avoid heavyweight installs unless needed: torch/YOLO/Diffusers/scikit-learn are optional extras and lazy-imported inside the commands that use them.

---

## Repository map (what is inside)
- ai-portfolio/ (portfolio projects)
  - 00_ui/portfolio_launcher
  - 01_nlp/nlp_sentiment_starter
  - 02_cv/realtime_yolo_starter
  - 03_image_gen/image_gen_starter
  - 04_tabular/tabular_classification_starter
  - 04_agents_mcp/mcp_sentiment_server
  - 04_agents_mcp/mcp_yolo_server
- notebooks/colab_runner.ipynb (pull + run a project on Colab GPU from .py sources)
- AGENTS.md (global Codex rules and guardrails)
- .github/ (PR template and CI configs for Codex review)

Each project folder repeats the same pattern: its own `AGENTS.md`, `pyproject.toml` console scripts, `configs/` for profiles, and optional `tests/`. Work inside one project at a time; do not share virtual environments.

---

## Prerequisites
- Python 3.12+
- Windows PowerShell (commands below assume PS)
- `uv` installed globally (official installer or `pip install uv`); do not introduce pip/conda/poetry envs here.

---

## How the command contract works
- Console scripts live under `[project.scripts]` in each `pyproject.toml`. Stable verbs: `smoke`, `train`, `eval`, `predict`, `lint`, `test`, `clean`, `setup`.
- CLI contract scope:
  - Demo projects: `smoke`, `train`, `eval`, `predict` (+ `lint/test/format` when dev extra is installed).
  - MCP servers: `smoke`, `serve` (profile flag is accepted but unused; see MCP READMEs).
  - UI launcher: `predict` opens the Gradio UI.
- Every command accepts `--profile local|colab` (or env `PROFILE`) on demo projects. The profile chooses which YAML config under `configs/` is loaded.
- Config keys: `data_dir`, `models_dir`, `outputs_dir`, `reports_dir`. Relative paths are resolved from the project root; absolute paths are respected. Missing config files raise a clear error before heavy work starts.
- Helper utilities in each `cli.py` create those folders automatically so repeated runs do not fail on missing directories.
- Heavy dependencies are optional extras; if a module is missing, commands fail fast with an install hint.

---

## Project-by-project walkthrough
Pick one project root and run commands inside it.

### 1) NLP Sentiment (`ai-portfolio/01_nlp/nlp_sentiment_starter`)
Goal: tiny sentiment classifier with a no-deps lexicon fallback and an optional scikit-learn baseline.
```powershell
cd .\ai-portfolio\01_nlp\nlp_sentiment_starter
uv sync                          # install base deps only
uv run smoke --profile local     # checks config + folders; torch is optional
uv run train --profile local     # trains LogisticRegression if sklearn exists, else writes lexicon json
uv run eval --profile local      # evaluates latest model -> reports/eval_report.json
uv run predict --profile local --text "this is surprisingly good"
```
Extras:
- `uv sync --extra ml` adds scikit-learn for the baseline pipeline.
- `uv sync --extra transformers` brings in a larger Transformers/Torch stack (not used by default).
- `uv sync --extra dev` enables `uv run lint` (ruff) and `uv run test` (pytest).

### 2) Real-time YOLO (`ai-portfolio/02_cv/realtime_yolo_starter`)
Goal: run YOLO inference on webcam/video/image with minimal ceremony.
```powershell
cd .\ai-portfolio\02_cv\realtime_yolo_starter
uv sync                          # base deps only; smoke does not need YOLO
uv run smoke --profile local
uv sync --extra yolo             # installs Ultralytics + OpenCV only when needed
uv run predict --profile local --source 0 --model yolo11n.pt --save
```
Notes:
- `predict` lazily imports OpenCV/Ultralytics; `--source 0` opens webcam, file paths or URLs also work.
- Outputs land in `outputs/yolo/predict/`; use `--save` to persist rendered frames instead of only viewing them.
- `train`/`eval` are intentional placeholders; add dataset scripts under `scripts/` if you want custom training or metrics.

### 3) Text-to-image (`ai-portfolio/03_image_gen/image_gen_starter`)
Goal: Diffusers-based text-to-image demo with a tiny default pipeline for fast testing.
```powershell
cd .\ai-portfolio\03_image_gen\image_gen_starter
uv sync                          # base deps only
uv run smoke --profile local
uv sync --extra diffusers        # installs Diffusers + Torch + Pillow for generation
uv run predict --profile local --prompt "a cute robot on a desk" --steps 2
```
Notes:
- Default model `hf-internal-testing/tiny-stable-diffusion-pipe` keeps runs lightweight; override via `--model <hf-id>`.
- Images are saved under `outputs/images/` (auto-created). `--seed` controls determinism; `--out` names the PNG.
- Training/eval are out of scope for this starter; add LoRA fine-tuning or metrics later if desired.

### 4) Tabular Classification (`ai-portfolio/04_tabular/tabular_classification_starter`)
Goal: small tabular classifier with optional scikit-learn baseline and pure-Python fallback.
```powershell
cd .\ai-portfolio\04_tabular\tabular_classification_starter
uv sync                          # base deps only
uv run smoke --profile local
uv sync --extra ml               # installs scikit-learn for the baseline pipeline
uv run train --profile local
uv run eval --profile local
uv run predict --profile local --features "0.5,0.1,0.3"
```
Notes:
- Smoke and fallback require no sklearn; baseline uses LogisticRegression + StandardScaler when installed.
- Outputs and reports mirror other starters: models under `models/`, evaluation under `reports/`.
- Extend with your own dataset by replacing the tiny in-code samples in `tab_app/cli.py`.

---

## Profiles and configs
- `configs/local.yaml` and `configs/colab.yaml` live in each project. Adjust paths to match your machine or mounted Drive.
- Switch environments with `--profile colab` or `PROFILE=colab uv run <command>`. Failing fast on missing configs prevents accidental writes.
- Outputs, models, and reports all respect these paths, so you can keep artifacts isolated per environment.

---

## Colab workflow (no .ipynb source needed)
Use `notebooks/colab_runner.ipynb` when you want GPU runs but only ship `.py` code.
1) Open the notebook in Colab and set `REPO_URL`, `REPO_DIR`, optional `BRANCH`, and `PROJECT_PATH` (for example `ai-portfolio/02_cv/realtime_yolo_starter`).
2) Run the cells; they clone, install `uv`, and execute `uv sync [--extra ...]` followed by the same console scripts as local.
3) If you want outputs persisted, mount Google Drive and point `configs/colab.yaml` paths to your Drive mount.

---

## Contribution and safety rules
- Keep PRs small: one feature or fix. Do not commit large artifacts (`.venv/`, `data/`, `models/`, `outputs/`, `runs/`, `*.pt`, `*.bin`, `*.ckpt`, `*.safetensors`).
- Use `uv` only; avoid pip/conda/poetry envs.
- Preferred checks: `uv run smoke --profile local`, `uv run lint`, `uv run test` (with `--extra dev`), plus project-specific `train`/`eval`/`predict` where relevant.
- Follow each project's `AGENTS.md` for Codex-specific constraints, Windows compatibility, and review expectations.

---

## Handy troubleshooting
- `program not found` when running `uv run <cmd>`: ensure `[tool.uv] package = true` exists in the project `pyproject.toml`, then rerun `uv sync`.
- Missing optional deps (torch/diffusers/yolo/sklearn): install the listed `--extra`; commands surface a clear error naming the missing module.
- Path issues: edit the appropriate `configs/*.yaml` before running; config is read first so you fail fast instead of half-running a command.
