# NLP Sentiment Starter

    A small sentiment analysis starter with an English command contract for Codex and local execution.

    ## Quickstart (local)

    1) Install uv (once on your machine).
    2) From this project folder:

    ```bash
    uv sync
    uv run smoke --profile local
    ```


### Optional extras
- Traditional ML baseline:
  ```bash
  uv sync --extra ml
  ```
- Transformers (larger, requires torch):
  ```bash
  uv sync --extra transformers
  ```
- Dev tools (lint/test):
  ```bash
  uv sync --extra dev
  ```

    ## Run examples


```bash
# quick smoke
uv run smoke --profile local

# train baseline model (uses scikit-learn if installed, otherwise falls back)
uv run train --profile local

# evaluate and write a small JSON report
uv run eval --profile local

# predict from text
uv run predict --profile local --text "this is surprisingly good"
```

    ## Profiles (local vs colab)
    - `--profile local|colab` (or env `PROFILE=local|colab`)
    - Edit `configs/local.yaml` and `configs/colab.yaml` to match your paths.

    ## Notes for Codex
    - See `AGENTS.md` for the command contract and repo rules.
