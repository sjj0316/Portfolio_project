# Tabular Classification Starter

    A lightweight tabular classification demo with Codex-friendly English commands and a pure-Python fallback.

    ## Quickstart (local)

    1) Install uv (once on your machine).
    2) From this project folder:

    ```bash
    uv sync
    uv run smoke --profile local
    ```


### Optional extras
- Scikit-learn baseline:
  ```bash
  uv sync --extra ml
  ```
- Dev tools:
  ```bash
  uv sync --extra dev
  ```

    ## Run examples


```bash
# smoke (no sklearn required)
uv run smoke --profile local

# train baseline model (uses scikit-learn if installed, otherwise pure-Python fallback)
uv run train --profile local

# evaluate and write a small JSON report
uv run eval --profile local

# predict from comma-separated numeric features
uv run predict --profile local --features "0.5,1.0,0.2"
```

    ## Profiles (local vs colab)
    - `--profile local|colab` (or env `PROFILE=local|colab`)
    - Edit `configs/local.yaml` and `configs/colab.yaml` to match your paths.

    ## Notes for Codex
    - See `AGENTS.md` for the command contract and repo rules.
