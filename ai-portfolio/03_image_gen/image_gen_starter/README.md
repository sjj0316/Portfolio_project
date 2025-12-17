# Image Generation Starter

    A tiny text-to-image starter using Diffusers (optional) with Codex-friendly English commands.

    ## Quickstart (local)

    1) Install uv (once on your machine).
    2) From this project folder:

    ```bash
    uv sync
    uv run smoke --profile local
    ```


### Optional extras
- Diffusers + Torch (text-to-image):
  ```bash
  uv sync --extra diffusers
  ```
- Dev tools:
  ```bash
  uv sync --extra dev
  ```

    ## Run examples


```bash
# smoke (no diffusers required)
uv run smoke --profile local

# text-to-image (requires diffusers extra)
uv run predict --profile local --prompt "a cute robot on a desk" --steps 2

# choose a model id (default is a tiny testing pipeline)
uv run predict --profile local --model hf-internal-testing/tiny-stable-diffusion-pipe --prompt "cat"
```

    ## Profiles (local vs colab)
    - `--profile local|colab` (or env `PROFILE=local|colab`)
    - Edit `configs/local.yaml` and `configs/colab.yaml` to match your paths.

    ## Notes for Codex
    - See `AGENTS.md` for the command contract and repo rules.
