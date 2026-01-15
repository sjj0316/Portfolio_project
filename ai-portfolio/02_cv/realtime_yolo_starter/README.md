# Realtime YOLO Starter

    A real-time video inference starter (webcam/video) with Codex-friendly English commands.

    ## Quickstart (local)

    1) Install uv (once on your machine).
    2) From this project folder:

    ```bash
    uv sync
    uv run smoke --profile local
    ```


### Optional extras
- YOLO + OpenCV:
  ```bash
  uv sync --extra yolo
  ```
- Dev tools:
  ```bash
  uv sync --extra dev
  ```

    ## Run examples


```bash
# smoke (no yolo dependency required)
uv run smoke --profile local

# webcam inference (requires yolo extra)
uv run predict --profile local --source 0 --model yolo11n.pt

# video file
uv run predict --profile local --source data/raw/sample.mp4 --model yolo11n.pt
```

    ## Profiles (local vs colab)
    - `--profile local|colab` (or env `PROFILE=local|colab`)
    - Edit `configs/local.yaml` and `configs/colab.yaml` to match your paths.

    ## Notes for Codex
    - See `AGENTS.md` for the command contract and repo rules.

## Preflight (first run)
- `uv sync`
- Install YOLO extra: `uv sync --extra yolo`
- Confirm camera permission if using `--source 0`.

## Colab checklist
- Update `configs/colab.yaml` to Drive paths.
- Run with `PROFILE=colab uv run smoke`.

## Portfolio story
- Problem: real-time object detection for webcam/video.
- Approach: YOLO inference via Ultralytics with lazy imports.
- Result: rendered outputs under `outputs/yolo/`.
- Limitations: no training/eval pipeline included.
- Next: add dataset training and evaluation scripts.
