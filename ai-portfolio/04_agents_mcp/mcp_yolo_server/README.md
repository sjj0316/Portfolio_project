# YOLO MCP Server

    An MCP server that exposes the YOLO predict workflow as a tool. YOLO inference runs only if the `yolo` extra is installed; smoke stays lightweight with no model downloads.

    ## Quickstart (local)

    1) Install uv (once on your machine).
    2) From this project folder:

    ```bash
    uv sync
    uv run smoke --profile local
    ```


### Optional extras
- YOLO + OpenCV (required for real inference):
  ```bash
  uv sync --extra yolo
  ```
- Dev tools:
  ```bash
  uv sync --extra dev
  ```

    ## Run examples


```bash
# smoke (no model download, verifies registration and input checks)
uv run smoke --profile local

# serve over streamable HTTP (requires yolo extra for real detection)
uv run serve --transport streamable-http

# serve over stdio (requires yolo extra for real detection)
uv run serve --transport stdio
```

    ## Inspector (optional)
    - Connect with: `npx -y @modelcontextprotocol/inspector --server-url http://127.0.0.1:8000`
    - Ensure `uv sync --extra yolo` before attempting real detections.

    ## Profiles (local vs colab)
    - `--profile local|colab` (or env `PROFILE=local|colab`)
    - Edit `configs/local.yaml` and `configs/colab.yaml` if you need custom paths.

    ## Notes for Codex
    - See `AGENTS.md` for the command contract and repo rules.
