# Sentiment MCP Server

    A minimal MCP server using FastMCP with a heuristic sentiment tool, health resource, and prompt template.

    ## Quickstart (local)

    1) Install uv (once on your machine).
    2) From this project folder:

    ```bash
    uv sync
    uv run smoke --profile local
    ```


### Optional extras
- Dev tools:
  ```bash
  uv sync --extra dev
  ```

    ## Run examples


```bash
# smoke (no server start, no network)
uv run smoke --profile local

# serve over streamable HTTP (default)
uv run serve --transport streamable-http

# serve over stdio (optional)
uv run serve --transport stdio
```

    ## Inspector (optional)
    - Connect with: `npx -y @modelcontextprotocol/inspector --server-url http://127.0.0.1:8000`

    ## Profiles (local vs colab)
    - `--profile local|colab` (or env `PROFILE=local|colab`) is accepted but currently unused (placeholder).
    - `configs/local.yaml` and `configs/colab.yaml` are kept for repo consistency.

    ## Notes for Codex
    - See `AGENTS.md` for the command contract and repo rules.
