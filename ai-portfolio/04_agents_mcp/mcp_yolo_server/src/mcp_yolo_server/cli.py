"""CLI entrypoints for the YOLO MCP server."""

from __future__ import annotations

import argparse
import subprocess
import sys

from mcp_yolo_server.server import build_app, run_server, yolo_detect


def setup_cmd() -> None:
    """Print recommended setup commands."""
    print("Recommended setup:")
    print("  uv sync")
    print("Optional extras:")
    print("  uv sync --extra dev")
    print("  uv sync --extra yolo         # required for real YOLO inference")


def lint_cmd() -> None:
    """Run ruff lint (requires dev extra)."""
    subprocess.call([sys.executable, "-m", "ruff", "check", "."])


def format_cmd() -> None:
    """Run ruff format (requires dev extra)."""
    subprocess.call([sys.executable, "-m", "ruff", "format", "."])


def test_cmd() -> None:
    """Run pytest (requires dev extra)."""
    subprocess.call([sys.executable, "-m", "pytest"])


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mcp-yolo", add_help=True)
    parser.add_argument("--profile", default="local", help="Optional profile (placeholder).")

    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Run a fast smoke test without downloading YOLO.")
    smoke.set_defaults(func=_smoke_cmd)

    serve = sub.add_parser("serve", help="Start the MCP server.")
    serve.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="Transport type (default: streamable-http).",
    )
    serve.add_argument("--host", default=None, help="Host for HTTP transport (default env MCP_HOST or 127.0.0.1).")
    serve.add_argument("--port", type=int, default=None, help="Port for HTTP transport (default env MCP_PORT or 8000).")
    serve.set_defaults(func=_serve_cmd)

    return parser.parse_args(argv)


def _smoke_cmd(_: argparse.Namespace) -> None:
    # YOLO 추론 없이 입력 검증 로직만 확인
    # 실제 YOLO 로드는 yolo extra 설치 후 사용
    _ = build_app()  # tool/resource/prompt 등록 확인
    try:
        yolo_detect(source="0", conf=0.5, iou=0.6)  # will raise if yolo extra missing
    except RuntimeError as exc:
        if "uv sync --extra yolo" not in str(exc):
            raise
        print("[info] yolo extra not installed; skipping actual inference (expected).")
    print("[ok] smoke registration/input validation")
    print("[done] smoke")


def _serve_cmd(args: argparse.Namespace) -> None:
    run_server(transport=args.transport, host=args.host, port=args.port)


def smoke() -> None:
    args = _parse_args(["smoke"])
    args.func(args)


def serve() -> None:
    args = _parse_args(sys.argv[1:])
    args.func(args)
