"""CLI entrypoints for the Sentiment MCP server."""

from __future__ import annotations

import argparse
import subprocess
import sys

from mcp_sentiment_server.server import analyze_sentiment, build_app, run_server


def setup_cmd() -> None:
    """Print recommended setup commands."""
    print("Recommended setup:")
    print("  uv sync")
    print("Optional extras:")
    print("  uv sync --extra dev")


def lint_cmd() -> None:
    """Run ruff lint (requires dev extra)."""
    subprocess.call(["python", "-m", "ruff", "check", "."])


def format_cmd() -> None:
    """Run ruff format (requires dev extra)."""
    subprocess.call(["python", "-m", "ruff", "format", "."])


def test_cmd() -> None:
    """Run pytest (requires dev extra)."""
    subprocess.call(["python", "-m", "pytest"])


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mcp-sentiment", add_help=True)
    parser.add_argument("--profile", default="local", help="Optional profile (unused placeholder).")

    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Run a fast smoke test without starting the server.")
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
    # 간단한 감성 분석 휴리스틱을 직접 호출해 결과 형태를 검증
    samples = ["I love this great product", "This is terrible and the worst"]
    for text in samples:
        res = analyze_sentiment(text)
        assert "label" in res and "score" in res, "result must contain label and score"
        assert res["label"] in {"positive", "neutral", "negative"}, "invalid label"
        assert 0.0 <= float(res["score"]) <= 1.0, "score out of range"
    print("[ok] smoke sentiment heuristics")
    print("[done] smoke")


def _serve_cmd(args: argparse.Namespace) -> None:
    run_server(transport=args.transport, host=args.host, port=args.port)


def smoke() -> None:
    args = _parse_args(["smoke"])
    args.func(args)


def serve() -> None:
    args = _parse_args(sys.argv[1:])
    args.func(args)
