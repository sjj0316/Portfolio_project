"""FastMCP sentiment server definition.

감성 분석 툴/리소스/프롬프트를 등록하고 transport(streamable-http|stdio)로 실행한다.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal

from mcp.server.fastmcp import FastMCP

TransportName = Literal["streamable-http", "stdio"]


@dataclass(frozen=True)
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    transport: TransportName = "streamable-http"


def analyze_sentiment(text: str) -> dict[str, Any]:
    """간단한 단어 가중치 기반 휴리스틱 감성 분석."""
    positives: dict[str, float] = {
        "good": 0.7,
        "great": 0.8,
        "love": 0.9,
        "amazing": 0.9,
        "happy": 0.8,
        "wonderful": 0.9,
        "like": 0.6,
    }
    negatives: dict[str, float] = {
        "bad": 0.7,
        "terrible": 0.9,
        "hate": 0.9,
        "sad": 0.7,
        "worst": 1.0,
        "awful": 0.9,
        "dislike": 0.6,
    }
    toks = [w.strip(".,!?").lower() for w in text.split()]
    raw_score = 0.0
    for tok in toks:
        if tok in positives:
            raw_score += positives[tok]
        if tok in negatives:
            raw_score -= negatives[tok]
    norm = max(1.0, len(toks))
    score = max(0.0, min(1.0, 0.5 + raw_score / (2 * norm)))
    label: Literal["positive", "neutral", "negative"]
    if score > 0.6:
        label = "positive"
    elif score < 0.4:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": round(score, 3)}


def build_app() -> FastMCP:
    """FastMCP 앱 생성 및 툴/리소스/프롬프트 등록."""
    app = FastMCP(name="Sentiment MCP Server", json_response=True)

    @app.tool()
    async def sentiment_analyze(text: str) -> dict[str, Any]:
        return analyze_sentiment(text)

    @app.resource("health://status")
    async def health() -> str:
        return "ok"

    @app.prompt("sentiment_prompt")
    async def sentiment_prompt(text: str) -> str:
        return f"다음 문장의 감성을 분석해줘: {text}"

    return app


def _run_http(app: FastMCP, host: str, port: int) -> None:
    # FastMCP 버전별 run API 차이를 흡수하기 위해 getattr 사용
    if hasattr(app, "run_http"):
        app.run_http(host=host, port=port)
    elif hasattr(app, "run"):
        app.run(transport="streamable-http", host=host, port=port)
    else:  # pragma: no cover - 방어적 코드
        raise RuntimeError("FastMCP does not expose an HTTP run method.")


def _run_stdio(app: FastMCP) -> None:
    if hasattr(app, "run_stdio"):
        app.run_stdio()
    elif hasattr(app, "run"):
        app.run(transport="stdio")
    else:  # pragma: no cover - 방어적 코드
        raise RuntimeError("FastMCP does not expose a stdio run method.")


def run_server(
    transport: TransportName = "streamable-http",
    host: str | None = None,
    port: int | None = None,
) -> None:
    """선택한 transport로 서버 실행."""
    cfg = AppConfig(
        host=host or os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(port or os.getenv("MCP_PORT", "8000")),
        transport=transport,
    )
    app = build_app()
    if cfg.transport == "stdio":
        _run_stdio(app)
    else:
        _run_http(app, cfg.host, cfg.port)


# 유닛 테스트/스모크에서 재사용할 동기 실행 헬퍼
def run_async(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))
