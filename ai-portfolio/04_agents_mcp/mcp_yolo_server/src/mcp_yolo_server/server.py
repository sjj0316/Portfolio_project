"""FastMCP YOLO server definition.

YOLO 추론은 optional extra(yolo) 설치 시에만 사용되며, smoke에서는 모델 다운로드 없이 등록/입력 검증만 수행한다.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

from mcp.server.fastmcp import FastMCP

TransportName = Literal["streamable-http", "stdio"]


@dataclass(frozen=True)
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    transport: TransportName = "streamable-http"


def _lazy_import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "YOLO dependencies not installed. Install with: uv sync --extra yolo"
        ) from exc
    return YOLO


def yolo_detect(
    source: str | int,
    conf: float = 0.25,
    iou: float = 0.45,
    model_path: str = "yolo11n.pt",
) -> dict[str, Any]:
    """Run YOLO detection if dependencies are available; otherwise raise with guidance."""
    YOLO = _lazy_import_ultralytics()
    model = YOLO(model_path)
    results = model.predict(source=source, conf=conf, iou=iou, verbose=False)
    boxes: list[dict[str, Any]] = []
    for res in results:
        if not hasattr(res, "boxes"):
            continue
        for box in getattr(res, "boxes"):
            xyxy = box.xyxy[0].tolist() if hasattr(box, "xyxy") else []
            cls = int(box.cls[0]) if hasattr(box, "cls") else -1
            score = float(box.conf[0]) if hasattr(box, "conf") else 0.0
            boxes.append({"xyxy": xyxy, "class": cls, "score": score})
    return {"boxes": boxes}


def build_app() -> FastMCP:
    """FastMCP 앱 생성 및 YOLO tool/리소스/프롬프트 등록."""
    app = FastMCP(name="YOLO MCP Server", json_response=True)

    @app.tool()
    async def yolo_detect_tool(source: str, conf: float = 0.25, iou: float = 0.45) -> dict[str, Any]:
        # 입력 검증: conf/iou 범위 확인
        if not 0.0 < conf <= 1.0:
            raise ValueError("conf must be in (0,1].")
        if not 0.0 < iou <= 1.0:
            raise ValueError("iou must be in (0,1].")
        # 실제 추론은 yolo extra 설치 시에만 실행
        return yolo_detect(source=source, conf=conf, iou=iou)

    @app.resource("health://status")
    async def health() -> str:
        return "ok"

    @app.prompt("yolo_prompt")
    async def yolo_prompt(source: str, conf: float = 0.25, iou: float = 0.45) -> str:
        return (
            "Run object detection on the provided source using YOLO. "
            f"source={source}, conf={conf}, iou={iou}"
        )

    return app


def _run_http(app: FastMCP, host: str, port: int) -> None:
    if hasattr(app, "run_http"):
        app.run_http(host=host, port=port)
    elif hasattr(app, "run"):
        app.run(transport="streamable-http", host=host, port=port)
    else:  # pragma: no cover - defensive
        raise RuntimeError("FastMCP does not expose an HTTP run method.")


def _run_stdio(app: FastMCP) -> None:
    if hasattr(app, "run_stdio"):
        app.run_stdio()
    elif hasattr(app, "run"):
        app.run(transport="stdio")
    else:  # pragma: no cover - defensive
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


def run_async(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))
