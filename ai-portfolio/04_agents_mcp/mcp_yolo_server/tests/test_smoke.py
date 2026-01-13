import pytest

from mcp_yolo_server.server import build_app, yolo_detect


def test_build_app_registers_components():
    app = build_app()
    assert any("yolo_detect_tool" in str(tool.name) for tool in app.tools), "tool missing"
    assert any(res.uri == "health://status" for res in app.resources), "resource missing"
    assert any(pr.name == "yolo_prompt" for pr in app.prompts), "prompt missing"


def test_yolo_detect_requires_extra():
    # Without yolo extra, expect a RuntimeError with guidance
    with pytest.raises(RuntimeError):
        yolo_detect(source="0", conf=0.5, iou=0.5)
