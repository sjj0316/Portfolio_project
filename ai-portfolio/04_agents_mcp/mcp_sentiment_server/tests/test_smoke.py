from mcp_sentiment_server.server import analyze_sentiment, build_app


def test_analyze_sentiment_shape():
    res = analyze_sentiment("i love this wonderful thing")
    assert set(res.keys()) == {"label", "score"}
    assert res["label"] in {"positive", "neutral", "negative"}
    assert 0.0 <= float(res["score"]) <= 1.0


def test_build_app_registers_artifacts():
    app = build_app()
    assert any("sentiment_analyze" in str(tool.name) for tool in app.tools), "tool missing"
    assert any(res.uri == "health://status" for res in app.resources), "resource missing"
    assert any(pr.name == "sentiment_prompt" for pr in app.prompts), "prompt missing"
