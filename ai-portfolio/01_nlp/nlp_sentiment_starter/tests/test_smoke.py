import sys

import pytest

import nlp_app.cli as cli

def test_load_cfg_local():
    cfg = cli.load_cfg("local")
    assert cfg.profile == "local"


def test_predict_requires_text(monkeypatch):
    monkeypatch.setattr(cli, "_ensure_dirs", lambda cfg: None)
    monkeypatch.setattr(sys, "argv", ["predict"])
    with pytest.raises(SystemExit):
        cli.predict_cmd()
