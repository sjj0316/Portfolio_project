import sys

import pytest

import gen_app.cli as cli

def test_cfg():
    cfg = cli.load_cfg("local")
    assert cfg.data_dir.name == "data"


def test_predict_requires_prompt(monkeypatch):
    monkeypatch.setattr(cli, "_ensure_dirs", lambda cfg: None)
    monkeypatch.setattr(sys, "argv", ["predict"])
    with pytest.raises(SystemExit):
        cli.predict_cmd()
