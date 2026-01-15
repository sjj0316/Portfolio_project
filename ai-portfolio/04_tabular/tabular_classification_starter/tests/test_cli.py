from pathlib import Path
import sys

import pytest

from tab_app import cli


def test_load_cfg_resolves_relative_paths(tmp_path: Path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "local.yaml"
    cfg_file.write_text(
        "data_dir: data\nmodels_dir: models\noutputs_dir: outputs\nreports_dir: reports\n",
        encoding="utf-8",
    )

    # monkeypatch project_root to point to a temp structure
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)

    cfg = cli.load_cfg("local")
    assert cfg.data_dir == (tmp_path / "data").resolve()
    assert cfg.models_dir == (tmp_path / "models").resolve()
    assert cfg.outputs_dir == (tmp_path / "outputs").resolve()
    assert cfg.reports_dir == (tmp_path / "reports").resolve()


def test_predict_requires_features(monkeypatch):
    monkeypatch.setattr(cli, "_ensure_dirs", lambda cfg: None)
    monkeypatch.setattr(sys, "argv", ["predict"])
    with pytest.raises(SystemExit):
        cli.predict_cmd()
