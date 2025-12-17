from gen_app.cli import load_cfg

def test_cfg():
    cfg = load_cfg("local")
    assert cfg.data_dir.name == "data"
