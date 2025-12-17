from nlp_app.cli import load_cfg

def test_load_cfg_local():
    cfg = load_cfg("local")
    assert cfg.profile == "local"
