from cv_app.cli import load_cfg

def test_load_cfg_local():
    assert load_cfg("local").profile == "local"
