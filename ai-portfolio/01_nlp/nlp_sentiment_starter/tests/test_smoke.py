from nlp_app import cli

def test_load_cfg_local():
    cfg = cli.load_cfg("local")
    assert cfg.profile == "local"


def test_check_cmd_exists():
    assert callable(cli.check_cmd)
