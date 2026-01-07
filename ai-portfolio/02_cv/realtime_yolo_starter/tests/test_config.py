from cv_app import cli

def test_load_cfg_local():
    assert cli.load_cfg("local").profile == "local"


def test_check_cmd_exists():
    assert callable(cli.check_cmd)
