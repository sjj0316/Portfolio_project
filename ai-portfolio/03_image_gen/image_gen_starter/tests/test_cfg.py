from gen_app import cli

def test_cfg():
    cfg = cli.load_cfg("local")
    assert cfg.data_dir.name == "data"


def test_parse_predict_args_flags():
    args = cli._parse_predict_args(["--prompt", "x", "--no-safety-checker"])
    assert args.prompt == "x"
    assert args.no_safety_checker is True


def test_check_cmd_exists():
    assert callable(cli.check_cmd)
