from pathlib import Path

from launcher_app.cli import build_projects, find_repo_root


def test_smoke_projects_exist() -> None:
    repo_root = find_repo_root(Path(__file__).resolve())
    projects = build_projects(repo_root)
    missing = [project for project in projects.values() if not project.path.exists()]
    assert not missing, f"Missing project paths: {[str(p.path) for p in missing]}"
