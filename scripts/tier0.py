from __future__ import annotations

import os
import subprocess
from pathlib import Path
import tomllib


DEFAULT_PROJECT_DIRS = [
    "ai-portfolio/00_ui/portfolio_launcher",
    "ai-portfolio/01_nlp/nlp_sentiment_starter",
    "ai-portfolio/02_cv/realtime_yolo_starter",
    "ai-portfolio/03_image_gen/image_gen_starter",
    "ai-portfolio/04_tabular/tabular_classification_starter",
    "ai-portfolio/04_agents_mcp/mcp_sentiment_server",
    "ai-portfolio/04_agents_mcp/mcp_yolo_server",
]


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> int:
    print(f"\n==> ({cwd}) {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    return proc.returncode


def load_scripts(pyproject: Path) -> set[str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {}) or {}
    return set(scripts.keys())


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [repo_root / p for p in DEFAULT_PROJECT_DIRS]

    env = os.environ.copy()
    if env.get("CI", "").lower() in ("1", "true", "yes"):
        env.setdefault("UV_FROZEN", "1")
        env.setdefault("UV_LOCKED", "1")

    failed: list[tuple[str, int]] = []

    for target in targets:
        if not target.exists():
            print(f"[skip] missing path: {target}")
            continue

        pyproject = target / "pyproject.toml"
        if not pyproject.exists():
            print(f"[skip] no pyproject.toml: {target}")
            continue

        scripts = load_scripts(pyproject)

        code = run(["uv", "sync", "--extra", "dev"], cwd=target, env=env)
        if code != 0:
            failed.append((str(target), code))
            continue

        if "test" in scripts:
            code = run(["uv", "run", "test"], cwd=target, env=env)
            if code != 0:
                failed.append((str(target), code))
                continue
        else:
            print(f"[skip] no script: test ({target})")

        if "smoke" in scripts:
            code = run(["uv", "run", "smoke", "--profile", "local"], cwd=target, env=env)
            if code != 0:
                failed.append((str(target), code))
                continue

    if failed:
        print("\nFAILED:")
        for path, code in failed:
            print(f" - {path} (exit={code})")
        return 1

    print("\nALL TIER-0 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
